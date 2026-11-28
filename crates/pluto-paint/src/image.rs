//! Image decoding and the browser-owned image cache (Stage 4 #6,
//! Stage 8 Rec 12).
//!
//! Decoding is the most crash-prone CPU work after JS, so it runs on
//! the shared rayon pool (or wherever the caller runs it off the UI
//! thread) and lands in the [`ImageCache`], which owns the
//! [`ResourceTable`] the display list references by id. A decoded
//! image is stored once as RGBA bytes; the renderer uploads it to a
//! wgpu texture from the table, never from the wire.
//!
//! Stage 8 Rec 12 (report §g): the cache is byte-capped at
//! [`IMAGE_CACHE_BYTE_CAP`] (64 MiB, matching the existing 64 MiB
//! resource budget) with LRU eviction — capped by bytes, not by item
//! count, because one 4K image is worth hundreds of icons — and
//! decode-once with dedup: identical source bytes (or identical
//! decoded pixels) return the existing id without decoding again.
//! Mip-level upload scales live here as pure helpers
//! ([`mip_level_count_for`], [`mip_level_for_scale`],
//! [`build_mip_chain`]): the renderer uploads mip levels, not exact
//! scales, and picks a level by draw scale. Animated images are
//! frame-advance dirtying only ([`AnimatedImage`]): advancing a
//! frame swaps the table entry to the next frame and marks that
//! image dirty so the next prepare re-uploads just that frame. Full
//! animation timing (loop counts, disposal modes) is out of scope.

use crate::{Resource, ResourceId, ResourceTable};
use image::AnimationDecoder;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// The image-cache byte cap (Stage 8 Rec 12, report §g): 64 MiB,
/// matching the existing 64 MiB resource budget
/// ([`crate::MAX_RESOURCE_BYTES`]).
pub const IMAGE_CACHE_BYTE_CAP: usize = 64 * 1024 * 1024;

/// Strict decode limits (Stage 11 Rec 12, bomb protection): no
/// image may claim more than 8192 px on a side — above any real web
/// image, below any bomb — and no decode may allocate more than the
/// image-cache cap. The dimension limits are strict (the decoder
/// rejects the header before allocating pixels); `max_alloc` is
/// best-effort per decoder, so the cache also rejects an oversize
/// result before inserting (see [`ImageCache::insert_decoded`]).
pub const MAX_IMAGE_DIMENSION: u32 = 8192;

/// The decode [`image::Limits`] for every Pluto decode: strict
/// 8192 px dimension caps and `max_alloc` at the image-cache cap.
/// (`Limits` is `#[non_exhaustive]`, so this builds from
/// `Default` and sets fields instead of a struct literal.)
pub fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(IMAGE_CACHE_BYTE_CAP as u64);
    limits
}

/// Most frames an animated image may carry (Stage 11 Rec 11): a
/// pathological GIF with thousands of frames is bounded before its
/// pixels are all held — the per-frame byte total is still capped
/// by [`IMAGE_CACHE_BYTE_CAP`] as well.
pub const MAX_ANIMATED_FRAMES: usize = 256;

/// Minimum frame delay for an animated image, in ms (Stage 11
/// Rec 11): a 0-delay frame means "as fast as possible" in the
/// format, which would spin the tab thread; browsers clamp it,
/// and so does Pluto.
pub const MIN_FRAME_DELAY_MS: u32 = 10;

/// Maximum frame rate for an animated image (Stage 11 Rec 11):
/// frames are never advanced faster than ~30 fps, whatever the
/// file's delays say.
pub const MAX_ANIMATION_FPS: u32 = 30;

/// FNV-1a 64-bit hash (deterministic across runs, unlike the
/// randomised `DefaultHasher` — dedup keys must compare equal for
/// identical bytes inserted at different times).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The dedup key for decoded pixels: dimensions mixed into the
/// pixel hash, so two images with the same bytes at different sizes
/// never collide.
fn pixel_hash(image: &DecodedImage) -> u64 {
    pixel_hash_parts(image.width, image.height, &image.rgba)
}

fn pixel_hash_parts(width: u32, height: u32, rgba: &[u8]) -> u64 {
    let mut hash = fnv1a(&width.to_le_bytes());
    hash ^= fnv1a(&height.to_le_bytes()).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^= fnv1a(rgba);
    // Mix once more so a hash of only the dimensions cannot equal
    // a hash of only pixels.
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    hash
}

/// One decoded image: RGBA8 bytes, row-major, `width * height * 4`
/// bytes. The layout is the `Resource::Image` layout, so a decoded
/// image drops straight into the resource table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl DecodedImage {
    /// Byte size of the decoded pixels (`width * height * 4`).
    pub fn byte_len(&self) -> usize {
        self.rgba.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("image decode failed: {0}")]
    Decode(#[from] image::ImageError),
    #[error("decoded image has zero width or height")]
    Empty,
    #[error("decoded image byte length {0} does not match {1}x{2} RGBA")]
    LengthMismatch(usize, u32, u32),
    /// The decoded pixels would exceed the image-cache cap (Stage
    /// 11 Rec 12): rejected before inserting, never after
    /// allocating past the cap.
    #[error("decoded image is {0} bytes, over the {IMAGE_CACHE_BYTE_CAP}-byte image cap")]
    TooLarge(usize),
    /// The animation has too many frames, or its frames' pixels
    /// would exceed the image-cache cap (Stage 11 Rec 11/12).
    #[error("animated image is too large: {0}")]
    AnimatedTooLarge(String),
}

/// Decode one image's bytes (PNG, JPEG, GIF, WebP, ICO, BMP —
/// the formats the workspace `image` dependency enables) to
/// RGBA8. Pure and thread-safe: call it on the rayon pool, never
/// on the UI thread.
///
/// Decodes through [`image::ImageReader`] with the strict
/// [`decode_limits`] (Stage 11 Rec 12): a header claiming more
/// than [`MAX_IMAGE_DIMENSION`] px on a side, or an allocation
/// past the cache cap, is rejected before pixels are allocated.
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    // Stage 15 (M7 Phase 3, Rec 12): SVG bytes rasterise
    // through the M7 rasteriser — the `image` crate has no SVG
    // decoder, so without this sniff an `<img src=*.svg>` lands
    // in the broken-image set. The sniff is structural (first
    // element `<svg>`), never a content-type trust: a PNG that
    // happens to contain the string is still a PNG (its first
    // bytes are the PNG magic, not `<svg`).
    if crate::svg::looks_like_svg(bytes) {
        return crate::svg::decode_svg_image(bytes);
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    reader.limits(decode_limits());
    let dynamic = reader.decode()?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Empty);
    }
    let raw = rgba.into_raw();
    if raw.len() != (width as usize) * (height as usize) * 4 {
        return Err(ImageError::LengthMismatch(raw.len(), width, height));
    }
    if raw.len() > IMAGE_CACHE_BYTE_CAP {
        return Err(ImageError::TooLarge(raw.len()));
    }
    Ok(DecodedImage {
        width,
        height,
        rgba: raw,
    })
}

/// One animated image's decoded frames (Stage 11 Rec 11): every
/// frame at its per-frame delay, in order. `loop_count` is `None`
/// for an infinite loop, `Some(n)` for n total plays (a GIF
/// `Repeat::Finite` / APNG `num_plays`); a static image decodes
/// to exactly one frame with `loop_count` `Some(1)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimatedFrames {
    pub frames: Vec<DecodedImage>,
    pub delays_ms: Vec<u32>,
    pub loop_count: Option<u32>,
}

/// Decode an image's frames: GIF via `AnimationDecoder`, APNG via
/// `PngDecoder::apng`, anything else as a single frame (see
/// [`decode_image`]). Frame delays are honoured (clamped to at
/// least [`MIN_FRAME_DELAY_MS`]); loop counts are read from the
/// file. Bounded before allocating past the cap: the dimension
/// limits apply to the header, the frame count is capped at
/// [`MAX_ANIMATED_FRAMES`], and the running pixel total is capped
/// at [`IMAGE_CACHE_BYTE_CAP`] — a bomb fails partway with its
/// partial frames dropped, never held.
pub fn decode_animated(bytes: &[u8]) -> Result<AnimatedFrames, ImageError> {
    use image::codecs::gif::GifDecoder;
    use image::codecs::png::PngDecoder;
    use image::ImageDecoder;

    let format = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.format());
    let single = |image: DecodedImage| AnimatedFrames {
        frames: vec![image],
        delays_ms: vec![100],
        loop_count: Some(1),
    };
    match format {
        Some(image::ImageFormat::Gif) => {
            let mut decoder = GifDecoder::new(std::io::Cursor::new(bytes))?;
            decoder.set_limits(decode_limits())?;
            let loop_count = match decoder.loop_count() {
                image::metadata::LoopCount::Infinite => None,
                image::metadata::LoopCount::Finite(n) => Some(n.get()),
            };
            collect_frames(decoder, loop_count)
        }
        Some(image::ImageFormat::Png) => {
            let mut decoder = PngDecoder::new(std::io::Cursor::new(bytes))?;
            // The limits ride into the APNG decoder with the
            // inner decoder (it wraps this same `PngDecoder`).
            decoder.set_limits(decode_limits())?;
            if decoder.is_apng()? {
                let apng = decoder.apng()?;
                let loop_count = match apng.loop_count() {
                    image::metadata::LoopCount::Infinite => None,
                    image::metadata::LoopCount::Finite(n) => Some(n.get()),
                };
                collect_frames(apng, loop_count)
            } else {
                Ok(single(decode_image(bytes)?))
            }
        }
        _ => Ok(single(decode_image(bytes)?)),
    }
}

/// Collect an animation decoder's frames into [`AnimatedFrames`],
/// enforcing the frame-count and pixel-total caps as frames land.
fn collect_frames<'a>(
    decoder: impl AnimationDecoder<'a>,
    loop_count: Option<u32>,
) -> Result<AnimatedFrames, ImageError> {
    let mut frames = Vec::new();
    let mut delays_ms = Vec::new();
    let mut total_bytes = 0usize;
    for frame in decoder.into_frames() {
        let frame = frame?;
        let buffer = frame.buffer();
        let (width, height) = buffer.dimensions();
        if width == 0 || height == 0 {
            return Err(ImageError::Empty);
        }
        let rgba = buffer.clone().into_raw();
        if rgba.len() != (width as usize) * (height as usize) * 4 {
            return Err(ImageError::LengthMismatch(rgba.len(), width, height));
        }
        total_bytes += rgba.len();
        if total_bytes > IMAGE_CACHE_BYTE_CAP {
            return Err(ImageError::AnimatedTooLarge(format!(
                "frames total {total_bytes} bytes, over the {IMAGE_CACHE_BYTE_CAP}-byte cap"
            )));
        }
        frames.push(DecodedImage {
            width,
            height,
            rgba,
        });
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay = numer.checked_div(denom).unwrap_or(0);
        delays_ms.push(delay.max(MIN_FRAME_DELAY_MS));
        if frames.len() > MAX_ANIMATED_FRAMES {
            return Err(ImageError::AnimatedTooLarge(format!(
                "more than {MAX_ANIMATED_FRAMES} frames"
            )));
        }
    }
    if frames.is_empty() {
        return Err(ImageError::Empty);
    }
    Ok(AnimatedFrames {
        frames,
        delays_ms,
        loop_count,
    })
}

/// Decode several images in parallel on the shared rayon pool.
/// Order is preserved: result `i` is the decode of input `i`.
pub fn decode_images_parallel(inputs: &[&[u8]]) -> Vec<Result<DecodedImage, ImageError>> {
    inputs.par_iter().map(|bytes| decode_image(bytes)).collect()
}

/// How many mip levels a `width`×`height` texture has: 1 +
/// floor(log2(max(w, h))). A 256×256 image has 9 levels (256 … 1);
/// a 1×1 image has 1. Zero dimensions have 0 levels.
pub fn mip_level_count_for(width: u32, height: u32) -> u32 {
    let max = width.max(height);
    if max == 0 {
        0
    } else {
        32 - max.leading_zeros()
    }
}

/// Which mip level to sample for a draw scale (`draw_px /
/// texture_px`, the smaller axis when they differ): level 0 at
/// scale ≥ 1 (drawn at or above native size), one level per halving
/// below that, clamped to `max_level` (the last level,
/// `mip_level_count_for(w, h) - 1`). Upload-scaling uses mip levels,
/// not exact scales, so scaled draws reuse the same uploaded levels
/// (report §g). A scale of 0 (or non-finite) takes the smallest level.
pub fn mip_level_for_scale(scale: f32, max_level: u32) -> u32 {
    if !scale.is_finite() || scale <= 0.0 {
        return max_level;
    }
    if scale >= 1.0 {
        return 0;
    }
    ((-scale.log2()).floor() as u32).min(max_level)
}

/// One 2×2 box-filter downscale step: each output pixel is the
/// mean of the up-to-four source pixels covering it (edges clamp,
/// so odd dimensions shrink by one pixel, never below 1×1).
pub fn downscale_rgba(image: &DecodedImage) -> DecodedImage {
    let (w, h) = (image.width, image.height);
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut rgba = vec![0u8; (nw as usize) * (nh as usize) * 4];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx < w && sy < h {
                        let off = ((sy as usize) * (w as usize) + (sx as usize)) * 4;
                        for (c, s) in sum.iter_mut().enumerate() {
                            *s += image.rgba[off + c] as u32;
                        }
                        count += 1;
                    }
                }
            }
            let off = ((y as usize) * (nw as usize) + (x as usize)) * 4;
            for c in 0..4 {
                rgba[off + c] = (sum[c] / count) as u8;
            }
        }
    }
    DecodedImage {
        width: nw,
        height: nh,
        rgba,
    }
}

/// Resize an image to an exact size with an area-average (box)
/// filter: each output pixel is the mean of the source pixels its
/// rectangle covers. Used for favicons (downscale to 16×16 /
/// 32×32, Stage 11 Rec 7) — unlike the mip chain's halving steps,
/// this lands on the requested size exactly. Pure and thread-safe.
pub fn resize_rgba(image: &DecodedImage, width: u32, height: u32) -> DecodedImage {
    let (w, h) = (image.width, image.height);
    if width == 0 || height == 0 || w == 0 || h == 0 {
        return DecodedImage {
            width,
            height,
            rgba: vec![0u8; (width as usize) * (height as usize) * 4],
        };
    }
    if (w, h) == (width, height) {
        return image.clone();
    }
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    for y in 0..height {
        // Source rows covered by output row y: [y0, y1).
        let y0 = (y as usize * h as usize) / height as usize;
        let y1 = ((y as usize + 1) * h as usize / height as usize).max(y0 + 1);
        for x in 0..width {
            let x0 = (x as usize * w as usize) / width as usize;
            let x1 = ((x as usize + 1) * w as usize / width as usize).max(x0 + 1);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let off = (sy * w as usize + sx) * 4;
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += image.rgba[off + c] as u32;
                    }
                    count += 1;
                }
            }
            let off = ((y as usize) * (width as usize) + (x as usize)) * 4;
            for c in 0..4 {
                rgba[off + c] = (sum[c] / count) as u8;
            }
        }
    }
    DecodedImage {
        width,
        height,
        rgba,
    }
}

/// The full mip chain for an image: level 0 is the image itself,
/// each further level is a [`downscale_rgba`] of the previous one,
/// down to 1×1. Length is [`mip_level_count_for`]. The renderer
/// uploads every level, so a draw at any scale samples an uploaded
/// level instead of resampling level 0.
pub fn build_mip_chain(image: &DecodedImage) -> Vec<DecodedImage> {
    let mut chain = vec![image.clone()];
    while chain
        .last()
        .is_some_and(|last| last.width > 1 || last.height > 1)
    {
        let next = downscale_rgba(chain.last().expect("non-empty chain"));
        chain.push(next);
    }
    chain
}

/// Total bytes of a mip chain (level 0 plus all downscaled levels —
/// about 4/3 of level 0 for a square image).
pub fn mip_chain_bytes(chain: &[DecodedImage]) -> usize {
    chain.iter().map(|level| level.byte_len()).sum()
}

/// One animated image (Stage 8 Rec 12, report §g): its frames at
/// their per-frame delays. Frame-advance dirtying only — the
/// `ImageCache` advances the frame and swaps the table entry; no
/// timer, loop count, or disposal mode lives here (out of scope).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimatedImage {
    frames: Vec<DecodedImage>,
    delays_ms: Vec<u32>,
    current: usize,
}

impl AnimatedImage {
    /// Build from frames + delays. `None` when there are no frames
    /// or the frames differ in size (animated frames share one
    /// size). A missing delay defaults to 100 ms; extra delays are
    /// ignored.
    pub fn new(frames: Vec<DecodedImage>, delays_ms: Vec<u32>) -> Option<Self> {
        if frames.is_empty() {
            return None;
        }
        let (w, h) = (frames[0].width, frames[0].height);
        if frames.iter().any(|f| f.width != w || f.height != h) {
            return None;
        }
        let delays_ms = (0..frames.len())
            .map(|i| delays_ms.get(i).copied().unwrap_or(100))
            .collect();
        Some(Self {
            frames,
            delays_ms,
            current: 0,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn current_index(&self) -> usize {
        self.current
    }

    pub fn current_frame(&self) -> &DecodedImage {
        &self.frames[self.current]
    }

    /// The delay of frame `index` in ms, or `None` out of range.
    pub fn delay_ms(&self, index: usize) -> Option<u32> {
        self.delays_ms.get(index).copied()
    }

    /// Advance to the next frame (wrapping). Returns `true` when
    /// the frame actually changed — a one-frame "animation" never
    /// advances, so it never dirties anything.
    pub fn advance(&mut self) -> bool {
        if self.frames.len() <= 1 {
            return false;
        }
        self.current = (self.current + 1) % self.frames.len();
        true
    }
}

/// The browser-owned image cache: decoded images keyed by
/// [`ResourceId`], backed by the [`ResourceTable`] display items
/// reference. Ids are assigned in insertion order, starting at 1
/// (0 is reserved as "no resource").
///
/// Stage 8 Rec 12: byte-capped LRU. `bytes_used` never exceeds
/// `cap_bytes` — and Stage 11 Rec 12: a single image whose RGBA
/// exceeds the cap is REFUSED before inserting (never stored
/// alone, never allocated past the cap), so a bomb cannot land in
/// the table at all. `get` does not touch recency (the table is
/// read without the cache); [`get_touch`] does, for callers that
/// count a read as use.
#[derive(Debug, Default)]
pub struct ImageCache {
    table: ResourceTable,
    next_id: u32,
    cap_bytes: usize,
    bytes_used: usize,
    /// Front = most recently used, back = eviction candidate.
    lru: VecDeque<ResourceId>,
    /// Source-bytes hash -> id (decode-once for `insert_bytes`).
    by_source: HashMap<u64, ResourceId>,
    /// Decoded-pixels hash -> id (decode-once for `insert_decoded`).
    by_pixels: HashMap<u64, ResourceId>,
    source_hash: HashMap<ResourceId, u64>,
    pixel_hash: HashMap<ResourceId, u64>,
    evictions: u64,
    dedup_hits: u64,
    animated: HashMap<ResourceId, AnimatedImage>,
    /// Animated ids whose current frame has not been uploaded yet
    /// (set by [`advance_animated`], cleared by
    /// [`clear_frame_dirty`]).
    frame_dirty: HashSet<ResourceId>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self::with_cap(IMAGE_CACHE_BYTE_CAP)
    }

    /// A cache with an explicit byte cap (tests use small caps to
    /// force eviction without 64 MiB of pixels).
    pub fn with_cap(cap_bytes: usize) -> Self {
        Self {
            table: ResourceTable::default(),
            next_id: 1,
            cap_bytes,
            bytes_used: 0,
            lru: VecDeque::new(),
            by_source: HashMap::new(),
            by_pixels: HashMap::new(),
            source_hash: HashMap::new(),
            pixel_hash: HashMap::new(),
            evictions: 0,
            dedup_hits: 0,
            animated: HashMap::new(),
            frame_dirty: HashSet::new(),
        }
    }

    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    /// Images evicted for the cap so far.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Inserts that returned an existing id without decoding.
    pub fn dedup_hits(&self) -> u64 {
        self.dedup_hits
    }

    /// Decode `bytes` and store the result, returning its id.
    /// Identical source bytes return the existing id without
    /// decoding again (decode-once, report §g). `Err` when the
    /// decode fails or the decoded pixels exceed the cap (Stage
    /// 11 Rec 12 — refused before inserting).
    pub fn insert_bytes(&mut self, bytes: &[u8]) -> Result<ResourceId, ImageError> {
        let hash = fnv1a(bytes);
        if let Some(&id) = self.by_source.get(&hash)
            && self.table.get(id).is_some()
        {
            self.dedup_hits += 1;
            self.touch(id);
            return Ok(id);
        }
        let decoded = decode_image(bytes)?;
        self.insert_decoded_with_source(decoded, Some(hash))
    }

    /// Store an already-decoded image, returning its id. Identical
    /// pixels (same size and bytes) return the existing id.
    /// `Err` when the pixels exceed the cap (Stage 11 Rec 12).
    pub fn insert_decoded(&mut self, image: DecodedImage) -> Result<ResourceId, ImageError> {
        self.insert_decoded_with_source(image, None)
    }

    fn insert_decoded_with_source(
        &mut self,
        image: DecodedImage,
        source_hash: Option<u64>,
    ) -> Result<ResourceId, ImageError> {
        let phash = pixel_hash(&image);
        if let Some(&id) = self.by_pixels.get(&phash)
            && self.table.get(id).is_some()
        {
            self.dedup_hits += 1;
            if let Some(hash) = source_hash {
                self.by_source.insert(hash, id);
                self.source_hash.insert(id, hash);
            }
            self.touch(id);
            return Ok(id);
        }
        let size = image.byte_len();
        // Stage 11 Rec 12: refuse an image larger than the cap
        // before it lands in the table — no eviction dance can
        // make room for it, and the IPC frame could not carry it
        // either.
        if size > self.cap_bytes {
            return Err(ImageError::TooLarge(size));
        }
        self.evict_until_fits(size);
        let id = ResourceId(self.next_id);
        // Ids are u32 and insertion is one-per-call; wrapping is not a
        // practical concern (4 billion images), but never hand out 0.
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.table.insert(
            id,
            Resource::Image {
                width: image.width,
                height: image.height,
                rgba: image.rgba,
            },
        );
        self.bytes_used += size;
        self.lru.push_front(id);
        self.by_pixels.insert(phash, id);
        self.pixel_hash.insert(id, phash);
        if let Some(hash) = source_hash {
            self.by_source.insert(hash, id);
            self.source_hash.insert(id, hash);
        }
        Ok(id)
    }

    /// Store a resource under a caller-chosen id (Stage 15, M7
    /// Phase 2, Rec 5): the worker's rasteriser names its Blobs
    /// with deterministic ids (`svg_blob_id` /
    /// `bg_gradient_blob_id`), so the tab thread inserts them
    /// under that id instead of the sequential `next_id`.
    /// Replacing an existing entry keeps the byte accounting,
    /// LRU order, and hash maps correct: the old entry's
    /// bookkeeping is removed first (not counted as an eviction),
    /// then room is made — the id being inserted is already out
    /// of the LRU, so `evict_until_fits` can never evict it.
    /// The pixels are registered in the dedup map like any
    /// insert. `false` when the id is 0 or the resource exceeds
    /// the cap (refused, like `insert_decoded`); the sequential
    /// `next_id` is untouched — the deterministic ids live in
    /// the 0x4000_0000+ range, far above any sequential id in
    /// practice, so the two namespaces never collide.
    pub fn insert_with_id(&mut self, id: ResourceId, resource: Resource) -> bool {
        let Resource::Image {
            width,
            height,
            rgba,
        } = &resource;
        let size = rgba.len();
        if id == ResourceId(0) || size > self.cap_bytes {
            return false;
        }
        if self.table.get(id).is_some() {
            self.remove_entry(id);
        }
        self.evict_until_fits(size);
        let phash = pixel_hash_parts(*width, *height, rgba);
        self.table.insert(id, resource);
        self.bytes_used += size;
        self.lru.push_front(id);
        self.by_pixels.insert(phash, id);
        self.pixel_hash.insert(id, phash);
        true
    }

    /// Store an animated image: frame 0 lands in the table under
    /// the returned id, the rest are held for
    /// [`advance_animated`]. `None` when there are no frames, the
    /// frames differ in size (see [`AnimatedImage::new`]), or
    /// frame 0's pixels exceed the cap (Stage 11 Rec 12).
    /// Animated inserts bypass dedup (frames are identity, not
    /// content) but still respect the byte cap.
    pub fn insert_animated(
        &mut self,
        frames: Vec<DecodedImage>,
        delays_ms: Vec<u32>,
    ) -> Option<ResourceId> {
        let animated = AnimatedImage::new(frames, delays_ms)?;
        let first = animated.current_frame().clone();
        let size = first.byte_len();
        if size > self.cap_bytes {
            return None;
        }
        self.evict_until_fits(size);
        let id = ResourceId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.table.insert(
            id,
            Resource::Image {
                width: first.width,
                height: first.height,
                rgba: first.rgba,
            },
        );
        self.bytes_used += size;
        self.lru.push_front(id);
        self.animated.insert(id, animated);
        Some(id)
    }

    /// Advance an animated image to its next frame (wrapping) and
    /// swap the table entry to that frame's pixels. Returns `true`
    /// only when the frame actually changed; on `true` the id is
    /// marked frame-dirty ([`is_frame_dirty`]) so the renderer's
    /// next prepare re-uploads just that image. A non-animated id,
    /// an unknown id, or a one-frame animation returns `false` and
    /// dirties nothing. Advancing never re-decodes: the frames
    /// were decoded once at insert. The frame is *not* advanced
    /// when the image is off-screen — the caller (which knows
    /// visibility) simply does not call this.
    pub fn advance_animated(&mut self, id: ResourceId) -> bool {
        let Some(animated) = self.animated.get_mut(&id) else {
            return false;
        };
        if !animated.advance() {
            return false;
        }
        let frame = animated.current_frame().clone();
        // Frames share one size, so the byte count is unchanged.
        self.table.insert(
            id,
            Resource::Image {
                width: frame.width,
                height: frame.height,
                rgba: frame.rgba,
            },
        );
        self.frame_dirty.insert(id);
        self.touch(id);
        true
    }

    /// The current frame index of an animated image, or `None`
    /// when the id is not animated.
    pub fn animated_frame(&self, id: ResourceId) -> Option<usize> {
        self.animated.get(&id).map(|a| a.current_index())
    }

    pub fn is_animated(&self, id: ResourceId) -> bool {
        self.animated.contains_key(&id)
    }

    /// True when an animated frame advance is waiting to be
    /// uploaded (set by [`advance_animated`]).
    pub fn is_frame_dirty(&self, id: ResourceId) -> bool {
        self.frame_dirty.contains(&id)
    }

    /// Clear the frame-dirty flag after the renderer re-uploads
    /// the image. Returns `true` when the flag was set.
    pub fn clear_frame_dirty(&mut self, id: ResourceId) -> bool {
        self.frame_dirty.remove(&id)
    }

    /// The table the display list references (serialisable with it).
    pub fn table(&self) -> &ResourceTable {
        &self.table
    }

    pub fn get(&self, id: ResourceId) -> Option<&Resource> {
        self.table.get(id)
    }

    /// Like [`get`], but counts as use for LRU purposes.
    pub fn get_touch(&mut self, id: ResourceId) -> Option<&Resource> {
        self.table.get(id)?;
        self.touch(id);
        self.table.get(id)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    fn touch(&mut self, id: ResourceId) {
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        self.lru.push_front(id);
    }

    /// Evict least-recently-used images until `size` more bytes
    /// fit under the cap. A single image larger than the cap still
    /// stores (alone, after evicting everything else).
    fn evict_until_fits(&mut self, size: usize) {
        while self.bytes_used + size > self.cap_bytes {
            let Some(id) = self.lru.back().copied() else {
                break;
            };
            self.evict(id);
        }
    }

    fn evict(&mut self, id: ResourceId) {
        self.remove_entry(id);
        self.evictions += 1;
    }

    /// Drop one entry's table row and all its bookkeeping (LRU,
    /// hashes, animation, bytes) without counting an eviction —
    /// the shared removal behind [`Self::evict`] and the replace
    /// path of [`Self::insert_with_id`].
    fn remove_entry(&mut self, id: ResourceId) {
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        if let Some(Resource::Image { rgba, .. }) = self.table.resources.remove(&id) {
            self.bytes_used = self.bytes_used.saturating_sub(rgba.len());
        }
        if let Some(hash) = self.pixel_hash.remove(&id)
            && self.by_pixels.get(&hash) == Some(&id) {
                self.by_pixels.remove(&hash);
            }
        if let Some(hash) = self.source_hash.remove(&id)
            && self.by_source.get(&hash) == Some(&id) {
                self.by_source.remove(&hash);
            }
        self.animated.remove(&id);
        self.frame_dirty.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder};

    /// Encode a small RGBA image to PNG bytes, so the decode tests
    /// exercise real format bytes without a checked-in fixture.
    fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(rgba, width, height, ExtendedColorType::Rgba8)
            .expect("test PNG encodes");
        out
    }

    /// A solid-colour decoded image of `size`×`size` (64 bytes at 4×4).
    fn solid(size: u32, px: [u8; 4]) -> DecodedImage {
        DecodedImage {
            width: size,
            height: size,
            rgba: px.repeat((size * size) as usize),
        }
    }

    #[test]
    fn decode_png_round_trips_exact_pixels() {
        // 2x2: red, green, blue, transparent white.
        let rgba: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 0,
        ];
        let bytes = png_bytes(2, 2, &rgba);
        let decoded = decode_image(&bytes).expect("PNG decodes");
        assert_eq!((decoded.width, decoded.height), (2, 2));
        assert_eq!(decoded.rgba, rgba);
    }

    #[test]
    fn decode_rejects_garbage_and_truncated_bytes() {
        assert!(decode_image(b"not an image at all").is_err());
        let bytes = png_bytes(2, 2, &[255u8; 16]);
        assert!(decode_image(&bytes[..bytes.len() / 2]).is_err());
        assert!(decode_image(&[]).is_err());
    }

    #[test]
    fn parallel_decode_matches_serial_decode() {
        let inputs: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                let px = [i * 30, 100, 200, 255];
                let rgba = [px, px, px, px].concat();
                png_bytes(2, 2, &rgba)
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|b| b.as_slice()).collect();
        let parallel = decode_images_parallel(&refs);
        let serial: Vec<_> = refs.iter().map(|b| decode_image(b)).collect();
        assert_eq!(parallel.len(), serial.len());
        for (p, s) in parallel.iter().zip(serial.iter()) {
            assert_eq!(p.as_ref().ok(), s.as_ref().ok());
        }
        // And every parallel result is the right image.
        for (i, result) in parallel.iter().enumerate() {
            let img = result.as_ref().expect("parallel decode succeeds");
            assert_eq!(img.rgba[0], (i as u8) * 30);
        }
    }

    #[test]
    fn image_cache_assigns_ids_and_serves_the_table() {
        let mut cache = ImageCache::new();
        assert!(cache.is_empty());
        let red = png_bytes(1, 1, &[255, 0, 0, 255]);
        let green = png_bytes(1, 1, &[0, 255, 0, 255]);
        let id_red = cache.insert_bytes(&red).expect("red decodes");
        let id_green = cache.insert_bytes(&green).expect("green decodes");
        assert_ne!(id_red, id_green);
        assert_ne!(id_red, ResourceId(0), "0 is never handed out");
        assert_eq!(cache.len(), 2);
        match cache.get(id_red).expect("red is cached") {
            Resource::Image {
                width,
                height,
                rgba,
            } => {
                assert_eq!((*width, *height), (1, 1));
                assert_eq!(rgba, &[255, 0, 0, 255]);
            }
        }
        assert!(cache.get(ResourceId(999)).is_none());
        // The table round-trips through postcard with its images.
        let bytes = postcard::to_allocvec(cache.table()).expect("table serialises");
        let back: ResourceTable = postcard::from_bytes(&bytes).expect("table deserialises");
        assert_eq!(back.len(), 2);
        assert_eq!(back.get(id_green), cache.get(id_green));
    }

    #[test]
    fn image_cache_enforces_byte_cap_with_lru_eviction() {
        // 4x4 RGBA = 64 bytes each; a 128-byte cap holds exactly two.
        let mut cache = ImageCache::with_cap(128);
        let a = cache.insert_decoded(solid(4, [255, 0, 0, 255])).expect("a inserts");
        let b = cache.insert_decoded(solid(4, [0, 255, 0, 255])).expect("b inserts");
        assert_eq!(cache.bytes_used(), 128);
        assert_eq!(cache.evictions(), 0);
        // A third image evicts the least recently used (a).
        let c = cache.insert_decoded(solid(4, [0, 0, 255, 255])).expect("c inserts");
        assert_eq!(cache.evictions(), 1);
        assert!(cache.get(a).is_none(), "LRU image a is evicted");
        assert!(cache.get(b).is_some());
        assert!(cache.get(c).is_some());
        assert_eq!(cache.bytes_used(), 128, "cap holds after eviction");
        assert_eq!(cache.len(), 2);
        // Touching b makes c the LRU; the next insert evicts c, not b.
        cache.get_touch(b).expect("b is cached");
        let d = cache.insert_decoded(solid(4, [255, 255, 0, 255])).expect("d inserts");
        assert!(cache.get(c).is_none(), "untouched c is evicted next");
        assert!(cache.get(b).is_some(), "touched b survives");
        assert!(cache.get(d).is_some());
        assert_eq!(cache.bytes_used(), 128);
    }

    #[test]
    fn image_cache_dedups_identical_bytes_and_pixels() {
        let mut cache = ImageCache::new();
        let red = png_bytes(1, 1, &[255, 0, 0, 255]);
        let id1 = cache.insert_bytes(&red).expect("red decodes");
        // Same source bytes again: no re-decode, same id, no growth.
        let id2 = cache.insert_bytes(&red).expect("red decodes again");
        assert_eq!(id1, id2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), 4);
        assert_eq!(cache.dedup_hits(), 1);
        // Same pixels via the decoded path dedup too.
        let id3 = cache
            .insert_decoded(DecodedImage {
                width: 1,
                height: 1,
                rgba: vec![255, 0, 0, 255],
            })
            .expect("same pixels dedup");
        assert_eq!(id3, id1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.dedup_hits(), 2);
        // Different pixels are a different image.
        let id4 = cache
            .insert_decoded(DecodedImage {
                width: 1,
                height: 1,
                rgba: vec![0, 255, 0, 255],
            })
            .expect("different pixels insert");
        assert_ne!(id4, id1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn mip_level_count_and_selection() {
        assert_eq!(mip_level_count_for(256, 256), 9);
        assert_eq!(mip_level_count_for(1, 1), 1);
        assert_eq!(mip_level_count_for(100, 50), 7); // floor(log2 100) = 6
        assert_eq!(mip_level_count_for(0, 0), 0);
        assert_eq!(mip_level_count_for(1024, 768), 11);
        let max = mip_level_count_for(256, 256) - 1; // 8
        assert_eq!(mip_level_for_scale(1.0, max), 0);
        assert_eq!(mip_level_for_scale(2.0, max), 0, "upscaled: level 0");
        assert_eq!(mip_level_for_scale(0.5, max), 1);
        assert_eq!(mip_level_for_scale(0.25, max), 2);
        assert_eq!(mip_level_for_scale(0.1, max), 3);
        assert_eq!(mip_level_for_scale(0.0, max), max);
        assert_eq!(mip_level_for_scale(f32::NAN, max), max);
        assert_eq!(mip_level_for_scale(0.001, max), max, "clamped to last");
    }

    #[test]
    fn mip_chain_downscales_and_counts_bytes() {
        // 4x4 quadrants: red / green / blue / white; the 2x2 level
        // keeps the quadrant colours, the 1x1 level is their mean.
        let rgba: Vec<u8> = (0..16)
            .flat_map(|i| {
                let (x, y) = (i % 4, i / 4);
                match (x < 2, y < 2) {
                    (true, true) => [255, 0, 0, 255],
                    (false, true) => [0, 255, 0, 255],
                    (true, false) => [0, 0, 255, 255],
                    (false, false) => [255, 255, 255, 255],
                }
            })
            .collect();
        let image = DecodedImage {
            width: 4,
            height: 4,
            rgba,
        };
        let chain = build_mip_chain(&image);
        assert_eq!(chain.len(), 3, "4 -> 2 -> 1");
        assert_eq!((chain[0].width, chain[0].height), (4, 4));
        assert_eq!((chain[1].width, chain[1].height), (2, 2));
        assert_eq!(
            chain[1].rgba,
            [
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255
            ]
        );
        assert_eq!((chain[2].width, chain[2].height), (1, 1));
        assert_eq!(
            chain[2].rgba,
            [127, 127, 127, 255],
            "mean of the four quadrants"
        );
        assert_eq!(
            mip_chain_bytes(&chain),
            64 + 16 + 4,
            "level bytes sum, ~4/3 of level 0"
        );
        // Odd dimensions shrink by one, never below 1x1.
        let odd = solid(3, [10, 20, 30, 255]);
        let down = downscale_rgba(&odd);
        assert_eq!((down.width, down.height), (1, 1));
        assert_eq!(down.rgba, [10, 20, 30, 255]);
    }

    #[test]
    fn animated_frame_advance_dirties_only_that_image() {
        let mut cache = ImageCache::new();
        let red = solid(2, [255, 0, 0, 255]);
        let green = solid(2, [0, 255, 0, 255]);
        let blue = solid(2, [0, 0, 255, 255]);
        let animated = cache
            .insert_animated(vec![red.clone(), green.clone()], vec![100, 200])
            .expect("animated inserts");
        let still = cache.insert_decoded(blue.clone()).expect("still inserts");
        assert_eq!(cache.animated_frame(animated), Some(0));
        assert!(!cache.is_frame_dirty(animated));
        assert!(!cache.is_frame_dirty(still));
        // Advancing swaps the table entry to frame 1 and dirties
        // only the animated image; the still image is untouched.
        assert!(cache.advance_animated(animated));
        assert_eq!(cache.animated_frame(animated), Some(1));
        assert!(cache.is_frame_dirty(animated));
        assert!(!cache.is_frame_dirty(still));
        match cache.get(animated).expect("frame 1 is cached") {
            Resource::Image { rgba, .. } => assert_eq!(rgba, &green.rgba),
        }
        match cache.get(still).expect("still is cached") {
            Resource::Image { rgba, .. } => assert_eq!(rgba, &blue.rgba),
        }
        // The renderer clears the flag after re-uploading.
        assert!(cache.clear_frame_dirty(animated));
        assert!(!cache.is_frame_dirty(animated));
        assert!(!cache.clear_frame_dirty(animated), "already clear");
        // Advance wraps back to frame 0.
        assert!(cache.advance_animated(animated));
        assert_eq!(cache.animated_frame(animated), Some(0));
        match cache.get(animated).expect("frame 0 again") {
            Resource::Image { rgba, .. } => assert_eq!(rgba, &red.rgba),
        }
        // A still image and an unknown id never advance or dirty.
        assert!(!cache.advance_animated(still));
        assert!(!cache.advance_animated(ResourceId(999)));
        assert!(!cache.is_frame_dirty(still));
        // A one-frame "animation" never advances.
        let one = cache
            .insert_animated(vec![red.clone()], vec![100])
            .expect("one-frame inserts");
        assert!(!cache.advance_animated(one));
        assert!(!cache.is_frame_dirty(one));
        // Bad animated inserts are refused, not panicked.
        assert!(cache.insert_animated(vec![], vec![]).is_none());
        assert!(
            cache
                .insert_animated(vec![red, solid(3, [0, 0, 0, 255])], vec![100, 100])
                .is_none(),
            "frames must share one size"
        );
    }

    /// CRC-32 (IEEE) for the hand-built PNG headers below — the
    /// test needs a valid IHDR CRC without pulling a CRC crate
    /// into the test build.
    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        let mut crc = 0xffff_ffffu32;
        for &b in data {
            crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
        }
        crc ^ 0xffff_ffff
    }

    /// A PNG whose IHDR claims `width` × `height` but carries no
    /// pixel data: the decoder must reject the header before
    /// allocating for it.
    fn png_header_only(width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![137, 80, 78, 71, 13, 10, 26, 10];
        // IHDR: length 13, type, width, height, bit depth 8,
        // colour type 6 (RGBA), compression/filter/interlace 0.
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(b"IHDR");
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        out.extend_from_slice(&13u32.to_be_bytes());
        out.extend_from_slice(&ihdr);
        out.extend_from_slice(&crc32(&ihdr).to_be_bytes());
        // IEND.
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"IEND");
        out.extend_from_slice(&crc32(b"IEND").to_be_bytes());
        out
    }

    #[test]
    fn decode_rejects_oversize_png_header_without_allocating() {
        // A 200-megapixel header (200000 × 1000): the strict
        // dimension limit rejects it at the header, before any
        // pixel buffer exists — the error is a limits error, not
        // an allocation failure.
        let bomb = png_header_only(200_000, 1000);
        let err = decode_image(&bomb).expect_err("200 MP header rejected");
        assert!(
            matches!(err, ImageError::Decode(_)),
            "limits rejection surfaces as a decode error: {err:?}"
        );
        // And a header just over the 8192 px cap is rejected too.
        assert!(decode_image(&png_header_only(8193, 10)).is_err());
        // A header at the cap is NOT rejected for dimensions: it
        // fails later, on the missing pixel data — the error is
        // about the truncated file, never about a limit.
        let at_cap = decode_image(&png_header_only(8192, 10)).expect_err("no pixels");
        assert!(
            !at_cap.to_string().to_ascii_lowercase().contains("limit"),
            "at-cap header fails on missing data, not on a limit: {at_cap}"
        );
        // A normal 4K image decodes fine (small stand-in: 2x2
        // PNG proves the limited path still decodes real files).
        let bytes = png_bytes(2, 2, &[255u8; 16]);
        assert!(decode_image(&bytes).is_ok());
    }

    #[test]
    fn image_cache_refuses_image_over_the_cap() {
        // A 128-byte cap: a 4x4 image (64 bytes) fits, an 8x8
        // image (256 bytes) is refused before inserting — the
        // table never holds an image over its cap.
        let mut cache = ImageCache::with_cap(128);
        let big = solid(8, [255, 0, 0, 255]);
        assert_eq!(big.byte_len(), 256);
        assert!(matches!(
            cache.insert_decoded(big),
            Err(ImageError::TooLarge(256))
        ));
        assert!(cache.is_empty());
        assert_eq!(cache.bytes_used(), 0);
        // The same refusal through the bytes path (a real PNG
        // whose RGBA exceeds the cap).
        let png = png_bytes(8, 8, &[1u8; 8 * 8 * 4]);
        assert!(matches!(
            cache.insert_bytes(&png),
            Err(ImageError::TooLarge(256))
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn insert_with_id_stores_replaces_and_refuses() {
        // Stage 15 Rec 5: the worker's rasteriser names its Blobs
        // with deterministic ids, so the tab thread inserts them
        // under that id instead of the sequential counter.
        let mut cache = ImageCache::with_cap(1024);
        let id = ResourceId(0x4000_0001);
        let red = solid(2, [255, 0, 0, 255]);
        assert!(cache.insert_with_id(
            id,
            Resource::Image {
                width: red.width,
                height: red.height,
                rgba: red.rgba.clone(),
            }
        ));
        assert_eq!(cache.get(id), Some(&Resource::Image {
            width: 2,
            height: 2,
            rgba: red.rgba.clone(),
        }));
        assert_eq!(cache.bytes_used(), 16);
        assert_eq!(cache.evictions(), 0);
        // Replace with different pixels under the same id: one
        // entry, bytes re-accounted, no eviction counted.
        let green = solid(1, [0, 255, 0, 255]);
        assert!(cache.insert_with_id(
            id,
            Resource::Image {
                width: green.width,
                height: green.height,
                rgba: green.rgba.clone(),
            }
        ));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), 4);
        assert_eq!(cache.evictions(), 0);
        assert_eq!(cache.get(id), Some(&Resource::Image {
            width: 1,
            height: 1,
            rgba: green.rgba,
        }));
        // Oversize is refused; id 0 is refused.
        let big = Resource::Image {
            width: 100,
            height: 100,
            rgba: vec![0u8; 40_000],
        };
        assert!(!cache.insert_with_id(ResourceId(0x4000_0002), big));
        assert!(!cache.insert_with_id(
            ResourceId(0),
            Resource::Image {
                width: 1,
                height: 1,
                rgba: vec![0, 0, 0, 255],
            }
        ));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn insert_with_id_evicts_lru_but_never_itself() {
        // A 64-byte cap: two 16-byte sequential images fill half
        // the cap; a 64-byte Blob must evict both and land — and
        // replacing that Blob afterwards must not evict itself.
        let mut cache = ImageCache::with_cap(64);
        let a = cache.insert_decoded(solid(2, [1, 0, 0, 255])).unwrap();
        let b = cache.insert_decoded(solid(2, [2, 0, 0, 255])).unwrap();
        assert_eq!(cache.bytes_used(), 32);
        let blob = ResourceId(0x4000_0001);
        assert!(cache.insert_with_id(
            blob,
            Resource::Image {
                width: 8,
                height: 2,
                rgba: vec![9u8; 64],
            }
        ));
        assert_eq!(cache.bytes_used(), 64);
        assert!(cache.get(blob).is_some());
        assert!(cache.get(a).is_none());
        assert!(cache.get(b).is_none());
        // Replace the Blob with a smaller one: it stays.
        assert!(cache.insert_with_id(
            blob,
            Resource::Image {
                width: 2,
                height: 2,
                rgba: vec![9u8; 16],
            }
        ));
        assert_eq!(cache.bytes_used(), 16);
        assert!(cache.get(blob).is_some());
    }

    #[test]
    fn decode_animated_gif_reads_frames_delays_and_loop_count() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, RgbaImage};
        // Two 2x2 frames: red for 100 ms, green for 250 ms,
        // looping forever.
        let mut bytes = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut bytes);
            enc.set_repeat(Repeat::Infinite).expect("repeat sets");
            let red = RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
            let green = RgbaImage::from_pixel(2, 2, image::Rgba([0, 255, 0, 255]));
            enc.encode_frame(Frame::from_parts(
                red,
                0,
                0,
                Delay::from_numer_denom_ms(100, 1),
            ))
            .expect("frame 0 encodes");
            enc.encode_frame(Frame::from_parts(
                green,
                0,
                0,
                Delay::from_numer_denom_ms(250, 1),
            ))
            .expect("frame 1 encodes");
        }
        let animated = decode_animated(&bytes).expect("GIF decodes animated");
        assert_eq!(animated.frames.len(), 2);
        assert_eq!(animated.delays_ms, vec![100, 250]);
        assert_eq!(animated.loop_count, None, "infinite loop");
        assert_eq!(animated.frames[0].rgba, [255, 0, 0, 255].repeat(4));
        assert_eq!(animated.frames[1].rgba, [0, 255, 0, 255].repeat(4));
        // And the plain decode of the same GIF gives frame 0.
        let still = decode_image(&bytes).expect("GIF decodes as a still");
        assert_eq!(still.rgba, animated.frames[0].rgba);
    }

    #[test]
    fn decode_animated_static_png_is_one_frame() {
        // A static PNG is not APNG: one frame, one play.
        let bytes = png_bytes(2, 2, &[9u8; 16]);
        let animated = decode_animated(&bytes).expect("PNG decodes");
        assert_eq!(animated.frames.len(), 1);
        assert_eq!(animated.loop_count, Some(1));
        assert_eq!(animated.frames[0].rgba, vec![9u8; 16]);
    }

    #[test]
    fn resize_rgba_lands_on_exact_size() {
        // A solid 4x4 red image resized to 16x16 stays solid red
        // at exactly 16x16 (favicon downscale shape).
        let image = solid(4, [255, 0, 0, 255]);
        let resized = resize_rgba(&image, 16, 16);
        assert_eq!((resized.width, resized.height), (16, 16));
        assert!(resized.rgba.iter().all(|&b| b == 255 || b == 0));
        assert_eq!(resized.rgba[0..4], [255, 0, 0, 255]);
        // Quadrant colours average correctly on a downscale to
        // 1x1 (the mean of the four quadrants).
        let rgba: Vec<u8> = (0..16)
            .flat_map(|i| {
                let (x, y) = (i % 4, i / 4);
                match (x < 2, y < 2) {
                    (true, true) => [255, 0, 0, 255],
                    (false, true) => [0, 255, 0, 255],
                    (true, false) => [0, 0, 255, 255],
                    (false, false) => [255, 255, 255, 255],
                }
            })
            .collect();
        let quadrants = DecodedImage {
            width: 4,
            height: 4,
            rgba,
        };
        let one = resize_rgba(&quadrants, 1, 1);
        assert_eq!((one.width, one.height), (1, 1));
        assert_eq!(one.rgba, [127, 127, 127, 255]);
        // Same size is a clone, not a resample.
        assert_eq!(resize_rgba(&image, 4, 4), image);
    }
}
