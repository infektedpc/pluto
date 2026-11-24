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
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// The image-cache byte cap (Stage 8 Rec 12, report §g): 64 MiB,
/// matching the existing 64 MiB resource budget
/// ([`crate::MAX_RESOURCE_BYTES`]).
pub const IMAGE_CACHE_BYTE_CAP: usize = 64 * 1024 * 1024;

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
    let mut hash = fnv1a(&image.width.to_le_bytes());
    hash ^= fnv1a(&image.height.to_le_bytes()).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^= fnv1a(&image.rgba);
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
}

/// Decode one image's bytes (PNG, JPEG, GIF, WebP — the formats the
/// workspace `image` dependency enables) to RGBA8. Pure and
/// thread-safe: call it on the rayon pool, never on the UI thread.
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    let dynamic = image::load_from_memory(bytes)?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Empty);
    }
    let raw = rgba.into_raw();
    if raw.len() != (width as usize) * (height as usize) * 4 {
        return Err(ImageError::LengthMismatch(raw.len(), width, height));
    }
    Ok(DecodedImage {
        width,
        height,
        rgba: raw,
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
                        for c in 0..4 {
                            sum[c] += image.rgba[off + c] as u32;
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

/// The full mip chain for an image: level 0 is the image itself,
/// each further level is a [`downscale_rgba`] of the previous one,
/// down to 1×1. Length is [`mip_level_count_for`]. The renderer
/// uploads every level, so a draw at any scale samples an uploaded
/// level instead of resampling level 0.
pub fn build_mip_chain(image: &DecodedImage) -> Vec<DecodedImage> {
    let mut chain = vec![image.clone()];
    while chain
        .last()
        .map_or(false, |last| last.width > 1 || last.height > 1)
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
/// `cap_bytes` except for a single image larger than the cap, which
/// is stored alone (evicting everything else) rather than refused —
/// the display list already caps a single resource at 64 MiB, so
/// this only bites under a test-sized cap. `get` does not touch
/// recency (the table is read without the cache); [`get_touch`]
/// does, for callers that count a read as use.
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
    /// decoding again (decode-once, report §g).
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
        Ok(self.insert_decoded_with_source(decoded, Some(hash)))
    }

    /// Store an already-decoded image, returning its id. Identical
    /// pixels (same size and bytes) return the existing id.
    pub fn insert_decoded(&mut self, image: DecodedImage) -> ResourceId {
        self.insert_decoded_with_source(image, None)
    }

    fn insert_decoded_with_source(
        &mut self,
        image: DecodedImage,
        source_hash: Option<u64>,
    ) -> ResourceId {
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
            return id;
        }
        let size = image.byte_len();
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
        id
    }

    /// Store an animated image: frame 0 lands in the table under
    /// the returned id, the rest are held for
    /// [`advance_animated`]. `None` when there are no frames or
    /// the frames differ in size (see [`AnimatedImage::new`]).
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
        if self.table.get(id).is_none() {
            return None;
        }
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
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        if let Some(Resource::Image { rgba, .. }) = self.table.resources.remove(&id) {
            self.bytes_used = self.bytes_used.saturating_sub(rgba.len());
        }
        if let Some(hash) = self.pixel_hash.remove(&id) {
            if self.by_pixels.get(&hash) == Some(&id) {
                self.by_pixels.remove(&hash);
            }
        }
        if let Some(hash) = self.source_hash.remove(&id) {
            if self.by_source.get(&hash) == Some(&id) {
                self.by_source.remove(&hash);
            }
        }
        self.animated.remove(&id);
        self.frame_dirty.remove(&id);
        self.evictions += 1;
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
        let a = cache.insert_decoded(solid(4, [255, 0, 0, 255]));
        let b = cache.insert_decoded(solid(4, [0, 255, 0, 255]));
        assert_eq!(cache.bytes_used(), 128);
        assert_eq!(cache.evictions(), 0);
        // A third image evicts the least recently used (a).
        let c = cache.insert_decoded(solid(4, [0, 0, 255, 255]));
        assert_eq!(cache.evictions(), 1);
        assert!(cache.get(a).is_none(), "LRU image a is evicted");
        assert!(cache.get(b).is_some());
        assert!(cache.get(c).is_some());
        assert_eq!(cache.bytes_used(), 128, "cap holds after eviction");
        assert_eq!(cache.len(), 2);
        // Touching b makes c the LRU; the next insert evicts c, not b.
        cache.get_touch(b).expect("b is cached");
        let d = cache.insert_decoded(solid(4, [255, 255, 0, 255]));
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
        let id3 = cache.insert_decoded(DecodedImage {
            width: 1,
            height: 1,
            rgba: vec![255, 0, 0, 255],
        });
        assert_eq!(id3, id1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.dedup_hits(), 2);
        // Different pixels are a different image.
        let id4 = cache.insert_decoded(DecodedImage {
            width: 1,
            height: 1,
            rgba: vec![0, 255, 0, 255],
        });
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
            .map(|i| {
                let (x, y) = (i % 4, i / 4);
                match (x < 2, y < 2) {
                    (true, true) => [255, 0, 0, 255],
                    (false, true) => [0, 255, 0, 255],
                    (true, false) => [0, 0, 255, 255],
                    (false, false) => [255, 255, 255, 255],
                }
            })
            .flatten()
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
        let still = cache.insert_decoded(blue.clone());
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
}
