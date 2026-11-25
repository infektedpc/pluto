//! The worker's image pipeline (Stage 11 Rec 1, 7, 11, 12).
//!
//! The load task only *reports* image URLs (see [`crate::page::LoadSinks`]);
//! everything below runs on the tab thread:
//!
//! * [`ImageStore::request`] dedups by URL (one fetch per URL per
//!   page; a `Ready` entry whose pixels were evicted from the cache
//!   re-fetches).
//! * [`ImageStore::spawn_fetch`] fetches through the worker's
//!   loader — `Destination::Image`, so the Stage 9 gates (mixed
//!   content, CSP, blocker) and the Stage 10 HTTP cache all apply
//!   exactly as for any other subresource — then decodes on the
//!   shared rayon pool, never on the tab thread and never on the
//!   UI thread. Decode is `Send`-safe; font state never leaves
//!   the layout thread.
//! * [`ImageStore::apply_outcome`] inserts the decoded frames into
//!   the [`ImageCache`] (which enforces the Rec 12 limits: 8192 px
//!   strict caps, 64 MiB cache cap) and marks the store dirty, so
//!   the next tick rebuilds the layout with the real intrinsic
//!   sizes. A failed / oversize / broken image marks the URL
//!   `Broken`: layout renders alt text or a zero-size box, never
//!   a failed navigation.
//! * [`ImageStore::advance_animations`] drives GIF/APNG playback on
//!   the tab-thread timer: per-frame delays and loop counts are
//!   honoured, the rate is capped at ~30 fps, and off-screen
//!   animations are paused (the tab thread supplies the visible
//!   set from a `Geometry` job — see `renderer.rs`).
//!
//! IPC shape (the decision from the Stage 11 report §5, made on day
//! one and documented in `pluto-ipc`): decode worker-side, deliver
//! each resource out-of-band in its own [`RendererToBrowser::ResourceReady`]
//! frame — one resource per frame, so a single resource always fits
//! the 64 MiB frame cap — and send `SubmitDisplayList` with an
//! empty resource table. The shell accumulates the per-tab table
//! and validates the list against it. [`ImageStore::resource_fits_frame`]
//! refuses a resource whose encoded frame would exceed the cap
//! (oversize → broken-image placeholder, never a dropped frame).

use pluto_html::{Dom, NodeId, select_img_url_from_dom};
use pluto_ipc::transport::MAX_FRAME_BYTES;
use pluto_ipc::{CredentialsMode, Destination, Priority, RequestMode, ResourceRequest, origin_of};
use pluto_layout::{ImageInfo, ImageMap};
use pluto_net::ResourceLoader;
use pluto_paint::{
    AnimatedFrames, ImageCache, Resource, ResourceId, decode_animated, resize_rgba,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

/// One image the load (preload scan or parser) reported. The tab
/// thread owns the fetch + decode; the load task only sends this.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    /// Absolute URL of the selected candidate (Rec 10: exactly
    /// one candidate per `<img>`, never every srcset entry).
    pub url: String,
    /// The document's final URL (after redirects): the fetch's
    /// origin / top-level site resolve against it. Carried here
    /// because the tab thread does not know the final URL until
    /// the load commits, and image fetches start during the
    /// streaming parse.
    pub base_url: String,
    /// A srcset / `<picture>` candidate: the mixed-content gate
    /// BLOCKS (never auto-upgrades) it on a secure page.
    pub is_imageset: bool,
    /// The element's `integrity` attribute, if any (Stage 9
    /// Rec 14, verified over the decoded bytes in `pluto-net`).
    pub integrity: Option<String>,
}

/// The outcome of one image fetch + decode, sent back to the tab
/// thread. `epoch` is the navigation epoch the fetch started
/// under; the tab thread drops outcomes from a stale epoch (a
/// newer navigation superseded the page that wanted the image).
#[derive(Debug)]
pub enum ImageOutcome {
    Decoded {
        url: String,
        epoch: u64,
        frames: AnimatedFrames,
    },
    Failed {
        url: String,
        epoch: u64,
    },
}

/// What the tab thread learns from applying an outcome: the
/// resource id the display list references, plus the intrinsic
/// size layout needs. `animated` means the cache holds more
/// frames behind this id (`advance_animated` swaps them in).
#[derive(Debug, Clone)]
pub struct AppliedImage {
    pub url: String,
    pub id: ResourceId,
    pub width: u32,
    pub height: u32,
    pub animated: bool,
}

/// Per-URL fetch state, so the same URL is never fetched twice
/// for one page.
#[derive(Debug, Clone)]
enum ImageState {
    Loading,
    Ready { id: ResourceId, width: u32, height: u32 },
    Broken,
}

/// One animated image's playback state, keyed by its cache id in
/// [`ImageStore::animated`]. The frames themselves live in the
/// [`ImageCache`] (inserted once, swapped in place by
/// `advance_animated` under the same id); this only tracks timing.
#[derive(Debug, Clone)]
struct AnimatedState {
    /// Per-frame delays in ms (clamped to ≥ 10 ms at decode).
    delays_ms: Vec<u32>,
    /// `None` = loop forever; `Some(n)` = play `n` times after
    /// the first pass, then rest on frame 0.
    loop_count: Option<u32>,
    /// Full passes completed (a pass ends when the frame wraps
    /// back to 0).
    loops_done: u32,
    /// When the current frame's delay expires.
    next_due: Instant,
}

/// Favicons are downscaled to this size before they cross IPC
/// (Rec 7): the tab strip shows them at 16 px, and a 512 px
/// source icon must not cost a 1 MB frame.
pub const FAVICON_SIZE: u32 = 32;

/// Margin subtracted from the IPC frame cap when deciding whether
/// a resource fits in one `ResourceReady` frame (Rec 1): the
/// postcard envelope (variant tag, epoch, id, length prefixes)
/// is a few dozen bytes; 4 KiB is generous headroom.
pub const FRAME_OVERHEAD_BYTES: usize = 4096;

/// Request ids for image fetches (own counter — the scheduler
/// dedups by URL + destination + credentials + top-level site,
/// not by id, so this only needs to be unique per process).
static NEXT_IMAGE_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Build the fetch request for one image (same shape as the
/// preload scanner's fetches in `page.rs`, so the Stage 5
/// scheduler dedups scanner-discovered and tab-thread fetches
/// of one URL into a single download).
pub fn image_resource_request(
    url: &str,
    is_imageset: bool,
    integrity: Option<String>,
    base_url: &str,
    tab_id: u64,
    epoch: u64,
) -> ResourceRequest {
    let origin = origin_of(base_url);
    ResourceRequest {
        request_id: NEXT_IMAGE_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
        tab_id,
        epoch,
        url: url.to_string(),
        method: "GET".to_string(),
        headers: Vec::new(),
        is_imageset,
        destination: Destination::Image,
        priority: Priority::Medium,
        mode: RequestMode::NoCors,
        credentials: CredentialsMode::SameOrigin,
        origin: origin.clone(),
        top_level_site: origin,
        upgrade_insecure_requests: false,
        integrity,
        body: None,
    }
}

/// The tab thread's image store: decoded pixels + per-URL state
/// + animation timers + per-origin favicon bookkeeping.
pub struct ImageStore {
    /// Decoded pixels, LRU-capped at 64 MiB (Rec 12).
    pub cache: ImageCache,
    by_url: HashMap<String, ImageState>,
    /// Animated images, keyed by cache id.
    animated: HashMap<ResourceId, AnimatedState>,
    /// Origins whose favicon was already attempted this session
    /// (Rec 7: a missing icon is not re-requested every
    /// navigation).
    pub favicon_attempted: HashSet<String>,
    /// Origins whose favicon fetch failed / decoded broken.
    pub favicon_missing: HashSet<String>,
    /// Set when an outcome landed since the last layout pass:
    /// the tab thread rebuilds (full, coalesced to one per
    /// tick) so the new intrinsic sizes reach the tree.
    pub images_dirty: bool,
}

impl ImageStore {
    pub fn new() -> Self {
        Self {
            cache: ImageCache::new(),
            by_url: HashMap::new(),
            animated: HashMap::new(),
            favicon_attempted: HashSet::new(),
            favicon_missing: HashSet::new(),
            images_dirty: false,
        }
    }

    /// Record a reported image URL. Returns `true` when the tab
    /// thread should spawn a fetch task for it: a fresh URL, or
    /// one whose pixels were evicted from the cache since it
    /// last decoded. `false` for in-flight / ready / broken.
    pub fn request(&mut self, req: &ImageRequest) -> bool {
        match self.by_url.get(&req.url) {
            Some(ImageState::Loading) | Some(ImageState::Broken) => false,
            Some(ImageState::Ready { id, .. }) => {
                if self.cache.get(*id).is_some() {
                    false
                } else {
                    // Evicted since it decoded: re-fetch.
                    self.by_url.insert(req.url.clone(), ImageState::Loading);
                    true
                }
            }
            None => {
                self.by_url.insert(req.url.clone(), ImageState::Loading);
                true
            }
        }
    }

    /// Intrinsic size of an already-decoded URL, if its pixels
    /// are still cached. The tab thread uses this when JS asks
    /// for an image that is already `Ready`: the `load` event
    /// fires straight away with the cached size instead of
    /// waiting for a fetch that will never come.
    pub fn ready_size(&self, url: &str) -> Option<(u32, u32)> {
        match self.by_url.get(url) {
            Some(ImageState::Ready { id, width, height }) if self.cache.get(*id).is_some() => {
                Some((*width, *height))
            }
            _ => None,
        }
    }

    /// True when this URL already failed (fetch, decode, or
    /// cache insert): a JS request for it reports `error`
    /// straight away instead of re-fetching.
    pub fn is_broken(&self, url: &str) -> bool {
        matches!(self.by_url.get(url), Some(ImageState::Broken))
    }

    /// Fetch + decode one image on detached tasks (call only
    /// after [`request`](Self::request) returned `true`). The
    /// fetch runs through the loader (all Stage 9 gates apply);
    /// the decode runs on the shared rayon pool via a oneshot
    /// bridge (the fetch task awaits the decode without blocking
    /// the tab thread). The outcome carries the navigation epoch
    /// so a stale fetch is dropped, never applied.
    pub fn spawn_fetch(
        fetch: crate::broker::FetchSource,
        req: ImageRequest,
        tab_id: u64,
        epoch: u64,
        cancel: CancellationToken,
        outcome_tx: UnboundedSender<ImageOutcome>,
    ) {
        let request = image_resource_request(
            &req.url,
            req.is_imageset,
            req.integrity.clone(),
            &req.base_url,
            tab_id,
            epoch,
        );
        let url = req.url.clone();
        tokio::task::spawn(async move {
            let outcome = match fetch.fetch_with(request, cancel).await {
                Ok(response) => {
                    let bytes = response.body;
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    rayon::spawn(move || {
                        let _ = tx.send(decode_animated(&bytes));
                    });
                    match rx.await {
                        Ok(Ok(frames)) => ImageOutcome::Decoded { url, epoch, frames },
                        // Awaiting the decode can only fail if the
                        // rayon task panicked (decode itself never
                        // panics — limits reject before allocating);
                        // a decode error is a broken image either way.
                        _ => ImageOutcome::Failed { url, epoch },
                    }
                }
                Err(_) => ImageOutcome::Failed { url, epoch },
            };
            let _ = outcome_tx.send(outcome);
        });
    }

    /// Apply a fetch outcome: insert the decoded frames into the
    /// cache and mark the URL Ready (or Broken on any failure).
    /// `is_favicon` downscales the frame to [`FAVICON_SIZE`]
    /// before inserting (Rec 7). Returns the applied image, or
    /// `None` when the URL is Broken (caller: render alt text /
    /// placeholder, never fail the navigation). Sets
    /// `images_dirty` on any state change of a *page* image; a
    /// favicon never appears in the page's display list, so its
    /// arrival or failure must not trigger a re-layout (a failed
    /// `/favicon.ico` fetch otherwise re-submits the whole page
    /// on the next tick — visible as a spurious extra display
    /// list in the event stream).
    pub fn apply_outcome(
        &mut self,
        outcome: ImageOutcome,
        is_favicon: bool,
    ) -> Option<AppliedImage> {
        match outcome {
            ImageOutcome::Decoded { url, frames, .. } => {
                let mut frames = frames;
                if is_favicon
                    && let Some(first) = frames.frames.first()
                    && (first.width > FAVICON_SIZE || first.height > FAVICON_SIZE)
                {
                    let small = resize_rgba(first, FAVICON_SIZE, FAVICON_SIZE);
                    frames.frames[0] = small;
                }
                let animated = frames.frames.len() > 1;
                let id = if animated {
                    self.cache
                        .insert_animated(frames.frames, frames.delays_ms.clone())
                } else {
                    let first = frames.frames.into_iter().next()?;
                    match self.cache.insert_decoded(first) {
                        Ok(id) => Some(id),
                        Err(_) => None,
                    }
                };
                let Some(id) = id else {
                    // Over the cache cap (or uninsertable): broken.
                    self.by_url.insert(url, ImageState::Broken);
                    if !is_favicon {
                        self.images_dirty = true;
                    }
                    return None;
                };
                let first = self.cache.get(id)?;
                let Resource::Image { width, height, .. } = first;
                let (width, height) = (*width, *height);
                if animated {
                    self.animated.insert(
                        id,
                        AnimatedState {
                            delays_ms: frames.delays_ms.clone(),
                            loop_count: frames.loop_count,
                            loops_done: 0,
                            next_due: Instant::now()
                                + Duration::from_millis(
                                    frames.delays_ms.first().copied().unwrap_or(100) as u64,
                                ),
                        },
                    );
                }
                self.by_url.insert(
                    url.clone(),
                    ImageState::Ready { id, width, height },
                );
                if !is_favicon {
                    self.images_dirty = true;
                }
                Some(AppliedImage {
                    url,
                    id,
                    width,
                    height,
                    animated,
                })
            }
            ImageOutcome::Failed { url, .. } => {
                self.by_url.insert(url, ImageState::Broken);
                if !is_favicon {
                    self.images_dirty = true;
                }
                None
            }
        }
    }

    /// A new navigation started: animations belong to the old
    /// page's nodes (drop them), in-flight fetches are stale
    /// (their outcomes carry the old epoch and are dropped on
    /// arrival — clear `Loading` so the new page re-requests),
    /// and a previously-broken URL may succeed now (clear
    /// `Broken`). `Ready` entries whose pixels are still cached
    /// survive: a logo shared by two pages decodes once.
    /// Favicon attempt/miss sets are per-origin per-session and
    /// survive (Rec 7).
    pub fn note_navigation(&mut self) {
        self.animated.clear();
        let stale: Vec<String> = self
            .by_url
            .iter()
            .filter(|(_, state)| {
                !matches!(state, ImageState::Ready { id, .. } if self.cache.get(*id).is_some())
            })
            .map(|(url, _)| url.clone())
            .collect();
        for url in stale {
            self.by_url.remove(&url);
        }
        self.images_dirty = false;
    }

    /// Advance the due + visible animations (Rec 11): each
    /// advance swaps the next frame into the cache under the
    /// same id, so the tab thread re-sends that id in a
    /// `ResourceReady` and re-submits the cached display list —
    /// no layout is involved. Returns the advanced ids. Loop
    /// counts are honoured: a finished animation is removed
    /// from the timer and rests on frame 0. The rate is capped
    /// by the decoded delays (each ≥ 10 ms, so ≤ 100 fps in
    /// theory) — and by the tick that calls this (every ~16 ms),
    /// so ~30 fps in practice; an animation never advances more
    /// than one frame per call, so a slow tick cannot fast-
    /// forward through frames.
    pub fn advance_animations(
        &mut self,
        now: Instant,
        visible: &HashSet<ResourceId>,
    ) -> Vec<ResourceId> {
        let mut advanced = Vec::new();
        let mut finished = Vec::new();
        for (&id, anim) in self.animated.iter_mut() {
            if now < anim.next_due || !visible.contains(&id) {
                continue;
            }
            if !self.cache.advance_animated(id) {
                continue;
            }
            advanced.push(id);
            let frame = self.cache.animated_frame(id).unwrap_or(0);
            if frame == 0 {
                anim.loops_done += 1;
                if let Some(count) = anim.loop_count
                    && anim.loops_done > count
                {
                    finished.push(id);
                    continue;
                }
            }
            let delay = anim.delays_ms.get(frame).copied().unwrap_or(100);
            anim.next_due = now + Duration::from_millis(delay as u64);
        }
        for id in finished {
            self.animated.remove(&id);
        }
        advanced
    }

    /// Build the [`ImageMap`] for one layout pass from the
    /// current document: `<img>` nodes resolve their URL via
    /// [`select_img_url_from_dom`] (srcset / `<picture>` selection
    /// at the real viewport) and map to their decoded pixels;
    /// background-image URLs come from the merged sheet, keyed by
    /// the raw CSS url string (layout looks them up by that key).
    /// Unknown URLs (not yet decoded) contribute nothing — the
    /// image lands on a later pass, when `images_dirty` rebuilds.
    pub fn build_image_map(
        &self,
        dom: &Dom,
        sheet: &pluto_css::Stylesheet,
        base_url: &str,
        viewport_width: f32,
        dpr: f32,
    ) -> ImageMap {
        let mut map = ImageMap::new();
        if let Some(root) = dom.root() {
            self.collect_img_nodes(dom, root, base_url, viewport_width, dpr, &mut map);
        }
        for raw in sheet.background_image_urls() {
            let resolved = pluto_html::resolve_url(base_url, &raw);
            if let Some(ImageState::Ready { id, width, height }) = self.by_url.get(&resolved) {
                map.backgrounds.insert(
                    raw,
                    ImageInfo {
                        resource: id.0,
                        width: *width,
                        height: *height,
                    },
                );
            }
        }
        map
    }

    fn collect_img_nodes(
        &self,
        dom: &Dom,
        id: NodeId,
        base_url: &str,
        viewport_width: f32,
        dpr: f32,
        map: &mut ImageMap,
    ) {
        if dom.tag(id) == Some("img") {
            match select_img_url_from_dom(dom, id, viewport_width, dpr) {
                Some((url, _)) => {
                    let resolved = pluto_html::resolve_url(base_url, &url);
                    match self.by_url.get(&resolved) {
                        Some(ImageState::Ready { id: rid, width, height }) => {
                            map.by_node.insert(
                                id,
                                ImageInfo {
                                    resource: rid.0,
                                    width: *width,
                                    height: *height,
                                },
                            );
                        }
                        Some(ImageState::Broken) => {
                            map.broken.insert(id);
                        }
                        _ => {}
                    }
                }
                None => {
                    // No src at all: broken with no alt fallback
                    // decision — layout treats a missing entry as
                    // "not yet decoded", so mark it broken here
                    // so a srcless <img> renders its alt text
                    // instead of a 300x150 hole.
                    map.broken.insert(id);
                }
            }
        }
        for &child in dom.children(id) {
            self.collect_img_nodes(dom, child, base_url, viewport_width, dpr, map);
        }
    }

    /// The favicon URL for this document (Rec 7): the first
    /// `<link>` whose `rel` carries an `icon` token (including
    /// `shortcut icon` and `apple-touch-icon`), else the origin's
    /// `/favicon.ico`. Returns `None` only when the document has
    /// no origin to fall back to (about:blank / data:).
    pub fn favicon_url_for(dom: &Dom, base_url: &str) -> Option<String> {
        if let Some(root) = dom.root() {
            let mut found: Option<String> = None;
            let mut apple: Option<String> = None;
            collect_favicon_links(dom, root, &mut found, &mut apple);
            if let Some(href) = found.or(apple) {
                return Some(pluto_html::resolve_url(base_url, &href));
            }
        }
        let origin = origin_of(base_url);
        if origin.is_empty() || origin == "null" {
            return None;
        }
        Some(format!("{origin}/favicon.ico"))
    }

    /// True when `rgba_len` bytes of pixels fit in one
    /// `ResourceReady` frame (Rec 1): the frame cap minus the
    /// postcard envelope. The tab thread refuses a larger
    /// resource instead of sending a frame the transport
    /// would reject (oversize → broken-image placeholder).
    pub fn resource_fits_frame(rgba_len: usize) -> bool {
        rgba_len + FRAME_OVERHEAD_BYTES < MAX_FRAME_BYTES
    }
}

fn collect_favicon_links(
    dom: &Dom,
    id: NodeId,
    found: &mut Option<String>,
    apple: &mut Option<String>,
) {
    if found.is_some() && apple.is_some() {
        return;
    }
    if dom.tag(id) == Some("link")
        && let Some(rel) = dom.attr(id, "rel")
        && let Some(href) = dom.attr(id, "href")
        && !href.trim().is_empty()
    {
        let tokens: Vec<String> = rel
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        if tokens.iter().any(|t| t == "icon" || t == "shortcut") {
            if found.is_none() {
                *found = Some(href.to_string());
            }
        } else if tokens.iter().any(|t| t == "apple-touch-icon") && apple.is_none() {
            *apple = Some(href.to_string());
        }
    }
    for &child in dom.children(id) {
        collect_favicon_links(dom, child, found, apple);
    }
}

/// The document's `<title>` text (Rec 8): the first `<title>`
/// element's text children, concatenated. Empty when the
/// document has no title — the shell then keeps the URL as the
/// tab title.
pub fn document_title(dom: &Dom) -> String {
    fn walk(dom: &Dom, id: NodeId, out: &mut String) -> bool {
        if dom.tag(id) == Some("title") {
            for &child in dom.children(id) {
                if let Some(text) = dom.text(child) {
                    out.push_str(text);
                }
            }
            return true;
        }
        for &child in dom.children(id) {
            if walk(dom, child, out) {
                return true;
            }
        }
        false
    }
    let mut out = String::new();
    if let Some(root) = dom.root() {
        walk(dom, root, &mut out);
    }
    out
}

impl Default for ImageStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder};
    use pluto_html::parse;
    use pluto_paint::{MAX_ANIMATION_FPS, MIN_FRAME_DELAY_MS};

    /// Encode a small RGBA image to real PNG bytes (same helper
    /// shape as pluto-paint's decode tests — no checked-in
    /// fixture, no hand-built CRCs).
    fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(rgba, width, height, ExtendedColorType::Rgba8)
            .expect("test PNG encodes");
        out
    }

    fn png_2x2_red() -> Vec<u8> {
        let px = [255u8, 0, 0, 255];
        let rgba: Vec<u8> = px.repeat(4);
        png_bytes(2, 2, &rgba)
    }

    fn gif_2frames() -> Vec<u8> {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};
        let mut out = Vec::new();
        let mut enc = GifEncoder::new(&mut out);
        enc.set_repeat(Repeat::Infinite).unwrap();
        for color in [Rgba([255, 0, 0, 255]), Rgba([0, 255, 0, 255])] {
            let img = RgbaImage::from_pixel(2, 2, color);
            let frame = Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1));
            enc.encode_frame(frame).unwrap();
        }
        drop(enc);
        out
    }

    #[test]
    fn request_dedups_and_refetches_after_eviction() {
        let mut store = ImageStore::new();
        let req = ImageRequest {
            url: "https://example.com/a.png".to_string(),
            base_url: "https://example.com/page".to_string(),
            is_imageset: false,
            integrity: None,
        };
        assert!(store.request(&req), "first request fetches");
        assert!(!store.request(&req), "in-flight request dedups");
        // Decode the PNG through the real path and apply it.
        let frames = decode_animated(&png_2x2_red()).expect("2x2 PNG decodes");
        let applied = store
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: req.url.clone(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("PNG applies");
        assert_eq!((applied.width, applied.height), (2, 2));
        assert!(!applied.animated);
        assert!(store.images_dirty);
        assert!(!store.request(&req), "ready + cached dedups");
        // Evict the pixels: a fresh small-cap cache loses it.
        store.cache = ImageCache::with_cap(1);
        assert!(store.request(&req), "evicted image re-fetches");
    }

    #[test]
    fn apply_failed_marks_broken() {
        let mut store = ImageStore::new();
        let url = "https://example.com/nope.png".to_string();
        assert!(store.apply_outcome(ImageOutcome::Failed { url: url.clone(), epoch: 1 }, false).is_none());
        let map = store.build_image_map(
            &parse("<img src=\"/nope.png\">"),
            &pluto_css::Stylesheet::default(),
            "https://example.com/",
            1024.0,
            1.0,
        );
        assert_eq!(map.broken.len(), 1, "broken img lands in the map");
    }

    #[test]
    fn build_image_map_selects_srcset_and_backgrounds() {
        let mut store = ImageStore::new();
        let frames = decode_animated(&png_2x2_red()).expect("PNG decodes");
        store
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: "https://example.com/small.png".to_string(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("applies");
        let frames = decode_animated(&png_2x2_red()).expect("PNG decodes");
        store
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: "https://example.com/bg.png".to_string(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("applies");
        let dom = parse(
            "<img src=\"/a.png\" srcset=\"/small.png 1x, /big.png 2x\" alt=\"pic\">",
        );
        let sheet = pluto_css::Stylesheet::parse("div { background-image: url(bg.png); }");
        let map = store.build_image_map(&dom, &sheet, "https://example.com/", 1024.0, 1.0);
        assert_eq!(map.by_node.len(), 1, "one img node mapped");
        assert!(map.backgrounds.contains_key("bg.png"), "raw CSS url keys the background");
        assert!(map.broken.is_empty());
    }

    #[test]
    fn favicon_prefers_link_icon_then_origin_fallback() {
        let dom = parse("<link rel=\"icon\" href=\"/fav.ico\">");
        assert_eq!(
            ImageStore::favicon_url_for(&dom, "https://example.com/page"),
            Some("https://example.com/fav.ico".to_string())
        );
        let dom = parse("<p>no link</p>");
        assert_eq!(
            ImageStore::favicon_url_for(&dom, "https://example.com/page"),
            Some("https://example.com/favicon.ico".to_string())
        );
        let dom = parse("<link rel=\"apple-touch-icon\" href=\"/apple.png\">");
        assert_eq!(
            ImageStore::favicon_url_for(&dom, "https://example.com/page"),
            Some("https://example.com/apple.png".to_string())
        );
    }

    #[test]
    fn document_title_reads_first_title_element() {
        let dom = parse("<title>Hello World</title><p>x</p>");
        assert_eq!(document_title(&dom), "Hello World");
        let dom = parse("<p>no title</p>");
        assert_eq!(document_title(&dom), "");
    }

    #[test]
    fn resource_fits_frame_refuses_a_4k_image_plus_margin() {
        // One 4K RGBA frame is ~33 MB — it fits alone; two would
        // not fit one frame, which is why resources travel one
        // per frame (see the module docs).
        let one_4k = 3840usize * 2160 * 4;
        assert!(ImageStore::resource_fits_frame(one_4k));
        assert!(!ImageStore::resource_fits_frame(MAX_FRAME_BYTES));
        assert!(!ImageStore::resource_fits_frame(
            MAX_FRAME_BYTES - FRAME_OVERHEAD_BYTES
        ));
    }

    #[test]
    fn animated_gif_advances_and_honours_loop_count() {
        // A 2-frame GIF that loops twice, then rests: the store
        // advances it through its frames and stops.
        let mut store = ImageStore::new();
        let mut frames = decode_animated(&gif_2frames()).expect("GIF decodes");
        assert_eq!(frames.frames.len(), 2);
        frames.loop_count = Some(1);
        let applied = store
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: "https://example.com/anim.gif".to_string(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("GIF applies");
        assert!(applied.animated);
        let id = applied.id;
        let mut visible = HashSet::new();
        visible.insert(id);
        // Frame 0's delay is 100 ms: not due yet.
        assert!(store
            .advance_animations(Instant::now(), &visible)
            .is_empty());
        // Due: advances to frame 1.
        let due = Instant::now() + Duration::from_millis(150);
        let advanced = store.advance_animations(due, &visible);
        assert_eq!(advanced, vec![id]);
        assert_eq!(store.cache.animated_frame(id), Some(1));
        // Frame 1 due: wraps to 0 — one loop done (count 1, so
        // loops_done == 1 is not > 1: it keeps playing once more).
        let due2 = due + Duration::from_millis(150);
        assert_eq!(store.advance_animations(due2, &visible), vec![id]);
        assert_eq!(store.cache.animated_frame(id), Some(0));
        // Second wrap: loops_done == 2 > 1 — the final advance
        // still lands (the wrap that ends the last pass), and
        // the animation is removed from the timer after it, so
        // the NEXT tick finds nothing to advance.
        let due3 = due2 + Duration::from_millis(150);
        assert_eq!(store.advance_animations(due3, &visible), vec![id]);
        let due4 = due3 + Duration::from_millis(150);
        assert_eq!(store.advance_animations(due4, &visible), vec![id]);
        let due5 = due4 + Duration::from_millis(150);
        assert!(
            store.advance_animations(due5, &visible).is_empty(),
            "a finished animation stops advancing"
        );
        // Off-screen animations never advance.
        let mut store2 = ImageStore::new();
        let frames = decode_animated(&gif_2frames()).expect("GIF decodes");
        let applied = store2
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: "https://example.com/anim.gif".to_string(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("GIF applies");
        let empty: HashSet<ResourceId> = HashSet::new();
        let _ = applied;
        assert!(store2
            .advance_animations(Instant::now() + Duration::from_secs(10), &empty)
            .is_empty());
    }

    #[test]
    fn note_navigation_keeps_cached_ready_but_drops_loading_and_broken() {
        let mut store = ImageStore::new();
        let frames = decode_animated(&png_2x2_red()).expect("PNG decodes");
        store
            .apply_outcome(
                ImageOutcome::Decoded {
                    url: "https://example.com/kept.png".to_string(),
                    epoch: 1,
                    frames,
                },
                false,
            )
            .expect("applies");
        store.request(&ImageRequest {
            url: "https://example.com/loading.png".to_string(),
            base_url: "https://example.com/page".to_string(),
            is_imageset: false,
            integrity: None,
        });
        store.apply_outcome(
            ImageOutcome::Failed {
                url: "https://example.com/broken.png".to_string(),
                epoch: 1,
            },
            false,
        );
        store.note_navigation();
        assert!(
            !store.request(&ImageRequest {
                url: "https://example.com/kept.png".to_string(),
                base_url: "https://example.com/page".to_string(),
                is_imageset: false,
                integrity: None,
            }),
            "cached ready survives a navigation"
        );
        assert!(
            store.request(&ImageRequest {
                url: "https://example.com/loading.png".to_string(),
                base_url: "https://example.com/page".to_string(),
                is_imageset: false,
                integrity: None,
            }),
            "stale loading re-requests on the new page"
        );
        assert!(
            store.request(&ImageRequest {
                url: "https://example.com/broken.png".to_string(),
                base_url: "https://example.com/page".to_string(),
                is_imageset: false,
                integrity: None,
            }),
            "a broken URL may succeed on the new page"
        );
        assert!(store.animated.is_empty(), "animations belong to the old page");
    }

    #[test]
    fn animation_rate_constants_hold() {
        // The driver never advances faster than the decoded
        // delays allow (≥ 10 ms per frame) and the tab tick
        // (~16 ms) caps the real rate near 30 fps.
        assert!(MIN_FRAME_DELAY_MS >= 10);
        assert_eq!(MAX_ANIMATION_FPS, 30);
    }

    #[test]
    fn image_resource_request_carries_tab_and_epoch() {
        let req = image_resource_request(
            "https://example.com/a.png",
            true,
            None,
            "https://example.com/page",
            7,
            42,
        );
        assert_eq!(req.tab_id, 7);
        assert_eq!(req.epoch, 42);
        assert!(req.is_imageset);
        assert_eq!(req.destination, Destination::Image);
        assert_eq!(req.origin, "https://example.com");
        assert_eq!(req.top_level_site, "https://example.com");
    }
}
