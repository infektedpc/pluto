//! HTML5 2D canvas backing store (Stage 15, M7 Phase 3, Rec 8/9).
//!
//! [`Canvas2d`] is the worker-side canvas bitmap + context state:
//! the JS bindings (pluto-js) talk to it through the [`DomHost`]
//! canvas call (plain data only — pluto-js never imports this
//! crate), the production host (`pluto-browser`) owns one per
//! `<canvas>` element, and the tab thread drains dirty bitmaps
//! into `ResourceReady` + `Blob` (the same producer path Phase 2
//! built for rasterised SVG — see [`crate::svg`]).
//!
//! Raster resolution is fixed: the bitmap is the canvas's
//! `width` × `height` attributes (300×150 default, honoured by
//! layout since Phase 1). Setting `width` / `height` from JS
//! resets the bitmap and the whole context state, per spec.
//! Updates are full-frame only, and a bitmap is re-sent only
//! when its `dirty` flag is set (a static canvas costs one
//! `ResourceReady` frame, not one per tick — the 64 MiB /
//! ~498 MB/s-at-1080p60 ceiling from the research report).
//!
//! Drawing reuses Phase 2's machinery: lyon tessellation
//! ([`crate::path`]) for fills/strokes and the CPU rasteriser
//! ([`crate::raster`]) for the bitmap. Gradient paints evaluate
//! per pixel through a coverage mask (the radial-gradient lesson
//! from `svg.rs` — per-vertex colours cannot express a radial
//! gradient). Text is shaped with cosmic-text and rasterised
//! with `SwashCache` (the same pair the renderer uses for glyphs).
//!
//! Tainting (Rec 9): [`Canvas2d::origin_clean`] starts `true`.
//! Drawing a cross-origin image without CORS approval taints the
//! canvas (the host decides what counts as cross-origin — see
//! [`crate::svg`] / `pluto-net::cors::tainting_for` for the fetch
//! model this extends); a tainted canvas refuses pixel readback
//! ([`Canvas2d::get_image_data`], [`Canvas2d::to_data_url`]
//! return [`CanvasError::Tainted`]). `putImageData` writes pixels
//! and exposes nothing, so — per spec — it is allowed on a
//! tainted canvas. A canvas drawn from another tainted canvas
//! inherits the taint.
//!
//! Documented approximations: `arc` flattens to line segments;
//! `fillText`/`strokeText` use the system font (the `font`
//! string's size is honoured, its family is not — no webfont
//! loading for canvas text in this phase); non-uniform CTM
//! scales apply the x-scale to glyph size; `globalAlpha` /
//! `globalCompositeOperation` / shadows / `setLineDash` /
//! `arcTo` / `ellipse` / `measureText` are not in this phase
//! (the JS binding does not expose them, so nothing looks real
//! while doing nothing).

use crate::path::{
    FillRule, SvgPath, VectorVertex, parse_line_cap, parse_line_join, path_from_rect,
    tessellate_fill, tessellate_stroke,
};
use crate::raster::{rasterize_coverage, rasterize_triangles_into};
use crate::svg::{MAX_RASTER_PIXELS, affine_inverse};
use crate::{DecodedImage, ResourceId};
use pluto_css::{Affine, Value, parse_value};
use pluto_html::NodeId as DomId;
use slotmap::Key as _;

/// Resource-id base for canvas bitmaps. Deterministic per
/// element (`BASE | slot-index`), like [`crate::svg::svg_blob_id`]
/// — a scroll rebuild names the id the tab thread already holds.
pub const CANVAS_BLOB_ID_BASE: u32 = 0x6000_0000;

/// Deterministic blob id for a `<canvas>` element's bitmap.
pub fn canvas_blob_id(id: DomId) -> ResourceId {
    ResourceId(CANVAS_BLOB_ID_BASE | (id.data().as_ffi() as u32 & 0x00FF_FFFF))
}

/// Resource-id base for video frames (Stage 16, M8 Phase 2,
/// Rec 5). Beside the canvas base: a video frame Blob is named
/// `BASE | slot-index` for its element, so a scroll rebuild
/// names the id the shell already holds (it re-uploads the
/// latest frame under the same id).
pub const VIDEO_BLOB_ID_BASE: u32 = 0x7000_0000;

/// Deterministic blob id for a `<video>` element's current frame.
pub fn video_blob_id(id: DomId) -> ResourceId {
    ResourceId(VIDEO_BLOB_ID_BASE | (id.data().as_ffi() as u32 & 0x00FF_FFFF))
}

/// True when `id` names a video-frame Blob (Stage 16, M8
/// Phase 2, Rec 5): the frame cache evicts these first.
pub fn is_video_blob_id(id: ResourceId) -> bool {
    (id.0 & 0xFF00_0000) == VIDEO_BLOB_ID_BASE
}

/// What a canvas operation can fail with. `Tainted` is the Rec 9
/// readback refusal (surfaced to JS as a `SecurityError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanvasError {
    /// The canvas is tainted: pixel readback is refused.
    Tainted,
    /// The requested bitmap would exceed [`MAX_RASTER_PIXELS`].
    TooLarge,
    /// The canvas has a zero width or height (no bitmap).
    Empty,
}

/// Parse a CSS colour spelling (`#rgb`, `#rrggbb`, `red`,
/// `rgb(…)` / `rgba(…)`) as used for `fillStyle` /
/// `strokeStyle`. Only the spellings the style layer already
/// understands are supported (hex + named colours +
/// `rgb()`/`rgba()` via [`pluto_css::parse_value`]).
pub fn parse_canvas_color(value: &str) -> Option<[u8; 4]> {
    match parse_value(value) {
        Value::Color(r, g, b) => Some([r, g, b, 255]),
        Value::ColorA(r, g, b, a) => Some([r, g, b, (a.clamp(0.0, 1.0) * 255.0) as u8]),
        _ => None,
    }
}

/// Format a colour as the `fillStyle` getter returns it: `#rrggbb`
/// for opaque colours, `rgba(r, g, b, a)` otherwise.
pub fn format_canvas_color(color: [u8; 4]) -> String {
    if color[3] == 255 {
        format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2])
    } else {
        format!(
            "rgba({}, {}, {}, {})",
            color[0],
            color[1],
            color[2],
            color[3] as f32 / 255.0
        )
    }
}

/// A canvas gradient (from `createLinearGradient` /
/// `createRadialGradient`): geometry in user space + stops.
/// `createRadialGradient` has two circles (start + end); a
/// linear gradient is the segment between its two points.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasGradientKind {
    Linear {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    },
    Radial {
        x1: f32,
        y1: f32,
        r1: f32,
        x2: f32,
        y2: f32,
        r2: f32,
    },
}

/// One canvas gradient: kind + stops (`offset` in 0..=1, colour
/// straight-alpha RGBA bytes). Stops are kept sorted by offset
/// (added out of order, they sort on insert — the spec sorts
/// them by offset at paint time).
#[derive(Debug, Clone, PartialEq)]
pub struct CanvasGradient {
    pub kind: CanvasGradientKind,
    pub stops: Vec<(f32, [u8; 4])>,
}

impl CanvasGradient {
    pub fn new(kind: CanvasGradientKind) -> Self {
        Self {
            kind,
            stops: Vec::new(),
        }
    }

    /// Add a stop (clamped to 0..=1; stops stay sorted).
    pub fn add_stop(&mut self, offset: f32, color: [u8; 4]) {
        let offset = offset.clamp(0.0, 1.0);
        self.stops.push((offset, color));
        self.stops
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// The colour at user-space point `p` (straight-alpha floats).
    /// Linear: projection onto the segment. Radial: solve for the
    /// `t` whose interpolated circle passes through `p`
    /// (`|p - c(t)| = r(t)`); outside 0..=1 the ends clamp.
    pub fn color_at(&self, p: [f32; 2]) -> [f32; 4] {
        if self.stops.is_empty() {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let t = match self.kind {
            CanvasGradientKind::Linear { x1, y1, x2, y2 } => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let len2 = dx * dx + dy * dy;
                if len2 <= 0.0 {
                    0.0
                } else {
                    ((p[0] - x1) * dx + (p[1] - y1) * dy) / len2
                }
            }
            CanvasGradientKind::Radial {
                x1,
                y1,
                r1,
                x2,
                y2,
                r2,
            } => {
                // |p - (c1 + t (c2 - c1))| = r1 + t (r2 - r1):
                // a quadratic in t. The root in [0, 1] is the
                // parameter; a degenerate (concentric, equal radii)
                // gradient is t = 0.
                let dcx = x2 - x1;
                let dcy = y2 - y1;
                let dr = r2 - r1;
                let px = p[0] - x1;
                let py = p[1] - y1;
                let a = dcx * dcx + dcy * dcy - dr * dr;
                let b = -2.0 * (px * dcx + py * dcy) - 2.0 * r1 * dr;
                let c = px * px + py * py - r1 * r1;
                if a.abs() < f32::EPSILON {
                    if b.abs() < f32::EPSILON {
                        0.0
                    } else {
                        (-c / b).clamp(0.0, 1.0)
                    }
                } else {
                    let disc = b * b - 4.0 * a * c;
                    if disc < 0.0 {
                        0.0
                    } else {
                        let sq = disc.sqrt();
                        let t1 = (-b - sq) / (2.0 * a);
                        let t2 = (-b + sq) / (2.0 * a);
                        if (0.0..=1.0).contains(&t1) {
                            t1
                        } else if (0.0..=1.0).contains(&t2) {
                            t2
                        } else {
                            t1.clamp(0.0, 1.0)
                        }
                    }
                }
            }
        };
        let t = t.clamp(0.0, 1.0);
        let at = |(_, c): (f32, [u8; 4])| -> [f32; 4] {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
                c[3] as f32 / 255.0,
            ]
        };
        if self.stops.len() == 1 || t <= self.stops[0].0 {
            return at(self.stops[0]);
        }
        let last = self.stops.len() - 1;
        if t >= self.stops[last].0 {
            return at(self.stops[last]);
        }
        for w in self.stops.windows(2) {
            let (a, b) = (w[0], w[1]);
            if t >= a.0 && t <= b.0 {
                let span = (b.0 - a.0).max(f32::EPSILON);
                let f = (t - a.0) / span;
                let ca = at(a);
                let cb = at(b);
                return [
                    ca[0] + (cb[0] - ca[0]) * f,
                    ca[1] + (cb[1] - ca[1]) * f,
                    ca[2] + (cb[2] - ca[2]) * f,
                    ca[3] + (cb[3] - ca[3]) * f,
                ];
            }
        }
        at(self.stops[last])
    }
}

/// A canvas paint: a solid colour or a gradient.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasPaint {
    Solid([u8; 4]),
    Gradient(CanvasGradient),
}

impl CanvasPaint {
    /// The colour at user-space point `p` (straight-alpha floats).
    pub fn color_at(&self, p: [f32; 2]) -> [f32; 4] {
        match self {
            CanvasPaint::Solid(c) => [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
                c[3] as f32 / 255.0,
            ],
            CanvasPaint::Gradient(g) => g.color_at(p),
        }
    }

    pub fn is_gradient(&self) -> bool {
        matches!(self, CanvasPaint::Gradient(_))
    }
}

impl Default for CanvasPaint {
    fn default() -> Self {
        CanvasPaint::Solid([0, 0, 0, 255])
    }
}

/// One saved context state (`save()` / `restore()`): everything
/// `save()` snapshots per spec (transform, paints, line style,
/// font, text alignment, clip mask). The path is NOT saved.
#[derive(Debug, Clone)]
struct SavedState {
    transform: Affine,
    fill: CanvasPaint,
    stroke: CanvasPaint,
    line_width: f32,
    line_cap: lyon::tessellation::LineCap,
    line_join: lyon::tessellation::LineJoin,
    miter_limit: f32,
    clip_mask: Option<Vec<u8>>,
    font: String,
    font_size: f32,
    text_align: String,
    text_baseline: String,
}

/// The 2D canvas: bitmap + context state. All coordinates the
/// methods take are user-space (the CTM maps them to device
/// pixels); the bitmap is `width` × `height` device pixels.
pub struct Canvas2d {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    origin_clean: bool,
    dirty: bool,
    transform: Affine,
    fill: CanvasPaint,
    stroke: CanvasPaint,
    line_width: f32,
    line_cap: lyon::tessellation::LineCap,
    line_join: lyon::tessellation::LineJoin,
    miter_limit: f32,
    path: SvgPath,
    /// Current point of the path (`None` = no subpath yet).
    current: Option<[f32; 2]>,
    /// Start point of the current subpath (for `closePath`).
    subpath_start: Option<[f32; 2]>,
    /// Clip mask in device pixels (`None` = no clip). A mask is
    /// the intersection of every `clip()` so far, in device
    /// space (the CTM at `clip()` time is baked in).
    clip_mask: Option<Vec<u8>>,
    saved: Vec<SavedState>,
    font: String,
    font_size: f32,
    text_align: String,
    text_baseline: String,
}

impl Canvas2d {
    /// A fresh canvas: transparent-black bitmap, origin-clean,
    /// default context state (spec defaults: black fill, black
    /// stroke, lineWidth 1, butt/miter, `10px sans-serif`).
    /// Returns `None` when the bitmap would exceed
    /// [`MAX_RASTER_PIXELS`] (the caller keeps the old canvas).
    pub fn new(width: u32, height: u32) -> Option<Self> {
        if (width as usize).saturating_mul(height as usize)
            > MAX_RASTER_PIXELS
        {
            return None;
        }
        Some(Self {
            width,
            height,
            rgba: vec![0u8; (width as usize) * (height as usize) * 4],
            origin_clean: true,
            dirty: false,
            transform: Affine::identity(),
            fill: CanvasPaint::Solid([0, 0, 0, 255]),
            stroke: CanvasPaint::Solid([0, 0, 0, 255]),
            line_width: 1.0,
            line_cap: lyon::tessellation::LineCap::Butt,
            line_join: lyon::tessellation::LineJoin::Miter,
            miter_limit: 10.0,
            path: SvgPath::new(),
            current: None,
            subpath_start: None,
            clip_mask: None,
            saved: Vec::new(),
            font: "10px sans-serif".to_string(),
            font_size: 10.0,
            text_align: "start".to_string(),
            text_baseline: "alphabetic".to_string(),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// True when the canvas has never drawn cross-origin content
    /// (Rec 9 origin-clean flag).
    pub fn origin_clean(&self) -> bool {
        self.origin_clean
    }

    /// Mark the canvas tainted (a cross-origin drawImage, or a
    /// draw from a tainted canvas). Irreversible until resize.
    pub fn taint(&mut self) {
        self.origin_clean = false;
    }

    /// True when the bitmap changed since the last drain (the tab
    /// thread only re-sends dirty bitmaps).
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// The bitmap as a decoded image (a copy — the caller owns
    /// the pixels it announces), clearing the dirty flag.
    pub fn take_bitmap(&mut self) -> DecodedImage {
        self.dirty = false;
        DecodedImage {
            width: self.width,
            height: self.height,
            rgba: self.rgba.clone(),
        }
    }

    /// Read the bitmap without clearing the dirty flag.
    pub fn bitmap(&self) -> DecodedImage {
        DecodedImage {
            width: self.width,
            height: self.height,
            rgba: self.rgba.clone(),
        }
    }

    /// Resize (spec: setting `width` / `height` clears the
    /// bitmap AND resets the whole context state). Returns false
    /// (and changes nothing) when the new size would exceed
    /// [`MAX_RASTER_PIXELS`]. A zero width or height is a legal
    /// empty canvas (no bitmap — drawing is a no-op).
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        if (width as usize).saturating_mul(height as usize)
            > MAX_RASTER_PIXELS
        {
            return false;
        }
        *self = Self::new(width, height).expect("size already checked");
        self.dirty = true;
        true
    }

    // --- state ---------------------------------------------------

    pub fn fill_paint(&self) -> &CanvasPaint {
        &self.fill
    }

    pub fn stroke_paint(&self) -> &CanvasPaint {
        &self.stroke
    }

    pub fn set_fill(&mut self, paint: CanvasPaint) {
        self.fill = paint;
    }

    pub fn set_stroke(&mut self, paint: CanvasPaint) {
        self.stroke = paint;
    }

    pub fn line_width(&self) -> f32 {
        self.line_width
    }

    pub fn set_line_width(&mut self, w: f32) {
        if w > 0.0 && w.is_finite() {
            self.line_width = w;
        }
    }

    pub fn line_cap(&self) -> &'static str {
        match self.line_cap {
            lyon::tessellation::LineCap::Butt => "butt",
            lyon::tessellation::LineCap::Round => "round",
            lyon::tessellation::LineCap::Square => "square",
        }
    }

    pub fn set_line_cap(&mut self, cap: &str) {
        self.line_cap = parse_line_cap(Some(cap));
    }

    pub fn line_join(&self) -> &'static str {
        match self.line_join {
            lyon::tessellation::LineJoin::Miter => "miter",
            lyon::tessellation::LineJoin::Round => "round",
            lyon::tessellation::LineJoin::Bevel => "bevel",
            _ => "miter",
        }
    }

    pub fn set_line_join(&mut self, join: &str) {
        self.line_join = parse_line_join(Some(join));
    }

    pub fn font(&self) -> &str {
        &self.font
    }

    /// Set the `font` string; the size is parsed from it (`NNpx`,
    /// falling back to 10). The family is not honoured (system
    /// font — documented).
    pub fn set_font(&mut self, font: &str) {
        self.font = font.to_string();
        for tok in font.split_whitespace() {
            if let Some(num) = tok.strip_suffix("px")
                && let Ok(v) = num.parse::<f32>()
                && v > 0.0
            {
                self.font_size = v;
                break;
            }
        }
    }

    pub fn font_size(&self) -> f32 {
        self.font_size
    }

    pub fn text_align(&self) -> &str {
        &self.text_align
    }

    pub fn set_text_align(&mut self, align: &str) {
        self.text_align = align.to_string();
    }

    pub fn text_baseline(&self) -> &str {
        &self.text_baseline
    }

    pub fn set_text_baseline(&mut self, baseline: &str) {
        self.text_baseline = baseline.to_string();
    }

    pub fn transform(&self) -> Affine {
        self.transform
    }

    // --- transform + save/restore --------------------------------

    pub fn save(&mut self) {
        self.saved.push(SavedState {
            transform: self.transform,
            fill: self.fill.clone(),
            stroke: self.stroke.clone(),
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            clip_mask: self.clip_mask.clone(),
            font: self.font.clone(),
            font_size: self.font_size,
            text_align: self.text_align.clone(),
            text_baseline: self.text_baseline.clone(),
        });
    }

    pub fn restore(&mut self) {
        if let Some(s) = self.saved.pop() {
            self.transform = s.transform;
            self.fill = s.fill;
            self.stroke = s.stroke;
            self.line_width = s.line_width;
            self.line_cap = s.line_cap;
            self.line_join = s.line_join;
            self.miter_limit = s.miter_limit;
            self.clip_mask = s.clip_mask;
            self.font = s.font;
            self.font_size = s.font_size;
            self.text_align = s.text_align;
            self.text_baseline = s.text_baseline;
        }
    }

    pub fn translate(&mut self, x: f32, y: f32) {
        self.transform = Affine::translate(x, y).then(self.transform);
    }

    pub fn scale(&mut self, x: f32, y: f32) {
        self.transform = Affine::scale(x, y).then(self.transform);
    }

    pub fn rotate(&mut self, angle_rad: f32) {
        self.transform =
            Affine::rotate(angle_rad.to_degrees()).then(self.transform);
    }

    pub fn set_transform(&mut self, m: Affine) {
        self.transform = m;
    }

    pub fn reset_transform(&mut self) {
        self.transform = Affine::identity();
    }

    // --- paths -----------------------------------------------------

    pub fn begin_path(&mut self) {
        self.path = SvgPath::new();
        self.current = None;
        self.subpath_start = None;
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        self.path.move_to([x, y]);
        self.current = Some([x, y]);
        self.subpath_start = Some([x, y]);
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.current.is_none() {
            self.move_to(x, y);
            return;
        }
        self.path.line_to([x, y]);
        self.current = Some([x, y]);
    }

    pub fn close_path(&mut self) {
        if self.current.is_none() {
            return;
        }
        self.path.close();
        self.current = self.subpath_start;
    }

    pub fn rect_path(&mut self, x: f32, y: f32, w: f32, h: f32) {
        if w == 0.0 || h == 0.0 {
            return;
        }
        let rect = path_from_rect(x, y, w, h, 0.0, 0.0);
        self.path.segments.extend(rect.segments);
        self.move_to(x, y);
    }

    /// `arc(x, y, r, start, end, anticlockwise)`: flattened to
    /// line segments (tolerance ~4 px of arc length per segment,
    /// at least 8 segments, at most 256).
    pub fn arc(&mut self, x: f32, y: f32, r: f32, start: f32, end: f32, anticlockwise: bool) {
        if r < 0.0 || !r.is_finite() {
            return;
        }
        let a0 = start;
        let mut a1 = end;
        if anticlockwise {
            while a1 > a0 {
                a1 -= 2.0 * std::f32::consts::TAU;
            }
            if (a0 - a1) >= 2.0 * std::f32::consts::TAU {
                a1 = a0 - 2.0 * std::f32::consts::TAU;
            }
        } else {
            while a1 < a0 {
                a1 += 2.0 * std::f32::consts::TAU;
            }
            if (a1 - a0) >= 2.0 * std::f32::consts::TAU {
                a1 = a0 + 2.0 * std::f32::consts::TAU;
            }
        }
        let span = (a1 - a0).abs();
        if span == 0.0 {
            return;
        }
        let steps = ((span * r / 4.0).ceil() as usize).clamp(8, 256);
        let start_pt = [x + r * a0.cos(), y + r * a0.sin()];
        if self.current.is_none() {
            self.move_to(start_pt[0], start_pt[1]);
        } else {
            self.line_to(start_pt[0], start_pt[1]);
        }
        for i in 1..=steps {
            let a = a0 + (a1 - a0) * (i as f32 / steps as f32);
            self.line_to(x + r * a.cos(), y + r * a.sin());
        }
        if span >= 2.0 * std::f32::consts::TAU - f32::EPSILON {
            self.close_path();
        }
    }

    pub fn quadratic_curve_to(&mut self, cpx: f32, cpy: f32, x: f32, y: f32) {
        if self.current.is_none() {
            self.move_to(cpx, cpy);
        }
        self.path.quad_to([cpx, cpy], [x, y]);
        self.current = Some([x, y]);
    }

    pub fn bezier_curve_to(
        &mut self,
        cp1x: f32,
        cp1y: f32,
        cp2x: f32,
        cp2y: f32,
        x: f32,
        y: f32,
    ) {
        if self.current.is_none() {
            self.move_to(cp1x, cp1y);
        }
        self.path.cubic_to([cp1x, cp1y], [cp2x, cp2y], [x, y]);
        self.current = Some([x, y]);
    }

    /// Fill the current path with the fill paint (nonzero rule
    /// unless `evenodd`).
    pub fn fill(&mut self, evenodd: bool) {
        let path = self.path.clone();
        let rule = if evenodd {
            FillRule::EvenOdd
        } else {
            FillRule::NonZero
        };
        self.paint_fill_path(&path, rule);
    }

    /// Stroke the current path with the stroke paint.
    pub fn stroke(&mut self) {
        let path = self.path.clone();
        self.paint_stroke_path(&path);
    }

    /// Intersect the current path with the clip region (the CTM
    /// at `clip()` time is baked into the mask).
    pub fn clip(&mut self) {
        let path = self.path.clone();
        if path.is_empty() {
            return;
        }
        let transformed = transform_path(&path, self.transform);
        let Ok(tris) = tessellate_fill(&transformed, FillRule::NonZero, |_| [1.0; 4]) else {
            return;
        };
        let mask = rasterize_coverage(&tris, self.width, self.height);
        self.clip_mask = Some(match self.clip_mask.take() {
            None => mask,
            Some(old) => old
                .iter()
                .zip(mask.iter())
                .map(|(a, b)| (*a).min(*b))
                .collect(),
        });
    }

    // --- rects -----------------------------------------------------

    pub fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        if w == 0.0 || h == 0.0 {
            return;
        }
        let path = path_from_rect(x, y, w, h, 0.0, 0.0);
        self.paint_fill_path(&path, FillRule::NonZero);
    }

    pub fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        if w == 0.0 || h == 0.0 {
            return;
        }
        let path = path_from_rect(x, y, w, h, 0.0, 0.0);
        self.paint_stroke_path(&path);
    }

    /// Clear a rect to transparent black (ignores the fill paint
    /// and the CTM — no: `clearRect` DOES apply the CTM per spec;
    /// it clears the transformed rect). Pixels outside the rect
    /// are untouched.
    pub fn clear_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        if w == 0.0 || h == 0.0 || self.width == 0 || self.height == 0 {
            return;
        }
        let path = path_from_rect(x, y, w, h, 0.0, 0.0);
        let transformed = transform_path(&path, self.transform);
        let Ok(tris) = tessellate_fill(&transformed, FillRule::NonZero, |_| [1.0; 4]) else {
            return;
        };
        let mask = rasterize_coverage(&tris, self.width, self.height);
        let w = self.width as usize;
        let mut changed = false;
        for (i, &m) in mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            if let Some(cm) = &self.clip_mask
                && cm[i] == 0
            {
                continue;
            }
            let idx = i * 4;
            if self.rgba[idx..idx + 4] != [0, 0, 0, 0] {
                self.rgba[idx..idx + 4].copy_from_slice(&[0, 0, 0, 0]);
                changed = true;
            }
            let _ = w;
        }
        if changed {
            self.dirty = true;
        }
    }

    // --- images ----------------------------------------------------

    /// Draw a source bitmap into a destination rect (user
    /// space, CTM applies). `src_rect` selects the source
    /// region; `dst` is `(x, y, w, h)` in user space. Nearest-
    /// neighbour sampling through the inverse CTM (so rotated /
    /// scaled draws work, at nearest quality). Returns false
    /// when nothing was drawn (empty rects / zero source).
    /// The caller decides tainting (see [`Canvas2d::taint`]).
    pub fn draw_image(
        &mut self,
        src: &DecodedImage,
        src_rect: (f32, f32, f32, f32),
        dst: (f32, f32, f32, f32),
    ) -> bool {
        if src.width == 0 || src.height == 0 || self.width == 0 || self.height == 0 {
            return false;
        }
        let (sx, sy, sw, sh) = src_rect;
        let (dx, dy, dw, dh) = dst;
        if sw <= 0.0 || sh <= 0.0 || dw == 0.0 || dh == 0.0 {
            return false;
        }
        // Device-space corners of the dest rect, then its bbox.
        let corners = [
            self.transform.apply(dx, dy),
            self.transform.apply(dx + dw, dy),
            self.transform.apply(dx, dy + dh),
            self.transform.apply(dx + dw, dy + dh),
        ];
        let min_x = corners
            .iter()
            .map(|c| c.0)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as i32;
        let max_x = corners
            .iter()
            .map(|c| c.0)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min(self.width as f32) as i32;
        let min_y = corners
            .iter()
            .map(|c| c.1)
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0) as i32;
        let max_y = corners
            .iter()
            .map(|c| c.1)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .min(self.height as f32) as i32;
        if min_x >= max_x || min_y >= max_y {
            return false;
        }
        let Some(inv) = affine_inverse(self.transform) else {
            return false;
        };
        let sw_px = src.width as f32;
        let sh_px = src.height as f32;
        let w = self.width as usize;
        let mut changed = false;
        for y in min_y..max_y {
            for x in min_x..max_x {
                let (ux, uy) = inv.apply(x as f32 + 0.5, y as f32 + 0.5);
                let u = (ux - dx) / dw;
                let v = (uy - dy) / dh;
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                    continue;
                }
                let spx = ((sx + u * sw).floor() as i32).clamp(0, src.width as i32 - 1);
                let spy = ((sy + v * sh).floor() as i32).clamp(0, src.height as i32 - 1);
                let sidx = ((spy as usize) * (src.width as usize) + (spx as usize)) * 4;
                let sa = src.rgba[sidx + 3] as f32 / 255.0;
                if sa <= 0.0 {
                    continue;
                }
                if let Some(cm) = &self.clip_mask
                    && cm[(y as usize) * w + (x as usize)] == 0
                {
                    continue;
                }
                let didx = ((y as usize) * w + (x as usize)) * 4;
                let da = self.rgba[didx + 3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                if out_a <= 0.0 {
                    continue;
                }
                for j in 0..3 {
                    let sc = src.rgba[sidx + j] as f32 / 255.0;
                    let dc = self.rgba[didx + j] as f32 / 255.0;
                    let out = (sc * sa + dc * da * (1.0 - sa)) / out_a;
                    self.rgba[didx + j] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                self.rgba[didx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
                changed = true;
            }
            let _ = sw_px;
            let _ = sh_px;
        }
        if changed {
            self.dirty = true;
        }
        changed
    }

    // --- readback (Rec 9) -----------------------------------------

    /// `getImageData(x, y, w, h)`: the pixels in the rect
    /// (transparent black outside the bitmap). Refused when
    /// tainted — the caller surfaces `SecurityError`.
    pub fn get_image_data(
        &self,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
    ) -> Result<DecodedImage, CanvasError> {
        if !self.origin_clean {
            return Err(CanvasError::Tainted);
        }
        if w == 0 || h == 0 {
            return Ok(DecodedImage {
                width: w,
                height: h,
                rgba: Vec::new(),
            });
        }
        let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
        for row in 0..h as i32 {
            let sy = y + row;
            if sy < 0 || sy >= self.height as i32 {
                continue;
            }
            for col in 0..w as i32 {
                let sx = x + col;
                if sx < 0 || sx >= self.width as i32 {
                    continue;
                }
                let sidx = ((sy as usize) * (self.width as usize) + (sx as usize)) * 4;
                let didx = ((row as usize) * (w as usize) + (col as usize)) * 4;
                out[didx..didx + 4].copy_from_slice(&self.rgba[sidx..sidx + 4]);
            }
        }
        Ok(DecodedImage {
            width: w,
            height: h,
            rgba: out,
        })
    }

    /// `putImageData(data, dx, dy)`: writes pixels ignoring the
    /// CTM (per spec) and ignoring the clip. Allowed on a tainted
    /// canvas — it exposes nothing. `data` must be
    /// `width * height * 4` bytes (a shorter buffer writes what
    /// fits; the binding validates the shape first).
    pub fn put_image_data(&mut self, data: &DecodedImage, dx: i32, dy: i32) {
        if self.width == 0 || self.height == 0 || data.width == 0 || data.height == 0 {
            return;
        }
        let mut changed = false;
        for row in 0..data.height as i32 {
            let ty = dy + row;
            if ty < 0 || ty >= self.height as i32 {
                continue;
            }
            for col in 0..data.width as i32 {
                let tx = dx + col;
                if tx < 0 || tx >= self.width as i32 {
                    continue;
                }
                let sidx = ((row as usize) * (data.width as usize) + (col as usize)) * 4;
                let didx = ((ty as usize) * (self.width as usize) + (tx as usize)) * 4;
                if self.rgba[didx..didx + 4] != data.rgba[sidx..sidx + 4] {
                    self.rgba[didx..didx + 4].copy_from_slice(&data.rgba[sidx..sidx + 4]);
                    changed = true;
                }
            }
        }
        if changed {
            self.dirty = true;
        }
    }

    /// `toDataURL()`: the bitmap as a `data:image/png;base64,…`
    /// URL. Refused when tainted.
    pub fn to_data_url(&self) -> Result<String, CanvasError> {
        if !self.origin_clean {
            return Err(CanvasError::Tainted);
        }
        let mut png = Vec::new();
        {
            use image::ImageEncoder;
            use image::codecs::png::PngEncoder;
            use image::ExtendedColorType;
            let enc = PngEncoder::new(&mut png);
            enc.write_image(&self.rgba, self.width, self.height, ExtendedColorType::Rgba8)
                .map_err(|_| CanvasError::Empty)?;
        }
        Ok(format!("data:image/png;base64,{}", base64_encode(&png)))
    }

    // --- text ------------------------------------------------------

    /// `fillText` / `strokeText`: shape `text` at `(x, y)` (the
    /// baseline, per `textBaseline`) with the fill / stroke paint.
    /// The system font is used (the `font` size is honoured, its
    /// family is not — documented). Text is laid out in user
    /// space; the CTM's uniform scale sizes the glyphs, its
    /// translation positions them (rotation/skew of glyphs is
    /// not applied — documented approximation).
    pub fn fill_text(&mut self, text: &str, x: f32, y: f32, stroke: bool) {
        if text.is_empty() || self.width == 0 || self.height == 0 {
            return;
        }
        let paint = if stroke {
            self.stroke.clone()
        } else {
            self.fill.clone()
        };
        // Horizontal alignment: measure first, then shift.
        let text_w = measure_text(text, self.font_size);
        let scale = self.transform.a.hypot(self.transform.b).max(0.001);
        let size = self.font_size * scale;
        let mut bx = x;
        match self.text_align.as_str() {
            "center" => bx -= text_w / 2.0,
            "right" | "end" => bx -= text_w,
            _ => {}
        }
        // Baseline: y is the baseline for `alphabetic` /
        // `ideographic`; `top` puts the em top at y, `middle`
        // centres the em box on y, `bottom` puts the em bottom
        // at y (em ascent ≈ 0.8 × size, descent ≈ 0.2 × size).
        let baseline_y = match self.text_baseline.as_str() {
            "top" | "hanging" => y + size * 0.8,
            "middle" => y + size * 0.3,
            "bottom" => y - size * 0.2,
            _ => y,
        };
        let (ox, oy) = self.transform.apply(bx, baseline_y);
        rasterize_text_into(
            &mut self.rgba,
            self.width,
            self.height,
            text,
            size,
            ox,
            oy,
            &paint,
            self.clip_mask.as_deref(),
        );
        self.dirty = true;
    }

    /// The advance width of `text` at the current font size
    /// (user units).
    pub fn measure_text(&self, text: &str) -> f32 {
        measure_text(text, self.font_size)
    }

    // --- internals ---------------------------------------------------

    /// Fill a user-space path with the fill paint (CTM applies).
    fn paint_fill_path(&mut self, path: &SvgPath, rule: FillRule) {
        if path.is_empty() || self.width == 0 || self.height == 0 {
            return;
        }
        let transformed = transform_path(path, self.transform);
        let paint = self.fill.clone();
        if paint.is_gradient() {
            let Ok(tris) = tessellate_fill(&transformed, rule, |_| [1.0; 4]) else {
                return;
            };
            self.paint_gradient_tris(&tris, &paint);
        } else {
            let color = paint.color_at([0.0, 0.0]);
            let Ok(tris) = tessellate_fill(&transformed, rule, move |_| color) else {
                return;
            };
            self.paint_solid_tris(&tris);
        }
    }

    /// Stroke a user-space path with the stroke paint (CTM
    /// applies; `lineWidth` is in user units, so the transformed
    /// path is tessellated at the same width — the CTM's scale
    /// widens the stroke, per spec).
    fn paint_stroke_path(&mut self, path: &SvgPath) {
        if path.is_empty() || self.width == 0 || self.height == 0 {
            return;
        }
        let transformed = transform_path(path, self.transform);
        let paint = self.stroke.clone();
        if paint.is_gradient() {
            let Ok(tris) = tessellate_stroke(
                &transformed,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                |_| [1.0; 4],
            ) else {
                return;
            };
            self.paint_gradient_tris(&tris, &paint);
        } else {
            let color = paint.color_at([0.0, 0.0]);
            let Ok(tris) = tessellate_stroke(
                &transformed,
                self.line_width,
                self.line_cap,
                self.line_join,
                self.miter_limit,
                move |_| color,
            ) else {
                return;
            };
            self.paint_solid_tris(&tris);
        }
    }

    /// Composite solid-colour triangles into the bitmap (through
    /// the clip mask when one is set).
    fn paint_solid_tris(&mut self, tris: &[VectorVertex]) {
        if tris.is_empty() {
            return;
        }
        match &self.clip_mask {
            None => {
                rasterize_triangles_into(&mut self.rgba, tris, self.width, self.height);
                self.dirty = true;
            }
            Some(mask) => {
                let mut tmp = vec![0u8; self.rgba.len()];
                rasterize_triangles_into(&mut tmp, tris, self.width, self.height);
                blend_masked(&mut self.rgba, &tmp, mask);
                self.dirty = true;
            }
        }
    }

    /// Composite gradient triangles: coverage mask, then the
    /// gradient evaluated per covered pixel (user-space point =
    /// device pixel through the inverse CTM), through the clip.
    fn paint_gradient_tris(&mut self, tris: &[VectorVertex], paint: &CanvasPaint) {
        if tris.is_empty() {
            return;
        }
        let mask = rasterize_coverage(tris, self.width, self.height);
        let Some(inv) = affine_inverse(self.transform) else {
            return;
        };
        let w = self.width as usize;
        let mut changed = false;
        for (i, &m) in mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            if let Some(cm) = &self.clip_mask
                && cm[i] == 0
            {
                continue;
            }
            let x = (i % w) as f32 + 0.5;
            let y = (i / w) as f32 + 0.5;
            let (ux, uy) = inv.apply(x, y);
            let c = paint.color_at([ux, uy]);
            let sa = (c[3] * m as f32 / 255.0).clamp(0.0, 1.0);
            if sa <= 0.0 {
                continue;
            }
            let idx = i * 4;
            let da = self.rgba[idx + 3] as f32 / 255.0;
            let out_a = sa + da * (1.0 - sa);
            if out_a <= 0.0 {
                continue;
            }
            for (j, &sc) in c.iter().enumerate().take(3) {
                let sc = sc.clamp(0.0, 1.0);
                let dc = self.rgba[idx + j] as f32 / 255.0;
                let out = (sc * sa + dc * da * (1.0 - sa)) / out_a;
                self.rgba[idx + j] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            self.rgba[idx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
            changed = true;
        }
        if changed {
            self.dirty = true;
        }
    }
}

/// Apply an affine to every point of a path (same helper shape
/// as [`crate::svg`]'s — the canvas CTM composes the same way).
fn transform_path(path: &SvgPath, m: Affine) -> SvgPath {
    use crate::path::PathSeg;
    let mut out = SvgPath::new();
    let pt = |p: [f32; 2]| -> [f32; 2] {
        let (x, y) = m.apply(p[0], p[1]);
        [x, y]
    };
    for seg in &path.segments {
        match *seg {
            PathSeg::MoveTo(p) => out.move_to(pt(p)),
            PathSeg::LineTo(p) => out.line_to(pt(p)),
            PathSeg::QuadTo(c, p) => out.quad_to(pt(c), pt(p)),
            PathSeg::CubicTo(c1, c2, p) => out.cubic_to(pt(c1), pt(c2), pt(p)),
            PathSeg::Close => out.close(),
        }
    }
    out
}

/// Source-over `src` (straight alpha, transparent black where it
/// never painted) onto `dst`, only where `mask` is set.
fn blend_masked(dst: &mut [u8], src: &[u8], mask: &[u8]) {
    for (i, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let idx = i * 4;
        let sa = src[idx + 3] as f32 / 255.0;
        if sa <= 0.0 {
            continue;
        }
        let da = dst[idx + 3] as f32 / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            continue;
        }
        for j in 0..3 {
            let sc = src[idx + j] as f32 / 255.0;
            let dc = dst[idx + j] as f32 / 255.0;
            let out = (sc * sa + dc * da * (1.0 - sa)) / out_a;
            dst[idx + j] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        dst[idx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
}

/// Standard base64 (no line breaks) — the `data:` URL encoder
/// for [`Canvas2d::to_data_url`]. No new dependency: the alphabet
/// is 64 bytes and the padding rule is three lines.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

// --- text (cosmic-text) ---------------------------------------------

/// The shared font system for canvas text: one `FontSystem` +
/// `SwashCache` per process (the renderer owns its own pair;
/// this one serves worker-side canvas rasterisation). Lazy —
/// `FontSystem::new()` scans system fonts, which is slow enough
/// to pay only when a page actually draws text.
fn font_system() -> &'static std::sync::Mutex<(
    cosmic_text::FontSystem,
    cosmic_text::SwashCache,
)> {
    use std::sync::{Mutex, OnceLock};
    static FONTS: OnceLock<Mutex<(cosmic_text::FontSystem, cosmic_text::SwashCache)>> =
        OnceLock::new();
    FONTS.get_or_init(|| {
        Mutex::new((
            cosmic_text::FontSystem::new(),
            cosmic_text::SwashCache::new(),
        ))
    })
}

/// The advance width of `text` at `font_size` (user units):
/// shape once, sum the run widths.
pub fn measure_text(text: &str, font_size: f32) -> f32 {
    if text.is_empty() || font_size <= 0.0 {
        return 0.0;
    }
    let mut guard = font_system().lock().unwrap_or_else(|e| e.into_inner());
    let (fs, _) = &mut *guard;
    let mut buffer = cosmic_text::Buffer::new(
        fs,
        cosmic_text::Metrics::new(font_size, font_size * 1.2),
    );
    buffer.set_text(
        text,
        &cosmic_text::Attrs::new(),
        cosmic_text::Shaping::Advanced,
        None,
    );
    buffer.set_size(None, None);
    buffer.shape_until_scroll(fs, false);
    buffer
        .layout_runs()
        .map(|run| run.line_w)
        .fold(0.0f32, f32::max)
}

/// Rasterise `text` into `rgba` (`width` × `height`, top-left
/// origin) at device-space origin `(ox, oy)` (the baseline), in
/// `paint`'s colour (gradients evaluate at the glyph origin —
/// text over a gradient is a solid-colour approximation,
/// documented). `clip` masks the output when set.
#[allow(clippy::too_many_arguments)]
fn rasterize_text_into(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    text: &str,
    size: f32,
    ox: f32,
    oy: f32,
    paint: &CanvasPaint,
    clip: Option<&[u8]>,
) {
    let color = paint.color_at([ox, oy]);
    let (r, g, b) = (
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    let alpha = color[3].clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return;
    }
    let mut guard = font_system().lock().unwrap_or_else(|e| e.into_inner());
    let (fs, swash) = &mut *guard;
    let mut buffer = cosmic_text::Buffer::new(fs, cosmic_text::Metrics::new(size, size * 1.2));
    buffer.set_text(text, &cosmic_text::Attrs::new(), cosmic_text::Shaping::Advanced, None);
    buffer.set_size(None, None);
    buffer.shape_until_scroll(fs, false);
    let w = width as usize;
    for run in buffer.layout_runs() {
        for glyph in run.glyphs {
            let physical = glyph.physical((ox, run.line_y + oy - size * 0.8), 1.0);
            let Some(image) = swash.get_image(fs, physical.cache_key) else {
                continue;
            };
            if image.content != cosmic_text::SwashContent::Mask {
                continue;
            }
            let placement = image.placement;
            if placement.width == 0 || placement.height == 0 {
                continue;
            }
            // Glyph bitmap origin: (physical.x + left,
            // physical.y - top) — the same anchor the renderer's
            // `glyph_quad` uses (see `renderer.rs`).
            let gx = physical.x + placement.left;
            let gy = physical.y - placement.top;
            let pw = placement.width as usize;
            let ph = placement.height as usize;
            for row in 0..ph {
                let ty = gy + row as i32;
                if ty < 0 || ty >= height as i32 {
                    continue;
                }
                for col in 0..pw {
                    let tx = gx + col as i32;
                    if tx < 0 || tx >= width as i32 {
                        continue;
                    }
                    let coverage = image.data[row * pw + col] as f32 / 255.0;
                    if coverage <= 0.0 {
                        continue;
                    }
                    let idx = (ty as usize) * w + (tx as usize);
                    if let Some(cm) = clip
                        && cm[idx] == 0
                    {
                        continue;
                    }
                    let sa = alpha * coverage;
                    let didx = idx * 4;
                    let da = rgba[didx + 3] as f32 / 255.0;
                    let out_a = sa + da * (1.0 - sa);
                    if out_a <= 0.0 {
                        continue;
                    }
                    for (j, &sc) in [r, g, b].iter().enumerate() {
                        let scf = sc as f32 / 255.0;
                        let dc = rgba[didx + j] as f32 / 255.0;
                        let out = (scf * sa + dc * da * (1.0 - sa)) / out_a;
                        rgba[didx + j] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                    rgba[didx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(c: &Canvas2d, x: u32, y: u32) -> [u8; 4] {
        let i = ((y as usize) * (c.width as usize) + (x as usize)) * 4;
        [c.rgba[i], c.rgba[i + 1], c.rgba[i + 2], c.rgba[i + 3]]
    }

    #[test]
    fn fill_rect_paints_pixels_and_clear_rect_empties_them() {
        let mut c = Canvas2d::new(10, 10).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.fill_rect(2.0, 2.0, 4.0, 4.0);
        assert_eq!(px(&c, 3, 3), [255, 0, 0, 255]);
        assert_eq!(px(&c, 0, 0), [0, 0, 0, 0]);
        assert!(c.dirty());
        c.clear_rect(2.0, 2.0, 4.0, 4.0);
        assert_eq!(px(&c, 3, 3), [0, 0, 0, 0]);
    }

    #[test]
    fn stroke_rect_outlines_without_filling() {
        let mut c = Canvas2d::new(10, 10).unwrap();
        c.set_stroke(CanvasPaint::Solid([0, 0, 255, 255]));
        c.set_line_width(2.0);
        c.stroke_rect(1.0, 1.0, 8.0, 8.0);
        assert_eq!(px(&c, 1, 5)[2], 255, "left edge is blue");
        assert_eq!(px(&c, 5, 5)[3], 0, "centre stays empty");
    }

    #[test]
    fn path_fill_and_stroke_follow_the_current_path() {
        let mut c = Canvas2d::new(20, 20).unwrap();
        c.set_fill(CanvasPaint::Solid([0, 255, 0, 255]));
        c.begin_path();
        c.move_to(2.0, 2.0);
        c.line_to(18.0, 2.0);
        c.line_to(10.0, 18.0);
        c.close_path();
        c.fill(false);
        assert_eq!(px(&c, 10, 6), [0, 255, 0, 255]);
        assert_eq!(px(&c, 1, 1)[3], 0);
    }

    #[test]
    fn arc_fills_a_disc() {
        let mut c = Canvas2d::new(20, 20).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.begin_path();
        c.arc(10.0, 10.0, 8.0, 0.0, std::f32::consts::TAU, false);
        c.fill(false);
        assert_eq!(px(&c, 10, 10), [255, 0, 0, 255]);
        assert_eq!(px(&c, 1, 1)[3], 0);
    }

    #[test]
    fn save_restore_and_transforms_apply() {
        let mut c = Canvas2d::new(20, 20).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.save();
        c.translate(10.0, 0.0);
        c.fill_rect(0.0, 0.0, 5.0, 5.0);
        c.restore();
        c.fill_rect(0.0, 0.0, 5.0, 5.0);
        assert_eq!(px(&c, 12, 2), [255, 0, 0, 255], "translated rect");
        assert_eq!(px(&c, 2, 2), [255, 0, 0, 255], "restored rect");
        assert_eq!(px(&c, 7, 2)[3], 0, "gap between the two rects");
    }

    #[test]
    fn linear_gradient_fill_interpolates() {
        let mut c = Canvas2d::new(10, 1).unwrap();
        let mut g = CanvasGradient::new(CanvasGradientKind::Linear {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 0.0,
        });
        g.add_stop(0.0, [255, 0, 0, 255]);
        g.add_stop(1.0, [0, 0, 255, 255]);
        c.set_fill(CanvasPaint::Gradient(g));
        c.fill_rect(0.0, 0.0, 10.0, 1.0);
        assert!(px(&c, 0, 0)[0] > 200, "left is red");
        assert!(px(&c, 9, 0)[2] > 200, "right is blue");
    }

    #[test]
    fn radial_gradient_fill_is_bright_in_the_centre() {
        let mut c = Canvas2d::new(9, 9).unwrap();
        let mut g = CanvasGradient::new(CanvasGradientKind::Radial {
            x1: 4.0,
            y1: 4.0,
            r1: 0.0,
            x2: 4.0,
            y2: 4.0,
            r2: 4.0,
        });
        g.add_stop(0.0, [255, 255, 255, 255]);
        g.add_stop(1.0, [0, 0, 0, 255]);
        c.set_fill(CanvasPaint::Gradient(g));
        c.fill_rect(0.0, 0.0, 9.0, 9.0);
        assert!(px(&c, 4, 4)[0] > 200, "centre is white");
        assert!(px(&c, 0, 0)[0] < 55, "corner is black");
    }

    #[test]
    fn clip_restricts_painting() {
        let mut c = Canvas2d::new(10, 10).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.begin_path();
        c.rect_path(0.0, 0.0, 5.0, 10.0);
        c.clip();
        c.fill_rect(0.0, 0.0, 10.0, 10.0);
        assert_eq!(px(&c, 2, 5)[0], 255);
        assert_eq!(px(&c, 7, 5)[3], 0, "outside the clip stays empty");
    }

    #[test]
    fn resize_resets_bitmap_and_state() {
        let mut c = Canvas2d::new(10, 10).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.fill_rect(0.0, 0.0, 10.0, 10.0);
        assert!(c.resize(4, 4));
        assert_eq!((c.width(), c.height()), (4, 4));
        assert_eq!(px(&c, 0, 0), [0, 0, 0, 0], "bitmap cleared");
        assert_eq!(
            c.fill_paint(),
            &CanvasPaint::Solid([0, 0, 0, 255]),
            "state reset to black"
        );
    }

    #[test]
    fn oversize_canvas_is_refused() {
        assert!(Canvas2d::new(5000, 5000).is_none(), "25 Mpx > 16.7 Mpx cap");
        assert!(Canvas2d::new(4096, 4096).is_none(), "4096^2 = 16,777,216 px > cap");
        assert!(Canvas2d::new(3840, 2160).is_some(), "4K fits the cap");
    }

    #[test]
    fn tainted_canvas_refuses_readback_but_accepts_put() {
        let mut c = Canvas2d::new(4, 4).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.fill_rect(0.0, 0.0, 4.0, 4.0);
        assert!(c.get_image_data(0, 0, 4, 4).is_ok());
        assert!(c.to_data_url().is_ok());
        c.taint();
        assert!(!c.origin_clean());
        assert_eq!(c.get_image_data(0, 0, 4, 4), Err(CanvasError::Tainted));
        assert_eq!(c.to_data_url(), Err(CanvasError::Tainted));
        // putImageData writes without reading: allowed.
        c.put_image_data(
            &DecodedImage {
                width: 1,
                height: 1,
                rgba: vec![0, 255, 0, 255],
            },
            0,
            0,
        );
        assert_eq!(px(&c, 0, 0), [0, 255, 0, 255]);
    }

    #[test]
    fn draw_image_blits_and_scales() {
        let mut c = Canvas2d::new(10, 10).unwrap();
        let src = DecodedImage {
            width: 2,
            height: 2,
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
        };
        assert!(c.draw_image(&src, (0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 4.0, 4.0)));
        assert_eq!(px(&c, 0, 0), [255, 0, 0, 255]);
        assert_eq!(px(&c, 3, 0), [0, 255, 0, 255]);
        assert_eq!(px(&c, 0, 3), [0, 0, 255, 255]);
        assert_eq!(px(&c, 5, 5)[3], 0);
    }

    #[test]
    fn get_and_put_image_data_round_trip() {
        let mut c = Canvas2d::new(4, 4).unwrap();
        c.set_fill(CanvasPaint::Solid([10, 20, 30, 255]));
        c.fill_rect(1.0, 1.0, 2.0, 2.0);
        let img = c.get_image_data(0, 0, 4, 4).unwrap();
        assert_eq!(img.rgba[(4 + 1) * 4], 10);
        assert_eq!(img.rgba[0], 0, "outside the rect is transparent");
        // Put a green pixel at (3, 3) via ImageData.
        let data = DecodedImage {
            width: 1,
            height: 1,
            rgba: vec![0, 200, 0, 255],
        };
        c.put_image_data(&data, 3, 3);
        assert_eq!(px(&c, 3, 3), [0, 200, 0, 255]);
    }

    #[test]
    fn to_data_url_is_a_png_data_url() {
        let mut c = Canvas2d::new(2, 2).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.fill_rect(0.0, 0.0, 2.0, 2.0);
        let url = c.to_data_url().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.len() > 30);
    }

    #[test]
    fn fill_text_paints_glyph_pixels() {
        let mut c = Canvas2d::new(60, 20).unwrap();
        c.set_fill(CanvasPaint::Solid([0, 0, 0, 255]));
        c.set_font("16px sans-serif");
        c.fill_text("Hi", 2.0, 16.0, false);
        let painted = c.rgba.as_chunks::<4>().0.iter().any(|p| p[3] > 0);
        assert!(painted, "fillText must paint some pixels");
    }

    #[test]
    fn color_parsing_covers_css_spellings() {
        assert_eq!(parse_canvas_color("red"), Some([255, 0, 0, 255]));
        assert_eq!(parse_canvas_color("#00ff00"), Some([0, 255, 0, 255]));
        assert_eq!(parse_canvas_color("rgb(1, 2, 3)"), Some([1, 2, 3, 255]));
        assert_eq!(parse_canvas_color("rgba(1, 2, 3, 0.5)"), Some([1, 2, 3, 127]));
        assert_eq!(parse_canvas_color("not-a-colour"), None);
    }

    #[test]
    fn quadratic_and_bezier_curves_fill() {
        let mut c = Canvas2d::new(20, 20).unwrap();
        c.set_fill(CanvasPaint::Solid([255, 0, 0, 255]));
        c.begin_path();
        c.move_to(2.0, 18.0);
        c.quadratic_curve_to(10.0, 2.0, 18.0, 18.0);
        c.close_path();
        c.fill(false);
        assert_eq!(px(&c, 10, 12)[0], 255);
        let mut c2 = Canvas2d::new(20, 20).unwrap();
        c2.set_fill(CanvasPaint::Solid([0, 0, 255, 255]));
        c2.begin_path();
        c2.move_to(2.0, 18.0);
        c2.bezier_curve_to(2.0, 2.0, 18.0, 2.0, 18.0, 18.0);
        c2.close_path();
        c2.fill(false);
        assert_eq!(px(&c2, 10, 12)[2], 255);
    }
}
