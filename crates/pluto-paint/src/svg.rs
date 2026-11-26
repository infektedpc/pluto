//! Inline SVG subtree → shapes → vector triangles or raster
//! bitmap (Stage 15, M7 Phase 2, Rec 3/5/7).
//!
//! [`collect_svg_content`] walks one `<svg>` element's subtree
//! (paths, rects, circles, ellipses, lines, polylines, polygons,
//! groups, nested `<svg>`) into [`SvgShape`]s in user units,
//! resolving `fill` / `stroke` from the computed style (Phase 1
//! folded presentation attributes into the cascade) and
//! `<linearGradient>` / `<radialGradient>` defs by id.
//!
//! Two display routes share this walk:
//! - **Vector** ([`tessellate_svg_shapes`]): lyon tessellates
//!   each shape's fill/stroke to coloured triangles. Only solid
//!   fills and two-stop linear gradients go this way (a two-stop
//!   linear gradient is exact under barycentric interpolation).
//! - **Raster** ([`rasterize_svg_content`]): the same triangles
//!   are filled into an RGBA bitmap by [`crate::raster`] — the
//!   route for radial gradients, multi-stop linear gradients, and
//!   any shape the tessellator refuses. Fixed raster resolution:
//!   the bitmap is the box's used size, re-rastered on resize
//!   (the display list rebuilds then); full-frame only.
//!
//! Deferred (documented, not silent): `<text>`, `<foreignObject>`
//! content in the raster route (layout paints its boxes — see
//! Rec 10), filters/masks/clipPath, patterns, SMIL,
//! `gradient href` inheritance, `spreadMethod` other than `pad`,
//! `stroke-dasharray` (renders solid), `preserveAspectRatio`
//! `slice` alignment is honoured but the overflowing content is
//! clipped to the viewport (correct), nested `<svg>` `viewBox`
//! (its `x`/`y`/`transform` apply; its `viewBox` is ignored),
//! external `<use>` (same-document only — Rec 10), `<symbol>`
//! `width`/`height` on `<use>` (the symbol paints at its
//! intrinsic size).

use crate::path::{
    FillRule, PathError, SvgPath, VectorVertex, parse_fill_rule, parse_line_cap,
    parse_line_join, parse_path_data, path_from_circle, path_from_ellipse, path_from_line,
    path_from_points, path_from_rect, tessellate_fill, tessellate_stroke,
};
use crate::raster::{rasterize_coverage, rasterize_triangles_into};
use crate::{DecodedImage, Resource, ResourceId};
use pluto_css::{Affine, Gradient, GradientKind, GradientStop, RadialExtent, RadialShape};
use pluto_html::{Dom, NodeId as DomId};
use pluto_layout::{DisplayKind, NodeStyle, Paint};
use slotmap::Key as _;
use slotmap::SparseSecondaryMap;
use std::collections::HashMap;

/// Resource-id base for rasterised inline-SVG bitmaps. Ids are
/// deterministic per element (`BASE | slot-index`), never drawn
/// from [`ImageCache`](crate::ImageCache)'s sequential counter
/// (which starts at 1 and would need 2^30 inserts to collide) —
/// so a scroll re-collect names the same id the tab thread
/// already holds, and no `ResourceReady` re-send is needed. The
/// slot index is masked to 24 bits (16.7M nodes per document is
/// not a real bound to hit); a document with more nodes than
/// that would alias ids across elements — accepted, documented.
pub const SVG_BLOB_ID_BASE: u32 = 0x4000_0000;
/// Same, for rasterised CSS radial-gradient backgrounds (a
/// different base so an `<svg>` and a radial background on the
/// same element never share an id).
pub const BG_GRADIENT_BLOB_ID_BASE: u32 = 0x5000_0000;

/// Max rasterised SVG pixels (~16.7 Mpx): one 4K bitmap is
/// ~33 MB of RGBA, which fits a single 64 MiB IPC frame; anything
/// larger is refused and the SVG paints through the vector path
/// (or nothing, if it needs the rasteriser — the same refusal as
/// an oversize decoded image).
/// The largest raster bitmap, in pixels: (64 MiB - 4 KiB) / 4
/// bytes per pixel. The 64 MiB is the IPC frame cap
/// (`transport::MAX_FRAME_BYTES`) and the 4 KiB is the
/// `ResourceReady` frame overhead
/// (`ImageStore::FRAME_OVERHEAD_BYTES`) — so every bitmap this
/// layer produces fits in one `ResourceReady` frame, and the tab
/// thread never has to refuse a Blob the list already names.
pub const MAX_RASTER_PIXELS: usize = 16_776_192;

/// Deterministic blob id for an inline `<svg>` element.
pub fn svg_blob_id(id: DomId) -> ResourceId {
    ResourceId(SVG_BLOB_ID_BASE | (id.data().as_ffi() as u32 & 0x00FF_FFFF))
}

/// Deterministic blob id for a CSS radial-gradient background.
pub fn bg_gradient_blob_id(id: DomId) -> ResourceId {
    ResourceId(BG_GRADIENT_BLOB_ID_BASE | (id.data().as_ffi() as u32 & 0x00FF_FFFF))
}

/// A length in an SVG gradient definition: a fraction (of the
/// bounding box, or of the viewport when `userSpaceOnUse`) or an
/// absolute user-space value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SvgLength {
    Fraction(f32),
    User(f32),
}

/// Which coordinate system an SVG gradient is defined in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientUnits {
    ObjectBoundingBox,
    UserSpaceOnUse,
}

/// The geometry of an SVG gradient definition.
#[derive(Debug, Clone, PartialEq)]
pub enum GradientDefKind {
    Linear {
        x1: SvgLength,
        y1: SvgLength,
        x2: SvgLength,
        y2: SvgLength,
    },
    Radial {
        cx: SvgLength,
        cy: SvgLength,
        r: SvgLength,
    },
}

/// One parsed `<linearGradient>` / `<radialGradient>` element:
/// geometry + stops + `gradientTransform`. `spreadMethod` other
/// than `pad` is treated as `pad` (documented). A gradient
/// referencing another via `href` is not resolved (deferred) —
/// a def with no stops of its own paints nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientDef {
    pub kind: GradientDefKind,
    pub units: GradientUnits,
    pub transform: Affine,
    pub stops: Vec<GradientStop>,
}

/// A resolved SVG paint: none, a solid colour (straight-alpha
/// RGBA floats), or a gradient definition.
#[derive(Debug, Clone, PartialEq)]
pub enum SvgPaint {
    None,
    Solid([f32; 4]),
    Gradient(GradientDef),
}

/// One paintable SVG shape, in user units (the viewBox transform
/// is applied by the emitter, not here — so the same shapes feed
/// the vector path and the rasteriser unchanged).
#[derive(Debug, Clone, PartialEq)]
pub struct SvgShape {
    pub path: SvgPath,
    pub fill: SvgPaint,
    pub stroke: SvgPaint,
    /// `opacity` x `fill-opacity` (1.0 when unset): applied to
    /// gradient fills at tessellation time (solid fills bake it
    /// into their colour in [`resolve_paint`]).
    pub fill_opacity: f32,
    /// `opacity` x `stroke-opacity`, same for strokes.
    pub stroke_opacity: f32,
    pub stroke_width: f32,
    pub fill_rule: FillRule,
    pub cap: lyon::tessellation::LineCap,
    pub join: lyon::tessellation::LineJoin,
    pub miter_limit: f32,
}

/// The collected content of one `<svg>` subtree: its shapes in
/// document order, plus the svg viewport size (the box's used
/// size — the basis for `userSpaceOnUse` percentages).
#[derive(Debug, Clone, PartialEq)]
pub struct SvgContent {
    pub shapes: Vec<SvgShape>,
    pub viewport: (f32, f32),
    /// `<svg><image>` elements, in document order (user units):
    /// the display list resolves each `href` to a resource id
    /// through the image map the tab thread built (Stage 11
    /// gates already fetched + decoded them — Rec 10). Empty
    /// when the svg has no `<image>` children.
    pub images: Vec<SvgImage>,
}

/// One `<svg><image>` element: its rect (user units) and the
/// `href` / `xlink:href` as written (unresolved — the caller
/// resolves it against the document base, like `<img>`).
#[derive(Debug, Clone, PartialEq)]
pub struct SvgImage {
    pub dom: DomId,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub href: String,
}

/// Parse one SVG number-or-percentage attribute: `50%` is a
/// fraction of `basis`, a bare number is itself. `None` when the
/// value does not parse.
fn attr_number(value: &str, basis: f32) -> Option<f32> {
    let s = value.trim();
    if let Some(num) = s.strip_suffix('%') {
        num.trim().parse::<f32>().ok().map(|v| v / 100.0 * basis)
    } else {
        s.parse::<f32>().ok()
    }
}

/// Read a numeric attribute (`cx`, `width`, ...) as an `f32`,
/// defaulting to `default`. Percentages resolve against `basis`.
fn attr_f32(dom: &Dom, id: DomId, name: &str, default: f32, basis: f32) -> f32 {
    dom.attr(id, name)
        .and_then(|v| attr_number(v, basis))
        .unwrap_or(default)
}

/// Parse a gradient coordinate (`x1`, `r`, ...): a `%` is a
/// fraction; a bare number is a fraction in `objectBoundingBox`
/// mode and a user-space value in `userSpaceOnUse` mode.
fn parse_gradient_length(value: &str, units: GradientUnits) -> Option<SvgLength> {
    let s = value.trim();
    if let Some(num) = s.strip_suffix('%') {
        num.trim()
            .parse::<f32>()
            .ok()
            .map(|v| SvgLength::Fraction(v / 100.0))
    } else {
        s.parse::<f32>().ok().map(|v| match units {
            GradientUnits::ObjectBoundingBox => SvgLength::Fraction(v),
            GradientUnits::UserSpaceOnUse => SvgLength::User(v),
        })
    }
}

/// Parse a `viewBox` attribute (`min-x min-y width height`).
pub fn parse_view_box(value: &str) -> Option<[f32; 4]> {
    let mut nums = Vec::with_capacity(4);
    for tok in value.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        nums.push(tok.parse::<f32>().ok()?);
    }
    if nums.len() == 4 && nums[2] > 0.0 && nums[3] > 0.0 {
        Some([nums[0], nums[1], nums[2], nums[3]])
    } else {
        None
    }
}

/// The user-space → box-pixel transform for an `<svg>` viewport:
/// `viewBox` + `preserveAspectRatio` (`xMidYMid meet` by
/// default). `None` viewBox (or a degenerate one) is the identity
/// — user units are pixels. Pure.
pub fn view_box_transform(
    box_w: f32,
    box_h: f32,
    view_box: Option<[f32; 4]>,
    preserve: Option<&str>,
) -> Affine {
    let Some([min_x, min_y, vb_w, vb_h]) = view_box else {
        return Affine::identity();
    };
    if vb_w <= 0.0 || vb_h <= 0.0 || box_w <= 0.0 || box_h <= 0.0 {
        return Affine::identity();
    }
    let (align, mode) = match preserve {
        Some(p) => {
            let mut parts = p.split_whitespace();
            (parts.next().unwrap_or("xMidYMid"), parts.next().unwrap_or("meet"))
        }
        None => ("xMidYMid", "meet"),
    };
    if align.eq_ignore_ascii_case("none") {
        let sx = box_w / vb_w;
        let sy = box_h / vb_h;
        return Affine {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: -min_x * sx,
            f: -min_y * sy,
        };
    }
    let s = if mode.eq_ignore_ascii_case("slice") {
        (box_w / vb_w).max(box_h / vb_h)
    } else {
        (box_w / vb_w).min(box_h / vb_h)
    };
    let align_x = if align.starts_with("xMin") {
        0.0
    } else if align.starts_with("xMax") {
        1.0
    } else {
        0.5
    };
    let align_y = if align.contains("YMin") {
        0.0
    } else if align.contains("YMax") {
        1.0
    } else {
        0.5
    };
    Affine {
        a: s,
        b: 0.0,
        c: 0.0,
        d: s,
        e: align_x * (box_w - vb_w * s) - min_x * s,
        f: align_y * (box_h - vb_h * s) - min_y * s,
    }
}

/// Invert a 2D affine, or `None` when it is singular (a
/// zero-scale gradient transform falls back to the identity at
/// the call site).
pub fn affine_inverse(m: Affine) -> Option<Affine> {
    let det = m.a * m.d - m.b * m.c;
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    Some(Affine {
        a: m.d / det,
        b: -m.b / det,
        c: -m.c / det,
        d: m.a / det,
        e: (m.c * m.f - m.d * m.e) / det,
        f: (m.b * m.e - m.a * m.f) / det,
    })
}

/// Parse an SVG `transform` attribute (`translate()`,
/// `scale()`, `rotate()`, `skewX()`, `skewY()`, `matrix()`) into
/// one [`Affine`]. Functions apply left-to-right (each one in
/// the coordinate system the previous ones established). An
/// unparseable list is the identity (never an error — a bad
/// transform must not take the whole icon down).
pub fn parse_svg_transform(value: &str) -> Affine {
    let bytes = value.as_bytes();
    let mut pos = 0usize;
    let mut acc = Affine::identity();
    while pos < bytes.len() {
        while pos < bytes.len() && !(bytes[pos] as char).is_ascii_alphabetic() {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let start = pos;
        while pos < bytes.len() && (bytes[pos] as char).is_ascii_alphabetic() {
            pos += 1;
        }
        let name = &value[start..pos];
        while pos < bytes.len() && (bytes[pos] as char).is_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b'(' {
            break;
        }
        pos += 1;
        let mut nums = Vec::new();
        let mut cur = String::new();
        while pos < bytes.len() && bytes[pos] != b')' {
            let c = bytes[pos] as char;
            if c == ',' || c.is_whitespace() {
                if !cur.is_empty() {
                    if let Ok(v) = cur.parse::<f32>() {
                        nums.push(v);
                    }
                    cur.clear();
                }
            } else {
                cur.push(c);
            }
            pos += 1;
        }
        if !cur.is_empty() {
            if let Ok(v) = cur.parse::<f32>() {
                nums.push(v);
            }
        }
        if pos < bytes.len() {
            pos += 1; // ')'
        }
        let f = match name {
            "matrix" if nums.len() == 6 => Some(Affine {
                a: nums[0],
                b: nums[1],
                c: nums[2],
                d: nums[3],
                e: nums[4],
                f: nums[5],
            }),
            "translate" => match nums.len() {
                1 => Some(Affine::translate(nums[0], 0.0)),
                2 => Some(Affine::translate(nums[0], nums[1])),
                _ => None,
            },
            "scale" => match nums.len() {
                1 => Some(Affine::scale(nums[0], nums[0])),
                2 => Some(Affine::scale(nums[0], nums[1])),
                _ => None,
            },
            "rotate" => match nums.len() {
                1 => Some(Affine::rotate(nums[0])),
                3 => Some(
                    Affine::translate(-nums[1], -nums[2])
                        .then(Affine::rotate(nums[0]))
                        .then(Affine::translate(nums[1], nums[2])),
                ),
                _ => None,
            },
            "skewX" if nums.len() == 1 => Some(Affine::skew_x(nums[0])),
            "skewY" if nums.len() == 1 => Some(Affine::skew_y(nums[0])),
            _ => None,
        };
        if let Some(f) = f {
            // `f.then(acc)`: the new function applies in the
            // coordinate system `acc` already established — the
            // SVG left-to-right order.
            acc = f.then(acc);
        }
    }
    acc
}

/// The colour at parameter `t` over a stop list (same algorithm
/// as [`pluto_css::Gradient::color_at`], over a bare slice).
/// Stops are clamped to the ends outside 0..=1; between stops the
/// colour interpolates linearly. Pure.
pub fn gradient_stops_color_at(stops: &[GradientStop], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let at = |s: GradientStop| {
        [
            s.color[0] as f32 / 255.0,
            s.color[1] as f32 / 255.0,
            s.color[2] as f32 / 255.0,
            s.color[3] as f32 / 255.0,
        ]
    };
    if stops.len() == 1 || t <= stops[0].offset {
        return at(stops[0]);
    }
    let last = stops.len() - 1;
    if t >= stops[last].offset {
        return at(stops[last]);
    }
    for w in stops.windows(2) {
        let (a, b) = (w[0], w[1]);
        if t >= a.offset && t <= b.offset {
            let span = (b.offset - a.offset).max(f32::EPSILON);
            let f = (t - a.offset) / span;
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
    at(stops[last])
}

/// Resolve a gradient length against a bounding box (for
/// `objectBoundingBox`) or the viewport (for `userSpaceOnUse`):
/// a fraction resolves against `object_min`/`object_size` in the
/// first case and against 0/`viewport_size` in the second; a user
/// value is itself.
fn resolve_gradient_length(
    len: SvgLength,
    units: GradientUnits,
    object_min: f32,
    object_size: f32,
    viewport_size: f32,
) -> f32 {
    match (len, units) {
        (SvgLength::Fraction(f), GradientUnits::ObjectBoundingBox) => object_min + f * object_size,
        (SvgLength::Fraction(f), GradientUnits::UserSpaceOnUse) => f * viewport_size,
        (SvgLength::User(u), _) => u,
    }
}

/// The colour of an SVG gradient at user-space point `p`, for a
/// shape whose bounding box (user units) is `bbox` and whose svg
/// viewport is `viewport`. `opacity` (the shape's fill/stroke
/// opacity product) scales the alpha. `gradientTransform` is
/// honoured (in the gradient's own space: normalised bbox space
/// for `objectBoundingBox`, user space for `userSpaceOnUse`).
/// Pure.
pub fn svg_gradient_color(
    def: &GradientDef,
    bbox: [f32; 4],
    viewport: (f32, f32),
    opacity: f32,
    p: [f32; 2],
) -> [f32; 4] {
    let inv = affine_inverse(def.transform).unwrap_or_else(Affine::identity);
    let t = match &def.kind {
        GradientDefKind::Linear { x1, y1, x2, y2 } => {
            let (p1, p2, pg) = match def.units {
                GradientUnits::ObjectBoundingBox => {
                    // gradientTransform lives in normalised bbox
                    // space: normalise the point first, then
                    // invert the transform, then measure against
                    // the (fraction) endpoints.
                    let pn = if bbox[2] > 0.0 && bbox[3] > 0.0 {
                        (
                            (p[0] - bbox[0]) / bbox[2],
                            (p[1] - bbox[1]) / bbox[3],
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let pg = inv.apply(pn.0, pn.1);
                    let f1 = match *x1 {
                        SvgLength::Fraction(f) => f,
                        SvgLength::User(u) => u,
                    };
                    let f1y = match *y1 {
                        SvgLength::Fraction(f) => f,
                        SvgLength::User(u) => u,
                    };
                    let f2 = match *x2 {
                        SvgLength::Fraction(f) => f,
                        SvgLength::User(u) => u,
                    };
                    let f2y = match *y2 {
                        SvgLength::Fraction(f) => f,
                        SvgLength::User(u) => u,
                    };
                    ((f1, f1y), (f2, f2y), pg)
                }
                GradientUnits::UserSpaceOnUse => {
                    let p1 = (
                        resolve_gradient_length(*x1, def.units, 0.0, 0.0, viewport.0),
                        resolve_gradient_length(*y1, def.units, 0.0, 0.0, viewport.1),
                    );
                    let p2 = (
                        resolve_gradient_length(*x2, def.units, 0.0, 0.0, viewport.0),
                        resolve_gradient_length(*y2, def.units, 0.0, 0.0, viewport.1),
                    );
                    let pg = inv.apply(p[0], p[1]);
                    (p1, p2, pg)
                }
            };
            let dx = p2.0 - p1.0;
            let dy = p2.1 - p1.1;
            let len2 = dx * dx + dy * dy;
            if len2 <= 0.0 {
                0.0
            } else {
                ((pg.0 - p1.0) * dx + (pg.1 - p1.1) * dy) / len2
            }
        }
        GradientDefKind::Radial { cx, cy, r } => {
            let (c, pg, radius) = match def.units {
                GradientUnits::ObjectBoundingBox => {
                    let cf = |l: SvgLength| match l {
                        SvgLength::Fraction(f) => f,
                        SvgLength::User(u) => u,
                    };
                    let pn = if bbox[2] > 0.0 && bbox[3] > 0.0 {
                        (
                            (p[0] - bbox[0]) / bbox[2],
                            (p[1] - bbox[1]) / bbox[3],
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let pg = inv.apply(pn.0, pn.1);
                    ((cf(*cx), cf(*cy)), pg, cf(*r))
                }
                GradientUnits::UserSpaceOnUse => {
                    let c = (
                        resolve_gradient_length(*cx, def.units, 0.0, 0.0, viewport.0),
                        resolve_gradient_length(*cy, def.units, 0.0, 0.0, viewport.1),
                    );
                    let radius = match *r {
                        SvgLength::Fraction(f) => {
                            // A % radius in userSpaceOnUse is a
                            // fraction of the viewport's normalised
                            // diagonal.
                            f * ((viewport.0 * viewport.0 + viewport.1 * viewport.1) / 2.0)
                                .sqrt()
                        }
                        SvgLength::User(u) => u,
                    };
                    (c, inv.apply(p[0], p[1]), radius)
                }
            };
            if radius <= 0.0 {
                0.0
            } else {
                let dx = pg.0 - c.0;
                let dy = pg.1 - c.1;
                (dx * dx + dy * dy).sqrt() / radius
            }
        }
    };
    let mut color = gradient_stops_color_at(&def.stops, t);
    color[3] *= opacity.clamp(0.0, 1.0);
    color
}

/// Parse one `<linearGradient>` / `<radialGradient>` element
/// into a [`GradientDef`]. Stops come from its `<stop>` children
/// (`offset`, `stop-color` attribute or computed `stop-color`,
/// `stop-opacity`); a def with no stops is `None`.
fn parse_gradient_def(
    dom: &Dom,
    id: DomId,
    styles: &SparseSecondaryMap<DomId, NodeStyle>,
) -> Option<GradientDef> {
    let tag = dom.tag(id)?;
    let units = match dom.attr(id, "gradientUnits") {
        Some(u) if u.trim() == "userSpaceOnUse" => GradientUnits::UserSpaceOnUse,
        _ => GradientUnits::ObjectBoundingBox,
    };
    let transform = dom
        .attr(id, "gradientTransform")
        .map(parse_svg_transform)
        .unwrap_or_else(Affine::identity);
    let gl = |name: &str| dom.attr(id, name).and_then(|v| parse_gradient_length(v, units));
    let kind = match tag {
        "linearGradient" => GradientDefKind::Linear {
            x1: gl("x1").unwrap_or(SvgLength::Fraction(0.0)),
            y1: gl("y1").unwrap_or(SvgLength::Fraction(0.0)),
            x2: gl("x2").unwrap_or(SvgLength::Fraction(1.0)),
            y2: gl("y2").unwrap_or(SvgLength::Fraction(0.0)),
        },
        "radialGradient" => GradientDefKind::Radial {
            cx: gl("cx").unwrap_or(SvgLength::Fraction(0.5)),
            cy: gl("cy").unwrap_or(SvgLength::Fraction(0.5)),
            r: gl("r").unwrap_or(SvgLength::Fraction(0.5)),
        },
        _ => return None,
    };
    let mut stops: Vec<GradientStop> = Vec::new();
    for &child in dom.children(id) {
        if dom.tag(child) != Some("stop") {
            continue;
        }
        let offset = dom
            .attr(child, "offset")
            .and_then(|v| {
                let s = v.trim();
                if let Some(num) = s.strip_suffix('%') {
                    num.trim().parse::<f32>().ok().map(|n| n / 100.0)
                } else {
                    s.parse::<f32>().ok()
                }
            })
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        // `stop-color` is a presentation attribute (Phase 1 folds
        // it into the computed style); the attribute wins when
        // present, the style is the fallback.
        let color = dom
            .attr(child, "stop-color")
            .and_then(|v| parse_css_color(v))
            .or_else(|| {
                styles
                    .get(child)
                    .map(|ns| ns.computed.stop_color)
                    .map(|c| [c[0], c[1], c[2], 255])
            })
            .unwrap_or([0, 0, 0, 255]);
        let opacity = dom
            .attr(child, "stop-opacity")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        stops.push(GradientStop {
            offset,
            color: [
                color[0],
                color[1],
                color[2],
                (color[3] as f32 * opacity) as u8,
            ],
        });
    }
    if stops.is_empty() {
        return None;
    }
    stops.sort_by(|a, b| a.offset.partial_cmp(&b.offset).unwrap_or(std::cmp::Ordering::Equal));
    Some(GradientDef {
        kind,
        units,
        transform,
        stops,
    })
}

/// Parse a CSS colour spelling (`#rgb`, `#rrggbb`, `red`, ...) as
/// used in `stop-color` / presentation attributes. Only the
/// spellings the style layer already understands are supported
/// here (hex + named colours via [`pluto_css::parse_value`]);
/// anything else is `None` and the caller falls back.
fn parse_css_color(value: &str) -> Option<[u8; 4]> {
    match pluto_css::parse_value(value) {
        pluto_css::Value::Color(r, g, b) => Some([r, g, b, 255]),
        pluto_css::Value::ColorA(r, g, b, a) => {
            Some([r, g, b, (a.clamp(0.0, 1.0) * 255.0) as u8])
        }
        _ => None,
    }
}

/// Apply an affine to every point of a path (group / element
/// `transform` attributes compose in user space).
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

/// Resolve one `fill` / `stroke` computed paint to an
/// [`SvgPaint`], given the gradient defs. `opacity` is the
/// element's `opacity` × `fill-opacity` / `stroke-opacity`
/// product (already resolved by the caller).
fn resolve_paint(
    paint: &Paint,
    style_color: [u8; 3],
    opacity: f32,
    gradients: &HashMap<String, GradientDef>,
) -> SvgPaint {
    match paint {
        Paint::None => SvgPaint::None,
        Paint::Color(rgb) => SvgPaint::Solid([
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
            opacity.clamp(0.0, 1.0),
        ]),
        Paint::CurrentColor => SvgPaint::Solid([
            style_color[0] as f32 / 255.0,
            style_color[1] as f32 / 255.0,
            style_color[2] as f32 / 255.0,
            opacity.clamp(0.0, 1.0),
        ]),
        Paint::Url(url) => {
            let id = url.trim().strip_prefix('#').unwrap_or(url.as_str());
            match gradients.get(id) {
                Some(def) => SvgPaint::Gradient(def.clone()),
                None => SvgPaint::None,
            }
        }
    }
}

/// Collect one `<svg>` subtree's paintable content. `viewport`
/// is the svg box's used size (the basis for `userSpaceOnUse`
/// percentages and for `%` geometry). Shapes are returned in
/// document order, in user units.
pub fn collect_svg_content(
    dom: &Dom,
    svg_id: DomId,
    styles: &SparseSecondaryMap<DomId, NodeStyle>,
    viewport: (f32, f32),
) -> SvgContent {
    // First pass: gradient defs by id (anywhere in the subtree,
    // including `<defs>`).
    let mut gradients: HashMap<String, GradientDef> = HashMap::new();
    collect_gradients(dom, svg_id, styles, &mut gradients);
    // Second pass: shapes, in document order, starting at the
    // root's children — the outermost <svg> is not a group (its
    // x/y/transform are layout concerns, not user-space offsets;
    // a nested <svg> is, and `walk_shapes` translates it).
    let mut shapes = Vec::new();
    let mut images = Vec::new();
    let mut use_stack: Vec<DomId> = Vec::new();
    for &child in dom.children(svg_id) {
        walk_shapes(
            dom,
            child,
            styles,
            viewport,
            &gradients,
            Affine::identity(),
            &mut shapes,
            &mut images,
            &mut use_stack,
        );
    }
    SvgContent {
        shapes,
        viewport,
        images,
    }
}

/// The `href` / `xlink:href` of an SVG element (`<use>`,
/// `<image>`, `<a>`): plain `href` wins, legacy `xlink:href` is
/// the fallback (the parser stores it under that spelling —
/// see `page.rs::find_link_href`).
fn svg_href(dom: &Dom, id: DomId) -> Option<String> {
    dom.attr(id, "href")
        .or_else(|| dom.attr(id, "xlink:href"))
        .map(|s| s.to_string())
}

/// First element in the document whose `id` attribute is `name`,
/// in arena order (the same-document lookup `<use href="#id">`
/// and gradient `url(#id)` resolution use — the parser keeps no
/// id index, so this is a linear scan; SVGs are small).
fn find_element_by_id(dom: &Dom, name: &str) -> Option<DomId> {
    dom.all_node_ids()
        .into_iter()
        .find(|&id| dom.attr(id, "id").is_some_and(|v| v == name))
}

fn collect_gradients(
    dom: &Dom,
    id: DomId,
    styles: &SparseSecondaryMap<DomId, NodeStyle>,
    out: &mut HashMap<String, GradientDef>,
) {
    if matches!(dom.tag(id), Some("linearGradient" | "radialGradient")) {
        if let Some(name) = dom.attr(id, "id") {
            if let Some(def) = parse_gradient_def(dom, id, styles) {
                out.entry(name.to_string()).or_insert(def);
            }
        }
    }
    for &child in dom.children(id) {
        collect_gradients(dom, child, styles, out);
    }
}

/// Tags whose subtrees never paint (deferred work, or
/// non-rendering elements). `<text>` is deferred (no SVG text
/// path yet); `<foreignObject>` paints through layout (its
/// children get boxes — Rec 10), not through the shape walk.
const NON_RENDERING: &[&str] = &[
    "defs",
    "title",
    "desc",
    "script",
    "style",
    "metadata",
    "linearGradient",
    "radialGradient",
    "clipPath",
    "mask",
    "pattern",
    "symbol",
    "text",
    "foreignObject",
];

fn walk_shapes(
    dom: &Dom,
    id: DomId,
    styles: &SparseSecondaryMap<DomId, NodeStyle>,
    viewport: (f32, f32),
    gradients: &HashMap<String, GradientDef>,
    transform: Affine,
    out: &mut Vec<SvgShape>,
    images: &mut Vec<SvgImage>,
    use_stack: &mut Vec<DomId>,
) {
    let Some(tag) = dom.tag(id) else {
        return;
    };
    // `display: none` hides the whole subtree.
    if let Some(ns) = styles.get(id)
        && ns.computed.display == DisplayKind::None
    {
        return;
    }
    // Stage 15 (M7 Phase 3, Rec 10): `<use href="#id">` clones
    // the referenced element into the tree (same-document only;
    // an external href is skipped — the loader path is a later
    // slice). A use referencing itself or an ancestor use's
    // target is a cycle: refused, never looped (`use_stack`
    // holds the targets currently being expanded).
    if tag == "use" {
        let Some(href) = svg_href(dom, id) else {
            return;
        };
        let Some(frag) = href.strip_prefix('#') else {
            return;
        };
        let Some(target) = find_element_by_id(dom, frag) else {
            return;
        };
        if target == id || use_stack.contains(&target) {
            return;
        }
        let x = attr_f32(dom, id, "x", 0.0, viewport.0);
        let y = attr_f32(dom, id, "y", 0.0, viewport.1);
        // Content transform: x/y translate first, then the
        // use's own `transform`, then the ancestors'.
        let mut t = transform;
        if let Some(tr) = dom.attr(id, "transform") {
            t = parse_svg_transform(tr).then(t);
        }
        t = Affine::translate(x, y).then(t);
        use_stack.push(target);
        walk_use_target(
            dom, target, styles, viewport, gradients, t, out, images, use_stack,
        );
        use_stack.pop();
        return;
    }
    // Stage 15 (M7 Phase 3, Rec 10): `<svg><image>` records its
    // rect + href; the display list resolves the href to a
    // resource id (the image pipeline fetched + decoded it under
    // the Stage 9 gates — see `collect_svg_content`'s caller).
    // The group/use transform is applied as a translate/scale
    // (exact for translate/scale; a rotated group shears the
    // rect — documented approximation).
    if tag == "image" {
        let Some(href) = svg_href(dom, id) else {
            return;
        };
        if href.trim().is_empty() {
            return;
        }
        let x = attr_f32(dom, id, "x", 0.0, viewport.0);
        let y = attr_f32(dom, id, "y", 0.0, viewport.1);
        let w = attr_f32(dom, id, "width", 0.0, viewport.0);
        let h = attr_f32(dom, id, "height", 0.0, viewport.1);
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let (tx, ty) = transform.apply(x, y);
        images.push(SvgImage {
            dom: id,
            x: tx,
            y: ty,
            width: w * transform.a.abs(),
            height: h * transform.d.abs(),
            href,
        });
        return;
    }
    if NON_RENDERING.contains(&tag) {
        return;
    }
    // Grouping elements: compose their `transform` and recurse.
    // A nested `<svg>` also translates by its `x`/`y` (its own
    // `viewBox` is ignored in this phase — documented).
    if matches!(tag, "g" | "a" | "switch" | "svg") {
        let mut t = transform;
        if tag == "svg" {
            let x = attr_f32(dom, id, "x", 0.0, viewport.0);
            let y = attr_f32(dom, id, "y", 0.0, viewport.1);
            if x != 0.0 || y != 0.0 {
                t = Affine::translate(x, y).then(t);
            }
        }
        if let Some(tr) = dom.attr(id, "transform") {
            t = parse_svg_transform(tr).then(t);
        }
        for &child in dom.children(id) {
            walk_shapes(dom, child, styles, viewport, gradients, t, out, images, use_stack);
        }
        return;
    }
    // Shape geometry, in user units. `%` lengths resolve against
    // the viewport (width basis for x-ish, height for y-ish, the
    // normalised diagonal for `r`).
    let diag = ((viewport.0 * viewport.0 + viewport.1 * viewport.1) / 2.0).sqrt();
    let path = match tag {
        "path" => match dom.attr(id, "d").map(parse_path_data) {
            Some(Ok(p)) => p,
            _ => return,
        },
        "rect" => {
            let x = attr_f32(dom, id, "x", 0.0, viewport.0);
            let y = attr_f32(dom, id, "y", 0.0, viewport.1);
            let w = attr_f32(dom, id, "width", 0.0, viewport.0);
            let h = attr_f32(dom, id, "height", 0.0, viewport.1);
            let rx = attr_f32(dom, id, "rx", 0.0, viewport.0);
            let ry = attr_f32(dom, id, "ry", rx, viewport.1);
            path_from_rect(x, y, w, h, rx, ry)
        }
        "circle" => {
            let cx = attr_f32(dom, id, "cx", 0.0, viewport.0);
            let cy = attr_f32(dom, id, "cy", 0.0, viewport.1);
            let r = attr_f32(dom, id, "r", 0.0, diag);
            path_from_circle(cx, cy, r)
        }
        "ellipse" => {
            let cx = attr_f32(dom, id, "cx", 0.0, viewport.0);
            let cy = attr_f32(dom, id, "cy", 0.0, viewport.1);
            let rx = attr_f32(dom, id, "rx", 0.0, viewport.0);
            let ry = attr_f32(dom, id, "ry", 0.0, viewport.1);
            path_from_ellipse(cx, cy, rx, ry)
        }
        "line" => {
            let x1 = attr_f32(dom, id, "x1", 0.0, viewport.0);
            let y1 = attr_f32(dom, id, "y1", 0.0, viewport.1);
            let x2 = attr_f32(dom, id, "x2", 0.0, viewport.0);
            let y2 = attr_f32(dom, id, "y2", 0.0, viewport.1);
            path_from_line(x1, y1, x2, y2)
        }
        "polyline" | "polygon" => match dom.attr(id, "points").map(|p| {
            path_from_points(p, tag == "polygon")
        }) {
            Some(Ok(p)) => p,
            _ => return,
        },
        _ => return,
    };
    if path.is_empty() {
        return;
    }
    // The element's own `transform` (composed after the
    // ancestors': the element's local transform applies first).
    let mut path = path;
    let mut t = transform;
    if let Some(tr) = dom.attr(id, "transform") {
        t = parse_svg_transform(tr).then(t);
    }
    if t != Affine::identity() {
        path = transform_path(&path, t);
    }
    // Paints from the computed style (Phase 1 folded presentation
    // attributes in); no style entry means the initial values
    // (fill black, stroke none, width 1).
    let style = styles.get(id).map(|ns| ns.computed.clone());
    let (fill_paint, stroke_paint, stroke_width, style_color) = match &style {
        Some(s) => (
            s.fill.clone(),
            s.stroke.clone(),
            s.stroke_width,
            s.color,
        ),
        None => (
            Paint::Color([0, 0, 0]),
            Paint::None,
            1.0,
            [0, 0, 0],
        ),
    };
    let opacity = attr_f32(dom, id, "opacity", 1.0, 1.0);
    let fill_opacity = opacity * attr_f32(dom, id, "fill-opacity", 1.0, 1.0);
    let stroke_opacity = opacity * attr_f32(dom, id, "stroke-opacity", 1.0, 1.0);
    let fill = resolve_paint(&fill_paint, style_color, fill_opacity, gradients);
    let stroke = resolve_paint(&stroke_paint, style_color, stroke_opacity, gradients);
    if matches!(fill, SvgPaint::None) && matches!(stroke, SvgPaint::None) {
        return;
    }
    out.push(SvgShape {
        path,
        fill,
        stroke,
        fill_opacity,
        stroke_opacity,
        stroke_width,
        fill_rule: parse_fill_rule(dom.attr(id, "fill-rule")),
        cap: parse_line_cap(dom.attr(id, "stroke-linecap")),
        join: parse_line_join(dom.attr(id, "stroke-linejoin")),
        miter_limit: dom
            .attr(id, "stroke-miterlimit")
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(4.0),
    });
}

/// Walk a `<use>` target as content: containers (`symbol` /
/// `defs` / `g` / nested `<svg>`) expand like groups (a
/// `<symbol>` reached only through `<use>` is a group here, not
/// a non-rendering element); a shape, a nested `<use>`, or an
/// `<image>` goes through the main walk (which re-checks the
/// cycle stack). The target's own `transform` / nested-svg
/// `x`/`y` compose the same way as a group's.
#[allow(clippy::too_many_arguments)]
fn walk_use_target(
    dom: &Dom,
    id: DomId,
    styles: &SparseSecondaryMap<DomId, NodeStyle>,
    viewport: (f32, f32),
    gradients: &HashMap<String, GradientDef>,
    transform: Affine,
    out: &mut Vec<SvgShape>,
    images: &mut Vec<SvgImage>,
    use_stack: &mut Vec<DomId>,
) {
    let Some(tag) = dom.tag(id) else {
        return;
    };
    if matches!(tag, "symbol" | "defs" | "g" | "a" | "switch" | "svg") {
        let mut t = transform;
        if tag == "svg" {
            let x = attr_f32(dom, id, "x", 0.0, viewport.0);
            let y = attr_f32(dom, id, "y", 0.0, viewport.1);
            if x != 0.0 || y != 0.0 {
                t = Affine::translate(x, y).then(t);
            }
        }
        if let Some(tr) = dom.attr(id, "transform") {
            t = parse_svg_transform(tr).then(t);
        }
        for &child in dom.children(id) {
            walk_shapes(dom, child, styles, viewport, gradients, t, out, images, use_stack);
        }
        return;
    }
    walk_shapes(dom, id, styles, viewport, gradients, transform, out, images, use_stack);
}

/// True when any shape needs the rasteriser: a radial gradient,
/// or a linear gradient with more than two stops (per-vertex
/// colours are exact only for two stops). Everything else goes
/// through the vector pipeline.
pub fn svg_needs_raster(content: &SvgContent) -> bool {
    fn needs(paint: &SvgPaint) -> bool {
        match paint {
            SvgPaint::Gradient(def) => match &def.kind {
                GradientDefKind::Radial { .. } => true,
                GradientDefKind::Linear { .. } => def.stops.len() > 2,
            },
            _ => false,
        }
    }
    content
        .shapes
        .iter()
        .any(|s| needs(&s.fill) || needs(&s.stroke))
}

/// The colour function for one shape's gradient paint: `p` is in
/// user units (the tessellator's space). `opacity` is the shape's
/// fill/stroke opacity product. Only linear gradients reach here
/// from the vector path (the caller routes the rest to the
/// rasteriser); radial gradients evaluate through the same
/// function for the rasteriser.
fn shape_gradient_color(
    def: &GradientDef,
    bbox: [f32; 4],
    viewport: (f32, f32),
    opacity: f32,
) -> impl Fn([f32; 2]) -> [f32; 4] + '_ {
    move |p| svg_gradient_color(def, bbox, viewport, opacity, p)
}

/// Tessellate every shape's fill + stroke into flat coloured
/// triangles, in user units. Gradient fills use per-vertex
/// colours (exact for two-stop linear gradients). A shape whose
/// paint is a gradient the vector path cannot express (radial,
/// >2 stops) is skipped here — the caller rasterises those.
/// Any tessellation error aborts the whole svg (the caller falls
/// back to the rasteriser, which tessellates with the same cap).
pub fn tessellate_svg_shapes(content: &SvgContent) -> Result<Vec<VectorVertex>, PathError> {
    let mut out = Vec::new();
    for shape in &content.shapes {
        out.extend(tessellate_svg_shape(shape, content.viewport)?);
    }
    Ok(out)
}

/// Tessellate one shape's fill + stroke into flat coloured
/// triangles, in user units. The vector path's half of
/// [`tessellate_svg_shapes`]: gradient fills use per-vertex
/// colours (exact for two-stop linear gradients). A shape whose
/// paint is a gradient the vector path cannot express (radial,
/// >2 stops) contributes nothing here — the caller rasterises
/// those. Any tessellation error aborts this shape (the caller
/// skips it and keeps the rest).
pub fn tessellate_svg_shape(
    shape: &SvgShape,
    viewport: (f32, f32),
) -> Result<Vec<VectorVertex>, PathError> {
    let mut out = Vec::new();
    {
        let bbox = match shape.path.bounds() {
            Some(b) => b,
            None => return Ok(out),
        };
        match &shape.fill {
            SvgPaint::None => {}
            SvgPaint::Solid(color) => {
                let color = *color;
                out.extend(tessellate_fill(&shape.path, shape.fill_rule, move |_| color)?);
            }
            SvgPaint::Gradient(def) => {
                // The vector path only carries two-stop linear
                // gradients; anything else is the rasteriser's.
                if matches!(def.kind, GradientDefKind::Linear { .. }) && def.stops.len() <= 2 {
                    let f = shape_gradient_color(
                        def,
                        bbox,
                        viewport,
                        shape.fill_opacity,
                    );
                    out.extend(tessellate_fill(&shape.path, shape.fill_rule, f)?);
                }
            }
        }
        match &shape.stroke {
            SvgPaint::None => {}
            SvgPaint::Solid(color) => {
                let color = *color;
                out.extend(tessellate_stroke(
                    &shape.path,
                    shape.stroke_width,
                    shape.cap,
                    shape.join,
                    shape.miter_limit,
                    move |_| color,
                )?);
            }
            SvgPaint::Gradient(def) => {
                if matches!(def.kind, GradientDefKind::Linear { .. }) && def.stops.len() <= 2 {
                    let f = shape_gradient_color(
                        def,
                        bbox,
                        viewport,
                        shape.stroke_opacity,
                    );
                    out.extend(tessellate_stroke(
                        &shape.path,
                        shape.stroke_width,
                        shape.cap,
                        shape.join,
                        shape.miter_limit,
                        f,
                    )?);
                }
            }
        }
    }
    Ok(out)
}

/// Rasterise one svg's content into an RGBA bitmap of `width` ×
/// `height` (the box's used size, in whole pixels). `view` maps
/// user units to bitmap pixels (the viewBox transform). `None`
/// when the bitmap is empty, oversize (> [`MAX_RASTER_PIXELS`]
/// px), or a shape fails to tessellate. Shapes paint in document
/// order (later shapes over earlier ones).
pub fn rasterize_svg_content(
    content: &SvgContent,
    view: Affine,
    width: u32,
    height: u32,
) -> Option<DecodedImage> {
    if width == 0 || height == 0 {
        return None;
    }
    if (width as usize).checked_mul(height as usize)? > MAX_RASTER_PIXELS {
        return None;
    }
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    let inv_view = affine_inverse(view)?;
    for shape in &content.shapes {
        rasterize_shape(
            &mut rgba,
            shape,
            content.viewport,
            view,
            inv_view,
            width,
            height,
        )
        .ok()?;
    }
    Some(DecodedImage {
        width,
        height,
        rgba,
    })
}

/// Rasterise one shape (fill then stroke) into the bitmap.
/// Solid paints composite directly from their tessellated
/// triangles; gradient paints rasterise to a coverage mask and
/// evaluate the gradient per covered pixel — per-vertex colours
/// cannot express a radial gradient (every corner of a rect is
/// at offset ~1, so the centre would come out the end-stop
/// colour), and a multi-stop linear gradient is only exact
/// per-pixel too.
fn rasterize_shape(
    rgba: &mut [u8],
    shape: &SvgShape,
    viewport: (f32, f32),
    view: Affine,
    inv_view: Affine,
    width: u32,
    height: u32,
) -> Result<(), PathError> {
    let Some(bbox) = shape.path.bounds() else {
        return Ok(());
    };
    let map_view = |tris: &mut Vec<VectorVertex>| {
        for v in tris.iter_mut() {
            let (x, y) = view.apply(v.position[0], v.position[1]);
            v.position = [x, y];
        }
    };
    let mut paint = |tris: Vec<VectorVertex>,
                     paint: &SvgPaint,
                     opacity: f32|
     -> Result<(), PathError> {
        match paint {
            SvgPaint::None => Ok(()),
            SvgPaint::Solid(_) => {
                let mut tris = tris;
                map_view(&mut tris);
                rasterize_triangles_into(rgba, &tris, width, height);
                Ok(())
            }
            SvgPaint::Gradient(def) => {
                let mut tris = tris;
                map_view(&mut tris);
                let mask = rasterize_coverage(&tris, width, height);
                composite_gradient(rgba, &mask, def, bbox, viewport, opacity, inv_view, width);
                Ok(())
            }
        }
    };
    match &shape.fill {
        SvgPaint::None => {}
        SvgPaint::Solid(color) => {
            let color = *color;
            let tris = tessellate_fill(&shape.path, shape.fill_rule, move |_| color)?;
            paint(tris, &shape.fill, shape.fill_opacity)?;
        }
        SvgPaint::Gradient(_) => {
            let tris =
                tessellate_fill(&shape.path, shape.fill_rule, |_| [1.0, 1.0, 1.0, 1.0])?;
            paint(tris, &shape.fill, shape.fill_opacity)?;
        }
    }
    match &shape.stroke {
        SvgPaint::None => {}
        SvgPaint::Solid(color) => {
            let color = *color;
            let tris = tessellate_stroke(
                &shape.path,
                shape.stroke_width,
                shape.cap,
                shape.join,
                shape.miter_limit,
                move |_| color,
            )?;
            paint(tris, &shape.stroke, shape.stroke_opacity)?;
        }
        SvgPaint::Gradient(_) => {
            let tris = tessellate_stroke(
                &shape.path,
                shape.stroke_width,
                shape.cap,
                shape.join,
                shape.miter_limit,
                |_| [1.0, 1.0, 1.0, 1.0],
            )?;
            paint(tris, &shape.stroke, shape.stroke_opacity)?;
        }
    }
    Ok(())
}

/// Source-over a gradient onto the bitmap wherever `mask` is
/// covered, evaluating the gradient at each pixel's user-space
/// position (the pixel centre mapped back through `inv_view`).
/// `opacity` is the shape's fill/stroke opacity product.
#[allow(clippy::too_many_arguments)]
fn composite_gradient(
    rgba: &mut [u8],
    mask: &[u8],
    def: &GradientDef,
    bbox: [f32; 4],
    viewport: (f32, f32),
    opacity: f32,
    inv_view: Affine,
    width: u32,
) {
    let w = width as usize;
    for (i, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let x = (i % w) as f32 + 0.5;
        let y = (i / w) as f32 + 0.5;
        let (ux, uy) = inv_view.apply(x, y);
        let c = svg_gradient_color(def, bbox, viewport, opacity, [ux, uy]);
        let sa = (c[3] * m as f32 / 255.0).clamp(0.0, 1.0);
        if sa <= 0.0 {
            continue;
        }
        let idx = i * 4;
        let da = rgba[idx + 3] as f32 / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            continue;
        }
        for j in 0..3 {
            let sc = c[j].clamp(0.0, 1.0);
            let dc = rgba[idx + j] as f32 / 255.0;
            let out = (sc * sa + dc * da * (1.0 - sa)) / out_a;
            rgba[idx + j] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        rgba[idx + 3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
}

/// True when `bytes` look like an SVG document (Stage 15, M7
/// Phase 3, Rec 12): after a UTF-8 BOM / whitespace / XML
/// declaration / comments, the first element is `<svg`. The
/// `image` crate has no SVG decoder, so [`crate::image::
/// decode_image`] sniffs with this first and routes SVG bytes
/// to [`decode_svg_image`].
pub fn looks_like_svg(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);
    let mut rest = text.trim_start();
    if let Some(stripped) = rest.strip_prefix('\u{feff}') {
        rest = stripped.trim_start();
    }
    // Skip an XML declaration and comments/doctype before the root.
    loop {
        if let Some(after) = rest.strip_prefix("<?xml") {
            match after.find("?>") {
                Some(end) => rest = after[end + 2..].trim_start(),
                None => return false,
            }
        } else if rest.starts_with("<!--") {
            match rest.find("-->") {
                Some(end) => rest = rest[end + 3..].trim_start(),
                None => return false,
            }
        } else if rest.starts_with("<!DOCTYPE") || rest.starts_with("<!doctype") {
            match rest.find('>') {
                Some(end) => rest = rest[end + 1..].trim_start(),
                None => return false,
            }
        } else {
            break;
        }
    }
    rest.starts_with("<svg") && rest[4..].starts_with(|c: char| c.is_whitespace() || c == '>')
}

/// Decode standalone SVG bytes (an `<img src=*.svg>`, Rec 12)
/// into an RGBA bitmap at the SVG's intrinsic size: `width` /
/// `height` attributes when present, else the `viewBox` size,
/// else the 300×150 default. The bytes go through the same
/// `collect_svg_content` + rasteriser as inline SVG (no styles
/// — an external SVG has no document stylesheet, so the initial
/// values apply); `<foreignObject>` content is skipped in this
/// route (its boxes belong to a document layout, which an
/// external image has none of). The full Stage 9 fetch gates
/// already ran before the bytes reached here — this is decode
/// only, never a fetch bypass.
pub fn decode_svg_image(bytes: &[u8]) -> Result<DecodedImage, crate::image::ImageError> {
    use crate::image::ImageError;
    let text = String::from_utf8_lossy(bytes);
    let dom = pluto_html::parse(&text);
    let svg_id = dom
        .all_node_ids()
        .into_iter()
        .find(|&id| dom.tag(id) == Some("svg"))
        .ok_or(ImageError::Empty)?;
    let parse_len = |name: &str| -> Option<f32> {
        dom.attr(svg_id, name)
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| *v > 0.0)
    };
    let view_box = dom.attr(svg_id, "viewBox").and_then(parse_view_box);
    let w = parse_len("width")
        .or_else(|| view_box.map(|vb| vb[2]))
        .unwrap_or(300.0);
    let h = parse_len("height")
        .or_else(|| view_box.map(|vb| vb[3]))
        .unwrap_or(150.0);
    let (w, h) = (w.round().max(1.0) as u32, h.round().max(1.0) as u32);
    if (w as usize).checked_mul(h as usize).unwrap_or(usize::MAX) > MAX_RASTER_PIXELS {
        return Err(ImageError::TooLarge(
            (w as usize) * (h as usize) * 4,
        ));
    }
    let styles = SparseSecondaryMap::default();
    let content = collect_svg_content(&dom, svg_id, &styles, (w as f32, h as f32));
    let view = view_box_transform(
        w as f32,
        h as f32,
        view_box,
        dom.attr(svg_id, "preserveAspectRatio"),
    );
    match rasterize_svg_content(&content, view, w, h) {
        Some(img) => {
            if img.rgba.len() > crate::image::IMAGE_CACHE_BYTE_CAP {
                return Err(ImageError::TooLarge(img.rgba.len()));
            }
            Ok(img)
        }
        None => Err(ImageError::Empty),
    }
}

/// A fingerprint of one CSS gradient value (kind + stops):
/// the raster cache keys a radial `background-image` on it (plus
/// the element id and bitmap size), so a changed gradient
/// re-rasterises and an unchanged one reuses its bitmap. FNV-1a
/// over the raw bits — deterministic across runs, which is what
/// a cache key needs.
pub fn gradient_fingerprint(gradient: &Gradient) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut mix = |bits: u64| {
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    match &gradient.kind {
        GradientKind::Linear { angle_deg } => {
            mix(0);
            mix(angle_deg.to_bits() as u64);
        }
        GradientKind::Radial {
            center,
            shape,
            extent,
        } => {
            mix(1);
            mix(center.0.to_bits() as u64);
            mix(center.1.to_bits() as u64);
            mix(match shape {
                RadialShape::Circle => 0,
                RadialShape::Ellipse => 1,
            });
            mix(match extent {
                RadialExtent::ClosestSide => 0,
                RadialExtent::FarthestSide => 1,
                RadialExtent::ClosestCorner => 2,
                RadialExtent::FarthestCorner => 3,
            });
        }
    }
    for stop in &gradient.stops {
        mix(stop.offset.to_bits() as u64);
        for &b in &stop.color {
            mix(b as u64);
        }
    }
    hash
}

/// A fingerprint of one `<svg>` subtree: tags + the SVG
/// attributes that affect paint, in document order. The raster
/// cache keys on it (plus the element id and bitmap size), so a
/// JS mutation that changes a `d` or a `fill` re-rasterises, and
/// an unchanged subtree reuses its bitmap. `Dom` exposes no
/// attribute enumeration, so the fingerprint hashes a fixed list
/// of the attribute names paint reads.
pub fn svg_fingerprint(dom: &Dom, svg_id: DomId) -> u64 {
    const ATTRS: &[&str] = &[
        "d", "points", "cx", "cy", "r", "rx", "ry", "x", "y", "width", "height", "x1", "y1",
        "x2", "y2", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin",
        "stroke-miterlimit", "fill-rule", "fill-opacity", "stroke-opacity", "opacity",
        "transform", "viewBox", "preserveAspectRatio", "stop-color", "stop-opacity", "offset",
        "gradientTransform", "gradientUnits", "href", "xlink:href", "id", "class", "style", "display",
    ];
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    fn walk(dom: &Dom, id: DomId, mix: &mut impl FnMut(&[u8])) {
        if let Some(tag) = dom.tag(id) {
            mix(tag.as_bytes());
        }
        for name in ATTRS {
            if let Some(v) = dom.attr(id, name) {
                mix(name.as_bytes());
                mix(v.as_bytes());
            }
        }
        for &child in dom.children(id) {
            walk(dom, child, mix);
        }
    }
    walk(dom, svg_id, &mut mix);
    hash
}

/// The worker-side memo for rasterised SVG bitmaps, owned by the
/// layout actor: keyed by (element, bitmap size, content
/// fingerprint), so a scroll re-collect reuses the bitmap (and
/// the tab thread skips the `ResourceReady` re-send — the id is
/// deterministic, so the shell already holds the pixels), while
/// a resize or a content change re-rasterises. Bounded: past 128
/// entries the whole cache clears (simple, and a page with 128
/// live SVGs re-rasterises once — bounded work either way).
#[derive(Debug, Default)]
pub struct SvgRasterCache {
    map: HashMap<(u32, u32, u32, u64), (ResourceId, Resource)>,
}

impl SvgRasterCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn key(dom: DomId, width: u32, height: u32, fingerprint: u64) -> (u32, u32, u32, u64) {
        (
            dom.data().as_ffi() as u32,
            width,
            height,
            fingerprint,
        )
    }

    pub fn get(
        &self,
        dom: DomId,
        width: u32,
        height: u32,
        fingerprint: u64,
    ) -> Option<(ResourceId, &Resource)> {
        self.map
            .get(&Self::key(dom, width, height, fingerprint))
            .map(|(id, r)| (*id, r))
    }

    pub fn insert(
        &mut self,
        dom: DomId,
        width: u32,
        height: u32,
        fingerprint: u64,
        id: ResourceId,
        resource: Resource,
    ) {
        if self.map.len() >= 128 {
            self.map.clear();
        }
        self.map
            .insert(Self::key(dom, width, height, fingerprint), (id, resource));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::parse_value;
    use pluto_html::parse as parse_html;
    use pluto_layout::{ElementState, LayoutTree, Theme};

    fn svg_dom(html: &str) -> (Dom, DomId) {
        let dom = parse_html(html);
        // Find the first <svg> element (the document root's child
        // in these test documents).
        let mut found = None;
        fn walk(dom: &Dom, id: DomId, found: &mut Option<DomId>) {
            if found.is_some() {
                return;
            }
            if dom.tag(id) == Some("svg") {
                *found = Some(id);
                return;
            }
            for &child in dom.children(id) {
                walk(dom, child, found);
            }
        }
        let root = dom
            .all_node_ids()
            .into_iter()
            .find(|&id| dom.tag(id) == Some("html"))
            .expect("html root");
        walk(&dom, root, &mut found);
        (dom, found.expect("an <svg> element"))
    }

    fn layout_for(dom: &Dom) -> LayoutTree {
        let sheet = pluto_css::Stylesheet::parse("");
        LayoutTree::build_with_state(dom, &sheet, 800.0, 600.0, &ElementState::default(), Theme::Light)
            .expect("layout builds")
    }

    #[test]
    fn view_box_transform_meet_centers_and_slice_crops() {
        // 200x100 box, 100x100 viewBox, meet: scale 1, centered
        // horizontally (50px margins each side).
        let t = view_box_transform(200.0, 100.0, Some([0.0, 0.0, 100.0, 100.0]), None);
        assert_eq!(t.apply(0.0, 0.0), (50.0, 0.0));
        assert_eq!(t.apply(100.0, 100.0), (150.0, 100.0));
        // slice: scale 2, centered vertically (-50px top).
        let t = view_box_transform(
            200.0,
            100.0,
            Some([0.0, 0.0, 100.0, 100.0]),
            Some("xMidYMid slice"),
        );
        assert_eq!(t.apply(0.0, 0.0), (0.0, -50.0));
        // none: non-uniform scale, no centering.
        let t = view_box_transform(
            200.0,
            100.0,
            Some([0.0, 0.0, 100.0, 100.0]),
            Some("none"),
        );
        assert_eq!(t.apply(100.0, 100.0), (200.0, 100.0));
        // No viewBox: identity.
        assert_eq!(
            view_box_transform(200.0, 100.0, None, None),
            Affine::identity()
        );
        // viewBox with an offset: scale 2, and the viewBox
        // origin maps to the viewport origin (no slack to
        // centre within, so the alignment is a no-op).
        let t = view_box_transform(100.0, 100.0, Some([10.0, 20.0, 50.0, 50.0]), None);
        assert_eq!(t.apply(10.0, 20.0), (0.0, 0.0));
        assert_eq!(t.apply(60.0, 70.0), (100.0, 100.0));
    }

    #[test]
    fn svg_transform_parses_and_composes_left_to_right() {
        // translate(10, 20) then scale(2): a point is scaled
        // first (in the translated system), then translated.
        let t = parse_svg_transform("translate(10 20) scale(2)");
        assert_eq!(t.apply(1.0, 1.0), (12.0, 22.0));
        let r = parse_svg_transform("rotate(90)");
        let (x, y) = r.apply(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-5 && (y - 1.0).abs() < 1e-5);
        let m = parse_svg_transform("matrix(1 0 0 1 5 6)");
        assert_eq!(m.apply(0.0, 0.0), (5.0, 6.0));
        // Garbage is the identity, never an error.
        assert_eq!(parse_svg_transform("bogus(1 2"), Affine::identity());
    }

    #[test]
    fn collect_svg_content_resolves_shapes_and_paints() {
        let html = r#"<svg viewBox="0 0 100 100" width="100" height="100">
            <rect x="10" y="10" width="80" height="80" fill="red"/>
            <circle cx="50" cy="50" r="20" fill="blue" stroke="black" stroke-width="2"/>
            <path d="M 0 0 L 10 10" stroke="green" fill="none"/>
            <g transform="translate(5 5)"><rect x="0" y="0" width="10" height="10" fill="yellow"/></g>
        </svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert_eq!(content.shapes.len(), 4);
        // The rect: solid red fill, no stroke.
        assert!(matches!(content.shapes[0].fill, SvgPaint::Solid(c) if c[0] > 0.9));
        assert!(matches!(content.shapes[0].stroke, SvgPaint::None));
        // The circle: blue fill + black stroke, width 2.
        assert!(matches!(content.shapes[1].stroke, SvgPaint::Solid(_)));
        assert_eq!(content.shapes[1].stroke_width, 2.0);
        // The path: no fill, green stroke.
        assert!(matches!(content.shapes[2].fill, SvgPaint::None));
        assert!(matches!(content.shapes[2].stroke, SvgPaint::Solid(c) if c[1] > 0.3));
        // The grouped rect: translated by (5, 5).
        let bounds = content.shapes[3].path.bounds().expect("bounds");
        assert!((bounds[0] - 5.0).abs() < 1e-5 && (bounds[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn gradient_def_parses_and_evaluates() {
        let html = r#"<svg viewBox="0 0 100 100" width="100" height="100"><defs>
            <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stop-color="red"/>
                <stop offset="1" stop-color="blue"/>
            </linearGradient>
            <radialGradient id="g2" cx="0.5" cy="0.5" r="0.5">
                <stop offset="0" stop-color="white"/>
                <stop offset="1" stop-color="black"/>
            </radialGradient>
        </defs>
        <rect x="0" y="0" width="100" height="100" fill="url(#g1)"/>
        <circle cx="50" cy="50" r="40" fill="url(#g2)"/>
        </svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert_eq!(content.shapes.len(), 2);
        // The linear gradient is vector-eligible (2 stops); the
        // radial one is what forces the raster route below.
        assert!(!svg_needs_raster(&SvgContent {
            shapes: vec![content.shapes[0].clone()],
            viewport: content.viewport,
            images: Vec::new(),
        }));
        match &content.shapes[0].fill {
            SvgPaint::Gradient(def) => {
                assert_eq!(def.stops.len(), 2);
                let bbox = content.shapes[0].path.bounds().unwrap();
                let left = svg_gradient_color(def, bbox, (100.0, 100.0), 1.0, [0.0, 50.0]);
                let right = svg_gradient_color(def, bbox, (100.0, 100.0), 1.0, [100.0, 50.0]);
                assert!(left[0] > 0.9 && left[2] < 0.1, "left is red: {left:?}");
                assert!(right[2] > 0.9 && right[0] < 0.1, "right is blue: {right:?}");
            }
            other => panic!("rect fill should be a gradient, got {other:?}"),
        }
        // The radial gradient forces the raster route.
        assert!(matches!(content.shapes[1].fill, SvgPaint::Gradient(_)));
        assert!(svg_needs_raster(&content));
        // A missing gradient id paints nothing.
        let html2 = r#"<svg viewBox="0 0 10 10"><rect x="0" y="0" width="10" height="10" fill="url(#nope)"/></svg>"#;
        let (dom2, svg2) = svg_dom(html2);
        let tree2 = layout_for(&dom2);
        let content2 = collect_svg_content(&dom2, svg2, &tree2.node_styles, (10.0, 10.0));
        assert!(content2.shapes.is_empty(), "unresolvable url paints nothing");
    }

    #[test]
    fn rasterize_known_svg_produces_expected_pixels() {
        // A GitHub-style icon set: a star path, a circle, and a
        // rect — rasterised at 16x16, the icon size.
        let html = r#"<svg viewBox="0 0 16 16" width="16" height="16">
            <path d="M8 1 L9.6 5.6 L14.5 5.8 L10.8 9 L12.2 13.8 L8 11 L3.8 13.8 L5.2 9 L1.5 5.8 L6.4 5.6 Z" fill="black"/>
            <circle cx="13" cy="3" r="2" fill="red"/>
        </svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (16.0, 16.0));
        assert_eq!(content.shapes.len(), 2);
        assert!(!svg_needs_raster(&content));
        let view = view_box_transform(16.0, 16.0, Some([0.0, 0.0, 16.0, 16.0]), None);
        let img = rasterize_svg_content(&content, view, 16, 16).expect("rasterises");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.rgba.len(), 16 * 16 * 4);
        // The star's centre (8, 8) is black; the red dot's centre
        // (13, 3) is red; a corner is transparent.
        let px = |x: usize, y: usize| -> [u8; 4] {
            let i = (y * 16 + x) * 4;
            [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2], img.rgba[i + 3]]
        };
        let centre = px(8, 8);
        assert_eq!(centre[3], 255, "star centre is opaque: {centre:?}");
        assert!(centre[0] < 40 && centre[1] < 40 && centre[2] < 40);
        let dot = px(13, 3);
        assert!(dot[0] > 200 && dot[3] == 255, "red dot: {dot:?}");
        assert_eq!(px(0, 0)[3], 0, "corner is transparent");
        // Non-empty: a real icon has both painted and clear pixels.
        let painted = img.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(painted > 10 && painted < 256, "painted pixels: {painted}");
    }

    #[test]
    fn rasterize_radial_gradient_svg() {
        let html = r#"<svg viewBox="0 0 100 100" width="100" height="100"><defs>
            <radialGradient id="rg" cx="0.5" cy="0.5" r="0.5">
                <stop offset="0" stop-color="white"/>
                <stop offset="1" stop-color="black"/>
            </radialGradient>
        </defs><rect x="0" y="0" width="100" height="100" fill="url(#rg)"/></svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert!(svg_needs_raster(&content));
        let view = view_box_transform(100.0, 100.0, Some([0.0, 0.0, 100.0, 100.0]), None);
        let img = rasterize_svg_content(&content, view, 100, 100).expect("rasterises");
        let centre = (50 * 100 + 50) * 4;
        assert!(img.rgba[centre] > 200, "radial centre is white");
        assert!(img.rgba[0] < 55, "radial corner is black");
    }

    #[test]
    fn raster_refuses_oversize_bitmaps() {
        let html = r#"<svg viewBox="0 0 10 10"><rect x="0" y="0" width="10" height="10" fill="red"/></svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (10.0, 10.0));
        let view = Affine::identity();
        // 5000x5000 = 25M px > 16.7M cap.
        assert!(rasterize_svg_content(&content, view, 5000, 5000).is_none());
        assert!(rasterize_svg_content(&content, view, 10, 10).is_some());
    }

    #[test]
    fn svg_fingerprint_changes_with_content() {
        let a = r#"<svg viewBox="0 0 10 10"><rect x="0" y="0" width="5" height="5" fill="red"/></svg>"#;
        let b = r#"<svg viewBox="0 0 10 10"><rect x="0" y="0" width="5" height="5" fill="blue"/></svg>"#;
        let (dom_a, svg_a) = svg_dom(a);
        let (dom_b, svg_b) = svg_dom(b);
        let fa = svg_fingerprint(&dom_a, svg_a);
        let fb = svg_fingerprint(&dom_b, svg_b);
        assert_ne!(fa, fb);
        assert_eq!(fa, svg_fingerprint(&dom_a, svg_a), "deterministic");
    }

    #[test]
    fn parse_value_gradient_is_shared_with_css() {
        // The CSS gradient type parses through the same value
        // parser the layout cascade uses (Rec 7 wiring check).
        assert!(matches!(
            parse_value("linear-gradient(90deg, red, blue)"),
            pluto_css::Value::Gradient(_)
        ));
    }

    // ---- Stage 15 (M7 Phase 3, Rec 10/12) ----

    #[test]
    fn use_clones_the_referenced_shape() {
        let html = r##"<svg viewBox="0 0 100 100">
            <defs><rect id="box" x="10" y="10" width="20" height="20" fill="red"/></defs>
            <use href="#box" x="30" y="0"/>
        </svg>"##;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        // The def itself paints nothing; the <use> clones it once.
        assert_eq!(content.shapes.len(), 1, "one cloned rect");
        let bbox = content.shapes[0].path.bounds().unwrap();
        assert_eq!((bbox[0], bbox[1]), (40.0, 10.0), "use x/y translate the clone");
    }

    #[test]
    fn use_with_xlink_href_and_symbol_expands() {
        let html = r##"<svg viewBox="0 0 100 100">
            <symbol id="dot"><circle cx="5" cy="5" r="5" fill="blue"/></symbol>
            <use xlink:href="#dot" x="10" y="10"/>
        </svg>"##;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert_eq!(content.shapes.len(), 1);
        let bbox = content.shapes[0].path.bounds().unwrap();
        assert_eq!((bbox[0], bbox[1]), (10.0, 10.0));
    }

    #[test]
    fn use_cycle_is_refused_not_looped() {
        // A <use> referencing itself, and two <use>s referencing
        // each other: both must collect zero shapes and return.
        let html = r##"<svg viewBox="0 0 100 100">
            <use id="self" href="#self" x="0" y="0"/>
            <use id="a" href="#b" x="0" y="0"/>
            <use id="b" href="#a" x="0" y="0"/>
        </svg>"##;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert!(content.shapes.is_empty(), "cyclic uses paint nothing");
    }

    #[test]
    fn use_with_external_href_is_skipped() {
        let html = r#"<svg viewBox="0 0 100 100">
            <use href="icons.svg#box" x="0" y="0"/>
        </svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert!(content.shapes.is_empty(), "external <use> is same-document-only");
    }

    #[test]
    fn svg_image_is_collected_with_rect_and_href() {
        let html = r#"<svg viewBox="0 0 100 100">
            <image href="photo.png" x="5" y="6" width="40" height="30"/>
            <image xlink:href="other.png" x="0" y="0" width="10" height="10"/>
        </svg>"#;
        let (dom, svg) = svg_dom(html);
        let tree = layout_for(&dom);
        let content = collect_svg_content(&dom, svg, &tree.node_styles, (100.0, 100.0));
        assert_eq!(content.images.len(), 2);
        assert_eq!(content.images[0].href, "photo.png");
        assert_eq!(
            (content.images[0].x, content.images[0].y, content.images[0].width, content.images[0].height),
            (5.0, 6.0, 40.0, 30.0)
        );
        assert_eq!(content.images[1].href, "other.png");
    }

    #[test]
    fn decode_svg_image_rasterises_at_intrinsic_size() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10"><rect x="0" y="0" width="20" height="10" fill="red"/></svg>"#;
        assert!(looks_like_svg(svg));
        let img = decode_svg_image(svg).expect("SVG decodes");
        assert_eq!((img.width, img.height), (20, 10));
        // No document stylesheet rides an external SVG, so the
        // `fill="red"` presentation attribute does not apply here
        // (initial fill is black) — the assert is that the rect
        // painted at all (opaque), not its colour.
        assert_eq!(img.rgba[3], 255, "top-left pixel is opaque");
        assert_eq!(img.rgba[0], 0, "initial fill is black, not red");
    }

    #[test]
    fn decode_svg_image_uses_view_box_without_width_height() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 6"><circle cx="4" cy="3" r="3" fill="blue"/></svg>"#;
        let img = decode_svg_image(svg).expect("SVG decodes");
        assert_eq!((img.width, img.height), (8, 6));
    }

    #[test]
    fn looks_like_svg_rejects_png_and_plain_text() {
        assert!(!looks_like_svg(b"not an image at all"));
        assert!(!looks_like_svg(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]));
        assert!(looks_like_svg(b"  <?xml version=\"1.0\"?>\n<svg></svg>"));
        assert!(!looks_like_svg(b"<html><svg></svg></html>"));
    }
}
