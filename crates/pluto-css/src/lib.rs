//! Parsing & cascade (layer 3): CSS parser.
//!
//! M2 scope: complex selectors (combinators, attribute selectors,
//! structural/link/dynamic/logical pseudo-classes, pseudo-elements),
//! specificity, cascade data (origin / `!important` / layers / source
//! order), conditional group rules (`@media` / `@supports`), `@import`,
//! `@font-face`, `@layer`, `@keyframes` (structured steps), and a real UA
//! stylesheet ([`UA_CSS`]) parsed by this same parser.
//!
//! The parser is fault-tolerant per CSS Syntax L3: a bad declaration is
//! dropped to the next `;`, a bad selector drops the whole rule (one
//! invalid complex selector invalidates the whole selector list), an
//! unknown at-rule is skipped to `;` or its balanced `}`, and an
//! unclosed block is closed at EOF. Everything skipped is recorded in
//! [`Stylesheet::errors`] with a byte offset where cheap.
//!
//! Matching note: [`Selector::matches`] only answers for compound-only
//! selectors against a bare `(tag, id, classes)` triple. Anything
//! needing tree context (combinators, structural/dynamic pseudo-classes,
//! attribute selectors, `:has()`) returns `false` here; the real
//! right-to-left matcher lives in `pluto-layout` (wave 2) and consumes
//! [`Selector::compounds`] / [`Selector::combinators`] directly.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// A parsed CSS value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A bare keyword: `block`, `none`, `auto`, `flex`, ...
    Keyword(String),
    /// A `px` length (bare `0` counts as `Length(0.0)`).
    Length(f32),
    /// A `%` value.
    Percentage(f32),
    /// An `em` length, resolved against the element's own computed
    /// font size at style time.
    Em(f32),
    /// A `rem` length, resolved against the root element's font size.
    Rem(f32),
    /// A `vh` length, resolved against the viewport height at layout
    /// time (a *used* value, not a computed one).
    Vh(f32),
    /// A `vw` length, resolved against the viewport width at layout time.
    Vw(f32),
    /// An RGB color, alpha always opaque.
    Color(u8, u8, u8),
    /// An RGBA color (`rgba()` / `hsla()` / 4- and 8-digit hex). The
    /// alpha channel is preserved here; consumers that cannot paint
    /// alpha yet should use the RGB channels and note the drop.
    ColorA(u8, u8, u8, f32),
    /// `border-radius: 1-4 lengths` (elliptical `/` radii are dropped).
    BorderRadius(BorderRadius),
    /// `box-shadow: <shadow>, ...` (`none` is the empty list).
    BoxShadow(Vec<BoxShadow>),
    /// `url(...)` — a resource reference, unquoted (Stage 11 Rec 3:
    /// `background-image`). The URL is stored verbatim; resolution
    /// against the document base is the caller's job.
    Url(String),
    /// A time duration in seconds (`0.3s` / `300ms`). Negative
    /// durations are invalid and stay `Raw`.
    Time(f32),
    /// An angle in degrees (`45deg` / `0.5turn` / `1rad`).
    Angle(f32),
    /// A `transform` function list (`none` is the empty list).
    Transform(TransformList),
    /// A parsed `transition` shorthand / longhand list.
    Transitions(Vec<Transition>),
    /// A parsed `animation` shorthand / longhand list.
    Animations(Vec<Animation>),
    /// `var(--name)` / `var(--name, <fallback>)`. Resolved at
    /// style time by the layout pass, not here.
    Var {
        name: String,
        fallback: Option<Box<Value>>,
    },
    /// `calc(...)`.
    Calc(CalcExpr),
    /// `clamp(min, val, max)`.
    Clamp {
        min: Box<CalcExpr>,
        val: Box<CalcExpr>,
        max: Box<CalcExpr>,
    },
    /// A CSS gradient (`linear-gradient()` / `radial-gradient()`),
    /// parsed to stops + geometry (Stage 15, M7 Phase 2, Rec 7).
    /// Stop offsets are resolved 0..=1 at parse time (missing
    /// offsets are distributed evenly, per CSS); `px` stop offsets
    /// are not supported and fail the parse (the value falls back
    /// to `Raw`). Shared by CSS `background-image` and, through
    /// the paint layer, SVG `<linearGradient>` /
    /// `<radialGradient>` (whose own defs live in pluto-paint and
    /// convert to this type).
    Gradient(Gradient),
    /// Anything not modelled yet (font stacks, ...), kept verbatim.
    Raw(String),
}

/// One gradient color stop: `offset` is a fraction along the
/// gradient line / radius (0..=1, resolved at parse time), `color`
/// is RGBA (alpha 255 when the stop had no alpha).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStop {
    pub offset: f32,
    pub color: [u8; 4],
}

/// The shape of a radial gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialShape {
    Circle,
    Ellipse,
}

/// The sizing keyword of a radial gradient (default
/// `FarthestCorner`, per CSS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialExtent {
    ClosestSide,
    FarthestSide,
    ClosestCorner,
    FarthestCorner,
}

/// A parsed gradient: its geometry plus its stops.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
}

/// The geometry of a gradient. A linear gradient's angle is in
/// degrees, clockwise from "to top" (0deg = to top, 90deg = to
/// right, 180deg = to bottom — the CSS default). A radial
/// gradient's center is fractions of the box (0.5, 0.5 = center);
/// its radii are derived from `shape` + `extent` against the box
/// at paint time, so the value stays box-independent.
#[derive(Debug, Clone, PartialEq)]
pub enum GradientKind {
    Linear { angle_deg: f32 },
    Radial {
        center: (f32, f32),
        shape: RadialShape,
        extent: RadialExtent,
    },
}

impl Gradient {
    /// The colour at `t` (0..=1 along the gradient line / radius),
    /// as premultiplied-by-nothing RGBA floats. Stops are clamped
    /// to the ends outside 0..=1; between stops the colour
    /// interpolates linearly (per channel, alpha included). Pure.
    pub fn color_at(&self, t: f32) -> [f32; 4] {
        let t = t.clamp(0.0, 1.0);
        if self.stops.is_empty() {
            return [0.0, 0.0, 0.0, 0.0];
        }
        if self.stops.len() == 1 || t <= self.stops[0].offset {
            return stop_rgba(self.stops[0]);
        }
        let last = self.stops.len() - 1;
        if t >= self.stops[last].offset {
            return stop_rgba(self.stops[last]);
        }
        for w in self.stops.windows(2) {
            let (a, b) = (w[0], w[1]);
            if t >= a.offset && t <= b.offset {
                let span = (b.offset - a.offset).max(f32::EPSILON);
                let f = (t - a.offset) / span;
                let ca = stop_rgba(a);
                let cb = stop_rgba(b);
                return [
                    ca[0] + (cb[0] - ca[0]) * f,
                    ca[1] + (cb[1] - ca[1]) * f,
                    ca[2] + (cb[2] - ca[2]) * f,
                    ca[3] + (cb[3] - ca[3]) * f,
                ];
            }
        }
        stop_rgba(self.stops[last])
    }

    /// The parameter `t` of a point `p` on a linear gradient over
    /// the box `[x, y, w, h]`: the projection of `p` onto the
    /// gradient axis (the line through the box center at
    /// `angle_deg`, from the "start" corner to the "end" corner),
    /// normalised so the axis spans 0..=1 across the box. Pure.
    /// For a radial gradient this is meaningless — use
    /// [`Gradient::radial_t`] instead.
    pub fn linear_t(&self, p: (f32, f32), rect: (f32, f32, f32, f32)) -> f32 {
        let GradientKind::Linear { angle_deg } = self.kind else {
            return 0.0;
        };
        let (x, y, w, h) = rect;
        let rad = angle_deg.to_radians();
        // CSS angles are clockwise from "to top": the axis
        // direction is (sin, -cos).
        let dx = rad.sin();
        let dy = -rad.cos();
        let cx = x + w / 2.0;
        let cy = y + h / 2.0;
        // Half the axis length across the box: the projection of
        // the half-diagonal onto the axis.
        let half = (w * dx.abs() + h * dy.abs()) / 2.0;
        if half <= 0.0 {
            return 0.0;
        }
        ((p.0 - cx) * dx + (p.1 - cy) * dy) / (2.0 * half) + 0.5
    }

    /// The parameter `t` of a point `p` on a radial gradient over
    /// the box `[x, y, w, h]`: the distance from the center,
    /// normalised by the radius in `p`'s direction (elliptical
    /// when the shape is an ellipse). Pure.
    pub fn radial_t(&self, p: (f32, f32), rect: (f32, f32, f32, f32)) -> f32 {
        let GradientKind::Radial {
            center,
            shape,
            extent,
        } = self.kind
        else {
            return 0.0;
        };
        let (x, y, w, h) = rect;
        let cx = x + center.0 * w;
        let cy = y + center.1 * h;
        let (rx, ry) = radial_radii(shape, extent, center, w, h);
        if rx <= 0.0 || ry <= 0.0 {
            return 0.0;
        }
        let dx = (p.0 - cx) / rx;
        let dy = (p.1 - cy) / ry;
        (dx * dx + dy * dy).sqrt()
    }
}

fn stop_rgba(stop: GradientStop) -> [f32; 4] {
    [
        stop.color[0] as f32 / 255.0,
        stop.color[1] as f32 / 255.0,
        stop.color[2] as f32 / 255.0,
        stop.color[3] as f32 / 255.0,
    ]
}

/// The (rx, ry) radii of a radial gradient over a `w`×`h` box
/// with center `center` (fractions of the box), from its shape +
/// extent keyword. Pure. Follows CSS Images 4: the circle radius
/// is the distance to the relevant side/corner; an ellipse through
/// a corner has radii `sqrt(2)` × the corner's axis distances.
pub fn radial_radii(
    shape: RadialShape,
    extent: RadialExtent,
    center: (f32, f32),
    w: f32,
    h: f32,
) -> (f32, f32) {
    let cx = center.0 * w;
    let cy = center.1 * h;
    // Distances from the center to each side.
    let (left, right, top, bottom) = (cx, w - cx, cy, h - cy);
    // Distances from the center to each corner (axis components).
    let corners = [
        (left, top),
        (right, top),
        (left, bottom),
        (right, bottom),
    ];
    let corner_dist = |dx: f32, dy: f32| (dx * dx + dy * dy).sqrt();
    match shape {
        RadialShape::Circle => {
            let r = match extent {
                RadialExtent::ClosestSide => left.min(right).min(top).min(bottom),
                RadialExtent::FarthestSide => left.max(right).max(top).max(bottom),
                RadialExtent::ClosestCorner => corners
                    .iter()
                    .map(|&(dx, dy)| corner_dist(dx, dy))
                    .fold(f32::INFINITY, f32::min),
                RadialExtent::FarthestCorner => corners
                    .iter()
                    .map(|&(dx, dy)| corner_dist(dx, dy))
                    .fold(0.0f32, f32::max),
            };
            (r, r)
        }
        RadialShape::Ellipse => {
            let (rx, ry) = match extent {
                RadialExtent::ClosestSide => (left.min(right), top.min(bottom)),
                RadialExtent::FarthestSide => (left.max(right), top.max(bottom)),
                RadialExtent::ClosestCorner | RadialExtent::FarthestCorner => {
                    let pick = if extent == RadialExtent::ClosestCorner {
                        corners
                            .iter()
                            .min_by(|a, b| {
                                corner_dist(a.0, a.1)
                                    .partial_cmp(&corner_dist(b.0, b.1))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .copied()
                            .unwrap_or((left, top))
                    } else {
                        corners
                            .iter()
                            .max_by(|a, b| {
                                corner_dist(a.0, a.1)
                                    .partial_cmp(&corner_dist(b.0, b.1))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .copied()
                            .unwrap_or((left, top))
                    };
                    // The ellipse through the corner, radii
                    // proportional to the corner's axis distances:
                    // (dx/rx)^2 + (dy/ry)^2 = 1 with rx = dx*s,
                    // ry = dy*s gives s = sqrt(2).
                    let s = std::f32::consts::SQRT_2;
                    (pick.0 * s, pick.1 * s)
                }
            };
            (rx, ry)
        }
    }
}

/// Light/dark colour-scheme theme, for `prefers-color-scheme`
/// media evaluation (M6 Rec 12). The shell owns the theme signal;
/// the layout entry points default to [`Theme::Light`] until it is
/// threaded through (a later phase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    #[default]
    Light,
    Dark,
}

/// An angle in degrees.
pub type Angle = f32;

/// A 2D affine transform, row-major `matrix(a, b, c, d, e, f)`:
/// `x' = a*x + c*y + e`, `y' = b*x + d*y + f`. One shared type
/// for the value model (pluto-css), the layout model
/// (pluto-layout) and — later — the display list (pluto-paint).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Affine {
    /// The identity transform.
    pub fn identity() -> Self {
        Affine {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// A translation by `(x, y)`.
    pub fn translate(x: f32, y: f32) -> Self {
        Affine {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: x,
            f: y,
        }
    }

    /// A scale by `(sx, sy)`.
    pub fn scale(sx: f32, sy: f32) -> Self {
        Affine {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    /// A rotation by `degrees` (clockwise, CSS convention).
    pub fn rotate(degrees: f32) -> Self {
        let r = degrees.to_radians();
        let (s, c) = r.sin_cos();
        Affine {
            a: c,
            b: s,
            c: -s,
            d: c,
            e: 0.0,
            f: 0.0,
        }
    }

    /// A skew along the x axis by `degrees`.
    pub fn skew_x(degrees: f32) -> Self {
        Affine {
            a: 1.0,
            b: 0.0,
            c: degrees.to_radians().tan(),
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// A skew along the y axis by `degrees`.
    pub fn skew_y(degrees: f32) -> Self {
        Affine {
            a: 1.0,
            b: degrees.to_radians().tan(),
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Compose: `self.then(other)` applies `self` first, then
    /// `other` (the matrix product `other * self`).
    pub fn then(self, other: Affine) -> Self {
        Affine {
            a: other.a * self.a + other.c * self.b,
            b: other.b * self.a + other.d * self.b,
            c: other.a * self.c + other.c * self.d,
            d: other.b * self.c + other.d * self.d,
            e: other.a * self.e + other.c * self.f + other.e,
            f: other.b * self.e + other.d * self.f + other.f,
        }
    }

    /// Apply the transform to a point.
    pub fn apply(self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    /// The inverse transform, or `None` when the matrix is
    /// singular (determinant ~0).
    pub fn invert(self) -> Option<Affine> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-12 {
            return None;
        }
        Some(Affine {
            a: self.d / det,
            b: -self.b / det,
            c: -self.c / det,
            d: self.a / det,
            e: (self.c * self.f - self.d * self.e) / det,
            f: (self.b * self.e - self.a * self.f) / det,
        })
    }
}

/// One `transform` function. Lengths are [`CssLength`] (the
/// pluto-css length type — layout's `Dim` is the same shape one
/// crate up); angles are degrees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransformFunction {
    Translate(CssLength, CssLength),
    TranslateX(CssLength),
    TranslateY(CssLength),
    Scale(f32, f32),
    ScaleX(f32),
    ScaleY(f32),
    Rotate(Angle),
    SkewX(Angle),
    SkewY(Angle),
    Matrix([f32; 6]),
}

/// A `transform` value: the function list, in author order
/// (`none` is the empty list). Serializable (via
/// [`TransformFunction`]) so it can ride a computed style or a
/// display item.
pub type TransformList = Vec<TransformFunction>;

impl TransformFunction {
    /// The affine matrix for one function. Percentage translates
    /// resolve against `basis` (`(width, height)` of the border
    /// box); `em`/`rem` against the font sizes, `vh`/`vw` against
    /// the viewport.
    fn to_affine(
        &self,
        basis: (f32, f32),
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
    ) -> Affine {
        let len = |l: CssLength, axis_basis: f32| match l {
            CssLength::Px(v) => v,
            CssLength::Percentage(v) => v / 100.0 * axis_basis,
            CssLength::Em(v) => v * font_size,
            CssLength::Rem(v) => v * root_font_size,
            CssLength::Vh(v) => v / 100.0 * viewport.1,
            CssLength::Vw(v) => v / 100.0 * viewport.0,
        };
        match *self {
            TransformFunction::Translate(x, y) => {
                Affine::translate(len(x, basis.0), len(y, basis.1))
            }
            TransformFunction::TranslateX(x) => Affine::translate(len(x, basis.0), 0.0),
            TransformFunction::TranslateY(y) => Affine::translate(0.0, len(y, basis.1)),
            TransformFunction::Scale(sx, sy) => Affine::scale(sx, sy),
            TransformFunction::ScaleX(sx) => Affine::scale(sx, 1.0),
            TransformFunction::ScaleY(sy) => Affine::scale(1.0, sy),
            TransformFunction::Rotate(a) => Affine::rotate(a),
            TransformFunction::SkewX(a) => Affine::skew_x(a),
            TransformFunction::SkewY(a) => Affine::skew_y(a),
            TransformFunction::Matrix(m) => Affine {
                a: m[0],
                b: m[1],
                c: m[2],
                d: m[3],
                e: m[4],
                f: m[5],
            },
        }
    }
}

/// Helpers on a transform list.
pub trait TransformListExt {
    /// Compose the functions in author order, wrapped around
    /// `origin`: `T(origin) * M * T(-origin)`. Lengths resolve
    /// with the 16px / zero-basis defaults — percentages and
    /// viewport units resolve to 0; layout's
    /// [`TransformListExt::to_affine_in`] is the resolving form.
    fn to_affine(&self, origin: (f32, f32)) -> Affine;
    /// Like [`TransformListExt::to_affine`], but percentage
    /// translates resolve against `basis` (`(width, height)` of
    /// the border box), `em`/`rem` against the font sizes and
    /// `vh`/`vw` against the viewport.
    fn to_affine_in(
        &self,
        origin: (f32, f32),
        basis: (f32, f32),
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
    ) -> Affine;
}

impl TransformListExt for TransformList {
    fn to_affine(&self, origin: (f32, f32)) -> Affine {
        self.to_affine_in(origin, (0.0, 0.0), 16.0, 16.0, (0.0, 0.0))
    }

    fn to_affine_in(
        &self,
        origin: (f32, f32),
        basis: (f32, f32),
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
    ) -> Affine {
        let mut m = Affine::identity();
        for f in self {
            m = m.then(f.to_affine(basis, font_size, root_font_size, viewport));
        }
        // `T(origin) * M * T(-origin)` in matrix order: the
        // -origin translation is applied first, then `M`, then
        // +origin.
        Affine::translate(-origin.0, -origin.1)
            .then(m)
            .then(Affine::translate(origin.0, origin.1))
    }
}

/// Where a `steps()` jump lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPosition {
    Start,
    End,
}

/// A timing function for transitions and animations.
#[derive(Debug, Clone, PartialEq)]
pub enum Easing {
    Linear,
    Ease,
    EaseIn,
    EaseOut,
    EaseInOut,
    CubicBezier(f32, f32, f32, f32),
    Steps(u32, StepPosition),
}

impl Easing {
    /// Parse a timing-function token: a keyword, `cubic-bezier()`
    /// or `steps()`.
    pub fn parse(raw: &str) -> Option<Easing> {
        let s = raw.trim();
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "linear" => return Some(Easing::Linear),
            "ease" => return Some(Easing::Ease),
            "ease-in" => return Some(Easing::EaseIn),
            "ease-out" => return Some(Easing::EaseOut),
            "ease-in-out" => return Some(Easing::EaseInOut),
            _ => {}
        }
        if lower.starts_with("cubic-bezier(") && lower.ends_with(')') {
            let inner = &s[13..s.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 4 {
                let nums: Option<Vec<f32>> =
                    parts.iter().map(|p| p.trim().parse::<f32>().ok()).collect();
                if let Some(n) = nums {
                    return Some(Easing::CubicBezier(n[0], n[1], n[2], n[3]));
                }
            }
            return None;
        }
        if lower.starts_with("steps(") && lower.ends_with(')') {
            let inner = &s[6..s.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.is_empty() || parts.len() > 2 {
                return None;
            }
            let n: u32 = parts[0].trim().parse().ok()?;
            if n == 0 {
                return None;
            }
            let pos = match parts.get(1).map(|p| p.trim().to_ascii_lowercase()) {
                None => StepPosition::End,
                Some(p) if p == "start" || p == "jump-start" => StepPosition::Start,
                Some(p) if p == "end" || p == "jump-end" => StepPosition::End,
                Some(_) => return None,
            };
            return Some(Easing::Steps(n, pos));
        }
        None
    }

    /// Sample the easing at progress `t` in `[0, 1]`.
    pub fn sample(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match *self {
            Easing::Linear => t,
            Easing::Ease => Easing::CubicBezier(0.25, 0.1, 0.25, 1.0).sample(t),
            Easing::EaseIn => Easing::CubicBezier(0.42, 0.0, 1.0, 1.0).sample(t),
            Easing::EaseOut => Easing::CubicBezier(0.0, 0.0, 0.58, 1.0).sample(t),
            Easing::EaseInOut => Easing::CubicBezier(0.42, 0.0, 0.58, 1.0).sample(t),
            Easing::CubicBezier(x1, y1, x2, y2) => cubic_bezier_sample(x1, y1, x2, y2, t),
            Easing::Steps(n, pos) => {
                let n = n as f32;
                match pos {
                    // Jumps at the end of each interval.
                    StepPosition::End => ((t * n).floor() / n).clamp(0.0, 1.0),
                    // Jumps at the start of each interval: progress
                    // 0 already shows the first step.
                    StepPosition::Start => {
                        if t <= 0.0 {
                            (1.0 / n).clamp(0.0, 1.0)
                        } else {
                            ((t * n).ceil() / n).clamp(0.0, 1.0)
                        }
                    }
                }
            }
        }
    }
}

impl fmt::Display for Easing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Easing::Linear => write!(f, "linear"),
            Easing::Ease => write!(f, "ease"),
            Easing::EaseIn => write!(f, "ease-in"),
            Easing::EaseOut => write!(f, "ease-out"),
            Easing::EaseInOut => write!(f, "ease-in-out"),
            Easing::CubicBezier(x1, y1, x2, y2) => {
                write!(f, "cubic-bezier({x1}, {y1}, {x2}, {y2})")
            }
            Easing::Steps(n, StepPosition::Start) => write!(f, "steps({n}, start)"),
            Easing::Steps(n, StepPosition::End) => write!(f, "steps({n}, end)"),
        }
    }
}

/// Sample a cubic-bezier easing at input progress `x`: solve for
/// the curve parameter `u` with `bezier_x(u) = x` (Newton, with a
/// bisection fallback when the derivative vanishes), then return
/// `bezier_y(u)`.
fn cubic_bezier_sample(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Polynomial coefficients for one axis: P(u) =
    // ((a*u + b)*u + c)*u.
    let (ax, bx, cx) = (1.0 - 3.0 * x2 + 3.0 * x1, 3.0 * x2 - 6.0 * x1, 3.0 * x1);
    let (ay, by, cy) = (1.0 - 3.0 * y2 + 3.0 * y1, 3.0 * y2 - 6.0 * y1, 3.0 * y1);
    let sample_x = |u: f32| ((ax * u + bx) * u + cx) * u;
    let sample_y = |u: f32| ((ay * u + by) * u + cy) * u;
    let sample_dx = |u: f32| (3.0 * ax * u + 2.0 * bx) * u + cx;
    // Newton from u = x; fall back to bisection on failure.
    let mut u = x;
    let mut solved = false;
    for _ in 0..8 {
        let err = sample_x(u) - x;
        if err.abs() < 1e-6 {
            solved = true;
            break;
        }
        let d = sample_dx(u);
        if d.abs() < 1e-6 {
            break;
        }
        let next = u - err / d;
        if !(0.0..=1.0).contains(&next) {
            break;
        }
        u = next;
    }
    if !solved {
        let mut lo = 0.0f32;
        let mut hi = 1.0f32;
        u = x;
        for _ in 0..20 {
            if sample_x(u) < x {
                lo = u;
            } else {
                hi = u;
            }
            u = (lo + hi) / 2.0;
        }
    }
    sample_y(u).clamp(0.0, 1.0)
}

/// One `transition` entry: which property, how long, with what
/// easing, after what delay. Durations/delays are seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub property: String,
    pub duration: f32,
    pub timing: Easing,
    pub delay: f32,
}

impl Default for Transition {
    fn default() -> Self {
        Transition {
            property: "all".to_string(),
            duration: 0.0,
            timing: Easing::Ease,
            delay: 0.0,
        }
    }
}

/// How many times an animation repeats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterationCount {
    Finite(f32),
    Infinite,
}

/// Which way an animation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationDirection {
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

/// What an animation applies outside its active interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    None,
    Forwards,
    Backwards,
    Both,
}

/// Whether an animation is running or paused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    Running,
    Paused,
}

/// One `animation` entry. Durations/delays are seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Animation {
    pub name: String,
    pub duration: f32,
    pub timing: Easing,
    pub delay: f32,
    pub iteration_count: IterationCount,
    pub direction: AnimationDirection,
    pub fill_mode: FillMode,
    pub play_state: PlayState,
}

impl Default for Animation {
    fn default() -> Self {
        Animation {
            name: "none".to_string(),
            duration: 0.0,
            timing: Easing::Ease,
            delay: 0.0,
            iteration_count: IterationCount::Finite(1.0),
            direction: AnimationDirection::Normal,
            fill_mode: FillMode::None,
            play_state: PlayState::Running,
        }
    }
}

/// One keyframe step: its selectors plus the declarations that
/// apply at those offsets.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeStep {
    pub selectors: Vec<KeyframeSelector>,
    pub declarations: Vec<Declaration>,
}

/// A keyframe selector: `from` / `to` / `50%`.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyframeSelector {
    From,
    To,
    Percentage(f32),
}

/// A `calc()` / `clamp()` expression tree. `Var` carries its
/// fallback expression, if any; `Mul`/`Div` only ever scale by a
/// plain number (anything else is unparseable and the whole value
/// falls back to `Raw`).
#[derive(Debug, Clone, PartialEq)]
pub enum CalcExpr {
    Len(CssLength),
    Pct(f32),
    Num(f32),
    Var(String, Option<Box<CalcExpr>>),
    Add(Box<CalcExpr>, Box<CalcExpr>),
    Sub(Box<CalcExpr>, Box<CalcExpr>),
    Mul(Box<CalcExpr>, f32),
    Div(Box<CalcExpr>, f32),
}

/// The context a `calc()` evaluates against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalcCtx {
    pub font_size: f32,
    pub root_font_size: f32,
    pub viewport: (f32, f32),
    /// What a percentage resolves against (`None` = unresolvable
    /// here — the caller has no basis, so the expression yields
    /// `Pct` and the declaration falls back).
    pub pct_basis: Option<f32>,
}

/// The result of evaluating a [`CalcExpr`]: used pixels, or an
/// unresolved percentage (only when no basis was available).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalcNumber {
    Px(f32),
    Pct(f32),
}

impl CalcExpr {
    /// Evaluate against `ctx`. `None` means unresolvable (a `%`
    /// with no basis, a bare number where a length is needed, a
    /// `var()` with no fallback, mixed px/% arithmetic, division
    /// by zero) — the declaration falls back to its initial value.
    pub fn eval(&self, ctx: &CalcCtx) -> Option<CalcNumber> {
        match self {
            CalcExpr::Len(l) => Some(match *l {
                CssLength::Px(v) => CalcNumber::Px(v),
                CssLength::Em(v) => CalcNumber::Px(v * ctx.font_size),
                CssLength::Rem(v) => CalcNumber::Px(v * ctx.root_font_size),
                CssLength::Vh(v) => CalcNumber::Px(v / 100.0 * ctx.viewport.1),
                CssLength::Vw(v) => CalcNumber::Px(v / 100.0 * ctx.viewport.0),
                CssLength::Percentage(p) => return CalcExpr::Pct(p).eval(ctx),
            }),
            CalcExpr::Pct(p) => Some(match ctx.pct_basis {
                Some(b) => CalcNumber::Px(p / 100.0 * b),
                None => CalcNumber::Pct(*p),
            }),
            CalcExpr::Num(_) => None,
            CalcExpr::Var(_, fallback) => {
                fallback.as_ref().and_then(|f| f.eval(ctx))
            }
            CalcExpr::Add(a, b) => {
                let (x, y) = (a.eval(ctx)?, b.eval(ctx)?);
                match (x, y) {
                    (CalcNumber::Px(u), CalcNumber::Px(v)) => Some(CalcNumber::Px(u + v)),
                    (CalcNumber::Pct(u), CalcNumber::Pct(v)) => Some(CalcNumber::Pct(u + v)),
                    _ => None,
                }
            }
            CalcExpr::Sub(a, b) => {
                let (x, y) = (a.eval(ctx)?, b.eval(ctx)?);
                match (x, y) {
                    (CalcNumber::Px(u), CalcNumber::Px(v)) => Some(CalcNumber::Px(u - v)),
                    (CalcNumber::Pct(u), CalcNumber::Pct(v)) => Some(CalcNumber::Pct(u - v)),
                    _ => None,
                }
            }
            CalcExpr::Mul(e, n) => match e.eval(ctx)? {
                CalcNumber::Px(v) => Some(CalcNumber::Px(v * n)),
                CalcNumber::Pct(p) => Some(CalcNumber::Pct(p * n)),
            },
            CalcExpr::Div(e, n) => {
                if *n == 0.0 {
                    return None;
                }
                match e.eval(ctx)? {
                    CalcNumber::Px(v) => Some(CalcNumber::Px(v / n)),
                    CalcNumber::Pct(p) => Some(CalcNumber::Pct(p / n)),
                }
            }
        }
    }

    /// Convert a plain [`Value`] to a [`CalcExpr`], for `var()`
    /// substitution inside a `calc()`: lengths/percentages/
    /// numbers map directly, a nested `calc()` unwraps, a `var()`
    /// maps to its node (resolved by the caller first); anything
    /// else (colours, keywords, ...) is not a length and yields
    /// `None`.
    pub fn from_value(value: &Value) -> Option<CalcExpr> {
        Some(match value {
            Value::Length(v) => CalcExpr::Len(CssLength::Px(*v)),
            Value::Percentage(p) => CalcExpr::Pct(*p),
            Value::Em(v) => CalcExpr::Len(CssLength::Em(*v)),
            Value::Rem(v) => CalcExpr::Len(CssLength::Rem(*v)),
            Value::Vh(v) => CalcExpr::Len(CssLength::Vh(*v)),
            Value::Vw(v) => CalcExpr::Len(CssLength::Vw(*v)),
            Value::Calc(e) => e.clone(),
            Value::Var { name, fallback } => CalcExpr::Var(
                name.clone(),
                fallback
                    .as_ref()
                    .and_then(|f| CalcExpr::from_value(f))
                    .map(Box::new),
            ),
            _ => return None,
        })
    }
}

impl fmt::Display for CalcExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcExpr::Len(l) => write!(f, "{l}"),
            CalcExpr::Pct(p) => write!(f, "{p}%"),
            CalcExpr::Num(n) => write!(f, "{n}"),
            CalcExpr::Var(name, None) => write!(f, "var({name})"),
            CalcExpr::Var(name, Some(fb)) => write!(f, "var({name}, {fb})"),
            CalcExpr::Add(a, b) => write!(f, "({a} + {b})"),
            CalcExpr::Sub(a, b) => write!(f, "({a} - {b})"),
            CalcExpr::Mul(e, n) => write!(f, "({e} * {n})"),
            CalcExpr::Div(e, n) => write!(f, "({e} / {n})"),
        }
    }
}

/// Every `url(...)` reference in a raw CSS value, in order
/// (Stage 11 Rec 3): quoted or unquoted, `url(` matched
/// case-insensitively. Used for `background` shorthand values
/// (where the URL hides inside a multi-token `Raw`) and for
/// collecting background-image URLs from a stylesheet.
pub fn extract_css_urls(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = raw.to_ascii_lowercase();
    let mut rest = lower.as_str();
    let mut offset = 0usize;
    while let Some(pos) = rest.find("url(") {
        let start = offset + pos + 4;
        let inner = &raw[start..];
        let end = inner.find(')').unwrap_or(inner.len());
        let url = inner[..end]
            .trim()
            .trim_matches(|c| c == '"' || c == '\'')
            .trim()
            .to_string();
        if !url.is_empty() {
            out.push(url);
        }
        let next = start + end + 1;
        if next >= raw.len() {
            break;
        }
        rest = &lower[next..];
        offset = next;
    }
    out
}

/// One length that can appear inside a structured value
/// ([`BorderRadius`], [`BoxShadow`], [`TransformFunction`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CssLength {
    Px(f32),
    Percentage(f32),
    Em(f32),
    Rem(f32),
    Vh(f32),
    Vw(f32),
}

impl CssLength {
    /// Convert a typed [`Value`] to a [`CssLength`], or `None` when the
    /// value is not a length in any unit this type models.
    pub fn from_value(value: &Value) -> Option<Self> {
        Some(match value {
            Value::Length(v) => CssLength::Px(*v),
            Value::Percentage(v) => CssLength::Percentage(*v),
            Value::Em(v) => CssLength::Em(*v),
            Value::Rem(v) => CssLength::Rem(*v),
            Value::Vh(v) => CssLength::Vh(*v),
            Value::Vw(v) => CssLength::Vw(*v),
            _ => return None,
        })
    }

    /// Resolve this length to used pixels. `font_size` is the
    /// element's own computed font size (`em`), `root_font_size`
    /// the root element's (`rem`), `viewport` the layout viewport
    /// (`vw`/`vh`); a percentage resolves against `percent_base`,
    /// which the caller supplies per axis (width for horizontal
    /// radii, height for vertical ones).
    pub fn resolve_px(
        self,
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
        percent_base: f32,
    ) -> f32 {
        match self {
            CssLength::Px(v) => v,
            CssLength::Percentage(v) => v / 100.0 * percent_base,
            CssLength::Em(v) => v * font_size,
            CssLength::Rem(v) => v * root_font_size,
            CssLength::Vh(v) => v / 100.0 * viewport.1,
            CssLength::Vw(v) => v / 100.0 * viewport.0,
        }
    }
}

/// One corner's resolved radii, in used pixels: `(rx, ry)` — the
/// horizontal radius (a percentage of the box width) and the vertical
/// radius (a percentage of the box height). Kept as plain numbers so
/// the display list can carry them over IPC.
pub type CornerRadii = (f32, f32);

/// A box's four resolved corner radii, clockwise from the top-left.
/// Produced by [`BorderRadius::resolve`] at build time (Stage 8):
/// percentages resolved per axis, then normalised by the
/// corner-overlap rule, so paint and the renderer never re-derive
/// them and never see an overlapping pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCorners {
    pub top_left: CornerRadii,
    pub top_right: CornerRadii,
    pub bottom_right: CornerRadii,
    pub bottom_left: CornerRadii,
}

impl ResolvedCorners {
    /// All radii zero: a square box.
    pub const ZERO: Self = Self {
        top_left: (0.0, 0.0),
        top_right: (0.0, 0.0),
        bottom_right: (0.0, 0.0),
        bottom_left: (0.0, 0.0),
    };

    /// True when every radius is zero.
    pub fn is_zero(self) -> bool {
        self.top_left == (0.0, 0.0)
            && self.top_right == (0.0, 0.0)
            && self.bottom_right == (0.0, 0.0)
            && self.bottom_left == (0.0, 0.0)
    }

    /// Every radius grown by `spread` (a box-shadow spread expands
    /// the shape the shadow is cast from, radii included), clamped
    /// at zero.
    pub fn expanded_by(self, spread: f32) -> Self {
        let grow = |(rx, ry): CornerRadii| ((rx + spread).max(0.0), (ry + spread).max(0.0));
        Self {
            top_left: grow(self.top_left),
            top_right: grow(self.top_right),
            bottom_right: grow(self.bottom_right),
            bottom_left: grow(self.bottom_left),
        }
    }

    /// Every radius scaled by `factor` (the corner-overlap
    /// normalisation step).
    pub fn scaled_by(self, factor: f32) -> Self {
        let scale = |(rx, ry): CornerRadii| (rx * factor, ry * factor);
        Self {
            top_left: scale(self.top_left),
            top_right: scale(self.top_right),
            bottom_right: scale(self.bottom_right),
            bottom_left: scale(self.bottom_left),
        }
    }
}

impl BorderRadius {
    /// Resolve this `border-radius` against a box of `width` ×
    /// `height`: each corner's single length becomes `(rx, ry)` —
    /// a percentage is of the width for `rx` and of the height for
    /// `ry` — and the result is normalised by the corner-overlap
    /// rule (CSS Backgrounds and Borders 3, §4.5 "Overlapping
    /// Curves", verified live against
    /// https://drafts.csswg.org/css-backgrounds-3/#corner-overlap
    /// on 2026-11-24): `f = min(Li / Si)` over the four sides,
    /// where `Si` is the sum of the two radii on side `i` (the
    /// horizontal radii on the top/bottom, the vertical radii on
    /// the left/right) and `Ltop = Lbottom = width`,
    /// `Lleft = Lright = height`; when `f < 1` every radius is
    /// multiplied by `f`. Negative lengths are invalid in CSS and
    /// clamp to zero here rather than failing the frame.
    pub fn resolve(
        self,
        width: f32,
        height: f32,
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
    ) -> ResolvedCorners {
        let corner = |len: CssLength| {
            let rx = len
                .resolve_px(font_size, root_font_size, viewport, width)
                .max(0.0);
            let ry = len
                .resolve_px(font_size, root_font_size, viewport, height)
                .max(0.0);
            (rx, ry)
        };
        let resolved = ResolvedCorners {
            top_left: corner(self.top_left),
            top_right: corner(self.top_right),
            bottom_right: corner(self.bottom_right),
            bottom_left: corner(self.bottom_left),
        };
        // §4.5: a side's factor is its length over the sum of the
        // two radii on that side (a zero sum means the side cannot
        // overflow, so it does not constrain `f`).
        let side = |a: f32, b: f32, len: f32| {
            let sum = a + b;
            if sum > 0.0 {
                len / sum
            } else {
                f32::INFINITY
            }
        };
        let f = side(resolved.top_left.0, resolved.top_right.0, width)
            .min(side(
                resolved.top_right.1,
                resolved.bottom_right.1,
                height,
            ))
            .min(side(
                resolved.bottom_right.0,
                resolved.bottom_left.0,
                width,
            ))
            .min(side(resolved.bottom_left.1, resolved.top_left.1, height));
        if f < 1.0 {
            resolved.scaled_by(f)
        } else {
            resolved
        }
    }
}

/// The largest `box-shadow` blur Pluto renders (Stage 8): above it
/// the display list fails validation instead of silently drawing a
/// cheaper shadow than the author asked for. Chromium caps at 128;
/// Pluto's 64 keeps the erf falloff (`~3σ`) inside a sane quad.
pub const MAX_BOX_SHADOW_BLUR: f32 = 64.0;

/// One `box-shadow` layer, fully resolved to used pixels: the shape
/// the renderer draws it from (the box's border box, expanded by
/// `spread` and translated by the offsets at draw time).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResolvedShadow {
    pub inset: bool,
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
    pub spread: f32,
    /// `(r, g, b, a)` in 0..=1; `currentcolor` already resolved by
    /// the caller to the element's text color.
    pub color: [f32; 4],
}

impl BoxShadow {
    /// Resolve this shadow to used pixels. Offsets may be negative
    /// (a shadow may fall up/left); blur and spread clamp at zero
    /// (negative values are invalid in CSS). Blur is *not* clamped
    /// to [`MAX_BOX_SHADOW_BLUR`] here: a blur above the cap is a
    /// validation error on the display list, not a silent clamp.
    pub fn resolve(
        self,
        font_size: f32,
        root_font_size: f32,
        viewport: (f32, f32),
        current_color: [u8; 3],
    ) -> ResolvedShadow {
        let len = |l: CssLength| l.resolve_px(font_size, root_font_size, viewport, 0.0);
        let color = match self.color {
            Some((r, g, b, a)) => [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a,
            ],
            None => [
                current_color[0] as f32 / 255.0,
                current_color[1] as f32 / 255.0,
                current_color[2] as f32 / 255.0,
                1.0,
            ],
        };
        ResolvedShadow {
            inset: self.inset,
            offset_x: len(self.offset_x),
            offset_y: len(self.offset_y),
            blur: len(self.blur).max(0.0),
            spread: len(self.spread).max(0.0),
            color,
        }
    }
}

/// `border-radius` corners, clockwise from the top-left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderRadius {
    pub top_left: CssLength,
    pub top_right: CssLength,
    pub bottom_right: CssLength,
    pub bottom_left: CssLength,
}

/// One `box-shadow` layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxShadow {
    pub inset: bool,
    pub offset_x: CssLength,
    pub offset_y: CssLength,
    pub blur: CssLength,
    pub spread: CssLength,
    /// `(r, g, b, alpha)`; `None` means "current color".
    pub color: Option<(u8, u8, u8, f32)>,
}

/// An `an+b` argument to `:nth-child()` and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnPlusB {
    pub a: i32,
    pub b: i32,
}

/// An attribute-selector operator: `[a]`, `[a=v]`, `[a~=v]`, `[a|=v]`,
/// `[a^=v]`, `[a$=v]`, `[a*=v]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrOp {
    Exists,
    Equals,
    Includes,
    Dash,
    Prefix,
    Suffix,
    Substring,
}

/// A pseudo-class. Structural and link variants are static (checkable
/// from the DOM); dynamic variants need event/hit-test state; logical
/// variants take forgiving selector lists; `:has()` is parsed and
/// stored but matching is not implemented (see
/// [`Selector::has_unimplemented`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoClass {
    Root,
    FirstChild,
    LastChild,
    NthChild(AnPlusB),
    NthLastChild(AnPlusB),
    OnlyChild,
    Empty,
    FirstOfType,
    LastOfType,
    NthOfType(AnPlusB),
    NthLastOfType(AnPlusB),
    OnlyOfType,
    Link,
    Visited,
    AnyLink,
    Hover,
    Active,
    Focus,
    FocusVisible,
    FocusWithin,
    // Form pseudo-classes (Stage 13 Rec 8, M5): matched from the
    // worker-projected `ElementState` form sets (see
    // `pluto-layout`), never from the DOM directly — the matcher
    // only sees element ids.
    Valid,
    Invalid,
    Required,
    Optional,
    Disabled,
    Checked,
    ReadOnly,
    Not(Vec<Selector>),
    Is(Vec<Selector>),
    Where(Vec<Selector>),
    Has(Vec<Selector>),
}

impl PseudoClass {
    /// Specificity contribution: `:where()` is zero, `:is()` / `:not()` /
    /// `:has()` take the max of their arguments, everything else is one
    /// pseudo-class (0, 1, 0).
    pub fn specificity(&self) -> (u32, u32, u32) {
        match self {
            PseudoClass::Where(_) => (0, 0, 0),
            PseudoClass::Is(args) | PseudoClass::Not(args) | PseudoClass::Has(args) => args
                .iter()
                .map(|s| s.specificity())
                .max()
                .unwrap_or((0, 0, 0)),
            _ => (0, 1, 0),
        }
    }
}

/// A pseudo-element (`::before`, ...). Parsed and stored on the
/// selector; generated boxes are a layout concern, not a CSS one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PseudoElement {
    Before,
    After,
    FirstLine,
    FirstLetter,
    Marker,
    Selection,
}

/// One simple selector: part of a compound selector, all parts of which
/// must match the same element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimpleSelector {
    Universal,
    Type(String),
    Class(String),
    Id(String),
    Attribute {
        name: String,
        op: AttrOp,
        value: Option<String>,
        /// `Some(false)` for the `i` flag, `Some(true)` for `s`,
        /// `None` for the default (case-sensitive, except quirks mode).
        case_sensitive: Option<bool>,
    },
    PseudoClass(PseudoClass),
    /// Kept in the compound only transiently while parsing; after a
    /// successful parse it is moved to [`Selector::pseudo_element`].
    /// It still appears here in a partially built selector.
    PseudoElement(PseudoElement),
}

impl SimpleSelector {
    /// Specificity contribution of one simple selector.
    pub fn specificity(&self) -> (u32, u32, u32) {
        match self {
            SimpleSelector::Universal => (0, 0, 0),
            SimpleSelector::Type(_) => (0, 0, 1),
            SimpleSelector::Class(_) => (0, 1, 0),
            SimpleSelector::Id(_) => (1, 0, 0),
            SimpleSelector::Attribute { .. } => (0, 1, 0),
            SimpleSelector::PseudoClass(pc) => pc.specificity(),
            SimpleSelector::PseudoElement(_) => (0, 0, 1),
        }
    }

    /// Indexing rank for [`Selector::key_selector`]: id beats
    /// class/attribute/pseudo-class beats type beats universal.
    fn key_rank(&self) -> u8 {
        match self {
            SimpleSelector::Id(_) => 4,
            SimpleSelector::Class(_)
            | SimpleSelector::Attribute { .. }
            | SimpleSelector::PseudoClass(_) => 3,
            SimpleSelector::Type(_) => 2,
            SimpleSelector::Universal => 1,
            SimpleSelector::PseudoElement(_) => 0,
        }
    }
}

/// A compound selector: `div`, `.note`, `#main`, `*`, `div.note`,
/// `a[href^="https"]:hover` — simple selectors with no whitespace
/// between them, all matching the same element.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompoundSelector {
    pub parts: Vec<SimpleSelector>,
}

/// A combinator joining two compounds in a complex selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    /// `A B` — B is a descendant of A.
    Descendant,
    /// `A > B` — B is a child of A.
    Child,
    /// `A + B` — B immediately follows A.
    NextSibling,
    /// `A ~ B` — B follows A among siblings.
    SubsequentSibling,
}

/// A complex selector: compounds joined by combinators, e.g.
/// `div.note > p:first-child`, plus an optional trailing
/// pseudo-element (`p::before`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Selector {
    /// Compounds left to right; `combinators[i]` joins
    /// `compounds[i]` and `compounds[i + 1]`.
    pub compounds: Vec<CompoundSelector>,
    pub combinators: Vec<Combinator>,
    pub pseudo_element: Option<PseudoElement>,
    /// True when the selector contains `:has(...)`. `:has()` is parsed
    /// and stored, but matching is NOT implemented: a selector with
    /// this flag must never be treated as matching.
    pub has_unimplemented: bool,
}

impl Selector {
    /// Specificity as a lexicographic triple `(ids, classes, types)`:
    /// IDs; classes + attributes + pseudo-classes; types +
    /// pseudo-elements. Compared lexicographically, never as a single
    /// number — 256 classes do not beat one ID. `:is()` / `:not()` /
    /// `:has()` contribute the max of their arguments; `:where()`
    /// contributes zero.
    pub fn specificity(&self) -> (u32, u32, u32) {
        let mut spec = (0u32, 0u32, 0u32);
        for compound in &self.compounds {
            for part in &compound.parts {
                let (a, b, c) = part.specificity();
                spec.0 += a;
                spec.1 += b;
                spec.2 += c;
            }
        }
        if self.pseudo_element.is_some() {
            spec.2 += 1;
        }
        spec
    }

    /// The key selector for rule indexing (Stylo-style): the most
    /// selective simple selector of the rightmost compound —
    /// id > class/attribute/pseudo-class > type > universal. Layout
    /// buckets rules by this and tests only the matching buckets.
    pub fn key_selector(&self) -> Option<&SimpleSelector> {
        self.compounds
            .last()?
            .parts
            .iter()
            .max_by_key(|part| part.key_rank())
    }

    /// True when every part matches this element (`tag` lowercased,
    /// `id` and `classes` as given). Compound-only selectors only:
    /// any combinator, pseudo-element, `:has()`, attribute selector,
    /// or pseudo-class needing DOM/event context returns `false` —
    /// the real right-to-left matcher (wave 2, in `pluto-layout`)
    /// handles those against the DOM.
    pub fn matches(&self, tag: &str, id: Option<&str>, classes: &[&str]) -> bool {
        if self.compounds.len() != 1 || !self.combinators.is_empty() {
            return false;
        }
        if self.pseudo_element.is_some() || self.has_unimplemented {
            return false;
        }
        self.compounds[0].parts.iter().all(|part| match part {
            SimpleSelector::Universal => true,
            SimpleSelector::Type(t) => t == tag,
            SimpleSelector::Class(c) => classes.contains(&c.as_str()),
            SimpleSelector::Id(want) => id == Some(want.as_str()),
            // Attribute selectors and pseudo-classes need DOM or
            // event state this triple does not carry.
            _ => false,
        })
    }
}

/// Cascade origin (Cascade 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Origin {
    /// The browser's own defaults ([`UA_CSS`]).
    Ua,
    /// The user's own stylesheet (not wired up yet).
    User,
    /// The document's own stylesheets (what the parser produces).
    #[default]
    Author,
}

impl Origin {
    /// Origin + importance rank, lowest first (Cascade 5 §6.1):
    /// UA normal < user normal < author normal < author `!important`
    /// < user `!important` < UA `!important`. Note the reversal:
    /// important user beats important author.
    pub fn rank(self, important: bool) -> u8 {
        match (self, important) {
            (Origin::Ua, false) => 0,
            (Origin::User, false) => 1,
            (Origin::Author, false) => 2,
            (Origin::Author, true) => 3,
            (Origin::User, true) => 4,
            (Origin::Ua, true) => 5,
        }
    }
}

/// One `property: value;` pair, plus whether it was `!important`.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub property: String,
    pub value: Value,
    pub important: bool,
}

/// A `@media` condition: `and` / `,` lists plus `max-width` /
/// `min-width` (and height) features with `px`/`em` values. Anything
/// else is kept as [`MediaCondition::Unknown`] and evaluates to
/// `None` (unknown), never silently to true or false.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaCondition {
    /// A bare media type: `screen`, `print`, `all`, ...
    MediaType(String),
    /// `(max-width: 700px)` — `value` is `None` for boolean features
    /// such as `(color)`.
    Feature { name: String, value: Option<Value> },
    And(Vec<MediaCondition>),
    /// A comma list.
    Or(Vec<MediaCondition>),
    Not(Box<MediaCondition>),
    Unknown(String),
}

impl MediaCondition {
    /// Parse a media query list. Never fails: unparseable input becomes
    /// [`MediaCondition::Unknown`].
    pub fn parse(raw: &str) -> Self {
        let text = raw.trim();
        if text.is_empty() {
            return MediaCondition::Unknown(String::new());
        }
        let comma_parts = split_top_level(text, ',');
        if comma_parts.len() > 1 {
            return MediaCondition::Or(
                comma_parts
                    .iter()
                    .map(|p| MediaCondition::parse(p))
                    .collect(),
            );
        }
        let and_parts = split_top_level_keyword(text, "and");
        if and_parts.len() > 1 {
            return MediaCondition::And(
                and_parts
                    .iter()
                    .map(|p| MediaCondition::parse(p))
                    .collect(),
            );
        }
        let lower = text.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("not ") {
            let rest_len = rest.len();
            let _ = rest_len;
            let inner = &text[4..];
            return MediaCondition::Not(Box::new(MediaCondition::parse(inner)));
        }
        if let Some(rest) = lower.strip_prefix("only ") {
            let _ = rest;
            return MediaCondition::parse(&text[5..]);
        }
        if text.starts_with('(') && text.ends_with(')') {
            let inner = text[1..text.len() - 1].trim();
            if let Some((name, value)) = inner.split_once(':') {
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim();
                return MediaCondition::Feature {
                    name,
                    value: if value.is_empty() {
                        None
                    } else {
                        Some(parse_value(value))
                    },
                };
            }
            return MediaCondition::Feature {
                name: inner.to_ascii_lowercase(),
                value: None,
            };
        }
        if text
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '-')
        {
            return MediaCondition::MediaType(lower);
        }
        MediaCondition::Unknown(text.to_string())
    }

    /// Evaluate against a viewport and a theme. `None` means
    /// "cannot tell" (unknown feature or type); `em` in a feature
    /// resolves against the initial 16px font size.
    /// `prefers-color-scheme: dark` / `: light` compare against
    /// `theme`; a bare `(prefers-color-scheme)` (no value) is
    /// `None`.
    pub fn eval(
        &self,
        viewport_width: f32,
        viewport_height: f32,
        theme: Theme,
    ) -> Option<bool> {
        match self {
            MediaCondition::MediaType(t) => match t.as_str() {
                "screen" | "all" => Some(true),
                "print" => Some(false),
                _ => None,
            },
            MediaCondition::Feature { name, value } => {
                if name == "prefers-color-scheme" {
                    return match value {
                        None => None,
                        Some(Value::Keyword(k)) if k == "dark" => {
                            Some(theme == Theme::Dark)
                        }
                        Some(Value::Keyword(k)) if k == "light" => {
                            Some(theme == Theme::Light)
                        }
                        Some(_) => None,
                    };
                }
                let px = |v: &Value| -> Option<f32> {
                    match v {
                        Value::Length(x) => Some(*x),
                        Value::Em(x) => Some(x * 16.0),
                        Value::Percentage(p) => Some(p / 100.0 * viewport_width),
                        _ => None,
                    }
                };
                let v = value.as_ref()?;
                let limit = px(v)?;
                match name.as_str() {
                    "max-width" => Some(viewport_width <= limit),
                    "min-width" => Some(viewport_width >= limit),
                    "max-height" => Some(viewport_height <= limit),
                    "min-height" => Some(viewport_height >= limit),
                    "width" => Some((viewport_width - limit).abs() < 0.5),
                    "height" => Some((viewport_height - limit).abs() < 0.5),
                    _ => None,
                }
            }
            MediaCondition::And(parts) => {
                let mut unknown = false;
                for p in parts {
                    match p.eval(viewport_width, viewport_height, theme) {
                        Some(false) => return Some(false),
                        None => unknown = true,
                        Some(true) => {}
                    }
                }
                if unknown {
                    None
                } else {
                    Some(true)
                }
            }
            MediaCondition::Or(parts) => {
                let mut unknown = false;
                for p in parts {
                    match p.eval(viewport_width, viewport_height, theme) {
                        Some(true) => return Some(true),
                        None => unknown = true,
                        Some(false) => {}
                    }
                }
                if unknown {
                    None
                } else {
                    Some(false)
                }
            }
            MediaCondition::Not(inner) => inner
                .eval(viewport_width, viewport_height, theme)
                .map(|b| !b),
            MediaCondition::Unknown(_) => None,
        }
    }
}

/// A raw `@supports` condition plus a best-effort evaluation for
/// `(property: value)` pairs the parser understands. Anything else
/// (`selector(...)`, unknown functions, unparseable syntax) evaluates
/// to `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportsCondition {
    pub raw: String,
}

impl SupportsCondition {
    pub fn parse(raw: &str) -> Self {
        SupportsCondition {
            raw: raw.trim().to_string(),
        }
    }

    /// Best-effort eval. A `(property: value)` pair is `Some(true)`
    /// when the property is known and the value parses to a typed
    /// (non-`Raw`) value, `Some(false)` when the property is unknown,
    /// and `None` when the value cannot be judged (e.g. a font stack
    /// kept as `Raw`). `and` / `or` / `not` combine per spec, with
    /// `None` propagating unless another branch decides the result.
    pub fn eval(&self) -> Option<bool> {
        eval_supports(&self.raw)
    }
}

/// Properties the parser recognises for `@supports` evaluation and
/// for the `font` shorthand's keyword checks. This is a judgement
/// list, not an exhaustive CSS property table.
pub const KNOWN_PROPERTIES: &[&str] = &[
    "display",
    "position",
    "top",
    "right",
    "bottom",
    "left",
    "width",
    "height",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "border",
    "border-radius",
    "border-collapse",
    "border-spacing",
    "box-shadow",
    "box-sizing",
    "color",
    "background",
    "background-color",
    "background-image",
    "background-size",
    "background-repeat",
    "background-position",
    "font",
    "font-family",
    "font-size",
    "font-style",
    "font-variant",
    "font-weight",
    "line-height",
    "text-align",
    "text-decoration",
    "white-space",
    "vertical-align",
    "overflow",
    "overflow-x",
    "overflow-y",
    "overscroll-behavior",
    "overscroll-behavior-x",
    "overscroll-behavior-y",
    "flex",
    "flex-direction",
    "flex-wrap",
    "flex-grow",
    "flex-shrink",
    "flex-basis",
    "justify-content",
    "align-items",
    "align-content",
    "grid",
    "grid-template-columns",
    "grid-template-rows",
    "grid-column",
    "grid-row",
    "grid-area",
    "grid-auto-flow",
    "gap",
    "grid-gap",
    "row-gap",
    "column-gap",
    "list-style",
    "list-style-type",
    "float",
    "clear",
    "visibility",
    "opacity",
    "z-index",
    "cursor",
    "transform",
    "transform-origin",
    "transform-box",
    "will-change",
    "transition",
    "transition-property",
    "transition-duration",
    "transition-timing-function",
    "transition-delay",
    "animation",
    "animation-name",
    "animation-duration",
    "animation-timing-function",
    "animation-delay",
    "animation-iteration-count",
    "animation-direction",
    "animation-fill-mode",
    "animation-play-state",
    "inset",
    "content",
];

fn eval_supports(text: &str) -> Option<bool> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let lower = text.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("not ") {
        let _ = rest;
        return eval_supports(&text[4..]).map(|b| !b);
    }
    let and_parts = split_top_level_keyword(text, "and");
    if and_parts.len() > 1 {
        let mut unknown = false;
        for p in &and_parts {
            match eval_supports(p) {
                Some(false) => return Some(false),
                None => unknown = true,
                Some(true) => {}
            }
        }
        return if unknown { None } else { Some(true) };
    }
    let or_parts = split_top_level_keyword(text, "or");
    if or_parts.len() > 1 {
        let mut unknown = false;
        for p in &or_parts {
            match eval_supports(p) {
                Some(true) => return Some(true),
                None => unknown = true,
                Some(false) => {}
            }
        }
        return if unknown { None } else { Some(false) };
    }
    if text.starts_with('(') && text.ends_with(')') && balanced_parens(text) {
        let inner = text[1..text.len() - 1].trim();
        let inner_lower = inner.to_ascii_lowercase();
        if inner_lower.starts_with("selector(") || inner_lower.starts_with("font-tech(") {
            return None;
        }
        if let Some((prop, value)) = split_top_level_once(inner, ':') {
            let prop = prop.trim().to_ascii_lowercase();
            let value = value.trim();
            if value.is_empty() {
                return Some(false);
            }
            if !KNOWN_PROPERTIES.contains(&prop.as_str()) {
                return Some(false);
            }
            return match parse_value(value) {
                Value::Raw(_) => None,
                _ => Some(true),
            };
        }
        return None;
    }
    None
}

fn balanced_parens(text: &str) -> bool {
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// One `@import` rule. Fetching is the integration's job, not the
/// parser's: this only records what was imported, from where, and
/// under what conditions. `layer` is `Some("")` for the anonymous
/// `layer` keyword, `Some(name)` for `layer(name)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Import {
    pub url: String,
    pub layer: Option<String>,
    pub supports: Option<String>,
    pub media: Option<String>,
}

/// One `@font-face` descriptor block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FontFace {
    pub family: Option<String>,
    pub src: Option<String>,
    pub unicode_range: Option<String>,
    pub display: Option<String>,
    pub weight: Option<String>,
    pub style: Option<String>,
}

/// One `@keyframes` rule: name plus its parsed steps (M6 Rec 10:
/// the raw blob is gone — steps carry selectors + declarations).
/// No consumer runs animations yet (the engine is a later phase),
/// but the structure is what it will read.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyframes {
    pub name: String,
    pub steps: Vec<KeyframeStep>,
}

/// One `selector, selector { declarations }` rule, with its cascade
/// data attached. `media` / `supports` are the conditions of the
/// enclosing `@media` / `@supports` group rules, if any — rules are
/// NOT flattened at parse time, because viewport/feature state can
/// change after parsing.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: Vec<Declaration>,
    pub origin: Origin,
    /// Cascade layer name, or `None` for unlayered rules. Unlayered
    /// rules beat layered rules for normal declarations.
    pub layer: Option<String>,
    /// Global sequence across the whole stylesheet, in document order
    /// (UA sheet: 0..). Last of equal key wins.
    pub source_order: u32,
    pub media: Option<MediaCondition>,
    pub supports: Option<String>,
}

impl Rule {
    /// The cascade sort key for one declaration of this rule, as a
    /// [`CascadeKey`]. `layer_order` is resolved by the caller from
    /// [`Stylesheet::layer_order`]: higher wins for normal
    /// declarations, unlayered is `i32::MAX`. (For `!important`
    /// declarations the layer order reverses — the caller must invert
    /// it; see Cascade 5.) Specificity here is the max over the
    /// rule's selectors; the per-matching-selector key is the
    /// layout pass's job.
    pub fn sort_key(&self, decl: &Declaration, layer_order: i32, inline: bool) -> CascadeKey {
        CascadeKey::new(self, decl, layer_order, inline)
    }
}

/// The cascade sort key for one declaration: `(origin rank,
/// important, layer order, inline, specificity, source order)`,
/// compared lexicographically — only consult the next criterion on a
/// tie. Higher wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CascadeKey {
    pub origin_rank: u8,
    pub important: bool,
    pub layer_order: i32,
    pub inline: bool,
    pub specificity: (u32, u32, u32),
    pub source_order: u32,
}

impl CascadeKey {
    pub fn new(rule: &Rule, decl: &Declaration, layer_order: i32, inline: bool) -> Self {
        let (origin_rank, important, lo, inl, specificity, source_order) =
            sort_key(rule, decl, layer_order, inline);
        CascadeKey {
            origin_rank,
            important,
            layer_order: lo,
            inline: inl,
            specificity,
            source_order,
        }
    }
}

/// The cascade sort key for one declaration of `rule`, as a plain
/// tuple: `(origin rank, important, layer order, inline,
/// specificity, source order)`. See [`CascadeKey`] for the layer
/// order convention and the max-specificity approximation.
pub fn sort_key(
    rule: &Rule,
    decl: &Declaration,
    layer_order: i32,
    inline: bool,
) -> (u8, bool, i32, bool, (u32, u32, u32), u32) {
    let specificity = rule
        .selectors
        .iter()
        .map(|s| s.specificity())
        .max()
        .unwrap_or((0, 0, 0));
    (
        rule.origin.rank(decl.important),
        decl.important,
        layer_order,
        inline,
        specificity,
        rule.source_order,
    )
}

/// A parsed stylesheet: rules in source order (cascade order is
/// applied by the styler, not stored here), plus what was skipped and
/// why, plus the at-rule data the styler and loader need.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    /// The `rules_by_pseudo_class` index (Stage 8 Rec 9): which
    /// rules mention `:hover` / `:active` / `:focus`, and how far
    /// a state change on one element can reach. It also carries
    /// the Stage 10 Rec 4 `:has()` argument inventory
    /// (`PseudoClassIndex::has`). Rebuilt by
    /// [`Stylesheet::rebuild_pseudo_class_index`], which parsing
    /// and sheet merging call; pushing rules into `rules` by hand
    /// must rebuild it too, or hover/active/focus restyles will
    /// miss rules.
    pub pseudo_class_index: PseudoClassIndex,
    /// Human-readable notes for every skipped rule, declaration, or
    /// at-rule. Empty means the whole sheet parsed cleanly.
    pub errors: Vec<String>,
    /// Cascade layer names in declaration order (first declaration
    /// fixes the order). Block-form `@layer` also declares here.
    pub layers: Vec<String>,
    /// `@import` records, in document order. Fetching is not done here.
    pub imports: Vec<Import>,
    /// `@font-face` descriptor blocks, in document order.
    pub font_faces: Vec<FontFace>,
    /// `@keyframes` rules, in document order (name + parsed steps).
    pub keyframes: Vec<Keyframes>,
}

impl Stylesheet {
    /// Parse CSS text. Never fails outright: malformed rules are
    /// skipped and recorded in [`Stylesheet::errors`].
    pub fn parse(css: &str) -> Self {
        Self::parse_with_origin(css, Origin::Author)
    }

    /// Parse the built-in UA stylesheet ([`UA_CSS`]) with origin
    /// [`Origin::Ua`] and `source_order` starting at 0. This replaces
    /// the hardcoded defaults previously kept in `pluto-layout`.
    pub fn ua() -> Stylesheet {
        Self::parse_with_origin(UA_CSS, Origin::Ua)
    }

    fn parse_with_origin(css: &str, origin: Origin) -> Self {
        let mut sheet = Stylesheet::default();
        let mut order = 0u32;
        let mut seen_rule = false;
        let ctx = RuleCtx::default();
        parse_rules(css, &mut sheet, &mut order, &ctx, origin, &mut seen_rule, true);
        sheet.rebuild_pseudo_class_index();
        sheet
    }

    /// Rebuild the [`PseudoClassIndex`] from the current rules
    /// (Stage 8 Rec 9). Call after pushing rules by hand; parsing
    /// and sheet merging call it already.
    pub fn rebuild_pseudo_class_index(&mut self) {
        self.pseudo_class_index = PseudoClassIndex::for_rules(&self.rules);
    }

    /// Resolve a layer name to its sort position: the index in
    /// [`Stylesheet::layers`] (later-declared wins for normal
    /// declarations), `i32::MAX` for unlayered rules, and
    /// `layers.len()` for a name that was never declared (defensive;
    /// block-form `@layer` always declares first).
    pub fn layer_order(&self, layer: &Option<String>) -> i32 {
        match layer {
            None => i32::MAX,
            Some(name) => self
                .layers
                .iter()
                .position(|l| l == name)
                .map(|i| i as i32)
                .unwrap_or(self.layers.len() as i32),
        }
    }

    /// Every `background-image` URL this sheet references, in rule
    /// order, deduped (Stage 11 Rec 3): `background-image:
    /// url(...)` declarations (typed [`Value::Url`]) plus URLs
    /// hiding in `background` shorthand values (scanned with
    /// [`extract_css_urls`]). The worker fetches + decodes these
    /// through the image pipeline, so a background image is a
    /// resource like any other — `none` contributes nothing. URLs
    /// are returned verbatim (unresolved); the caller resolves
    /// them against the document base.
    pub fn background_image_urls(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for rule in &self.rules {
            for decl in &rule.declarations {
                let urls: Vec<String> = match (&decl.property[..], &decl.value) {
                    ("background-image", Value::Url(url)) => vec![url.clone()],
                    ("background-image", Value::Keyword(k)) if k == "none" => Vec::new(),
                    ("background-image", Value::Raw(raw)) => extract_css_urls(raw),
                    ("background", Value::Raw(raw)) => extract_css_urls(raw),
                    _ => Vec::new(),
                };
                for url in urls {
                    if !out.contains(&url) {
                        out.push(url);
                    }
                }
            }
        }
        out
    }
}

/// Which elements a pseudo-class state change may restyle, beyond
/// the changed element itself (Stage 8 Rec 9). Over-marking is
/// safe (a bigger subtree relayout); under-marking is a missed
/// restyle, so the scan below is conservative.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PseudoReach {
    /// A rule styles descendants/children of the changed element
    /// (e.g. `a:hover b`, `a:hover > b`).
    pub descendants: bool,
    /// A rule styles following siblings (e.g. `a:hover + b`).
    pub siblings: bool,
    /// A rule styles ancestors of the changed element. Only
    /// `:focus-within` does this: focusing an element matches its
    /// ancestors.
    pub ancestors: bool,
}

impl std::ops::BitOrAssign for PseudoReach {
    fn bitor_assign(&mut self, other: Self) {
        self.descendants |= other.descendants;
        self.siblings |= other.siblings;
        self.ancestors |= other.ancestors;
    }
}

/// The `rules_by_pseudo_class` index (Stage 8 Rec 9): for each
/// dynamic pseudo-class, the rule indices whose selectors mention
/// it, plus how far a state change on one element can reach. A
/// hover/active/focus change re-matches only these rules, on the
/// changed elements plus the reach (descendants/siblings/ancestors
/// when a candidate rule has a combinator reaching them); the
/// resulting computed-style changes flow through the existing
/// RestyleHint/RestyleDamage path.
///
/// The `has` field (Stage 10 Rec 4) is the same idea for `:has()`:
/// parse-time bookkeeping of which simple selectors appear in each
/// `:has()` argument, so a later `:has()` build can ask
/// [`HasInvalidationIndex::could_affect`] before walking ancestors.
/// Matching is still NOT implemented — a selector containing
/// `:has()` never matches — so `has` records bookkeeping only and
/// changes no restyle behaviour.
#[derive(Debug, Clone, Default)]
pub struct PseudoClassIndex {
    pub hover: Vec<usize>,
    pub active: Vec<usize>,
    pub focus: Vec<usize>,
    pub hover_reach: PseudoReach,
    pub active_reach: PseudoReach,
    pub focus_reach: PseudoReach,
    /// `:has()` argument inventory (Stage 10 Rec 4): one entry
    /// per rule whose selectors contain `:has()`. Empty when no
    /// `:has()` rule exists.
    pub has: HasInvalidationIndex,
    /// How far a `:has()`-relevant change can reach: ancestors
    /// only — a `:has()` anchor is always an ancestor of the
    /// element its argument matches. Set (ancestors) exactly when
    /// `has` is non-empty; the anchor re-check itself is still a
    /// stub (see [`HasInvalidationIndex::recheck_anchors`]).
    pub has_reach: PseudoReach,
}

impl PseudoClassIndex {
    /// Build the index for `rules` (rule indices are positions in
    /// the slice). `:has()` arguments are skipped — a selector
    /// containing `:has()` never matches, so its rules can never
    /// apply. `:is()`/`:not()`/`:where()` arguments are scanned:
    /// they match, so a pseudo-class inside one counts, with the
    /// reach taken from the argument's own combinators after the
    /// pseudo plus (conservatively) the outer combinators after
    /// the compound holding the argument.
    pub fn for_rules(rules: &[Rule]) -> Self {
        let mut index = Self::default();
        for (i, rule) in rules.iter().enumerate() {
            for selector in &rule.selectors {
                if selector.has_unimplemented {
                    continue;
                }
                let mut scan = ScanOut::default();
                scan_selector(selector, &mut scan, None);
                if scan.hover {
                    push_unique(&mut index.hover, i);
                    index.hover_reach |= scan.hover_reach;
                }
                if scan.active {
                    push_unique(&mut index.active, i);
                    index.active_reach |= scan.active_reach;
                }
                if scan.focus {
                    push_unique(&mut index.focus, i);
                    index.focus_reach |= scan.focus_reach;
                }
            }
        }
        // `:has()` bookkeeping (Stage 10 Rec 4) is built from the
        // same rules, in the same rebuild: selectors with `:has()`
        // were skipped above (they never match), but their argument
        // inventory is recorded here for the later matching build.
        index.has = HasInvalidationIndex::for_rules(rules);
        if !index.has.is_empty() {
            index.has_reach.ancestors = true;
        }
        index
    }
}

fn push_unique(list: &mut Vec<usize>, value: usize) {
    if !list.contains(&value) {
        list.push(value);
    }
}

/// Per-selector scan output: which dynamic pseudo-classes the
/// selector mentions, and how far each one's state change reaches.
#[derive(Debug, Default)]
struct ScanOut {
    hover: bool,
    active: bool,
    focus: bool,
    hover_reach: PseudoReach,
    active_reach: PseudoReach,
    focus_reach: PseudoReach,
}

/// Scan one complex selector for dynamic pseudo-class mentions.
/// `outer_after` is the combinators after the compound holding
/// this selector when it appears as an `:is()`/`:not()`/`:where()`
/// argument (`None` at the top level): they reach from the element
/// the argument is attached to, so they count toward the reach
/// too.
fn scan_selector(sel: &Selector, out: &mut ScanOut, outer_after: Option<&[Combinator]>) {
    for (ci, compound) in sel.compounds.iter().enumerate() {
        // Combinators after this compound (joining it to the next
        // one, and everything beyond): a rule styles the changed
        // element's descendants/siblings only through these.
        let after_here: &[Combinator] = sel.combinators.get(ci..).unwrap_or(&[]);
        for part in &compound.parts {
            let SimpleSelector::PseudoClass(pc) = part else {
                continue;
            };
            match pc {
                PseudoClass::Hover => {
                    out.hover = true;
                    out.hover_reach |= reach_from(after_here, outer_after);
                }
                PseudoClass::Active => {
                    out.active = true;
                    out.active_reach |= reach_from(after_here, outer_after);
                }
                PseudoClass::Focus | PseudoClass::FocusVisible => {
                    out.focus = true;
                    out.focus_reach |= reach_from(after_here, outer_after);
                }
                PseudoClass::FocusWithin => {
                    out.focus = true;
                    out.focus_reach.ancestors = true;
                    out.focus_reach |= reach_from(after_here, outer_after);
                }
                PseudoClass::Is(args) | PseudoClass::Not(args) | PseudoClass::Where(args) => {
                    for arg in args {
                        scan_selector(arg, out, Some(after_here));
                    }
                }
                // `:has()` never matches — skip its arguments.
                // Everything else is static (structural/link) and
                // cannot change on hover/active/focus.
                _ => {}
            }
        }
    }
}

/// The reach of combinators after a pseudo-class: descendant/child
/// combinators reach descendants, sibling combinators reach
/// following siblings.
fn reach_from(after_here: &[Combinator], outer_after: Option<&[Combinator]>) -> PseudoReach {
    let mut reach = PseudoReach::default();
    for c in after_here.iter().chain(outer_after.into_iter().flatten()) {
        match c {
            Combinator::Descendant | Combinator::Child => reach.descendants = true,
            Combinator::NextSibling | Combinator::SubsequentSibling => reach.siblings = true,
        }
    }
    reach
}

// ---- Stage 10 Rec 4: `:has()` parse-time invalidation bookkeeping ----
//
// `:has()` matching is NOT implemented (a selector containing it
// never matches — see `Selector::has_unimplemented`), and a naive
// implementation would re-run matching on every ancestor after
// every DOM/state change, multiplying the cost of the full restyle
// M2 already does on every build. This section records, at parse
// time, only what a later matching build needs to avoid that scan:
// for each `:has()` rule, which simple selectors appear in its
// arguments. On a change the engine asks
// `HasInvalidationIndex::could_affect` first, and only walks up the
// ancestors re-checking anchors when the changed feature appears
// in some argument. The walk itself is still a stub
// (`HasInvalidationIndex::recheck_anchors` returns empty), so this
// changes no behaviour whatsoever.

/// One simple-selector feature that a DOM or state change can
/// alter, and that a `:has()` argument may mention. The query side
/// of the bookkeeping: a changed element contributes one feature
/// per simple selector it (newly) satisfies, and the index answers
/// whether any `:has()` argument mentions it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HasArgFeature {
    /// An element type name, lowercased (`img`, `div`).
    Type(String),
    /// A class name (`active`).
    Class(String),
    /// An id name (`main`).
    Id(String),
    /// An attribute name, lowercased (`href`); the value is not
    /// part of the gate — any change to a named attribute may
    /// affect an argument that mentions it.
    Attribute(String),
    /// A pseudo-class name, lowercased (`hover`, `first-child`).
    PseudoClass(String),
    /// An unspecified element change (insertion/removal, or a
    /// change the caller cannot name): matches any `:has()` rule,
    /// conservatively. Over-marking is safe; under-marking would
    /// miss a restyle.
    Universal,
}

impl HasArgFeature {
    pub fn type_selector(name: &str) -> Self {
        Self::Type(name.to_ascii_lowercase())
    }
    pub fn class(name: &str) -> Self {
        Self::Class(name.to_string())
    }
    pub fn id(name: &str) -> Self {
        Self::Id(name.to_string())
    }
    pub fn attribute(name: &str) -> Self {
        Self::Attribute(name.to_ascii_lowercase())
    }
    pub fn pseudo_class(name: &str) -> Self {
        Self::PseudoClass(name.to_ascii_lowercase())
    }
}

/// The simple selectors appearing in one `:has()` argument
/// selector, as sets (duplicates collapse; sets are sorted, so two
/// parses of the same sheet build identical inventories). This is
/// the per-argument inventory a later `:has()` build reads without
/// re-parsing: the argument selectors themselves stay stored in
/// the rule, this is only the gate over them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HasArgInventory {
    pub types: BTreeSet<String>,
    pub classes: BTreeSet<String>,
    pub ids: BTreeSet<String>,
    /// Attribute *names* only (see [`HasArgFeature::Attribute`]).
    pub attributes: BTreeSet<String>,
    /// Pseudo-class names (`hover`, `active`, `first-child`, ...),
    /// including ones nested inside `:is()` / `:not()` / `:where()`
    /// in the argument. Structural names are recorded, not
    /// evaluated: a sibling/child-structure change alters them
    /// without any feature on the changed element itself, so the
    /// later build must treat structure changes as affecting any
    /// argument that mentions one — the gate alone cannot see it.
    pub pseudo_classes: BTreeSet<String>,
    /// The argument contains a real `*` (not the parser's implied
    /// subject — see `collect_has_simple_selectors`): any element
    /// may be in scope, so any feature may affect it.
    pub has_universal: bool,
    /// The argument itself contains `:has(...)` (nested). Its
    /// selectors are folded into this inventory; the later build
    /// re-expands them from the stored rule selectors.
    pub has_nested_has: bool,
}

impl HasArgInventory {
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
            && self.classes.is_empty()
            && self.ids.is_empty()
            && self.attributes.is_empty()
            && self.pseudo_classes.is_empty()
            && !self.has_universal
            && !self.has_nested_has
    }

    /// True when a change carrying `feature` could affect an
    /// anchor guarded by this argument. A real `*` in the argument
    /// answers true for every feature (conservative — see
    /// `has_universal`); otherwise the feature must be listed.
    pub fn contains(&self, feature: &HasArgFeature) -> bool {
        if self.has_universal {
            return true;
        }
        match feature {
            HasArgFeature::Type(t) => self.types.contains(t),
            HasArgFeature::Class(c) => self.classes.contains(c),
            HasArgFeature::Id(i) => self.ids.contains(i),
            HasArgFeature::Attribute(a) => self.attributes.contains(a),
            HasArgFeature::PseudoClass(p) => self.pseudo_classes.contains(p),
            HasArgFeature::Universal => !self.is_empty(),
        }
    }

    /// Union `other` into this inventory.
    pub fn merge(&mut self, other: &Self) {
        self.types.extend(other.types.iter().cloned());
        self.classes.extend(other.classes.iter().cloned());
        self.ids.extend(other.ids.iter().cloned());
        self.attributes.extend(other.attributes.iter().cloned());
        self.pseudo_classes
            .extend(other.pseudo_classes.iter().cloned());
        self.has_universal |= other.has_universal;
        self.has_nested_has |= other.has_nested_has;
    }
}

/// One `:has()` argument selector, with its inventory and shape.
/// `selector` is the index into the rule's selector list,
/// `argument` the index into that `:has()`'s forgiving selector
/// list — together they locate the argument in the stored rule
/// without re-parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HasArgEntry {
    pub selector: usize,
    pub argument: usize,
    pub inventory: HasArgInventory,
    /// The argument selector's full combinator chain, in order.
    pub combinators: Vec<Combinator>,
    /// True when the argument is explicitly anchored to the anchor
    /// element: its first combinator is `>` / `+` / `~` (the
    /// parser rewrites a leading `>` as `* > ...`, so this is
    /// `combinators[0]` being a non-descendant combinator).
    /// False means the implicit-descendant (broad) form, e.g.
    /// `body:has(div)` — the argument may match at any depth
    /// below the anchor, so a later re-check must walk all
    /// ancestors, not just the parent.
    pub anchored: bool,
}

impl HasArgEntry {
    /// The broad form: the argument is not explicitly anchored
    /// (see [`HasArgEntry::anchored`]).
    pub fn broad(&self) -> bool {
        !self.anchored
    }
}

/// The `:has()` bookkeeping for one rule: every `:has()` argument
/// in every selector of the rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HasRuleEntry {
    pub rule: usize,
    pub args: Vec<HasArgEntry>,
}

impl HasRuleEntry {
    /// Union of all this rule's argument inventories.
    pub fn inventory(&self) -> HasArgInventory {
        let mut inv = HasArgInventory::default();
        for arg in &self.args {
            inv.merge(&arg.inventory);
        }
        inv
    }

    /// True when any argument of this rule is broad (see
    /// [`HasArgEntry::broad`]).
    pub fn is_broad(&self) -> bool {
        self.args.iter().any(|a| a.broad())
    }
}

/// The `:has()` invalidation index (Stage 10 Rec 4): per-rule
/// argument inventories for a whole stylesheet, plus their union.
/// Built by [`PseudoClassIndex::for_rules`] (and therefore by
/// every `Stylesheet` parse and
/// [`Stylesheet::rebuild_pseudo_class_index`]); pushing rules by
/// hand must rebuild it too, exactly like the `:hover` index.
/// Bookkeeping only — no matching, no behaviour change.
#[derive(Debug, Clone, Default)]
pub struct HasInvalidationIndex {
    /// One entry per rule containing `:has()`, in rule order.
    pub rules: Vec<HasRuleEntry>,
    /// Union of every rule's argument inventory: the gate.
    pub all: HasArgInventory,
}

impl HasInvalidationIndex {
    /// Build the index for `rules` (rule indices are positions in
    /// the slice). Finds `:has()` anywhere in a selector —
    /// including inside `:is()` / `:not()` / `:where()` — and
    /// records each argument's simple selectors.
    pub fn for_rules(rules: &[Rule]) -> Self {
        let mut index = Self::default();
        for (i, rule) in rules.iter().enumerate() {
            let mut entry = HasRuleEntry {
                rule: i,
                args: Vec::new(),
            };
            for (si, selector) in rule.selectors.iter().enumerate() {
                collect_has_args(selector, si, &mut entry.args);
            }
            if !entry.args.is_empty() {
                index.all.merge(&entry.inventory());
                index.rules.push(entry);
            }
        }
        index
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The gate: true exactly when `feature` appears in some
    /// `:has()` argument (or some argument contains a real `*`,
    /// which is conservative-true for every feature). False when
    /// the sheet has no `:has()` rule at all. A later `:has()`
    /// build calls this first on a change and only walks ancestors
    /// when it answers true.
    pub fn could_affect(&self, feature: &HasArgFeature) -> bool {
        !self.is_empty() && self.all.contains(feature)
    }

    /// True when any of `features` could affect an anchor.
    pub fn could_affect_any(&self, features: &[HasArgFeature]) -> bool {
        features.iter().any(|f| self.could_affect(f))
    }

    /// Rule indices whose `:has()` arguments mention `feature`
    /// (or contain a real `*`), in rule order — the candidate
    /// rules a later build re-checks on a change carrying it.
    /// Pure bookkeeping: no matching is performed.
    pub fn candidate_rules(&self, feature: &HasArgFeature) -> Vec<usize> {
        self.rules
            .iter()
            .filter(|entry| entry.args.iter().any(|a| a.inventory.contains(feature)))
            .map(|entry| entry.rule)
            .collect()
    }

    /// TODO(Stage 10 Rec 4): the anchor re-check. For a change
    /// carrying `features`, walk each changed node's ancestors and
    /// re-match the candidate `:has()` anchors
    /// (`candidate_rules`), returning the rule indices whose
    /// anchors need restyling. NOT IMPLEMENTED: `:has()` matching
    /// is stubbed (`selector_matches` rejects `has_unimplemented`;
    /// `PseudoClass::Has(_) => false`), so no anchor can match
    /// today and this returns empty — behaviour identical to
    /// before this bookkeeping existed. The later matching build
    /// replaces this body; the gate (`could_affect`) and the
    /// inventories above are already complete for it.
    pub fn recheck_anchors(&self, _features: &[HasArgFeature]) -> Vec<usize> {
        Vec::new()
    }
}

/// Collect every `:has()` argument in `sel` (recursing into
/// `:is()` / `:not()` / `:where()` arguments, where a `:has()`
/// may also hide) into `out`.
fn collect_has_args(sel: &Selector, selector_idx: usize, out: &mut Vec<HasArgEntry>) {
    for compound in &sel.compounds {
        for part in &compound.parts {
            let SimpleSelector::PseudoClass(pc) = part else {
                continue;
            };
            match pc {
                PseudoClass::Has(args) => {
                    for (ai, arg) in args.iter().enumerate() {
                        out.push(HasArgEntry {
                            selector: selector_idx,
                            argument: ai,
                            inventory: inventory_for_selector(arg),
                            combinators: arg.combinators.clone(),
                            anchored: matches!(
                                arg.combinators.first(),
                                Some(
                                    Combinator::Child
                                        | Combinator::NextSibling
                                        | Combinator::SubsequentSibling
                                )
                            ),
                        });
                    }
                }
                PseudoClass::Is(args) | PseudoClass::Not(args) | PseudoClass::Where(args) => {
                    for arg in args {
                        collect_has_args(arg, selector_idx, out);
                    }
                }
                _ => {}
            }
        }
    }
}

/// The inventory of one `:has()` argument selector.
fn inventory_for_selector(sel: &Selector) -> HasArgInventory {
    let mut inv = HasArgInventory::default();
    collect_has_simple_selectors(sel, &mut inv);
    inv
}

/// Record every simple selector in `sel` into `inv`, recursing
/// into `:is()` / `:not()` / `:where()` / nested `:has()`.
/// The parser rewrites a leading relative combinator as `* > ...`
/// (see `parse_forgiving_selector_list`), so a bare `*` as the
/// first compound of a multi-compound argument is the implied
/// subject, not a selector the author wrote: it is not counted as
/// `has_universal` (the rest of the argument discriminates). A
/// lone `*` argument, or a `*` anywhere else, is real and counts.
fn collect_has_simple_selectors(sel: &Selector, inv: &mut HasArgInventory) {
    for (ci, compound) in sel.compounds.iter().enumerate() {
        let implied_subject = ci == 0 && !sel.combinators.is_empty();
        for part in &compound.parts {
            match part {
                SimpleSelector::Universal => {
                    if !implied_subject {
                        inv.has_universal = true;
                    }
                }
                SimpleSelector::Type(t) => {
                    inv.types.insert(t.clone());
                }
                SimpleSelector::Class(c) => {
                    inv.classes.insert(c.clone());
                }
                SimpleSelector::Id(i) => {
                    inv.ids.insert(i.clone());
                }
                SimpleSelector::Attribute { name, .. } => {
                    inv.attributes.insert(name.to_ascii_lowercase());
                }
                SimpleSelector::PseudoClass(pc) => collect_has_pseudo(pc, inv),
                SimpleSelector::PseudoElement(_) => {}
            }
        }
    }
}

/// Record one pseudo-class from a `:has()` argument into `inv`.
fn collect_has_pseudo(pc: &PseudoClass, inv: &mut HasArgInventory) {
    match pc {
        PseudoClass::Has(args) => {
            inv.has_nested_has = true;
            for arg in args {
                collect_has_simple_selectors(arg, inv);
            }
        }
        PseudoClass::Is(args) | PseudoClass::Not(args) | PseudoClass::Where(args) => {
            for arg in args {
                collect_has_simple_selectors(arg, inv);
            }
        }
        other => {
            inv.pseudo_classes.insert(pseudo_class_name(other));
        }
    }
}

/// The canonical (lowercase, hyphenated) name of a pseudo-class
/// variant, for inventory and query purposes.
fn pseudo_class_name(pc: &PseudoClass) -> String {
    match pc {
        PseudoClass::Root => "root",
        PseudoClass::FirstChild => "first-child",
        PseudoClass::LastChild => "last-child",
        PseudoClass::NthChild(_) => "nth-child",
        PseudoClass::NthLastChild(_) => "nth-last-child",
        PseudoClass::OnlyChild => "only-child",
        PseudoClass::Empty => "empty",
        PseudoClass::FirstOfType => "first-of-type",
        PseudoClass::LastOfType => "last-of-type",
        PseudoClass::NthOfType(_) => "nth-of-type",
        PseudoClass::NthLastOfType(_) => "nth-last-of-type",
        PseudoClass::OnlyOfType => "only-of-type",
        PseudoClass::Link => "link",
        PseudoClass::Visited => "visited",
        PseudoClass::AnyLink => "any-link",
        PseudoClass::Hover => "hover",
        PseudoClass::Active => "active",
        PseudoClass::Focus => "focus",
        PseudoClass::FocusVisible => "focus-visible",
        PseudoClass::FocusWithin => "focus-within",
        PseudoClass::Valid => "valid",
        PseudoClass::Invalid => "invalid",
        PseudoClass::Required => "required",
        PseudoClass::Optional => "optional",
        PseudoClass::Disabled => "disabled",
        PseudoClass::Checked => "checked",
        PseudoClass::ReadOnly => "read-only",
        // Logical variants are recursed into by the caller and
        // never reach here; name them defensively anyway.
        PseudoClass::Not(_) => "not",
        PseudoClass::Is(_) => "is",
        PseudoClass::Where(_) => "where",
        PseudoClass::Has(_) => "has",
    }
    .to_string()
}

/// Rule context threaded through nested group rules: the enclosing
/// layer and conditions. Carried, not flattened.
#[derive(Debug, Clone, Default)]
struct RuleCtx {
    layer: Option<String>,
    media: Option<MediaCondition>,
    supports: Option<String>,
}

/// The browser's built-in stylesheet, as real CSS text parsed by the
/// same parser as author sheets (origin UA, source order 0..). Values
/// follow the usual UA defaults: body margin 8px, `h1` 2em bold with
/// 0.67em margins, `p`/`ul` 1em vertical margins, `ul` disc with
/// 40px left padding, links `#0000EE` underlined, `head`/`title`/
/// `style`/`script`/`meta`/`link`/`base` hidden, table/form defaults.
pub const UA_CSS: &str = r#"
html { display: block; }
head, title, style, script, meta, link, base { display: none; }
body { display: block; margin: 8px; }
div, section, article, header, footer, nav, main, aside, figure, figcaption, blockquote, pre, hr, form, fieldset, address, dl, dt, dd, details, summary { display: block; }
span, a, b, i, em, strong, small, code, sub, sup, u, s, mark, abbr, cite, q, time, kbd, samp, var, bdi, bdo, data, wbr, label, output { display: inline; }
h1 { display: block; font-size: 2em; font-weight: bold; margin: 0.67em 0; }
h2 { display: block; font-size: 1.5em; font-weight: bold; margin: 0.83em 0; }
h3 { display: block; font-size: 1.17em; font-weight: bold; margin: 1em 0; }
h4 { display: block; font-size: 1em; font-weight: bold; margin: 1.33em 0; }
h5 { display: block; font-size: 0.83em; font-weight: bold; margin: 1.67em 0; }
h6 { display: block; font-size: 0.67em; font-weight: bold; margin: 2.33em 0; }
p { display: block; margin: 1em 0; }
ul { display: block; list-style-type: disc; margin: 1em 0; padding-left: 40px; }
ol { display: block; list-style-type: decimal; margin: 1em 0; padding-left: 40px; }
li { display: list-item; }
a { color: #0000EE; text-decoration: underline; }
a:visited { color: #551A8B; }
b, strong { font-weight: bold; }
i, em { font-style: italic; }
small { font-size: 0.83em; }
blockquote { display: block; margin: 1em 40px; }
figure { display: block; margin: 1em 40px; }
pre { display: block; margin: 1em 0; white-space: pre; }
hr { display: block; margin: 0.5em auto; border: 1px inset; }
fieldset { display: block; margin: 0 2px; padding: 0.35em 0.75em 0.625em; border: 1px solid; }
table { display: table; border-collapse: separate; border-spacing: 2px; }
thead { display: table-header-group; }
tbody { display: table-row-group; }
tfoot { display: table-footer-group; }
tr { display: table-row; }
td, th { display: table-cell; padding: 1px; }
th { font-weight: bold; text-align: center; }
caption { display: table-caption; text-align: center; }
col { display: table-column; }
colgroup { display: table-column-group; }
button, input, select, textarea { display: inline-block; }
button { padding: 1px 6px; }
input { padding: 1px; }
input, select, textarea { border: 1px solid; background: #ffffff; color: #000000; }
input { min-width: 150px; }
textarea { white-space: pre; }
input[type="checkbox"], input[type="radio"] { min-width: 0; }
img { display: inline; }
/* Stage 15 (M7 Phase 1, Rec 2): `<svg>` and `<canvas>` are replaced
   elements — atomic leaves sized by their intrinsic dimensions, never
   inline containers whose children would leak text into flow. The
   `inline-block` display routes them through `build_atomic` (the same
   leaf the `img` branch builds); layout also special-cases the tags
   by namespace, so an author `display: inline` cannot re-open the
   leak. */
svg, canvas { display: inline-block; }
"#;

// ---- top-level parsing ----

/// Parse rules from `text` into `sheet`, threading the cascade
/// context (`ctx`), the global source-order counter, and the
/// "have we seen a real rule yet" flag used by `@import`.
/// `top_level` is false inside group rules (`@import` is only valid
/// at the top level).
fn parse_rules(
    text: &str,
    sheet: &mut Stylesheet,
    order: &mut u32,
    ctx: &RuleCtx,
    origin: Origin,
    seen_rule: &mut bool,
    top_level: bool,
) {
    let mut cur = Cursor::new(text);
    loop {
        cur.skip_ws_comments();
        if cur.eof() {
            break;
        }
        let start_pos = cur.pos;
        if cur.rest().starts_with('@') {
            handle_at_rule(
                &mut cur, sheet, order, ctx, origin, seen_rule, top_level, start_pos,
            );
            continue;
        }
        let Some(prelude) = cur.read_until('{') else {
            if !cur.rest().trim().is_empty() {
                sheet.errors.push(format!(
                    "rule without '{{' skipped at byte {start_pos}: {:?}",
                    cur.rest().trim()
                ));
            }
            break;
        };
        let block_start = cur.pos;
        let (block, closed) = match cur.read_balanced_block() {
            Some(b) => (b, true),
            None => {
                sheet.errors.push(format!(
                    "unclosed rule block at byte {start_pos} (closed at EOF)"
                ));
                (&text[block_start..], false)
            }
        };
        let selectors = match parse_selector_list(prelude, &mut sheet.errors) {
            Some(sels) => sels,
            None => {
                if !closed {
                    break;
                }
                continue;
            }
        };
        let mut rule = Rule {
            selectors,
            declarations: Vec::new(),
            origin,
            layer: ctx.layer.clone(),
            source_order: *order,
            media: ctx.media.clone(),
            supports: ctx.supports.clone(),
        };
        *order += 1;
        *seen_rule = true;
        parse_declarations(block, &mut rule, &mut sheet.errors);
        sheet.rules.push(rule);
        if !closed {
            break;
        }
    }
}

/// Handle one at-rule; `cur` is positioned at the `@`.
#[allow(clippy::too_many_arguments)]
fn handle_at_rule(
    cur: &mut Cursor<'_>,
    sheet: &mut Stylesheet,
    order: &mut u32,
    ctx: &RuleCtx,
    origin: Origin,
    seen_rule: &mut bool,
    top_level: bool,
    start_pos: usize,
) {
    // Read the at-rule name: `@media`, `@-webkit-keyframes`, ...
    let rest = cur.rest();
    let mut name_end = 1;
    for (i, c) in rest.char_indices().skip(1) {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            name_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    let name = rest[1..name_end].to_ascii_lowercase();
    cur.pos += name_end;
    let Some((off, term)) = cur.find_top_level(&[';', '{']) else {
        sheet
            .errors
            .push(format!("unterminated at-rule at byte {start_pos}: @{name}"));
        cur.pos = cur.s.len();
        return;
    };
    let prelude = cur.rest()[..off].trim().to_string();
    match term {
        ';' => {
            cur.pos += off + 1;
            match name.as_str() {
                "charset" => {
                    // Ignored: UTF-8 is assumed. Never an error.
                }
                "import" => {
                    if !top_level || *seen_rule {
                        sheet.errors.push(format!(
                            "@import ignored at byte {start_pos}: must precede all rules"
                        ));
                    } else {
                        match parse_import(&prelude) {
                            Some(import) => sheet.imports.push(import),
                            None => sheet.errors.push(format!(
                                "unparseable @import skipped at byte {start_pos}: {prelude:?}"
                            )),
                        }
                    }
                }
                "namespace" => {
                    sheet.errors.push(format!(
                        "@namespace skipped at byte {start_pos} (XML only): {prelude:?}"
                    ));
                    *seen_rule = true;
                }
                "layer" => {
                    // Statement form: `@layer a, b;` declares order.
                    // First declaration fixes it.
                    for layer in prelude.split(',') {
                        let layer = layer.trim();
                        if !layer.is_empty() && !sheet.layers.contains(&layer.to_string()) {
                            sheet.layers.push(layer.to_string());
                        }
                    }
                }
                _ => {
                    sheet.errors.push(format!(
                        "unknown at-rule skipped at byte {start_pos}: @{name}"
                    ));
                    *seen_rule = true;
                }
            }
        }
        '{' => {
            cur.pos += off + 1; // step past the '{'
            let block_start = cur.pos;
            let (block, closed) = match cur.read_balanced_block() {
                Some(b) => (b, true),
                None => {
                    sheet.errors.push(format!(
                        "unclosed at-rule block at byte {start_pos} (closed at EOF): @{name}"
                    ));
                    (&cur.s[block_start..], false)
                }
            };
            match name.as_str() {
                "media" => {
                    *seen_rule = true;
                    let cond = MediaCondition::parse(&prelude);
                    let mut inner = ctx.clone();
                    inner.media = Some(match ctx.media.clone() {
                        Some(outer) => MediaCondition::And(vec![outer, cond]),
                        None => cond,
                    });
                    parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                }
                "supports" => {
                    *seen_rule = true;
                    let mut inner = ctx.clone();
                    inner.supports = Some(match ctx.supports.clone() {
                        Some(outer) => format!("({outer}) and ({prelude})"),
                        None => prelude.clone(),
                    });
                    parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                }
                "layer" => {
                    *seen_rule = true;
                    // Block form: `@layer name { ... }` (one name;
                    // anonymous `@layer { ... }` gets "").
                    let layer_name = prelude.trim().to_string();
                    if layer_name.contains(',') {
                        sheet.errors.push(format!(
                            "@layer block with multiple names skipped at byte {start_pos}: {prelude:?}"
                        ));
                    } else {
                        if !sheet.layers.contains(&layer_name) {
                            sheet.layers.push(layer_name.clone());
                        }
                        let mut inner = ctx.clone();
                        inner.layer = Some(layer_name);
                        parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                    }
                }
                "font-face" => {
                    *seen_rule = true;
                    sheet.font_faces.push(parse_font_face(block));
                }
                "container" | "scope" | "property" => {
                    *seen_rule = true;
                    sheet.errors.push(format!(
                        "@{name} skipped at byte {start_pos} (not implemented yet)"
                    ));
                }
                _ if name == "keyframes" || name.ends_with("-keyframes") => {
                    *seen_rule = true;
                    let kf_name = prelude
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    sheet
                        .keyframes
                        .push(parse_keyframes_block(&kf_name, block));
                }
                _ => {
                    *seen_rule = true;
                    sheet.errors.push(format!(
                        "unknown at-rule skipped at byte {start_pos}: @{name}"
                    ));
                }
            }
            if !closed {
                // EOF was hit inside the block; nothing follows.
                cur.pos = cur.s.len();
            }
        }
        _ => unreachable!(),
    }
}

/// Parse an `@import` prelude (text after `@import`, before `;`):
/// `url("x.css")` or `"x.css"`, then optional `layer` /
/// `layer(name)`, `supports(...)`, and a media query list.
fn parse_import(prelude: &str) -> Option<Import> {
    let mut rest = prelude.trim();
    let url;
    if rest.to_ascii_lowercase().starts_with("url(") {
        let inner_start = 4;
        let mut depth = 1i32; // already inside url('s paren
        let mut end = None;
        for (i, c) in rest.char_indices().skip(inner_start) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end?;
        let inner = rest[inner_start..end].trim();
        url = unquote(inner).unwrap_or_else(|| inner.to_string());
        rest = rest[end + 1..].trim_start();
    } else if rest.starts_with('"') || rest.starts_with('\'') {
        let (u, after) = parse_quoted(rest)?;
        url = u;
        rest = after.trim_start();
    } else {
        return None;
    }
    if url.is_empty() {
        return None;
    }
    let mut import = Import {
        url,
        ..Import::default()
    };
    // Optional `layer` / `layer(name)`, then `supports(...)`, then media.
    loop {
        let lower = rest.to_ascii_lowercase();
        if lower.starts_with("layer(") {
            let end = rest.find(')')?;
            import.layer = Some(rest[6..end].trim().to_string());
            rest = rest[end + 1..].trim_start();
        } else if lower.starts_with("layer")
            && rest[5..]
                .chars()
                .next()
                .map(|c| c.is_whitespace())
                .unwrap_or(true)
        {
            import.layer = Some(String::new());
            rest = rest[5..].trim_start();
        } else if lower.starts_with("supports(") {
            let inner_start = 9;
            let mut depth = 1i32; // already inside supports('s paren
            let mut end = None;
            for (i, c) in rest.char_indices().skip(inner_start) {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = end?;
            import.supports = Some(rest[inner_start..end].trim().to_string());
            rest = rest[end + 1..].trim_start();
        } else {
            break;
        }
    }
    if !rest.is_empty() {
        import.media = Some(rest.to_string());
    }
    Some(import)
}

/// Parse an `@keyframes` block into structured steps: each
/// `from` / `to` / `50%` selector group (comma-separated) with
/// the declarations inside its braces. A step with no parseable
/// selector or no declarations is dropped; an unclosed step ends
/// the block.
fn parse_keyframes_block(name: &str, block: &str) -> Keyframes {
    let mut steps = Vec::new();
    let mut rest = block;
    loop {
        let rest_trimmed = rest.trim_start();
        if rest_trimmed.is_empty() {
            break;
        }
        rest = rest_trimmed;
        let Some(open) = rest.find('{') else {
            break;
        };
        let sel_text = &rest[..open];
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in rest[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else {
            break;
        };
        let decl_block = &rest[open + 1..close];
        let selectors: Vec<KeyframeSelector> = sel_text
            .split(',')
            .filter_map(|t| {
                let t = t.trim().to_ascii_lowercase();
                match t.as_str() {
                    "from" => Some(KeyframeSelector::From),
                    "to" => Some(KeyframeSelector::To),
                    _ => t
                        .strip_suffix('%')
                        .and_then(|n| n.trim().parse::<f32>().ok())
                        .map(KeyframeSelector::Percentage),
                }
            })
            .collect();
        let mut rule = Rule::default();
        let mut errors = Vec::new();
        parse_declarations(decl_block, &mut rule, &mut errors);
        if !selectors.is_empty() && !rule.declarations.is_empty() {
            steps.push(KeyframeStep {
                selectors,
                declarations: rule.declarations,
            });
        }
        rest = &rest[close + 1..];
    }
    Keyframes {
        name: name.to_string(),
        steps,
    }
}

/// Parse a `@font-face` descriptor block into a [`FontFace`].
/// Descriptors are `name: value;` pairs; unknown ones are ignored.
fn parse_font_face(block: &str) -> FontFace {
    let mut face = FontFace::default();
    for chunk in split_top_level(block, ';') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let Some((prop, raw)) = split_top_level_once(chunk, ':') else {
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let value = unquote(raw.trim()).unwrap_or_else(|| raw.trim().to_string());
        match prop.as_str() {
            "font-family" => face.family = Some(value),
            "src" => face.src = Some(value),
            "unicode-range" => face.unicode_range = Some(value),
            "font-display" => face.display = Some(value),
            "font-weight" => face.weight = Some(value),
            "font-style" => face.style = Some(value),
            _ => {}
        }
    }
    face
}

/// Parse a quoted string at the start of `s`: returns the unquoted
/// value and the remainder after the closing quote.
fn parse_quoted(s: &str) -> Option<(String, &str)> {
    let quote = s.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    let mut chars = s[1..].char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                if let Some((_, esc)) = chars.next() {
                    out.push(esc);
                }
            }
            c if c == quote => {
                return Some((out, &s[1 + i + c.len_utf8()..]));
            }
            c => out.push(c),
        }
    }
    None
}

/// Strip matching quotes from a string, if it is quoted.
fn unquote(s: &str) -> Option<String> {
    parse_quoted(s).map(|(v, _)| v)
}

// ---- cursor over the stylesheet text ----

/// Byte cursor over a stylesheet (or a block of one), aware of
/// comments, strings, and nesting depth.
struct Cursor<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Cursor { s, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn skip_ws_comments(&mut self) {
        loop {
            let trimmed = self.rest().trim_start();
            self.pos = self.s.len() - trimmed.len();
            if self.rest().starts_with("/*") {
                match self.rest().find("*/") {
                    Some(p) => self.pos += p + 2,
                    None => {
                        self.pos = self.s.len();
                        return;
                    }
                }
            } else {
                return;
            }
        }
    }

    /// Read up to `target` (outside strings); returns the slice
    /// before it and leaves the cursor just past `target`. `None`
    /// when never found. Note: only strings are skipped — brackets
    /// and parens do NOT hide the target, because a `{` in a rule
    /// prelude always opens the block even inside an (unclosed)
    /// attribute selector or functional pseudo-class.
    fn read_until(&mut self, target: char) -> Option<&'a str> {
        let start = self.pos;
        let mut iter = self.s[self.pos..].char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                _ if c == target => {
                    let end = self.pos + off;
                    let out = &self.s[start..end];
                    self.pos = end + c.len_utf8();
                    return Some(out);
                }
                _ => {}
            }
        }
        None
    }

    /// Read a balanced `{ ... }` block; the cursor must be just past
    /// the opening `{`. Returns the inner text and leaves the cursor
    /// just past the matching `}`. `None` when the block never closes
    /// (cursor ends at EOF).
    fn read_balanced_block(&mut self) -> Option<&'a str> {
        let start = self.pos;
        let mut depth = 0i32;
        let mut iter = self.s[self.pos..].char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                '/' if iter.peek().map(|(_, p)| *p) == Some('*') => {
                    iter.next();
                    let mut prev = ' ';
                    for (_, cc) in iter.by_ref() {
                        if prev == '*' && cc == '/' {
                            break;
                        }
                        prev = cc;
                    }
                }
                '{' => depth += 1,
                '}' => {
                    if depth == 0 {
                        let end = self.pos + off;
                        let inner = &self.s[start..end];
                        self.pos = end + 1;
                        return Some(inner);
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
        self.pos = self.s.len();
        None
    }

    /// Offset (relative to the cursor) and character of the first
    /// top-level `;` or `{` (outside strings/parens/brackets).
    /// Used to tell statement at-rules from block at-rules.
    fn find_top_level(&self, targets: &[char]) -> Option<(usize, char)> {
        let mut depth = 0i32;
        let mut iter = self.rest().char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ if depth == 0 && targets.contains(&c) => {
                    return Some((off, c));
                }
                _ => {}
            }
        }
        None
    }
}

/// Skip a quoted string in a `char_indices` iterator (the opening
/// quote was already consumed).
fn skip_string(
    iter: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    quote: char,
) {
    while let Some((_, c)) = iter.next() {
        if c == '\\' {
            iter.next();
            continue;
        }
        if c == quote {
            break;
        }
    }
}

/// Split `s` on top-level occurrences of `sep` (outside strings,
/// parens, and brackets). The separator itself is dropped.
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                skip_string(&mut iter, c);
            }
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(&s[start..off]);
                start = off + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Split on the first top-level occurrence of `sep`.
fn split_top_level_once(s: &str, sep: char) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                skip_string(&mut iter, c);
            }
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ if c == sep && depth == 0 => {
                return Some((&s[..off], &s[off + c.len_utf8()..]));
            }
            _ => {}
        }
    }
    None
}

/// Split on top-level whitespace (outside strings and parens).
fn split_top_level_ws(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                if start.is_none() {
                    start = Some(off);
                }
                skip_string(&mut iter, c);
            }
            '(' | '[' => {
                if start.is_none() {
                    start = Some(off);
                }
                depth += 1;
            }
            ')' | ']' => depth -= 1,
            _ if c.is_whitespace() && depth == 0 => {
                if let Some(st) = start.take() {
                    out.push(&s[st..off]);
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(off);
                }
            }
        }
    }
    if let Some(st) = start {
        out.push(&s[st..]);
    }
    out
}

/// Split on a top-level keyword (`and` / `or`) matched as a whole
/// word, case-insensitively, outside strings and parens.
fn split_top_level_keyword<'a>(s: &'a str, keyword: &str) -> Vec<&'a str> {
    let lower = s.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_string: Option<char> = None;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if let Some(q) = in_string {
            if c == '\\' {
                i += 2;
                continue;
            }
            if c as u8 == q as u8 {
                in_string = None;
            }
            i += 1;
            continue;
        }
        match c {
            '"' | '\'' => {
                in_string = Some(c);
                i += 1;
            }
            '(' | '[' => {
                depth += 1;
                i += 1;
            }
            ')' | ']' => {
                depth -= 1;
                i += 1;
            }
            _ if depth == 0 && lower[i..].starts_with(keyword) => {
                let before_ok = i == 0
                    || !(bytes[i - 1] as char).is_ascii_alphanumeric()
                        && bytes[i - 1] != b'-'
                        && bytes[i - 1] != b'_';
                let after = i + keyword.len();
                let after_ok = after >= bytes.len()
                    || !(bytes[after] as char).is_ascii_alphanumeric()
                        && bytes[after] != b'-'
                        && bytes[after] != b'_';
                if before_ok && after_ok {
                    out.push(&s[start..i]);
                    start = after;
                    i = after;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    out.push(&s[start..]);
    out
}

// ---- selector parsing ----

/// Parse a selector list. ONE invalid complex selector invalidates
/// the WHOLE list (CSS Syntax L3): returns `None` and records one
/// error, and the caller drops the entire rule.
/// Parse a selector list for `querySelector` (wave 2b): the same
/// parser the stylesheet uses, without a sheet attached. ONE
/// invalid complex selector invalidates the WHOLE list (CSS Syntax
/// L3), so an invalid selector returns an empty list — the caller
/// treats it as "no match", never as a partial match.
pub fn parse_selectors(text: &str) -> Vec<Selector> {
    let mut errors = Vec::new();
    parse_selector_list(text, &mut errors).unwrap_or_default()
}

fn parse_selector_list(text: &str, errors: &mut Vec<String>) -> Option<Vec<Selector>> {
    let mut out = Vec::new();
    for item in split_top_level(text, ',') {
        let item = item.trim();
        if item.is_empty() {
            errors.push(format!(
                "invalid selector list (empty item), whole rule dropped: {text:?}"
            ));
            return None;
        }
        match parse_complex_selector(item) {
            Some(sel) => out.push(sel),
            None => {
                errors.push(format!(
                    "invalid selector {item:?} in list, whole rule dropped: {text:?}"
                ));
                return None;
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse a forgiving selector list (inside `:is()` / `:not()` /
/// `:where()` / `:has()`): invalid entries are dropped, the valid
/// ones are kept.
fn parse_forgiving_selector_list(text: &str) -> Vec<Selector> {
    let mut out = Vec::new();
    for item in split_top_level(text, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        // `:has()` takes relative selectors (`> p`); a leading
        // combinator is stored against an implied universal subject.
        let item = if item.starts_with('>') || item.starts_with('+') || item.starts_with('~') {
            format!("* {item}")
        } else {
            item.to_string()
        };
        if let Some(sel) = parse_complex_selector(&item) {
            out.push(sel);
        }
    }
    out
}

/// Parse one complex selector: compounds joined by combinators.
fn parse_complex_selector(text: &str) -> Option<Selector> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = SelectorParser {
        chars: &chars,
        i: 0,
    };
    let mut sel = Selector::default();
    p.skip_ws();
    if p.eof() {
        return None;
    }
    if matches!(p.peek(), Some('>') | Some('+') | Some('~')) {
        return None; // leading combinator
    }
    sel.compounds.push(p.parse_compound()?);
    loop {
        let had_ws = p.skip_ws();
        if p.eof() {
            break;
        }
        let comb = match p.peek() {
            Some('>') => {
                p.i += 1;
                Combinator::Child
            }
            Some('+') => {
                p.i += 1;
                Combinator::NextSibling
            }
            Some('~') => {
                p.i += 1;
                Combinator::SubsequentSibling
            }
            Some(_) if had_ws => Combinator::Descendant,
            _ => return None,
        };
        p.skip_ws();
        if p.eof() {
            return None; // trailing combinator
        }
        sel.compounds.push(p.parse_compound()?);
        sel.combinators.push(comb);
    }
    // A pseudo-element may only appear on the last compound, as its
    // final part; move it onto the selector.
    for compound in sel.compounds.iter().take(sel.compounds.len().saturating_sub(1)) {
        if compound
            .parts
            .iter()
            .any(|part| matches!(part, SimpleSelector::PseudoElement(_)))
        {
            return None;
        }
    }
    let pseudo = sel
        .compounds
        .last()
        .and_then(|last| last.parts.last())
        .and_then(|part| match part {
            SimpleSelector::PseudoElement(pe) => Some(*pe),
            _ => None,
        });
    if let Some(pe) = pseudo {
        let last = sel.compounds.last_mut().unwrap();
        last.parts.pop();
        sel.pseudo_element = Some(pe);
        if last.parts.is_empty() {
            // `::before` alone is not a selector; keep the
            // compound only if something else matched. A bare
            // pseudo-element has no originating element here.
            return None;
        }
    }
    sel.has_unimplemented = sel.compounds.iter().any(|c| {
        c.parts.iter().any(|part| {
            matches!(part, SimpleSelector::PseudoClass(PseudoClass::Has(_)))
        })
    });
    Some(sel)
}

struct SelectorParser<'a> {
    chars: &'a [char],
    i: usize,
}

impl<'a> SelectorParser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn eof(&self) -> bool {
        self.i >= self.chars.len()
    }

    /// Skip whitespace; true when any was skipped.
    fn skip_ws(&mut self) -> bool {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.i += 1;
        }
        self.i > start
    }

    fn parse_compound(&mut self) -> Option<CompoundSelector> {
        let mut parts = Vec::new();
        let mut saw_pseudo_element = false;
        loop {
            match self.peek() {
                None => break,
                Some(c) if c.is_whitespace() || c == '>' || c == '+' || c == '~' || c == ',' => {
                    break
                }
                Some('*') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    self.i += 1;
                    parts.push(SimpleSelector::Universal);
                }
                Some('#') | Some('.') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    let kind = self.peek().unwrap();
                    self.i += 1;
                    let name = self.parse_name()?;
                    parts.push(if kind == '#' {
                        SimpleSelector::Id(name)
                    } else {
                        SimpleSelector::Class(name)
                    });
                }
                Some('[') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    parts.push(self.parse_attribute()?);
                }
                Some(':') => match self.parse_pseudo()? {
                    PseudoOut::Element(pe) => {
                        if saw_pseudo_element {
                            return None;
                        }
                        saw_pseudo_element = true;
                        parts.push(SimpleSelector::PseudoElement(pe));
                    }
                    PseudoOut::Class(pc) => {
                        if saw_pseudo_element {
                            return None;
                        }
                        parts.push(SimpleSelector::PseudoClass(pc));
                    }
                },
                Some(c) if is_ident_start(c) => {
                    if saw_pseudo_element {
                        return None;
                    }
                    let name = self.parse_name()?;
                    parts.push(SimpleSelector::Type(name.to_ascii_lowercase()));
                }
                _ => return None,
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(CompoundSelector { parts })
        }
    }

    /// An identifier: `[A-Za-z0-9_-]` plus non-ASCII, with `\x`
    /// escapes. `None` when empty.
    fn parse_name(&mut self) -> Option<String> {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                self.i += 1;
                let esc = self.peek()?;
                out.push(esc);
                self.i += 1;
            } else if c.is_ascii_alphanumeric() || c == '-' || c == '_' || !c.is_ascii() {
                out.push(c);
                self.i += 1;
            } else {
                break;
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn parse_attribute(&mut self) -> Option<SimpleSelector> {
        debug_assert_eq!(self.peek(), Some('['));
        self.i += 1;
        self.skip_ws();
        let name = self.parse_name()?;
        self.skip_ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Some(SimpleSelector::Attribute {
                name,
                op: AttrOp::Exists,
                value: None,
                case_sensitive: None,
            });
        }
        let op = match (self.peek(), self.chars.get(self.i + 1).copied()) {
            (Some('='), _) => {
                self.i += 1;
                AttrOp::Equals
            }
            (Some('~'), Some('=')) => {
                self.i += 2;
                AttrOp::Includes
            }
            (Some('|'), Some('=')) => {
                self.i += 2;
                AttrOp::Dash
            }
            (Some('^'), Some('=')) => {
                self.i += 2;
                AttrOp::Prefix
            }
            (Some('$'), Some('=')) => {
                self.i += 2;
                AttrOp::Suffix
            }
            (Some('*'), Some('=')) => {
                self.i += 2;
                AttrOp::Substring
            }
            _ => return None,
        };
        self.skip_ws();
        let value = match self.peek() {
            Some('"') | Some('\'') => {
                let quote = self.peek().unwrap();
                self.i += 1;
                let mut out = String::new();
                loop {
                    match self.peek() {
                        None => return None,
                        Some(c) if c == quote => {
                            self.i += 1;
                            break;
                        }
                        Some('\\') => {
                            self.i += 1;
                            let esc = self.peek()?;
                            out.push(esc);
                            self.i += 1;
                        }
                        Some(c) => {
                            out.push(c);
                            self.i += 1;
                        }
                    }
                }
                out
            }
            Some(_) => self.parse_name()?,
            None => return None,
        };
        self.skip_ws();
        // An `i`/`s` case flag only counts when followed by `]` (or
        // whitespace then `]`).
        let mut case_sensitive = None;
        if let Some(flag) = self.peek()
            && matches!(flag, 'i' | 'I' | 's' | 'S')
            && self
                .chars
                .get(self.i + 1)
                .copied()
                .is_some_and(|c| c == ']' || c.is_whitespace())
        {
            case_sensitive = Some(matches!(flag, 's' | 'S'));
            self.i += 1;
            self.skip_ws();
        }
        if self.peek() != Some(']') {
            return None;
        }
        self.i += 1;
        Some(SimpleSelector::Attribute {
            name,
            op,
            value: Some(value),
            case_sensitive,
        })
    }

    /// Read the balanced `( ... )` after a functional pseudo-class.
    fn read_paren_arg(&mut self) -> Option<String> {
        if self.peek() != Some('(') {
            return None;
        }
        self.i += 1;
        let start = self.i;
        let mut depth = 1i32;
        while let Some(c) = self.peek() {
            match c {
                '"' | '\'' => {
                    let quote = c;
                    self.i += 1;
                    loop {
                        match self.peek() {
                            None => return None,
                            Some('\\') => self.i += 2,
                            Some(c2) if c2 == quote => {
                                self.i += 1;
                                break;
                            }
                            Some(_) => self.i += 1,
                        }
                    }
                }
                '(' => {
                    depth += 1;
                    self.i += 1;
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        let arg: String = self.chars[start..self.i].iter().collect();
                        self.i += 1;
                        return Some(arg);
                    }
                    self.i += 1;
                }
                _ => self.i += 1,
            }
        }
        None
    }

    fn parse_pseudo(&mut self) -> Option<PseudoOut> {
        debug_assert_eq!(self.peek(), Some(':'));
        self.i += 1;
        if self.peek() == Some(':') {
            self.i += 1;
            let name = self.parse_name()?.to_ascii_lowercase();
            return match name.as_str() {
                "before" => Some(PseudoOut::Element(PseudoElement::Before)),
                "after" => Some(PseudoOut::Element(PseudoElement::After)),
                "first-line" => Some(PseudoOut::Element(PseudoElement::FirstLine)),
                "first-letter" => Some(PseudoOut::Element(PseudoElement::FirstLetter)),
                "marker" => Some(PseudoOut::Element(PseudoElement::Marker)),
                "selection" => Some(PseudoOut::Element(PseudoElement::Selection)),
                _ => None,
            };
        }
        let name = self.parse_name()?.to_ascii_lowercase();
        // Legacy single-colon pseudo-elements.
        match name.as_str() {
            "before" => return Some(PseudoOut::Element(PseudoElement::Before)),
            "after" => return Some(PseudoOut::Element(PseudoElement::After)),
            "first-line" => return Some(PseudoOut::Element(PseudoElement::FirstLine)),
            "first-letter" => return Some(PseudoOut::Element(PseudoElement::FirstLetter)),
            _ => {}
        }
        let pc = match name.as_str() {
            "root" => PseudoClass::Root,
            "first-child" => PseudoClass::FirstChild,
            "last-child" => PseudoClass::LastChild,
            "only-child" => PseudoClass::OnlyChild,
            "empty" => PseudoClass::Empty,
            "first-of-type" => PseudoClass::FirstOfType,
            "last-of-type" => PseudoClass::LastOfType,
            "only-of-type" => PseudoClass::OnlyOfType,
            "link" => PseudoClass::Link,
            "visited" => PseudoClass::Visited,
            "any-link" => PseudoClass::AnyLink,
            "hover" => PseudoClass::Hover,
            "active" => PseudoClass::Active,
            "focus" => PseudoClass::Focus,
            "focus-visible" => PseudoClass::FocusVisible,
            "focus-within" => PseudoClass::FocusWithin,
            "valid" => PseudoClass::Valid,
            "invalid" => PseudoClass::Invalid,
            "required" => PseudoClass::Required,
            "optional" => PseudoClass::Optional,
            "disabled" => PseudoClass::Disabled,
            "checked" => PseudoClass::Checked,
            "read-only" => PseudoClass::ReadOnly,
            "nth-child" | "nth-last-child" | "nth-of-type" | "nth-last-of-type" => {
                let arg = self.read_paren_arg()?;
                let ab = parse_an_plus_b(&arg)?;
                match name.as_str() {
                    "nth-child" => PseudoClass::NthChild(ab),
                    "nth-last-child" => PseudoClass::NthLastChild(ab),
                    "nth-of-type" => PseudoClass::NthOfType(ab),
                    _ => PseudoClass::NthLastOfType(ab),
                }
            }
            "not" | "is" | "where" | "has" => {
                let arg = self.read_paren_arg()?;
                let list = parse_forgiving_selector_list(&arg);
                if list.is_empty() {
                    return None;
                }
                match name.as_str() {
                    "not" => PseudoClass::Not(list),
                    "is" => PseudoClass::Is(list),
                    "where" => PseudoClass::Where(list),
                    _ => PseudoClass::Has(list),
                }
            }
            _ => return None, // unknown pseudo-class invalidates the selector
        };
        Some(PseudoOut::Class(pc))
    }
}

enum PseudoOut {
    Element(PseudoElement),
    Class(PseudoClass),
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '-' || !c.is_ascii()
}

/// Parse an `an+b` argument: `odd`, `even`, `3`, `n`, `-n+3`, `2n+1`.
fn parse_an_plus_b(s: &str) -> Option<AnPlusB> {
    let t = s.trim().to_ascii_lowercase();
    if t == "odd" {
        return Some(AnPlusB { a: 2, b: 1 });
    }
    if t == "even" {
        return Some(AnPlusB { a: 2, b: 0 });
    }
    if let Ok(n) = t.parse::<i32>() {
        return Some(AnPlusB { a: 0, b: n });
    }
    let t: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    let npos = t.find('n')?;
    let (a_part, rest) = t.split_at(npos);
    let a = match a_part {
        "" | "+" => 1,
        "-" => -1,
        _ => a_part.parse::<i32>().ok()?,
    };
    let rest = &rest[1..];
    let b = if rest.is_empty() {
        0
    } else {
        rest.parse::<i32>().ok()?
    };
    Some(AnPlusB { a, b })
}

// ---- declarations & shorthands ----

fn decl(property: String, value: Value, important: bool) -> Declaration {
    Declaration {
        property,
        value,
        important,
    }
}

fn parse_declarations(block: &str, rule: &mut Rule, errors: &mut Vec<String>) {
    for chunk in split_top_level(block, ';') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let Some((prop, raw)) = split_top_level_once(chunk, ':') else {
            errors.push(format!("declaration without ':' skipped: {chunk:?}"));
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let mut raw = raw.trim();
        if prop.is_empty() || raw.is_empty() {
            errors.push(format!("empty declaration skipped: {chunk:?}"));
            continue;
        }
        // A second top-level ':' in the value means a missing `;`
        // merged two declarations into one invalid one: both are lost.
        if split_top_level_once(raw, ':').is_some() {
            errors.push(format!(
                "declaration merged by missing ';' dropped: {chunk:?}"
            ));
            continue;
        }
        let mut important = false;
        if let Some(stripped) = strip_important(raw) {
            if stripped.is_empty() {
                errors.push(format!("empty declaration skipped: {chunk:?}"));
                continue;
            }
            raw = stripped;
            important = true;
        }
        match prop.as_str() {
            "margin" | "padding" => match expand_shorthand(&prop, raw, important) {
                Some(decls) => rule.declarations.extend(decls),
                None => rule
                    .declarations
                    .push(decl(prop, parse_value(raw), important)),
            },
            "font" => match expand_font_shorthand(raw, important) {
                Some(decls) => rule.declarations.extend(decls),
                None => errors.push(format!("invalid font shorthand dropped: {chunk:?}")),
            },
            "border-radius" => match parse_border_radius(raw) {
                Some(v) => rule.declarations.push(decl(prop, v, important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            "box-shadow" => match parse_box_shadow(raw) {
                Some(v) => rule.declarations.push(decl(prop, v, important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            // M6 Rec 10: transition / animation shorthands expand
            // to structured values; the longhands keep their
            // parsed single values and are zipped by the layout
            // pass (`apply_decl`).
            "transition" => match parse_transition_shorthand(raw) {
                Some(list) => rule
                    .declarations
                    .push(decl(prop, Value::Transitions(list), important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            "animation" => match parse_animation_shorthand(raw) {
                Some(list) => rule
                    .declarations
                    .push(decl(prop, Value::Animations(list), important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            _ => rule
                .declarations
                .push(decl(prop, parse_value(raw), important)),
        }
    }
}

/// Strip a trailing `!important` (case-insensitive, whitespace
/// allowed around the `!`). Returns the value without it.
fn strip_important(raw: &str) -> Option<&str> {
    let t = raw.trim_end();
    let lower = t.to_ascii_lowercase();
    let stripped = lower.strip_suffix("important")?;
    let stripped = stripped.trim_end();
    let stripped = stripped.strip_suffix('!')?;
    Some(t[..stripped.len()].trim_end())
}

/// Expand `margin`/`padding: 1-4 values` into the four longhands.
/// Returns `None` when a token is not a length, percentage, or `auto`
/// (the declaration is then kept as a single `Raw` value).
fn expand_shorthand(prop: &str, raw: &str, important: bool) -> Option<Vec<Declaration>> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        match parse_value(t) {
            Value::Length(v) => vals.push(Value::Length(v)),
            Value::Percentage(v) => vals.push(Value::Percentage(v)),
            Value::Em(v) => vals.push(Value::Em(v)),
            Value::Rem(v) => vals.push(Value::Rem(v)),
            Value::Vh(v) => vals.push(Value::Vh(v)),
            Value::Vw(v) => vals.push(Value::Vw(v)),
            Value::Keyword(k) if k == "auto" => vals.push(Value::Keyword(k)),
            _ => return None,
        }
    }
    let (top, right, bottom, left) = match vals.len() {
        1 => (
            vals[0].clone(),
            vals[0].clone(),
            vals[0].clone(),
            vals[0].clone(),
        ),
        2 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[0].clone(),
            vals[1].clone(),
        ),
        3 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[2].clone(),
            vals[1].clone(),
        ),
        4 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[2].clone(),
            vals[3].clone(),
        ),
        _ => unreachable!(),
    };
    Some(vec![
        decl(format!("{prop}-top"), top, important),
        decl(format!("{prop}-right"), right, important),
        decl(format!("{prop}-bottom"), bottom, important),
        decl(format!("{prop}-left"), left, important),
    ])
}

const FONT_SIZE_KEYWORDS: &[&str] = &[
    "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "xxx-large",
    "larger", "smaller",
];

/// Expand `font: [style] [variant] [weight] size[/line-height] family`
/// into its longhands. Size + family are required; anything else
/// invalid returns `None` (the declaration is dropped).
fn expand_font_shorthand(raw: &str, important: bool) -> Option<Vec<Declaration>> {
    let tokens = split_top_level_ws(raw);
    if tokens.is_empty() {
        return None;
    }
    let mut i = 0;
    let mut style: Option<String> = None;
    let mut variant: Option<String> = None;
    let mut weight: Option<Value> = None;
    // Style / variant / weight keywords before the size.
    while i < tokens.len() {
        let tok = tokens[i];
        let lower = tok.to_ascii_lowercase();
        match lower.as_str() {
            "italic" | "oblique" => {
                if style.is_some() {
                    return None;
                }
                style = Some(lower);
                i += 1;
            }
            "small-caps" => {
                if variant.is_some() {
                    return None;
                }
                variant = Some(lower);
                i += 1;
            }
            "bold" | "bolder" | "lighter" => {
                if weight.is_some() {
                    return None;
                }
                weight = Some(Value::Keyword(lower));
                i += 1;
            }
            "normal" => {
                i += 1; // `normal` for style/variant/weight: no decl needed
            }
            _ => {
                if let Ok(n) = tok.parse::<f32>() {
                    if (1.0..=1000.0).contains(&n) && weight.is_none() {
                        weight = Some(Value::Raw(tok.to_string()));
                        i += 1;
                    } else {
                        return None;
                    }
                } else {
                    break; // must be the size token
                }
            }
        }
    }
    if i >= tokens.len() {
        return None; // no size, no family
    }
    // Size token, optionally with `/line-height` (which may also be
    // split across tokens: `12px / 1.5`).
    let mut size_tok = tokens[i].to_string();
    i += 1;
    let mut line_height: Option<Value> = None;
    if size_tok.ends_with('/') {
        size_tok.pop();
        if i < tokens.len() {
            line_height = Some(parse_value(tokens[i]));
            i += 1;
        } else {
            return None;
        }
    } else if let Some((size_part, lh_part)) = size_tok.split_once('/') {
        let lh_part = lh_part.to_string();
        size_tok = size_part.to_string();
        if lh_part.is_empty() {
            if i < tokens.len() {
                line_height = Some(parse_value(tokens[i]));
                i += 1;
            } else {
                return None;
            }
        } else {
            line_height = Some(parse_value(&lh_part));
        }
    }
    let size = match parse_value(&size_tok) {
        v @ (Value::Length(_)
        | Value::Percentage(_)
        | Value::Em(_)
        | Value::Rem(_)
        | Value::Vh(_)
        | Value::Vw(_)) => v,
        Value::Keyword(k) if FONT_SIZE_KEYWORDS.contains(&k.as_str()) => Value::Keyword(k),
        _ => return None,
    };
    if matches!(line_height.as_ref(), Some(Value::Raw(_))) {
        return None;
    }
    if i >= tokens.len() {
        return None; // no family
    }
    let family = tokens[i..].join(" ");
    if family.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    if let Some(s) = style {
        out.push(decl(
            "font-style".to_string(),
            Value::Keyword(s),
            important,
        ));
    }
    if let Some(v) = variant {
        out.push(decl(
            "font-variant".to_string(),
            Value::Keyword(v),
            important,
        ));
    }
    if let Some(w) = weight {
        out.push(decl("font-weight".to_string(), w, important));
    }
    out.push(decl("font-size".to_string(), size, important));
    if let Some(lh) = line_height {
        out.push(decl("line-height".to_string(), lh, important));
    }
    out.push(decl(
        "font-family".to_string(),
        Value::Raw(family),
        important,
    ));
    Some(out)
}

/// Parse `border-radius: 1-4 lengths`. The elliptical part after `/`
/// is dropped (not modelled yet). `None` keeps the `Raw` fallback.
fn parse_border_radius(raw: &str) -> Option<Value> {
    let before_slash = raw.split('/').next()?;
    let tokens: Vec<&str> = before_slash.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        vals.push(CssLength::from_value(&parse_value(t))?);
    }
    let (tl, tr, br, bl) = match vals.len() {
        1 => (vals[0], vals[0], vals[0], vals[0]),
        2 => (vals[0], vals[1], vals[0], vals[1]),
        3 => (vals[0], vals[1], vals[2], vals[1]),
        4 => (vals[0], vals[1], vals[2], vals[3]),
        _ => unreachable!(),
    };
    Some(Value::BorderRadius(BorderRadius {
        top_left: tl,
        top_right: tr,
        bottom_right: br,
        bottom_left: bl,
    }))
}

/// Parse `box-shadow: <shadow>, ...`. `None` keeps the `Raw` fallback.
fn parse_box_shadow(raw: &str) -> Option<Value> {
    if raw.trim().eq_ignore_ascii_case("none") {
        return Some(Value::BoxShadow(Vec::new()));
    }
    let mut shadows = Vec::new();
    for part in split_top_level(raw, ',') {
        shadows.push(parse_one_box_shadow(part.trim())?);
    }
    if shadows.is_empty() {
        return None;
    }
    Some(Value::BoxShadow(shadows))
}

fn parse_one_box_shadow(s: &str) -> Option<BoxShadow> {
    let mut inset = false;
    let mut lengths = Vec::new();
    let mut color = None;
    for tok in split_top_level_ws(s) {
        if tok.eq_ignore_ascii_case("inset") {
            inset = true;
            continue;
        }
        if let Some(l) = CssLength::from_value(&parse_value(tok)) {
            lengths.push(l);
            continue;
        }
        if color.is_none() && let Some(c) = parse_color_token(tok) {
            color = Some(c);
            continue;
        }
        return None;
    }
    if lengths.len() < 2 || lengths.len() > 4 {
        return None;
    }
    Some(BoxShadow {
        inset,
        offset_x: lengths[0],
        offset_y: lengths[1],
        blur: lengths.get(2).copied().unwrap_or(CssLength::Px(0.0)),
        spread: lengths.get(3).copied().unwrap_or(CssLength::Px(0.0)),
        color,
    })
}

fn parse_color_token(tok: &str) -> Option<(u8, u8, u8, f32)> {
    match parse_value(tok) {
        Value::Color(r, g, b) => Some((r, g, b, 1.0)),
        Value::ColorA(r, g, b, a) => Some((r, g, b, a)),
        _ => None,
    }
}

// ---- values ----

/// Interpret one raw value string as a typed [`Value`].
pub fn parse_value(raw: &str) -> Value {
    let s = raw.trim();
    let lower = s.to_ascii_lowercase();
    // NOTE: `rem` must be checked before `em` ("2rem" ends in "em").
    if let Some(num) = lower.strip_suffix("rem")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Rem(v);
    }
    if let Some(num) = lower.strip_suffix("em")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Em(v);
    }
    if let Some(num) = lower.strip_suffix("vh")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Vh(v);
    }
    if let Some(num) = lower.strip_suffix("vw")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Vw(v);
    }
    if let Some(num) = lower.strip_suffix("px")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Length(v);
    }
    if let Some(num) = lower.strip_suffix('%')
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Percentage(v);
    }
    // Time: `ms` before `s` ("300ms" ends in "s" too). A negative
    // duration is not a valid duration — it falls through to `Raw`.
    if let Some(num) = lower.strip_suffix("ms")
        && let Ok(v) = num.trim().parse::<f32>()
        && v >= 0.0
    {
        return Value::Time(v / 1000.0);
    }
    if let Some(num) = lower.strip_suffix('s')
        && let Ok(v) = num.trim().parse::<f32>()
        && v >= 0.0
    {
        return Value::Time(v);
    }
    // Angles, in degrees: `deg` / `turn` / `rad`.
    if let Some(num) = lower.strip_suffix("deg")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Angle(v);
    }
    if let Some(num) = lower.strip_suffix("turn")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Angle(v * 360.0);
    }
    if let Some(num) = lower.strip_suffix("rad")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Angle(v * 180.0 / std::f32::consts::PI);
    }
    if let Ok(v) = s.parse::<f32>() {
        return Value::Length(v); // bare number: unitless zero, e.g. `margin: 0`
    }
    if let Some(hex) = lower.strip_prefix('#')
        && let Some((r, g, b, a)) = parse_hex_color(hex)
    {
        if a >= 1.0 {
            return Value::Color(r, g, b);
        }
        return Value::ColorA(r, g, b, a);
    }
    if lower.starts_with("rgba(")
        && lower.ends_with(')')
        && let Some(c) = parse_rgba(&lower[5..lower.len() - 1])
    {
        return Value::ColorA(c.0, c.1, c.2, c.3);
    }
    if lower.starts_with("rgb(")
        && lower.ends_with(')')
        && let Some(c) = parse_rgb(&lower[4..lower.len() - 1])
    {
        return Value::Color(c.0, c.1, c.2);
    }
    if lower.starts_with("hsla(")
        && lower.ends_with(')')
        && let Some(c) = parse_hsl(&lower[5..lower.len() - 1], true)
    {
        return Value::ColorA(c.0, c.1, c.2, c.3);
    }
    if lower.starts_with("hsl(")
        && lower.ends_with(')')
        && let Some(c) = parse_hsl(&lower[4..lower.len() - 1], false)
    {
        return Value::Color(c.0, c.1, c.2);
    }
    if let Some(c) = named_color(&lower) {
        return Value::Color(c.0, c.1, c.2);
    }
    if lower.starts_with("url(") && lower.ends_with(')') {
        // Stage 11 Rec 3: `background-image: url(...)` parses to
        // a typed value instead of `Raw`, so the cascade can read
        // it without re-parsing text.
        let inner = s[4..s.len() - 1]
            .trim()
            .trim_matches(|c| c == '"' || c == '\'')
            .trim();
        if !inner.is_empty() {
            return Value::Url(inner.to_string());
        }
    }
    // `var()` / `calc()` / `clamp()` / transform functions, before
    // the keyword fallthrough (they all contain parens, so the
    // keyword arm below would never claim them — but `Raw` would).
    if lower.starts_with("var(") && lower.ends_with(')') {
        if let Some(v) = parse_var_value(&s[4..s.len() - 1]) {
            return v;
        }
    }
    if lower.starts_with("calc(") && lower.ends_with(')') {
        if let Some(e) = parse_calc_str(&s[5..s.len() - 1]) {
            return Value::Calc(e);
        }
    }
    if lower.starts_with("clamp(") && lower.ends_with(')') {
        let inner = &s[6..s.len() - 1];
        let parts = split_top_level(inner, ',');
        if parts.len() == 3 {
            if let (Some(min), Some(val), Some(max)) = (
                parse_calc_str(parts[0]),
                parse_calc_str(parts[1]),
                parse_calc_str(parts[2]),
            ) {
                return Value::Clamp {
                    min: Box::new(min),
                    val: Box::new(val),
                    max: Box::new(max),
                };
            }
        }
    }
    // CSS gradients (Stage 15, M7 Phase 2, Rec 7): parse before
    // the generic function branch below — `linear-gradient(`
    // contains a paren, so `parse_transform_list` would claim it
    // first. An unparseable gradient falls through to `Raw`.
    if lower.starts_with("linear-gradient(") && lower.ends_with(')') {
        if let Some(g) = parse_gradient(&s[16..s.len() - 1], true) {
            return Value::Gradient(g);
        }
    }
    if lower.starts_with("radial-gradient(") && lower.ends_with(')') {
        if let Some(g) = parse_gradient(&s[16..s.len() - 1], false) {
            return Value::Gradient(g);
        }
    }
    if s.contains('(') {
        if let Some(list) = parse_transform_list(s) {
            return Value::Transform(list);
        }
    }
    if s.split_whitespace().count() == 1
        && s.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
    {
        return Value::Keyword(lower);
    }
    Value::Raw(s.to_string())
}

/// Parse the inside of `var(...)`: `--name` plus an optional
/// fallback (itself a parsed [`Value`]; a nested `var()` in the
/// fallback is allowed — `parse_value` recurses for it).
fn parse_var_value(inner: &str) -> Option<Value> {
    let (name, fallback) = match split_top_level_once(inner, ',') {
        Some((n, f)) => (n.trim(), Some(f.trim())),
        None => (inner.trim(), None),
    };
    if !name.starts_with("--") || name.len() < 3 {
        return None;
    }
    Some(Value::Var {
        name: name.to_string(),
        fallback: fallback.map(|f| Box::new(parse_value(f))),
    })
}

/// Parse one angle token (`45deg` / `0.5turn` / `1rad`) to
/// degrees, or `None` when it is not an angle.
fn parse_angle_value(tok: &str) -> Option<f32> {
    match parse_value(tok) {
        Value::Angle(a) => Some(a),
        _ => None,
    }
}

/// Parse the inside of `linear-gradient(...)` /
/// `radial-gradient(...)` (without the parens) into a
/// [`Gradient`]. `linear` selects the linear form. Missing stop
/// offsets are distributed evenly (first 0, last 1); offsets may
/// be `%` or a bare number (read as a fraction); `px` offsets are
/// not supported and fail the parse. A gradient with fewer than
/// two stops fails the parse.
fn parse_gradient(inner: &str, linear: bool) -> Option<Gradient> {
    let parts = split_top_level(inner, ',');
    if parts.len() < 2 {
        return None;
    }
    let mut rest: &[&str] = &parts;
    let mut kind = if linear {
        GradientKind::Linear { angle_deg: 180.0 }
    } else {
        GradientKind::Radial {
            center: (0.5, 0.5),
            shape: RadialShape::Ellipse,
            extent: RadialExtent::FarthestCorner,
        }
    };
    // The first part may be geometry, not a stop: an angle / `to
    // ...` direction (linear), or a shape / extent / `at ...`
    // position (radial). A part whose first token parses as a
    // colour is a stop, not geometry.
    let first = parts[0].trim();
    let first_is_stop = parse_gradient_stop(first).is_some();
    if !first_is_stop {
        if linear {
            if let Some(angle) = parse_angle_value(first) {
                kind = GradientKind::Linear { angle_deg: angle };
                rest = &parts[1..];
            } else if let Some(angle) = parse_to_direction(first) {
                kind = GradientKind::Linear { angle_deg: angle };
                rest = &parts[1..];
            } else {
                // Unrecognised first part: not a gradient we
                // understand — fail rather than paint wrong.
                return None;
            }
        } else {
            let lower = first.to_ascii_lowercase();
            if lower.contains("circle")
                || lower.contains("ellipse")
                || lower.contains("closest")
                || lower.contains("farthest")
                || lower.contains(" at ")
                || lower.ends_with(" at")
                || lower.starts_with("at ")
            {
                kind = parse_radial_prelude(first)?;
                rest = &parts[1..];
            } else {
                return None;
            }
        }
    }
    let mut stops: Vec<(Option<f32>, [u8; 4])> =
        Vec::with_capacity(rest.len());
    for part in rest {
        let (offset, color) = parse_gradient_stop(part.trim())?;
        stops.push((offset, color));
    }
    if stops.len() < 2 {
        return None;
    }
    // Distribute missing offsets, per CSS: the first stop
    // defaults to 0, the last to 1, and runs between known
    // offsets interpolate evenly.
    if stops[0].0.is_none() {
        stops[0].0 = Some(0.0);
    }
    let last = stops.len() - 1;
    if stops[last].0.is_none() {
        stops[last].0 = Some(1.0);
    }
    let mut resolved: Vec<GradientStop> = Vec::with_capacity(stops.len());
    let mut i = 0;
    while i < stops.len() {
        if stops[i].0.is_some() {
            resolved.push(GradientStop {
                offset: stops[i].0.unwrap(),
                color: stops[i].1,
            });
            i += 1;
            continue;
        }
        // A run of missing offsets starting at i, ending at the
        // next known offset (both ends are known now: the first
        // and last stops were pinned above).
        let mut j = i;
        while j < stops.len() && stops[j].0.is_none() {
            j += 1;
        }
        let start = stops[i - 1].0.unwrap();
        let end = stops[j].0.unwrap();
        let run = j - i;
        for k in 0..run {
            let f = (k + 1) as f32 / (run + 1) as f32;
            resolved.push(GradientStop {
                offset: start + (end - start) * f,
                color: stops[i + k].1,
            });
        }
        i = j;
    }
    // Offsets must be non-decreasing (a stop behind the previous
    // one clamps to it, per CSS).
    let mut prev = f32::NEG_INFINITY;
    for stop in &mut resolved {
        if stop.offset < prev {
            stop.offset = prev;
        }
        prev = stop.offset;
    }
    Some(Gradient {
        kind,
        stops: resolved,
    })
}

/// Parse one gradient stop (`red`, `red 50%`, `#fff 0.2`): the
/// colour plus an optional offset (fraction). `None` when the
/// first token is not a colour.
fn parse_gradient_stop(part: &str) -> Option<(Option<f32>, [u8; 4])> {
    let mut toks = part.split_whitespace();
    let color_tok = toks.next()?;
    let color = match parse_value(color_tok) {
        Value::Color(r, g, b) => [r, g, b, 255],
        Value::ColorA(r, g, b, a) => [r, g, b, (a.clamp(0.0, 1.0) * 255.0) as u8],
        _ => return None,
    };
    let offset = match toks.next() {
        None => None,
        Some(tok) => {
            let tok = tok.trim();
            if let Some(num) = tok.strip_suffix('%')
                && let Ok(v) = num.trim().parse::<f32>()
            {
                Some(v / 100.0)
            } else if let Ok(v) = tok.parse::<f32>() {
                Some(v)
            } else {
                // A second token that is neither % nor a bare
                // number (e.g. `10px`) — unsupported.
                return None;
            }
        }
    };
    if toks.next().is_some() {
        return None;
    }
    Some((offset, color))
}

/// Parse a `to <side>` direction (`to top`, `to bottom right`,
/// ...) to degrees (0 = to top, 90 = to right, 180 = to bottom,
/// 270 = to left).
fn parse_to_direction(s: &str) -> Option<f32> {
    let lower = s.trim().to_ascii_lowercase();
    let rest = lower.strip_prefix("to ")?.trim();
    let words: Vec<&str> = rest.split_whitespace().collect();
    let (mut vertical, mut horizontal): (Option<f32>, Option<f32>) = (None, None);
    for w in words {
        match w {
            "top" => vertical = Some(-1.0),
            "bottom" => vertical = Some(1.0),
            "left" => horizontal = Some(-1.0),
            "right" => horizontal = Some(1.0),
            _ => return None,
        }
    }
    let (v, h) = (vertical.unwrap_or(0.0), horizontal.unwrap_or(0.0));
    if v == 0.0 && h == 0.0 {
        return None;
    }
    // atan2 of (horizontal, -vertical): CSS 0deg is "to top".
    Some(h.atan2(-v).to_degrees().rem_euclid(360.0))
}

/// Parse a radial prelude (`circle`, `ellipse at 20% 30%`,
/// `farthest-side`, `circle closest-corner at left top`, ...).
fn parse_radial_prelude(s: &str) -> Option<GradientKind> {
    let lower = s.trim().to_ascii_lowercase();
    let mut shape = RadialShape::Ellipse;
    let mut extent = RadialExtent::FarthestCorner;
    let mut center = (0.5f32, 0.5f32);
    // Split off an `at <position>` tail first.
    let (head, at) = match lower.split_once(" at ") {
        Some((h, a)) => (h, Some(a)),
        None => (lower.as_str(), None),
    };
    for tok in head.split_whitespace() {
        match tok {
            "circle" => shape = RadialShape::Circle,
            "ellipse" => shape = RadialShape::Ellipse,
            "closest-side" => extent = RadialExtent::ClosestSide,
            "farthest-side" => extent = RadialExtent::FarthestSide,
            "closest-corner" => extent = RadialExtent::ClosestCorner,
            "farthest-corner" => extent = RadialExtent::FarthestCorner,
            "" => {}
            _ => return None,
        }
    }
    if let Some(pos) = at {
        let toks: Vec<&str> = pos.split_whitespace().collect();
        if toks.is_empty() || toks.len() > 2 {
            return None;
        }
        let parse_pos = |t: &str| -> Option<f32> {
            match t {
                "left" | "top" => Some(0.0),
                "center" => Some(0.5),
                "right" | "bottom" => Some(1.0),
                _ => {
                    if let Some(num) = t.strip_suffix('%')
                        && let Ok(v) = num.parse::<f32>()
                    {
                        Some(v / 100.0)
                    } else {
                        None
                    }
                }
            }
        };
        let x = parse_pos(toks[0])?;
        let y = if toks.len() == 2 {
            parse_pos(toks[1])?
        } else {
            0.5
        };
        center = (x, y);
    }
    Some(GradientKind::Radial {
        center,
        shape,
        extent,
    })
}

/// Parse a `transform` function list: `translate(...) scale(...)
/// rotate(...) skew(...) matrix(...)` (comma- or space-separated
/// arguments), `none` → the empty list. Returns `None` when the
/// text is not a transform list at all (so `parse_value` falls
/// through to `Raw`).
fn parse_transform_list(s: &str) -> Option<TransformList> {
    let lower = s.trim().to_ascii_lowercase();
    if lower == "none" {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let len = |tok: &str| -> Option<CssLength> {
        CssLength::from_value(&parse_value(tok))
    };
    let num = |tok: &str| -> Option<f32> { tok.trim().parse::<f32>().ok() };
    loop {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Function name, up to '('.
        let name_start = i;
        while i < bytes.len() && (bytes[i] as char).is_ascii_alphabetic() {
            i += 1;
        }
        if i == name_start || i >= bytes.len() || bytes[i] != b'(' {
            return None;
        }
        let name = s[name_start..i].to_ascii_lowercase();
        // Balanced args up to the matching ')'.
        let mut depth = 0i32;
        let mut end = None;
        let mut j = i;
        while j < bytes.len() {
            match bytes[j] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        let Some(end) = end else {
            return None;
        };
        let args_raw = &s[i + 1..end];
        let args: Vec<&str> = args_raw
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .collect();
        let f = match name.as_str() {
            "translate" => {
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                let x = len(args[0])?;
                let y = if args.len() == 2 {
                    len(args[1])?
                } else {
                    CssLength::Px(0.0)
                };
                TransformFunction::Translate(x, y)
            }
            "translatex" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::TranslateX(len(args[0])?)
            }
            "translatey" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::TranslateY(len(args[0])?)
            }
            "scale" => {
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                let sx = num(args[0])?;
                let sy = if args.len() == 2 { num(args[1])? } else { sx };
                TransformFunction::Scale(sx, sy)
            }
            "scalex" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::ScaleX(num(args[0])?)
            }
            "scaley" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::ScaleY(num(args[0])?)
            }
            "rotate" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::Rotate(parse_angle_value(args[0])?)
            }
            "skew" => {
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                out.push(TransformFunction::SkewX(parse_angle_value(args[0])?));
                if args.len() == 2 {
                    out.push(TransformFunction::SkewY(parse_angle_value(args[1])?));
                }
                i = end + 1;
                continue;
            }
            "skewx" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::SkewX(parse_angle_value(args[0])?)
            }
            "skewy" => {
                if args.len() != 1 {
                    return None;
                }
                TransformFunction::SkewY(parse_angle_value(args[0])?)
            }
            "matrix" => {
                if args.len() != 6 {
                    return None;
                }
                let mut m = [0.0f32; 6];
                for (k, t) in args.iter().enumerate() {
                    m[k] = num(t)?;
                }
                TransformFunction::Matrix(m)
            }
            _ => return None,
        };
        out.push(f);
        i = end + 1;
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// Parse a `transition` shorthand value (comma-separated list of
/// `property duration timing delay` items, in any order per item)
/// into structured [`Transition`]s.
fn parse_transition_shorthand(raw: &str) -> Option<Vec<Transition>> {
    let mut out = Vec::new();
    for item in split_top_level(raw, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let mut t = Transition::default();
        let mut seen_time = false;
        let mut seen_property = false;
        for tok in split_top_level_ws(item) {
            match parse_value(tok) {
                Value::Time(d) => {
                    if !seen_time {
                        t.duration = d;
                        seen_time = true;
                    } else {
                        t.delay = d;
                    }
                }
                _ => {
                    if let Some(e) = Easing::parse(tok) {
                        t.timing = e;
                    } else if !seen_property {
                        t.property = tok.to_ascii_lowercase();
                        seen_property = true;
                    }
                }
            }
        }
        out.push(t);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse an `animation` shorthand value (comma-separated list) into
/// structured [`Animation`]s. Tokens are classified by shape: a
/// time is duration (first) then delay, an easing keyword /
/// function is the timing, `infinite` / a bare number is the
/// iteration count, the direction / fill / play-state keywords
/// name themselves, and the first remaining token is the name.
fn parse_animation_shorthand(raw: &str) -> Option<Vec<Animation>> {
    let mut out = Vec::new();
    for item in split_top_level(raw, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let mut a = Animation::default();
        let mut seen_time = false;
        let mut seen_name = false;
        for tok in split_top_level_ws(item) {
            let lower = tok.to_ascii_lowercase();
            match parse_value(tok) {
                Value::Time(d) => {
                    if !seen_time {
                        a.duration = d;
                        seen_time = true;
                    } else {
                        a.delay = d;
                    }
                }
                _ => {
                    if let Some(e) = Easing::parse(tok) {
                        a.timing = e;
                        continue;
                    }
                    match lower.as_str() {
                        "infinite" => a.iteration_count = IterationCount::Infinite,
                        "normal" => a.direction = AnimationDirection::Normal,
                        "reverse" => a.direction = AnimationDirection::Reverse,
                        "alternate" => a.direction = AnimationDirection::Alternate,
                        "alternate-reverse" => {
                            a.direction = AnimationDirection::AlternateReverse
                        }
                        "none" if !seen_name => {
                            // `none` as a fill keyword only when a
                            // name was already seen; as the first
                            // bare token it is the animation name.
                            a.name = "none".to_string();
                            seen_name = true;
                        }
                        "forwards" => a.fill_mode = FillMode::Forwards,
                        "backwards" => a.fill_mode = FillMode::Backwards,
                        "both" => a.fill_mode = FillMode::Both,
                        "running" => a.play_state = PlayState::Running,
                        "paused" => a.play_state = PlayState::Paused,
                        _ => {
                            if let Ok(n) = tok.parse::<f32>() {
                                a.iteration_count = IterationCount::Finite(n);
                            } else if !seen_name {
                                a.name = tok.to_string();
                                seen_name = true;
                            }
                        }
                    }
                }
            }
        }
        out.push(a);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse a `calc(...)` inner expression into a [`CalcExpr`].
/// Grammar: `expr := term (('+'|'-') term)*`,
/// `term := factor (('*'|'/') factor)*`,
/// `factor := number[unit] | '(' expr ')' | var(...)`.
/// `*` / `/` only scale by a plain number (a length on either
/// side of a number is folded into `Mul`/`Div`); anything else
/// fails the whole parse (`None` → the value stays `Raw`).
fn parse_calc_str(s: &str) -> Option<CalcExpr> {
    let mut p = CalcParser {
        bytes: s.as_bytes(),
        pos: 0,
    };
    let expr = p.parse_expr()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return None;
    }
    Some(expr)
}

struct CalcParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> CalcParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && (self.bytes[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_expr(&mut self) -> Option<CalcExpr> {
        let mut lhs = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = CalcExpr::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(b'-') => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = CalcExpr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_term(&mut self) -> Option<CalcExpr> {
        let mut lhs = self.parse_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    match (lhs, rhs) {
                        (CalcExpr::Num(n), e) => lhs = CalcExpr::Mul(Box::new(e), n),
                        (e, CalcExpr::Num(n)) => lhs = CalcExpr::Mul(Box::new(e), n),
                        _ => return None,
                    }
                }
                Some(b'/') => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    match rhs {
                        CalcExpr::Num(n) => lhs = CalcExpr::Div(Box::new(lhs), n),
                        _ => return None,
                    }
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    fn parse_factor(&mut self) -> Option<CalcExpr> {
        self.skip_ws();
        match self.peek()? {
            b'(' => {
                self.pos += 1;
                let e = self.parse_expr()?;
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return None;
                }
                self.pos += 1;
                Some(e)
            }
            b'v' | b'V' => {
                // `var(--name)` / `var(--name, <fallback expr>)`.
                let rest = &self.bytes[self.pos..];
                let lower: String = rest
                    .iter()
                    .take(4)
                    .map(|b| (*b as char).to_ascii_lowercase())
                    .collect();
                if lower != "var(" {
                    return None;
                }
                self.pos += 4;
                self.skip_ws();
                let name_start = self.pos;
                while let Some(c) = self.peek() {
                    if c == b',' || c == b')' {
                        break;
                    }
                    self.pos += 1;
                }
                let name = std::str::from_utf8(&self.bytes[name_start..self.pos])
                    .ok()?
                    .trim()
                    .to_string();
                if !name.starts_with("--") {
                    return None;
                }
                self.skip_ws();
                let fallback = if self.peek() == Some(b',') {
                    self.pos += 1;
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return None;
                }
                self.pos += 1;
                Some(CalcExpr::Var(name, fallback))
            }
            _ => self.parse_number(),
        }
    }

    /// A number with an optional length unit: `10px`, `50%`,
    /// `2em`, `1.5` (bare number). A leading `+`/`-` is part of
    /// the number when it directly precedes digits.
    fn parse_number(&mut self) -> Option<CalcExpr> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) {
            self.pos += 1;
        }
        let mut seen_digit = false;
        let mut seen_dot = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => {
                    seen_digit = true;
                    self.pos += 1;
                }
                b'.' if !seen_dot => {
                    seen_dot = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        if !seen_digit {
            self.pos = start;
            return None;
        }
        let num_str = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
        let v: f32 = num_str.parse().ok()?;
        let unit_start = self.pos;
        while let Some(c) = self.peek() {
            if (c as char).is_ascii_alphabetic() || c == b'%' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let unit = std::str::from_utf8(&self.bytes[unit_start..self.pos])
            .ok()?
            .to_ascii_lowercase();
        Some(match unit.as_str() {
            "" => CalcExpr::Num(v),
            "px" => CalcExpr::Len(CssLength::Px(v)),
            "%" => CalcExpr::Pct(v),
            "em" => CalcExpr::Len(CssLength::Em(v)),
            "rem" => CalcExpr::Len(CssLength::Rem(v)),
            "vh" => CalcExpr::Len(CssLength::Vh(v)),
            "vw" => CalcExpr::Len(CssLength::Vw(v)),
            _ => return None,
        })
    }
}

/// `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`. Alpha is 1.0 unless the
/// 4/8-digit form says otherwise.
fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8, f32)> {
    let nibble = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let b = hex.as_bytes();
    match b.len() {
        3 => Some((
            nibble(b[0])? * 17,
            nibble(b[1])? * 17,
            nibble(b[2])? * 17,
            1.0,
        )),
        4 => Some((
            nibble(b[0])? * 17,
            nibble(b[1])? * 17,
            nibble(b[2])? * 17,
            nibble(b[3])? as f32 * 17.0 / 255.0,
        )),
        6 => Some((
            nibble(b[0])? * 16 + nibble(b[1])?,
            nibble(b[2])? * 16 + nibble(b[3])?,
            nibble(b[4])? * 16 + nibble(b[5])?,
            1.0,
        )),
        8 => Some((
            nibble(b[0])? * 16 + nibble(b[1])?,
            nibble(b[2])? * 16 + nibble(b[3])?,
            nibble(b[4])? * 16 + nibble(b[5])?,
            (nibble(b[6])? * 16 + nibble(b[7])?) as f32 / 255.0,
        )),
        _ => None,
    }
}

fn parse_channel(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct.trim().parse().ok()?;
        return Some((v / 100.0 * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    s.parse::<u8>().ok()
}

fn parse_alpha(s: &str) -> Option<f32> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct.trim().parse().ok()?;
        return Some((v / 100.0).clamp(0.0, 1.0));
    }
    s.parse::<f32>().ok().map(|v| v.clamp(0.0, 1.0))
}

fn parse_rgb(inner: &str) -> Option<(u8, u8, u8)> {
    let mut parts = inner.split(',');
    let r = parse_channel(parts.next()?)?;
    let g = parse_channel(parts.next()?)?;
    let b = parse_channel(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b))
}

fn parse_rgba(inner: &str) -> Option<(u8, u8, u8, f32)> {
    let mut parts = inner.split(',');
    let r = parse_channel(parts.next()?)?;
    let g = parse_channel(parts.next()?)?;
    let b = parse_channel(parts.next()?)?;
    let a = parse_alpha(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b, a))
}

/// `hsl(h, s%, l%)` / `hsla(h, s%, l%, a)`. Hue may carry a `deg`
/// suffix; `s`/`l` are percentages.
fn parse_hsl(inner: &str, with_alpha: bool) -> Option<(u8, u8, u8, f32)> {
    let mut parts = inner.split(',');
    let h_raw = parts.next()?.trim();
    let h_raw = h_raw.strip_suffix("deg").unwrap_or(h_raw);
    let h: f32 = h_raw.parse().ok()?;
    let s_raw = parts.next()?.trim();
    let s_pct: f32 = s_raw.strip_suffix('%')?.trim().parse().ok()?;
    let l_raw = parts.next()?.trim();
    let l_pct: f32 = l_raw.strip_suffix('%')?.trim().parse().ok()?;
    let a = if with_alpha {
        parse_alpha(parts.next()?)?
    } else {
        1.0
    };
    if parts.next().is_some() {
        return None;
    }
    let (r, g, b) = hsl_to_rgb(h, s_pct / 100.0, l_pct / 100.0);
    Some((r, g, b, a))
}

fn hsl_to_rgb(h_deg: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = (((h_deg % 360.0) + 360.0) % 360.0) / 360.0;
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let conv = |v: f32| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (conv(r), conv(g), conv(b))
}

fn named_color(name: &str) -> Option<(u8, u8, u8)> {
    Some(match name {
        "black" => (0, 0, 0),
        "white" => (255, 255, 255),
        "red" => (255, 0, 0),
        "green" => (0, 128, 0),
        "lime" => (0, 255, 0),
        "blue" => (0, 0, 255),
        "yellow" => (255, 255, 0),
        "cyan" | "aqua" => (0, 255, 255),
        "magenta" | "fuchsia" => (255, 0, 255),
        "gray" | "grey" => (128, 128, 128),
        "silver" => (192, 192, 192),
        "maroon" => (128, 0, 0),
        "navy" => (0, 0, 128),
        "olive" => (128, 128, 0),
        "teal" => (0, 128, 128),
        "purple" => (128, 0, 128),
        "orange" => (255, 165, 0),
        "pink" => (255, 192, 203),
        "brown" => (165, 42, 42),
        "gold" => (255, 215, 0),
        "beige" => (245, 245, 220),
        _ => return None,
    })
}

impl fmt::Display for CssLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CssLength::Px(v) => write!(f, "{v}px"),
            CssLength::Percentage(v) => write!(f, "{v}%"),
            CssLength::Em(v) => write!(f, "{v}em"),
            CssLength::Rem(v) => write!(f, "{v}rem"),
            CssLength::Vh(v) => write!(f, "{v}vh"),
            CssLength::Vw(v) => write!(f, "{v}vw"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Keyword(k) => write!(f, "{k}"),
            Value::Length(v) => write!(f, "{v}px"),
            Value::Percentage(v) => write!(f, "{v}%"),
            Value::Em(v) => write!(f, "{v}em"),
            Value::Rem(v) => write!(f, "{v}rem"),
            Value::Vh(v) => write!(f, "{v}vh"),
            Value::Vw(v) => write!(f, "{v}vw"),
            Value::Color(r, g, b) => write!(f, "rgb({r}, {g}, {b})"),
            Value::ColorA(r, g, b, a) => write!(f, "rgba({r}, {g}, {b}, {a})"),
            Value::BorderRadius(br) => {
                let corners = [br.top_left, br.top_right, br.bottom_right, br.bottom_left];
                if corners.iter().all(|c| *c == corners[0]) {
                    write!(f, "{}", corners[0])
                } else if corners[0] == corners[2] && corners[1] == corners[3] {
                    write!(f, "{} {}", corners[0], corners[1])
                } else if corners[1] == corners[3] {
                    write!(f, "{} {} {}", corners[0], corners[1], corners[2])
                } else {
                    write!(
                        f,
                        "{} {} {} {}",
                        corners[0], corners[1], corners[2], corners[3]
                    )
                }
            }
            Value::BoxShadow(shadows) => {
                if shadows.is_empty() {
                    return write!(f, "none");
                }
                let mut first = true;
                for sh in shadows {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    if sh.inset {
                        write!(f, "inset ")?;
                    }
                    write!(f, "{} {}", sh.offset_x, sh.offset_y)?;
                    if sh.blur != CssLength::Px(0.0) || sh.spread != CssLength::Px(0.0) {
                        write!(f, " {}", sh.blur)?;
                    }
                    if sh.spread != CssLength::Px(0.0) {
                        write!(f, " {}", sh.spread)?;
                    }
                    if let Some((r, g, b, a)) = sh.color {
                        if a >= 1.0 {
                            write!(f, " rgb({r}, {g}, {b})")?;
                        } else {
                            write!(f, " rgba({r}, {g}, {b}, {a})")?;
                        }
                    }
                }
                Ok(())
            }
            Value::Url(url) => write!(f, "url({url})"),
            Value::Time(v) => write!(f, "{v}s"),
            Value::Angle(v) => write!(f, "{v}deg"),
            Value::Transform(list) => {
                if list.is_empty() {
                    return write!(f, "none");
                }
                let mut first = true;
                for func in list {
                    if !first {
                        write!(f, " ")?;
                    }
                    first = false;
                    match func {
                        TransformFunction::Translate(x, y) => {
                            write!(f, "translate({x}, {y})")?
                        }
                        TransformFunction::TranslateX(x) => write!(f, "translateX({x})")?,
                        TransformFunction::TranslateY(y) => write!(f, "translateY({y})")?,
                        TransformFunction::Scale(sx, sy) => write!(f, "scale({sx}, {sy})")?,
                        TransformFunction::ScaleX(sx) => write!(f, "scaleX({sx})")?,
                        TransformFunction::ScaleY(sy) => write!(f, "scaleY({sy})")?,
                        TransformFunction::Rotate(a) => write!(f, "rotate({a}deg)")?,
                        TransformFunction::SkewX(a) => write!(f, "skewX({a}deg)")?,
                        TransformFunction::SkewY(a) => write!(f, "skewY({a}deg)")?,
                        TransformFunction::Matrix(m) => write!(
                            f,
                            "matrix({}, {}, {}, {}, {}, {})",
                            m[0], m[1], m[2], m[3], m[4], m[5]
                        )?,
                    }
                }
                Ok(())
            }
            Value::Transitions(list) => {
                let mut first = true;
                for t in list {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(
                        f,
                        "{} {}s {} {}s",
                        t.property, t.duration, t.timing, t.delay
                    )?;
                }
                Ok(())
            }
            Value::Animations(list) => {
                let mut first = true;
                for a in list {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(
                        f,
                        "{} {}s {} {}s",
                        a.name, a.duration, a.timing, a.delay
                    )?;
                }
                Ok(())
            }
            Value::Var { name, fallback } => match fallback {
                Some(fb) => write!(f, "var({name}, {fb})"),
                None => write!(f, "var({name})"),
            },
            Value::Calc(e) => write!(f, "calc({e})"),
            Value::Clamp { min, val, max } => write!(f, "clamp({min}, {val}, {max})"),
            Value::Gradient(g) => {
                let stops: Vec<String> = g
                    .stops
                    .iter()
                    .map(|s| {
                        format!(
                            "rgba({}, {}, {}, {}) {}%",
                            s.color[0],
                            s.color[1],
                            s.color[2],
                            s.color[3] as f32 / 255.0,
                            s.offset * 100.0
                        )
                    })
                    .collect();
                match &g.kind {
                    GradientKind::Linear { angle_deg } => {
                        write!(f, "linear-gradient({angle_deg}deg, {})", stops.join(", "))
                    }
                    GradientKind::Radial {
                        center,
                        shape,
                        extent,
                    } => write!(
                        f,
                        "radial-gradient({:?} {:?} at {}% {}%, {})",
                        shape,
                        extent,
                        center.0 * 100.0,
                        center.1 * 100.0,
                        stops.join(", ")
                    ),
                }
            }
            Value::Raw(r) => write!(f, "{r}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real example.com `<style>` block (see pluto-html tests).
    const EXAMPLE_CSS: &str = r#"
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
        border-radius: 0.5em;
        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);
    }
    a:link, a:visited {
        color: #38488f;
        text-decoration: none;
    }
    @media (max-width: 700px) {
        div {
            margin: 0 auto;
            width: auto;
        }
    }
    "#;

    fn decl<'a>(rule: &'a Rule, prop: &str) -> Option<&'a Value> {
        rule.declarations
            .iter()
            .find(|d| d.property == prop)
            .map(|d| &d.value)
    }

    fn decl_entry<'a>(rule: &'a Rule, prop: &str) -> Option<&'a Declaration> {
        rule.declarations.iter().find(|d| d.property == prop)
    }

    #[test]
    fn parses_example_com_stylesheet() {
        let sheet = Stylesheet::parse(EXAMPLE_CSS);
        // M2 truth: everything parses now — body, div, a:link/a:visited,
        // and the @media div rule. Zero errors.
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 4);

        let body = &sheet.rules[0];
        assert_eq!(body.selectors.len(), 1);
        assert!(body.selectors[0].matches("body", None, &[]));
        assert_eq!(
            decl(body, "background-color"),
            Some(&Value::Color(240, 240, 242))
        );
        assert_eq!(decl(body, "margin-top"), Some(&Value::Length(0.0)));
        assert_eq!(decl(body, "margin-left"), Some(&Value::Length(0.0)));
        // font-family is a multi-word stack: kept verbatim.
        assert!(matches!(decl(body, "font-family"), Some(Value::Raw(_))));

        let div = &sheet.rules[1];
        assert_eq!(decl(div, "width"), Some(&Value::Length(600.0)));
        assert_eq!(
            decl(div, "background-color"),
            Some(&Value::Color(253, 253, 255))
        );
        // `margin: 5em auto` now expands (em is typed in M2).
        assert_eq!(decl(div, "margin-top"), Some(&Value::Em(5.0)));
        assert_eq!(
            decl(div, "margin-right"),
            Some(&Value::Keyword("auto".into()))
        );
        assert_eq!(decl(div, "margin-bottom"), Some(&Value::Em(5.0)));
        assert_eq!(decl(div, "padding-top"), Some(&Value::Em(2.0)));
        assert_eq!(
            decl(div, "border-radius"),
            Some(&Value::BorderRadius(BorderRadius {
                top_left: CssLength::Em(0.5),
                top_right: CssLength::Em(0.5),
                bottom_right: CssLength::Em(0.5),
                bottom_left: CssLength::Em(0.5),
            }))
        );
        assert_eq!(
            decl(div, "box-shadow"),
            Some(&Value::BoxShadow(vec![BoxShadow {
                inset: false,
                offset_x: CssLength::Px(2.0),
                offset_y: CssLength::Px(3.0),
                blur: CssLength::Px(7.0),
                spread: CssLength::Px(2.0),
                color: Some((0, 0, 0, 0.02)),
            }]))
        );

        // a:link, a:visited now parse (link pseudo-classes stored).
        let link = &sheet.rules[2];
        assert_eq!(link.selectors.len(), 2);
        assert_eq!(
            decl(link, "color"),
            Some(&Value::Color(0x38, 0x48, 0x8f))
        );

        // The @media rule is kept with its condition attached.
        let media_div = &sheet.rules[3];
        assert_eq!(
            media_div.media,
            Some(MediaCondition::Feature {
                name: "max-width".to_string(),
                value: Some(Value::Length(700.0)),
            })
        );
        assert_eq!(
            media_div.media.as_ref().unwrap().eval(600.0, 800.0, Theme::Light),
            Some(true)
        );
        assert_eq!(
            media_div.media.as_ref().unwrap().eval(800.0, 800.0, Theme::Light),
            Some(false)
        );
        assert_eq!(decl(media_div, "width"), Some(&Value::Keyword("auto".into())));
        assert_eq!(decl(media_div, "margin-top"), Some(&Value::Length(0.0)));
    }

    #[test]
    fn shorthand_expansion() {
        let sheet = Stylesheet::parse("div { margin: 1px 2px 3px 4px; padding: 10px 20px; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "margin-top"), Some(&Value::Length(1.0)));
        assert_eq!(decl(rule, "margin-right"), Some(&Value::Length(2.0)));
        assert_eq!(decl(rule, "margin-bottom"), Some(&Value::Length(3.0)));
        assert_eq!(decl(rule, "margin-left"), Some(&Value::Length(4.0)));
        assert_eq!(decl(rule, "padding-top"), Some(&Value::Length(10.0)));
        assert_eq!(decl(rule, "padding-bottom"), Some(&Value::Length(10.0)));
        assert_eq!(decl(rule, "padding-right"), Some(&Value::Length(20.0)));
        assert_eq!(decl(rule, "padding-left"), Some(&Value::Length(20.0)));
        assert!(decl(rule, "margin").is_none());
    }

    #[test]
    fn selectors_and_matching() {
        let sheet = Stylesheet::parse("div, .note, #main, *, div.note { display: block; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(rule.selectors.len(), 5);
        assert!(rule.selectors[0].matches("div", None, &[]));
        assert!(!rule.selectors[0].matches("p", None, &[]));
        assert!(rule.selectors[1].matches("p", None, &["note"]));
        assert!(!rule.selectors[1].matches("p", None, &["other"]));
        assert!(rule.selectors[2].matches("div", Some("main"), &[]));
        assert!(!rule.selectors[2].matches("div", Some("other"), &[]));
        assert!(rule.selectors[3].matches("anything", None, &[]));
        assert!(rule.selectors[4].matches("div", None, &["note"]));
        assert!(!rule.selectors[4].matches("p", None, &["note"]));
        assert_eq!(decl(rule, "display"), Some(&Value::Keyword("block".into())));
    }

    #[test]
    fn comments_and_fault_tolerance() {
        let sheet = Stylesheet::parse(
            "/* header */ div { color: red; bogus } p { width: 10px; }",
        );
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.errors.len(), 1);
        assert_eq!(decl(&sheet.rules[0], "color"), Some(&Value::Color(255, 0, 0)));
        assert_eq!(decl(&sheet.rules[1], "width"), Some(&Value::Length(10.0)));
    }

    #[test]
    fn colors() {
        assert_eq!(parse_value("#fff"), Value::Color(255, 255, 255));
        assert_eq!(parse_value("#f0f0f2"), Value::Color(240, 240, 242));
        assert_eq!(parse_value("rgb(1, 2, 3)"), Value::Color(1, 2, 3));
        assert_eq!(parse_value("red"), Value::Color(255, 0, 0));
        assert_eq!(parse_value("50%"), Value::Percentage(50.0));
        assert_eq!(parse_value("12px"), Value::Length(12.0));
        assert_eq!(parse_value("auto"), Value::Keyword("auto".into()));
        // M2: `2em` is typed now (was Raw in M1).
        assert_eq!(parse_value("2em"), Value::Em(2.0));
        assert_eq!(parse_value("1.5rem"), Value::Rem(1.5));
        assert_eq!(parse_value("25vh"), Value::Vh(25.0));
        assert_eq!(parse_value("100vw"), Value::Vw(100.0));
        assert_eq!(
            parse_value("rgba(10, 20, 30, 0.5)"),
            Value::ColorA(10, 20, 30, 0.5)
        );
        // hsla(0, 100%, 50%, 0.25) is red at quarter alpha.
        assert_eq!(
            parse_value("hsla(0, 100%, 50%, 0.25)"),
            Value::ColorA(255, 0, 0, 0.25)
        );
        assert_eq!(parse_value("hsl(120, 100%, 50%)"), Value::Color(0, 255, 0));
        assert_eq!(
            parse_value("#ff000080"),
            Value::ColorA(255, 0, 0, 128.0 / 255.0)
        );
    }

    // ---- R5: complex selectors ----

    #[test]
    fn combinators_parse() {
        let sheet = Stylesheet::parse(
            "div p { color: red; } div > p { color: green; } \
             h1 + p { color: blue; } h1 ~ p { color: yellow; } \
             div.note > p:first-child { color: pink; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 5);
        let combos = [
            Combinator::Descendant,
            Combinator::Child,
            Combinator::NextSibling,
            Combinator::SubsequentSibling,
            Combinator::Child,
        ];
        for (i, want) in combos.iter().enumerate() {
            let sel = &sheet.rules[i].selectors[0];
            assert_eq!(sel.compounds.len(), 2, "rule {i}");
            assert_eq!(sel.combinators, vec![*want], "rule {i}");
        }
        // Compound-only matches() still works; combinator selectors
        // need DOM context and return false here.
        assert!(!sheet.rules[0].selectors[0].matches("p", None, &[]));
        assert!(!sheet.rules[4].selectors[0].matches("p", None, &[]));
        // The :first-child pseudo-class is stored on the last compound.
        let last = &sheet.rules[4].selectors[0];
        assert!(matches!(
            last.compounds[1].parts.last(),
            Some(SimpleSelector::PseudoClass(PseudoClass::FirstChild))
        ));
    }

    #[test]
    fn specificity_cases() {
        let spec = |css: &str| {
            let sheet = Stylesheet::parse(&format!("{css} {{ color: red; }}"));
            assert!(sheet.errors.is_empty(), "{css}: {:?}", sheet.errors);
            sheet.rules[0].selectors[0].specificity()
        };
        assert_eq!(spec("#a"), (1, 0, 0));
        assert_eq!(spec(".a.b"), (0, 2, 0));
        assert_eq!(spec("div"), (0, 0, 1));
        assert_eq!(spec("div.note#main"), (1, 1, 1));
        assert_eq!(spec("*"), (0, 0, 0));
        assert_eq!(spec("ul li a"), (0, 0, 3));
        assert_eq!(spec("a:hover"), (0, 1, 1));
        assert_eq!(spec("[href]"), (0, 1, 0));
        assert_eq!(spec("p::before"), (0, 0, 2));
        // :where() contributes zero; :is()/:not() take the max arg.
        assert_eq!(spec(":where(.a)"), (0, 0, 0));
        assert_eq!(spec(":where(#a)"), (0, 0, 0));
        assert_eq!(spec(":is(.a, #b)"), (1, 0, 0));
        assert_eq!(spec(":is(.a, .b)"), (0, 1, 0));
        assert_eq!(spec(":not(.a)"), (0, 1, 0));
        assert_eq!(spec("div:not(#a)"), (1, 0, 1));
        assert_eq!(spec(":nth-child(2n+1)"), (0, 1, 0));
        // Lexicographic: 256 classes do NOT beat one id.
        let many_classes = ".c".repeat(256);
        assert_eq!(spec(&many_classes), (0, 256, 0));
        assert!(spec(&many_classes) < spec("#a"));
    }

    #[test]
    fn attribute_selectors_parse() {
        let sheet = Stylesheet::parse(
            r#"[href] { color: red; } a[href^="https"] { color: green; }
               [data-x~="y" i] { color: blue; } [lang|="en"] { color: pink; }
               a[href$=".pdf" s] { color: yellow; } [title*="hi"] { color: purple; }"#,
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 6);
        let attr = |rule: usize| match &sheet.rules[rule].selectors[0].compounds[0].parts[..] {
            [SimpleSelector::Attribute { op, .. }] => *op,
            [_, SimpleSelector::Attribute { op, .. }] => *op,
            _ => panic!("rule {rule} has no attribute selector"),
        };
        assert_eq!(attr(0), AttrOp::Exists);
        assert_eq!(attr(1), AttrOp::Prefix);
        assert_eq!(attr(2), AttrOp::Includes);
        assert_eq!(attr(3), AttrOp::Dash);
        assert_eq!(attr(4), AttrOp::Suffix);
        assert_eq!(attr(5), AttrOp::Substring);
        // Case flags: `i` is insensitive, `s` is sensitive.
        let flag = |rule: usize| {
            let parts = &sheet.rules[rule].selectors[0].compounds[0].parts;
            match &parts[..] {
                [SimpleSelector::Attribute {
                    case_sensitive, ..
                }] => *case_sensitive,
                [_, SimpleSelector::Attribute {
                    case_sensitive, ..
                }] => *case_sensitive,
                _ => panic!("rule {rule}"),
            }
        };
        assert_eq!(flag(2), Some(false));
        assert_eq!(flag(4), Some(true));
    }

    #[test]
    fn pseudo_classes_and_elements_parse() {
        let sheet = Stylesheet::parse(
            "p::before { content: \"x\"; } a:hover { color: red; } \
             li:nth-child(2n+1) { color: green; } :root { color: blue; } \
             p:empty { color: pink; } div:has(> p) { color: yellow; } \
             a:any-link { color: purple; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 7);
        assert_eq!(
            sheet.rules[0].selectors[0].pseudo_element,
            Some(PseudoElement::Before)
        );
        assert!(matches!(
            sheet.rules[1].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::Hover)
        ));
        assert!(matches!(
            sheet.rules[2].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::NthChild(AnPlusB { a: 2, b: 1 }))
        ));
        // :has() is parsed, stored, and flagged as unimplemented.
        let has = &sheet.rules[5].selectors[0];
        assert!(has.has_unimplemented);
        assert!(!has.matches("div", None, &[]));
        assert!(matches!(
            sheet.rules[6].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::AnyLink)
        ));
    }

    #[test]
    fn key_selector_picks_most_selective() {
        let sheet = Stylesheet::parse("div#main .note { color: red; } div p { color: blue; }");
        assert!(sheet.errors.is_empty());
        assert!(matches!(
            sheet.rules[0].selectors[0].key_selector(),
            Some(SimpleSelector::Class(c)) if c == "note"
        ));
        assert!(matches!(
            sheet.rules[1].selectors[0].key_selector(),
            Some(SimpleSelector::Type(t)) if t == "p"
        ));
    }

    #[test]
    fn one_bad_selector_kills_the_whole_list() {
        // CHANGED BEHAVIOUR (M2, CSS Syntax L3): previously the good
        // selectors were kept; now one invalid complex invalidates
        // the whole list and the rule is dropped.
        let sheet = Stylesheet::parse(
            "div, .bad[foo, p { color: red; } span { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);
        assert!(sheet.errors[0].contains("whole rule dropped"));
        assert!(sheet.rules[0].selectors[0].matches("span", None, &[]));

        let sheet = Stylesheet::parse("div, span:hover-ish { color: red; }");
        assert_eq!(sheet.rules.len(), 0);
        assert_eq!(sheet.errors.len(), 1);
    }

    // ---- R6: cascade data ----

    #[test]
    fn important_flag_parsed() {
        let sheet =
            Stylesheet::parse("p { color: red !important; margin: 0 !IMPORTANT; width: 1px; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert!(decl_entry(rule, "color").unwrap().important);
        assert!(decl_entry(rule, "margin-top").unwrap().important);
        assert!(!decl_entry(rule, "width").unwrap().important);
    }

    #[test]
    fn cascade_key_ordering() {
        let sheet = Stylesheet::parse(
            "p { color: red; } p { color: blue !important; }",
        );
        let (normal, important) = (&sheet.rules[0], &sheet.rules[1]);
        let normal_decl = &normal.declarations[0];
        let important_decl = &important.declarations[0];
        let kn = normal.sort_key(normal_decl, i32::MAX, false);
        let ki = important.sort_key(important_decl, i32::MAX, false);
        assert!(ki > kn);
        assert_eq!(kn.origin_rank, 2); // author normal
        assert_eq!(ki.origin_rank, 3); // author !important
        // Source order breaks ties between equal keys.
        let later = Rule {
            selectors: normal.selectors.clone(),
            declarations: vec![],
            origin: Origin::Author,
            layer: None,
            source_order: 99,
            media: None,
            supports: None,
        };
        assert_eq!(later.source_order, 99);
    }

    #[test]
    fn layer_statement_and_block_order() {
        let sheet = Stylesheet::parse(
            "@layer b, a; @layer a { p { color: red; } } @layer b { p { color: blue; } } \
             p { color: green; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        // First declaration fixes the order.
        assert_eq!(sheet.layers, vec!["b".to_string(), "a".to_string()]);
        assert_eq!(sheet.rules.len(), 3);
        assert_eq!(sheet.rules[0].layer, Some("a".to_string()));
        assert_eq!(sheet.rules[1].layer, Some("b".to_string()));
        assert_eq!(sheet.rules[2].layer, None);
        assert_eq!(sheet.layer_order(&Some("b".to_string())), 0);
        assert_eq!(sheet.layer_order(&Some("a".to_string())), 1);
        assert_eq!(sheet.layer_order(&None), i32::MAX);
    }

    #[test]
    fn import_recorded_and_late_import_ignored() {
        let sheet = Stylesheet::parse(
            r#"@import url("base.css") layer(foo) supports(display: grid) screen and (max-width: 600px);
               @import "theme.css";
               p { color: red; }"#,
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.imports.len(), 2);
        assert_eq!(sheet.imports[0].url, "base.css");
        assert_eq!(sheet.imports[0].layer, Some("foo".to_string()));
        assert_eq!(
            sheet.imports[0].supports,
            Some("display: grid".to_string())
        );
        assert_eq!(
            sheet.imports[0].media,
            Some("screen and (max-width: 600px)".to_string())
        );
        assert_eq!(sheet.imports[1].url, "theme.css");
        assert_eq!(sheet.imports[1].layer, None);

        let late = Stylesheet::parse("p { color: red; } @import url(\"x.css\");");
        assert!(late.imports.is_empty());
        assert_eq!(late.errors.len(), 1);
        assert!(late.errors[0].contains("@import ignored"));
    }

    #[test]
    fn charset_and_namespace_do_not_choke() {
        let sheet = Stylesheet::parse(
            "@charset \"UTF-8\"; @namespace url(http://www.w3.org/1999/xhtml); p { color: red; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        // @charset is silently ignored; @namespace is recorded, skipped.
        assert_eq!(sheet.errors.len(), 1);
        assert!(sheet.errors[0].contains("@namespace"));
    }

    // ---- R6/R7: @media / @supports / @font-face / @keyframes ----

    #[test]
    fn media_conditions_parse_and_eval() {
        let sheet = Stylesheet::parse(
            "@media screen and (min-width: 100px) { p { color: red; } } \
             @media (max-width: 700px), (min-width: 1200px) { p { color: blue; } }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 2);
        let and = sheet.rules[0].media.as_ref().unwrap();
        assert!(matches!(and, MediaCondition::And(_)));
        assert_eq!(and.eval(50.0, 800.0, Theme::Light), Some(false));
        assert_eq!(and.eval(200.0, 800.0, Theme::Light), Some(true));
        let or = sheet.rules[1].media.as_ref().unwrap();
        assert!(matches!(or, MediaCondition::Or(_)));
        assert_eq!(or.eval(600.0, 800.0, Theme::Light), Some(true));
        assert_eq!(or.eval(800.0, 800.0, Theme::Light), Some(false));
        assert_eq!(or.eval(1300.0, 800.0, Theme::Light), Some(true));
        // `em` in a media feature resolves against 16px.
        let em = MediaCondition::parse("(max-width: 40em)");
        assert_eq!(em.eval(640.0, 800.0, Theme::Light), Some(true));
        assert_eq!(em.eval(641.0, 800.0, Theme::Light), Some(false));
        // Unknown features evaluate to None, never true/false.
        assert_eq!(
            MediaCondition::parse("(pointer: fine)").eval(800.0, 600.0, Theme::Light),
            None
        );
        assert_eq!(
            MediaCondition::parse("print").eval(800.0, 600.0, Theme::Light),
            Some(false)
        );
    }

    #[test]
    fn supports_condition_best_effort_eval() {
        assert_eq!(
            SupportsCondition::parse("(display: grid)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("(display: grid) and (color: red)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("(foo: bar)").eval(),
            Some(false)
        );
        assert_eq!(
            SupportsCondition::parse("(display: grid) or (foo: bar)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("not (foo: bar)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("selector(.foo)").eval(),
            None
        );
        // Rules inside @supports keep the raw condition attached.
        let sheet = Stylesheet::parse("@supports (display: grid) { p { color: red; } }");
        assert!(sheet.errors.is_empty());
        assert_eq!(
            sheet.rules[0].supports,
            Some("(display: grid)".to_string())
        );
    }

    #[test]
    fn font_face_and_keyframes_stored() {
        let sheet = Stylesheet::parse(
            "@font-face { font-family: Foo; src: url(foo.woff2); font-display: swap; \
             font-weight: 400; } \
             @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } \
             p { color: red; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.font_faces.len(), 1);
        assert_eq!(sheet.font_faces[0].family, Some("Foo".to_string()));
        assert_eq!(sheet.font_faces[0].src, Some("url(foo.woff2)".to_string()));
        assert_eq!(sheet.font_faces[0].display, Some("swap".to_string()));
        assert_eq!(sheet.font_faces[0].weight, Some("400".to_string()));
        assert_eq!(sheet.keyframes.len(), 1);
        assert_eq!(sheet.keyframes[0].name, "spin");
        assert_eq!(sheet.keyframes[0].steps.len(), 2);
        assert_eq!(
            sheet.keyframes[0].steps[0].selectors,
            vec![KeyframeSelector::From]
        );
        assert_eq!(
            sheet.keyframes[0].steps[1].selectors,
            vec![KeyframeSelector::To]
        );
        assert_eq!(
            sheet.keyframes[0].steps[0].declarations[0].property,
            "transform"
        );
        assert_eq!(
            sheet.keyframes[0].steps[1].declarations[0].value,
            Value::Transform(vec![TransformFunction::Rotate(360.0)])
        );
        assert_eq!(sheet.rules.len(), 1);
    }

    #[test]
    fn time_and_angle_values_parse() {
        assert_eq!(parse_value("0.3s"), Value::Time(0.3));
        assert_eq!(parse_value("300ms"), Value::Time(0.3));
        assert_eq!(parse_value("0s"), Value::Time(0.0));
        assert_eq!(parse_value("2s"), Value::Time(2.0));
        // A negative duration is not valid: it stays Raw.
        assert_eq!(parse_value("-0.3s"), Value::Raw("-0.3s".to_string()));
        assert_eq!(parse_value("45deg"), Value::Angle(45.0));
        assert_eq!(parse_value("0.5turn"), Value::Angle(180.0));
        assert!((match parse_value("1rad") {
            Value::Angle(a) => a,
            _ => panic!("expected Angle"),
        } - 57.29578)
            .abs()
            < 1e-4);
    }

    #[test]
    fn transform_functions_parse_and_compose() {
        // Each function parses to its typed form.
        assert_eq!(
            parse_value("translate(10px, 20px)"),
            Value::Transform(vec![TransformFunction::Translate(
                CssLength::Px(10.0),
                CssLength::Px(20.0),
            )])
        );
        assert_eq!(
            parse_value("translateX(5%)"),
            Value::Transform(vec![TransformFunction::TranslateX(
                CssLength::Percentage(5.0),
            )])
        );
        assert_eq!(
            parse_value("scale(2, 3)"),
            Value::Transform(vec![TransformFunction::Scale(2.0, 3.0)])
        );
        assert_eq!(
            parse_value("scale(2)"),
            Value::Transform(vec![TransformFunction::Scale(2.0, 2.0)])
        );
        assert_eq!(
            parse_value("rotate(90deg)"),
            Value::Transform(vec![TransformFunction::Rotate(90.0)])
        );
        assert_eq!(
            parse_value("skewX(45deg)"),
            Value::Transform(vec![TransformFunction::SkewX(45.0)])
        );
        assert_eq!(
            parse_value("matrix(1, 2, 3, 4, 5, 6)"),
            Value::Transform(vec![TransformFunction::Matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
        );
        assert_eq!(parse_value("none"), Value::Keyword("none".to_string()));
        // Composition: rotate(90deg) maps (1, 0) to (0, 1).
        let list = match parse_value("rotate(90deg)") {
            Value::Transform(l) => l,
            _ => panic!("expected Transform"),
        };
        let (x, y) = list.to_affine((0.0, 0.0)).apply(1.0, 0.0);
        assert!(x.abs() < 1e-5, "x={x}");
        assert!((y - 1.0).abs() < 1e-5, "y={y}");
        // Order: translate then scale applies the translate first.
        let list = match parse_value("translate(10px, 0px) scale(2)") {
            Value::Transform(l) => l,
            _ => panic!("expected Transform"),
        };
        let (x, y) = list.to_affine((0.0, 0.0)).apply(1.0, 0.0);
        assert!((x - 22.0).abs() < 1e-5, "x={x}");
        assert!(y.abs() < 1e-5, "y={y}");
        // Origin wrapping: rotate(90deg) about (10, 0) leaves the
        // origin point itself fixed.
        let rot = match parse_value("rotate(90deg)") {
            Value::Transform(l) => l,
            _ => panic!("expected Transform"),
        };
        let (x, y) = rot.to_affine((10.0, 0.0)).apply(10.0, 0.0);
        assert!((x - 10.0).abs() < 1e-4, "x={x}");
        assert!(y.abs() < 1e-4, "y={y}");
        // ...and maps a point 10px to the right of the origin to
        // 10px below it.
        let (x, y) = rot.to_affine((10.0, 0.0)).apply(20.0, 0.0);
        assert!((x - 10.0).abs() < 1e-4, "x={x}");
        assert!((y - 10.0).abs() < 1e-4, "y={y}");
        // `invert` round-trips.
        let m = Affine::translate(3.0, 4.0).then(Affine::rotate(30.0));
        let inv = m.invert().expect("invertible");
        let (x, y) = inv.apply(m.apply(7.0, -2.0).0, m.apply(7.0, -2.0).1);
        assert!((x - 7.0).abs() < 1e-4, "x={x}");
        assert!((y + 2.0).abs() < 1e-4, "y={y}");
        assert!(Affine::scale(0.0, 1.0).invert().is_none());
    }

    #[test]
    fn easing_samples() {
        assert_eq!(Easing::Linear.sample(0.5), 0.5);
        assert_eq!(Easing::Ease.sample(0.0), 0.0);
        assert_eq!(Easing::Ease.sample(1.0), 1.0);
        // `cubic-bezier(0, 0, 1, 1)` is the identity.
        assert!((Easing::CubicBezier(0.0, 0.0, 1.0, 1.0).sample(0.37) - 0.37).abs() < 1e-4);
        // `ease` runs fast through the middle (above the diagonal),
        // slow at both ends.
        assert!(Easing::Ease.sample(0.5) > 0.5);
        assert!(Easing::EaseIn.sample(0.5) < 0.5);
        assert!(Easing::EaseOut.sample(0.5) > 0.5);
        // Steps: end jumps at interval ends, start at interval starts.
        assert_eq!(Easing::Steps(4, StepPosition::End).sample(0.0), 0.0);
        assert_eq!(Easing::Steps(4, StepPosition::End).sample(0.3), 0.25);
        assert_eq!(Easing::Steps(4, StepPosition::Start).sample(0.0), 0.25);
        assert_eq!(Easing::Steps(4, StepPosition::Start).sample(0.3), 0.5);
        assert_eq!(Easing::parse("linear"), Some(Easing::Linear));
        assert_eq!(Easing::parse("ease-in"), Some(Easing::EaseIn));
        assert_eq!(
            Easing::parse("cubic-bezier(0.1, 0.2, 0.3, 0.4)"),
            Some(Easing::CubicBezier(0.1, 0.2, 0.3, 0.4))
        );
        assert_eq!(
            Easing::parse("steps(4, start)"),
            Some(Easing::Steps(4, StepPosition::Start))
        );
        assert_eq!(Easing::parse("bounce"), None);
    }

    #[test]
    fn transition_shorthand_parses() {
        let sheet = Stylesheet::parse("div { transition: opacity .3s ease-in .1s; }");
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let v = sheet.rules[0]
            .declarations
            .iter()
            .find(|d| d.property == "transition")
            .map(|d| &d.value)
            .expect("transition decl");
        assert_eq!(
            v,
            &Value::Transitions(vec![Transition {
                property: "opacity".to_string(),
                duration: 0.3,
                timing: Easing::EaseIn,
                delay: 0.1,
            }])
        );
        // Comma-separated lists, and `ms` durations.
        let sheet = Stylesheet::parse(
            "div { transition: opacity 300ms, transform 1s linear; }",
        );
        let v = sheet.rules[0]
            .declarations
            .iter()
            .find(|d| d.property == "transition")
            .map(|d| &d.value)
            .expect("transition decl");
        assert_eq!(
            v,
            &Value::Transitions(vec![
                Transition {
                    property: "opacity".to_string(),
                    duration: 0.3,
                    timing: Easing::Ease,
                    delay: 0.0,
                },
                Transition {
                    property: "transform".to_string(),
                    duration: 1.0,
                    timing: Easing::Linear,
                    delay: 0.0,
                },
            ])
        );
    }

    #[test]
    fn animation_shorthand_parses() {
        let sheet = Stylesheet::parse(
            "div { animation: spin 2s linear infinite alternate both running; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let v = sheet.rules[0]
            .declarations
            .iter()
            .find(|d| d.property == "animation")
            .map(|d| &d.value)
            .expect("animation decl");
        assert_eq!(
            v,
            &Value::Animations(vec![Animation {
                name: "spin".to_string(),
                duration: 2.0,
                timing: Easing::Linear,
                delay: 0.0,
                iteration_count: IterationCount::Infinite,
                direction: AnimationDirection::Alternate,
                fill_mode: FillMode::Both,
                play_state: PlayState::Running,
            }])
        );
        let sheet = Stylesheet::parse("div { animation: fade 0.5s ease-in-out 0.2s 3 reverse; }");
        let v = sheet.rules[0]
            .declarations
            .iter()
            .find(|d| d.property == "animation")
            .map(|d| &d.value)
            .expect("animation decl");
        assert_eq!(
            v,
            &Value::Animations(vec![Animation {
                name: "fade".to_string(),
                duration: 0.5,
                timing: Easing::EaseInOut,
                delay: 0.2,
                iteration_count: IterationCount::Finite(3.0),
                direction: AnimationDirection::Reverse,
                fill_mode: FillMode::None,
                play_state: PlayState::Running,
            }])
        );
    }

    #[test]
    fn keyframes_percentage_and_multi_selectors_parse() {
        let sheet = Stylesheet::parse(
            "@keyframes bounce { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.keyframes.len(), 1);
        assert_eq!(sheet.keyframes[0].steps.len(), 2);
        assert_eq!(
            sheet.keyframes[0].steps[0].selectors,
            vec![
                KeyframeSelector::Percentage(0.0),
                KeyframeSelector::Percentage(100.0),
            ]
        );
        assert_eq!(
            sheet.keyframes[0].steps[1].selectors,
            vec![KeyframeSelector::Percentage(50.0)]
        );
        assert_eq!(
            sheet.keyframes[0].steps[1].declarations[0].value,
            Value::Transform(vec![TransformFunction::TranslateY(CssLength::Px(-10.0))])
        );
    }

    #[test]
    fn var_and_calc_parse_and_eval() {
        assert_eq!(
            parse_value("var(--x)"),
            Value::Var {
                name: "--x".to_string(),
                fallback: None,
            }
        );
        assert_eq!(
            parse_value("var(--x, 10px)"),
            Value::Var {
                name: "--x".to_string(),
                fallback: Some(Box::new(Value::Length(10.0))),
            }
        );
        assert_eq!(
            parse_value("var(--x, var(--y, 5px))"),
            Value::Var {
                name: "--x".to_string(),
                fallback: Some(Box::new(Value::Var {
                    name: "--y".to_string(),
                    fallback: Some(Box::new(Value::Length(5.0))),
                })),
            }
        );
        assert_eq!(
            parse_value("calc(100% - 20px)"),
            Value::Calc(CalcExpr::Sub(
                Box::new(CalcExpr::Pct(100.0)),
                Box::new(CalcExpr::Len(CssLength::Px(20.0))),
            ))
        );
        // `calc(100% - 20px)` evaluates where a basis exists.
        let ctx = CalcCtx {
            font_size: 16.0,
            root_font_size: 16.0,
            viewport: (800.0, 600.0),
            pct_basis: Some(200.0),
        };
        assert_eq!(
            match parse_value("calc(100% - 20px)") {
                Value::Calc(e) => e.eval(&ctx),
                _ => panic!("expected Calc"),
            },
            Some(CalcNumber::Px(180.0))
        );
        // ...and stays an unresolved percentage without one.
        let no_basis = CalcCtx {
            pct_basis: None,
            ..ctx
        };
        assert_eq!(
            match parse_value("calc(100% - 20px)") {
                Value::Calc(e) => e.eval(&no_basis),
                _ => panic!("expected Calc"),
            },
            None
        );
        // Pure-px calc, `*` / `/` by plain numbers, `em`.
        let px = |s: &str| match parse_value(s) {
            Value::Calc(e) => e.eval(&ctx),
            _ => panic!("expected Calc"),
        };
        assert_eq!(px("calc(100px + 20px)"), Some(CalcNumber::Px(120.0)));
        assert_eq!(px("calc(10px * 2)"), Some(CalcNumber::Px(20.0)));
        assert_eq!(px("calc(2 * 10px)"), Some(CalcNumber::Px(20.0)));
        assert_eq!(px("calc(100px / 4)"), Some(CalcNumber::Px(25.0)));
        assert_eq!(px("calc(2em + 8px)"), Some(CalcNumber::Px(40.0)));
        assert_eq!(px("calc((10px + 5px) * 2)"), Some(CalcNumber::Px(30.0)));
        // `*` by a length is unparseable → Raw.
        assert_eq!(
            parse_value("calc(10px * 2px)"),
            Value::Raw("calc(10px * 2px)".to_string())
        );
        // `clamp()` clamps the value between min and max.
        let clamp = match parse_value("clamp(10px, 50px, 20px)") {
            Value::Clamp { min, val, max } => {
                let lo = min.eval(&ctx).unwrap();
                let v = val.eval(&ctx).unwrap();
                let hi = max.eval(&ctx).unwrap();
                match (lo, v, hi) {
                    (CalcNumber::Px(a), CalcNumber::Px(b), CalcNumber::Px(c)) => {
                        b.clamp(a, c)
                    }
                    _ => panic!("expected px"),
                }
            }
            _ => panic!("expected Clamp"),
        };
        assert_eq!(clamp, 20.0);
    }

    #[test]
    fn prefers_color_scheme_media_eval() {
        let dark = MediaCondition::parse("(prefers-color-scheme: dark)");
        assert_eq!(dark.eval(800.0, 600.0, Theme::Dark), Some(true));
        assert_eq!(dark.eval(800.0, 600.0, Theme::Light), Some(false));
        let light = MediaCondition::parse("(prefers-color-scheme: light)");
        assert_eq!(light.eval(800.0, 600.0, Theme::Dark), Some(false));
        assert_eq!(light.eval(800.0, 600.0, Theme::Light), Some(true));
        // Bare `(prefers-color-scheme)` (no value) is unknown.
        assert_eq!(
            MediaCondition::parse("(prefers-color-scheme)").eval(800.0, 600.0, Theme::Dark),
            None
        );
        // Combined with a width feature, both must hold.
        let combo = MediaCondition::parse(
            "screen and (max-width: 700px) and (prefers-color-scheme: dark)",
        );
        assert_eq!(combo.eval(600.0, 800.0, Theme::Dark), Some(true));
        assert_eq!(combo.eval(600.0, 800.0, Theme::Light), Some(false));
        assert_eq!(combo.eval(800.0, 800.0, Theme::Dark), Some(false));
    }

    // ---- R7: UA stylesheet ----

    #[test]
    fn ua_stylesheet_parses_cleanly() {
        let ua = Stylesheet::ua();
        assert!(ua.errors.is_empty(), "errors: {:?}", ua.errors);
        assert!(!ua.rules.is_empty());
        // Every UA rule is origin Ua with source_order 0.. in order.
        for (i, rule) in ua.rules.iter().enumerate() {
            assert_eq!(rule.origin, Origin::Ua);
            assert_eq!(rule.source_order, i as u32);
        }
        let has = |tag: &str| {
            ua.rules.iter().any(|r| {
                r.selectors
                    .iter()
                    .any(|s| s.compounds.len() == 1 && s.compounds[0].parts.iter().any(|p| matches!(p, SimpleSelector::Type(t) if t == tag)))
            })
        };
        assert!(has("body"));
        assert!(has("h1"));
        assert!(has("a"));
        assert!(has("ul"));
        assert!(has("table"));
        let body = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.iter().any(|s| {
                    s.compounds.len() == 1
                        && s.compounds[0].parts == vec![SimpleSelector::Type("body".into())]
                })
            })
            .unwrap();
        assert_eq!(decl(body, "margin-top"), Some(&Value::Length(8.0)));
        let h1 = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.iter().any(|s| {
                    s.compounds.len() == 1
                        && s.compounds[0].parts == vec![SimpleSelector::Type("h1".into())]
                })
            })
            .unwrap();
        assert_eq!(decl(h1, "font-size"), Some(&Value::Em(2.0)));
        assert_eq!(decl(h1, "margin-top"), Some(&Value::Em(0.67)));
        let a = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.len() == 1
                    && r.selectors[0].compounds.len() == 1
                    && r.selectors[0].compounds[0].parts
                        == vec![SimpleSelector::Type("a".into())]
            })
            .unwrap();
        assert_eq!(decl(a, "color"), Some(&Value::Color(0, 0, 238)));
    }

    // ---- R13/R16: values & shorthands ----

    #[test]
    fn font_shorthand_expands() {
        let sheet = Stylesheet::parse("p { font: italic bold 12px/1.5 sans-serif; }");
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let rule = &sheet.rules[0];
        assert_eq!(
            decl(rule, "font-style"),
            Some(&Value::Keyword("italic".into()))
        );
        assert_eq!(
            decl(rule, "font-weight"),
            Some(&Value::Keyword("bold".into()))
        );
        assert_eq!(decl(rule, "font-size"), Some(&Value::Length(12.0)));
        assert_eq!(decl(rule, "line-height"), Some(&Value::Length(1.5)));
        assert_eq!(decl(rule, "font-family"), Some(&Value::Raw("sans-serif".into())));
        assert!(decl(rule, "font").is_none());

        let minimal = Stylesheet::parse("p { font: 16px serif; }");
        assert!(minimal.errors.is_empty());
        assert_eq!(
            decl(&minimal.rules[0], "font-size"),
            Some(&Value::Length(16.0))
        );
        assert_eq!(
            decl(&minimal.rules[0], "font-family"),
            Some(&Value::Raw("serif".into()))
        );
        assert!(decl(&minimal.rules[0], "line-height").is_none());

        // Missing size/family: invalid, dropped with an error.
        let bad = Stylesheet::parse("p { font: bold; color: red; }");
        assert_eq!(bad.rules[0].declarations.len(), 1);
        assert_eq!(bad.errors.len(), 1);
        assert_eq!(
            decl(&bad.rules[0], "color"),
            Some(&Value::Color(255, 0, 0))
        );
    }

    #[test]
    fn shorthand_with_em_expands() {
        let sheet = Stylesheet::parse("div { margin: 5em auto; padding: 2em; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "margin-top"), Some(&Value::Em(5.0)));
        assert_eq!(
            decl(rule, "margin-right"),
            Some(&Value::Keyword("auto".into()))
        );
        assert_eq!(decl(rule, "padding-top"), Some(&Value::Em(2.0)));
        assert_eq!(decl(rule, "padding-left"), Some(&Value::Em(2.0)));
    }

    #[test]
    fn border_radius_resolve_normalises_by_spec_overlap_rule() {
        // CSS Backgrounds and Borders 3 §4.5, worked example (verified
        // live 2026-11-24): a 6em x 2em box with
        // `border-radius: 0.5em 2em 0.5em 2em` at 16px type is reduced
        // by factor 0.8, to 0.4em and 1.6em.
        let sheet = Stylesheet::parse("div { border-radius: 0.5em 2em 0.5em 2em; }");
        let Value::BorderRadius(br) = decl(&sheet.rules[0], "border-radius").unwrap() else {
            panic!("border-radius parses");
        };
        let resolved = br.resolve(96.0, 32.0, 16.0, 16.0, (1024.0, 768.0));
        assert_eq!(resolved.top_left, (6.4, 6.4));
        assert_eq!(resolved.top_right, (25.6, 25.6));
        assert_eq!(resolved.bottom_right, (6.4, 6.4));
        assert_eq!(resolved.bottom_left, (25.6, 25.6));
        // The same radii on a tall enough box are untouched.
        let roomy = br.resolve(96.0, 40.0, 16.0, 16.0, (1024.0, 768.0));
        assert_eq!(roomy.top_left, (8.0, 8.0));
        assert_eq!(roomy.top_right, (32.0, 32.0));
    }

    #[test]
    fn border_radius_percentages_resolve_per_axis() {
        // `border-radius: 50%` on a 200x100 box: rx is 50% of the
        // width, ry 50% of the height — and the pair fits exactly,
        // so no scale-down.
        let sheet = Stylesheet::parse("div { border-radius: 50%; }");
        let Value::BorderRadius(br) = decl(&sheet.rules[0], "border-radius").unwrap() else {
            panic!("border-radius parses");
        };
        let resolved = br.resolve(200.0, 100.0, 16.0, 16.0, (1024.0, 768.0));
        assert_eq!(resolved.top_left, (100.0, 50.0));
        // `border-radius: 100%` overflows both axes by 2x: f = 0.5.
        let sheet = Stylesheet::parse("div { border-radius: 100%; }");
        let Value::BorderRadius(br) = decl(&sheet.rules[0], "border-radius").unwrap() else {
            panic!("border-radius parses");
        };
        let resolved = br.resolve(200.0, 100.0, 16.0, 16.0, (1024.0, 768.0));
        assert_eq!(resolved.top_left, (100.0, 50.0));
        // A zero box never divides by zero: all radii stay zero.
        let resolved = br.resolve(0.0, 0.0, 16.0, 16.0, (1024.0, 768.0));
        assert!(resolved.is_zero());
    }

    #[test]
    fn box_shadow_resolve_keeps_offsets_and_caps_blur_at_validation() {
        let sheet = Stylesheet::parse(
            "div { box-shadow: -2px 3px 7px 2px rgba(0,0,0,0.5), inset 0 0 100px black; }",
        );
        let Value::BoxShadow(shadows) = decl(&sheet.rules[0], "box-shadow").unwrap() else {
            panic!("box-shadow parses");
        };
        let first = shadows[0].resolve(16.0, 16.0, (1024.0, 768.0), [0, 0, 0]);
        assert_eq!(
            (first.offset_x, first.offset_y, first.blur, first.spread),
            (-2.0, 3.0, 7.0, 2.0)
        );
        assert_eq!(first.color, [0.0, 0.0, 0.0, 0.5]);
        assert!(!first.inset);
        // The blur is resolved, not clamped: the cap is enforced by
        // display-list validation, so a 100px blur still resolves
        // here and is rejected there.
        let second = shadows[1].resolve(16.0, 16.0, (1024.0, 768.0), [255, 0, 0]);
        assert!(second.inset);
        assert_eq!(second.blur, 100.0);
        assert!(second.blur > MAX_BOX_SHADOW_BLUR);
        assert_eq!(second.color, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn border_radius_and_box_shadow_parse() {
        let sheet = Stylesheet::parse(
            "div { border-radius: 4px 8px; box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02); }",
        );
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(
            decl(rule, "border-radius"),
            Some(&Value::BorderRadius(BorderRadius {
                top_left: CssLength::Px(4.0),
                top_right: CssLength::Px(8.0),
                bottom_right: CssLength::Px(4.0),
                bottom_left: CssLength::Px(8.0),
            }))
        );
        assert_eq!(
            decl(rule, "box-shadow"),
            Some(&Value::BoxShadow(vec![BoxShadow {
                inset: false,
                offset_x: CssLength::Px(2.0),
                offset_y: CssLength::Px(3.0),
                blur: CssLength::Px(7.0),
                spread: CssLength::Px(2.0),
                color: Some((0, 0, 0, 0.02)),
            }]))
        );
        let inset = Stylesheet::parse("div { box-shadow: inset 0 1px 2px #000; }");
        assert!(inset.errors.is_empty());
        match decl(&inset.rules[0], "box-shadow") {
            Some(Value::BoxShadow(shadows)) => {
                assert_eq!(shadows.len(), 1);
                assert!(shadows[0].inset);
                assert_eq!(shadows[0].color, Some((0, 0, 0, 1.0)));
            }
            other => panic!("unexpected box-shadow: {other:?}"),
        }
        // Unparseable shapes fall back to Raw, never an error.
        let bad = Stylesheet::parse("div { border-radius: bogus; box-shadow: bogus; }");
        assert!(bad.errors.is_empty());
        assert!(matches!(decl(&bad.rules[0], "border-radius"), Some(Value::Raw(_))));
        assert!(matches!(decl(&bad.rules[0], "box-shadow"), Some(Value::Raw(_))));
    }

    #[test]
    fn grid_and_flex_properties_accepted() {
        let sheet = Stylesheet::parse(
            "div { display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: auto; \
             gap: 10px; grid-column: 1 / 3; flex-direction: column; flex-wrap: wrap; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "display"), Some(&Value::Keyword("grid".into())));
        assert!(decl(rule, "grid-template-columns").is_some());
        assert!(decl(rule, "grid-template-rows").is_some());
        assert_eq!(decl(rule, "gap"), Some(&Value::Length(10.0)));
        assert!(decl(rule, "grid-column").is_some());
        assert_eq!(
            decl(rule, "flex-direction"),
            Some(&Value::Keyword("column".into()))
        );
        assert_eq!(
            decl(rule, "flex-wrap"),
            Some(&Value::Keyword("wrap".into()))
        );
    }

    #[test]
    fn grid_auto_flow_property_known_and_value_shapes() {
        // Stage 10 Rec 5: the property is recognised (so `@supports
        // (grid-auto-flow: ...)` evaluates instead of failing
        // unknown), and its value shapes are the ones layout's
        // parser consumes: single keywords as `Keyword`, the
        // two-word forms as `Raw`.
        assert!(KNOWN_PROPERTIES.contains(&"grid-auto-flow"));
        assert_eq!(parse_value("column"), Value::Keyword("column".into()));
        assert_eq!(parse_value("dense"), Value::Keyword("dense".into()));
        assert_eq!(
            parse_value("row dense"),
            Value::Raw("row dense".into())
        );
        assert_eq!(
            parse_value("column dense"),
            Value::Raw("column dense".into())
        );
        assert_eq!(
            SupportsCondition::parse("(grid-auto-flow: column)").eval(),
            Some(true)
        );
    }

    // ---- error-recovery granularity ----

    #[test]
    fn error_recovery_granularity() {
        // Bad declaration: only that declaration is dropped.
        let sheet = Stylesheet::parse("p { color: red; bogus:; width: 10px; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].declarations.len(), 2);
        assert_eq!(sheet.errors.len(), 1);

        // Missing semicolon merges two declarations: both are lost,
        // the rest of the rule survives.
        let sheet = Stylesheet::parse("p { color: red width: 10px; height: 5px; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            None,
            "merged declaration must be dropped"
        );
        assert_eq!(
            decl(&sheet.rules[0], "height"),
            Some(&Value::Length(5.0))
        );
        assert_eq!(sheet.errors.len(), 1);

        // Bad selector: the whole rule is dropped, other rules survive.
        let sheet = Stylesheet::parse("div > > p { color: red; } p { color: blue; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);

        // Unknown at-rule (statement and block): skipped, rest survives.
        let sheet = Stylesheet::parse(
            "@foo bar; p { color: red; } @bar { div { width: 1px; } } p { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.errors.len(), 2);

        // Unclosed block: closed at EOF, the rule is still kept.
        let sheet = Stylesheet::parse("p { color: red; ");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            Some(&Value::Color(255, 0, 0))
        );
        assert_eq!(sheet.errors.len(), 1);

        // @container is parsed-and-skipped with correct brace matching.
        let sheet = Stylesheet::parse(
            "@container (min-width: 400px) { p { color: red; } } p { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            Some(&Value::Color(0, 0, 255))
        );
    }

    #[test]
    fn pseudo_class_index_covers_hover_active_focus_and_reach() {
        // A plain rule mentions nothing; a bare `:hover` reaches
        // nothing beyond its element; a descendant combinator after
        // the pseudo reaches descendants; `+`/`~` reach siblings.
        let sheet = Stylesheet::parse(
            "p { color: red; } \
             a:hover { color: blue; } \
             div:hover span { color: green; } \
             div:hover + p { color: yellow; } \
             button:active { color: pink; } \
             input:focus { color: orange; } \
             form:focus-within { color: purple; } \
             div:has(a:hover) { color: grey; }",
        );
        let index = &sheet.pseudo_class_index;
        assert_eq!(index.hover, vec![1, 2, 3]);
        assert_eq!(index.active, vec![4]);
        assert_eq!(index.focus, vec![5, 6]);
        assert!(index.hover_reach.descendants); // `div:hover span`
        assert!(index.hover_reach.siblings); // `div:hover + p`
        assert!(!index.hover_reach.ancestors);
        assert!(!index.active_reach.descendants && !index.active_reach.siblings);
        assert!(index.focus_reach.ancestors); // `:focus-within`
        // `:has()` never matches, so its `:hover` is not indexed.
        assert!(!index.hover.contains(&7));
    }

    #[test]
    fn pseudo_class_index_reaches_through_is_not_and_ignores_plain_combinators() {
        // A combinator BEFORE the pseudo does not reach: hovering
        // the `b` in `div b:hover` restyles only `b` itself.
        // `:is()`/`:not()` arguments count; a combinator after the
        // argument's pseudo still reaches.
        let sheet = Stylesheet::parse(
            "div b:hover { color: red; } \
             a:is(span:hover) { color: blue; } \
             a:not(:hover) { color: green; } \
             div:is(span:hover b) { color: yellow; }",
        );
        let index = &sheet.pseudo_class_index;
        assert_eq!(index.hover, vec![0, 1, 2, 3]);
        // Only the last rule reaches: its `:hover` sits before a
        // descendant combinator inside the `:is()` argument.
        assert!(index.hover_reach.descendants);
        assert!(!index.hover_reach.siblings);
    }

    #[test]
    fn pseudo_class_index_rebuilds_after_hand_pushed_rules() {
        let mut sheet = Stylesheet::parse("p { color: red; }");
        assert!(sheet.pseudo_class_index.hover.is_empty());
        let mut pushed = Stylesheet::parse("a:hover { color: blue; }");
        sheet.rules.append(&mut pushed.rules);
        // Stale until rebuilt: the index still says no hover rule.
        assert!(sheet.pseudo_class_index.hover.is_empty());
        sheet.rebuild_pseudo_class_index();
        assert_eq!(sheet.pseudo_class_index.hover, vec![1]);
    }

    // ---- Stage 10 Rec 4: `:has()` parse-time invalidation bookkeeping ----

    #[test]
    fn has_index_records_argument_selectors() {
        let sheet = Stylesheet::parse(
            "div:has(> img) { color: red; } \
             li:has(.active) { color: blue; } \
             body:has(div) { color: green; } \
             p { color: black; } \
             a:has(img, .thumb) { color: yellow; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let index = &sheet.pseudo_class_index.has;
        // Only the four `:has()` rules are indexed; the plain `p`
        // rule (3) is not.
        assert_eq!(
            index.rules.iter().map(|e| e.rule).collect::<Vec<_>>(),
            vec![0, 1, 2, 4]
        );
        assert!(!index.is_empty());
        // Anchors are ancestors of the changed element.
        assert!(sheet.pseudo_class_index.has_reach.ancestors);
        assert!(!sheet.pseudo_class_index.has_reach.descendants);
        assert!(!sheet.pseudo_class_index.has_reach.siblings);

        // `div:has(> img)`: one argument, anchored child, the
        // parser's implied `*` subject is NOT counted as universal.
        let r0 = &index.rules[0];
        assert_eq!(r0.args.len(), 1);
        assert_eq!(r0.args[0].selector, 0);
        assert_eq!(r0.args[0].argument, 0);
        assert!(r0.args[0].anchored);
        assert!(!r0.args[0].broad());
        assert_eq!(r0.args[0].combinators, vec![Combinator::Child]);
        assert!(r0.args[0].inventory.types.contains("img"));
        assert!(!r0.args[0].inventory.has_universal);
        assert!(!r0.is_broad());

        // `li:has(.active)`: implicit descendant — broad.
        let r1 = &index.rules[1];
        assert_eq!(r1.args.len(), 1);
        assert!(!r1.args[0].anchored);
        assert!(r1.args[0].broad());
        assert!(r1.args[0].combinators.is_empty());
        assert!(r1.args[0].inventory.classes.contains("active"));
        assert!(r1.is_broad());

        // `body:has(div)`: broad un-anchored argument.
        let r2 = &index.rules[2];
        assert!(r2.is_broad());
        assert!(r2.args[0].inventory.types.contains("div"));

        // `a:has(img, .thumb)`: two arguments, located by index.
        let r4 = &index.rules[3];
        assert_eq!(r4.args.len(), 2);
        assert_eq!(r4.args[0].argument, 0);
        assert!(r4.args[0].inventory.types.contains("img"));
        assert_eq!(r4.args[1].argument, 1);
        assert!(r4.args[1].inventory.classes.contains("thumb"));

        // The union covers every argument's selectors.
        assert!(index.all.types.contains("img"));
        assert!(index.all.types.contains("div"));
        assert!(index.all.classes.contains("active"));
        assert!(index.all.classes.contains("thumb"));
    }

    #[test]
    fn has_query_matches_only_mentioned_features() {
        let sheet = Stylesheet::parse(
            "div:has(> img) { color: red; } \
             li:has(.active) { color: blue; } \
             body:has(div) { color: green; } \
             a:has(img, .thumb) { color: yellow; } \
             section:has([href]) { color: pink; } \
             nav:has(a:hover) { color: orange; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let index = &sheet.pseudo_class_index.has;
        let img = HasArgFeature::type_selector("img");
        let div = HasArgFeature::type_selector("div");
        let span = HasArgFeature::type_selector("span");
        let active = HasArgFeature::class("active");
        let thumb = HasArgFeature::class("thumb");
        let missing = HasArgFeature::class("missing");
        let href = HasArgFeature::attribute("href");
        let hover = HasArgFeature::pseudo_class("hover");
        let focus = HasArgFeature::pseudo_class("focus");

        assert!(index.could_affect(&img));
        assert!(index.could_affect(&div));
        assert!(index.could_affect(&active));
        assert!(index.could_affect(&thumb));
        assert!(index.could_affect(&href));
        // A `:hover` inside a `:has()` argument is inventoried
        // (it can change the anchor later), even though the
        // `:hover` index itself skips `:has()` rules.
        assert!(index.could_affect(&hover));
        assert!(!sheet.pseudo_class_index.hover.contains(&5));

        assert!(!index.could_affect(&span));
        assert!(!index.could_affect(&missing));
        assert!(!index.could_affect(&focus));
        assert!(!index.could_affect(&HasArgFeature::id("main")));

        assert!(index.could_affect_any(&[span.clone(), active.clone()]));
        assert!(!index.could_affect_any(&[span.clone(), missing.clone()]));
        assert!(!index.could_affect_any(&[]));

        // Candidate rules are the rules whose arguments mention
        // the feature, in rule order.
        assert_eq!(index.candidate_rules(&img), vec![0, 3]);
        assert_eq!(index.candidate_rules(&active), vec![1]);
        assert_eq!(index.candidate_rules(&div), vec![2]);
        assert_eq!(index.candidate_rules(&href), vec![4]);
        assert_eq!(index.candidate_rules(&hover), vec![5]);
        assert!(index.candidate_rules(&span).is_empty());
    }

    #[test]
    fn has_index_folds_nested_has_and_logical_args() {
        let sheet = Stylesheet::parse(
            "div:has(:is(.a, #b)) { color: red; } \
             div:has(span:has(img)) { color: blue; } \
             div:has(> *) { color: green; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let index = &sheet.pseudo_class_index.has;
        // `:is()` arguments count: the class and id are features.
        let r0 = &index.rules[0];
        assert!(r0.args[0].inventory.classes.contains("a"));
        assert!(r0.args[0].inventory.ids.contains("b"));
        assert!(index.could_affect(&HasArgFeature::id("b")));
        // Nested `:has()` folds into the enclosing inventory and
        // is flagged, so the gate sees the inner selectors too.
        let r1 = &index.rules[1];
        assert!(r1.args[0].inventory.has_nested_has);
        assert!(r1.args[0].inventory.types.contains("span"));
        assert!(r1.args[0].inventory.types.contains("img"));
        assert!(index.could_affect(&HasArgFeature::type_selector("img")));
        // A real `*` (here the child of the implied subject) is
        // universal: any feature may affect that rule.
        let r2 = &index.rules[2];
        assert!(r2.args[0].inventory.has_universal);
        assert!(index.could_affect(&HasArgFeature::type_selector("anything")));
        assert_eq!(
            index.candidate_rules(&HasArgFeature::type_selector("anything")),
            vec![2]
        );
    }

    #[test]
    fn has_index_rebuilds_after_hand_pushed_rules() {
        let mut sheet = Stylesheet::parse("p { color: red; }");
        assert!(sheet.pseudo_class_index.has.is_empty());
        let mut pushed = Stylesheet::parse("div:has(> img) { color: blue; }");
        sheet.rules.append(&mut pushed.rules);
        // Stale until rebuilt: the index still says no `:has()`.
        assert!(sheet.pseudo_class_index.has.is_empty());
        sheet.rebuild_pseudo_class_index();
        assert_eq!(sheet.pseudo_class_index.has.rules.len(), 1);
        assert!(sheet.pseudo_class_index.has.could_affect(
            &HasArgFeature::type_selector("img")
        ));
        assert!(!sheet.pseudo_class_index.has.could_affect(
            &HasArgFeature::type_selector("span")
        ));
    }

    #[test]
    fn has_selectors_still_never_match() {
        // Regression (Stage 10 Rec 4): bookkeeping must not change
        // behaviour — `:has()` selectors still never match, and
        // the stubbed anchor re-check returns nothing even when
        // the gate passes.
        let sheet = Stylesheet::parse(
            "div:has(> img) { color: red; } \
             li:has(.active) { color: blue; } \
             body:has(div) { color: green; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        for rule in &sheet.rules {
            for sel in &rule.selectors {
                assert!(sel.has_unimplemented);
                assert!(!sel.matches("div", None, &[]));
                assert!(!sel.matches("li", None, &["active"]));
            }
        }
        let index = &sheet.pseudo_class_index.has;
        // The gate passes for a mentioned feature ...
        let img = HasArgFeature::type_selector("img");
        assert!(index.could_affect(&img));
        assert_eq!(index.candidate_rules(&img), vec![0]);
        // ... but the re-check is still the not-yet-implemented
        // stub: no anchor is ever re-checked, no rule restyles.
        assert!(index.recheck_anchors(&[img]).is_empty());
        assert!(index
            .recheck_anchors(&[HasArgFeature::class("active")])
            .is_empty());
        // And a sheet with no `:has()` at all gates everything off.
        let plain = Stylesheet::parse("p { color: red; }");
        assert!(plain.pseudo_class_index.has.is_empty());
        assert!(!plain
            .pseudo_class_index
            .has
            .could_affect(&HasArgFeature::type_selector("img")));
    }

    // Stage 11 Rec 3: background-image parsing + URL collection.

    #[test]
    fn background_image_url_value_parses() {
        assert_eq!(
            parse_value("url(bg.png)"),
            Value::Url("bg.png".to_string())
        );
        assert_eq!(
            parse_value("url(\"quoted.png\")"),
            Value::Url("quoted.png".to_string())
        );
        assert_eq!(
            parse_value("URL('UP.PNG')"),
            Value::Url("UP.PNG".to_string())
        );
        assert_eq!(parse_value("none"), Value::Keyword("none".to_string()));
        assert_eq!(
            parse_value("url(  spaced.png  )"),
            Value::Url("spaced.png".to_string())
        );
    }

    #[test]
    fn extract_css_urls_finds_shorthand_urls() {
        assert_eq!(
            extract_css_urls("url(a.png) no-repeat center"),
            vec!["a.png".to_string()]
        );
        assert_eq!(
            extract_css_urls("url(a.png), url(b.png)"),
            vec!["a.png".to_string(), "b.png".to_string()]
        );
        assert!(extract_css_urls("red").is_empty());
        assert!(extract_css_urls("none").is_empty());
    }

    #[test]
    fn background_image_urls_collects_from_rules() {
        let sheet = Stylesheet::parse(
            "div { background-image: url(hero.png); } \
             p { background: url(tile.png) repeat; } \
             span { background-image: none; } \
             a { background-image: url(hero.png); }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        // Deduped, in rule order; `none` contributes nothing.
        assert_eq!(
            sheet.background_image_urls(),
            vec!["hero.png".to_string(), "tile.png".to_string()]
        );
        let plain = Stylesheet::parse("p { color: red; }");
        assert!(plain.background_image_urls().is_empty());
    }

    // Stage 15 (M7 Phase 2, Rec 7): CSS gradient values.
    fn gradient_value(s: &str) -> Gradient {
        match parse_value(s) {
            Value::Gradient(g) => g,
            other => panic!("{s} should parse as a gradient, got {other:?}"),
        }
    }

    #[test]
    fn linear_gradient_angle_and_stops_parse() {
        let g = gradient_value("linear-gradient(45deg, red, blue)");
        assert_eq!(g.kind, GradientKind::Linear { angle_deg: 45.0 });
        assert_eq!(g.stops.len(), 2);
        assert_eq!(g.stops[0], GradientStop { offset: 0.0, color: [255, 0, 0, 255] });
        assert_eq!(g.stops[1], GradientStop { offset: 1.0, color: [0, 0, 255, 255] });
        // Default direction is `to bottom` (180deg).
        let d = gradient_value("linear-gradient(red, blue)");
        assert_eq!(d.kind, GradientKind::Linear { angle_deg: 180.0 });
        // `to ...` directions.
        assert_eq!(
            gradient_value("linear-gradient(to top right, red, blue)").kind,
            GradientKind::Linear { angle_deg: 45.0 }
        );
        assert_eq!(
            gradient_value("linear-gradient(to left, red, blue)").kind,
            GradientKind::Linear { angle_deg: 270.0 }
        );
        // Explicit offsets + a middle stop without one (distributed).
        let m = gradient_value("linear-gradient(90deg, red 0%, green, blue 100%)");
        assert_eq!(m.stops[0].offset, 0.0);
        assert_eq!(m.stops[1].offset, 0.5);
        assert_eq!(m.stops[2].offset, 1.0);
        assert_eq!(m.stops[1].color, [0, 128, 0, 255]);
    }

    #[test]
    fn radial_gradient_shape_position_and_extent_parse() {
        let g = gradient_value("radial-gradient(circle at 20% 30%, red, blue)");
        assert_eq!(
            g.kind,
            GradientKind::Radial {
                center: (0.2, 0.3),
                shape: RadialShape::Circle,
                extent: RadialExtent::FarthestCorner,
            }
        );
        let e = gradient_value("radial-gradient(closest-side, red, blue)");
        assert_eq!(
            e.kind,
            GradientKind::Radial {
                center: (0.5, 0.5),
                shape: RadialShape::Ellipse,
                extent: RadialExtent::ClosestSide,
            }
        );
        // One stop, px offsets, and unknown directions are not
        // gradients — they fall back to Raw, never an error.
        assert!(matches!(parse_value("linear-gradient(red)"), Value::Raw(_)));
        assert!(matches!(
            parse_value("linear-gradient(red 10px, blue)"),
            Value::Raw(_)
        ));
    }

    #[test]
    fn gradient_color_at_and_linear_t_are_exact() {
        let g = gradient_value("linear-gradient(90deg, red, blue)");
        // 90deg = to the right: t runs left (0) to right (1).
        // (Float noise: 90deg in f32 radians is a hair off pi/2,
        // so compare approximately.)
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(approx(g.linear_t((0.0, 50.0), (0.0, 0.0, 100.0, 100.0)), 0.0));
        assert!(approx(g.linear_t((100.0, 50.0), (0.0, 0.0, 100.0, 100.0)), 1.0));
        assert!(approx(g.linear_t((50.0, 50.0), (0.0, 0.0, 100.0, 100.0)), 0.5));
        assert_eq!(g.color_at(0.0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(g.color_at(1.0), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(g.color_at(0.5), [0.5, 0.0, 0.5, 1.0]);
        // Radial: center is t = 0, the farthest corner is t = 1
        // for the default (ellipse, farthest-corner) gradient.
        let r = gradient_value("radial-gradient(red, blue)");
        assert_eq!(r.radial_t((50.0, 50.0), (0.0, 0.0, 100.0, 100.0)), 0.0);
        let corner = r.radial_t((100.0, 100.0), (0.0, 0.0, 100.0, 100.0));
        assert!((corner - 1.0).abs() < 1e-5, "corner t = {corner}");
    }
}