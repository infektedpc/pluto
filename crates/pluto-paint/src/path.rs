//! SVG path data model + tessellation (Stage 15, M7 Phase 2,
//! Rec 3/4).
//!
//! Pluto owns its path type: [`SvgPath`] is parsed from `d`,
//! `points`, and shape geometry (`cx`/`cy`/`r`, `x`/`y`/`width`/
//! `height`, `x1`/`y1`/`x2`/`y2`) and tessellated to triangle
//! meshes by lyon ([`tessellate_fill`], [`tessellate_stroke`]).
//! Lyon is the roadmap's first candidate and was verified
//! available in the offline registry cache (lyon 1.0.19,
//! lyon_tessellation 1.0.23) before the dependency was added —
//! no hand-rolled earcut fallback was needed. resvg/usvg are
//! deliberately not used: they would pull a whole SVG stack and
//! violate the from-scratch rule in spirit.
//!
//! The tessellator emits [`VectorVertex`] triangles (position +
//! premultiplied-nothing RGBA, per-vertex colour so a two-stop
//! linear gradient is exact by barycentric interpolation). The
//! display list carries them as
//! [`DisplayItem::VectorPath`](crate::DisplayItem::VectorPath)
//! and the renderer draws them through the vector pipeline
//! (group 0 only, like the rect pipeline). The same triangles
//! feed the CPU rasteriser ([`crate::raster`]) — one tessellation
//! path for both display routes.

use pluto_css::Gradient;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A parsed SVG path: an owned list of segments in absolute
/// user-space coordinates (the parser resolves relative commands
/// and arc flags at parse time, so consumers never re-implement
/// them).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvgPath {
    pub segments: Vec<PathSeg>,
}

/// One path segment, absolute coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSeg {
    MoveTo([f32; 2]),
    LineTo([f32; 2]),
    QuadTo([f32; 2], [f32; 2]),
    CubicTo([f32; 2], [f32; 2], [f32; 2]),
    Close,
}

impl SvgPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn move_to(&mut self, p: [f32; 2]) {
        self.segments.push(PathSeg::MoveTo(p));
    }

    pub fn line_to(&mut self, p: [f32; 2]) {
        self.segments.push(PathSeg::LineTo(p));
    }

    pub fn quad_to(&mut self, c: [f32; 2], p: [f32; 2]) {
        self.segments.push(PathSeg::QuadTo(c, p));
    }

    pub fn cubic_to(&mut self, c1: [f32; 2], c2: [f32; 2], p: [f32; 2]) {
        self.segments.push(PathSeg::CubicTo(c1, c2, p));
    }

    pub fn close(&mut self) {
        self.segments.push(PathSeg::Close);
    }

    /// The axis-aligned bounding box `[x, y, w, h]`, or `None`
    /// for an empty path. Control points are included (a bezier's
    /// true bounds are inside its control hull, so this is a safe
    /// over-approximation — right for culling and gradient boxes).
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;
        let mut pt = |p: [f32; 2]| {
            any = true;
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        };
        for seg in &self.segments {
            match *seg {
                PathSeg::MoveTo(p) | PathSeg::LineTo(p) => pt(p),
                PathSeg::QuadTo(c, p) => {
                    pt(c);
                    pt(p);
                }
                PathSeg::CubicTo(c1, c2, p) => {
                    pt(c1);
                    pt(c2);
                    pt(p);
                }
                PathSeg::Close => {}
            }
        }
        if !any {
            return None;
        }
        Some([min_x, min_y, max_x - min_x, max_y - min_y])
    }
}

/// What went wrong parsing path data or tessellating it.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PathError {
    #[error("path data is empty")]
    Empty,
    #[error("path data does not start with a moveto command")]
    NoMoveto,
    #[error("unknown path command '{0}'")]
    UnknownCommand(char),
    #[error("expected a number at offset {0}")]
    ExpectedNumber(usize),
    #[error("expected a flag (0 or 1) at offset {0}")]
    ExpectedFlag(usize),
    #[error("path has too many segments (over {0})")]
    TooManySegments(usize),
    #[error("tessellation failed: {0}")]
    Tessellation(String),
    #[error("tessellation produced too many vertices (over {0})")]
    TooManyVertices(usize),
}

/// A hostile `d` must not make the tessellator allocate
/// unboundedly: the parser refuses past this many segments, and
/// the tessellator refuses past [`MAX_TESSELLATED_VERTICES`]
/// vertices (a shape past the cap falls back to the rasteriser,
/// which tessellates the same path with the same cap).
pub const MAX_PATH_SEGMENTS: usize = 100_000;
/// Max tessellated vertices per shape (~21k triangles, ~1.5 MB of
/// `VectorVertex` — far above any icon, far below the 64 MiB IPC
/// frame the display list must fit in).
pub const MAX_TESSELLATED_VERTICES: usize = 65_536;

/// The SVG fill rule: `nonzero` (the default) or `evenodd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

/// Parse a `fill-rule` / `clip-rule` value; anything else (or
/// nothing) is the SVG default, `nonzero`.
pub fn parse_fill_rule(s: Option<&str>) -> FillRule {
    match s {
        Some(v) if v.trim().eq_ignore_ascii_case("evenodd") => FillRule::EvenOdd,
        _ => FillRule::NonZero,
    }
}

/// Parse an SVG path-data string (`d` attribute) into an
/// [`SvgPath`]. The full command set is supported (M/m, L/l, H/h,
/// V/v, C/c, S/s, Q/q, T/t, A/a, Z/z), including implicit
/// repetition, implicit lineto after moveto, and arc-to-cubic
/// conversion. Relative commands resolve against the current
/// point at parse time.
///
/// Known limitation (documented, not silent): an arc flag
/// written with no separator after the rotation
/// (`A 30 50 001162.55 ...`) parses the rotation greedily — the
/// spec's single-character flag rule needs lookahead the parser
/// does not do. Space- or comma-separated arc arguments (what
/// real icon sets emit) parse exactly.
pub fn parse_path_data(d: &str) -> Result<SvgPath, PathError> {
    let bytes = d.as_bytes();
    let mut p = Parser {
        bytes,
        pos: 0,
        path: SvgPath::new(),
        current: [0.0, 0.0],
        subpath_start: [0.0, 0.0],
        prev_cubic_ctrl: None,
        prev_quad_ctrl: None,
        prev_cmd: '\0',
    };
    p.skip_sep();
    if p.pos >= bytes.len() {
        return Err(PathError::Empty);
    }
    // The first command must be a moveto (spec: a path starting
    // with anything else is an error).
    let first = bytes[p.pos] as char;
    if first != 'M' && first != 'm' {
        return Err(PathError::NoMoveto);
    }
    while p.pos < bytes.len() {
        p.skip_sep();
        if p.pos >= bytes.len() {
            break;
        }
        let c = bytes[p.pos] as char;
        if !c.is_ascii_alphabetic() {
            return Err(PathError::UnknownCommand(c));
        }
        p.pos += 1;
        p.skip_sep();
        p.command(c)?;
        if p.path.segments.len() > MAX_PATH_SEGMENTS {
            return Err(PathError::TooManySegments(MAX_PATH_SEGMENTS));
        }
    }
    Ok(p.path)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    path: SvgPath,
    current: [f32; 2],
    subpath_start: [f32; 2],
    prev_cubic_ctrl: Option<[f32; 2]>,
    prev_quad_ctrl: Option<[f32; 2]>,
    prev_cmd: char,
}

impl<'a> Parser<'a> {
    /// Skip a separator: whitespace, then at most one comma, then
    /// whitespace (the SVG `comma-wsp` production).
    fn skip_sep(&mut self) {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] as char).is_whitespace()
        {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b',' {
            self.pos += 1;
        }
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] as char).is_whitespace()
        {
            self.pos += 1;
        }
    }

    /// Whether the next token could start a number (a digit, a
    /// sign, or a `.` followed by a digit) — used for implicit
    /// command repetition.
    fn next_is_number(&self) -> bool {
        if self.pos >= self.bytes.len() {
            return false;
        }
        let c = self.bytes[self.pos] as char;
        c.is_ascii_digit()
            || c == '-'
            || c == '+'
            || (c == '.'
                && self.pos + 1 < self.bytes.len()
                && (self.bytes[self.pos + 1] as char).is_ascii_digit())
    }

    /// Parse one SVG number at the cursor (no separator skipping:
    /// the caller skips first). The grammar is `sign? (digits+
    /// "."? digits* | "." digits+) (e|E sign? digits+)?` — and a
    /// number ends where the grammar ends, so `10-20` is two
    /// numbers and `10.5.5` is `10.5` then `.5`.
    fn number(&mut self) -> Result<f32, PathError> {
        let start = self.pos;
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+')
        {
            self.pos += 1;
        }
        let mut digits = 0;
        while self.pos < self.bytes.len() && (self.bytes[self.pos] as char).is_ascii_digit() {
            self.pos += 1;
            digits += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.bytes.len()
                && (self.bytes[self.pos] as char).is_ascii_digit()
            {
                self.pos += 1;
                digits += 1;
            }
        }
        if digits == 0 {
            return Err(PathError::ExpectedNumber(start));
        }
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
        {
            let save = self.pos;
            self.pos += 1;
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+')
            {
                self.pos += 1;
            }
            let mut exp_digits = 0;
            while self.pos < self.bytes.len()
                && (self.bytes[self.pos] as char).is_ascii_digit()
            {
                self.pos += 1;
                exp_digits += 1;
            }
            if exp_digits == 0 {
                // `1e` is not a number — rewind to before the `e`
                // so the error points at the real problem.
                self.pos = save;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| PathError::ExpectedNumber(start))?;
        let v: f32 = text
            .parse()
            .map_err(|_| PathError::ExpectedNumber(start))?;
        if !v.is_finite() {
            return Err(PathError::ExpectedNumber(start));
        }
        Ok(v)
    }

    fn flag(&mut self) -> Result<bool, PathError> {
        self.skip_sep();
        let at = self.pos;
        if self.pos >= self.bytes.len() {
            return Err(PathError::ExpectedFlag(at));
        }
        match self.bytes[self.pos] {
            b'0' => {
                self.pos += 1;
                Ok(false)
            }
            b'1' => {
                self.pos += 1;
                Ok(true)
            }
            _ => Err(PathError::ExpectedFlag(at)),
        }
    }

    /// One coordinate pair (two numbers, comma-wsp separated).
    fn pair(&mut self) -> Result<[f32; 2], PathError> {
        let x = self.number()?;
        self.skip_sep();
        let y = self.number()?;
        Ok([x, y])
    }

    fn command(&mut self, cmd: char) -> Result<(), PathError> {
        let relative = cmd.is_ascii_lowercase();
        let rel = |p: [f32; 2], cur: [f32; 2]| -> [f32; 2] {
            if relative {
                [cur[0] + p[0], cur[1] + p[1]]
            } else {
                p
            }
        };
        match cmd {
            'M' | 'm' => {
                // The first pair is a moveto; further pairs are
                // implicit lineto (same relativity).
                let p = rel(self.pair()?, self.current);
                self.path.move_to(p);
                self.current = p;
                self.subpath_start = p;
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.prev_cmd = cmd;
                loop {
                    self.skip_sep();
                    if !self.next_is_number() {
                        break;
                    }
                    let p = rel(self.pair()?, self.current);
                    self.path.line_to(p);
                    self.current = p;
                    self.prev_cubic_ctrl = None;
                    self.prev_quad_ctrl = None;
                    self.prev_cmd = if relative { 'l' } else { 'L' };
                }
            }
            'L' | 'l' => loop {
                self.skip_sep();
                let p = rel(self.pair()?, self.current);
                self.path.line_to(p);
                self.current = p;
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'H' | 'h' => loop {
                self.skip_sep();
                let x = self.number()?;
                let p = if relative {
                    [self.current[0] + x, self.current[1]]
                } else {
                    [x, self.current[1]]
                };
                self.path.line_to(p);
                self.current = p;
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'V' | 'v' => loop {
                self.skip_sep();
                let y = self.number()?;
                let p = if relative {
                    [self.current[0], self.current[1] + y]
                } else {
                    [self.current[0], y]
                };
                self.path.line_to(p);
                self.current = p;
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'C' | 'c' => loop {
                self.skip_sep();
                let c1 = rel(self.pair()?, self.current);
                self.skip_sep();
                let c2 = rel(self.pair()?, self.current);
                self.skip_sep();
                let p = rel(self.pair()?, self.current);
                self.path.cubic_to(c1, c2, p);
                self.prev_cubic_ctrl = Some(c2);
                self.prev_quad_ctrl = None;
                self.current = p;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'S' | 's' => loop {
                self.skip_sep();
                // The first control point reflects the previous
                // cubic's second control point (when the previous
                // command was C/c/S/s); otherwise it is the
                // current point.
                let c1 = match (self.prev_cmd, self.prev_cubic_ctrl) {
                    ('C' | 'c' | 'S' | 's', Some(prev)) => {
                        [2.0 * self.current[0] - prev[0], 2.0 * self.current[1] - prev[1]]
                    }
                    _ => self.current,
                };
                let c2 = rel(self.pair()?, self.current);
                self.skip_sep();
                let p = rel(self.pair()?, self.current);
                self.path.cubic_to(c1, c2, p);
                self.prev_cubic_ctrl = Some(c2);
                self.prev_quad_ctrl = None;
                self.current = p;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'Q' | 'q' => loop {
                self.skip_sep();
                let c = rel(self.pair()?, self.current);
                self.skip_sep();
                let p = rel(self.pair()?, self.current);
                self.path.quad_to(c, p);
                self.prev_quad_ctrl = Some(c);
                self.prev_cubic_ctrl = None;
                self.current = p;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'T' | 't' => loop {
                self.skip_sep();
                let c = match (self.prev_cmd, self.prev_quad_ctrl) {
                    ('Q' | 'q' | 'T' | 't', Some(prev)) => {
                        [2.0 * self.current[0] - prev[0], 2.0 * self.current[1] - prev[1]]
                    }
                    _ => self.current,
                };
                let p = rel(self.pair()?, self.current);
                self.path.quad_to(c, p);
                self.prev_quad_ctrl = Some(c);
                self.prev_cubic_ctrl = None;
                self.current = p;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'A' | 'a' => loop {
                self.skip_sep();
                let rx = self.number()?.abs();
                self.skip_sep();
                let ry = self.number()?.abs();
                self.skip_sep();
                let rotation = self.number()?;
                let large = self.flag()?;
                let sweep = self.flag()?;
                self.skip_sep();
                let end = rel(self.pair()?, self.current);
                if rx == 0.0 || ry == 0.0 || end == self.current {
                    // A degenerate arc is a straight line (spec
                    // F.6.5.1); a zero-length arc adds nothing.
                    if end != self.current {
                        self.path.line_to(end);
                    }
                } else {
                    for (c1, c2, p) in
                        arc_to_cubic(self.current, rx, ry, rotation, large, sweep, end)
                    {
                        self.path.cubic_to(c1, c2, p);
                    }
                }
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.current = end;
                self.prev_cmd = cmd;
                self.skip_sep();
                if !self.next_is_number() {
                    break;
                }
            },
            'Z' | 'z' => {
                self.path.close();
                self.current = self.subpath_start;
                self.prev_cubic_ctrl = None;
                self.prev_quad_ctrl = None;
                self.prev_cmd = cmd;
            }
            other => return Err(PathError::UnknownCommand(other)),
        }
        Ok(())
    }
}

/// Convert an SVG elliptical arc (endpoint parameterisation) to
/// cubic beziers (SVG spec F.6.5, endpoint → center → beziers).
/// Each returned cubic is `(control1, control2, end)`; the arc is
/// split into ≤ 90° spans, so the cubic approximation error stays
/// small. Pure.
pub fn arc_to_cubic(
    start: [f32; 2],
    rx: f32,
    ry: f32,
    x_axis_rotation: f32,
    large_arc: bool,
    sweep: bool,
    end: [f32; 2],
) -> Vec<([f32; 2], [f32; 2], [f32; 2])> {
    let phi = x_axis_rotation.to_radians();
    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
    // Step 1: (x1', y1') — the start point in the rotated frame.
    let dx2 = (start[0] - end[0]) / 2.0;
    let dy2 = (start[1] - end[1]) / 2.0;
    let x1p = cos_phi * dx2 + sin_phi * dy2;
    let y1p = -sin_phi * dx2 + cos_phi * dy2;
    // Step 2: radii too small to reach the end are scaled up.
    let mut rx = rx.abs();
    let mut ry = ry.abs();
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    // Step 3: the center (cx', cy') in the rotated frame.
    let num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    let den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let mut factor = (num / den).max(0.0).sqrt();
    if large_arc == sweep {
        factor = -factor;
    }
    let cxp = factor * (rx * y1p / ry);
    let cyp = factor * (-ry * x1p / rx);
    // Step 4: the center in user space.
    let cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0;
    let cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0;
    // Step 5: the start and sweep angles.
    let angle = |ux: f32, uy: f32, vx: f32, vy: f32| -> f32 {
        let dot = ux * vx + uy * vy;
        let len = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        let mut a = (dot / len).clamp(-1.0, 1.0).acos();
        if ux * vy - uy * vx < 0.0 {
            a = -a;
        }
        a
    };
    let theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry);
    let mut dtheta = angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    );
    if !sweep && dtheta > 0.0 {
        dtheta -= 2.0 * std::f32::consts::PI;
    } else if sweep && dtheta < 0.0 {
        dtheta += 2.0 * std::f32::consts::PI;
    }
    // Split into ≤ 90° spans; each span is one cubic with the
    // standard kappa = 4/3 * tan(span/4).
    let spans = (dtheta.abs() / (std::f32::consts::FRAC_PI_2)).ceil().max(1.0) as usize;
    let span = dtheta / spans as f32;
    let mut out = Vec::with_capacity(spans);
    let mut theta = theta1;
    for _ in 0..spans {
        let next = theta + span;
        let kappa = 4.0 / 3.0 * (span / 4.0).tan();
        let (sin_t, cos_t) = (theta.sin(), theta.cos());
        let (sin_n, cos_n) = (next.sin(), next.cos());
        // Endpoints and control points in the rotated frame,
        // then rotated back to user space.
        let pt = |t: f32| -> [f32; 2] {
            let (s, c) = (t.sin(), t.cos());
            let x = cx + rx * c * cos_phi - ry * s * sin_phi;
            let y = cy + rx * c * sin_phi + ry * s * cos_phi;
            [x, y]
        };
        let p0 = pt(theta);
        let p3 = pt(next);
        let c1 = [
            p0[0] + kappa * (-rx * sin_t * cos_phi - ry * cos_t * sin_phi),
            p0[1] + kappa * (-rx * sin_t * sin_phi + ry * cos_t * cos_phi),
        ];
        let c2 = [
            p3[0] + kappa * (rx * sin_n * cos_phi + ry * cos_n * sin_phi),
            p3[1] + kappa * (rx * sin_n * sin_phi - ry * cos_n * cos_phi),
        ];
        out.push((c1, c2, p3));
        theta = next;
    }
    out
}

/// The kappa constant for approximating a quarter circle/ellipse
/// with one cubic bezier (4/3 * tan(pi/8)).
const KAPPA: f32 = 0.552_284_8;

/// Build a path from a `<rect>` (x, y, width, height, optional
/// corner radii rx/ry). A zero-size rect is an empty path. Corner
/// radii are clamped to half the side, per SVG.
pub fn path_from_rect(x: f32, y: f32, w: f32, h: f32, rx: f32, ry: f32) -> SvgPath {
    let mut path = SvgPath::new();
    if w <= 0.0 || h <= 0.0 {
        return path;
    }
    let rx = rx.max(0.0).min(w / 2.0);
    let ry = ry.max(0.0).min(h / 2.0);
    if rx == 0.0 || ry == 0.0 {
        path.move_to([x, y]);
        path.line_to([x + w, y]);
        path.line_to([x + w, y + h]);
        path.line_to([x, y + h]);
        path.close();
        return path;
    }
    // Rounded rect: four corner arcs (one cubic each) joined by
    // straight edges.
    path.move_to([x + rx, y]);
    path.line_to([x + w - rx, y]);
    path.cubic_to(
        [x + w - rx + rx * KAPPA, y],
        [x + w, y + ry - ry * KAPPA],
        [x + w, y + ry],
    );
    path.line_to([x + w, y + h - ry]);
    path.cubic_to(
        [x + w, y + h - ry + ry * KAPPA],
        [x + w - rx + rx * KAPPA, y + h],
        [x + w - rx, y + h],
    );
    path.line_to([x + rx, y + h]);
    path.cubic_to(
        [x + rx - rx * KAPPA, y + h],
        [x, y + h - ry + ry * KAPPA],
        [x, y + h - ry],
    );
    path.line_to([x, y + ry]);
    path.cubic_to(
        [x, y + ry - ry * KAPPA],
        [x + rx - rx * KAPPA, y],
        [x + rx, y],
    );
    path.close();
    path
}

/// Build a path from a `<circle>` (cx, cy, r): four cubics.
pub fn path_from_circle(cx: f32, cy: f32, r: f32) -> SvgPath {
    path_from_ellipse(cx, cy, r, r)
}

/// Build a path from an `<ellipse>` (cx, cy, rx, ry): four
/// cubics, each a quarter ellipse.
pub fn path_from_ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> SvgPath {
    let mut path = SvgPath::new();
    if rx <= 0.0 || ry <= 0.0 {
        return path;
    }
    let k = KAPPA;
    path.move_to([cx + rx, cy]);
    path.cubic_to(
        [cx + rx, cy + ry * k],
        [cx + rx * k, cy + ry],
        [cx, cy + ry],
    );
    path.cubic_to(
        [cx - rx * k, cy + ry],
        [cx - rx, cy + ry * k],
        [cx - rx, cy],
    );
    path.cubic_to(
        [cx - rx, cy - ry * k],
        [cx - rx * k, cy - ry],
        [cx, cy - ry],
    );
    path.cubic_to(
        [cx + rx * k, cy - ry],
        [cx + rx, cy - ry * k],
        [cx + rx, cy],
    );
    path.close();
    path
}

/// Build an open path from a `<line>` (x1, y1, x2, y2): a
/// moveto + lineto, no fill. Strokes only.
pub fn path_from_line(x1: f32, y1: f32, x2: f32, y2: f32) -> SvgPath {
    let mut path = SvgPath::new();
    path.move_to([x1, y1]);
    path.line_to([x2, y2]);
    path
}

/// Build a path from a `points` attribute (`<polyline>` /
/// `<polygon>`): whitespace/comma-separated coordinate pairs.
/// `close` (polygon) appends a Close. An odd trailing number is
/// an error.
pub fn path_from_points(points: &str, close: bool) -> Result<SvgPath, PathError> {
    let bytes = points.as_bytes();
    let mut p = Parser {
        bytes,
        pos: 0,
        path: SvgPath::new(),
        current: [0.0, 0.0],
        subpath_start: [0.0, 0.0],
        prev_cubic_ctrl: None,
        prev_quad_ctrl: None,
        prev_cmd: '\0',
    };
    let mut first = true;
    loop {
        p.skip_sep();
        if p.pos >= bytes.len() {
            break;
        }
        let pair = p.pair()?;
        if first {
            p.path.move_to(pair);
            first = false;
        } else {
            p.path.line_to(pair);
        }
        if p.path.segments.len() > MAX_PATH_SEGMENTS {
            return Err(PathError::TooManySegments(MAX_PATH_SEGMENTS));
        }
    }
    if first {
        return Err(PathError::Empty);
    }
    if close {
        p.path.close();
    }
    Ok(p.path)
}

/// One tessellated triangle vertex: a position in layout space
/// and its RGBA colour (components 0..=1, straight alpha). The
/// colour is per-vertex, so a two-stop linear gradient is exact
/// under barycentric interpolation and a solid fill is just a
/// constant colour. Serialisable — it rides the display list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// The SVG stroke cap / join spellings.
pub fn parse_line_cap(s: Option<&str>) -> lyon::tessellation::LineCap {
    use lyon::tessellation::LineCap;
    match s {
        Some(v) if v.trim().eq_ignore_ascii_case("round") => LineCap::Round,
        Some(v) if v.trim().eq_ignore_ascii_case("square") => LineCap::Square,
        _ => LineCap::Butt,
    }
}

/// The SVG stroke join spellings.
pub fn parse_line_join(s: Option<&str>) -> lyon::tessellation::LineJoin {
    use lyon::tessellation::LineJoin;
    match s {
        Some(v) if v.trim().eq_ignore_ascii_case("round") => LineJoin::Round,
        Some(v) if v.trim().eq_ignore_ascii_case("bevel") => LineJoin::Bevel,
        _ => LineJoin::Miter,
    }
}

/// Convert an [`SvgPath`] to a lyon path (absolute coordinates).
fn to_lyon_path(path: &SvgPath) -> lyon::path::Path {
    use lyon::path::math::point;
    let mut b = lyon::path::Path::builder();
    let mut in_subpath = false;
    for seg in &path.segments {
        match *seg {
            PathSeg::MoveTo(p) => {
                if in_subpath {
                    b.end(false);
                }
                b.begin(point(p[0], p[1]));
                in_subpath = true;
            }
            PathSeg::LineTo(p) => {
                b.line_to(point(p[0], p[1]));
            }
            PathSeg::QuadTo(c, p) => {
                b.quadratic_bezier_to(point(c[0], c[1]), point(p[0], p[1]));
            }
            PathSeg::CubicTo(c1, c2, p) => {
                b.cubic_bezier_to(
                    point(c1[0], c1[1]),
                    point(c2[0], c2[1]),
                    point(p[0], p[1]),
                );
            }
            PathSeg::Close => {
                b.close();
                in_subpath = false;
            }
        }
    }
    if in_subpath {
        b.end(false);
    }
    b.build()
}

/// Tessellate a path's fill into triangles. `color_at` colours
/// each vertex from its position — pass a constant for a solid
/// fill, or a gradient evaluation for a linear gradient (exact
/// for two stops; the caller routes multi-stop / radial
/// gradients to the rasteriser instead). `rule` is the SVG
/// fill-rule (default nonzero). Returns the triangles as a flat
/// vertex list (three vertices per triangle, in order). The
/// tessellator flattens curves at tolerance 0.1 (lyon's default)
/// — sub-pixel for icon-scale paths.
pub fn tessellate_fill(
    path: &SvgPath,
    rule: FillRule,
    color_at: impl Fn([f32; 2]) -> [f32; 4],
) -> Result<Vec<VectorVertex>, PathError> {
    use lyon::tessellation::{
        BuffersBuilder, FillOptions, FillRule as LyonRule, FillTessellator, FillVertex,
        VertexBuffers,
    };
    if path.is_empty() {
        return Ok(Vec::new());
    }
    let lyon_path = to_lyon_path(path);
    let mut buffers: VertexBuffers<VectorVertex, u32> = VertexBuffers::new();
    {
        let mut builder = BuffersBuilder::new(&mut buffers, |v: FillVertex| VectorVertex {
            position: v.position().to_array(),
            color: color_at(v.position().to_array()),
        });
        let mut tessellator = FillTessellator::new();
        // FillOptions is #[non_exhaustive] (no literal
        // construction outside lyon), but its fields are public:
        // mutate a default.
        let mut options = FillOptions::default();
        options.fill_rule = match rule {
            FillRule::NonZero => LyonRule::NonZero,
            FillRule::EvenOdd => LyonRule::EvenOdd,
        };
        tessellator
            .tessellate_path(&lyon_path, &options, &mut builder)
            .map_err(|e| PathError::Tessellation(e.to_string()))?;
    }
    if buffers.vertices.len() > MAX_TESSELLATED_VERTICES {
        return Err(PathError::TooManyVertices(MAX_TESSELLATED_VERTICES));
    }
    // Expand the index buffer to a flat triangle list (the
    // display list carries no index buffer).
    let mut out = Vec::with_capacity(buffers.indices.len());
    for &i in &buffers.indices {
        let v = buffers
            .vertices
            .get(i as usize)
            .ok_or_else(|| PathError::Tessellation("tessellator index out of range".into()))?;
        out.push(*v);
    }
    Ok(out)
}

/// Tessellate a path's stroke into triangles. `width` is in the
/// path's user units; cap/join/miter follow the SVG attributes
/// (defaults butt/miter/4). A `stroke-dasharray` is NOT honoured
/// here — dashed strokes render solid in this phase (the caller
/// notes it); dashing is a later slice. Returns a flat triangle
/// list, like [`tessellate_fill`].
pub fn tessellate_stroke(
    path: &SvgPath,
    width: f32,
    cap: lyon::tessellation::LineCap,
    join: lyon::tessellation::LineJoin,
    miter_limit: f32,
    color_at: impl Fn([f32; 2]) -> [f32; 4],
) -> Result<Vec<VectorVertex>, PathError> {
    use lyon::tessellation::{
        BuffersBuilder, StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
    };
    if path.is_empty() || width <= 0.0 {
        return Ok(Vec::new());
    }
    let lyon_path = to_lyon_path(path);
    let mut buffers: VertexBuffers<VectorVertex, u32> = VertexBuffers::new();
    {
        let mut builder = BuffersBuilder::new(&mut buffers, |v: StrokeVertex| VectorVertex {
            position: v.position().to_array(),
            color: color_at(v.position().to_array()),
        });
        let mut tessellator = StrokeTessellator::new();
        let options = StrokeOptions::default()
            .with_line_width(width)
            .with_line_cap(cap)
            .with_line_join(join)
            .with_miter_limit(miter_limit.max(1.0));
        tessellator
            .tessellate_path(&lyon_path, &options, &mut builder)
            .map_err(|e| PathError::Tessellation(e.to_string()))?;
    }
    if buffers.vertices.len() > MAX_TESSELLATED_VERTICES {
        return Err(PathError::TooManyVertices(MAX_TESSELLATED_VERTICES));
    }
    let mut out = Vec::with_capacity(buffers.indices.len());
    for &i in &buffers.indices {
        let v = buffers
            .vertices
            .get(i as usize)
            .ok_or_else(|| PathError::Tessellation("tessellator index out of range".into()))?;
        out.push(*v);
    }
    Ok(out)
}

/// Clip coloured triangles to an axis-aligned rect `[x, y, w, h]`
/// (Sutherland–Hodgman, one half-plane at a time), interpolating
/// colours linearly along clipped edges. Triangles fully outside
/// vanish; a triangle crossing the edge gains vertices. Pure —
/// used to clip SVG content to the `<svg>` viewport (spec: the
/// viewport clips its content) without touching the display
/// list's clip stack.
pub fn clip_colored_triangles(
    vertices: &[VectorVertex],
    rect: [f32; 4],
) -> Vec<VectorVertex> {
    let [rx, ry, rw, rh] = rect;
    let (x0, y0, x1, y1) = (rx, ry, rx + rw, ry + rh);
    let mut out = Vec::with_capacity(vertices.len());
    for tri in vertices.as_chunks::<3>().0 {
        let mut poly: Vec<VectorVertex> = tri.to_vec();
        for (axis, bound, keep_above) in [
            (0usize, x0, true),
            (0, x1, false),
            (1, y0, true),
            (1, y1, false),
        ] {
            if poly.is_empty() {
                break;
            }
            let mut next: Vec<VectorVertex> = Vec::with_capacity(poly.len() + 1);
            let inside = |v: &VectorVertex| {
                if keep_above {
                    v.position[axis] >= bound
                } else {
                    v.position[axis] <= bound
                }
            };
            let intersect = |a: &VectorVertex, b: &VectorVertex| -> VectorVertex {
                let da = a.position[axis] - bound;
                let db = b.position[axis] - bound;
                let t = if (da - db).abs() < f32::EPSILON {
                    0.0
                } else {
                    da / (da - db)
                };
                let t = t.clamp(0.0, 1.0);
                let lerp = |i: usize| a.position[i] + (b.position[i] - a.position[i]) * t;
                let lerp_c = |i: usize| a.color[i] + (b.color[i] - a.color[i]) * t;
                VectorVertex {
                    position: [lerp(0), lerp(1)],
                    color: [lerp_c(0), lerp_c(1), lerp_c(2), lerp_c(3)],
                }
            };
            for i in 0..poly.len() {
                let cur = poly[i];
                let prev = poly[(i + poly.len() - 1) % poly.len()];
                let cur_in = inside(&cur);
                let prev_in = inside(&prev);
                if cur_in {
                    if !prev_in {
                        next.push(intersect(&prev, &cur));
                    }
                    next.push(cur);
                } else if prev_in {
                    next.push(intersect(&prev, &cur));
                }
            }
            poly = next;
        }
        // Fan-triangulate the clipped polygon.
        if poly.len() >= 3 {
            for i in 1..poly.len() - 1 {
                out.push(poly[0]);
                out.push(poly[i]);
                out.push(poly[i + 1]);
            }
        }
    }
    out
}

/// Build the tessellated triangles for a linear-gradient fill of
/// a rect `[x, y, w, h]`: one strip per stop segment, so the
/// colour is exact for any stop count (a single quad would
/// interpolate straight across the middle stops). The strip
/// boundaries are the gradient's iso-lines clipped to the rect.
/// Only linear gradients are supported — radial returns empty
/// (the caller rasterises those).
/// A linear gradient over a rect `[x, y, w, h]`, as flat
/// coloured triangles: one strip per stop segment, so the colour
/// is exact for any stop count (a single quad would interpolate
/// straight across the middle stops). Each strip is the rect
/// polygon clipped to the stop pair's `t` range and fan
/// triangulated, with each vertex coloured by its own `t` — the
/// colour function is linear in position, so barycentric
/// interpolation inside a strip is exact. Only linear gradients
/// are supported — radial returns empty (the caller rasterises
/// those).
///
/// The clip (not the old iso-line approach) is deliberate: the
/// iso-line at `t = 0` or `t = 1` passes exactly through a rect
/// corner, and whether the segment survives depends on f32
/// rounding of `cos(90°)` — a rect at (8, 8) produced no
/// triangles at all. Clipping the polygon is exact for every
/// angle, including the degenerate end stops.
pub fn linear_gradient_quad(x: f32, y: f32, w: f32, h: f32, gradient: &Gradient) -> Vec<VectorVertex> {
    let pluto_css::GradientKind::Linear { angle_deg } = gradient.kind else {
        return Vec::new();
    };
    if w <= 0.0 || h <= 0.0 || gradient.stops.len() < 2 {
        return Vec::new();
    }
    // The gradient parameter, as in `Gradient::linear_t`: the
    // axis direction, the rect centre, and the half-extent of
    // the axis over the rect.
    let rad = angle_deg.to_radians();
    let d = [rad.sin(), -rad.cos()];
    let c = [x + w / 2.0, y + h / 2.0];
    let half = (w * d[0].abs() + h * d[1].abs()) / 2.0;
    if half <= 0.0 {
        return Vec::new();
    }
    let t_of = |p: [f32; 2]| {
        ((p[0] - c[0]) * d[0] + (p[1] - c[1]) * d[1] + half) / (2.0 * half)
    };
    let mut out = Vec::new();
    for pair in gradient.stops.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        // The rect polygon clipped to the strip `a.offset..=b.offset`.
        let mut poly = vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
        poly = clip_to_value(poly, &t_of, a.offset, true);
        poly = clip_to_value(poly, &t_of, b.offset, false);
        if poly.len() < 3 {
            continue;
        }
        // Fan triangulation (the clipped polygon is convex: a
        // rect cut by two parallel lines). Each vertex carries
        // the colour at its own `t`, so the strip is exact.
        for i in 1..poly.len() - 1 {
            for p in [poly[0], poly[i], poly[i + 1]] {
                out.push(VectorVertex {
                    position: p,
                    color: gradient.color_at(t_of(p)),
                });
            }
        }
    }
    out
}

/// Clip a polygon to the half-plane `value(p) >= bound`
/// (`keep_above`) or `value(p) <= bound`, interpolating the
/// crossing points (Sutherland–Hodgman, one edge at a time).
/// The polygon must be convex; the result is convex.
fn clip_to_value(
    poly: Vec<[f32; 2]>,
    value: &dyn Fn([f32; 2]) -> f32,
    bound: f32,
    keep_above: bool,
) -> Vec<[f32; 2]> {
    if poly.is_empty() {
        return poly;
    }
    let inside = |p: [f32; 2]| {
        let v = value(p);
        if keep_above {
            v >= bound
        } else {
            v <= bound
        }
    };
    let mut out = Vec::new();
    for i in 0..poly.len() {
        let cur = poly[i];
        let prev = poly[(i + poly.len() - 1) % poly.len()];
        let cur_in = inside(cur);
        let prev_in = inside(prev);
        if cur_in != prev_in {
            // Crossing on the edge prev -> cur.
            let vp = value(prev);
            let vc = value(cur);
            let f = if (vc - vp).abs() > f32::EPSILON {
                (bound - vp) / (vc - vp)
            } else {
                0.0
            };
            out.push([
                prev[0] + (cur[0] - prev[0]) * f,
                prev[1] + (cur[1] - prev[1]) * f,
            ]);
        }
        if cur_in {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(color: [f32; 4]) -> impl Fn([f32; 2]) -> [f32; 4] {
        move |_| color
    }

    #[test]
    fn path_parser_covers_all_commands() {
        // Absolute + relative moveto/lineto, implicit repetition,
        // H/V, C/S, Q/T, A, Z — one path exercising everything.
        let d = "M 10 10 L 20 10 20 20 H 30 V 40 C 40 50 50 60 60 70 \
                 S 80 90 90 100 Q 110 120 130 140 T 150 160 \
                 A 10 10 0 0 1 170 170 Z \
                 m 5 5 l 5 0 h 5 v 5 c 1 1 2 2 3 3 s 1 1 2 2 q 1 1 2 2 t 3 3 a 5 5 0 0 0 5 5 z";
        let path = parse_path_data(d).expect("parses");
        assert!(!path.is_empty());
        // First subpath: M + 2 L (one explicit, one implicit) + H
        // + V + C + S + Q + T + A (a semicircle, so 2 cubics) + Z.
        let first_close = path
            .segments
            .iter()
            .position(|s| matches!(s, PathSeg::Close))
            .expect("first close");
        assert_eq!(first_close, 11);
        // Second subpath starts with a relative moveto at
        // (15, 15): `Z` reset the current point to the first
        // subpath's start (10, 10), and the relative moveto adds
        // (5, 5) to that.
        assert_eq!(path.segments[12], PathSeg::MoveTo([15.0, 15.0]));
        // Relative lineto: (20, 15).
        assert_eq!(path.segments[13], PathSeg::LineTo([20.0, 15.0]));
    }

    #[test]
    fn path_parser_rejects_bad_input() {
        assert_eq!(parse_path_data(""), Err(PathError::Empty));
        assert_eq!(parse_path_data("L 10 10"), Err(PathError::NoMoveto));
        assert!(matches!(
            parse_path_data("M 10 10 X 20 20"),
            Err(PathError::UnknownCommand('X'))
        ));
        assert!(matches!(
            parse_path_data("M 10"),
            Err(PathError::ExpectedNumber(_))
        ));
        assert!(matches!(
            parse_path_data("M 10 10 A 5 5 0 2 1 20 20"),
            Err(PathError::ExpectedFlag(_))
        ));
    }

    #[test]
    fn path_parser_handles_implicit_separators() {
        // `10-20` is two numbers; `10.5.5` is `10.5` then `.5`.
        let p = parse_path_data("M10-20L30.5.5").expect("parses");
        assert_eq!(p.segments[0], PathSeg::MoveTo([10.0, -20.0]));
        assert_eq!(p.segments[1], PathSeg::LineTo([30.5, 0.5]));
    }

    #[test]
    fn arc_to_cubic_matches_endpoints_and_midpoint() {
        // A quarter-circle arc from (10, 0) to (0, 10) around the
        // origin: the cubic chain must start/end exactly there,
        // and its midpoint must sit on the circle.
        let cubics = arc_to_cubic([10.0, 0.0], 10.0, 10.0, 0.0, false, true, [0.0, 10.0]);
        assert_eq!(cubics.len(), 1);
        let (c1, c2, end) = cubics[0];
        assert!((end[0] - 0.0).abs() < 1e-5 && (end[1] - 10.0).abs() < 1e-5);
        // Cubic midpoint (t = 0.5) on the circle, radius 10.
        let mid = |t: f32| -> [f32; 2] {
            let mt = 1.0 - t;
            let b = |i: usize| {
                mt * mt * mt * [10.0, 0.0][i]
                    + 3.0 * mt * mt * t * c1[i]
                    + 3.0 * mt * t * t * c2[i]
                    + t * t * t * end[i]
            };
            [b(0), b(1)]
        };
        let m = mid(0.5);
        let r = (m[0] * m[0] + m[1] * m[1]).sqrt();
        assert!((r - 10.0).abs() < 0.05, "mid radius {r}");
    }

    #[test]
    fn arc_large_and_sweep_choose_opposite_sides() {
        // Same endpoints, large-arc on/off: the large arc must
        // swing far around the chord (its cubic endpoints reach
        // y = -18.66 for r = 10), while the small arc stays close
        // (endpoints within y >= -1.34).
        let small = arc_to_cubic([0.0, 0.0], 10.0, 10.0, 0.0, false, true, [10.0, 0.0]);
        let large = arc_to_cubic([0.0, 0.0], 10.0, 10.0, 0.0, true, true, [10.0, 0.0]);
        assert!(!small.is_empty() && !large.is_empty());
        let min_end_y = |cubics: &[([f32; 2], [f32; 2], [f32; 2])]| -> f32 {
            cubics
                .iter()
                .map(|(_, _, end)| end[1])
                .fold(f32::INFINITY, f32::min)
        };
        assert!(min_end_y(&small) > -2.0, "small arc min y");
        assert!(min_end_y(&large) < -10.0, "large arc min y");
    }

    #[test]
    fn shape_builders_produce_closed_paths() {
        let rect = path_from_rect(0.0, 0.0, 10.0, 20.0, 0.0, 0.0);
        assert_eq!(rect.segments.len(), 5); // M + 3 L + Z
        let rounded = path_from_rect(0.0, 0.0, 10.0, 20.0, 2.0, 2.0);
        assert_eq!(rounded.segments.len(), 10); // M + L + C + L + C + L + C + L + C + Z
        let circle = path_from_circle(5.0, 5.0, 5.0);
        assert_eq!(circle.segments.len(), 6); // M + 4 C + Z
        assert_eq!(circle.bounds(), Some([0.0, 0.0, 10.0, 10.0]));
        let line = path_from_line(1.0, 2.0, 3.0, 4.0);
        assert_eq!(line.segments.len(), 2);
        let poly = path_from_points("0,0 10,0 10,10", true).expect("points parse");
        assert_eq!(poly.segments.len(), 4); // M + 2 L + Z
        assert!(path_from_rect(0.0, 0.0, 0.0, 10.0, 0.0, 0.0).is_empty());
        assert!(path_from_circle(0.0, 0.0, 0.0).is_empty());
    }

    #[test]
    fn fill_tessellation_produces_triangles_and_honours_fill_rule() {
        // A pentagram (self-intersecting): nonzero and evenodd
        // must differ — the center pentagon is filled under
        // nonzero, holed under evenodd.
        let star = "M 50 0 L 79 91 L 2 35 L 98 35 L 21 91 Z";
        let path = parse_path_data(star).expect("star parses");
        let red = [1.0, 0.0, 0.0, 1.0];
        let nonzero =
            tessellate_fill(&path, FillRule::NonZero, solid(red)).expect("tessellates");
        let evenodd =
            tessellate_fill(&path, FillRule::EvenOdd, solid(red)).expect("tessellates");
        assert!(!nonzero.is_empty());
        assert!(!evenodd.is_empty());
        assert_eq!(nonzero.len() % 3, 0);
        assert_eq!(evenodd.len() % 3, 0);
        assert_ne!(
            nonzero.len(),
            evenodd.len(),
            "fill rules must tessellate differently on a star"
        );
        // Every vertex is inside the star's bounds and red.
        for v in &nonzero {
            assert!(v.position[0] >= 0.0 && v.position[0] <= 100.0);
            assert!(v.position[1] >= 0.0 && v.position[1] <= 100.0);
            assert_eq!(v.color, red);
        }
    }

    #[test]
    fn stroke_tessellation_produces_triangles() {
        let path = parse_path_data("M 0 0 L 100 0").expect("line parses");
        let black = [0.0, 0.0, 0.0, 1.0];
        let tris = tessellate_stroke(
            &path,
            10.0,
            lyon::tessellation::LineCap::Butt,
            lyon::tessellation::LineJoin::Miter,
            4.0,
            solid(black),
        )
        .expect("tessellates");
        assert!(!tris.is_empty());
        assert_eq!(tris.len() % 3, 0);
        // The stroke is 10 wide around y = 0: vertices span
        // y in [-5, 5] and x in [0, 100].
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &tris {
            min_y = min_y.min(v.position[1]);
            max_y = max_y.max(v.position[1]);
        }
        assert!((min_y + 5.0).abs() < 0.01, "min_y {min_y}");
        assert!((max_y - 5.0).abs() < 0.01, "max_y {max_y}");
    }

    #[test]
    fn clip_colored_triangles_clips_to_rect() {
        let red = [1.0, 0.0, 0.0, 1.0];
        // One big triangle spanning x -10..110, clipped to 0..100.
        let tris = vec![
            VectorVertex { position: [-10.0, 50.0], color: red },
            VectorVertex { position: [110.0, 0.0], color: red },
            VectorVertex { position: [110.0, 100.0], color: red },
        ];
        let clipped = clip_colored_triangles(&tris, [0.0, 0.0, 100.0, 100.0]);
        assert!(!clipped.is_empty());
        assert_eq!(clipped.len() % 3, 0);
        for v in &clipped {
            assert!(v.position[0] >= -1e-4 && v.position[0] <= 100.0 + 1e-4);
            assert!(v.position[1] >= -1e-4 && v.position[1] <= 100.0 + 1e-4);
        }
    }

    #[test]
    fn linear_gradient_quad_is_exact_for_two_stops() {
        use pluto_css::{Gradient as CssGradient, GradientKind, GradientStop};
        let g = CssGradient {
            kind: GradientKind::Linear { angle_deg: 90.0 },
            stops: vec![
                GradientStop { offset: 0.0, color: [255, 0, 0, 255] },
                GradientStop { offset: 1.0, color: [0, 0, 255, 255] },
            ],
        };
        let tris = linear_gradient_quad(0.0, 0.0, 100.0, 50.0, &g);
        assert_eq!(tris.len(), 6);
        // Left edge is red, right edge is blue, middle is purple.
        for v in &tris {
            if v.position[0] < 0.01 {
                assert_eq!(v.color, [1.0, 0.0, 0.0, 1.0]);
            } else if v.position[0] > 99.99 {
                assert_eq!(v.color, [0.0, 0.0, 1.0, 1.0]);
            }
        }
    }

    #[test]
    fn linear_gradient_quad_survives_offset_rects_diagonals_and_three_stops() {
        use pluto_css::{Gradient as CssGradient, GradientKind, GradientStop};
        let two = |angle: f32| CssGradient {
            kind: GradientKind::Linear { angle_deg: angle },
            stops: vec![
                GradientStop { offset: 0.0, color: [255, 0, 0, 255] },
                GradientStop { offset: 1.0, color: [0, 0, 255, 255] },
            ],
        };
        // Regression: the old iso-line version produced ZERO
        // triangles for a rect at (8, 8) — the t=0 iso-line
        // passes exactly through a corner, and f32 rounding of
        // cos(90deg) decided whether the segment survived.
        let tris = linear_gradient_quad(8.0, 8.0, 100.0, 100.0, &two(90.0));
        assert_eq!(tris.len(), 6, "offset rect, 90deg");
        for v in &tris {
            assert!(v.position[0] >= 8.0 - 1e-4 && v.position[0] <= 108.0 + 1e-4);
            assert!(v.position[1] >= 8.0 - 1e-4 && v.position[1] <= 108.0 + 1e-4);
        }
        // A diagonal: the end iso-lines touch single corners, so
        // the strip is a hexagon fan, not a quad — still exact,
        // still inside the rect.
        let tris = linear_gradient_quad(0.0, 0.0, 100.0, 100.0, &two(45.0));
        assert!(!tris.is_empty() && tris.len().is_multiple_of(3), "45deg");
        for v in &tris {
            assert!(v.position[0] >= -1e-4 && v.position[0] <= 100.0 + 1e-4);
            assert!(v.position[1] >= -1e-4 && v.position[1] <= 100.0 + 1e-4);
        }
        // Three stops: two strips, twelve vertices; the middle
        // stop's colour appears on the shared boundary.
        let three = CssGradient {
            kind: GradientKind::Linear { angle_deg: 90.0 },
            stops: vec![
                GradientStop { offset: 0.0, color: [255, 0, 0, 255] },
                GradientStop { offset: 0.5, color: [0, 255, 0, 255] },
                GradientStop { offset: 1.0, color: [0, 0, 255, 255] },
            ],
        };
        let tris = linear_gradient_quad(0.0, 0.0, 100.0, 100.0, &three);
        assert_eq!(tris.len(), 12, "three stops = two strips");
        // Approximately green: the vertex `t` carries f32 noise
        // (~2e-8 from cos(90deg)), so the colour is green to
        // within 1e-6, not bit-exact.
        assert!(
            tris.iter().any(|v| v.color[1] > 0.999
                && v.color[0] < 0.001
                && v.color[2] < 0.001
                && (v.position[0] - 50.0).abs() < 0.01),
            "the middle stop paints the midline green"
        );
        // A radial gradient is the rasteriser's job, not this fn's.
        let radial = CssGradient {
            kind: GradientKind::Radial {
                center: (0.5, 0.5),
                shape: pluto_css::RadialShape::Circle,
                extent: pluto_css::RadialExtent::ClosestSide,
            },
            stops: vec![GradientStop { offset: 0.0, color: [255, 0, 0, 255] }],
        };
        assert!(linear_gradient_quad(0.0, 0.0, 100.0, 100.0, &radial).is_empty());
    }
}
