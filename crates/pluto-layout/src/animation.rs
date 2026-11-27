//! CSS transitions, keyframe animations, and the Web Animations
//! subset (`Element.animate` / `getAnimations`) — M6 Rec 11.
//!
//! Progress is sampled from a **monotonic [`Instant`]** at each tab
//! tick — never by counting ticks. The tab tick is 16 ms and a
//! vsync interval is 16.67 ms, so counting ticks would run every
//! animation ~4% fast. `advance(now)` takes the tick's `Instant`
//! and derives all progress from `now - started`, so a missed or
//! late tick only ever shifts the sample, never the rate.
//!
//! The engine is pure: it knows nothing about the DOM, the style
//! resolver, or the renderer. The tab thread owns one engine, feeds
//! it style changes (transition starts) and animation specs, and
//! applies the sampled values it returns. Values are
//! [`AnimatedValue`]s — numbers, resolved lengths, colours, and
//! transform lists — so the actor can write them straight back into
//! a [`ComputedStyle`] without re-parsing CSS text.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use pluto_css::{
    Animation, CalcCtx, CalcNumber, CssLength, Easing, KeyframeSelector, Keyframes,
    TransformFunction, TransformList, Transition, Value,
};
/// Re-exported for the tab thread, which builds [`AnimTiming`]
/// from WAAPI specs (M6 Phase 3, Rec 11).
pub use pluto_css::{AnimationDirection, FillMode, IterationCount};
use pluto_html::NodeId as DomId;

use crate::ComputedStyle;

/// A sampled animatable value: always in *used* form (lengths are
/// resolved pixels, colours are RGBA, transforms are function
/// lists), so interpolation never has to re-resolve units.
#[derive(Debug, Clone, PartialEq)]
pub enum AnimatedValue {
    /// A bare number (`opacity`, `font-size`).
    Number(f32),
    /// A resolved length in px.
    Length(f32),
    /// RGBA colour.
    Colour(u8, u8, u8, f32),
    /// A `transform` function list.
    Transform(TransformList),
}

impl AnimatedValue {
    /// Resolve a parsed CSS [`Value`] to an animatable value, or
    /// `None` when the value is not animatable (keywords, URLs,
    /// `calc()` that will not resolve against `ctx`, ...).
    pub fn from_value(value: &Value, ctx: &CalcCtx) -> Option<Self> {
        match value {
            Value::Length(v) => Some(AnimatedValue::Length(*v)),
            Value::Em(v) => Some(AnimatedValue::Length(*v * ctx.font_size)),
            Value::Rem(v) => Some(AnimatedValue::Length(*v * ctx.root_font_size)),
            Value::Vh(v) => Some(AnimatedValue::Length(*v / 100.0 * ctx.viewport.1)),
            Value::Vw(v) => Some(AnimatedValue::Length(*v / 100.0 * ctx.viewport.0)),
            Value::Percentage(v) => ctx
                .pct_basis
                .map(|b| AnimatedValue::Length(*v / 100.0 * b)),
            Value::Calc(expr) => match expr.eval(ctx)? {
                CalcNumber::Px(v) => Some(AnimatedValue::Length(v)),
                CalcNumber::Pct(_) => None,
            },
            Value::Clamp { min, val, max } => {
                let px = |e: &pluto_css::CalcExpr| match e.eval(ctx)? {
                    CalcNumber::Px(v) => Some(v),
                    CalcNumber::Pct(_) => None,
                };
                let (lo, v, hi) = (px(min)?, px(val)?, px(max)?);
                Some(AnimatedValue::Length(v.clamp(lo.min(hi), hi.max(lo))))
            }
            Value::Color(r, g, b) => Some(AnimatedValue::Colour(*r, *g, *b, 1.0)),
            Value::ColorA(r, g, b, a) => Some(AnimatedValue::Colour(*r, *g, *b, *a)),
            Value::Transform(list) => Some(AnimatedValue::Transform(list.clone())),
            Value::Raw(s) => s.trim().parse::<f32>().ok().map(AnimatedValue::Number),
            _ => None,
        }
    }

    /// Interpolate towards `other` at progress `t` in `[0, 1]`.
    /// `None` means the pair is not interpolable (mismatched
    /// kinds, or transform lists whose functions differ) — the
    /// caller falls back to a discrete switch at `t = 0.5`.
    pub fn lerp(&self, other: &Self, t: f32, ctx: &CalcCtx) -> Option<Self> {
        let t = t.clamp(0.0, 1.0);
        match (self, other) {
            (AnimatedValue::Number(a), AnimatedValue::Number(b)) => {
                Some(AnimatedValue::Number(a + (b - a) * t))
            }
            (AnimatedValue::Length(a), AnimatedValue::Length(b)) => {
                Some(AnimatedValue::Length(a + (b - a) * t))
            }
            (
                AnimatedValue::Colour(ar, ag, ab, aa),
                AnimatedValue::Colour(br, bg, bb, ba),
            ) => Some(AnimatedValue::Colour(
                lerp_u8(*ar, *br, t),
                lerp_u8(*ag, *bg, t),
                lerp_u8(*ab, *bb, t),
                aa + (ba - aa) * t,
            )),
            (AnimatedValue::Transform(a), AnimatedValue::Transform(b)) => {
                lerp_transform(a, b, t, ctx).map(AnimatedValue::Transform)
            }
            _ => None,
        }
    }

    /// The value as CSS text, for the `getComputedStyle` cache and
    /// for `Element.animate` keyframe reporting. Lengths serialise
    /// as `px`; colours as `rgb()` / `rgba()`.
    pub fn to_css(&self) -> String {
        match self {
            AnimatedValue::Number(v) => format!("{v}"),
            AnimatedValue::Length(v) => format!("{v}px"),
            AnimatedValue::Colour(r, g, b, a) => {
                if (*a - 1.0).abs() < 1e-6 {
                    format!("rgb({r}, {g}, {b})")
                } else {
                    format!("rgba({r}, {g}, {b}, {a})")
                }
            }
            AnimatedValue::Transform(list) => transform_to_css(list),
        }
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

fn css_length_px(l: CssLength, ctx: &CalcCtx) -> f32 {
    match l {
        CssLength::Px(v) => v,
        CssLength::Percentage(v) => ctx.pct_basis.map_or(0.0, |b| v / 100.0 * b),
        CssLength::Em(v) => v * ctx.font_size,
        CssLength::Rem(v) => v * ctx.root_font_size,
        CssLength::Vh(v) => v / 100.0 * ctx.viewport.1,
        CssLength::Vw(v) => v / 100.0 * ctx.viewport.0,
    }
}

/// Interpolate two transform lists function-by-function. The
/// lists must have the same length and matching function kinds;
/// lengths resolve to px through `ctx` and re-emit as `Px`.
/// Anything else is a discrete change (`None`).
fn lerp_transform(
    a: &[TransformFunction],
    b: &[TransformFunction],
    t: f32,
    ctx: &CalcCtx,
) -> Option<TransformList> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for (fa, fb) in a.iter().zip(b.iter()) {
        let f = match (fa, fb) {
            (
                TransformFunction::Translate(ax, ay),
                TransformFunction::Translate(bx, by),
            ) => TransformFunction::Translate(
                CssLength::Px(lerp_len(*ax, *bx, t, ctx)),
                CssLength::Px(lerp_len(*ay, *by, t, ctx)),
            ),
            (TransformFunction::TranslateX(ax), TransformFunction::TranslateX(bx)) => {
                TransformFunction::TranslateX(CssLength::Px(lerp_len(*ax, *bx, t, ctx)))
            }
            (TransformFunction::TranslateY(ay), TransformFunction::TranslateY(by)) => {
                TransformFunction::TranslateY(CssLength::Px(lerp_len(*ay, *by, t, ctx)))
            }
            (TransformFunction::Scale(ax, ay), TransformFunction::Scale(bx, by)) => {
                TransformFunction::Scale(ax + (bx - ax) * t, ay + (by - ay) * t)
            }
            (TransformFunction::ScaleX(ax), TransformFunction::ScaleX(bx)) => {
                TransformFunction::ScaleX(ax + (bx - ax) * t)
            }
            (TransformFunction::ScaleY(ay), TransformFunction::ScaleY(by)) => {
                TransformFunction::ScaleY(ay + (by - ay) * t)
            }
            (TransformFunction::Rotate(aa), TransformFunction::Rotate(ba)) => {
                TransformFunction::Rotate(aa + (ba - aa) * t)
            }
            (TransformFunction::SkewX(aa), TransformFunction::SkewX(ba)) => {
                TransformFunction::SkewX(aa + (ba - aa) * t)
            }
            (TransformFunction::SkewY(aa), TransformFunction::SkewY(ba)) => {
                TransformFunction::SkewY(aa + (ba - aa) * t)
            }
            (TransformFunction::Matrix(ma), TransformFunction::Matrix(mb)) => {
                let mut m = [0.0f32; 6];
                for i in 0..6 {
                    m[i] = ma[i] + (mb[i] - ma[i]) * t;
                }
                TransformFunction::Matrix(m)
            }
            _ => return None,
        };
        out.push(f);
    }
    Some(out)
}

fn lerp_len(a: CssLength, b: CssLength, t: f32, ctx: &CalcCtx) -> f32 {
    let pa = css_length_px(a, ctx);
    let pb = css_length_px(b, ctx);
    pa + (pb - pa) * t
}

fn transform_to_css(list: &[TransformFunction]) -> String {
    if list.is_empty() {
        return "none".to_string();
    }
    let mut parts = Vec::with_capacity(list.len());
    for f in list {
        let len = |l: CssLength| match l {
            CssLength::Px(v) => format!("{v}px"),
            CssLength::Percentage(v) => format!("{v}%"),
            CssLength::Em(v) => format!("{v}em"),
            CssLength::Rem(v) => format!("{v}rem"),
            CssLength::Vh(v) => format!("{v}vh"),
            CssLength::Vw(v) => format!("{v}vw"),
        };
        parts.push(match f {
            TransformFunction::Translate(x, y) => {
                format!("translate({}, {})", len(*x), len(*y))
            }
            TransformFunction::TranslateX(x) => format!("translateX({})", len(*x)),
            TransformFunction::TranslateY(y) => format!("translateY({})", len(*y)),
            TransformFunction::Scale(x, y) => format!("scale({x}, {y})"),
            TransformFunction::ScaleX(x) => format!("scaleX({x})"),
            TransformFunction::ScaleY(y) => format!("scaleY({y})"),
            TransformFunction::Rotate(a) => format!("rotate({a}deg)"),
            TransformFunction::SkewX(a) => format!("skewX({a}deg)"),
            TransformFunction::SkewY(a) => format!("skewY({a}deg)"),
            TransformFunction::Matrix(m) => format!(
                "matrix({}, {}, {}, {}, {}, {})",
                m[0], m[1], m[2], m[3], m[4], m[5]
            ),
        });
    }
    parts.join(" ")
}

/// The properties the engine can animate, in one place so the
/// actor, the computed-style cache, and the transition detector
/// agree on the set.
pub const ANIMATABLE_PROPERTIES: &[&str] = &[
    "opacity",
    "transform",
    "color",
    "background-color",
    "width",
    "height",
    "left",
    "right",
    "top",
    "bottom",
    "font-size",
];

/// The animatable properties of `style` as `(property, css text)`
/// pairs, for the tab thread's `getComputedStyle` cache. Only
/// properties that resolve through `ctx` are included (a `%` width
/// with no basis is omitted — the cache keeps the last value it
/// had, or nothing).
pub fn animatable_pairs(style: &ComputedStyle, ctx: &CalcCtx) -> Vec<(String, String)> {
    ANIMATABLE_PROPERTIES
        .iter()
        .filter_map(|p| {
            animated_value_of_style(style, p, ctx).map(|v| (p.to_string(), v.to_css()))
        })
        .collect()
}

/// Read the underlying (non-animated) value of `property` out of
/// a computed style, for transition `from` snapshots and for the
/// base values a keyframe animation returns to when its fill is
/// `none`. Lengths resolve through `ctx` (percentages against the
/// border-box basis the caller supplies). `None` means the property
/// is not animatable through this engine.
pub fn animated_value_of_style(
    style: &ComputedStyle,
    property: &str,
    ctx: &CalcCtx,
) -> Option<AnimatedValue> {
    use crate::Dim;
    let dim_px = |d: Option<Dim>| -> Option<AnimatedValue> {
        match d? {
            Dim::Px(v) => Some(AnimatedValue::Length(v)),
            Dim::Pct(f) => ctx.pct_basis.map(|b| AnimatedValue::Length(f * b)),
            Dim::Vh(f) => Some(AnimatedValue::Length(f * ctx.viewport.1)),
            Dim::Vw(f) => Some(AnimatedValue::Length(f * ctx.viewport.0)),
        }
    };
    match property {
        "opacity" => Some(AnimatedValue::Number(style.opacity)),
        "transform" => Some(AnimatedValue::Transform(style.transform.clone())),
        "color" => Some(AnimatedValue::Colour(
            style.color[0],
            style.color[1],
            style.color[2],
            1.0,
        )),
        "background-color" => style
            .background
            .map(|[r, g, b]| AnimatedValue::Colour(r, g, b, 1.0)),
        "width" => dim_px(style.width),
        "height" => dim_px(style.height),
        "left" => dim_px(style.inset[3]),
        "right" => dim_px(style.inset[1]),
        "top" => dim_px(style.inset[0]),
        "bottom" => dim_px(style.inset[2]),
        "font-size" => Some(AnimatedValue::Number(style.font_size)),
        _ => None,
    }
}

/// Write a sampled [`AnimatedValue`] back into a computed style.
/// The inverse of [`animated_value_of_style`]; the actor applies
/// this to the resolved styles before classification, so an
/// in-flight value flows through the same damage path as any other
/// style change. Unknown properties are ignored.
pub fn apply_animated_value(style: &mut ComputedStyle, property: &str, value: &AnimatedValue) {
    use crate::Dim;
    match (property, value) {
        // `opacity: 0` parses as a length (bare `0` is
        // `Length(0.0)`), so keyframe values arrive as `Length`
        // while style snapshots arrive as `Number` — accept both.
        ("opacity", AnimatedValue::Number(v)) | ("opacity", AnimatedValue::Length(v)) => {
            style.opacity = v.clamp(0.0, 1.0)
        }
        ("transform", AnimatedValue::Transform(list)) => style.transform = list.clone(),
        ("color", AnimatedValue::Colour(r, g, b, _)) => style.color = [*r, *g, *b],
        ("background-color", AnimatedValue::Colour(r, g, b, _)) => {
            style.background = Some([*r, *g, *b]);
        }
        ("width", AnimatedValue::Length(v)) => style.width = Some(Dim::Px(*v)),
        ("height", AnimatedValue::Length(v)) => style.height = Some(Dim::Px(*v)),
        ("left", AnimatedValue::Length(v)) => style.inset[3] = Some(Dim::Px(*v)),
        ("right", AnimatedValue::Length(v)) => style.inset[1] = Some(Dim::Px(*v)),
        ("top", AnimatedValue::Length(v)) => style.inset[0] = Some(Dim::Px(*v)),
        ("bottom", AnimatedValue::Length(v)) => style.inset[2] = Some(Dim::Px(*v)),
        ("font-size", AnimatedValue::Number(v)) => style.font_size = *v,
        _ => {}
    }
}

/// The timing parameters shared by CSS animations and WAAPI
/// animations (both are the same machine; only the event surface
/// differs).
#[derive(Debug, Clone, PartialEq)]
pub struct AnimTiming {
    pub duration: f32,
    pub delay: f32,
    pub timing: Easing,
    pub iteration_count: IterationCount,
    pub direction: AnimationDirection,
    pub fill_mode: FillMode,
}

impl From<&Animation> for AnimTiming {
    fn from(a: &Animation) -> Self {
        AnimTiming {
            duration: a.duration,
            delay: a.delay,
            timing: a.timing.clone(),
            iteration_count: a.iteration_count,
            direction: a.direction,
            fill_mode: a.fill_mode,
        }
    }
}

/// One running CSS transition: `from` → `to` over `duration`,
/// starting after `delay`, sampled with `spec.timing`.
#[derive(Debug, Clone)]
struct RunningTransition {
    property: String,
    spec: Transition,
    from: AnimatedValue,
    to: AnimatedValue,
    started: Instant,
}

/// One running keyframe animation (CSS `animation` or WAAPI):
/// normalised keyframe steps plus the underlying base values it
/// returns to when its fill is `none`.
#[derive(Debug, Clone)]
struct RunningKeyframes {
    timing: AnimTiming,
    /// The CSS animation name (`Some`), or `None` for WAAPI.
    name: Option<String>,
    /// `(offset 0..1, [(property, value)])`, sorted by offset.
    steps: Vec<(f32, Vec<(String, AnimatedValue)>)>,
    /// Underlying values for the animated properties.
    base: HashMap<String, AnimatedValue>,
    started: Instant,
    /// When this entry's own pause began (effective time — see
    /// [`AnimationEngine::effective_now`]), or `None` while running.
    /// A paused entry samples its frozen progress; `resume` folds
    /// the paused interval into `accumulated_pause`.
    paused_at: Option<Instant>,
    /// Total time this entry spent paused so far.
    accumulated_pause: Duration,
    /// WAAPI animation id, or `None` for a CSS `animation`.
    waapi_id: Option<u64>,
    /// `true` once the post-fill value has been emitted, so a
    /// finished animation does not re-report every tick.
    finish_emitted: bool,
}

/// What [`AnimationEngine::advance`] found at one tick.
#[derive(Debug, Default)]
pub struct EngineOutcome {
    /// `(element, [(property, sampled value)])` — the caller
    /// applies these to the resolved styles and marks the element
    /// dirty. Only properties whose sampled value changed since
    /// the last tick are included.
    pub changed: Vec<(DomId, Vec<(String, AnimatedValue)>)>,
    /// `(element, event type)` — `"transitionend"` /
    /// `"animationend"`, dispatched through the tab's JS.
    pub finished_events: Vec<(DomId, &'static str)>,
    /// WAAPI animation ids that finished this tick, per element —
    /// the tab thread updates the matching `Animation` object's
    /// `playState` / `currentTime` from these.
    pub waapi_finished: Vec<(DomId, u64)>,
}

#[derive(Debug)]
pub struct AnimationEngine {
    transitions: HashMap<DomId, Vec<RunningTransition>>,
    animations: HashMap<DomId, Vec<RunningKeyframes>>,
    /// The resolution context for sampled values (viewport, root
    /// font size) — the tab thread refreshes it when the viewport
    /// changes. Percentages in transform lists resolve against a
    /// zero basis here; layout-time values use the element's box.
    ctx: CalcCtx,
    /// Paused while the tab is hidden (animations freeze; the
    /// clock resumes where it left off, it does not jump).
    paused_since: Option<Instant>,
    accumulated_pause: Duration,
}

impl Default for AnimationEngine {
    fn default() -> Self {
        AnimationEngine {
            transitions: HashMap::new(),
            animations: HashMap::new(),
            ctx: CalcCtx {
                font_size: 16.0,
                root_font_size: 16.0,
                viewport: (0.0, 0.0),
                pct_basis: None,
            },
            paused_since: None,
            accumulated_pause: Duration::ZERO,
        }
    }
}

impl AnimationEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty() && self.animations.is_empty()
    }

    /// Refresh the resolution context (viewport / font sizes).
    pub fn set_ctx(&mut self, ctx: CalcCtx) {
        self.ctx = ctx;
    }

    /// Start (or restart) a transition on `id`. A running
    /// transition on the same property is replaced — its current
    /// sampled value becomes the new `from` only when the caller
    /// passes it; the engine itself just swaps the entry.
    pub fn start_transition(
        &mut self,
        id: DomId,
        spec: Transition,
        from: AnimatedValue,
        to: AnimatedValue,
        now: Instant,
    ) {
        let list = self.transitions.entry(id).or_default();
        list.retain(|t| t.property != spec.property);
        list.push(RunningTransition {
            property: spec.property.clone(),
            spec,
            from,
            to,
            started: now,
        });
    }

    /// Start a CSS `animation` on `id`, resolving its keyframe
    /// steps against `ctx` (lengths become px; unresolvable
    /// declarations are dropped from their step). The animation
    /// replaces any running CSS animation of the same name on the
    /// element — WAAPI animations are independent and untouched.
    pub fn start_animation(
        &mut self,
        id: DomId,
        spec: Animation,
        keyframes: &Keyframes,
        base: HashMap<String, AnimatedValue>,
        ctx: &CalcCtx,
        now: Instant,
    ) {
        let steps = normalise_steps(keyframes, ctx);
        if steps.is_empty() {
            return;
        }
        let list = self.animations.entry(id).or_default();
        list.retain(|a| {
            !(a.waapi_id.is_none() && a.name.as_deref() == Some(spec.name.as_str()))
        });
        list.push(RunningKeyframes {
            timing: AnimTiming::from(&spec),
            name: Some(spec.name.clone()),
            steps,
            base,
            started: now,
            paused_at: None,
            accumulated_pause: Duration::ZERO,
            waapi_id: None,
            finish_emitted: false,
        });
    }

    /// Start a WAAPI animation (`Element.animate`). `steps` arrive
    /// already normalised (the tab thread parses the keyframe
    /// objects); `anim_id` is the id the JS side tracks.
    pub fn start_waapi(
        &mut self,
        id: DomId,
        anim_id: u64,
        timing: AnimTiming,
        steps: Vec<(f32, Vec<(String, AnimatedValue)>)>,
        base: HashMap<String, AnimatedValue>,
        now: Instant,
    ) {
        if steps.is_empty() {
            return;
        }
        let list = self.animations.entry(id).or_default();
        list.retain(|a| a.waapi_id != Some(anim_id));
        list.push(RunningKeyframes {
            timing,
            name: None,
            steps,
            base,
            started: now,
            paused_at: None,
            accumulated_pause: Duration::ZERO,
            waapi_id: Some(anim_id),
            finish_emitted: false,
        });
    }

    /// The entry matching `pred` on `id`, if any.
    fn find_entry_mut(
        &mut self,
        id: DomId,
        pred: impl Fn(&RunningKeyframes) -> bool,
    ) -> Option<&mut RunningKeyframes> {
        self.animations
            .get_mut(&id)?
            .iter_mut()
            .find(|a| pred(a))
    }

    /// Elapsed seconds for one entry at effective time `now`,
    /// less its own paused intervals (a paused entry is frozen).
    fn entry_elapsed(a: &RunningKeyframes, now: Instant) -> f32 {
        let end = a.paused_at.unwrap_or(now);
        end.saturating_duration_since(a.started)
            .saturating_sub(a.accumulated_pause)
            .as_secs_f32()
    }

    /// Total active seconds of one entry's iterations (`INFINITY`
    /// for an infinite count).
    fn entry_total(a: &RunningKeyframes) -> f32 {
        let dur = a.timing.duration.max(0.0);
        match a.timing.iteration_count {
            IterationCount::Finite(n) => dur * n.max(0.0),
            IterationCount::Infinite => f32::INFINITY,
        }
    }

    /// Freeze one WAAPI animation at its current progress.
    pub fn pause_waapi(&mut self, id: DomId, anim_id: u64, now: Instant) -> bool {
        let eff = self.effective_now(now);
        match self.find_entry_mut(id, |a| a.waapi_id == Some(anim_id)) {
            Some(a) if a.paused_at.is_none() => {
                a.paused_at = Some(eff);
                true
            }
            _ => false,
        }
    }

    /// Resume a paused WAAPI animation (the paused interval folds
    /// into its accumulated pause, so it does not jump).
    pub fn resume_waapi(&mut self, id: DomId, anim_id: u64, now: Instant) -> bool {
        let eff = self.effective_now(now);
        match self.find_entry_mut(id, |a| a.waapi_id == Some(anim_id)) {
            Some(a) => match a.paused_at.take() {
                Some(p) => {
                    a.accumulated_pause += eff.saturating_duration_since(p);
                    true
                }
                None => false,
            },
            None => false,
        }
    }

    /// Jump a WAAPI animation to the end of its active interval:
    /// the next `advance` emits the final values and the finish.
    /// A finite animation ends at its total; an infinite one at
    /// the end of its current iteration. Works while paused.
    pub fn finish_waapi(&mut self, id: DomId, anim_id: u64, now: Instant) -> bool {
        let eff = self.effective_now(now);
        let Some(a) = self.find_entry_mut(id, |a| a.waapi_id == Some(anim_id)) else {
            return false;
        };
        let dur = a.timing.duration.max(0.0);
        let target = {
            let total = Self::entry_total(a);
            if total.is_finite() {
                total
            } else if dur > 0.0 {
                let elapsed = Self::entry_elapsed(a, eff) - a.timing.delay;
                (elapsed / dur).floor().max(0.0).mul_add(dur, dur)
            } else {
                0.0
            }
        };
        // Rewind `started` so the entry's elapsed time equals
        // `delay + target` at `eff` (paused time excluded).
        let want = a.timing.delay + target;
        a.started = eff - a.accumulated_pause - Duration::from_secs_f32(want.max(0.0));
        a.paused_at = None;
        a.finish_emitted = false;
        true
    }

    /// Cancel a WAAPI animation by id; returns its underlying base
    /// values when it was running (the caller restores them — fill
    /// `none` semantics), or `None` when it was not running.
    pub fn cancel_waapi(
        &mut self,
        id: DomId,
        anim_id: u64,
    ) -> Option<HashMap<String, AnimatedValue>> {
        let list = self.animations.get_mut(&id)?;
        let pos = list.iter().position(|a| a.waapi_id == Some(anim_id))?;
        Some(list.remove(pos).base)
    }

    /// Freeze one CSS animation by name at its current progress.
    pub fn pause_css(&mut self, id: DomId, name: &str, now: Instant) -> bool {
        let eff = self.effective_now(now);
        match self.find_entry_mut(id, |a| {
            a.waapi_id.is_none() && a.name.as_deref() == Some(name)
        }) {
            Some(a) if a.paused_at.is_none() => {
                a.paused_at = Some(eff);
                true
            }
            _ => false,
        }
    }

    /// Resume a paused CSS animation by name.
    pub fn resume_css(&mut self, id: DomId, name: &str, now: Instant) -> bool {
        let eff = self.effective_now(now);
        match self.find_entry_mut(id, |a| {
            a.waapi_id.is_none() && a.name.as_deref() == Some(name)
        }) {
            Some(a) => match a.paused_at.take() {
                Some(p) => {
                    a.accumulated_pause += eff.saturating_duration_since(p);
                    true
                }
                None => false,
            },
            None => false,
        }
    }

    /// Jump a CSS animation by name to the end of its active
    /// interval (same rule as [`Self::finish_waapi`]).
    pub fn finish_css(&mut self, id: DomId, name: &str, now: Instant) -> bool {
        let eff = self.effective_now(now);
        let Some(a) = self.find_entry_mut(id, |a| {
            a.waapi_id.is_none() && a.name.as_deref() == Some(name)
        }) else {
            return false;
        };
        let dur = a.timing.duration.max(0.0);
        let target = {
            let total = Self::entry_total(a);
            if total.is_finite() {
                total
            } else if dur > 0.0 {
                let elapsed = Self::entry_elapsed(a, eff) - a.timing.delay;
                (elapsed / dur).floor().max(0.0).mul_add(dur, dur)
            } else {
                0.0
            }
        };
        let want = a.timing.delay + target;
        a.started = eff - a.accumulated_pause - Duration::from_secs_f32(want.max(0.0));
        a.paused_at = None;
        a.finish_emitted = false;
        true
    }

    /// Cancel a CSS animation by name; returns its underlying base
    /// values when it was running.
    pub fn cancel_css(
        &mut self,
        id: DomId,
        name: &str,
    ) -> Option<HashMap<String, AnimatedValue>> {
        let list = self.animations.get_mut(&id)?;
        let pos = list
            .iter()
            .position(|a| a.waapi_id.is_none() && a.name.as_deref() == Some(name))?;
        Some(list.remove(pos).base)
    }

    /// Drop every running transition/animation on `id` (element
    /// removed, style reset).
    pub fn cancel_for(&mut self, id: DomId) {
        self.transitions.remove(&id);
        self.animations.remove(&id);
    }

    /// Drop everything — the navigation boundary.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.animations.clear();
        self.paused_since = None;
        self.accumulated_pause = Duration::ZERO;
    }

    /// Freeze the clock (tab hidden). `advance` keeps returning
    /// the frozen values until [`resume`](Self::resume).
    pub fn pause(&mut self, now: Instant) {
        if self.paused_since.is_none() {
            self.paused_since = Some(now);
        }
    }

    /// Resume the clock: the paused interval is added to the
    /// accumulated pause, so no animation jumps.
    pub fn resume(&mut self, now: Instant) {
        if let Some(since) = self.paused_since.take() {
            self.accumulated_pause += now.saturating_duration_since(since);
        }
    }

    /// The monotonic clock the engine samples against: wall time
    /// minus every paused interval.
    fn effective_now(&self, now: Instant) -> Instant {
        let mut eff = now - self.accumulated_pause;
        if let Some(since) = self.paused_since {
            // Frozen at the pause point.
            eff = since - self.accumulated_pause;
        }
        eff
    }

    /// Advance to `now`, sampling every running transition and
    /// animation from the monotonic clock. Finished entries are
    /// removed; their final values and events are reported once.
    pub fn advance(&mut self, now: Instant) -> EngineOutcome {
        let now = self.effective_now(now);
        let mut out = EngineOutcome::default();

        // Transitions.
        let mut empty_t = Vec::new();
        for (id, list) in self.transitions.iter_mut() {
            let mut i = 0;
            while i < list.len() {
                let tr = &list[i];
                let elapsed =
                    now.saturating_duration_since(tr.started).as_secs_f32() - tr.spec.delay;
                if elapsed < 0.0 {
                    i += 1;
                    continue;
                }
                let dur = tr.spec.duration.max(0.0);
                if dur <= 0.0 || elapsed >= dur {
                    let value = tr.to.clone();
                    let prop = tr.property.clone();
                    push_changed(&mut out.changed, *id, prop, value);
                    out.finished_events.push((*id, "transitionend"));
                    list.remove(i);
                } else {
                    let p = tr.spec.timing.sample(elapsed / dur);
                    let value = tr
                        .from
                        .lerp(&tr.to, p, &self.ctx)
                        .unwrap_or(if p < 0.5 { tr.from.clone() } else { tr.to.clone() });
                    let prop = tr.property.clone();
                    push_changed(&mut out.changed, *id, prop, value);
                    i += 1;
                }
            }
            if list.is_empty() {
                empty_t.push(*id);
            }
        }
        for id in empty_t {
            self.transitions.remove(&id);
        }

        // Keyframe animations (CSS + WAAPI share the machine).
        let mut empty_a = Vec::new();
        for (id, list) in self.animations.iter_mut() {
            let mut i = 0;
            while i < list.len() {
                let a = &mut list[i];
                let ctx = self.ctx;
                let local = Self::entry_elapsed(a, now) - a.timing.delay;
                let dur = a.timing.duration.max(0.0);
                let total = Self::entry_total(a);
                if local < 0.0 {
                    // In the delay: fill backwards/both shows the
                    // first keyframe, otherwise nothing applies.
                    if matches!(a.timing.fill_mode, FillMode::Backwards | FillMode::Both) {
                        let vals = first_step_values(a);
                        push_changed_many(&mut out.changed, *id, vals);
                    }
                    i += 1;
                    continue;
                }
                if dur <= 0.0 || local >= total {
                    // Finished: hold the last keyframe under a
                    // forwards fill, else return to the base
                    // values — exactly once.
                    if !a.finish_emitted {
                        let vals = if matches!(
                            a.timing.fill_mode,
                            FillMode::Forwards | FillMode::Both
                        ) {
                            last_step_values(a)
                        } else {
                            a.base
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        };
                        push_changed_many(&mut out.changed, *id, vals);
                        match a.waapi_id {
                            Some(wid) => out.waapi_finished.push((*id, wid)),
                            None => out.finished_events.push((*id, "animationend")),
                        }
                        a.finish_emitted = true;
                    }
                    list.remove(i);
                    continue;
                }
                let iteration = (local / dur).floor();
                let mut frac = local / dur - iteration;
                frac = match a.timing.direction {
                    AnimationDirection::Normal => frac,
                    AnimationDirection::Reverse => 1.0 - frac,
                    AnimationDirection::Alternate => {
                        if (iteration as u64).is_multiple_of(2) {
                            frac
                        } else {
                            1.0 - frac
                        }
                    }
                    AnimationDirection::AlternateReverse => {
                        if (iteration as u64).is_multiple_of(2) {
                            1.0 - frac
                        } else {
                            frac
                        }
                    }
                };
                let eased = a.timing.timing.sample(frac);
                let vals = sample_steps(&a.steps, eased, &ctx);
                push_changed_many(&mut out.changed, *id, vals);
                i += 1;
            }
            if list.is_empty() {
                empty_a.push(*id);
            }
        }
        for id in empty_a {
            self.animations.remove(&id);
        }
        out
    }

    /// Snapshot for `getAnimations`: the names of the CSS
    /// animations still running on `id`.
    pub fn running_css_animations(&self, id: DomId) -> Vec<String> {
        self.animations
            .get(&id)
            .map(|list| {
                list.iter()
                    .filter(|a| a.waapi_id.is_none())
                    .filter_map(|a| a.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// WAAPI state for `getAnimations`: `(anim_id, running)` per
    /// WAAPI entry on `id`.
    pub fn waapi_ids(&self, id: DomId) -> Vec<u64> {
        self.animations
            .get(&id)
            .map(|list| list.iter().filter_map(|a| a.waapi_id).collect())
            .unwrap_or_default()
    }
}

/// Append one sampled value to the outcome, merging with an
/// earlier entry for the same element.
fn push_changed(
    changed: &mut Vec<(DomId, Vec<(String, AnimatedValue)>)>,
    id: DomId,
    property: String,
    value: AnimatedValue,
) {
    if let Some((_, vals)) = changed.iter_mut().find(|(e, _)| *e == id) {
        vals.push((property, value));
    } else {
        changed.push((id, vec![(property, value)]));
    }
}

fn push_changed_many(
    changed: &mut Vec<(DomId, Vec<(String, AnimatedValue)>)>,
    id: DomId,
    values: Vec<(String, AnimatedValue)>,
) {
    for (prop, value) in values {
        push_changed(changed, id, prop, value);
    }
}

/// Normalise a parsed `@keyframes` rule into sorted
/// `(offset, [(property, value)])` steps: `from` = 0, `to` = 1,
/// multiple selectors on one step expand to one entry each,
/// duplicate offsets merge (later declarations win).
fn normalise_steps(keyframes: &Keyframes, ctx: &CalcCtx) -> Vec<(f32, Vec<(String, AnimatedValue)>)> {
    let mut steps: Vec<(f32, Vec<(String, AnimatedValue)>)> = Vec::new();
    for step in &keyframes.steps {
        let values: Vec<(String, AnimatedValue)> = step
            .declarations
            .iter()
            .filter_map(|d| {
                AnimatedValue::from_value(&d.value, ctx).map(|v| (d.property.clone(), v))
            })
            .collect();
        if values.is_empty() {
            continue;
        }
        for sel in &step.selectors {
            let offset = match sel {
                KeyframeSelector::From => 0.0,
                KeyframeSelector::To => 1.0,
                KeyframeSelector::Percentage(p) => (*p / 100.0).clamp(0.0, 1.0),
            };
            if let Some((_, existing)) = steps.iter_mut().find(|(o, _)| (*o - offset).abs() < 1e-6)
            {
                for (prop, val) in &values {
                    if let Some((_, old)) =
                        existing.iter_mut().find(|(p, _)| *p == *prop)
                    {
                        *old = val.clone();
                    } else {
                        existing.push((prop.clone(), val.clone()));
                    }
                }
            } else {
                steps.push((offset, values.clone()));
            }
        }
    }
    steps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    steps
}

/// Sample the normalised steps at eased progress `e`: per
/// property, bracket its own (offset, value) timeline and
/// interpolate; before the first defined offset the first value
/// holds, after the last the last value holds.
fn sample_steps(
    steps: &[(f32, Vec<(String, AnimatedValue)>)],
    e: f32,
    ctx: &CalcCtx,
) -> Vec<(String, AnimatedValue)> {
    let mut props: HashMap<String, Vec<(f32, AnimatedValue)>> = HashMap::new();
    for (offset, values) in steps {
        for (prop, val) in values {
            props
                .entry(prop.clone())
                .or_default()
                .push((*offset, val.clone()));
        }
    }
    let mut out = Vec::new();
    for (prop, timeline) in props {
        let mut timeline = timeline;
        timeline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let value = if e <= timeline[0].0 {
            timeline[0].1.clone()
        } else if e >= timeline[timeline.len() - 1].0 {
            timeline[timeline.len() - 1].1.clone()
        } else {
            let mut val = timeline[timeline.len() - 1].1.clone();
            for w in timeline.windows(2) {
                let (o0, v0) = &w[0];
                let (o1, v1) = &w[1];
                if e >= *o0 && e <= *o1 {
                    let t = if (*o1 - *o0).abs() < 1e-6 {
                        1.0
                    } else {
                        (e - *o0) / (*o1 - *o0)
                    };
                    val = v0
                        .lerp(v1, t, ctx)
                        .unwrap_or(if t < 0.5 { v0.clone() } else { v1.clone() });
                    break;
                }
            }
            val
        };
        out.push((prop, value));
    }
    out
}

fn first_step_values(a: &RunningKeyframes) -> Vec<(String, AnimatedValue)> {
    a.steps
        .first()
        .map(|(_, vals)| vals.clone())
        .unwrap_or_default()
}

fn last_step_values(a: &RunningKeyframes) -> Vec<(String, AnimatedValue)> {
    a.steps
        .last()
        .map(|(_, vals)| vals.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::{Easing, Keyframes, Stylesheet};

    fn ctx() -> CalcCtx {
        CalcCtx {
            font_size: 16.0,
            root_font_size: 16.0,
            viewport: (800.0, 600.0),
            pct_basis: None,
        }
    }



    /// Read a numeric sampled value regardless of whether it
    /// arrived as `Number` (style snapshot) or `Length`
    /// (`opacity: 0` parses as a bare-zero length).
    fn num(v: &AnimatedValue) -> f32 {
        match v {
            AnimatedValue::Number(n) | AnimatedValue::Length(n) => *n,
            other => panic!("not numeric: {other:?}"),
        }
    }

    /// A real element id from a tiny parsed document.
    fn test_dom_id() -> DomId {
        let dom = pluto_html::parse("<div></div>");
        crate::find_element(&dom, dom.root().unwrap(), "div").unwrap()
    }

    fn opacity_frames() -> Keyframes {
        let mut sheet = Stylesheet::parse(
            "@keyframes fade { from { opacity: 0; } to { opacity: 1; } }",
        );
        sheet.keyframes.drain(..).next().expect("one rule")
    }

    /// Rec 11: progress comes from the monotonic clock, never
    /// from counting ticks — sampling at exact offsets lands on
    /// exact values however many ticks elapsed between samples.
    #[test]
    fn transition_samples_from_monotonic_time() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "opacity".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
            },
            AnimatedValue::Number(0.0),
            AnimatedValue::Number(1.0),
            start,
        );
        // One sample 250 ms in — not "16 ticks later".
        let out = engine.advance(start + Duration::from_millis(250));
        assert_eq!(out.changed.len(), 1);
        assert_eq!(out.finished_events.len(), 0);
        let (_, vals) = &out.changed[0];
        assert_eq!(vals.len(), 1);
        assert!((num(&vals[0].1) - 0.25).abs() < 1e-6);
        let out = engine.advance(start + Duration::from_millis(1000));
        assert_eq!(out.finished_events, vec![(id, "transitionend")]);
        assert!(engine.is_empty());
    }

    #[test]
    fn transition_delay_holds_from_value() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "opacity".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.5,
            },
            AnimatedValue::Number(0.0),
            AnimatedValue::Number(1.0),
            start,
        );
        // Still in the delay: nothing reported.
        let out = engine.advance(start + Duration::from_millis(250));
        assert!(out.changed.is_empty());
        // 250 ms into the 1 s run: 0.25.
        let out = engine.advance(start + Duration::from_millis(750));
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn transition_easing_shapes_progress() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "opacity".to_string(),
                duration: 1.0,
                timing: Easing::EaseIn,
                delay: 0.0,
            },
            AnimatedValue::Number(0.0),
            AnimatedValue::Number(1.0),
            start,
        );
        let out = engine.advance(start + Duration::from_millis(500));
        let (_, vals) = &out.changed[0];
        let v = num(&vals[0].1);
        assert!(v < 0.5, "ease-in at halfway is below 0.5: {v}");
        assert!(v > 0.0);
    }

    #[test]
    fn keyframe_animation_interpolates_and_ends() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        let frames = opacity_frames();
        let mut base = HashMap::new();
        base.insert("opacity".to_string(), AnimatedValue::Number(0.0));
        engine.start_animation(
            id,
            Animation {
                name: "fade".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
                iteration_count: IterationCount::Finite(1.0),
                direction: AnimationDirection::Normal,
                fill_mode: FillMode::None,
                play_state: pluto_css::PlayState::Running,
            },
            &frames,
            base,
            &ctx(),
            start,
        );
        let out = engine.advance(start + Duration::from_millis(500));
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.5).abs() < 1e-6);
        // Past the end with fill none: back to the base value,
        // plus the animationend event.
        let out = engine.advance(start + Duration::from_millis(1000));
        assert_eq!(out.finished_events, vec![(id, "animationend")]);
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.0).abs() < 1e-6);
        assert!(engine.is_empty());
    }

    #[test]
    fn keyframe_animation_fill_forwards_holds_last_value() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        let frames = opacity_frames();
        let mut base = HashMap::new();
        base.insert("opacity".to_string(), AnimatedValue::Number(0.0));
        engine.start_animation(
            id,
            Animation {
                name: "fade".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
                iteration_count: IterationCount::Finite(1.0),
                direction: AnimationDirection::Normal,
                fill_mode: FillMode::Forwards,
                play_state: pluto_css::PlayState::Running,
            },
            &frames,
            base,
            &ctx(),
            start,
        );
        let out = engine.advance(start + Duration::from_millis(1500));
        assert_eq!(out.finished_events, vec![(id, "animationend")]);
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn animation_alternate_reverses_second_iteration() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        let frames = opacity_frames();
        engine.start_animation(
            id,
            Animation {
                name: "fade".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
                iteration_count: IterationCount::Finite(2.0),
                direction: AnimationDirection::Alternate,
                fill_mode: FillMode::None,
                play_state: pluto_css::PlayState::Running,
            },
            &frames,
            HashMap::new(),
            &ctx(),
            start,
        );
        // 1.5 s in = halfway through the reversed second
        // iteration: 0.5 again, coming down.
        let out = engine.advance(start + Duration::from_millis(1500));
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn paused_engine_does_not_advance() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "opacity".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
            },
            AnimatedValue::Number(0.0),
            AnimatedValue::Number(1.0),
            start,
        );
        engine.pause(start + Duration::from_millis(100));
        // 900 ms of wall time pass while paused: the sample is
        // still the frozen 100 ms value.
        let out = engine.advance(start + Duration::from_millis(1000));
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.1).abs() < 1e-6);
        // Resume: the clock continues from 100 ms, not from 1000.
        engine.resume(start + Duration::from_millis(1000));
        let out = engine.advance(start + Duration::from_millis(1200));
        let (_, vals) = &out.changed[0];
        assert!((num(&vals[0].1) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn clear_removes_all_animations() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "opacity".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
            },
            AnimatedValue::Number(0.0),
            AnimatedValue::Number(1.0),
            start,
        );
        engine.clear();
        assert!(engine.is_empty());
        assert!(engine.advance(start + Duration::from_millis(500)).changed.is_empty());
    }

    #[test]
    fn transform_list_lerps_translate() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_transition(
            id,
            Transition {
                property: "transform".to_string(),
                duration: 1.0,
                timing: Easing::Linear,
                delay: 0.0,
            },
            AnimatedValue::Transform(vec![TransformFunction::TranslateX(CssLength::Px(0.0))]),
            AnimatedValue::Transform(vec![TransformFunction::TranslateX(CssLength::Px(100.0))]),
            start,
        );
        let out = engine.advance(start + Duration::from_millis(500));
        let (_, vals) = &out.changed[0];
        assert_eq!(vals[0].1.to_css(), "translateX(50px)");
    }

    #[test]
    fn waapi_animation_reports_finish_by_id() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_waapi(
            id,
            7,
            AnimTiming {
                duration: 1.0,
                delay: 0.0,
                timing: Easing::Linear,
                iteration_count: IterationCount::Finite(1.0),
                direction: AnimationDirection::Normal,
                fill_mode: FillMode::None,
            },
            vec![(
                0.0,
                vec![("opacity".to_string(), AnimatedValue::Number(0.0))],
            ), (
                1.0,
                vec![("opacity".to_string(), AnimatedValue::Number(1.0))],
            )],
            HashMap::new(),
            start,
        );
        assert_eq!(engine.waapi_ids(id), vec![7]);
        let out = engine.advance(start + Duration::from_millis(1000));
        assert_eq!(out.waapi_finished, vec![(id, 7)]);
        assert!(out.finished_events.is_empty());
        assert!(engine.cancel_waapi(id, 7).is_none());
    }

    fn waapi_timing(duration: f32) -> AnimTiming {
        AnimTiming {
            duration,
            delay: 0.0,
            timing: Easing::Linear,
            iteration_count: IterationCount::Finite(1.0),
            direction: AnimationDirection::Normal,
            fill_mode: FillMode::None,
        }
    }

    fn opacity_steps() -> Vec<(f32, Vec<(String, AnimatedValue)>)> {
        vec![
            (0.0, vec![("opacity".to_string(), AnimatedValue::Number(0.0))]),
            (1.0, vec![("opacity".to_string(), AnimatedValue::Number(1.0))]),
        ]
    }

    fn sampled_opacity(out: &EngineOutcome, id: DomId) -> Option<f32> {
        out.changed.iter().find(|(e, _)| *e == id).and_then(|(_, vals)| {
            vals.iter()
                .find(|(p, _)| p == "opacity")
                .map(|(_, v)| num(v))
        })
    }

    /// Rec 11: pausing a WAAPI animation freezes its progress at
    /// the pause point (later wall time changes nothing); resuming
    /// continues from there — the paused interval never counts.
    #[test]
    fn waapi_pause_freezes_and_resume_continues() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_waapi(id, 3, waapi_timing(1.0), opacity_steps(), HashMap::new(), start);
        let out = engine.advance(start + Duration::from_millis(500));
        assert!((sampled_opacity(&out, id).unwrap() - 0.5).abs() < 1e-6);
        assert!(engine.pause_waapi(id, 3, start + Duration::from_millis(500)));
        let out = engine.advance(start + Duration::from_millis(1400));
        assert!((sampled_opacity(&out, id).unwrap() - 0.5).abs() < 1e-6);
        assert!(engine.resume_waapi(id, 3, start + Duration::from_millis(1400)));
        let out = engine.advance(start + Duration::from_millis(1900));
        assert_eq!(out.waapi_finished, vec![(id, 3)]);
        assert!(engine.waapi_ids(id).is_empty());
    }

    /// Rec 11: `finish()` jumps to the end of the active interval
    /// — the next tick reports the finish (and the final value
    /// under a forwards fill, or the base under fill `none`).
    #[test]
    fn waapi_finish_jumps_to_end() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        engine.start_waapi(id, 9, waapi_timing(1.0), opacity_steps(), HashMap::new(), start);
        let out = engine.advance(start + Duration::from_millis(200));
        assert!((sampled_opacity(&out, id).unwrap() - 0.2).abs() < 1e-6);
        assert!(engine.finish_waapi(id, 9, start + Duration::from_millis(200)));
        let out = engine.advance(start + Duration::from_millis(210));
        assert_eq!(out.waapi_finished, vec![(id, 9)]);
        assert!(engine.waapi_ids(id).is_empty());
    }

    /// Rec 11: cancelling a WAAPI animation hands its underlying
    /// base values back, so the caller can restore them.
    #[test]
    fn waapi_cancel_returns_base_values() {
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        let mut base = HashMap::new();
        base.insert("opacity".to_string(), AnimatedValue::Number(0.3));
        engine.start_waapi(id, 5, waapi_timing(1.0), opacity_steps(), base, start);
        let base = engine.cancel_waapi(id, 5).expect("running");
        assert!((num(&base["opacity"]) - 0.3).abs() < 1e-6);
        assert!(engine.waapi_ids(id).is_empty());
        assert!(engine.cancel_waapi(id, 5).is_none());
    }

    /// Rec 11: CSS animations pause/resume/finish by name — same
    /// machine as WAAPI, keyed on the animation name instead.
    #[test]
    fn css_pause_resume_and_finish_by_name() {
        use pluto_css::{Animation, PlayState};
        let mut engine = AnimationEngine::new();
        let start = Instant::now();
        let id = test_dom_id();
        let frames = opacity_frames();
        let spec = Animation {
            name: "fade".to_string(),
            duration: 1.0,
            timing: Easing::Linear,
            delay: 0.0,
            iteration_count: IterationCount::Finite(1.0),
            direction: AnimationDirection::Normal,
            fill_mode: FillMode::None,
            play_state: PlayState::Running,
        };
        engine.start_animation(id, spec, &frames, HashMap::new(), &ctx(), start);
        assert_eq!(engine.running_css_animations(id), vec!["fade".to_string()]);
        let out = engine.advance(start + Duration::from_millis(400));
        assert!((sampled_opacity(&out, id).unwrap() - 0.4).abs() < 1e-6);
        assert!(engine.pause_css(id, "fade", start + Duration::from_millis(400)));
        let out = engine.advance(start + Duration::from_millis(900));
        assert!((sampled_opacity(&out, id).unwrap() - 0.4).abs() < 1e-6);
        assert!(engine.resume_css(id, "fade", start + Duration::from_millis(900)));
        assert!(engine.finish_css(id, "fade", start + Duration::from_millis(1000)));
        let out = engine.advance(start + Duration::from_millis(1010));
        assert_eq!(out.finished_events, vec![(id, "animationend")]);
        assert!(engine.running_css_animations(id).is_empty());
        assert!(!engine.pause_css(id, "fade", start + Duration::from_millis(1010)));
    }
}
