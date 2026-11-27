//! Layout & text (layer 4): taffy + cosmic-text bridge.
//!
//! M2 scope: a real selector matcher + cascade (right-to-left complex
//! selectors, rule indexing by key selector, origin / `!important` /
//! layer / specificity / source-order sorting, the parsed UA stylesheet
//! as the base sheet, inline `style=""`), a computed/used split
//! (`em`/`rem` resolve in style, `%`/`auto`/`vh`/`vw` resolve in
//! layout), box generation with anonymous block boxes and atomic
//! inline-level boxes, per-block-container inline layout with strut
//! line-height and `text-align` applied at line placement, pre-shaping
//! (option A: parallel pre-shape with per-thread `FontSystem`s producing
//! plain `Send` data, then a pure-arithmetic measure closure inside the
//! serial taffy pass), grid mapping, overflow/scroll containers with
//! retained offsets and display-list clipping, hit-testing, and
//! `border-radius` / `box-shadow` stored for Stage 8 paint.
//!
//! Parallelism boundary (Stage 6 report §d, verified against the
//! taffy 0.15.1 + cosmic-text 0.19.0 registry sources — see the
//! "Verified against registry sources" notes on the items below):
//! parallel style resolution → parallel pre-shaping → **serial** taffy
//! (`compute_layout_with_measure` takes `&mut self` + a plain `FnMut`)
//! → serial display-list build. There is no `Mutex<FontSystem>` in the
//! production layout path; a serialising `SharedFontSystem` exists only
//! for single-threaded/test paths (see `SharedFontSystem`).
//!
//! Verified against registry sources (2026-11-24, read from
//! `~/.cargo/registry/src/index.crates.io-*/`):
//! - cosmic-text 0.19.0 `src/font/system.rs`: `FontSystem::new()` calls
//!   `new_with_fonts(empty)`; `new_with_fonts(fonts: impl IntoIterator<
//!   Item = fontdb::Source>)` always calls `db.load_system_fonts()`
//!   first and then loads the given sources. So every per-thread
//!   `FontSystem` sees the same system fonts; a font loaded on the
//!   owner thread is visible to worker instances only if it is passed
//!   as a shared `fontdb::Source` (webfonts: not loaded anywhere yet —
//!   the hook is `FontSystem::new_with_fonts(shared_sources)`).
//! - cosmic-text 0.19.0 `src/shape_run_cache.rs` + `src/lib.rs:133`:
//!   `ShapeRunCache` / `ShapeRunKey` exist and are exported
//!   unconditionally, but the *use* of the cache inside shaping
//!   (`src/shape.rs`, `shape_run_cached`) is gated on the
//!   `shape-run-cache` feature, which Pluto does not enable — so the
//!   cache type exists but is inert here. Pre-shaping therefore keeps
//!   its own run cache keyed by (text hash, font attrs, size); see
//!   `preshape_leaves`.
//! - cosmic-text 0.19.0 `src/buffer.rs:571` `Buffer::shape_until_scroll`
//!   shapes/layouts lines up to `scroll.vertical + height_opt`
//!   (`height_opt` is `None` → `f32::INFINITY` when the buffer size
//!   was set with `None` height), so a buffer sized `(width, None)`
//!   shapes everything; `layout_runs` must only be read after it.
//! - taffy 0.15.1 `src/compute/leaf.rs` `compute_leaf_layout`: leaves
//!   return `top_margin`/`bottom_margin` =
//!   `CollapsibleMarginSet::ZERO` always, and
//!   `margins_can_collapse_through` is `false` in the early-return
//!   path and `!has_styles_preventing_being_collapsed_through &&
//!   size.height == 0.0 && measured_size.height == 0.0` in the general
//!   path — i.e. `false` for any non-empty text leaf. The report's
//!   "zero + false" holds for text leaves; `LayoutOutput::HIDDEN`
//!   (`src/tree/layout.rs:511`) is exactly that shape (zero size,
//!   zero margins, `margins_can_collapse_through: false`), which is
//!   what the measure closure returns for leaves.
//! - taffy 0.15.1 has no parallel entry point; `TaffyTree::mark_dirty`
//!   exists and is the incremental hook (see `LayoutTree::mark_dirty`).
//!
//! Style resolution runs as a two-phase pass before the taffy tree is
//! built (Stage 4 #5): phase 1 collects every element into
//! breadth-first levels from the root element, phase 2 resolves each
//! level's styles — in parallel on rayon's shared pool, parents before
//! children, so a level only ever reads styles resolved in earlier
//! levels — into an owned `DomId -> Arc<ComputedStyle>` map plus a
//! `SparseSecondaryMap<DomId, NodeStyle>` (the Servo `ElementData`
//! shape: computed style + restyle hint + restyle damage +
//! selector flags). M2 still does a full restyle on every build; the
//! hint/damage fields are populated so Stage 7's incremental path has
//! the data. The `Builder` then looks styles up from that map instead
//! of resolving them inline. The same level loop can run sequentially
//! (`resolve_document_styles(..., parallel: false)`), so the two paths
//! stay comparable and a serial fallback is real.
//!
//! Quirks mode: `pluto-html` has no `quirks_mode` flag yet (checked
//! 2026-11-24: no `quirks`/`dirty` fields on `Dom`), so matching
//! assumes no-quirks (exact case-sensitive class/id matching). When
//! the parser agent adds `Dom::quirks_mode`, the matcher is the one
//! place to read it — flagged as a follow-up, not worked around here.
//! Likewise `Dom` has no dirty bits yet, so R20's Double Dirty Bit is
//! data-only on the layout side (`LayoutTree::mark_dirty` drives
//! taffy's own dirty flag); consuming DOM dirty bits is a follow-up
//! once `pluto-html` grows them — no fields were added to `pluto-html`.
//!
//! Stage 8 Recommendation 6 — dirty-subtree layout v1 (see
//! `incremental`): `LayoutTree::relayout_incremental` consumes the
//! DOM dirty set Stage 7 populated, classifies the damage
//! (paint-only / position-only / layout / reconstruct), re-lays out
//! only a proven-safe subtree when it can (definite width — the
//! height may be auto — `contain: layout`, replaced element, or
//! fixed-size leaf, with no float intrusion or percentage
//! resolution escaping the root; margin collapse is recomputed by
//! taffy for the dirty ancestors, so it needs no predicate guard),
//! translates position-only moves without touching taffy, and
//! falls back to a full rebuild for anything unsafe. In debug
//! builds every incremental result is asserted equal to a cold
//! full layout. Style re-resolution is subtree-only since Stage 12
//! Rec 6 (the dirty LCA's subtree, seeded from the retained
//! styles above it); the whole-document pass runs only for cold
//! builds.

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping, Weight};
use pluto_css::{
    Affine, AnPlusB, Animation, AnimationDirection, AttrOp, BorderRadius, BoxShadow, CalcCtx,
    CalcExpr, CalcNumber, CascadeKey, Combinator, Easing, FillMode, Gradient, IterationCount,
    Keyframes, Origin, PlayState, PseudoClass, ResolvedCorners, Rule, Selector, SimpleSelector,
    Stylesheet, SupportsCondition, TransformList, TransformListExt, Transition, Value,
    parse_value,
};
use pluto_html::{Dom, NodeId as DomId};
use rayon::prelude::*;
use slotmap::Key as _;
use slotmap::SparseSecondaryMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use taffy::{
    AvailableSpace, BoxSizing, Dimension, Display, FlexDirection, FlexWrap, GridAutoFlow,
    GridTemplateComponent, LayoutInput, LayoutOutput, LengthPercentage, LengthPercentageAuto,
    Line, MaxTrackSizingFunction, MinTrackSizingFunction, NodeId as TaffyId, Overflow,
    Position, Size, Style, TaffyError, TaffyTree, TrackSizingFunction,
};
use taffy::GridPlacement;
use taffy::prelude::{line as grid_line, span as grid_span};

/// Stage 8 Recommendation 6 — dirty-subtree layout v1: damage
/// classification, the safe-root predicate, the incremental relayout
/// entry point, and the position-only fast path.
mod incremental;

/// M6 Rec 11 — the animation engine: CSS transitions, keyframe
/// animations, and the `Element.animate` / `getAnimations` subset.
pub mod animation;

pub use incremental::{
    classify_damage, classify_style_change, is_safe_layout_root, unsafe_layout_root_reason,
    DamageKind, RelayoutKind, RelayoutOutcome,
};

/// Re-exported for callers that build trees: the theme a layout
/// pass evaluates `prefers-color-scheme` against (M6 Rec 12).
pub use pluto_css::Theme;

/// One styled run inside a text leaf.
#[derive(Debug, Clone)]
pub struct TextSpan {
    pub text: String,
    pub color: [u8; 3],
    pub bold: bool,
}

/// One word's advance, from pre-shaping: the width of the word's
/// glyphs, plus whether a space follows it (its width is
/// `ShapedText::space_width`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordAdvance {
    pub advance: f32,
    pub trailing_space: bool,
}

/// Plain `Send` data for one text run, produced by pre-shaping before
/// taffy runs. The measure closure is pure arithmetic over this: it
/// wraps words at the available width and answers min/max-content from
/// cached advances. No cosmic-text call happens inside measure.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapedText {
    /// Words per source line (`\n`-separated segment).
    pub segments: Vec<Vec<WordAdvance>>,
    pub space_width: f32,
    pub min_content: f32,
    pub max_content: f32,
    pub line_height: f32,
}

/// The text of one taffy leaf: merged spans of a block's inline content.
#[derive(Debug, Clone)]
pub struct LeafText {
    pub spans: Vec<TextSpan>,
    pub font_size: f32,
    /// Computed line height in px (from `line-height`).
    pub line_height: f32,
    /// Pre-shaped data, filled in by `preshape_leaves` before taffy
    /// runs. `None` only before that pass (or for leaves created by
    /// tests directly).
    pub shaped: Option<Arc<ShapedText>>,
}

/// Taffy node context: `Some` for text leaves, `None` for elements.
pub type NodeContext = Option<LeafText>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dim {
    Px(f32),
    /// Fraction of the containing block (0.0..1.0).
    Pct(f32),
    /// Fraction of the viewport height — a *used* value, resolved in
    /// `to_taffy_style` against the viewport.
    Vh(f32),
    /// Fraction of the viewport width — a *used* value.
    Vw(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spacing {
    Px(f32),
    /// Fraction of the containing block width (0.0..1.0).
    Pct(f32),
    /// Fraction of the viewport height — resolved in `to_taffy_style`.
    Vh(f32),
    /// Fraction of the viewport width — resolved in `to_taffy_style`.
    Vw(f32),
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayKind {
    Block,
    Flex,
    Grid,
    Inline,
    InlineBlock,
    FlowRoot,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowKind {
    Visible,
    Hidden,
    Scroll,
    Auto,
    Clip,
}

impl OverflowKind {
    /// `overflow` values that make the box a scroll container (and a
    /// BFC root). `clip` clips but creates neither.
    pub fn is_scroll_container(self) -> bool {
        matches!(self, OverflowKind::Hidden | OverflowKind::Scroll | OverflowKind::Auto)
    }
}

/// `overscroll-behavior` (Stage 8 Rec 9): whether a scroll that
/// reaches this container's limit chains to the parent scroller
/// (and eventually the viewport). `auto` chains (the default);
/// `contain` and `none` stop the chain at this container — the
/// leftover delta is dropped, not passed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverscrollBehavior {
    #[default]
    Auto,
    Contain,
    None,
}

impl OverscrollBehavior {
    /// Whether a scroll reaching this container's limit stops
    /// here instead of chaining to the parent scroller.
    pub fn stops_chaining(self) -> bool {
        matches!(self, OverscrollBehavior::Contain | OverscrollBehavior::None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    /// M2 renders `justify` as `left` (no justification engine yet).
    Justify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhiteSpace {
    Normal,
    Pre,
    Nowrap,
    PreWrap,
    PreLine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionKind {
    Static,
    Relative,
    Absolute,
    /// Viewport-relative; taffy has no `fixed`, so it is laid out as
    /// absolute and the shell must re-anchor it to the viewport.
    Fixed,
    /// No sticky support in taffy; laid out as relative.
    Sticky,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxSizingKind {
    ContentBox,
    BorderBox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// One grid track from `grid-template-columns` / `-rows`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridTrack {
    Fixed(Dim),
    Fraction(f32),
    Auto,
}

/// One end of a `grid-column` / `grid-row` placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlace {
    Auto,
    Line(i16),
    Span(u16),
}

/// The computed style of one element: the cascade winner for every
/// property, with relative units resolved as far as style resolution
/// can take them — `em` against the element's own computed font size,
/// A paint server for SVG `fill` / `stroke` (Stage 15, M7 Phase 1,
/// Rec 6): a solid colour, none, a `url(...)` reference to a
/// `<linearGradient>` / `<radialGradient>` / `<pattern>` (paint
/// resolves the reference in a later phase — gradients are Rec 7),
/// or `currentColor`, resolved against the element's computed `color`
/// after the cascade (see `resolve_style`'s post-pass).
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    /// No paint (`fill: none` / `stroke: none`).
    None,
    /// A solid RGB colour.
    Color([u8; 3]),
    /// A `url(...)` paint-server reference, as written.
    Url(String),
    /// `currentColor` — resolved post-cascade.
    CurrentColor,
}

/// `rem` against the root font size, `vh`/`vw` left as used values for
/// layout. `%`/`auto` are likewise left for layout. This is the
/// computed/used split (R8): nothing here depends on layout geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputedStyle {
    pub display: DisplayKind,
    pub width: Option<Dim>,
    pub height: Option<Dim>,
    pub min_width: Option<Dim>,
    pub max_width: Option<Dim>,
    pub min_height: Option<Dim>,
    pub max_height: Option<Dim>,
    pub margin: [Spacing; 4],  // top, right, bottom, left
    pub padding: [Spacing; 4], // top, right, bottom, left
    /// `top` / `right` / `bottom` / `left` offsets (`None` = auto).
    pub inset: [Option<Dim>; 4],
    pub font_size: f32,
    /// Computed line height in px (already resolved: number and `%`
    /// are multiples of the font size, `normal` is 1.2x).
    pub line_height: f32,
    pub font_family: String,
    pub font_style: FontStyle,
    pub color: [u8; 3],
    pub background: Option<[u8; 3]>,
    /// SVG `fill` (Stage 15, M7 Phase 1, Rec 6): inherited,
    /// initial black. Paint reads it; layout only carries it.
    pub fill: Paint,
    /// SVG `stroke` (Rec 6): inherited, initial none.
    pub stroke: Paint,
    /// SVG `stroke-width` in px (Rec 6): inherited, initial 1.0.
    /// Percentages (relative to the viewport diagonal) are not
    /// resolved here — they keep the previous value.
    pub stroke_width: f32,
    /// SVG `stop-color` (Rec 6): inherited, initial black. Used
    /// by gradient stops (Rec 7); stored now so the cascade and
    /// presentation attributes have somewhere to land.
    pub stop_color: [u8; 3],
    pub bold: bool,
    pub text_align: TextAlign,
    pub white_space: WhiteSpace,
    pub overflow_x: OverflowKind,
    pub overflow_y: OverflowKind,
    /// `overscroll-behavior-x` / `-y` (Stage 8 Rec 9): whether a
    /// scroll reaching this container's limit chains onward.
    pub overscroll_behavior_x: OverscrollBehavior,
    pub overscroll_behavior_y: OverscrollBehavior,
    pub position: PositionKind,
    pub box_sizing: BoxSizingKind,
    /// `border-radius` corners in px (percentages resolve against the
    /// box at paint time, so they are kept as `CssLength` here).
    pub border_radius: Option<BorderRadius>,
    /// `box-shadow` layers, kept as parsed; geometry-neutral.
    pub box_shadow: Vec<BoxShadow>,
    pub flex_direction: FlexDirection,
    pub flex_wrap: FlexWrap,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: Option<Dim>,
    pub grid_template_columns: Vec<GridTrack>,
    pub grid_template_rows: Vec<GridTrack>,
    pub grid_gap: (Dim, Dim), // (row, column)
    /// (start, end) placement per axis; `Auto` = auto-placement.
    pub grid_column: (GridPlace, GridPlace),
    pub grid_row: (GridPlace, GridPlace),
    /// `grid-auto-flow` (Stage 10 Rec 5): row/column auto-placement
    /// axis plus the dense packing algorithm. Initial `Row`, which
    /// is also what taffy defaults to.
    pub grid_auto_flow: GridAutoFlow,
    /// `contain: layout` (also implied by `contain: strict` /
    /// `content`): the subtree is layout-contained, so it can be a
    /// layout root even without a definite size (Stage 8 Rec 6).
    /// `contain: size` alone does NOT set this.
    pub contain_layout: bool,
    /// `background-image` URL as written in the CSS (Stage 11
    /// Rec 3): the raw `url(...)` text, unresolved — the worker
    /// resolves it against the document base and looks the decoded
    /// resource up in the [`ImageMap`].
    pub background_image: Option<String>,
    /// A CSS gradient `background-image` (Stage 15, M7 Phase 2,
    /// Rec 7): the parsed gradient. `background-image` is one
    /// property, so this and `background_image` are mutually
    /// exclusive — each arm that sets one clears the other (see
    /// `apply_decl`).
    pub background_gradient: Option<Gradient>,
    /// `background-size` (Stage 11 Rec 3).
    pub background_size: BackgroundSize,
    /// `background-repeat` (Stage 11 Rec 3; two-value form M6).
    pub background_repeat: BackgroundRepeat,
    /// `background-position` `(x, y)` (Stage 11 Rec 3).
    pub background_position: (BackgroundPosition, BackgroundPosition),
    /// `z-index`: `None` = `auto` (no explicit stacking level).
    pub z_index: Option<i32>,
    /// `opacity`, clamped to `0..=1` (initial 1.0).
    pub opacity: f32,
    /// `transform` function list (initial `none` = empty).
    pub transform: TransformList,
    /// `transform-origin` (initial `50% 50%`). Percentages
    /// resolve against the border box at box-build time, never
    /// here.
    pub transform_origin: TransformOrigin,
    /// `transform-box` (initial `view-box`; only view-box
    /// semantics are implemented — the value is stored so the
    /// later paint phase can read it).
    pub transform_box: TransformBox,
    /// `will-change: transform` was declared. A hint only: it
    /// does NOT create a stacking context (see
    /// [`is_stacking_context`]).
    pub will_change_transform: bool,
    /// `transition` entries, in author order.
    pub transitions: Vec<Transition>,
    /// `animation` entries, in author order.
    pub animations: Vec<Animation>,
    /// Custom properties (`--*`) for this element: its own
    /// declarations over the inherited chain (see `resolve_style`).
    pub custom_properties: HashMap<String, Value>,
}

impl ComputedStyle {
    /// Initial values (CSS initial + the inherited-value carriers a
    /// root element starts from). `display` is `Inline`: the spec
    /// default for an element the UA stylesheet does not mention.
    pub fn initial() -> Self {
        ComputedStyle {
            display: DisplayKind::Inline,
            width: None,
            height: None,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            margin: [Spacing::Px(0.0); 4],
            padding: [Spacing::Px(0.0); 4],
            inset: [None; 4],
            font_size: 16.0,
            line_height: 16.0 * 1.2,
            font_family: "sans-serif".to_string(),
            font_style: FontStyle::Normal,
            color: [0, 0, 0],
            background: None,
            fill: Paint::Color([0, 0, 0]),
            stroke: Paint::None,
            stroke_width: 1.0,
            stop_color: [0, 0, 0],
            bold: false,
            text_align: TextAlign::Left,
            white_space: WhiteSpace::Normal,
            overflow_x: OverflowKind::Visible,
            overflow_y: OverflowKind::Visible,
            overscroll_behavior_x: OverscrollBehavior::Auto,
            overscroll_behavior_y: OverscrollBehavior::Auto,
            position: PositionKind::Static,
            box_sizing: BoxSizingKind::ContentBox,
            border_radius: None,
            box_shadow: Vec::new(),
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::NoWrap,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: None,
            grid_template_columns: Vec::new(),
            grid_template_rows: Vec::new(),
            grid_gap: (Dim::Px(0.0), Dim::Px(0.0)),
            grid_column: (GridPlace::Auto, GridPlace::Auto),
            grid_row: (GridPlace::Auto, GridPlace::Auto),
            grid_auto_flow: GridAutoFlow::Row,
            contain_layout: false,
            background_image: None,
            background_gradient: None,
            background_size: BackgroundSize::Auto,
            background_repeat: BackgroundRepeat::Repeat,
            background_position: (BackgroundPosition::Start, BackgroundPosition::Start),
            z_index: None,
            opacity: 1.0,
            transform: Vec::new(),
            transform_origin: TransformOrigin {
                x: OriginComponent::Percentage(50.0),
                y: OriginComponent::Percentage(50.0),
            },
            transform_box: TransformBox::ViewBox,
            will_change_transform: false,
            transitions: Vec::new(),
            animations: Vec::new(),
            custom_properties: HashMap::new(),
        }
    }
}

/// `transform-box`: which box the transform reference box is.
/// Only `view-box` semantics are implemented (the initial value);
/// the other values are stored so a later phase can honour them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformBox {
    ContentBox,
    BorderBox,
    FillBox,
    StrokeBox,
    ViewBox,
}

/// One axis of `transform-origin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OriginComponent {
    Length(Dim),
    Percentage(f32),
    Keyword(OriginKeyword),
}

/// A `transform-origin` keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginKeyword {
    Left,
    Center,
    Right,
    Top,
    Bottom,
}

/// `transform-origin`: the pivot a transform rotates / scales
/// about, resolved against the border box at box-build time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformOrigin {
    pub x: OriginComponent,
    pub y: OriginComponent,
}

impl TransformOrigin {
    /// Resolve one axis to used pixels against a border-box side
    /// of `size` (width for x, height for y).
    pub fn resolve_axis(component: OriginComponent, size: f32, viewport: (f32, f32)) -> f32 {
        match component {
            OriginComponent::Length(d) => match d {
                Dim::Px(v) => v,
                Dim::Pct(f) => f * size,
                Dim::Vh(v) => v / 100.0 * viewport.1,
                Dim::Vw(v) => v / 100.0 * viewport.0,
            },
            OriginComponent::Percentage(p) => p / 100.0 * size,
            OriginComponent::Keyword(k) => match k {
                OriginKeyword::Left | OriginKeyword::Top => 0.0,
                OriginKeyword::Center => size / 2.0,
                OriginKeyword::Right | OriginKeyword::Bottom => size,
            },
        }
    }

    /// Resolve both axes against a `width` × `height` border box.
    pub fn resolve(self, width: f32, height: f32, viewport: (f32, f32)) -> (f32, f32) {
        (
            Self::resolve_axis(self.x, width, viewport),
            Self::resolve_axis(self.y, height, viewport),
        )
    }
}

/// Whether `style` makes its box a stacking context (M6 Rec 5,
/// Phase 1 model only — paint order is a later phase).
///
/// Explicit: `position != static` with a non-`auto` `z-index`,
/// or a flex/grid item with a non-`auto` `z-index`.
/// Implicit: `opacity < 1.0`, or a non-`none` transform.
///
/// Deliberately NOT in the predicate: `will-change` (a hint
/// only — it must not create a context), and `isolation` /
/// `filter` / `mix-blend-mode`, which are unmodelled in this
/// phase, so they cannot create one either.
pub fn is_stacking_context(style: &ComputedStyle, is_flex_or_grid_item: bool) -> bool {
    if style.z_index.is_some()
        && (style.position != PositionKind::Static || is_flex_or_grid_item)
    {
        return true;
    }
    if style.opacity < 1.0 {
        return true;
    }
    if !style.transform.is_empty() {
        return true;
    }
    false
}

/// `background-size` (Stage 11 Rec 3): `cover` / `contain` scale
/// to the box; `Lengths` is explicit `(width, height)` where
/// `None` on one axis means "auto" (scale by the intrinsic
/// ratio); `Auto` (both auto) is the intrinsic size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackgroundSize {
    Cover,
    Contain,
    Lengths(Option<Dim>, Option<Dim>),
    Auto,
}

/// `background-repeat` (Stage 11 Rec 3; two-value form M6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundRepeat {
    Repeat,
    NoRepeat,
    RepeatX,
    RepeatY,
    /// Two-value form: `(repeat_x, repeat_y)` — e.g.
    /// `repeat no-repeat` tiles x only.
    RepeatXY(bool, bool),
}

/// One axis of `background-position` (Stage 11 Rec 3): keyword
/// (`left`/`center`/`right`, `top`/`center`/`bottom`) or an
/// explicit offset. Initial is `Start` on both axes (`0% 0%`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackgroundPosition {
    Start,
    Center,
    End,
    Length(Dim),
}

/// One decoded image, as layout sees it (Stage 11, M3): the
/// resource id — a plain `u32`, because layout must not depend
/// on pluto-paint (paint re-wraps it as `ResourceId`) — plus the
/// intrinsic size the HTML sizing rules need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    pub resource: u32,
    pub width: u32,
    pub height: u32,
}

/// One video element's decoded state, as layout sees it
/// (Stage 16 Phase 2, Rec 5): the intrinsic size once
/// `HAVE_METADATA` (from the demuxer), and whether a frame is
/// currently presentable (a decoded frame has been presented
/// for this element — paint then draws the frame Blob instead
/// of the poster).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub has_frame: bool,
}

/// Images the worker has decoded for this document (Stage 11,
/// M3): per-`<img>` node (intrinsic size + resource id), the
/// nodes whose image is known-broken (alt text renders instead),
/// and background-image URLs (raw CSS url → intrinsic size +
/// resource id, so paint can place `cover` / `contain`).
/// `videos` (Stage 16 Phase 2, Rec 5) carries the same for
/// `<video>` elements: intrinsic size + frame liveness. The
/// poster image itself rides `by_node` (the video node maps to
/// its poster resource, like an `<img>`), so paint needs no
/// second lookup for it.
#[derive(Debug, Clone, Default)]
pub struct ImageMap {
    pub by_node: HashMap<DomId, ImageInfo>,
    pub broken: HashSet<DomId>,
    pub backgrounds: HashMap<String, ImageInfo>,
    pub videos: HashMap<DomId, VideoInfo>,
}

impl ImageMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nodes whose per-`<img>` entry differs from `old` — a new
    /// image, a changed intrinsic size, a newly-broken (or
    /// newly-fixed) image, or a vanished entry. An image arrival
    /// dirties exactly these nodes for layout (Stage 12 Rec 5):
    /// the incremental path takes them as extra damage, so an
    /// image landing inside a safe subtree re-lays out that
    /// subtree instead of forcing a full rebuild.
    pub fn changed_node_ids(&self, old: &ImageMap) -> Vec<DomId> {
        let mut out: Vec<DomId> = Vec::new();
        for (&id, info) in &self.by_node {
            if old.by_node.get(&id) != Some(info) && !out.contains(&id) {
                out.push(id);
            }
        }
        for &id in old.by_node.keys() {
            if !self.by_node.contains_key(&id) && !out.contains(&id) {
                out.push(id);
            }
        }
        for &id in self.broken.symmetric_difference(&old.broken) {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        // Stage 16 Phase 2 (Rec 5): a video whose metadata
        // arrived (or whose frame liveness flipped) dirties
        // its node the same way an image arrival does.
        for (&id, info) in &self.videos {
            if old.videos.get(&id) != Some(info) && !out.contains(&id) {
                out.push(id);
            }
        }
        for &id in old.videos.keys() {
            if !self.videos.contains_key(&id) && !out.contains(&id) {
                out.push(id);
            }
        }
        out
    }

    /// Whether any background-image entry differs from `old`.
    /// Backgrounds are keyed by URL, not node, so a change cannot
    /// name the boxes that use it — the caller must full-rebuild
    /// (the boxes carry the resource id + intrinsic size from the
    /// map they were built with).
    pub fn backgrounds_changed(&self, old: &ImageMap) -> bool {
        self.backgrounds != old.backgrounds
    }
}

/// What restyling a node needs (Servo `RestyleHint` shape). M2
/// populates these; the traversal still does a full restyle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestyleHint {
    pub restyle_self: bool,
    pub restyle_subtree: bool,
    pub restyle_descendants: bool,
    pub restyle_later_siblings: bool,
}

impl RestyleHint {
    /// M2's full-restyle hint: everything, every time.
    pub fn full() -> Self {
        RestyleHint {
            restyle_self: true,
            restyle_subtree: true,
            restyle_descendants: true,
            restyle_later_siblings: true,
        }
    }
}

/// What layout must redo after restyling (Servo `RestyleDamage` shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestyleDamage {
    pub repaint: bool,
    pub reflow: bool,
    pub reconstruct: bool,
}

impl RestyleDamage {
    /// M2's full-restyle damage: repaint + reflow, no reconstruction.
    pub fn full() -> Self {
        RestyleDamage {
            repaint: true,
            reflow: true,
            reconstruct: false,
        }
    }
}

/// Which selector features matched this element — the data a future
/// invalidation map will read (R19). Computed during matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SelectorFlags {
    pub has_dynamic_pseudo: bool,
    pub has_sibling_combinator: bool,
    pub has_descendant_combinator: bool,
}

/// Per-node style data (Servo `ElementData` shape), hung off the DOM
/// in a secondary map so the hot `Node` stays small and `Arc` sharing
/// keeps the rayon style pass sound.
#[derive(Debug, Clone)]
pub struct NodeStyle {
    pub computed: Arc<ComputedStyle>,
    pub hint: RestyleHint,
    pub damage: RestyleDamage,
    pub selector_flags: SelectorFlags,
}

/// Dynamic pseudo-class state for one style/layout pass: which
/// elements are currently hovered / active / focused, and which
/// links point at URLs the session has visited. The shell owns
/// this (hit-testing feeds `:hover`; the worker's committed-URL
/// set feeds `:visited` — see `TabState::visited_urls` in
/// pluto-browser, which projects the set onto element ids before
/// each layout pass).
#[derive(Debug, Clone, Default)]
pub struct ElementState {
    pub hover: HashSet<DomId>,
    pub active: HashSet<DomId>,
    pub focus: HashSet<DomId>,
    /// Links whose resolved `href` is in the session's visited
    /// set (worker-side projection — the matcher itself never
    /// sees a URL, only element ids, so visited-ness cannot leak
    /// through a layout-dependent query).
    pub visited: HashSet<DomId>,
    /// Form pseudo-class projections (Stage 13 Rec 8): the worker
    /// projects the `Dom` form state onto element ids before each
    /// style pass, exactly like `:visited` — the matcher never
    /// reads a control's value itself.
    pub invalid: HashSet<DomId>,
    pub required: HashSet<DomId>,
    pub disabled: HashSet<DomId>,
    pub checked: HashSet<DomId>,
    pub read_only: HashSet<DomId>,
}

impl ElementState {
    pub fn with_hover(mut self, id: DomId) -> Self {
        self.hover.insert(id);
        self
    }

    pub fn with_active(mut self, id: DomId) -> Self {
        self.active.insert(id);
        self
    }

    pub fn with_focus(mut self, id: DomId) -> Self {
        self.focus.insert(id);
        self
    }

    pub fn with_visited(mut self, id: DomId) -> Self {
        self.visited.insert(id);
        self
    }

    pub fn with_invalid(mut self, id: DomId) -> Self {
        self.invalid.insert(id);
        self
    }

    pub fn with_required(mut self, id: DomId) -> Self {
        self.required.insert(id);
        self
    }

    pub fn with_disabled(mut self, id: DomId) -> Self {
        self.disabled.insert(id);
        self
    }

    pub fn with_checked(mut self, id: DomId) -> Self {
        self.checked.insert(id);
        self
    }

    pub fn with_read_only(mut self, id: DomId) -> Self {
        self.read_only.insert(id);
        self
    }
}

/// One shaped glyph, in page coordinates (origin = glyph origin, i.e.
/// the point `SwashImage.placement` is relative to).
#[derive(Debug, Clone, Copy)]
pub struct ShapedGlyph {
    pub cache_key: cosmic_text::CacheKey,
    pub x: f32,
    pub y: f32,
    pub color: [u8; 3],
}

/// What kind of replaced-element content a [`LayoutBox`] carries
/// (Stage 15, M7 Phase 1, Rec 2). Deliberately separate from
/// [`LayoutBox::image`]: `image` is a decoded-image resource id the
/// paint phase looks up, while this only marks "this box is an
/// `<svg>` / `<canvas>` leaf" so paint (a later phase) can find the
/// boxes whose content it must rasterise — reusing `image` would
/// conflate the two and send paint looking up a resource id that
/// does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvgCanvasKind {
    /// An `<svg>` element (SVG namespace).
    Svg,
    /// A `<canvas>` element (HTML namespace).
    Canvas,
    /// A `<video>` element (HTML namespace; Stage 16 Phase 2,
    /// Rec 5): a replaced-element leaf like canvas, sized by
    /// the decoded intrinsic size (300×150 default) — paint
    /// draws the current frame Blob, the poster, or black.
    Video,
}

/// One laid-out box: an element or a text leaf, in page coordinates.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub taffy: TaffyId,
    /// `None` for anonymous text leaves (merged inline content).
    pub dom: Option<DomId>,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub style: ComputedStyle,
    pub text: Option<LeafText>,
    /// Shaped glyphs for text leaves (empty otherwise).
    pub glyphs: Vec<ShapedGlyph>,
    /// Effective clip rect `[x, y, w, h]` in page coordinates, from
    /// the nearest clipping ancestor (`None` = unclipped). Any
    /// `overflow` other than `visible` clips (hidden/scroll/auto
    /// clip to the padding box; `clip` clips and forbids scrolling,
    /// so it creates no scroll container). The display list turns
    /// clip changes into PushClip/PopClip.
    pub clip: Option<[f32; 4]>,
    /// The taffy node of the nearest clipping ancestor that produced
    /// `clip` (`None` when unclipped). The position-only fast path
    /// (Stage 8 Rec 6) moves a clip rect only when its source moves
    /// with the translated subtree.
    pub clip_source: Option<TaffyId>,
    /// Resolved corner radii of the clip rect's own box (`None` =
    /// square clip): the nearest clipping ancestor's
    /// `border-radius`, resolved against that ancestor's box. The
    /// renderer applies these as a rounded-rect clip; with no
    /// borders modelled, the padding box and the border box are the
    /// same rect, so the radii need no border-width adjustment.
    pub clip_radii: Option<ResolvedCorners>,
    /// This box's own resolved `border-radius` (zero when unset):
    /// the background fill and the box-shadow shape are drawn with
    /// these radii.
    pub radii: ResolvedCorners,
    /// Nearest scroll-container ancestor (`None` = the viewport).
    pub scroll_ancestor: Option<DomId>,
    /// Decoded `<img>` resource id for this box (`None` when the
    /// element is not an `<img>` or its image has not arrived yet),
    /// from the [`ImageMap`] the tree was built with (Stage 11,
    /// M3). The id is a plain `u32` — paint re-wraps it as
    /// `ResourceId`.
    pub image: Option<u32>,
    /// Background-image resource id for this box, plus the
    /// intrinsic size paint needs for `cover` / `contain` /
    /// `no-repeat` placement (Stage 11 Rec 3). `None` when the
    /// element has no background image or it has not arrived yet.
    pub background_image: Option<u32>,
    pub background_image_size: Option<(u32, u32)>,
    /// `<svg>` / `<canvas>` content marker (Stage 15, M7 Phase 1,
    /// Rec 2): `Some(Svg)` / `Some(Canvas)` exactly when this box
    /// is such an element's atomic leaf, `None` otherwise. Paint
    /// reads this to find the boxes whose vector / bitmap content
    /// it must produce; layout sizes the leaf, it does not draw it.
    /// `Some(Video)` (Stage 16 Phase 2, Rec 5) is the same marker
    /// for a `<video>` leaf — paint draws the frame Blob, the
    /// poster, or black; `video_live` says which.
    pub svg_canvas: Option<SvgCanvasKind>,
    /// True when this `<video>` box has a presentable frame
    /// (Stage 16 Phase 2, Rec 5): from the [`ImageMap`] the tree
    /// was built with. Paint draws the frame Blob when true, the
    /// poster (`image`) when false.
    pub video_live: bool,
    /// The DOM id of the nearest ancestor box (`None` for the
    /// root box).
    pub parent: Option<DomId>,
    /// The accumulated transform mapping this box's local
    /// coordinates to page coordinates: the parent box's matrix
    /// composed with this box's own origin-resolved transform
    /// (`T(origin) * M * T(-origin)`). Identity when nothing in
    /// the ancestor chain (or this box) is transformed. This is
    /// the transform part only — the box's layout offset lives
    /// in `x`/`y`, and paint composes the two.
    pub matrix: Affine,
    /// Whether this box is a stacking context (see
    /// [`is_stacking_context`]). Model only in this phase —
    /// paint order is a later phase.
    pub is_stacking_context: bool,
    /// Whether this box is viewport-fixed (M6 Phase 3, Rec 1):
    /// `position: fixed` with no transformed ancestor, so it
    /// resolves against the viewport — not against the nearest
    /// positioned ancestor, and not against any scroll container.
    /// Its `x`/`y` are viewport coordinates (insets resolved
    /// against the viewport); paint must exempt it from the
    /// viewport scroll subtraction (see `PushFixed`), and
    /// hit-testing tests it in viewport coordinates.
    pub is_viewport_fixed: bool,
}

#[derive(Debug, Clone)]
struct BoxMeta {
    dom: Option<DomId>,
    style: ComputedStyle,
    children: Vec<TaffyId>,
    text: Option<LeafText>,
    /// `<svg>` / `<canvas>` marker, copied onto the [`LayoutBox`]
    /// in `collect_boxes` (Stage 15, M7 Phase 1, Rec 2).
    svg_canvas: Option<SvgCanvasKind>,
}

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("taffy error: {0}")]
    Taffy(#[from] TaffyError),
    #[error("document has no root element")]
    NoRoot,
}

/// Whether `style` (plus `is_root`) makes its box a block formatting
/// context root (R14). Taffy's block algorithm then handles the
/// geometry; Pluto must not emulate BFCs with padding or wrappers.
/// Note: flex/grid *items* are also BFC roots, but that needs the
/// parent's display, which this predicate does not see — the builder
/// treats flex/grid children as BFC roots by construction (they are
/// laid out by the flex/grid algorithm, not the block algorithm).
pub fn is_bfc_root(style: &ComputedStyle, is_root: bool) -> bool {
    is_root
        || style.display == DisplayKind::FlowRoot
        || style.display == DisplayKind::InlineBlock
        || matches!(
            style.position,
            PositionKind::Absolute | PositionKind::Fixed
        )
        || style.overflow_x.is_scroll_container()
        || style.overflow_y.is_scroll_container()
}

/// True when `id` is an `<svg>` element in the SVG namespace
/// (Stage 15, M7 Phase 1, Rec 2): tag spelling alone is not enough —
/// an unknown HTML `<svg>` (no namespace) is not an SVG element.
pub fn is_svg_element(dom: &Dom, id: DomId) -> bool {
    dom.tag(id) == Some("svg")
        && dom.namespace(id).is_some_and(|ns| ns == pluto_html::NS_SVG)
}

/// True when `id` is a `<canvas>` element in the HTML namespace
/// (Stage 15, M7 Phase 1, Rec 2).
pub fn is_canvas_element(dom: &Dom, id: DomId) -> bool {
    dom.tag(id) == Some("canvas")
        && dom
            .namespace(id)
            .is_some_and(|ns| ns.is_empty() || ns == pluto_html::NS_HTML)
}

/// True when `id` is a `<video>` element in the HTML namespace
/// (Stage 16 Phase 2, Rec 5).
pub fn is_video_element(dom: &Dom, id: DomId) -> bool {
    dom.tag(id) == Some("video")
        && dom
            .namespace(id)
            .is_some_and(|ns| ns.is_empty() || ns == pluto_html::NS_HTML)
}

/// True when `id` is an `<audio>` element in the HTML namespace
/// (Stage 16 Phase 2, Rec 5).
pub fn is_audio_element(dom: &Dom, id: DomId) -> bool {
    dom.tag(id) == Some("audio")
        && dom
            .namespace(id)
            .is_some_and(|ns| ns.is_empty() || ns == pluto_html::NS_HTML)
}

/// Parse an SVG `width` / `height` / `canvas` attribute: a bare
/// number, or a number with a `px` suffix. Percentages are NOT a
/// fixed size (they resolve against the containing block, which
/// layout does not know here) — they return `None` so the caller
/// falls back to the intrinsic default. Non-positive values are
/// invalid per spec and also return `None`.
fn parse_svg_attr_length(value: Option<&str>) -> Option<f32> {
    let v = value?.trim();
    let v = v.strip_suffix("px").unwrap_or(v).trim();
    if v.ends_with('%') {
        return None;
    }
    v.parse::<f32>().ok().filter(|n| *n > 0.0)
}

/// Parse an SVG `viewBox` attribute (`min-x min-y width height`,
/// whitespace- and/or comma-separated) into its width and height.
/// `None` when absent, malformed, or with a non-positive width or
/// height (an invalid viewBox is ignored per spec, not an error).
fn parse_view_box(value: Option<&str>) -> Option<(f32, f32)> {
    let v = value?;
    let mut nums = v
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>());
    let _min_x = nums.next()?.ok()?;
    let _min_y = nums.next()?.ok()?;
    let w = nums.next()?.ok()?;
    let h = nums.next()?.ok()?;
    if w > 0.0 && h > 0.0 { Some((w, h)) } else { None }
}

/// Intrinsic size for an `<svg>` element (Stage 15, M7 Phase 1,
/// Rec 2): `(width, height, aspect_ratio)`.
///
/// * `width`/`height` attributes win per axis; the other axis
///   follows the `viewBox` ratio when one is present, else the
///   300×150 replaced-element default on that axis.
/// * No attributes but a `viewBox`: the 300×150 default size, with
///   the viewBox ratio so a CSS-constrained axis keeps it.
/// * Neither: 300×150, no ratio.
/// * `preserveAspectRatio="none"` disables the ratio (the viewport
///   stretches); any other value — including absent — keeps it.
///   `meet` vs `slice` only changes how the viewBox maps into the
///   viewport at paint time, not the box size.
fn svg_intrinsic_size(dom: &Dom, id: DomId) -> (f32, f32, Option<f32>) {
    let attr_w = parse_svg_attr_length(dom.attr(id, "width"));
    let attr_h = parse_svg_attr_length(dom.attr(id, "height"));
    let view_box = parse_view_box(dom.attr(id, "viewBox"));
    let ratio = view_box.map(|(w, h)| w / h);
    let ratio = match dom.attr(id, "preserveAspectRatio") {
        Some(v) if v.split_whitespace().next() == Some("none") => None,
        _ => ratio,
    };
    let (w, h) = match (attr_w, attr_h) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, ratio.map_or(150.0, |r| w / r)),
        (None, Some(h)) => (ratio.map_or(300.0, |r| h * r), h),
        (None, None) => (300.0, 150.0),
    };
    // With both attributes the ratio is the attributes' own, not
    // the viewBox's — but `preserveAspectRatio="none"` still
    // disables it (`ratio` is already `None` then).
    let ratio = match (attr_w, attr_h) {
        (Some(_), Some(_)) => ratio.map(|_| w / h),
        _ => ratio,
    };
    (w, h, ratio)
}

/// Owns the Taffy layout tree and the cosmic-text font system for one
/// document, plus the laid-out boxes in document order.
pub struct LayoutTree {
    pub taffy: TaffyTree<NodeContext>,
    /// The font system, shared across rebuilds of this tree (Stage 8
    /// Rec 9): a fresh `FontSystem` re-scans the system fonts, so a
    /// full rebuild reuses the retained one instead of creating a
    /// new one. `None` only transiently, inside `full_rebuild`
    /// between the `take` and the fresh build (or its failure — the
    /// actor then falls back to a fresh scan via `build_full_tree`).
    pub font_system: Option<FontSystem>,
    pub root: Option<TaffyId>,
    pub boxes: Vec<LayoutBox>,
    pub viewport: (f32, f32),
    /// Canvas background: the html element's background, else the body's
    /// (CSS canvas rule), else white — a browser's canvas defaults to
    /// white when neither element sets one. Painted over the whole
    /// viewport by the display list, not just the element's box.
    pub canvas_background: [u8; 3],
    /// Per-node style data (computed style + hint/damage/flags).
    pub node_styles: SparseSecondaryMap<DomId, NodeStyle>,
    /// Retained scroll offsets per scroll container, mutated by input
    /// (`set_scroll_offset`), never recomputed by layout.
    pub scroll_offsets: HashMap<DomId, (f32, f32)>,
    /// The root element's computed font size (`rem` basis for
    /// border-radius / box-shadow resolution in [`collect_boxes`]).
    pub root_font_size: f32,
    /// `DomId -> taffy node` for hit-testing / `mark_dirty`.
    pub dom_to_taffy: HashMap<DomId, TaffyId>,
    /// Position-only translations recorded by `translate_subtree`
    /// (Stage 8 Rec 6), keyed by the moved subtree's taffy root:
    /// `(dx, dy)` to add to that subtree's boxes when they are
    /// re-collected. The taffy layouts stay stale after a move, so
    /// this map is the record of it; any path that rebuilds a node
    /// from the DOM (subtree / full layout) drops its entry, because
    /// the fresh taffy layout already carries the new position.
    position_overrides: HashMap<TaffyId, (f32, f32)>,
    meta: HashMap<TaffyId, BoxMeta>,
    /// Images decoded for this document (Stage 11, M3): the map
    /// the tree was built with. `recollect` reads it, so a
    /// re-layout after an image arrival picks the new image up
    /// without the caller re-threading the map.
    pub images: ImageMap,
    /// The `prefers-color-scheme` theme this tree was built with
    /// (M6 Phase 3, Rec 12): restyles after a theme change rebuild
    /// against the new theme instead of the hardcoded Light the
    /// incremental path used before.
    pub theme: Theme,
    /// The viewport (page) scroll offset, in page px (M6 Phase 3,
    /// Rec 1/3): mirrored from the shell by
    /// [`LayoutTree::set_viewport_scroll`]. Boxes are in page
    /// coordinates, so this is NOT subtracted in `collect_boxes` —
    /// it drives sticky offsets and viewport-fixed hit-testing.
    pub viewport_scroll: (f32, f32),
    /// Scroll-derived sticky offsets (M6 Phase 3, Rec 3), keyed by
    /// taffy node: `(dx, dy)` added to the sticky subtree's boxes
    /// when they are re-collected. Recomputed from scratch on every
    /// `recollect` (scroll or layout change); never set by hand.
    sticky_offsets: HashMap<TaffyId, (f32, f32)>,
    /// Hashes of web-font byte blobs already loaded into
    /// `font_system` (M6 Phase 3, Rec 12): `@font-face` downloads
    /// arrive per navigation and must not be re-loaded on every
    /// rebuild.
    loaded_font_hashes: HashSet<u64>,
    /// In-flight sampled animation values (M6 Phase 3, Rec 11):
    /// `(element, [(property, value)])`, owned by the tab thread
    /// and set on the actor's tree before each layout. The actor
    /// applies them over the resolved (underlying) styles before
    /// classification, so an animated transform routes through
    /// the CompositingOnly path and an animated width re-lays out
    /// like any other style change. The retained `node_styles`
    /// stay underlying — overrides never persist there.
    pub animation_overrides: Vec<(DomId, Vec<(String, animation::AnimatedValue)>)>,
}

impl LayoutTree {
    pub fn new() -> Self {
        Self {
            taffy: TaffyTree::new(),
            font_system: Some(FontSystem::new()),
            root: None,
            boxes: Vec::new(),
            viewport: (0.0, 0.0),
            canvas_background: [255, 255, 255],
            node_styles: SparseSecondaryMap::new(),
            scroll_offsets: HashMap::new(),
            root_font_size: 16.0,
            dom_to_taffy: HashMap::new(),
            position_overrides: HashMap::new(),
            meta: HashMap::new(),
            images: ImageMap::default(),
            theme: Theme::Light,
            viewport_scroll: (0.0, 0.0),
            sticky_offsets: HashMap::new(),
            loaded_font_hashes: HashSet::new(),
            animation_overrides: Vec::new(),
        }
    }

    /// Lay out `dom` with `sheet` in a viewport of the given size.
    /// Returns the tree with `boxes` in document order, positions in
    /// page coordinates (origin = viewport top-left). `sheet` is the
    /// document's author sheet; the parsed UA stylesheet
    /// (`Stylesheet::ua()`) is always applied underneath it.
    /// Lay out `dom` with `sheet` in a viewport of the given size,
    /// evaluating `prefers-color-scheme` against `theme` (M6 Rec 12).
    /// The shell theme signal is a later phase; callers that have no
    /// theme yet pass [`Theme::Light`].
    pub fn build(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        Self::build_with_state(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            &ElementState::default(),
            theme,
        )
    }

    /// Like `build`, with dynamic pseudo-class state (`:hover` /
    /// `:active` / `:focus`) for this pass.
    pub fn build_with_state(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        // A fresh `FontSystem` scans the system fonts (~1.5 s):
        // fine for a first layout, but a full rebuild must not pay
        // it again — see `build_with_state_reusing_font_system`.
        Self::build_with_state_and_font_system(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            state,
            FontSystem::new(),
            &ImageMap::default(),
            theme,
        )
    }

    /// Like [`build_with_state`](Self::build_with_state), but with
    /// the images decoded so far (Stage 11, M3): `<img>` boxes get
    /// intrinsic sizing from the map, broken images get alt text,
    /// and background-image boxes get their resource id. An image
    /// arrival rebuilds with the fuller map (full rebuild,
    /// coalesced per tick — the incremental path does not take an
    /// image map; see the worker's `images_dirty` flag).
    pub fn build_with_state_and_images(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        images: &ImageMap,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        Self::build_with_state_and_font_system(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            state,
            FontSystem::new(),
            images,
            theme,
        )
    }

    /// Like [`build_with_state`](Self::build_with_state), but the
    /// tree reuses an existing `FontSystem` (Stage 8 Rec 9): the
    /// layout actor hands back the retained tree's font system on
    /// a full rebuild, so a hover restyle that lands on the
    /// document root does not re-scan the system fonts per pointer
    /// event. Fonts are system-global and shaping is per text run,
    /// so reuse across documents is safe — the incremental path
    /// already reuses the retained tree's font system across
    /// mutations of the same document.
    pub fn build_with_state_reusing_font_system(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        font_system: FontSystem,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        Self::build_with_state_and_font_system(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            state,
            font_system,
            &ImageMap::default(),
            theme,
        )
    }

    /// Like [`build_with_state_reusing_font_system`](Self::build_with_state_reusing_font_system),
    /// but with the images decoded so far (Stage 11, M3) — the
    /// entry the worker uses for image-arrival rebuilds.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_state_reusing_font_system_and_images(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        font_system: FontSystem,
        images: &ImageMap,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        Self::build_with_state_and_font_system(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            state,
            font_system,
            images,
            theme,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_state_and_font_system(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        font_system: FontSystem,
        images: &ImageMap,
        theme: Theme,
    ) -> Result<Self, LayoutError> {
        let ua = Stylesheet::ua();
        // Two-phase style resolution (Stage 4 #5): every element's
        // style is resolved up front, in parallel, before any taffy
        // node is built. The builder looks styles up from this map.
        let (styles, node_styles) = resolve_document_styles(
            dom,
            &ua,
            sheet,
            true,
            (viewport_width, viewport_height),
            state,
            theme,
        );
        // The root element's font size is the `rem` basis for
        // border-radius / box-shadow resolution (Stage 8). Read it
        // here: `styles` moves into the builder below.
        let root_dom_pre = find_element(dom, dom.root().unwrap(), "html");
        let root_font_size = root_dom_pre
            .and_then(|r| styles.get(&r))
            .map(|s| s.font_size)
            .unwrap_or(16.0);
        let mut taffy = TaffyTree::new();
        let mut meta: HashMap<TaffyId, BoxMeta> = HashMap::new();
        let mut leaves: Vec<TaffyId> = Vec::new();
        let mut builder = Builder {
            dom,
            styles,
            taffy: &mut taffy,
            meta: &mut meta,
            leaves: &mut leaves,
            viewport: (viewport_width, viewport_height),
            images,
        };
        let root_dom = find_element(dom, dom.root().unwrap(), "html");
        let root = match root_dom {
            Some(html) => builder
                .build_node(html, &ComputedStyle::initial())
                .ok_or(LayoutError::NoRoot)?,
            None => {
                // Fragment without <html>: lay out the document's children
                // directly under a synthetic block root.
                let default_style = ComputedStyle::initial();
                let children =
                    builder.build_children(dom.root().unwrap(), &default_style);
                let node = builder
                    .taffy
                    .new_with_children(to_taffy_style(&default_style, builder.viewport), &children)?;
                builder.meta.insert(
                    node,
                    BoxMeta {
                        dom: None,
                        style: default_style,
                        children,
                        text: None,
                        svg_canvas: None,
                    },
                );
                node
            }
        };
        // The builder only borrowed the tree; drop it so the locals
        // below are usable again.
        drop(builder);

        // R11 pre-shaping: shape every distinct text run once, in
        // parallel on rayon workers each owning a FontSystem, and
        // store plain `Send` data in the leaf contexts. The measure
        // closure below then does no cosmic-text work at all.
        preshape_leaves(&mut taffy, &mut meta, &leaves);

        // Text leaves are measured arithmetically from their
        // pre-shaped data (see `measure_text`); taffy itself stays
        // serial by design.
        taffy.compute_layout_with_measure(
            root,
            Size {
                width: AvailableSpace::Definite(viewport_width),
                height: AvailableSpace::MaxContent,
            },
            |input, _node, ctx, _style| measure_text(ctx, input),
        )?;

        let dom_to_taffy: HashMap<DomId, TaffyId> = meta
            .iter()
            .filter_map(|(&t, m)| m.dom.map(|d| (d, t)))
            .collect();
        // Canvas background: html's, else body's (CSS canvas rule),
        // else white — the browser default when neither sets one.
        let mut tree = Self {
            taffy,
            font_system: Some(font_system),
            root: Some(root),
            boxes: Vec::new(),
            viewport: (viewport_width, viewport_height),
            canvas_background: [255, 255, 255],
            node_styles,
            scroll_offsets: HashMap::new(),
            root_font_size,
            dom_to_taffy,
            position_overrides: HashMap::new(),
            meta,
            images: images.clone(),
            theme,
            viewport_scroll: (0.0, 0.0),
            sticky_offsets: HashMap::new(),
            loaded_font_hashes: HashSet::new(),
            animation_overrides: Vec::new(),
        };
        // Re-collect boxes (which also shapes glyphs at final width).
        tree.recollect();
        let html_bg = tree
            .boxes
            .iter()
            .find(|b| b.taffy == root)
            .and_then(|b| b.style.background);
        let body_bg = root_dom
            .and_then(|r| find_element(dom, r, "body"))
            .and_then(|body_id| tree.boxes.iter().find(|b| b.dom == Some(body_id)))
            .and_then(|b| b.style.background);
        tree.canvas_background = html_bg.or(body_bg).unwrap_or([255, 255, 255]);
        Ok(tree)
    }

    /// Re-collect `boxes` from the taffy tree, applying the retained
    /// scroll offsets and re-shaping glyphs at their final positions.
    /// Called after layout and after every `set_scroll_offset`.
    /// Sticky offsets are computed from the first pass and applied
    /// in a second collect (M6 Phase 3, Rec 3): a sticky box lays
    /// out in flow, then shifts by a scroll-derived offset clamped
    /// to its containing block — never through taffy insets.
    fn recollect(&mut self) {
        if self.root.is_none() {
            return;
        }
        self.sticky_offsets.clear();
        let mut boxes = self.collect_all();
        let sticky = self.sticky_offsets_for(&boxes);
        if !sticky.is_empty() {
            self.sticky_offsets = sticky;
            boxes = self.collect_all();
        }
        // `None` only transiently inside `full_rebuild` (see the
        // field docs): every other path holds a font system.
        let fs = self
            .font_system
            .as_mut()
            .expect("LayoutTree holds a FontSystem outside a full rebuild");
        for bx in boxes.iter_mut() {
            if bx.text.is_some() {
                bx.glyphs = shape_leaf(fs, bx);
            }
        }
        self.boxes = boxes;
    }

    /// One `collect_boxes` pass over the taffy tree, with the
    /// current `sticky_offsets` applied.
    fn collect_all(&self) -> Vec<LayoutBox> {
        let Some(root) = self.root else {
            return Vec::new();
        };
        let mut boxes = Vec::new();
        collect_boxes(
            &self.taffy,
            &self.meta,
            root,
            (0.0, 0.0),
            (0.0, 0.0),
            &self.scroll_offsets,
            None,
            None,
            None,
            None,
            self.viewport,
            self.root_font_size,
            &self.position_overrides,
            &self.sticky_offsets,
            &self.images,
            None,
            Affine::identity(),
            false,
            false,
            &mut boxes,
        );
        boxes
    }

    /// The scroll-derived sticky offsets for `boxes` (M6 Phase 3,
    /// Rec 3), keyed by taffy node. For each `position: sticky`
    /// box, in its scroll space — the content space of its nearest
    /// scroll container (or page space for viewport scrolling),
    /// with the origin at that container's content-box top-left:
    /// `top` sticks the box `top` px below the scrollport top once
    /// the scroll passes its flow position, and releases it when
    /// its bottom would pass its containing block's bottom edge
    /// (same on the horizontal axis with `left` / `right`). The
    /// containing block is the nearest ancestor box; a sticky box
    /// inside a viewport-fixed subtree is already pinned and gets
    /// no offset. Percentages in insets resolve against the
    /// containing block's size; `auto` insets stick at 0 on that
    /// side only when the opposite inset is also `auto`.
    fn sticky_offsets_for(&self, boxes: &[LayoutBox]) -> HashMap<TaffyId, (f32, f32)> {
        let mut by_dom: HashMap<DomId, usize> = HashMap::new();
        for (i, bx) in boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        // Flow (unscrolled, page-space) position of a box: its
        // collected position plus the scroll offsets of the scroll
        // containers above it (a container's own box does NOT move
        // with its own offset, so its own offset is never added
        // back — only its ancestors' are).
        let flow_pos = |bx: &LayoutBox| -> (f32, f32) {
            let mut ox = 0.0f32;
            let mut oy = 0.0f32;
            let mut cur = bx.parent;
            while let Some(dom) = cur {
                let Some(&pi) = by_dom.get(&dom) else {
                    break;
                };
                let p = &boxes[pi];
                if p.style.overflow_x.is_scroll_container()
                    || p.style.overflow_y.is_scroll_container()
                {
                    let (ax, ay) =
                        self.scroll_offsets.get(&dom).copied().unwrap_or((0.0, 0.0));
                    ox += ax;
                    oy += ay;
                }
                cur = p.parent;
            }
            (bx.x + ox, bx.y + oy)
        };
        let has_fixed_ancestor = |bx: &LayoutBox| -> bool {
            let mut cur = bx.parent;
            while let Some(dom) = cur {
                let Some(&pi) = by_dom.get(&dom) else {
                    break;
                };
                if boxes[pi].is_viewport_fixed {
                    return true;
                }
                cur = boxes[pi].parent;
            }
            false
        };
        let mut out: HashMap<TaffyId, (f32, f32)> = HashMap::new();
        for bx in boxes {
            if bx.dom.is_none() {
                continue;
            }
            if bx.style.position != PositionKind::Sticky || has_fixed_ancestor(bx) {
                continue;
            }
            // Scroll offset, visible size, and content origin of
            // the scrollport the box scrolls in (its nearest scroll
            // container, else the viewport — origin 0,0 in page
            // space). All sticky math below is in this scroll
            // space, so a container at page y=8 still sticks its
            // header at its own top, not at y=0.
            let (sx, sy, vis_w, vis_h, origin_x, origin_y) = match bx.scroll_ancestor {
                Some(c) => {
                    let (ox, oy) = self.scroll_offsets.get(&c).copied().unwrap_or((0.0, 0.0));
                    let (w, h, ox0, oy0) = by_dom
                        .get(&c)
                        .map(|&i| {
                            let (fx, fy) = flow_pos(&boxes[i]);
                            (boxes[i].width, boxes[i].height, fx, fy)
                        })
                        .unwrap_or((self.viewport.0, self.viewport.1, 0.0, 0.0));
                    (ox, oy, w, h, ox0, oy0)
                }
                None => (
                    self.viewport_scroll.0,
                    self.viewport_scroll.1,
                    self.viewport.0,
                    self.viewport.1,
                    0.0,
                    0.0,
                ),
            };
            let (flow_x, flow_y) = flow_pos(bx);
            let (flow_x, flow_y) = (flow_x - origin_x, flow_y - origin_y);
            // Containing block: nearest ancestor box, in the same
            // scroll space (unbounded when there is none — the
            // root, whose content runs to the end of the page).
            let (cb_left, cb_top, cb_right, cb_bottom) = match bx
                .parent
                .and_then(|p| by_dom.get(&p))
            {
                Some(&ci) => {
                    let cb = &boxes[ci];
                    let (cx, cy) = flow_pos(cb);
                    (
                        cx - origin_x,
                        cy - origin_y,
                        cx - origin_x + cb.width,
                        cy - origin_y + cb.height,
                    )
                }
                None => (0.0, 0.0, f32::INFINITY, f32::INFINITY),
            };
            let resolve = |d: Option<Dim>, basis: f32| {
                d.map(|dim| match dim {
                    Dim::Px(v) => v,
                    Dim::Pct(f) => f * basis,
                    Dim::Vh(v) => v / 100.0 * self.viewport.1,
                    Dim::Vw(v) => v / 100.0 * self.viewport.0,
                })
            };
            let cb_w = if cb_right.is_finite() {
                cb_right - cb_left
            } else {
                self.viewport.0
            };
            let cb_h = if cb_bottom.is_finite() {
                cb_bottom - cb_top
            } else {
                self.viewport.1
            };
            let dy = match (
                resolve(bx.style.inset[0], cb_h),
                resolve(bx.style.inset[2], cb_h),
            ) {
                (Some(top), _) => {
                    let max = (cb_bottom - flow_y - bx.height).max(0.0);
                    (sy - (flow_y - top)).clamp(0.0, max)
                }
                (None, Some(bottom)) => {
                    let min = cb_top - flow_y;
                    ((sy + vis_h) - bottom - (flow_y + bx.height)).clamp(min, 0.0)
                }
                (None, None) => 0.0,
            };
            let dx = match (
                resolve(bx.style.inset[3], cb_w),
                resolve(bx.style.inset[1], cb_w),
            ) {
                (Some(left), _) => {
                    let max = (cb_right - flow_x - bx.width).max(0.0);
                    (sx - (flow_x - left)).clamp(0.0, max)
                }
                (None, Some(right)) => {
                    let min = cb_left - flow_x;
                    ((sx + vis_w) - right - (flow_x + bx.width)).clamp(min, 0.0)
                }
                (None, None) => 0.0,
            };
            if dx != 0.0 || dy != 0.0 {
                out.insert(bx.taffy, (dx, dy));
            }
        }
        out
    }

    /// Set the viewport (page) scroll offset (M6 Phase 3, Rec 3):
    /// sticky boxes re-anchor against it, and viewport-fixed boxes
    /// hit-test against it. No layout is recomputed — boxes are
    /// re-collected with the new scroll applied. The offset is
    /// stored as given; the caller (the worker's viewport-scroll
    /// mirror) clamps it to the page maximum.
    ///
    /// Returns `true` when the scroll moved a sticky box (the
    /// sticky offsets changed) — only then does the re-collected
    /// display list differ from the previous one. A pure viewport
    /// scroll with no sticky content leaves the list identical
    /// (fixed boxes are viewport-anchored, containers scroll
    /// through their own offsets), so the worker skips the submit
    /// and the shell just scrolls its cached list.
    pub fn set_viewport_scroll(&mut self, x: f32, y: f32) -> bool {
        let before = self.sticky_offsets.clone();
        self.viewport_scroll = (x, y);
        self.recollect();
        self.sticky_offsets != before
    }

    /// Change the theme this tree resolves `prefers-color-scheme`
    /// against (M6 Phase 3, Rec 12). Only records the theme — the
    /// caller rebuilds (full rebuild, no refetch) for the restyle
    /// to take effect.
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Load web-font bytes into the tree's font system (M6 Phase 3,
    /// Rec 12, `@font-face`): each blob is loaded once (deduped by
    /// a hash of its bytes — the same URL fetched twice still loads
    /// once). Returns true when at least one new face landed, in
    /// which case the caller must rebuild: shaping changes with the
    /// available faces. Invalid blobs are ignored by fontdb, so a
    /// corrupt download degrades to the fallback font, never an
    /// error.
    pub fn load_fonts(&mut self, fonts: &[Vec<u8>]) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let Some(fs) = self.font_system.as_mut() else {
            return false;
        };
        let mut loaded = false;
        for bytes in fonts {
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            let hash = hasher.finish();
            if !self.loaded_font_hashes.insert(hash) {
                continue;
            }
            fs.db_mut().load_font_data(bytes.clone());
            loaded = true;
        }
        loaded
    }

    /// Boxes in document order.
    pub fn boxes(&self) -> &[LayoutBox] {
        &self.boxes
    }

    /// Total content height (bottom of the lowest box).
    pub fn content_height(&self) -> f32 {
        self.boxes
            .iter()
            .map(|b| b.y + b.height)
            .fold(0.0, f32::max)
    }

    /// Total content width (right edge of the rightmost box), the
    /// horizontal counterpart of [`content_height`].
    pub fn content_width(&self) -> f32 {
        self.boxes
            .iter()
            .map(|b| b.x + b.width)
            .fold(0.0, f32::max)
    }

    /// Maximum scroll offset of one scroll container, per axis
    /// (`scrollHeight - clientHeight`, `scrollWidth -
    /// clientWidth`): how far the container's content reaches past
    /// its own box, in the container's scroll space. `None` when
    /// `id` has no box (not a laid-out element). Content is every
    /// box whose nearest scroll container — counting a scroll
    /// container's own box as content of its parent — is this
    /// container: a nested scroll container contributes its own
    /// border box, not its inner content (that content scrolls
    /// inside the nested container). Boxes already carry their
    /// container's current offset (see `recollect`), so the
    /// container's own offset is added back — offsets of ancestors
    /// above it shift the container's box by the same amount, so
    /// they cancel in the difference.
    pub fn scrollable_size(&self, id: DomId) -> Option<(f32, f32)> {
        let container = *self.boxes_for_dom(id).first()?;
        let (ox, oy) = self.scroll_offsets.get(&id).copied().unwrap_or((0.0, 0.0));
        // Taffy parent of each box, from the meta children lists.
        let mut parent: HashMap<TaffyId, TaffyId> = HashMap::new();
        for (node, m) in &self.meta {
            for &child in &m.children {
                parent.insert(child, *node);
            }
        }
        let by_taffy: HashMap<TaffyId, &LayoutBox> =
            self.boxes.iter().map(|b| (b.taffy, b)).collect();
        let is_container = |b: &LayoutBox| {
            b.style.overflow_x.is_scroll_container() || b.style.overflow_y.is_scroll_container()
        };
        // The container's own extent: content shorter than the box
        // scrolls nowhere.
        let mut max_right = container.width;
        let mut max_bottom = container.height;
        // M6 Phase 3 (Rec 1): viewport-fixed boxes are not content
        // of any scroll container — neither they nor their
        // descendants may inflate a container's maximum. A fixed
        // box's descendants scroll with the fixed box, not with
        // the container underneath it.
        let mut fixed_subtree: HashSet<TaffyId> = HashSet::new();
        let mut stack: Vec<TaffyId> = self
            .boxes
            .iter()
            .filter(|b| b.is_viewport_fixed)
            .map(|b| b.taffy)
            .collect();
        while let Some(node) = stack.pop() {
            if !fixed_subtree.insert(node) {
                continue;
            }
            if let Some(m) = self.meta.get(&node) {
                stack.extend(m.children.iter().copied());
            }
        }
        for b in &self.boxes {
            if fixed_subtree.contains(&b.taffy) {
                continue;
            }
            if b.dom == Some(id) {
                continue;
            }
            // Nearest scroll container above this box (a scroll
            // container's own box counts as content of its parent).
            let ancestor = if is_container(b) {
                parent
                    .get(&b.taffy)
                    .and_then(|p| by_taffy.get(p))
                    .and_then(|p| p.scroll_ancestor)
            } else {
                b.scroll_ancestor
            };
            if ancestor != Some(id) {
                continue;
            }
            max_right = max_right.max(b.x + b.width + ox - container.x);
            max_bottom = max_bottom.max(b.y + b.height + oy - container.y);
        }
        Some((
            (max_right - container.width).max(0.0),
            (max_bottom - container.height).max(0.0),
        ))
    }

    /// Hit-test: the topmost box (in *paint* order, M6 Phase 2
    /// Rec 4 — paint order and hit order must never disagree)
    /// whose rect contains `(x, y)`. Boxes already carry their
    /// scroll offsets (applied in `recollect`), so no offset
    /// subtraction is needed here. Text leaves have no `DomId`
    /// and never win. A transformed box is tested through the
    /// inverse of its page transform (Rec 7/8): the point is
    /// mapped back into the box's layout rect, so a click inside
    /// the axis-aligned bounding box but outside the rotated rect
    /// misses. A box whose transform is singular (e.g. a
    /// `scale(0)`) has no inverse and is unhittable. The point
    /// must also lie inside the box's effective clip (mapped back
    /// to layout space the same way) — a point under a clipped
    /// box's overflow region does not hit it.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<DomId> {
        let transforms = self.page_transforms();
        let order = self.paint_order();
        // M6 Phase 3 (Rec 1): the point arrives in page
        // coordinates (the shell added the viewport scroll), but a
        // viewport-fixed box lives in viewport coordinates — test
        // it against the point with the viewport scroll removed.
        let (vx, vy) = (x - self.viewport_scroll.0, y - self.viewport_scroll.1);
        for &i in order.iter().rev() {
            let b = &self.boxes[i];
            let Some(dom) = b.dom else {
                continue;
            };
            let (px, py) = if b.is_viewport_fixed { (vx, vy) } else { (x, y) };
            let inv = match transforms[i].invert() {
                Some(inv) => inv,
                None => continue,
            };
            let (lx, ly) = inv.apply(px, py);
            if !(lx >= b.x && lx < b.x + b.width && ly >= b.y && ly < b.y + b.height) {
                continue;
            }
            if let Some(clip) = b.clip
                && !(lx >= clip[0]
                    && lx < clip[0] + clip[2]
                    && ly >= clip[1]
                    && ly < clip[1] + clip[3])
            {
                continue;
            }
            return Some(dom);
        }
        None
    }

    /// The stable wire key for a DOM id (M6 Phase 2, Rec 9): the
    /// slotmap key's ffi value. The display list's
    /// `PushTransform`/`PushOpacity` items and the `CompositingUpdate`
    /// IPC message agree on this key — both are produced
    /// worker-side from the same tree, so the key never needs to
    /// be interpreted, only matched.
    pub fn dom_key(id: DomId) -> u64 {
        id.data().as_ffi()
    }

    /// Paint order (CSS 2.1 Appendix E, M6 Phase 2 Rec 4): box
    /// indices in the order their items paint — a permutation of
    /// `boxes()`, computed from the stacking-context tree, not
    /// document order. The walk, per stacking context:
    ///
    /// 1. the context root's own box;
    /// 2. child stacking contexts with negative `z-index`, in
    ///    `z-index` order (then tree order);
    /// 3. in-flow, non-positioned members, in tree order
    ///    (blocks and inlines together — floats and borders do
    ///    not exist yet, so there is no float step to split out;
    ///    when floats land they slot in between steps 3 and 4);
    /// 4. positioned members with `z-index: auto` and child
    ///    stacking contexts with `z-index: auto`/`0`, interleaved
    ///    in tree order (Appendix E step 8);
    /// 5. child stacking contexts with positive `z-index`, in
    ///    `z-index` order (then tree order).
    ///
    /// A member is any box whose nearest stacking-context
    /// ancestor is the context being walked; a member's own
    /// descendants that belong to a deeper context paint inside
    /// that context's group, never with the member. `z-index` is
    /// ignored on static boxes (they are never stacking contexts
    /// and never positioned, so they stay in step 3). The root
    /// element is always the root stacking context, whatever its
    /// style says.
    pub fn paint_order(&self) -> Vec<usize> {
        if self.boxes.is_empty() {
            return Vec::new();
        }
        // Nearest stacking-context ancestor for every box, by
        // walking the `parent` chain (the root box, index 0, is
        // the root context when no ancestor qualifies).
        let mut by_dom: HashMap<DomId, usize> = HashMap::new();
        for (i, bx) in self.boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        let mut context_of: Vec<usize> = Vec::with_capacity(self.boxes.len());
        for bx in &self.boxes {
            let mut ctx = 0;
            let mut cur = bx.parent;
            while let Some(dom) = cur {
                let Some(&pi) = by_dom.get(&dom) else {
                    break;
                };
                if self.boxes[pi].is_stacking_context {
                    ctx = pi;
                    break;
                }
                cur = self.boxes[pi].parent;
            }
            context_of.push(ctx);
        }
        // Members per context, in tree order (`boxes()` is
        // pre-order, so index order is tree order).
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &ctx) in context_of.iter().enumerate() {
            if i != ctx {
                members.entry(ctx).or_default().push(i);
            }
        }
        let mut out = Vec::with_capacity(self.boxes.len());
        paint_context(0, &members, &self.boxes, &mut out);
        debug_assert_eq!(out.len(), self.boxes.len());
        out
    }

    /// The page-space transform for every box (M6 Phase 2,
    /// Rec 7/8): one affine per box, aligned with `boxes()`,
    /// mapping the box's *layout* coordinates to page
    /// coordinates. For a box with transformed top-left `P` and
    /// accumulated matrix `M`, the mapping is `P + M_lin·(p -
    /// (x, y))` folded into a single matrix — paint applies it to
    /// item coordinates, hit-testing inverts it. `P` is the page
    /// position of the box's local origin, and it composes as
    /// `P(child) = P(parent) + M_lin(parent)·offset +
    /// (M(child).e - M(parent).e, M(child).f - M(parent).f)`:
    /// the parent's linear part rotates the child's offset, and
    /// the translation difference is the child's own transform's
    /// contribution (a rotated box's own origin moves its
    /// top-left; a child with no transform of its own
    /// contributes nothing). The linear part only for the
    /// offset: the offset is a parent-local *vector*, and the
    /// parent's translation already lives in the parent's `P`.
    /// The root's top-left is its layout position plus its own
    /// matrix's translation. A box whose parent box is missing
    /// (unreachable in a well-formed tree) falls back to its
    /// layout position.
    pub fn page_transforms(&self) -> Vec<Affine> {
        let mut by_dom: HashMap<DomId, usize> = HashMap::new();
        for (i, bx) in self.boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        // Transformed top-lefts, in pre-order (parents precede
        // children, so each parent's top-left is ready).
        let mut tops: Vec<(f32, f32)> = Vec::with_capacity(self.boxes.len());
        for bx in &self.boxes {
            // M6 Phase 3 (Rec 1): a viewport-fixed box is rooted
            // at the viewport — its parent's page offset must not
            // leak into its top-left (its `x`/`y` are already
            // viewport coordinates).
            let top = match bx.parent.and_then(|p| by_dom.get(&p)) {
                Some(&pi) if !bx.is_viewport_fixed => {
                    let parent = &self.boxes[pi];
                    let (ppx, ppy) = tops[pi];
                    let m = parent.matrix;
                    let (dx, dy) = (bx.x - parent.x, bx.y - parent.y);
                    (
                        ppx + m.a * dx + m.c * dy + (bx.matrix.e - m.e),
                        ppy + m.b * dx + m.d * dy + (bx.matrix.f - m.f),
                    )
                }
                _ => (bx.x + bx.matrix.e, bx.y + bx.matrix.f),
            };
            tops.push(top);
        }
        self.boxes
            .iter()
            .zip(tops)
            .map(|(bx, (px, py))| {
                let m = bx.matrix;
                Affine {
                    a: m.a,
                    b: m.b,
                    c: m.c,
                    d: m.d,
                    // Fold `P + M_lin·(p - (x, y))` into the
                    // matrix: e = Px - (a*x + c*y),
                    // f = Py - (b*x + d*y). Only the linear part
                    // folds in — `M`'s own translation already
                    // lives in `P`.
                    e: px - (m.a * bx.x + m.c * bx.y),
                    f: py - (m.b * bx.x + m.d * bx.y),
                }
            })
            .collect()
    }

    /// The compositing state of `ids` (M6 Phase 2, Rec 9): one
    /// `(key, matrix, opacity)` triple per id that has a box —
    /// the key is [`LayoutTree::dom_key`], the matrix the box's
    /// page transform, the opacity its computed opacity. The tab
    /// thread turns these into the `CompositingUpdate` IPC
    /// message; ids with no box (unrendered elements) are
    /// skipped, not errors.
    pub fn composited_state(&self, ids: &[DomId]) -> Vec<(u64, Affine, f32)> {
        let transforms = self.page_transforms();
        let mut by_dom: HashMap<DomId, usize> = HashMap::new();
        for (i, bx) in self.boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        ids.iter()
            .filter_map(|&id| {
                let &i = by_dom.get(&id)?;
                let bx = &self.boxes[i];
                Some((Self::dom_key(id), transforms[i], bx.style.opacity))
            })
            .collect()
    }

    /// The compositing state of `ids` and every element in their
    /// box subtrees (M6 Phase 2, Rec 9): a transform change on an
    /// element moves its whole subtree — every descendant box's
    /// page transform changes with it — so a `CompositingUpdate`
    /// must carry the new state for the changed elements AND
    /// their descendants, or the descendants would keep their
    /// stale matrices (a rotated container would move its
    /// background and leave its text behind). Text leaves share
    /// their parent element's page transform (the same map), so
    /// they ride on the parent's entry and need no entry of
    /// their own. An opacity-only change needs no subtree
    /// expansion, but including the descendants is idempotent
    /// (their opacity is unchanged), so one method serves both.
    pub fn composited_subtree_state(&self, ids: &[DomId]) -> Vec<(u64, Affine, f32)> {
        let transforms = self.page_transforms();
        let mut by_dom: HashMap<DomId, usize> = HashMap::new();
        for (i, bx) in self.boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        let wanted: HashSet<DomId> = ids.iter().copied().collect();
        let mut expanded: Vec<DomId> = Vec::new();
        let mut seen: HashSet<DomId> = HashSet::new();
        for bx in &self.boxes {
            let Some(dom) = bx.dom else {
                continue;
            };
            if !seen.insert(dom) {
                continue;
            }
            // Walk the box's ancestor chain: the element is
            // affected when it, or any ancestor, is dirty.
            let mut cur = Some(dom);
            let mut hit = false;
            while let Some(d) = cur {
                if wanted.contains(&d) {
                    hit = true;
                    break;
                }
                cur = by_dom.get(&d).and_then(|&i| self.boxes[i].parent);
            }
            if hit {
                expanded.push(dom);
            }
        }
        expanded
            .iter()
            .map(|&id| {
                let i = by_dom[&id];
                let bx = &self.boxes[i];
                (Self::dom_key(id), transforms[i], bx.style.opacity)
            })
            .collect()
    }

    /// All boxes belonging to one DOM node (element boxes; text leaves
    /// have no `DomId`). The shell's retained rect map for hit-testing
    /// and link handling.
    pub fn boxes_for_dom(&self, id: DomId) -> Vec<&LayoutBox> {
        self.boxes.iter().filter(|b| b.dom == Some(id)).collect()
    }

    /// Retained scroll offset of a scroll container, if any.
    pub fn scroll_offset(&self, id: DomId) -> Option<(f32, f32)> {
        self.scroll_offsets.get(&id).copied()
    }

    /// Set a scroll container's offset (input-driven) and re-collect
    /// boxes with the new offsets applied. Layout is NOT recomputed:
    /// offsets only shift already-laid-out descendants. The offset
    /// clamps to the container's current maximum — offsets
    /// re-applied after a resize (a fresh rebuild or an incremental
    /// re-layout) must not survive past the new maximum. An id with
    /// no box has no measurable maximum, so its offset is stored
    /// as given.
    pub fn set_scroll_offset(&mut self, id: DomId, x: f32, y: f32) {
        let (x, y) = match self.scrollable_size(id) {
            Some((max_x, max_y)) => (x.clamp(0.0, max_x), y.clamp(0.0, max_y)),
            None => (x, y),
        };
        self.scroll_offsets.insert(id, (x, y));
        self.recollect();
    }

    /// Scroll containers in this tree: `(dom id, border-box rect)`.
    pub fn scroll_containers(&self) -> Vec<(DomId, [f32; 4])> {
        self.boxes
            .iter()
            .filter_map(|b| {
                let dom = b.dom?;
                if b.style.overflow_x.is_scroll_container()
                    || b.style.overflow_y.is_scroll_container()
                {
                    Some((dom, [b.x, b.y, b.width, b.height]))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Taffy's own dirty hook (R20): mark one node's taffy layout
    /// dirty. M2 always rebuilds the whole tree in `build`, so this
    /// is a hook for Stage 7's incremental path, not a re-layout
    /// trigger — it returns `false` when the node has no taffy node.
    /// `Dom` has no dirty bits yet (see the module docs), so there is
    /// nothing on the DOM side to consume here.
    pub fn mark_dirty(&mut self, id: DomId) -> bool {
        match self.dom_to_taffy.get(&id) {
            Some(&node) => self.taffy.mark_dirty(node).is_ok(),
            None => false,
        }
    }
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

/// One stacking context's span of the Appendix E walk (M6 Phase
/// 2, Rec 4): the context root's own box first, then its members
/// in the step order documented on [`LayoutTree::paint_order`].
/// `members` maps a context root's box index to its member box
/// indices, in tree order. Recursion is over stacking contexts
/// only — a member that is itself a context recurses, a plain
/// member is emitted once, in its step's pass.
fn paint_context(
    ctx: usize,
    members: &HashMap<usize, Vec<usize>>,
    boxes: &[LayoutBox],
    out: &mut Vec<usize>,
) {
    out.push(ctx);
    let Some(ctx_members) = members.get(&ctx) else {
        return;
    };
    let mut neg: Vec<usize> = Vec::new();
    let mut inflow: Vec<usize> = Vec::new();
    let mut pos_auto: Vec<usize> = Vec::new();
    let mut sc_auto0: Vec<usize> = Vec::new();
    let mut pos: Vec<usize> = Vec::new();
    for &m in ctx_members {
        let bx = &boxes[m];
        if bx.is_stacking_context {
            match bx.style.z_index {
                Some(z) if z < 0 => neg.push(m),
                Some(z) if z > 0 => pos.push(m),
                _ => sc_auto0.push(m),
            }
        } else if bx.style.position == PositionKind::Static {
            inflow.push(m);
        } else {
            // Positioned, `z-index: auto`, not a stacking
            // context: Appendix E step 8, in tree order.
            pos_auto.push(m);
        }
    }
    // Negative, then positive `z-index`: `z-index` order, ties
    // broken by tree order (box index order is tree order).
    neg.sort_by_key(|&m| (boxes[m].style.z_index.unwrap_or(0), m));
    pos.sort_by_key(|&m| (boxes[m].style.z_index.unwrap_or(0), m));
    for m in neg {
        paint_context(m, members, boxes, out);
    }
    out.extend(inflow);
    // Step 8: positioned `auto` members and `auto`/`0` stacking
    // contexts interleaved in tree order.
    let mut step8: Vec<usize> = pos_auto;
    step8.extend(sc_auto0);
    step8.sort_unstable();
    for m in step8 {
        if boxes[m].is_stacking_context {
            paint_context(m, members, boxes, out);
        } else {
            out.push(m);
        }
    }
    for m in pos {
        paint_context(m, members, boxes, out);
    }
}

/// Depth-first search for the first element with this tag.
fn find_element(dom: &Dom, id: DomId, tag: &str) -> Option<DomId> {
    for &child in dom.children(id) {
        if dom.tag(child) == Some(tag) {
            return Some(child);
        }
        if let Some(found) = find_element(dom, child, tag) {
            return Some(found);
        }
    }
    None
}

struct Builder<'a, 't> {
    dom: &'a Dom,
    /// Pre-resolved styles for every element (two-phase pass, see
    /// `resolve_document_styles`). `Arc`-shared: computed styles are
    /// immutable once resolved, the Stylo pattern.
    styles: HashMap<DomId, Arc<ComputedStyle>>,
    /// Borrowed from the caller: `build_with_state` passes a fresh
    /// tree, the incremental path passes the retained tree, so a
    /// subtree rebuild reuses the live taffy nodes instead of
    /// rebuilding the world (Stage 8 Rec 6).
    taffy: &'t mut TaffyTree<NodeContext>,
    meta: &'t mut HashMap<TaffyId, BoxMeta>,
    /// Text-leaf taffy nodes, in build order (pre-shape input).
    leaves: &'t mut Vec<TaffyId>,
    viewport: (f32, f32),
    /// Images decoded so far (Stage 11, M3): `<img>` intrinsic
    /// sizes + broken-image set + background-image resources.
    images: &'a ImageMap,
}

impl<'a, 't> Builder<'a, 't> {
    /// Style for `id` from the pre-resolved map. Falls back to the
    /// parent style if the id was never collected — e.g. a node
    /// outside the `<html>` subtree — so behaviour stays identical;
    /// in practice the map covers every element the builder visits.
    fn style_for(&self, id: DomId, parent_style: &ComputedStyle) -> ComputedStyle {
        match self.styles.get(&id) {
            Some(style) => (**style).clone(),
            None => parent_style.clone(),
        }
    }

    /// Build a taffy node for one element (block / flex / grid /
    /// flow-root). Returns `None` for `display: none`, for inline
    /// elements (handled by the parent's inline pass), and for
    /// inline-block / images (atomic boxes, handled by the parent).
    fn build_node(
        &mut self,
        id: DomId,
        parent_style: &ComputedStyle,
    ) -> Option<TaffyId> {
        // Stage 15 (M7 Phase 1, Rec 2): an `<svg>` / `<canvas>` with
        // an author `display: block` is still a replaced-element
        // leaf — its children must never build (no boxes, no leaked
        // SVG `<text>`). Stage 16 Phase 2 (Rec 5): the same for
        // `<video>` / `<audio>` — a video is a replaced leaf
        // whatever its display, and audio takes no space at all.
        if is_svg_element(self.dom, id)
            || is_canvas_element(self.dom, id)
            || is_video_element(self.dom, id)
            || is_audio_element(self.dom, id)
        {
            let style = self.style_for(id, parent_style);
            return self.build_atomic(id, &style);
        }
        let style = self.style_for(id, parent_style);
        match style.display {
            DisplayKind::None | DisplayKind::Inline | DisplayKind::InlineBlock => None,
            _ => {
                let children = self.build_children(id, &style);
                let taffy_style = to_taffy_style(&style, self.viewport);
                let node = if children.is_empty() {
                    self.taffy
                        .new_leaf_with_context(taffy_style, None)
                        .ok()?
                } else {
                    self.taffy.new_with_children(taffy_style, &children).ok()?
                };
                self.meta.insert(
                    node,
                    BoxMeta {
                        dom: Some(id),
                        style,
                        children,
                        text: None,
                        svg_canvas: None,
                    },
                );
                Some(node)
            }
        }
    }

    /// Build a block container's children: block-level children become
    /// taffy nodes; consecutive text nodes and inline elements merge
    /// into anonymous text leaves (one leaf per run of inline content
    /// — the anonymous block boxes of R10); atomic inline-level boxes
    /// (`inline-block`, `img`) become their own leaves, sized by
    /// content. `<br>` contributes a forced line break to the current
    /// inline run; it never becomes a taffy leaf.
    fn build_children(&mut self, parent: DomId, parent_style: &ComputedStyle) -> Vec<TaffyId> {
        let mut out = Vec::new();
        let mut spans: Vec<TextSpan> = Vec::new();
        for &child in self.dom.children(parent) {
            if let Some(text) = self.dom.text(child) {
                if let Some(processed) =
                    process_text(text, parent_style.white_space)
                {
                    spans.push(TextSpan {
                        text: processed,
                        color: parent_style.color,
                        bold: parent_style.bold,
                    });
                }
                continue;
            }
            let Some(tag) = self.dom.tag(child) else {
                continue;
            };
            if tag == "br" {
                // Forced line break in the enclosing inline run.
                spans.push(TextSpan {
                    text: "\n".to_string(),
                    color: parent_style.color,
                    bold: false,
                });
                continue;
            }
            let style = self.style_for(child, parent_style);
            match style.display {
                DisplayKind::None => {}
                DisplayKind::Inline => {
                    self.append_inline(child, &style, parent_style, &mut spans, &mut out);
                }
                DisplayKind::InlineBlock => {
                    self.flush_spans(&mut spans, parent_style, &mut out);
                    if let Some(node) = self.build_atomic(child, &style) {
                        out.push(node);
                    }
                }
                _ => {
                    self.flush_spans(&mut spans, parent_style, &mut out);
                    if let Some(node) = self.build_node(child, parent_style) {
                        out.push(node);
                    }
                }
            }
        }
        self.flush_spans(&mut spans, parent_style, &mut out);
        out
    }

    /// Recursively collect an inline element's text into `spans`. A
    /// block-level descendant flushes the spans and becomes a sibling
    /// block child instead (browsers split inline boxes the same way).
    /// M6 Phase 3 (Rec 2): a positioned inline element
    /// (`position: absolute` / `fixed`) does NOT fold its text into
    /// the parent leaf — it gets its own box, like a block child.
    /// (A `relative` inline stays inline: it lays out in flow and
    /// only offsets.)
    fn append_inline(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
        parent_style: &ComputedStyle,
        spans: &mut Vec<TextSpan>,
        out: &mut Vec<TaffyId>,
    ) {
        if matches!(
            style.position,
            PositionKind::Absolute | PositionKind::Fixed
        ) {
            self.flush_spans(spans, parent_style, out);
            if let Some(node) = self.build_positioned_inline(id, style) {
                out.push(node);
            }
            return;
        }
        if self.dom.tag(id) == Some("img")
            || is_svg_element(self.dom, id)
            || is_canvas_element(self.dom, id)
            || is_video_element(self.dom, id)
            || is_audio_element(self.dom, id)
        {
            // Replaced element (Stage 11 Rec 2 for `<img>`, Stage
            // 15 Rec 2 for `<svg>` / `<canvas>`, Stage 16 Rec 5
            // for `<video>` / `<audio>`): an `<img>` is atomic
            // even though the UA sheet says `display: inline` —
            // called here as the inline element itself, it has no
            // text children to collect, so it becomes its own leaf
            // instead of vanishing. Same for SVG / canvas / video
            // / audio, whose children must never contribute inline
            // text (an author `display: inline` on a `<video>`
            // must not re-open the leak the UA `inline-block`
            // rule closes).
            self.flush_spans(spans, parent_style, out);
            if let Some(node) = self.build_atomic(id, style) {
                out.push(node);
            }
            return;
        }
        for &child in self.dom.children(id) {
            if let Some(text) = self.dom.text(child) {
                if let Some(processed) = process_text(text, style.white_space) {
                    spans.push(TextSpan {
                        text: processed,
                        color: style.color,
                        bold: style.bold,
                    });
                }
                continue;
            }
            let Some(tag) = self.dom.tag(child) else {
                continue;
            };
            if tag == "br" {
                spans.push(TextSpan {
                    text: "\n".to_string(),
                    color: style.color,
                    bold: false,
                });
                continue;
            }
            let child_style = self.style_for(child, style);
            match child_style.display {
                DisplayKind::None => {}
                DisplayKind::Inline => {
                    if tag == "img"
                        || is_svg_element(self.dom, child)
                        || is_canvas_element(self.dom, child)
                        || is_video_element(self.dom, child)
                        || is_audio_element(self.dom, child)
                    {
                        // Replaced element (Stage 11 Rec 2 for
                        // `<img>`, Stage 15 Rec 2 for `<svg>` /
                        // `<canvas>`, Stage 16 Rec 5 for `<video>`
                        // / `<audio>`): an `<img>` is atomic even
                        // though the UA sheet says `display: inline`
                        // — it never contributes inline text, so
                        // recursing into it (as for a span) would
                        // drop it entirely. Same for SVG / canvas /
                        // video / audio, whose children must never
                        // leak text.
                        self.flush_spans(spans, parent_style, out);
                        if let Some(node) = self.build_atomic(child, &child_style) {
                            out.push(node);
                        }
                    } else {
                        self.append_inline(child, &child_style, parent_style, spans, out);
                    }
                }
                DisplayKind::InlineBlock => {
                    self.flush_spans(spans, parent_style, out);
                    if let Some(node) = self.build_atomic(child, &child_style) {
                        out.push(node);
                    }
                }
                _ => {
                    self.flush_spans(spans, parent_style, out);
                    if let Some(node) = self.build_node(child, style) {
                        out.push(node);
                    }
                }
            }
        }
    }

    /// Build a taffy node for a positioned inline element (M6 Phase
    /// 3, Rec 2): like [`Builder::build_node`]'s block arm, but
    /// reached for `display: inline` elements whose position takes
    /// them out of flow. Its children build the same way a block's
    /// do (inline runs merge into text leaves, block children
    /// become nodes); `to_taffy_style` already maps `display:
    /// inline` to a block display for taffy, so the node lays out
    /// as a box, not a run.
    fn build_positioned_inline(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
    ) -> Option<TaffyId> {
        let children = self.build_children(id, style);
        let taffy_style = to_taffy_style(style, self.viewport);
        let node = if children.is_empty() {
            self.taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?
        } else {
            self.taffy.new_with_children(taffy_style, &children).ok()?
        };
        self.meta.insert(
            node,
            BoxMeta {
                dom: Some(id),
                style: style.clone(),
                children,
                text: None,
                svg_canvas: None,
            },
        );
        Some(node)
    }

    /// An atomic leaf for an `<svg>` / `<canvas>` replaced element
    /// (Stage 15, M7 Phase 1, Rec 2): a taffy leaf sized by the
    /// intrinsic dimensions — SVG from `width` / `height` /
    /// `viewBox` (+ `preserveAspectRatio`), canvas from its
    /// `width` / `height` attributes with the 300×150 default —
    /// with CSS `width` / `height` winning over both, exactly like
    /// the `<img>` branch. The `aspect_ratio` lets taffy derive
    /// the auto axis from the set one instead of collapsing it.
    /// The leaf carries `svg_canvas` in its `BoxMeta` so
    /// `collect_boxes` can mark the `LayoutBox`; it draws nothing
    /// yet (paint is a later phase), and its children never build.
    fn replaced_svg_canvas_leaf(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
        kind: SvgCanvasKind,
    ) -> Option<TaffyId> {
        let (w, h, ratio) = match kind {
            SvgCanvasKind::Svg => svg_intrinsic_size(self.dom, id),
            SvgCanvasKind::Canvas => {
                let w = parse_svg_attr_length(self.dom.attr(id, "width")).unwrap_or(300.0);
                let h = parse_svg_attr_length(self.dom.attr(id, "height")).unwrap_or(150.0);
                (w, h, Some(w / h))
            }
            // `<video>` never routes here (`build_atomic` has
            // its own branch, with the HTML width/height
            // attribute scaling the `<img>` branch does); the
            // arm exists for exhaustiveness and answers with
            // the same size the branch would.
            SvgCanvasKind::Video => {
                let (w, h) = self
                    .images
                    .videos
                    .get(&id)
                    .map(|v| (v.width as f32, v.height as f32))
                    .unwrap_or((300.0, 150.0));
                (w, h, Some(w / h))
            }
        };
        let mut taffy_style = to_taffy_style(style, self.viewport);
        if w > 0.0 && h > 0.0
            && let Some(r) = ratio {
                taffy_style.aspect_ratio = Some(r);
            }
        if style.width.is_none() && style.height.is_none() {
            taffy_style.size.width = Dimension::length(w);
            taffy_style.size.height = Dimension::length(h);
        }
        // Stage 15 (M7 Phase 3, Rec 10): `<foreignObject>` children
        // get real boxes (see `build_foreign_object_children`); the
        // svg node becomes a taffy parent (position relative) so
        // its absolute foreignObject children anchor to it. An svg
        // without foreignObjects stays a leaf, exactly as before.
        let fo_children = if kind == SvgCanvasKind::Svg {
            self.build_foreign_object_children(id, style, (w, h))
        } else {
            Vec::new()
        };
        let node = if fo_children.is_empty() {
            self.taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?
        } else {
            taffy_style.position = Position::Relative;
            self.taffy
                .new_with_children(taffy_style, &fo_children)
                .ok()?
        };
        self.meta.insert(
            node,
            BoxMeta {
                dom: Some(id),
                style: style.clone(),
                children: fo_children,
                text: None,
                svg_canvas: Some(kind),
            },
        );
        Some(node)
    }

    /// Stage 15 (M7 Phase 3, Rec 10): the `<foreignObject>`
    /// elements of an `<svg>` (top-level only — not nested inside
    /// another foreignObject) as taffy nodes, each a real viewport
    /// for its HTML children: sized/positioned by its `x` / `y` /
    /// `width` / `height` attributes, `overflow: clip` so its
    /// children paint inside the rect and never outside it (the
    /// clip rides `collect_boxes` like any overflow clip). `%`
    /// lengths resolve against the svg's intrinsic size — the
    /// used size is not known at build time (documented
    /// approximation; px values are exact). A foreignObject
    /// nested inside HTML content is out of scope.
    fn build_foreign_object_children(
        &mut self,
        svg_id: DomId,
        svg_style: &ComputedStyle,
        intrinsic: (f32, f32),
    ) -> Vec<TaffyId> {
        let mut fos = Vec::new();
        self.collect_foreign_objects(svg_id, &mut fos);
        fos.into_iter()
            .filter_map(|fo| self.build_foreign_object(fo, svg_style, intrinsic))
            .collect()
    }

    /// Collect top-level `<foreignObject>` descendants of `id`
    /// (recursing into SVG grouping elements only).
    fn collect_foreign_objects(&self, id: DomId, out: &mut Vec<DomId>) {
        for &child in self.dom.children(id) {
            if self.dom.tag(child) == Some("foreignObject") {
                out.push(child);
                continue;
            }
            if matches!(
                self.dom.tag(child),
                Some("g" | "svg" | "a" | "switch")
            ) {
                self.collect_foreign_objects(child, out);
            }
        }
    }

    /// One `<foreignObject>` as an absolute taffy node inside its
    /// `<svg>`: its HTML children build inside it (via
    /// `build_children`, so text, divs, and nested replaced
    /// elements all lay out normally, at the foreignObject's
    /// origin).
    fn build_foreign_object(
        &mut self,
        id: DomId,
        parent_style: &ComputedStyle,
        intrinsic: (f32, f32),
    ) -> Option<TaffyId> {
        let mut style = self.style_for(id, parent_style);
        style.overflow_x = OverflowKind::Clip;
        style.overflow_y = OverflowKind::Clip;
        let len = |dom: &Dom, name: &str, basis: f32, default: f32| -> f32 {
            dom.attr(id, name)
                .and_then(|v| {
                    let s = v.trim();
                    if let Some(num) = s.strip_suffix('%') {
                        num.trim().parse::<f32>().ok().map(|n| n / 100.0 * basis)
                    } else {
                        parse_svg_attr_length(Some(s))
                    }
                })
                .unwrap_or(default)
        };
        let x = len(self.dom, "x", intrinsic.0, 0.0);
        let y = len(self.dom, "y", intrinsic.1, 0.0);
        let w = len(self.dom, "width", intrinsic.0, 0.0);
        let h = len(self.dom, "height", intrinsic.1, 0.0);
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        let children = self.build_children(id, &style);
        let mut taffy_style = to_taffy_style(&style, self.viewport);
        taffy_style.position = Position::Absolute;
        taffy_style.inset.left = LengthPercentageAuto::length(x);
        taffy_style.inset.top = LengthPercentageAuto::length(y);
        taffy_style.size.width = Dimension::length(w);
        taffy_style.size.height = Dimension::length(h);
        let node = if children.is_empty() {
            self.taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?
        } else {
            self.taffy
                .new_with_children(taffy_style, &children)
                .ok()?
        };
        self.meta.insert(
            node,
            BoxMeta {
                dom: Some(id),
                style,
                children,
                text: None,
                svg_canvas: None,
            },
        );
        Some(node)
    }

    /// An atomic inline-level box (`display: inline-block`, or an
    /// `<img>`): a taffy leaf sized by its content. Its inline text
    /// merges into one leaf (same shaping as a block's text leaf); a
    /// block-level descendant falls back to a full node. `<img>` has
    /// no text, so it is a zero-size leaf unless width/height styles
    /// size it (image decoding itself is M3).
    fn build_atomic(&mut self, id: DomId, style: &ComputedStyle) -> Option<TaffyId> {
        let tag = self.dom.tag(id).unwrap_or("");
        // Stage 15 (M7 Phase 1, Rec 2): `<svg>` and `<canvas>` are
        // replaced elements — atomic leaves, sized by their intrinsic
        // dimensions. Their children NEVER build: SVG shapes get no
        // boxes, and `<svg><text>` must not leak its text into the
        // parent's anonymous text leaf (the generic tail below would
        // recurse into the children through `append_inline`).
        if is_svg_element(self.dom, id) {
            return self.replaced_svg_canvas_leaf(id, style, SvgCanvasKind::Svg);
        }
        if is_canvas_element(self.dom, id) {
            return self.replaced_svg_canvas_leaf(id, style, SvgCanvasKind::Canvas);
        }
        if tag == "img" {
            // Stage 11 Rec 2: HTML sizing rules. A broken image
            // with alt text renders as a text leaf (the alt text,
            // in the image's style); broken without alt is a
            // zero-size leaf, as before.
            if self.images.broken.contains(&id)
                && let Some(alt) = self.dom.attr(id, "alt").filter(|a| !a.trim().is_empty()) {
                    let leaf = LeafText {
                        spans: vec![TextSpan {
                            text: alt.to_string(),
                            color: style.color,
                            bold: style.bold,
                        }],
                        font_size: style.font_size,
                        line_height: style.line_height,
                        shaped: None,
                    };
                    let node = self
                        .taffy
                        .new_leaf_with_context(to_taffy_style(style, self.viewport), Some(leaf.clone()))
                        .ok()?;
                    self.meta.insert(
                        node,
                        BoxMeta {
                            dom: Some(id),
                            style: style.clone(),
                            children: Vec::new(),
                            text: Some(leaf),
                            svg_canvas: None,
                        },
                    );
                    self.leaves.push(node);
                    return Some(node);
                }
            // Intrinsic size (decoded) or the 300x150 default when
            // the image has not arrived yet; width/height
            // attributes override one axis and scale the other by
            // the intrinsic ratio; CSS width/height win over both.
            let (iw, ih) = self
                .images
                .by_node
                .get(&id)
                .map(|info| (info.width as f32, info.height as f32))
                .unwrap_or((300.0, 150.0));
            let mut w = iw;
            let mut h = ih;
            let attr_w = self
                .dom
                .attr(id, "width")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| *v > 0.0);
            let attr_h = self
                .dom
                .attr(id, "height")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| *v > 0.0);
            if let Some(aw) = attr_w {
                w = aw;
                if attr_h.is_none() && iw > 0.0 {
                    h = ih * (aw / iw);
                }
            }
            if let Some(ah) = attr_h {
                h = ah;
                if attr_w.is_none() && ih > 0.0 {
                    w = iw * (ah / ih);
                }
            }
            let mut taffy_style = to_taffy_style(style, self.viewport);
            if self.images.broken.contains(&id)
                && style.width.is_none()
                && style.height.is_none()
            {
                // Broken image with no alt text (the alt case
                // returned above): a zero-size leaf. CSS width /
                // height still win when set.
                taffy_style.size.width = Dimension::length(0.0);
                taffy_style.size.height = Dimension::length(0.0);
            } else {
                // `aspect_ratio` lets taffy derive the auto axis from
                // the set one (CSS width with auto height, and vice
                // versa) instead of collapsing it to zero.
                if w > 0.0 && h > 0.0 {
                    taffy_style.aspect_ratio = Some(w / h);
                }
                if style.width.is_none() && style.height.is_none() {
                    taffy_style.size.width = Dimension::length(w);
                    taffy_style.size.height = Dimension::length(h);
                }
            }
            let node = self
                .taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?;
            self.meta.insert(
                node,
                BoxMeta {
                    dom: Some(id),
                    style: style.clone(),
                    children: Vec::new(),
                    text: None,
                    svg_canvas: None,
                },
            );
            return Some(node);
        }
        // Stage 16 Phase 2 (Rec 5): `<video>` is a replaced-
        // element leaf, sized by the decoded intrinsic size
        // once HAVE_METADATA, 300×150 before it (mirroring
        // `<img>`); width/height attributes override one axis
        // and scale the other by the intrinsic ratio; CSS
        // width/height win over both. Fallback children never
        // build — the leaf replaces them, like `<img>`. The
        // `Video` marker on the BoxMeta tells paint this box
        // draws a frame Blob / poster / black.
        if tag == "video" {
            let (iw, ih) = self
                .images
                .videos
                .get(&id)
                .map(|info| (info.width as f32, info.height as f32))
                .unwrap_or((300.0, 150.0));
            let mut w = iw;
            let mut h = ih;
            let attr_w = self
                .dom
                .attr(id, "width")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| *v > 0.0);
            let attr_h = self
                .dom
                .attr(id, "height")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| *v > 0.0);
            if let Some(aw) = attr_w {
                w = aw;
                if attr_h.is_none() && iw > 0.0 {
                    h = ih * (aw / iw);
                }
            }
            if let Some(ah) = attr_h {
                h = ah;
                if attr_w.is_none() && ih > 0.0 {
                    w = iw * (ah / ih);
                }
            }
            let mut taffy_style = to_taffy_style(style, self.viewport);
            if w > 0.0 && h > 0.0 {
                taffy_style.aspect_ratio = Some(w / h);
            }
            if style.width.is_none() && style.height.is_none() {
                taffy_style.size.width = Dimension::length(w);
                taffy_style.size.height = Dimension::length(h);
            }
            let node = self
                .taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?;
            self.meta.insert(
                node,
                BoxMeta {
                    dom: Some(id),
                    style: style.clone(),
                    children: Vec::new(),
                    text: None,
                    svg_canvas: Some(SvgCanvasKind::Video),
                },
            );
            return Some(node);
        }
        // Stage 16 Phase 2 (Rec 5): `<audio>` renders as a
        // zero-size leaf — no visual at all — unless controls
        // exist, and the controls UI is deferred (M8 does not
        // build it). Forcing 0×0 here, whatever the style
        // says, is the honest shape of that deferral.
        if tag == "audio" {
            let mut taffy_style = to_taffy_style(style, self.viewport);
            taffy_style.size.width = Dimension::length(0.0);
            taffy_style.size.height = Dimension::length(0.0);
            let node = self
                .taffy
                .new_leaf_with_context(taffy_style, None)
                .ok()?;
            self.meta.insert(
                node,
                BoxMeta {
                    dom: Some(id),
                    style: style.clone(),
                    children: Vec::new(),
                    text: None,
                    svg_canvas: None,
                },
            );
            return Some(node);
        }
        // Form controls (Stage 13 Rec 4): a control's leaf shows
        // its LIVE value from the `Dom` form state, not its static
        // children — typing must repaint, and a `<select>` paints
        // only the selected option's text (the generic inline path
        // below would concatenate every option). Value-driven
        // damage clears the taffy cache on the node via the
        // renderer's dirty set (trap 18, Stage 12 Rec 4/6): the
        // leaf is rebuilt from the `Dom` on every dirty re-layout,
        // so a stale measurement can never survive a value change.
        if tag == "select" {
            let options = self.dom.select_options(id);
            let idx = self.dom.form_selected_index(id).unwrap_or(-1);
            let text = if idx >= 0 && (idx as usize) < options.len() {
                let opt = options[idx as usize];
                self.dom
                    .descendant_text(opt)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                String::new()
            };
            return Some(self.control_text_leaf(id, style, &text));
        }
        if tag == "textarea" {
            let text = self.dom.form_value(id).unwrap_or_default();
            return Some(self.control_text_leaf(id, style, &text));
        }
        if tag == "input" {
            let ty = self.dom.input_type(id);
            match ty.as_str() {
                "checkbox" | "radio" | "hidden" | "submit" | "button" | "reset"
                | "image" | "file" => {}
                "password" => {
                    let len = self.dom.form_value(id).map(|v| v.chars().count()).unwrap_or(0);
                    let masked = "\u{2022}".repeat(len);
                    return Some(self.control_text_leaf(id, style, &masked));
                }
                _ => {
                    let mut text = self.dom.form_value(id).unwrap_or_default();
                    if text.is_empty() {
                        text = self
                            .dom
                            .attr(id, "placeholder")
                            .unwrap_or("")
                            .to_string();
                    }
                    return Some(self.control_text_leaf(id, style, &text));
                }
            }
        }
        let mut spans: Vec<TextSpan> = Vec::new();
        let mut out: Vec<TaffyId> = Vec::new();
        self.append_inline(id, style, style, &mut spans, &mut out);
        if !out.is_empty() {
            // Block content inside an inline-block: fall back to a
            // full node (the inline run is discarded — M2 limitation,
            // noted in the report).
            return self.build_node(id, style);
        }
        if spans.is_empty() {
            let node = self
                .taffy
                .new_leaf_with_context(to_taffy_style(style, self.viewport), None)
                .ok()?;
            self.meta.insert(
                node,
                BoxMeta {
                    dom: Some(id),
                    style: style.clone(),
                    children: Vec::new(),
                    text: None,
                    svg_canvas: None,
                },
            );
            return Some(node);
        }
        let leaf = LeafText {
            spans,
            font_size: style.font_size,
            line_height: style.line_height,
            shaped: None,
        };
        let node = self
            .taffy
            .new_leaf_with_context(to_taffy_style(style, self.viewport), Some(leaf.clone()))
            .ok()?;
        self.meta.insert(
            node,
            BoxMeta {
                dom: Some(id),
                style: style.clone(),
                children: Vec::new(),
                text: Some(leaf),
                svg_canvas: None,
            },
        );
        self.leaves.push(node);
        Some(node)
    }

    /// One text leaf for a form control (Stage 13 Rec 4): the
    /// control's live value (or the selected option's text, or the
    /// placeholder) as a single span. Empty text still makes a
    /// leaf — the UA sheet gives the control its intrinsic size
    /// (min-width / border), so an unstyled `<input>` is a visible
    /// box, not a ~2px sliver.
    fn control_text_leaf(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
        text: &str,
    ) -> TaffyId {
        let leaf = if text.is_empty() {
            None
        } else {
            Some(LeafText {
                spans: vec![TextSpan {
                    text: text.to_string(),
                    color: style.color,
                    bold: style.bold,
                }],
                font_size: style.font_size,
                line_height: style.line_height,
                shaped: None,
            })
        };
        let has_text = leaf.is_some();
        let node = self
            .taffy
            .new_leaf_with_context(to_taffy_style(style, self.viewport), leaf.clone())
            .expect("taffy insert cannot fail");
        self.meta.insert(
            node,
            BoxMeta {
                dom: Some(id),
                style: style.clone(),
                children: Vec::new(),
                text: leaf,
                svg_canvas: None,
            },
        );
        if has_text {
            self.leaves.push(node);
        }
        node
    }

    /// Turn accumulated inline spans into a taffy text leaf (an
    /// anonymous block box wrapping one run of inline content).
    fn flush_spans(
        &mut self,
        spans: &mut Vec<TextSpan>,
        parent_style: &ComputedStyle,
        out: &mut Vec<TaffyId>,
    ) {
        if spans.is_empty() {
            return;
        }
        let leaf = LeafText {
            spans: std::mem::take(spans),
            font_size: parent_style.font_size,
            line_height: parent_style.line_height,
            shaped: None,
        };
        let node = self
            .taffy
            .new_leaf_with_context(Style::default(), Some(leaf.clone()))
            .expect("taffy insert cannot fail");
        self.meta.insert(
            node,
            BoxMeta {
                dom: None,
                style: parent_style.clone(),
                children: Vec::new(),
                text: Some(leaf),
                svg_canvas: None,
            },
        );
        self.leaves.push(node);
        out.push(node);
    }
}

/// Apply `white-space` to one text node before shaping. `normal`
/// collapses every whitespace run to a single space (and drops
/// whitespace-only nodes); `pre` keeps the text verbatim;
/// `nowrap`/`pre-wrap`/`pre-line` are stored data in M2 — wrapping
/// still happens, so they collapse like `normal` here except that
/// `pre-line` keeps newlines. Returns `None` for text that contributes
/// nothing.
fn process_text(text: &str, ws: WhiteSpace) -> Option<String> {
    match ws {
        WhiteSpace::Pre => {
            if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            }
        }
        WhiteSpace::PreLine => {
            // Collapse spaces/tabs, keep newlines.
            let mut out = String::with_capacity(text.len());
            let mut in_space = false;
            for c in text.chars() {
                if c == '\n' {
                    out.push('\n');
                    in_space = false;
                } else if c == ' ' || c == '\t' {
                    if !in_space {
                        out.push(' ');
                        in_space = true;
                    }
                } else {
                    out.push(c);
                    in_space = false;
                }
            }
            if out.trim().is_empty() {
                None
            } else {
                Some(out)
            }
        }
        WhiteSpace::Normal | WhiteSpace::Nowrap | WhiteSpace::PreWrap => {
            // Collapse all whitespace runs to a single space; drop
            // whitespace-only nodes (their spaces collapse away at
            // line boundaries in M2's simple model).
            let collapsed: String = text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if collapsed.is_empty() {
                None
            } else {
                Some(collapsed)
            }
        }
    }
}

// ---- R11 pre-shaping ----

// One rayon worker's `FontSystem`, created lazily per pool thread.
// `FontSystem::new()` loads the system fonts (see the module docs);
// every worker therefore sees the same font set as the owner thread.
// Webfonts must be injected into every instance via
// `FontSystem::new_with_fonts` from shared `fontdb::Source`s — no
// webfonts are loaded anywhere yet, so plain `new()` is consistent.
thread_local! {
    static WORKER_FONT_SYSTEM: RefCell<FontSystem> = RefCell::new(FontSystem::new());
}

fn with_worker_font_system<R>(f: impl FnOnce(&mut FontSystem) -> R) -> R {
    WORKER_FONT_SYSTEM.with(|fs| f(&mut fs.borrow_mut()))
}

/// Serialising `FontSystem` for single-threaded / test paths ONLY.
/// Never used in production layout: it would serialise all shaping
/// across tabs for no benefit (the measure path is already serial).
/// Kept so a debug harness can shape without a rayon pool.
#[allow(dead_code)]
pub struct SharedFontSystem(pub std::sync::Mutex<FontSystem>);

#[allow(dead_code)]
impl SharedFontSystem {
    pub fn new() -> Self {
        SharedFontSystem(std::sync::Mutex::new(FontSystem::new()))
    }

    pub fn shape_text(&self, leaf: &LeafText) -> ShapedText {
        let mut fs = self.0.lock().expect("font system lock");
        shape_text(&mut fs, leaf)
    }
}

impl Default for SharedFontSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Shape one leaf's text once, at unconstrained width, into plain
/// `Send` data. Word advances come from the shaped glyphs' `x`/`w`
/// (glyph clusters carry byte offsets into the run text, so word
/// boundaries are exact); `max_content` is the widest run's `line_w`.
fn shape_text(font_system: &mut FontSystem, leaf: &LeafText) -> ShapedText {
    let full_text: String = leaf.spans.iter().map(|s| s.text.as_str()).collect();
    let line_count = full_text.split('\n').count().max(1);
    let mut segments: Vec<Vec<WordAdvance>> = vec![Vec::new(); line_count];
    let mut space_width = leaf.font_size * 0.25;
    let mut max_content = 0.0f32;
    if !full_text.is_empty() {
        let mut buffer = Buffer::new(
            font_system,
            Metrics::new(leaf.font_size, leaf.line_height),
        );
        set_buffer_text(&mut buffer, leaf);
        buffer.set_size(None, None);
        buffer.shape_until_scroll(font_system, false);
        for run in buffer.layout_runs() {
            max_content = max_content.max(run.line_w);
            let seg = &mut segments[run.line_i.min(line_count - 1)];
            let mut word_start_x: Option<f32> = None;
            let mut word_end_x = 0.0f32;
            let close_word = |seg: &mut Vec<WordAdvance>,
                                  word_start_x: &mut Option<f32>,
                                  word_end_x: f32,
                                  trailing_space: bool| {
                if let Some(sx) = word_start_x.take() {
                    seg.push(WordAdvance {
                        advance: (word_end_x - sx).max(0.0),
                        trailing_space,
                    });
                }
            };
            for g in run.glyphs {
                let cluster = run.text.get(g.start..g.end).unwrap_or("");
                if cluster.chars().all(|c| c == ' ' || c == '\t') {
                    // A space glyph ends the current word.
                    if g.w > 0.0 {
                        space_width = g.w;
                    }
                    close_word(seg, &mut word_start_x, word_end_x, true);
                } else {
                    if word_start_x.is_none() {
                        word_start_x = Some(g.x);
                    }
                    word_end_x = g.x + g.w;
                }
            }
            close_word(seg, &mut word_start_x, word_end_x, false);
        }
    }
    let min_content = segments
        .iter()
        .flatten()
        .map(|w| w.advance)
        .fold(0.0f32, f32::max);
    ShapedText {
        segments,
        space_width,
        min_content,
        max_content,
        line_height: leaf.line_height,
    }
}

/// Pre-shape pass (R11, option A): shape each *distinct* text run
/// once, in parallel over rayon's pool (one `FontSystem` per worker),
/// and store the plain `Send` result in the leaf contexts. Runs are
/// deduplicated by (text, font size, line height, bold pattern) so
/// repeated paragraphs shape once. Runs the cache keyed by that hash.
fn preshape_leaves(
    taffy: &mut TaffyTree<NodeContext>,
    meta: &mut HashMap<TaffyId, BoxMeta>,
    leaves: &[TaffyId],
) {
    // Deduplicate leaves by content.
    let mut unique: HashMap<u64, (LeafText, Vec<TaffyId>)> = HashMap::new();
    for &id in leaves {
        let Some(leaf) = meta.get(&id).and_then(|m| m.text.clone()) else {
            continue;
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for span in &leaf.spans {
            span.text.hash(&mut hasher);
            span.bold.hash(&mut hasher);
        }
        leaf.font_size.to_bits().hash(&mut hasher);
        leaf.line_height.to_bits().hash(&mut hasher);
        let key = hasher.finish();
        unique.entry(key).or_insert_with(|| (leaf, Vec::new())).1.push(id);
    }
    // Parallel shape: each worker owns its FontSystem (thread_local).
    let shaped: Vec<(Vec<TaffyId>, Arc<ShapedText>)> = unique
        .into_values()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(leaf, ids)| {
            let shaped = with_worker_font_system(|fs| shape_text(fs, &leaf));
            (ids, Arc::new(shaped))
        })
        .collect();
    // Write the shaped data back into the meta map and the taffy
    // node contexts (both must agree — measure reads the context).
    for (ids, shaped) in shaped {
        for id in ids {
            if let Some(m) = meta.get_mut(&id)
                && let Some(leaf) = m.text.as_mut() {
                    leaf.shaped = Some(shaped.clone());
                }
            if let Some(ctx) = taffy.get_node_context_mut(id)
                && let Some(leaf) = ctx.as_mut() {
                    leaf.shaped = Some(shaped.clone());
                }
        }
    }
}

/// Measure one text leaf from its pre-shaped data — pure arithmetic,
/// no cosmic-text calls, no locks. Taffy calls this whenever it needs
/// the leaf's size; wrapping is word-by-word at the available width.
fn measure_text(ctx: Option<&mut NodeContext>, input: LayoutInput) -> LayoutOutput {
    let leaf = match ctx {
        Some(c) => c.as_ref(),
        None => None,
    };
    let Some(leaf) = leaf else {
        // Non-text leaf (empty element): size from known dimensions, else 0.
        return LayoutOutput {
            size: Size {
                width: input.known_dimensions.width.unwrap_or(0.0),
                height: input.known_dimensions.height.unwrap_or(0.0),
            },
            ..LayoutOutput::HIDDEN
        };
    };
    let Some(shaped) = leaf.shaped.as_ref() else {
        // Pre-shaping did not run (direct test construction): fall
        // back to one strut line rather than failing layout.
        return LayoutOutput {
            size: Size {
                width: input.known_dimensions.width.unwrap_or(0.0),
                height: input.known_dimensions.height.unwrap_or(leaf.line_height),
            },
            ..LayoutOutput::HIDDEN
        };
    };
    let known_w = input.known_dimensions.width;
    let (width, height) = match known_w {
        Some(w) => (w, wrap_height(shaped, w)),
        None => match input.available_space.width {
            AvailableSpace::Definite(w) => (w, wrap_height(shaped, w)),
            AvailableSpace::MinContent => (
                shaped.min_content,
                wrap_height(shaped, shaped.min_content.max(1.0)),
            ),
            AvailableSpace::MaxContent => (
                shaped.max_content,
                shaped.segments.len() as f32 * shaped.line_height,
            ),
        },
    };
    LayoutOutput {
        size: Size { width, height },
        ..LayoutOutput::HIDDEN
    }
}

/// Height of the shaped text wrapped at `width`: each source-line
/// segment wraps independently, and every segment is at least one
/// line tall (a line box with no text still has the strut height).
fn wrap_height(shaped: &ShapedText, width: f32) -> f32 {
    let mut lines = 0usize;
    for segment in &shaped.segments {
        if segment.is_empty() {
            lines += 1;
            continue;
        }
        let mut line_count = 1usize;
        let mut line_w = 0.0f32;
        for (i, word) in segment.iter().enumerate() {
            let add = if i == 0 {
                word.advance
            } else {
                shaped.space_width + word.advance
            };
            if i > 0 && line_w + add > width && line_w > 0.0 {
                line_count += 1;
                line_w = word.advance;
            } else {
                line_w += add;
            }
        }
        lines += line_count;
    }
    lines as f32 * shaped.line_height
}

/// Fill a `Buffer` with a leaf's spans (single span: `set_text`,
/// several: `set_rich_text` so wrapping sees the whole paragraph).
fn set_buffer_text(buffer: &mut Buffer, leaf: &LeafText) {
    if leaf.spans.len() == 1 {
        let span = &leaf.spans[0];
        let attrs = Attrs::new().weight(if span.bold {
            Weight::BOLD
        } else {
            Weight::NORMAL
        });
        buffer.set_text(&span.text, &attrs, Shaping::Advanced, None);
    } else {
        let spans: Vec<(&str, Attrs)> = leaf
            .spans
            .iter()
            .map(|span| {
                (
                    span.text.as_str(),
                    Attrs::new().weight(if span.bold {
                        Weight::BOLD
                    } else {
                        Weight::NORMAL
                    }),
                )
            })
            .collect();
        buffer.set_rich_text(spans, &Attrs::new(), Shaping::Advanced, None);
    }
}

/// Shape one laid-out text box into glyphs at its final size, on the
/// owner thread with the tree's own `FontSystem`. The same inputs as
/// the pre-shape pass, so positions agree with layout. `text-align`
/// is applied when *placing* lines (R10): each run is offset within
/// the box width (center/right); `justify` renders as left in M2.
fn shape_leaf(font_system: &mut FontSystem, bx: &LayoutBox) -> Vec<ShapedGlyph> {
    let Some(leaf) = bx.text.as_ref() else {
        return Vec::new();
    };
    let mut buffer = Buffer::new(
        font_system,
        Metrics::new(leaf.font_size, leaf.line_height),
    );
    set_buffer_text(&mut buffer, leaf);
    buffer.set_size(Some(bx.width), None);
    // See `measure_text`: `layout_runs` does not shape on its own.
    buffer.shape_until_scroll(font_system, false);

    // Byte ranges of each span in the concatenated text, for color lookup.
    let mut ranges: Vec<(std::ops::Range<usize>, [u8; 3])> = Vec::new();
    let mut off = 0;
    for span in &leaf.spans {
        ranges.push((off..off + span.text.len(), span.color));
        off += span.text.len();
    }
    let full_text: String = leaf.spans.iter().map(|s| s.text.as_str()).collect();
    // Start byte of each '\n'-separated line in the concatenated text.
    let mut line_starts = vec![0usize];
    for (i, b) in full_text.bytes().enumerate() {
        if b == b'\n' {
            line_starts.push(i + 1);
        }
    }

    let mut glyphs = Vec::new();
    for run in buffer.layout_runs() {
        // `run.text` is the whole source line in cosmic-text 0.19,
        // not the visual line's substring — do not slice or search
        // it. Each glyph carries its own byte offset within the
        // source line (`LayoutGlyph.start`), which is what locates
        // the glyph for its span color.
        let align_dx = match bx.style.text_align {
            TextAlign::Left | TextAlign::Justify => 0.0,
            TextAlign::Center => (bx.width - run.line_w) / 2.0,
            TextAlign::Right => bx.width - run.line_w,
        };
        let line_start = line_starts.get(run.line_i).copied().unwrap_or(0);
        for g in run.glyphs {
            let abs = line_start + g.start;
            let color = ranges
                .iter()
                .find(|(r, _)| r.contains(&abs))
                .map(|(_, c)| *c)
                .unwrap_or([0, 0, 0]);
            let physical = g.physical((bx.x + align_dx, bx.y + run.line_y), 1.0);
            glyphs.push(ShapedGlyph {
                cache_key: physical.cache_key,
                x: physical.x as f32,
                y: physical.y as f32,
                color,
            });
        }
    }
    glyphs
}

/// Intersect two `[x, y, w, h]` rects (empty when they do not overlap).
fn intersect_clip(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    [x, y, (x2 - x).max(0.0), (y2 - y).max(0.0)]
}

/// Walk the taffy tree from `node`, collecting boxes with absolute
/// (page) coordinates. Scroll offsets of scroll-container ancestors
/// are subtracted as the walk descends (offsets are retained input
/// state, applied here — never recomputed by layout), and each box
/// records the effective clip rect from its nearest clipping
/// ancestor. `overflow: hidden|scroll|auto` clips to the padding box
/// (and makes a scroll container); `overflow: clip` clips the same
/// way but creates no scroll container; `overflow: visible` clips
/// nothing. Each box also resolves its own `border-radius` against
/// its box (see [`BorderRadius::resolve`]), which the display list
/// uses for rounded fills, shadows, and rounded clips.
#[allow(clippy::too_many_arguments)]
fn collect_boxes(
    taffy: &TaffyTree<NodeContext>,
    meta: &HashMap<TaffyId, BoxMeta>,
    node: TaffyId,
    offset: (f32, f32),
    // The same accumulation as `offset`, but with no scroll
    // offset ever subtracted (M6 Phase 3, Rec 1): a
    // viewport-fixed box's fallback position, and the base its
    // descendants accumulate from.
    offset_ns: (f32, f32),
    scroll_offsets: &HashMap<DomId, (f32, f32)>,
    parent_clip: Option<[f32; 4]>,
    parent_clip_radii: Option<ResolvedCorners>,
    parent_clip_source: Option<TaffyId>,
    parent_scroll_ancestor: Option<DomId>,
    viewport: (f32, f32),
    root_font_size: f32,
    // Position-only translations recorded by `translate_subtree`
    // (Stage 8 Rec 6): the taffy layouts are stale after a move,
    // so the override is the record of it. Each node's own override
    // is added to its position; children inherit it through `offset`,
    // and clips are computed from the translated rects, so an inner
    // clip moves with its subtree while an outer clip stays put.
    position_overrides: &HashMap<TaffyId, (f32, f32)>,
    // Scroll-derived sticky offsets (M6 Phase 3, Rec 3), keyed by
    // taffy node: added to the sticky subtree's position like a
    // position override. Never through taffy insets.
    sticky: &HashMap<TaffyId, (f32, f32)>,
    // Images decoded so far (Stage 11, M3): `<img>` resource ids
    // and background-image resource ids land on the boxes here.
    images: &ImageMap,
    // M6 Phase 1: the DOM id of the nearest ancestor box (None
    // for the root), the ancestor chain's accumulated transform,
    // and whether this node is a flex/grid item (its parent's
    // display is Flex/Grid) — the stacking-context predicate
    // needs the last one.
    parent_dom: Option<DomId>,
    parent_matrix: Affine,
    is_flex_or_grid_item: bool,
    // Whether any ancestor has a non-`none` transform (M6 Phase
    // 3, Rec 1): a transformed ancestor is the containing block
    // for `fixed` descendants, so they stop being
    // viewport-relative and fall back to the absolute path.
    ancestor_transformed: bool,
    out: &mut Vec<LayoutBox>,
) {
    let Some(layout) = taffy.layout(node).ok() else {
        return;
    };
    let Some(m) = meta.get(&node) else {
        return;
    };
    let (ox, oy) = position_overrides.get(&node).copied().unwrap_or((0.0, 0.0));
    let (sox, soy) = sticky.get(&node).copied().unwrap_or((0.0, 0.0));
    let mut x = offset.0 + layout.location.x + ox + sox;
    let mut y = offset.1 + layout.location.y + oy + soy;
    let width = layout.size.width;
    let height = layout.size.height;
    // The unscrolled position (same accumulation, no scroll
    // subtraction, no sticky shift — sticky is scroll-derived).
    let x_ns = offset_ns.0 + layout.location.x + ox;
    let y_ns = offset_ns.1 + layout.location.y + oy;
    // M6 Phase 3 (Rec 1): a `fixed` box with no transformed
    // ancestor resolves against the viewport — its insets are
    // resolved against the viewport, and no ancestor scroll
    // offset or parent position applies. This is the re-anchor
    // the old comments promised but never built: laying fixed
    // out as absolute in taffy is only the sizing step, never
    // the positioning step. `left` wins over `right`, `top` over
    // `bottom`; with both insets `auto` the taffy position
    // (unscrolled) is the fallback — the static position is not
    // modelled, and this preserves the pre-M6 unscrolled
    // behaviour for that case.
    let is_viewport_fixed = m.dom.is_some()
        && m.style.position == PositionKind::Fixed
        && !ancestor_transformed;
    if is_viewport_fixed {
        let resolve = |d: Option<Dim>, basis: f32| {
            d.map(|dim| match dim {
                Dim::Px(v) => v,
                Dim::Pct(f) => f * basis,
                Dim::Vh(v) => v / 100.0 * viewport.1,
                Dim::Vw(v) => v / 100.0 * viewport.0,
            })
        };
        let inset = m.style.inset;
        x = match (resolve(inset[3], viewport.0), resolve(inset[1], viewport.0)) {
            (Some(left), _) => left + ox,
            (None, Some(right)) => viewport.0 - right - width + ox,
            (None, None) => x_ns,
        };
        y = match (resolve(inset[0], viewport.1), resolve(inset[2], viewport.1)) {
            (Some(top), _) => top + oy,
            (None, Some(bottom)) => viewport.1 - bottom - height + oy,
            (None, None) => y_ns,
        };
    }
    // A viewport-fixed box is clipped by the viewport alone (M6
    // Phase 3, Rec 1): no ancestor overflow clip reaches it or
    // its descendants — the clip chain restarts here.
    let (parent_clip, parent_clip_radii, parent_clip_source) = if is_viewport_fixed {
        (None, None, None)
    } else {
        (parent_clip, parent_clip_radii, parent_clip_source)
    };
    let radii = m
        .style
        .border_radius
        .map(|br| {
            br.resolve(
                width,
                height,
                m.style.font_size,
                root_font_size,
                viewport,
            )
        })
        .unwrap_or(ResolvedCorners::ZERO);
    let is_scroll_container = m.style.overflow_x.is_scroll_container()
        || m.style.overflow_y.is_scroll_container();
    // Any non-visible overflow clips (including `clip`, which is
    // not a scroll container); the clip rect is the box's own
    // padding box — with no borders modelled, the border box.
    let clips_content = m.style.overflow_x != OverflowKind::Visible
        || m.style.overflow_y != OverflowKind::Visible;
    let (clip, clip_radii, clip_source) = if clips_content {
        let own_rect = [x, y, width, height];
        let clip = Some(match parent_clip {
            Some(pc) => intersect_clip(pc, own_rect),
            None => own_rect,
        });
        let clip_radii = if radii.is_zero() { None } else { Some(radii) };
        (clip, clip_radii, Some(node))
    } else {
        (parent_clip, parent_clip_radii, parent_clip_source)
    };
    // Only a real scroll container becomes a scroll ancestor:
    // `overflow: clip` clips but never scrolls. A viewport-fixed
    // box is its own scroll root (M6 Phase 3, Rec 1): its
    // descendants scroll with it, never with the container
    // underneath — except when the fixed box itself is a scroll
    // container, in which case it is its descendants' ancestor
    // (and the `scroll_ancestor` handed to the children below is
    // its own id, so no ancestor above it leaks through).
    let scroll_ancestor = if is_scroll_container {
        m.dom.or(parent_scroll_ancestor)
    } else if is_viewport_fixed {
        None
    } else {
        parent_scroll_ancestor
    };
    // Stage 11 Rec 1/3: image + background-image resource ids
    // for this box, from the image map the tree was built with.
    // A background image is keyed by the raw CSS url in the
    // computed style; paint needs its intrinsic size for
    // cover/contain placement, so both ride the box.
    let image = m.dom.and_then(|d| images.by_node.get(&d)).map(|i| i.resource);
    // Stage 16 Phase 2 (Rec 5): a video box is live when the
    // map says a frame is presentable for its node.
    let video_live = m
        .dom
        .and_then(|d| images.videos.get(&d))
        .map(|v| v.has_frame)
        .unwrap_or(false);
    let (background_image, background_image_size) = m
        .style
        .background_image
        .as_ref()
        .and_then(|url| images.backgrounds.get(url))
        .map(|info| (Some(info.resource), Some((info.width, info.height))))
        .unwrap_or((None, None));
    // M6 Phase 1: this box's own transform, origin-resolved
    // against its border box (`transform-origin` % is of the
    // width/height), composed after the parent chain's matrix:
    // `parent matrix * own transform`, applied own-first. The
    // box's layout offset (`x`/`y`) is NOT folded in — the
    // matrix is the transform part only (identity when nothing
    // in the chain is transformed), and paint composes it with
    // the box's position.
    //
    // M6 Phase 2 (Rec 7/8): a box with no `dom` is a text leaf
    // whose style is its parent element's clone, not its own —
    // `transform`, `opacity`, `z-index` and `position` in it are
    // the parent's. Composing the cloned transform again would
    // double-apply it (a leaf under a 45-degree parent would
    // render at 90 degrees), and the cloned opacity/transform
    // would wrongly make the leaf a stacking context of its own.
    // The leaf has no transform and no context of its own: the
    // parent chain's matrix already carries the transform, and
    // the leaf paints inside the parent's opacity group.
    let origin = m
        .style
        .transform_origin
        .resolve(width, height, viewport);
    let own_matrix = if m.dom.is_none() || m.style.transform.is_empty() {
        Affine::identity()
    } else {
        m.style.transform.to_affine_in(
            origin,
            (width, height),
            m.style.font_size,
            root_font_size,
            viewport,
        )
    };
    let matrix = own_matrix.then(parent_matrix);
    let stacking = m.dom.is_some() && is_stacking_context(&m.style, is_flex_or_grid_item);
    out.push(LayoutBox {
        taffy: node,
        dom: m.dom,
        x,
        y,
        width,
        height,
        style: m.style.clone(),
        text: m.text.clone(),
        glyphs: Vec::new(),
        clip,
        clip_source,
        clip_radii,
        radii,
        scroll_ancestor,
        image,
        background_image,
        background_image_size,
        svg_canvas: m.svg_canvas,
        video_live,
        parent: parent_dom,
        matrix,
        is_stacking_context: stacking,
        is_viewport_fixed,
    });
    // Descendants of a scroll container move with its offset.
    let (sx, sy) = match m.dom {
        Some(dom) if is_scroll_container => {
            scroll_offsets.get(&dom).copied().unwrap_or((0.0, 0.0))
        }
        _ => (0.0, 0.0),
    };
    for &child in &m.children {
        collect_boxes(
            taffy,
            meta,
            child,
            (x - sx, y - sy),
            // The unscrolled base: this node's unscrolled
            // position, with no scroll subtraction at any level
            // (M6 Phase 3, Rec 1). A fixed child's fallback reads
            // this; a normal child's own scroll subtraction still
            // applies through `offset` above. A viewport-fixed
            // node's unscrolled position IS its resolved viewport
            // position — its descendants accumulate from there.
            if is_viewport_fixed { (x, y) } else { (x_ns, y_ns) },
            scroll_offsets,
            clip,
            clip_radii,
            clip_source,
            scroll_ancestor,
            viewport,
            root_font_size,
            position_overrides,
            sticky,
            images,
            // The nearest ancestor box with a DOM id: this node's
            // own id when it has one, else the one passed in
            // (anonymous text leaves carry no id of their own).
            m.dom.or(parent_dom),
            matrix,
            matches!(m.style.display, DisplayKind::Flex | DisplayKind::Grid),
            // A transformed ancestor (this node included, text
            // leaves excluded — their style is their parent's
            // clone) is the containing block for `fixed`
            // descendants below it.
            ancestor_transformed
                || (m.dom.is_some() && !m.style.transform.is_empty()),
            out,
        );
    }
}

// ---- two-phase style resolution (Stage 4 #5) ----

/// Phase 1: collect every element node of the DOM into breadth-first
/// levels from `roots` (level 0). Each entry knows its parent element
/// (`None` for a root). Only elements are collected — text nodes carry
/// no style and never pass inheritance on — but `display: none`
/// subtrees are collected too: their styles are computed and unused,
/// which keeps collection independent of style values. Which nodes get
/// boxes is unchanged; the builder still skips `display: none`.
/// A root may carry its parent (a subtree restyle, Stage 12 Rec 6:
/// the root's parent style is seeded from the retained map); roots
/// without one resolve against the initial style.
fn collect_style_levels(
    dom: &Dom,
    roots: &[(DomId, Option<DomId>)],
) -> Vec<Vec<(DomId, Option<DomId>)>> {
    let mut levels: Vec<Vec<(DomId, Option<DomId>)>> = Vec::new();
    let mut frontier: Vec<(DomId, Option<DomId>)> = roots
        .iter()
        .filter(|&&(id, _)| dom.tag(id).is_some())
        .copied()
        .collect();
    while !frontier.is_empty() {
        let mut next: Vec<(DomId, Option<DomId>)> = Vec::new();
        for &(id, _) in &frontier {
            for &child in dom.children(id) {
                if dom.tag(child).is_some() {
                    next.push((child, Some(id)));
                }
            }
        }
        levels.push(std::mem::take(&mut frontier));
        frontier = next;
    }
    levels
}

/// One rule + one of its selectors, as stored in the rule index.
struct IndexedRule<'a> {
    rule: &'a Rule,
    selector_idx: usize,
    /// Layer order from the sheet the rule came from (unlayered =
    /// `i32::MAX`); callers invert it for `!important` declarations.
    layer_order: i32,
    /// Global source order: UA rules keep theirs, author rules are
    /// offset past all UA rules, so document order holds across sheets.
    source_order: u32,
}

/// Rule index (Stylo-style): rules bucketed by their key selector
/// (`Selector::key_selector`), so an element only tests the buckets
/// it can match instead of every rule. A (rule, selector) pair lands
/// in exactly one bucket.
struct RuleIndex<'a> {
    by_id: HashMap<String, Vec<IndexedRule<'a>>>,
    by_class: HashMap<String, Vec<IndexedRule<'a>>>,
    by_tag: HashMap<String, Vec<IndexedRule<'a>>>,
    /// Key selectors that cannot bucket by id/class/tag (attribute,
    /// pseudo-class, universal, pseudo-element).
    other: Vec<IndexedRule<'a>>,
}

impl<'a> RuleIndex<'a> {
    /// Build the index over the UA sheet then the author sheet, in
    /// document order. Rules whose `@media` / `@supports` condition
    /// does not hold at this viewport / theme are dropped here
    /// (conditions are evaluated at style time, not flattened at
    /// parse time — a later viewport or theme change rebuilds the
    /// index).
    fn build(
        ua: &'a Stylesheet,
        author: &'a Stylesheet,
        viewport: (f32, f32),
        theme: Theme,
    ) -> Self {
        let mut index = RuleIndex {
            by_id: HashMap::new(),
            by_class: HashMap::new(),
            by_tag: HashMap::new(),
            other: Vec::new(),
        };
        let mut offset = 0u32;
        for sheet in [ua, author] {
            for rule in &sheet.rules {
                if !rule_applies(rule, viewport, theme) {
                    continue;
                }
                let layer_order = sheet.layer_order(&rule.layer);
                for (selector_idx, sel) in rule.selectors.iter().enumerate() {
                    let entry = IndexedRule {
                        rule,
                        selector_idx,
                        layer_order,
                        source_order: offset + rule.source_order,
                    };
                    match sel.key_selector() {
                        Some(SimpleSelector::Id(id)) => index
                            .by_id
                            .entry(id.clone())
                            .or_default()
                            .push(entry),
                        Some(SimpleSelector::Class(class)) => index
                            .by_class
                            .entry(class.clone())
                            .or_default()
                            .push(entry),
                        Some(SimpleSelector::Type(tag)) => index
                            .by_tag
                            .entry(tag.clone())
                            .or_default()
                            .push(entry),
                        _ => index.other.push(entry),
                    }
                }
            }
            offset += sheet.rules.len() as u32;
        }
        index
    }

    /// Candidate (rule, selector) pairs an element can match: its id
    /// bucket, each of its class buckets, its tag bucket, and the
    /// unbucketed remainder. The caller still runs the full matcher —
    /// the index only narrows the candidate set.
    fn candidates(&self, dom: &Dom, id: DomId) -> Vec<&IndexedRule<'a>> {
        let mut out: Vec<&IndexedRule<'a>> = Vec::new();
        if let Some(want) = dom.attr(id, "id")
            && let Some(bucket) = self.by_id.get(want) {
                out.extend(bucket.iter());
            }
        if let Some(classes) = dom.attr(id, "class") {
            for class in classes.split_whitespace() {
                if let Some(bucket) = self.by_class.get(class) {
                    out.extend(bucket.iter());
                }
            }
        }
        if let Some(tag) = dom.tag(id)
            && let Some(bucket) = self.by_tag.get(tag) {
                out.extend(bucket.iter());
            }
        out.extend(self.other.iter());
        out
    }
}

/// Whether a rule's enclosing `@media` / `@supports` conditions hold.
/// Unknown media features evaluate to `None` and the rule is skipped
/// (a query that cannot be judged does not apply) — documented M2
/// choice; `@supports` unknown values likewise skip the rule.
fn rule_applies(rule: &Rule, viewport: (f32, f32), theme: Theme) -> bool {
    if let Some(media) = &rule.media
        && media.eval(viewport.0, viewport.1, theme) != Some(true) {
            return false;
        }
    if let Some(supports) = &rule.supports
        && SupportsCondition::parse(supports).eval() != Some(true) {
            return false;
        }
    true
}

/// Context threaded through one element's style resolution.
struct StyleCtx<'a> {
    index: &'a RuleIndex<'a>,
    state: &'a ElementState,
    viewport: (f32, f32),
    /// The root element's computed font size (`rem` resolves against
    /// it; 16.0 until the root level is resolved).
    root_font_size: f32,
}

/// Phase 2: resolve one style per entry of `levels`, level by level.
/// A level's parents were all resolved in earlier levels, so each
/// entry's parent computed style is read from the already-resolved
/// map and the entries of one level are independent: with
/// `parallel` they resolve on rayon's shared global pool via
/// `par_iter` over `&Dom` / `&Stylesheet` (both `Sync`, asserted in
/// tests); without it the same loop runs sequentially, so the serial
/// path is the same code path and a real fallback, not a comment.
/// Returns the owned map from element to its (immutable, `Arc`-shared)
/// computed style, plus the per-node `NodeStyle` secondary map.
/// `seed_parent_style` / `root_font_size_seed` serve a subtree
/// restyle (Stage 12 Rec 6): level 0's parent style comes from the
/// retained map (its parent sits above the subtree), and `rem`
/// resolves against the retained root font size. Both are `None` /
/// 16.0 for a whole-document pass, and the seed only ever applies
/// at level 0 — a deeper level's parent must be in the map being
/// built, or the levels are malformed.
#[allow(clippy::too_many_arguments)]
fn resolve_styles(
    dom: &Dom,
    index: &RuleIndex,
    levels: &[Vec<(DomId, Option<DomId>)>],
    parallel: bool,
    viewport: (f32, f32),
    state: &ElementState,
    seed_parent_style: Option<Arc<ComputedStyle>>,
    root_font_size_seed: f32,
    theme: Theme,
) -> (
    HashMap<DomId, Arc<ComputedStyle>>,
    SparseSecondaryMap<DomId, NodeStyle>,
) {
    let mut resolved: HashMap<DomId, Arc<ComputedStyle>> = HashMap::new();
    let mut node_styles: SparseSecondaryMap<DomId, NodeStyle> = SparseSecondaryMap::new();
    let mut root_font_size = root_font_size_seed;
    // `theme` is consumed upstream by `RuleIndex::build`
    // (`rule_applies`); it is threaded through this signature so
    // the whole resolution chain carries it for later phases
    // (e.g. per-element media re-evaluation).
    let _ = theme;
    for (level_no, level) in levels.iter().enumerate() {
        // Parent computed styles first (serial, cheap): the parallel
        // closure then only reads `&Dom` / `&Stylesheet` plus its own
        // inputs, never the map being built.
        let inputs: Vec<(DomId, Arc<ComputedStyle>)> = level
            .iter()
            .map(|&(id, parent)| {
                let parent_style = match parent {
                    Some(p) => resolved
                        .get(&p)
                        .cloned()
                        .or_else(|| {
                            if level_no == 0 {
                                seed_parent_style.clone()
                            } else {
                                None
                            }
                        })
                        .expect("parent style resolved in an earlier level"),
                    None => Arc::new(ComputedStyle::initial()),
                };
                (id, parent_style)
            })
            .collect();
        let ctx = StyleCtx {
            index,
            state,
            viewport,
            root_font_size,
        };
        let resolve_one =
            |&(id, ref parent): &(DomId, Arc<ComputedStyle>)| {
                let (style, flags) = resolve_style(dom, id, &ctx, parent);
                (id, Arc::new(style), flags)
            };
        let resolved_level: Vec<(DomId, Arc<ComputedStyle>, SelectorFlags)> = if parallel {
            inputs.par_iter().map(resolve_one).collect()
        } else {
            inputs.iter().map(resolve_one).collect()
        };
        for (id, style, flags) in resolved_level {
            node_styles.insert(
                id,
                NodeStyle {
                    computed: style.clone(),
                    // M2 does a full restyle on every build; the
                    // hint/damage are populated so Stage 7's
                    // incremental path has the data to read.
                    hint: RestyleHint::full(),
                    damage: RestyleDamage::full(),
                    selector_flags: flags,
                },
            );
            resolved.insert(id, style);
        }
        // After the root level, the root element's font size is known
        // and `rem` resolves against it for every later level.
        if level_no == 0 {
            for &(id, _) in level {
                if dom.tag(id) == Some("html") {
                    if let Some(style) = resolved.get(&id) {
                        root_font_size = style.font_size;
                    }
                    break;
                }
            }
        }
    }
    (resolved, node_styles)
}

/// Resolve styles for the whole document: phase 1 from the `<html>`
/// element, or — for a fragment without one — from the document
/// root's element children, then phase 2 over the UA sheet + the
/// author sheet. `LayoutTree::build` calls this with `parallel: true`.
fn resolve_document_styles(
    dom: &Dom,
    ua: &Stylesheet,
    sheet: &Stylesheet,
    parallel: bool,
    viewport: (f32, f32),
    state: &ElementState,
    theme: Theme,
) -> (
    HashMap<DomId, Arc<ComputedStyle>>,
    SparseSecondaryMap<DomId, NodeStyle>,
) {
    let root = dom.root().expect("Dom always has a root node");
    let roots: Vec<(DomId, Option<DomId>)> = match find_element(dom, root, "html") {
        Some(html) => vec![(html, None)],
        None => dom
            .children(root)
            .iter()
            .copied()
            .filter(|&id| dom.tag(id).is_some())
            .map(|id| (id, None))
            .collect(),
    };
    let levels = collect_style_levels(dom, &roots);
    let index = RuleIndex::build(ua, sheet, viewport, theme);
    resolve_styles(dom, &index, &levels, parallel, viewport, state, None, 16.0, theme)
}

/// Resolve styles for one subtree only (Stage 12 Rec 6): the dirty
/// LCA's subtree, seeded with the root's parent style and the root
/// font size from the retained map. Sound because the parents above
/// the root are not dirty, so their styles are unchanged; sibling
/// and `nth-child` effects stay inside the LCA subtree (an insertion
/// marks the parent dirty, pulling the LCA up to cover the siblings);
/// and `:has()` is stubbed/inert, so no descendant change can
/// restyle an ancestor. Returns the FULL merged maps — retained
/// entries outside the subtree kept (`Arc` values, so the merge is
/// pointer clones; the resolution/matching work is what is skipped)
/// — so every caller that reads the whole map keeps working
/// unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_subtree_styles(
    dom: &Dom,
    ua: &Stylesheet,
    sheet: &Stylesheet,
    parallel: bool,
    viewport: (f32, f32),
    state: &ElementState,
    root: DomId,
    retained: &SparseSecondaryMap<DomId, NodeStyle>,
    theme: Theme,
) -> (
    HashMap<DomId, Arc<ComputedStyle>>,
    SparseSecondaryMap<DomId, NodeStyle>,
) {
    // The root's parent style seeds level 0 (its parent sits above
    // the subtree); a root with no element parent resolves against
    // the initial style, as in the document pass.
    let seed_parent_style = parent_element(dom, root)
        .and_then(|p| retained.get(p))
        .map(|ns| ns.computed.clone());
    // `rem` resolves against the `<html>` font size. When the root
    // IS `<html>`, its own size is being resolved in level 0, so the
    // seed is the initial 16.0 (the level-0 update then sets it from
    // the fresh style, exactly as in the document pass); otherwise
    // the retained `<html>` size is the basis.
    let root_font_size_seed = if dom.tag(root) == Some("html") {
        16.0
    } else {
        dom.root()
            .and_then(|r| find_element(dom, r, "html"))
            .and_then(|html| retained.get(html))
            .map(|ns| ns.computed.font_size)
            .unwrap_or(16.0)
    };
    let levels = collect_style_levels(dom, &[(root, parent_element(dom, root))]);
    let index = RuleIndex::build(ua, sheet, viewport, theme);
    let (sub_styles, sub_node_styles) = resolve_styles(
        dom,
        &index,
        &levels,
        parallel,
        viewport,
        state,
        seed_parent_style,
        root_font_size_seed,
        theme,
    );
    let mut styles: HashMap<DomId, Arc<ComputedStyle>> =
        HashMap::with_capacity(retained.len() + sub_styles.len());
    for (id, ns) in retained.iter() {
        styles.insert(id, ns.computed.clone());
    }
    styles.extend(sub_styles);
    let mut node_styles = retained.clone();
    for (id, ns) in sub_node_styles.iter() {
        node_styles.insert(id, ns.clone());
    }
    (styles, node_styles)
}

// ---- selector matching (right-to-left) ----

/// Parent *element* of `id` (`None` for the root element / document).
fn parent_element(dom: &Dom, id: DomId) -> Option<DomId> {
    let parent = dom.get(id)?.parent?;
    if dom.tag(parent).is_some() {
        Some(parent)
    } else {
        None
    }
}

/// Element siblings of `id` (elements only — text nodes are not
/// siblings for selector purposes), in document order.
fn element_siblings(dom: &Dom, id: DomId) -> Vec<DomId> {
    let parent = match dom.get(id).and_then(|n| n.parent) {
        Some(p) => p,
        None => return vec![id],
    };
    dom.children(parent)
        .iter()
        .copied()
        .filter(|&c| dom.tag(c).is_some())
        .collect()
}

/// Previous element sibling, if any.
fn prev_element_sibling(dom: &Dom, id: DomId) -> Option<DomId> {
    let siblings = element_siblings(dom, id);
    siblings
        .iter()
        .position(|&s| s == id)
        .and_then(|i| if i > 0 { Some(siblings[i - 1]) } else { None })
}

/// Does `n` (1-based) satisfy `an+b`? `a == 0` means exactly `b`.
fn nth_matches(n: i32, ab: AnPlusB) -> bool {
    if ab.a == 0 {
        return n == ab.b;
    }
    let d = n - ab.b;
    d % ab.a == 0 && d / ab.a >= 0
}

/// True when `id` is a link anchor for `:link` / `:any-link`:
/// an HTML `<a>` / `<area>` / `<link>` with `href`, or an SVG
/// `<a>` with `href` (SVG2) or the legacy `xlink:href` (Stage 15,
/// M7 Phase 1, Rec 1 — namespace-aware; a same-spelled `<a>` in
/// any other namespace is not a link).
fn is_link_anchor(dom: &Dom, id: DomId) -> bool {
    match dom.tag(id) {
        Some("a") => {
            if dom.attr(id, "href").is_some() {
                return true;
            }
            dom.namespace(id).is_some_and(|ns| ns == pluto_html::NS_SVG)
                && dom.attr(id, "xlink:href").is_some()
        }
        Some("area" | "link") => dom.attr(id, "href").is_some(),
        _ => false,
    }
}

/// Right-to-left complex-selector matching: test the key (rightmost)
/// compound first, then walk ancestors / siblings per combinator.
/// A selector with a pseudo-element never matches a real element
/// (generated boxes are parked until Stage 8), and `:has()` never
/// matches (`has_unimplemented`).
pub fn selector_matches(
    dom: &Dom,
    id: DomId,
    sel: &Selector,
    state: &ElementState,
) -> bool {
    if sel.has_unimplemented || sel.pseudo_element.is_some() || sel.compounds.is_empty() {
        return false;
    }
    match_compound_from(dom, id, sel, sel.compounds.len() - 1, state)
}

fn match_compound_from(
    dom: &Dom,
    id: DomId,
    sel: &Selector,
    idx: usize,
    state: &ElementState,
) -> bool {
    if !match_compound(dom, id, &sel.compounds[idx], state) {
        return false;
    }
    if idx == 0 {
        return true;
    }
    match sel.combinators[idx - 1] {
        Combinator::Child => match parent_element(dom, id) {
            Some(p) => match_compound_from(dom, p, sel, idx - 1, state),
            None => false,
        },
        Combinator::Descendant => {
            let mut cur = parent_element(dom, id);
            while let Some(p) = cur {
                if match_compound_from(dom, p, sel, idx - 1, state) {
                    return true;
                }
                cur = parent_element(dom, p);
            }
            false
        }
        Combinator::NextSibling => match prev_element_sibling(dom, id) {
            Some(s) => match_compound_from(dom, s, sel, idx - 1, state),
            None => false,
        },
        Combinator::SubsequentSibling => {
            let mut cur = prev_element_sibling(dom, id);
            while let Some(s) = cur {
                if match_compound_from(dom, s, sel, idx - 1, state) {
                    return true;
                }
                cur = prev_element_sibling(dom, s);
            }
            false
        }
    }
}

/// Every simple selector in the compound must match the same element.
fn match_compound(
    dom: &Dom,
    id: DomId,
    compound: &pluto_css::CompoundSelector,
    state: &ElementState,
) -> bool {
    compound
        .parts
        .iter()
        .all(|part| match_simple(dom, id, part, state))
}

fn match_simple(
    dom: &Dom,
    id: DomId,
    part: &SimpleSelector,
    state: &ElementState,
) -> bool {
    match part {
        SimpleSelector::Universal => true,
        // Type selectors are ASCII case-insensitive in HTML.
        SimpleSelector::Type(t) => dom
            .tag(id)
            .is_some_and(|tag| tag.eq_ignore_ascii_case(t)),
        // No-quirks assumption (see module docs): exact match.
        SimpleSelector::Class(c) => dom
            .attr(id, "class")
            .is_some_and(|classes| classes.split_whitespace().any(|x| x == c)),
        SimpleSelector::Id(want) => dom.attr(id, "id") == Some(want.as_str()),
        SimpleSelector::Attribute {
            name,
            op,
            value,
            case_sensitive,
        } => {
            let Some(actual) = dom.attr(id, name) else {
                return false;
            };
            let sensitive = case_sensitive.unwrap_or(true);
            let eq = |a: &str, b: &str| {
                if sensitive {
                    a == b
                } else {
                    a.eq_ignore_ascii_case(b)
                }
            };
            match op {
                AttrOp::Exists => true,
                AttrOp::Equals => value.as_ref().is_some_and(|v| eq(actual, v)),
                AttrOp::Includes => value.as_ref().is_some_and(|v| {
                    actual.split_whitespace().any(|tok| eq(tok, v))
                }),
                AttrOp::Dash => value.as_ref().is_some_and(|v| {
                    eq(actual, v)
                        || (!sensitive
                            && actual.len() > v.len()
                            && actual[..v.len()].eq_ignore_ascii_case(v)
                            && actual.as_bytes()[v.len()] == b'-')
                        || (sensitive
                            && actual.len() > v.len()
                            && actual.starts_with(v.as_str())
                            && actual.as_bytes()[v.len()] == b'-')
                }),
                AttrOp::Prefix => value.as_ref().is_some_and(|v| {
                    if sensitive {
                        actual.starts_with(v.as_str())
                    } else {
                        actual.len() >= v.len()
                            && actual[..v.len()].eq_ignore_ascii_case(v)
                    }
                }),
                AttrOp::Suffix => value.as_ref().is_some_and(|v| {
                    if sensitive {
                        actual.ends_with(v.as_str())
                    } else {
                        actual.len() >= v.len()
                            && actual[actual.len() - v.len()..].eq_ignore_ascii_case(v)
                    }
                }),
                AttrOp::Substring => value.as_ref().is_some_and(|v| {
                    if sensitive {
                        actual.contains(v.as_str())
                    } else {
                        actual.to_ascii_lowercase().contains(&v.to_ascii_lowercase())
                    }
                }),
            }
        }
        SimpleSelector::PseudoClass(pc) => match_pseudo_class(dom, id, pc, state),
        // A pseudo-element inside a compound cannot match an element.
        SimpleSelector::PseudoElement(_) => false,
    }
}

/// Whether `id` is a control `:valid` / `:optional` can apply
/// to (Stage 13 Rec 8): inputs, textareas, selects, and buttons.
fn is_form_control(dom: &Dom, id: DomId) -> bool {
    matches!(dom.tag(id), Some("input" | "textarea" | "select" | "button"))
}

fn match_pseudo_class(
    dom: &Dom,
    id: DomId,
    pc: &PseudoClass,
    state: &ElementState,
) -> bool {
    match pc {
        // `:root` is the document element (`<html>`).
        PseudoClass::Root => {
            dom.tag(id) == Some("html") && parent_element(dom, id).is_none()
        }
        PseudoClass::FirstChild => {
            element_siblings(dom, id).first().copied() == Some(id)
        }
        PseudoClass::LastChild => {
            element_siblings(dom, id).last().copied() == Some(id)
        }
        PseudoClass::NthChild(ab) => {
            let siblings = element_siblings(dom, id);
            siblings
                .iter()
                .position(|&s| s == id)
                .is_some_and(|i| nth_matches(i as i32 + 1, *ab))
        }
        PseudoClass::NthLastChild(ab) => {
            let siblings = element_siblings(dom, id);
            siblings
                .iter()
                .position(|&s| s == id)
                .is_some_and(|i| nth_matches((siblings.len() - i) as i32, *ab))
        }
        PseudoClass::OnlyChild => element_siblings(dom, id).len() == 1,
        // No child nodes at all (comments do not exist in this DOM).
        PseudoClass::Empty => dom.children(id).is_empty(),
        PseudoClass::FirstOfType => {
            let tag = dom.tag(id);
            element_siblings(dom, id)
                .iter().find(|&&s| dom.tag(s) == tag)
                .copied()
                == Some(id)
        }
        PseudoClass::LastOfType => {
            let tag = dom.tag(id);
            element_siblings(dom, id)
                .iter().rfind(|&&s| dom.tag(s) == tag)
                .copied()
                == Some(id)
        }
        PseudoClass::NthOfType(ab) => {
            let tag = dom.tag(id);
            let same: Vec<DomId> = element_siblings(dom, id)
                .into_iter()
                .filter(|&s| dom.tag(s) == tag)
                .collect();
            same.iter()
                .position(|&s| s == id)
                .is_some_and(|i| nth_matches(i as i32 + 1, *ab))
        }
        PseudoClass::NthLastOfType(ab) => {
            let tag = dom.tag(id);
            let same: Vec<DomId> = element_siblings(dom, id)
                .into_iter()
                .filter(|&s| dom.tag(s) == tag)
                .collect();
            same.iter()
                .position(|&s| s == id)
                .is_some_and(|i| nth_matches((same.len() - i) as i32, *ab))
        }
        PseudoClass::OnlyOfType => {
            let tag = dom.tag(id);
            element_siblings(dom, id)
                .iter()
                .filter(|&&s| dom.tag(s) == tag)
                .count()
                == 1
        }
        // `:link` = an anchor with an `href` that has NOT been
        // visited — `:link` and `:visited` are exclusive per spec.
        // `:any-link` is the union: any anchor with an `href`,
        // visited or not. Namespace-aware (Stage 15, M7 Phase 1,
        // Rec 1): an SVG `<a>` also links through the legacy
        // `xlink:href`.
        PseudoClass::Link => {
            is_link_anchor(dom, id) && !state.visited.contains(&id)
        }
        PseudoClass::AnyLink => is_link_anchor(dom, id),
        // `:visited` = an anchor the worker's visited projection
        // flagged (see `ElementState::visited`): the link's
        // resolved href is in the session's committed-URL set.
        // The matcher only ever sees element ids — visited-ness
        // never leaves the style pass (no JS binding, geometry,
        // or IPC field carries it).
        PseudoClass::Visited => state.visited.contains(&id),
        PseudoClass::Hover => state.hover.contains(&id),
        PseudoClass::Active => state.active.contains(&id),
        PseudoClass::Focus | PseudoClass::FocusVisible => state.focus.contains(&id),
        // Form pseudo-classes (Stage 13 Rec 8). `:valid` is the
        // complement of `:invalid` for controls that can validate
        // (anything in either set is a form control); `:optional`
        // is the complement of `:required` on the same terms.
        // `:read-only` is the complement of editable: text
        // controls without `readonly` are NOT `:read-only` — the
        // projection only flags genuinely read-only controls, so
        // the matcher treats "not in the set" as read-only only
        // for controls (see `is_form_control` below).
        PseudoClass::Valid => {
            is_form_control(dom, id) && !state.invalid.contains(&id)
        }
        PseudoClass::Invalid => state.invalid.contains(&id),
        PseudoClass::Required => state.required.contains(&id),
        PseudoClass::Optional => {
            is_form_control(dom, id) && !state.required.contains(&id)
        }
        PseudoClass::Disabled => state.disabled.contains(&id),
        PseudoClass::Checked => state.checked.contains(&id),
        PseudoClass::ReadOnly => state.read_only.contains(&id),
        PseudoClass::FocusWithin => {
            if state.focus.contains(&id) {
                return true;
            }
            let mut stack: Vec<DomId> = dom.children(id).to_vec();
            while let Some(child) = stack.pop() {
                if state.focus.contains(&child) {
                    return true;
                }
                stack.extend(dom.children(child));
            }
            false
        }
        PseudoClass::Not(args) => !args
            .iter()
            .any(|s| selector_matches(dom, id, s, state)),
        PseudoClass::Is(args) | PseudoClass::Where(args) => {
            args.iter().any(|s| selector_matches(dom, id, s, state))
        }
        // `:has()` is parsed and stored but never matches (the
        // selector's `has_unimplemented` flag already rejects the
        // whole selector; this arm is the belt to the braces).
        PseudoClass::Has(_) => false,
    }
}

/// Selector flags for one matched selector (invalidation-map data).
fn flags_for_selector(sel: &Selector) -> SelectorFlags {
    let mut flags = SelectorFlags::default();
    for compound in &sel.compounds {
        for part in &compound.parts {
            if let SimpleSelector::PseudoClass(pc) = part
                && matches!(
                    pc,
                    PseudoClass::Hover
                        | PseudoClass::Active
                        | PseudoClass::Focus
                        | PseudoClass::FocusVisible
                        | PseudoClass::FocusWithin
                        // Form pseudo-classes (Stage 13 Rec 8) are
                        // dynamic too: validity / required-ness /
                        // checked-ness change as the user types.
                        | PseudoClass::Valid
                        | PseudoClass::Invalid
                        | PseudoClass::Required
                        | PseudoClass::Optional
                        | PseudoClass::Disabled
                        | PseudoClass::Checked
                        | PseudoClass::ReadOnly
                ) {
                    flags.has_dynamic_pseudo = true;
                }
        }
    }
    for comb in &sel.combinators {
        match comb {
            Combinator::NextSibling | Combinator::SubsequentSibling => {
                flags.has_sibling_combinator = true
            }
            Combinator::Descendant | Combinator::Child => {
                flags.has_descendant_combinator = true
            }
        }
    }
    flags
}

// ---- style resolution (cascade) ----

/// Resolve one element's computed style: start from the parent's
/// computed style (inherited properties carry over), reset the
/// non-inherited properties to initial, then apply the cascade
/// winners in ascending `CascadeKey` order (later = higher wins).
/// `font-size` and friends apply in a first pass so `em` in margins
/// resolves against the element's *own* computed font size.
fn resolve_style(
    dom: &Dom,
    id: DomId,
    ctx: &StyleCtx,
    parent: &Arc<ComputedStyle>,
) -> (ComputedStyle, SelectorFlags) {
    let mut style = (**parent).clone();
    // Non-inherited properties fall back to initial (the inherited
    // ones — color, font-*, line-height, text-align, white-space —
    // keep the parent's computed value from the clone above).
    let init = ComputedStyle::initial();
    style.display = init.display;
    style.width = None;
    style.height = None;
    style.min_width = None;
    style.max_width = None;
    style.min_height = None;
    style.max_height = None;
    style.margin = init.margin;
    style.padding = init.padding;
    style.inset = init.inset;
    style.background = None;
    style.overflow_x = OverflowKind::Visible;
    style.overflow_y = OverflowKind::Visible;
    style.position = PositionKind::Static;
    style.box_sizing = BoxSizingKind::ContentBox;
    style.border_radius = None;
    style.box_shadow = Vec::new();
    style.flex_direction = FlexDirection::Row;
    style.flex_wrap = FlexWrap::NoWrap;
    style.flex_grow = 0.0;
    style.flex_shrink = 1.0;
    style.flex_basis = None;
    style.grid_template_columns = Vec::new();
    style.grid_template_rows = Vec::new();
    style.grid_gap = (Dim::Px(0.0), Dim::Px(0.0));
    style.grid_column = (GridPlace::Auto, GridPlace::Auto);
    style.grid_row = (GridPlace::Auto, GridPlace::Auto);
    style.grid_auto_flow = GridAutoFlow::Row;
    style.z_index = None;
    style.opacity = 1.0;
    style.transform = Vec::new();
    style.transform_origin = init.transform_origin;
    style.transform_box = TransformBox::ViewBox;
    style.will_change_transform = false;
    style.transitions = Vec::new();
    style.animations = Vec::new();
    // `custom_properties` is inherited, NOT reset: the merged map
    // is built below from the parent's chain plus this element's
    // own `--*` declarations.

    // Collect matching declarations with their cascade keys.
    // (Owned property/value: the inline-style sheet below is a
    // temporary, so borrowed values would not live long enough.)
    let mut decls: Vec<(CascadeKey, String, Value)> = Vec::new();
    let mut flags = SelectorFlags::default();
    for cand in ctx.index.candidates(dom, id) {
        let sel = &cand.rule.selectors[cand.selector_idx];
        if !selector_matches(dom, id, sel, ctx.state) {
            continue;
        }
        let f = flags_for_selector(sel);
        flags.has_dynamic_pseudo |= f.has_dynamic_pseudo;
        flags.has_sibling_combinator |= f.has_sibling_combinator;
        flags.has_descendant_combinator |= f.has_descendant_combinator;
        for decl in &cand.rule.declarations {
            // For `!important` the layer order reverses (Cascade 5):
            // earlier layers win, and unlayered important loses to
            // layered important — hence the negation.
            let layer_order = if decl.important {
                cand.layer_order.wrapping_neg()
            } else {
                cand.layer_order
            };
            decls.push((
                CascadeKey {
                    origin_rank: cand.rule.origin.rank(decl.important),
                    important: decl.important,
                    layer_order,
                    inline: false,
                    specificity: sel.specificity(),
                    source_order: cand.source_order,
                },
                decl.property.clone(),
                decl.value.clone(),
            ));
        }
    }
    // Inline `style=""`: parsed as declarations of a dummy rule.
    // Inline beats all rule-based declarations of the same
    // origin/importance (the `inline` flag in the sort key).
    if let Some(attr) = dom.attr(id, "style")
        && !attr.trim().is_empty() {
            let inline_sheet = Stylesheet::parse(&format!("* {{ {attr} }}"));
            for rule in &inline_sheet.rules {
                for decl in &rule.declarations {
                    decls.push((
                        CascadeKey {
                            origin_rank: Origin::Author.rank(decl.important),
                            important: decl.important,
                            layer_order: if decl.important {
                                i32::MAX.wrapping_neg()
                            } else {
                                i32::MAX
                            },
                            inline: true,
                            specificity: (0, 0, 0),
                            source_order: u32::MAX,
                        },
                        decl.property.clone(),
                        decl.value.clone(),
                    ));
                }
            }
        }
    // Presentation attributes (Stage 15, M7 Phase 1, Rec 6): an
    // SVG element's `fill` / `stroke` / `stroke-width` /
    // `stop-color` attributes are author-level declarations with
    // specificity 0 (SVG2 §6.5 / CSS Cascade 5 §6.1) — below every
    // author rule, above the UA sheet. They sort there via the
    // lowest author key: author origin, non-important, the lowest
    // layer order (layered and unlayered rules all sort above),
    // non-inline, zero specificity, first source order. Only SVG
    // elements have presentation attributes — an HTML element's
    // `fill` attribute must NOT leak into the cascade (namespace
    // threaded by Rec 1). Values are parsed as single CSS values
    // (`parse_value`), never embedded in a stylesheet string, so a
    // hostile `fill="red; background: blue"` cannot inject a
    // second declaration.
    if dom
        .namespace(id)
        .is_some_and(|ns| ns == pluto_html::NS_SVG)
    {
        for property in ["fill", "stroke", "stroke-width", "stop-color"] {
            if let Some(attr) = dom.attr(id, property)
                && !attr.trim().is_empty()
            {
                decls.push((
                    CascadeKey {
                        origin_rank: Origin::Author.rank(false),
                        important: false,
                        layer_order: i32::MIN,
                        inline: false,
                        specificity: (0, 0, 0),
                        source_order: 0,
                    },
                    property.to_string(),
                    parse_value(attr),
                ));
            }
        }
    }
    decls.sort_by_key(|(key, _, _)| *key);

    // M6 Rec 12: custom properties first. The element's own `--*`
    // declarations (in cascade order — the sorted list, so the
    // winner is last) layer over the inherited chain seeded from
    // the parent's map. The merged map is what `var()` lookups
    // read below.
    //
    // NOTE: a `var()` change on `:root` is document-wide restyle
    // by nature — subtree restyle does NOT short-circuit custom
    // property changes (a later phase may refine this; the
    // incremental path re-resolves the dirty subtree, and a
    // `:root` change dirties the root, so the restyle still
    // reaches every descendant).
    let mut custom: HashMap<String, Value> = parent.custom_properties.clone();
    for (_, property, value) in decls.iter() {
        if property.starts_with("--") {
            custom.insert(property.clone(), value.clone());
        }
    }
    style.custom_properties = custom.clone();

    let parent_font_size = parent.font_size;
    let root_font_size = ctx.root_font_size;
    let viewport = ctx.viewport;
    // Pass 1: font properties first — `em` elsewhere resolves
    // against the element's own computed font size, which is only
    // known after `font-size` applies.
    for (_, property, value) in decls.iter().filter(|(_, p, _)| {
        matches!(
            p.as_str(),
            "font-size" | "font-family" | "font-weight" | "font-style" | "font-variant" | "line-height"
        )
    }) {
        // Custom properties are already stored above; they are
        // resolved at use time, not at definition time.
        if property.starts_with("--") {
            continue;
        }
        // An unresolvable `var()` with no fallback makes the
        // declaration invalid at computed-value time: skip it,
        // keeping the inherited / initial value.
        let Some(resolved) = resolve_vars(value, &custom) else {
            continue;
        };
        apply_decl(
            &mut style,
            property,
            &resolved,
            parent_font_size,
            root_font_size,
            viewport,
        );
    }
    // Pass 2: everything else, in cascade order.
    for (_, property, value) in decls.iter().filter(|(_, p, _)| {
        !matches!(
            p.as_str(),
            "font-size" | "font-family" | "font-weight" | "font-style" | "font-variant" | "line-height"
        )
    }) {
        if property.starts_with("--") {
            continue;
        }
        let Some(resolved) = resolve_vars(value, &custom) else {
            continue;
        };
        apply_decl(
            &mut style,
            property,
            &resolved,
            parent_font_size,
            root_font_size,
            viewport,
        );
    }
    // `currentColor` post-pass (Stage 15, M7 Phase 1, Rec 6):
    // `fill` / `stroke` may have resolved to `Paint::CurrentColor`
    // in either pass, but `color` applies in cascade order too, so
    // resolving at `apply_decl` time could read a stale `color`.
    // Resolve now, against the final computed `color`.
    if style.fill == Paint::CurrentColor {
        style.fill = Paint::Color(style.color);
    }
    if style.stroke == Paint::CurrentColor {
        style.stroke = Paint::Color(style.color);
    }
    (style, flags)
}

/// Resolve `var()` references in one declaration value against
/// `custom` (the element's merged custom-property map — its own
/// `--*` declarations over the inherited chain). Returns `None`
/// when the value is unresolvable with no fallback (invalid at
/// computed-value time — the caller skips the declaration).
/// Non-`var()` values pass through unchanged.
///
/// `Raw` strings containing `var(` are left unresolved
/// (documented limitation): the raw text is not re-parsed here,
/// so e.g. `background: var(--x)` keeps its text and the
/// property arm finds no colour / URL in it.
fn resolve_vars(value: &Value, custom: &HashMap<String, Value>) -> Option<Value> {
    resolve_vars_depth(value, custom, 0)
}

fn resolve_vars_depth(
    value: &Value,
    custom: &HashMap<String, Value>,
    depth: usize,
) -> Option<Value> {
    if depth > 8 {
        return None;
    }
    match value {
        Value::Var { name, fallback } => {
            if let Some(stored) = custom.get(name) {
                let stored = stored.clone();
                resolve_vars_depth(&stored, custom, depth + 1)
            } else if let Some(fb) = fallback {
                resolve_vars_depth(fb, custom, depth + 1)
            } else {
                None
            }
        }
        Value::Calc(expr) => {
            let resolved = resolve_calc_vars(expr, custom, depth)?;
            Some(Value::Calc(resolved))
        }
        Value::Clamp { min, val, max } => Some(Value::Clamp {
            min: Box::new(resolve_calc_vars(min, custom, depth)?),
            val: Box::new(resolve_calc_vars(val, custom, depth)?),
            max: Box::new(resolve_calc_vars(max, custom, depth)?),
        }),
        _ => Some(value.clone()),
    }
}

/// Resolve `var()` nodes inside a [`CalcExpr`], substituting the
/// looked-up value (converted with [`CalcExpr::from_value`]) or
/// the fallback expression. `None` when a `var()` has neither.
fn resolve_calc_vars(
    expr: &CalcExpr,
    custom: &HashMap<String, Value>,
    depth: usize,
) -> Option<CalcExpr> {
    if depth > 8 {
        return None;
    }
    match expr {
        CalcExpr::Var(name, fallback) => {
            if let Some(stored) = custom.get(name) {
                let stored = stored.clone();
                let as_expr = CalcExpr::from_value(&stored)?;
                resolve_calc_vars(&as_expr, custom, depth + 1)
            } else if let Some(fb) = fallback {
                resolve_calc_vars(fb, custom, depth + 1)
            } else {
                None
            }
        }
        CalcExpr::Add(a, b) => Some(CalcExpr::Add(
            Box::new(resolve_calc_vars(a, custom, depth)?),
            Box::new(resolve_calc_vars(b, custom, depth)?),
        )),
        CalcExpr::Sub(a, b) => Some(CalcExpr::Sub(
            Box::new(resolve_calc_vars(a, custom, depth)?),
            Box::new(resolve_calc_vars(b, custom, depth)?),
        )),
        CalcExpr::Mul(e, n) => Some(CalcExpr::Mul(
            Box::new(resolve_calc_vars(e, custom, depth)?),
            *n,
        )),
        CalcExpr::Div(e, n) => Some(CalcExpr::Div(
            Box::new(resolve_calc_vars(e, custom, depth)?),
            *n,
        )),
        _ => Some(expr.clone()),
    }
}

fn apply_decl(
    style: &mut ComputedStyle,
    property: &str,
    value: &Value,
    parent_font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) {
    match property {
        "display" => {
            if let Value::Keyword(k) = value {
                style.display = match k.as_str() {
                    "block" => DisplayKind::Block,
                    "flex" => DisplayKind::Flex,
                    "grid" => DisplayKind::Grid,
                    "inline" => DisplayKind::Inline,
                    "inline-block" => DisplayKind::InlineBlock,
                    "flow-root" => DisplayKind::FlowRoot,
                    "none" => DisplayKind::None,
                    // Taffy has no table layout: table boxes lay out
                    // as blocks in M2 (documented limitation).
                    "list-item" | "table" | "table-row" | "table-cell"
                    | "table-caption" | "table-header-group" | "table-row-group"
                    | "table-footer-group" | "table-column" | "table-column-group" => {
                        DisplayKind::Block
                    }
                    _ => return,
                };
            }
        }
        "position" => {
            if let Value::Keyword(k) = value {
                style.position = match k.as_str() {
                    "static" => PositionKind::Static,
                    "relative" => PositionKind::Relative,
                    "absolute" => PositionKind::Absolute,
                    "fixed" => PositionKind::Fixed,
                    "sticky" => PositionKind::Sticky,
                    _ => return,
                };
            }
        }
        "width" => style.width = dim_value(value, style.font_size, root_font_size, viewport),
        "height" => style.height = dim_value(value, style.font_size, root_font_size, viewport),
        "min-width" => {
            style.min_width = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "max-width" => {
            style.max_width = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "min-height" => {
            style.min_height = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "max-height" => {
            style.max_height = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "top" => {
            style.inset[0] = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "right" => {
            style.inset[1] = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "bottom" => {
            style.inset[2] = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "left" => {
            style.inset[3] = dim_value(value, style.font_size, root_font_size, viewport)
        }
        "margin-top" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.margin[0] = s;
            }
        }
        "margin-right" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.margin[1] = s;
            }
        }
        "margin-bottom" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.margin[2] = s;
            }
        }
        "margin-left" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.margin[3] = s;
            }
        }
        "margin" => {
            if let Value::Raw(raw) = value
                && let Some(sides) =
                    parse_shorthand(raw, style.font_size, root_font_size, viewport)
                {
                    style.margin = sides;
                }
        }
        "padding-top" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.padding[0] = s;
            }
        }
        "padding-right" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.padding[1] = s;
            }
        }
        "padding-bottom" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.padding[2] = s;
            }
        }
        "padding-left" => {
            if let Some(s) = spacing_value(value, style.font_size, root_font_size, viewport) {
                style.padding[3] = s;
            }
        }
        "padding" => {
            if let Value::Raw(raw) = value
                && let Some(sides) =
                    parse_shorthand(raw, style.font_size, root_font_size, viewport)
                {
                    style.padding = sides;
                }
        }
        "font-size" => match value {
            Value::Length(v) => style.font_size = *v,
            Value::Percentage(p) => style.font_size = parent_font_size * p / 100.0,
            Value::Em(v) => style.font_size = parent_font_size * v,
            Value::Rem(v) => style.font_size = root_font_size * v,
            Value::Vh(v) => style.font_size = viewport.1 * v / 100.0,
            Value::Vw(v) => style.font_size = viewport.0 * v / 100.0,
            Value::Keyword(k) => {
                style.font_size = match k.as_str() {
                    "xx-small" => 9.0,
                    "x-small" => 10.0,
                    "small" => 13.0,
                    "medium" => 16.0,
                    "large" => 18.0,
                    "x-large" => 24.0,
                    "xx-large" | "xxx-large" => 32.0,
                    "larger" => parent_font_size * 1.2,
                    "smaller" => parent_font_size / 1.2,
                    _ => return,
                }
            }
            Value::Raw(raw) => {
                if let Some(Spacing::Px(v)) =
                    parse_spacing_single(raw, parent_font_size, root_font_size, viewport)
                {
                    style.font_size = v;
                }
            }
            _ => {}
        },
        "font-family" => match value {
            Value::Raw(raw) | Value::Keyword(raw) => {
                let first = raw.split(',').next().unwrap_or("").trim();
                let first = first.trim_matches(|c| c == '"' || c == '\'');
                if !first.is_empty() {
                    style.font_family = first.to_string();
                }
            }
            _ => {}
        },
        "font-weight" => match value {
            Value::Keyword(k) => {
                style.bold = matches!(k.as_str(), "bold" | "bolder")
                    || k.parse::<f32>().map(|n| n >= 600.0).unwrap_or(false);
                if k == "normal" || k == "lighter" {
                    style.bold = false;
                }
            }
            Value::Raw(raw) => {
                if let Ok(n) = raw.trim().parse::<f32>() {
                    style.bold = n >= 600.0;
                }
            }
            _ => {}
        },
        "font-style" => {
            if let Value::Keyword(k) = value {
                style.font_style = match k.as_str() {
                    "italic" => FontStyle::Italic,
                    "oblique" => FontStyle::Oblique,
                    _ => FontStyle::Normal,
                };
            }
        }
        "line-height" => match value {
            // NOTE: the CSS parser keeps bare numbers as
            // `Length` (it cannot tell `1.6` from `1.6px`), so a
            // small `Length` here is read as a multiplier — the
            // `font: 16px/1.6` form depends on it. A `Length`
            // >= 10 is read as px. Documented heuristic.
            Value::Length(v) => {
                style.line_height = if *v > 0.0 && *v < 10.0 {
                    v * style.font_size
                } else {
                    *v
                }
            }
            Value::Percentage(p) => style.line_height = style.font_size * p / 100.0,
            Value::Em(v) => style.line_height = style.font_size * v,
            Value::Rem(v) => style.line_height = root_font_size * v,
            Value::Vh(v) => style.line_height = viewport.1 * v / 100.0,
            Value::Vw(v) => style.line_height = viewport.0 * v / 100.0,
            Value::Keyword(k) if k == "normal" => {
                style.line_height = style.font_size * 1.2
            }
            Value::Raw(raw) => {
                if let Ok(n) = raw.trim().parse::<f32>() {
                    style.line_height = n * style.font_size;
                }
            }
            _ => {}
        },
        "color" => match value {
            Value::Color(r, g, b) => style.color = [*r, *g, *b],
            // Alpha is dropped: M1 paint has no alpha channel.
            Value::ColorA(r, g, b, _) => style.color = [*r, *g, *b],
            _ => {}
        },
        // Stage 15 (M7 Phase 1, Rec 6): SVG paint properties.
        // `fill` / `stroke` accept a colour, `none`, a `url(...)`
        // paint-server reference, or `currentColor` (resolved
        // against the computed `color` in `resolve_style`'s
        // post-pass, because `color` may apply after `fill` in
        // cascade order). `stroke-width` accepts a length or a
        // bare number (`Raw`, from `parse_value`); a percentage
        // is relative to the viewport diagonal and is left for a
        // later phase. `stop-color` accepts a colour or
        // `currentColor` (resolved here against the inherited
        // `color` — the post-pass only rewrites `fill`/`stroke`).
        "fill" => {
            if let Some(paint) = paint_value(value) {
                style.fill = paint;
            }
        }
        "stroke" => {
            if let Some(paint) = paint_value(value) {
                style.stroke = paint;
            }
        }
        "stroke-width" => match value {
            Value::Length(v) => style.stroke_width = v.max(0.0),
            Value::Em(v) => style.stroke_width = (v * style.font_size).max(0.0),
            Value::Raw(raw) => {
                if let Ok(v) = raw.trim().parse::<f32>() {
                    style.stroke_width = v.max(0.0);
                }
            }
            _ => {}
        },
        "stop-color" => match value {
            Value::Color(r, g, b) => style.stop_color = [*r, *g, *b],
            Value::ColorA(r, g, b, _) => style.stop_color = [*r, *g, *b],
            Value::Keyword(k) if k.eq_ignore_ascii_case("currentcolor") => {
                style.stop_color = style.color;
            }
            _ => {}
        },
        "background-color" | "background" => match value {
            Value::Color(r, g, b) => style.background = Some([*r, *g, *b]),
            Value::ColorA(r, g, b, _) => style.background = Some([*r, *g, *b]),
            // Stage 11 Rec 3: `background: url(...)` is a
            // shorthand — the image rides the same property.
            // A shorthand resets the whole background-image, so
            // it clears a gradient too (Stage 15 Rec 7).
            Value::Url(url) => {
                style.background_image = Some(url.clone());
                style.background_gradient = None;
            }
            Value::Gradient(g) => {
                style.background_gradient = Some(g.clone());
                style.background_image = None;
            }
            Value::Raw(raw) => {
                if let Some(url) = pluto_css::extract_css_urls(raw).into_iter().next() {
                    style.background_image = Some(url);
                    style.background_gradient = None;
                }
            }
            _ => {}
        },
        // Stage 11 Rec 3: background-image properties.
        "background-image" => match value {
            Value::Url(url) => {
                style.background_image = Some(url.clone());
                style.background_gradient = None;
            }
            // Stage 15 (M7 Phase 2, Rec 7): a gradient
            // background-image replaces a URL one (one property).
            Value::Gradient(g) => {
                style.background_gradient = Some(g.clone());
                style.background_image = None;
            }
            Value::Keyword(k) if k == "none" => {
                style.background_image = None;
                style.background_gradient = None;
            }
            Value::Raw(raw) => {
                if let Some(url) = pluto_css::extract_css_urls(raw).into_iter().next() {
                    style.background_image = Some(url);
                    style.background_gradient = None;
                }
            }
            _ => {}
        },
        "background-size" => {
            style.background_size = match value {
                Value::Keyword(k) if k == "cover" => BackgroundSize::Cover,
                Value::Keyword(k) if k == "contain" => BackgroundSize::Contain,
                Value::Keyword(k) if k == "auto" => BackgroundSize::Auto,
                Value::Raw(raw) => {
                    let mut parts = raw.split_whitespace();
                    let first = parts
                        .next()
                        .and_then(|t| dim_value(&parse_value(t), style.font_size, root_font_size, viewport));
                    let second = parts
                        .next()
                        .and_then(|t| dim_value(&parse_value(t), style.font_size, root_font_size, viewport));
                    match (first, second) {
                        (Some(w), Some(h)) => BackgroundSize::Lengths(Some(w), Some(h)),
                        (Some(w), None) => BackgroundSize::Lengths(Some(w), None),
                        (None, _) => BackgroundSize::Auto,
                    }
                }
                v => match dim_value(v, style.font_size, root_font_size, viewport) {
                    Some(d) => BackgroundSize::Lengths(Some(d), None),
                    None => BackgroundSize::Auto,
                },
            };
        }
        "background-repeat" => {
            if let Value::Keyword(k) = value {
                style.background_repeat = match k.as_str() {
                    "no-repeat" => BackgroundRepeat::NoRepeat,
                    "repeat-x" => BackgroundRepeat::RepeatX,
                    "repeat-y" => BackgroundRepeat::RepeatY,
                    _ => BackgroundRepeat::Repeat,
                };
            } else if let Value::Raw(raw) = value {
                // Two-value form (`repeat no-repeat`, `no-repeat
                // repeat`): each axis repeats independently. A
                // single keyword cannot arrive here (it parses to
                // `Keyword` above).
                let parts: Vec<&str> = raw.split_whitespace().collect();
                let axis = |tok: &str| match tok {
                    "repeat" => Some(true),
                    "no-repeat" => Some(false),
                    _ => None,
                };
                match parts.as_slice() {
                    [a, b] => {
                        if let (Some(x), Some(y)) = (axis(a), axis(b)) {
                            style.background_repeat = BackgroundRepeat::RepeatXY(x, y);
                        }
                    }
                    [a] => match *a {
                        "repeat-x" => style.background_repeat = BackgroundRepeat::RepeatX,
                        "repeat-y" => style.background_repeat = BackgroundRepeat::RepeatY,
                        _ => {
                            if let Some(x) = axis(a) {
                                style.background_repeat = if x {
                                    BackgroundRepeat::Repeat
                                } else {
                                    BackgroundRepeat::NoRepeat
                                };
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        "background-position" => {
            // Keywords: `left`/`center`/`right` (x),
            // `top`/`center`/`bottom` (y); one token sets x and
            // centres y; lengths resolve via `dim_value`.
            // Multi-token values arrive as `Raw`.
            let parse_pos = |tok: &str, axis_is_x: bool| -> Option<BackgroundPosition> {
                match tok {
                    "left" if axis_is_x => Some(BackgroundPosition::Start),
                    "top" if !axis_is_x => Some(BackgroundPosition::Start),
                    "center" => Some(BackgroundPosition::Center),
                    "right" if axis_is_x => Some(BackgroundPosition::End),
                    "bottom" if !axis_is_x => Some(BackgroundPosition::End),
                    // A horizontal keyword in the y slot (or vice
                    // versa) is invalid CSS: `None` falls back to
                    // centring at the call site.
                    _ => dim_value(&parse_value(tok), style.font_size, root_font_size, viewport)
                        .map(BackgroundPosition::Length),
                }
            };
            let (x, y) = match value {
                Value::Keyword(k) => (
                    parse_pos(k, true).unwrap_or(BackgroundPosition::Start),
                    BackgroundPosition::Center,
                ),
                Value::Raw(raw) => {
                    let mut parts = raw.split_whitespace();
                    let first = parts.next().unwrap_or("left");
                    let second = parts.next();
                    let x = parse_pos(first, true).unwrap_or(BackgroundPosition::Start);
                    let y = match second {
                        Some(tok) => parse_pos(tok, false).unwrap_or(BackgroundPosition::Center),
                        None => BackgroundPosition::Center,
                    };
                    (x, y)
                }
                v => (
                    dim_value(v, style.font_size, root_font_size, viewport)
                        .map(BackgroundPosition::Length)
                        .unwrap_or(BackgroundPosition::Start),
                    BackgroundPosition::Center,
                ),
            };
            style.background_position = (x, y);
        }
        "text-align" => {
            if let Value::Keyword(k) = value {
                style.text_align = match k.as_str() {
                    "center" => TextAlign::Center,
                    "right" | "end" => TextAlign::Right,
                    "justify" => TextAlign::Justify,
                    _ => TextAlign::Left,
                };
            }
        }
        "white-space" => {
            if let Value::Keyword(k) = value {
                style.white_space = match k.as_str() {
                    "pre" => WhiteSpace::Pre,
                    "nowrap" => WhiteSpace::Nowrap,
                    "pre-wrap" => WhiteSpace::PreWrap,
                    "pre-line" => WhiteSpace::PreLine,
                    _ => WhiteSpace::Normal,
                };
            }
        }
        "contain" => {
            // `contain: layout` (and `strict` / `content`, which imply
            // it) marks the subtree layout-contained — a layout-root
            // signal for Stage 8 Rec 6. `size` / `paint` / `style`
            // alone do not imply layout containment.
            let has_layout = match value {
                Value::Keyword(k) => matches!(k.as_str(), "layout" | "strict" | "content"),
                Value::Raw(raw) => raw
                    .split_whitespace()
                    .any(|w| matches!(w, "layout" | "strict" | "content")),
                _ => false,
            };
            style.contain_layout = has_layout;
        }
        "overflow" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overflow_kind(k) {
                    style.overflow_x = o;
                    style.overflow_y = o;
                }
        }
        "overflow-x" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overflow_kind(k) {
                    style.overflow_x = o;
                }
        }
        "overflow-y" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overflow_kind(k) {
                    style.overflow_y = o;
                }
        }
        "overscroll-behavior" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_x = o;
                    style.overscroll_behavior_y = o;
                }
        }
        "overscroll-behavior-x" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_x = o;
                }
        }
        "overscroll-behavior-y" => {
            if let Value::Keyword(k) = value
                && let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_y = o;
                }
        }
        "box-sizing" => {
            if let Value::Keyword(k) = value {
                style.box_sizing = match k.as_str() {
                    "border-box" => BoxSizingKind::BorderBox,
                    _ => BoxSizingKind::ContentBox,
                };
            }
        }
        "border-radius" => {
            if let Value::BorderRadius(br) = value {
                style.border_radius = Some(*br);
            }
        }
        "box-shadow" => {
            if let Value::BoxShadow(shadows) = value {
                style.box_shadow = shadows.clone();
            }
        }
        "flex-direction" => {
            if let Value::Keyword(k) = value {
                style.flex_direction = match k.as_str() {
                    "row-reverse" => FlexDirection::RowReverse,
                    "column" => FlexDirection::Column,
                    "column-reverse" => FlexDirection::ColumnReverse,
                    _ => FlexDirection::Row,
                };
            }
        }
        "flex-wrap" => {
            if let Value::Keyword(k) = value {
                style.flex_wrap = match k.as_str() {
                    "wrap" => FlexWrap::Wrap,
                    "wrap-reverse" => FlexWrap::WrapReverse,
                    _ => FlexWrap::NoWrap,
                };
            }
        }
        "flex-grow" => {
            if let Some(n) = number_value(value) {
                style.flex_grow = n;
            }
        }
        "flex-shrink" => {
            if let Some(n) = number_value(value) {
                style.flex_shrink = n;
            }
        }
        "flex-basis" => {
            style.flex_basis = match value {
                Value::Keyword(k) if k == "auto" => None,
                _ => dim_value(value, style.font_size, root_font_size, viewport),
            }
        }
        "grid-template-columns" => {
            if let Value::Raw(raw) = value {
                style.grid_template_columns = parse_track_list(raw, style.font_size, root_font_size, viewport);
            }
        }
        "grid-template-rows" => {
            if let Value::Raw(raw) = value {
                style.grid_template_rows = parse_track_list(raw, style.font_size, root_font_size, viewport);
            }
        }
        "gap" | "grid-gap" => match value {
            Value::Raw(raw) => {
                let parts: Vec<&str> = raw.split_whitespace().collect();
                let row = parts.first().and_then(|t| {
                    parse_spacing_single(t, style.font_size, root_font_size, viewport)
                        .and_then(spacing_to_dim)
                });
                let col = parts
                    .get(1)
                    .and_then(|t| {
                        parse_spacing_single(t, style.font_size, root_font_size, viewport)
                            .and_then(spacing_to_dim)
                    })
                    .or(row);
                if let (Some(r), Some(c)) = (row, col) {
                    style.grid_gap = (r, c);
                }
            }
            _ => {
                if let Some(d) = dim_value(value, style.font_size, root_font_size, viewport) {
                    style.grid_gap = (d, d);
                }
            }
        },
        "row-gap" => {
            if let Some(d) = dim_value(value, style.font_size, root_font_size, viewport) {
                style.grid_gap.0 = d;
            }
        }
        "column-gap" => {
            if let Some(d) = dim_value(value, style.font_size, root_font_size, viewport) {
                style.grid_gap.1 = d;
            }
        }
        "grid-column" => {
            if let Value::Raw(raw) = value {
                style.grid_column = parse_grid_placement(raw);
            }
        }
        "grid-row" => {
            if let Value::Raw(raw) = value {
                style.grid_row = parse_grid_placement(raw);
            }
        }
        "grid-column-start" => {
            if let Value::Raw(raw) = value {
                style.grid_column.0 = parse_grid_place(raw);
            }
        }
        "grid-column-end" => {
            if let Value::Raw(raw) = value {
                style.grid_column.1 = parse_grid_place(raw);
            }
        }
        "grid-row-start" => {
            if let Value::Raw(raw) = value {
                style.grid_row.0 = parse_grid_place(raw);
            }
        }
        "grid-row-end" => {
            if let Value::Raw(raw) = value {
                style.grid_row.1 = parse_grid_place(raw);
            }
        }
        "grid-auto-flow" => {
            // Single keywords arrive as `Keyword`, two-word forms
            // (`row dense`) as `Raw`; both go through the same
            // parser. An invalid value drops the declaration (the
            // `display` arm's early `return`): the previous /
            // initial row flow stands.
            let raw = match value {
                Value::Keyword(k) => k.as_str(),
                Value::Raw(raw) => raw.as_str(),
                _ => return,
            };
            if let Some(flow) = parse_grid_auto_flow(raw) { style.grid_auto_flow = flow }
        }
        // M6 Phase 1: stacking / transform / transition / animation
        // model (paint order, the animation engine and the
        // `CompositingOnly` classifier are later phases — these
        // fields are deliberately NOT in
        // `layout_properties_changed`).
        "z-index" => match value {
            Value::Keyword(k) if k == "auto" => style.z_index = None,
            // Bare numbers arrive as `Length` (the parser cannot
            // tell `2` from `2px`); only integral ones are valid
            // `z-index` values.
            Value::Length(n) if n.fract() == 0.0 => style.z_index = Some(*n as i32),
            _ => {}
        },
        "opacity" => match value {
            Value::Length(n) => style.opacity = n.clamp(0.0, 1.0),
            Value::Percentage(p) => style.opacity = (p / 100.0).clamp(0.0, 1.0),
            _ => {}
        },
        "transform" => match value {
            Value::Transform(list) => style.transform = list.clone(),
            Value::Keyword(k) if k == "none" => style.transform = Vec::new(),
            _ => {}
        },
        "transform-origin" => {
            if let Some(origin) =
                parse_transform_origin(value, style.font_size, root_font_size, viewport)
            {
                style.transform_origin = origin;
            }
        }
        "transform-box" => {
            if let Value::Keyword(k) = value {
                style.transform_box = match k.as_str() {
                    "content-box" => TransformBox::ContentBox,
                    "border-box" => TransformBox::BorderBox,
                    "fill-box" => TransformBox::FillBox,
                    "stroke-box" => TransformBox::StrokeBox,
                    _ => TransformBox::ViewBox,
                };
            }
        }
        "will-change" => {
            // `will-change: transform` is a hint only — it does
            // NOT create a stacking context. `opacity` / `auto` /
            // `contents` parse without effect.
            let has_transform = match value {
                Value::Keyword(k) => k == "transform",
                Value::Raw(raw) => raw.split_whitespace().any(|w| w == "transform"),
                _ => false,
            };
            style.will_change_transform = has_transform;
        }
        "transition" => {
            if let Value::Transitions(list) = value {
                style.transitions = list.clone();
            }
        }
        "transition-property" => {
            let items = longhand_values(value);
            let names: Option<Vec<String>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => Some(k.clone()),
                    Value::Raw(r) => Some(r.trim().to_string()),
                    _ => None,
                })
                .collect();
            if let Some(names) = names {
                apply_transition_longhand(&mut style.transitions, &names, |t, name| {
                    t.property = name.clone()
                });
            }
        }
        "transition-duration" => {
            let times = longhand_times(value);
            if let Some(times) = times {
                apply_transition_longhand(&mut style.transitions, &times, |t, d| {
                    t.duration = *d
                });
            }
        }
        "transition-timing-function" => {
            let items = longhand_values(value);
            let easings: Option<Vec<Easing>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => Easing::parse(k),
                    Value::Raw(r) => Easing::parse(r),
                    _ => None,
                })
                .collect();
            if let Some(easings) = easings {
                apply_transition_longhand(&mut style.transitions, &easings, |t, e| {
                    t.timing = e.clone()
                });
            }
        }
        "transition-delay" => {
            let times = longhand_times(value);
            if let Some(times) = times {
                apply_transition_longhand(&mut style.transitions, &times, |t, d| {
                    t.delay = *d
                });
            }
        }
        "animation" => {
            if let Value::Animations(list) = value {
                style.animations = list.clone();
            }
        }
        "animation-name" => {
            let items = longhand_values(value);
            let names: Option<Vec<String>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => Some(k.clone()),
                    Value::Raw(r) => Some(r.trim().to_string()),
                    _ => None,
                })
                .collect();
            if let Some(names) = names {
                apply_animation_longhand(&mut style.animations, &names, |a, name| {
                    a.name = name.clone()
                });
            }
        }
        "animation-duration" => {
            let times = longhand_times(value);
            if let Some(times) = times {
                apply_animation_longhand(&mut style.animations, &times, |a, d| {
                    a.duration = *d
                });
            }
        }
        "animation-timing-function" => {
            let items = longhand_values(value);
            let easings: Option<Vec<Easing>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => Easing::parse(k),
                    Value::Raw(r) => Easing::parse(r),
                    _ => None,
                })
                .collect();
            if let Some(easings) = easings {
                apply_animation_longhand(&mut style.animations, &easings, |a, e| {
                    a.timing = e.clone()
                });
            }
        }
        "animation-delay" => {
            let times = longhand_times(value);
            if let Some(times) = times {
                apply_animation_longhand(&mut style.animations, &times, |a, d| {
                    a.delay = *d
                });
            }
        }
        "animation-iteration-count" => {
            let items = longhand_values(value);
            let counts: Option<Vec<IterationCount>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) if k == "infinite" => {
                        Some(IterationCount::Infinite)
                    }
                    Value::Length(n) => Some(IterationCount::Finite(*n)),
                    Value::Raw(r) => r
                        .trim()
                        .parse::<f32>()
                        .ok()
                        .map(IterationCount::Finite),
                    _ => None,
                })
                .collect();
            if let Some(counts) = counts {
                apply_animation_longhand(&mut style.animations, &counts, |a, c| {
                    a.iteration_count = *c
                });
            }
        }
        "animation-direction" => {
            let items = longhand_values(value);
            let dirs: Option<Vec<AnimationDirection>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => match k.as_str() {
                        "normal" => Some(AnimationDirection::Normal),
                        "reverse" => Some(AnimationDirection::Reverse),
                        "alternate" => Some(AnimationDirection::Alternate),
                        "alternate-reverse" => Some(AnimationDirection::AlternateReverse),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            if let Some(dirs) = dirs {
                apply_animation_longhand(&mut style.animations, &dirs, |a, d| {
                    a.direction = *d
                });
            }
        }
        "animation-fill-mode" => {
            let items = longhand_values(value);
            let fills: Option<Vec<FillMode>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => match k.as_str() {
                        "none" => Some(FillMode::None),
                        "forwards" => Some(FillMode::Forwards),
                        "backwards" => Some(FillMode::Backwards),
                        "both" => Some(FillMode::Both),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            if let Some(fills) = fills {
                apply_animation_longhand(&mut style.animations, &fills, |a, f| {
                    a.fill_mode = *f
                });
            }
        }
        "animation-play-state" => {
            let items = longhand_values(value);
            let states: Option<Vec<PlayState>> = items
                .iter()
                .map(|v| match v {
                    Value::Keyword(k) => match k.as_str() {
                        "running" => Some(PlayState::Running),
                        "paused" => Some(PlayState::Paused),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            if let Some(states) = states {
                apply_animation_longhand(&mut style.animations, &states, |a, s| {
                    a.play_state = *s
                });
            }
        }
        "inset" => {
            // 1–4 values → top/right/bottom/left, same expansion
            // shape as `margin`/`padding`. A single value arrives
            // typed; several arrive as `Raw`.
            let tokens: Vec<Value> = match value {
                Value::Raw(raw) => raw.split_whitespace().map(parse_value).collect(),
                v => vec![v.clone()],
            };
            if !tokens.is_empty() && tokens.len() <= 4 {
                let dims: Vec<Option<Dim>> = tokens
                    .iter()
                    .map(|t| dim_value(t, style.font_size, root_font_size, viewport))
                    .collect();
                let (top, right, bottom, left) = match dims.len() {
                    1 => (dims[0], dims[0], dims[0], dims[0]),
                    2 => (dims[0], dims[1], dims[0], dims[1]),
                    3 => (dims[0], dims[1], dims[2], dims[1]),
                    4 => (dims[0], dims[1], dims[2], dims[3]),
                    _ => unreachable!(),
                };
                style.inset = [top, right, bottom, left];
            }
        }
        // Deliberately unmodelled in M2 (stored nowhere, never an
        // error): float/clear (no float layout yet), border widths
        // (no border painting yet), text-decoration, list-style-type,
        // visibility, justify-content/align-*,
        // cursor, content. They are ignored
        // here; adding them is data, not a rewrite.
        _ => {}
    }
}

/// Parse `transform-origin`: 1–2 components, each a length, a
/// percentage, or a keyword (`left` / `center` / `right` for x,
/// `top` / `center` / `bottom` for y). One value centres the
/// other axis. `None` when the value is invalid (the declaration
/// is then ignored, keeping the previous origin).
fn parse_transform_origin(
    value: &Value,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Option<TransformOrigin> {
    let tokens: Vec<Value> = match value {
        Value::Keyword(k) => vec![Value::Keyword(k.clone())],
        Value::Raw(raw) => raw.split_whitespace().map(parse_value).collect(),
        v => vec![v.clone()],
    };
    if tokens.is_empty() || tokens.len() > 2 {
        return None;
    }
    let component = |v: &Value| -> Option<OriginComponent> {
        match v {
            Value::Keyword(k) => match k.as_str() {
                "left" => Some(OriginComponent::Keyword(OriginKeyword::Left)),
                "center" => Some(OriginComponent::Keyword(OriginKeyword::Center)),
                "right" => Some(OriginComponent::Keyword(OriginKeyword::Right)),
                "top" => Some(OriginComponent::Keyword(OriginKeyword::Top)),
                "bottom" => Some(OriginComponent::Keyword(OriginKeyword::Bottom)),
                _ => None,
            },
            Value::Percentage(p) => Some(OriginComponent::Percentage(*p)),
            _ => dim_value(v, font_size, root_font_size, viewport)
                .map(OriginComponent::Length),
        }
    };
    let first = component(&tokens[0])?;
    if tokens.len() == 1 {
        return Some(match first {
            OriginComponent::Keyword(OriginKeyword::Top)
            | OriginComponent::Keyword(OriginKeyword::Bottom) => TransformOrigin {
                x: OriginComponent::Keyword(OriginKeyword::Center),
                y: first,
            },
            _ => TransformOrigin {
                x: first,
                y: OriginComponent::Keyword(OriginKeyword::Center),
            },
        });
    }
    let second = component(&tokens[1])?;
    // Two keywords may arrive in either order (`top left`);
    // lengths / percentages are positional (first = x).
    let (x, y) = match (first, second) {
        (
            OriginComponent::Keyword(OriginKeyword::Top)
            | OriginComponent::Keyword(OriginKeyword::Bottom),
            OriginComponent::Keyword(
                OriginKeyword::Left | OriginKeyword::Center | OriginKeyword::Right,
            ),
        ) => (second, first),
        (a, b) => (a, b),
    };
    Some(TransformOrigin { x, y })
}

/// Split a comma-separated longhand list (`transition-duration:
/// .3s, .5s`) into its per-item values. A single value arrives
/// typed; a list arrives as `Raw` and is split at top level
/// (commas inside `cubic-bezier()` do not split).
fn longhand_values(value: &Value) -> Vec<Value> {
    match value {
        Value::Raw(raw) => {
            // Split on top-level commas without re-entering the
            // css parser's private splitter: a small depth walk.
            let mut out = Vec::new();
            let mut depth = 0i32;
            let mut start = 0usize;
            for (i, c) in raw.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    ',' if depth == 0 => {
                        out.push(parse_value(raw[start..i].trim()));
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            out.push(parse_value(raw[start..].trim()));
            out
        }
        v => vec![v.clone()],
    }
}

/// A `transition-duration` / `transition-delay` /
/// `animation-duration` / `animation-delay` longhand as seconds.
/// `None` when any item is not a time (the whole declaration is
/// then ignored).
fn longhand_times(value: &Value) -> Option<Vec<f32>> {
    longhand_values(value)
        .iter()
        .map(|v| match v {
            Value::Time(t) => Some(*t),
            _ => None,
        })
        .collect()
}

/// Apply one `transition-*` longhand list to the transitions vec:
/// items zip by index, repeating the list when it is shorter than
/// the vec (CSS list repetition); an empty vec is grown from the
/// list. `apply` sets the one field this longhand owns.
fn apply_transition_longhand<T>(
    transitions: &mut Vec<Transition>,
    items: &[T],
    mut apply: impl FnMut(&mut Transition, &T),
) {
    if items.is_empty() {
        return;
    }
    let n = transitions.len().max(items.len());
    while transitions.len() < n {
        transitions.push(Transition::default());
    }
    for i in 0..n {
        apply(&mut transitions[i], &items[i % items.len()]);
    }
}

/// The `animation-*` counterpart of
/// [`apply_transition_longhand`].
fn apply_animation_longhand<T>(
    animations: &mut Vec<Animation>,
    items: &[T],
    mut apply: impl FnMut(&mut Animation, &T),
) {
    if items.is_empty() {
        return;
    }
    let n = animations.len().max(items.len());
    while animations.len() < n {
        animations.push(Animation::default());
    }
    for i in 0..n {
        apply(&mut animations[i], &items[i % items.len()]);
    }
}

fn overflow_kind(k: &str) -> Option<OverflowKind> {
    Some(match k {
        "visible" => OverflowKind::Visible,
        "hidden" => OverflowKind::Hidden,
        "scroll" => OverflowKind::Scroll,
        "auto" => OverflowKind::Auto,
        "clip" => OverflowKind::Clip,
        _ => return None,
    })
}

/// `overscroll-behavior` keyword to value (Stage 8 Rec 9). The
/// two-value form (`overscroll-behavior: contain auto`) is not
/// parsed — the shorthand takes one keyword for both axes.
fn overscroll_behavior(k: &str) -> Option<OverscrollBehavior> {
    Some(match k {
        "auto" => OverscrollBehavior::Auto,
        "contain" => OverscrollBehavior::Contain,
        "none" => OverscrollBehavior::None,
        _ => return None,
    })
}

fn number_value(value: &Value) -> Option<f32> {
    match value {
        Value::Length(v) => Some(*v),
        Value::Raw(raw) => raw.trim().parse().ok(),
        _ => None,
    }
}

/// A width/height/min/max value. `em`/`rem` resolve now (computed
/// values, against the element's own / root font size); `%` and
/// `vh`/`vw` stay unresolved for layout (`Dim::Pct` / `Dim::Vh` /
/// `Dim::Vw`); `auto` is `None`. `calc()` / `clamp()` evaluate
/// here; a `%` inside them has no containing-block basis at style
/// time, so a mixed calc is unresolvable and yields `None` (the
/// declaration falls back to `auto` / initial).
/// Map a parsed CSS value to an SVG [`Paint`] (Stage 15, M7 Phase 1,
/// Rec 6): a colour, `none`, a `url(...)` paint-server reference, or
/// `currentColor`. Anything else (e.g. a gradient function, which is
/// Rec 7) returns `None` and the declaration is dropped.
fn paint_value(value: &Value) -> Option<Paint> {
    match value {
        Value::Color(r, g, b) => Some(Paint::Color([*r, *g, *b])),
        // Alpha is dropped: paint has no alpha channel yet.
        Value::ColorA(r, g, b, _) => Some(Paint::Color([*r, *g, *b])),
        Value::Keyword(k) if k == "none" => Some(Paint::None),
        Value::Keyword(k) if k.eq_ignore_ascii_case("currentcolor") => {
            Some(Paint::CurrentColor)
        }
        Value::Url(url) => Some(Paint::Url(url.clone())),
        _ => None,
    }
}

fn dim_value(
    value: &Value,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Option<Dim> {
    dim_value_with_basis(value, font_size, root_font_size, viewport, None)
}

/// [`dim_value`] with a percentage basis, for call sites that
/// have one (a `calc(100% - 20px)` only evaluates when the basis
/// is known).
fn dim_value_with_basis(
    value: &Value,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
    pct_basis: Option<f32>,
) -> Option<Dim> {
    match value {
        Value::Length(v) => Some(Dim::Px(*v)),
        Value::Percentage(p) => Some(Dim::Pct(p / 100.0)),
        Value::Em(v) => Some(Dim::Px(v * font_size)),
        Value::Rem(v) => Some(Dim::Px(v * root_font_size)),
        Value::Vh(v) => Some(Dim::Vh(*v)),
        Value::Vw(v) => Some(Dim::Vw(*v)),
        Value::Keyword(k) if k == "auto" => None,
        Value::Calc(expr) => {
            let ctx = CalcCtx {
                font_size,
                root_font_size,
                viewport,
                pct_basis,
            };
            match expr.eval(&ctx)? {
                CalcNumber::Px(v) => Some(Dim::Px(v)),
                CalcNumber::Pct(p) => Some(Dim::Pct(p / 100.0)),
            }
        }
        Value::Clamp { min, val, max } => {
            let ctx = CalcCtx {
                font_size,
                root_font_size,
                viewport,
                pct_basis,
            };
            let (lo, v, hi) = (min.eval(&ctx)?, val.eval(&ctx)?, max.eval(&ctx)?);
            match (lo, v, hi) {
                (CalcNumber::Px(a), CalcNumber::Px(b), CalcNumber::Px(c)) => {
                    Some(Dim::Px(b.clamp(a, c)))
                }
                (CalcNumber::Pct(a), CalcNumber::Pct(b), CalcNumber::Pct(c)) => {
                    Some(Dim::Pct(b.clamp(a, c) / 100.0))
                }
                _ => None,
            }
        }
        Value::Raw(raw) => {
            parse_spacing_single(raw, font_size, root_font_size, viewport)
                .and_then(spacing_to_dim)
        }
        _ => None,
    }
}

fn spacing_to_dim(sp: Spacing) -> Option<Dim> {
    Some(match sp {
        Spacing::Px(v) => Dim::Px(v),
        Spacing::Pct(f) => Dim::Pct(f),
        Spacing::Vh(v) => Dim::Vh(v),
        Spacing::Vw(v) => Dim::Vw(v),
        Spacing::Auto => return None,
    })
}

/// A margin/padding value, resolving `em`/`rem` against the font
/// sizes; `vh`/`vw` stay as used values for `to_taffy_style`.
/// `calc()` / `clamp()` evaluate through `dim_value` (same
/// no-basis limitation as there).
fn spacing_value(
    value: &Value,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Option<Spacing> {
    match value {
        Value::Length(v) => Some(Spacing::Px(*v)),
        Value::Percentage(p) => Some(Spacing::Pct(p / 100.0)),
        Value::Em(v) => Some(Spacing::Px(v * font_size)),
        Value::Rem(v) => Some(Spacing::Px(v * root_font_size)),
        Value::Vh(v) => Some(Spacing::Vh(*v)),
        Value::Vw(v) => Some(Spacing::Vw(*v)),
        Value::Keyword(k) if k == "auto" => Some(Spacing::Auto),
        Value::Calc(_) | Value::Clamp { .. } => {
            dim_value(value, font_size, root_font_size, viewport).map(|d| match d {
                Dim::Px(v) => Spacing::Px(v),
                Dim::Pct(f) => Spacing::Pct(f),
                Dim::Vh(v) => Spacing::Vh(v),
                Dim::Vw(v) => Spacing::Vw(v),
            })
        }
        Value::Raw(raw) => parse_spacing_single(raw, font_size, root_font_size, viewport),
        _ => None,
    }
}

fn parse_spacing_single(
    s: &str,
    font_size: f32,
    root_font_size: f32,
    // Kept for signature symmetry with the other value parsers;
    // `vh`/`vw` stay unresolved here (they resolve in `to_taffy_style`).
    _viewport: (f32, f32),
) -> Option<Spacing> {
    let s = s.trim().to_ascii_lowercase();
    if s == "auto" {
        return Some(Spacing::Auto);
    }
    // NOTE: `rem` must be checked before `em` ("2rem" ends in "em").
    if let Some(n) = s.strip_suffix("rem") {
        return n.trim().parse().ok().map(|v: f32| Spacing::Px(v * root_font_size));
    }
    if let Some(n) = s.strip_suffix("em") {
        return n.trim().parse().ok().map(|v: f32| Spacing::Px(v * font_size));
    }
    if let Some(n) = s.strip_suffix("vh") {
        return n.trim().parse().ok().map(Spacing::Vh);
    }
    if let Some(n) = s.strip_suffix("vw") {
        return n.trim().parse().ok().map(Spacing::Vw);
    }
    if let Some(n) = s.strip_suffix("px") {
        return n.trim().parse().ok().map(Spacing::Px);
    }
    if let Some(n) = s.strip_suffix('%') {
        return n.trim().parse().ok().map(|v: f32| Spacing::Pct(v / 100.0));
    }
    s.parse::<f32>().ok().map(Spacing::Px)
}

/// Expand a 1-4 token `margin`/`padding` shorthand (`Raw` form, i.e. it
/// contained a unit the CSS parser kept verbatim). `auto` is kept
/// for margins; for padding it becomes 0 (handled in `to_taffy_style`).
fn parse_shorthand(
    raw: &str,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Option<[Spacing; 4]> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        vals.push(parse_spacing_single(t, font_size, root_font_size, viewport)?);
    }
    let (top, right, bottom, left) = match vals.len() {
        1 => (vals[0], vals[0], vals[0], vals[0]),
        2 => (vals[0], vals[1], vals[0], vals[1]),
        3 => (vals[0], vals[1], vals[2], vals[1]),
        4 => (vals[0], vals[1], vals[2], vals[3]),
        _ => unreachable!(),
    };
    Some([top, right, bottom, left])
}

/// Parse `grid-template-columns` / `-rows`: whitespace-separated
/// tracks — `100px`, `50%`, `2em`, `1fr`, `auto`. `minmax(...)` and
/// `repeat(...)` are not modelled in M2 and parse as `Auto` (the
/// track still exists, just auto-sized).
fn parse_track_list(
    raw: &str,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Vec<GridTrack> {
    raw.split_whitespace()
        .map(|tok| {
            let lower = tok.to_ascii_lowercase();
            if lower == "auto" || lower.starts_with("minmax(") || lower.starts_with("repeat(") {
                GridTrack::Auto
            } else if let Some(n) = lower.strip_suffix("fr") {
                n.trim().parse().map(GridTrack::Fraction).unwrap_or(GridTrack::Auto)
            } else {
                match parse_spacing_single(tok, font_size, root_font_size, viewport)
                    .and_then(spacing_to_dim)
                {
                    Some(d) => GridTrack::Fixed(d),
                    None => GridTrack::Auto,
                }
            }
        })
        .collect()
}

/// Parse `grid-auto-flow`: `[ row | column ] || dense` — `row`,
/// `column`, `row dense`, `column dense`, and the `dense`-only form
/// (which means `row dense`, per the grammar and taffy's `FromCss`).
/// Keywords are case-insensitive. Anything else (`row column`,
/// `dense dense`, `sideways`, empty, three tokens, ...) is `None`
/// and the caller drops the declaration.
fn parse_grid_auto_flow(raw: &str) -> Option<GridAutoFlow> {
    let mut axis: Option<bool> = None; // false = row, true = column
    let mut dense = false;
    for tok in raw.split_whitespace() {
        match tok.to_ascii_lowercase().as_str() {
            "row" | "column" => {
                if axis.is_some() {
                    return None;
                }
                axis = Some(tok.eq_ignore_ascii_case("column"));
            }
            "dense" => {
                if dense {
                    return None;
                }
                dense = true;
            }
            _ => return None,
        }
    }
    Some(match (axis, dense) {
        (Some(false), false) => GridAutoFlow::Row,
        (Some(true), false) => GridAutoFlow::Column,
        (Some(false), true) | (None, true) => GridAutoFlow::RowDense,
        (Some(true), true) => GridAutoFlow::ColumnDense,
        (None, false) => return None,
    })
}

fn parse_grid_place(raw: &str) -> GridPlace {
    let s = raw.trim().to_ascii_lowercase();
    if s == "auto" || s.is_empty() {
        return GridPlace::Auto;
    }
    if let Some(n) = s.strip_prefix("span ") {
        if let Ok(v) = n.trim().parse::<u16>() {
            return GridPlace::Span(v);
        }
        return GridPlace::Auto;
    }
    if let Ok(v) = s.parse::<i16>() {
        return GridPlace::Line(v);
    }
    GridPlace::Auto
}

/// Parse `grid-column` / `grid-row`: `start / end`, a single place,
/// or `span n`.
fn parse_grid_placement(raw: &str) -> (GridPlace, GridPlace) {
    let parts: Vec<&str> = raw.split('/').collect();
    match parts.len() {
        1 => (parse_grid_place(parts[0]), GridPlace::Auto),
        _ => (
            parse_grid_place(parts[0]),
            parse_grid_place(parts[1]),
        ),
    }
}

/// `ComputedStyle` (+ viewport) → taffy `Style`: the used-value step.
/// `%` resolves against the containing block inside taffy; `vh`/`vw`
/// resolve here against the viewport; `em`/`rem` are already px.
fn to_taffy_style(style: &ComputedStyle, viewport: (f32, f32)) -> Style {
    let mut s = Style {
        display: match style.display {
            DisplayKind::Block | DisplayKind::Inline => Display::Block,
            DisplayKind::Flex => Display::Flex,
            DisplayKind::Grid => Display::Grid,
            DisplayKind::FlowRoot => Display::FlowRoot,
            DisplayKind::InlineBlock => Display::Block,
            DisplayKind::None => Display::None,
        },
        position: match style.position {
            PositionKind::Static => Position::Static,
            PositionKind::Relative => Position::Relative,
            // Taffy has no `fixed`: absolute is the closest layout
            // behaviour; the shell re-anchors fixed boxes to the viewport.
            PositionKind::Absolute | PositionKind::Fixed => Position::Absolute,
            // No sticky in taffy: relative is the closest behaviour.
            PositionKind::Sticky => Position::Relative,
        },
        box_sizing: match style.box_sizing {
            BoxSizingKind::ContentBox => BoxSizing::ContentBox,
            BoxSizingKind::BorderBox => BoxSizing::BorderBox,
        },
        ..Default::default()
    };
    let dim = |d: Dim| match d {
        Dim::Px(v) => Dimension::length(v),
        Dim::Pct(f) => Dimension::percent(f),
        Dim::Vh(v) => Dimension::length(v / 100.0 * viewport.1),
        Dim::Vw(v) => Dimension::length(v / 100.0 * viewport.0),
    };
    let dim_opt = |d: Option<Dim>| d.map(dim).unwrap_or(Dimension::auto());
    s.size.width = dim_opt(style.width);
    s.size.height = dim_opt(style.height);
    // min/max sizes are `LengthPercentageAuto` in taffy, not `Dimension`.
    let lpa = |d: Option<Dim>| match d {
        Some(Dim::Px(v)) => LengthPercentageAuto::length(v),
        Some(Dim::Pct(f)) => LengthPercentageAuto::percent(f),
        Some(Dim::Vh(v)) => LengthPercentageAuto::length(v / 100.0 * viewport.1),
        Some(Dim::Vw(v)) => LengthPercentageAuto::length(v / 100.0 * viewport.0),
        None => LengthPercentageAuto::auto(),
    };
    s.min_size.width = lpa(style.min_width);
    s.min_size.height = lpa(style.min_height);
    s.max_size.width = lpa(style.max_width);
    s.max_size.height = lpa(style.max_height);
    let margin = |sp: Spacing| match sp {
        Spacing::Px(v) => LengthPercentageAuto::length(v),
        Spacing::Pct(f) => LengthPercentageAuto::percent(f),
        Spacing::Vh(v) => LengthPercentageAuto::length(v / 100.0 * viewport.1),
        Spacing::Vw(v) => LengthPercentageAuto::length(v / 100.0 * viewport.0),
        Spacing::Auto => LengthPercentageAuto::auto(),
    };
    let padding = |sp: Spacing| match sp {
        Spacing::Px(v) => LengthPercentage::length(v),
        Spacing::Pct(f) => LengthPercentage::percent(f),
        Spacing::Vh(v) => LengthPercentage::length(v / 100.0 * viewport.1),
        Spacing::Vw(v) => LengthPercentage::length(v / 100.0 * viewport.0),
        Spacing::Auto => LengthPercentage::length(0.0),
    };
    s.margin.top = margin(style.margin[0]);
    s.margin.right = margin(style.margin[1]);
    s.margin.bottom = margin(style.margin[2]);
    s.margin.left = margin(style.margin[3]);
    s.padding.top = padding(style.padding[0]);
    s.padding.right = padding(style.padding[1]);
    s.padding.bottom = padding(style.padding[2]);
    s.padding.left = padding(style.padding[3]);
    let inset = |d: Option<Dim>| match d {
        Some(Dim::Px(v)) => LengthPercentageAuto::length(v),
        Some(Dim::Pct(f)) => LengthPercentageAuto::percent(f),
        Some(Dim::Vh(v)) => LengthPercentageAuto::length(v / 100.0 * viewport.1),
        Some(Dim::Vw(v)) => LengthPercentageAuto::length(v / 100.0 * viewport.0),
        None => LengthPercentageAuto::auto(),
    };
    s.inset.top = inset(style.inset[0]);
    s.inset.right = inset(style.inset[1]);
    s.inset.bottom = inset(style.inset[2]);
    s.inset.left = inset(style.inset[3]);
    // M6 Phase 1 (report §4 sharpest trap): taffy 0.15 cannot
    // override `is_containing_block`, so a transformed element
    // does NOT become a containing block for absolutely-positioned
    // descendants in taffy. Mapping a transformed static box to
    // `Position::Relative` makes taffy resolve abs descendants
    // against it — the correct CSS containing block — without
    // moving the element itself (relative with auto insets lays
    // out identically to static). The insets are forced back to
    // `auto` here: a static box's `top`/`left` must not move it.
    // The predicate change in `has_outside_resolving_descendant`
    // (incremental.rs) lands in the same change, so the two never
    // disagree.
    if !style.transform.is_empty() && style.position == PositionKind::Static {
        s.position = Position::Relative;
        s.inset.top = LengthPercentageAuto::auto();
        s.inset.right = LengthPercentageAuto::auto();
        s.inset.bottom = LengthPercentageAuto::auto();
        s.inset.left = LengthPercentageAuto::auto();
    }
    let overflow = |o: OverflowKind| match o {
        OverflowKind::Visible => Overflow::Visible,
        OverflowKind::Hidden => Overflow::Hidden,
        // Taffy has no `auto`: `Scroll` is the closest behaviour
        // (clip + scrollbar); the "iff overflow" part is M2-approximate.
        OverflowKind::Scroll | OverflowKind::Auto => Overflow::Scroll,
        OverflowKind::Clip => Overflow::Clip,
    };
    s.overflow.x = overflow(style.overflow_x);
    s.overflow.y = overflow(style.overflow_y);
    s.flex_direction = style.flex_direction;
    s.flex_wrap = style.flex_wrap;
    s.flex_grow = style.flex_grow;
    s.flex_shrink = style.flex_shrink;
    if let Some(basis) = style.flex_basis {
        s.flex_basis = dim(basis);
    }
    // Grid (R12): taffy implements the grid algorithm — Pluto only
    // maps the properties.
    let track = |t: GridTrack| {
        let (min, max) = match t {
            GridTrack::Auto => (
                MinTrackSizingFunction::auto(),
                MaxTrackSizingFunction::auto(),
            ),
            GridTrack::Fraction(f) => (
                MinTrackSizingFunction::auto(),
                MaxTrackSizingFunction::fr(f),
            ),
            GridTrack::Fixed(d) => {
                let max = match d {
                    Dim::Px(v) => MaxTrackSizingFunction::length(v),
                    Dim::Pct(f) => MaxTrackSizingFunction::percent(f),
                    Dim::Vh(v) => MaxTrackSizingFunction::length(v / 100.0 * viewport.1),
                    Dim::Vw(v) => MaxTrackSizingFunction::length(v / 100.0 * viewport.0),
                };
                (MinTrackSizingFunction::auto(), max)
            }
        };
        GridTemplateComponent::Single(TrackSizingFunction { min, max })
    };
    s.grid_template_columns = style
        .grid_template_columns
        .iter()
        .copied()
        .map(track)
        .collect();
    s.grid_template_rows = style
        .grid_template_rows
        .iter()
        .copied()
        .map(track)
        .collect();
    let gap = |d: Dim| match d {
        Dim::Px(v) => LengthPercentage::length(v),
        Dim::Pct(f) => LengthPercentage::percent(f),
        Dim::Vh(v) => LengthPercentage::length(v / 100.0 * viewport.1),
        Dim::Vw(v) => LengthPercentage::length(v / 100.0 * viewport.0),
    };
    s.gap.width = gap(style.grid_gap.1);
    s.gap.height = gap(style.grid_gap.0);
    let place = |p: GridPlace| match p {
        GridPlace::Auto => GridPlacement::Auto,
        GridPlace::Line(i) => grid_line(i),
        GridPlace::Span(n) => grid_span(n),
    };
    s.grid_column = Line {
        start: place(style.grid_column.0),
        end: place(style.grid_column.1),
    };
    s.grid_row = Line {
        start: place(style.grid_row.0),
        end: place(style.grid_row.1),
    };
    // GridAutoFlow (Stage 10 Rec 5): taffy implements
    // auto-placement; Pluto only maps the property.
    s.grid_auto_flow = style.grid_auto_flow;
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::GradientKind;
    use pluto_css::TransformFunction;
    use pluto_html::parse as parse_html;

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
    }
    a:link, a:visited {
        color: #38488f;
    }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>"#;

    fn example_tree() -> LayoutTree {
        let dom = parse_html(EXAMPLE_COM);
        let style = find_element(&dom, dom.root().unwrap(), "style").unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        let sheet = Stylesheet::parse(&css);
        LayoutTree::build(&dom, &sheet, 1024.0, 768.0, Theme::Light).expect("layout builds")
    }

    fn box_for<'a>(tree: &'a LayoutTree, dom: &Dom, tag: &str) -> &'a LayoutBox {
        tree.boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.tag(id) == Some(tag)))
            .unwrap_or_else(|| panic!("no box for <{tag}>"))
    }

    /// Resolve styles for `html` + `css` at the given viewport.
    fn styles_for(
        html: &str,
        css: &str,
        viewport: (f32, f32),
    ) -> HashMap<DomId, Arc<ComputedStyle>> {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(css);
        let ua = Stylesheet::ua();
        resolve_document_styles(&dom, &ua, &sheet, false, viewport, &ElementState::default(), Theme::Light).0
    }

    #[test]
    fn example_com_layout() {
        let dom = parse_html(EXAMPLE_COM);
        let tree = example_tree();
        let body = box_for(&tree, &dom, "body");
        // The div's 5em top margin collapses through body/html (CSS margin
        // collapsing), so the body border box itself starts at y=80; the
        // grey canvas background still covers y=0 (see the paint test).
        assert_eq!((body.x, body.y), (0.0, 80.0));
        assert_eq!(body.width, 1024.0);
        assert_eq!(body.style.background, Some([240, 240, 242]));

        let div = box_for(&tree, &dom, "div");
        // CSS `box-sizing` initial is `content-box` (M2 maps it;
        // M1 inherited taffy's `BorderBox` default): width 600px is
        // the content box, so the border box is 600 + 2*32 padding.
        assert_eq!(div.width, 664.0);
        // width 664 + auto side margins centers the div in 1024.
        assert_eq!(div.x, 180.0);
        // 5em top margin at 16px font.
        assert_eq!(div.y, 80.0);
        assert_eq!(div.style.background, Some([253, 253, 255]));

        let h1 = box_for(&tree, &dom, "h1");
        // Inside the div: 2em (32px) padding, then the h1's own 0.67em
        // top margin (0.67 * 32 = 21.44, rounded by taffy to 21).
        assert_eq!(h1.x, 180.0 + 32.0);
        assert_eq!(h1.y, 80.0 + 32.0 + 21.0);
        assert_eq!(h1.style.font_size, 32.0);
        // The h1's text lives in its child text leaf (anonymous box),
        // not on the element box itself.
        let h1_text = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("Example Domain")))
            })
            .expect("h1 text leaf");
        assert!(!h1_text.glyphs.is_empty(), "h1 text was shaped");

        // Both paragraphs laid out below the heading, inside the div.
        let ps: Vec<&LayoutBox> = tree
            .boxes
            .iter()
            .filter(|b| b.dom.is_some_and(|id| dom.tag(id) == Some("p")))
            .collect();
        assert_eq!(ps.len(), 2);
        assert!(ps[0].y > h1.y + h1.height);
        assert!(ps[1].y > ps[0].y + ps[0].height);
        assert!(ps[0].x >= div.x + 32.0 && ps[0].x + ps[0].width <= div.x + div.width - 32.0);

        // The link text is shaped in its own text leaf. M2 applies the
        // stylesheet's `a:link` rule now (R8): #38488f beats the UA
        // default link blue, so the glyphs carry #38488f.
        let link_leaf = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("More information")))
            })
            .expect("link text leaf");
        assert!(
            link_leaf.glyphs.iter().any(|g| g.color == [56, 72, 143]),
            "link glyphs carry the author a:link color #38488f"
        );

        assert!(tree.content_height() > 200.0);
        assert!(tree.content_height() < 768.0);
    }

    #[test]
    fn display_none_skips_subtree() {
        let dom = parse_html(
            r#"<div class="hide"><p>one</p></div><div><p>two</p></div>"#,
        );
        let sheet = Stylesheet::parse(".hide { display: none; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let texts: Vec<String> = tree
            .boxes
            .iter()
            .filter_map(|b| b.text.as_ref())
            .flat_map(|t| t.spans.iter().map(|s| s.text.clone()))
            .collect();
        assert!(!texts.iter().any(|t| t.contains("one")));
        assert!(texts.iter().any(|t| t.contains("two")));
    }

    fn id_of_class(tree: &LayoutTree, dom: &Dom, class: &str) -> DomId {
        tree.boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some(class)))
            .and_then(|b| b.dom)
            .unwrap_or_else(|| panic!("no box for .{class}"))
    }

    /// `scrollable_size` (Stage 10 Rec 2): a container's maximum
    /// scroll offset is its content past its own box, per axis —
    /// and a scroll offset already applied to the boxes does not
    /// change it (the offset is added back).
    #[test]
    fn scrollable_size_measures_content_past_the_box() {
        let dom = parse_html(
            r#"<div class="outer"><div class="tall"></div><div class="wide"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".outer { height: 100px; overflow: auto; } \
             .tall { height: 250px; } .wide { width: 900px; height: 10px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        // Tall child bottoms at 250, wide child at 260 x 900, in a
        // 100px box (body margins cancel in the relative math; the
        // wide child's 900 - 784 outer width = 116).
        assert_eq!(tree.scrollable_size(outer), Some((116.0, 160.0)));
        // Scrolling the container shifts its children's boxes, but
        // the maximum is a property of the content, not the offset.
        tree.set_scroll_offset(outer, 0.0, 5.0);
        assert_eq!(tree.scrollable_size(outer), Some((116.0, 160.0)));
        tree.set_scroll_offset(outer, 40.0, 0.0);
        assert_eq!(tree.scrollable_size(outer), Some((116.0, 160.0)));
    }

    /// `scrollable_size` (Stage 10 Rec 2): a nested scroll
    /// container contributes its own border box to its parent's
    /// content — not its inner content, which scrolls inside it.
    #[test]
    fn scrollable_size_counts_a_nested_container_by_its_own_box() {
        let dom = parse_html(
            r#"<div class="outer"><div class="nested"><div class="deep"></div></div><div class="sib"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".outer { height: 100px; overflow: auto; } \
             .nested { height: 60px; overflow: auto; } .deep { height: 300px; } \
             .sib { height: 50px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        let nested = id_of_class(&tree, &dom, "nested");
        // The nested container's 300px of inner content scrolls
        // inside it (max 240); the outer sees only the nested box
        // (60) plus the sibling (50) — 110 in a 100px box.
        assert_eq!(tree.scrollable_size(nested), Some((0.0, 240.0)));
        assert_eq!(tree.scrollable_size(outer), Some((0.0, 10.0)));
    }

    /// `set_scroll_offset` (Stage 10 follow-up): an offset past
    /// the container's maximum clamps to it — offsets re-applied
    /// after a resize must not survive past the new maximum — and
    /// a negative offset clamps to zero.
    #[test]
    fn set_scroll_offset_clamps_to_the_containers_maximum() {
        let dom = parse_html(
            r#"<div class="outer"><div class="tall"></div><div class="wide"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".outer { height: 100px; overflow: auto; } \
             .tall { height: 250px; } .wide { width: 900px; height: 10px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        assert_eq!(tree.scrollable_size(outer), Some((116.0, 160.0)));
        tree.set_scroll_offset(outer, 500.0, 9999.0);
        assert_eq!(tree.scroll_offset(outer), Some((116.0, 160.0)));
        tree.set_scroll_offset(outer, -5.0, -5.0);
        assert_eq!(tree.scroll_offset(outer), Some((0.0, 0.0)));
        // Within the maximum, the offset stores as given.
        tree.set_scroll_offset(outer, 40.0, 30.0);
        assert_eq!(tree.scroll_offset(outer), Some((40.0, 30.0)));
    }

    /// `scrollable_size` (Stage 10 Rec 2): content that fits
    /// scrolls nowhere, and a node with no box has no maximum.
    #[test]
    fn scrollable_size_is_zero_when_the_content_fits() {
        let dom = parse_html(
            r#"<div class="hide"></div><div class="outer"><div class="small"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".hide { display: none; } \
             .outer { height: 100px; overflow: auto; } .small { height: 50px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        assert_eq!(tree.scrollable_size(outer), Some((0.0, 0.0)));
        let hidden = find_element(&dom, dom.root().unwrap(), "div").unwrap();
        assert_eq!(tree.scrollable_size(hidden), None);
    }

    // ---- M6 Phase 3 (Rec 1/2/3): fixed / absolute / sticky ----

    /// Rec 1: a `fixed` box resolves its insets against the
    /// viewport — not against the nearest positioned ancestor, and
    /// not against any scroll container. `top`/`left` win over
    /// `bottom`/`right`.
    #[test]
    fn fixed_box_resolves_insets_against_the_viewport() {
        let dom = parse_html(
            r#"<div class="spacer"></div><div class="fixed">F</div><div class="fixed2">G</div>"#,
        );
        let sheet = Stylesheet::parse(
            ".spacer { height: 2000px; } \
             .fixed { position: fixed; top: 10px; left: 20px; width: 100px; height: 50px; } \
             .fixed2 { position: fixed; right: 30px; bottom: 40px; width: 100px; height: 50px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let fixed = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("fixed")))
            .unwrap();
        assert_eq!((fixed.x, fixed.y), (20.0, 10.0));
        assert_eq!((fixed.width, fixed.height), (100.0, 50.0));
        assert!(fixed.is_viewport_fixed);
        let fixed2 = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("fixed2")))
            .unwrap();
        assert_eq!((fixed2.x, fixed2.y), (800.0 - 30.0 - 100.0, 600.0 - 40.0 - 50.0));
        assert!(fixed2.is_viewport_fixed);
    }

    /// Rec 1: a `fixed` box inside a scroll container does not
    /// move with it, and does not inflate the container's maximum
    /// scroll — the fixed box is viewport content, not container
    /// content.
    #[test]
    fn fixed_box_ignores_scroll_container_offset_and_size() {
        let dom = parse_html(
            r#"<div class="outer"><div class="tall"></div><div class="fixed">F</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".outer { height: 100px; overflow: auto; } \
             .tall { height: 250px; } \
             .fixed { position: fixed; top: 0px; left: 0px; width: 50px; height: 500px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        // Only the 250px tall child counts: 250 - 100 = 150. The
        // 500px fixed box must not add a single px.
        assert_eq!(tree.scrollable_size(outer), Some((0.0, 150.0)));
        tree.set_scroll_offset(outer, 0.0, 100.0);
        let fixed = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("fixed")))
            .unwrap();
        assert_eq!((fixed.x, fixed.y), (0.0, 0.0));
        assert_eq!(tree.scrollable_size(outer), Some((0.0, 150.0)));
    }

    /// Rec 1: a transformed ancestor is the containing block for
    /// `fixed` descendants — they stop being viewport-relative
    /// and lay out as absolute against it.
    #[test]
    fn fixed_box_under_transformed_ancestor_behaves_absolute() {
        let dom = parse_html(r#"<div class="outer"><div class="fixed">F</div></div>"#);
        let sheet = Stylesheet::parse(
            ".outer { transform: translateX(10px); } \
             .fixed { position: fixed; top: 10px; left: 20px; width: 100px; height: 50px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let fixed = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("fixed")))
            .unwrap();
        assert!(!fixed.is_viewport_fixed, "transformed ancestor re-parents fixed");
        // Absolute path: the outer box sits at the body margin
        // (8, 8); the fixed child resolves against its padding
        // box, so (8 + 20, 8 + 10). The 10px transform rides on
        // the paint matrix, not on x/y.
        assert_eq!((fixed.x, fixed.y), (28.0, 18.0));
    }

    /// Rec 1: hit-testing a viewport-fixed box uses viewport
    /// coordinates — the shell passes page coordinates, so the
    /// viewport scroll is removed for fixed boxes only.
    #[test]
    fn fixed_box_hit_tests_in_viewport_coordinates() {
        let dom = parse_html(
            r#"<div class="fixed">F</div><div class="content"></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".fixed { position: fixed; top: 10px; left: 20px; width: 100px; height: 50px; } \
             .content { height: 2000px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let fixed = id_of_class(&tree, &dom, "fixed");
        let content = id_of_class(&tree, &dom, "content");
        // Page point (25, 115) is inside the content div in page
        // space — but in viewport space (25, 15) it is inside the
        // fixed box, which paints on top.
        tree.set_viewport_scroll(0.0, 100.0);
        assert_eq!(tree.hit_test(25.0, 115.0), Some(fixed));
        // Page point (25, 65) is viewport (25, -35): above the
        // fixed box, so the content underneath wins.
        assert_eq!(tree.hit_test(25.0, 65.0), Some(content));
    }

    /// Rec 2: a positioned `display: inline` element gets its own
    /// box instead of vanishing into the parent's text leaf — and
    /// the surrounding inline text still lays out.
    #[test]
    fn positioned_inline_element_gets_a_box() {
        let dom = parse_html(r#"<p class="wrap">before <span class="abs">X</span> after</p>"#);
        let sheet = Stylesheet::parse(
            ".wrap { position: relative; } \
             .abs { position: absolute; left: 30px; top: 40px; width: 50px; height: 20px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let abs = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("abs")))
            .expect("positioned span has a box");
        let wrap = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("wrap")))
            .unwrap();
        assert_eq!((abs.x, abs.y), (wrap.x + 30.0, wrap.y + 40.0));
        assert_eq!((abs.width, abs.height), (50.0, 20.0));
        let texts: Vec<String> = tree
            .boxes
            .iter()
            .filter_map(|b| b.text.as_ref())
            .flat_map(|t| t.spans.iter().map(|s| s.text.clone()))
            .collect();
        assert!(texts.iter().any(|t| t.contains("before")));
        assert!(texts.iter().any(|t| t.contains("after")));
        assert!(texts.iter().any(|t| t.contains('X')));
    }

    /// Rec 2: absolute geometry against the containing block —
    /// percentage insets resolve against the CB, and
    /// over-constrained insets resolve with `left` winning.
    #[test]
    fn absolute_geometry_against_containing_block() {
        let dom = parse_html(
            r#"<div class="cb"><div class="a"></div><div class="b"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".cb { position: relative; width: 300px; height: 200px; } \
             .a { position: absolute; left: 10%; top: 20px; width: 50%; height: 30px; } \
             .b { position: absolute; left: 10px; right: 20px; top: 0px; width: 100px; height: 10px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let cb = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("cb")))
            .unwrap();
        let a = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("a")))
            .unwrap();
        assert_eq!((a.x, a.y), (cb.x + 30.0, cb.y + 20.0));
        assert_eq!((a.width, a.height), (150.0, 30.0));
        let b = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("b")))
            .unwrap();
        assert_eq!((b.x, b.y), (cb.x + 10.0, cb.y));
        assert_eq!(b.width, 100.0);
    }

    /// Rec 2: the `inset` shorthand (parsed in Phase 1) positions
    /// an absolute box end to end — `top right bottom left`.
    #[test]
    fn inset_shorthand_positions_absolute_box() {
        let dom = parse_html(r#"<div class="cb"><div class="abs"></div></div>"#);
        let sheet = Stylesheet::parse(
            ".cb { position: relative; width: 300px; height: 200px; } \
             .abs { position: absolute; inset: 10px 20px 30px 40px; width: 50px; height: 60px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let cb = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("cb")))
            .unwrap();
        let abs = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("abs")))
            .unwrap();
        assert_eq!((abs.x, abs.y), (cb.x + 40.0, cb.y + 10.0));
        assert_eq!((abs.width, abs.height), (50.0, 60.0));
    }

    /// Rec 3: a sticky navbar lays out in flow, sticks at `top: 0`
    /// while the page scrolls, and releases when its bottom would
    /// pass its containing block's bottom edge.
    #[test]
    fn sticky_navbar_sticks_at_top_and_releases_at_cb_end() {
        let dom = parse_html(
            r#"<div class="page"><div class="nav">Nav</div><div class="filler"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".page { height: 2000px; } .nav { position: sticky; top: 0px; height: 50px; } \
             .filler { height: 1900px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let nav = |tree: &LayoutTree| {
            tree.boxes
                .iter()
                .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("nav")))
                .unwrap()
                .clone()
        };
        // In flow at the top of the page (body margin 8).
        assert_eq!(nav(&tree).y, 8.0);
        // Scrolled 100: stuck at the viewport top. Layout `y`
        // is a page coordinate — the viewport scroll is subtracted
        // at paint — so the stuck box sits at y == 100, painted
        // at 100 - 100 == 0. The scroll moved the sticky box, so
        // the worker must resubmit the re-collected list.
        assert!(tree.set_viewport_scroll(0.0, 100.0));
        assert_eq!(nav(&tree).y, 100.0);
        // Scrolled to the containing block's end: the nav's bottom
        // aligns with the page bottom in page coordinates
        // (2008); painted at 2008 - 1960 == 48.
        tree.set_viewport_scroll(0.0, 1960.0);
        let n = nav(&tree);
        assert_eq!(n.y + n.height, 2008.0);
        // Back to the top: unstuck, back in flow.
        tree.set_viewport_scroll(0.0, 0.0);
        assert_eq!(nav(&tree).y, 8.0);
    }

    /// Rec 3 follow-up: a viewport scroll with no sticky content
    /// moves nothing — `set_viewport_scroll` reports `false`, so
    /// the worker skips the resubmit and the event stream stays
    /// clean (a spurious submit here poisoned the wheel/resize
    /// worker tests' event order).
    #[test]
    fn viewport_scroll_without_sticky_reports_no_movement() {
        let dom = parse_html(
            r#"<div class="page"><div class="filler"></div></div>"#,
        );
        let sheet = Stylesheet::parse(".page { height: 2000px; } .filler { height: 1900px; }");
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        assert!(!tree.set_viewport_scroll(0.0, 100.0));
        assert!(!tree.set_viewport_scroll(0.0, 200.0));
        assert!(!tree.set_viewport_scroll(0.0, 0.0));
    }

    /// Rec 3: a sticky header inside a scroll container sticks at
    /// the container's top (not the viewport's) and releases at
    /// the container's bottom edge.
    #[test]
    fn sticky_header_sticks_inside_scroll_container() {
        let dom = parse_html(
            r#"<div class="outer"><div class="head">H</div><div class="tall"></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".outer { height: 100px; overflow: auto; } \
             .head { position: sticky; top: 0px; height: 20px; } .tall { height: 300px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let outer = id_of_class(&tree, &dom, "outer");
        let head = |tree: &LayoutTree| {
            tree.boxes
                .iter()
                .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("head")))
                .unwrap()
                .clone()
        };
        let outer_box = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(outer))
            .unwrap()
            .clone();
        assert_eq!(head(&tree).y, outer_box.y);
        // Scrolled 50: stuck at the container's top.
        tree.set_scroll_offset(outer, 0.0, 50.0);
        assert_eq!(head(&tree).y, outer_box.y);
        // Scrolled to the container's end (max 220): released —
        // the header's content position caps at the containing
        // block's bottom minus its height (100 - 20 = 80), which
        // is above the visible window [220, 320], so it has
        // scrolled off the top: painted at 8 + (80 - 220) = -132.
        tree.set_scroll_offset(outer, 0.0, 220.0);
        let h = head(&tree);
        assert_eq!(h.y, outer_box.y + (80.0 - 220.0));
    }

    #[test]
    fn flex_row_lays_children_side_by_side() {
        let dom = parse_html(
            r#"<div class="row"><div class="a">A</div><div class="b">B</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".row { display: flex; } .a { width: 100px; } .b { width: 100px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let a = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("a")))
            .unwrap();
        let b = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("b")))
            .unwrap();
        assert_eq!(a.y, b.y);
        assert_eq!(b.x, a.x + 100.0);
        assert_eq!(a.width, 100.0);
    }

    // ---- two-phase (parallel) style resolution (Stage 4 #5) ----

    /// Compile-time proof that the parallel pass may share `&Dom` /
    /// `&Stylesheet` across rayon threads. `Dom` is a slotmap arena
    /// of plain owned data (`String` / `Vec` / `NodeId`), so it is
    /// `Sync`; if a future change ever makes it non-`Sync` this test
    /// fails to compile and the parallel design must stop there —
    /// never work around it with unsafe code.
    fn assert_sync<T: Sync>() {}

    #[test]
    fn shared_style_inputs_are_sync() {
        assert_sync::<Dom>();
        assert_sync::<Stylesheet>();
        assert_sync::<ComputedStyle>();
    }

    /// The `<style>` text of a document, parsed (same extraction the
    /// layout tests use).
    fn stylesheet_from(dom: &Dom) -> Stylesheet {
        let style = find_element(dom, dom.root().unwrap(), "style").unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        Stylesheet::parse(&css)
    }

    fn assert_same_styles(
        a: &HashMap<DomId, Arc<ComputedStyle>>,
        b: &HashMap<DomId, Arc<ComputedStyle>>,
    ) {
        assert_eq!(
            a.len(),
            b.len(),
            "parallel and serial maps cover different numbers of elements"
        );
        for (id, style) in a {
            let other = b
                .get(id)
                .unwrap_or_else(|| panic!("serial map has no style for {id:?}"));
            assert_eq!(**style, **other, "computed style differs for {id:?}");
        }
    }

    /// First element descendant of `id` (or `id` itself) with this class.
    fn find_by_class(dom: &Dom, id: DomId, class: &str) -> Option<DomId> {
        if dom.attr(id, "class") == Some(class) {
            return Some(id);
        }
        for &child in dom.children(id) {
            if let Some(found) = find_by_class(dom, child, class) {
                return Some(found);
            }
        }
        None
    }

    /// All element descendants of `id` with this tag, in document order.
    fn find_all_by_tag(dom: &Dom, id: DomId, tag: &str) -> Vec<DomId> {
        let mut out = Vec::new();
        fn walk(dom: &Dom, id: DomId, tag: &str, out: &mut Vec<DomId>) {
            for &child in dom.children(id) {
                if dom.tag(child) == Some(tag) {
                    out.push(child);
                }
                walk(dom, child, tag, out);
            }
        }
        walk(dom, id, tag, &mut out);
        out
    }

    fn resolve_both(
        dom: &Dom,
        sheet: &Stylesheet,
    ) -> (
        HashMap<DomId, Arc<ComputedStyle>>,
        HashMap<DomId, Arc<ComputedStyle>>,
    ) {
        let ua = Stylesheet::ua();
        let state = ElementState::default();
        let parallel =
            resolve_document_styles(dom, &ua, sheet, true, (1024.0, 768.0), &state, Theme::Light).0;
        let serial =
            resolve_document_styles(dom, &ua, sheet, false, (1024.0, 768.0), &state, Theme::Light).0;
        (parallel, serial)
    }

    #[test]
    fn parallel_and_serial_styles_match_example_com() {
        let dom = parse_html(EXAMPLE_COM);
        let sheet = stylesheet_from(&dom);

        // Levels are parent-before-children: every entry's parent
        // appears in an earlier level.
        let html = find_element(&dom, dom.root().unwrap(), "html").unwrap();
        let levels = collect_style_levels(&dom, &[(html, None)]);
        let mut seen: Vec<DomId> = Vec::new();
        for level in &levels {
            for &(id, parent) in level {
                if let Some(p) = parent {
                    assert!(seen.contains(&p), "parent of {id:?} not in an earlier level");
                }
            }
            seen.extend(level.iter().map(|&(id, _)| id));
        }

        let (parallel, serial) = resolve_both(&dom, &sheet);
        assert_same_styles(&parallel, &serial);
        // html head title meta style body div h1 p p a
        assert_eq!(parallel.len(), 11);
        // Spot-checks through the parallel map: the div keeps its
        // stylesheet width, the h1 its UA 32px font.
        let div = find_element(&dom, dom.root().unwrap(), "div").unwrap();
        assert_eq!(parallel[&div].width, Some(Dim::Px(600.0)));
        let h1 = find_element(&dom, dom.root().unwrap(), "h1").unwrap();
        assert_eq!(parallel[&h1].font_size, 32.0);
    }

    #[test]
    fn parallel_and_serial_styles_match_wide_document() {
        // 200 sibling <p> elements at one level under a wrapper, each
        // with a <span> child: wide enough that rayon's `par_iter`
        // actually splits the level across threads.
        let mut html = String::from("<html><body><div class=\"wrap\">");
        for i in 0..200 {
            html.push_str(&format!(
                "<p class=\"item\">Item {i} <span>inner {i}</span></p>"
            ));
        }
        html.push_str("</div></body></html>");
        let dom = parse_html(&html);
        let sheet = Stylesheet::parse(
            ".wrap { font-size: 20px; color: #112233; } \
             .item { font-size: 150%; margin: 1em; } \
             span { color: #ff0000; }",
        );
        let (parallel, serial) = resolve_both(&dom, &sheet);
        assert_same_styles(&parallel, &serial);
        // html + head + body + div.wrap + 200 p + 200 span: Pluto's
        // parse_document synthesises html/head/body around the input,
        // so the head is a real (if empty) element in the style map.
        assert_eq!(parallel.len(), 404);

        // Inheritance through the parallel map: a `.item` inherits
        // the wrap's 20px font, so 150% is 30px and `1em` is 30px.
        let root = dom.root().unwrap();
        let wrap = find_by_class(&dom, root, "wrap").unwrap();
        assert_eq!(parallel[&wrap].font_size, 20.0);
        assert_eq!(parallel[&wrap].color, [0x11, 0x22, 0x33]);
        let item = find_by_class(&dom, root, "item").unwrap();
        assert_eq!(parallel[&item].font_size, 30.0);
        assert_eq!(parallel[&item].color, [0x11, 0x22, 0x33]);
        assert_eq!(parallel[&item].margin, [Spacing::Px(30.0); 4]);
        let span = find_element(&dom, item, "span").unwrap();
        assert_eq!(parallel[&span].font_size, 30.0);
        assert_eq!(parallel[&span].color, [255, 0, 0]);
    }

    #[test]
    fn parallel_and_serial_styles_match_fragment() {
        // No <html> in the input: Pluto's parse_document
        // synthesises html/head/body around the fragment, and the
        // style pass collects those too.
        let dom = parse_html(
            r#"<div class="hide"><p>one</p></div><div><p>two</p></div>"#,
        );
        let sheet = Stylesheet::parse(".hide { display: none; }");
        let (parallel, serial) = resolve_both(&dom, &sheet);
        assert_same_styles(&parallel, &serial);
        // Both divs and both paragraphs are collected even though
        // the first subtree is `display: none` (computed, unused) —
        // plus html/head/body, which Pluto's parse_document
        // synthesises around the fragment (7 total, not 4).
        assert_eq!(parallel.len(), 7);
        let hide = find_by_class(&dom, dom.root().unwrap(), "hide").unwrap();
        assert_eq!(parallel[&hide].display, DisplayKind::None);
    }

    #[test]
    fn subtree_restyle_matches_whole_document_styles() {
        // Stage 12 Rec 6: restyling one subtree — rooted at
        // <html> itself, at a middle <div>, and at a deep <p> —
        // must produce exactly the whole-document styles for every
        // element. The document exercises the seeding paths:
        // descendant selectors (`.wrap p`), `nth-child`, inherited
        // color, `em`/`rem` font sizes, and a class rule that only
        // the subtree's parent carries.
        let dom = parse_html(
            r#"<html><body><div class="wrap"><p class="a">one</p><p class="b">two</p><p class="c">three <span>inner</span></p></div><div class="after"><p>below</p></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(
            "html { font-size: 20px; } \
             .wrap { color: #112233; font-size: 1rem; } \
             .wrap p { margin: 1em; } \
             .wrap p:nth-child(2) { color: #ff0000; } \
             .b { font-size: 150%; } \
             span { color: #00ff00; }",
        );
        let ua = Stylesheet::ua();
        let state = ElementState::default();
        let viewport = (800.0, 600.0);
        let (full_styles, _) =
            resolve_document_styles(&dom, &ua, &sheet, false, viewport, &state, Theme::Light);
        let tree = LayoutTree::build(&dom, &sheet, viewport.0, viewport.1, Theme::Light)
            .expect("layout builds");
        let root = dom.root().unwrap();
        let html = find_element(&dom, root, "html").unwrap();
        let wrap = find_by_class(&dom, root, "wrap").unwrap();
        let p_c = find_by_class(&dom, root, "c").unwrap();
        for (name, sub_root) in [("html", html), ("wrap", wrap), ("p.c", p_c)] {
            let (sub_styles, _) = resolve_subtree_styles(
                &dom,
                &ua,
                &sheet,
                false,
                viewport,
                &state,
                sub_root,
                &tree.node_styles,
                Theme::Light,
            );
            assert_eq!(
                sub_styles.len(),
                full_styles.len(),
                "{name}: subtree map covers every element"
            );
            for (id, style) in &full_styles {
                let got = sub_styles
                    .get(id)
                    .unwrap_or_else(|| panic!("{name}: no style for {id:?}"));
                assert_eq!(got.as_ref(), style.as_ref(), "{name}: style differs for {id:?}");
            }
        }
        // Spot-check the values the seeding must get right: the
        // wrap's 1rem is the html 20px; `.b` at 150% of it is 30px;
        // the second <p> is red by nth-child; the span is green.
        let b = find_by_class(&dom, root, "b").unwrap();
        assert_eq!(full_styles[&wrap].font_size, 20.0);
        assert_eq!(full_styles[&b].font_size, 30.0);
        assert_eq!(full_styles[&b].color, [255, 0, 0]);
        let span = find_element(&dom, p_c, "span").unwrap();
        assert_eq!(full_styles[&span].color, [0, 255, 0]);
    }

    // ---- R8 cascade + selectors ----

    #[test]
    fn cascade_specificity_and_important() {
        // #x (1,0,0) beats .note (0,1,0) beats p (0,0,1); but an
        // `!important` author declaration beats all normal ones.
        let dom = parse_html(r#"<p id="x" class="note">hi</p>"#);
        let styles = styles_for(
            r#"<p id="x" class="note">hi</p>"#,
            "p { color: #ff0000; } .note { color: #00ff00; } \
             #x { color: #0000ff; } p { color: #ffff00 !important; }",
            (800.0, 600.0),
        );
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        assert_eq!(styles[&p].color, [255, 255, 0]);
    }

    #[test]
    fn cascade_specificity_without_important() {
        // Without `!important`, #x wins over .note over p.
        let dom = parse_html(r#"<p id="x" class="note">hi</p>"#);
        let styles = styles_for(
            r#"<p id="x" class="note">hi</p>"#,
            "p { color: #ff0000; } .note { color: #00ff00; } #x { color: #0000ff; }",
            (800.0, 600.0),
        );
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        assert_eq!(styles[&p].color, [0, 0, 255]);
    }

    #[test]
    fn cascade_layers_unlayered_beats_layered() {
        // Later layer beats earlier layer, but unlayered beats all
        // layered rules for normal declarations.
        let styles = styles_for(
            "<p>hi</p>",
            "@layer a, b; @layer a { p { color: #ff0000; } } \
             @layer b { p { color: #00ff00; } } p { color: #0000ff; }",
            (800.0, 600.0),
        );
        let dom = parse_html("<p>hi</p>");
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        assert_eq!(styles[&p].color, [0, 0, 255]);
    }

    #[test]
    fn inline_style_beats_rules() {
        let dom = parse_html(r#"<p style="color: #ff0000;">hi</p>"#);
        let styles = styles_for(
            r#"<p style="color: #ff0000;">hi</p>"#,
            "p { color: #00ff00; }",
            (800.0, 600.0),
        );
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        assert_eq!(styles[&p].color, [255, 0, 0]);
    }

    #[test]
    fn combinators_descendant_child_sibling() {
        let html = r#"<div class="outer"><p class="a">A</p><section><p class="b">B</p></section><p class="c">C</p></div>
            <div><p class="s1">1</p><p class="s2">2</p><span class="s3">3</span></div>"#;
        let dom = parse_html(html);
        let styles = styles_for(
            html,
            ".outer p { color: #111111; } .outer > p { font-size: 20px; } \
             p + p { color: #ff0000; } p ~ span { color: #00ff00; }",
            (800.0, 600.0),
        );
        let root = dom.root().unwrap();
        // .a and .c are direct children of .outer; .b is nested.
        let a = find_by_class(&dom, root, "a").unwrap();
        let b = find_by_class(&dom, root, "b").unwrap();
        let c = find_by_class(&dom, root, "c").unwrap();
        assert_eq!(styles[&a].font_size, 20.0);
        assert_eq!(styles[&b].font_size, 16.0);
        assert_eq!(styles[&c].font_size, 20.0);
        assert_eq!(styles[&a].color, [0x11, 0x11, 0x11]);
        assert_eq!(styles[&b].color, [0x11, 0x11, 0x11]);
        // .s2 immediately follows a <p>; .s3 follows a <p> among siblings.
        let s1 = find_by_class(&dom, root, "s1").unwrap();
        let s2 = find_by_class(&dom, root, "s2").unwrap();
        let s3 = find_by_class(&dom, root, "s3").unwrap();
        assert_eq!(styles[&s1].color, [0, 0, 0]);
        assert_eq!(styles[&s2].color, [255, 0, 0]);
        assert_eq!(styles[&s3].color, [0, 255, 0]);
    }

    #[test]
    fn attribute_selectors_and_nth_child() {
        let html = r#"<a href="https://example.com" class="x">link</a>
            <ul><li>1</li><li>2</li><li>3</li></ul>"#;
        let dom = parse_html(html);
        let styles = styles_for(
            html,
            r#"a[href^="https"] { color: #ff0000; } a[href$=".com"] { font-size: 20px; }
               li:nth-child(2) { color: #00ff00; } li:first-child { font-weight: bold; }"#,
            (800.0, 600.0),
        );
        let a = find_element(&dom, dom.root().unwrap(), "a").unwrap();
        assert_eq!(styles[&a].color, [255, 0, 0]);
        assert_eq!(styles[&a].font_size, 20.0);
        let ul = find_element(&dom, dom.root().unwrap(), "ul").unwrap();
        let lis = dom.children(ul);
        assert_eq!(styles[&lis[1]].color, [0, 255, 0]);
        assert_eq!(styles[&lis[0]].color, [0, 0, 0]);
        assert!(styles[&lis[0]].bold);
        assert!(!styles[&lis[1]].bold);
    }

    #[test]
    fn not_selector_and_visited_matches_visited_only() {
        let html = r#"<p class="a">A</p><p class="b">B</p><a href="https://example.com">L</a><a href="https://example.org">M</a>"#;
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(
            "p:not(.b) { color: #ff0000; } a:visited { color: #00ff00; }",
        );
        let ua = Stylesheet::ua();
        let root = dom.root().unwrap();
        let a = find_by_class(&dom, root, "a").unwrap();
        let b = find_by_class(&dom, root, "b").unwrap();
        let links: Vec<DomId> = find_all_by_tag(&dom, root, "a");
        assert_eq!(links.len(), 2);
        let (visited_link, unvisited_link) = (links[0], links[1]);
        // No visited state: `:visited` matches nothing — both links
        // keep the UA default blue, not the green rule.
        let plain =
            resolve_document_styles(&dom, &ua, &sheet, false, (800.0, 600.0), &ElementState::default(), Theme::Light).0;
        assert_eq!(plain[&a].color, [255, 0, 0]);
        assert_eq!(plain[&b].color, [0, 0, 0]);
        assert_eq!(plain[&visited_link].color, [0, 0, 238]);
        assert_eq!(plain[&unvisited_link].color, [0, 0, 238]);
        // One link visited: the `a:visited` rule applies to it only.
        let visited =
            resolve_document_styles(&dom, &ua, &sheet, false, (800.0, 600.0), &ElementState::default().with_visited(visited_link), Theme::Light).0;
        assert_eq!(visited[&visited_link].color, [0, 255, 0]);
        assert_eq!(visited[&unvisited_link].color, [0, 0, 238]);
    }

    #[test]
    fn ua_visited_rule_colors_visited_links_purple() {
        let html = r#"<a href="https://example.com">L</a><a href="https://example.org">M</a>"#;
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("");
        let ua = Stylesheet::ua();
        let root = dom.root().unwrap();
        let links: Vec<DomId> = find_all_by_tag(&dom, root, "a");
        assert_eq!(links.len(), 2);
        let (visited_link, unvisited_link) = (links[0], links[1]);
        let styles =
            resolve_document_styles(&dom, &ua, &sheet, false, (800.0, 600.0), &ElementState::default().with_visited(visited_link), Theme::Light).0;
        // UA sheet: `a { color: #0000EE; } a:visited { color: #551A8B; }`.
        assert_eq!(styles[&visited_link].color, [0x55, 0x1a, 0x8b]);
        assert_eq!(styles[&unvisited_link].color, [0, 0, 238]);
    }

    #[test]
    fn link_and_visited_are_exclusive_any_link_is_the_union() {
        let html = r#"<a href="https://example.com">L</a><a href="https://example.org">M</a>"#;
        let dom = parse_html(html);
        let ua = Stylesheet::ua();
        let root = dom.root().unwrap();
        let links: Vec<DomId> = find_all_by_tag(&dom, root, "a");
        assert_eq!(links.len(), 2);
        let (visited_link, unvisited_link) = (links[0], links[1]);
        // `:link` red vs `:visited` green, same specificity: a
        // visited link matches only `:visited`, an unvisited link
        // only `:link` — the two are exclusive per spec.
        let sheet = Stylesheet::parse(
            "a:link { color: #ff0000; } a:visited { color: #00ff00; }",
        );
        let styles = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (800.0, 600.0),
            &ElementState::default().with_visited(visited_link),
            Theme::Light,
        )
        .0;
        assert_eq!(styles[&visited_link].color, [0, 255, 0]);
        assert_eq!(styles[&unvisited_link].color, [255, 0, 0]);
        // `:any-link` is the union: both links match it, visited
        // or not.
        let sheet = Stylesheet::parse("a:any-link { color: #0000ff; }");
        let styles = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (800.0, 600.0),
            &ElementState::default().with_visited(visited_link),
            Theme::Light,
        )
        .0;
        assert_eq!(styles[&visited_link].color, [0, 0, 255]);
        assert_eq!(styles[&unvisited_link].color, [0, 0, 255]);
    }

    #[test]
    fn hover_state_applies_dynamic_rule() {
        let html = r#"<a href="https://example.com">L</a>"#;
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("a:hover { color: #ff0000; }");
        let ua = Stylesheet::ua();
        let link = find_element(&dom, dom.root().unwrap(), "a").unwrap();
        let plain = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (800.0, 600.0),
            &ElementState::default(),
            Theme::Light,
        )
        .0;
        assert_eq!(plain[&link].color, [0, 0, 238]);
        let hovered = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (800.0, 600.0),
            &ElementState::default().with_hover(link),
            Theme::Light,
        )
        .0;
        assert_eq!(hovered[&link].color, [255, 0, 0]);
    }

    #[test]
    fn media_query_gates_rules_by_viewport() {
        let html = "<p>hi</p>";
        let css = "@media (max-width: 500px) { p { color: #ff0000; } }";
        let narrow = styles_for(html, css, (400.0, 600.0));
        let wide = styles_for(html, css, (800.0, 600.0));
        let dom = parse_html(html);
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        assert_eq!(narrow[&p].color, [255, 0, 0]);
        assert_eq!(wide[&p].color, [0, 0, 0]);
    }

    #[test]
    fn ua_stylesheet_replaces_hardcoded_defaults() {
        // No author sheet at all: h1 is 2em bold with 0.67em margins,
        // body has 8px margins, links are UA blue — all from UA_CSS.
        let dom = parse_html("<html><body><h1>T</h1><a href=\"/x\">L</a></body></html>");
        let styles = styles_for(
            "<html><body><h1>T</h1><a href=\"/x\">L</a></body></html>",
            "",
            (800.0, 600.0),
        );
        let h1 = find_element(&dom, dom.root().unwrap(), "h1").unwrap();
        assert_eq!(styles[&h1].font_size, 32.0);
        assert!(styles[&h1].bold);
        assert_eq!(styles[&h1].margin[0], Spacing::Px(0.67 * 32.0));
        let body = find_element(&dom, dom.root().unwrap(), "body").unwrap();
        assert_eq!(styles[&body].margin, [Spacing::Px(8.0); 4]);
        let a = find_element(&dom, dom.root().unwrap(), "a").unwrap();
        assert_eq!(styles[&a].color, [0, 0, 238]);
        // A tag UA_CSS does not mention falls back to inline display.
        let dom2 = parse_html("<my-widget>hi</my-widget>");
        let styles2 = styles_for("<my-widget>hi</my-widget>", "", (800.0, 600.0));
        let w = find_element(&dom2, dom2.root().unwrap(), "my-widget").unwrap();
        assert_eq!(styles2[&w].display, DisplayKind::Inline);
    }

    // ---- R10/R13 inline layout + parked gaps ----

    #[test]
    fn br_is_a_forced_break_not_a_leaf() {
        let dom = parse_html("<p>one<br>two</p>");
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        // No box exists for <br> itself.
        assert!(!tree.boxes.iter().any(|b| {
            b.dom.is_some_and(|id| dom.tag(id) == Some("br"))
        }));
        let leaf = tree
            .boxes
            .iter()
            .find(|b| {
                b.text.as_ref().is_some_and(|t| {
                    t.spans.iter().any(|s| s.text.contains("one"))
                })
            })
            .expect("paragraph text leaf");
        assert!(leaf
            .text
            .as_ref()
            .unwrap()
            .spans
            .iter()
            .any(|s| s.text.contains("two")));
        // Two lines at the computed line-height (16 * 1.2).
        assert!((leaf.height - 2.0 * 16.0 * 1.2).abs() < 0.5, "leaf height {}", leaf.height);
        assert!(!leaf.glyphs.is_empty());
    }

    #[test]
    fn text_align_center_places_glyphs_centered() {
        let dom = parse_html(
            r#"<div style="text-align: center; width: 400px;">hi</div>"#,
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.width, 400.0);
        let leaf = tree
            .boxes
            .iter()
            .find(|b| b.text.is_some())
            .expect("text leaf");
        assert!(!leaf.glyphs.is_empty());
        let min_x = leaf.glyphs.iter().map(|g| g.x).fold(f32::INFINITY, f32::min);
        let max_x = leaf.glyphs.iter().map(|g| g.x).fold(f32::NEG_INFINITY, f32::max);
        // Glyph run centred on the div's centre (div x=8 + 200).
        let centre = (min_x + max_x) / 2.0;
        assert!((centre - 208.0).abs() < 15.0, "glyph centre {centre}");
    }

    #[test]
    fn pre_preserves_whitespace() {
        let dom = parse_html("<pre>  a   b  </pre>");
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let leaf = tree
            .boxes
            .iter()
            .find(|b| b.text.is_some())
            .expect("pre text leaf");
        assert_eq!(
            leaf.text.as_ref().unwrap().spans[0].text,
            "  a   b  ",
            "pre whitespace preserved verbatim"
        );
        assert_eq!(leaf.style.white_space, WhiteSpace::Pre);
    }

    #[test]
    fn max_width_clamps_and_vh_resolves() {
        let dom = parse_html(r#"<div class="w">x</div><div class="pad">y</div>"#);
        let sheet = Stylesheet::parse(
            ".w { width: 800px; max-width: 400px; } .pad { padding-top: 25vh; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let w = find_by_class(&dom, root, "w").unwrap();
        let wbox = tree.boxes.iter().find(|b| b.dom == Some(w)).unwrap();
        assert_eq!(wbox.width, 400.0);
        let pad = find_by_class(&dom, root, "pad").unwrap();
        let padbox = tree.boxes.iter().find(|b| b.dom == Some(pad)).unwrap();
        assert_eq!(padbox.style.padding[0], Spacing::Vh(25.0));
        // 25vh of a 600px viewport = 150px of top padding.
        assert!((padbox.height - 150.0 - 16.0 * 1.2).abs() < 1.0, "pad height {}", padbox.height);
    }

    /// The live example.com page as served 2026-11-23 (577 bytes):
    /// `max-width:26em`, `margin:auto`, `padding:25vh`, `text-align:
    /// center`, `font:16px/1.6 system-ui` — a centred 26em column.
    const NEW_EXAMPLE_COM: &str = "<!doctype html><html lang=en><head><meta charset=utf-8><link rel=icon href=data:,><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>Example Domain</title><style>html{color-scheme:light dark;background:light-dark(#eee,#222)}body{font:16px/1.6 system-ui,sans-serif;max-width:26em;margin:auto;padding:25vh 2em 2em;text-align:center}</style></head><body><p>This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.</p><script src=/s.js></script></body></html>";

    #[test]
    fn new_example_com_centred_column() {
        let dom = parse_html(NEW_EXAMPLE_COM);
        let style = find_element(&dom, dom.root().unwrap(), "style").unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        let sheet = Stylesheet::parse(&css);
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0, Theme::Light).expect("layout builds");
        let body = box_for(&tree, &dom, "body");
        // 26em at 16px = 416px content width; with `content-box`
        // the 2em side padding adds 64px, so the border box is 480
        // wide and auto margins centre it.
        assert_eq!(body.width, 480.0);
        assert_eq!(body.x, (1024.0 - 480.0) / 2.0);
        // font shorthand: 16px / 1.6.
        assert_eq!(body.style.font_size, 16.0);
        assert!((body.style.line_height - 25.6).abs() < 0.01);
        assert_eq!(body.style.text_align, TextAlign::Center);
        // 25vh top padding at a 768px viewport = 192px, plus the
        // paragraph's 1em top margin. (Body y is 0: the author
        // `margin:auto` overrides the UA 8px body margin.)
        let p = box_for(&tree, &dom, "p");
        assert!((p.y - (192.0 + 16.0)).abs() < 1.0, "p y {}", p.y);
        // The paragraph glyphs are centred on the viewport centre.
        let leaf = tree
            .boxes
            .iter()
            .find(|b| b.text.is_some())
            .expect("paragraph text leaf");
        assert!(!leaf.glyphs.is_empty());
        let min_x = leaf.glyphs.iter().map(|g| g.x).fold(f32::INFINITY, f32::min);
        let max_x = leaf.glyphs.iter().map(|g| g.x).fold(f32::NEG_INFINITY, f32::max);
        let centre = (min_x + max_x) / 2.0;
        assert!((centre - 512.0).abs() < 30.0, "glyph centre {centre}");
    }

    // ---- R12 grid ----

    #[test]
    fn grid_two_columns() {
        let dom = parse_html(
            r#"<div class="grid"><div class="a">A</div><div class="b">B</div><div class="c">C</div><div class="d">D</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".grid { display: grid; grid-template-columns: 100px 100px; width: 200px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let at = |class: &str| {
            let id = find_by_class(&dom, root, class).unwrap();
            tree.boxes.iter().find(|b| b.dom == Some(id)).unwrap().clone()
        };
        let a = at("a");
        let b = at("b");
        let c = at("c");
        let d = at("d");
        assert_eq!(a.width, 100.0);
        assert_eq!(b.x, a.x + 100.0);
        assert_eq!(a.y, b.y);
        assert_eq!(c.x, a.x);
        assert_eq!(d.x, b.x);
        assert!(c.y > a.y + a.height - 1.0);
        assert_eq!(c.y, d.y);
    }

    #[test]
    fn grid_fraction_tracks_share_space() {
        let dom = parse_html(
            r#"<div class="grid"><div class="a">A</div><div class="b">B</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".grid { display: grid; grid-template-columns: 1fr 1fr; width: 300px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let a = find_by_class(&dom, root, "a").unwrap();
        let bid = find_by_class(&dom, root, "b").unwrap();
        let abox = tree.boxes.iter().find(|b| b.dom == Some(a)).unwrap();
        let bbox = tree.boxes.iter().find(|b| b.dom == Some(bid)).unwrap();
        assert!((abox.width - 150.0).abs() < 0.5, "a width {}", abox.width);
        assert!((bbox.x - (abox.x + 150.0)).abs() < 0.5, "b x {}", bbox.x);
    }

    // ---- Stage 10 Rec 5: `grid-auto-flow` ----

    #[test]
    fn grid_auto_flow_parses_all_forms() {
        assert_eq!(parse_grid_auto_flow("row"), Some(GridAutoFlow::Row));
        assert_eq!(
            parse_grid_auto_flow("column"),
            Some(GridAutoFlow::Column)
        );
        assert_eq!(
            parse_grid_auto_flow("row dense"),
            Some(GridAutoFlow::RowDense)
        );
        assert_eq!(
            parse_grid_auto_flow("column dense"),
            Some(GridAutoFlow::ColumnDense)
        );
        // The `dense`-only form means `row dense` (grammar:
        // `[ row | column ] || dense`); order does not matter and
        // keywords are case-insensitive.
        assert_eq!(
            parse_grid_auto_flow("dense"),
            Some(GridAutoFlow::RowDense)
        );
        assert_eq!(
            parse_grid_auto_flow("dense row"),
            Some(GridAutoFlow::RowDense)
        );
        assert_eq!(
            parse_grid_auto_flow("dense column"),
            Some(GridAutoFlow::ColumnDense)
        );
        assert_eq!(
            parse_grid_auto_flow("ROW DENSE"),
            Some(GridAutoFlow::RowDense)
        );
        // Invalid: dropped by the caller, never a flow.
        for bad in [
            "",
            "sideways",
            "row column",
            "row row",
            "column column",
            "dense dense",
            "row dense dense",
            "dense row column",
            "row dense column",
            "auto",
        ] {
            assert_eq!(parse_grid_auto_flow(bad), None, "input {bad:?}");
        }
    }

    #[test]
    fn grid_auto_flow_column_places_column_first() {
        let dom = parse_html(
            r#"<div class="grid"><div class="a">A</div><div class="b">B</div><div class="c">C</div><div class="d">D</div></div>"#,
        );
        // Two explicit rows: with zero explicit rows column flow
        // wraps immediately (the auto-placement cursor is already
        // past the last explicit row), putting every item in its
        // own implicit column — so the rows must be explicit for
        // column-first placement to be observable.
        let sheet = Stylesheet::parse(
            ".grid { display: grid; grid-template-columns: 100px 100px; grid-template-rows: 20px 20px; grid-auto-flow: column; width: 200px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let at = |class: &str| {
            let id = find_by_class(&dom, root, class).unwrap();
            tree.boxes.iter().find(|b| b.dom == Some(id)).unwrap().clone()
        };
        let (a, b, c, d) = (at("a"), at("b"), at("c"), at("d"));
        // Column flow fills column 1 first (a, b stacked), then
        // column 2 (c, d stacked) — the opposite of the default
        // row flow proved by `grid_two_columns`.
        assert_eq!(b.x, a.x);
        assert_eq!(b.y, a.y + 20.0);
        assert_eq!(c.x, a.x + 100.0);
        assert_eq!(c.y, a.y);
        assert_eq!(d.x, c.x);
        assert_eq!(d.y, b.y);
    }

    #[test]
    fn grid_auto_flow_invalid_value_dropped() {
        let dom = parse_html(
            r#"<div class="grid"><div class="a">A</div><div class="b">B</div><div class="c">C</div><div class="d">D</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".grid { display: grid; grid-template-columns: 100px 100px; grid-auto-flow: sideways; width: 200px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let at = |class: &str| {
            let id = find_by_class(&dom, root, class).unwrap();
            tree.boxes.iter().find(|b| b.dom == Some(id)).unwrap().clone()
        };
        let (a, b, c) = (at("a"), at("b"), at("c"));
        // Invalid value dropped: default row flow — b sits beside
        // a in row 1, c starts row 2 under a.
        assert_eq!(b.x, a.x + 100.0);
        assert_eq!(b.y, a.y);
        assert_eq!(c.x, a.x);
        assert!(c.y > a.y, "c below a: c.y {} a.y {}", c.y, a.y);
    }

    // ---- R14 overflow/scroll + R15 hit-testing ----

    #[test]
    fn scroll_container_clips_and_offsets() {
        let dom = parse_html(
            r#"<div class="scroll"><div class="tall">hello</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".scroll { overflow: hidden; height: 50px; width: 200px; } .tall { height: 200px; }",
        );
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let scroll = find_by_class(&dom, root, "scroll").unwrap();
        let tall = find_by_class(&dom, root, "tall").unwrap();
        assert!(is_bfc_root(
            &tree.boxes.iter().find(|b| b.dom == Some(scroll)).unwrap().style,
            false
        ));
        let tall_before = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(tall))
            .unwrap()
            .clone();
        // The tall child is clipped to the scroll container's rect.
        // The container sits at (8,8), not (0,0): Pluto's
        // parse_document synthesises a <body> around the fragment,
        // and the UA stylesheet gives it its 8px margin.
        assert_eq!(tall_before.clip, Some([8.0, 8.0, 200.0, 50.0]));
        assert_eq!(tall_before.scroll_ancestor, Some(scroll));
        // Hit-testing finds the tall child (later = on top).
        assert_eq!(tree.hit_test(10.0, 10.0), Some(tall));
        assert_eq!(tree.hit_test(700.0, 500.0), None);
        // Scrolling the container shifts the child; layout is not
        // recomputed (the container's own box does not move).
        tree.set_scroll_offset(scroll, 0.0, 30.0);
        let tall_after = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(tall))
            .unwrap()
            .clone();
        assert!((tall_after.y - (tall_before.y - 30.0)).abs() < 0.01);
        let scroll_after = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(scroll))
            .unwrap()
            .clone();
        assert_eq!(scroll_after.y, 8.0);
        assert_eq!(tree.scroll_offset(scroll), Some((0.0, 30.0)));
        assert_eq!(tree.scroll_containers().len(), 1);
        // mark_dirty hook: known node true, unknown node false.
        assert!(tree.mark_dirty(tall));
        assert!(!tree.mark_dirty(dom.root().unwrap()));
    }

    // ---- M6 Phase 2 (Rec 4/5/7/8): Appendix E paint order,
    // stacking contexts, page transforms, transformed hit-test ----

    /// The paint-order position of one element's box (its index in
    /// `tree.paint_order()`), for asserting relative order.
    fn paint_pos(tree: &LayoutTree, dom: &Dom, class: &str) -> usize {
        let root = dom.root().unwrap();
        let id = find_by_class(dom, root, class).unwrap();
        let idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(id))
            .expect("box for class");
        tree.paint_order()
            .iter()
            .position(|&i| i == idx)
            .expect("box in paint order")
    }

    #[test]
    fn paint_order_is_a_permutation_of_boxes() {
        let dom = parse_html(
            "<html><head></head><body><div class=\"a\">A</div><div class=\"b\">B</div></body></html>",
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let mut order = tree.paint_order();
        order.sort_unstable();
        let expected: Vec<usize> = (0..tree.boxes.len()).collect();
        assert_eq!(order, expected, "paint order covers every box once");
    }

    #[test]
    fn paint_order_follows_z_index_not_document_order() {
        // Three overlapping absolutely-positioned siblings in
        // document order a, b, c with z-index 3, 1, 2: Appendix E
        // paints them b (1), c (2), a (3) — and hit-testing agrees.
        let dom = parse_html(
            r#"<div class="wrap"><div class="a">A</div><div class="b">B</div><div class="c">C</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".wrap { position: relative; width: 200px; height: 200px; } \
             .a { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 3; background-color: #ff0000; } \
             .b { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 1; background-color: #00ff00; } \
             .c { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 2; background-color: #0000ff; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let (a, _b, _c) = (
            find_by_class(&dom, root, "a").unwrap(),
            find_by_class(&dom, root, "b").unwrap(),
            find_by_class(&dom, root, "c").unwrap(),
        );
        assert!(
            paint_pos(&tree, &dom, "b") < paint_pos(&tree, &dom, "c")
                && paint_pos(&tree, &dom, "c") < paint_pos(&tree, &dom, "a"),
            "z-index order b < c < a, got b={} c={} a={}",
            paint_pos(&tree, &dom, "b"),
            paint_pos(&tree, &dom, "c"),
            paint_pos(&tree, &dom, "a")
        );
        // All three overlap at (30, 30) (inside the wrap at the
        // 8px body margin); the topmost by paint order wins.
        assert_eq!(tree.hit_test(30.0, 30.0), Some(a));
        // `z-index` is ignored on a static box: a static sibling
        // with z-index 99 still paints in-flow, below a
        // positioned z-index: 1 sibling.
        let dom = parse_html(
            r#"<div class="wrap"><div class="s">S</div><div class="p">P</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".wrap { position: relative; width: 200px; height: 200px; } \
             .s { width: 100px; height: 100px; z-index: 99; background-color: #ff0000; } \
             .p { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 1; background-color: #00ff00; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        assert!(
            paint_pos(&tree, &dom, "s") < paint_pos(&tree, &dom, "p"),
            "static z-index: 99 paints below positioned z-index: 1"
        );
    }

    #[test]
    fn paint_order_negative_z_behind_parent_background() {
        // A negative-z child paints before (behind) its parent's
        // in-flow content — but after the parent's own background,
        // which is the context root's box and paints first. The
        // parent needs `z-index: 0` to be a stacking context at
        // all: a negative-z child of a non-context parent paints
        // in the *root* context's negative step, behind the
        // parent's own background.
        let dom = parse_html(
            r#"<div class="parent"><div class="neg">N</div><div class="flow">F</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".parent { position: relative; z-index: 0; width: 200px; height: 200px; background-color: #ffffff; } \
             .neg { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: -1; background-color: #ff0000; } \
             .flow { width: 50px; height: 50px; background-color: #00ff00; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        assert!(
            paint_pos(&tree, &dom, "parent") < paint_pos(&tree, &dom, "neg")
                && paint_pos(&tree, &dom, "neg") < paint_pos(&tree, &dom, "flow"),
            "parent bg < negative-z child < in-flow content"
        );
        // ...and the negative-z child still hit-tests: (98, 98)
        // is inside the 100x100 negative child but outside the
        // 50x50 in-flow sibling, so the negative child is topmost
        // there.
        let root = dom.root().unwrap();
        let neg = find_by_class(&dom, root, "neg").unwrap();
        assert_eq!(tree.hit_test(8.0 + 90.0, 8.0 + 90.0), Some(neg));
    }

    #[test]
    fn paint_order_opacity_creates_stacking_context() {
        // `opacity: 0.5` is an implicit stacking context: the
        // translucent parent's children paint inside its group —
        // after the parent's own box and before the positioned
        // sibling that follows the parent in the outer context.
        let dom = parse_html(
            r#"<div class="ghost"><div class="kid">K</div></div><div class="sib">S</div>"#,
        );
        let sheet = Stylesheet::parse(
            ".ghost { opacity: 0.5; width: 100px; height: 100px; background-color: #ff0000; } \
             .kid { width: 50px; height: 50px; background-color: #00ff00; } \
             .sib { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 1; background-color: #0000ff; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let ghost = tree
            .boxes
            .iter()
            .find(|b| {
                b.dom.is_some_and(|id| {
                    dom.tag(id) == Some("div")
                        && dom.get(id)
                            .and_then(|n| n.attr("class"))
                            .is_some_and(|c| c == "ghost")
                })
            })
            .expect("ghost box");
        assert!(ghost.is_stacking_context, "opacity < 1 is a stacking context");
        assert!(
            paint_pos(&tree, &dom, "ghost") < paint_pos(&tree, &dom, "kid")
                && paint_pos(&tree, &dom, "kid") < paint_pos(&tree, &dom, "sib"),
            "ghost < kid < positioned sibling"
        );
    }

    #[test]
    fn page_transform_composes_through_rotated_ancestor() {
        // A 90-degree rotation about the parent's centre maps the
        // parent's local +x axis to page +y; the child's page
        // transform must carry that rotation, and its transformed
        // top-left must be the parent's top-left plus the rotated
        // offset. Parent 100x100 at (8, 8) (body margin), child
        // 10x10 at layout offset (80, 0) inside the parent.
        let dom = parse_html(
            r#"<div class="parent"><div class="child">C</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".parent { width: 100px; height: 100px; transform: rotate(90deg); transform-origin: 50% 50%; } \
             .child { width: 10px; height: 10px; margin-left: 80px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let parent_id = find_by_class(&dom, root, "parent").unwrap();
        let child_id = find_by_class(&dom, root, "child").unwrap();
        let transforms = tree.page_transforms();
        let parent_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(parent_id))
            .unwrap();
        let child_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(child_id))
            .unwrap();
        let parent_box = &tree.boxes[parent_idx];
        let child_box = &tree.boxes[child_idx];
        assert_eq!((parent_box.x, parent_box.y), (8.0, 8.0));
        // The parent's own matrix is a 90-degree rotation about
        // its centre (origin 50,50): local (0,0) maps to (100,0)
        // in parent-local space.
        let (px, py) = parent_box.matrix.apply(0.0, 0.0);
        assert!((px - 100.0).abs() < 0.01 && (py - 0.0).abs() < 0.01, "parent matrix {px} {py}");
        // The child's page transform carries the rotation: its
        // local +x maps to page +y.
        let child_t = transforms[child_idx];
        let (cx, cy) = child_t.apply(child_box.x, child_box.y);
        let (cx1, cy1) = child_t.apply(child_box.x + 10.0, child_box.y);
        assert!((cx1 - cx).abs() < 0.01, "child local +x is page +y, got {cx},{cy} -> {cx1},{cy1}");
        assert!((cy1 - cy - 10.0).abs() < 0.01, "child local +x is page +y, got {cx},{cy} -> {cx1},{cy1}");
        // The child's transformed top-left: parent top-left plus
        // the parent matrix applied to the (80, 0) offset — the
        // rotation turns (80, 0) into (100, 80) in parent-local
        // space, so the child sits at page (8 + 100, 8 + 80) less
        // the rotation of its own origin... its top-left corner
        // is the image of its layout top-left under its own page
        // transform.
        let (tx, ty) = child_t.apply(child_box.x, child_box.y);
        assert!((tx - 108.0).abs() < 0.01 && (ty - 88.0).abs() < 0.01, "child top-left {tx} {ty}");
        // Two levels: a grandchild inside the child composes
        // through the same rotation — its offset (5, 0) inside
        // the child maps to page (0, +5).
        let dom = parse_html(
            r#"<div class="parent"><div class="child"><div class="grand">G</div></div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".parent { width: 100px; height: 100px; transform: rotate(90deg); transform-origin: 50% 50%; } \
             .child { width: 10px; height: 10px; margin-left: 80px; } \
             .grand { width: 5px; height: 5px; margin-left: 5px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let grand_id = find_by_class(&dom, dom.root().unwrap(), "grand").unwrap();
        let grand_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(grand_id))
            .unwrap();
        let grand_box = &tree.boxes[grand_idx];
        let grand_t = tree.page_transforms()[grand_idx];
        let (gx, gy) = grand_t.apply(grand_box.x, grand_box.y);
        assert!((gx - 108.0).abs() < 0.01 && (gy - 93.0).abs() < 0.01, "grandchild top-left {gx} {gy}");
    }

    #[test]
    fn hit_test_goes_through_rotated_box() {
        // A 100x100 box rotated 45 degrees about its centre: its
        // axis-aligned bounding box is ~141x141, but the corners
        // of that bbox are outside the rotated rect. A click at a
        // bbox corner misses; a click at the centre hits.
        let dom = parse_html(r#"<div class="spin">S</div>"#);
        let sheet = Stylesheet::parse(
            ".spin { position: absolute; left: 100px; top: 100px; width: 100px; height: 100px; transform: rotate(45deg); }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let spin = find_by_class(&dom, root, "spin").unwrap();
        let spin_box = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(spin))
            .expect("spin box")
            .clone();
        // Points are derived from the box's own geometry, not
        // assumed coordinates: absolute positioning resolves
        // against the body's box, so the layout position is the
        // body's (8, 8) plus the 100px offsets.
        let (cx, cy) = (spin_box.x + 50.0, spin_box.y + 50.0);
        // Centre of the box hits.
        assert_eq!(tree.hit_test(cx, cy), Some(spin));
        // The bbox corner (the layout top-left) is outside the
        // rotated diamond.
        assert_eq!(tree.hit_test(spin_box.x, spin_box.y), None);
        // Just inside the diamond's top edge hits: the top vertex
        // is the image of the layout top-left under the page
        // transform, and 12px below it (toward the centre) is
        // inside.
        let transforms = tree.page_transforms();
        let spin_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin))
            .unwrap();
        let (vx, vy) = transforms[spin_idx].apply(spin_box.x, spin_box.y);
        assert_eq!(tree.hit_test(vx, vy + 12.0), Some(spin));
        // And a point just outside the top vertex misses.
        assert_eq!(tree.hit_test(vx, vy - 12.0), None);
        // A singular transform (scale 0) is unhittable.
        let dom = parse_html(r#"<div class="flat">F</div>"#);
        let sheet = Stylesheet::parse(
            ".flat { position: absolute; left: 100px; top: 100px; width: 100px; height: 100px; transform: scale(0); }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let flat = find_by_class(&dom, dom.root().unwrap(), "flat").unwrap();
        assert_eq!(tree.hit_test(150.0, 150.0), None);
        let _ = flat;
    }

    #[test]
    fn text_leaf_under_rotated_parent_is_not_double_rotated() {
        // A text leaf clones its parent element's computed style,
        // so without special handling it would compose the
        // parent's transform a second time (its own matrix would
        // be rotate(45) * rotate(45)) and paint its glyphs at
        // 90 degrees. The leaf has no transform of its own: its
        // matrix must equal the parent's, its page transform the
        // same map (same linear part, same image of any point),
        // and it is never a stacking context of its own — it
        // paints inside the parent's group.
        let dom = parse_html(r#"<div class="spin">S</div>"#);
        let sheet = Stylesheet::parse(
            ".spin { position: absolute; left: 100px; top: 100px; width: 100px; height: 100px; transform: rotate(45deg); }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light)
            .expect("layout builds");
        let spin = find_by_class(&dom, dom.root().unwrap(), "spin").unwrap();
        let spin_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin))
            .unwrap();
        let leaf_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom.is_none() && b.text.is_some())
            .expect("text leaf");
        let spin_box = &tree.boxes[spin_idx];
        let leaf_box = &tree.boxes[leaf_idx];
        assert_eq!(leaf_box.matrix, spin_box.matrix);
        assert!(!leaf_box.is_stacking_context);
        let transforms = tree.page_transforms();
        let (lx, ly) = transforms[leaf_idx].apply(leaf_box.x, leaf_box.y);
        let (sx, sy) = transforms[spin_idx].apply(leaf_box.x, leaf_box.y);
        assert!((lx - sx).abs() < 0.01 && (ly - sy).abs() < 0.01,
            "leaf page transform ({lx}, {ly}) != parent ({sx}, {sy})");
    }

    #[test]
    fn composited_state_reports_matrix_and_opacity() {
        let dom = parse_html(r#"<div class="ghost">G</div>"#);
        let sheet = Stylesheet::parse(
            ".ghost { width: 100px; height: 100px; opacity: 0.5; transform: translate(10px, 20px); }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let ghost = find_by_class(&dom, dom.root().unwrap(), "ghost").unwrap();
        let state = tree.composited_state(&[ghost]);
        assert_eq!(state.len(), 1);
        let (key, matrix, opacity) = state[0];
        assert_eq!(key, LayoutTree::dom_key(ghost));
        assert!((opacity - 0.5).abs() < 1e-6);
        // The page transform carries the translate: the layout
        // top-left (8, 8) maps to (18, 28).
        let ghost_box = tree.boxes.iter().find(|b| b.dom == Some(ghost)).unwrap();
        let (tx, ty) = matrix.apply(ghost_box.x, ghost_box.y);
        assert!((tx - 18.0).abs() < 0.01 && (ty - 28.0).abs() < 0.01, "translated {tx} {ty}");
        // Ids with no box are skipped, not errors: `<head>` is
        // unrendered, and the document root itself has no box.
        let head = find_element(&dom, dom.root().unwrap(), "head").unwrap();
        assert!(tree.composited_state(&[head]).is_empty());
        assert!(tree
            .composited_state(&[dom.root().unwrap()])
            .is_empty());
    }

    #[test]
    fn composited_subtree_state_covers_descendants() {
        // A transform change on a container moves its whole
        // subtree: the update must carry the child element too,
        // with the child's own new page transform — not just the
        // container's. The text leaf needs no entry of its own
        // (it shares its parent element's page transform).
        let dom = parse_html(
            r#"<div class="wrap"><div class="inner">I</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".wrap { width: 200px; height: 200px; transform: translate(10px, 0px); } \
             .inner { width: 50px; height: 50px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light)
            .expect("layout builds");
        let root = dom.root().unwrap();
        let wrap = find_by_class(&dom, root, "wrap").unwrap();
        let inner = find_by_class(&dom, root, "inner").unwrap();
        let state = tree.composited_subtree_state(&[wrap]);
        let keys: Vec<u64> = state.iter().map(|s| s.0).collect();
        assert!(keys.contains(&LayoutTree::dom_key(wrap)));
        assert!(keys.contains(&LayoutTree::dom_key(inner)));
        assert_eq!(state.len(), 2, "only the wrap and its child");
        // The inner's matrix carries the wrap's translate.
        let inner_box = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(inner))
            .unwrap();
        let (_, im, _) = state
            .iter()
            .find(|s| s.0 == LayoutTree::dom_key(inner))
            .unwrap();
        let (tx, ty) = im.apply(inner_box.x, inner_box.y);
        assert!((tx - (inner_box.x + 10.0)).abs() < 0.01, "inner moved {tx}");
        assert!((ty - inner_box.y).abs() < 0.01, "inner y {ty}");
    }

    #[test]
    fn overscroll_behavior_parses_to_computed_style() {
        // Stage 8 Rec 9: `overscroll-behavior` (and the -x/-y
        // longhands) reach the computed style; the default is
        // `auto` (chains), `contain`/`none` stop the chain.
        let dom = parse_html(
            r#"<div class="a"></div><div class="b"></div><div class="c"></div><div class="d"></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".a { overflow: auto; overscroll-behavior: none; } \
             .b { overflow: auto; overscroll-behavior-x: contain; overscroll-behavior-y: auto; } \
             .c { overflow: auto; } \
             .d { overflow: auto; overscroll-behavior: bogus; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let style_of = |class: &str| {
            let id = find_by_class(&dom, root, class).unwrap();
            tree.boxes
                .iter()
                .find(|b| b.dom == Some(id))
                .unwrap()
                .style
                .clone()
        };
        let a = style_of("a");
        assert_eq!(a.overscroll_behavior_x, OverscrollBehavior::None);
        assert_eq!(a.overscroll_behavior_y, OverscrollBehavior::None);
        assert!(a.overscroll_behavior_x.stops_chaining());
        let b = style_of("b");
        assert_eq!(b.overscroll_behavior_x, OverscrollBehavior::Contain);
        assert_eq!(b.overscroll_behavior_y, OverscrollBehavior::Auto);
        assert!(!b.overscroll_behavior_y.stops_chaining());
        let c = style_of("c");
        assert_eq!(c.overscroll_behavior_x, OverscrollBehavior::Auto);
        assert_eq!(c.overscroll_behavior_y, OverscrollBehavior::Auto);
        // An unknown keyword is dropped: the initial value stands.
        let d = style_of("d");
        assert_eq!(d.overscroll_behavior_x, OverscrollBehavior::Auto);
    }

    #[test]
    fn bfc_root_predicate() {
        let mut style = ComputedStyle::initial();
        assert!(!is_bfc_root(&style, false));
        assert!(is_bfc_root(&style, true));
        style.display = DisplayKind::FlowRoot;
        assert!(is_bfc_root(&style, false));
        style.display = DisplayKind::Block;
        style.overflow_x = OverflowKind::Hidden;
        assert!(is_bfc_root(&style, false));
        style.overflow_x = OverflowKind::Clip;
        assert!(!is_bfc_root(&style, false));
        style.position = PositionKind::Absolute;
        assert!(is_bfc_root(&style, false));
    }

    // ---- R16 border-radius / box-shadow stored ----

    #[test]
    fn border_radius_and_box_shadow_stored() {
        let dom = parse_html(r#"<div class="r">x</div>"#);
        let styles = styles_for(
            r#"<div class="r">x</div>"#,
            ".r { border-radius: 8px; box-shadow: 2px 2px 4px #000000; }",
            (800.0, 600.0),
        );
        let r = find_by_class(&dom, dom.root().unwrap(), "r").unwrap();
        assert!(styles[&r].border_radius.is_some());
        assert_eq!(styles[&r].box_shadow.len(), 1);
        assert!(!styles[&r].box_shadow[0].inset);
        // And the layout box carries them through to the display list
        // boundary (paint consumes them in Stage 8).
        let sheet = Stylesheet::parse(
            ".r { border-radius: 8px; box-shadow: 2px 2px 4px #000000; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let rbox = tree.boxes.iter().find(|b| b.dom == Some(r)).unwrap();
        assert!(rbox.style.border_radius.is_some());
        assert_eq!(rbox.style.box_shadow.len(), 1);
    }

    // ---- Stage 8: resolved radii, overflow: clip ----

    #[test]
    fn overflow_clip_clips_but_creates_no_scroll_container() {
        // Stage 8 Rec 3: `overflow: clip` clips to the padding box
        // and forbids scrolling — the child is clipped, but the box
        // is not a scroll ancestor and not a scroll container.
        let dom = parse_html(
            r#"<div class="clip"><div class="tall">hello</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".clip { overflow: clip; height: 50px; width: 200px; } .tall { height: 200px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let clip = find_by_class(&dom, root, "clip").unwrap();
        let tall = find_by_class(&dom, root, "tall").unwrap();
        let tall_box = tree.boxes.iter().find(|b| b.dom == Some(tall)).unwrap();
        assert_eq!(tall_box.clip, Some([8.0, 8.0, 200.0, 50.0]));
        assert_eq!(tall_box.scroll_ancestor, None);
        assert!(tree.scroll_containers().is_empty());
        assert!(!is_bfc_root(
            &tree.boxes.iter().find(|b| b.dom == Some(clip)).unwrap().style,
            false
        ));
    }

    #[test]
    fn boxes_carry_resolved_radii_and_clip_radii() {
        // Stage 8 Rec 3/4: a box's own radii are resolved against
        // its box (here 200x50 with 50% → rx 100, ry 25, no
        // scale-down: top 200/200 = 1, left 50/50 = 1), and a child
        // of a rounded scroll container inherits the clip radii.
        let dom = parse_html(
            r#"<div class="scroll"><div class="tall">hello</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".scroll { overflow: hidden; height: 50px; width: 200px; border-radius: 50%; } \
             .tall { height: 200px; border-radius: 4px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let root = dom.root().unwrap();
        let scroll = find_by_class(&dom, root, "scroll").unwrap();
        let tall = find_by_class(&dom, root, "tall").unwrap();
        let scroll_box = tree.boxes.iter().find(|b| b.dom == Some(scroll)).unwrap();
        assert_eq!(scroll_box.radii.top_left, (100.0, 25.0));
        assert_eq!(scroll_box.clip_radii, Some(scroll_box.radii));
        let tall_box = tree.boxes.iter().find(|b| b.dom == Some(tall)).unwrap();
        assert_eq!(tall_box.radii.top_left, (4.0, 4.0));
        assert_eq!(tall_box.clip, Some([8.0, 8.0, 200.0, 50.0]));
        assert_eq!(tall_box.clip_radii, Some(scroll_box.radii));
        // A square box carries zero radii and no clip radii.
        let dom = parse_html(r#"<div class="plain">x</div>"#);
        let sheet = Stylesheet::parse(".plain { width: 100px; height: 100px; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let plain = find_by_class(&dom, dom.root().unwrap(), "plain").unwrap();
        let plain_box = tree.boxes.iter().find(|b| b.dom == Some(plain)).unwrap();
        assert!(plain_box.radii.is_zero());
        assert_eq!(plain_box.clip_radii, None);
    }

    // ---- R19 node styles ----

    #[test]
    fn node_styles_populated_with_full_hints() {
        let dom = parse_html("<html><body><p>hi</p></body></html>");
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let p = find_element(&dom, dom.root().unwrap(), "p").unwrap();
        let ns = tree.node_styles.get(p).expect("node style for <p>");
        assert_eq!(ns.computed.font_size, 16.0);
        assert!(ns.hint.restyle_self && ns.hint.restyle_subtree);
        assert!(ns.damage.repaint && ns.damage.reflow);
    }

    // ---- R11 pre-shape: serial fallback path only (test) ----

    #[test]
    fn shared_font_system_shapes_without_rayon() {
        // The Mutex fallback exists for single-threaded/test paths;
        // production layout never takes it (see module docs).
        let fs = SharedFontSystem::new();
        let leaf = LeafText {
            spans: vec![TextSpan {
                text: "hello world".to_string(),
                color: [0, 0, 0],
                bold: false,
            }],
            font_size: 16.0,
            line_height: 19.2,
            shaped: None,
        };
        let shaped = fs.shape_text(&leaf);
        assert_eq!(shaped.segments.len(), 1);
        assert_eq!(shaped.segments[0].len(), 2);
        assert!(shaped.min_content > 0.0);
        assert!(shaped.max_content >= shaped.min_content);
    }

    // ---- Stage 11 (M3): images in layout ----

    fn img_tree(html: &str, images: &ImageMap) -> (Dom, LayoutTree) {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            images,
            Theme::Light,
        )
        .expect("layout builds");
        (dom, tree)
    }

    fn find_img(dom: &Dom) -> DomId {
        find_element(dom, dom.root().unwrap(), "img").expect("img in dom")
    }

    #[test]
    fn img_uses_intrinsic_size_then_attrs_then_css() {
        // No image yet: the 300x150 default.
        let (dom, tree) = img_tree(
            "<html><body><img src=\"a.png\"></body></html>",
            &ImageMap::default(),
        );
        let img = box_for(&tree, &dom, "img");
        assert_eq!((img.width, img.height), (300.0, 150.0));
        assert_eq!(img.image, None);

        // Decoded 400x100: intrinsic size, resource id on the box.
        let html = "<html><body><img src=\"a.png\"></body></html>";
        let dom2 = parse_html(html);
        let mut images = ImageMap::default();
        images.by_node.insert(
            find_img(&dom2),
            ImageInfo {
                resource: 7,
                width: 400,
                height: 100,
            },
        );
        let (dom, tree) = img_tree(html, &images);
        let img = box_for(&tree, &dom, "img");
        assert_eq!((img.width, img.height), (400.0, 100.0));
        assert_eq!(img.image, Some(7));

        // width attribute only: height scales by the intrinsic ratio.
        let (dom, tree) = img_tree(
            "<html><body><img src=\"a.png\" width=\"200\"></body></html>",
            &images,
        );
        // The re-parsed dom gets the same slotmap ids (same
        // insertion order), so the map built from the earlier
        // parse still addresses this img.
        assert_eq!(find_img(&dom), find_img(&dom2));
        let img = box_for(&tree, &dom, "img");
        assert_eq!((img.width, img.height), (200.0, 50.0));

        // Both attributes: used as-is, no ratio scaling.
        let (dom, tree) = img_tree(
            "<html><body><img src=\"a.png\" width=\"200\" height=\"80\"></body></html>",
            &images,
        );
        let img = box_for(&tree, &dom, "img");
        assert_eq!((img.width, img.height), (200.0, 80.0));
    }

    #[test]
    fn img_css_width_scales_height_by_ratio() {
        let mut images = ImageMap::default();
        let dom = parse_html("<html><body><img src=\"a.png\"></body></html>");
        let img_id = find_element(&dom, dom.root().unwrap(), "img").expect("img in dom");
        images.by_node.insert(
            img_id,
            ImageInfo {
                resource: 3,
                width: 400,
                height: 100,
            },
        );
        let sheet = Stylesheet::parse("img { width: 200px; }");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light
        )
        .expect("layout builds");
        let img = box_for(&tree, &dom, "img");
        // CSS width 200 with auto height: aspect_ratio derives
        // height 50 from the 4:1 intrinsic ratio.
        assert_eq!((img.width, img.height), (200.0, 50.0));
    }

    #[test]
    fn broken_img_renders_alt_text() {
        let dom = parse_html("<html><body><img src=\"x.png\" alt=\"photo of a cat\"></body></html>");
        let img_id = find_element(&dom, dom.root().unwrap(), "img").expect("img in dom");
        let mut images = ImageMap::default();
        images.broken.insert(img_id);
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light
        )
        .expect("layout builds");
        // The alt text is a text leaf on the img box itself.
        let img = box_for(&tree, &dom, "img");
        let alt = img
            .text
            .as_ref()
            .expect("broken img with alt has a text leaf");
        assert!(alt.spans.iter().any(|s| s.text == "photo of a cat"));
        assert_eq!(img.image, None);

        // Broken without alt: zero-size leaf, no text.
        let dom = parse_html("<html><body><img src=\"x.png\"></body></html>");
        let img_id = find_element(&dom, dom.root().unwrap(), "img").expect("img in dom");
        let mut images = ImageMap::default();
        images.broken.insert(img_id);
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light
        )
        .expect("layout builds");
        let img = box_for(&tree, &dom, "img");
        assert!(img.text.is_none());
        assert_eq!((img.width, img.height), (0.0, 0.0));
    }

    #[test]
    fn background_image_lands_on_box_with_intrinsic_size() {
        let mut images = ImageMap::default();
        images.backgrounds.insert(
            "bg.png".to_string(),
            ImageInfo {
                resource: 9,
                width: 800,
                height: 600,
            },
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: url(bg.png); background-size: cover; }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light
        )
        .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.background_image, Some(9));
        assert_eq!(div.background_image_size, Some((800, 600)));
        assert_eq!(div.style.background_size, BackgroundSize::Cover);
        assert_eq!(div.style.background_image.as_deref(), Some("bg.png"));

        // No decoded background yet: the box carries nothing.
        let dom = parse_html("<html><body><div></div></body></html>");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &ImageMap::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.background_image, None);
        assert_eq!(div.background_image_size, None);
    }

    #[test]
    fn background_gradient_lands_on_box_and_url_clears_it() {
        // Stage 15 Rec 7: a `linear-gradient()` background-image
        // lands in the computed style, and the url/gradient
        // slots are one property — setting one clears the other.
        let dom = parse_html("<html><body><div></div></body></html>");
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: linear-gradient(to right, red, blue); }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &ImageMap::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        let gradient = div
            .style
            .background_gradient
            .as_ref()
            .expect("gradient lands in the computed style");
        assert_eq!(gradient.stops.len(), 2);
        assert!(matches!(gradient.kind, GradientKind::Linear { .. }));
        assert_eq!(div.style.background_image, None);

        // A later url() rule wins and clears the gradient.
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: linear-gradient(to right, red, blue); } div { background-image: url(bg.png); }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &ImageMap::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.style.background_gradient, None);
        assert_eq!(div.style.background_image.as_deref(), Some("bg.png"));

        // And the other way round: a gradient after a url clears it.
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: url(bg.png); } div { background-image: radial-gradient(circle, white, black); }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &ImageMap::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert!(div.style.background_gradient.is_some());
        assert!(matches!(
            div.style.background_gradient.as_ref().unwrap().kind,
            GradientKind::Radial { .. }
        ));
        assert_eq!(div.style.background_image, None);
    }

    #[test]
    fn background_position_and_repeat_parse() {
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { background-position: right bottom; background-repeat: no-repeat; background-size: 50px 25px; }",
            (1024.0, 768.0),
        );
        // styles_for re-parses, so the div id must match the
        // fresh parse's (same insertion order → same slotmap ids).
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(
            div.background_position,
            (BackgroundPosition::End, BackgroundPosition::End)
        );
        assert_eq!(div.background_repeat, BackgroundRepeat::NoRepeat);
        assert_eq!(
            div.background_size,
            BackgroundSize::Lengths(Some(Dim::Px(50.0)), Some(Dim::Px(25.0)))
        );
    }

    // ---- M6 Phase 1 ----

    #[test]
    fn z_index_and_opacity_parse() {
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { z-index: -1; opacity: 0.5; }",
            (800.0, 600.0),
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.z_index, Some(-1));
        assert_eq!(div.opacity, 0.5);
        // `auto` → None; opacity clamps to 0..=1.
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { z-index: auto; opacity: 1.5; }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.z_index, None);
        assert_eq!(div.opacity, 1.0);
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { opacity: 50%; }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.opacity, 0.5);
    }

    #[test]
    fn transform_origin_and_will_change_parse() {
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { transform: rotate(45deg); transform-origin: 25% 75%; transform-box: fill-box; will-change: transform; }",
            (800.0, 600.0),
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(
            div.transform,
            vec![TransformFunction::Rotate(45.0)]
        );
        assert_eq!(
            div.transform_origin,
            TransformOrigin {
                x: OriginComponent::Percentage(25.0),
                y: OriginComponent::Percentage(75.0),
            }
        );
        assert_eq!(div.transform_box, TransformBox::FillBox);
        assert!(div.will_change_transform);
        // One origin value centres the other axis; `will-change:
        // opacity` does not set the transform hint.
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { transform-origin: left; will-change: opacity; }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(
            div.transform_origin,
            TransformOrigin {
                x: OriginComponent::Keyword(OriginKeyword::Left),
                y: OriginComponent::Keyword(OriginKeyword::Center),
            }
        );
        assert!(!div.will_change_transform);
        // Defaults: empty transform, 50% 50% origin, view-box.
        let styles = styles_for("<html><body><div></div></body></html>", "div {}", (800.0, 600.0));
        let div = styles.get(&div_id).expect("div style");
        assert!(div.transform.is_empty());
        assert_eq!(
            div.transform_origin,
            TransformOrigin {
                x: OriginComponent::Percentage(50.0),
                y: OriginComponent::Percentage(50.0),
            }
        );
        assert_eq!(div.transform_box, TransformBox::ViewBox);
    }

    #[test]
    fn transition_and_animation_shorthand_and_longhands() {
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { transition: opacity .3s ease-in .1s; animation: spin 2s linear infinite; }",
            (800.0, 600.0),
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.transitions.len(), 1);
        assert_eq!(div.transitions[0].property, "opacity");
        assert!((div.transitions[0].duration - 0.3).abs() < 1e-6);
        assert_eq!(div.transitions[0].timing, Easing::EaseIn);
        assert!((div.transitions[0].delay - 0.1).abs() < 1e-6);
        assert_eq!(div.animations.len(), 1);
        assert_eq!(div.animations[0].name, "spin");
        assert_eq!(div.animations[0].duration, 2.0);
        assert_eq!(div.animations[0].timing, Easing::Linear);
        assert_eq!(
            div.animations[0].iteration_count,
            IterationCount::Infinite
        );
        // Longhands zip onto the shorthand entry; a longhand
        // alone creates a default entry.
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { transition-duration: .5s; transition-delay: .2s; animation-name: fade; animation-duration: 1s; animation-direction: reverse; }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.transitions.len(), 1);
        assert!((div.transitions[0].duration - 0.5).abs() < 1e-6);
        assert!((div.transitions[0].delay - 0.2).abs() < 1e-6);
        assert_eq!(div.animations.len(), 1);
        assert_eq!(div.animations[0].name, "fade");
        assert_eq!(div.animations[0].duration, 1.0);
        assert_eq!(div.animations[0].direction, AnimationDirection::Reverse);
    }

    #[test]
    fn var_resolves_with_fallback_and_root_chain() {
        let html = "<html><body><div class=\"a\"><div class=\"b\"></div><div class=\"c\"></div></div></body></html>";
        let css = "html { --main: 20px; } \
                   .a { width: var(--main); } \
                   .b { width: var(--nope, 30px); height: var(--nope); } \
                   .c { width: calc(var(--main) + 10px); }";
        let styles = styles_for(html, css, (800.0, 600.0));
        let dom = parse_html(html);
        let root = dom.root().unwrap();
        let a = find_by_class(&dom, root, "a").expect("a");
        let b = find_by_class(&dom, root, "b").expect("b");
        let c = find_by_class(&dom, root, "c").expect("c");
        let a = styles.get(&a).expect("a style");
        let b = styles.get(&b).expect("b style");
        let c = styles.get(&c).expect("c style");
        // `:root`-inherited `--main` resolves for the descendant.
        assert_eq!(a.width, Some(Dim::Px(20.0)));
        // Fallback used when the var is undefined...
        assert_eq!(b.width, Some(Dim::Px(30.0)));
        // ...and an unresolvable var with no fallback drops the
        // declaration (height stays initial `None`).
        assert_eq!(b.height, None);
        // A `var()` inside `calc()` resolves first, then evaluates.
        assert_eq!(c.width, Some(Dim::Px(30.0)));
    }

    #[test]
    fn calc_and_clamp_evaluate_in_dim_value() {
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { width: calc(100px + 20px); height: clamp(10px, 50px, 20px); }",
            (800.0, 600.0),
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.width, Some(Dim::Px(120.0)));
        assert_eq!(div.height, Some(Dim::Px(20.0)));
        // `calc(100% - 20px)` has no containing-block basis at
        // style time: unresolvable → falls back to `auto`.
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { width: calc(100% - 20px); }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(div.width, None);
    }

    #[test]
    fn prefers_color_scheme_rules_apply_only_with_theme() {
        let html = "<html><body><div></div></body></html>";
        let css = "@media (prefers-color-scheme: dark) { div { background-color: #000000; } } \
                   div { width: 100px; height: 100px; }";
        let sheet = Stylesheet::parse(css);
        for (theme, want) in [(Theme::Dark, Some([0, 0, 0])), (Theme::Light, None)] {
            let dom = parse_html(html);
            let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, theme)
                .expect("layout builds");
            let div = box_for(&tree, &dom, "div");
            assert_eq!(div.style.background, want, "theme {theme:?}");
        }
    }

    #[test]
    fn origin_percent_resolves_against_border_box() {
        let dom = parse_html("<html><body><div></div></body></html>");
        let sheet = Stylesheet::parse(
            "div { width: 200px; height: 100px; transform: rotate(90deg); transform-origin: 50% 50%; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light)
            .expect("layout builds");
        let div = box_for(&tree, &dom, "div");
        assert_eq!((div.width, div.height), (200.0, 100.0));
        // The transform makes the box a stacking context.
        assert!(div.is_stacking_context);
        // The origin (100, 50) is the fixed point of the rotation.
        let (x, y) = div.matrix.apply(100.0, 50.0);
        assert!((x - 100.0).abs() < 1e-4, "x={x}");
        assert!((y - 50.0).abs() < 1e-4, "y={y}");
        // A point 100px right of the origin lands 100px below it.
        let (x, y) = div.matrix.apply(200.0, 50.0);
        assert!((x - 100.0).abs() < 1e-4, "x={x}");
        assert!((y - 150.0).abs() < 1e-4, "y={y}");
        // `parent` is the body box's DOM id.
        let body_id = find_element(&dom, dom.root().unwrap(), "body").expect("body");
        assert_eq!(div.parent, Some(body_id));
    }

    #[test]
    fn abs_descendant_of_transformed_static_ancestor() {
        // The sharpest-trap test (report §4): the abs child must
        // resolve against the transformed static ancestor, not
        // the outer relative box.
        let html = "<html><body><div class=\"outer\"><div class=\"t\"><div class=\"abs\">x</div></div></div></body></html>";
        let css = "body { margin: 0; } .outer { position: relative; width: 400px; height: 400px; padding: 60px 0 0 50px; } \
                   .t { width: 200px; height: 200px; transform: translate(5px, 5px); } \
                   .abs { position: absolute; left: 10px; top: 20px; width: 50px; height: 50px; }";
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(css);
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light)
            .expect("layout builds");
        let root = dom.root().unwrap();
        let t = find_by_class(&dom, root, "t").expect("t");
        let abs = find_by_class(&dom, root, "abs").expect("abs");
        let t_box = tree
            .boxes()
            .iter()
            .find(|b| b.dom == Some(t))
            .expect("t box");
        let abs_box = tree
            .boxes()
            .iter()
            .find(|b| b.dom == Some(abs))
            .expect("abs box");
        // The transform is visual only: `.t` still lays out at
        // the padding offset, and its matrix carries translate.
        assert_eq!((t_box.x, t_box.y), (50.0, 60.0));
        assert!((t_box.matrix.e - 5.0).abs() < 1e-4);
        assert!((t_box.matrix.f - 5.0).abs() < 1e-4);
        // Against `.t` (50, 60) + (10, 20) — NOT against `.outer`
        // (which would give (10, 20)).
        assert_eq!((abs_box.x, abs_box.y), (60.0, 80.0));
    }

    #[test]
    fn background_repeat_two_value_parses() {
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { background-repeat: repeat no-repeat; }",
            (800.0, 600.0),
        );
        let dom = parse_html("<html><body><div></div></body></html>");
        let div_id = find_element(&dom, dom.root().unwrap(), "div").expect("div in dom");
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(
            div.background_repeat,
            BackgroundRepeat::RepeatXY(true, false)
        );
        let styles = styles_for(
            "<html><body><div></div></body></html>",
            "div { background-repeat: no-repeat repeat; }",
            (800.0, 600.0),
        );
        let div = styles.get(&div_id).expect("div style");
        assert_eq!(
            div.background_repeat,
            BackgroundRepeat::RepeatXY(false, true)
        );
    }

    #[test]
    fn stacking_context_predicate() {
        let mut style = ComputedStyle::initial();
        // Nothing by default.
        assert!(!is_stacking_context(&style, false));
        // Explicit: positioned + z-index.
        style.position = PositionKind::Relative;
        assert!(!is_stacking_context(&style, false));
        style.z_index = Some(0);
        assert!(is_stacking_context(&style, false));
        // Static + z-index alone is not a context...
        style.position = PositionKind::Static;
        assert!(!is_stacking_context(&style, false));
        // ...unless it is a flex/grid item.
        assert!(is_stacking_context(&style, true));
        // Implicit: opacity < 1, or a non-empty transform.
        style.z_index = None;
        style.opacity = 0.5;
        assert!(is_stacking_context(&style, false));
        style.opacity = 1.0;
        style.transform = vec![TransformFunction::Rotate(10.0)];
        assert!(is_stacking_context(&style, false));
        // `will-change: transform` is a hint only — no context.
        style.transform = Vec::new();
        style.will_change_transform = true;
        assert!(!is_stacking_context(&style, false));
    }

    // ---- Stage 15 (M7 Phase 1, Rec 2): <svg> / <canvas> replaced elements ----

    /// Build `html` with author `css` and return the tree + dom.
    fn svg_tree(html: &str, css: &str) -> (Dom, LayoutTree) {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(css);
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0, Theme::Light)
            .expect("layout builds");
        (dom, tree)
    }

    #[test]
    fn svg_intrinsic_size_from_width_height_attributes() {
        let (dom, tree) = svg_tree(
            r#"<svg width="100" height="50"><rect width="10" height="10"/></svg>"#,
            "",
        );
        let svg = box_for(&tree, &dom, "svg");
        assert_eq!((svg.width, svg.height), (100.0, 50.0));
        assert_eq!(svg.svg_canvas, Some(SvgCanvasKind::Svg));
        // The <rect> child builds no box.
        assert!(
            tree.boxes
                .iter()
                .all(|b| b.dom.is_none_or(|id| dom.tag(id) != Some("rect"))),
            "SVG children must not get boxes"
        );
    }

    #[test]
    fn svg_view_box_only_defaults_to_300x150() {
        let (dom, tree) = svg_tree(r#"<svg viewBox="0 0 100 50"></svg>"#, "");
        let svg = box_for(&tree, &dom, "svg");
        assert_eq!((svg.width, svg.height), (300.0, 150.0));
        assert_eq!(svg.svg_canvas, Some(SvgCanvasKind::Svg));
    }

    #[test]
    fn svg_view_box_ratio_derives_the_missing_axis() {
        let (dom, tree) = svg_tree(r#"<svg width="200" viewBox="0 0 100 50"></svg>"#, "");
        assert_eq!((box_for(&tree, &dom, "svg").width, box_for(&tree, &dom, "svg").height), (200.0, 100.0));
        let (dom, tree) = svg_tree(r#"<svg height="100" viewBox="0 0 100 50"></svg>"#, "");
        assert_eq!((box_for(&tree, &dom, "svg").width, box_for(&tree, &dom, "svg").height), (200.0, 100.0));
        // One attribute, no viewBox: the other axis is the default.
        let (dom, tree) = svg_tree(r#"<svg width="200"></svg>"#, "");
        assert_eq!((box_for(&tree, &dom, "svg").width, box_for(&tree, &dom, "svg").height), (200.0, 150.0));
    }

    #[test]
    fn foreign_object_gets_a_viewport_box_and_clip() {
        // Stage 15 (M7 Phase 3, Rec 10): <foreignObject> is a
        // real viewport — its HTML children lay out inside its
        // rect, clipped to it — while the svg's shape children
        // still build no boxes.
        let (dom, tree) = svg_tree(
            r#"<svg width="200" height="100"><foreignObject x="10" y="20" width="100" height="50"><div style="width: 80px; height: 40px;">hi</div></foreignObject><rect width="10" height="10"/></svg>"#,
            "",
        );
        let fo = box_for(&tree, &dom, "foreignObject");
        assert_eq!((fo.x, fo.y, fo.width, fo.height), (18.0, 28.0, 100.0, 50.0));
        assert!(fo.clip.is_some(), "foreignObject clips its content");
        let clip = fo.clip.unwrap();
        assert_eq!((clip[0], clip[1], clip[2], clip[3]), (18.0, 28.0, 100.0, 50.0));
        let div = box_for(&tree, &dom, "div");
        assert_eq!((div.x, div.y), (18.0, 28.0), "div lays out at the foreignObject origin");
        assert!(
            tree.boxes
                .iter()
                .all(|b| b.dom.is_none_or(|id| dom.tag(id) != Some("rect"))),
            "SVG shape children still build no boxes"
        );
    }

    #[test]
    fn foreign_object_text_lays_out_inside_the_svg() {
        let (dom, tree) = svg_tree(
            r#"<svg width="200" height="100"><foreignObject x="0" y="0" width="200" height="100"><p>hello</p></foreignObject></svg>"#,
            "",
        );
        assert!(
            tree.boxes.iter().any(|b| b.text.as_ref().is_some_and(|t| t
                .spans
                .iter()
                .any(|s| s.text.contains("hello")))),
            "foreignObject text lays out as HTML"
        );
        assert_eq!(box_for(&tree, &dom, "svg").svg_canvas, Some(SvgCanvasKind::Svg));
    }

    #[test]
    fn svg_css_size_wins_over_attributes() {
        let (dom, tree) = svg_tree(
            r#"<svg width="100" height="50"></svg>"#,
            "svg { width: 400px; height: 300px; }",
        );
        let svg = box_for(&tree, &dom, "svg");
        assert_eq!((svg.width, svg.height), (400.0, 300.0));
    }

    #[test]
    fn svg_text_does_not_leak_into_flow() {
        let (dom, tree) = svg_tree(
            r#"<p>before<svg viewBox="0 0 10 10"><text>hello</text></svg>after</p>"#,
            "",
        );
        assert!(
            tree.boxes.iter().all(|b| !b.text.as_ref().is_some_and(|t| t
                .spans
                .iter()
                .any(|s| s.text.contains("hello")))),
            "SVG <text> must not leak as HTML text"
        );
        assert!(
            tree.boxes.iter().any(|b| b.text.as_ref().is_some_and(|t| t
                .spans
                .iter()
                .any(|s| s.text.contains("before")))),
            "surrounding text still lays out"
        );
        assert_eq!(box_for(&tree, &dom, "svg").svg_canvas, Some(SvgCanvasKind::Svg));
    }

    #[test]
    fn svg_stays_atomic_under_author_display() {
        // Author `display: inline` must not re-open the leak the
        // UA `inline-block` rule closes; `display: block` must not
        // build the children either.
        let (dom, tree) = svg_tree(
            r#"<svg style="display: inline" width="50" height="40"><text>leak</text></svg>"#,
            "",
        );
        assert_eq!((box_for(&tree, &dom, "svg").width, box_for(&tree, &dom, "svg").height), (50.0, 40.0));
        assert!(
            tree.boxes.iter().all(|b| !b.text.as_ref().is_some_and(|t| t
                .spans
                .iter()
                .any(|s| s.text.contains("leak")))),
            "inline SVG <text> must not leak"
        );
        let (dom, tree) = svg_tree(
            r#"<svg style="display: block" width="50" height="40"><circle r="5"/></svg>"#,
            "",
        );
        assert_eq!((box_for(&tree, &dom, "svg").width, box_for(&tree, &dom, "svg").height), (50.0, 40.0));
        assert!(
            tree.boxes
                .iter()
                .all(|b| b.dom.is_none_or(|id| dom.tag(id) != Some("circle"))),
            "block SVG children must not get boxes"
        );
    }

    #[test]
    fn canvas_default_and_attribute_sizes() {
        let (dom, tree) = svg_tree(r#"<canvas></canvas>"#, "");
        let canvas = box_for(&tree, &dom, "canvas");
        assert_eq!((canvas.width, canvas.height), (300.0, 150.0));
        assert_eq!(canvas.svg_canvas, Some(SvgCanvasKind::Canvas));
        let (dom, tree) = svg_tree(r#"<canvas width="640" height="480"></canvas>"#, "");
        assert_eq!(
            (box_for(&tree, &dom, "canvas").width, box_for(&tree, &dom, "canvas").height),
            (640.0, 480.0)
        );
        let (dom, tree) = svg_tree(
            r#"<canvas width="640" height="480"></canvas>"#,
            "canvas { width: 100px; }",
        );
        // CSS width wins; the height follows the attribute ratio.
        let canvas = box_for(&tree, &dom, "canvas");
        assert_eq!(canvas.width, 100.0);
        assert_eq!(canvas.height, 75.0);
    }

    // ---- Stage 15 (M7 Phase 1, Rec 6): fill / stroke / presentation attributes ----

    #[test]
    fn fill_stroke_from_css() {
        let (dom, tree) = svg_tree(
            r#"<svg><path d="M0 0h10"/></svg>"#,
            "svg { fill: red; stroke: #0000ff; stroke-width: 2px; stop-color: green; }",
        );
        let style = &box_for(&tree, &dom, "svg").style;
        assert_eq!(style.fill, Paint::Color([255, 0, 0]));
        assert_eq!(style.stroke, Paint::Color([0, 0, 255]));
        assert_eq!(style.stroke_width, 2.0);
        assert_eq!(style.stop_color, [0, 128, 0]);
    }

    #[test]
    fn fill_none_and_url_and_current_color() {
        let (dom, tree) = svg_tree(r#"<svg></svg>"#, "svg { fill: none; }");
        assert_eq!(box_for(&tree, &dom, "svg").style.fill, Paint::None);
        let (dom, tree) = svg_tree(r#"<svg></svg>"#, "svg { fill: url(#grad); }");
        assert_eq!(
            box_for(&tree, &dom, "svg").style.fill,
            Paint::Url("#grad".to_string())
        );
        let (dom, tree) = svg_tree(
            r#"<svg style="color: blue" fill="currentColor"></svg>"#,
            "svg { fill: currentColor; stroke: currentColor; }",
        );
        // The presentation attribute and the CSS rule both say
        // currentColor; the post-pass resolves both against the
        // element's own computed color.
        let style = &box_for(&tree, &dom, "svg").style;
        assert_eq!(style.fill, Paint::Color([0, 0, 255]));
        assert_eq!(style.stroke, Paint::Color([0, 0, 255]));
    }

    #[test]
    fn presentation_attributes_lose_to_author_rules() {
        // Author `fill: red` beats the `fill="green"` presentation
        // attribute (specificity 0, below author rules).
        let (dom, tree) = svg_tree(
            r#"<svg fill="green" stroke="blue" stroke-width="3" stop-color="blue"></svg>"#,
            "svg { fill: red; }",
        );
        let style = &box_for(&tree, &dom, "svg").style;
        assert_eq!(style.fill, Paint::Color([255, 0, 0]));
        assert_eq!(style.stroke, Paint::Color([0, 0, 255]));
        assert_eq!(style.stroke_width, 3.0);
        assert_eq!(style.stop_color, [0, 0, 255]);
        // With no author rule the presentation attributes apply.
        let (dom, tree) = svg_tree(r#"<svg fill="green"></svg>"#, "");
        assert_eq!(
            box_for(&tree, &dom, "svg").style.fill,
            Paint::Color([0, 128, 0])
        );
    }

    #[test]
    fn presentation_attributes_do_not_leak_into_html() {
        // An HTML element's `fill` / `stroke` attributes are not
        // presentation attributes — they must not reach the cascade.
        let (dom, tree) = svg_tree(r#"<div fill="red" stroke="blue">x</div>"#, "");
        let style = &box_for(&tree, &dom, "div").style;
        assert_eq!(style.fill, Paint::Color([0, 0, 0]));
        assert_eq!(style.stroke, Paint::None);
    }

    #[test]
    fn fill_inherits_into_svg_children() {
        // `fill` is inherited: a <path> inside a fill="red" <svg>
        // resolves to red even though it never gets a box.
        let dom = parse_html(r#"<svg fill="red"><path d="M0 0h10"/></svg>"#);
        let sheet = Stylesheet::parse("");
        let ua = Stylesheet::ua();
        let (styles, _) = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (1024.0, 768.0),
            &ElementState::default(),
            Theme::Light,
        );
        let path = find_element(&dom, dom.root().unwrap(), "path").unwrap();
        assert_eq!(styles[&path].fill, Paint::Color([255, 0, 0]));
    }

    #[test]
    fn svg_anchor_with_xlink_href_matches_any_link() {
        // Stage 15 (M7 Phase 1, Rec 1): an SVG <a> links through
        // the legacy xlink:href too, so :any-link matches it.
        let dom = parse_html(
            r#"<svg><a xlink:href="https://example.com"><text>t</text></a></svg>"#,
        );
        let sheet = Stylesheet::parse("a:any-link { color: #123456; }");
        let ua = Stylesheet::ua();
        let (styles, _) = resolve_document_styles(
            &dom,
            &ua,
            &sheet,
            false,
            (1024.0, 768.0),
            &ElementState::default(),
            Theme::Light,
        );
        let a = find_element(&dom, dom.root().unwrap(), "a").unwrap();
        assert_eq!(styles[&a].color, [0x12, 0x34, 0x56]);
    }

    // ---- Stage 16 Phase 2 (Rec 5): <video> / <audio> leaves ----

    /// Build `html` with an image map carrying `videos`.
    fn video_tree(html: &str, videos: ImageMap) -> (Dom, LayoutTree) {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &videos,
            Theme::Light,
        )
        .expect("layout builds");
        (dom, tree)
    }

    #[test]
    fn video_defaults_to_300x150_before_metadata() {
        let (dom, tree) = video_tree(
            "<html><body><video src=\"v.mp4\">fallback text</video></body></html>",
            ImageMap::new(),
        );
        let video = box_for(&tree, &dom, "video");
        assert_eq!((video.width, video.height), (300.0, 150.0));
        assert_eq!(video.svg_canvas, Some(SvgCanvasKind::Video));
        assert!(!video.video_live);
        // Fallback children never build: the fallback text has
        // no box anywhere in the tree.
        assert!(
            !tree.boxes().iter().any(|b| b.text.as_ref().is_some_and(|t| t
                .spans
                .iter()
                .any(|s| s.text.contains("fallback")))),
            "no fallback text under the video leaf"
        );
    }

    #[test]
    fn video_uses_intrinsic_size_and_liveness_from_the_map() {
        let dom = parse_html("<html><body><video src=\"v.mp4\"></video></body></html>");
        let video_id = find_element(&dom, dom.root().unwrap(), "video").expect("video");
        let mut images = ImageMap::new();
        images.videos.insert(
            video_id,
            VideoInfo {
                width: 640,
                height: 360,
                has_frame: true,
            },
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let video = box_for(&tree, &dom, "video");
        assert_eq!((video.width, video.height), (640.0, 360.0));
        assert!(video.video_live);
    }

    #[test]
    fn video_width_attribute_scales_by_intrinsic_ratio() {
        let dom = parse_html(
            "<html><body><video src=\"v.mp4\" width=\"320\"></video></body></html>",
        );
        let video_id = find_element(&dom, dom.root().unwrap(), "video").expect("video");
        let mut images = ImageMap::new();
        images.videos.insert(
            video_id,
            VideoInfo {
                width: 640,
                height: 360,
                has_frame: false,
            },
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let video = box_for(&tree, &dom, "video");
        assert_eq!((video.width, video.height), (320.0, 180.0));
    }

    #[test]
    fn audio_is_a_zero_size_leaf() {
        let (dom, tree) = video_tree(
            "<html><body><audio src=\"a.mp3\" controls></audio><p>after</p></body></html>",
            ImageMap::new(),
        );
        let audio = box_for(&tree, &dom, "audio");
        assert_eq!((audio.width, audio.height), (0.0, 0.0));
        // The following paragraph still lays out (the audio took
        // no space and broke nothing).
        let p = box_for(&tree, &dom, "p");
        assert!(p.height > 0.0);
    }
}
