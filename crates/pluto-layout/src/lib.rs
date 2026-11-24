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
//! only a proven-safe subtree when it can (definite size,
//! `contain: layout`, replaced element, or fixed-size leaf, with no
//! margin collapse, float intrusion, or percentage resolution
//! escaping the root), translates position-only moves without
//! touching taffy, and falls back to a full rebuild for anything
//! unsafe. In debug builds every incremental result is asserted
//! equal to a cold full layout. Style re-resolution stays
//! whole-document in v1 (the parallel pass is cheap; the serial
//! taffy pass is where incrementality pays).

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping, Weight};
use pluto_css::{
    AnPlusB, AttrOp, BorderRadius, BoxShadow, CascadeKey, Combinator, Origin, PseudoClass,
    ResolvedCorners, Rule, Selector, SimpleSelector, Stylesheet, SupportsCondition, Value,
};
use pluto_html::{Dom, NodeId as DomId};
use rayon::prelude::*;
use slotmap::SparseSecondaryMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use taffy::{
    AvailableSpace, BoxSizing, Dimension, Display, FlexDirection, FlexWrap,
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

pub use incremental::{
    classify_damage, classify_style_change, is_safe_layout_root, unsafe_layout_root_reason,
    DamageKind, RelayoutKind, RelayoutOutcome,
};

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
    /// `contain: layout` (also implied by `contain: strict` /
    /// `content`): the subtree is layout-contained, so it can be a
    /// layout root even without a definite size (Stage 8 Rec 6).
    /// `contain: size` alone does NOT set this.
    pub contain_layout: bool,
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
            contain_layout: false,
        }
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
/// elements are currently hovered / active / focused. The shell owns
/// this (hit-testing feeds `:hover`); `:visited` is intentionally
/// absent — it is always false until a history exists (see
/// `match_pseudo_class`).
#[derive(Debug, Clone, Default)]
pub struct ElementState {
    pub hover: HashSet<DomId>,
    pub active: HashSet<DomId>,
    pub focus: HashSet<DomId>,
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
}

#[derive(Debug, Clone)]
struct BoxMeta {
    dom: Option<DomId>,
    style: ComputedStyle,
    children: Vec<TaffyId>,
    text: Option<LeafText>,
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
        }
    }

    /// Lay out `dom` with `sheet` in a viewport of the given size.
    /// Returns the tree with `boxes` in document order, positions in
    /// page coordinates (origin = viewport top-left). `sheet` is the
    /// document's author sheet; the parsed UA stylesheet
    /// (`Stylesheet::ua()`) is always applied underneath it.
    pub fn build(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Result<Self, LayoutError> {
        Self::build_with_state(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            &ElementState::default(),
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
    ) -> Result<Self, LayoutError> {
        Self::build_with_state_and_font_system(
            dom,
            sheet,
            viewport_width,
            viewport_height,
            state,
            font_system,
        )
    }

    fn build_with_state_and_font_system(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        font_system: FontSystem,
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
    fn recollect(&mut self) {
        let Some(root) = self.root else {
            return;
        };
        let mut boxes = Vec::new();
        collect_boxes(
            &self.taffy,
            &self.meta,
            root,
            (0.0, 0.0),
            &self.scroll_offsets,
            None,
            None,
            None,
            None,
            self.viewport,
            self.root_font_size,
            &self.position_overrides,
            &mut boxes,
        );
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

    /// Hit-test: the topmost (latest in document order) box whose rect
    /// contains `(x, y)`. Boxes already carry their scroll offsets
    /// (applied in `recollect`), so no offset subtraction is needed
    /// here. Text leaves have no `DomId` and never win.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<DomId> {
        self.boxes.iter().rev().find_map(|b| {
            let dom = b.dom?;
            if x >= b.x && x < b.x + b.width && y >= b.y && y < b.y + b.height {
                Some(dom)
            } else {
                None
            }
        })
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
    /// offsets only shift already-laid-out descendants.
    pub fn set_scroll_offset(&mut self, id: DomId, x: f32, y: f32) {
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
    fn append_inline(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
        parent_style: &ComputedStyle,
        spans: &mut Vec<TextSpan>,
        out: &mut Vec<TaffyId>,
    ) {
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
                    self.append_inline(child, &child_style, parent_style, spans, out);
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

    /// An atomic inline-level box (`display: inline-block`, or an
    /// `<img>`): a taffy leaf sized by its content. Its inline text
    /// merges into one leaf (same shaping as a block's text leaf); a
    /// block-level descendant falls back to a full node. `<img>` has
    /// no text, so it is a zero-size leaf unless width/height styles
    /// size it (image decoding itself is M3).
    fn build_atomic(&mut self, id: DomId, style: &ComputedStyle) -> Option<TaffyId> {
        let tag = self.dom.tag(id).unwrap_or("");
        if tag == "img" {
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
                },
            );
            return Some(node);
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
            },
        );
        self.leaves.push(node);
        Some(node)
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
            if let Some(m) = meta.get_mut(&id) {
                if let Some(leaf) = m.text.as_mut() {
                    leaf.shaped = Some(shaped.clone());
                }
            }
            if let Some(ctx) = taffy.get_node_context_mut(id) {
                if let Some(leaf) = ctx.as_mut() {
                    leaf.shaped = Some(shaped.clone());
                }
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
    out: &mut Vec<LayoutBox>,
) {
    let Some(layout) = taffy.layout(node).ok() else {
        return;
    };
    let Some(m) = meta.get(&node) else {
        return;
    };
    let (ox, oy) = position_overrides.get(&node).copied().unwrap_or((0.0, 0.0));
    let x = offset.0 + layout.location.x + ox;
    let y = offset.1 + layout.location.y + oy;
    let width = layout.size.width;
    let height = layout.size.height;
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
    // `overflow: clip` clips but never scrolls.
    let scroll_ancestor = if is_scroll_container {
        m.dom.or(parent_scroll_ancestor)
    } else {
        parent_scroll_ancestor
    };
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
            scroll_offsets,
            clip,
            clip_radii,
            clip_source,
            scroll_ancestor,
            viewport,
            root_font_size,
            position_overrides,
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
fn collect_style_levels(
    dom: &Dom,
    roots: &[DomId],
) -> Vec<Vec<(DomId, Option<DomId>)>> {
    let mut levels: Vec<Vec<(DomId, Option<DomId>)>> = Vec::new();
    let mut frontier: Vec<(DomId, Option<DomId>)> = roots
        .iter()
        .filter(|&&id| dom.tag(id).is_some())
        .map(|&id| (id, None))
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
    /// does not hold at this viewport are dropped here (conditions
    /// are evaluated at style time, not flattened at parse time — a
    /// later viewport change rebuilds the index).
    fn build(
        ua: &'a Stylesheet,
        author: &'a Stylesheet,
        viewport: (f32, f32),
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
                if !rule_applies(rule, viewport) {
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
        if let Some(want) = dom.attr(id, "id") {
            if let Some(bucket) = self.by_id.get(want) {
                out.extend(bucket.iter());
            }
        }
        if let Some(classes) = dom.attr(id, "class") {
            for class in classes.split_whitespace() {
                if let Some(bucket) = self.by_class.get(class) {
                    out.extend(bucket.iter());
                }
            }
        }
        if let Some(tag) = dom.tag(id) {
            if let Some(bucket) = self.by_tag.get(tag) {
                out.extend(bucket.iter());
            }
        }
        out.extend(self.other.iter());
        out
    }
}

/// Whether a rule's enclosing `@media` / `@supports` conditions hold.
/// Unknown media features evaluate to `None` and the rule is skipped
/// (a query that cannot be judged does not apply) — documented M2
/// choice; `@supports` unknown values likewise skip the rule.
fn rule_applies(rule: &Rule, viewport: (f32, f32)) -> bool {
    if let Some(media) = &rule.media {
        if media.eval(viewport.0, viewport.1) != Some(true) {
            return false;
        }
    }
    if let Some(supports) = &rule.supports {
        if SupportsCondition::parse(supports).eval() != Some(true) {
            return false;
        }
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
fn resolve_styles(
    dom: &Dom,
    index: &RuleIndex,
    levels: &[Vec<(DomId, Option<DomId>)>],
    parallel: bool,
    viewport: (f32, f32),
    state: &ElementState,
) -> (
    HashMap<DomId, Arc<ComputedStyle>>,
    SparseSecondaryMap<DomId, NodeStyle>,
) {
    let mut resolved: HashMap<DomId, Arc<ComputedStyle>> = HashMap::new();
    let mut node_styles: SparseSecondaryMap<DomId, NodeStyle> = SparseSecondaryMap::new();
    let mut root_font_size = 16.0f32;
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
                        .expect("parent style resolved in an earlier level")
                        .clone(),
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
) -> (
    HashMap<DomId, Arc<ComputedStyle>>,
    SparseSecondaryMap<DomId, NodeStyle>,
) {
    let root = dom.root().expect("Dom always has a root node");
    let roots: Vec<DomId> = match find_element(dom, root, "html") {
        Some(html) => vec![html],
        None => dom
            .children(root)
            .iter()
            .copied()
            .filter(|&id| dom.tag(id).is_some())
            .collect(),
    };
    let levels = collect_style_levels(dom, &roots);
    let index = RuleIndex::build(ua, sheet, viewport);
    resolve_styles(dom, &index, &levels, parallel, viewport, state)
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
                .iter()
                .filter(|&&s| dom.tag(s) == tag)
                .next()
                .copied()
                == Some(id)
        }
        PseudoClass::LastOfType => {
            let tag = dom.tag(id);
            element_siblings(dom, id)
                .iter()
                .filter(|&&s| dom.tag(s) == tag)
                .next_back()
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
        // `:link` = an anchor with an `href`. `:any-link` is the same
        // set here (no `:visited` distinction without history).
        PseudoClass::Link | PseudoClass::AnyLink => {
            matches!(dom.tag(id), Some("a" | "area" | "link"))
                && dom.attr(id, "href").is_some()
        }
        // `:visited` is ALWAYS FALSE until a history exists (Stage 6
        // report §c). The UA sheet's `a:visited` rule therefore never
        // applies in M2 — by design, and documented here, not a bug.
        PseudoClass::Visited => false,
        PseudoClass::Hover => state.hover.contains(&id),
        PseudoClass::Active => state.active.contains(&id),
        PseudoClass::Focus | PseudoClass::FocusVisible => state.focus.contains(&id),
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
            if let SimpleSelector::PseudoClass(pc) = part {
                if matches!(
                    pc,
                    PseudoClass::Hover
                        | PseudoClass::Active
                        | PseudoClass::Focus
                        | PseudoClass::FocusVisible
                        | PseudoClass::FocusWithin
                ) {
                    flags.has_dynamic_pseudo = true;
                }
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
    if let Some(attr) = dom.attr(id, "style") {
        if !attr.trim().is_empty() {
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
    }
    decls.sort_by_key(|(key, _, _)| *key);

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
        apply_decl(
            &mut style,
            property,
            value,
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
        apply_decl(
            &mut style,
            property,
            value,
            parent_font_size,
            root_font_size,
            viewport,
        );
    }
    (style, flags)
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
            if let Value::Raw(raw) = value {
                if let Some(sides) =
                    parse_shorthand(raw, style.font_size, root_font_size, viewport)
                {
                    style.margin = sides;
                }
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
            if let Value::Raw(raw) = value {
                if let Some(sides) =
                    parse_shorthand(raw, style.font_size, root_font_size, viewport)
                {
                    style.padding = sides;
                }
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
            Value::Color(..) | Value::ColorA(..) | Value::BorderRadius(..)
            | Value::BoxShadow(..) => {}
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
        "background-color" | "background" => match value {
            Value::Color(r, g, b) => style.background = Some([*r, *g, *b]),
            Value::ColorA(r, g, b, _) => style.background = Some([*r, *g, *b]),
            _ => {}
        },
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
            if let Value::Keyword(k) = value {
                if let Some(o) = overflow_kind(k) {
                    style.overflow_x = o;
                    style.overflow_y = o;
                }
            }
        }
        "overflow-x" => {
            if let Value::Keyword(k) = value {
                if let Some(o) = overflow_kind(k) {
                    style.overflow_x = o;
                }
            }
        }
        "overflow-y" => {
            if let Value::Keyword(k) = value {
                if let Some(o) = overflow_kind(k) {
                    style.overflow_y = o;
                }
            }
        }
        "overscroll-behavior" => {
            if let Value::Keyword(k) = value {
                if let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_x = o;
                    style.overscroll_behavior_y = o;
                }
            }
        }
        "overscroll-behavior-x" => {
            if let Value::Keyword(k) = value {
                if let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_x = o;
                }
            }
        }
        "overscroll-behavior-y" => {
            if let Value::Keyword(k) = value {
                if let Some(o) = overscroll_behavior(k) {
                    style.overscroll_behavior_y = o;
                }
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
        // Deliberately unmodelled in M2 (stored nowhere, never an
        // error): float/clear (no float layout yet), border widths
        // (no border painting yet), text-decoration, list-style-type,
        // visibility, opacity, z-index, justify-content/align-*,
        // transform/transition, cursor, content. They are ignored
        // here; adding them is data, not a rewrite.
        _ => {}
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
/// `Dim::Vw`); `auto` is `None`.
fn dim_value(
    value: &Value,
    font_size: f32,
    root_font_size: f32,
    viewport: (f32, f32),
) -> Option<Dim> {
    match value {
        Value::Length(v) => Some(Dim::Px(*v)),
        Value::Percentage(p) => Some(Dim::Pct(p / 100.0)),
        Value::Em(v) => Some(Dim::Px(v * font_size)),
        Value::Rem(v) => Some(Dim::Px(v * root_font_size)),
        Value::Vh(v) => Some(Dim::Vh(*v)),
        Value::Vw(v) => Some(Dim::Vw(*v)),
        Value::Keyword(k) if k == "auto" => None,
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
    let mut s = Style::default();
    s.display = match style.display {
        DisplayKind::Block | DisplayKind::Inline => Display::Block,
        DisplayKind::Flex => Display::Flex,
        DisplayKind::Grid => Display::Grid,
        DisplayKind::FlowRoot => Display::FlowRoot,
        DisplayKind::InlineBlock => Display::Block,
        DisplayKind::None => Display::None,
    };
    s.position = match style.position {
        PositionKind::Static => Position::Static,
        PositionKind::Relative => Position::Relative,
        // Taffy has no `fixed`: absolute is the closest layout
        // behaviour; the shell re-anchors fixed boxes to the viewport.
        PositionKind::Absolute | PositionKind::Fixed => Position::Absolute,
        // No sticky in taffy: relative is the closest behaviour.
        PositionKind::Sticky => Position::Relative,
    };
    s.box_sizing = match style.box_sizing {
        BoxSizingKind::ContentBox => BoxSizing::ContentBox,
        BoxSizingKind::BorderBox => BoxSizing::BorderBox,
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
    s
}

#[cfg(test)]
mod tests {
    use super::*;
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
        LayoutTree::build(&dom, &sheet, 1024.0, 768.0).expect("layout builds")
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
        resolve_document_styles(&dom, &ua, &sheet, false, viewport, &ElementState::default()).0
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        let texts: Vec<String> = tree
            .boxes
            .iter()
            .filter_map(|b| b.text.as_ref())
            .flat_map(|t| t.spans.iter().map(|s| s.text.clone()))
            .collect();
        assert!(!texts.iter().any(|t| t.contains("one")));
        assert!(texts.iter().any(|t| t.contains("two")));
    }

    #[test]
    fn flex_row_lays_children_side_by_side() {
        let dom = parse_html(
            r#"<div class="row"><div class="a">A</div><div class="b">B</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".row { display: flex; } .a { width: 100px; } .b { width: 100px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
            resolve_document_styles(dom, &ua, sheet, true, (1024.0, 768.0), &state).0;
        let serial =
            resolve_document_styles(dom, &ua, sheet, false, (1024.0, 768.0), &state).0;
        (parallel, serial)
    }

    #[test]
    fn parallel_and_serial_styles_match_example_com() {
        let dom = parse_html(EXAMPLE_COM);
        let sheet = stylesheet_from(&dom);

        // Levels are parent-before-children: every entry's parent
        // appears in an earlier level.
        let html = find_element(&dom, dom.root().unwrap(), "html").unwrap();
        let levels = collect_style_levels(&dom, &[html]);
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
    fn not_selector_and_visited_never_matches() {
        let html = r#"<p class="a">A</p><p class="b">B</p><a href="https://example.com">L</a>"#;
        let dom = parse_html(html);
        let styles = styles_for(
            html,
            "p:not(.b) { color: #ff0000; } a:visited { color: #00ff00; }",
            (800.0, 600.0),
        );
        let root = dom.root().unwrap();
        let a = find_by_class(&dom, root, "a").unwrap();
        let b = find_by_class(&dom, root, "b").unwrap();
        assert_eq!(styles[&a].color, [255, 0, 0]);
        assert_eq!(styles[&b].color, [0, 0, 0]);
        // `:visited` is always false in M2 (no history): the link
        // keeps the UA default blue, not the green rule.
        let link = find_element(&dom, dom.root().unwrap(), "a").unwrap();
        assert_eq!(styles[&link].color, [0, 0, 238]);
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        let root = dom.root().unwrap();
        let a = find_by_class(&dom, root, "a").unwrap();
        let bid = find_by_class(&dom, root, "b").unwrap();
        let abox = tree.boxes.iter().find(|b| b.dom == Some(a)).unwrap();
        let bbox = tree.boxes.iter().find(|b| b.dom == Some(bid)).unwrap();
        assert!((abox.width - 150.0).abs() < 0.5, "a width {}", abox.width);
        assert!((bbox.x - (abox.x + 150.0)).abs() < 0.5, "b x {}", bbox.x);
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
        let mut tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
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
}
