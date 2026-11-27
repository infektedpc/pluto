//! Stage 8 Recommendation 6 — dirty-subtree layout v1.
//!
//! The Double Dirty Bit flags Stage 7 populated (`Dom::dirty_set`)
//! say *which* nodes changed; this module decides *how much* layout
//! that change costs:
//!
//! 1. **Classify** the damage — paint-only (colour, background,
//!    radius, shadow), position-only (offsets on an out-of-flow
//!    box), layout-affecting, or reconstruct (structure changed) —
//!    from the existing `RestyleDamage` and, when that is coarse
//!    (Stage 7 populates it as `full()`), from a diff of the old and
//!    new computed styles.
//! 2. **Layout path, v1**: find the highest dirty node (the lowest
//!    common ancestor of all dirty nodes), map it to the nearest
//!    ancestor that owns a taffy node (the *effective root*), then
//!    walk up to the nearest ancestor that passes the safe-root
//!    predicate — the dirty node itself is often not a safe root
//!    (an auto-height `<p>` is not), while its fixed-size container
//!    usually is. Rebuild only that subtree's taffy nodes —
//!    everything else keeps its cached taffy layout. If no ancestor
//!    up to the document root is safe, fall back to a full rebuild,
//!    so v1 is correct by construction.
//! 3. **Position-only fast path**: an offset-only move on an
//!    absolutely/fixed-positioned box translates the subtree's boxes
//!    (and their glyphs and clips) without running taffy at all. The
//!    move is recorded in `LayoutTree::position_overrides` so a later
//!    re-collect (e.g. a scroll) reproduces it — taffy 0.15 has no
//!    layout setter, so the override map is the record of the move.
//! 4. **Debug invariant**: after every incremental result, debug
//!    builds lay the document out cold and assert the boxes are
//!    identical (Cephal's "incremental == cold, bit for bit"
//!    discipline). If the check ever fires, the fix is to narrow the
//!    safe-root predicate, never to patch the geometry.
//!
//! What v1 deliberately does NOT do: subtree-only style resolution
//! (styles are still resolved whole-document — the parallel pass is
//! cheap; the serial taffy pass is where incrementality pays);
//! compositing-only damage (there is no opacity/transform in
//! `ComputedStyle` yet, so it is currently unrepresentable — see
//! `DamageKind::CompositingOnly`); general Spineless traversal (out
//! of Stage 8 scope).
//!
//! The caller owns the dirty set: after a successful
//! `relayout_incremental`, call `Dom::clear_dirty()` (the tab
//! thread already does this after a re-layout — see
//! `js_host::clear_dirty`). The DOM passed in must be the live,
//! already-mutated document; node ids are slotmap-stable across
//! mutations, so the retained tree's `dom_to_taffy` map stays valid.

use super::*;
use crate::animation::{
    ANIMATABLE_PROPERTIES, AnimatedValue, animatable_pairs, animated_value_of_style,
    apply_animated_value,
};
use pluto_html::DirtySet;

/// What a restyle costs, in increasing severity. The order matters:
/// `max_damage` picks the worst of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageKind {
    /// Nothing changed (a dirty bit with no style difference — the
    /// dirt came from a descendant or from structure, so the
    /// aggregator in `relayout_incremental` promotes this to
    /// `Layout`; it is never a final answer on its own).
    None,
    /// Colour, background, border-radius, or box-shadow only: no
    /// geometry changes. The boxes' styles are refreshed and the
    /// boxes re-collected; taffy never runs.
    PaintOnly,
    /// Only `top`/`right`/`bottom`/`left` changed, on an
    /// absolutely- or fixed-positioned box: the subtree only moves.
    /// Handled by the position-only fast path.
    PositionOnly,
    /// A layout-affecting property changed: the dirty subtree is
    /// re-laid out (when its root is safe) or the whole tree is.
    Layout,
    /// Structure changed (a node was inserted, removed, replaced, or
    /// reparented, or an element entered/left `display: none`): the
    /// dirty subtree is rebuilt.
    Reconstruct,
    /// Transform / opacity only: no layout, no repaint — the shell
    /// updates a transform/opacity uniform. Currently
    /// unrepresentable: `ComputedStyle` models neither property yet
    /// (both are deliberately unmodelled in M2), so no classifier
    /// produces this today; it exists so the path is named when they
    /// land.
    CompositingOnly,
}

fn damage_rank(kind: DamageKind) -> u8 {
    match kind {
        DamageKind::None => 0,
        DamageKind::PaintOnly | DamageKind::CompositingOnly => 1,
        DamageKind::PositionOnly => 2,
        DamageKind::Layout => 3,
        DamageKind::Reconstruct => 4,
    }
}

fn max_damage(a: DamageKind, b: DamageKind) -> DamageKind {
    if damage_rank(a) >= damage_rank(b) {
        a
    } else {
        b
    }
}

/// Classify the damage Stage 7 recorded for one node (the
/// `RestyleDamage` shape). `reconstruct` wins over `reflow`, which
/// wins over `repaint`; all-false is `None`.
pub fn classify_damage(damage: RestyleDamage) -> DamageKind {
    if damage.reconstruct {
        DamageKind::Reconstruct
    } else if damage.reflow {
        DamageKind::Layout
    } else if damage.repaint {
        DamageKind::PaintOnly
    } else {
        DamageKind::None
    }
}

/// True when any layout-affecting property differs. `inset` is
/// deliberately excluded (handled separately: offset-only changes on
/// an out-of-flow box are `PositionOnly`), as are the paint-only
/// properties (`color`, `background`, `border_radius`,
/// `box_shadow`).
fn layout_properties_changed(old: &ComputedStyle, new: &ComputedStyle) -> bool {
    old.display != new.display
        || old.width != new.width
        || old.height != new.height
        || old.min_width != new.min_width
        || old.max_width != new.max_width
        || old.min_height != new.min_height
        || old.max_height != new.max_height
        || old.margin != new.margin
        || old.padding != new.padding
        || old.position != new.position
        || old.font_size != new.font_size
        || old.line_height != new.line_height
        || old.font_family != new.font_family
        || old.font_style != new.font_style
        || old.bold != new.bold
        || old.text_align != new.text_align
        || old.white_space != new.white_space
        || old.overflow_x != new.overflow_x
        || old.overflow_y != new.overflow_y
        || old.box_sizing != new.box_sizing
        || old.flex_direction != new.flex_direction
        || old.flex_wrap != new.flex_wrap
        || old.flex_grow != new.flex_grow
        || old.flex_shrink != new.flex_shrink
        || old.flex_basis != new.flex_basis
        || old.grid_template_columns != new.grid_template_columns
        || old.grid_template_rows != new.grid_template_rows
        || old.grid_gap != new.grid_gap
        || old.grid_column != new.grid_column
        || old.grid_row != new.grid_row
        || old.contain_layout != new.contain_layout
}

/// Classify a computed-style change geometrically, by diffing the
/// old and new styles. Stage 7 populates `RestyleDamage` as
/// `full()` everywhere, so this diff is what actually drives the
/// path selection in `relayout_incremental`.
///
/// A `color` change is `PaintOnly` here — geometrically it is — but
/// the relayout path routes it through the subtree rebuild anyway:
/// glyph colours are baked into the shaped leaves, so a colour
/// change needs new spans, not just a style refresh.
///
/// M6 Phase 2 (Rec 9): `z-index` changes reorder paint without
/// layout (`PaintOnly` — the display list is rebuilt, taffy never
/// runs), and transform/opacity-only changes are
/// `CompositingOnly` — no layout, no list rebuild, the shell
/// patches the retained list from a `CompositingUpdate`. But a
/// transform/opacity change that creates or destroys a stacking
/// context (opacity crossing 1.0, a transform appearing or
/// disappearing) reorders paint, so it is `PaintOnly`, not
/// `CompositingOnly`: the patch cannot move the element between
/// paint steps. `will-change` is a hint only and changes nothing
/// visual; `transition`/`animation` change future behaviour, not
/// the current frame — both classify `None` here.
pub fn classify_style_change(old: &ComputedStyle, new: &ComputedStyle) -> DamageKind {
    // Entering or leaving `display: none` adds or removes boxes.
    if old.display != new.display {
        if old.display == DisplayKind::None || new.display == DisplayKind::None {
            return DamageKind::Reconstruct;
        }
        return DamageKind::Layout;
    }
    if layout_properties_changed(old, new) {
        return DamageKind::Layout;
    }
    // Offsets alone, on an out-of-flow box, only move it. (A
    // relative offset is layout-affecting in taffy, so it falls
    // through to `Layout` below.)
    if old.inset != new.inset {
        if matches!(old.position, PositionKind::Absolute | PositionKind::Fixed) {
            return DamageKind::PositionOnly;
        }
        return DamageKind::Layout;
    }
    // A `z-index` change reorders paint without layout. (On a
    // static box the value is ignored — but the classifier does
    // not know flex/grid-item status, and a spurious list rebuild
    // is merely a repaint, never wrong; see the doc above.)
    if old.z_index != new.z_index {
        return DamageKind::PaintOnly;
    }
    // Transform / opacity only: the cheap path — unless the
    // change crosses a stacking-context boundary, which reorders
    // paint (opacity crossing 1.0, or a transform appearing or
    // disappearing), or a paint property changed in the same
    // restyle (a background repaint cannot ride a compositing
    // patch — the list must be rebuilt). `transform-origin`/
    // `transform-box` only change the resolved matrix, so they
    // ride along.
    if old.transform != new.transform
        || old.transform_origin != new.transform_origin
        || old.transform_box != new.transform_box
        || old.opacity != new.opacity
    {
        let crosses_context = (old.opacity < 1.0) != (new.opacity < 1.0)
            || old.transform.is_empty() != new.transform.is_empty();
        let paint_changed = old.color != new.color
            || old.background != new.background
            || old.border_radius != new.border_radius
            || old.box_shadow != new.box_shadow;
        if crosses_context || paint_changed {
            return DamageKind::PaintOnly;
        }
        return DamageKind::CompositingOnly;
    }
    if old.color != new.color
        || old.background != new.background
        || old.border_radius != new.border_radius
        || old.box_shadow != new.box_shadow
    {
        return DamageKind::PaintOnly;
    }
    DamageKind::None
}

// ---- safe-root predicate (Stage 8 Rec 6 §d.2) ----

fn is_positioned(position: PositionKind) -> bool {
    matches!(
        position,
        PositionKind::Relative
            | PositionKind::Absolute
            | PositionKind::Fixed
            | PositionKind::Sticky
    )
}

/// True when the subtree of `root` contains an absolutely- or
/// fixed-positioned descendant whose containing block is outside
/// the root (no positioned ancestor between it and the root, and
/// the root itself is not positioned). Such a descendant resolves
/// its size and offsets against that outside block, so re-laying
/// out the subtree in isolation could not place it. `display:
/// none` subtrees are skipped — they produce no boxes.
///
/// M6 Phase 3 (Rec 1): the old comment here said "`fixed`
/// descendants are fine: they resolve against the viewport" —
///
/// that was the false safe-root exemption the Stage 14 report
/// names. It is only true when no transformed ancestor sits
/// above the root: a transformed ancestor is the containing
/// block for `fixed` descendants, and the predicate cannot see
/// above the root, so a barrier-less `fixed` descendant is
/// unsafe — its containing block may be the viewport (constant
/// during the pass) or a transformed ancestor outside the root
/// (not constant). The conservative choice is to flag it like
/// `absolute`: the subtree falls back to a full rebuild. A
/// `fixed` descendant behind a positioned or transformed barrier
/// inside the root resolves inside the root and stays safe.
///
/// M6 Phase 1: a transformed ancestor is a containing block too
/// (CSS Transforms 1 — a non-`none` transform makes the element a
/// containing block for abs descendants), and `to_taffy_style`
/// maps transformed static boxes to `Position::Relative` so taffy
/// agrees. The predicate counts a transformed ancestor as a
/// barrier in the same step, so the two can never disagree (the
/// report §4 sharpest trap: widening one without the other makes
/// `incremental == cold` pass while both are wrong vs CSS).
fn has_outside_resolving_descendant(
    dom: &Dom,
    styles: &HashMap<DomId, Arc<ComputedStyle>>,
    root: DomId,
    root_positioned: bool,
) -> bool {
    fn walk(
        dom: &Dom,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        id: DomId,
        positioned_above: bool,
        root_positioned: bool,
    ) -> bool {
        for &child in dom.children(id) {
            // Text / comment nodes carry no style.
            let Some(style) = styles.get(&child) else {
                continue;
            };
            if style.display == DisplayKind::None {
                continue;
            }
            if matches!(style.position, PositionKind::Absolute | PositionKind::Fixed)
                && !root_positioned
                && !positioned_above
            {
                return true;
            }
            if walk(
                dom,
                styles,
                child,
                positioned_above
                    || is_positioned(style.position)
                    || !style.transform.is_empty(),
                root_positioned,
            ) {
                return true;
            }
        }
        false
    }
    walk(dom, styles, root, false, root_positioned)
}

/// The safe-root predicate: `None` when `root` is a valid
/// incremental layout root, `Some(reason)` when it is not. The
/// checks are the report's, in order:
///
/// 1. Size independence — a definite width, `contain: layout`, a
///    replaced element, or a fixed-size leaf — so the root's size
///    cannot change because of what changed inside it. (Stage 12
///    Rec 4 stage 1: the width is the requirement, not the height.
///    A definite px width cannot change with the contents, and a
///    height change propagates correctly: taffy `mark_dirty` is
///    recursive up the ancestor chain (taffy 0.15.1
///    `TaffyTree::mark_dirty`), so the parent and every ancestor
///    re-run their layout and reposition siblings from the new
///    size; siblings whose available space changed miss taffy's
///    cache, which is keyed on available space, and are
///    re-measured; `recollect()` rebuilds every clip from the
///    fresh boxes. The debug incremental==cold assertion is the
///    tripwire.)
/// 2. Margin collapse with outside — no predicate guard (Stage 12
///    Rec 4 stage 2). Taffy computes margin collapse during
///    layout, and the dirty ancestors re-run their layout on a
///    subtree rebuild, so the collapse is re-evaluated for every
///    partner: the parent, a preceding sibling, and collapse
///    through the parent to the grandparent. The old guard (BFC
///    root / out-of-flow / padding) was conservative; the debug
///    incremental==cold assertion is the tripwire if the
///    recomputation ever diverges.
/// 3. No floats intruding outside — vacuous in Pluto today (`float`
///    / `clear` are deliberately unmodelled), and a BFC root would
///    contain floats anyway; the check becomes real when floats land.
/// 4. No percentage descendants resolving outside the root (see
///    `has_outside_resolving_descendant`).
///
/// Anything unsafe falls back to full layout — v1 only accelerates
/// the cases it can prove safe.
pub fn unsafe_layout_root_reason(
    dom: &Dom,
    styles: &HashMap<DomId, Arc<ComputedStyle>>,
    root: DomId,
) -> Option<&'static str> {
    let Some(style) = styles.get(&root) else {
        return Some("root has no computed style");
    };
    // The root must own a taffy node: inline / `display: none`
    // boxes never reach the predicate (the effective-root search
    // excludes them), but the predicate stays total for direct
    // callers.
    if matches!(style.display, DisplayKind::Inline | DisplayKind::None) {
        return Some("root is not a block-level box (inline or display:none)");
    }
    // (1) Size independence (Stage 12 Rec 4 stage 1: definite
    // width — the height may be auto, a percentage, or viewport
    // relative; all of those are functions of the parent or the
    // viewport, not of the subtree's contents, and a height change
    // propagates through taffy's dirty ancestors).
    let definite_width = matches!(style.width, Some(Dim::Px(_)));
    let definite_size =
        definite_width && matches!(style.height, Some(Dim::Px(_)));
    // Stage 15 (M7 Phase 1, Rec 2): `<svg>` / `<canvas>` are
    // replaced-element leaves like `<img>` — their size does not
    // come from their subtree (which builds no boxes at all), so a
    // subtree rooted at one is size-independent. The predicate and
    // the cold build change together here (Stage 14 trap class):
    // cold used to drop SVG children entirely, so the
    // incremental==cold tripwire could not have caught a predicate
    // that treated them as ordinary containers.
    let replaced = matches!(dom.tag(root), Some("img"))
        || is_svg_element(dom, root)
        || is_canvas_element(dom, root);
    // A leaf with a definite size: the fourth alternative, listed
    // separately for the report even though it is subsumed by
    // `definite_size` (a leaf has no contents to size from).
    let fixed_size_leaf = definite_size && dom.children(root).iter().all(|&c| dom.tag(c).is_none());
    if !definite_width && !style.contain_layout && !replaced && !fixed_size_leaf {
        return Some(
            "root has no definite width, no `contain: layout`, is not a replaced element, \
             and is not a fixed-size leaf",
        );
    }
    // (2) Margin collapse with outside — no guard (Stage 12 Rec 4
    // stage 2): taffy recomputes the collapse for every dirty
    // ancestor on a subtree rebuild; see the doc comment above.
    // (3) Floats: vacuous, see the doc comment above.
    // (4) No percentage descendants resolving outside the root.
    // M6 Phase 1: a transformed root is a containing block for
    // its abs descendants too (same barrier as a positioned root).
    if has_outside_resolving_descendant(
        dom,
        styles,
        root,
        is_positioned(style.position) || !style.transform.is_empty(),
    ) {
        return Some(
            "subtree has an absolutely-positioned descendant whose containing block \
             is outside the root",
        );
    }
    None
}

/// `true` when `root` is a valid incremental layout root.
pub fn is_safe_layout_root(
    dom: &Dom,
    styles: &HashMap<DomId, Arc<ComputedStyle>>,
    root: DomId,
) -> bool {
    unsafe_layout_root_reason(dom, styles, root).is_none()
}

// ---- relayout ----

/// Which path `relayout_incremental` took — the widening metric
/// (Stage 12 Rec 3): every submitted frame records its kind in the
/// frame-time histogram, so the fallback rate (FullLayout /
/// (SubtreeLayout + FullLayout)) is visible in the periodic stderr
/// summary instead of guessed at. `ColdBuild` is a fresh tree for a
/// new document/viewport — not a fallback, so it counts in neither
/// side of the fallback rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RelayoutKind {
    /// Nothing was dirty, or the dirty subtree has no boxes.
    #[default]
    NothingToDo,
    /// Only paint properties changed: box styles refreshed, boxes
    /// re-collected, taffy never ran.
    PaintOnly,
    /// Only out-of-flow offsets changed: boxes translated, taffy
    /// never ran.
    PositionOnly,
    /// Only transform / opacity changed, without crossing a
    /// stacking-context boundary (M6 Phase 2, Rec 9): box styles
    /// refreshed and boxes re-collected (so the retained matrices
    /// are current), taffy never ran, and NO display list is
    /// rebuilt — the tab thread sends a `CompositingUpdate` with
    /// just the changed elements' state instead.
    CompositingOnly,
    /// The dirty subtree was rebuilt and re-laid out; everything
    /// else kept its cached taffy layout.
    SubtreeLayout,
    /// The safe-root predicate (or the shape of the change) refused
    /// the incremental path; the tree was rebuilt cold.
    FullLayout,
    /// A fresh tree was built for a new document/viewport (no
    /// retained tree to be incremental against) — not a fallback.
    ColdBuild,
}

impl RelayoutKind {
    /// Number of variants — the histogram's per-kind counter array
    /// is sized by this, so the two cannot drift apart silently.
    pub const COUNT: usize = 7;

    /// Index into the histogram's per-kind counters, in
    /// declaration order.
    pub fn as_index(self) -> usize {
        match self {
            RelayoutKind::NothingToDo => 0,
            RelayoutKind::PaintOnly => 1,
            RelayoutKind::PositionOnly => 2,
            RelayoutKind::CompositingOnly => 3,
            RelayoutKind::SubtreeLayout => 4,
            RelayoutKind::FullLayout => 5,
            RelayoutKind::ColdBuild => 6,
        }
    }
}

/// The outcome of `relayout_incremental`: which path ran, and why.
#[derive(Debug, Clone)]
pub struct RelayoutOutcome {
    pub kind: RelayoutKind,
    /// Why this path was chosen (static diagnostic string).
    pub reason: &'static str,
    /// The dirty elements whose transform/opacity changed, for a
    /// `CompositingOnly` outcome (M6 Phase 2, Rec 9): the tab
    /// thread asks the actor for their new compositing state and
    /// sends it as a `CompositingUpdate`. Empty for every other
    /// kind.
    pub composited: Vec<DomId>,
    /// Transitions that start on this layout (M6 Phase 3, Rec
    /// 11): the tab thread feeds each one to its animation
    /// engine with the `from` value the element had before this
    /// pass. Empty when no transitioned property changed.
    pub pending_transitions: Vec<PendingTransition>,
    /// Keyframe animations that start (or restart) on this
    /// layout: the tab thread feeds each one to its engine with
    /// the resolved `@keyframes` rule. Empty when no `animation`
    /// entry is new or changed.
    pub pending_animations: Vec<PendingAnimation>,
    /// The animatable properties of the dirty elements after this
    /// pass, as `(element, [(property, css text)])` — the tab
    /// thread merges these into its `getComputedStyle` cache, so
    /// a restyle is visible to JS even while no animation runs.
    pub changed_styles: Vec<(DomId, Vec<(String, String)>)>,
}

/// One transition the actor detected on this layout: `spec`'s
/// property changed from `from` to `to` (underlying values — the
/// tab thread substitutes the currently sampled value for `from`
/// when a transition on the same property is already running).
#[derive(Debug, Clone)]
pub struct PendingTransition {
    pub id: DomId,
    pub spec: Transition,
    pub from: AnimatedValue,
    pub to: AnimatedValue,
}

/// One keyframe animation the actor detected on this layout: the
/// `animation` entry plus the `@keyframes` rule it names
/// (resolved from the sheet by the actor, so the tab thread never
/// has to re-parse the sheet).
#[derive(Debug, Clone)]
pub struct PendingAnimation {
    pub id: DomId,
    pub spec: Animation,
    pub keyframes: Keyframes,
    /// The underlying (non-animated) values of the animated
    /// properties, for the fill-`none` return — the tab thread
    /// feeds them to the engine as the animation's base.
    pub base: Vec<(String, AnimatedValue)>,
}

impl RelayoutOutcome {
    fn of(kind: RelayoutKind, reason: &'static str) -> Self {
        RelayoutOutcome {
            kind,
            reason,
            composited: Vec::new(),
            pending_transitions: Vec::new(),
            pending_animations: Vec::new(),
            changed_styles: Vec::new(),
        }
    }

    fn with_composited(
        kind: RelayoutKind,
        reason: &'static str,
        composited: Vec<DomId>,
    ) -> Self {
        RelayoutOutcome {
            kind,
            reason,
            composited,
            pending_transitions: Vec::new(),
            pending_animations: Vec::new(),
            changed_styles: Vec::new(),
        }
    }
}

/// What the dirty root maps to in the retained tree.
enum EffectiveRoot {
    /// A taffy-owning node to rebuild (never the document root).
    Node(DomId),
    /// The dirty subtree contains no rendered node at all (e.g. a
    /// `<title>` change inside `<head>`): nothing to lay out.
    NotRendered,
    /// The dirty root is the document root itself: only a full
    /// rebuild can cover it.
    DocumentRoot,
}

/// Map a dirty node to the element it dirties for layout: text (and
/// comment) nodes dirty their parent element; elements dirty
/// themselves.
fn layout_dirty_element(dom: &Dom, id: DomId) -> Option<DomId> {
    if dom.tag(id).is_some() {
        return Some(id);
    }
    let parent = dom.get(id)?.parent?;
    if dom.tag(parent).is_some() {
        Some(parent)
    } else {
        None
    }
}

/// Every dirty layout/style node, mapped to elements, deduplicated,
/// in dirty-set order. Detached nodes are skipped: a removal marks
/// the removed child dirty (`note_mutation`) and `detach` clears its
/// parent link, so the child has no ancestor chain left — the
/// parent, which the caller marks dirty, is the node that matters.
/// (If *no* dirty node is attached, the caller falls back to a full
/// rebuild.) `extra` carries nodes damaged by other means — an
/// image arrival's changed `<img>` nodes (Stage 12 Rec 5) — which
/// take the same mapping and filtering.
fn dirty_elements(dom: &Dom, dirty: &DirtySet, extra: &[DomId]) -> Vec<DomId> {
    let mut out: Vec<DomId> = Vec::new();
    for &id in dirty
        .layout
        .iter()
        .chain(dirty.style.iter())
        .chain(extra.iter())
    {
        if let Some(el) = layout_dirty_element(dom, id) {
            if !is_attached(dom, el) {
                continue;
            }
            if !out.contains(&el) {
                out.push(el);
            }
        }
    }
    out
}

/// True when `id`'s ancestor chain reaches the document root —
/// i.e. the node is still in the document, not a detached subtree.
fn is_attached(dom: &Dom, id: DomId) -> bool {
    let root = dom.root();
    let mut cur = Some(id);
    while let Some(n) = cur {
        if Some(n) == root {
            return true;
        }
        cur = dom.get(n).and_then(|x| x.parent);
    }
    false
}

/// True when `node` is `ancestor` or a descendant of it.
fn is_descendant_or_self(dom: &Dom, node: DomId, ancestor: DomId) -> bool {
    let mut cur = Some(node);
    while let Some(id) = cur {
        if id == ancestor {
            return true;
        }
        cur = dom.get(id).and_then(|n| n.parent);
    }
    false
}

fn lowest_common_ancestor(dom: &Dom, a: DomId, b: DomId) -> Option<DomId> {
    let mut ancestors: HashSet<DomId> = HashSet::new();
    let mut cur = Some(a);
    while let Some(id) = cur {
        ancestors.insert(id);
        cur = dom.get(id).and_then(|n| n.parent);
    }
    let mut cur = Some(b);
    while let Some(id) = cur {
        if ancestors.contains(&id) {
            return Some(id);
        }
        cur = dom.get(id).and_then(|n| n.parent);
    }
    None
}

/// The lowest common ancestor of all dirty elements — the highest
/// node whose subtree covers every dirty node. `None` when no dirty
/// node maps to an element in the document (all stale). `extra`
/// carries nodes damaged by other means (Stage 12 Rec 5: an image
/// arrival's changed `<img>` nodes).
fn lowest_common_dirty_root(dom: &Dom, dirty: &DirtySet, extra: &[DomId]) -> Option<DomId> {
    let ids = dirty_elements(dom, dirty, extra);
    let mut it = ids.into_iter();
    let mut acc = it.next()?;
    for id in it {
        acc = lowest_common_ancestor(dom, acc, id)?;
    }
    Some(acc)
}

fn root_font_size_of(dom: &Dom, styles: &HashMap<DomId, Arc<ComputedStyle>>) -> f32 {
    let Some(root) = dom.root() else {
        return 16.0;
    };
    find_element(dom, root, "html")
        .and_then(|r| styles.get(&r))
        .map(|s| s.font_size)
        .unwrap_or(16.0)
}

/// One axis of a position-only move: the delta when the start offset
/// (`left` / `top`) is an explicit pixel value on both sides, else
/// the delta from the end offset (`right` / `bottom`, which moves
/// the box the other way), else `0.0` when both ends are unchanged,
/// else `None` (the move is not computable from the style diff —
/// e.g. `auto` became a length — and the caller falls back to the
/// subtree path).
fn axis_delta(
    old_start: Option<Dim>,
    new_start: Option<Dim>,
    old_end: Option<Dim>,
    new_end: Option<Dim>,
) -> Option<f32> {
    match (old_start, new_start) {
        (Some(Dim::Px(a)), Some(Dim::Px(b))) => Some(b - a),
        _ => match (old_end, new_end) {
            (Some(Dim::Px(a)), Some(Dim::Px(b))) => Some(a - b),
            _ => {
                if old_start == new_start && old_end == new_end {
                    Some(0.0)
                } else {
                    None
                }
            }
        },
    }
}

impl LayoutTree {
    /// Re-lay out `dom` against this tree's retained taffy tree,
    /// using the dirty set Stage 7 populated on the DOM. `dom` is the
    /// live, already-mutated document; `sheet` is the author sheet;
    /// `state` carries the dynamic pseudo-class state for the pass.
    /// `images` is the fresh image map for this pass (`None` when
    /// no image arrived): it replaces the retained map up front,
    /// and the `<img>` nodes whose entries changed are damage too
    /// (Stage 12 Rec 5) — an image landing inside a safe subtree
    /// re-lays out that subtree, not the whole document.
    ///
    /// The path taken is reported in the outcome: paint-only changes
    /// refresh box styles without running taffy; offset-only moves
    /// on an out-of-flow box translate boxes without running taffy;
    /// layout-affecting changes rebuild only the dirty subtree when
    /// its root passes the safe-root predicate; anything else —
    /// an unsafe root, the document root itself, a viewport change —
    /// falls back to a full rebuild. In debug builds every
    /// incremental result is asserted equal to a cold full layout.
    ///
    /// The caller owns the dirty set: call `Dom::clear_dirty()`
    /// after a successful relayout.
    pub fn relayout_incremental(
        &mut self,
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        images: Option<&ImageMap>,
    ) -> Result<RelayoutOutcome, LayoutError> {
        // Stage 12 Rec 5: the fresh image map replaces the retained
        // one BEFORE any rebuild path runs, so a full rebuild (or
        // the debug cold twin) builds with the new images, not the
        // stale ones. The changed <img> nodes are extra damage
        // (see `ImageMap::changed_node_ids`); a changed background
        // cannot name its nodes, so it forces a full rebuild.
        let image_changed: Vec<DomId> = match images {
            Some(map) => {
                let changed = map.changed_node_ids(&self.images);
                let backgrounds_changed = map.backgrounds_changed(&self.images);
                self.images = map.clone();
                if backgrounds_changed {
                    return self.full_rebuild(
                        dom,
                        sheet,
                        viewport_width,
                        viewport_height,
                        state,
                        "background image arrived",
                    );
                }
                changed
            }
            None => Vec::new(),
        };
        // A viewport change moves the root's available space, so
        // every cached layout is suspect; likewise with no retained
        // tree at all.
        if self.root.is_none() || self.viewport != (viewport_width, viewport_height) {
            return self.full_rebuild(
                dom,
                sheet,
                viewport_width,
                viewport_height,
                state,
                "no retained layout tree, or the viewport changed",
            );
        }
        let dirty = dom.dirty_set();
        if dirty.is_empty() && image_changed.is_empty() {
            return Ok(RelayoutOutcome::of(
                RelayoutKind::NothingToDo,
                "dirty set is empty",
            ));
        }
        // The dirty LCA comes first: the restyle roots at it
        // (Stage 12 Rec 6), so only the dirty subtree pays the
        // cascade.
        let Some(dirty_root) = lowest_common_dirty_root(dom, dirty, &image_changed) else {
            return self.full_rebuild(
                dom,
                sheet,
                viewport_width,
                viewport_height,
                state,
                "dirty nodes are no longer in the document",
            );
        };
        // Stage 12 Rec 6: restyle only the dirty subtree — the
        // LCA's subtree, seeded from the retained styles above it.
        // The returned maps are the full merged ones, so the damage
        // diff below and the subtree rebuild read them exactly as
        // they read the whole-document maps before.
        let ua = Stylesheet::ua();
        let (mut styles, node_styles) = resolve_subtree_styles(
            dom,
            &ua,
            sheet,
            true,
            (viewport_width, viewport_height),
            state,
            dirty_root,
            &self.node_styles,
            self.theme,
        );
        // M6 Phase 3 (Rec 11): in-flight sampled values ride over
        // the resolved (underlying) styles before classification,
        // so an animated transform/opacity takes the
        // CompositingOnly path and an animated width re-lays out
        // like any other style change. Only `styles` (the
        // classification map) is overridden — `node_styles` stays
        // underlying, so the next pass still diffs against the
        // real cascade, not last frame's sample.
        for (id, pairs) in &self.animation_overrides {
            if let Some(old) = styles.get(id) {
                let mut new_style = (**old).clone();
                for (prop, val) in pairs {
                    apply_animated_value(&mut new_style, prop, val);
                }
                styles.insert(*id, Arc::new(new_style));
            }
        }
        // Classify the change: this tree's `node_styles` are the old
        // styles, the freshly resolved map the new ones.
        let dirty_els = dirty_elements(dom, dirty, &image_changed);
        let mut aggregate = DamageKind::None;
        let mut color_changed = false;
        // M6 Phase 2 (Rec 9): the elements that classified
        // `CompositingOnly`, and whether any element classified
        // `PaintOnly`. `max_damage` ties `PaintOnly` and
        // `CompositingOnly` at rank 1, so the aggregate alone
        // cannot prove a batch is purely compositing — a `PaintOnly`
        // element in the batch needs the list rebuilt, and the
        // cheap path must not swallow its repaint.
        let mut composited_ids: Vec<DomId> = Vec::new();
        let mut any_paint_only = false;
        for &id in &dirty_els {
            match (self.node_styles.get(id), styles.get(&id)) {
                (Some(old), Some(new)) => {
                    if old.computed.color != new.color {
                        color_changed = true;
                    }
                    let kind = classify_style_change(&old.computed, new);
                    if kind == DamageKind::CompositingOnly {
                        composited_ids.push(id);
                    }
                    if kind == DamageKind::PaintOnly {
                        any_paint_only = true;
                    }
                    // A dirty node whose own style did not change was
                    // dirtied by a descendant or by structure: it
                    // needs at least a subtree re-layout.
                    let kind = if kind == DamageKind::None {
                        DamageKind::Layout
                    } else {
                        kind
                    };
                    aggregate = max_damage(aggregate, kind);
                }
                // Inserted (no old style) or removed (no new style):
                // structure changed.
                _ => aggregate = DamageKind::Reconstruct,
            }
        }
        // M6 Phase 3 (Rec 11): transitions / animations that start
        // on this pass, plus the animatable values for the
        // `getComputedStyle` cache. Computed from the old
        // (retained) and new (resolved, override-applied) styles.
        let mut pending_transitions: Vec<PendingTransition> = Vec::new();
        let mut pending_animations: Vec<PendingAnimation> = Vec::new();
        let mut changed_styles: Vec<(DomId, Vec<(String, String)>)> = Vec::new();
        for &id in &dirty_els {
            let (Some(old), Some(new)) = (self.node_styles.get(id), styles.get(&id)) else {
                continue;
            };
            let old = &old.computed;
            let new = &**new;
            let box_width = self
                .boxes
                .iter()
                .find(|b| b.dom == Some(id))
                .map(|b| b.width);
            let ctx = CalcCtx {
                font_size: new.font_size,
                root_font_size: self.root_font_size,
                viewport: (viewport_width, viewport_height),
                pct_basis: box_width,
            };
            for spec in &new.transitions {
                if spec.duration <= 0.0 {
                    continue;
                }
                let props: Vec<&str> = if spec.property == "all" {
                    ANIMATABLE_PROPERTIES.to_vec()
                } else {
                    vec![spec.property.as_str()]
                };
                for prop in props {
                    let (Some(from), Some(to)) = (
                        animated_value_of_style(old, prop, &ctx),
                        animated_value_of_style(new, prop, &ctx),
                    ) else {
                        continue;
                    };
                    if from != to {
                        pending_transitions.push(PendingTransition {
                            id,
                            spec: spec.clone(),
                            from,
                            to,
                        });
                    }
                }
            }
            for spec in &new.animations {
                if spec.name == "none" {
                    continue;
                }
                let already = old
                    .animations
                    .iter()
                    .any(|o| o.name == spec.name && o == spec);
                if already {
                    continue;
                }
                if let Some(kf) = sheet.keyframes.iter().find(|k| k.name == spec.name) {
                    let base = ANIMATABLE_PROPERTIES
                        .iter()
                        .filter_map(|p| {
                            animated_value_of_style(new, p, &ctx)
                                .map(|v| (p.to_string(), v))
                        })
                        .collect();
                    pending_animations.push(PendingAnimation {
                        id,
                        spec: spec.clone(),
                        keyframes: kf.clone(),
                        base,
                    });
                }
            }
            let pairs = animatable_pairs(new, &ctx);
            if !pairs.is_empty() {
                changed_styles.push((id, pairs));
            }
        }
        // Every outcome below carries the pending work — a full
        // rebuild still restyles, so its transitions start too.
        let attach = |mut o: RelayoutOutcome| -> RelayoutOutcome {
            o.pending_transitions = pending_transitions.clone();
            o.pending_animations = pending_animations.clone();
            o.changed_styles = changed_styles.clone();
            o
        };
        match self.effective_layout_root(dom, dirty_root) {
            EffectiveRoot::NotRendered => {
                self.node_styles = node_styles;
                Ok(attach(RelayoutOutcome::of(
                    RelayoutKind::NothingToDo,
                    "dirty subtree has no boxes (display:none / head)",
                )))
            }
            EffectiveRoot::DocumentRoot => self
                .full_rebuild(
                    dom,
                    sheet,
                    viewport_width,
                    viewport_height,
                    state,
                    "dirty root is the document root",
                )
                .map(attach),
            EffectiveRoot::Node(eff) => {
                // A display change can move the effective root: an
                // element that became inline (or `display: none`) no
                // longer owns a taffy node, so its owning block
                // container is the root to rebuild instead.
                let Some(eff) = self.adjust_effective_root(dom, &styles, eff) else {
                    return self
                        .full_rebuild(
                            dom,
                            sheet,
                            viewport_width,
                            viewport_height,
                            state,
                            "display change moved the layout root out of the tree",
                        )
                        .map(attach);
                };
                if self.dom_to_taffy.get(&eff).copied() == self.root {
                    return self
                        .full_rebuild(
                            dom,
                            sheet,
                            viewport_width,
                            viewport_height,
                            state,
                            "dirty root is the document root",
                        )
                        .map(attach);
                }
                // The cheap paths need proof that nothing structural
                // or textual changed — only style values did. The
                // gate is conservative: any doubt takes the subtree
                // path, which is always correct.
                let structure_unchanged = self.subtree_structure_unchanged(dom, &styles, eff);
                // Compositing-only (M6 Phase 2, Rec 9): every dirty
                // element changed only transform/opacity, without
                // crossing a stacking-context boundary. Refresh box
                // styles and re-collect — the retained boxes carry
                // the new matrices, so the tab thread can read the
                // changed elements' state and send it as a
                // `CompositingUpdate` — taffy never runs and NO
                // display list is rebuilt. The `any_paint_only`
                // guard matters: a `PaintOnly` element in the same
                // batch needs the list rebuilt, and this path must
                // not swallow its repaint.
                if aggregate == DamageKind::CompositingOnly
                    && !any_paint_only
                    && !composited_ids.is_empty()
                    && structure_unchanged
                {
                    self.refresh_meta_styles(&styles);
                    self.node_styles = node_styles;
                    self.recollect();
                    self.debug_assert_matches_cold(
                        dom,
                        sheet,
                        (viewport_width, viewport_height),
                        state,
                    );
                    return Ok(attach(RelayoutOutcome::with_composited(
                        RelayoutKind::CompositingOnly,
                        "only transform/opacity changed; no layout, no list rebuild",
                        composited_ids,
                    )));
                }
                // Paint-only, with no text-colour change (glyph
                // colours are baked into shaped leaves, so a colour
                // change needs the subtree path): refresh box styles
                // and re-collect; taffy never runs.
                if aggregate == DamageKind::PaintOnly && !color_changed && structure_unchanged {
                    self.refresh_meta_styles(&styles);
                    self.node_styles = node_styles;
                    self.recollect();
                    self.refresh_canvas_background(dom);
                    self.debug_assert_matches_cold(
                        dom,
                        sheet,
                        (viewport_width, viewport_height),
                        state,
                    );
                    return Ok(attach(RelayoutOutcome::of(
                        RelayoutKind::PaintOnly,
                        "only paint properties changed; no layout",
                    )));
                }
                // Position-only: exactly one dirty element, an
                // out-of-flow box whose offsets moved by a computable
                // pixel delta, and no structural or text change.
                if aggregate == DamageKind::PositionOnly
                    && structure_unchanged
                    && dirty_els.len() == 1
                    && let Some((dx, dy)) = self.position_only_delta(dirty_els[0], &styles)
                {
                    let id = dirty_els[0];
                    // Keep this node's meta style in sync (its
                    // inset changed); the taffy style is left
                    // stale on purpose — the override map is the
                    // record of the move.
                    if let Some(&t) = self.dom_to_taffy.get(&id)
                        && let Some(m) = self.meta.get_mut(&t)
                        && let Some(new) = styles.get(&id)
                    {
                        m.style = (**new).clone();
                    }
                    if self.translate_subtree(id, dx, dy) {
                        self.node_styles = node_styles;
                        self.debug_assert_matches_cold(
                            dom,
                            sheet,
                            (viewport_width, viewport_height),
                            state,
                        );
                        return Ok(attach(RelayoutOutcome::of(
                            RelayoutKind::PositionOnly,
                            "only out-of-flow offsets changed; boxes translated",
                        )));
                    }
                    // `translate_subtree` said no (the node owns
                    // no taffy node — e.g. an absolutely
                    // positioned inline element): fall through to
                    // the subtree path below.
                }
                // Layout / reconstruct: rebuild from the nearest
                // safe ancestor of the effective root when there is
                // one, else fall back to a full rebuild.
                match self.nearest_safe_ancestor(dom, &styles, eff) {
                    Some(safe) => self
                        .relayout_subtree(
                            dom,
                            sheet,
                            &styles,
                            node_styles,
                            safe,
                            (viewport_width, viewport_height),
                            state,
                        )
                        .map(attach),
                    None => {
                        let reason = unsafe_layout_root_reason(dom, &styles, eff)
                            .unwrap_or("no safe layout root above the dirty node");
                        self.full_rebuild(
                            dom,
                            sheet,
                            viewport_width,
                            viewport_height,
                            state,
                            reason,
                        )
                        .map(attach)
                    }
                }
            }
        }
    }

    /// The position-only fast path (Stage 8 Rec 6 §d.3): translate
    /// the subtree rooted at `id` by `(dx, dy)` without running
    /// taffy at all. Box rects, glyph positions, and clips move; a
    /// clip bounded by an outer container stays bounded (clips are
    /// recomputed from the translated ancestors, not blindly
    /// shifted). The move is recorded in `position_overrides` so a
    /// later re-collect reproduces it.
    ///
    /// Contract: the caller guarantees the move is position-only —
    /// sizes, styles, and structure are unchanged, and the subtree
    /// contains no scroll-container offset change. The next layout
    /// pass that rebuilds this subtree from the DOM drops the
    /// override (the fresh taffy layout already carries the new
    /// position). Returns `false` when `id` owns no taffy node.
    pub fn translate_subtree(&mut self, id: DomId, dx: f32, dy: f32) -> bool {
        let Some(&node) = self.dom_to_taffy.get(&id) else {
            return false;
        };
        if dx == 0.0 && dy == 0.0 {
            return true;
        }
        // Record the move so a later re-collect reproduces it (taffy
        // has no layout setter — the override map is the record).
        let entry = self.position_overrides.entry(node).or_insert((0.0, 0.0));
        entry.0 += dx;
        entry.1 += dy;
        // Translate the live boxes in place.
        let subtree: HashSet<TaffyId> = self.taffy_subtree_ids(node).into_iter().collect();
        for bx in self.boxes.iter_mut() {
            if subtree.contains(&bx.taffy) {
                bx.x += dx;
                bx.y += dy;
                for g in bx.glyphs.iter_mut() {
                    g.x += dx;
                    g.y += dy;
                }
            }
        }
        self.recompute_clips_for(&subtree);
        true
    }

    /// Walk up from `eff` (inclusive) to the nearest ancestor that
    /// owns a taffy node and passes the safe-root predicate — the
    /// root a subtree rebuild starts from. The dirty node itself is
    /// often not a safe root (an auto-height `<p>` is not); its
    /// fixed-size container usually is. `None` when the walk reaches
    /// the document root without finding one.
    fn nearest_safe_ancestor(
        &self,
        dom: &Dom,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        eff: DomId,
    ) -> Option<DomId> {
        let mut cur = Some(eff);
        while let Some(id) = cur {
            if let Some(&taffy_node) = self.dom_to_taffy.get(&id) {
                if Some(taffy_node) == self.root {
                    return None;
                }
                if unsafe_layout_root_reason(dom, styles, id).is_none() {
                    return Some(id);
                }
            }
            cur = dom.get(id).and_then(|n| n.parent);
        }
        None
    }

    /// The nearest ancestor-or-self of `dirty_root` that owns a
    /// taffy node in the retained tree — the node a subtree rebuild
    /// would start from. Text and inline elements own no taffy node,
    /// so the root is the block container that owns their leaf.
    fn effective_layout_root(&self, dom: &Dom, dirty_root: DomId) -> EffectiveRoot {
        // Walk up to the nearest taffy-owning ancestor: an inline
        // element owns no taffy node, but its text lives in the
        // nearest block ancestor's node, so that ancestor is the
        // rebuild root. (The old gate checked only whether the
        // dirty SUBTREE owned a taffy node, so a text change inside
        // an inline span — `<span>9</span>` becoming `10` — was
        // dropped as "not rendered" and the new text never
        // appeared. The span's subtree has no taffy node, but its
        // block ancestor does.)
        let mut cur = Some(dirty_root);
        while let Some(id) = cur {
            if let Some(&taffy_node) = self.dom_to_taffy.get(&id) {
                if Some(taffy_node) == self.root {
                    // The root owns the layout: a full rebuild —
                    // unless the dirty subtree contains no rendered
                    // node at all (e.g. a `<title>` change inside
                    // `<head>`), in which case there is nothing to
                    // lay out.
                    if self.subtree_has_taffy_node(dom, dirty_root) {
                        return EffectiveRoot::DocumentRoot;
                    }
                    return EffectiveRoot::NotRendered;
                }
                return EffectiveRoot::Node(id);
            }
            cur = dom.get(id).and_then(|n| n.parent);
        }
        // No taffy-owning ancestor at all: the dirty node is inside
        // an unrendered subtree (or detached — the caller filters
        // those). NOTE: a change inside a `display: none` subtree
        // used to return `NotRendered` here via the subtree gate;
        // it now rebuilds the nearest block ancestor instead — the
        // result is identical (the subtree contributes no leaves),
        // just one subtree rebuild slower. Correctness first; the
        // fast path can come back with a style-aware check.
        EffectiveRoot::NotRendered
    }

    /// Walk `eff` up to the nearest ancestor that still owns a taffy
    /// node under the *new* styles: an element whose display became
    /// `inline` or `none` no longer owns one, so its owning block
    /// container is the rebuild root instead. `None` when the walk
    /// leaves the tree (the caller falls back to a full rebuild).
    fn adjust_effective_root(
        &self,
        dom: &Dom,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        eff: DomId,
    ) -> Option<DomId> {
        let mut cur = Some(eff);
        while let Some(id) = cur {
            match styles.get(&id).map(|s| s.display) {
                Some(DisplayKind::Inline) | Some(DisplayKind::None) => {
                    // No taffy node under the new styles: walk up.
                }
                Some(_) if self.dom_to_taffy.contains_key(&id) => {
                    return Some(id);
                }
                Some(_) => {
                    // New block-level node with no retained taffy
                    // node (it was inline before): its parent must
                    // be rebuilt instead.
                }
                None => {
                    // Not an element in the new styles: walk up
                    // defensively.
                }
            }
            cur = dom.get(id).and_then(|n| n.parent);
        }
        None
    }

    fn subtree_has_taffy_node(&self, dom: &Dom, root: DomId) -> bool {
        if self.dom_to_taffy.contains_key(&root) {
            return true;
        }
        dom.children(root)
            .iter()
            .any(|&c| self.subtree_has_taffy_node(dom, c))
    }

    /// The cheap-path gate: true only when the subtree of `eff`
    /// changed in style values alone — no element inserted (G1), no
    /// element removed or reparented (G2), no text changed (G3).
    /// Every check is conservative: a failure takes the subtree
    /// path, which is always correct. `node_styles` here must still
    /// be the OLD map (the caller overwrites it only after the
    /// gate has run).
    fn subtree_structure_unchanged(
        &self,
        dom: &Dom,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        eff: DomId,
    ) -> bool {
        // (G1) Every element in the new subtree existed at the last
        // build — the style pass covers `display: none` subtrees too,
        // so a missing old `NodeStyle` means "inserted".
        let mut stack = vec![eff];
        while let Some(id) = stack.pop() {
            if dom.tag(id).is_some() && self.node_styles.get(id).is_none() {
                return false;
            }
            stack.extend(dom.children(id).iter().copied());
        }
        // (G2) Every element in the old taffy subtree is still a
        // descendant-or-self of `eff` in the new DOM. Removed nodes
        // have their parent link cleared by `detach`, and reparented
        // nodes walk up to a different root — both fail this test.
        let Some(&eff_taffy) = self.dom_to_taffy.get(&eff) else {
            return false;
        };
        for id in self.taffy_subtree_ids(eff_taffy) {
            if let Some(dom_id) = self.meta.get(&id).and_then(|m| m.dom)
                && !is_descendant_or_self(dom, dom_id, eff)
            {
                return false;
            }
        }
        // (G3) The processed text is unchanged.
        self.subtree_text_unchanged(dom, styles, eff)
    }

    /// (G3): the processed text of `eff`'s subtree — the spans of
    /// the old text leaves, in document order, against a fresh walk
    /// of the new DOM applying the same `white-space` processing the
    /// builder does (a text node's parent element's style governs
    /// it; `<br>` is a forced line break; `display: none` subtrees
    /// contribute nothing).
    fn subtree_text_unchanged(
        &self,
        dom: &Dom,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        eff: DomId,
    ) -> bool {
        let Some(&eff_taffy) = self.dom_to_taffy.get(&eff) else {
            return false;
        };
        let subtree: HashSet<TaffyId> = self.taffy_subtree_ids(eff_taffy).into_iter().collect();
        let mut old_text = String::new();
        for bx in self.boxes.iter() {
            if subtree.contains(&bx.taffy)
                && let Some(leaf) = bx.text.as_ref()
            {
                for span in &leaf.spans {
                    old_text.push_str(&span.text);
                }
            }
        }
        let mut new_text = String::new();
        if !collect_processed_text(dom, styles, eff, &mut new_text) {
            return false;
        }
        old_text == new_text
    }

    /// The pixel delta for a position-only move of `id`, from the
    /// old style (this tree's `node_styles`) and the new style.
    /// `None` when the move is not computable from the style diff.
    fn position_only_delta(
        &self,
        id: DomId,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
    ) -> Option<(f32, f32)> {
        let old = self.node_styles.get(id)?.computed.clone();
        let new = styles.get(&id)?;
        // `inset` order is [top, right, bottom, left].
        let dx = axis_delta(old.inset[3], new.inset[3], old.inset[1], new.inset[1])?;
        let dy = axis_delta(old.inset[0], new.inset[0], old.inset[2], new.inset[2])?;
        Some((dx, dy))
    }

    /// Taffy ids in the subtree rooted at `root`, pre-order (root
    /// first). Reversed, it is a valid bottom-up removal order.
    fn taffy_subtree_ids(&self, root: TaffyId) -> Vec<TaffyId> {
        let mut out = vec![root];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(m) = self.meta.get(&id) {
                for &child in &m.children {
                    out.push(child);
                    stack.push(child);
                }
            }
        }
        out
    }

    /// Drop a whole taffy subtree (children first), with its meta,
    /// `dom_to_taffy` entries, and position overrides.
    fn remove_taffy_subtree(&mut self, root: TaffyId) {
        let ids = self.taffy_subtree_ids(root);
        for &id in ids.iter().rev() {
            // `remove` detaches the node from its parent and marks
            // the parent dirty; children go first so no parent is
            // removed while its children still point at it.
            let _ = self.taffy.remove(id);
            self.meta.remove(&id);
            self.position_overrides.remove(&id);
        }
        let removed: HashSet<TaffyId> = ids.into_iter().collect();
        self.dom_to_taffy.retain(|_, &mut t| !removed.contains(&t));
    }

    /// Rebuild and re-lay out the subtree rooted at `eff` (which
    /// passed the safe-root predicate). The rest of the tree keeps
    /// its cached taffy layouts; taffy recomputes the new nodes and
    /// their ancestors.
    #[allow(clippy::too_many_arguments)]
    fn relayout_subtree(
        &mut self,
        dom: &Dom,
        sheet: &Stylesheet,
        styles: &HashMap<DomId, Arc<ComputedStyle>>,
        node_styles: SparseSecondaryMap<DomId, NodeStyle>,
        eff: DomId,
        viewport: (f32, f32),
        state: &ElementState,
    ) -> Result<RelayoutOutcome, LayoutError> {
        let eff_taffy = self.dom_to_taffy[&eff];
        let taffy_parent = match self.taffy.parent(eff_taffy) {
            Some(p) => p,
            None => {
                return self.full_rebuild(
                    dom,
                    sheet,
                    viewport.0,
                    viewport.1,
                    state,
                    "layout root has no taffy parent",
                );
            }
        };
        // The parent's children lists, saved before removal:
        // `taffy.remove` detaches the old node from its parent, so
        // the lists must be rebuilt from these copies, with the old
        // id swapped for the new one in place.
        let parent_kids = self.taffy.children(taffy_parent)?;
        let parent_meta_kids = self
            .meta
            .get(&taffy_parent)
            .map(|m| m.children.clone())
            .unwrap_or_default();
        // The parent's computed style is the fallback for `style_for`
        // inside the subtree build.
        let parent_style = self
            .meta
            .get(&taffy_parent)
            .and_then(|m| m.dom)
            .and_then(|d| styles.get(&d))
            .map(|s| (**s).clone())
            .unwrap_or_else(ComputedStyle::initial);
        let new_style = styles
            .get(&eff)
            .map(|s| (**s).clone())
            .unwrap_or_else(ComputedStyle::initial);
        // Drop the old subtree, then build the new one in its place
        // with a Builder borrowing the retained tree.
        self.remove_taffy_subtree(eff_taffy);
        let mut new_leaves: Vec<TaffyId> = Vec::new();
        let new_node = {
            let mut builder = Builder {
                dom,
                styles: styles.clone(),
                taffy: &mut self.taffy,
                meta: &mut self.meta,
                leaves: &mut new_leaves,
                viewport,
                // The retained tree's image map: a subtree rebuild
                // keeps the images decoded so far (Stage 11, M3).
                images: &self.images,
            };
            match new_style.display {
                DisplayKind::None | DisplayKind::Inline => None,
                DisplayKind::InlineBlock => builder.build_atomic(eff, &new_style),
                _ => builder.build_node(eff, &parent_style),
            }
        };
        // Reattach: swap the old id for the new one in both children
        // lists (or drop it, for `display: none` / inline). Either
        // way the parent is marked dirty, so its layout re-runs.
        let reattach = |kids: Vec<TaffyId>| -> Vec<TaffyId> {
            match new_node {
                Some(node) => kids
                    .into_iter()
                    .map(|c| if c == eff_taffy { node } else { c })
                    .collect(),
                None => kids.into_iter().filter(|&c| c != eff_taffy).collect(),
            }
        };
        let new_kids = reattach(parent_kids);
        self.taffy.set_children(taffy_parent, &new_kids)?;
        if let Some(m) = self.meta.get_mut(&taffy_parent) {
            m.children = reattach(parent_meta_kids);
        }
        // New leaves need pre-shaping before taffy measures them.
        preshape_leaves(&mut self.taffy, &mut self.meta, &new_leaves);
        // Taffy recomputes the dirty nodes (the new subtree and its
        // ancestors); clean subtrees return their cached layouts.
        self.taffy.compute_layout_with_measure(
            self.root.expect("retained tree has a root"),
            Size {
                width: AvailableSpace::Definite(viewport.0),
                height: AvailableSpace::MaxContent,
            },
            |input, _node, ctx, _style| measure_text(ctx, input),
        )?;
        self.dom_to_taffy = self
            .meta
            .iter()
            .filter_map(|(&t, m)| m.dom.map(|d| (d, t)))
            .collect();
        self.node_styles = node_styles;
        self.root_font_size = root_font_size_of(dom, styles);
        self.recollect();
        self.refresh_canvas_background(dom);
        self.debug_assert_matches_cold(dom, sheet, viewport, state);
        Ok(RelayoutOutcome::of(
            RelayoutKind::SubtreeLayout,
            "safe layout root: rebuilt only the dirty subtree",
        ))
    }

    /// The full-layout fallback: rebuild cold, keeping the retained
    /// scroll offsets (the incremental path preserves them, so the
    /// fallback must too).
    fn full_rebuild(
        &mut self,
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
        state: &ElementState,
        reason: &'static str,
    ) -> Result<RelayoutOutcome, LayoutError> {
        let scroll_offsets = std::mem::take(&mut self.scroll_offsets);
        // M6 Phase 3 (Rec 1/3/12): a fresh tree starts with no
        // viewport scroll, the Light theme, and no loaded web
        // fonts — carry the retained ones across, or a rebuild
        // would silently drop the viewport scroll (sticky/fixed
        // hit-testing), the dark theme, and every downloaded font.
        let theme = self.theme;
        let viewport_scroll = self.viewport_scroll;
        let loaded_font_hashes = std::mem::take(&mut self.loaded_font_hashes);
        // M6 Phase 3 (Rec 11): the in-flight animation values
        // survive a rebuild like the theme and the scroll — the
        // fresh tree resolves underlying styles only, so the
        // overrides are re-applied to its meta styles below.
        let animation_overrides = std::mem::take(&mut self.animation_overrides);
        // Reuse this tree's FontSystem (Stage 8 Rec 9): a fresh one
        // re-scans the system fonts (~1.5 s, more in debug), and a
        // plain page has no safe layout root, so even ordinary
        // JS-mutation relayouts take this path — a hover restyle
        // would otherwise pay a font scan per pointer event. The
        // `take` leaves `None` behind; the fresh build restores it,
        // and on failure the caller falls back to a fresh scan.
        let fonts = self.font_system.take();
        // Stage 12 Rec 5: the rebuild takes the retained image map
        // (`self.images` is already the fresh one when this runs
        // from `relayout_incremental` with an image arrival — the
        // old no-images variants built with an empty map, so an
        // image tick silently dropped every decoded image from the
        // fresh tree).
        let fresh = match fonts {
            Some(fs) => Self::build_with_state_reusing_font_system_and_images(
                dom,
                sheet,
                viewport_width,
                viewport_height,
                state,
                fs,
                &self.images,
                theme,
            ),
            None => Self::build_with_state_and_images(
                dom,
                sheet,
                viewport_width,
                viewport_height,
                state,
                &self.images,
                theme,
            ),
        }?;
        *self = fresh;
        self.scroll_offsets = scroll_offsets;
        self.viewport_scroll = viewport_scroll;
        self.loaded_font_hashes = loaded_font_hashes;
        self.animation_overrides = animation_overrides;
        // The fresh build collected boxes without the retained
        // offsets; re-collect so the boxes carry them. The
        // viewport scroll matters too: sticky offsets are
        // scroll-derived, so a retained viewport scroll also
        // forces a re-collect.
        if !self.scroll_offsets.is_empty() || self.viewport_scroll != (0.0, 0.0) {
            self.recollect();
        }
        // Re-apply in-flight animation values over the fresh
        // tree's underlying styles (see above).
        if !self.animation_overrides.is_empty() {
            self.apply_animation_overrides_to_meta();
        }
        Ok(RelayoutOutcome::of(RelayoutKind::FullLayout, reason))
    }

    /// Apply the retained `animation_overrides` to the fresh
    /// tree's meta styles after a full rebuild (the fresh build
    /// resolves underlying styles only), then re-collect so the
    /// boxes carry the sampled values. Text leaves pick the
    /// overridden style up through `refresh_meta_styles`' parent
    /// fallback, exactly as at build time. `pub` so the renderer's
    /// actor can apply it after a cold `build_full_tree` too.
    pub fn apply_animation_overrides_to_meta(&mut self) {
        if self.animation_overrides.is_empty() {
            return;
        }
        let mut overridden: HashMap<DomId, Arc<ComputedStyle>> = HashMap::new();
        for (id, pairs) in &self.animation_overrides {
            let Some(&t) = self.dom_to_taffy.get(id) else {
                continue;
            };
            let Some(m) = self.meta.get(&t) else {
                continue;
            };
            let mut style = m.style.clone();
            for (prop, val) in pairs {
                apply_animated_value(&mut style, prop, val);
            }
            overridden.insert(*id, Arc::new(style));
        }
        self.refresh_meta_styles(&overridden);
        self.recollect();
    }

    /// Refresh the meta styles from a freshly resolved map, so a
    /// paint-only re-collect paints the new values. Element boxes
    /// take their new style; anonymous text leaves (no `DomId`) take
    /// their taffy parent's new style, as at build time
    /// (`flush_spans` clones the parent style).
    fn refresh_meta_styles(&mut self, styles: &HashMap<DomId, Arc<ComputedStyle>>) {
        let updates: Vec<(TaffyId, ComputedStyle)> = self
            .meta
            .iter()
            .filter_map(|(&t, m)| {
                if let Some(new) = m.dom.and_then(|d| styles.get(&d)) {
                    return Some((t, (**new).clone()));
                }
                let parent = self.taffy.parent(t)?;
                let pdom = self.meta.get(&parent).and_then(|pm| pm.dom)?;
                let new = styles.get(&pdom)?;
                Some((t, (**new).clone()))
            })
            .collect();
        for (t, style) in updates {
            if let Some(m) = self.meta.get_mut(&t) {
                m.style = style;
            }
        }
    }

    /// Canvas background: the html element's background, else the
    /// body's, else white — the same rule `build` uses, re-run after
    /// a re-collect so a paint-only change to either is honoured.
    fn refresh_canvas_background(&mut self, dom: &Dom) {
        let html_bg = self
            .root
            .and_then(|r| self.boxes.iter().find(|b| b.taffy == r))
            .and_then(|b| b.style.background);
        let body_bg = dom
            .root()
            .and_then(|r| find_element(dom, r, "body"))
            .and_then(|body_id| self.boxes.iter().find(|b| b.dom == Some(body_id)))
            .and_then(|b| b.style.background);
        self.canvas_background = html_bg.or(body_bg).unwrap_or([255, 255, 255]);
    }

    /// Recompute clips for `subtree` boxes from the (already
    /// translated) ancestor rects, mirroring `collect_boxes`: a box's
    /// clip is the intersection of its clipping ancestors' rects, so
    /// a clip bounded by an outer container stays bounded when the
    /// inner box moves. `clip_radii` comes from the nearest clipping
    /// ancestor, as at build time.
    fn recompute_clips_for(&mut self, subtree: &HashSet<TaffyId>) {
        let rects: HashMap<TaffyId, [f32; 4]> = self
            .boxes
            .iter()
            .map(|b| (b.taffy, [b.x, b.y, b.width, b.height]))
            .collect();
        let radii_of: HashMap<TaffyId, ResolvedCorners> =
            self.boxes.iter().map(|b| (b.taffy, b.radii)).collect();
        for bx in self.boxes.iter_mut() {
            if !subtree.contains(&bx.taffy) {
                continue;
            }
            let mut clip: Option<[f32; 4]> = None;
            let mut clip_radii: Option<ResolvedCorners> = None;
            let mut clip_source: Option<TaffyId> = None;
            let mut cur = Some(bx.taffy);
            while let Some(id) = cur {
                let Some(m) = self.meta.get(&id) else {
                    break;
                };
                if m.style.overflow_x != OverflowKind::Visible
                    || m.style.overflow_y != OverflowKind::Visible
                {
                    let Some(&rect) = rects.get(&id) else {
                        break;
                    };
                    clip = Some(match clip {
                        Some(c) => intersect_clip(c, rect),
                        None => rect,
                    });
                    if clip_source.is_none() {
                        clip_source = Some(id);
                        let r = radii_of.get(&id).copied().unwrap_or(ResolvedCorners::ZERO);
                        clip_radii = if r.is_zero() { None } else { Some(r) };
                    }
                }
                cur = self.taffy.parent(id);
            }
            bx.clip = clip;
            bx.clip_radii = clip_radii;
            bx.clip_source = clip_source;
        }
    }

    /// Debug-only invariant (Stage 8 Rec 6 §d.5): incremental layout
    /// must produce exactly the same boxes as a cold full layout.
    /// Any divergence means the safe-root predicate let an unsafe
    /// root through — narrow the predicate, never patch the geometry.
    /// Compiles to a no-op in release builds.
    #[cfg(debug_assertions)]
    fn debug_assert_matches_cold(
        &self,
        dom: &Dom,
        sheet: &Stylesheet,
        viewport: (f32, f32),
        state: &ElementState,
    ) {
        // M6 Phase 3 (Rec 11): the cold twin resolves underlying
        // styles only — it cannot reproduce in-flight animated
        // values, and must not be asked to. A geometry-changing
        // override (animated width / left) would false-fail the
        // tripwire, so the check is skipped while overrides are
        // active. This is the report §4 trap in the other
        // direction: cold is not the reference for animated
        // geometry; the engine's own tests are.
        if !self.animation_overrides.is_empty() {
            return;
        }
        // Stage 12 Rec 5: the cold twin builds with the same image
        // map as this tree — the old no-images build would false-
        // fail the assertion on any image tick (the incremental
        // tree carries the images, the twin did not).
        let mut cold = match Self::build_with_state_and_images(
            dom,
            sheet,
            viewport.0,
            viewport.1,
            state,
            &self.images,
            self.theme,
        ) {
            Ok(tree) => tree,
            Err(e) => panic!("incremental==cold check: cold build failed: {e}"),
        };
        // The cold tree starts with no retained scroll offsets; give
        // it this tree's so the comparison is apples-to-apples —
        // and the same viewport scroll, so fixed/sticky boxes (whose
        // positions depend on it) compare too.
        cold.scroll_offsets.clone_from(&self.scroll_offsets);
        cold.viewport_scroll = self.viewport_scroll;
        if !cold.scroll_offsets.is_empty() || cold.viewport_scroll != (0.0, 0.0) {
            cold.recollect();
        }
        assert_eq!(
            cold.boxes.len(),
            self.boxes.len(),
            "incremental==cold: incremental produced {} boxes, cold produced {}",
            self.boxes.len(),
            cold.boxes.len()
        );
        for (i, (c, inc)) in cold.boxes.iter().zip(self.boxes.iter()).enumerate() {
            assert_eq!(
                c.dom,
                inc.dom,
                "incremental==cold: box {i} dom differs (cold {c:?} vs incremental {inc:?})",
                c = c.dom,
                inc = inc.dom
            );
            for (name, a, b) in [
                ("x", c.x, inc.x),
                ("y", c.y, inc.y),
                ("width", c.width, inc.width),
                ("height", c.height, inc.height),
            ] {
                assert!(
                    (a - b).abs() < 0.001,
                    "incremental==cold: box {i} {name} differs: incremental {b}, cold {a}"
                );
            }
            assert_eq!(
                c.glyphs.len(),
                inc.glyphs.len(),
                "incremental==cold: box {i} glyph count differs (incremental {}, cold {})",
                inc.glyphs.len(),
                c.glyphs.len()
            );
            // M6 Phase 2: the old comparison is blind to paint
            // order and compositing state — a z-index or transform
            // regression passes it while painting wrong. Compare
            // the accumulated matrices and the compositing style
            // (opacity / z-index) per box, and the Appendix E
            // paint order of the whole tree, so the cheap paths
            // (PaintOnly / CompositingOnly) are covered too.
            for (name, a, b) in [
                ("matrix.a", c.matrix.a, inc.matrix.a),
                ("matrix.b", c.matrix.b, inc.matrix.b),
                ("matrix.c", c.matrix.c, inc.matrix.c),
                ("matrix.d", c.matrix.d, inc.matrix.d),
                ("matrix.e", c.matrix.e, inc.matrix.e),
                ("matrix.f", c.matrix.f, inc.matrix.f),
                ("opacity", c.style.opacity, inc.style.opacity),
            ] {
                assert!(
                    (a - b).abs() < 0.001,
                    "incremental==cold: box {i} {name} differs: incremental {b}, cold {a}"
                );
            }
            assert_eq!(
                c.style.z_index, inc.style.z_index,
                "incremental==cold: box {i} z-index differs (incremental {:?}, cold {:?})",
                inc.style.z_index, c.style.z_index
            );
            assert_eq!(
                c.is_stacking_context, inc.is_stacking_context,
                "incremental==cold: box {i} stacking-context flag differs"
            );
            assert_eq!(
                c.is_viewport_fixed, inc.is_viewport_fixed,
                "incremental==cold: box {i} viewport-fixed flag differs"
            );
        }
        // Paint order must agree too: it is a permutation of the
        // box indices, so comparing the two orders catches a
        // reordering the per-box checks cannot see.
        assert_eq!(
            cold.paint_order(),
            self.paint_order(),
            "incremental==cold: paint order differs"
        );
    }

    #[cfg(not(debug_assertions))]
    fn debug_assert_matches_cold(
        &self,
        _dom: &Dom,
        _sheet: &Stylesheet,
        _viewport: (f32, f32),
        _state: &ElementState,
    ) {
    }
}

/// Walk `id`'s subtree in document order, collecting the processed
/// text the builder would put in its leaves (see
/// `LayoutTree::subtree_text_unchanged`). Returns `false` when the
/// walk cannot be trusted (a missing style), which fails the gate
/// toward the subtree path.
fn collect_processed_text(
    dom: &Dom,
    styles: &HashMap<DomId, Arc<ComputedStyle>>,
    id: DomId,
    out: &mut String,
) -> bool {
    for &child in dom.children(id) {
        if let Some(text) = dom.text(child) {
            let Some(parent_style) = styles.get(&id) else {
                return false;
            };
            if let Some(processed) = process_text(text, parent_style.white_space) {
                out.push_str(&processed);
            }
            continue;
        }
        let Some(tag) = dom.tag(child) else {
            continue;
        };
        if tag == "br" {
            out.push('\n');
            continue;
        }
        let Some(style) = styles.get(&child) else {
            return false;
        };
        if style.display == DisplayKind::None {
            continue;
        }
        if !collect_processed_text(dom, styles, child, out) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_html::parse as parse_html;

    const VIEWPORT: (f32, f32) = (800.0, 600.0);

    /// A fixed-size, overflow-hidden div — the canonical safe root.
    const SAFE_DIV_CSS: &str = ".box { width: 300px; height: 200px; overflow: hidden; }";

    fn build(html: &str, css: &str) -> (Dom, Stylesheet, LayoutTree) {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(css);
        let tree =
            LayoutTree::build(&dom, &sheet, VIEWPORT.0, VIEWPORT.1, Theme::Light)
                .expect("layout builds");
        (dom, sheet, tree)
    }

    fn cold(dom: &Dom, sheet: &Stylesheet) -> LayoutTree {
        LayoutTree::build(dom, sheet, VIEWPORT.0, VIEWPORT.1, Theme::Light)
            .expect("cold layout builds")
    }

    /// Geometry equality between two trees (the same check the debug
    /// assertion runs, so the property also holds in release builds).
    fn assert_same_geometry(a: &LayoutTree, b: &LayoutTree) {
        assert_eq!(a.boxes.len(), b.boxes.len(), "box count differs");
        for (i, (x, y)) in a.boxes.iter().zip(b.boxes.iter()).enumerate() {
            assert_eq!(x.dom, y.dom, "box {i} dom differs");
            assert!(
                (x.x - y.x).abs() < 0.001,
                "box {i} x differs: {} vs {}",
                x.x,
                y.x
            );
            assert!(
                (x.y - y.y).abs() < 0.001,
                "box {i} y differs: {} vs {}",
                x.y,
                y.y
            );
            assert!(
                (x.width - y.width).abs() < 0.001,
                "box {i} width differs: {} vs {}",
                x.width,
                y.width
            );
            assert!(
                (x.height - y.height).abs() < 0.001,
                "box {i} height differs: {} vs {}",
                x.height,
                y.height
            );
        }
    }

    fn el(dom: &Dom, tag: &str) -> DomId {
        find_element(dom, dom.root().unwrap(), tag).unwrap_or_else(|| panic!("no <{tag}>"))
    }

    fn first_text(dom: &Dom, parent: DomId) -> DomId {
        dom.children(parent)
            .iter()
            .copied()
            .find(|&c| dom.text(c).is_some())
            .expect("no text child")
    }

    fn styles_of(dom: &Dom, sheet: &Stylesheet) -> HashMap<DomId, Arc<ComputedStyle>> {
        let ua = Stylesheet::ua();
        resolve_document_styles(
            dom,
            &ua,
            sheet,
            false,
            VIEWPORT,
            &ElementState::default(),
            Theme::Light,
        )
        .0
    }

    /// Mutate (already done by the caller), relayout incrementally,
    /// clear the dirty set — the tab thread's contract.
    fn relayout(tree: &mut LayoutTree, dom: &mut Dom, sheet: &Stylesheet) -> RelayoutOutcome {
        relayout_with_images(tree, dom, sheet, None)
    }

    /// Like [`relayout`], with a fresh image map for the pass
    /// (Stage 12 Rec 5) — `None` means no image arrived.
    fn relayout_with_images(
        tree: &mut LayoutTree,
        dom: &mut Dom,
        sheet: &Stylesheet,
        images: Option<&ImageMap>,
    ) -> RelayoutOutcome {
        let out = tree
            .relayout_incremental(
                dom,
                sheet,
                VIEWPORT.0,
                VIEWPORT.1,
                &ElementState::default(),
                images,
            )
            .expect("relayout works");
        dom.clear_dirty();
        out
    }

    /// A cold twin built with an image map (Stage 12 Rec 5): the
    /// reference geometry for image-arrival relayouts.
    fn cold_with_images(dom: &Dom, sheet: &Stylesheet, images: &ImageMap) -> LayoutTree {
        LayoutTree::build_with_state_and_images(
            dom,
            sheet,
            VIEWPORT.0,
            VIEWPORT.1,
            &ElementState::default(),
            images,
            Theme::Light,
        )
        .expect("cold layout builds")
    }

    fn box_for<'a>(tree: &'a LayoutTree, dom: &Dom, tag: &str) -> &'a LayoutBox {
        tree.boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.tag(id) == Some(tag)))
            .unwrap_or_else(|| panic!("no box for <{tag}>"))
    }

    // ---- (a) safe-root predicate ----

    #[test]
    fn safe_root_accepts_definite_size_bfc() {
        let (dom, sheet, _) = build(r#"<div class="box"><p>hi</p></div>"#, SAFE_DIV_CSS);
        let styles = styles_of(&dom, &sheet);
        let div = el(&dom, "div");
        assert!(is_safe_layout_root(&dom, &styles, div));
        assert_eq!(unsafe_layout_root_reason(&dom, &styles, div), None);
    }

    #[test]
    fn safe_root_accepts_contain_layout() {
        let (dom, sheet, _) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { contain: layout; padding: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        let div = el(&dom, "div");
        assert!(styles[&div].contain_layout, "contain: layout was parsed");
        assert!(is_safe_layout_root(&dom, &styles, div));
    }

    #[test]
    fn safe_root_accepts_replaced_img() {
        let (dom, sheet, _) = build(
            r#"<div><img style="display: inline-block; width: 40px; height: 30px;"></div>"#,
            "",
        );
        let styles = styles_of(&dom, &sheet);
        let img = el(&dom, "img");
        assert!(is_safe_layout_root(&dom, &styles, img));
    }

    #[test]
    fn safe_root_accepts_fixed_size_leaf() {
        let (dom, sheet, _) = build(
            r#"<div class="box"></div>"#,
            ".box { width: 100px; height: 50px; overflow: hidden; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "div")));
    }

    #[test]
    fn safe_root_rejects_auto_size() {
        let (dom, sheet, _) = build(r#"<div class="box"><p>hi</p></div>"#, ".box {}");
        let styles = styles_of(&dom, &sheet);
        let reason = unsafe_layout_root_reason(&dom, &styles, el(&dom, "div")).expect("unsafe");
        assert!(reason.contains("definite width"), "reason: {reason}");
    }

    #[test]
    fn safe_root_accepts_definite_width_auto_height() {
        // Stage 12 Rec 4 stage 1: a definite px width with auto
        // height is a safe root — the width cannot change with
        // the contents, and the height change propagates through
        // taffy's dirty ancestors.
        let (dom, sheet, _) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; overflow: hidden; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "div")));
    }

    #[test]
    fn safe_root_accepts_margin_collapse() {
        // Stage 12 Rec 4 stage 2: a definite-size root whose
        // first child's margins collapse through it is a safe
        // root now — taffy recomputes the collapse for every
        // dirty ancestor on a subtree rebuild. (This test used to
        // assert rejection.)
        let (dom, sheet, _) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; height: 200px; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "div")));
    }

    #[test]
    fn safe_root_rejects_abspos_descendant_outside_cb() {
        let (dom, sheet, _) = build(
            r#"<div class="box"><div class="abs">x</div></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .abs { position: absolute; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        let reason = unsafe_layout_root_reason(&dom, &styles, el(&dom, "div")).expect("unsafe");
        assert!(reason.contains("containing block"), "reason: {reason}");
    }

    #[test]
    fn safe_root_accepts_abspos_descendant_when_root_positioned() {
        let (dom, sheet, _) = build(
            r#"<div class="box"><div class="abs">x</div></div>"#,
            ".box { position: relative; width: 300px; height: 200px; overflow: hidden; } .abs { position: absolute; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "div")));
    }

    #[test]
    fn safe_root_rejects_abspos_descendant_under_transformed_ancestor() {
        // M6 Phase 1: a transformed ancestor is a containing
        // block for abs descendants, so a root below it is unsafe
        // when the abs descendant resolves against that outside
        // transformed ancestor. (Without the predicate change the
        // root would be accepted and the subtree laid out in
        // isolation against the wrong containing block.)
        let (dom, sheet, _) = build(
            r#"<div class="outer"><section class="box"><div class="abs">x</div></section></div>"#,
            ".outer { transform: translateX(10px); } .box { width: 300px; height: 200px; overflow: hidden; } .abs { position: absolute; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        let reason =
            unsafe_layout_root_reason(&dom, &styles, el(&dom, "section")).expect("unsafe");
        assert!(reason.contains("containing block"), "reason: {reason}");
    }

    #[test]
    fn safe_root_accepts_abspos_descendant_when_root_transformed() {
        // The transformed root itself IS the containing block, so
        // the abs descendant resolves inside the root.
        let (dom, sheet, _) = build(
            r#"<section class="box"><div class="abs">x</div></section>"#,
            ".box { transform: translateX(10px); width: 300px; height: 200px; overflow: hidden; } .abs { position: absolute; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "section")));
    }

    #[test]
    fn safe_root_rejects_fixed_descendant_outside_cb() {
        // M6 Phase 3 (Rec 1): a `fixed` descendant resolves
        // against the viewport — not against the root — unless a
        // positioned or transformed ancestor inside the root
        // re-parents it. The old predicate claimed fixed
        // descendants were always fine because they resolve
        // against the viewport; that exemption is false when a
        // transformed ancestor sits above the root, and the
        // predicate cannot see it — so the conservative rule
        // flags any barrier-less fixed descendant.
        let (dom, sheet, _) = build(
            r#"<div class="box"><div class="fix">x</div></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .fix { position: fixed; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        let reason = unsafe_layout_root_reason(&dom, &styles, el(&dom, "div")).expect("unsafe");
        assert!(reason.contains("containing block"), "reason: {reason}");
    }

    #[test]
    fn safe_root_accepts_fixed_descendant_when_root_positioned() {
        // The positioned root itself IS the containing block for
        // the fixed descendant, so it resolves inside the root.
        let (dom, sheet, _) = build(
            r#"<div class="box"><div class="fix">x</div></div>"#,
            ".box { position: relative; width: 300px; height: 200px; overflow: hidden; } .fix { position: fixed; left: 10px; top: 10px; }",
        );
        let styles = styles_of(&dom, &sheet);
        assert!(is_safe_layout_root(&dom, &styles, el(&dom, "div")));
    }

    #[test]
    fn safe_root_rejects_inline() {
        let (dom, sheet, _) = build(
            r#"<p><span style="width: 100px; height: 50px;">hi</span></p>"#,
            "",
        );
        let styles = styles_of(&dom, &sheet);
        let span = el(&dom, "span");
        assert!(!is_safe_layout_root(&dom, &styles, span));
    }

    // ---- damage classification ----

    #[test]
    fn classify_damage_levels() {
        assert_eq!(
            classify_damage(RestyleDamage {
                repaint: true,
                reflow: true,
                reconstruct: true,
            }),
            DamageKind::Reconstruct
        );
        assert_eq!(
            classify_damage(RestyleDamage {
                repaint: true,
                reflow: true,
                reconstruct: false,
            }),
            DamageKind::Layout
        );
        assert_eq!(
            classify_damage(RestyleDamage {
                repaint: true,
                reflow: false,
                reconstruct: false,
            }),
            DamageKind::PaintOnly
        );
        assert_eq!(classify_damage(RestyleDamage::default()), DamageKind::None);
        assert_eq!(classify_damage(RestyleDamage::full()), DamageKind::Layout);
    }

    #[test]
    fn classify_style_change_cases() {
        let base = ComputedStyle::initial();
        let mut paint = base.clone();
        paint.background = Some([255, 0, 0]);
        assert_eq!(classify_style_change(&base, &paint), DamageKind::PaintOnly);
        let mut color = base.clone();
        color.color = [1, 2, 3];
        assert_eq!(classify_style_change(&base, &color), DamageKind::PaintOnly);
        let mut layout = base.clone();
        layout.width = Some(Dim::Px(100.0));
        assert_eq!(classify_style_change(&base, &layout), DamageKind::Layout);
        let mut font = base.clone();
        font.font_size = 20.0;
        assert_eq!(classify_style_change(&base, &font), DamageKind::Layout);
        let mut hide = base.clone();
        hide.display = DisplayKind::None;
        assert_eq!(classify_style_change(&base, &hide), DamageKind::Reconstruct);
        let mut show = base.clone();
        show.display = DisplayKind::None;
        let mut shown = ComputedStyle::initial();
        shown.display = DisplayKind::Block;
        assert_eq!(
            classify_style_change(&show, &shown),
            DamageKind::Reconstruct
        );
        let mut flex = base.clone();
        flex.display = DisplayKind::Flex;
        assert_eq!(classify_style_change(&base, &flex), DamageKind::Layout);
        let mut abspos = base.clone();
        abspos.position = PositionKind::Absolute;
        abspos.inset = [Some(Dim::Px(10.0)), None, None, Some(Dim::Px(20.0))];
        let mut moved = abspos.clone();
        moved.inset = [Some(Dim::Px(30.0)), None, None, Some(Dim::Px(50.0))];
        assert_eq!(
            classify_style_change(&abspos, &moved),
            DamageKind::PositionOnly
        );
        let mut static_inset = base.clone();
        static_inset.inset = [Some(Dim::Px(10.0)), None, None, None];
        let mut static_moved = static_inset.clone();
        static_moved.inset = [Some(Dim::Px(30.0)), None, None, None];
        assert_eq!(
            classify_style_change(&static_inset, &static_moved),
            DamageKind::Layout
        );
        assert_eq!(classify_style_change(&base, &base), DamageKind::None);
        // M6 Phase 2 (Rec 9): transform / opacity only.
        use pluto_css::{CssLength, TransformFunction};
        // A transform value change on an already-transformed box
        // (still a stacking context either way) is the cheap path.
        let mut transformed = base.clone();
        transformed.transform =
            vec![TransformFunction::Translate(CssLength::Px(0.0), CssLength::Px(0.0))];
        let mut moved_transform = transformed.clone();
        moved_transform.transform =
            vec![TransformFunction::Translate(CssLength::Px(10.0), CssLength::Px(0.0))];
        assert_eq!(
            classify_style_change(&transformed, &moved_transform),
            DamageKind::CompositingOnly
        );
        // An opacity change between two translucent values is too.
        let mut translucent = base.clone();
        translucent.opacity = 0.5;
        let mut more_translucent = translucent.clone();
        more_translucent.opacity = 0.8;
        assert_eq!(
            classify_style_change(&translucent, &more_translucent),
            DamageKind::CompositingOnly
        );
        // But appearing/disappearing a transform creates or
        // destroys a stacking context, which reorders paint.
        assert_eq!(
            classify_style_change(&base, &transformed),
            DamageKind::PaintOnly
        );
        assert_eq!(
            classify_style_change(&transformed, &base),
            DamageKind::PaintOnly
        );
        // Likewise opacity crossing 1.0.
        assert_eq!(
            classify_style_change(&base, &translucent),
            DamageKind::PaintOnly
        );
        assert_eq!(
            classify_style_change(&translucent, &base),
            DamageKind::PaintOnly
        );
        // A transform-origin change only re-pivots the matrix.
        let mut origin = transformed.clone();
        origin.transform_origin = TransformOrigin {
            x: OriginComponent::Percentage(0.0),
            y: OriginComponent::Percentage(0.0),
        };
        assert_eq!(
            classify_style_change(&transformed, &origin),
            DamageKind::CompositingOnly
        );
        // `z-index` alone reorders paint without layout — never
        // swallowed, never a relayout.
        let mut zed = base.clone();
        zed.position = PositionKind::Absolute;
        zed.z_index = Some(1);
        let mut zed2 = zed.clone();
        zed2.z_index = Some(2);
        assert_eq!(
            classify_style_change(&zed, &zed2),
            DamageKind::PaintOnly
        );
        // `will-change` is a hint: no visual change, no damage.
        let mut hinted = base.clone();
        hinted.will_change_transform = true;
        assert_eq!(classify_style_change(&base, &hinted), DamageKind::None);
    }

    // ---- (b2) compositing-only fast path (M6 Phase 2, Rec 9) ----

    #[test]
    fn transform_only_change_is_compositing_only() {
        // A `transform` edit on a transformed element takes the
        // cheap path: no taffy, no list rebuild, and the outcome
        // names the changed element for the `CompositingUpdate`.
        // The debug `incremental == cold` check inside the path
        // also proves the retained matrices match a cold build.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="spin">S</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .spin { width: 100px; height: 100px; transform: translateX(0px); }",
        );
        let spin = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            spin,
            "style",
            "width: 100px; height: 100px; transform: translateX(30px);",
        );
        dom.mark_dirty(spin);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::CompositingOnly,
            "reason: {}",
            out.reason
        );
        assert_eq!(out.composited, vec![spin]);
        // The retained box's matrix carries the new translate.
        let bx = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(spin))
            .expect("spin box");
        let transforms = tree.page_transforms();
        let idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin))
            .unwrap();
        let (tx, ty) = transforms[idx].apply(bx.x, bx.y);
        assert!((tx - (bx.x + 30.0)).abs() < 0.01, "translated x {tx}");
        assert!((ty - bx.y).abs() < 0.01, "translated y {ty}");
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn opacity_only_change_is_compositing_only() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="ghost">G</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .ghost { width: 100px; height: 100px; opacity: 0.5; }",
        );
        let ghost = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            ghost,
            "style",
            "width: 100px; height: 100px; opacity: 0.8;",
        );
        dom.mark_dirty(ghost);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::CompositingOnly,
            "reason: {}",
            out.reason
        );
        assert_eq!(out.composited, vec![ghost]);
        let bx = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(ghost))
            .expect("ghost box");
        assert!((bx.style.opacity - 0.8).abs() < 1e-6);
    }

    // ---- M6 Phase 3 (Rec 11): transitions / animations / overrides ----

    #[test]
    fn opacity_change_reports_pending_transition() {
        // A `transition`ed property change reports the pending
        // transition (from/to underlying values) on the outcome,
        // so the tab thread can start its engine — and the new
        // value rides in `changed_styles` for the computed-style
        // cache.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="ghost">G</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .ghost { width: 100px; height: 100px; opacity: 0.5; transition: opacity 1s linear; }",
        );
        let ghost = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            ghost,
            "style",
            "width: 100px; height: 100px; opacity: 0.8;",
        );
        dom.mark_dirty(ghost);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::CompositingOnly,
            "reason: {}",
            out.reason
        );
        assert_eq!(out.pending_transitions.len(), 1, "one pending transition");
        let pt = &out.pending_transitions[0];
        assert_eq!(pt.id, ghost);
        assert_eq!(pt.spec.property, "opacity");
        assert_eq!(pt.from, AnimatedValue::Number(0.5));
        assert_eq!(pt.to, AnimatedValue::Number(0.8));
        let cached = out
            .changed_styles
            .iter()
            .find(|(id, _)| *id == ghost)
            .expect("changed_styles names the ghost");
        assert!(
            cached.1.iter().any(|(p, v)| p == "opacity" && v == "0.8"),
            "cache pairs: {:?}",
            cached.1
        );
    }

    #[test]
    fn new_animation_reports_pending_animation() {
        // A restyle that adds an `animation` entry reports the
        // pending animation with its resolved `@keyframes` rule.
        let (mut dom, sheet_a, mut tree) = build(
            r#"<div class="wrap"><div class="ghost">G</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .ghost { width: 100px; height: 100px; opacity: 0.5; }",
        );
        let ghost = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        let sheet_b = Stylesheet::parse(
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .ghost { width: 100px; height: 100px; opacity: 0.5; animation: fade 1s linear; } \
             @keyframes fade { from { opacity: 0; } to { opacity: 1; } }",
        );
        dom.mark_dirty(ghost);
        let out = relayout(&mut tree, &mut dom, &sheet_b);
        assert_eq!(out.pending_animations.len(), 1, "one pending animation");
        let pa = &out.pending_animations[0];
        assert_eq!(pa.id, ghost);
        assert_eq!(pa.spec.name, "fade");
        assert_eq!(pa.keyframes.name, "fade");
        assert_eq!(pa.keyframes.steps.len(), 2);
        // The same animation restyled identically does NOT
        // re-report: a second pass with the same sheet is quiet.
        dom.clear_dirty();
        dom.mark_dirty(ghost);
        let out2 = relayout(&mut tree, &mut dom, &sheet_b);
        assert!(
            out2.pending_animations.is_empty(),
            "no re-report: {:?}",
            out2.pending_animations
        );
        let _ = sheet_a;
    }

    #[test]
    fn animation_overrides_apply_over_resolved_styles() {
        // In-flight sampled values ride over the resolved styles:
        // a dirty pass with an opacity override on the ghost
        // paints 0.3 even though the cascade says 0.5 — and the
        // retained `node_styles` stay underlying (0.5), so the
        // next pass still diffs against the real cascade.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="ghost">G</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .ghost { width: 100px; height: 100px; opacity: 0.5; }",
        );
        let ghost = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        tree.animation_overrides = vec![(
            ghost,
            vec![("opacity".to_string(), AnimatedValue::Number(0.3))],
        )];
        dom.mark_dirty(ghost);
        let _out = relayout(&mut tree, &mut dom, &sheet);
        let bx = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(ghost))
            .expect("ghost box");
        assert!((bx.style.opacity - 0.3).abs() < 1e-6, "override applied");
        let retained = tree.node_styles.get(ghost).expect("retained style");
        assert!(
            (retained.computed.opacity - 0.5).abs() < 1e-6,
            "retained style stays underlying"
        );
    }

    #[test]
    fn z_index_only_change_is_paint_only_not_swallowed() {
        // The trap: a `z-index`-only change used to classify
        // `None` and vanish. It must rebuild the list (paint
        // order changes) without laying out.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="a">A</div><div class="b">B</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .a { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 1; } \
             .b { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 2; }",
        );
        let a = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            a,
            "style",
            "position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 3;",
        );
        dom.mark_dirty(a);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::PaintOnly, "reason: {}", out.reason);
        // And the new order is visible in the tree: a now paints
        // after b.
        let order = tree.paint_order();
        let pos = |id: DomId| {
            let idx = tree.boxes.iter().position(|b| b.dom == Some(id)).unwrap();
            order.iter().position(|&i| i == idx).unwrap()
        };
        let b = dom
            .children(el(&dom, "div"))
            .iter()
            .copied().find(|&c| dom.tag(c) == Some("div") && c != a)
            .unwrap();
        assert!(pos(b) < pos(a), "b now paints before a");
    }

    #[test]
    fn transform_plus_background_change_is_paint_only() {
        // A transform change bundled with a paint change needs
        // the list rebuilt (the background repaint cannot ride a
        // compositing patch), so the aggregate is `PaintOnly`,
        // not `CompositingOnly`.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="spin">S</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .spin { width: 100px; height: 100px; transform: translateX(0px); background-color: #ff0000; }",
        );
        let spin = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            spin,
            "style",
            "width: 100px; height: 100px; transform: translateX(30px); background-color: #00ff00;",
        );
        dom.mark_dirty(spin);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::PaintOnly, "reason: {}", out.reason);
    }

    // ---- (b) position-only fast path ----

    #[test]
    fn position_only_move_matches_cold() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="abs">M</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .abs { position: absolute; left: 50px; top: 60px; width: 100px; height: 50px; }",
        );
        let before = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains('M')))
            })
            .expect("abs text leaf")
            .clone();
        // The abs child is the div inside the wrapper div.
        let abs = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        dom.set_attribute(
            abs,
            "style",
            "position: absolute; left: 80px; top: 80px; width: 100px; height: 50px;",
        );
        dom.mark_dirty(abs);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::PositionOnly,
            "reason: {}",
            out.reason
        );
        let abs_after = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains('M')))
            })
            .expect("abs text leaf after move");
        assert!((abs_after.x - (before.x + 30.0)).abs() < 0.001);
        assert!((abs_after.y - (before.y + 20.0)).abs() < 0.001);
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn translate_subtree_matches_cold() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="wrap"><div class="abs">M</div></div>"#,
            ".wrap { position: relative; width: 400px; height: 300px; } \
             .abs { position: absolute; left: 50px; top: 60px; width: 100px; height: 50px; }",
        );
        let abs = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        assert!(tree.translate_subtree(abs, 30.0, 20.0));
        // The cold twin: the same document with the inset changed.
        dom.set_attribute(
            abs,
            "style",
            "position: absolute; left: 80px; top: 80px; width: 100px; height: 50px;",
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
        // And a later re-collect (e.g. a scroll) keeps the move:
        // the override map is the record of it.
        tree.set_scroll_offset(el(&dom, "div"), 0.0, 0.0);
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn translate_subtree_clip_stays_bounded() {
        // The moved node sits inside an outer `overflow: hidden`
        // container and clips its own child: the clip is the
        // intersection of the two, so translating the inner clip
        // rect blindly would push its far edge past the outer one.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="outer"><div class="abs"><div class="inner">deep text here</div></div></div>"#,
            ".outer { position: relative; width: 200px; height: 200px; overflow: hidden; } \
             .abs { position: absolute; left: 150px; top: 10px; width: 100px; height: 100px; overflow: hidden; } \
             .inner { width: 90px; height: 20px; }",
        );
        let abs = dom
            .children(el(&dom, "div"))
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("div"))
            .unwrap();
        assert!(tree.translate_subtree(abs, 40.0, 0.0));
        dom.set_attribute(
            abs,
            "style",
            "position: absolute; left: 190px; top: 10px; width: 100px; height: 100px; overflow: hidden;",
        );
        let cold_tree = cold(&dom, &sheet);
        assert_same_geometry(&tree, &cold_tree);
        // The inner box's clip must still end at the outer
        // container's right edge (200), not at 240.
        let inner = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("deep")))
            })
            .expect("inner text leaf");
        let clip = inner.clip.expect("inner leaf is clipped");
        // The outer container sits at x=8 (the body's UA margin) and
        // is 200 wide, so its right edge — and the clip's — is 208,
        // not 200: the clip is bounded by the outer container even
        // though the inner box moved past it.
        let outer = box_for(&tree, &dom, "div");
        assert!(
            (clip[0] + clip[2] - (outer.x + outer.width)).abs() < 0.001,
            "clip: {clip:?}, outer right edge: {}",
            outer.x + outer.width
        );
        assert!((clip[0] - 198.0).abs() < 0.001, "clip: {clip:?}");
    }

    // ---- (c) incremental == cold on representative mutations ----

    #[test]
    fn incremental_text_change_in_safe_root() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hello</p></div><div class="after"><p>below</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; }",
        );
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(
            text,
            "hello world, this is a much longer line of text that wraps",
        );
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_auto_height_root_grows_and_shifts_sibling() {
        // Stage 12 Rec 4 stage 1: a text mutation inside a
        // definite-width auto-height root takes the subtree path,
        // the root grows, and the sibling below it moves down —
        // exactly the case the old definite-height requirement
        // refused (it fell back to a full rebuild).
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>short</p></div><div class="after"><p>below</p></div>"#,
            ".box { width: 300px; overflow: hidden; }",
        );
        let body = el(&dom, "body");
        let divs: Vec<DomId> = dom
            .children(body)
            .iter()
            .copied()
            .filter(|&c| dom.tag(c) == Some("div"))
            .collect();
        assert_eq!(divs.len(), 2);
        let (box_div, after_div) = (divs[0], divs[1]);
        let box_before = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(box_div))
            .expect("box div")
            .clone();
        let after_before = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(after_div))
            .expect("after div")
            .clone();
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(
            text,
            "a much longer line of text that wraps onto several lines inside the 300px box",
        );
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        let box_after = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(box_div))
            .expect("box div after");
        let after_after = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(after_div))
            .expect("after div after");
        assert!(
            box_after.height > box_before.height,
            "the root grew: {} -> {}",
            box_before.height,
            box_after.height
        );
        assert!(
            after_after.y > after_before.y,
            "the sibling moved down: {} -> {}",
            after_before.y,
            after_after.y
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_margin_collapse_through_root() {
        // Stage 12 Rec 4 stage 2: the old rejection case — a
        // definite-size root with no BFC/padding, whose first
        // child's margins collapse through it — now takes the
        // subtree path, and the collapsed margins come out exactly
        // as the cold build computes them.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>short</p></div><div class="after"><p>below</p></div>"#,
            ".box { width: 300px; height: 200px; }",
        );
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(
            text,
            "a much longer line of text that wraps onto several lines inside the 300px box",
        );
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_margin_collapse_sibling_gap_preserved() {
        // Stage 12 Rec 4 stage 2: adjacent-sibling margins
        // collapse across the rebuild boundary — the gap between
        // the growing root and the sibling below it is the
        // collapsed margin, before and after, not the sum.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>short</p></div><div class="after"><p>below</p></div>"#,
            ".box { width: 300px; margin-bottom: 20px; } .after { margin-top: 30px; }",
        );
        let body = el(&dom, "body");
        let divs: Vec<DomId> = dom
            .children(body)
            .iter()
            .copied()
            .filter(|&c| dom.tag(c) == Some("div"))
            .collect();
        assert_eq!(divs.len(), 2);
        let (box_div, after_div) = (divs[0], divs[1]);
        let gap_before = {
            let bx = tree
                .boxes
                .iter()
                .find(|b| b.dom == Some(box_div))
                .expect("box div");
            let af = tree
                .boxes
                .iter()
                .find(|b| b.dom == Some(after_div))
                .expect("after div");
            af.y - (bx.y + bx.height)
        };
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(
            text,
            "a much longer line of text that wraps onto several lines inside the 300px box",
        );
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        let gap_after = {
            let bx = tree
                .boxes
                .iter()
                .find(|b| b.dom == Some(box_div))
                .expect("box div after");
            let af = tree
                .boxes
                .iter()
                .find(|b| b.dom == Some(after_div))
                .expect("after div after");
            af.y - (bx.y + bx.height)
        };
        assert!(
            (gap_after - gap_before).abs() < 0.001,
            "collapsed sibling gap changed across the rebuild: {gap_before} -> {gap_after}"
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_text_change_with_image_arrival_stays_subtree() {
        // Stage 12 Rec 5: a text mutation AND an image arrival in
        // the same tick — before Rec 5 the image forced a full
        // rebuild; now the subtree path runs with the fresh map,
        // and the <img> gets its intrinsic size.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>short</p><img src="https://example.com/a.png"></div><div class="after"><p>below</p></div>"#,
            SAFE_DIV_CSS,
        );
        let img = el(&dom, "img");
        let mut map = ImageMap::new();
        map.by_node.insert(
            img,
            ImageInfo {
                resource: 1,
                width: 40,
                height: 30,
            },
        );
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(
            text,
            "a much longer line of text that wraps onto several lines inside the 300px box",
        );
        dom.mark_dirty(text);
        let out = relayout_with_images(&mut tree, &mut dom, &sheet, Some(&map));
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        let img_box = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(img))
            .expect("img box");
        assert!(
            (img_box.width - 40.0).abs() < 0.001,
            "img width: {}",
            img_box.width
        );
        assert!(
            (img_box.height - 30.0).abs() < 0.001,
            "img height: {}",
            img_box.height
        );
        assert_same_geometry(&tree, &cold_with_images(&dom, &sheet, &map));
    }

    #[test]
    fn incremental_image_only_arrival_in_safe_root_is_subtree() {
        // Stage 12 Rec 5: an image-only tick (empty dirty set) —
        // the changed <img> is the damage, and inside a safe root
        // it re-lays out just that subtree with the fresh map.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><img src="https://example.com/a.png"></div>"#,
            SAFE_DIV_CSS,
        );
        let img = el(&dom, "img");
        let mut map = ImageMap::new();
        map.by_node.insert(
            img,
            ImageInfo {
                resource: 1,
                width: 40,
                height: 30,
            },
        );
        let out = relayout_with_images(&mut tree, &mut dom, &sheet, Some(&map));
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        let img_box = tree
            .boxes
            .iter()
            .find(|b| b.dom == Some(img))
            .expect("img box");
        assert!(
            (img_box.width - 40.0).abs() < 0.001,
            "img width: {}",
            img_box.width
        );
        assert!(
            (img_box.height - 30.0).abs() < 0.001,
            "img height: {}",
            img_box.height
        );
        assert_same_geometry(&tree, &cold_with_images(&dom, &sheet, &map));
    }

    #[test]
    fn incremental_background_arrival_forces_full_rebuild() {
        // Stage 12 Rec 5: a background-image arrival cannot name
        // its nodes (the map is keyed by URL), so it forces the
        // full rebuild — same as before Rec 5 — and the fresh
        // resource id lands on the box.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; height: 200px; background-image: url(bg.png); background-size: cover; }",
        );
        let mut map = ImageMap::new();
        map.backgrounds.insert(
            "bg.png".to_string(),
            ImageInfo {
                resource: 9,
                width: 800,
                height: 600,
            },
        );
        let out = relayout_with_images(&mut tree, &mut dom, &sheet, Some(&map));
        assert_eq!(
            out.kind,
            RelayoutKind::FullLayout,
            "reason: {}",
            out.reason
        );
        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.background_image, Some(9));
        assert_eq!(div.background_image_size, Some((800, 600)));
        assert_same_geometry(&tree, &cold_with_images(&dom, &sheet, &map));
    }

    #[test]
    fn incremental_attribute_change_paint_only() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .a { background-color: #ff0000; } .b { background-color: #0000ff; }",
        );
        let div = el(&dom, "div");
        dom.set_attribute(div, "class", "box a");
        dom.mark_dirty(div);
        // First lay out with the red class, so the next change is a
        // real paint-only diff.
        let _ = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            box_for(&tree, &dom, "div").style.background,
            Some([255, 0, 0])
        );
        dom.set_attribute(div, "class", "box b");
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::PaintOnly, "reason: {}", out.reason);
        assert_eq!(
            box_for(&tree, &dom, "div").style.background,
            Some([0, 0, 255])
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_color_change_goes_through_subtree() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .a { color: #ff0000; } .b { color: #0000ff; }",
        );
        let div = el(&dom, "div");
        dom.set_attribute(div, "class", "box a");
        dom.mark_dirty(div);
        let _ = relayout(&mut tree, &mut dom, &sheet);
        dom.set_attribute(div, "class", "box b");
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        // Geometrically paint-only, but glyph colours are baked into
        // shaped leaves, so v1 routes it through the subtree path.
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        let leaf = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("hi")))
            })
            .expect("text leaf");
        assert!(
            leaf.glyphs.iter().any(|g| g.color == [0, 0, 255]),
            "glyphs carry the new blue"
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_subtree_insert() {
        let (mut dom, sheet, mut tree) =
            build(r#"<div class="box"><p>one</p></div>"#, SAFE_DIV_CSS);
        let div = el(&dom, "div");
        let p = dom.create_element("p");
        let text = dom.create_text("two");
        dom.append_child(p, text);
        dom.append_child(div, p);
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert!(tree.boxes.iter().any(|b| {
            b.text
                .as_ref()
                .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("two")))
        }));
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_subtree_remove() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>one</p><p>two</p></div>"#,
            SAFE_DIV_CSS,
        );
        let div = el(&dom, "div");
        let ps: Vec<DomId> = dom
            .children(div)
            .iter()
            .copied()
            .filter(|&c| dom.tag(c) == Some("p"))
            .collect();
        assert_eq!(ps.len(), 2);
        dom.remove_child(div, ps[1]);
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert!(!tree.boxes.iter().any(|b| {
            b.text
                .as_ref()
                .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("two")))
        }));
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_style_change_size() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div><div class="after"><p>below</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; }",
        );
        let div = el(&dom, "div");
        dom.set_attribute(
            div,
            "style",
            "width: 350px; height: 200px; overflow: hidden;",
        );
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert!((box_for(&tree, &dom, "div").width - 350.0).abs() < 0.001);
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn paint_only_with_insert_takes_subtree_path() {
        // A paint-only change AND an insert in one batch: the cheap
        // path's structural gate must catch the insert.
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .a { background-color: #ff0000; } .b { background-color: #0000ff; }",
        );
        let div = el(&dom, "div");
        dom.set_attribute(div, "class", "box a");
        dom.mark_dirty(div);
        let _ = relayout(&mut tree, &mut dom, &sheet);
        dom.set_attribute(div, "class", "box b");
        let p = dom.create_element("p");
        let text = dom.create_text("new");
        dom.append_child(p, text);
        dom.append_child(div, p);
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::SubtreeLayout,
            "reason: {}",
            out.reason
        );
        assert_eq!(
            box_for(&tree, &dom, "div").style.background,
            Some([0, 0, 255])
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    // ---- (d) unsafe roots fall back to full layout ----

    #[test]
    fn incremental_unsafe_root_falls_back() {
        let (mut dom, sheet, mut tree) =
            build(r#"<div><p>hello</p></div><div><p>below</p></div>"#, "");
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(text, "hello world, a much longer line of text that wraps");
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::FullLayout, "reason: {}", out.reason);
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_abspos_descendant_falls_back() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><div class="abs">x</div><p>hello</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .abs { position: absolute; left: 10px; top: 10px; }",
        );
        let p = el(&dom, "p");
        let text = first_text(&dom, p);
        dom.set_text(text, "hello world, a much longer line of text that wraps");
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::FullLayout, "reason: {}", out.reason);
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_display_none_toggle_falls_back() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="box"><p>hi</p></div><div><p>below</p></div>"#,
            ".box { width: 300px; height: 200px; overflow: hidden; } .hide { display: none; }",
        );
        let div = el(&dom, "div");
        dom.set_attribute(div, "class", "box hide");
        dom.mark_dirty(div);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::FullLayout, "reason: {}", out.reason);
        assert!(tree.boxes.iter().all(|b| b.dom != Some(div)));
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_nothing_dirty() {
        let (dom, sheet, mut tree) = build(r#"<div class="box"><p>hi</p></div>"#, SAFE_DIV_CSS);
        let out = tree
            .relayout_incremental(
                &dom,
                &sheet,
                VIEWPORT.0,
                VIEWPORT.1,
                &ElementState::default(),
                None,
            )
            .expect("relayout works");
        assert_eq!(out.kind, RelayoutKind::NothingToDo);
    }

    #[test]
    fn incremental_head_change_is_nothing() {
        let (mut dom, sheet, mut tree) = build(
            r#"<html><head><title>old</title></head><body><p>hi</p></body></html>"#,
            "",
        );
        let title = el(&dom, "title");
        let text = first_text(&dom, title);
        dom.set_text(text, "new title");
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(
            out.kind,
            RelayoutKind::NothingToDo,
            "reason: {}",
            out.reason
        );
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_text_change_inside_inline_span_rebuilds() {
        // Regression (Stage 8 Rec 6/9): a text change inside an
        // inline span — `<span>9</span>` becoming `10`, the counter
        // click — must re-shape the span's text. The span owns no
        // taffy node (its text lives in the parent block's node),
        // so the old "dirty subtree has no boxes" gate dropped the
        // change as NothingToDo and the old text stayed on screen.
        // The debug assertion does not catch this: the NotRendered
        // arm returns before it runs.
        let (mut dom, sheet, mut tree) = build(r#"<p>count: <span>9</span></p>"#, "");
        let span = el(&dom, "span");
        let text = first_text(&dom, span);
        dom.set_text(text, "10");
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_ne!(
            out.kind,
            RelayoutKind::NothingToDo,
            "reason: {}",
            out.reason
        );
        // The incremental tree matches a cold build of the new DOM
        // (the span's box widens by one glyph).
        assert_same_geometry(&tree, &cold(&dom, &sheet));
    }

    #[test]
    fn incremental_scroll_offset_survives_fallback() {
        let (mut dom, sheet, mut tree) = build(
            r#"<div class="scroll"><p>line one</p><p>line two</p><p>line three</p><p>line four</p><p>line five</p></div><div><p>hello</p></div>"#,
            ".scroll { width: 200px; height: 60px; overflow: auto; }",
        );
        let scroll = el(&dom, "div");
        tree.set_scroll_offset(scroll, 0.0, 20.0);
        assert_eq!(tree.scroll_offset(scroll), Some((0.0, 20.0)));
        // An unsafe mutation elsewhere forces the full fallback: the
        // paragraph in the plain auto-height sibling div.
        let body = el(&dom, "body");
        let divs: Vec<DomId> = dom
            .children(body)
            .iter()
            .copied()
            .filter(|&c| dom.tag(c) == Some("div"))
            .collect();
        assert_eq!(divs.len(), 2);
        let last_p = dom
            .children(divs[1])
            .iter()
            .copied()
            .find(|&c| dom.tag(c) == Some("p"))
            .unwrap();
        let text = first_text(&dom, last_p);
        dom.set_text(text, "hello world, a much longer line of text that wraps");
        dom.mark_dirty(text);
        let out = relayout(&mut tree, &mut dom, &sheet);
        assert_eq!(out.kind, RelayoutKind::FullLayout, "reason: {}", out.reason);
        assert_eq!(tree.scroll_offset(scroll), Some((0.0, 20.0)));
        // The cold twin needs the same retained offset to compare.
        let mut cold_tree = cold(&dom, &sheet);
        cold_tree.set_scroll_offset(scroll, 0.0, 20.0);
        assert_same_geometry(&tree, &cold_tree);
    }
}
