//! `ResizeObserver` registry + gather/broadcast loop (Stage 8,
//! Recommendation 8).
//!
//! Pure Rust, no JS types: the registry holds observations, the
//! gather step compares current sizes against `lastReportedSizes`,
//! and the caller (the `TabJs` rendering step, or the browser's
//! worker step over layout-tree sizes) delivers the batches and
//! re-queries sizes between rounds. Delivery itself lives in
//! `bindings::observers`, which turns entries into JS objects and
//! invokes the callbacks.
//!
//! # Loop shape (from the Stage 8 report, §(e))
//!
//! One step is a sequence of *rounds*. Each round gathers the
//! active observations — size differs from `lastReportedSizes`, or
//! no size has ever been reported and the observed box is non-zero
//! (a 0×0 target gets no initial entry, matching "does not fire
//! for 0×0 unrendered targets"; a later resize *to* 0×0 does fire,
//! matching "fires on … `display:none`"). Active observations are
//! broadcast shallowest-first (by DOM depth, then observer
//! creation order), grouped by observer, and `lastReportedSizes`
//! is updated to the delivered size. Then sizes are re-queried:
//! a callback that resized something makes that observation
//! active again, and the next round picks it up.
//!
//! The loop ends when a round finds nothing active. If callbacks
//! keep changing sizes round after round, the step stops after
//! [`MAX_ROUNDS`] rounds and reports
//! [`ResizeStepOutcome::loop_error`] — Pluto's form of the spec's
//! resize-loop error ("ResizeObserver loop completed with
//! undelivered notifications."). The spec terminates its loop by
//! depth-shrinking; Pluto keeps the depth ordering (shallowest
//! first) and adds the round bound as the termination guarantee,
//! because a callback can keep changing a same-depth size forever
//! and depth-shrinking alone does not bound that.
//!
//! Unrendered targets (the size provider returns `None`) are never
//! active: no entry is delivered for them, and no initial entry
//! fires until they are rendered.

use crate::host::DomKey;

/// Which box an observation watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeBoxKind {
    /// Content box (the default).
    #[default]
    ContentBox,
    /// Border box.
    BorderBox,
    /// Device-pixel content box (content box × device pixel ratio;
    /// Pluto has no DPR plumbing yet, so this equals the content
    /// box — see [`ObservedSizes::from_border`]).
    DevicePixelContentBox,
}

impl ResizeBoxKind {
    /// Parse the `box` option (`"content-box"` /
    /// `"border-box"` / `"device-pixel-content-box"`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "content-box" => Some(ResizeBoxKind::ContentBox),
            "border-box" => Some(ResizeBoxKind::BorderBox),
            "device-pixel-content-box" => Some(ResizeBoxKind::DevicePixelContentBox),
            _ => None,
        }
    }
}

/// The sizes of one observed target, as the entry reports them.
/// All pairs are `(inline size, block size)` in CSS px, except
/// `device_pixel`, which is in device px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservedSizes {
    /// Border-box size.
    pub border: (f64, f64),
    /// Content-box size.
    pub content: (f64, f64),
    /// Device-pixel content-box size.
    pub device_pixel: (f64, f64),
    /// Content-box offset from the border-box origin
    /// (`(padding-left, padding-top)`); the legacy `contentRect`'s
    /// `x` / `y`.
    pub content_offset: (f64, f64),
}

impl ObservedSizes {
    /// Sizes when only the border box is known (the JS-layer size
    /// provider, which has no padding data): the content box is
    /// taken as the border box, and the device-pixel size equals
    /// the content size (device pixel ratio 1 — Pluto has no DPR
    /// plumbing yet).
    pub fn from_border(border: (f64, f64)) -> Self {
        Self {
            border,
            content: border,
            device_pixel: border,
            content_offset: (0.0, 0.0),
        }
    }

    /// The size the observed box kind watches.
    pub fn observed(&self, kind: ResizeBoxKind) -> (f64, f64) {
        match kind {
            ResizeBoxKind::ContentBox => self.content,
            ResizeBoxKind::BorderBox => self.border,
            ResizeBoxKind::DevicePixelContentBox => self.device_pixel,
        }
    }
}

/// One entry to deliver to a `ResizeObserver` callback.
#[derive(Debug, Clone, PartialEq)]
pub struct ResizeObserverEntry {
    /// The observed element.
    pub target: DomKey,
    /// Its sizes at gather time.
    pub sizes: ObservedSizes,
    /// DOM depth of the target (used for shallowest-first order).
    pub depth: usize,
}

/// One observation: an observer watching one target.
#[derive(Debug, Clone)]
struct Observation {
    observer_id: u64,
    target: DomKey,
    kind: ResizeBoxKind,
    last_reported: Option<ObservedSizes>,
}

/// What one gather/broadcast step did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResizeStepOutcome {
    /// At least one callback batch was delivered.
    pub delivered: bool,
    /// Callbacks kept changing sizes past the round bound: the
    /// resize-loop error fired and undelivered observations remain
    /// active for the next step.
    pub loop_error: bool,
    /// How many broadcast rounds ran.
    pub rounds: usize,
}

/// Maximum broadcast rounds in one step before the resize-loop
/// error fires.
pub const MAX_ROUNDS: usize = 5;

/// The `ResizeObserver` registry: every observation of every
/// observer in one tab, plus their `lastReportedSizes`.
#[derive(Debug, Default)]
pub struct ResizeObserverRegistry {
    observations: Vec<Observation>,
    /// Observer ids in creation order (first `observe()` call).
    observer_order: Vec<u64>,
}

impl ResizeObserverRegistry {
    /// Observe `target` with `observer_id` (replaces an existing
    /// observation for the same observer + target, per spec).
    pub fn observe(&mut self, observer_id: u64, target: DomKey, kind: ResizeBoxKind) {
        self.observations
            .retain(|o| !(o.observer_id == observer_id && o.target == target));
        self.observations.push(Observation {
            observer_id,
            target,
            kind,
            last_reported: None,
        });
        if !self.observer_order.contains(&observer_id) {
            self.observer_order.push(observer_id);
        }
    }

    /// Stop observing `target` for `observer_id`.
    pub fn unobserve(&mut self, observer_id: u64, target: DomKey) {
        self.observations
            .retain(|o| !(o.observer_id == observer_id && o.target == target));
    }

    /// Drop every observation of `observer_id`.
    pub fn disconnect(&mut self, observer_id: u64) {
        self.observations.retain(|o| o.observer_id != observer_id);
    }

    /// Every (observer, target, box) currently observed, in
    /// registration order. The browser's worker step uses this to
    /// ask the layout actor for exactly these targets' sizes.
    pub fn targets(&self) -> Vec<(u64, DomKey, ResizeBoxKind)> {
        self.observations
            .iter()
            .map(|o| (o.observer_id, o.target, o.kind))
            .collect()
    }

    /// True when any observation exists.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// One broadcast round: gather the active observations and
    /// update their `lastReportedSizes` to the delivered sizes.
    /// Returns the batches to deliver, grouped by observer in
    /// observer-creation order, entries shallowest-first within a
    /// batch. Returns an empty vec when nothing is active.
    ///
    /// `sizes` answers the current sizes for a target (`None` =
    /// unrendered — never active); `depth` answers its DOM depth.
    pub fn gather_round(
        &mut self,
        sizes: &dyn Fn(DomKey) -> Option<ObservedSizes>,
        depth: &dyn Fn(DomKey) -> usize,
    ) -> Vec<(u64, Vec<ResizeObserverEntry>)> {
        let mut active: Vec<(usize, usize, u64, ResizeObserverEntry)> = Vec::new();
        for obs in self.observations.iter() {
            let Some(size) = sizes(obs.target) else {
                continue;
            };
            let observed = size.observed(obs.kind);
            let is_active = match obs.last_reported {
                Some(last) => last.observed(obs.kind) != observed,
                // Initial entry: only for a rendered, non-zero box.
                None => observed != (0.0, 0.0),
            };
            if is_active {
                active.push((
                    depth(obs.target),
                    self.observer_order
                        .iter()
                        .position(|&id| id == obs.observer_id)
                        .unwrap_or(usize::MAX),
                    obs.observer_id,
                    ResizeObserverEntry {
                        target: obs.target,
                        sizes: size,
                        depth: depth(obs.target),
                    },
                ));
            }
        }
        // Shallowest first; ties break by observer creation order.
        active.sort_by_key(|(d, o, _, _)| (*d, *o));
        let mut batches: Vec<(u64, Vec<ResizeObserverEntry>)> = Vec::new();
        for (_, _, observer_id, entry) in active {
            // Update lastReported to the delivered size now, so a
            // re-gather after the callback only sees observations
            // the callback actually changed.
            if let Some(obs) = self
                .observations
                .iter_mut()
                .find(|o| o.observer_id == observer_id && o.target == entry.target)
            {
                obs.last_reported = Some(entry.sizes);
            }
            match batches.iter_mut().find(|(id, _)| *id == observer_id) {
                Some((_, entries)) => entries.push(entry),
                None => batches.push((observer_id, vec![entry])),
            }
        }
        // Batches in observer-creation order.
        batches.sort_by_key(|(id, _)| {
            self.observer_order
                .iter()
                .position(|o| o == id)
                .unwrap_or(usize::MAX)
        });
        batches
    }

    /// True when any observation is currently active (its size
    /// differs from `lastReportedSizes`, or it has never been
    /// reported and is non-zero). Used between rounds to decide
    /// whether another round is needed.
    pub fn has_active(&self, sizes: &dyn Fn(DomKey) -> Option<ObservedSizes>) -> bool {
        self.observations.iter().any(|obs| {
            let Some(size) = sizes(obs.target) else {
                return false;
            };
            let observed = size.observed(obs.kind);
            match obs.last_reported {
                Some(last) => last.observed(obs.kind) != observed,
                None => observed != (0.0, 0.0),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Fake DomKeys: the registry only needs distinct keys, so build
    // them from a real Dom (slotmap keys cannot be fabricated).
    fn keys() -> (DomKey, DomKey, DomKey) {
        let mut dom = pluto_html::Dom::new();
        let a = dom.create_element("div");
        let b = dom.create_element("div");
        let c = dom.create_element("div");
        (a, b, c)
    }

    fn sizes_map(
        entries: Vec<(DomKey, ObservedSizes)>,
    ) -> Rc<RefCell<HashMap<DomKey, ObservedSizes>>> {
        Rc::new(RefCell::new(entries.into_iter().collect()))
    }

    #[test]
    fn initial_entry_then_no_spurious_second_round() {
        let (a, _, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        let sizes = sizes_map(vec![(a, ObservedSizes::from_border((100.0, 50.0)))]);
        let batches = reg.gather_round(&|k| sizes.borrow().get(&k).copied(), &|_| 0);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0, 1);
        assert_eq!(batches[0].1.len(), 1);
        assert_eq!(batches[0].1[0].sizes.border, (100.0, 50.0));
        // Same sizes: nothing active any more.
        assert!(
            reg.gather_round(&|k| sizes.borrow().get(&k).copied(), &|_| 0)
                .is_empty()
        );
        assert!(!reg.has_active(&|k| sizes.borrow().get(&k).copied()));
    }

    #[test]
    fn size_change_detected_and_last_reported_updated() {
        let (a, _, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::ContentBox);
        let sizes = sizes_map(vec![(a, ObservedSizes::from_border((100.0, 50.0)))]);
        let provider = |k: DomKey| sizes.borrow().get(&k).copied();
        assert_eq!(reg.gather_round(&provider, &|_| 0).len(), 1);
        sizes
            .borrow_mut()
            .insert(a, ObservedSizes::from_border((200.0, 50.0)));
        let batches = reg.gather_round(&provider, &|_| 0);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].1[0].sizes.content, (200.0, 50.0));
        // Delivered: quiet again at the new size.
        assert!(reg.gather_round(&provider, &|_| 0).is_empty());
    }

    #[test]
    fn zero_size_gets_no_initial_entry_but_resize_to_zero_fires() {
        let (a, _, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        let sizes = sizes_map(vec![(a, ObservedSizes::from_border((0.0, 0.0)))]);
        let provider = |k: DomKey| sizes.borrow().get(&k).copied();
        assert!(reg.gather_round(&provider, &|_| 0).is_empty());
        sizes
            .borrow_mut()
            .insert(a, ObservedSizes::from_border((80.0, 40.0)));
        assert_eq!(reg.gather_round(&provider, &|_| 0).len(), 1);
        sizes
            .borrow_mut()
            .insert(a, ObservedSizes::from_border((0.0, 0.0)));
        let batches = reg.gather_round(&provider, &|_| 0);
        assert_eq!(batches.len(), 1, "resize to zero fires");
        assert_eq!(batches[0].1[0].sizes.border, (0.0, 0.0));
    }

    #[test]
    fn unrendered_target_never_fires() {
        let (a, _, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        let sizes: Rc<RefCell<HashMap<DomKey, ObservedSizes>>> =
            Rc::new(RefCell::new(HashMap::new()));
        assert!(
            reg.gather_round(&|k| sizes.borrow().get(&k).copied(), &|_| 0)
                .is_empty()
        );
    }

    #[test]
    fn batches_group_by_observer_in_creation_order_shallowest_first() {
        let (a, b, c) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(2, c, ResizeBoxKind::BorderBox);
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        reg.observe(1, b, ResizeBoxKind::BorderBox);
        let sizes = sizes_map(vec![
            (a, ObservedSizes::from_border((10.0, 10.0))),
            (b, ObservedSizes::from_border((20.0, 20.0))),
            (c, ObservedSizes::from_border((30.0, 30.0))),
        ]);
        let depth = |k: DomKey| if k == b { 2 } else { 1 };
        let batches = reg.gather_round(&|k| sizes.borrow().get(&k).copied(), &depth);
        assert_eq!(batches.len(), 2);
        // Observer 2 was created first, so its batch comes first…
        assert_eq!(batches[0].0, 2);
        assert_eq!(batches[1].0, 1);
        // …and within observer 1, the shallower target comes first.
        assert_eq!(batches[1].1[0].target, a);
        assert_eq!(batches[1].1[1].target, b);
    }

    #[test]
    fn disconnect_and_unobserve_stop_delivery() {
        let (a, b, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        reg.observe(1, b, ResizeBoxKind::BorderBox);
        reg.unobserve(1, a);
        let sizes = sizes_map(vec![
            (a, ObservedSizes::from_border((10.0, 10.0))),
            (b, ObservedSizes::from_border((20.0, 20.0))),
        ]);
        let batches = reg.gather_round(&|k| sizes.borrow().get(&k).copied(), &|_| 0);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].1[0].target, b);
        reg.disconnect(1);
        assert!(reg.is_empty());
        assert!(reg.targets().is_empty());
    }

    #[test]
    fn box_kind_selects_which_size_is_watched() {
        let (a, _, _) = keys();
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, a, ResizeBoxKind::BorderBox);
        let mut full = ObservedSizes::from_border((100.0, 50.0));
        full.content = (90.0, 40.0);
        let sizes = sizes_map(vec![(a, full)]);
        let provider = |k: DomKey| sizes.borrow().get(&k).copied();
        assert_eq!(reg.gather_round(&provider, &|_| 0).len(), 1);
        // Content changes, border does not: border-box observation
        // stays quiet.
        let mut changed = full;
        changed.content = (95.0, 40.0);
        sizes.borrow_mut().insert(a, changed);
        assert!(reg.gather_round(&provider, &|_| 0).is_empty());
        // Border changes: fires.
        let mut changed2 = changed;
        changed2.border = (110.0, 50.0);
        sizes.borrow_mut().insert(a, changed2);
        assert_eq!(reg.gather_round(&provider, &|_| 0).len(), 1);
    }
}

// ------------------------------------------------------------------ IntersectionObserver

/// One `rootMargin` edge: px, or % of the root's width (left/right)
///
/// / height (top/bottom).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginValue {
    /// Absolute CSS px (may be negative — margins may shrink the root).
    Px(f64),
    /// Percentage of the root's width (horizontal edges) or height
    /// (vertical edges), as a fraction (0.1 = 10%).
    Percent(f64),
}

impl Default for MarginValue {
    fn default() -> Self {
        MarginValue::Px(0.0)
    }
}

impl MarginValue {
    /// Resolve against `whole` (the root's width or height, per edge).
    fn resolve(self, whole: f64) -> f64 {
        match self {
            MarginValue::Px(v) => v,
            MarginValue::Percent(f) => f * whole,
        }
    }
}

/// `IntersectionObserver`'s `rootMargin`: a CSS margin shorthand
/// expanding (or, when negative, shrinking) the root before the
/// intersection is computed. Percentages resolve against the root's
/// width (left/right) or height (top/bottom).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RootMargin {
    pub top: MarginValue,
    pub right: MarginValue,
    pub bottom: MarginValue,
    pub left: MarginValue,
}

impl RootMargin {
    /// Parse a CSS margin shorthand: 1–4 whitespace-separated
    /// values, each `<number>px`, `<number>%`, or a bare `0`.
    /// Negative values are allowed. `None` on any invalid token.
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_ascii_whitespace().collect();
        if parts.is_empty() || parts.len() > 4 {
            return None;
        }
        let mut values = Vec::with_capacity(4);
        for part in &parts {
            values.push(parse_margin_value(part)?);
        }
        let m = match values.len() {
            1 => Self {
                top: values[0],
                right: values[0],
                bottom: values[0],
                left: values[0],
            },
            2 => Self {
                top: values[0],
                right: values[1],
                bottom: values[0],
                left: values[1],
            },
            3 => Self {
                top: values[0],
                right: values[1],
                bottom: values[2],
                left: values[1],
            },
            _ => Self {
                top: values[0],
                right: values[1],
                bottom: values[2],
                left: values[3],
            },
        };
        Some(m)
    }

    /// Expand `rect` (`[x, y, width, height]`) by these margins.
    /// Percentages resolve against the rect's own width/height.
    /// The result may have negative width/height (margins larger
    /// than the root) — the geometry step treats that as an empty
    /// intersection.
    pub fn expand(&self, rect: [f64; 4]) -> [f64; 4] {
        let top = self.top.resolve(rect[3]);
        let right = self.right.resolve(rect[2]);
        let bottom = self.bottom.resolve(rect[3]);
        let left = self.left.resolve(rect[2]);
        [
            rect[0] - left,
            rect[1] - top,
            rect[2] + left + right,
            rect[3] + top + bottom,
        ]
    }
}

/// Parse one margin token: `<number>px`, `<number>%`, or a bare
/// `0` (the only bare number CSS allows for a length).
fn parse_margin_value(token: &str) -> Option<MarginValue> {
    if let Some(num) = token.strip_suffix("px") {
        num.parse::<f64>().ok().map(MarginValue::Px)
    } else if let Some(num) = token.strip_suffix('%') {
        num.parse::<f64>().ok().map(|v| MarginValue::Percent(v / 100.0))
    } else {
        match token.parse::<f64>() {
            Ok(0.0) => Some(MarginValue::Px(0.0)),
            _ => None,
        }
    }
}

/// The browser-side geometry for one observed target, computed on
/// the layout actor's retained tree (see
/// `pluto-browser::observers::intersection_geometry_for`): the
/// target's border box and the (margin-expanded) root bounds in
/// page coordinates, the clipped intersection, and the ratio.
/// `is_intersecting` follows the spec text — true on any overlap
/// (ratio > 0) — not the Chrome/Firefox quirk the Stage 8 report
/// flags in its open questions (§248: those engines keep
/// `isIntersecting` false until the ratio reaches the lowest
/// non-zero threshold; Pluto follows the spec).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputedIntersection {
    /// The target's border box `[x, y, width, height]`, page coords.
    pub target_rect: [f64; 4],
    /// The root's bounds after `rootMargin`, page coords.
    pub root_bounds: [f64; 4],
    /// The clipped intersection `[x, y, width, height]`, page
    /// coords; `[0, 0, 0, 0]` when there is no overlap.
    pub intersection_rect: [f64; 4],
    /// `intersection area / target area`, 0..=1 (0 for a zero-area
    /// target, which never intersects).
    pub ratio: f64,
    /// True when the ratio is greater than 0.
    pub is_intersecting: bool,
}

impl ComputedIntersection {
    /// Shift the rects from page coordinates into
    /// viewport-relative coordinates (Stage 16 Phase 1, Rec 10):
    /// the spec's `boundingClientRect` / `rootBounds` /
    /// `intersectionRect` are all relative to the viewport, so
    /// the viewport scroll offset comes off each one. An empty
    /// intersection (`[0, 0, 0, 0]`, no overlap) stays empty —
    /// shifting it would invent a position for a rect that has
    /// none. The ratio is scroll-independent and untouched.
    pub fn to_viewport_relative(&self, scroll: (f32, f32)) -> Self {
        let (sx, sy) = (scroll.0 as f64, scroll.1 as f64);
        let shift = |r: [f64; 4]| [r[0] - sx, r[1] - sy, r[2], r[3]];
        let intersection_rect =
            if self.intersection_rect[2] > 0.0 && self.intersection_rect[3] > 0.0 {
                shift(self.intersection_rect)
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
        Self {
            target_rect: shift(self.target_rect),
            root_bounds: shift(self.root_bounds),
            intersection_rect,
            ratio: self.ratio,
            is_intersecting: self.is_intersecting,
        }
    }
}

/// One entry to deliver to an `IntersectionObserver` callback.
#[derive(Debug, Clone, PartialEq)]
pub struct IntersectionObserverEntry {
    /// The observed element.
    pub target: DomKey,
    /// Timestamp of the computation (ms since the tab started —
    /// the worker passes its tab clock), not of delivery.
    pub time: f64,
    /// The root's bounds after `rootMargin` (the viewport's
    /// visible region for a viewport root), VIEWPORT-relative
    /// (Stage 16 Phase 1, Rec 10 — the spec's rects are
    /// viewport-relative, not page coordinates).
    pub root_bounds: [f64; 4],
    /// The target's border box, viewport-relative.
    pub bounding_client_rect: [f64; 4],
    /// The clipped intersection, viewport-relative
    /// (`[0, 0, 0, 0]` when not intersecting).
    pub intersection_rect: [f64; 4],
    /// `intersection area / target area`.
    pub ratio: f64,
    /// True when the ratio is greater than 0.
    pub is_intersecting: bool,
}

/// One observation: an observer watching one target against one
/// root (or the viewport).
#[derive(Debug, Clone)]
struct IntersectionObservation {
    observer_id: u64,
    target: DomKey,
    root: Option<DomKey>,
    root_margin: RootMargin,
    /// Thresholds, sorted and deduplicated (an entry is queued
    /// when the ratio crosses any of them).
    thresholds: Vec<f64>,
    /// `(ratio, is_intersecting)` of the last delivered entry, or
    /// `None` when no entry has been delivered yet (the first
    /// computed geometry always delivers — the initial entry).
    last_reported: Option<(f64, bool)>,
}

/// The options one `IntersectionObserver` was constructed with
/// (Stage 16 Phase 1, Rec 10): per OBSERVER, not per `observe()`
/// call — the spec parses `root` / `rootMargin` / `threshold` in
/// the constructor, and `observe(target)` takes no options.
/// Stored at construction so `observe()` can look them up.
#[derive(Debug, Clone)]
pub struct IntersectionObserverOptions {
    pub root: Option<DomKey>,
    pub root_margin: RootMargin,
    /// Thresholds, sorted and deduplicated.
    pub thresholds: Vec<f64>,
}

impl Default for IntersectionObserverOptions {
    fn default() -> Self {
        Self {
            root: None,
            root_margin: RootMargin::default(),
            thresholds: vec![0.0],
        }
    }
}

/// The `IntersectionObserver` registry: every observation of every
/// observer in one tab, plus their `lastReported` ratios and the
/// pending entries awaiting the queued delivery task.
///
/// Delivery model (Stage 8 report §(e)): the worker's rendering
/// step computes the geometry after layout and calls [`Self::gather`];
/// entries that crossed a threshold band (or flipped
/// `isIntersecting`) land in the pending queue, and the caller
/// enqueues ONE host task that drains the queue — so the callbacks
/// run after paint, on a later turn, never inline in the step. A
/// callback that writes to the DOM therefore affects the *next*
/// frame, unlike a `ResizeObserver` callback (which may affect the
/// same frame).
#[derive(Debug, Default)]
pub struct IntersectionObserverRegistry {
    observations: Vec<IntersectionObservation>,
    /// Constructor options per observer id (Stage 16 Phase 1,
    /// Rec 10). Survives `disconnect()` — an observer can
    /// `observe()` again after disconnecting, with the same
    /// options it was constructed with.
    observer_options: std::collections::HashMap<u64, IntersectionObserverOptions>,
    /// Observer ids in creation order (first `observe()` call).
    observer_order: Vec<u64>,
    /// Entries gathered since the last delivery, grouped by
    /// observer in gather order.
    pending: Vec<(u64, Vec<IntersectionObserverEntry>)>,
    /// True while a delivery task is already queued for the pending
    /// entries (set by [`Self::gather`], cleared by
    /// [`Self::take_pending`]).
    delivery_queued: bool,
}

impl IntersectionObserverRegistry {
    /// Record an observer's constructor options (Stage 16 Phase 1,
    /// Rec 10). `thresholds` are sorted and deduplicated. Called
    /// from `__pluto_new`, before any `observe()`; an observer
    /// with no recorded options observes with the defaults
    /// (viewport root, no margin, threshold `[0]`).
    pub fn configure_observer(
        &mut self,
        observer_id: u64,
        root: Option<DomKey>,
        root_margin: RootMargin,
        thresholds: Vec<f64>,
    ) {
        let mut thresholds = thresholds;
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup_by(|a, b| *a == *b);
        self.observer_options.insert(
            observer_id,
            IntersectionObserverOptions {
                root,
                root_margin,
                thresholds,
            },
        );
    }

    /// Observe `target` with `observer_id` (replaces an existing
    /// observation for the same observer + target, per spec),
    /// using the options the observer was constructed with
    /// (Stage 16 Phase 1, Rec 10 — options live on the
    /// constructor, not on `observe()`).
    pub fn observe(&mut self, observer_id: u64, target: DomKey) {
        let options = self
            .observer_options
            .get(&observer_id)
            .cloned()
            .unwrap_or_default();
        self.observations
            .retain(|o| !(o.observer_id == observer_id && o.target == target));
        self.observations.push(IntersectionObservation {
            observer_id,
            target,
            root: options.root,
            root_margin: options.root_margin,
            thresholds: options.thresholds,
            last_reported: None,
        });
        if !self.observer_order.contains(&observer_id) {
            self.observer_order.push(observer_id);
        }
    }

    /// Stop observing `target` for `observer_id`. Already-queued
    /// pending entries for that observer + target are dropped —
    /// `unobserve()` must not deliver an entry for a target the
    /// page just stopped watching (Stage 16 Phase 1, Rec 10).
    pub fn unobserve(&mut self, observer_id: u64, target: DomKey) {
        self.observations
            .retain(|o| !(o.observer_id == observer_id && o.target == target));
        self.drop_pending_for(observer_id, Some(target));
    }

    /// Drop every observation of `observer_id`, and every
    /// already-queued pending entry for it (Stage 16 Phase 1,
    /// Rec 10). The constructor options survive, so the observer
    /// can `observe()` again with the same options.
    pub fn disconnect(&mut self, observer_id: u64) {
        self.observations.retain(|o| o.observer_id != observer_id);
        self.drop_pending_for(observer_id, None);
    }

    /// Drop pending entries for `observer_id` — all of them when
    /// `target` is `None`, only that target's when `Some`.
    fn drop_pending_for(&mut self, observer_id: u64, target: Option<DomKey>) {
        for (id, entries) in self.pending.iter_mut() {
            if *id == observer_id {
                match target {
                    Some(t) => entries.retain(|e| e.target != t),
                    None => entries.clear(),
                }
            }
        }
        self.pending.retain(|(_, entries)| !entries.is_empty());
        if self.pending.is_empty() {
            // Nothing left to deliver: the queued task (if any)
            // drains an empty queue, so the flag can clear now
            // and the next gather re-arms it.
            self.delivery_queued = false;
        }
    }

    /// True when any observation exists.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Every (observer, target, root, rootMargin) currently
    /// observed, in registration order. The browser's worker step
    /// uses this to ask the layout actor for exactly these
    /// targets' intersections.
    pub fn targets(&self) -> Vec<(u64, DomKey, Option<DomKey>, RootMargin)> {
        self.observations
            .iter()
            .map(|o| (o.observer_id, o.target, o.root, o.root_margin))
            .collect()
    }

    /// Gather entries from the freshly computed geometry: an entry
    /// is queued only when the target crosses a threshold band or
    /// flips `isIntersecting` — except the first computed geometry
    /// for an observation, which always delivers (the initial
    /// entry). Unrendered targets (`geometry` returns `None`) get
    /// no entry, and their initial entry still waits for the first
    /// rendered geometry.
    ///
    /// Queued entries land in the pending queue. Returns `true`
    /// when the caller must enqueue a delivery task (entries were
    /// queued and no task is already pending); `false` when
    /// nothing was queued, or a task already covers the queue.
    pub fn gather(
        &mut self,
        geometry: &dyn Fn(u64, DomKey) -> Option<ComputedIntersection>,
        time_ms: f64,
    ) -> bool {
        let mut queued_any = false;
        for obs in self.observations.iter_mut() {
            let Some(geo) = geometry(obs.observer_id, obs.target) else {
                continue;
            };
            let active = match obs.last_reported {
                // Initial entry: always.
                None => true,
                Some((prev_ratio, prev_is)) => {
                    prev_is != geo.is_intersecting
                        || obs
                            .thresholds
                            .iter()
                            .any(|&t| (prev_ratio < t) != (geo.ratio < t))
                }
            };
            if !active {
                continue;
            }
            obs.last_reported = Some((geo.ratio, geo.is_intersecting));
            let entry = IntersectionObserverEntry {
                target: obs.target,
                time: time_ms,
                root_bounds: geo.root_bounds,
                bounding_client_rect: geo.target_rect,
                intersection_rect: geo.intersection_rect,
                ratio: geo.ratio,
                is_intersecting: geo.is_intersecting,
            };
            match self
                .pending
                .iter_mut()
                .find(|(id, _)| *id == obs.observer_id)
            {
                Some((_, entries)) => entries.push(entry),
                None => self.pending.push((obs.observer_id, vec![entry])),
            }
            queued_any = true;
        }
        // Batches in observer-creation order.
        self.pending.sort_by_key(|(id, _)| {
            self.observer_order
                .iter()
                .position(|o| o == id)
                .unwrap_or(usize::MAX)
        });
        if queued_any && !self.delivery_queued {
            self.delivery_queued = true;
            true
        } else {
            false
        }
    }

    /// True when entries are waiting for the delivery task.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Drain every pending entry, grouped by observer in
    /// observer-creation order (entries for one observer from
    /// several gathers merge into one batch, so the callback is
    /// invoked once per observer per task). Clears the
    /// delivery-task flag.
    pub fn take_pending(&mut self) -> Vec<(u64, Vec<IntersectionObserverEntry>)> {
        self.delivery_queued = false;
        std::mem::take(&mut self.pending)
    }

    /// Drain `observer_id`'s pending entries (`takeRecords()`):
    /// the entries gathered for that observer since the last
    /// delivery, in gather order.
    pub fn take_records(&mut self, observer_id: u64) -> Vec<IntersectionObserverEntry> {
        let mut out = Vec::new();
        self.pending.retain(|(id, entries)| {
            if *id == observer_id {
                out.extend(entries.iter().cloned());
                false
            } else {
                true
            }
        });
        out
    }
}

#[cfg(test)]
mod intersection_tests {
    use super::*;

    fn keys() -> (DomKey, DomKey) {
        let mut dom = pluto_html::Dom::new();
        let a = dom.create_element("div");
        let b = dom.create_element("div");
        (a, b)
    }

    fn geo(ratio: f64, is: bool) -> ComputedIntersection {
        ComputedIntersection {
            target_rect: [0.0, 0.0, 100.0, 100.0],
            root_bounds: [0.0, 0.0, 1024.0, 768.0],
            intersection_rect: [0.0, 0.0, 100.0, 100.0 * ratio],
            ratio,
            is_intersecting: is,
        }
    }

    fn geo_map(pairs: Vec<(DomKey, ComputedIntersection)>) -> std::collections::HashMap<DomKey, ComputedIntersection> {
        pairs.into_iter().collect()
    }

    #[test]
    fn root_margin_parses_shorthand() {
        let all = RootMargin::parse("10px").expect("one value");
        assert_eq!(all.top, MarginValue::Px(10.0));
        assert_eq!(all.right, MarginValue::Px(10.0));
        assert_eq!(all.bottom, MarginValue::Px(10.0));
        assert_eq!(all.left, MarginValue::Px(10.0));
        let two = RootMargin::parse("10px 20%").expect("two values");
        assert_eq!(two.top, MarginValue::Px(10.0));
        assert_eq!(two.right, MarginValue::Percent(0.2));
        assert_eq!(two.bottom, MarginValue::Px(10.0));
        assert_eq!(two.left, MarginValue::Percent(0.2));
        let three = RootMargin::parse("1px 2px 3px").expect("three values");
        assert_eq!((three.top, three.right, three.bottom, three.left), (
            MarginValue::Px(1.0),
            MarginValue::Px(2.0),
            MarginValue::Px(3.0),
            MarginValue::Px(2.0),
        ));
        let four = RootMargin::parse("1px 2px 3px 4px").expect("four values");
        assert_eq!(four.left, MarginValue::Px(4.0));
        // Bare zero is a valid length; negative values are allowed.
        assert_eq!(RootMargin::parse("0").expect("bare zero").top, MarginValue::Px(0.0));
        assert_eq!(
            RootMargin::parse("-5px").expect("negative").top,
            MarginValue::Px(-5.0)
        );
        assert!(RootMargin::parse("").is_none());
        assert!(RootMargin::parse("1px 2px 3px 4px 5px").is_none());
        assert!(RootMargin::parse("10").is_none());
        assert!(RootMargin::parse("10em").is_none());
        assert!(RootMargin::parse("abc").is_none());
    }

    #[test]
    fn root_margin_expands_rect_with_percentages() {
        let m = RootMargin::parse("10% 20px").expect("margin");
        // % resolves against width (left/right) / height (top/bottom).
        let expanded = m.expand([100.0, 100.0, 200.0, 100.0]);
        assert_eq!(expanded, [80.0, 90.0, 240.0, 120.0]);
        let shrink = RootMargin::parse("-10px").expect("negative margin");
        assert_eq!(shrink.expand([0.0, 0.0, 100.0, 100.0]), [10.0, 10.0, 80.0, 80.0]);
    }

    #[test]
    fn intersection_initial_entry_then_quiet() {
        let (a, _) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.0]);
        reg.observe(1, a);
        let map = geo_map(vec![(a, geo(1.0, true))]);
        let provider = |_: u64, k: DomKey| map.get(&k).copied();
        assert!(reg.gather(&provider, 100.0), "initial entry queues a task");
        assert!(!reg.gather(&provider, 200.0), "identical geometry: quiet");
        let batches = reg.take_pending();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].1.len(), 1, "one initial entry");
        assert_eq!(batches[0].1[0].ratio, 1.0);
        assert_eq!(batches[0].1[0].time, 100.0, "time is the computation time");
        assert!(batches[0].1[0].is_intersecting);
        assert!(!reg.has_pending());
    }

    #[test]
    fn intersection_threshold_band_crossing() {
        let (a, _) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.5]);
        reg.observe(1, a);
        // Initial entry at ratio 0 (out of view).
        let mut map = geo_map(vec![(a, geo(0.0, false))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 0.0));
        reg.take_pending();
        // 0.25: isIntersecting flips false -> true, so an entry
        // fires even though no threshold was crossed.
        map.insert(a, geo(0.25, true));
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 16.0), "flip fires");
        reg.take_pending();
        // 0.4: same band, still intersecting — quiet.
        map.insert(a, geo(0.4, true));
        assert!(!reg.gather(&|_, k| map.get(&k).copied(), 32.0));
        // 0.6: crosses 0.5 upward — fires.
        map.insert(a, geo(0.6, true));
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 48.0), "crossing up fires");
        reg.take_pending();
        // 0.55: same band — quiet.
        map.insert(a, geo(0.55, true));
        assert!(!reg.gather(&|_, k| map.get(&k).copied(), 64.0));
        // 0.4: crosses 0.5 downward — fires.
        map.insert(a, geo(0.4, true));
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 80.0), "crossing down fires");
        reg.take_pending();
        // 0.0: flips back to not intersecting — fires.
        map.insert(a, geo(0.0, false));
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 96.0), "flip back fires");
        let batches = reg.take_pending();
        assert_eq!(batches[0].1[0].ratio, 0.0);
        assert!(!batches[0].1[0].is_intersecting);
    }

    #[test]
    fn intersection_unrendered_target_gets_no_entry() {
        let (a, _) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.0]);
        reg.observe(1, a);
        // No geometry at all: nothing queued, and the initial
        // entry still waits.
        assert!(!reg.gather(&|_, _| None, 0.0));
        assert!(!reg.has_pending());
        // Once rendered, the initial entry fires.
        let map = geo_map(vec![(a, geo(0.5, true))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 16.0));
    }

    #[test]
    fn intersection_take_records_drains_pending() {
        let (a, b) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.0]);
        reg.observe(1, a);
        reg.configure_observer(2, None, RootMargin::default(), vec![0.0]);
        reg.observe(2, b);
        let map = geo_map(vec![(a, geo(1.0, true)), (b, geo(0.0, false))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 0.0));
        let recs = reg.take_records(1);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].target, a);
        // Observer 2's entry is untouched.
        assert!(reg.has_pending());
        let batches = reg.take_pending();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0, 2);
    }

    #[test]
    fn intersection_disconnect_and_unobserve_drop_pending() {
        let (a, b) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.0]);
        reg.observe(1, a);
        reg.configure_observer(2, None, RootMargin::default(), vec![0.0]);
        reg.observe(2, b);
        let map = geo_map(vec![(a, geo(1.0, true)), (b, geo(1.0, true))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 0.0));
        assert!(reg.has_pending());
        reg.disconnect(1);
        // Observer 1's entries are gone; observer 2's remain.
        let batches = reg.take_pending();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0, 2);
        // A fresh entry for observer 2, then unobserve drops it.
        let map = geo_map(vec![(b, geo(0.0, false))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 16.0));
        assert!(reg.has_pending());
        reg.unobserve(2, b);
        assert!(!reg.has_pending());
        assert!(reg.take_pending().is_empty());
    }

    #[test]
    fn intersection_observe_without_configure_uses_defaults() {
        // `observe()` with no recorded constructor options uses
        // the defaults (viewport root, threshold [0]).
        let (a, _) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.observe(1, a);
        let map = geo_map(vec![(a, geo(1.0, true))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 0.0));
        reg.take_pending();
        // 0.25: no threshold in [0] is crossed and
        // isIntersecting stays true — quiet.
        let map = geo_map(vec![(a, geo(0.25, true))]);
        assert!(!reg.gather(&|_, k| map.get(&k).copied(), 16.0));
    }

    #[test]
    fn computed_intersection_to_viewport_relative() {
        let g = ComputedIntersection {
            target_rect: [10.0, 200.0, 100.0, 50.0],
            root_bounds: [0.0, 100.0, 1024.0, 768.0],
            intersection_rect: [10.0, 200.0, 100.0, 50.0],
            ratio: 1.0,
            is_intersecting: true,
        };
        let v = g.to_viewport_relative((5.0, 100.0));
        assert_eq!(v.target_rect, [5.0, 100.0, 100.0, 50.0]);
        assert_eq!(v.root_bounds, [-5.0, 0.0, 1024.0, 768.0]);
        assert_eq!(v.intersection_rect, [5.0, 100.0, 100.0, 50.0]);
        assert_eq!(v.ratio, 1.0);
        // An empty intersection stays empty, not shifted.
        let g = ComputedIntersection {
            target_rect: [10.0, 200.0, 100.0, 50.0],
            root_bounds: [0.0, 100.0, 1024.0, 768.0],
            intersection_rect: [0.0, 0.0, 0.0, 0.0],
            ratio: 0.0,
            is_intersecting: false,
        };
        let v = g.to_viewport_relative((5.0, 100.0));
        assert_eq!(v.intersection_rect, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(v.target_rect, [5.0, 100.0, 100.0, 50.0]);
    }

    #[test]
    fn intersection_second_gather_before_delivery_merges() {
        let (a, _) = keys();
        let mut reg = IntersectionObserverRegistry::default();
        reg.configure_observer(1, None, RootMargin::default(), vec![0.0, 0.5]);
        reg.observe(1, a);
        let mut map = geo_map(vec![(a, geo(0.0, false))]);
        assert!(reg.gather(&|_, k| map.get(&k).copied(), 0.0), "first gather queues a task");
        // A second gather before the task runs: the new entry joins
        // the pending queue, and no second task is requested.
        map.insert(a, geo(1.0, true));
        assert!(!reg.gather(&|_, k| map.get(&k).copied(), 16.0), "task already queued");
        let batches = reg.take_pending();
        assert_eq!(batches.len(), 1, "one batch per observer per task");
        assert_eq!(batches[0].1.len(), 2, "both entries in the one batch");
    }
}
