//! The shell's compositor-side holder (Stage 4 #3, Stage 8 Rec 2):
//! pending/active display lists + the deadline rule, epoch'd.
//!
//! The worker thread builds display lists; the holder is the only
//! place the shell keeps one. An incoming list goes to the
//! *pending* slot, never straight to the screen: it activates only
//! at the frame boundary — when the shell builds the next frame's
//! widget tree and reads the holder (`display()`/`resources()`).
//! The swap is one `take` + replace, so a frame never sees half
//! a list, and the in-flight `Primitive` keeps its old `Arc`
//! until the next frame. A newer submit supersedes an older
//! pending list that never activated (only the newest is ever
//! drawn); an older-or-equal epoch is ignored outright, so a slow
//! worker can never resurrect a stale frame.
//!
//! The deadline rule: when a frame is built while a new list is
//! awaited (a navigation or resize asked for one and it has not
//! landed yet), the old active list is presented with the fresh
//! scroll offset — the scroll is read from the holder at frame
//! time, independent of the list, so a missed deadline costs a
//! stale frame, never a frozen scroll. Each such frame counts
//! into `frames_deadline_missed` in the process perf counters.
//!
//! Scroll lives here too, so a wheel event never round-trips to
//! layout: the viewport widget applies the offset as a view
//! transform (the renderer subtracts it in the vertex shader, and
//! the canvas fill is translated the other way so it stays glued
//! to the viewport). The offset is clamped to `[0, max]` on both
//! axes (Stage 10 Rec 2): the maximum comes from the submitted
//! display list — item 0 is the canvas fill over the viewport, and
//! the list's damage reaches `max(viewport, content)`, so `damage
//! - canvas` is the page's `content - viewport`, per axis. Until
//! a list lands the maximum is unbounded (no page yet); a list
//! that carries no canvas or damage leaves the previous maximum
//! in place.

use pluto_paint::{perf, DisplayItem, DisplayList, ResourceTable};
use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// A worker-built list waiting for the next frame boundary.
#[derive(Debug, Clone)]
struct PendingList {
    epoch: u64,
    display: Arc<DisplayList>,
    resources: Arc<ResourceTable>,
}

/// The active (drawn) list and its epoch, behind one
/// `RefCell`: activation happens on the `&self` read path
/// (`display()`/`resources()`, called from `view()`), and the
/// shell is single-threaded, so no reentrancy is possible. The
/// accessors return cloned `Arc`s (one refcount bump per frame)
/// because the `RefCell` borrow cannot outlive the call.
#[derive(Debug, Clone)]
struct ActiveList {
    display: Arc<DisplayList>,
    resources: Arc<ResourceTable>,
    epoch: u64,
}

/// The shell's view of the current page: what to draw, what its
/// image/blob ids resolve against, how new it is, and how far it
/// is scrolled.
#[derive(Debug, Clone)]
pub struct Compositor {
    active: RefCell<ActiveList>,
    /// The next list, waiting for the frame boundary. Same
    /// interior-mutability note as `active`.
    pending: RefCell<Option<PendingList>>,
    /// A navigation or resize asked for a new list that has not
    /// activated yet. Set by `note_expected_list` (shell-side,
    /// `&mut`), cleared by activation. While set, a frame built
    /// with no pending list is a deadline-missed frame.
    awaiting_new_list: Cell<bool>,
    /// Current viewport scroll offset, in page pixels. A `Cell`
    /// because `submit` (a `&self` path) re-clamps it when a new
    /// list lowers the maximum.
    scroll: Cell<(f32, f32)>,
    /// Maximum viewport scroll, per axis (`content - viewport`,
    /// clamped at zero — see the module docs). Unbounded until a
    /// display list lands: with no page, there is nothing to
    /// clamp against, and the worker's own clamp (its mirror of
    /// this offset) is the authority until then.
    max_scroll: Cell<(f32, f32)>,
}

impl Compositor {
    pub fn new() -> Self {
        Self {
            active: RefCell::new(ActiveList {
                display: Arc::new(DisplayList::new()),
                resources: Arc::new(ResourceTable::new()),
                epoch: 0,
            }),
            pending: RefCell::new(None),
            awaiting_new_list: Cell::new(false),
            scroll: Cell::new((0.0, 0.0)),
            max_scroll: Cell::new((f32::INFINITY, f32::INFINITY)),
        }
    }

    /// Offer a worker-built list. It lands in the pending slot
    /// when `epoch` is newer than the active one (and newer than
    /// any pending one — a newer submit supersedes an older
    /// pending list that never activated) and returns true. An
    /// older or equal epoch is ignored (a stale frame never
    /// replaces a newer one) and returns false. Nothing is drawn
    /// until the next frame boundary activates it.
    pub fn submit(
        &self,
        epoch: u64,
        display: Arc<DisplayList>,
        resources: Arc<ResourceTable>,
    ) -> bool {
        if epoch <= self.active.borrow().epoch {
            return false;
        }
        let mut pending = self.pending.borrow_mut();
        if let Some(current) = pending.as_ref()
            && epoch <= current.epoch
        {
            return false;
        }
        // A newer list is a newer page: its damage is the new
        // scroll maximum (a list that carries none leaves the
        // previous maximum in place). The stored scroll may now
        // exceed that maximum (a resize that shrank the page), so
        // re-clamp it here, at submit — leaving it stale until
        // the next scroll event self-heals it shows a blank gap
        // past the end of the content.
        if let Some(max) = max_scroll_for(&display) {
            self.max_scroll.set(max);
            let (sx, sy) = self.scroll.get();
            self.scroll
                .set((sx.clamp(0.0, max.0), sy.clamp(0.0, max.1)));
        }
        *pending = Some(PendingList {
            epoch,
            display,
            resources,
        });
        true
    }

    /// The epoch of the pending list, if one is waiting.
    /// Diagnostic API, exercised by tests (the shell reads the
    /// active epoch, not the pending one).
    #[allow(dead_code)]
    pub fn pending_epoch(&self) -> Option<u64> {
        self.pending.borrow().as_ref().map(|p| p.epoch)
    }

    /// Tell the holder a new list is expected — a navigation was
    /// committed or the viewport resized, so the next frames are
    /// deadline-missed frames until a list activates. Called by
    /// the shell (`&mut`, from `update()`); a JS-driven relayout
    /// the shell never hears about is counted worker-side instead.
    pub fn note_expected_list(&mut self) {
        self.awaiting_new_list.set(true);
    }

    /// Activate the pending list, if one is waiting: it becomes
    /// the active list, atomically, and the expectation clears.
    /// Returns true when a list activated. When no list is
    /// pending but one is awaited, the frame presents the old list
    /// — a deadline-missed frame — and the count goes up (the
    /// expectation stays until a list lands). Called implicitly by
    /// `display()`/`resources()` at the frame boundary; the miss
    /// is counted only on the `display()` call, so one frame
    /// counts once even though both accessors activate.
    pub fn activate(&self) -> bool {
        self.activate_inner(true)
    }

    fn activate_inner(&self, count_miss: bool) -> bool {
        let Some(next) = self.pending.borrow_mut().take() else {
            if count_miss
                && self.awaiting_new_list.get()
                && let Ok(mut counters) = perf::global().lock()
            {
                counters.frames_deadline_missed += 1;
            }
            return false;
        };
        let mut active = self.active.borrow_mut();
        active.epoch = next.epoch;
        active.display = next.display;
        active.resources = next.resources;
        self.awaiting_new_list.set(false);
        // A pending list activated at the frame boundary is a
        // represented frame (Stage 8 Rec 5, report §i) — the
        // counterpart to `frames_produced` in `prepare`: a list
        // superseded in pending is produced but never represented.
        if let Ok(mut counters) = perf::global().lock() {
            counters.frames_represented += 1;
        }
        true
    }

    /// Frames that presented the old list while a new one was
    /// awaited (the deadline rule working as intended).
    /// Diagnostic API, exercised by tests (the same count is
    /// printed by the perf summary line).
    #[allow(dead_code)]
    pub fn deadline_missed_frames(&self) -> u64 {
        perf::global()
            .lock()
            .map(|counters| counters.frames_deadline_missed)
            .unwrap_or(0)
    }

    /// The active display list, activating any pending list first
    /// (the frame boundary). The in-flight `Primitive` holds its
    /// own `Arc`, so this swap never affects a frame already
    /// being drawn.
    pub fn display(&self) -> Arc<DisplayList> {
        self.activate();
        Arc::clone(&self.active.borrow().display)
    }

    /// The active resource table, for the active display list.
    /// Activates any pending list first, like `display()` (the
    /// two are always read as a pair, display first, in `view()`),
    /// but without counting a deadline miss — `display()` already
    /// counted this frame's, if any.
    pub fn resources(&self) -> Arc<ResourceTable> {
        self.activate_inner(false);
        Arc::clone(&self.active.borrow().resources)
    }

    /// The epoch of the active (drawn) list.
    pub fn epoch(&self) -> u64 {
        self.active.borrow().epoch
    }

    /// Current scroll offset, in page pixels.
    pub fn scroll(&self) -> (f32, f32) {
        self.scroll.get()
    }

    /// Maximum scroll offset, in page pixels, from the newest
    /// submitted display list — `(∞, ∞)` until a list lands.
    /// Diagnostic API, exercised by tests.
    #[allow(dead_code)]
    pub fn max_scroll(&self) -> (f32, f32) {
        self.max_scroll.get()
    }

    /// Scroll by (dx, dy) page pixels, clamped to `[0, max]` on
    /// both axes, where `max` is the newest list's `content -
    /// viewport` (see the module docs). The worker's mirror of
    /// this offset applies the same clamp from the layout tree,
    /// so a delta the worker already clamped lands exactly.
    pub fn scroll_by(&mut self, dx: f32, dy: f32) {
        let (max_x, max_y) = self.max_scroll.get();
        let (sx, sy) = self.scroll.get();
        self.scroll.set((
            (sx + dx).clamp(0.0, max_x),
            (sy + dy).clamp(0.0, max_y),
        ));
    }

    /// Reset the scroll, e.g. on a new navigation.
    pub fn reset_scroll(&mut self) {
        self.scroll.set((0.0, 0.0));
    }
}

/// Maximum viewport scroll for a display list: the list's damage
/// is the union of the canvas (the viewport) and every box, so its
/// far edges are `max(viewport, content)` per axis — and item 0 is
/// the canvas fill over the viewport (`DisplayList::build` always
/// pushes it first), so `damage - canvas` is the page's `content -
/// viewport`, clamped at zero. `None` when the list carries no
/// damage or no canvas-shaped first item. A box-shadow sticking
/// out below the last box stretches the damage a few px past the
/// content, so this maximum can sit a hair above the worker's
/// `content_height`-based one — the shell's clamp is the looser
/// backstop, and the worker only ever sends deltas it already
/// clamped, so the two never disagree in practice.
fn max_scroll_for(display: &DisplayList) -> Option<(f32, f32)> {
    let (canvas_w, canvas_h) = match display.0.first() {
        Some(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width,
            height,
            ..
        }) if *width > 0.0 && *height > 0.0 => (*width, *height),
        _ => return None,
    };
    let damage = display.1?;
    let max_x = (damage.x + damage.width - canvas_w).max(0.0);
    let max_y = (damage.y + damage.height - canvas_h).max(0.0);
    Some((max_x, max_y))
}

impl Default for Compositor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_paint::DisplayItem;

    fn list_with_canvas(width: f32, height: f32) -> Arc<DisplayList> {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width,
            height,
            color: [1.0; 4],
        });
        Arc::new(list)
    }

    #[test]
    fn submit_goes_to_pending_and_activates_at_the_frame_boundary() {
        let holder = Compositor::new();
        assert_eq!(holder.epoch(), 0);
        assert!(holder.display().is_empty());
        let first = list_with_canvas(800.0, 600.0);
        // Submit lands in pending: the active list (and epoch)
        // do not change until a frame boundary activates it.
        assert!(holder.submit(1, Arc::clone(&first), Arc::new(ResourceTable::new())));
        assert_eq!(holder.epoch(), 0);
        assert_eq!(holder.pending_epoch(), Some(1));
        // Reading the holder IS the boundary: the pending list
        // activates and the read returns it.
        assert!(Arc::ptr_eq(&holder.display(), &first));
        assert_eq!(holder.epoch(), 1);
        assert_eq!(holder.pending_epoch(), None);
        // Same epoch again: ignored, the current list stays.
        assert!(!holder.submit(
            1,
            list_with_canvas(400.0, 300.0),
            Arc::new(ResourceTable::new())
        ));
        assert_eq!(holder.epoch(), 1);
        assert!(Arc::ptr_eq(&holder.display(), &first));
        // Older epoch: ignored too — a slow worker cannot
        // resurrect a stale frame.
        assert!(!holder.submit(
            0,
            list_with_canvas(100.0, 100.0),
            Arc::new(ResourceTable::new())
        ));
        assert!(Arc::ptr_eq(&holder.display(), &first));
        // A newer submit supersedes an older pending list that
        // never activated: only the newest is ever drawn.
        let superseded = list_with_canvas(640.0, 480.0);
        assert!(holder.submit(2, Arc::clone(&superseded), Arc::new(ResourceTable::new())));
        let second = list_with_canvas(1024.0, 768.0);
        assert!(holder.submit(3, Arc::clone(&second), Arc::new(ResourceTable::new())));
        assert_eq!(holder.pending_epoch(), Some(3));
        assert!(Arc::ptr_eq(&holder.display(), &second));
        assert_eq!(holder.epoch(), 3);
        assert!(!Arc::ptr_eq(&holder.display(), &superseded));
    }

    #[test]
    fn deadline_miss_counts_frames_presenting_the_old_list() {
        let mut holder = Compositor::new();
        let before = holder.deadline_missed_frames();
        // No list awaited: frames built on the initial empty list
        // are not deadline misses.
        let _ = holder.display();
        let _ = holder.display();
        assert_eq!(holder.deadline_missed_frames(), before);
        // A navigation expects a new list: every frame built
        // before it lands counts once (display + resources read
        // as a pair, but the frame counts once).
        holder.note_expected_list();
        let _ = holder.display();
        let _ = holder.resources();
        let _ = holder.display();
        assert_eq!(holder.deadline_missed_frames(), before + 2);
        // The list lands in pending, then activates on the next
        // read: the expectation clears and the counting stops.
        let first = list_with_canvas(800.0, 600.0);
        assert!(holder.submit(1, Arc::clone(&first), Arc::new(ResourceTable::new())));
        assert!(Arc::ptr_eq(&holder.display(), &first));
        let _ = holder.display();
        assert_eq!(holder.deadline_missed_frames(), before + 2);
    }

    #[test]
    fn scroll_clamps_at_zero_and_accumulates() {
        let mut holder = Compositor::new();
        assert_eq!(holder.scroll(), (0.0, 0.0));
        holder.scroll_by(0.0, 100.0);
        assert_eq!(holder.scroll(), (0.0, 100.0));
        holder.scroll_by(10.0, 50.0);
        assert_eq!(holder.scroll(), (10.0, 150.0));
        // Scrolling back past the top clamps at zero, per axis.
        holder.scroll_by(-20.0, -200.0);
        assert_eq!(holder.scroll(), (0.0, 0.0));
        holder.reset_scroll();
        assert_eq!(holder.scroll(), (0.0, 0.0));
        holder.scroll_by(5.0, 5.0);
        holder.reset_scroll();
        assert_eq!(holder.scroll(), (0.0, 0.0));
    }

    #[test]
    fn scroll_clamps_at_the_bottom_from_the_submitted_list() {
        let mut holder = Compositor::new();
        // No list yet: no maximum, the old unbounded behaviour.
        assert_eq!(holder.max_scroll(), (f32::INFINITY, f32::INFINITY));
        // Canvas 800x600, one rect reaching y=1000: the page is
        // 400px taller than the viewport, and no wider.
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            color: [1.0; 4],
        });
        list.push(DisplayItem::FillRect {
            x: 700.0,
            y: 900.0,
            width: 100.0,
            height: 100.0,
            color: [1.0; 4],
        });
        assert!(holder.submit(1, Arc::new(list), Arc::new(ResourceTable::new())));
        assert_eq!(holder.max_scroll(), (0.0, 400.0));
        // Scrolling past the end stops at the maximum, per axis.
        holder.scroll_by(1000.0, 1000.0);
        assert_eq!(holder.scroll(), (0.0, 400.0));
        holder.scroll_by(0.0, -1000.0);
        assert_eq!(holder.scroll(), (0.0, 0.0));
        // A newer, shorter list lowers the maximum again.
        let short = list_with_canvas(800.0, 600.0);
        assert!(holder.submit(2, short, Arc::new(ResourceTable::new())));
        assert_eq!(holder.max_scroll(), (0.0, 0.0));
        holder.scroll_by(0.0, 50.0);
        assert_eq!(holder.scroll(), (0.0, 0.0));
        // A stale list never moves the maximum.
        assert!(!holder.submit(
            1,
            list_with_canvas(400.0, 300.0),
            Arc::new(ResourceTable::new())
        ));
        assert_eq!(holder.max_scroll(), (0.0, 0.0));
    }

    #[test]
    fn submit_reclamps_scroll_when_the_page_shrinks() {
        let mut holder = Compositor::new();
        // Tall list: canvas 800x600, one rect reaching y=1000, so
        // the page maximum is (0, 400). Scroll to the bottom.
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            color: [1.0; 4],
        });
        list.push(DisplayItem::FillRect {
            x: 700.0,
            y: 900.0,
            width: 100.0,
            height: 100.0,
            color: [1.0; 4],
        });
        assert!(holder.submit(1, Arc::new(list), Arc::new(ResourceTable::new())));
        holder.scroll_by(0.0, 1000.0);
        assert_eq!(holder.scroll(), (0.0, 400.0));
        // A newer, medium list (rect to y=700, maximum (0, 100))
        // re-clamps the stored scroll at submit — not at the next
        // scroll event.
        let mut medium = DisplayList::new();
        medium.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            color: [1.0; 4],
        });
        medium.push(DisplayItem::FillRect {
            x: 700.0,
            y: 650.0,
            width: 100.0,
            height: 50.0,
            color: [1.0; 4],
        });
        assert!(holder.submit(2, Arc::new(medium), Arc::new(ResourceTable::new())));
        assert_eq!(holder.max_scroll(), (0.0, 100.0));
        assert_eq!(holder.scroll(), (0.0, 100.0));
        // And a list no taller than the canvas clamps it to zero.
        let short = list_with_canvas(800.0, 600.0);
        assert!(holder.submit(3, short, Arc::new(ResourceTable::new())));
        assert_eq!(holder.scroll(), (0.0, 0.0));
    }
}
