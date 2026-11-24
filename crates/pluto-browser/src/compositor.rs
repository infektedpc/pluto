//! The shell's compositor-side holder (Stage 4 #3): the last-good
//! display list + scroll offset, epoch'd.
//!
//! The worker thread builds display lists; the holder is the only
//! place the shell keeps one. An incoming list installs only when
//! its epoch is newer than the installed one, so a slow worker can
//! never resurrect a stale frame. The "double buffer" is the old
//! `Arc` still held by the in-flight `Primitive` while the new one
//! is installed here — the swap happens in `update`, between
//! frames, never mid-draw.
//!
//! Scroll lives here too, so a wheel event never round-trips to
//! layout: the viewport widget applies the offset as a view
//! transform (the renderer subtracts it in the vertex shader, and
//! the canvas fill is translated the other way so it stays glued
//! to the viewport). The offset is clamped to >= 0 (no overscroll
//! above or left). Clamping the bottom needs the content height,
//! which does not ride in `SubmitDisplayList` yet — until M2's
//! scroll work lands, the page can scroll past its end into the
//! canvas background.

use pluto_paint::{DisplayList, ResourceTable};
use std::sync::Arc;

/// The shell's view of the current page: what to draw, what its
/// image/blob ids resolve against, how new it is, and how far it
/// is scrolled.
#[derive(Debug, Clone)]
pub struct Compositor {
    display: Arc<DisplayList>,
    resources: Arc<ResourceTable>,
    epoch: u64,
    scroll: (f32, f32),
}

impl Compositor {
    pub fn new() -> Self {
        Self {
            display: Arc::new(DisplayList::new()),
            resources: Arc::new(ResourceTable::new()),
            epoch: 0,
            scroll: (0.0, 0.0),
        }
    }

    /// Offer a worker-built list. Installs it when `epoch` is newer
    /// than the installed one and returns true; an older or equal
    /// epoch is ignored (a stale frame never replaces a newer one)
    /// and returns false. The widget reads the holder every frame,
    /// so an install is picked up on the next frame.
    pub fn submit(
        &mut self,
        epoch: u64,
        display: Arc<DisplayList>,
        resources: Arc<ResourceTable>,
    ) -> bool {
        if epoch <= self.epoch {
            return false;
        }
        self.epoch = epoch;
        self.display = display;
        self.resources = resources;
        true
    }

    pub fn display(&self) -> &Arc<DisplayList> {
        &self.display
    }

    pub fn resources(&self) -> &Arc<ResourceTable> {
        &self.resources
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Current scroll offset, in page pixels.
    pub fn scroll(&self) -> (f32, f32) {
        self.scroll
    }

    /// Scroll by (dx, dy) page pixels, clamped to >= 0 on both axes.
    pub fn scroll_by(&mut self, dx: f32, dy: f32) {
        self.scroll.0 = (self.scroll.0 + dx).max(0.0);
        self.scroll.1 = (self.scroll.1 + dy).max(0.0);
    }

    /// Reset the scroll, e.g. on a new navigation.
    pub fn reset_scroll(&mut self) {
        self.scroll = (0.0, 0.0);
    }
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
    fn submit_installs_newer_epochs_only() {
        let mut holder = Compositor::new();
        assert_eq!(holder.epoch(), 0);
        assert!(holder.display().is_empty());
        let first = list_with_canvas(800.0, 600.0);
        assert!(holder.submit(1, Arc::clone(&first), Arc::new(ResourceTable::new())));
        assert_eq!(holder.epoch(), 1);
        assert!(Arc::ptr_eq(holder.display(), &first));
        // Same epoch again: ignored, the current list stays.
        assert!(!holder.submit(
            1,
            list_with_canvas(400.0, 300.0),
            Arc::new(ResourceTable::new())
        ));
        assert_eq!(holder.epoch(), 1);
        assert!(Arc::ptr_eq(holder.display(), &first));
        // Older epoch: ignored too — a slow worker cannot
        // resurrect a stale frame.
        assert!(!holder.submit(
            0,
            list_with_canvas(100.0, 100.0),
            Arc::new(ResourceTable::new())
        ));
        assert!(Arc::ptr_eq(holder.display(), &first));
        // Newer epoch: installs.
        let second = list_with_canvas(1024.0, 768.0);
        assert!(holder.submit(2, Arc::clone(&second), Arc::new(ResourceTable::new())));
        assert_eq!(holder.epoch(), 2);
        assert!(Arc::ptr_eq(holder.display(), &second));
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
}
