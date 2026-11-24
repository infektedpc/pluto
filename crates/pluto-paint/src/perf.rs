//! Frame-span / counter instrumentation (Stage 8 Rec 5, report §i).
//!
//! In-process only: per-frame spans kept as rolling samples with
//! p50/p95/max (never averages), plus counters for produced /
//! clean-skipped / deadline-missed frames, forced flushes, atlas
//! and damage figures, and first paint / first contentful paint.
//! The worker, the shell compositor, and the renderer all record
//! into one [`PerfCounters`] — the process-wide one behind
//! [`global()`] — and the shell logs [`PerfCounters::summary`] to
//! stderr periodically. There is deliberately no DevTools panel,
//! no trace export, and no CI perf gate here: spans + counters
//! first, visualisation second.
//!
//! A frame is "dropped" when the frame interval exceeds 2× the
//! 16.67 ms vsync budget, or when the worker misses the deadline
//! (its build ran longer than one frame — counted by the worker,
//! not guessed by the shell).

use crate::DamageRect;
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// How many samples each span keeps (four seconds at 60 Hz).
pub const SPAN_SAMPLES: usize = 240;

/// One 16.67 ms frame budget, for the dropped-frame rule.
pub const FRAME_BUDGET_MS: f32 = 1000.0 / 60.0;

/// The per-frame spans (report §i item 1), in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpanKind {
    /// Worker: parse + style + layout + display-list build.
    WorkerBuild,
    /// Serialise the list + send it to the shell.
    Ipc,
    /// Shell: glyph/image uploads + instance-buffer writes.
    ShellPrepare,
    /// Shell: command encoding for this frame.
    ShellEncode,
    /// Shell: submit to presentation.
    SubmitToPresent,
    /// Present-to-present interval.
    FrameInterval,
}

impl SpanKind {
    pub fn name(self) -> &'static str {
        match self {
            SpanKind::WorkerBuild => "worker_build_ms",
            SpanKind::Ipc => "ipc_ms",
            SpanKind::ShellPrepare => "shell_prepare_ms",
            SpanKind::ShellEncode => "shell_encode_ms",
            SpanKind::SubmitToPresent => "submit_to_present_ms",
            SpanKind::FrameInterval => "frame_interval_ms",
        }
    }

    const COUNT: usize = 6;
}

/// Rolling samples for one span, with percentile reads.
#[derive(Debug, Clone, Default)]
pub struct SpanStats {
    samples: VecDeque<f32>,
}

impl SpanStats {
    pub fn record(&mut self, ms: f32) {
        if ms.is_finite() && ms >= 0.0 {
            self.samples.push_back(ms);
            while self.samples.len() > SPAN_SAMPLES {
                self.samples.pop_front();
            }
        }
    }

    pub fn count(&self) -> usize {
        self.samples.len()
    }

    fn sorted(&self) -> Vec<f32> {
        let mut values: Vec<f32> = self.samples.iter().copied().collect();
        values.sort_by(f32::total_cmp);
        values
    }

    /// Nearest-rank percentile (`p` in 0..=100), or `None` when empty.
    pub fn percentile(&self, p: f32) -> Option<f32> {
        let values = self.sorted();
        if values.is_empty() {
            return None;
        }
        let rank = (p / 100.0 * values.len() as f32).ceil() as usize;
        values.get(rank.saturating_sub(1).min(values.len() - 1)).copied()
    }

    pub fn p50(&self) -> Option<f32> {
        self.percentile(50.0)
    }

    pub fn p95(&self) -> Option<f32> {
        self.percentile(95.0)
    }

    pub fn max(&self) -> Option<f32> {
        self.sorted().last().copied()
    }

    /// True when the last interval exceeded 2× the frame budget
    /// (the report's dropped-frame rule).
    pub fn last_was_dropped_frame(&self) -> bool {
        self.samples
            .back()
            .is_some_and(|interval| *interval > 2.0 * FRAME_BUDGET_MS)
    }
}

/// All Stage 8 frame instrumentation for one tab (or the process).
#[derive(Debug, Default)]
pub struct PerfCounters {
    /// Frames produced (a pending list activated at the frame
    /// boundary, or a first present).
    pub frames_produced: u64,
    /// Frames clean-skipped: nothing dirty, no rAF pending, no
    /// scroll change — no submit, no re-encode.
    pub frames_clean_skipped: u64,
    /// Frames whose worker build missed the deadline (the build ran
    /// longer than one frame budget, so the list activated a frame
    /// late). Counted by the worker, separately from clean skips.
    pub frames_deadline_missed: u64,
    /// Frames that re-presented the active list with a new scroll
    /// uniform (no new list pending — the deadline rule working as
    /// designed, not a miss).
    pub frames_represented: u64,
    /// Frame ticks that arrived while the tab was hidden (no
    /// rendering opportunity at all).
    pub frames_hidden_skipped: u64,
    /// Forced synchronous layout flushes (geometry reads).
    pub forced_flushes: u64,
    /// Forced flushes that found nothing dirty (a layout-thrash
    /// signal: the caller is reading in a loop).
    pub flushes_found_nothing_dirty: u64,
    /// Glyph atlas uploads this frame's prepare performed.
    pub atlas_uploads: u64,
    /// Glyph atlas evictions (Stage 8 Rec 11 LRU): glyphs evicted
    /// from either atlas to make room, cumulative.
    pub atlas_evictions: u64,
    /// Times the glyph atlas grew 1024→2048 (Stage 8 Rec 11),
    /// cumulative across both atlases.
    pub atlas_growths: u64,
    /// Colour-glyph (emoji) atlas uploads this frame's prepare
    /// performed (Stage 8 Rec 11, second atlas).
    pub colour_atlas_uploads: u64,
    /// Colour glyphs skipped because the colour atlas was full
    /// (Stage 8 Rec 11, cumulative).
    pub colour_glyphs_skipped: u64,
    /// Glyphs skipped because the atlas was full.
    pub glyphs_skipped: u64,
    /// Image textures evicted from the renderer's GPU image cache
    /// for the byte cap (Stage 8 Rec 12), cumulative.
    pub image_cache_evictions: u64,
    /// Bytes currently held by the renderer's GPU image cache
    /// (Stage 8 Rec 12, mip chains included).
    pub image_cache_bytes: usize,
    /// Image texture uploads this frame's prepare performed
    /// (Stage 8 Rec 12, mip level 0 plus its downscaled levels).
    pub image_uploads: u64,
    /// Display-list items CPU-culled against the clip/viewport.
    pub items_culled: u64,
    /// Items / glyphs / resources in the last submitted list.
    pub display_list_items: usize,
    pub display_list_glyphs: usize,
    pub display_list_resources: usize,
    /// Last frame's damage px² over viewport px² (0..=1+).
    pub damage_area_ratio: f32,
    /// Navigation-start → first submitted frame, in ms.
    pub first_paint_ms: Option<f32>,
    /// Navigation-start → first frame with non-empty text/image
    /// content, in ms.
    pub first_contentful_paint_ms: Option<f32>,
    spans: [SpanStats; SpanKind::COUNT],
    navigation_start: Option<Instant>,
}

impl PerfCounters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_span(&mut self, kind: SpanKind, ms: f32) {
        self.spans[kind as usize].record(ms);
    }

    pub fn span(&self, kind: SpanKind) -> &SpanStats {
        &self.spans[kind as usize]
    }

    /// A navigation started: first paint / first contentful paint
    /// are measured from here until the next frames land.
    pub fn note_navigation_start(&mut self) {
        self.navigation_start = Some(Instant::now());
        self.first_paint_ms = None;
        self.first_contentful_paint_ms = None;
    }

    /// A frame was submitted. `has_content` is true when the list
    /// carries non-empty text or image content. First paint is the
    /// first submitted frame after the navigation started; first
    /// contentful paint is the first such frame with content.
    pub fn note_frame(&mut self, has_content: bool) {
        let Some(start) = self.navigation_start else {
            return;
        };
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        if self.first_paint_ms.is_none() {
            self.first_paint_ms = Some(elapsed);
        }
        if has_content && self.first_contentful_paint_ms.is_none() {
            self.first_contentful_paint_ms = Some(elapsed);
        }
    }

    /// Record a submitted list's damage against the viewport.
    pub fn note_damage(&mut self, damage: Option<DamageRect>, viewport: (f32, f32)) {
        let viewport_area = (viewport.0 * viewport.1).max(1.0);
        self.damage_area_ratio = damage.map_or(0.0, |d| d.area() / viewport_area);
    }

    /// One stderr line with the counters and the span percentiles.
    /// `Option` reads print as `-` (no samples yet).
    pub fn summary(&self) -> String {
        let pct = |kind: SpanKind| {
            let span = self.span(kind);
            let fmt = |v: Option<f32>| {
                v.map_or("-".to_string(), |v| format!("{v:.2}"))
            };
            format!(
                "{} p50={} p95={} max={}",
                kind.name(),
                fmt(span.p50()),
                fmt(span.p95()),
                fmt(span.max())
            )
        };
        format!(
            "[pluto perf] frames produced={} clean_skipped={} deadline_missed={} \
             represented={} hidden_skipped={} | flushes={} (nothing_dirty={}) | \
             list items={} glyphs={} resources={} | atlas uploads={} evictions={} \
             growths={} colour_uploads={} colour_skipped={} glyphs_skipped={} \
             culled={} image_cache evictions={} bytes={} uploads={} damage_ratio={:.3} | first_paint={}ms \
             first_contentful_paint={}ms | {} | {} | {} | {} | {} | {}",
            self.frames_produced,
            self.frames_clean_skipped,
            self.frames_deadline_missed,
            self.frames_represented,
            self.frames_hidden_skipped,
            self.forced_flushes,
            self.flushes_found_nothing_dirty,
            self.display_list_items,
            self.display_list_glyphs,
            self.display_list_resources,
            self.atlas_uploads,
            self.atlas_evictions,
            self.atlas_growths,
            self.colour_atlas_uploads,
            self.colour_glyphs_skipped,
            self.glyphs_skipped,
            self.items_culled,
            self.image_cache_evictions,
            self.image_cache_bytes,
            self.image_uploads,
            self.damage_area_ratio,
            self.first_paint_ms
                .map_or("-".to_string(), |v| format!("{v:.1}")),
            self.first_contentful_paint_ms
                .map_or("-".to_string(), |v| format!("{v:.1}")),
            pct(SpanKind::WorkerBuild),
            pct(SpanKind::Ipc),
            pct(SpanKind::ShellPrepare),
            pct(SpanKind::ShellEncode),
            pct(SpanKind::SubmitToPresent),
            pct(SpanKind::FrameInterval),
        )
    }
}

/// The process-wide counters (worker, compositor, and renderer all
/// record here; the shell logs the summary). Tests use local
/// [`PerfCounters`] instances, never this global.
pub fn global() -> &'static Mutex<PerfCounters> {
    static GLOBAL: OnceLock<Mutex<PerfCounters>> = OnceLock::new();
    GLOBAL.get_or_init(|| Mutex::new(PerfCounters::new()))
}

/// Record one span sample in the process-wide counters.
pub fn record_span(kind: SpanKind, ms: f32) {
    if let Ok(mut counters) = global().lock() {
        counters.record_span(kind, ms);
    }
}

/// The process-wide summary line (for the shell's periodic log).
pub fn summary() -> String {
    global()
        .lock()
        .map_or("[pluto perf] counters unavailable".to_string(), |c| {
            c.summary()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_percentiles_use_nearest_rank_not_averages() {
        let mut stats = SpanStats::default();
        assert_eq!(stats.p50(), None);
        for v in 1..=10 {
            stats.record(v as f32);
        }
        assert_eq!(stats.count(), 10);
        assert_eq!(stats.p50(), Some(5.0));
        assert_eq!(stats.p95(), Some(10.0));
        assert_eq!(stats.max(), Some(10.0));
        // Averages would say 5.5 for p50; percentiles do not.
        assert_eq!(stats.percentile(0.0), Some(1.0));
        assert_eq!(stats.percentile(100.0), Some(10.0));
        // Non-finite / negative samples are ignored, not counted.
        stats.record(f32::NAN);
        stats.record(-1.0);
        assert_eq!(stats.count(), 10);
    }

    #[test]
    fn span_samples_roll_off_after_the_cap() {
        let mut stats = SpanStats::default();
        for v in 0..(SPAN_SAMPLES + 10) {
            stats.record(v as f32);
        }
        assert_eq!(stats.count(), SPAN_SAMPLES);
        // Samples 10..249 remain; nearest-rank p50 is index 119.
        assert_eq!(stats.p50(), Some(129.0));
    }

    #[test]
    fn dropped_frame_rule_is_two_frame_budgets() {
        let mut stats = SpanStats::default();
        stats.record(16.0);
        assert!(!stats.last_was_dropped_frame());
        stats.record(2.0 * FRAME_BUDGET_MS + 0.01);
        assert!(stats.last_was_dropped_frame());
    }

    #[test]
    fn first_paint_and_first_contentful_paint() {
        let mut counters = PerfCounters::new();
        // No navigation start: frames do not invent timings.
        counters.note_frame(true);
        assert_eq!(counters.first_paint_ms, None);
        counters.note_navigation_start();
        counters.note_frame(false);
        assert!(counters.first_paint_ms.is_some());
        assert_eq!(counters.first_contentful_paint_ms, None);
        counters.note_frame(false);
        assert_eq!(counters.first_contentful_paint_ms, None);
        counters.note_frame(true);
        assert!(counters.first_contentful_paint_ms.is_some());
        // Later frames never overwrite the firsts.
        let first = counters.first_paint_ms;
        counters.note_frame(true);
        assert_eq!(counters.first_paint_ms, first);
        // A new navigation resets both.
        counters.note_navigation_start();
        assert_eq!(counters.first_paint_ms, None);
        assert_eq!(counters.first_contentful_paint_ms, None);
    }

    #[test]
    fn damage_ratio_is_damage_over_viewport() {
        let mut counters = PerfCounters::new();
        counters.note_damage(
            Some(DamageRect {
                x: 0.0,
                y: 0.0,
                width: 400.0,
                height: 300.0,
            }),
            (800.0, 600.0),
        );
        assert!((counters.damage_area_ratio - 0.25).abs() < 0.001);
        counters.note_damage(None, (800.0, 600.0));
        assert_eq!(counters.damage_area_ratio, 0.0);
    }

    #[test]
    fn summary_renders_without_samples() {
        let counters = PerfCounters::new();
        let line = counters.summary();
        assert!(line.contains("frames produced=0"));
        assert!(line.contains("worker_build_ms p50=-"));
    }
}
