//! The host-owned event loop (Stage 7, Recommendation 3; HTML §8.1.7).
//!
//! The loop is the host's, not the engine's. The engine runs one task or
//! one microtask when told; it never decides ordering. This module owns
//! the ordering state: per-source task queues, the one microtask queue,
//! the timer wheel, the rAF registry, and the navigation epoch.
//!
//! # Per-turn algorithm (one turn of "run the event loop")
//!
//! 1. Pick one runnable task: the oldest task from the highest-priority
//!    non-empty source (priority is Pluto policy — the spec leaves the
//!    choice of queue implementation-defined — see [`TaskSource`]).
//!    Stale-epoch tasks are dropped, never run.
//! 2. Run it to completion (run-to-completion: nothing on this thread
//!    can interrupt it).
//! 3. Microtask checkpoint: drain the microtask queue completely,
//!    including microtasks queued by microtasks.
//! 4. Update the rendering, but ONLY at a rendering opportunity (a frame
//!    is due and something asked for one: a pending rAF callback or an
//!    explicit render request). Rendering is not "once per turn".
//!
//! The driving code lives in [`crate::TabJs`]; this module is the state
//! and the queue operations. JS callbacks themselves live in the realm
//! (in the registry's `timers` / `raf` Maps — see `tab.rs`), so
//! this state holds no engine values at all: only ids, instants, and
//! Rust closures. That keeps drop order trivial (no engine value can
//! outlive its runtime).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A task source (HTML §8.1.7.4, "generic task sources"). Each source
/// has its own queue; FIFO is guaranteed only *within* a source. The
/// loop picks which source to service next — that choice is Pluto
/// policy, not the spec's, and it is the hook that lets user input
/// pre-empt timers.
///
/// Priority order (highest first) is the declaration order of
/// [`TaskSource::PRIORITY`]: user interaction first (input must never
/// wait behind a timer), then DOM manipulation, networking, parser,
/// script, and timers last. Rendering is not a task source for dispatch:
/// rAF callbacks run inside update-the-rendering, never as tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskSource {
    /// Parser work (tokenizing bytes, processing tokens). Parser-blocking
    /// scripts run as `Script` tasks issued from a parser task.
    Parser,
    /// Classic / module / defer / async script execution.
    Script,
    /// Timer firings (`setTimeout` / `setInterval`).
    Timer,
    /// Network completions (`fetch` resolution, resource loads). A
    /// completion queues a task here; it never resolves a promise inline
    /// on the network thread.
    Networking,
    /// Input events (click, key, …) and their dispatch.
    UserInteraction,
    /// DOM mutation work and `MutationObserver` notifications (the
    /// observer itself is a later stage; the source slot is reserved).
    DomManipulation,
    /// Reserved: rendering work is driven by the rendering-opportunity
    /// check, not by task dispatch. Never enqueue tasks here.
    Rendering,
}

impl TaskSource {
    /// Sources in dispatch-priority order, highest first.
    pub const PRIORITY: [TaskSource; 7] = [
        TaskSource::UserInteraction,
        TaskSource::DomManipulation,
        TaskSource::Networking,
        TaskSource::Parser,
        TaskSource::Script,
        TaskSource::Timer,
        TaskSource::Rendering,
    ];

    /// Index into the per-source queue array.
    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// How long a frame lasts at 60 Hz. The rendering opportunity is
/// checked against this cadence (or the shell's vsync signal, when the
/// integration wave wires it in — the loop only needs "a frame is due").
pub const FRAME_INTERVAL: Duration = Duration::from_millis(16);

/// The nesting clamp (HTML §8.7): after this many levels of nested
/// timers, the delay is forced to at least [`MIN_NESTED_TIMER_DELAY`].
/// Nesting counts any mix of `setTimeout` / `setInterval`, including
/// repeats of an interval.
pub const NESTING_CLAMP_LEVEL: u32 = 5;

/// Minimum delay for a timer whose nesting level exceeds
/// [`NESTING_CLAMP_LEVEL`].
pub const MIN_NESTED_TIMER_DELAY: Duration = Duration::from_millis(4);

/// One unit of scheduled work. Tasks are taken out of their queue before
/// they run, so a task may enqueue further tasks (including into its own
/// source) without borrowing the queue it came from.
pub enum TaskKind {
    /// Evaluate a classic script in the realm's global scope.
    Script {
        /// Script source text.
        source: String,
        /// URL or filename for stack traces.
        url: String,
    },
    /// Fire a timer: look its callback up in the realm's
    /// the registry's `timers` Map and call it with no arguments.
    Timer {
        /// Timer id, as returned by `setTimeout` / `setInterval`.
        id: u64,
        /// Nesting level of this firing (for the clamp on timers this
        /// callback schedules).
        nesting: u32,
        /// True for `setInterval`: the callback stays in the Map for
        /// the next firing. One-shot timers consume their callback.
        repeats: bool,
    },
    /// Run an animation-frame callback: look it up in the realm's
    /// the registry's `raf` Map and call it with the frame timestamp (ms).
    /// rAF callbacks are only ever run from update-the-rendering, never
    /// dispatched as tasks; this variant exists so the rendering step
    /// can reuse the same runner.
    AnimationFrame {
        /// rAF callback id, as returned by `requestAnimationFrame`.
        id: u64,
        /// Shared frame timestamp for this frame (ms since realm
        /// creation, `performance.now()` clock).
        timestamp_ms: f64,
    },
    /// A Rust-side task. The closure receives the whole `TabJs` so it
    /// can enqueue work, evaluate scripts, or touch the host — it runs
    /// outside any engine entry, so calling into the realm is safe.
    /// (`Send` is not required: tasks never leave the worker thread.)
    Host(Box<dyn FnOnce(&mut crate::TabJs)>),
}

impl std::fmt::Debug for TaskKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskKind::Script { source, url } => f
                .debug_struct("Script")
                .field("source", source)
                .field("url", url)
                .finish(),
            TaskKind::Timer { id, nesting, repeats } => f
                .debug_struct("Timer")
                .field("id", id)
                .field("nesting", nesting)
                .field("repeats", repeats)
                .finish(),
            TaskKind::AnimationFrame { id, timestamp_ms } => f
                .debug_struct("AnimationFrame")
                .field("id", id)
                .field("timestamp_ms", timestamp_ms)
                .finish(),
            TaskKind::Host(_) => f.debug_tuple("Host").field(&"<fn>").finish(),
        }
    }
}

/// A task, tagged with its source and the navigation epoch it belongs
/// to. A task whose epoch is not the loop's current epoch is stale
/// (its document went away under it) and is dropped at dispatch.
#[derive(Debug)]
pub struct Task {
    /// Which queue this task came from.
    pub source: TaskSource,
    /// Navigation epoch this task was queued under.
    pub epoch: u64,
    /// What to run.
    pub kind: TaskKind,
}

impl Task {
    /// A task for the current epoch.
    pub fn new(source: TaskSource, epoch: u64, kind: TaskKind) -> Self {
        Self {
            source,
            epoch,
            kind,
        }
    }
}

/// One microtask. JS-originated microtasks (promise reactions,
/// `queueMicrotask` callbacks, `await` continuations) are NOT stored
/// here: they live in the engine's own pending-job queue and are drained
/// one at a time through `Ctx::execute_pending_job`, which keeps their
/// relative order exactly as the engine enqueued them (a separate Rust
/// queue could not observe that order). This queue holds only Rust-side
/// microtasks enqueued between turns (wave 2: fetch resolutions, and —
/// later — `MutationObserver` notifications).
///
/// Ordering caveat, documented honestly: a Rust-side microtask enqueued
/// *during* JS execution (via a binding) would run before JS jobs that
/// were already pending, which is not spec order. In this wave no
/// binding enqueues host microtasks during JS execution, so the case is
/// unreachable; wave 2 must route binding-originated microtasks through
/// promise jobs (as `queueMicrotask` already does) instead of this
/// queue.
pub enum Microtask {
    /// A Rust-side microtask.
    Host(Box<dyn FnOnce(&mut crate::TabJs)>),
}

impl std::fmt::Debug for Microtask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Microtask::Host(_) => f.debug_tuple("Host").field(&"<fn>").finish(),
        }
    }
}

/// One timer in the wheel.
#[derive(Debug, Clone)]
pub struct TimerEntry {
    /// Timer id (shared id space with intervals; `clearTimeout` and
    /// `clearInterval` cancel either).
    pub id: u64,
    /// When this timer fires (monotonic clock).
    pub fire_at: Instant,
    /// Repeat interval for `setInterval`; `None` for `setTimeout`.
    /// Repeats re-arm with nesting + 1, so a `setInterval(fn, 0)` is
    /// clamped to 4 ms after five firings, like a nested timeout chain.
    pub interval: Option<Duration>,
    /// Nesting level of this timer (see [`NESTING_CLAMP_LEVEL`]).
    pub nesting: u32,
    /// Navigation epoch this timer was scheduled under. Stale-epoch
    /// timers are dropped when they fire, never run.
    pub epoch: u64,
}

/// One pending `requestAnimationFrame` callback. The callback itself
/// lives in the registry's `raf` Map; this is the registry entry.
#[derive(Debug, Clone)]
pub struct RafEntry {
    /// Callback id, as returned by `requestAnimationFrame`.
    pub id: u64,
    /// Navigation epoch this callback was registered under.
    pub epoch: u64,
}

/// The loop's mutable state. Pure Rust: ids, instants, queues. No engine
/// values are stored here, so dropping this state never touches a JS
/// runtime.
#[derive(Debug)]
pub struct LoopState {
    /// Current navigation epoch. Starts at 1; `navigate()` increments
    /// it and clears every queue.
    pub epoch: u64,
    /// One queue per [`TaskSource`], indexed by [`TaskSource::index`].
    pub queues: [VecDeque<Task>; 7],
    /// Rust-side microtasks (see [`Microtask`]).
    pub microtasks: VecDeque<Microtask>,
    /// The timer wheel: unsorted; the next deadline is found by scan.
    /// (M4 scale — tens of timers, not thousands. A hierarchical wheel
    /// is a later optimisation if profiling ever demands it.)
    pub timers: Vec<TimerEntry>,
    /// Next timer id to hand out.
    pub next_timer_id: u64,
    /// Pending rAF callbacks, in registration order.
    pub raf: Vec<RafEntry>,
    /// Next rAF id to hand out.
    pub next_raf_id: u64,
    /// Nesting level of the timer task currently running (0 outside a
    /// timer task). `setTimeout` called from a timer callback schedules
    /// at `current_nesting + 1`.
    pub current_nesting: u32,
    /// When the next rendering opportunity occurs. Starts due
    /// immediately so the first frame renders without waiting for a
    /// vsync; afterwards it advances one [`FRAME_INTERVAL`] per frame.
    pub next_frame_at: Instant,
    /// Monotonic clock origin for `performance.now()` and rAF
    /// timestamps.
    pub created: Instant,
    /// Uncaught script errors (per-script boundary: a throwing script,
    /// timer, or microtask is recorded here and the loop continues).
    /// Wave 2 reports these via `window.onerror`; this wave records
    /// them so tests can assert error isolation.
    pub script_errors: Vec<String>,
    /// Set by the host when a new display list is wanted even with no
    /// rAF callbacks pending (e.g. a DOM mutation marked the document
    /// dirty). Cleared when a frame runs.
    pub render_requested: bool,
    /// Whether the tab is visible (Stage 8 Rec 1). A hidden tab gets
    /// no rendering opportunities: `maybe_render` returns early, so
    /// rAF callbacks stay queued and no frame is produced — while
    /// timers, tasks, and microtasks keep running (they are not
    /// rendering). Set by the shell through `TabJs::set_visible`.
    pub visible: bool,
}

impl Default for LoopState {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopState {
    /// Fresh state for a new realm / navigation.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            epoch: 1,
            queues: Default::default(),
            microtasks: VecDeque::new(),
            timers: Vec::new(),
            next_timer_id: 1,
            raf: Vec::new(),
            next_raf_id: 1,
            current_nesting: 0,
            next_frame_at: now,
            created: now,
            script_errors: Vec::new(),
            render_requested: false,
            visible: true,
        }
    }

    /// Drop everything scheduled for any document: the navigation
    /// boundary. Called by `navigate()` before the epoch increments.
    pub fn clear_for_navigation(&mut self) {
        for queue in &mut self.queues {
            queue.clear();
        }
        self.microtasks.clear();
        self.timers.clear();
        self.raf.clear();
        self.current_nesting = 0;
        self.render_requested = false;
        self.next_frame_at = Instant::now();
    }

    /// Enqueue a task for the current epoch.
    pub fn push_task(&mut self, source: TaskSource, kind: TaskKind) {
        let task = Task::new(source, self.epoch, kind);
        self.queues[source.index()].push_back(task);
    }

    /// Enqueue a Rust-side microtask.
    pub fn push_microtask(&mut self, microtask: Microtask) {
        self.microtasks.push_back(microtask);
    }

    /// True if any task, microtask, timer, or rAF callback is pending.
    pub fn has_work(&self) -> bool {
        self.queues.iter().any(|q| !q.is_empty())
            || !self.microtasks.is_empty()
            || !self.timers.is_empty()
            || !self.raf.is_empty()
    }

    /// The next timer deadline, if any timer is scheduled.
    pub fn next_timer_deadline(&self) -> Option<Instant> {
        self.timers.iter().map(|t| t.fire_at).min()
    }

    /// Schedule a timer. Returns its id. The delay is clamped:
    /// negative / NaN delays become 0, and a timer whose nesting level
    /// exceeds [`NESTING_CLAMP_LEVEL`] gets at least
    /// [`MIN_NESTED_TIMER_DELAY`]. Nesting is the caller's job to pass
    /// (the host functions in `tab.rs` compute it from
    /// `current_nesting`).
    pub fn schedule_timer(
        &mut self,
        delay: Duration,
        interval: Option<Duration>,
        nesting: u32,
    ) -> u64 {
        let mut delay = delay;
        if nesting > NESTING_CLAMP_LEVEL && delay < MIN_NESTED_TIMER_DELAY {
            delay = MIN_NESTED_TIMER_DELAY;
        }
        let id = self.next_timer_id;
        self.next_timer_id += 1;
        self.timers.push(TimerEntry {
            id,
            fire_at: Instant::now() + delay,
            interval,
            nesting,
            epoch: self.epoch,
        });
        id
    }

    /// Cancel a timer or interval by id. Returns true if one was found.
    pub fn cancel_timer(&mut self, id: u64) -> bool {
        let before = self.timers.len();
        self.timers.retain(|t| t.id != id);
        before != self.timers.len()
    }

    /// Register an rAF callback. Returns its id.
    pub fn register_raf(&mut self) -> u64 {
        let id = self.next_raf_id;
        self.next_raf_id += 1;
        self.raf.push(RafEntry {
            id,
            epoch: self.epoch,
        });
        id
    }

    /// Cancel a pending rAF callback by id. Returns true if one was found.
    pub fn cancel_raf(&mut self, id: u64) -> bool {
        let before = self.raf.len();
        self.raf.retain(|r| r.id != id);
        before != self.raf.len()
    }

    /// Move due timers into the Timer task queue (as `TaskKind::Timer`
    /// tasks, in fire-time order). Stale-epoch timers are dropped.
    /// Returns the number of tasks queued.
    pub fn fire_due_timers(&mut self) -> usize {
        let now = Instant::now();
        let epoch = self.epoch;
        let mut due: Vec<TimerEntry> = Vec::new();
        let mut keep: Vec<TimerEntry> = Vec::new();
        for entry in self.timers.drain(..) {
            if entry.fire_at <= now {
                due.push(entry);
            } else {
                keep.push(entry);
            }
        }
        self.timers = keep;
        due.sort_by_key(|t| t.fire_at);
        let mut queued = 0;
        for entry in due {
            if entry.epoch != epoch {
                continue;
            }
            if let Some(interval) = entry.interval {
                // Re-arm the interval before firing, with nesting + 1
                // (repeats count toward the clamp, per the HTML timer
                // model). The re-armed entry fires even if this firing's
                // callback throws — the callback runs later, as a task.
                // The clamp applies only once nesting exceeds the level.
                let nesting = entry.nesting.saturating_add(1);
                let mut fire_at = now + interval;
                if nesting > NESTING_CLAMP_LEVEL && fire_at < now + MIN_NESTED_TIMER_DELAY {
                    fire_at = now + MIN_NESTED_TIMER_DELAY;
                }
                self.timers.push(TimerEntry {
                    id: entry.id,
                    fire_at,
                    interval: Some(interval),
                    nesting,
                    epoch: entry.epoch,
                });
            }
            self.queues[TaskSource::Timer.index()].push_back(Task::new(
                TaskSource::Timer,
                entry.epoch,
                TaskKind::Timer {
                    id: entry.id,
                    nesting: entry.nesting,
                    repeats: entry.interval.is_some(),
                },
            ));
            queued += 1;
        }
        queued
    }

    /// Take the next runnable task by priority, or `None` if every
    /// queue is empty. Due timers are fired into the Timer queue first.
    pub fn next_task(&mut self) -> Option<Task> {
        self.fire_due_timers();
        for source in TaskSource::PRIORITY {
            if source == TaskSource::Rendering {
                continue;
            }
            if let Some(task) = self.queues[source.index()].pop_front() {
                return Some(task);
            }
        }
        None
    }

    /// Take all currently-registered rAF callbacks (for one frame).
    /// Callbacks registered while the frame runs stay queued for the
    /// next frame. Stale-epoch callbacks are dropped.
    pub fn take_raf_frame(&mut self) -> Vec<RafEntry> {
        let epoch = self.epoch;
        let taken = std::mem::take(&mut self.raf);
        taken.into_iter().filter(|r| r.epoch == epoch).collect()
    }

    /// `performance.now()`: ms since this loop state was created.
    pub fn now_ms(&self) -> f64 {
        self.created.elapsed().as_secs_f64() * 1000.0
    }
}
