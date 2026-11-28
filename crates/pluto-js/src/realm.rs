//! One realm per tab (Stage 7, Recommendation 4).
//!
//! A [`TabRealm`] owns one rquickjs `AsyncRuntime` + one `AsyncContext`
//! (one global, one set of intrinsics), created on the tab's worker
//! thread and dropped on navigation. No JS state is ever shared across
//! tabs, and the realm is only ever entered on its owning thread —
//! enforced by a debug assertion on every entry.
//!
//! # Threading
//!
//! rquickjs's `AsyncRuntime` wraps a single-threaded QuickJS runtime
//! behind a future-aware lock; it does not make the VM multi-threaded.
//! All entry here is synchronous: each public method drives the
//! rquickjs future it needs with [`block_on_sync`] — a plain
//! poll-to-completion loop on the owner thread, no executor. That is
//! load-bearing for the integration wave (wave 2b): the tab thread
//! runs page loads inside its network runtime's `block_on`, and the
//! load's script hooks call back into the realm from inside that
//! `block_on` — tokio's `block_on` cannot nest (it panics), so the
//! realm must not own a tokio runtime of its own. The futures driven
//! here only ever await the runtime's async-lock, which is
//! uncontended on the single owner thread, so they complete on the
//! first poll; a parked future would spin (yielding) rather than
//! complete — see `block_on_sync`. The *ordering* of work lives in
//! the host event loop ([`crate::TabJs`]), never in an executor.
//! One invariant follows, and breaking it deadlocks:
//!
//! * Never call a realm method from a host function invoked by JS
//!   (the `AsyncContext::with` lock is held for the whole JS call, so
//!   a nested entry would deadlock on it). Host functions touch only
//!   loop state; JS execution happens only in loop methods.
//!
//! # Budgets (Recommendation 9, first half)
//!
//! Per-runtime `set_memory_limit` + `set_interrupt_handler` at
//! creation (the Svix pattern). The interrupt handler is a wall-clock
//! check: it returns `true` (abort the VM) once the armed deadline
//! passes. Every evaluation arms the deadline for its own budget, so a
//! long task cannot eat the next script's budget.
//!
//! The interrupt handler alone cannot kill a script hung across
//! `await`s — a suspended script generates no VM ticks. The watchdog
//! covers that case: promise-returning evaluations drive the engine's
//! job queue synchronously, and if the queue empties while the promise
//! is still pending (it can then only settle via a host event this call
//! cannot drive), the realm is poisoned and must be dropped — its
//! suspended async state can never be unwound safely, so the host
//! recreates a fresh realm. Killing the realm is the honest
//! granularity — browsers kill the page's JS context on an
//! unresponsive-script kill, they do not surgically unwind one async
//! function.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use rquickjs::{
    AsyncContext, AsyncRuntime, CatchResultExt, Ctx, FromJs, Function, IntoJs, Module, Object,
    Promise, Type, Value, WriteOptions,
};
use rquickjs::function::This;

use crate::engine::{EvalKind, JsEngine, JsValue};
use crate::error::JsError;

/// Sentinel message for "the job queue emptied while a promise was
/// still pending" (see `TabRealm::await_promise`).
const WOULD_BLOCK: &str = "__pluto_would_block__";

/// Default heap cap per tab realm. A runaway script in one tab cannot
/// OOM another tab (each tab has its own runtime), and cannot OOM the
/// browser process past this plus its DOM.
pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;

/// Default wall-clock budget for one script evaluation (sync or async).
/// Long enough for any real page script, short enough that a hung tab
/// is killed instead of spinning the worker thread forever.
pub const DEFAULT_SCRIPT_BUDGET: Duration = Duration::from_secs(5);

/// Configuration for one [`TabRealm`]. The host (per-tab worker) builds
/// this once per navigation.
#[derive(Debug, Clone)]
pub struct RealmConfig {
    /// Heap cap for this runtime (0 = unlimited — never use for a tab).
    pub memory_limit_bytes: usize,
    /// Wall-clock budget for one script evaluation.
    pub script_budget: Duration,
    /// Where unhandled promise rejections are reported. If `None`,
    /// rejections are dropped on the floor (tests that care pass one).
    pub rejection_sink: Option<mpsc::Sender<String>>,
}

impl Default for RealmConfig {
    fn default() -> Self {
        Self {
            memory_limit_bytes: DEFAULT_MEMORY_LIMIT_BYTES,
            script_budget: DEFAULT_SCRIPT_BUDGET,
            rejection_sink: None,
        }
    }
}

/// One JS realm: one `AsyncRuntime` + one `AsyncContext`, owned by one
/// worker thread. Created on navigation, dropped on navigation.
///
/// Field order is load-bearing in the other direction: this struct holds
/// no `Persistent` engine values, so nothing here can outlive the
/// runtime — the drop-order hazard the rquickjs docs warn about cannot
/// occur (callbacks live in the realm's own JS Maps; see `tab.rs`).
pub struct TabRealm {
    /// The thread that created this realm. Every entry asserts on it
    /// (debug builds); the realm is never entered from anywhere else.
    owner: std::thread::ThreadId,
    /// The QuickJS runtime (heap, jobs, interrupt handler).
    runtime: AsyncRuntime,
    /// The realm's global context.
    context: AsyncContext,
    /// Monotonic origin for the armed deadline.
    created: Instant,
    /// Armed deadline, as milliseconds since `created`. `u64::MAX`
    /// when disarmed. Read by the interrupt handler (which runs on the
    /// VM thread — the owner thread — while JS executes).
    deadline_ms: Arc<AtomicU64>,
    /// Set when the watchdog fires. A poisoned realm must not be
    /// entered again; the host drops it and creates a fresh one.
    poisoned: bool,
    /// Budget applied to each evaluation.
    script_budget: Duration,
}

impl TabRealm {
    /// Create a realm on the current thread, with budgets armed per
    /// [`RealmConfig`]. The caller must be the tab's worker thread;
    /// the thread id is recorded here and asserted on every entry.
    pub fn new(config: &RealmConfig) -> Result<Self, JsError> {
        let runtime = AsyncRuntime::new()
            .map_err(|e| JsError::Internal(format!("AsyncRuntime::new: {e}")))?;
        let created = Instant::now();
        let deadline_ms = Arc::new(AtomicU64::new(u64::MAX));

        // Wall-clock interrupt handler (Svix pattern): the VM calls this
        // regularly while executing; returning true raises an
        // uncatchable exception and returns control to the caller.
        let handler_deadline = Arc::clone(&deadline_ms);
        let handler_created = created;
        let handler = move || {
            handler_created.elapsed().as_millis() as u64 > handler_deadline.load(Ordering::Relaxed)
        };

        // Unhandled-rejection tracker (HTML `HostPromiseRejectionTracker`):
        // a promise that rejects with no handler is reported here. The
        // third argument is the rejection reason; the fourth is true for
        // `rejectionhandled` (a handler attached late) and false for a
        // fresh unhandled rejection. Only the fresh ones are reported.
        let sink = config.rejection_sink.clone();
        let tracker: Option<rquickjs::runtime::RejectionTracker> =
            sink.map(make_rejection_tracker);

        let script_budget = config.script_budget;
        let memory_limit = config.memory_limit_bytes;
        let context = block_on_sync(async {
            if memory_limit > 0 {
                runtime.set_memory_limit(memory_limit).await;
            }
            runtime.set_interrupt_handler(Some(Box::new(handler))).await;
            if let Some(tracker) = tracker {
                runtime
                    .set_host_promise_rejection_tracker(Some(tracker))
                    .await;
            }
            AsyncContext::full(&runtime).await
        });
        let context = context.map_err(|e| JsError::Internal(format!("AsyncContext::full: {e}")))?;

        Ok(Self {
            owner: std::thread::current().id(),
            runtime,
            context,
            created,
            deadline_ms,
            poisoned: false,
            script_budget,
        })
    }

    /// Install the ES module loader (Stage 12 Rec 1): URL
    /// resolution + pre-fetched sources served from `sources`.
    /// One install per realm: a fresh realm has no loader, so
    /// `TabJs` calls this from `new` and from `navigate`, after
    /// the realm is created. The loader closures are `'static`
    /// (they hold `Arc`s), so they outlive this call and die with
    /// the realm.
    pub fn install_module_loader(
        &mut self,
        sources: crate::module_loader::ModuleSources,
    ) -> Result<(), JsError> {
        self.check_usable()?;
        let resolver = crate::module_loader::PlutoResolver;
        let loader = crate::module_loader::PlutoLoader::new(sources);
        block_on_sync(self.runtime.set_loader(resolver, loader));
        Ok(())
    }

    /// The budget applied to each evaluation.
    pub fn script_budget(&self) -> Duration {
        self.script_budget
    }

    /// True after the watchdog fired. A poisoned realm must be dropped,
    /// never entered again.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Debug assertion: JS entry only happens on the owning thread.
    /// This is a `debug_assert` by design — the check is free in
    /// release, and the ownership rule is enforced by construction
    /// (the realm lives on the worker thread) rather than by a runtime
    /// penalty on every call.
    #[inline]
    fn assert_owner(&self) {
        debug_assert_eq!(
            std::thread::current().id(),
            self.owner,
            "TabRealm entered from a non-owner thread"
        );
    }

    fn check_usable(&self) -> Result<(), JsError> {
        self.assert_owner();
        if self.poisoned {
            return Err(JsError::Poisoned);
        }
        Ok(())
    }

    /// Arm the wall-clock deadline for one evaluation, `budget` from now.
    fn arm_budget(&self, budget: Duration) {
        let deadline = self.created.elapsed().as_millis() as u64
            + budget.as_millis() as u64;
        self.deadline_ms.store(deadline, Ordering::Relaxed);
    }

    /// Disarm the deadline (no evaluation running).
    fn disarm_budget(&self) {
        self.deadline_ms.store(u64::MAX, Ordering::Relaxed);
    }

    /// True if the armed deadline has passed.
    fn budget_tripped(&self) -> bool {
        self.created.elapsed().as_millis() as u64 > self.deadline_ms.load(Ordering::Relaxed)
    }

    /// Run `f` with the realm's `Ctx`, on the owner thread. The closure
    /// runs synchronously while the runtime lock is held; it must not
    /// call back into the realm (see module docs, invariant 2).
    pub(crate) fn with_ctx<F, R>(&self, f: F) -> R
    where
        F: for<'js> FnOnce(Ctx<'js>) -> R + rquickjs::markers::ParallelSend,
        R: rquickjs::markers::ParallelSend,
    {
        self.assert_owner();
        block_on_sync(self.context.with(f))
    }

    /// Drain exactly one pending QuickJS promise job into the host
    /// microtask queue, in order.
    ///
    /// This is the job-draining API the Stage 7 report could not verify
    /// by name: **`Ctx::execute_pending_job(&self) -> bool`**
    /// (rquickjs 0.14.0, `rquickjs-core/src/context/ctx.rs`), which wraps
    /// QuickJS's `JS_ExecutePendingJob`. It runs one job — one promise
    /// reaction, one `queueMicrotask` callback, one `await` continuation —
    /// and returns true if a job ran (even if it threw; the wrapper
    /// catches a thrown job's exception so it cannot be mistaken for a
    /// later error), false when the job queue is empty. The async twin,
    /// `AsyncRuntime::execute_pending_job(&self)` (`runtime/async.rs`),
    /// does the same behind the runtime lock and additionally surfaces
    /// a thrown job as `AsyncJobException`; the loop uses the sync `Ctx`
    /// form here because the exception is already reported through the
    /// rejection tracker (see [`TabRealm::new`]), and the sync form
    /// cannot deadlock against the `with` lock.
    ///
    /// Never awaited ad-hoc on the tokio runtime: jobs are drained only
    /// here, from the host microtask checkpoint, in order.
    pub(crate) fn drain_one_job(&self) -> bool {
        self.assert_owner();
        self.with_ctx(|ctx| ctx.execute_pending_job())
    }

    /// True if the engine has a pending promise job (or a spawned Rust
    /// future). The loop's `run_until_idle` uses this so it does not
    /// exit while a promise job is still queued.
    pub(crate) fn is_job_pending(&self) -> bool {
        self.assert_owner();
        block_on_sync(self.runtime.is_job_pending())
    }

    /// Call a callback stored in one of the registry's callback Maps
    /// (`timers` / `raf`, see `tab.rs`) by id, with the global as
    /// `this` (browsers invoke timer/rAF callbacks with `this ===
    /// window`; the global stands in for `window` until wave 2
    /// installs the real one). A missing entry (cancelled callback) is
    /// a silent no-op. A throwing callback returns [`JsError::Exception`];
    /// the caller records it and the loop continues.
    pub(crate) fn call_map_function(
        &mut self,
        map_name: &str,
        id: u64,
        timestamp_ms: Option<f64>,
    ) -> Result<JsValue, JsError> {
        self.check_usable()?;
        self.arm_budget(self.script_budget);
        let result = self.with_ctx(|ctx| {
            let map: Object = match crate::bindings::registry_map(&ctx, map_name) {
                Ok(map) => map,
                Err(e) => {
                    return Err(format!("callback map {map_name} missing: {e}"));
                }
            };
            let get: Function = match map.get("get") {
                Ok(get) => get,
                Err(e) => return Err(format!("Map.get missing: {e}")),
            };
            // Map methods are extracted then called, so `this` must be
            // passed explicitly — an unbound `Map.prototype.get` throws
            // "Map object expected".
            let value: Value = match get.call((This(map.clone()), id as f64)) {
                Ok(value) => value,
                Err(e) => return Err(format!("Map.get failed: {e}")),
            };
            if !value.is_function() {
                // Cancelled (or already run): nothing to call.
                return Ok(JsValue::Undefined);
            }
            let func = Function::from_js(&ctx, value)
                .map_err(|e| format!("callback is not a function: {e}"))?;
            let result: rquickjs::Result<Value> = match timestamp_ms {
                Some(ts) => func.call((This(ctx.globals()), ts)),
                None => func.call((This(ctx.globals()),)),
            };
            match result.catch(&ctx) {
                Ok(value) => Ok(value_to_js(&ctx, value)),
                Err(caught) => Err(caught.to_string()),
            }
        });
        // Classify while the budget is still armed: `budget_tripped`
        // is only meaningful before `disarm_budget`.
        let classified = self.classify_call_result(result);
        self.disarm_budget();
        classified
    }

    /// Remove a callback from one of the realm's callback Maps.
    pub(crate) fn remove_map_function(&mut self, map_name: &str, id: u64) {
        if self.poisoned {
            return;
        }
        self.assert_owner();
        self.with_ctx(|ctx| {
            if let Ok(map) = crate::bindings::registry_map(&ctx, map_name)
                && let Ok(delete) = map.get::<_, Function>("delete")
            {
                // Unbound Map method: pass the map as `this`.
                let _: rquickjs::Result<Value> =
                    delete.call((This(map.clone()), id as f64));
            }
        });
    }

    /// Start a module evaluation WITHOUT awaiting it (wave 2b,
    /// loop-driven modules — Recommendation 9's remainder).
    ///
    /// `Module::evaluate` runs the module body synchronously up to
    /// its first suspension, then returns the module promise. That
    /// promise is stashed in the registry's `modules` Map under
    /// `id` (the loop polls it with [`Self::poll_module`]); it is
    /// never awaited here, so a module that top-level-awaits a host
    /// event (a timer, a `fetch`) does NOT poison the realm — the
    /// loop keeps turning, the host event fires, the promise
    /// settles, and the poll reports it. This is the fix for the
    /// wave-1 watchdog limit documented in `pluto-js/README.md`:
    /// `await_promise` WouldBlocks immediately on such a module,
    /// because no host event can fire while the worker thread is
    /// driving the job loop. Synchronous failures (syntax errors,
    /// a throw before the first await) are returned as `Err`.
    pub fn start_module(&mut self, url: &str, source: &str, id: u64) -> Result<(), JsError> {
        self.check_usable()?;
        self.arm_budget(self.script_budget);
        let result = self.with_ctx(|ctx| {
            let promise = match Module::evaluate(ctx.clone(), url, source) {
                Ok(promise) => promise,
                Err(e) => return Err(map_plain_error(&ctx, e)),
            };
            let value: Value = match promise.into_js(&ctx) {
                Ok(value) => value,
                Err(e) => return Err(map_plain_error(&ctx, e)),
            };
            let modules = match crate::bindings::registry_map(&ctx, "modules") {
                Ok(map) => map,
                Err(e) => return Err(e.to_string()),
            };
            match crate::bindings::map_set(&ctx, &modules, id, value) {
                Ok(()) => Ok(()),
                Err(e) => Err(e.to_string()),
            }
        });
        let classified = self.classify_call_result(result);
        self.disarm_budget();
        classified
    }

    /// Poll one loop-driven module started by
    /// [`Self::start_module`]: `None` while it is still pending,
    /// `Some(Ok(()))` once it fulfils, `Some(Err(reason))` once it
    /// rejects (a failed module rejects its dependents — the loop
    /// records the error and moves on, it never hangs). The registry
    /// entry is removed on settlement.
    pub fn poll_module(&self, id: u64) -> Option<Result<(), String>> {
        if self.poisoned {
            return Some(Err("realm is poisoned".to_string()));
        }
        self.assert_owner();
        self.with_ctx(|ctx| {
            let modules = crate::bindings::registry_map(&ctx, "modules").ok()?;
            let value = crate::bindings::map_get(&ctx, &modules, id).ok()?;
            if value.is_undefined() {
                return None;
            }
            let promise = Promise::from_js(&ctx, value).ok()?;
            // `result()` is `None` while pending, `Some(Ok)` on
            // fulfilment, `Some(Err)` on rejection (it also throws
            // the reason, so `ctx.catch()` retrieves it).
            match promise.result::<Value>() {
                None => None,
                Some(Ok(_)) => {
                    crate::bindings::map_delete(&ctx, &modules, id);
                    Some(Ok(()))
                }
                Some(Err(_)) => {
                    let reason_value = ctx.catch();
                    let reason = crate::bindings::format_console_arg(&ctx, &reason_value);
                    crate::bindings::map_delete(&ctx, &modules, id);
                    Some(Err(reason))
                }
            }
        })
    }

    /// Evaluate a promise-returning entry point (top-level await,
    /// module evaluation) and drive it to settlement synchronously.
    ///
    /// The promise is driven with the job-draining API
    /// (`Ctx::execute_pending_job`, one job at a time) until
    /// `Promise::result` reports it settled. This is deliberately NOT
    /// `Promise::into_future().await`: awaiting the promise as a Rust
    /// future parks the worker thread, and nothing runs the engine's
    /// job queue while parked, so the promise could never settle.
    ///
    /// Watchdog: if the job queue empties while the promise is still
    /// pending, the promise can only settle via a host event (a timer,
    /// a fetch) that this call cannot drive — in this wave that means
    /// it never settles here, so the realm is poisoned
    /// ([`JsError::WatchdogTimeout`]). Known wave-1 limit: a module
    /// that top-level-awaits a host event (timer, fetch) cannot be
    /// evaluated through this method; wave 2 must drive such modules
    /// from the loop across turns instead of awaiting them here.
    /// Ticking work (infinite loops) is still bounded by the interrupt
    /// handler, which fires while jobs run.
    fn await_promise<F>(&mut self, make: F) -> Result<JsValue, JsError>
    where
        F: for<'js> FnOnce(
            Ctx<'js>,
        ) -> rquickjs::Result<rquickjs::Promise<'js>>,
    {
        self.check_usable()?;
        self.arm_budget(self.script_budget);
        let result: Result<JsValue, String> = self.with_ctx(|ctx| {
            let promise = match make(ctx.clone()) {
                Ok(promise) => promise,
                Err(e) => return Err(map_plain_error(&ctx, e)),
            };
            let mut jobs = 0usize;
            loop {
                if let Some(settled) = promise.result::<Value>() {
                    // Fulfilment value: a script/module completion
                    // value is `undefined` per spec, and — verified
                    // empirically against QuickJS-NG 0.15.1 —
                    // `Ctx::eval_promise` (JS_Eval + JS_EVAL_FLAG_ASYNC)
                    // fulfils with `{ value: <completion> }`, not
                    // with the completion itself, so the fulfillment
                    // value is never surfaced as the script's value.
                    // Callers that need a value capture it into a
                    // global (see the probe) and read it back with a
                    // classic eval. Rejection still surfaces the
                    // reason as an exception.
                    return match settled.catch(&ctx) {
                        Ok(_) => Ok(JsValue::Undefined),
                        Err(caught) => Err(caught.to_string()),
                    };
                }
                if !ctx.execute_pending_job() {
                    return Err(WOULD_BLOCK.to_string());
                }
                jobs += 1;
                if jobs > 100_000 {
                    return Err(
                        "microtask drain exceeded 100000 jobs (infinite microtask loop?)"
                            .to_string(),
                    );
                }
            }
        });
        let tripped = self.budget_tripped();
        self.disarm_budget();
        match result {
            Ok(value) => Ok(value),
            Err(msg) if msg == WOULD_BLOCK => {
                self.poisoned = true;
                Err(JsError::WatchdogTimeout)
            }
            Err(msg) => {
                if tripped || msg.contains("interrupted") {
                    Err(JsError::Interrupted)
                } else if msg.contains("out of memory") {
                    Err(JsError::MemoryLimit)
                } else {
                    Err(JsError::Exception(msg))
                }
            }
        }
    }

    /// Map a `Result<T, String>` from inside a `with` closure to
    /// [`JsError`], applying the budget/memory-limit classification.
    fn classify_call_result<T>(&self, result: Result<T, String>) -> Result<T, JsError> {
        match result {
            Ok(value) => Ok(value),
            Err(msg) => {
                if self.budget_tripped() || msg.contains("interrupted") {
                    Err(JsError::Interrupted)
                } else if msg.contains("out of memory") {
                    Err(JsError::MemoryLimit)
                } else {
                    Err(JsError::Exception(msg))
                }
            }
        }
    }

    /// Evaluate `source` as a classic script and format its value
    /// to a string on this thread (Stage 17, M9 stages 1–2 — the
    /// DevTools REPL). The same shape as the `Classic` arm of
    /// [`JsEngine::eval`], except the live `Value` is rendered
    /// with [`crate::bindings::format_console_arg`] (like
    /// `String(value)`) before it leaves the `Ctx`: a REPL result
    /// is a formatted string, never a live JS value crossing
    /// threads. A throw is the caught value's string, with its
    /// stack when rquickjs appends one — the same string
    /// `eval` would have classified as an exception.
    pub(crate) fn eval_formatted(&mut self, source: &str) -> Result<String, JsError> {
        self.check_usable()?;
        self.arm_budget(self.script_budget);
        let result = self.with_ctx(|ctx| {
            let r: rquickjs::Result<Value> = ctx.eval(source);
            match r.catch(&ctx) {
                Ok(value) => Ok(crate::bindings::format_console_arg(&ctx, &value)),
                Err(caught) => Err(caught.to_string()),
            }
        });
        let classified = self.classify_call_result(result);
        self.disarm_budget();
        classified
    }
}

impl JsEngine for TabRealm {
    fn eval(&mut self, kind: EvalKind, source: &str, filename: &str) -> Result<JsValue, JsError> {
        match kind {
            EvalKind::Classic => {
                self.check_usable()?;
                self.arm_budget(self.script_budget);
                let result = self.with_ctx(|ctx| {
                    let r: rquickjs::Result<Value> = ctx.eval(source);
                    match r.catch(&ctx) {
                        Ok(value) => Ok(value_to_js(&ctx, value)),
                        Err(caught) => Err(caught.to_string()),
                    }
                });
                // Classify while the budget is still armed.
                let classified = self.classify_call_result(result);
                self.disarm_budget();
                // A classic script has no promise to await; jobs it
                // queued are drained by the loop's microtask checkpoint.
                // The filename is recorded for stack traces via the
                // eval options in a later wave; today `eval` uses the
                // default filename.
                let _ = filename;
                classified
            }
            EvalKind::ClassicAwait => self.await_promise(|ctx| ctx.eval_promise(source)),
            EvalKind::Module => self.eval_module(filename, source),
        }
    }

    fn eval_module(&mut self, url: &str, source: &str) -> Result<JsValue, JsError> {
        self.await_promise(|ctx| Module::evaluate(ctx, url, source))
    }

    fn compile_module(&mut self, url: &str, source: &str) -> Result<Vec<u8>, JsError> {
        self.check_usable()?;
        self.arm_budget(self.script_budget);
        let result = self.with_ctx(|ctx| {
            let module = match Module::declare(ctx.clone(), url, source) {
                Ok(module) => module,
                Err(e) => return Err(map_plain_error(&ctx, e)),
            };
            match module.write(WriteOptions::default()) {
                Ok(bytes) => Ok(bytes),
                Err(e) => Err(map_plain_error(&ctx, e)),
            }
        });
        // Classify while the budget is still armed.
        let classified = self.classify_call_result(result);
        self.disarm_budget();
        classified
    }

    fn eval_module_bytecode(&mut self, _url: &str, bytecode: &[u8]) -> Result<JsValue, JsError> {
        // SAFETY: `Module::load` requires valid QuickJS bytecode. The
        // only caller is the bytecode cache, which stores exactly the
        // bytes `compile_module` produced in this process for this
        // engine version — never foreign or on-disk bytes.
        self.await_promise(|ctx| {
            let module = unsafe { Module::load(ctx.clone(), bytecode) };
            match module {
                Ok(module) => module.eval().map(|(_module, promise)| promise),
                Err(e) => Err(e),
            }
        })
    }

    fn drain_jobs(&mut self) -> Result<usize, JsError> {
        self.check_usable()?;
        let mut ran = 0;
        // The sync drain swallows a thrown job's exception (the wrapper
        // catches it); reporting goes through the runtime's rejection
        // tracker, not through here. A job that throws therefore does
        // not stop the drain, and this method cannot report it — install
        // a rejection sink on the realm if thrown jobs must be visible.
        loop {
            if !self.drain_one_job() {
                break;
            }
            ran += 1;
            if ran > 100_000 {
                return Err(JsError::Internal(
                    "microtask drain exceeded 100000 jobs (infinite microtask loop?)".to_string(),
                ));
            }
        }
        Ok(ran)
    }

    fn memory_usage_bytes(&self) -> u64 {
        self.assert_owner();
        let usage = block_on_sync(self.runtime.memory_usage());
        usage.memory_used_size as u64
    }
}

/// Drive a future to completion synchronously, on the owner
/// thread, with no executor (wave 2b — see the module docs).
///
/// The rquickjs futures driven here (`AsyncContext::with`,
/// `AsyncContext::full`, `set_memory_limit`,
/// `set_interrupt_handler`, `is_job_pending`, `memory_usage`) only
/// ever await the runtime's async-lock, which is uncontended on the
/// single owner thread: they complete on the first poll, so this is
/// a straight-line call, not a spin. If one ever parks (a future
/// that waits on a timer, I/O, or a contended lock reached this
/// driver by mistake), the loop yields instead of burning CPU — and
/// hangs rather than completing, which is the honest failure for a
/// future this driver was never meant to run. Do NOT "fix" that by
/// reaching for a tokio `block_on` here: the realm is entered from
/// inside the tab thread's network `block_on`, where a nested
/// `block_on` panics.
fn block_on_sync<F: Future>(mut fut: F) -> F::Output {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    // SAFETY: the future is never moved after pinning.
    let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
    let mut spins = 0u32;
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => {
                spins += 1;
                if spins < 1_000 {
                    std::hint::spin_loop();
                } else {
                    // A parked future: yield so a sibling thread can
                    // release whatever it waits on (it never should
                    // park — see the docs above).
                    std::thread::yield_now();
                }
            }
        }
    }
}


/// Convert an engine `Value` to the engine-neutral [`JsValue`].
/// Structured values cross as markers; primitives cross by value.
pub(crate) fn value_to_js(_ctx: &Ctx<'_>, value: Value<'_>) -> JsValue {
    match value.type_of() {
        Type::Undefined => JsValue::Undefined,
        Type::Null => JsValue::Null,
        Type::Bool => JsValue::Bool(value.as_bool().unwrap_or(false)),
        Type::Int | Type::Float => JsValue::Number(value.as_number().unwrap_or(f64::NAN)),
        Type::String => JsValue::String(
            value
                .as_string()
                .and_then(|s| s.to_string().ok())
                .unwrap_or_default(),
        ),
        Type::Function | Type::Constructor => JsValue::Function,
        _ => JsValue::Object,
    }
}

/// Build the runtime's unhandled-rejection tracker: a promise that
/// rejects with no handler is reported to `sender` as a string. The
/// reason is stringified with `Value`'s own accessors (no `Ctx`
/// needed), which keeps the closure higher-ranked over the value
/// lifetime.
fn make_rejection_tracker(sender: mpsc::Sender<String>) -> rquickjs::runtime::RejectionTracker {
    Box::new(
        move |_ctx: Ctx<'_>, _promise: Value<'_>, reason: Value<'_>, handled: bool| {
            if handled {
                return;
            }
            let text = if let Some(s) = reason.as_string() {
                s.to_string()
                    .unwrap_or_else(|_| "<unprintable rejection>".to_string())
            } else if let Some(obj) = reason.as_object() {
                obj.get::<_, String>("message")
                    .unwrap_or_else(|_| format!("unhandled rejection: {reason:?}"))
            } else {
                format!("unhandled rejection: {reason:?}")
            };
            let _ = sender.send(text);
        },
    )
}

/// Map a plain rquickjs error (no pending JS exception to catch) to a
/// message string.
fn map_plain_error(ctx: &Ctx<'_>, err: rquickjs::Error) -> String {
    // A failed evaluation usually leaves the thrown JS value on
    // the context (loader/resolver failures, throws, syntax
    // errors), and that value carries the real message — the Rust
    // error alone stringifies to a generic "Exception generated
    // by QuickJS", which hides the cause (Stage 12 Rec 1: a
    // missing import must name the module). Prefer the caught
    // value when one is pending; fall back to the Rust error
    // when none is (OOM / allocation failures).
    let caught = ctx.catch();
    if !caught.is_undefined() && !caught.is_null() {
        return crate::bindings::format_console_arg(ctx, &caught);
    }
    err.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn test_realm() -> (TabRealm, mpsc::Receiver<String>) {
        let (tx, rx) = mpsc::channel();
        let config = RealmConfig {
            rejection_sink: Some(tx),
            ..RealmConfig::default()
        };
        (TabRealm::new(&config).unwrap(), rx)
    }

    #[test]
    fn eval_returns_primitives() {
        let (mut realm, _rx) = test_realm();
        assert_eq!(
            realm.eval(EvalKind::Classic, "1 + 2", "test.js").unwrap(),
            JsValue::Number(3.0)
        );
        assert_eq!(
            realm.eval(EvalKind::Classic, "'a' + 'b'", "test.js").unwrap(),
            JsValue::String("ab".to_string())
        );
        assert_eq!(
            realm.eval(EvalKind::Classic, "1 < 2", "test.js").unwrap(),
            JsValue::Bool(true)
        );
        assert_eq!(
            realm.eval(EvalKind::Classic, "undefined", "test.js").unwrap(),
            JsValue::Undefined
        );
        assert_eq!(
            realm.eval(EvalKind::Classic, "null", "test.js").unwrap(),
            JsValue::Null
        );
    }

    #[test]
    fn script_error_is_isolated() {
        let (mut realm, _rx) = test_realm();
        let err = realm
            .eval(EvalKind::Classic, "throw new Error('boom')", "test.js")
            .unwrap_err();
        assert!(matches!(err, JsError::Exception(_)));
        assert!(err.to_string().contains("boom"));
        // The realm survives a throwing script.
        assert_eq!(
            realm.eval(EvalKind::Classic, "40 + 2", "test.js").unwrap(),
            JsValue::Number(42.0)
        );
    }

    #[test]
    fn syntax_error_is_an_exception() {
        let (mut realm, _rx) = test_realm();
        let err = realm
            .eval(EvalKind::Classic, "const = 1;", "test.js")
            .unwrap_err();
        assert!(matches!(err, JsError::Exception(_)));
    }

    #[test]
    fn interrupt_handler_kills_a_spin_loop() {
        let (tx, _rx) = mpsc::channel();
        let config = RealmConfig {
            script_budget: Duration::from_millis(100),
            rejection_sink: Some(tx),
            ..RealmConfig::default()
        };
        let mut realm = TabRealm::new(&config).unwrap();
        let err = realm
            .eval(EvalKind::Classic, "while (true) {}", "test.js")
            .unwrap_err();
        assert_eq!(err, JsError::Interrupted);
        // The realm survives the kill.
        assert_eq!(
            realm.eval(EvalKind::Classic, "7 * 6", "test.js").unwrap(),
            JsValue::Number(42.0)
        );
    }

    #[test]
    fn watchdog_kills_a_hung_await_and_poisons_the_realm() {
        let (tx, _rx) = mpsc::channel();
        let config = RealmConfig {
            script_budget: Duration::from_millis(100),
            rejection_sink: Some(tx),
            ..RealmConfig::default()
        };
        let mut realm = TabRealm::new(&config).unwrap();
        let err = realm
            .eval(
                EvalKind::ClassicAwait,
                "await new Promise(() => {})",
                "test.js",
            )
            .unwrap_err();
        assert_eq!(err, JsError::WatchdogTimeout);
        assert!(realm.is_poisoned());
        assert_eq!(
            realm.eval(EvalKind::Classic, "1", "test.js").unwrap_err(),
            JsError::Poisoned
        );
    }

    #[test]
    fn unhandled_rejection_is_reported() {
        let (mut realm, rx) = test_realm();
        realm
            .eval(
                EvalKind::Classic,
                "Promise.resolve().then(() => { throw new Error('reject-boom'); });",
                "test.js",
            )
            .unwrap();
        realm.drain_jobs().unwrap();
        let reported: Vec<String> = rx.try_iter().collect();
        assert!(
            reported.iter().any(|m| m.contains("reject-boom")),
            "expected rejection report, got {reported:?}"
        );
    }

    #[test]
    fn memory_usage_is_reported() {
        let (realm, _rx) = test_realm();
        assert!(realm.memory_usage_bytes() > 0);
    }

    /// Top-level-await results are observed via a global, not via the
    /// eval promise's fulfilment value: `eval_promise` fulfils with
    /// `{ value: <completion> }` (verified empirically against
    /// QuickJS-NG 0.15.1), so `await_promise` returns `Undefined` and
    /// the probe reads async results back from a global.
    #[test]
    fn top_level_await_result_is_observed_via_global() {
        let (mut realm, _rx) = test_realm();
        realm
            .eval(
                EvalKind::ClassicAwait,
                "globalThis.__got = await Promise.resolve(42);",
                "test.js",
            )
            .unwrap();
        assert_eq!(
            realm.eval(EvalKind::Classic, "globalThis.__got", "test.js").unwrap(),
            JsValue::Number(42.0)
        );
    }
}
