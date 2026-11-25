//! The per-tab JS host: event loop + realm, wired together (Stage 7).
//!
//! [`TabJs`] is what the per-tab worker owns: one [`LoopState`] (the
//! event loop) and one [`TabRealm`] (one realm per tab, fresh per
//! navigation). Wave 2 builds the DOM bindings on top of this; this wave
//! proves the loop, the realm, and their ordering with pure JS.
//!
//! # Host globals installed in the realm
//!
//! The loop owns these host APIs (they are loop infrastructure, not DOM
//! bindings, so they land in this wave):
//!
//! * `queueMicrotask(cb)` — defined in JS as `Promise.resolve().then(cb)`.
//!   That routes the callback through the engine's own job queue, so its
//!   order relative to promise reactions is exactly the order the
//!   statements ran in. A separate Rust queue could not observe that
//!   order (see `event_loop::Microtask`).
//! * `setTimeout` / `clearTimeout` / `setInterval` / `clearInterval` —
//!   Rust host functions over the timer wheel. Callbacks live in the
//!   registry's `timers` Map (symbol-keyed — wave 2a fixed the
//!   wave-1 gap where the Maps were reachable by name); the Rust side
//!   stores only ids, never engine values.
//! * `requestAnimationFrame` / `cancelAnimationFrame` — callbacks live
//!   in the registry's `raf` Map and run inside update-the-rendering
//!   with a shared frame timestamp.
//! * `performance.now()` — ms since this realm was created.
//!
//! Callbacks are invoked with the global as `this` (standing in for
//! `window` until wave 2 installs the real one).
//!
//! # Navigation
//!
//! [`TabJs::navigate`] drops the realm and creates a fresh one (fresh
//! globals, fresh wrapper cache in wave 2), clears every queue, and
//! bumps the epoch. Tasks, timers, and rAF callbacks queued under the
//! old epoch are dropped at dispatch (and cleared outright by
//! `navigate`). A watchdog kill is handled the same way via
//! [`TabJs::recreate_realm`]: the poisoned realm is dropped, never
//! re-entered.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rquickjs::function::{Opt, This};
use rquickjs::{Ctx, Function, Object};

use crate::bindings::{
    BindingState, FetchBackend, FetchCompletion, call_onerror, install_bindings, registry_map,
};
use crate::bindings::dom::fire_image_event;
use crate::bindings::events::TypedEventKind;
use crate::bindings::permissions::PermissionsBackend;
use crate::bindings::storage::StorageBackend;
use crate::bytecode::{BytecodeCache, CacheHit};
use crate::engine::{EvalKind, JsEngine, JsValue};
use crate::error::JsError;
use crate::module_loader::ModuleSources;
use crate::event_loop::{
    FRAME_INTERVAL, LoopState, Microtask, TaskKind, TaskSource,
};
use crate::host::{DomHost, DomKey};
use crate::observers::{
    ComputedIntersection, RootMargin, MAX_ROUNDS, ObservedSizes, ResizeBoxKind, ResizeStepOutcome,
};
use crate::realm::{RealmConfig, TabRealm};

/// Bytecode cache capacity per tab realm (8 MiB of compiled modules).
pub const DEFAULT_BYTECODE_CAPACITY_BYTES: usize = 8 * 1024 * 1024;

/// Wall-clock cap for `run_until_idle`, so a test or a page that keeps
/// scheduling work cannot spin the worker thread forever.
const RUN_UNTIL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// What one [`TabJs::run_turn`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    /// A task ran to completion (then the checkpoint and rendering ran).
    RanTask,
    /// A stale-epoch task was dropped at dispatch (then the checkpoint
    /// and rendering ran).
    DroppedStaleTask,
    /// No task was runnable; only the checkpoint and rendering ran.
    Idle,
}

/// The JS host for one tab: the event loop plus its realm.
pub struct TabJs {
    /// Event-loop state, shared with the host-function closures
    /// installed in the realm (they enqueue timers / rAF callbacks).
    /// Pure Rust data — no engine values — so it can be dropped in any
    /// order relative to the realm.
    state: Rc<RefCell<LoopState>>,
    /// One realm per tab. Dropped and recreated on navigation.
    realm: TabRealm,
    /// Unhandled-rejection reports from the realm's tracker.
    rejection_rx: std::sync::mpsc::Receiver<String>,
    /// Config reused when the realm is recreated (navigation, watchdog).
    config: RealmConfig,
    /// Per-realm bytecode cache (fresh on navigation).
    bytecode: BytecodeCache,
    /// Wave-2a binding state (host document, fetch backend, wrapper
    /// identity maps, console capture). Plain Rust data — no JS
    /// values — shared with the binding host functions. Fresh per
    /// navigation; the host/backend below are re-attached to it.
    bindings: Rc<RefCell<BindingState>>,
    /// The host document, re-attached to each fresh [`BindingState`]
    /// on navigation (set by the per-tab worker).
    host: Option<Rc<RefCell<Box<dyn DomHost>>>>,
    /// The fetch backend, re-attached the same way.
    fetch_backend: Option<std::sync::Arc<dyn FetchBackend>>,
    /// The storage backend, re-attached the same way (Stage 9
    /// Rec 4: the shell attaches its profile-backed backend).
    storage_backend: Option<std::sync::Arc<dyn StorageBackend>>,
    /// The permissions backend, re-attached the same way
    /// (Stage 9 Rec 11: the shell attaches its profile-backed
    /// backend).
    permissions_backend: Option<std::sync::Arc<dyn PermissionsBackend>>,
    /// Fetch outcomes delivered by backends (any thread). Drained
    /// each turn into Networking-source resolution tasks.
    fetch_tx: std::sync::mpsc::Sender<FetchCompletion>,
    /// Receiving end of the fetch-completion channel.
    fetch_rx: std::sync::mpsc::Receiver<FetchCompletion>,
    /// Loop-driven modules (wave 2b): ids started by
    /// [`TabJs::start_module_async`] whose promises are still
    /// pending; polled each turn by [`TabJs::poll_modules`].
    pending_modules: Vec<u64>,
    /// Next loop-driven module id.
    next_module_id: u64,
    /// Pre-fetched module sources for this navigation (Stage 12
    /// Rec 1): the browser's load task fills this, the realm's
    /// loader reads it. Replaced on navigation, so a dying load
    /// task cannot leak old sources into the new document.
    module_sources: ModuleSources,
}

impl TabJs {
    /// Create a tab host on the current thread (the tab's worker
    /// thread). The realm and its host globals are installed here.
    pub fn new(config: &RealmConfig) -> Result<Self, JsError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut config = config.clone();
        config.rejection_sink = Some(tx);
        let state = Rc::new(RefCell::new(LoopState::new()));
        let bindings = Rc::new(RefCell::new(BindingState::new()));
        let (fetch_tx, fetch_rx) = std::sync::mpsc::channel();
        let mut realm = TabRealm::new(&config)?;
        Self::install_globals(&mut realm, &state)?;
        install_bindings(&mut realm, &state, &bindings, fetch_tx.clone())?;
        let mut tab = Self {
            state,
            realm,
            rejection_rx: rx,
            config,
            bytecode: BytecodeCache::new(DEFAULT_BYTECODE_CAPACITY_BYTES),
            bindings,
            host: None,
            fetch_backend: None,
            storage_backend: None,
            permissions_backend: None,
            fetch_tx,
            fetch_rx,
            pending_modules: Vec::new(),
            next_module_id: 1,
            module_sources: ModuleSources::new(),
        };
        tab.install_module_loader()?;
        Ok(tab)
    }

    /// Install the module loader on the current realm: sources
    /// shared with the load task. Called from `new` and from
    /// `navigate`.
    fn install_module_loader(&mut self) -> Result<(), JsError> {
        self.realm
            .install_module_loader(self.module_sources.clone())
    }

    /// Attach the host document (the per-tab worker's `Dom`). The
    /// bindings validate every `DomKey` against it; until it is
    /// attached, document bindings throw `InvalidStateError`.
    pub fn set_dom_host(&mut self, host: Box<dyn DomHost>) {
        let shared = Rc::new(RefCell::new(host));
        self.host = Some(Rc::clone(&shared));
        self.bindings.borrow_mut().host = Some(shared);
    }

    /// Attach the `fetch()` backend (the integration wave's
    /// `pluto-net` backend). Until it is attached, `fetch()` rejects.
    pub fn set_fetch_backend(&mut self, backend: std::sync::Arc<dyn FetchBackend>) {
        self.fetch_backend = Some(std::sync::Arc::clone(&backend));
        self.bindings.borrow_mut().fetch_backend = Some(backend);
    }

    /// Attach the storage backend (Stage 9 Rec 4: the shell's
    /// profile-backed `PlutoStorageBackend`). Until it is
    /// attached, storage bindings use the in-memory default.
    pub fn set_storage_backend(&mut self, backend: std::sync::Arc<dyn StorageBackend>) {
        self.storage_backend = Some(std::sync::Arc::clone(&backend));
        self.bindings.borrow_mut().storage_backend = backend;
    }

    /// Attach the permissions backend (Stage 9 Rec 11: the
    /// shell's profile-backed `PlutoPermissionsBackend`).
    /// Until it is attached, permission queries use the
    /// in-memory default.
    pub fn set_permissions_backend(
        &mut self,
        backend: std::sync::Arc<dyn PermissionsBackend>,
    ) {
        self.permissions_backend = Some(std::sync::Arc::clone(&backend));
        self.bindings.borrow_mut().permissions_backend = backend;
    }

    /// Ask for a rendering opportunity at the next frame, even with
    /// no rAF callback pending (the integration wave calls this when
    /// a DOM mutation marked the document dirty). `maybe_render`
    /// clears it when the frame runs.
    pub fn request_render(&mut self) {
        self.state.borrow_mut().render_requested = true;
    }

    /// True while any loop work remains (public form of the
    /// `run_until_idle` predicate, for the integration wave's
    /// frame-cadence driver).
    pub fn work_pending(&self) -> bool {
        self.has_pending_work()
    }

    /// True when the realm was killed by the watchdog (or a script
    /// error poisoned it): the integration wave recreates the realm
    /// rather than letting the tab wedge.
    pub fn realm_poisoned(&self) -> bool {
        self.realm.is_poisoned()
    }

    /// Start a module evaluation without awaiting it (wave 2b,
    /// loop-driven modules — see `TabRealm::start_module`). The
    /// module's top-level code runs up to its first suspension
    /// now; if it top-level-awaits a host event (timer, fetch),
    /// the loop drives it across turns and settlement is reported
    /// through `script_errors` (rejection) like any other script
    /// error. A synchronous throw is recorded the same way, not
    /// returned — use `script_errors` to read it. Only a
    /// realm-killing error (watchdog / poisoned) is returned.
    pub fn start_module_async(&mut self, url: &str, source: &str) -> Result<(), JsError> {
        let id = self.next_module_id;
        self.next_module_id += 1;
        match self.realm.start_module(url, source, id) {
            Ok(()) => {
                self.pending_modules.push(id);
                Ok(())
            }
            Err(e) => self.record_or_propagate_with_source(e, Some(url)),
        }
    }

    /// Poll loop-driven modules for settlement (called from
    /// `run_turn`, after the microtask checkpoint). A fulfilled
    /// module just leaves the pending list; a rejected one is
    /// recorded as a script error (failed modules reject their
    /// dependents — the loop never hangs on them).
    fn poll_modules(&mut self) -> Result<(), JsError> {
        let mut i = 0;
        while i < self.pending_modules.len() {
            let id = self.pending_modules[i];
            match self.realm.poll_module(id) {
                None => i += 1,
                Some(Ok(())) => {
                    self.pending_modules.remove(i);
                }
                Some(Err(reason)) => {
                    self.pending_modules.remove(i);
                    self.record_or_propagate(JsError::Exception(format!(
                        "module rejected: {reason}"
                    )))?;
                }
            }
        }
        Ok(())
    }

    /// Captured `console` output as `(level, message)` pairs.
    pub fn console_lines(&self) -> Vec<(String, String)> {
        self.bindings.borrow().console_lines.clone()
    }

    /// Dispatch an external event (a shell click, a key press) to a
    /// node: enqueued as a `UserInteraction`-source task, so the
    /// listeners run on the loop, not inline in the caller.
    pub fn dispatch_event(&mut self, target: DomKey, event_type: &str, bubbles: bool) {
        let event_type = event_type.to_string();
        self.enqueue_host_task(
            TaskSource::UserInteraction,
            Box::new(move |tab| {
                let _ = tab.dispatch_event_now(target, &event_type, bubbles);
            }),
        );
    }

    /// Run an external-event dispatch now (inside a loop task).
    pub(crate) fn dispatch_event_now(
        &mut self,
        target: DomKey,
        event_type: &str,
        bubbles: bool,
    ) -> Result<(), JsError> {
        let bindings = Rc::clone(&self.bindings);
        let event_type = event_type.to_string();
        self.realm
            .with_ctx(|ctx| {
                crate::bindings::events::dispatch_external(
                    &ctx, &bindings, target, &event_type, bubbles,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(JsError::Internal)
    }

    /// Dispatch an external *typed* event (a shell key press, mouse
    /// click, or text input) to a node: enqueued as a
    /// `UserInteraction`-source task, so the listeners run on the
    /// loop, not inline in the caller. The shell's input dispatch
    /// (renderer input handling) calls this instead of
    /// [`Self::dispatch_event`] for keyboard / mouse / input events
    /// — see `TypedEventKind`; the plain `dispatch_event` stays for
    /// untyped events (`click` from `element.click()`,
    /// `DOMContentLoaded`).
    pub fn dispatch_typed_event(
        &mut self,
        target: DomKey,
        event_type: &str,
        bubbles: bool,
        kind: TypedEventKind,
    ) {
        let event_type = event_type.to_string();
        self.enqueue_host_task(
            TaskSource::UserInteraction,
            Box::new(move |tab| {
                let _ = tab.dispatch_typed_event_now(target, &event_type, bubbles, kind);
            }),
        );
    }

    /// Dispatch a typed external event synchronously and report
    /// whether its default action was NOT prevented (Stage 13 Rec
    /// 2/7): the renderer calls this from its input path — a
    /// `keydown` listener's `preventDefault()` must suppress the
    /// following `input` / insertion, and a `submit` listener's
    /// `preventDefault()` must block the navigation — so the
    /// answer has to come back inline, not through the queued
    /// `dispatch_typed_event`. Runs the listeners on the calling
    /// thread (the tab thread), exactly like the loop task would.
    pub fn dispatch_typed_event_sync(
        &mut self,
        target: DomKey,
        event_type: &str,
        bubbles: bool,
        kind: TypedEventKind,
    ) -> Result<bool, JsError> {
        let bindings = Rc::clone(&self.bindings);
        let event_type = event_type.to_string();
        self.realm
            .with_ctx(|ctx| {
                crate::bindings::events::dispatch_typed_external_bool(
                    &ctx, &bindings, target, &event_type, bubbles, &kind,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(JsError::Internal)
    }

    /// Dispatch a plain external event synchronously, reporting
    /// whether its default action was NOT prevented (Stage 13 Rec
    /// 7: cancelable `submit` / `reset` events from the renderer).
    pub fn dispatch_event_sync(
        &mut self,
        target: DomKey,
        event_type: &str,
        bubbles: bool,
    ) -> Result<bool, JsError> {
        let bindings = Rc::clone(&self.bindings);
        let event_type = event_type.to_string();
        self.realm
            .with_ctx(|ctx| {
                crate::bindings::events::dispatch_external_bool(
                    &ctx, &bindings, target, &event_type, bubbles,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(JsError::Internal)
    }

    /// Dispatch a `storage` event to the window (Stage 13 Rec
    /// 10d): called by the tab thread when another tab mutated
    /// the shared store for this document's origin. Never to the
    /// mutating tab itself — the caller filters by tab id.
    pub fn dispatch_storage_event(
        &mut self,
        key: Option<&str>,
        old_value: Option<&str>,
        new_value: Option<&str>,
        url: &str,
    ) -> Result<(), JsError> {
        let bindings = Rc::clone(&self.bindings);
        let url = url.to_string();
        self.realm
            .with_ctx(|ctx| {
                crate::bindings::events::dispatch_storage_event(
                    &ctx, &bindings, key, old_value, new_value, &url,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(JsError::Internal)
    }

    /// Run a typed external-event dispatch now (inside a loop task).
    pub(crate) fn dispatch_typed_event_now(
        &mut self,
        target: DomKey,
        event_type: &str,
        bubbles: bool,
        kind: TypedEventKind,
    ) -> Result<(), JsError> {
        let bindings = Rc::clone(&self.bindings);
        let event_type = event_type.to_string();
        self.realm
            .with_ctx(|ctx| {
                crate::bindings::events::dispatch_typed_external(
                    &ctx, &bindings, target, &event_type, bubbles, &kind,
                )
                .map_err(|e| e.to_string())
            })
            .map_err(JsError::Internal)
    }

    /// Report an `Image()` / `<img>` load outcome from the shell's
    /// Stage 11 image store: marks the image complete with its
    /// intrinsic size and fires a `load` event on the element (plus
    /// its `onload` handler). Enqueued as a `UserInteraction`-source
    /// task so the event runs on the loop.
    pub fn notify_image_loaded(&mut self, target: DomKey, width: u32, height: u32) {
        self.enqueue_host_task(
            TaskSource::UserInteraction,
            Box::new(move |tab| {
                let bindings = Rc::clone(&tab.bindings);
                let _ = tab.realm.with_ctx(|ctx| {
                    fire_image_event(&ctx, &bindings, target, "load", width, height)
                        .map_err(|e| e.to_string())
                });
            }),
        );
    }

    /// Report an `Image()` / `<img>` load failure: fires an `error`
    /// event on the element (plus its `onerror` handler).
    pub fn notify_image_failed(&mut self, target: DomKey) {
        self.enqueue_host_task(
            TaskSource::UserInteraction,
            Box::new(move |tab| {
                let bindings = Rc::clone(&tab.bindings);
                let _ = tab.realm.with_ctx(|ctx| {
                    fire_image_event(&ctx, &bindings, target, "error", 0, 0)
                        .map_err(|e| e.to_string())
                });
            }),
        );
    }

    /// Current navigation epoch.
    pub fn epoch(&self) -> u64 {
        self.state.borrow().epoch
    }

    /// Uncaught script errors recorded so far (per-script boundary).
    /// Drains the rejection channel first, so unhandled rejections are
    /// included, and merges binding-recorded errors (throwing event
    /// listeners, `fetch` resolution failures).
    pub fn script_errors(&self) -> Vec<String> {
        self.drain_rejections();
        let mut errors = self.state.borrow().script_errors.clone();
        errors.extend(self.bindings.borrow().errors.iter().cloned());
        errors
    }

    /// Heap use of this tab's realm in bytes.
    pub fn memory_usage_bytes(&self) -> u64 {
        self.realm.memory_usage_bytes()
    }

    /// The bytecode cache, for the integration wave to load modules
    /// through (and for tests to inspect hit/miss counts).
    pub fn bytecode_cache(&mut self) -> &mut BytecodeCache {
        &mut self.bytecode
    }

    /// The shared pre-fetched module sources for this navigation.
    /// The browser's load task clones this handle and fills it
    /// while fetching; the realm's loader reads from it at link
    /// time. Fresh per navigation (see `navigate`).
    pub fn module_sources(&self) -> ModuleSources {
        self.module_sources.clone()
    }

    /// Load a module script through the bytecode cache: compile once
    /// per (URL, source), reuse on repeat loads. Returns the completion
    /// value and whether the bytecode was reused.
    pub fn load_module_cached(
        &mut self,
        url: &str,
        source: &str,
    ) -> Result<(JsValue, CacheHit), JsError> {
        self.bytecode.load_or_compile(&mut self.realm, url, source)
    }

    /// Enqueue a classic script as a `Script`-source task for the
    /// current epoch (the parser seam wave 2 / `pluto-html` calls).
    pub fn enqueue_script(&mut self, source: &str, url: &str) {
        self.state.borrow_mut().push_task(
            TaskSource::Script,
            TaskKind::Script {
                source: source.to_string(),
                url: url.to_string(),
            },
        );
    }

    /// Enqueue a Rust-side task (for the host: fetch completions,
    /// parser work, input dispatch in later waves).
    pub fn enqueue_host_task(
        &mut self,
        source: TaskSource,
        task: Box<dyn FnOnce(&mut TabJs)>,
    ) {
        self.state
            .borrow_mut()
            .push_task(source, TaskKind::Host(task));
    }

    /// Evaluate a classic script as a task: enqueue it and run one turn.
    /// A throwing script is recorded in [`TabJs::script_errors`], not
    /// returned — use [`TabJs::eval_value`] for an expression whose
    /// value (or throw) the caller needs. A watchdog kill or a poisoned
    /// realm IS returned: the host must recreate the realm.
    pub fn eval_script(&mut self, source: &str, url: &str) -> Result<(), JsError> {
        self.enqueue_script(source, url);
        self.run_turn()?;
        Ok(())
    }

    /// Evaluate an expression and return its value (host inspection,
    /// not a script task — used by tests and by the host to read page
    /// state back). A throw is returned as [`JsError::Exception`].
    /// Pending microtasks are drained first so the read sees settled
    /// state.
    pub fn eval_value(&mut self, source: &str) -> Result<JsValue, JsError> {
        self.microtask_checkpoint()?;
        self.realm.eval(EvalKind::Classic, source, "host-read.js")
    }

    /// Navigate: fresh realm, cleared queues, bumped epoch. Pending
    /// work for the old document is dropped, never run. The host
    /// document and fetch backend are re-attached to the fresh
    /// binding state; listeners, wrappers, and pending fetches die
    /// with the old realm.
    pub fn navigate(&mut self) -> Result<(), JsError> {
        // Rejections for the old document are not carried over.
        while self.rejection_rx.try_recv().is_ok() {}
        let old_epoch = self.state.borrow().epoch;
        let mut new_state = LoopState::new();
        new_state.epoch = old_epoch + 1;
        let new_state = Rc::new(RefCell::new(new_state));
        let new_bindings = Rc::new(RefCell::new(BindingState::new()));
        if let Some(host) = &self.host {
            new_bindings.borrow_mut().host = Some(Rc::clone(host));
        }
        if let Some(backend) = &self.fetch_backend {
            new_bindings.borrow_mut().fetch_backend = Some(std::sync::Arc::clone(backend));
        }
        if let Some(backend) = &self.storage_backend {
            new_bindings.borrow_mut().storage_backend = std::sync::Arc::clone(backend);
        }
        if let Some(backend) = &self.permissions_backend {
            new_bindings.borrow_mut().permissions_backend = std::sync::Arc::clone(backend);
        }
        let (fetch_tx, fetch_rx) = std::sync::mpsc::channel();
        let mut new_realm = TabRealm::new(&self.config)?;
        Self::install_globals(&mut new_realm, &new_state)?;
        install_bindings(&mut new_realm, &new_state, &new_bindings, fetch_tx.clone())?;
        // Drop the old realm after the new one is installed: its
        // host-function closures hold `Rc` clones of the old state
        // and old bindings, and dropping the realm frees them. The
        // old state/bindings hold no engine values, so the order is
        // safe either way — this just frees memory promptly.
        let old_realm = std::mem::replace(&mut self.realm, new_realm);
        drop(old_realm);
        self.state = new_state;
        self.bindings = new_bindings;
        self.fetch_tx = fetch_tx;
        self.fetch_rx = fetch_rx;
        self.bytecode = BytecodeCache::new(DEFAULT_BYTECODE_CAPACITY_BYTES);
        self.pending_modules.clear();
        self.module_sources = ModuleSources::new();
        self.install_module_loader()?;
        Ok(())
    }

    /// Recreate the realm after a watchdog kill. Identical to a
    /// navigation for scheduling purposes (epoch bump, queues cleared):
    /// the poisoned realm's pending callbacks are gone with it.
    pub fn recreate_realm(&mut self) -> Result<(), JsError> {
        self.navigate()
    }

    /// Set the tab's visibility (Stage 8 Rec 1): a hidden tab runs
    /// no rAF callbacks and produces no frames until visible again.
    /// Timers, tasks, and microtasks are unaffected — only the
    /// rendering opportunity is gated (see `maybe_render`). Pending
    /// rAF callbacks stay queued and fire on the first visible
    /// frame; `next_frame_at` is left alone, so that frame is due
    /// immediately.
    pub fn set_visible(&mut self, visible: bool) {
        self.state.borrow_mut().visible = visible;
    }

    /// Whether the tab is currently visible.
    pub fn is_visible(&self) -> bool {
        self.state.borrow().visible
    }

    /// Every (observer, target, box) currently observed by a
    /// `ResizeObserver` in this tab. The browser's worker step
    /// uses this to ask the layout actor for exactly these
    /// targets' sizes.
    pub fn resize_observation_targets(&self) -> Vec<(u64, DomKey, ResizeBoxKind)> {
        self.bindings.borrow().resize_registry.targets()
    }

    /// Run one `ResizeObserver` gather/broadcast step from the
    /// host's geometry (`geometry()` for the border box,
    /// `content_size()` for the content box). The rendering-step
    /// driver: tests call this directly; the per-tab worker calls
    /// [`Self::run_resize_observers_with_sizes`] with layout-tree
    /// sizes instead (same registry, same loop — the size source
    /// is the only difference).
    pub fn run_resize_observers(&mut self) -> ResizeStepOutcome {
        let Some(host) = self.bindings.borrow().host.clone() else {
            return ResizeStepOutcome::default();
        };
        self.run_resize_rounds(
            &|key| crate::bindings::observers::host_observed_sizes(&host, key),
            &|key| host.borrow().depth(key).unwrap_or(0),
        )
    }

    /// Run one `ResizeObserver` gather/broadcast step from sizes
    /// the browser measured on the layout tree (border + content
    /// boxes, per target) and DOM depths. Called by the per-tab
    /// worker's rendering update, after layout and before the
    /// display list is built.
    pub fn run_resize_observers_with_sizes(
        &mut self,
        sizes: &HashMap<DomKey, ObservedSizes>,
        depths: &HashMap<DomKey, usize>,
    ) -> ResizeStepOutcome {
        self.run_resize_rounds(
            &|key| sizes.get(&key).copied(),
            &|key| depths.get(&key).copied().unwrap_or(0),
        )
    }

    /// The gather/broadcast loop: gather a round, deliver its
    /// batches, re-query sizes (a callback may have resized
    /// something), repeat while anything is active — up to
    /// [`MAX_ROUNDS`] rounds, after which the resize-loop error
    /// fires and the step stops with observations still active
    /// for the next step.
    fn run_resize_rounds(
        &mut self,
        sizes: &dyn Fn(DomKey) -> Option<ObservedSizes>,
        depth: &dyn Fn(DomKey) -> usize,
    ) -> ResizeStepOutcome {
        let mut outcome = ResizeStepOutcome::default();
        for round in 1..=MAX_ROUNDS {
            let batches = self
                .bindings
                .borrow_mut()
                .resize_registry
                .gather_round(sizes, depth);
            if batches.is_empty() {
                break;
            }
            outcome.delivered = true;
            outcome.rounds = round;
            for (id, entries) in batches {
                let bindings = Rc::clone(&self.bindings);
                if let Err(e) = crate::bindings::observers::deliver_resize_entries(
                    &mut self.realm,
                    &bindings,
                    id,
                    &entries,
                ) {
                    self.state.borrow_mut().script_errors.push(e.to_string());
                }
            }
            if !self
                .bindings
                .borrow()
                .resize_registry
                .has_active(sizes)
            {
                break;
            }
            if round == MAX_ROUNDS {
                outcome.loop_error = true;
                self.state.borrow_mut().script_errors.push(
                    "ResizeObserver loop completed with undelivered notifications."
                        .to_string(),
                );
            }
        }
        outcome
    }

    /// Every (observer, target, root, rootMargin) currently observed
    /// by an `IntersectionObserver` in this tab. The browser's
    /// worker step uses this to ask the layout actor for exactly
    /// these targets' intersections.
    pub fn intersection_observation_targets(
        &self,
    ) -> Vec<(u64, DomKey, Option<DomKey>, RootMargin)> {
        self.bindings.borrow().intersection_registry.targets()
    }

    /// Run one `IntersectionObserver` gather from geometry the
    /// browser measured on the layout tree, keyed by
    /// (observer, target). Entries that crossed a threshold band
    /// (or flipped `isIntersecting`) are queued in the registry,
    /// and ONE host task is enqueued to deliver them — so the
    /// callbacks run after paint, on a later turn, never inline in
    /// the rendering step (Stage 8 report §(e): an
    /// `IntersectionObserver` callback that writes to the DOM must
    /// not affect the same frame). `time_ms` is the computation
    /// timestamp the entries carry (ms since the tab started).
    pub fn run_intersection_observers_with_geometry(
        &mut self,
        geometry: &HashMap<(u64, DomKey), ComputedIntersection>,
        time_ms: f64,
    ) {
        let enqueue = {
            let mut bindings = self.bindings.borrow_mut();
            bindings
                .intersection_registry
                .gather(&|id, key| geometry.get(&(id, key)).copied(), time_ms)
        };
        if enqueue {
            self.enqueue_host_task(
                TaskSource::DomManipulation,
                Box::new(|tab| tab.deliver_pending_intersection_entries()),
            );
        }
    }

    /// Drain the pending `IntersectionObserver` entries and invoke
    /// their callbacks, one call per observer. Runs as the queued
    /// host task enqueued by
    /// [`Self::run_intersection_observers_with_geometry`]. A
    /// throwing callback is recorded as a script error, never fatal.
    pub fn deliver_pending_intersection_entries(&mut self) {
        let batches = {
            self.bindings
                .borrow_mut()
                .intersection_registry
                .take_pending()
        };
        for (id, entries) in batches {
            let bindings = Rc::clone(&self.bindings);
            if let Err(e) = crate::bindings::observers::deliver_intersection_entries(
                &mut self.realm,
                &bindings,
                id,
                &entries,
            ) {
                self.state.borrow_mut().script_errors.push(e.to_string());
            }
        }
    }

    /// Run one turn of the event loop: one task to completion, then the
    /// microtask checkpoint, then update-the-rendering at a rendering
    /// opportunity. Returns what the turn did. A script error is
    /// recorded, not returned; only a watchdog kill / poisoned realm
    /// propagates as `Err`.
    pub fn run_turn(&mut self) -> Result<TurnOutcome, JsError> {
        self.drain_fetch_completions();
        let task = self.state.borrow_mut().next_task();
        let outcome = match task {
            Some(task) => {
                if task.epoch != self.state.borrow().epoch {
                    TurnOutcome::DroppedStaleTask
                } else {
                    self.run_task(task)?;
                    TurnOutcome::RanTask
                }
            }
            None => TurnOutcome::Idle,
        };
        self.microtask_checkpoint()?;
        self.poll_modules()?;
        self.drain_rejections();
        self.maybe_render()?;
        Ok(outcome)
    }

    /// Run turns until no work remains (no tasks, microtasks, pending
    /// engine jobs, timers, rAF callbacks, or in-flight fetches),
    /// sleeping until the next timer deadline / frame time when the remaining work is all in
    /// the future. Bounded by [`RUN_UNTIL_IDLE_TIMEOUT`]; returns an
    /// error on timeout rather than spinning forever.
    pub fn run_until_idle(&mut self) -> Result<(), JsError> {
        let started = Instant::now();
        loop {
            if !self.has_pending_work() {
                return Ok(());
            }
            if started.elapsed() > RUN_UNTIL_IDLE_TIMEOUT {
                return Err(JsError::Internal(
                    "run_until_idle timed out (work keeps arriving)".to_string(),
                ));
            }
            let outcome = self.run_turn()?;
            if outcome == TurnOutcome::Idle && self.has_pending_work() {
                // Everything left is in the future: sleep until the
                // next timer deadline or the next frame time.
                if let Some(wake) = self.next_wake() {
                    let now = Instant::now();
                    if wake > now {
                        std::thread::sleep(wake - now);
                    }
                } else {
                    // Work the loop cannot see a wake time for (should
                    // not happen: jobs would have been drained by the
                    // checkpoint). Yield instead of spinning.
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Like [`TabJs::run_until_idle`], but with a caller-supplied
    /// budget (wave 2b: the per-tab worker drives the loop on its
    /// ~60 Hz frame cadence and must not block the load path for 30
    /// s). Returns `Ok` when the loop goes idle AND when the budget
    /// expires with work still pending (a repeating timer or a
    /// never-settling fetch is not an error — the next frame's drive
    /// picks it up). Only a realm-killing error propagates.
    pub fn drive_until_idle_or_timeout(&mut self, budget: Duration) -> Result<(), JsError> {
        let started = Instant::now();
        loop {
            if !self.has_pending_work() {
                return Ok(());
            }
            if started.elapsed() >= budget {
                return Ok(());
            }
            let outcome = self.run_turn()?;
            if outcome == TurnOutcome::Idle && self.has_pending_work() {
                let remaining = budget.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Ok(());
                }
                if let Some(wake) = self.next_wake() {
                    let now = Instant::now();
                    if wake > now {
                        std::thread::sleep((wake - now).min(remaining));
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// True if any work remains: tasks, microtasks, pending engine
    /// jobs, timers, rAF callbacks, in-flight fetches, or pending
    /// loop-driven modules.
    fn has_pending_work(&self) -> bool {
        self.state.borrow().has_work()
            || self.realm.is_job_pending()
            || !self.bindings.borrow().pending_fetches.is_empty()
            || !self.pending_modules.is_empty()
    }

    /// Move fetch outcomes delivered by backends into Networking
    /// -source resolution tasks. The backend never resolves a promise
    /// itself; resolution happens here, on the loop.
    fn drain_fetch_completions(&mut self) {
        let completions: Vec<FetchCompletion> = self.fetch_rx.try_iter().collect();
        for completion in completions {
            self.enqueue_host_task(
                TaskSource::Networking,
                Box::new(move |tab| {
                    let bindings = Rc::clone(&tab.bindings);
                    let result = tab.realm.with_ctx(|ctx| {
                        crate::bindings::fetch::resolve_fetch(&ctx, &bindings, completion)
                            .map_err(|e| e.to_string())
                    });
                    if let Err(e) = result {
                        tab.bindings
                            .borrow_mut()
                            .errors
                            .push(format!("fetch completion failed: {e}"));
                    }
                }),
            );
        }
    }

    /// When the loop should next wake while idle: the earliest timer
    /// deadline, or the next frame time if a frame is wanted. A hidden
    /// tab never wakes for a frame (Stage 8 Rec 1: zero rendering
    /// opportunities while hidden); its timers still wake it.
    fn next_wake(&self) -> Option<Instant> {
        let state = self.state.borrow();
        let mut wake = state.next_timer_deadline();
        if state.visible && (!state.raf.is_empty() || state.render_requested) {
            wake = Some(match wake {
                Some(t) => t.min(state.next_frame_at),
                None => state.next_frame_at,
            });
        }
        wake
    }

    /// Run one task to completion. Script errors are recorded in
    /// `script_errors`; only watchdog / poisoned propagate.
    fn run_task(&mut self, task: crate::event_loop::Task) -> Result<(), JsError> {
        match task.kind {
            TaskKind::Script { source, url } => {
                match self.realm.eval(EvalKind::Classic, &source, &url) {
                    Ok(_) => Ok(()),
                    Err(e) => self.record_or_propagate_with_source(e, Some(&url)),
                }
            }
            TaskKind::Timer { id, nesting, repeats } => {
                self.state.borrow_mut().current_nesting = nesting;
                // Call first, then consume: the callback must still be
                // in the Map for the lookup to find it. A re-entrant
                // clearTimeout during the callback just makes the
                // post-call removal a no-op. Intervals keep their
                // callback in the Map for the next firing.
                let result = self
                    .realm
                    .call_map_function("timers", id, None);
                if !repeats {
                    self.realm.remove_map_function("timers", id);
                }
                self.state.borrow_mut().current_nesting = 0;
                match result {
                    Ok(_) => Ok(()),
                    Err(e) => self.record_or_propagate(e),
                }
            }
            TaskKind::AnimationFrame { id, timestamp_ms } => {
                let result = self
                    .realm
                    .call_map_function("raf", id, Some(timestamp_ms));
                self.realm.remove_map_function("raf", id);
                match result {
                    Ok(_) => Ok(()),
                    Err(e) => self.record_or_propagate(e),
                }
            }
            TaskKind::Host(task) => {
                task(self);
                Ok(())
            }
        }
    }

    /// Record a script error and continue, or propagate a
    /// realm-killing error (watchdog / poisoned). A page `window.onerror`
    /// handler runs first: if it returns `true`, the default report
    /// (recording in `script_errors`) is suppressed, per HTML.
    fn record_or_propagate(&mut self, err: JsError) -> Result<(), JsError> {
        self.record_or_propagate_with_source(err, None)
    }

    /// Like [`Self::record_or_propagate`], but the caller knows the
    /// script's URL (the `Script` task's `url`, the module's URL):
    /// when the exception's stack names only the engine's generic
    /// filename (`eval_script` — the realm evaluates classic
    /// scripts without a filename), the handler is told the real
    /// script URL instead. `None` keeps whatever the stack says
    /// (or `undefined` when the stack names nothing real).
    fn record_or_propagate_with_source(
        &mut self,
        err: JsError,
        source_url: Option<&str>,
    ) -> Result<(), JsError> {
        if err.is_script_error() {
            let full = err.to_string();
            let (message, source, lineno, colno) =
                parse_error_location(&full, source_url);
            let suppressed = {
                let bindings = Rc::clone(&self.bindings);
                call_onerror(
                    &mut self.realm,
                    &bindings,
                    &message,
                    source.as_deref(),
                    lineno,
                    colno,
                )
            };
            if !suppressed {
                self.state.borrow_mut().script_errors.push(full.clone());
            }
            if let Some(host) = self.bindings.borrow().host.clone() {
                host.borrow().report_error(&full);
            }
            Ok(())
        } else {
            Err(err)
        }
    }

    /// Microtask checkpoint: drain host microtasks and engine promise    /// jobs until both are empty, including microtasks queued by
    /// microtasks. A throwing microtask is recorded, not fatal — the
    /// checkpoint continues (a throwing microtask must not stop later
    /// microtasks from running). Bounded so a microtask loop that keeps
    /// queueing starves the loop loudly instead of forever.
    fn microtask_checkpoint(&mut self) -> Result<(), JsError> {
        let mut drained = 0usize;
        loop {
            // MutationObserver delivery (Stage 8 Rec 7): one
            // compound callback per observer per checkpoint, in
            // pending order. This runs at the start of the
            // checkpoint, before the engine's promise jobs — a
            // deliberate approximation of the spec's "one compound
            // microtask per observer": the record queue is drained
            // here, so a callback that mutates the DOM queues fresh
            // records that the next checkpoint iteration delivers
            // (and the 100_000 bound below counts each delivery
            // round, so an observer loop that keeps mutating
            // starves loudly instead of forever). A throwing
            // callback is recorded, never fatal.
            let had_pending = self
                .bindings
                .borrow()
                .host
                .as_ref()
                .is_some_and(|h| !h.borrow().pending_mutation_observers().is_empty());
            if had_pending {
                drained += 1;
                let bindings = Rc::clone(&self.bindings);
                if let Err(e) = crate::bindings::observers::deliver_mutation_observers(
                    &mut self.realm,
                    &bindings,
                ) {
                    self.state.borrow_mut().script_errors.push(e.to_string());
                }
            }
            let microtask = self.state.borrow_mut().microtasks.pop_front();
            match microtask {
                Some(Microtask::Host(task)) => {
                    task(self);
                    drained += 1;
                }
                None => {
                    if self.realm.drain_one_job() {
                        drained += 1;
                    } else {
                        return Ok(());
                    }
                }
            }
            if drained > 100_000 {
                return Err(JsError::Internal(
                    "microtask checkpoint exceeded 100000 microtasks (infinite microtask loop?)"
                        .to_string(),
                ));
            }
        }
    }

    /// Update the rendering, but only at a rendering opportunity: a
    /// frame is due AND (an rAF callback is pending OR a render was
    /// requested). A hidden tab gets no rendering opportunity at all
    /// (Stage 8 Rec 1): this returns early with the rAF queue and
    /// `render_requested` untouched, so the callbacks fire on the
    /// first frame after the tab becomes visible again. Runs each rAF
    /// callback with the shared frame timestamp, with a microtask
    /// checkpoint after each callback (per spec, each callback is
    /// invoked from an empty stack).
    fn maybe_render(&mut self) -> Result<(), JsError> {
        if !self.state.borrow().visible {
            return Ok(());
        }
        let now = Instant::now();
        let (due, wanted) = {
            let state = self.state.borrow();
            (
                now >= state.next_frame_at,
                !state.raf.is_empty() || state.render_requested,
            )
        };
        if !due || !wanted {
            return Ok(());
        }
        let timestamp_ms = self.state.borrow().now_ms();
        let frame = self.state.borrow_mut().take_raf_frame();
        self.state.borrow_mut().next_frame_at = now + FRAME_INTERVAL;
        self.state.borrow_mut().render_requested = false;
        for entry in frame {
            let result = self.run_task(crate::event_loop::Task::new(
                TaskSource::Rendering,
                entry.epoch,
                TaskKind::AnimationFrame {
                    id: entry.id,
                    timestamp_ms,
                },
            ));
            // A throwing rAF callback is recorded by run_task and the
            // rest still run; only a realm-killing error aborts the
            // frame (the `?` below).
            result?;
            self.microtask_checkpoint()?;
        }
        Ok(())
    }

    /// Move unhandled-rejection reports into `script_errors`.
    fn drain_rejections(&self) {
        let reports: Vec<String> = self.rejection_rx.try_iter().collect();
        if !reports.is_empty() {
            self.state.borrow_mut().script_errors.extend(reports);
        }
    }

    /// Install the loop's host globals into a fresh realm. See the
    /// module docs for what each global is and why it is shaped this way.
    fn install_globals(
        realm: &mut TabRealm,
        state: &Rc<RefCell<LoopState>>,
    ) -> Result<(), JsError> {
        let state = Rc::clone(state);
        realm
            .with_ctx(|ctx| install_globals_in(ctx, &state))
            .map_err(JsError::Internal)
    }
}

/// Split a `JsError` string into the `window.onerror` arguments
/// (Stage 12 Rec 9): `(message, source, lineno, colno)`.
///
/// The realm stringifies a caught exception as
/// `"JS exception: Error: boom\n    at f (file:line:col)\n …"`
/// (`rquickjs::Exception`'s `Display` appends the stack), so the
/// throw site is recoverable here without touching the realm: the
/// message is the first line, the location is the first stack
/// frame. A stack frame whose file is the engine's generic
/// filename (`eval_script`, `<eval>`, …) carries no real source:
/// the caller's `source_url` (the script / module URL) replaces
/// it, or the source is `None` (→ `undefined`) when the caller
/// knows none either. Errors with no stack (interrupted, OOM)
/// yield the message alone.
fn parse_error_location(
    text: &str,
    source_url: Option<&str>,
) -> (String, Option<String>, Option<u32>, Option<u32>) {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or_default();
    let message = first
        .strip_prefix("JS exception: ")
        .unwrap_or(first)
        .to_string();
    for line in lines {
        let trimmed = line.trim();
        let Some(frame) = trimmed.strip_prefix("at ") else {
            continue;
        };
        // `name (file:line:col)` or `file:line:col`.
        let loc = if let Some(open) = frame.rfind('(')
            && let Some(close) = frame.rfind(')')
            && close > open
        {
            &frame[open + 1..close]
        } else {
            frame
        };
        let Some((file, line_no, col_no)) = parse_frame_location(loc) else {
            continue;
        };
        let generic = matches!(file.as_str(), "eval_script" | "<eval>" | "eval" | "<anonymous>");
        let source = if generic {
            source_url.map(|s| s.to_string())
        } else {
            Some(file)
        };
        return (message, source, line_no, col_no);
    }
    (message, None, None, None)
}

/// Parse `file:line:col` (or `file:line`) from one stack frame,
/// splitting from the right so `:` inside the file (a URL scheme,
/// a Windows drive) survives.
fn parse_frame_location(loc: &str) -> Option<(String, Option<u32>, Option<u32>)> {
    let loc = loc.trim();
    let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
    match parts.len() {
        3 => {
            let col: u32 = parts[0].parse().ok()?;
            let line: u32 = parts[1].parse().ok()?;
            Some((parts[2].to_string(), Some(line), Some(col)))
        }
        2 => {
            let line: u32 = parts[0].parse().ok()?;
            Some((parts[1].to_string(), Some(line), None))
        }
        _ => None,
    }
}


/// Install the host globals into `ctx`. Runs once per realm, on the
/// owner thread, while the runtime lock is held. The closures capture
/// `Rc` clones of the loop state; they touch only that state — never
/// the realm — so they cannot deadlock against the `with` lock.
fn install_globals_in<'js>(
    ctx: Ctx<'js>,
    state: &Rc<RefCell<LoopState>>,
) -> Result<(), String> {
    let globals = ctx.globals();

    // Callback registries live in the symbol-keyed registry object
    // (`Symbol.for("pluto.registry")`, created lazily by
    // `bindings::registry`): the `timers` / `raf` Maps are reachable
    // only through the symbol, never by name (wave 2a fixed the
    // wave-1 gap). The Rust side never stores a callback value itself.
    // queueMicrotask via the engine's own job queue (see module docs).
    ctx.eval::<(), _>(
        "globalThis.queueMicrotask = function (cb) { \
           if (typeof cb !== 'function') \
             throw new TypeError('queueMicrotask: argument is not a function'); \
           Promise.resolve().then(cb); \
         };",
    )
    .map_err(|e| format!("install queueMicrotask: {e}"))?;

    // setTimeout / setInterval share one timer pool and one id space.
    let state_timeout = Rc::clone(state);
    let set_timeout = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>, delay: Opt<f64>| -> rquickjs::Result<f64> {
            let id = {
                let nesting = state_timeout.borrow().current_nesting.saturating_add(1);
                state_timeout
                    .borrow_mut()
                    .schedule_timer(duration_from_ms(delay.0.unwrap_or(0.0)), None, nesting)
            };
            store_callback(&ctx, "timers", id, cb).inspect_err(|_| {
                state_timeout.borrow_mut().cancel_timer(id);
            })?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install setTimeout: {e}"))?;
    globals
        .set("setTimeout", set_timeout)
        .map_err(|e| format!("install setTimeout: {e}"))?;

    let state_interval = Rc::clone(state);
    let set_interval = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>, delay: Opt<f64>| -> rquickjs::Result<f64> {
            let delay = duration_from_ms(delay.0.unwrap_or(0.0));
            let id = {
                let nesting = state_interval.borrow().current_nesting.saturating_add(1);
                state_interval
                    .borrow_mut()
                    .schedule_timer(delay, Some(delay), nesting)
            };
            store_callback(&ctx, "timers", id, cb).inspect_err(|_| {
                state_interval.borrow_mut().cancel_timer(id);
            })?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install setInterval: {e}"))?;
    globals
        .set("setInterval", set_interval)
        .map_err(|e| format!("install setInterval: {e}"))?;

    let state_clear = Rc::clone(state);
    let clear_timeout = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'_>, id: f64| -> rquickjs::Result<()> {
            state_clear.borrow_mut().cancel_timer(id as u64);
            remove_callback(&ctx, "timers", id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install clearTimeout: {e}"))?;
    globals
        .set("clearTimeout", clear_timeout)
        .map_err(|e| format!("install clearTimeout: {e}"))?;
    // clearInterval cancels from the same pool (spec: one timer pool).
    let state_clear_interval = Rc::clone(state);
    let clear_interval = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'_>, id: f64| -> rquickjs::Result<()> {
            state_clear_interval.borrow_mut().cancel_timer(id as u64);
            remove_callback(&ctx, "timers", id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install clearInterval: {e}"))?;
    globals
        .set("clearInterval", clear_interval)
        .map_err(|e| format!("install clearInterval: {e}"))?;

    // requestAnimationFrame / cancelAnimationFrame.
    let state_raf = Rc::clone(state);
    let request_raf = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>| -> rquickjs::Result<f64> {
            let id = state_raf.borrow_mut().register_raf();
            store_callback(&ctx, "raf", id, cb).inspect_err(|_| {
                state_raf.borrow_mut().cancel_raf(id);
            })?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install requestAnimationFrame: {e}"))?;
    globals
        .set("requestAnimationFrame", request_raf)
        .map_err(|e| format!("install requestAnimationFrame: {e}"))?;

    let state_cancel_raf = Rc::clone(state);
    let cancel_raf = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'_>, id: f64| -> rquickjs::Result<()> {
            state_cancel_raf.borrow_mut().cancel_raf(id as u64);
            remove_callback(&ctx, "raf", id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install cancelAnimationFrame: {e}"))?;
    globals
        .set("cancelAnimationFrame", cancel_raf)
        .map_err(|e| format!("install cancelAnimationFrame: {e}"))?;

    // performance.now(): ms since this realm was created.
    let state_now = Rc::clone(state);
    let perf = Object::new(ctx.clone()).map_err(|e| format!("install performance: {e}"))?;
    let now_fn = Function::new(ctx.clone(), move || -> rquickjs::Result<f64> {
        Ok(state_now.borrow().now_ms())
    })
    .map_err(|e| format!("install performance.now: {e}"))?;
    perf.set("now", now_fn)
        .map_err(|e| format!("install performance.now: {e}"))?;
    globals
        .set("performance", perf)
        .map_err(|e| format!("install performance: {e}"))?;

    Ok(())
}

/// Store a JS callback in one of the realm's callback Maps, by numeric id.
fn store_callback<'js>(
    ctx: &Ctx<'js>,
    map_name: &str,
    id: u64,
    cb: Function<'js>,
) -> rquickjs::Result<()> {
    let map: Object = registry_map(ctx, map_name)?;
    let set: Function = map.get("set")?;
    // Map methods are extracted then called, so `this` must be passed
    // explicitly — an unbound `Map.prototype.set` throws "Map object
    // expected".
    let _: rquickjs::Value = set.call((This(map.clone()), id as f64, cb))?;
    Ok(())
}

/// Remove a callback from one of the realm's callback Maps. Best effort:
/// a missing map or entry is not an error.
fn remove_callback(ctx: &Ctx<'_>, map_name: &str, id: u64) {
    if let Ok(map) = registry_map(ctx, map_name)
        && let Ok(delete) = map.get::<_, Function>("delete")
    {
        // Unbound Map method: pass the map as `this`.
        let _: rquickjs::Result<rquickjs::Value> =
            delete.call((This(map.clone()), id as f64));
    }
}

/// A `setTimeout` / `setInterval` delay in ms → `Duration`. Negative,
/// NaN, and infinite delays become 0 (per the HTML timer steps, a
/// timeout less than 0 is set to 0).
fn duration_from_ms(ms: f64) -> Duration {
    if ms.is_finite() && ms > 0.0 {
        Duration::from_secs_f64(ms / 1000.0)
    } else {
        Duration::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    fn test_tab() -> TabJs {
        TabJs::new(&RealmConfig::default()).unwrap()
    }

    /// Read a JS string back out of the tab.
    fn read_str(tab: &mut TabJs, source: &str) -> String {
        match tab.eval_value(source).unwrap() {
            JsValue::String(s) => s,
            other => panic!("expected a string from `{source}`, got {other:?}"),
        }
    }

    fn read_num(tab: &mut TabJs, source: &str) -> f64 {
        match tab.eval_value(source).unwrap() {
            JsValue::Number(n) => n,
            other => panic!("expected a number from `{source}`, got {other:?}"),
        }
    }

    #[test]
    fn ordering_timeout_vs_promise_vs_microtask_vs_raf() {
        let mut tab = test_tab();
        tab.eval_script(
            "const order = []; \
             setTimeout(() => order.push('timeout'), 0); \
             Promise.resolve().then(() => order.push('promise')); \
             queueMicrotask(() => order.push('microtask')); \
             requestAnimationFrame(() => order.push('raf')); \
             order.push('sync');",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(
            read_str(&mut tab, "order.join(',')"),
            "sync,promise,microtask,raf,timeout"
        );
    }

    #[test]
    fn microtask_queueing_microtask_drains_in_order() {
        let mut tab = test_tab();
        tab.eval_script(
            "const order = []; \
             queueMicrotask(() => { \
               order.push('m1'); \
               queueMicrotask(() => order.push('m2')); \
             }); \
             Promise.resolve().then(() => order.push('p1'));",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        // m1 runs first, queues m2 behind the already-queued p1.
        assert_eq!(read_str(&mut tab, "order.join(',')"), "m1,p1,m2");
    }

    #[test]
    fn timer_nesting_clamp() {
        let mut tab = test_tab();
        tab.eval_script(
            "const times = []; \
             function tick(n) { \
               times.push(performance.now()); \
               if (n < 6) setTimeout(() => tick(n + 1), 0); \
             } \
             setTimeout(() => tick(0), 0);",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "times.length"), 7.0);
        let times: Vec<f64> = read_str(&mut tab, "times.join(',')")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        // The first five hops are unclamped (each well under the clamp).
        assert!(
            times[4] - times[0] < 20.0,
            "unclamped hops took too long: {times:?}"
        );
        // The 6th and 7th hops nest past the 5-level clamp: >= 4 ms each.
        assert!(
            times[5] - times[4] >= 3.0,
            "6th nested timer was not clamped: {times:?}"
        );
        assert!(
            times[6] - times[5] >= 3.0,
            "7th nested timer was not clamped: {times:?}"
        );
    }

    #[test]
    fn interval_repeats_and_clears() {
        let mut tab = test_tab();
        tab.eval_script(
            "let n = 0; \
             const id = setInterval(() => { n++; if (n >= 3) clearInterval(id); }, 0);",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "n"), 3.0);
    }

    #[test]
    fn clear_timeout_cancels() {
        let mut tab = test_tab();
        tab.eval_script(
            "let n = 0; \
             const id = setTimeout(() => { n = 1; }, 0); \
             clearTimeout(id);",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "n"), 0.0);
    }

    #[test]
    fn throwing_timer_does_not_kill_the_loop() {
        let mut tab = test_tab();
        tab.eval_script(
            "const order = []; \
             setTimeout(() => { throw new Error('timer-boom'); }, 0); \
             setTimeout(() => order.push('second'), 0);",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_str(&mut tab, "order.join(',')"), "second");
        assert!(
            tab.script_errors()
                .iter()
                .any(|e| e.contains("timer-boom")),
            "expected timer-boom in errors: {:?}",
            tab.script_errors()
        );
    }

    #[test]
    fn throwing_script_does_not_kill_later_scripts() {
        let mut tab = test_tab();
        tab.eval_script("throw new Error('script-boom');", "a.js")
            .unwrap();
        tab.eval_script("globalThis.__x = 41 + 1;", "b.js").unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__x"), 42.0);
        assert!(
            tab.script_errors()
                .iter()
                .any(|e| e.contains("script-boom")),
            "expected script-boom in errors: {:?}",
            tab.script_errors()
        );
    }

    #[test]
    fn stale_epoch_tasks_are_dropped_at_dispatch() {
        static RAN: AtomicBool = AtomicBool::new(false);
        let mut tab = test_tab();
        tab.enqueue_host_task(TaskSource::Script, Box::new(|_| {
            RAN.store(true, AtomicOrdering::SeqCst);
        }));
        // The epoch moves without a navigation clear: the dispatch-time
        // check must drop the task.
        tab.state.borrow_mut().epoch += 1;
        tab.run_until_idle().unwrap();
        assert!(!RAN.load(AtomicOrdering::SeqCst));
    }

    #[test]
    fn hidden_tab_runs_no_raf_but_timers_still_fire() {
        // Stage 8 Rec 1: hidden = zero rendering opportunities.
        // The rAF callback stays queued while hidden and fires on
        // the first visible frame; a timer fires while hidden
        // (timers are not rendering).
        let mut tab = test_tab();
        // Hide first: `eval_script` itself runs one turn, and that
        // turn must not fire the rAF callback either.
        tab.set_visible(false);
        assert!(!tab.is_visible());
        tab.eval_script(
            "globalThis.__raf = 0; globalThis.__timer = 0; \
             requestAnimationFrame(() => { globalThis.__raf = 1; }); \
             setTimeout(() => { globalThis.__timer = 1; }, 0);",
            "test.js",
        )
        .unwrap();
        for _ in 0..10 {
            tab.run_turn().unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(read_num(&mut tab, "__timer"), 1.0, "timers run while hidden");
        assert_eq!(read_num(&mut tab, "__raf"), 0.0, "no rAF while hidden");
        tab.set_visible(true);
        assert!(tab.is_visible());
        tab.run_turn().unwrap();
        assert_eq!(
            read_num(&mut tab, "__raf"),
            1.0,
            "queued rAF fires when visible again"
        );
    }

    #[test]
    fn raf_callbacks_share_one_timestamp() {
        let mut tab = test_tab();
        tab.eval_script(
            "const ts = []; \
             requestAnimationFrame((t) => { \
               ts.push(t); \
               requestAnimationFrame((t2) => ts.push(t2)); \
             }); \
             requestAnimationFrame((t) => ts.push(t));",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        let ts: Vec<f64> = read_str(&mut tab, "ts.join(',')")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(ts.len(), 3);
        assert_eq!(ts[0], ts[1], "one frame shares one timestamp: {ts:?}");
        assert!(ts[2] > ts[0], "next frame is later: {ts:?}");
    }

    #[test]
    fn navigation_resets_the_realm_and_bumps_the_epoch() {
        let mut tab = test_tab();
        let epoch_before = tab.epoch();
        tab.eval_script("globalThis.__x = 41;", "a.js").unwrap();
        assert_eq!(read_num(&mut tab, "__x"), 41.0);
        tab.navigate().unwrap();
        assert_eq!(tab.epoch(), epoch_before + 1);
        assert_eq!(read_str(&mut tab, "typeof __x"), "undefined");
        // The new realm works.
        tab.eval_script("globalThis.__y = 2;", "b.js").unwrap();
        assert_eq!(read_num(&mut tab, "__y"), 2.0);
    }

    #[test]
    fn module_cache_hit_on_second_load() {
        let mut tab = test_tab();
        let source = "globalThis.__count = (globalThis.__count || 0) + 1; export const x = 1;";
        let (_, hit1) = tab
            .load_module_cached("https://example.com/mod.mjs", source)
            .unwrap();
        assert_eq!(hit1, CacheHit::Miss);
        let (_, hit2) = tab
            .load_module_cached("https://example.com/mod.mjs", source)
            .unwrap();
        assert_eq!(hit2, CacheHit::Hit);
        // The bytecode was reused (compiled once), but the module still
        // executes on each load — the cache saves compilation, not
        // execution.
        assert_eq!(read_num(&mut tab, "__count"), 2.0);
        let stats = tab.bytecode_cache().stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
    }

    /// Wave 2b (Rec 9 remainder): a module whose top-level await
    /// waits on a host event (here a timer) must NOT poison the
    /// realm — the loop drives it across turns. The wave-1
    /// `await_promise` path WouldBlocks on this immediately.
    #[test]
    fn module_top_level_await_on_timer_is_driven_by_the_loop() {
        let mut tab = test_tab();
        tab.start_module_async(
            "https://example.com/mod.mjs",
            "await new Promise((r) => setTimeout(r, 10)); \
             globalThis.__mod_done = 42;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__mod_done"), 42.0);
        assert!(!tab.realm.is_poisoned(), "realm must not be poisoned");
        assert!(tab.script_errors().is_empty(), "{:?}", tab.script_errors());
    }

    /// A module that rejects after awaiting a host event is
    /// recorded as a script error; the loop and later scripts live.
    #[test]
    fn module_rejection_after_host_await_is_recorded_not_fatal() {
        let mut tab = test_tab();
        tab.start_module_async(
            "https://example.com/bad.mjs",
            "await new Promise((r) => setTimeout(r, 0)); \
             throw new Error('module-boom');",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert!(
            tab.script_errors()
                .iter()
                .any(|e| e.contains("module-boom")),
            "expected module-boom in errors: {:?}",
            tab.script_errors()
        );
        tab.eval_script("globalThis.__after = 7;", "after.js").unwrap();
        assert_eq!(read_num(&mut tab, "__after"), 7.0);
    }

    /// A module that throws synchronously is recorded at the
    /// per-script boundary and does not kill later scripts.
    #[test]
    fn module_sync_throw_does_not_kill_later_scripts() {
        let mut tab = test_tab();
        tab.start_module_async("https://example.com/sync.mjs", "throw new Error('sync-boom');")
            .unwrap();
        tab.run_until_idle().unwrap();
        assert!(
            tab.script_errors()
                .iter()
                .any(|e| e.contains("sync-boom")),
            "expected sync-boom in errors: {:?}",
            tab.script_errors()
        );
        tab.eval_script("globalThis.__after = 8;", "after.js").unwrap();
        assert_eq!(read_num(&mut tab, "__after"), 8.0);
    }

    /// Stage 12 Rec 2: fixed JS workload for the `-Os` before/after
    /// measurement. Pure CPU (arithmetic loop, string building,
    /// numeric sort) — no host calls, so the time is engine time.
    /// The result is deterministic: three runs must agree exactly,
    /// which also proves the workload was not optimised away.
    /// The timings print to stderr (visible with `--nocapture`);
    /// the test itself only asserts correctness + a generous
    /// completion bound, so it stays green on slow CI runners.
    #[test]
    fn js_workload_benchmark_reports_timing() {
        const WORKLOAD: &str = r#"
            let acc = 0;
            for (let i = 0; i < 2000000; i++) {
                acc = (acc * 31 + i) % 1000003;
                acc ^= (acc >>> 7);
            }
            let s = "";
            for (let i = 0; i < 5000; i++) { s += "x" + (i % 10); }
            const a = [];
            for (let i = 0; i < 50000; i++) a.push((i * 2654435761) % 100000);
            a.sort((x, y) => x - y);
            globalThis.__bench = acc + s.length + a[0] + a[a.length - 1];
        "#;
        let mut results = Vec::new();
        let mut times_ms = Vec::new();
        for _ in 0..3 {
            let mut tab = test_tab();
            let start = std::time::Instant::now();
            tab.eval_script(WORKLOAD, "bench.js").unwrap();
            tab.run_until_idle().unwrap();
            times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            results.push(read_num(&mut tab, "__bench"));
            assert!(
                tab.script_errors().is_empty(),
                "workload errored: {:?}",
                tab.script_errors()
            );
        }
        assert_eq!(results[0], results[1], "workload must be deterministic");
        assert_eq!(results[0], results[2], "workload must be deterministic");
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "[pluto js-bench] runs: {:.1}ms / {:.1}ms / {:.1}ms (median {:.1}ms) result={}",
            times_ms[0], times_ms[1], times_ms[2], times_ms[1], results[0]
        );
    }

    /// Stage 12 Rec 1: a module importing a relative specifier
    /// resolves it against its own URL and runs the pre-fetched
    /// import's code (the browser fills `module_sources`; the test
    /// stands in for it).
    #[test]
    fn module_import_resolves_from_prefetched_sources() {
        let mut tab = test_tab();
        tab.module_sources().insert(
            "https://example.com/util.mjs",
            "export const answer = 41;".to_string(),
        );
        tab.start_module_async(
            "https://example.com/main.mjs",
            "import { answer } from './util.mjs'; globalThis.__got = answer + 1;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__got"), 42.0);
        assert!(tab.script_errors().is_empty(), "{:?}", tab.script_errors());
    }

    /// Stage 12 Rec 1: an import that was never pre-fetched makes
    /// the module reject cleanly — recorded as a script error, no
    /// realm poison, later scripts still run.
    #[test]
    fn module_missing_import_rejects_cleanly() {
        let mut tab = test_tab();
        tab.start_module_async(
            "https://example.com/main.mjs",
            "import './gone.mjs'; globalThis.__got = 1;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert!(
            tab.script_errors().iter().any(|e| e.contains("gone.mjs")),
            "expected gone.mjs in errors: {:?}",
            tab.script_errors()
        );
        assert!(!tab.realm.is_poisoned(), "realm must not be poisoned");
        tab.eval_script("globalThis.__after = 9;", "after.js").unwrap();
        assert_eq!(read_num(&mut tab, "__after"), 9.0);
    }

    /// Stage 12 Rec 1: a string-literal dynamic `import()` goes
    /// through the same loader and works when pre-fetched.
    #[test]
    fn module_dynamic_import_literal_resolves() {
        let mut tab = test_tab();
        tab.module_sources().insert(
            "https://example.com/chunk.mjs",
            "export const v = 5;".to_string(),
        );
        tab.start_module_async(
            "https://example.com/main.mjs",
            "const m = await import('./chunk.mjs'); globalThis.__got = m.v;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__got"), 5.0);
        assert!(tab.script_errors().is_empty(), "{:?}", tab.script_errors());
    }

    /// Stage 12 Rec 1: a bare specifier rejects with the import-map
    /// note (no import maps in M4), not a crash or a hang.
    #[test]
    fn module_bare_specifier_rejects_with_import_map_note() {
        let mut tab = test_tab();
        tab.start_module_async(
            "https://example.com/main.mjs",
            "import 'lodash'; globalThis.__got = 1;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert!(
            tab.script_errors()
                .iter()
                .any(|e| e.contains("import map")),
            "expected import-map note in errors: {:?}",
            tab.script_errors()
        );
        assert!(!tab.realm.is_poisoned(), "realm must not be poisoned");
    }

    /// Stage 12 Rec 1: transitive imports resolve — the entry
    /// imports A, A imports B, both pre-fetched.
    #[test]
    fn module_transitive_import_resolves() {
        let mut tab = test_tab();
        let sources = tab.module_sources();
        sources.insert(
            "https://example.com/a.mjs",
            "import { b } from './sub/b.mjs'; export const a = b + 1;".to_string(),
        );
        sources.insert(
            "https://example.com/sub/b.mjs",
            "export const b = 10;".to_string(),
        );
        tab.start_module_async(
            "https://example.com/main.mjs",
            "import { a } from './a.mjs'; globalThis.__got = a;",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__got"), 11.0);
        assert!(tab.script_errors().is_empty(), "{:?}", tab.script_errors());
    }

    /// Stage 12 Rec 9: a throwing inline script calls `onerror`
    /// with the message, the script URL, and the throw site's line
    /// and column (parsed out of the exception's stack — see
    /// `parse_error_location`).
    #[test]
    fn onerror_receives_source_line_and_column() {
        let mut tab = test_tab();
        tab.eval_script(
            "const seen = []; \
             window.onerror = (msg, src, line, col, err) => { \
               seen.push([msg, src, line, col, err]); return true; \
             }; \
             globalThis.__seen = seen;",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        // The throw sits on line 2, column 11 (`Error` starts at
        // column 11 of `throw new Error('loc-boom');`).
        tab.eval_script("var __x = 1;\nthrow new Error('loc-boom');", "s.js")
            .unwrap();
        tab.run_until_idle().unwrap();
        assert!(
            tab.script_errors().is_empty(),
            "suppressed error must not be recorded: {:?}",
            tab.script_errors()
        );
        assert_eq!(read_num(&mut tab, "__seen.length"), 1.0);
        assert_eq!(
            read_str(&mut tab, "__seen[0].join('|')"),
            "Error: loc-boom|s.js|2|11|"
        );
    }

    /// Stage 12 Rec 9: without a location the handler still runs
    /// with `undefined` source/line/col (an interrupted script has
    /// no throw site to report).
    #[test]
    fn onerror_without_location_gets_undefineds() {
        let mut tab = test_tab();
        tab.eval_script(
            "const seen = []; \
             window.onerror = (msg, src, line, col) => { \
               seen.push([msg, src, line, col]); return true; \
             }; \
             globalThis.__seen = seen;",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        // A throwing *timer* callback has a stack, but no script
        // URL is known for it — the source is the engine's generic
        // filename, which the loop maps to `undefined`.
        tab.eval_script(
            "setTimeout(() => { throw new Error('timer-boom'); }, 0);",
            "test.js",
        )
        .unwrap();
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__seen.length"), 1.0);
        // Message + `undefined` source (the timer callback has a
        // stack, so line/col are present, but no script URL).
        assert_eq!(
            read_str(&mut tab, "__seen[0].slice(0, 2).join('|')"),
            "Error: timer-boom|"
        );
        assert_eq!(read_num(&mut tab, "__seen[0][2]"), 1.0);
        assert!(read_num(&mut tab, "__seen[0][3]") > 0.0);
    }
}
