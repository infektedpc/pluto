//! Scripting (layer 6): the JS engine host for one tab.
//!
//! `pluto-js` owns the JS engine (rquickjs / QuickJS-NG), the
//! host-owned event loop, and the binding boundary — but no DOM code:
//! DOM access crosses through [`DomHost`] handles, and layout is
//! reached through that trait, never by depending on `pluto-layout`
//! directly.
//!
//! # The pieces (Stage 7, wave 1)
//!
//! * [`TabRealm`] — one rquickjs `AsyncRuntime` + `AsyncContext` per
//!   tab, created on navigation on the worker thread, dropped on
//!   navigation. Per-runtime memory limit + wall-clock interrupt
//!   handler, plus a watchdog for scripts hung across `await`s.
//! * [`TabJs`] — the event loop wired to a realm: per-source task
//!   queues, one microtask queue, the timer wheel (5-level nesting /
//!   4 ms clamp), rAF at rendering opportunities, the navigation
//!   epoch. Host globals (`setTimeout`, `queueMicrotask`,
//!   `requestAnimationFrame`, `performance.now`) are installed here.
//! * [`JsEngine`] / [`JsValue`] — the engine trait boundary. No
//!   QuickJS type appears in this crate's public API; an engine swap
//!   changes [`realm`] and nothing else.
//! * [`DomHost`] / [`DomKey`] — what the engine needs from the host
//!   document, for wave 2 to implement against the slotmap DOM.
//! * [`bindings`] — the hand-written DOM / WebIDL bindings (wave 2a):
//!   `window`, `document`, `Node` / `Element`, `EventTarget`,
//!   `fetch` / `Response` / `Headers`, plus the [`FetchBackend`]
//!   trait the integration wave implements with `pluto-net`.
//! * [`BytecodeCache`] — in-memory compiled-module cache (URL +
//!   SHA-256 of source, bounded LRU).
//! * [`probe::probe_engine`] — the feature checklist + Test262-style
//!   subset that gates engine upgrades. Run it with
//!   `cargo test -p pluto-js probe -- --nocapture`.
//!
//! See `crates/pluto-js/README.md` for the pinned engine version, the
//! probe verdict, the public API surface, and the performance notes.

pub mod bytecode;
pub mod bindings;
pub mod observers;
pub mod engine;
pub mod error;
pub mod event_loop;
pub mod host;
pub mod probe;
pub mod realm;
pub mod tab;

pub use bindings::{FetchBackend, FetchCompletion, FetchError, FetchRequest, FetchResponse};
pub use bytecode::{BytecodeCache, CacheHit, CacheStats, CacheKey};
pub use engine::{EvalKind, JsEngine, JsValue};
pub use error::JsError;
pub use event_loop::{FRAME_INTERVAL, TaskSource};
pub use observers::{
    ComputedIntersection, IntersectionObserverEntry, IntersectionObserverRegistry, MarginValue,
    RootMargin, MAX_ROUNDS, ObservedSizes, ResizeBoxKind, ResizeObserverEntry,
    ResizeObserverRegistry, ResizeStepOutcome,
};
pub use host::{DomHost, DomHostError, DomKey, Geometry};
pub use realm::{DEFAULT_MEMORY_LIMIT_BYTES, DEFAULT_SCRIPT_BUDGET, RealmConfig, TabRealm};
pub use tab::{TabJs, TurnOutcome};
