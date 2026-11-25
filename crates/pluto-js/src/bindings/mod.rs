//! DOM / WebIDL bindings (Stage 7, wave 2a — Recommendation 5, §(f)
//! items 1–5, in that order).
//!
//! Hand-written bindings over the wave-1 engine boundary. No QuickJS
//! type appears in any public signature of `pluto-js`; everything in
//! this module is `pub(crate)` except the [`FetchBackend`] trait and
//! the [`FetchRequest`] / [`FetchResponse`] / [`FetchError`] types,
//! which the integration wave needs to implement the backend.
//!
//! # Safety model (report §(c), non-negotiable)
//!
//! * A JS wrapper holds only a small integer id. The id maps (in
//!   [`BindingState`], plain Rust data) to a generational [`DomKey`];
//!   the key is validated against the live `Dom` on every crossing.
//!   A stale key surfaces as `null` for getters and a no-op /
//!   `TypeError` for setters — never use-after-free.
//! * Wrapper identity is a per-realm cache: the registry's `wrappers`
//!   `Map` (id → wrapper object), populated lazily, so
//!   `document.body === document.body`. Stale entries are evicted
//!   lazily on access.
//! * No Rust code in this module stores a JS value (`Persistent`,
//!   `Function`, `Object`) past one `with_ctx` entry. Listener
//!   functions, pending-fetch resolve/reject functions, and wrappers
//!   all live in JS-side `Map`s inside one registry object, reachable
//!   only through a symbol key (`Symbol.for("pluto.registry")`).
//!   Rust holds ids and keys only, so there is no Persistent
//!   drop-order hazard on navigation (wave 1's rule, kept).
//! * `this` is validated on every method; a wrong `this` throws
//!   `TypeError`. Argument conversion follows WebIDL where stated;
//!   departures are flagged `M4:` in code comments.
//! * Rust errors surface as JS exceptions with the right name
//!   (`TypeError`, `HierarchyRequestError`, `NotFoundError`,
//!   `InvalidStateError`, `SyntaxError` where noted).
//!
//! # Registry (symbol-keyed — wave-1 gap (a), fixed here)
//!
//! Wave 1 left the `__pluto_timers` / `__pluto_raf` Maps reachable by
//! name. They now live in the registry object, keyed by
//! `Symbol.for("pluto.registry")` on `globalThis` (non-enumerable via
//! string keys; a page that guesses the `Symbol.for` key can still
//! reach it — the symbol hides the registries from enumeration and
//! from name lookup, it is not a capability boundary). The same
//! registry holds `wrappers`, `listeners`, `fetches`, `responses`,
//! `headers`, and `prototypes`. `TabJs`'s timer/rAF code and this
//! module share [`registry`] / [`registry_map`].
//!
//! # What is NOT built here (Rec 11)
//!
//! Full WebIDL/codegen, workers, Shadow DOM, `IntersectionObserver`,
//! import maps, full fetch/CORS (CORS is M5 — see the `TODO(M5)` at
//! the fetch gate in `fetch.rs`), CSP (M10 — see the `TODO(M10)` at
//! the same gate). `MutationObserver` / `ResizeObserver` landed in
//! Stage 8 (see [`observers`]); storage (`localStorage` /
//! `sessionStorage`, Stage 9 Rec 4) and `navigator.permissions`
//! (Stage 9 Rec 11) landed in Stage 9 — see [`storage`] and
//! [`permissions`]. Each remaining item gets a one-line
//! `TODO(stage)` at its hook point.
//!
//! TODO(M5): WebIDL/codegen — the bindings in this module are
//! hand-written per interface; a generator would replace them when
//! the binding surface grows (forms, storage, workers).

pub mod dom;
pub mod events;
pub mod fetch;
pub mod observers;
pub mod permissions;
pub mod storage;
#[cfg(test)]
pub mod test_host;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rquickjs::function::This;
use rquickjs::{Atom, Ctx, FromJs, Function, Object, Symbol, Value};

use crate::bindings::permissions::{MemoryPermissionsBackend, PermissionName, PermissionsBackend};
use crate::bindings::storage::{MemoryStorageBackend, StorageBackend};
use crate::error::JsError;
use crate::event_loop::LoopState;
use crate::host::{DomHost, DomKey};
use crate::observers::{IntersectionObserverRegistry, ResizeObserverRegistry};
use crate::realm::TabRealm;

pub use fetch::{FetchBackend, FetchCompletion, FetchError, FetchRequest, FetchResponse, ResponseType};

/// `Symbol.for("pluto.registry")` — the registry key. Reconstructed
/// on every entry (a global symbol), so Rust never has to store it.
pub const REGISTRY_SYMBOL: &str = "pluto.registry";

/// `Symbol.for("pluto.domkey")` — the per-wrapper DomKey-id property.
pub const DOMKEY_SYMBOL: &str = "pluto.domkey";

/// `Symbol.for("pluto.response")` — the per-Response body id property.
pub const RESPONSE_SYMBOL: &str = "pluto.response";

/// `Symbol.for("pluto.headers")` — the per-Headers id property.
pub const HEADERS_SYMBOL: &str = "pluto.headers";

/// `Symbol.for("pluto.request")` — the per-Request id property.
pub const REQUEST_SYMBOL: &str = "pluto.request";

/// `Symbol.for("pluto.abortSignal")` — the per-AbortSignal id property.
pub const ABORT_SIGNAL_SYMBOL: &str = "pluto.abortSignal";

/// `Symbol.for("pluto.url")` — the per-URL / per-URLSearchParams id property.
pub const URL_SYMBOL: &str = "pluto.url";

/// `window`'s target id in the listener registry (the window has no
/// `DomKey`).
pub const WINDOW_TARGET_ID: u64 = u64::MAX;

/// Shared binding state. Plain Rust data only — no JS values — so it
/// can be dropped in any order relative to the realm. The JS values
/// (wrappers, listeners, pending fetches) live in the registry Maps.
pub struct BindingState {
    /// The host document, set by the per-tab worker via
    /// [`crate::TabJs::set_dom_host`]. `None` until then: bindings
    /// that need a document throw `InvalidStateError` (M4: the
    /// document is always attached before scripts run in the
    /// integration wave; tests attach a [`test_host::TestHost`]).
    pub host: Option<Rc<RefCell<Box<dyn DomHost>>>>,
    /// The fetch backend, set via [`crate::TabJs::set_fetch_backend`].
    /// `None` until then: `fetch()` rejects with a `TypeError`.
    pub fetch_backend: Option<std::sync::Arc<dyn FetchBackend>>,
    /// Captured `console` output as `(level, message)` pairs, for
    /// tests and for the host log hook. `console` also writes through
    /// `tracing` (see `dom.rs`).
    pub console_lines: Vec<(String, String)>,
    /// Script/callback errors recorded by bindings (listener throws,
    /// `onerror` reports). Drained into `TabJs::script_errors`.
    pub errors: Vec<String>,
    /// Next wrapper id (sequential; exact in an f64 JS Map key).
    pub next_wrapper_id: u64,
    /// Wrapper id → `DomKey` (the identity cache's Rust half).
    pub wrapper_keys: HashMap<u64, DomKey>,
    /// `DomKey` → wrapper id (reverse lookup).
    pub wrapper_ids: HashMap<DomKey, u64>,
    /// Next pending-fetch id.
    pub next_fetch_id: u64,
    /// Next response / headers object id.
    pub next_body_id: u64,
    /// In-flight fetch requests, by id (the backend outcome arrives
    /// later, on the Networking task queue).
    pub pending_fetches: HashMap<u64, FetchRequest>,
    /// Response bodies, by body id (bytes are Rust data, not JS).
    pub response_bodies: HashMap<u64, Vec<u8>>,
    /// Response headers, by body id.
    pub response_headers: HashMap<u64, Vec<(String, String)>>,
    /// Next `MutationObserver` id (sequential; exact in an f64 JS
    /// Map key). The callback + instance live in the registry's
    /// `mutationObservers` / `mutationObserverInstances` Maps.
    pub next_mutation_observer_id: u64,
    /// Next `ResizeObserver` id. The callback + instance live in
    /// the registry's `resizeObservers` /
    /// `resizeObserverInstances` Maps.
    pub next_resize_observer_id: u64,
    /// The `ResizeObserver` registry: every observation of every
    /// observer in this tab, plus their `lastReportedSizes`. The
    /// `TabJs` rendering step drives its gather/broadcast loop.
    pub resize_registry: ResizeObserverRegistry,
    /// Next `IntersectionObserver` id. The callback + instance live
    /// in the registry's `intersectionObservers` /
    /// `intersectionObserverInstances` Maps.
    pub next_intersection_observer_id: u64,
    /// The `IntersectionObserver` registry: every observation of
    /// every observer in this tab, plus their `lastReported`
    /// ratios and the pending entries awaiting the queued delivery
    /// task. The `TabJs` rendering step drives its gather; the
    /// delivery task drains the pending queue after paint.
    pub intersection_registry: IntersectionObserverRegistry,
    /// The storage backend behind `localStorage` /
    /// `sessionStorage` (Stage 9 Rec 4). Default: the in-memory
    /// [`MemoryStorageBackend`] (process-shared local buckets,
    /// per-realm session buckets). The integration wave replaces
    /// this with its profile-backed backends once `TabJs` gains a
    /// setter (see the wiring note in
    /// `pluto-browser::js_host`); `TabJs::navigate` re-attaches
    /// whatever backend is set there, like the fetch backend.
    pub storage_backend: std::sync::Arc<dyn StorageBackend>,
    /// The permission store behind `navigator.permissions`
    /// (Stage 9 Rec 11). Default: the in-memory
    /// [`MemoryPermissionsBackend`]. Replaced by the
    /// profile-persisted store the same way as
    /// [`BindingState::storage_backend`].
    pub permissions_backend: std::sync::Arc<dyn PermissionsBackend>,
    /// Next `PermissionStatus` id (sequential; exact in an f64 JS
    /// Map key). The status objects live in the registry's
    /// `permissionStatuses` Map.
    pub next_permission_status_id: u64,
    /// `PermissionStatus` id → (feature, sysex), so a state
    /// change can dispatch `onchange` to the right statuses.
    pub permission_status_features: HashMap<u64, (PermissionName, bool)>,
    /// `console.count` counters, by label (Stage 12 Rec 10).
    pub console_counts: HashMap<String, u64>,
    /// `console.time` timers, by label: when the timer started.
    pub console_timers: HashMap<String, std::time::Instant>,
    /// `console.group` nesting depth: captured lines are indented
    /// by two spaces per level.
    pub console_group_depth: usize,
    /// Next `Request` id.
    pub next_request_id: u64,
    /// `Request` data, by id (url / method / headers / body /
    /// signal). The JS object holds only the id.
    pub requests: HashMap<u64, fetch::StoredRequest>,
    /// Pending-fetch id → `AbortSignal` id that can cancel it.
    pub fetch_signals: HashMap<u64, u64>,
    /// `AbortSignal` state, by id (aborted flag + the fetch ids
    /// registered on it).
    pub abort_signals: HashMap<u64, fetch::AbortSignalState>,
    /// Next `AbortSignal` id.
    pub next_signal_id: u64,
    /// Next `URL` / `URLSearchParams` id.
    pub next_url_id: u64,
    /// `URL` / `URLSearchParams` data, by id.
    pub url_entries: HashMap<u64, dom::UrlEntry>,
    /// Response URL + `redirected`, by body id (set by
    /// `make_response`; read by the `url` / `redirected` accessors
    /// and by `Response.clone()`).
    pub response_urls: HashMap<u64, (String, bool)>,
    /// `Image()` / `<img>` load state, by node key: whether the
    /// current `src` has loaded, and its intrinsic size.
    pub image_states: HashMap<DomKey, dom::ImageLoadState>,
}

impl BindingState {
    pub fn new() -> Self {
        Self {
            host: None,
            fetch_backend: None,
            console_lines: Vec::new(),
            errors: Vec::new(),
            next_wrapper_id: 1,
            wrapper_keys: HashMap::new(),
            wrapper_ids: HashMap::new(),
            next_fetch_id: 1,
            next_body_id: 1,
            pending_fetches: HashMap::new(),
            response_bodies: HashMap::new(),
            response_headers: HashMap::new(),
            next_mutation_observer_id: 1,
            next_resize_observer_id: 1,
            resize_registry: ResizeObserverRegistry::default(),
            next_intersection_observer_id: 1,
            intersection_registry: IntersectionObserverRegistry::default(),
            storage_backend: std::sync::Arc::new(MemoryStorageBackend::default_for_realm()),
            permissions_backend: std::sync::Arc::new(MemoryPermissionsBackend::new()),
            next_permission_status_id: 1,
            permission_status_features: HashMap::new(),
            console_counts: HashMap::new(),
            console_timers: HashMap::new(),
            console_group_depth: 0,
            next_request_id: 1,
            requests: HashMap::new(),
            fetch_signals: HashMap::new(),
            abort_signals: HashMap::new(),
            next_signal_id: 1,
            next_url_id: 1,
            url_entries: HashMap::new(),
            response_urls: HashMap::new(),
            image_states: HashMap::new(),
        }
    }
}

impl Default for BindingState {
    fn default() -> Self {
        Self::new()
    }
}

/// The global registry symbol for this realm.
pub fn registry_symbol<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<Symbol<'js>> {
    Symbol::new_global(ctx.clone(), REGISTRY_SYMBOL)
}

/// The registry object on `globalThis`, creating it (with its Maps)
/// if this is the first call for the realm.
pub fn registry<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<Object<'js>> {
    let globals = ctx.globals();
    let sym = registry_symbol(ctx)?;
    let atom: Atom = sym.as_atom();
    if globals.contains_key(atom.clone())? {
        return globals.get(atom);
    }
    let reg = Object::new(ctx.clone())?;
    // One Map per registry use; timers/rAF are wave 1's, the rest are
    // wave 2a's. Created here so either install order works.
    for name in [
        "timers",
        "raf",
        "wrappers",
        "listeners",
        "fetches",
        "responses",
        "headers",
        "prototypes",
        "classLists",
        "styles",
        "mutationObservers",
        "mutationObserverInstances",
        "resizeObservers",
        "resizeObserverInstances",
        "intersectionObservers",
        "intersectionObserverInstances",
        // Stage 9 Rec 11: live `PermissionStatus` objects, by id,
        // so a state change can dispatch `onchange` to them.
        "permissionStatuses",
        // Loop-driven modules (wave 2b): the module promise lives
        // here while its top-level await waits on a host event; the
        // loop polls it by id (see `TabRealm::poll_module`).
        "modules",
        // Stage 12 Rec 7: cached per-element `dataset` objects, by
        // wrapper id (like `classLists` / `styles` above).
        "datasets",
        // Stage 12 Rec 7: the cached `searchParams` object per
        // `URL`, by URL id (`URL.searchParams` is `===`-stable).
        "urlSearchParams",
        // Stage 12 Rec 8: live `AbortSignal` objects, by id, so a
        // `Request.signal` read returns the same object.
        "abortSignals",
    ] {
        let map: Object = ctx.eval("new Map()")?;
        reg.set(name, map)?;
    }
    // Non-enumerable, non-writable, non-configurable: the Maps are
    // reachable only through the symbol, never by name.
    globals.prop(
        atom,
        rquickjs::object::Property::from(reg.clone()),
    )?;
    Ok(reg)
}

/// One named `Map` out of the registry.
pub fn registry_map<'js>(ctx: &Ctx<'js>, name: &str) -> rquickjs::Result<Object<'js>> {
    let reg = registry(ctx)?;
    reg.get(name)
}

/// Get a value out of a registry `Map` by numeric id.
pub fn map_get<'js>(
    _ctx: &Ctx<'js>,
    map: &Object<'js>,
    id: u64,
) -> rquickjs::Result<Value<'js>> {
    let get: Function = map.get("get")?;
    get.call((This(map.clone()), id as f64))
}

/// Set a value in a registry `Map` by numeric id.
pub fn map_set<'js>(
    _ctx: &Ctx<'js>,
    map: &Object<'js>,
    id: u64,
    value: Value<'js>,
) -> rquickjs::Result<()> {
    let set: Function = map.get("set")?;
    let _: Value = set.call((This(map.clone()), id as f64, value))?;
    Ok(())
}

/// Delete an entry from a registry `Map` by numeric id.
pub fn map_delete<'js>(_ctx: &Ctx<'js>, map: &Object<'js>, id: u64) {
    if let Ok(delete) = map.get::<_, Function>("delete") {
        let _: rquickjs::Result<Value> = delete.call((This(map.clone()), id as f64));
    }
}

/// Get a value out of a registry `Map` by string key.
pub fn map_get_str<'js>(
    _ctx: &Ctx<'js>,
    map: &Object<'js>,
    key: &str,
) -> rquickjs::Result<Value<'js>> {
    let get: Function = map.get("get")?;
    get.call((This(map.clone()), key))
}

/// Set a value in a registry `Map` by string key.
pub fn map_set_str<'js>(
    _ctx: &Ctx<'js>,
    map: &Object<'js>,
    key: &str,
    value: Value<'js>,
) -> rquickjs::Result<()> {
    let set: Function = map.get("set")?;
    let _: Value = set.call((This(map.clone()), key, value))?;
    Ok(())
}

/// Delete an entry from a registry `Map` by string key.
pub fn map_delete_str<'js>(_ctx: &Ctx<'js>, map: &Object<'js>, key: &str) {
    if let Ok(delete) = map.get::<_, Function>("delete") {
        let _: rquickjs::Result<Value> = delete.call((This(map.clone()), key));
    }
}

/// The `DomKey` id stored on a wrapper object (under the domkey
/// symbol), or `None` if the object is not a wrapper.
pub fn wrapper_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), DOMKEY_SYMBOL).ok()?;
    let atom: Atom = sym.as_atom();
    let value: Value = obj.get(atom).ok()?;
    value.as_number().map(|n| n as u64)
}

/// The `DomKey` for a `this` value, validating the wrapper shape.
/// `Err` is a `TypeError` for a wrong `this`.
pub fn dom_key_of_this<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: &Object<'js>,
) -> rquickjs::Result<DomKey> {
    let id = wrapper_id_of(ctx, this)
        .ok_or_else(|| throw_type_error(ctx, "called on a non-Node object"))?;
    state
        .borrow()
        .wrapper_keys
        .get(&id)
        .copied()
        .ok_or_else(|| throw_type_error(ctx, "called on a detached wrapper"))
}

/// Like `dom_key_of_this`, but returns `Ok(None)` for a stale
/// wrapper instead of throwing. A wrapper whose id was evicted by an
/// earlier stale access stays stale: getters use this and return
/// `null`, setters/methods keep `dom_key_of_this` and throw
/// `TypeError` for stale keys.
pub fn dom_key_of_this_opt<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: &Object<'js>,
) -> rquickjs::Result<Option<DomKey>> {
    let id = wrapper_id_of(ctx, this)
        .ok_or_else(|| throw_type_error(ctx, "called on a non-Node object"))?;
    let key = match state.borrow().wrapper_keys.get(&id).copied() {
        Some(k) => k,
        // Already evicted as stale by an earlier access: still stale.
        None => return Ok(None),
    };
    let host = dom::host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
    if !host.borrow().validate(key) {
        dom::evict_wrapper(ctx, state, key);
        return Ok(None);
    }
    Ok(Some(key))
}

/// Convert a Rust value to a JS `Value` (primitives, strings, and
/// wrapper types alike, via `IntoJs`). The conversion cannot fail
/// for the types used here; a failure would be a host bug.
pub fn js<'js, T>(ctx: &Ctx<'js>, value: T) -> Value<'js>
where
    T: rquickjs::IntoJs<'js>,
{
    value.into_js(ctx).expect("Rust→JS conversion")
}

/// JS `null` as a `Value`.
pub fn js_null<'js>(ctx: &Ctx<'js>) -> Value<'js> {
    js(ctx, rquickjs::Null)
}

/// Throw a JS exception of the given constructor name with `msg`.
/// Returns the `rquickjs::Error` to return from the host function.
pub fn throw_named(ctx: &Ctx<'_>, name: &str, msg: &str) -> rquickjs::Error {
    // Rust `{:?}` on a String yields a double-quoted literal whose
    // escapes (`\"`, `\\`, `\n`, `\u{...}`) are all valid JS string
    // escapes, so the message survives the round-trip verbatim for
    // the ASCII messages used here.
    let value: Value = match ctx.eval::<Value, _>(format!("new {name}({msg:?})")) {
        Ok(v) => v,
        Err(e) => return e,
    };
    ctx.throw(value)
}

/// `throw_named` for `TypeError`.
pub fn throw_type_error(ctx: &Ctx<'_>, msg: &str) -> rquickjs::Error {
    throw_named(ctx, "TypeError", msg)
}

/// `throw_named` for a DOMException name (`HierarchyRequestError`,
/// `NotFoundError`, `InvalidStateError`, `SyntaxError`, …).
/// M4: QuickJS has no `DOMException` global, so the exception is a
/// plain `Error` whose `name` is set to the DOM name — `err.name`
/// reads correctly in page JS, which is what M4 tests assert on.
pub fn throw_dom(ctx: &Ctx<'_>, name: &str, msg: &str) -> rquickjs::Error {
    let value: Value = match ctx.eval::<Value, _>(format!(
        "(() => {{ const e = new Error({msg:?}); e.name = {name:?}; return e; }})()"
    )) {
        Ok(v) => v,
        Err(e) => return e,
    };
    ctx.throw(value)
}

/// Map a [`crate::host::DomHostError`] to the DOMException it raises.
pub fn throw_host_error(ctx: &Ctx<'_>, err: crate::host::DomHostError) -> rquickjs::Error {
    use crate::host::DomHostError;
    match err {
        DomHostError::StaleKey => {
            throw_type_error(ctx, "the node is no longer in the document")
        }
        DomHostError::Hierarchy => throw_dom(
            ctx,
            "HierarchyRequestError",
            "the node cannot be inserted at the specified point in the hierarchy",
        ),
        DomHostError::NotFound => {
            throw_dom(ctx, "NotFoundError", "the node was not found")
        }
        DomHostError::InvalidState => {
            throw_dom(ctx, "InvalidStateError", "the document is in an invalid state")
        }
    }
}

/// Coerce a JS value to a Rust `String` per WebIDL `DOMString`
/// (via the `String()` constructor, so numbers/booleans convert
/// instead of throwing).
pub fn to_dom_string<'js>(ctx: &Ctx<'js>, value: &Value<'js>) -> rquickjs::Result<String> {
    if let Some(s) = value.as_string() {
        return Ok(s.to_string().unwrap_or_default());
    }
    let string_ctor: Function = ctx.globals().get("String")?;
    let coerced: Value = string_ctor.call((value.clone(),))?;
    Ok(coerced
        .as_string()
        .and_then(|s| s.to_string().ok())
        .unwrap_or_default())
}

/// Format a `console` argument for the host log (like `String(arg)`).
pub fn format_console_arg<'js>(ctx: &Ctx<'js>, value: &Value<'js>) -> String {
    to_dom_string(ctx, value).unwrap_or_else(|_| format!("{value:?}"))
}

/// Install all wave-2a bindings into a fresh realm. Called from
/// `TabJs::install_globals` (after the wave-1 loop globals) and on
/// every navigation, with a fresh [`BindingState`].
///
/// Installs, in order: the registry + prototypes, `window`/`self`,
/// `document`, `console`, `fetch`/`Response`/`Headers`,
/// `localStorage`/`sessionStorage` (Stage 9 Rec 4),
/// `navigator.permissions` + `navigator.storage` + the
/// deferred-device stubs (Stage 9 Rec 11), `Event`/`CustomEvent`,
/// and the `window.onerror` property.
pub fn install_bindings(
    realm: &mut TabRealm,
    state: &Rc<RefCell<LoopState>>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_tx: std::sync::mpsc::Sender<fetch::FetchCompletion>,
) -> Result<(), JsError> {
    let bindings = Rc::clone(bindings);
    let state = Rc::clone(state);
    realm
        .with_ctx(|ctx| install_bindings_in(ctx, &state, &bindings, fetch_tx))
        .map_err(JsError::Internal)
}

fn install_bindings_in<'js>(
    ctx: Ctx<'js>,
    state: &Rc<RefCell<LoopState>>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_tx: std::sync::mpsc::Sender<fetch::FetchCompletion>,
) -> Result<(), String> {
    // The registry first: everything else stores into it.
    let _ = registry(&ctx).map_err(|e| format!("install registry: {e}"))?;
    events::install_events(&ctx, bindings).map_err(|e| format!("install events: {e}"))?;
    dom::install_prototypes(&ctx, bindings).map_err(|e| format!("install prototypes: {e}"))?;
    dom::install_window(&ctx, bindings).map_err(|e| format!("install window: {e}"))?;
    dom::install_console(&ctx, bindings).map_err(|e| format!("install console: {e}"))?;
    dom::install_url(&ctx, bindings).map_err(|e| format!("install URL: {e}"))?;
    dom::install_dataset_helpers(&ctx, bindings)
        .map_err(|e| format!("install dataset helpers: {e}"))?;
    fetch::install_fetch(&ctx, state, bindings, fetch_tx).map_err(|e| format!("install fetch: {e}"))?;
    storage::install_storage(&ctx, bindings).map_err(|e| format!("install storage: {e}"))?;
    permissions::install_permissions(&ctx, bindings).map_err(|e| format!("install permissions: {e}"))?;
    observers::install_observers(&ctx, bindings).map_err(|e| format!("install observers: {e}"))?;
    Ok(())
}

/// Install `Symbol.iterator` on `proto` as a function returning a
/// plain `{ next() }` iterator over `this.entries()`.
///
/// The `entries` method itself returns an Array, which is iterable
/// but not an iterator: `for..of` calls `.next()` on the value the
/// `@@iterator` method returns, so wiring `@@iterator` straight to
/// `entries` throws `TypeError: not a function` on the first step.
/// This wrapper closes over the entries array and hands out one
/// `{ value, done }` pair per step.
pub fn install_entries_iterator<'js>(
    ctx: &Ctx<'js>,
    proto: &Object<'js>,
) -> rquickjs::Result<()> {
    // Pure-JS install (no Rust closure capturing a JS value):
    // a Rust `Function::new` closure holding a `Function<'js>` is a
    // GC-invisible root — the captured object survives to
    // `JS_FreeRuntime` and aborts the process. The JS closure below
    // captures only the entries array, which the GC can trace.
    let install: Function = ctx.eval(
        "(proto) => { proto[Symbol.iterator] = function() { const entries = this.entries(); let i = 0; return { next() { return i < entries.length ? { value: entries[i++], done: false } : { value: undefined, done: true }; } }; }; }",
    )?;
    let _: Value = install.call((proto.clone(),))?;
    Ok(())
}

/// Build an `Error` value with `name` / `message` without throwing
/// it (for promise rejections such as `AbortError`, where the error
/// is a value, not a throw).
pub fn make_dom_error<'js>(ctx: &Ctx<'js>, name: &str, msg: &str) -> rquickjs::Result<Value<'js>> {
    ctx.eval(format!(
        "(() => {{ const e = new Error({msg:?}); e.name = {name:?}; return e; }})()"
    ))
}

/// Call `window.onerror` (if the page set one) with
/// `(message, source, lineno, colno, error)`.
/// Returns `true` when the handler returned `true` (suppress the
/// default report). Called from `TabJs::record_or_propagate`.
/// A throwing `onerror` is reported, never fatal.
///
/// Stage 12 Rec 9: `source` / `lineno` / `colno` come from the
/// throwing script's stack (parsed by the loop out of the
/// `JsError` string — the realm stringifies the caught exception
/// with its stack attached); they are `None` (→ `undefined`) when
/// the error carries no location (interrupted / OOM / module
/// rejection). The `error` argument is still `undefined`: the
/// error object itself does not survive the realm boundary (the
/// realm keeps only its string form), so there is no value to
/// pass — a later wave that keeps the caught value in the realm
/// can fill it in.
pub fn call_onerror(
    realm: &mut TabRealm,
    bindings: &Rc<RefCell<BindingState>>,
    message: &str,
    source: Option<&str>,
    lineno: Option<u32>,
    colno: Option<u32>,
) -> bool {
    let message = message.to_string();
    let source = source.map(|s| s.to_string());
    let bindings = Rc::clone(bindings);
    realm.with_ctx(|ctx| {
        let globals = ctx.globals();
        let handler: Value = match globals.get("onerror") {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !handler.is_function() {
            return false;
        }
        let func = match rquickjs::Function::from_js(&ctx, handler) {
            Ok(f) => f,
            Err(_) => return false,
        };
        // onerror(message, source, lineno, colno, error).
        let source_value: Value = match &source {
            Some(s) => js(&ctx, s.clone()),
            None => js(&ctx, rquickjs::Undefined),
        };
        let lineno_value: Value = match lineno {
            Some(n) => js(&ctx, n as f64),
            None => js(&ctx, rquickjs::Undefined),
        };
        let colno_value: Value = match colno {
            Some(n) => js(&ctx, n as f64),
            None => js(&ctx, rquickjs::Undefined),
        };
        let result: rquickjs::Result<Value> = func.call((
            This(globals),
            message,
            source_value,
            lineno_value,
            colno_value,
            rquickjs::Undefined,
        ));
        match result {
            Ok(v) => v.as_bool().unwrap_or(false),
            Err(e) => {
                bindings
                    .borrow_mut()
                    .errors
                    .push(format!("window.onerror threw: {e}"));
                false
            }
        }
    })
}
