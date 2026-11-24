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
//! Full WebIDL/codegen, workers, Shadow DOM, `MutationObserver` /
//! `ResizeObserver` / `IntersectionObserver`, import maps, storage
//! APIs, full fetch/CORS (CORS is M5 — see the `TODO(M5)` at the
//! fetch gate in `fetch.rs`), CSP (M10 — see the `TODO(M10)` at the
//! same gate). Each gets a one-line `TODO(stage)` at its hook point.
//!
//! TODO(M5): WebIDL/codegen — the bindings in this module are
//! hand-written per interface; a generator would replace them when
//! the binding surface grows (forms, storage, workers).

pub mod dom;
pub mod events;
pub mod fetch;
#[cfg(test)]
pub mod test_host;
#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use rquickjs::function::This;
use rquickjs::{Atom, Ctx, FromJs, Function, Object, Symbol, Value};

use crate::error::JsError;
use crate::event_loop::LoopState;
use crate::host::{DomHost, DomKey};
use crate::realm::TabRealm;

pub use fetch::{FetchBackend, FetchCompletion, FetchError, FetchRequest, FetchResponse};

/// `Symbol.for("pluto.registry")` — the registry key. Reconstructed
/// on every entry (a global symbol), so Rust never has to store it.
pub const REGISTRY_SYMBOL: &str = "pluto.registry";

/// `Symbol.for("pluto.domkey")` — the per-wrapper DomKey-id property.
pub const DOMKEY_SYMBOL: &str = "pluto.domkey";

/// `Symbol.for("pluto.response")` — the per-Response body id property.
pub const RESPONSE_SYMBOL: &str = "pluto.response";

/// `Symbol.for("pluto.headers")` — the per-Headers id property.
pub const HEADERS_SYMBOL: &str = "pluto.headers";

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
        // Loop-driven modules (wave 2b): the module promise lives
        // here while its top-level await waits on a host event; the
        // loop polls it by id (see `TabRealm::poll_module`).
        "modules",
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
/// `Event`/`CustomEvent`, and the `window.onerror` property.
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
    fetch::install_fetch(&ctx, state, bindings, fetch_tx).map_err(|e| format!("install fetch: {e}"))?;
    Ok(())
}

/// Call `window.onerror` (if the page set one) with `message`.
/// Returns `true` when the handler returned `true` (suppress the
/// default report). Called from `TabJs::record_or_propagate`.
/// A throwing `onerror` is reported, never fatal.
pub fn call_onerror(
    realm: &mut TabRealm,
    bindings: &Rc<RefCell<BindingState>>,
    message: &str,
) -> bool {
    let message = message.to_string();
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
        // onerror(message, source, lineno, colno, error). M4 passes
        // only the message; source/line are undefined (the loop does
        // not track them yet — a later wave can fill them in).
        let result: rquickjs::Result<Value> = func.call((
            This(globals),
            message,
            rquickjs::Undefined,
            rquickjs::Undefined,
            rquickjs::Undefined,
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
