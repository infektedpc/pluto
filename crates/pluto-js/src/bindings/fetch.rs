//! `fetch` / `Response` / `Headers` (binding item 5).
//!
//! The JS surface lives here; the network backend is the
//! [`FetchBackend`] trait, implemented by the integration wave with
//! `pluto-net` + the Stage-5 `RequestResource`. Promise resolution
//! goes through the event loop's Networking task queue — never inline
//! on a network thread, never inline in the `fetch()` host call:
//! `fetch()` starts the backend and returns a pending promise; the
//! backend delivers its outcome over an `mpsc` channel (from any
//! thread); `TabJs` drains the channel each turn and enqueues a
//! Networking-source Host task that resolves or rejects the promise
//! (see [`resolve_fetch`]).
//!
//! ```text
//! fetch(url) ──► backend.start_fetch(req, tx) ──► [network thread]
//!      │                                              │
//!      │   TabJs::run_turn drains fetch_rx            ▼
//!      └──► pending Promise ──► Networking task ──► resolve(Response)
//! ```
//!
//! M4 subset: `Response` has `ok` / `status` / `statusText` /
//! `headers.get` / `text()` / `json()`; `Headers` has `get` (+ `has`
//! / `set` / `append`, which are trivial once `get` exists).
//! Streaming `body` is later.
//!
//! TODO(M5): CORS. The fetch gate below (`check_fetch_allowed`) is
//! where the CORS check goes: M4 tracks no document origin in
//! `TabJs`, so the check cannot run yet — `fetch()` must NOT be
//! wired to a real network backend until M5 lands the origin and the
//! check. The test backend is a fake; the trait documents the rule.
//! TODO(M10): CSP. The same gate is where the `connect-src` check
//! goes; do not add an `unsafe-eval`-style switch in M4.
//! TODO(M5): `init.mode` / `credentials` are parsed and ignored;
//! opaque `no-cors` responses are a later stage, not a silent pass.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use rquickjs::function::{Opt, This};
use rquickjs::object::{Accessor, Property};
use rquickjs::{Coerced, Ctx, Function, Object, Symbol, Value};

use crate::bindings::{
    BindingState, HEADERS_SYMBOL, RESPONSE_SYMBOL, js, js_null, map_delete, map_get, map_set,
    registry_map, throw_type_error, to_dom_string,
};
use crate::event_loop::LoopState;

/// One fetch request, as the binding hands it to the backend.
/// `url` is absolute (`http(s)://…`) or a same-origin path (`/…`);
/// anything else is rejected by the binding before it reaches here.
#[derive(Debug, Clone)]
pub struct FetchRequest {
    /// Pending-fetch id, assigned by the binding. The backend echoes
    /// it in its [`FetchCompletion`] so the loop resolves the right
    /// promise.
    pub id: u64,
    /// Request URL (absolute, or a same-origin path).
    pub url: String,
    /// HTTP method, uppercased (`GET` by default).
    pub method: String,
    /// Request headers, in order.
    pub headers: Vec<(String, String)>,
    /// Request body bytes, if any.
    pub body: Option<Vec<u8>>,
}

/// One fetch response, as the backend hands it back.
#[derive(Debug, Clone)]
pub struct FetchResponse {
    /// HTTP status code.
    pub status: u16,
    /// Reason phrase (may be empty; the binding fills a default).
    pub status_text: String,
    /// Response headers, in order.
    pub headers: Vec<(String, String)>,
    /// Full response body (M4: bodies are buffered, not streamed).
    pub body: Vec<u8>,
}

/// Why a fetch failed (surfaces as a rejected promise).
#[derive(Debug, Clone)]
pub enum FetchError {
    /// The network request itself failed.
    Network(String),
    /// The URL was invalid.
    InvalidUrl(String),
    /// The fetch was cancelled (navigation epoch bump / tab close).
    Cancelled,
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Network(m) => write!(f, "network error: {m}"),
            FetchError::InvalidUrl(m) => write!(f, "invalid URL: {m}"),
            FetchError::Cancelled => write!(f, "fetch cancelled"),
        }
    }
}

/// A completed fetch, delivered to the loop. `id` is the pending
/// fetch the outcome belongs to.
#[derive(Debug)]
pub struct FetchCompletion {
    /// Pending-fetch id (from [`BindingState::next_fetch_id`]).
    pub id: u64,
    /// The outcome.
    pub result: Result<FetchResponse, FetchError>,
}

/// The network backend for `fetch()`. Implemented by the integration
/// wave with `pluto-net` (Stage-5 `RequestResource`: destination
/// `fetch`, mode/credentials/origin/top-level-site/epoch).
///
/// Contract: `start_fetch` must return promptly (never block the
/// worker thread) and must send exactly one [`FetchCompletion`] on
/// `done` — from any thread — when the response (or failure) is
/// known. The binding resolves the promise from the loop's
/// Networking task queue; the backend must never resolve anything
/// itself, and must never call into the realm.
pub trait FetchBackend: Send + Sync {
    /// Start a fetch. Send one outcome on `done`, exactly once.
    fn start_fetch(&self, request: FetchRequest, done: mpsc::Sender<FetchCompletion>);
}

/// Install `fetch`, `Response`, and `Headers` on the global object.
pub fn install_fetch<'js>(
    ctx: &Ctx<'js>,
    _state: &Rc<RefCell<LoopState>>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_tx: mpsc::Sender<FetchCompletion>,
) -> Result<(), String> {
    let globals = ctx.globals();
    install_response_prototype(ctx, bindings)?;
    install_headers_prototype(ctx, bindings)?;

    let b = Rc::clone(bindings);
    let fetch_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, input: Value<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            fetch_impl(&ctx, &b, &fetch_tx, input, init.0)
        },
    )
    .map_err(|e| format!("install fetch: {e}"))?;
    globals
        .set("fetch", fetch_fn)
        .map_err(|e| format!("install fetch: {e}"))?;
    Ok(())
}

/// The fetch gate. M4: validates the URL shape only.
/// TODO(M5): CORS — check the request URL against the document
/// origin here (same-origin, or a CORS-safelisted cross-origin
/// request) and reject anything else. Do NOT ship a permissive
/// fetch: until this check exists, `fetch()` must only be used
/// with the test backend.
/// TODO(M10): CSP — check `connect-src` here.
///
/// Wave 2b note: the integration wave's `FetchBackend`
/// (`pluto-browser`) IS the M4 origin check — it resolves the URL
/// against the document base and rejects absolute cross-origin
/// URLs before any network access, so a real backend is wired
/// only behind that same-origin gate. The full CORS preflight /
/// `mode` handling still lands in M5 at this gate.
fn check_fetch_allowed(_url: &str) -> Result<(), String> {
    // No origin is tracked in M4, so there is nothing to check
    // against yet. The gate exists so the M5/M10 checks have exactly
    // one place to land; the README states the rule.
    Ok(())
}

/// Default reason phrase for a status code.
fn default_status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "",
    }
}

/// `window.fetch(input, init?)`.
fn fetch_impl<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_tx: &mpsc::Sender<FetchCompletion>,
    input: Value<'js>,
    init: Option<Value<'js>>,
) -> rquickjs::Result<Value<'js>> {
    // The promise first: every path below resolves or rejects it.
    let (promise, resolve, reject) = ctx.promise()?;
    let reject_with = |ctx: &Ctx<'js>, msg: &str| -> rquickjs::Result<()> {
        let err: Value = ctx.eval(format!("new TypeError({msg:?})"))?;
        let _: Value = reject.call((err,))?;
        Ok(())
    };

    // `input`: a URL string, or a `Request`-shaped object with a
    // `url` property (M4: no `Request` class yet — read the property).
    let url = if let Some(s) = input.as_string() {
        s.to_string().unwrap_or_default()
    } else if let Some(obj) = input.as_object() {
        match obj.get::<_, String>("url") {
            Ok(u) => u,
            Err(_) => {
                reject_with(ctx, "fetch: input is not a URL string or Request")?;
                return Ok(promise.into_value());
            }
        }
    } else {
        reject_with(ctx, "fetch: input is not a URL string or Request")?;
        return Ok(promise.into_value());
    };
    if !(url.starts_with("http://") || url.starts_with("https://") || url.starts_with('/')) {
        reject_with(ctx, "fetch: invalid URL")?;
        return Ok(promise.into_value());
    }
    if let Err(e) = check_fetch_allowed(&url) {
        reject_with(ctx, &e)?;
        return Ok(promise.into_value());
    }

    // `init`: method / headers / body. `mode` and `credentials`
    // are parsed and ignored — TODO(M5) (opaque responses).
    let mut method = "GET".to_string();
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut body: Option<Vec<u8>> = None;
    if let Some(init) = init
        && let Some(obj) = init.as_object() {
            if let Ok(m) = obj.get::<_, String>("method") {
                method = m.to_ascii_uppercase();
            }
            if let Ok(h) = obj.get::<_, Value>("headers") {
                headers = parse_headers_init(ctx, &h)?;
            }
            if let Ok(b) = obj.get::<_, Value>("body")
                && !b.is_null() && !b.is_undefined() {
                    body = Some(to_dom_string(ctx, &b)?.into_bytes());
                }
            // TODO(M5): `mode` / `credentials` — read here once the
            // CORS check exists; ignored until then.
            let _ = obj.get::<_, String>("mode");
            let _ = obj.get::<_, String>("credentials");
        }
    let id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_fetch_id;
        s.next_fetch_id += 1;
        id
    };
    let request = FetchRequest {
        id,
        url,
        method,
        headers,
        body,
    };
    bindings.borrow_mut().pending_fetches.insert(id, request.clone());
    // Stash resolve/reject in the registry until the outcome arrives.
    let fetches = registry_map(ctx, "fetches")?;
    let entry = Object::new(ctx.clone())?;
    entry.set("resolve", resolve)?;
    entry.set("reject", reject)?;
    map_set(ctx, &fetches, id, Value::from(entry))?;

    let backend = bindings.borrow().fetch_backend.clone();
    match backend {
        Some(backend) => {
            // The backend owns delivery: it sends exactly one
            // completion on the channel, from any thread. If the
            // receiver is gone (navigation), the send fails silently
            // and the pending entry is dropped with the realm.
            backend.start_fetch(request, fetch_tx.clone());
        }
        None => {
            // No backend attached: reject now (via the promise job
            // queue, so `.catch` ordering is the normal microtask
            // order).
            let fetches = registry_map(ctx, "fetches")?;
            if let Ok(entry) = map_get(ctx, &fetches, id)
                && let Some(obj) = entry.as_object()
                    && let Ok(reject_fn) = obj.get::<_, Function>("reject") {
                        let err: Value =
                            ctx.eval("new TypeError('fetch: no fetch backend installed')")?;
                        let _: rquickjs::Result<Value> = reject_fn.call((err,));
                    }
            map_delete(ctx, &fetches, id);
            bindings.borrow_mut().pending_fetches.remove(&id);
        }
    }
    Ok(promise.into_value())
}

/// Parse a `HeadersInit`: an object of name → value, or an array of
/// `[name, value]` pairs.
fn parse_headers_init<'js>(
    ctx: &Ctx<'js>,
    value: &Value<'js>,
) -> rquickjs::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    if let Some(obj) = value.as_object() {
        if let Some(arr) = obj.clone().into_array() {
            let len: usize = arr.len();
            for i in 0..len {
                if let Ok(pair) = arr.get::<Value>(i)
                    && let Some(pair_obj) = pair.as_object()
                        && let Some(pair_arr) = pair_obj.clone().into_array()
                            && pair_arr.len() >= 2 {
                                let n: Value = pair_arr.get(0)?;
                                let v: Value = pair_arr.get(1)?;
                                out.push((
                                    to_dom_string(ctx, &n)?,
                                    to_dom_string(ctx, &v)?,
                                ));
                            }
            }
            return Ok(out);
        }
        for key in obj.keys::<String>() {
            let key = key?;
            let v: Value = obj.get(key.clone())?;
            out.push((key, to_dom_string(ctx, &v)?));
        }
    }
    Ok(out)
}

/// Resolve (or reject) a pending fetch from a Networking-source Host
/// task. Called by `TabJs` with the loop's `Ctx`; builds the
/// `Response` object and calls the stashed resolve/reject. A throwing
/// resolve is reported, never fatal.
pub fn resolve_fetch<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    completion: FetchCompletion,
) -> rquickjs::Result<()> {
    let id = completion.id;
    bindings.borrow_mut().pending_fetches.remove(&id);
    let fetches = registry_map(ctx, "fetches")?;
    let entry = map_get(ctx, &fetches, id)?;
    map_delete(ctx, &fetches, id);
    let Some(entry_obj) = entry.as_object() else {
        return Ok(()); // Cancelled by navigation: nothing to resolve.
    };
    match completion.result {
        Ok(response) => {
            let response_obj = make_response(ctx, bindings, &response)?;
            let resolve: Function = entry_obj.get("resolve")?;
            let result: rquickjs::Result<Value> =
                resolve.call((Value::from(response_obj),));
            if let Err(e) = result {
                bindings
                    .borrow_mut()
                    .errors
                    .push(format!("fetch resolve threw: {e}"));
            }
        }
        Err(err) => {
            let reject: Function = entry_obj.get("reject")?;
            let reason: Value = ctx.eval(format!("new TypeError({:?})", err.to_string()))?;
            let result: rquickjs::Result<Value> = reject.call((reason,));
            if let Err(e) = result {
                bindings
                    .borrow_mut()
                    .errors
                    .push(format!("fetch reject threw: {e}"));
            }
        }
    }
    Ok(())
}

/// Build a `Response` wrapper for a completed fetch, storing its
/// body + headers in [`BindingState`] under a fresh body id.
pub fn make_response<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    response: &FetchResponse,
) -> rquickjs::Result<Object<'js>> {
    let body_id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_body_id;
        s.next_body_id += 1;
        s.response_bodies.insert(id, response.body.clone());
        s.response_headers.insert(id, response.headers.clone());
        id
    };
    let status_text = if response.status_text.is_empty() {
        default_status_text(response.status).to_string()
    } else {
        response.status_text.clone()
    };
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("response")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), RESPONSE_SYMBOL)?;
    obj.set(sym.as_atom(), body_id as f64)?;
    // Status lives on the object (plain data properties; the
    // prototype's accessors read them).
    obj.prop("status", Property::from(response.status as f64))?;
    obj.prop("statusText", Property::from(status_text))?;
    // The `headers` object shares the body id.
    let headers_obj = make_headers_object(ctx, bindings, body_id)?;
    obj.set("headers", headers_obj)?;
    Ok(obj)
}

/// Build a `Headers` wrapper sharing `body_id`'s header list.
fn make_headers_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
) -> rquickjs::Result<Object<'js>> {
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("headers")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), HEADERS_SYMBOL)?;
    obj.set(sym.as_atom(), body_id as f64)?;
    let _ = bindings;
    Ok(obj)
}

/// The body id stored on a `Response` / `Headers` object.
fn body_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>, symbol: &str) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), symbol).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

fn install_response_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let proto = Object::new(ctx.clone()).map_err(|e| format!("Response prototype: {e}"))?;

    // `ok`: 200 ≤ status < 300.
    proto
        .prop(
            "ok",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
                    let status: f64 = this.get("status").unwrap_or(0.0);
                    let _ = ctx;
                    Ok((200.0..300.0).contains(&status))
                },
            ),
        )
        .map_err(|e| format!("Response.ok: {e}"))?;
    // `status` / `statusText` are plain data properties on the
    // object (set by `make_response`); the prototype exposes them
    // as accessors that read the own property, so a missing value
    // reads as `undefined` instead of throwing.
    proto
        .prop(
            "status",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                    let _ = ctx;
                    this.get("status")
                },
            ),
        )
        .map_err(|e| format!("Response.status: {e}"))?;
    proto
        .prop(
            "statusText",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                    let _ = ctx;
                    this.get("statusText")
                },
            ),
        )
        .map_err(|e| format!("Response.statusText: {e}"))?;

    // `text()`: a promise resolved with the body as a string.
    let b = Rc::clone(bindings);
    let text_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, _reject) = ctx.promise()?;
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let text = body_id
                .and_then(|id| b.borrow().response_bodies.get(&id).cloned())
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default();
            let _: Value = resolve.call((text,))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("Response.text: {e}"))?;
    proto
        .set("text", text_fn)
        .map_err(|e| format!("Response.text: {e}"))?;

    // `json()`: a promise resolved with the parsed body, or rejected
    // with the `JSON.parse` error.
    let b = Rc::clone(bindings);
    let json_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, reject) = ctx.promise()?;
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let text = body_id
                .and_then(|id| b.borrow().response_bodies.get(&id).cloned())
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default();
            let json: Object = ctx.globals().get("JSON")?;
            let parse: Function = json.get("parse")?;
            match parse.call::<_, Value>((text,)) {
                Ok(value) => {
                    let _: Value = resolve.call((value,))?;
                }
                Err(_) => {
                    // The pending exception is the real `SyntaxError`
                    // from `JSON.parse`; surface its message instead
                    // of rquickjs's generic error rendering.
                    let caught = ctx.catch();
                    let msg = caught
                        .as_object()
                        .and_then(|o| o.get::<_, String>("message").ok())
                        .unwrap_or_else(|| "JSON parse error".to_string());
                    let reason: Value =
                        ctx.eval(format!("new SyntaxError({msg:?})"))?;
                    let _: Value = reject.call((reason,))?;
                }
            }
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("Response.json: {e}"))?;
    proto
        .set("json", json_fn)
        .map_err(|e| format!("Response.json: {e}"))?;

    protos
        .set("response", proto)
        .map_err(|e| format!("prototypes response: {e}"))?;

    // `Response` constructor: `new Response(body, init?)`. M4: the
    // body is a string (or null); `init` carries `status` /
    // `statusText` / `headers`. A Rust `Function::new` is callable
    // but not constructable in QuickJS, so the constructor is a JS
    // function that delegates to a Rust factory in the registry; the
    // factory returns the wrapper object, which `new` then adopts.
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, body: Opt<Value<'js>>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let bytes = match body.0 {
                Some(v) if !v.is_null() && !v.is_undefined() => {
                    to_dom_string(&ctx, &v)?.into_bytes()
                }
                _ => Vec::new(),
            };
            let mut status: u16 = 200;
            let mut status_text = String::new();
            let mut headers: Vec<(String, String)> = Vec::new();
            if let Some(init) = init.0
                && let Some(obj) = init.as_object() {
                    if let Ok(s) = obj.get::<_, f64>("status") {
                        status = s as u16;
                    }
                    if let Ok(t) = obj.get::<_, String>("statusText") {
                        status_text = t;
                    }
                    if let Ok(h) = obj.get::<_, Value>("headers") {
                        headers = parse_headers_init(&ctx, &h)?;
                    }
                }
            let response = FetchResponse {
                status,
                status_text,
                headers,
                body: bytes,
            };
            make_response(&ctx, &b, &response).map(Value::from)
        },
    )
    .map_err(|e| format!("Response factory: {e}"))?;
    let reg = crate::bindings::registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("responseFactory", factory)
        .map_err(|e| format!("install Response factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.Response = function Response(body, init) {            if (!(this instanceof Response)) throw new TypeError('Response constructor requires new');            return globalThis[Symbol.for('pluto.registry')].responseFactory(body, init);          };",
    )
    .map_err(|e| format!("install Response: {e}"))?;
    // The constructor's `.prototype` is the prototype object, so
    // `response instanceof Response` holds.
    let globals = ctx.globals();
    let response_ctor: Object = globals.get("Response").map_err(|e| e.to_string())?;
    response_ctor
        .set("prototype", protos.get::<_, Object>("response").map_err(|e| e.to_string())?)
        .map_err(|e| format!("Response.prototype: {e}"))?;
    Ok(())
}

fn install_headers_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let proto = Object::new(ctx.clone()).map_err(|e| format!("Headers prototype: {e}"))?;

    let b = Rc::clone(bindings);
    let get_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Ok(js_null(&ctx));
            };
            let headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            let want = name.to_ascii_lowercase();
            Ok(match headers
                .iter()
                .find(|(n, _)| n.to_ascii_lowercase() == want)
            {
                Some((_, v)) => js(&ctx, v.clone()),
                None => js_null(&ctx),
            })
        },
    )
    .map_err(|e| format!("Headers.get: {e}"))?;
    proto
        .set("get", get_fn)
        .map_err(|e| format!("Headers.get: {e}"))?;

    let b = Rc::clone(bindings);
    let has_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<bool> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Ok(false);
            };
            let headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            let want = name.to_ascii_lowercase();
            Ok(headers.iter().any(|(n, _)| n.to_ascii_lowercase() == want))
        },
    )
    .map_err(|e| format!("Headers.has: {e}"))?;
    proto
        .set("has", has_fn)
        .map_err(|e| format!("Headers.has: {e}"))?;

    let b = Rc::clone(bindings);
    let set_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Err(throw_type_error(&ctx, "called on a non-Headers object"));
            };
            let mut headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            let want = name.to_ascii_lowercase();
            match headers.iter_mut().find(|(n, _)| n.to_ascii_lowercase() == want) {
                Some((_, v)) => *v = value.to_string(),
                None => headers.push((name.to_string(), value.to_string())),
            }
            b.borrow_mut().response_headers.insert(id, headers);
            Ok(())
        },
    )
    .map_err(|e| format!("Headers.set: {e}"))?;
    proto
        .set("set", set_fn)
        .map_err(|e| format!("Headers.set: {e}"))?;

    let b = Rc::clone(bindings);
    let append_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Err(throw_type_error(&ctx, "called on a non-Headers object"));
            };
            let mut headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            headers.push((name.to_string(), value.to_string()));
            b.borrow_mut().response_headers.insert(id, headers);
            Ok(())
        },
    )
    .map_err(|e| format!("Headers.append: {e}"))?;
    proto
        .set("append", append_fn)
        .map_err(|e| format!("Headers.append: {e}"))?;

    protos
        .set("headers", proto)
        .map_err(|e| format!("prototypes headers: {e}"))?;

    // `Headers` constructor: `new Headers(init?)` — same JS
    // constructor + Rust factory shape as `Response` (a Rust
    // `Function::new` is not constructable).
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let headers = match init.0 {
                Some(v) => parse_headers_init(&ctx, &v)?,
                None => Vec::new(),
            };
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_body_id;
                s.next_body_id += 1;
                s.response_headers.insert(id, headers);
                s.response_bodies.insert(id, Vec::new());
                id
            };
            let obj = make_headers_object(&ctx, &b, id)?;
            Ok(Value::from(obj))
        },
    )
    .map_err(|e| format!("Headers factory: {e}"))?;
    let reg = crate::bindings::registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("headersFactory", factory)
        .map_err(|e| format!("install Headers factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.Headers = function Headers(init) {            if (!(this instanceof Headers)) throw new TypeError('Headers constructor requires new');            return globalThis[Symbol.for('pluto.registry')].headersFactory(init);          };",
    )
    .map_err(|e| format!("install Headers: {e}"))?;
    let globals = ctx.globals();
    let headers_ctor: Object = globals.get("Headers").map_err(|e| e.to_string())?;
    headers_ctor
        .set("prototype", protos.get::<_, Object>("headers").map_err(|e| e.to_string())?)
        .map_err(|e| format!("Headers.prototype: {e}"))?;
    Ok(())
}
