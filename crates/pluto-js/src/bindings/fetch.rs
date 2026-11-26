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
//! `type` / `headers.get` / `text()` / `json()`; `Headers` has
//! `get` (+ `has` / `set` / `append`, which are trivial once
//! `get` exists).
//!
//! Stage 16 Phase 3, Rec 12: `fetch()` resolves at headers with a
//! `Response` whose `body` is a live `ReadableStream` (see
//! [`streams`]); the backend delivers
//! [`FetchStreamEvent`]s (headers / chunk / done / error) over a
//! second mpsc channel, drained by `TabJs` like completions.
//! `text()` / `json()` / `blob()` / `arrayBuffer()` wait for the
//! stream to finish (via the registry's `bodyWaiters` Map) when
//! called before `Done`. The 10 MiB body cap moved from the
//! transport to the buffering point — it always protected memory,
//! and streaming readers are exempt by design
//! ([`BODY_BUFFER_CAP`]).
//!
//! Stage 9 Rec 2: `init.mode` / `init.credentials` are threaded
//! to the browser-side backend, which enforces them in `pluto-net`
//! (CORS preflight, credentials rules, opaque filtering); the
//! realm-side gate is a pass-through, and `Response.type` reports
//! the backend-computed tainting (`basic` / `cors` / `opaque`).
//! TODO(M10): CSP. The same gate is where the `connect-src` check
//! goes; do not add an `unsafe-eval`-style switch in M4.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use pluto_ipc::{CredentialsMode, Origin, RequestMode};
use rquickjs::function::{Opt, This};
use rquickjs::object::{Accessor, Property};
use rquickjs::{Array, ArrayBuffer, Coerced, Ctx, FromJs, Function, IntoJs, Object, Symbol, Value};

use crate::bindings::eventsource::resolve_event_source_stream_event;
use crate::bindings::streams;
use crate::bindings::{
    ABORT_SIGNAL_SYMBOL, BindingState, HEADERS_SYMBOL, REQUEST_SYMBOL, RESPONSE_SYMBOL, js,
    js_null, make_dom_error, map_delete, map_get, map_set, registry, registry_map,
    throw_type_error, to_dom_string,
};
use crate::event_loop::LoopState;

/// The body-buffer cap (Stage 16 Phase 3, Rec 12): a response
/// whose body is buffered (`.text()` etc., or a backend that only
/// speaks `start_fetch`) is capped at 10 MiB — the cap always
/// protected memory, so it moved to the buffering point. A
/// `ReadableStream` reader is exempt: chunks flow through the
/// stream without accumulating. A body that exceeds the cap while
/// buffering errors the stream (the `Response` already resolved;
/// reading its body rejects).
pub const BODY_BUFFER_CAP: usize = 10 * 1024 * 1024;

/// One `Request` object, as stored in [`BindingState::requests`].
/// The JS object holds only the id; `fetch()` reads this back.
#[derive(Debug, Clone)]
pub struct StoredRequest {
    /// Request URL (as passed to the constructor).
    pub url: String,
    /// HTTP method, uppercased.
    pub method: String,
    /// The `Headers` object's id (its list lives in
    /// [`BindingState::response_headers`], shared with responses).
    pub headers_id: u64,
    /// Request body bytes, if any.
    pub body: Option<Vec<u8>>,
    /// The `AbortSignal` id attached at construction, if any.
    pub signal_id: Option<u64>,
}

/// State of one `AbortSignal`, by id (see
/// [`BindingState::abort_signals`]).
#[derive(Debug, Clone, Default)]
pub struct AbortSignalState {
    /// True once `abort()` was called.
    pub aborted: bool,
    /// Pending-fetch ids registered on this signal.
    pub fetch_ids: Vec<u64>,
}

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
    /// `init.mode` (`"cors"` by default): how cross-origin access
    /// is treated. Threaded to the browser-side backend, which
    /// enforces it in `pluto-net` — never decided in the realm.
    pub mode: RequestMode,
    /// `init.credentials` (`"same-origin"` by default): whether
    /// cookies / auth ride along. Threaded to the backend with
    /// `mode`; `pluto-net` applies the CORS credentials rules.
    pub credentials: CredentialsMode,
}

/// The `Response.type` of a completed fetch (Fetch §2.2.6): what
/// script may learn about the response. Set by the backend from
/// the CORS tainting / origin comparison — never by script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseType {
    /// Same-origin response: status, URL, body, all headers
    /// except the forbidden response headers.
    Basic,
    /// Cross-origin response that passed the CORS check: status,
    /// URL, body, CORS-safelisted + exposed headers only.
    Cors,
    /// `no-cors` cross-origin response (or a CORS failure the
    /// backend surfaces as opaque): type is the ONLY signal —
    /// status 0, empty header list, null body.
    Opaque,
    /// A redirect with `redirect: manual`: same shape as opaque,
    /// but the URL list is exposed.
    OpaqueRedirect,
    /// A `new Response(body, init?)` construction: not a network
    /// response at all.
    Default,
}

impl ResponseType {
    /// The string script reads as `response.type`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseType::Basic => "basic",
            ResponseType::Cors => "cors",
            ResponseType::Opaque => "opaque",
            ResponseType::OpaqueRedirect => "opaqueredirect",
            ResponseType::Default => "default",
        }
    }
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
    /// What script may learn about this response
    /// (`response.type`). The backend sets it from the CORS
    /// tainting; an opaque response carries status 0, no headers,
    /// and an empty body alongside this.
    pub response_type: ResponseType,
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

/// One streaming fetch event (Stage 16 Phase 3, Rec 12): the
/// chunk channel. The backend sends `Headers` once (the promise
/// resolves then, with a `Response` whose `body` is a live
/// stream), then one `Chunk` per body chunk, then exactly one
/// terminal event (`Done`, or `Error` when the fetch fails before
/// or during delivery). The `id` is the pending-fetch id. All
/// variants carry plain Rust data — rquickjs values never cross
/// threads.
#[derive(Debug)]
pub enum FetchStreamEvent {
    /// Response headers arrived: status, reason, headers, and the
    /// backend-computed response type. The promise resolves now.
    Headers {
        id: u64,
        status: u16,
        status_text: String,
        headers: Vec<(String, String)>,
        response_type: ResponseType,
    },
    /// One body chunk (already body-decoded by the backend).
    Chunk { id: u64, bytes: Vec<u8> },
    /// The body is complete (terminal).
    Done { id: u64 },
    /// The fetch failed (terminal): before `Headers` (the promise
    /// rejects) or after it (the live stream errors).
    Error { id: u64, error: FetchError },
}

impl FetchStreamEvent {
    /// The pending-fetch id this event belongs to.
    pub fn fetch_id(&self) -> u64 {
        match self {
            FetchStreamEvent::Headers { id, .. }
            | FetchStreamEvent::Chunk { id, .. }
            | FetchStreamEvent::Done { id }
            | FetchStreamEvent::Error { id, .. } => *id,
        }
    }
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

    /// Start a streaming fetch (Stage 16 Phase 3, Rec 12): send
    /// `Headers` once, then one `Chunk` per body chunk, then
    /// exactly one terminal event (`Done` / `Error`) on
    /// `stream_tx`. The default implementation bridges a
    /// `start_fetch`-only backend: it runs the whole fetch on a
    /// helper thread and re-emits the buffered outcome as
    /// `Headers` → one `Chunk` → `Done` (so `fetch()` resolves at
    /// headers even for backends that buffer — the chunk is the
    /// whole body, and the stream closes at once). Backends with
    /// a real chunk channel override this.
    fn start_stream_fetch(
        &self,
        request: FetchRequest,
        stream_tx: mpsc::Sender<FetchStreamEvent>,
    ) {
        let (tx, rx) = mpsc::channel::<FetchCompletion>();
        let id = request.id;
        std::thread::spawn(move || {
            let Ok(completion) = rx.recv() else {
                return;
            };
            match completion.result {
                Ok(resp) => {
                    let _ = stream_tx.send(FetchStreamEvent::Headers {
                        id,
                        status: resp.status,
                        status_text: resp.status_text,
                        headers: resp.headers,
                        response_type: resp.response_type,
                    });
                    if !resp.body.is_empty() {
                        let _ = stream_tx.send(FetchStreamEvent::Chunk { id, bytes: resp.body });
                    }
                    let _ = stream_tx.send(FetchStreamEvent::Done { id });
                }
                Err(error) => {
                    let _ = stream_tx.send(FetchStreamEvent::Error { id, error });
                }
            }
        });
        self.start_fetch(request, tx);
    }

    /// Cancel an in-flight fetch (Stage 12 Rec 8: `AbortSignal`).
    /// Best effort: the backend should drop the request and send
    /// no completion for it. Default: no-op — the realm rejects
    /// the promise locally and ignores a late completion anyway.
    fn cancel_fetch(&self, _id: u64) {}

    /// The document origin this backend fetches for, if it tracks
    /// one (Stage 9 Rec 1). The realm-side gate
    /// ([`check_fetch_allowed`]) reads it for its early same-origin
    /// check; `None` (the test backend) means the realm cannot
    /// check and the backend's own gate is the only enforcement.
    /// The browser-side gate is always authoritative — the realm
    /// check is defense in depth, never a substitute, and the real
    /// check never runs inside the realm's JS: script only ever
    /// sees the rejection, never the origin value.
    fn document_origin(&self) -> Option<Origin> {
        None
    }
}

/// Install `fetch`, `Response`, and `Headers` on the global object.
/// `fetch_tx` carries whole-response completions (the legacy
/// `start_fetch` path, still drained by `TabJs`);
/// `stream_tx` carries chunk events for `start_stream_fetch` —
/// the path `fetch()` actually uses (Stage 16 Phase 3, Rec 12).
pub fn install_fetch<'js>(
    ctx: &Ctx<'js>,
    _state: &Rc<RefCell<LoopState>>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_tx: mpsc::Sender<FetchCompletion>,
    stream_tx: mpsc::Sender<FetchStreamEvent>,
) -> Result<(), String> {
    let globals = ctx.globals();
    install_response_prototype(ctx, bindings)?;
    install_headers_prototype(ctx, bindings)?;
    install_request_prototype(ctx, bindings)?;
    install_abort_prototypes(ctx, bindings)?;

    let b = Rc::clone(bindings);
    let fetch_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, input: Value<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            fetch_impl(&ctx, &b, &stream_tx, input, init.0)
        },
    )
    .map_err(|e| format!("install fetch: {e}"))?;
    globals
        .set("fetch", fetch_fn)
        .map_err(|e| format!("install fetch: {e}"))?;
    let _ = fetch_tx;
    Ok(())
}

/// The fetch gate (Stage 9 Rec 2): intentionally a pass-through.
/// The browser-side backend (`PlutoFetchBackend` in pluto-browser)
/// + the CORS engine in `pluto-net` are authoritative: the CORS
/// preflight / safelist check, opaque filtering, and the
/// credentials rules all run there, never in the realm — script
/// must never receive the internal response object. Relative URLs
/// resolve against the document base browser-side. A fetch with no
/// backend attached is rejected by the caller, not here.
/// TODO(M10): CSP — check `connect-src` here.
fn check_fetch_allowed(_url: &str, _document_origin: Option<&Origin>) -> Result<(), String> {
    // Stage 9 Rec 2: the realm-side gate is intentionally a
    // pass-through. The browser-side backend (`PlutoFetchBackend`
    // in pluto-browser) + the CORS engine in `pluto-net` are
    // authoritative: same-origin / CORS / opaque filtering and the
    // credentials rules all run there, never in the realm — script
    // must never receive the internal response object. A fetch with
    // no backend attached is rejected by the caller, not here.
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
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
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

    // `input`: a URL string, a `Request` object (Stage 12 Rec 8),
    // or a `Request`-shaped object with a `url` property.
    let mut method = "GET".to_string();
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut body: Option<Vec<u8>> = None;
    let mut signal_id: Option<u64> = None;
    let url = if let Some(s) = input.as_string() {
        s.to_string().unwrap_or_default()
    } else if let Some(obj) = input.as_object() {
        if let Some(req_id) = request_id_of(&ctx, obj) {
            let stored = match bindings.borrow().requests.get(&req_id).cloned() {
                Some(s) => s,
                None => {
                    reject_with(ctx, "fetch: input is not a valid Request")?;
                    return Ok(promise.into_value());
                }
            };
            method = stored.method;
            headers = bindings
                .borrow()
                .response_headers
                .get(&stored.headers_id)
                .cloned()
                .unwrap_or_default();
            body = stored.body;
            signal_id = stored.signal_id;
            stored.url
        } else {
            match obj.get::<_, String>("url") {
                Ok(u) => u,
                Err(_) => {
                    reject_with(ctx, "fetch: input is not a URL string or Request")?;
                    return Ok(promise.into_value());
                }
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
    // The realm-side same-origin gate (Stage 9 Rec 1): the document
    // origin comes from the attached backend; the backend's own
    // gate stays authoritative.
    let document_origin = bindings
        .borrow()
        .fetch_backend
        .as_ref()
        .and_then(|backend| backend.document_origin());
    if let Err(e) = check_fetch_allowed(&url, document_origin.as_ref()) {
        reject_with(ctx, &e)?;
        return Ok(promise.into_value());
    }

    // `init`: method / headers / body / mode / credentials / signal.
    // `mode` and `credentials` are threaded to the backend, which
    // enforces them in `pluto-net` (Stage 9 Rec 2): an unknown
    // enum value rejects with a TypeError, as the IDL conversion
    // would. Defaults: mode `"cors"`, credentials `"same-origin"`.
    let mut mode = RequestMode::Cors;
    let mut credentials = CredentialsMode::SameOrigin;
    if let Some(init) = init
        && let Some(obj) = init.as_object() {
            if let Ok(m) = obj.get::<_, String>("method") {
                method = m.to_ascii_uppercase();
            }
            if let Ok(h) = obj.get::<_, Value>("headers") {
                headers = parse_headers_init(&ctx, bindings, &h)?;
            }
            if let Ok(b) = obj.get::<_, Value>("body")
                && !b.is_null() && !b.is_undefined() {
                    body = Some(to_dom_string(&ctx, &b)?.into_bytes());
                }
            if let Ok(m) = obj.get::<_, String>("mode") {
                mode = match m.as_str() {
                    "navigate" => RequestMode::Navigate,
                    "same-origin" => RequestMode::SameOrigin,
                    "no-cors" => RequestMode::NoCors,
                    "cors" => RequestMode::Cors,
                    other => {
                        reject_with(
                            ctx,
                            &format!("fetch: invalid mode {other:?}"),
                        )?;
                        return Ok(promise.into_value());
                    }
                };
            }
            if let Ok(c) = obj.get::<_, String>("credentials") {
                credentials = match c.as_str() {
                    "omit" => CredentialsMode::Omit,
                    "same-origin" => CredentialsMode::SameOrigin,
                    "include" => CredentialsMode::Include,
                    other => {
                        reject_with(
                            ctx,
                            &format!("fetch: invalid credentials {other:?}"),
                        )?;
                        return Ok(promise.into_value());
                    }
                };
            }
            if let Ok(s) = obj.get::<_, Value>("signal")
                && !s.is_null() && !s.is_undefined() {
                    match s.as_object().and_then(|o| abort_signal_id_of(&ctx, o)) {
                        Some(id) => signal_id = Some(id),
                        None => {
                            reject_with(ctx, "fetch: signal is not an AbortSignal")?;
                            return Ok(promise.into_value());
                        }
                    }
                }
        }
    // An already-aborted signal rejects at once, without starting
    // the backend (Fetch §4.1: "if signal is aborted, reject").
    if let Some(id) = signal_id
        && bindings
            .borrow()
            .abort_signals
            .get(&id)
            .is_some_and(|s| s.aborted)
    {
        let reason = make_dom_error(&ctx, "AbortError", "The user aborted a request.")?;
        let _: Value = reject.call((reason,))?;
        return Ok(promise.into_value());
    }
    let id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_fetch_id;
        s.next_fetch_id += 1;
        id
    };
    if let Some(signal) = signal_id {
        let mut s = bindings.borrow_mut();
        s.fetch_signals.insert(id, signal);
        if let Some(state) = s.abort_signals.get_mut(&signal) {
            state.fetch_ids.push(id);
        }
    }
    let request = FetchRequest {
        id,
        url,
        method,
        headers,
        body,
        mode,
        credentials,
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
            // The backend owns delivery: it sends `Headers` /
            // `Chunk` / `Done` (or `Error`) on the stream channel,
            // from any thread. If the receiver is gone
            // (navigation), the sends fail silently and the pending
            // entry is dropped with the realm.
            backend.start_stream_fetch(request, stream_tx.clone());
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

/// Parse a `HeadersInit`: a `Headers` object (copied), an object
/// of name → value, or an array of `[name, value]` pairs.
fn parse_headers_init<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    value: &Value<'js>,
) -> rquickjs::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    if let Some(obj) = value.as_object() {
        // A `Headers` object copies its list.
        if let Some(id) = body_id_of(ctx, obj, HEADERS_SYMBOL) {
            return Ok(bindings
                .borrow()
                .response_headers
                .get(&id)
                .cloned()
                .unwrap_or_default());
        }
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
    // The request URL rides along so the `Response` knows its
    // `.url`; the signal registration is dropped whether the
    // fetch succeeded or failed (an aborted fetch never reaches
    // here — `abort()` already removed the entry).
    let url = bindings
        .borrow()
        .pending_fetches
        .get(&id)
        .map(|r| r.url.clone())
        .unwrap_or_default();
    bindings.borrow_mut().pending_fetches.remove(&id);
    // Bind each removal first: a `borrow_mut()` temporary in a
    // `let`-chain scrutinee lives through the whole chain, so a
    // second `borrow_mut()` inside it would panic.
    let signal_id = bindings.borrow_mut().fetch_signals.remove(&id);
    if let Some(signal_id) = signal_id
        && let Some(state) = bindings.borrow_mut().abort_signals.get_mut(&signal_id)
    {
        state.fetch_ids.retain(|f| *f != id);
    }
    let fetches = registry_map(ctx, "fetches")?;
    let entry = map_get(ctx, &fetches, id)?;
    map_delete(ctx, &fetches, id);
    let Some(entry_obj) = entry.as_object() else {
        return Ok(()); // Cancelled by navigation / abort: nothing to resolve.
    };
    match completion.result {
        Ok(response) => {
            let response_obj = make_response(ctx, bindings, &response, &url, false)?;
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

/// Route one chunk-channel event (Stage 16 Phase 3, Rec 12).
/// Called by `TabJs` from a Networking-source Host task, like
/// [`resolve_fetch`]. An `EventSource` fetch is routed to the SSE
/// parser first; anything else feeds a `fetch()` response stream:
/// `Headers` resolves the fetch promise with a `Response` whose
/// `body` is a live stream, `Chunk` appends to the buffer (with
/// the cap) and pushes to the stream, `Done` closes the stream
/// and wakes the body waiters, `Error` errors the stream (and
/// rejects the promise when headers never arrived).
pub fn resolve_stream_event<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
    event: FetchStreamEvent,
) -> rquickjs::Result<()> {
    if resolve_event_source_stream_event(ctx, bindings, stream_tx, &event)? {
        return Ok(());
    }
    let id = event.fetch_id();
    match event {
        FetchStreamEvent::Headers {
            status,
            status_text,
            headers,
            response_type,
            ..
        } => {
            let url = bindings
                .borrow()
                .pending_fetches
                .get(&id)
                .map(|r| r.url.clone())
                .unwrap_or_default();
            let response_obj = make_streaming_response(
                ctx,
                bindings,
                id,
                status,
                &status_text,
                headers,
                response_type,
                &url,
                false,
            )?;
            let fetches = registry_map(ctx, "fetches")?;
            let entry = map_get(ctx, &fetches, id)?;
            // The promise resolves now; the pending entry stays
            // until `Done` / `Error` (the fetch is still live —
            // `has_pending_work` counts it, and `abort()` can
            // still cancel it).
            if let Some(entry_obj) = entry.as_object()
                && let Ok(resolve) = entry_obj.get::<_, Function>("resolve")
            {
                let result: rquickjs::Result<Value> =
                    resolve.call((Value::from(response_obj),));
                if let Err(e) = result {
                    bindings
                        .borrow_mut()
                        .errors
                        .push(format!("fetch resolve threw: {e}"));
                }
                map_delete(ctx, &fetches, id);
            }
        }
        FetchStreamEvent::Chunk { bytes, .. } => {
            let Some(body_id) = bindings.borrow().fetch_bodies.get(&id).copied() else {
                return Ok(()); // Headers never arrived / already gone.
            };
            // The chunk fans out to the source body and every
            // `clone()` tee of it (see `body_tees`). Bind the tree
            // first: a `borrow()` temporary in the `for` iterator
            // lives through the whole loop, and `push_body_chunk`
            // borrows mutably — nesting them panics.
            let bodies = tee_tree(&bindings.borrow(), body_id);
            for bid in bodies {
                push_body_chunk(ctx, bindings, bid, &bytes)?;
            }
        }
        FetchStreamEvent::Done { .. } => {
            finish_stream_fetch(ctx, bindings, id, None)?;
        }
        FetchStreamEvent::Error { error, .. } => {
            // Before `Headers`: the promise rejects (no response
            // object exists yet). After `Headers`: the live stream
            // errors and the body waiters reject.
            let body_id = bindings.borrow().fetch_bodies.get(&id).copied();
            match body_id {
                None => {
                    bindings.borrow_mut().pending_fetches.remove(&id);
                    let fetches = registry_map(ctx, "fetches")?;
                    let entry = map_get(ctx, &fetches, id)?;
                    map_delete(ctx, &fetches, id);
                    if let Some(entry_obj) = entry.as_object()
                        && let Ok(reject) = entry_obj.get::<_, Function>("reject")
                    {
                        let reason: Value =
                            ctx.eval(format!("new TypeError({:?})", error.to_string()))?;
                        let result: rquickjs::Result<Value> = reject.call((reason,));
                        if let Err(e) = result {
                            bindings
                                .borrow_mut()
                                .errors
                                .push(format!("fetch reject threw: {e}"));
                        }
                    }
                }
                Some(body_id) => {
                    // The error finishes the source body and every
                    // tee of it (see `finish_stream_fetch`).
                    finish_stream_fetch(ctx, bindings, id, Some(&error.to_string()))?;
                }
            }
        }
    }
    Ok(())
}

/// A stream fetch finished (`Done` — `error` is `None` — or
/// `Error` after `Headers` — `error` is the message): finish the
/// source body and every `clone()` tee of it, then drop the
/// pending entry.
///
/// `Done`: close each stream, mark each body complete, wake each
/// body's waiters. `Error`: error each stream and reject each
/// body's waiters — the bodies are NOT marked complete, or a later
/// `.text()` would read a partial buffer instead of rejecting.
fn finish_stream_fetch<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_id: u64,
    error: Option<&str>,
) -> rquickjs::Result<()> {
    let body_id = bindings.borrow().fetch_bodies.get(&fetch_id).copied();
    if let Some(body_id) = body_id {
        // Bind the tree first (see the Chunk arm): the `borrow()`
        // temporary in the `for` iterator would live through the
        // loop and panic on the stream calls' `borrow_mut`.
        let bodies = tee_tree(&bindings.borrow(), body_id);
        for bid in bodies {
            // Bind first: a `borrow()` temporary in an `if let`
            // scrutinee lives through the whole body, and the
            // stream calls borrow mutably — nesting them panics.
            let stream_id = bindings.borrow().response_streams.get(&bid).copied();
            match error {
                Some(msg) => {
                    if let Some(stream_id) = stream_id {
                        let _ = streams::stream_error(ctx, bindings, stream_id, msg);
                    }
                    reject_body_waiters(ctx, bindings, bid, msg)?;
                }
                None => {
                    if let Some(stream_id) = stream_id {
                        streams::stream_close(ctx, bindings, stream_id)?;
                    }
                    bindings.borrow_mut().response_complete.insert(bid);
                    resolve_body_waiters(ctx, bindings, bid)?;
                }
            }
        }
        // The tee registrations die with the fetch (transitively —
        // a tee of a tee is in the tree too).
        let bodies = tee_tree(&bindings.borrow(), body_id);
        for bid in bodies {
            bindings.borrow_mut().body_tees.remove(&bid);
        }
    }
    bindings.borrow_mut().pending_fetches.remove(&fetch_id);
    bindings.borrow_mut().fetch_bodies.remove(&fetch_id);
    let signal_id = bindings.borrow_mut().fetch_signals.remove(&fetch_id);
    if let Some(signal_id) = signal_id
        && let Some(state) = bindings.borrow_mut().abort_signals.get_mut(&signal_id)
    {
        state.fetch_ids.retain(|f| *f != fetch_id);
    }
    Ok(())
}

/// The source body id plus every `clone()` tee of it, transitively
/// (a clone of a clone tees the clone — see `body_tees`).
fn tee_tree(bindings: &BindingState, root: u64) -> Vec<u64> {
    let mut out = vec![root];
    let mut i = 0;
    while i < out.len() {
        if let Some(tees) = bindings.body_tees.get(&out[i]) {
            out.extend(tees.iter().copied());
        }
        i += 1;
    }
    out
}

/// Append one chunk to a body's buffer (with the cap) and push it
/// to the body's stream. A body that grows past the cap while
/// buffering errors the stream (the `Response` already resolved;
/// reading its body rejects) — and a body whose stream already
/// errored drops further chunks. Readers are exempt from the cap:
/// the chunk still flows to the stream while it fits.
fn push_body_chunk<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
    bytes: &[u8],
) -> rquickjs::Result<()> {
    let stream_id = bindings.borrow().response_streams.get(&body_id).copied();
    let Some(stream_id) = stream_id else {
        return Ok(());
    };
    if bindings
        .borrow()
        .stream_states
        .get(&stream_id)
        .is_some_and(|s| s.error.is_some())
    {
        return Ok(()); // Already errored: drop further chunks.
    }
    let over_cap = {
        let mut s = bindings.borrow_mut();
        match s.response_bodies.get_mut(&body_id) {
            Some(buf) => {
                buf.extend_from_slice(bytes);
                buf.len() > BODY_BUFFER_CAP
            }
            None => false,
        }
    };
    if over_cap {
        let _ = streams::stream_error(
            ctx,
            bindings,
            stream_id,
            "fetch: response body exceeds the 10 MiB buffer cap",
        );
        reject_body_waiters(
            ctx,
            bindings,
            body_id,
            "fetch: response body exceeds the 10 MiB buffer cap",
        )?;
    } else {
        streams::stream_push_chunk(ctx, bindings, stream_id, bytes.to_vec())?;
    }
    Ok(())
}

/// Build a `Response` for a `Headers` event: an empty body buffer
/// + a live `ReadableStream` (fed by later `Chunk` events, closed
/// by `Done`). The fetch promise resolves with this object.
pub fn make_streaming_response<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_id: u64,
    status: u16,
    status_text: &str,
    headers: Vec<(String, String)>,
    response_type: ResponseType,
    url: &str,
    redirected: bool,
) -> rquickjs::Result<Object<'js>> {
    let body_id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_body_id;
        s.next_body_id += 1;
        s.response_bodies.insert(id, Vec::new());
        s.response_headers.insert(id, headers);
        s.response_urls.insert(id, (url.to_string(), redirected));
        s.fetch_bodies.insert(fetch_id, id);
        id
    };
    let (stream_obj, stream_id) = streams::new_stream(ctx, bindings, Some(fetch_id))?;
    bindings
        .borrow_mut()
        .response_streams
        .insert(body_id, stream_id);
    let status_text = if status_text.is_empty() {
        default_status_text(status).to_string()
    } else {
        status_text.to_string()
    };
    response_object(
        ctx,
        bindings,
        body_id,
        status,
        &status_text,
        response_type,
        url,
        redirected,
        Some(stream_obj),
    )
}

/// Wake every body waiter for `body_id` with the finished body
/// (the value computed by kind — `text` / `json` / `blob` /
/// `arrayBuffer` — from the buffer, exactly as the direct path
/// computes it).
fn resolve_body_waiters<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
) -> rquickjs::Result<()> {
    for (resolve, reject, kind) in take_body_waiters(ctx, body_id)? {
        match body_value(ctx, bindings, body_id, &kind) {
            Ok(value) => {
                let _: Value = resolve.call((value,))?;
            }
            Err(reason) => {
                let _: Value = reject.call((reason,))?;
            }
        }
    }
    Ok(())
}

/// Reject every body waiter for `body_id` with a `TypeError`.
fn reject_body_waiters<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
    msg: &str,
) -> rquickjs::Result<()> {
    for (_resolve, reject, _kind) in take_body_waiters(ctx, body_id)? {
        let reason: Value = ctx.eval(format!("new TypeError({msg:?})"))?;
        let _: Value = reject.call((reason,))?;
    }
    Ok(())
}

/// Take (and clear) the body waiters for `body_id` from the
/// registry's `bodyWaiters` Map. Each record is
/// `{ resolve, reject, kind }` with `kind` one of `"text"` /
/// `"json"` / `"blob"` / `"arrayBuffer"`.
fn take_body_waiters<'js>(
    ctx: &Ctx<'js>,
    body_id: u64,
) -> rquickjs::Result<Vec<(Function<'js>, Function<'js>, String)>> {
    let map = registry_map(ctx, "bodyWaiters")?;
    let value = map_get(ctx, &map, body_id)?;
    map_delete(ctx, &map, body_id);
    let Some(arr) = value.as_object().and_then(|o| o.clone().into_array()) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for i in 0..arr.len() {
        let Ok(record) = arr.get::<Value>(i) else {
            continue;
        };
        let Some(obj) = record.as_object() else {
            continue;
        };
        let Ok(resolve) = obj.get::<_, Function>("resolve") else {
            continue;
        };
        let Ok(reject) = obj.get::<_, Function>("reject") else {
            continue;
        };
        let kind = obj.get::<_, String>("kind").unwrap_or_default();
        out.push((resolve, reject, kind));
    }
    Ok(out)
}

/// The body value for a waiter kind, from the finished buffer —
/// the same computation the direct (already-complete) path does
/// in the `Response` prototype methods below. `Err` is the
/// rejection reason (a `SyntaxError` value for a bad `json()`).
fn body_value<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
    kind: &str,
) -> Result<Value<'js>, Value<'js>> {
    let bytes = bindings
        .borrow()
        .response_bodies
        .get(&body_id)
        .cloned()
        .unwrap_or_default();
    match kind {
        "json" => {
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let json: Object = ctx.globals().get("JSON").map_err(|_| js_null(ctx))?;
            let parse: Function = json.get("parse").map_err(|_| js_null(ctx))?;
            match parse.call::<_, Value>((text,)) {
                Ok(value) => Ok(value),
                Err(_) => {
                    let caught = ctx.catch();
                    let msg = caught
                        .as_object()
                        .and_then(|o| o.get::<_, String>("message").ok())
                        .unwrap_or_else(|| "JSON parse error".to_string());
                    match ctx.eval::<Value, _>(format!("new SyntaxError({msg:?})")) {
                        Ok(value) => Err(value),
                        Err(_) => Err(js_null(ctx)),
                    }
                }
            }
        }
        "blob" => {
            let ty = bindings
                .borrow()
                .response_headers
                .get(&body_id)
                .and_then(|h| {
                    h.iter()
                        .find(|(n, _)| n.eq_ignore_ascii_case("content-type"))
                        .map(|(_, v)| v.clone())
                })
                .unwrap_or_default();
            let blob = Object::new(ctx.clone()).map_err(|_| js_null(ctx))?;
            blob.set("size", bytes.len() as f64).map_err(|_| js_null(ctx))?;
            blob.set("type", ty).map_err(|_| js_null(ctx))?;
            Ok(Value::from(blob))
        }
        "arrayBuffer" => {
            let buffer = ArrayBuffer::new_copy(ctx.clone(), bytes.as_slice())
                .map_err(|_| js_null(ctx))?;
            buffer.into_js(ctx).map_err(|_| js_null(ctx))
        }
        _ => Ok(js(ctx, String::from_utf8_lossy(&bytes).into_owned())),
    }
}

/// Register a body waiter: the caller (`.text()` etc.) created a
/// promise whose resolve/reject wait for the stream to finish.
/// Returns `true` when the waiter was registered (the caller
/// returns the pending promise); `false` when the body is already
/// complete (the caller reads the buffer directly).
fn wait_for_body<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
    kind: &str,
    resolve: &Function<'js>,
    reject: &Function<'js>,
) -> rquickjs::Result<bool> {
    if bindings.borrow().response_complete.contains(&body_id) {
        return Ok(false);
    }
    let map = registry_map(ctx, "bodyWaiters")?;
    let value = map_get(ctx, &map, body_id)?;
    let arr = match value.as_object().and_then(|o| o.clone().into_array()) {
        Some(arr) => arr,
        None => Array::new(ctx.clone())?,
    };
    let record = Object::new(ctx.clone())?;
    record.set("resolve", resolve.clone())?;
    record.set("reject", reject.clone())?;
    record.set("kind", kind)?;
    arr.set(arr.len(), Value::from(record))?;
    map_set(ctx, &map, body_id, Value::from(arr))?;
    Ok(true)
}

/// Abort every fetch registered on `signal_id`: cancel the
/// backend request, reject the promise with an `AbortError`, and
/// drop the pending entry — so a late backend completion finds
/// no entry and is ignored. Then run the signal's `onabort`
/// handler, if one is set. A second `abort()` is a no-op.
pub fn abort_signal<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    signal_id: u64,
) -> rquickjs::Result<()> {
    let already_aborted = bindings
        .borrow()
        .abort_signals
        .get(&signal_id)
        .is_some_and(|s| s.aborted);
    if already_aborted {
        return Ok(());
    }
    let fetch_ids = {
        let mut s = bindings.borrow_mut();
        if let Some(state) = s.abort_signals.get_mut(&signal_id) {
            state.aborted = true;
            std::mem::take(&mut state.fetch_ids)
        } else {
            return Ok(());
        }
    };
    let backend = bindings.borrow().fetch_backend.clone();
    for fetch_id in fetch_ids {
        if let Some(backend) = &backend {
            backend.cancel_fetch(fetch_id);
        }
        // A fetch that already resolved at headers has a live
        // stream: error it (and every `clone()` tee of it) and
        // reject the body waiters (the promise is gone — it
        // resolved already — so the abort surfaces through the
        // stream / `.text()` instead).
        // Bind first: a `borrow()` temporary in an `if let`
        // scrutinee lives through the whole body, and the error /
        // waiter calls borrow mutably — nesting them panics.
        let body_id = bindings.borrow().fetch_bodies.get(&fetch_id).copied();
        if let Some(body_id) = body_id {
            // Bind the tree first (see the Chunk arm): the
            // `borrow()` temporary in the `for` iterator would live
            // through the loop and panic on the `borrow_mut`s.
            let bodies = tee_tree(&bindings.borrow(), body_id);
            for bid in bodies {
                let stream_id = bindings.borrow().response_streams.get(&bid).copied();
                if let Some(stream_id) = stream_id {
                    let _ = streams::stream_error(
                        ctx,
                        bindings,
                        stream_id,
                        "The user aborted a request.",
                    );
                }
                let _ = reject_body_waiters(
                    ctx,
                    bindings,
                    bid,
                    "The user aborted a request.",
                );
                bindings.borrow_mut().body_tees.remove(&bid);
            }
            bindings.borrow_mut().fetch_bodies.remove(&fetch_id);
        }
        bindings.borrow_mut().pending_fetches.remove(&fetch_id);
        bindings.borrow_mut().fetch_signals.remove(&fetch_id);
        let fetches = registry_map(ctx, "fetches")?;
        let entry = map_get(ctx, &fetches, fetch_id)?;
        map_delete(ctx, &fetches, fetch_id);
        if let Some(entry_obj) = entry.as_object()
            && let Ok(reject_fn) = entry_obj.get::<_, Function>("reject")
        {
            let reason = make_dom_error(&ctx, "AbortError", "The user aborted a request.")?;
            let result: rquickjs::Result<Value> = reject_fn.call((reason,));
            if let Err(e) = result {
                bindings
                    .borrow_mut()
                    .errors
                    .push(format!("fetch abort reject threw: {e}"));
            }
        }
    }
    // `onabort`: called with the signal as `this` and an `abort` event.
    let signals = registry_map(ctx, "abortSignals")?;
    let signal = map_get(ctx, &signals, signal_id)?;
    if let Some(signal_obj) = signal.as_object()
        && let Ok(handler) = signal_obj.get::<_, Value>("onabort")
        && handler.is_function()
    {
        let func = Function::from_js(ctx, handler)?;
        let event: Object = ctx.eval("new Event('abort')")?;
        let _: Value = func.call((
            This(Value::from(signal_obj.clone())),
            Value::from(event),
        ))?;
    }
    Ok(())
}

/// Build a `Response` wrapper for a completed fetch, storing its
/// body + headers in [`BindingState`] under a fresh body id. The
/// `body` stream is a closed pre-loaded stream of the buffered
/// bytes (a `Response` built from bytes has a body stream that is
/// already complete), and the body is marked complete — so
/// `.text()` etc. read the buffer directly.
///
/// `live` is `true` only for `clone()` of a still-streaming
/// response (see `body_tees`): the stream stays open (later
/// chunks fan out to it) and the body is NOT marked complete, so
/// `.text()` waits for `Done` instead of reading the prefix.
pub fn make_response<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    response: &FetchResponse,
    url: &str,
    redirected: bool,
) -> rquickjs::Result<Object<'js>> {
    make_response_live(ctx, bindings, response, url, redirected, false)
}

/// [`make_response`] with the stream left live (see above).
pub fn make_response_live<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    response: &FetchResponse,
    url: &str,
    redirected: bool,
    live: bool,
) -> rquickjs::Result<Object<'js>> {
    let body_id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_body_id;
        s.next_body_id += 1;
        s.response_bodies.insert(id, response.body.clone());
        s.response_headers.insert(id, response.headers.clone());
        s.response_urls
            .insert(id, (url.to_string(), redirected));
        id
    };
    let status_text = if response.status_text.is_empty() {
        default_status_text(response.status).to_string()
    } else {
        response.status_text.clone()
    };
    let (stream_obj, stream_id) = streams::new_stream(ctx, bindings, None)?;
    if !response.body.is_empty() {
        streams::stream_push_chunk(ctx, bindings, stream_id, response.body.clone())?;
    }
    if live {
        // The stream stays open for fanned-out chunks; the body is
        // not complete, so `.text()` etc. wait for `Done`.
    } else {
        streams::stream_close(ctx, bindings, stream_id)?;
        bindings.borrow_mut().response_complete.insert(body_id);
    }
    bindings
        .borrow_mut()
        .response_streams
        .insert(body_id, stream_id);
    response_object(
        ctx,
        bindings,
        body_id,
        response.status,
        &status_text,
        response.response_type,
        url,
        redirected,
        Some(stream_obj),
    )
}

/// The shared `Response` object builder: the body-id symbol, the
/// status / type / url properties, the `headers` object, and the
/// `body` stream (when one was built — every `Response` has one;
/// a streamed response's is live, a buffered response's is a
/// closed pre-loaded stream).
fn response_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    body_id: u64,
    status: u16,
    status_text: &str,
    response_type: ResponseType,
    url: &str,
    redirected: bool,
    stream: Option<Object<'js>>,
) -> rquickjs::Result<Object<'js>> {
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("response")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), RESPONSE_SYMBOL)?;
    obj.set(sym.as_atom(), body_id as f64)?;
    // Status lives on the object (plain data properties; the
    // prototype's accessors read them).
    obj.prop("status", Property::from(status as f64))?;
    obj.prop("statusText", Property::from(status_text))?;
    // `type` likewise: the backend-set tainting (`basic` / `cors` /
    // `opaque` / `opaqueredirect` / `default`), read by the
    // prototype's `type` accessor below.
    obj.prop("type", Property::from(response_type.as_str()))?;
    // `url` / `redirected` likewise (Fetch §2.2.6): the final URL
    // after redirects, and whether any redirect happened.
    obj.prop("url", Property::from(url))?;
    obj.prop("redirected", Property::from(redirected))?;
    // The `headers` object shares the body id.
    let headers_obj = make_headers_object(ctx, bindings, body_id)?;
    obj.set("headers", headers_obj)?;
    // `body`: the stream object itself (an own data property — it
    // shadows nothing, and the prototype has no `body` accessor).
    if let Some(stream) = stream {
        obj.set("body", stream)?;
    }
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

/// The `Response` / `Headers` object id helpers + `ResponseType`
/// parsing for `Response.clone()`.
fn response_type_from_str(s: &str) -> ResponseType {
    match s {
        "basic" => ResponseType::Basic,
        "cors" => ResponseType::Cors,
        "opaque" => ResponseType::Opaque,
        "opaqueredirect" => ResponseType::OpaqueRedirect,
        _ => ResponseType::Default,
    }
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

    // `type` is a plain data property on the object (set by
    // `make_response`); the prototype exposes it as an accessor
    // that reads the own property, so a missing value reads as
    // `undefined` instead of throwing.
    proto
        .prop(
            "type",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                    let _ = ctx;
                    this.get("type")
                },
            ),
        )
        .map_err(|e| format!("Response.type: {e}"))?;

    // `text()`: a promise resolved with the body as a string.
    // When the stream has not finished yet, the promise waits
    // (via `bodyWaiters`) for `Done` instead of reading a
    // half-filled buffer.
    let b = Rc::clone(bindings);
    let text_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, reject) = ctx.promise()?;
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let Some(body_id) = body_id else {
                let _: Value = resolve.call((String::new(),))?;
                return Ok(promise.into_value());
            };
            if wait_for_body(&ctx, &b, body_id, "text", &resolve, &reject)? {
                return Ok(promise.into_value());
            }
            let text = b
                .borrow()
                .response_bodies
                .get(&body_id)
                .cloned()
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
            let Some(body_id) = body_id else {
                let _: Value = resolve.call((js_null(&ctx),))?;
                return Ok(promise.into_value());
            };
            if wait_for_body(&ctx, &b, body_id, "json", &resolve, &reject)? {
                return Ok(promise.into_value());
            }
            let text = b
                .borrow()
                .response_bodies
                .get(&body_id)
                .cloned()
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

    // `url` / `redirected` are plain data properties on the object
    // (set by `make_response`); the prototype exposes them as
    // accessors that read the own property, so a missing value
    // reads as `undefined` instead of throwing.
    for prop in ["url", "redirected"] {
        proto
            .prop(
                prop,
                Accessor::from(
                    move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                        let _ = ctx;
                        this.get(prop)
                    },
                ),
            )
            .map_err(|e| format!("Response.{prop}: {e}"))?;
    }

    // `blob()`: a promise resolved with a `Blob`-shaped object
    // (`{ size, type }` — M4 has no `Blob` class yet; the size and
    // the content type are what pages read most). Waits for the
    // stream to finish, like `text()`.
    let b = Rc::clone(bindings);
    let blob_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, reject) = ctx.promise()?;
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let Some(body_id) = body_id else {
                let blob = Object::new(ctx.clone())?;
                blob.set("size", 0.0)?;
                blob.set("type", String::new())?;
                let _: Value = resolve.call((Value::from(blob),))?;
                return Ok(promise.into_value());
            };
            if wait_for_body(&ctx, &b, body_id, "blob", &resolve, &reject)? {
                return Ok(promise.into_value());
            }
            let (size, ty) = {
                let s = b.borrow();
                let size = s.response_bodies.get(&body_id).map(|v| v.len()).unwrap_or(0);
                let ty = s
                    .response_headers
                    .get(&body_id)
                    .and_then(|h| {
                        h.iter()
                            .find(|(n, _)| n.eq_ignore_ascii_case("content-type"))
                            .map(|(_, v)| v.clone())
                    })
                    .unwrap_or_default();
                (size, ty)
            };
            let blob = Object::new(ctx.clone())?;
            blob.set("size", size as f64)?;
            blob.set("type", ty)?;
            let _: Value = resolve.call((Value::from(blob),))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("Response.blob: {e}"))?;
    proto
        .set("blob", blob_fn)
        .map_err(|e| format!("Response.blob: {e}"))?;

    // `arrayBuffer()`: a promise resolved with the body bytes in a
    // real `ArrayBuffer`. Waits for the stream to finish, like
    // `text()`.
    let b = Rc::clone(bindings);
    let array_buffer_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, reject) = ctx.promise()?;
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let Some(body_id) = body_id else {
                let buffer = ArrayBuffer::new_copy(ctx.clone(), [0u8; 0])?;
                let buffer_value: Value = buffer.into_js(&ctx)?;
                let _: Value = resolve.call((buffer_value,))?;
                return Ok(promise.into_value());
            };
            if wait_for_body(&ctx, &b, body_id, "arrayBuffer", &resolve, &reject)? {
                return Ok(promise.into_value());
            }
            let bytes = b
                .borrow()
                .response_bodies
                .get(&body_id)
                .cloned()
                .unwrap_or_default();
            let buffer = ArrayBuffer::new_copy(ctx.clone(), bytes.as_slice())?;
            let buffer_value: Value = buffer.into_js(&ctx)?;
            let _: Value = resolve.call((buffer_value,))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("Response.arrayBuffer: {e}"))?;
    proto
        .set("arrayBuffer", array_buffer_fn)
        .map_err(|e| format!("Response.arrayBuffer: {e}"))?;

    // `clone()`: a new `Response` with an independent copy of the
    // body + headers (reading one clone never disturbs the other).
    // When the source body is still streaming, the clone tees it
    // (see `body_tees`): the clone starts with the bytes buffered
    // so far, and later chunks fan out to both — a clone made
    // immediately after `fetch()` resolves still reads the full
    // body. A clone of a finished body is just the buffered copy.
    let b = Rc::clone(bindings);
    let clone_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let body_id = body_id_of(&ctx, &this, RESPONSE_SYMBOL);
            let (status, status_text, ty, headers, body, url, redirected) = match body_id {
                Some(id) => {
                    let s = b.borrow();
                    let status = this.get::<_, f64>("status").unwrap_or(200.0) as u16;
                    let status_text = this
                        .get::<_, String>("statusText")
                        .unwrap_or_default();
                    let ty = response_type_from_str(
                        &this.get::<_, String>("type").unwrap_or_default(),
                    );
                    let headers = s.response_headers.get(&id).cloned().unwrap_or_default();
                    let body = s.response_bodies.get(&id).cloned().unwrap_or_default();
                    let (url, redirected) = s
                        .response_urls
                        .get(&id)
                        .cloned()
                        .unwrap_or_default();
                    (status, status_text, ty, headers, body, url, redirected)
                }
                None => (
                    200u16,
                    String::new(),
                    ResponseType::Default,
                    Vec::new(),
                    Vec::new(),
                    String::new(),
                    false,
                ),
            };
            let response = FetchResponse {
                status,
                status_text,
                headers,
                body,
                response_type: ty,
            };
            // A clone of a still-streaming body gets a LIVE stream
            // (see `make_response_live`) and registers as a tee of
            // the source, so later chunks fan out to it.
            let live = body_id
                .is_some_and(|id| !b.borrow().response_complete.contains(&id));
            let clone_obj =
                make_response_live(&ctx, &b, &response, &url, redirected, live)?;
            if live {
                if let (Some(source_id), Some(clone_id)) = (
                    body_id,
                    body_id_of(&ctx, &clone_obj, RESPONSE_SYMBOL),
                ) {
                    b.borrow_mut()
                        .body_tees
                        .entry(source_id)
                        .or_default()
                        .push(clone_id);
                }
            }
            Ok(Value::from(clone_obj))
        },
    )
    .map_err(|e| format!("Response.clone: {e}"))?;
    proto
        .set("clone", clone_fn)
        .map_err(|e| format!("Response.clone: {e}"))?;

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
                        headers = parse_headers_init(&ctx, &b, &h)?;
                    }
                }
            let response = FetchResponse {
                status,
                status_text,
                headers,
                body: bytes,
                // A constructed response is not a network response:
                // its type is always "default" (Fetch §2.2.6).
                response_type: ResponseType::Default,
            };
            make_response(&ctx, &b, &response, "", false).map(Value::from)
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
            // `get` returns the values for the name combined with
            // `", "` (Fetch §5.2.6) — a single value reads unchanged.
            let values: Vec<&str> = headers
                .iter()
                .filter(|(n, _)| n.to_ascii_lowercase() == want)
                .map(|(_, v)| v.as_str())
                .collect();
            Ok(if values.is_empty() {
                js_null(&ctx)
            } else {
                js(&ctx, values.join(", "))
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
            // `set` replaces *all* values for the name with one.
            headers.retain(|(n, _)| n.to_ascii_lowercase() != want);
            headers.push((name.to_string(), value.to_string()));
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

    // `delete(name)`: drop every value for the name.
    let b = Rc::clone(bindings);
    let delete_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Err(throw_type_error(&ctx, "called on a non-Headers object"));
            };
            let mut headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            let want = name.to_ascii_lowercase();
            headers.retain(|(n, _)| n.to_ascii_lowercase() != want);
            b.borrow_mut().response_headers.insert(id, headers);
            Ok(())
        },
    )
    .map_err(|e| format!("Headers.delete: {e}"))?;
    proto
        .set("delete", delete_fn)
        .map_err(|e| format!("Headers.delete: {e}"))?;

    // Iteration (`keys` / `values` / `entries` / `forEach` /
    // `for..of`): names in first-appearance insertion order, with
    // duplicate values combined with `", "` (Fetch §5.2.6 — the
    // list is sorted by name in the spec; M4 keeps insertion
    // order, which is what the report's test asserts).
    for (name, which) in [("keys", 0u8), ("values", 1u8), ("entries", 2u8)] {
        let b = Rc::clone(bindings);
        let f = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                    return Ok(Value::from(Array::new(ctx.clone())?));
                };
                let headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
                let arr = Array::new(ctx.clone())?;
                for (i, (n, v)) in normalized_headers(&headers).into_iter().enumerate() {
                    let item = match which {
                        0 => Value::from(js(&ctx, n)),
                        1 => Value::from(js(&ctx, v)),
                        _ => {
                            let pair = Array::new(ctx.clone())?;
                            pair.set(0, Value::from(js(&ctx, n)))?;
                            pair.set(1, Value::from(js(&ctx, v)))?;
                            Value::from(pair)
                        }
                    };
                    arr.set(i, item)?;
                }
                Ok(Value::from(arr))
            },
        )
        .map_err(|e| format!("Headers.{name}: {e}"))?;
        proto.set(name, f).map_err(|e| format!("Headers.{name}: {e}"))?;
    }

    // `forEach(callback, thisArg?)`: `callback(value, key, headers)`.
    let b = Rc::clone(bindings);
    let for_each_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, callback: Function<'js>, this_arg: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let Some(id) = body_id_of(&ctx, &this, HEADERS_SYMBOL) else {
                return Ok(());
            };
            let headers = b.borrow().response_headers.get(&id).cloned().unwrap_or_default();
            let this_value = this_arg.0.unwrap_or_else(|| Value::from(this.clone()));
            for (n, v) in normalized_headers(&headers) {
                let _: Value = callback.call((
                    This(this_value.clone()),
                    js(&ctx, v),
                    js(&ctx, n),
                    Value::from(this.clone()),
                ))?;
            }
            Ok(())
        },
    )
    .map_err(|e| format!("Headers.forEach: {e}"))?;
    proto
        .set("forEach", for_each_fn)
        .map_err(|e| format!("Headers.forEach: {e}"))?;

    // `for (const [k, v] of headers)`: the default iterator is
    // `entries` (via a real `{ next() }` iterator — see
    // `install_entries_iterator`).
    super::install_entries_iterator(&ctx, &proto)
        .map_err(|e| format!("Headers iterator: {e}"))?;

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
                Some(v) => parse_headers_init(&ctx, &b, &v)?,
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

/// Collapse a header list for iteration: one entry per name
/// (lowercased), in first-appearance order, values combined with
/// `", "`.
fn normalized_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        match out.iter_mut().find(|(n, _)| *n == lower) {
            Some((_, v)) => {
                v.push_str(", ");
                v.push_str(value);
            }
            None => out.push((lower, value.clone())),
        }
    }
    out
}

/// The `Request` id stored on an object, if it is one.
fn request_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), REQUEST_SYMBOL).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

/// The `AbortSignal` id stored on an object, if it is one.
fn abort_signal_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), ABORT_SIGNAL_SYMBOL).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

/// Build a `Request` wrapper for a stored request id.
fn make_request_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    id: u64,
) -> rquickjs::Result<Object<'js>> {
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("request")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), REQUEST_SYMBOL)?;
    obj.set(sym.as_atom(), id as f64)?;
    let _ = bindings;
    Ok(obj)
}

/// Install the `Request` prototype + constructor (Stage 12 Rec 8).
/// `new Request(input, init?)`: `input` is a URL string or another
/// `Request` (cloned); `init` overrides `method` / `headers` /
/// `body` / `signal`. A `Request` is usable as `fetch()` input —
/// see `fetch_impl`.
fn install_request_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let proto = Object::new(ctx.clone()).map_err(|e| format!("Request prototype: {e}"))?;

    // `url` / `method`: read the stored request.
    for prop in ["url", "method"] {
        let b = Rc::clone(bindings);
        let get = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = request_id_of(&ctx, &this) else {
                return Ok(js(&ctx, ""));
            };
            Ok(js(
                &ctx,
                b.borrow()
                    .requests
                    .get(&id)
                    .map(|r| {
                        if prop == "url" {
                            r.url.clone()
                        } else {
                            r.method.clone()
                        }
                    })
                    .unwrap_or_default(),
            ))
        };
        proto
            .prop(prop, Accessor::from(get))
            .map_err(|e| format!("Request.{prop}: {e}"))?;
    }

    // `headers`: the `Headers` object sharing the stored list
    // (cached per request, so `req.headers === req.headers`).
    let b = Rc::clone(bindings);
    let get_headers = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(id) = request_id_of(&ctx, &this) else {
            return Err(throw_type_error(&ctx, "called on a non-Request object"));
        };
        let headers_id = b
            .borrow()
            .requests
            .get(&id)
            .map(|r| r.headers_id)
            .unwrap_or(0);
        let map = registry_map(&ctx, "headers")?;
        let existing = map_get(&ctx, &map, headers_id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone().into_value());
        }
        let headers_obj = make_headers_object(&ctx, &b, headers_id)?;
        map_set(&ctx, &map, headers_id, Value::from(headers_obj.clone()))?;
        Ok(Value::from(headers_obj))
    };
    proto
        .prop("headers", Accessor::from(get_headers))
        .map_err(|e| format!("Request.headers: {e}"))?;

    // `signal`: the attached `AbortSignal` object (`null` when none).
    let b = Rc::clone(bindings);
    let get_signal = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(id) = request_id_of(&ctx, &this) else {
            return Ok(js_null(&ctx));
        };
        let signal_id = b.borrow().requests.get(&id).and_then(|r| r.signal_id);
        match signal_id {
            Some(signal_id) => {
                let signals = registry_map(&ctx, "abortSignals")?;
                map_get(&ctx, &signals, signal_id)
            }
            None => Ok(js_null(&ctx)),
        }
    };
    proto
        .prop("signal", Accessor::from(get_signal))
        .map_err(|e| format!("Request.signal: {e}"))?;

    protos
        .set("request", proto)
        .map_err(|e| format!("prototypes request: {e}"))?;

    // `Request` constructor: `new Request(input, init?)` — same JS
    // constructor + Rust factory shape as `Response` (a Rust
    // `Function::new` is not constructable).
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, input: Value<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let mut stored = if let Some(s) = input.as_string() {
                StoredRequest {
                    url: s.to_string().unwrap_or_default(),
                    method: "GET".to_string(),
                    headers_id: 0,
                    body: None,
                    signal_id: None,
                }
            } else if let Some(obj) = input.as_object()
                && let Some(req_id) = request_id_of(&ctx, obj)
            {
                match b.borrow().requests.get(&req_id).cloned() {
                    Some(s) => s,
                    None => {
                        return Err(throw_type_error(&ctx, "Request: invalid Request"));
                    }
                }
            } else {
                return Err(throw_type_error(
                    &ctx,
                    "Request: input is not a URL string or Request",
                ));
            };
            // A fresh headers list for this request (a clone's list
            // starts as a copy of the source's).
            let headers_id = {
                let mut s = b.borrow_mut();
                let id = s.next_body_id;
                s.next_body_id += 1;
                let headers = if stored.headers_id == 0 {
                    Vec::new()
                } else {
                    s.response_headers
                        .get(&stored.headers_id)
                        .cloned()
                        .unwrap_or_default()
                };
                s.response_headers.insert(id, headers);
                s.response_bodies.insert(id, Vec::new());
                id
            };
            stored.headers_id = headers_id;
            if let Some(init) = init.0
                && let Some(obj) = init.as_object()
            {
                if let Ok(m) = obj.get::<_, String>("method") {
                    stored.method = m.to_ascii_uppercase();
                }
                if let Ok(h) = obj.get::<_, Value>("headers") {
                    let headers = parse_headers_init(&ctx, &b, &h)?;
                    b.borrow_mut().response_headers.insert(headers_id, headers);
                }
                if let Ok(body) = obj.get::<_, Value>("body")
                    && !body.is_null() && !body.is_undefined()
                {
                    stored.body = Some(to_dom_string(&ctx, &body)?.into_bytes());
                }
                if let Ok(s) = obj.get::<_, Value>("signal")
                    && !s.is_null() && !s.is_undefined()
                {
                    match s.as_object().and_then(|o| abort_signal_id_of(&ctx, o)) {
                        Some(id) => stored.signal_id = Some(id),
                        None => {
                            return Err(throw_type_error(
                                &ctx,
                                "Request: signal is not an AbortSignal",
                            ));
                        }
                    }
                }
            }
            if !(stored.url.starts_with("http://")
                || stored.url.starts_with("https://")
                || stored.url.starts_with('/'))
            {
                return Err(throw_type_error(&ctx, "Request: invalid URL"));
            }
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_request_id;
                s.next_request_id += 1;
                s.requests.insert(id, stored);
                id
            };
            make_request_object(&ctx, &b, id).map(Value::from)
        },
    )
    .map_err(|e| format!("Request factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("requestFactory", factory)
        .map_err(|e| format!("install Request factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.Request = function Request(input, init) { \
           if (!(this instanceof Request)) throw new TypeError('Request constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].requestFactory(input, init); \
         };",
    )
    .map_err(|e| format!("install Request: {e}"))?;
    let globals = ctx.globals();
    let request_ctor: Object = globals.get("Request").map_err(|e| e.to_string())?;
    request_ctor
        .set("prototype", protos.get::<_, Object>("request").map_err(|e| e.to_string())?)
        .map_err(|e| format!("Request.prototype: {e}"))?;
    Ok(())
}

/// Install `AbortController` / `AbortSignal` (Stage 12 Rec 8).
/// `AbortSignal` objects live in the registry's `abortSignals`
/// Map (by id) so `Request.signal` returns the same object;
/// `abort()` marks the state, cancels + rejects the registered
/// fetches (see [`abort_signal`]), and runs `onabort`.
fn install_abort_prototypes<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let signal_proto = Object::new(ctx.clone()).map_err(|e| format!("AbortSignal prototype: {e}"))?;

    // `aborted`: read the signal state.
    let b = Rc::clone(bindings);
    let get_aborted = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
        let Some(id) = abort_signal_id_of(&ctx, &this) else {
            return Ok(false);
        };
        Ok(b.borrow()
            .abort_signals
            .get(&id)
            .is_some_and(|s| s.aborted))
    };
    signal_proto
        .prop("aborted", Accessor::from(get_aborted))
        .map_err(|e| format!("AbortSignal.aborted: {e}"))?;

    // `throwIfAborted()`: throw an `AbortError` when aborted.
    let b = Rc::clone(bindings);
    let throw_if_aborted = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let Some(id) = abort_signal_id_of(&ctx, &this) else {
                return Ok(());
            };
            if b.borrow()
                .abort_signals
                .get(&id)
                .is_some_and(|s| s.aborted)
            {
                let err = make_dom_error(&ctx, "AbortError", "The user aborted a request.")?;
                return Err(ctx.throw(err));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("AbortSignal.throwIfAborted: {e}"))?;
    signal_proto
        .set("throwIfAborted", throw_if_aborted)
        .map_err(|e| format!("AbortSignal.throwIfAborted: {e}"))?;

    protos
        .set("abortSignal", signal_proto)
        .map_err(|e| format!("prototypes abortSignal: {e}"))?;

    // `AbortController` constructor: `new AbortController()` — a
    // plain object with `.signal` (a fresh `AbortSignal`) and an
    // `.abort()` method. The controller object carries the signal
    // id under the same symbol, so `abort()` finds its signal.
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            let signal_id = {
                let mut s = b.borrow_mut();
                let id = s.next_signal_id;
                s.next_signal_id += 1;
                s.abort_signals.insert(id, AbortSignalState::default());
                id
            };
            let protos = registry_map(&ctx, "prototypes")?;
            let signal_proto: Object = protos.get("abortSignal")?;
            let signal = Object::new_proto(ctx.clone(), Some(&signal_proto))?;
            let sym = Symbol::new_global(ctx.clone(), ABORT_SIGNAL_SYMBOL)?;
            signal.set(sym.as_atom(), signal_id as f64)?;
            // `onabort` starts as `null` (an IDL event-handler attribute).
            signal.set("onabort", js_null(&ctx))?;
            let signals = registry_map(&ctx, "abortSignals")?;
            map_set(&ctx, &signals, signal_id, Value::from(signal.clone()))?;

            let controller = Object::new(ctx.clone())?;
            controller.set("signal", Value::from(signal))?;
            controller.set(sym.as_atom(), signal_id as f64)?;
            let b2 = Rc::clone(&b);
            let abort_fn = Function::new(
                ctx.clone(),
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
                    let Some(id) = abort_signal_id_of(&ctx, &this) else {
                        return Ok(());
                    };
                    abort_signal(&ctx, &b2, id)
                },
            )?;
            controller.set("abort", abort_fn)?;
            Ok(Value::from(controller))
        },
    )
    .map_err(|e| format!("AbortController factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("abortControllerFactory", factory)
        .map_err(|e| format!("install AbortController factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.AbortController = function AbortController() { \
           if (!(this instanceof AbortController)) throw new TypeError('AbortController constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].abortControllerFactory(); \
         };",
    )
    .map_err(|e| format!("install AbortController: {e}"))?;
    let globals = ctx.globals();
    let controller_ctor: Object = globals
        .get("AbortController")
        .map_err(|e| e.to_string())?;
    // `AbortController.prototype` is a plain object (only `abort`
    // lives on the instance); `AbortSignal` is exposed as a
    // constructor-shaped object whose prototype is the signal
    // prototype, so `signal instanceof AbortSignal` holds — but
    // `new AbortSignal()` itself throws (spec: no public constructor).
    let controller_proto = Object::new(ctx.clone())
        .map_err(|e| format!("AbortController prototype: {e}"))?;
    controller_ctor
        .set("prototype", controller_proto)
        .map_err(|e| format!("AbortController.prototype: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.AbortSignal = function AbortSignal() { \
           throw new TypeError('AbortSignal constructor requires new'); \
         };",
    )
    .map_err(|e| format!("install AbortSignal: {e}"))?;
    let signal_ctor: Object = globals.get("AbortSignal").map_err(|e| e.to_string())?;
    signal_ctor
        .set(
            "prototype",
            protos.get::<_, Object>("abortSignal").map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("AbortSignal.prototype: {e}"))?;
    Ok(())
}
