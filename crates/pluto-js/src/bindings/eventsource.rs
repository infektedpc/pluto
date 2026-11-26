//! `EventSource` + the SSE parser (Stage 16 Phase 3, Rec 12).
//!
//! An `EventSource` is a long-lived `text/event-stream` fetch
//! driven by the same chunk channel as `fetch()` streaming: the
//! backend's `start_stream_fetch` delivers `FetchStreamEvent`s,
//! `TabJs` routes them here by fetch id (see
//! [`BindingState::event_source_fetches`]), and the parser below
//! turns bytes into events. `message` / `error` / `open` dispatch
//! through the normal `EventTarget` machinery (the object is a
//! standalone target, like `WebSocket`).
//!
//! Reconnect: a failed or server-closed stream fires `error` and
//! reconnects after the `retry:` value (default 3000 ms) via the
//! realm's own `setTimeout` — the timer is a normal loop timer,
//! so navigation clears it with everything else. `close()` sets
//! `CLOSED` and cancels the in-flight fetch; no reconnect follows.
//! A `Cancelled` stream error (navigation / `close()`) never
//! reconnects. `Last-Event-ID` is sent on reconnect when the
//! server set an id.
//!
//! The fetch is CORS-mode with `Accept: text/event-stream`. A
//! non-200 status or a non-`text/event-stream` content type fails
//! the connection (spec: the event stream is only established for
//! 200 + `text/event-stream`).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use rquickjs::function::{Opt, This};
use rquickjs::object::Accessor;
use rquickjs::{Ctx, Function, Object, Value};

use crate::bindings::dom::host_of;
use crate::bindings::events::{
    dispatch_to_target, install_event_handler_property, mint_standalone_target,
};
use crate::bindings::fetch::{FetchError, FetchRequest, FetchStreamEvent};
use crate::bindings::{
    BindingState, map_get, map_set, registry, registry_map, throw_dom, throw_type_error,
    to_dom_string,
};
use pluto_ipc::{CredentialsMode, RequestMode};

/// Ready states (`EventSource.CONNECTING` etc.).
pub const ES_CONNECTING: u16 = 0;
pub const ES_OPEN: u16 = 1;
pub const ES_CLOSED: u16 = 2;

/// Default reconnection time (spec: 3000 ms).
pub const ES_DEFAULT_RETRY_MS: u64 = 3000;

/// The JS-visible state of one `EventSource` (plain Rust data).
#[derive(Debug, Clone)]
pub struct EventSourceState {
    /// The stream URL.
    pub url: String,
    /// The last event id the server sent ("" when none).
    pub last_event_id: String,
    /// The reconnection time in ms (server `retry:` or 3000).
    pub retry_ms: u64,
    /// Ready state (`ES_*`).
    pub ready_state: u16,
    /// The in-flight stream fetch id, if any.
    pub fetch_id: Option<u64>,
    /// Whether the constructor asked for credentials.
    pub with_credentials: bool,
    /// The standalone-target registry id (for dispatch).
    pub target_id: u64,
}

/// One parsed server-sent event.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    /// The event type (`"message"` when the server sent none).
    pub event_type: String,
    /// The event data (lines joined with `\n`, trailing newline
    /// removed).
    pub data: String,
    /// The id the event carried, if any.
    pub last_event_id: Option<String>,
}

/// The SSE parser: a line buffer plus the in-progress event's
/// fields. Lines end with `\n`, `\r\n`, or `\r`; a `\r` at the
/// end of a chunk is held until the next chunk (it may be half of
/// a `\r\n`). A blank line dispatches the event.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of the current (incomplete) line.
    buf: Vec<u8>,
    /// Accumulated `data:` lines for the current event.
    data: String,
    /// The current event's `event:` field, if any.
    event_type: Option<String>,
    /// The current event's `id:` field, if any.
    last_event_id: Option<String>,
    /// The `retry:` value seen in this event, if any.
    retry: Option<u64>,
}

impl SseParser {
    /// Feed bytes; returns the events completed by this chunk.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        loop {
            let Some((line, consumed)) = take_line(&self.buf) else {
                break;
            };
            self.buf.drain(..consumed);
            if let Some(event) = self.handle_line(line) {
                out.push(event);
            }
        }
        out
    }

    /// The stream ended: flush a final unterminated line, if any.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let line = std::mem::take(&mut self.buf);
        self.handle_line(line)
            .map(|e| vec![e])
            .unwrap_or_default()
    }

    /// Handle one complete line (without its terminator).
    /// Returns the dispatched event, if the line was blank.
    fn handle_line(&mut self, line: Vec<u8>) -> Option<SseEvent> {
        let text = String::from_utf8_lossy(&line).into_owned();
        if text.is_empty() {
            return self.dispatch();
        }
        if text.starts_with(':') {
            return None; // Comment line.
        }
        let (field, value) = match text.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (text.as_str(), ""),
        };
        match field {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => self.event_type = Some(value.to_string()),
            "id" => {
                // An id containing NUL is ignored (spec).
                if !value.contains('\0') {
                    self.last_event_id = Some(value.to_string());
                }
            }
            "retry" => {
                if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
                    self.retry = value.parse().ok();
                }
            }
            _ => {}
        }
        None
    }

    /// Dispatch the current event (a blank line): the data buffer
    /// with its trailing newline removed, the event type (or
    /// `"message"`), the id (if any). A dispatch with an empty
    /// data buffer is a no-op (spec) — the `retry:` of that block
    /// still applies, and it stays in `self.retry` for the caller
    /// to read via [`take_retry`](SseParser::take_retry).
    fn dispatch(&mut self) -> Option<SseEvent> {
        let data = std::mem::take(&mut self.data);
        let event_type = std::mem::take(&mut self.event_type);
        let last_event_id = std::mem::take(&mut self.last_event_id);
        if data.is_empty() {
            return None;
        }
        let mut data = data;
        if data.ends_with('\n') {
            data.pop();
        }
        Some(SseEvent {
            event_type: event_type.unwrap_or_else(|| "message".to_string()),
            data,
            last_event_id,
        })
    }

    /// The `retry:` value the last dispatch carried, if any
    /// (consumed by the caller after handling the event).
    pub fn take_retry(&mut self) -> Option<u64> {
        self.retry.take()
    }
}

/// Take one complete line out of `buf` (without its terminator),
/// plus the bytes consumed. A trailing `\r` with nothing after it
/// is NOT a complete line yet — it may be half of `\r\n`.
fn take_line(buf: &[u8]) -> Option<(Vec<u8>, usize)> {
    let mut i = 0;
    while i < buf.len() {
        match buf[i] {
            b'\n' => return Some((buf[..i].to_vec(), i + 1)),
            b'\r' => {
                if i + 1 < buf.len() {
                    // `\r\n` consumes both; a lone `\r` is a
                    // terminator too.
                    let consumed = if buf[i + 1] == b'\n' { i + 2 } else { i + 1 };
                    return Some((buf[..i].to_vec(), consumed));
                }
                // A trailing `\r` may be half of `\r\n` — wait.
                return None;
            }
            _ => i += 1,
        }
    }
    None
}

/// Install `EventSource` (constructor + prototype) on the global
/// object. `stream_tx` is the channel the backend's stream events
/// travel on — the same one `fetch()` streaming uses.
pub fn install_event_source<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: mpsc::Sender<FetchStreamEvent>,
) -> Result<(), String> {
    let globals = ctx.globals();
    install_event_source_prototype(ctx, bindings)
        .map_err(|e| format!("EventSource proto: {e}"))?;
    // The constructor is a real JS function (a `Function::new`
    // closure is not constructible with `new` — QuickJS gives it
    // no [[Construct]]), following the `URL` pattern in dom.rs: a
    // Rust factory in the registry, and a JS constructor that
    // returns the factory's object.
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, url: Value<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            event_source_new(&ctx, &b, &stream_tx, url, init.0)
        },
    )
    .map_err(|e| format!("EventSource factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("esFactory", factory)
        .map_err(|e| format!("install EventSource factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.EventSource = function EventSource(url, init) { \
           if (!(this instanceof EventSource)) throw new TypeError('EventSource constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].esFactory(url, init); \
         };",
    )
    .map_err(|e| format!("install EventSource: {e}"))?;
    let es_ctor: Object = globals.get("EventSource").map_err(|e| e.to_string())?;
    es_ctor
        .set(
            "prototype",
            registry_map(ctx, "prototypes")
                .map_err(|e| e.to_string())?
                .get::<_, Object>("eventSource")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("EventSource.prototype: {e}"))?;
    es_ctor
        .set("CONNECTING", ES_CONNECTING as f64)
        .map_err(|e| e.to_string())?;
    es_ctor.set("OPEN", ES_OPEN as f64).map_err(|e| e.to_string())?;
    es_ctor
        .set("CLOSED", ES_CLOSED as f64)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// `new EventSource(url, { withCredentials }?)`.
fn event_source_new<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
    url: Value<'js>,
    init: Option<Value<'js>>,
) -> rquickjs::Result<Value<'js>> {
    let url_str = to_dom_string(ctx, &url)?;
    let host = host_of(bindings).map_err(|e| throw_type_error(&ctx, &e))?;
    let base = host
        .borrow()
        .document_url()
        .unwrap_or_else(|| "about:blank".to_string());
    let parsed = url::Url::parse(&url_str)
        .ok()
        .or_else(|| url::Url::parse(&base).ok()?.join(&url_str).ok())
        .ok_or_else(|| throw_dom(&ctx, "SyntaxError", "EventSource: the URL could not be parsed"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(throw_dom(
            &ctx,
            "SyntaxError",
            "EventSource: the URL scheme must be http: or https:",
        ));
    }
    let with_credentials = match init {
        Some(value) if value.is_object() => {
            let obj: Object = value.as_object().cloned().ok_or_else(|| {
                throw_type_error(&ctx, "EventSource: init must be an object")
            })?;
            obj.get::<_, bool>("withCredentials").unwrap_or(false)
        }
        // The JS constructor always passes two arguments, so a
        // missing `init` arrives as `undefined` — that is no init.
        Some(value) if value.is_null() || value.is_undefined() => false,
        Some(_) => {
            return Err(throw_type_error(&ctx, "EventSource: init must be an object"));
        }
        None => false,
    };
    let id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_event_source_id;
        s.next_event_source_id += 1;
        id
    };
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("eventSource")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let target_id = mint_standalone_target(ctx, bindings, &obj)?;
    bindings.borrow_mut().event_sources.insert(
        id,
        EventSourceState {
            url: parsed.to_string(),
            last_event_id: String::new(),
            retry_ms: ES_DEFAULT_RETRY_MS,
            ready_state: ES_CONNECTING,
            fetch_id: None,
            with_credentials,
            target_id,
        },
    );
    bindings
        .borrow_mut()
        .sse_parsers
        .insert(id, SseParser::default());
    let sources = registry_map(ctx, "eventSources")?;
    map_set(ctx, &sources, id, Value::from(obj.clone()))?;
    for name in ["open", "message", "error"] {
        install_event_handler_property(ctx, bindings, &obj, target_id, name)
            .map_err(|e| throw_type_error(&ctx, &e))?;
    }
    obj.prop(
        "url",
        rquickjs::object::Property::from(parsed.to_string()),
    )?;
    // The hidden reconnect entry point: the `setTimeout` callback
    // calls it (see `schedule_reconnect`). A script that finds it
    // can only restart the stream — the same thing a failed fetch
    // does on its own.
    let b = Rc::clone(bindings);
    let stream_tx_reconnect = stream_tx.clone();
    let reconnect_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<()> {
            start_event_source_stream(&ctx, &b, &stream_tx_reconnect, id)
                .map_err(|e| throw_type_error(&ctx, &e.to_string()))?;
            Ok(())
        },
    )?;
    obj.set("__pluto_reconnect", reconnect_fn)?;
    start_event_source_stream(ctx, bindings, stream_tx, id)
        .map_err(|e| throw_type_error(&ctx, &e.to_string()))?;
    Ok(Value::from(obj))
}

/// The `EventSource` prototype: `close()` + the `readyState` /
/// `url` / `withCredentials` accessors.
fn install_event_source_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    // An `EventTarget`: chain from the shared EventTarget
    // prototype so `addEventListener` / `removeEventListener` /
    // `dispatchEvent` work (named SSE events need them — without
    // this, `es.addEventListener('custom', …)` threw and a script
    // died before its `setTimeout` line ran).
    let event_target_proto: Object = protos
        .get("eventTarget")
        .map_err(|e| format!("EventTarget proto: {e}"))?;
    let proto = Object::new_proto(ctx.clone(), Some(&event_target_proto))
        .map_err(|e| format!("EventSource proto: {e}"))?;

    // `readyState`: the numeric state.
    let b = Rc::clone(bindings);
    proto
        .prop(
            "readyState",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                    Ok(es_id_of(&ctx, &this)
                        .and_then(|id| {
                            b.borrow().event_sources.get(&id).map(|s| s.ready_state as f64)
                        })
                        .unwrap_or(ES_CLOSED as f64))
                },
            ),
        )
        .map_err(|e| format!("EventSource.readyState: {e}"))?;

    // `withCredentials`: the constructor flag.
    let b = Rc::clone(bindings);
    proto
        .prop(
            "withCredentials",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
                    Ok(es_id_of(&ctx, &this)
                        .and_then(|id| {
                            b.borrow().event_sources.get(&id).map(|s| s.with_credentials)
                        })
                        .unwrap_or(false))
                },
            ),
        )
        .map_err(|e| format!("EventSource.withCredentials: {e}"))?;

    // `close()`: stop reconnecting and cancel the in-flight fetch.
    let b = Rc::clone(bindings);
    let close_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let Some(id) = es_id_of(&ctx, &this) else {
                return Ok(());
            };
            let fetch_id = {
                let mut s = b.borrow_mut();
                let Some(state) = s.event_sources.get_mut(&id) else {
                    return Ok(());
                };
                state.ready_state = ES_CLOSED;
                state.fetch_id.take()
            };
            if let Some(fetch_id) = fetch_id {
                b.borrow_mut().event_source_fetches.remove(&fetch_id);
                b.borrow_mut().pending_fetches.remove(&fetch_id);
                if let Some(backend) = b.borrow().fetch_backend.clone() {
                    backend.cancel_fetch(fetch_id);
                }
            }
            Ok(())
        },
    )
    .map_err(|e| format!("EventSource.close: {e}"))?;
    proto
        .set("close", close_fn)
        .map_err(|e| format!("EventSource.close: {e}"))?;

    protos
        .set("eventSource", proto)
        .map_err(|e| format!("prototypes eventSource: {e}"))?;
    Ok(())
}

/// The `EventSource` id on an object (stored under the same
/// target-id symbol as `WebSocket` — the object is a standalone
/// target, and its registry id doubles as its source id).
fn es_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    crate::bindings::events::standalone_target_id_of(ctx, obj)
}

/// The `EventSource` object for `id`, if it is still registered.
pub fn event_source_object<'js>(
    ctx: &Ctx<'js>,
    id: u64,
) -> rquickjs::Result<Option<Object<'js>>> {
    let map = registry_map(ctx, "eventSources")?;
    let value = map_get(ctx, &map, id)?;
    Ok(value.as_object().cloned())
}

/// Start (or restart) the stream fetch for `es_id`: a CORS-mode
/// GET with `Accept: text/event-stream` and `Last-Event-ID` when
/// the server set one. A no-op when the source is closed or a
/// fetch is already in flight (reconnect races the old stream's
/// terminal event).
pub fn start_event_source_stream<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
    es_id: u64,
) -> Result<(), String> {
    let (url, last_event_id, with_credentials) = {
        let s = bindings.borrow();
        let Some(state) = s.event_sources.get(&es_id) else {
            return Ok(());
        };
        if state.ready_state == ES_CLOSED {
            return Ok(());
        }
        if state.fetch_id.is_some() {
            return Ok(()); // A fetch is already in flight.
        }
        (
            state.url.clone(),
            state.last_event_id.clone(),
            state.with_credentials,
        )
    };
    let fetch_id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_fetch_id;
        s.next_fetch_id += 1;
        id
    };
    let mut headers = vec![("Accept".to_string(), "text/event-stream".to_string())];
    if !last_event_id.is_empty() {
        headers.push(("Last-Event-ID".to_string(), last_event_id));
    }
    let request = FetchRequest {
        id: fetch_id,
        url: url.clone(),
        method: "GET".to_string(),
        headers,
        body: None,
        mode: RequestMode::Cors,
        credentials: if with_credentials {
            CredentialsMode::Include
        } else {
            CredentialsMode::SameOrigin
        },
    };
    {
        let mut s = bindings.borrow_mut();
        s.pending_fetches.insert(fetch_id, request.clone());
        s.event_source_fetches.insert(fetch_id, es_id);
        if let Some(state) = s.event_sources.get_mut(&es_id) {
            state.fetch_id = Some(fetch_id);
            state.ready_state = ES_CONNECTING;
        }
    }
    let backend = bindings
        .borrow()
        .fetch_backend
        .clone()
        .ok_or_else(|| "EventSource: no fetch backend attached".to_string())?;
    backend.start_stream_fetch(request, stream_tx.clone());
    let _ = ctx;
    Ok(())
}

/// Route one stream event to the `EventSource` it belongs to
/// (called from `fetch::resolve_stream_event`). Returns `true`
/// when the event belonged to an `EventSource` (the caller then
/// does nothing else with it). Takes the event by reference — the
/// caller still owns it when this returns `false`.
pub fn resolve_event_source_stream_event<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
    event: &FetchStreamEvent,
) -> rquickjs::Result<bool> {
    let fetch_id = event.fetch_id();
    let Some(es_id) = bindings.borrow().event_source_fetches.get(&fetch_id).copied() else {
        return Ok(false);
    };
    match event {
        FetchStreamEvent::Headers {
            status, headers, ..
        } => {
            // The stream is only established for 200 +
            // `text/event-stream` (spec); anything else fails the
            // connection (error + reconnect).
            let content_type = headers
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case("content-type"))
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            let ok = *status == 200
                && content_type
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("text/event-stream");
            if !ok {
                fail_connection(ctx, bindings, stream_tx, es_id, "EventSource: bad status or content type")?;
                return Ok(true);
            }
            if let Some(state) = bindings.borrow_mut().event_sources.get_mut(&es_id) {
                state.ready_state = ES_OPEN;
            }
            dispatch_simple(ctx, bindings, es_id, "open")?;
        }
        FetchStreamEvent::Chunk { bytes, .. } => {
            let events = {
                let mut s = bindings.borrow_mut();
                match s.sse_parsers.get_mut(&es_id) {
                    Some(parser) => parser.feed(bytes),
                    None => Vec::new(),
                }
            };
            for event in events {
                apply_sse_event(ctx, bindings, es_id, event)?;
            }
            // A `retry:` in a data-less block still applies (see
            // `SseParser::dispatch`).
            let retry = {
                let mut s = bindings.borrow_mut();
                s.sse_parsers
                    .get_mut(&es_id)
                    .and_then(|p| p.take_retry())
            };
            if let Some(retry) = retry
                && let Some(state) = bindings.borrow_mut().event_sources.get_mut(&es_id)
            {
                state.retry_ms = retry;
            }
        }
        FetchStreamEvent::Done { .. } => {
            // The server closed the stream: a failure (spec) —
            // error + reconnect.
            finish_fetch(bindings, fetch_id, es_id);
            fail_connection(ctx, bindings, stream_tx, es_id, "EventSource: stream closed by server")?;
        }
        FetchStreamEvent::Error { error, .. } => {
            finish_fetch(bindings, fetch_id, es_id);
            match error {
                // A cancelled fetch never reconnects (navigation /
                // `close()` already owns the outcome).
                FetchError::Cancelled => {}
                FetchError::Network(msg) | FetchError::InvalidUrl(msg) => {
                    fail_connection(ctx, bindings, stream_tx, es_id, msg)?;
                }
            }
        }
    }
    Ok(true)
}

/// The fetch for `es_id` ended: forget the fetch id (the source
/// stays, for reconnect).
fn finish_fetch(bindings: &Rc<RefCell<BindingState>>, fetch_id: u64, es_id: u64) {
    let mut s = bindings.borrow_mut();
    s.event_source_fetches.remove(&fetch_id);
    s.pending_fetches.remove(&fetch_id);
    if let Some(state) = s.event_sources.get_mut(&es_id) {
        state.fetch_id = None;
    }
}

/// Apply one parsed SSE event: update the last-event-id / retry,
/// then dispatch it (`message` by default, or the server's type)
/// with `data` / `origin` / `lastEventId`.
fn apply_sse_event<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    es_id: u64,
    event: SseEvent,
) -> rquickjs::Result<()> {
    if event.data.is_empty() && event.event_type.is_empty() {
        return Ok(()); // A data-less block: only its `retry:` applied.
    }
    let (target_id, url) = {
        let mut s = bindings.borrow_mut();
        if let Some(id) = &event.last_event_id
            && let Some(state) = s.event_sources.get_mut(&es_id)
        {
            state.last_event_id = id.clone();
        }
        let retry = s
            .sse_parsers
            .get_mut(&es_id)
            .and_then(|p| p.take_retry());
        if let Some(retry) = retry
            && let Some(state) = s.event_sources.get_mut(&es_id)
        {
            state.retry_ms = retry;
        }
        let Some(state) = s.event_sources.get(&es_id) else {
            return Ok(());
        };
        (state.target_id, state.url.clone())
    };
    let event_obj: Object = ctx.eval(format!("new Event({:?}, {{}})", event.event_type))?;
    event_obj.set("data", event.data)?;
    let origin = pluto_ipc::Origin::from_url(&url).serialize();
    event_obj.set("origin", origin)?;
    event_obj.set(
        "lastEventId",
        event.last_event_id.clone().unwrap_or_default(),
    )?;
    dispatch_to_target(ctx, bindings, target_id, &event_obj, &event.event_type)?;
    Ok(())
}

/// Dispatch a plain event (no fields) to the source.
fn dispatch_simple<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    es_id: u64,
    ty: &str,
) -> rquickjs::Result<()> {
    let Some(target_id) = bindings
        .borrow()
        .event_sources
        .get(&es_id)
        .map(|s| s.target_id)
    else {
        return Ok(());
    };
    let event_obj: Object = ctx.eval(format!("new Event({ty:?}, {{}})"))?;
    dispatch_to_target(ctx, bindings, target_id, &event_obj, ty)?;
    Ok(())
}

/// Fail the connection: fire `error`, then schedule a reconnect
/// (unless the source is closed) after the current retry delay,
/// via the realm's `setTimeout` (a normal loop timer — navigation
/// clears it with everything else).
fn fail_connection<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_tx: &mpsc::Sender<FetchStreamEvent>,
    es_id: u64,
    message: &str,
) -> rquickjs::Result<()> {
    // `error` first (spec: the error event fires on failure).
    let target_id = bindings
        .borrow()
        .event_sources
        .get(&es_id)
        .map(|s| s.target_id);
    if let Some(target_id) = target_id {
        let event_obj: Object = ctx.eval("new Event('error', {})")?;
        event_obj.set("message", message)?;
        dispatch_to_target(ctx, bindings, target_id, &event_obj, "error")?;
    }
    let (closed, retry_ms) = {
        let s = bindings.borrow();
        match s.event_sources.get(&es_id) {
            Some(state) => (state.ready_state == ES_CLOSED, state.retry_ms),
            None => (true, ES_DEFAULT_RETRY_MS),
        }
    };
    if closed {
        return Ok(());
    }
    // Reconnect through `setTimeout`: the callback re-enters via
    // the hidden `__pluto_reconnect` method on the source object.
    let Some(obj) = event_source_object(ctx, es_id)? else {
        return Ok(());
    };
    let schedule: Function =
        ctx.eval("(es, delay) => setTimeout(() => es.__pluto_reconnect(), delay)")?;
    let _: Value = schedule.call((Value::from(obj), retry_ms as f64))?;
    let _ = stream_tx;
    Ok(())
}

/// Drop an `EventSource`'s state (navigation / realm teardown).
pub fn drop_event_source(bindings: &Rc<RefCell<BindingState>>, id: u64) {
    let mut s = bindings.borrow_mut();
    s.event_sources.remove(&id);
    s.sse_parsers.remove(&id);
}
