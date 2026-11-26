//! `WebSocket` (Stage 16 Phase 3, Rec 11): the JS binding over a
//! browser-side socket.
//!
//! The socket lives in the browser process (the broker pump —
//! `NetworkBroker::serve_websocket`), never in the renderer. The
//! binding talks to it through [`WebSocketBackend`], whose methods
//! only ever move plain Rust data over an mpsc channel — rquickjs
//! values never cross threads, exactly like the fetch bridge.
//! [`TabJs`] drains the backend's events into Networking tasks
//! (`resolve_ws_event` below), and only the tab-thread loop
//! touches JS.
//!
//! What the binding does NOT do: client masking and ping/pong live
//! in tungstenite (Rust), not in JS; `binaryType` defaults to
//! `"arraybuffer"` — there is still no `Blob` class, so a binary
//! message is an `ArrayBuffer` by default. Setting `binaryType`
//! to `"blob"` delivers the same Blob-shaped `{ size, type }`
//! object `fetch()`'s `.blob()` returns (an honest stand-in
//! until a real `Blob` exists); any other value is ignored per
//! spec. `bufferedAmount` reads `0`: `send()` hands the bytes to
//! the backend synchronously, so there is no JS-visible queue —
//! the in-flight bytes live in the browser-side pump.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use rquickjs::function::{Opt, This};
use rquickjs::object::Accessor;
use rquickjs::{Ctx, FromJs, Function, Object, TypedArray, Value};

use crate::bindings::dom::host_of;
use crate::bindings::events::{
    dispatch_to_target, install_event_handler_property, mint_standalone_target,
};
use crate::bindings::{
    BindingState, js, map_get, map_set, registry, registry_map, throw_dom, throw_type_error,
    to_dom_string,
};

/// Ready states (`WebSocket.CONNECTING` etc.).
pub const WS_CONNECTING: u16 = 0;
pub const WS_OPEN: u16 = 1;
pub const WS_CLOSING: u16 = 2;
pub const WS_CLOSED: u16 = 3;

/// A WebSocket connect request: plain data for the backend.
#[derive(Debug, Clone)]
pub struct WsConnectRequest {
    /// Socket id (from [`BindingState::next_ws_id`]).
    pub id: u64,
    /// The `ws:` / `wss:` URL.
    pub url: String,
    /// Requested subprotocols, in preference order.
    pub protocols: Vec<String>,
    /// The document origin, serialized (the handshake's `Origin`
    /// header).
    pub origin: String,
}

/// One backend event, delivered to the loop. `id` is the socket the
/// event belongs to.
#[derive(Debug, Clone)]
pub enum WsEvent {
    Opened {
        id: u64,
        /// The subprotocol the server selected ("" when none).
        protocol: String,
    },
    Message {
        id: u64,
        data: Vec<u8>,
        is_binary: bool,
    },
    Closed {
        id: u64,
        code: Option<u16>,
        reason: String,
        clean: bool,
    },
    /// Handshake / protocol failure before `Opened` (after `Opened`,
    /// failures arrive as `Closed { clean: false }`).
    Error {
        id: u64,
        message: String,
    },
}

/// The socket backend. Implemented by the integration wave
/// (`PlutoWebSocketBackend` in `pluto-browser`): `connect` must
/// return promptly and deliver events on `events` — from any
/// thread — until the socket closes; `send` / `close` are
/// fire-and-forget forwards to the browser-side pump. The backend
/// must never call into the realm.
pub trait WebSocketBackend: Send + Sync {
    /// Open the socket. Events arrive on `events`, exactly one
    /// terminal event (`Closed` or `Error`) per socket.
    fn connect(&self, request: WsConnectRequest, events: mpsc::Sender<WsEvent>);
    /// Queue one message for the socket (no-op when the socket is
    /// already gone — the binding checked the state first).
    fn send(&self, id: u64, data: Vec<u8>, is_binary: bool);
    /// Close the socket (no-op when already gone).
    fn close(&self, id: u64, code: Option<u16>, reason: String);
}

/// The JS-visible state of one socket (plain Rust data — the JS
/// object holds only the id).
#[derive(Debug, Clone)]
pub struct WsState {
    /// Ready state (`WS_*`).
    pub ready_state: u16,
    /// `"arraybuffer"` (default) or `"blob"`.
    pub binary_type: String,
    /// The URL the socket was opened with.
    pub url: String,
    /// The subprotocol the server selected ("" until open).
    pub protocol: String,
    /// The standalone-target registry id (for dispatch).
    pub target_id: u64,
}

/// The socket id stored on a `WebSocket` object.
pub const WS_SYMBOL: &str = "pluto.webSocket";

/// Install `WebSocket` (constructor + prototype) on the global
/// object. `ws_tx` is the channel the backend's events travel on
/// — `TabJs` drains it into Networking tasks.
pub fn install_websocket<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    ws_tx: mpsc::Sender<WsEvent>,
) -> Result<(), String> {
    let globals = ctx.globals();
    install_websocket_prototype(ctx, bindings).map_err(|e| format!("WebSocket proto: {e}"))?;
    // The constructor is a real JS function (a `Function::new`
    // closure is not constructible with `new` — QuickJS gives it
    // no [[Construct]]), following the `URL` pattern in dom.rs: a
    // Rust factory in the registry, and a JS constructor that
    // returns the factory's object.
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, url: Value<'js>, protocols: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            websocket_new(&ctx, &b, &ws_tx, url, protocols.0)
        },
    )
    .map_err(|e| format!("WebSocket factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("wsFactory", factory)
        .map_err(|e| format!("install WebSocket factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.WebSocket = function WebSocket(url, protocols) { \
           if (!(this instanceof WebSocket)) throw new TypeError('WebSocket constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].wsFactory(url, protocols); \
         };",
    )
    .map_err(|e| format!("install WebSocket: {e}"))?;
    // Ready-state constants on the constructor, per WebIDL.
    let ws_ctor: Object = globals.get("WebSocket").map_err(|e| e.to_string())?;
    ws_ctor
        .set(
            "prototype",
            registry_map(ctx, "prototypes")
                .map_err(|e| e.to_string())?
                .get::<_, Object>("webSocket")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("WebSocket.prototype: {e}"))?;
    ws_ctor
        .set("CONNECTING", WS_CONNECTING as f64)
        .map_err(|e| e.to_string())?;
    ws_ctor.set("OPEN", WS_OPEN as f64).map_err(|e| e.to_string())?;
    ws_ctor
        .set("CLOSING", WS_CLOSING as f64)
        .map_err(|e| e.to_string())?;
    ws_ctor
        .set("CLOSED", WS_CLOSED as f64)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// `new WebSocket(url, protocols?)`.
fn websocket_new<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    ws_tx: &mpsc::Sender<WsEvent>,
    url: Value<'js>,
    protocols: Option<Value<'js>>,
) -> rquickjs::Result<Value<'js>> {
    let url_str = to_dom_string(ctx, &url)?;
    // Resolve against the document base; a bad URL is a
    // `SyntaxError`, and a non-`ws:`/`wss:` scheme is a
    // `SyntaxError` too (spec: the URL must use a WebSocket
    // scheme).
    let host = host_of(bindings).map_err(|e| throw_type_error(&ctx, &e))?;
    let base = host
        .borrow()
        .document_url()
        .unwrap_or_else(|| "about:blank".to_string());
    let parsed = url::Url::parse(&url_str)
        .ok()
        .or_else(|| url::Url::parse(&base).ok()?.join(&url_str).ok())
        .ok_or_else(|| throw_dom(&ctx, "SyntaxError", "WebSocket: the URL could not be parsed"))?;
    if !matches!(parsed.scheme(), "ws" | "wss") {
        return Err(throw_dom(
            &ctx,
            "SyntaxError",
            "WebSocket: the URL scheme must be ws: or wss:",
        ));
    }
    // Subprotocols: a string, or an array of strings. A duplicate
    // is a `SyntaxError` (spec).
    let mut subprotocols: Vec<String> = Vec::new();
    if let Some(value) = protocols
        && !value.is_undefined()
        && !value.is_null()
    {
        if let Some(s) = value.as_string() {
            subprotocols.push(s.to_string().unwrap_or_default());
        } else if let Some(arr) = value.as_object().and_then(|o| o.clone().into_array()) {
            let len: usize = arr.len();
            for i in 0..len {
                let item: Value = arr.get(i)?;
                let s = to_dom_string(ctx, &item)?;
                if subprotocols.contains(&s) {
                    return Err(throw_dom(
                        &ctx,
                        "SyntaxError",
                        "WebSocket: duplicate subprotocol",
                    ));
                }
                subprotocols.push(s);
            }
        } else {
            return Err(throw_type_error(
                &ctx,
                "WebSocket: protocols must be a string or an array of strings",
            ));
        }
    }
    let backend = bindings
        .borrow()
        .ws_backend
        .clone()
        .ok_or_else(|| throw_type_error(&ctx, "WebSocket: no WebSocket backend attached"))?;
    let doc_origin = host.borrow().document_url().unwrap_or_default();
    let origin = pluto_ipc::Origin::from_url(&doc_origin).serialize();
    let id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_ws_id;
        s.next_ws_id += 1;
        id
    };
    // The object: prototype + id symbol + standalone-target
    // registration (so it is an `EventTarget`), then the `on*`
    // handler properties.
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("webSocket")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = rquickjs::Symbol::new_global(ctx.clone(), WS_SYMBOL)?;
    obj.set(sym.as_atom(), id as f64)?;
    let target_id = mint_standalone_target(ctx, bindings, &obj)?;
    bindings.borrow_mut().ws_states.insert(
        id,
        WsState {
            ready_state: WS_CONNECTING,
            binary_type: "arraybuffer".to_string(),
            url: parsed.to_string(),
            protocol: String::new(),
            target_id,
        },
    );
    let sockets = registry_map(ctx, "webSockets")?;
    map_set(ctx, &sockets, id, Value::from(obj.clone()))?;
    for name in ["open", "message", "error", "close"] {
        install_event_handler_property(ctx, bindings, &obj, target_id, name)
            .map_err(|e| throw_type_error(&ctx, &e))?;
    }
    // `url`: the resolved URL, as a plain data property.
    obj.prop(
        "url",
        rquickjs::object::Property::from(parsed.to_string()),
    )?;
    backend.connect(
        WsConnectRequest {
            id,
            url: parsed.to_string(),
            protocols: subprotocols,
            origin,
        },
        ws_tx.clone(),
    );
    Ok(Value::from(obj))
}

/// The `WebSocket` prototype: `send` / `close` + the `readyState`
/// / `binaryType` / `bufferedAmount` / `protocol` accessors.
fn install_websocket_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    // A `WebSocket` is an `EventTarget`: chain from the shared
    // prototype so `addEventListener` / `removeEventListener` /
    // `dispatchEvent` work (same fix as EventSource).
    let event_target_proto: Object = protos
        .get("eventTarget")
        .map_err(|e| format!("EventTarget proto: {e}"))?;
    let proto = Object::new_proto(ctx.clone(), Some(&event_target_proto))
        .map_err(|e| format!("WebSocket proto: {e}"))?;

    // `readyState`: the numeric state.
    let b = Rc::clone(bindings);
    proto
        .prop(
            "readyState",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                    Ok(ws_id_of(&ctx, &this)
                        .and_then(|id| b.borrow().ws_states.get(&id).map(|s| s.ready_state as f64))
                        .unwrap_or(WS_CLOSED as f64))
                },
            ),
        )
        .map_err(|e| format!("WebSocket.readyState: {e}"))?;

    // `binaryType`: `"arraybuffer"` (default) or `"blob"`; any
    // other value is ignored per spec.
    let b = Rc::clone(bindings);
    let b_get = Rc::clone(&b);
    proto
        .prop(
            "binaryType",
            Accessor::new(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<String> {
                    Ok(ws_id_of(&ctx, &this)
                        .and_then(|id| b_get.borrow().ws_states.get(&id).map(|s| s.binary_type.clone()))
                        .unwrap_or_else(|| "arraybuffer".to_string()))
                },
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
                    let _ = ctx;
                    if let Some(s) = value.as_string().and_then(|s| s.to_string().ok())
                        && matches!(s.as_str(), "blob" | "arraybuffer")
                        && let Some(id) = ws_id_of(&ctx, &this)
                        && let Some(state) = b.borrow_mut().ws_states.get_mut(&id)
                    {
                        state.binary_type = s;
                    }
                    Ok(())
                },
            ),
        )
        .map_err(|e| format!("WebSocket.binaryType: {e}"))?;

    // `bufferedAmount`: 0 — `send()` hands off synchronously (see
    // the module docs).
    proto
        .prop(
            "bufferedAmount",
            Accessor::from(
                move |ctx: Ctx<'js>, This(_this): This<Object<'js>>| -> rquickjs::Result<f64> {
                    let _ = ctx;
                    Ok(0.0)
                },
            ),
        )
        .map_err(|e| format!("WebSocket.bufferedAmount: {e}"))?;

    // `protocol`: the subprotocol the server selected ("" until open).
    let b = Rc::clone(bindings);
    proto
        .prop(
            "protocol",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<String> {
                    Ok(ws_id_of(&ctx, &this)
                        .and_then(|id| b.borrow().ws_states.get(&id).map(|s| s.protocol.clone()))
                        .unwrap_or_default())
                },
            ),
        )
        .map_err(|e| format!("WebSocket.protocol: {e}"))?;

    // `extensions`: "" — Pluto negotiates no extensions.
    proto
        .prop(
            "extensions",
            Accessor::from(
                move |ctx: Ctx<'js>, This(_this): This<Object<'js>>| -> rquickjs::Result<String> {
                    let _ = ctx;
                    Ok(String::new())
                },
            ),
        )
        .map_err(|e| format!("WebSocket.extensions: {e}"))?;

    // `send(data)`: a string, an `ArrayBuffer`, or a typed-array
    // view. `CONNECTING` throws `InvalidStateError`;
    // `CLOSING` / `CLOSED` silently drops (spec: the data is
    // queued only while open — a closed socket has nowhere to
    // queue it).
    let b = Rc::clone(bindings);
    let send_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, data: Value<'js>| -> rquickjs::Result<()> {
            let Some(id) = ws_id_of(&ctx, &this) else {
                return Err(throw_type_error(&ctx, "send: not a WebSocket"));
            };
            let ready = b
                .borrow()
                .ws_states
                .get(&id)
                .map(|s| s.ready_state)
                .unwrap_or(WS_CLOSED);
            if ready == WS_CONNECTING {
                return Err(throw_dom(
                    &ctx,
                    "InvalidStateError",
                    "WebSocket is not open yet",
                ));
            }
            if ready != WS_OPEN {
                return Ok(());
            }
            let (bytes, is_binary) = ws_message_bytes(&ctx, &data)?;
            if let Some(backend) = b.borrow().ws_backend.clone() {
                backend.send(id, bytes, is_binary);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("WebSocket.send: {e}"))?;
    proto
        .set("send", send_fn)
        .map_err(|e| format!("WebSocket.send: {e}"))?;

    // `close(code?, reason?)`: validates the code (1000 or
    // 3000–4999) and the reason (≤ 123 UTF-8 bytes), then moves to
    // `CLOSING` and asks the backend to send the close frame. A
    // second `close()` is a no-op.
    let b = Rc::clone(bindings);
    let close_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, code: Opt<f64>, reason: Opt<String>| -> rquickjs::Result<()> {
            let Some(id) = ws_id_of(&ctx, &this) else {
                return Err(throw_type_error(&ctx, "close: not a WebSocket"));
            };
            let code = code.0.map(|c| c as u16);
            if let Some(code) = code
                && !(code == 1000 || (3000..=4999).contains(&code))
            {
                return Err(throw_dom(
                    &ctx,
                    "InvalidAccessError",
                    "WebSocket close code must be 1000 or 3000-4999",
                ));
            }
            let reason = reason.0.unwrap_or_default();
            if reason.len() > 123 {
                return Err(throw_dom(
                    &ctx,
                    "SyntaxError",
                    "WebSocket close reason must be 123 bytes or fewer",
                ));
            }
            let mut s = b.borrow_mut();
            let Some(state) = s.ws_states.get_mut(&id) else {
                return Ok(());
            };
            if state.ready_state == WS_CLOSING || state.ready_state == WS_CLOSED {
                return Ok(());
            }
            state.ready_state = WS_CLOSING;
            drop(s);
            if let Some(backend) = b.borrow().ws_backend.clone() {
                backend.close(id, code, reason);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("WebSocket.close: {e}"))?;
    proto
        .set("close", close_fn)
        .map_err(|e| format!("WebSocket.close: {e}"))?;

    protos
        .set("webSocket", proto)
        .map_err(|e| format!("prototypes webSocket: {e}"))?;
    Ok(())
}

/// The socket id on a `WebSocket` object.
fn ws_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = rquickjs::Symbol::new_global(ctx.clone(), WS_SYMBOL).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

/// Encode `send()` data: a string is text (UTF-8), an
/// `ArrayBuffer` / typed-array view is binary.
fn ws_message_bytes<'js>(ctx: &Ctx<'js>, data: &Value<'js>) -> rquickjs::Result<(Vec<u8>, bool)> {
    if let Some(s) = data.as_string() {
        return Ok((s.to_string().unwrap_or_default().into_bytes(), false));
    }
    if let Some(bytes) = data
        .as_object()
        .and_then(|o| TypedArray::<u8>::from_object(o.clone()).ok())
    {
        let raw = bytes
            .as_raw()
            .map(|r| unsafe { r.as_ref().to_vec() })
            .unwrap_or_default();
        return Ok((raw, true));
    }
    if let Some(buf) = data.as_object().and_then(|o| {
        rquickjs::ArrayBuffer::from_js(ctx, Value::from(o.clone())).ok()
    }) {
        return Ok((unsafe { buf.as_bytes() }.unwrap_or_default().to_vec(), true));
    }
    Err(throw_type_error(
        &ctx,
        "WebSocket.send: data must be a string, ArrayBuffer, or typed array",
    ))
}

/// Resolve (or reject) one backend event from a Networking-source
/// Host task. Called by `TabJs` with the loop's `Ctx`; builds the
/// event object and runs the dispatch walk. A throwing listener is
/// reported, never fatal.
pub fn resolve_ws_event<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    event: WsEvent,
) -> rquickjs::Result<()> {
    match event {
        WsEvent::Opened { id, protocol } => {
            let Some((target_id, _)) = transition(bindings, id, WS_OPEN, Some(&protocol)) else {
                return Ok(()); // Socket dropped: nothing to dispatch to.
            };
            let event_obj: Object = ctx.eval("new Event('open', {})")?;
            dispatch_to_target(ctx, bindings, target_id, &event_obj, "open")?;
        }
        WsEvent::Message {
            id,
            data,
            is_binary,
        } => {
            let Some((target_id, binary_type)) = ({
                let s = bindings.borrow();
                s.ws_states
                    .get(&id)
                    .map(|st| (st.target_id, st.binary_type.clone()))
            }) else {
                return Ok(());
            };
            let event_obj: Object = ctx.eval("new Event('message', {})")?;
            let payload: Value = if is_binary {
                if binary_type == "blob" {
                    // The Blob-shaped stand-in (see the module
                    // docs): `{ size, type }`, no real `Blob`.
                    let blob = Object::new(ctx.clone())?;
                    blob.set("size", data.len() as f64)?;
                    blob.set("type", "")?;
                    Value::from(blob)
                } else {
                    let uint8 = TypedArray::<u8>::new_copy(ctx.clone(), data.as_slice())?;
                    uint8.into_value()
                }
            } else {
                js(&ctx, String::from_utf8_lossy(&data).into_owned())
            };
            event_obj.set("data", payload)?;
            dispatch_to_target(ctx, bindings, target_id, &event_obj, "message")?;
        }
        WsEvent::Closed {
            id,
            code,
            reason,
            clean,
        } => {
            let Some((target_id, _)) = transition(bindings, id, WS_CLOSED, None) else {
                return Ok(());
            };
            let event_obj: Object = ctx.eval("new Event('close', {})")?;
            event_obj.set("code", code.map(|c| c as f64).unwrap_or(1005.0))?;
            event_obj.set("reason", reason)?;
            event_obj.set("wasClean", clean)?;
            dispatch_to_target(ctx, bindings, target_id, &event_obj, "close")?;
        }
        WsEvent::Error { id, message } => {
            // A failed connection fires `error` and then `close`
            // (code 1006, not clean), per spec.
            let Some((target_id, _)) = transition(bindings, id, WS_CLOSED, None) else {
                return Ok(());
            };
            let event_obj: Object = ctx.eval("new Event('error', {})")?;
            event_obj.set("message", message)?;
            dispatch_to_target(ctx, bindings, target_id, &event_obj, "error")?;
            let close_obj: Object = ctx.eval("new Event('close', {})")?;
            close_obj.set("code", 1006.0)?;
            close_obj.set("reason", "")?;
            close_obj.set("wasClean", false)?;
            dispatch_to_target(ctx, bindings, target_id, &close_obj, "close")?;
        }
    }
    Ok(())
}

/// Move a socket to `ready_state` (recording the negotiated
/// protocol when given); returns its target id + binary type, or
/// `None` when the socket is already gone.
fn transition(
    bindings: &Rc<RefCell<BindingState>>,
    id: u64,
    ready_state: u16,
    protocol: Option<&str>,
) -> Option<(u64, String)> {
    let mut s = bindings.borrow_mut();
    let state = s.ws_states.get_mut(&id)?;
    state.ready_state = ready_state;
    if let Some(protocol) = protocol {
        state.protocol = protocol.to_string();
    }
    Some((state.target_id, state.binary_type.clone()))
}

/// The `WebSocket` object for `id`, if it is still registered.
pub fn ws_object<'js>(
    ctx: &Ctx<'js>,
    id: u64,
) -> rquickjs::Result<Option<Object<'js>>> {
    let map = registry_map(ctx, "webSockets")?;
    let value = map_get(ctx, &map, id)?;
    Ok(value.as_object().cloned())
}

/// Drop a socket's state (navigation / realm teardown).
pub fn drop_ws(bindings: &Rc<RefCell<BindingState>>, id: u64) {
    bindings.borrow_mut().ws_states.remove(&id);
}
