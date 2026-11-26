//! `ReadableStream` + the chunk channel (Stage 16 Phase 3, Rec 12).
//!
//! The stream is the JS-visible end of the chunk channel: the
//! backend (`FetchBackend::start_stream_fetch`) sends
//! `FetchStreamEvent`s over an mpsc channel, `TabJs` drains them
//! into Networking tasks, and this module turns them into queued
//! chunks / close / error on a stream object.
//!
//! Shape (same pattern as `fetch`'s registry `fetches` map): the
//! bytes + flags live in [`BindingState::stream_states`] (plain
//! Rust data — no JS values), the stream object lives in the
//! registry's `streams` Map, and pending `read()` resolvers live
//! in the registry's `streamWaiters` Map as
//! `{ resolve, reject, kind: "read" | "closed" }`. A chunk resolves
//! exactly one `read` waiter (or queues when nobody is waiting);
//! close resolves every waiter (`read` → `{ done: true }`,
//! `closed` → `undefined`); error rejects every waiter.
//!
//! What this is NOT: there is no BYOB reader, no `pipeTo` /
//! `pipeThrough`, no `WritableStream` / `TransformStream`, and no
//! backpressure signal to the producer beyond the bounded mpsc
//! channel the backend sends on — a reader that stops reading
//! stalls the network read instead of growing memory (the queue
//! here only holds chunks that arrived while nobody was waiting,
//! and the channel bound is what stops the producer).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use rquickjs::function::This;
use rquickjs::object::{Accessor, Property};
use rquickjs::{Array, Ctx, Function, Object, Symbol, TypedArray, Value};

use crate::bindings::{
    BindingState, js, map_delete, map_get, map_set, registry, registry_map, throw_type_error,
};

/// `Symbol.for("pluto.stream")` — the stream id on a
/// `ReadableStream` / reader object.
pub const STREAM_SYMBOL: &str = "pluto.stream";

/// `Symbol.for("pluto.streamClosed")` — the cached `closed`
/// promise on a reader object.
const STREAM_CLOSED_SYMBOL: &str = "pluto.streamClosed";

/// One live stream: the queued chunks + the terminal flags.
/// The chunks are bytes only — the `Uint8Array` is built at
/// `read()` time, so a queued chunk costs one clone, not a JS
/// object per chunk.
#[derive(Debug, Default)]
pub struct StreamState {
    /// Chunks that arrived while no `read()` was waiting.
    pub queue: VecDeque<Vec<u8>>,
    /// `true` once `Done` arrived (or `cancel()` ran).
    pub done: bool,
    /// The error message once the stream errored, if any.
    pub error: Option<String>,
    /// The fetch feeding this stream, if any (`Response.body`):
    /// `cancel()` cancels it through the backend.
    pub fetch_id: Option<u64>,
    /// `true` while a reader holds the lock.
    pub locked: bool,
}

/// The stream id stored on a `ReadableStream` / reader object.
pub fn stream_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), STREAM_SYMBOL).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

/// Install `ReadableStream` (constructor + prototype) on the
/// global object. `new ReadableStream()` creates an empty stream
/// with no source — reads pend forever (there is nothing feeding
/// it); the streams pages use come from `fetch()` /
/// `Response.body`, built by [`new_stream`].
pub fn install_streams<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    install_stream_prototype(ctx, bindings).map_err(|e| format!("ReadableStream proto: {e}"))?;
    // The constructor is a real JS function (a `Function::new`
    // closure is not constructible with `new` — QuickJS gives it
    // no [[Construct]]), following the `URL` pattern in dom.rs: a
    // Rust factory in the registry, and a JS constructor that
    // returns the factory's object.
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            let (obj, _id) = new_stream(&ctx, &b, None)
                .map_err(|e| throw_type_error(&ctx, &e.to_string()))?;
            Ok(Value::from(obj))
        },
    )
    .map_err(|e| format!("ReadableStream factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("streamFactory", factory)
        .map_err(|e| format!("install ReadableStream factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.ReadableStream = function ReadableStream() { \
           if (!(this instanceof ReadableStream)) throw new TypeError('ReadableStream constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].streamFactory(); \
         };",
    )
    .map_err(|e| format!("install ReadableStream: {e}"))?;
    let stream_ctor: Object = globals
        .get("ReadableStream")
        .map_err(|e| e.to_string())?;
    stream_ctor
        .set(
            "prototype",
            registry_map(ctx, "prototypes")
                .map_err(|e| e.to_string())?
                .get::<_, Object>("readableStream")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("ReadableStream.prototype: {e}"))?;
    Ok(())
}

/// The `ReadableStream` prototype: `getReader()` / `cancel()`.
fn install_stream_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let proto = Object::new(ctx.clone()).map_err(|e| format!("ReadableStream proto: {e}"))?;

    // `getReader()`: one active reader at a time (spec: a second
    // `getReader()` while locked throws `TypeError`).
    let b = Rc::clone(bindings);
    let get_reader = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = stream_id_of(&ctx, &this) else {
                return Err(throw_type_error(&ctx, "getReader: not a ReadableStream"));
            };
            {
                let mut s = b.borrow_mut();
                let Some(state) = s.stream_states.get_mut(&id) else {
                    return Err(throw_type_error(&ctx, "getReader: unknown stream"));
                };
                if state.locked {
                    return Err(throw_type_error(
                        &ctx,
                        "getReader: stream is already locked",
                    ));
                }
                state.locked = true;
            }
            make_reader(&ctx, &b, id).map(Value::from)
        },
    )
    .map_err(|e| format!("ReadableStream.getReader: {e}"))?;
    proto
        .set("getReader", get_reader)
        .map_err(|e| format!("ReadableStream.getReader: {e}"))?;

    // `cancel()`: stop the underlying fetch (if any), close the
    // stream so pending reads resolve `{ done: true }`, and return
    // a promise resolved once that is done.
    let b = Rc::clone(bindings);
    let cancel_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, _reject) = ctx.promise()?;
            let Some(id) = stream_id_of(&ctx, &this) else {
                let _: Value = resolve.call(((),))?;
                return Ok(promise.into_value());
            };
            cancel_stream(&ctx, &b, id)?;
            let _: Value = resolve.call(((),))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("ReadableStream.cancel: {e}"))?;
    proto
        .set("cancel", cancel_fn)
        .map_err(|e| format!("ReadableStream.cancel: {e}"))?;

    // `locked`: whether a reader holds the lock.
    let b = Rc::clone(bindings);
    proto
        .prop(
            "locked",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
                    let locked = stream_id_of(&ctx, &this)
                        .and_then(|id| b.borrow().stream_states.get(&id).map(|s| s.locked))
                        .unwrap_or(false);
                    Ok(locked)
                },
            ),
        )
        .map_err(|e| format!("ReadableStream.locked: {e}"))?;

    protos
        .set("readableStream", proto)
        .map_err(|e| format!("prototypes readableStream: {e}"))?;
    Ok(())
}

/// Build a reader object for `stream_id`: `read()` /
/// `releaseLock()` / `cancel()` / `closed`.
fn make_reader<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
) -> rquickjs::Result<Object<'js>> {
    let reader = Object::new(ctx.clone())?;
    let sym = Symbol::new_global(ctx.clone(), STREAM_SYMBOL)?;
    reader.set(sym.as_atom(), stream_id as f64)?;

    // `read()`: one chunk, or `{ done: true }`, or a promise that
    // resolves when the next chunk lands.
    let b = Rc::clone(bindings);
    let read_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, reject) = ctx.promise()?;
            let Some(id) = stream_id_of(&ctx, &this) else {
                let reason: Value = ctx.eval("new TypeError('read: not a stream reader')")?;
                let _: Value = reject.call((reason,))?;
                return Ok(promise.into_value());
            };
            let outcome = {
                let mut s = b.borrow_mut();
                match s.stream_states.get_mut(&id) {
                    None => ReadOutcome::Error("read: unknown stream".to_string()),
                    Some(state) => {
                        // The error wins over `done`: a stream that
                        // errored and then closed (the fetch Error
                        // path closes after erroring) must reject,
                        // not read as a clean EOF.
                        if let Some(msg) = &state.error {
                            ReadOutcome::Error(msg.clone())
                        } else if let Some(bytes) = state.queue.pop_front() {
                            ReadOutcome::Chunk(bytes)
                        } else if state.done {
                            ReadOutcome::Done
                        } else {
                            ReadOutcome::Wait
                        }
                    }
                }
            };
            match outcome {
                ReadOutcome::Chunk(bytes) => {
                    let result = chunk_result(&ctx, bytes)?;
                    let _: Value = resolve.call((Value::from(result),))?;
                }
                ReadOutcome::Done => {
                    let result = done_result(&ctx)?;
                    let _: Value = resolve.call((Value::from(result),))?;
                }
                ReadOutcome::Error(msg) => {
                    let reason: Value = ctx.eval(format!("new TypeError({msg:?})"))?;
                    let _: Value = reject.call((reason,))?;
                }
                ReadOutcome::Wait => {
                    push_waiter(&ctx, id, resolve, reject, "read")?;
                }
            }
            Ok(promise.into_value())
        },
    )
    ?;
    reader
        .set("read", read_fn)
        ?;

    // `releaseLock()`: drop the lock so another reader can attach.
    // Pending `read()` promises stay pending — they resolve when a
    // chunk lands, whoever reads next (spec: the lock only gates
    // `getReader`, not the queued reads).
    let b = Rc::clone(bindings);
    let release_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            if let Some(id) = stream_id_of(&ctx, &this) {
                if let Some(state) = b.borrow_mut().stream_states.get_mut(&id) {
                    state.locked = false;
                }
            }
            Ok(())
        },
    )
    ?;
    reader
        .set("releaseLock", release_fn)
        ?;

    // `cancel()`: cancel the stream (same as the stream's own
    // `cancel()`), then release the lock.
    let b = Rc::clone(bindings);
    let cancel_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, _reject) = ctx.promise()?;
            if let Some(id) = stream_id_of(&ctx, &this) {
                cancel_stream(&ctx, &b, id)?;
                if let Some(state) = b.borrow_mut().stream_states.get_mut(&id) {
                    state.locked = false;
                }
            }
            let _: Value = resolve.call(((),))?;
            Ok(promise.into_value())
        },
    )
    ?;
    reader
        .set("cancel", cancel_fn)
        ?;

    // `closed`: one promise per reader (cached on the reader under
    // a symbol), resolved on close, rejected on error.
    let b = Rc::clone(bindings);
    reader
        .prop(
            "closed",
            Accessor::from(
                move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                    let sym = Symbol::new_global(ctx.clone(), STREAM_CLOSED_SYMBOL)?;
                    if let Ok(existing) = this.get::<_, Value>(sym.as_atom())
                        && !existing.is_undefined()
                    {
                        return Ok(existing);
                    }
                    let (promise, resolve, reject) = ctx.promise()?;
                    let Some(id) = stream_id_of(&ctx, &this) else {
                        return Ok(promise.into_value());
                    };
                    let (done, error) = {
                        let s = b.borrow();
                        match s.stream_states.get(&id) {
                            Some(state) => (state.done, state.error.clone()),
                            None => (true, Some("unknown stream".to_string())),
                        }
                    };
                    if done {
                        let _: Value = resolve.call(((),))?;
                    } else if let Some(msg) = error {
                        let reason: Value = ctx.eval(format!("new TypeError({msg:?})"))?;
                        let _: Value = reject.call((reason,))?;
                    } else {
                        push_waiter(&ctx, id, resolve, reject, "closed")?;
                    }
                    let value = promise.into_value();
                    this.set(sym.as_atom(), value.clone())?;
                    Ok(value)
                },
            ),
        )
        ?;
    Ok(reader)
}

/// What one `read()` found.
enum ReadOutcome {
    Chunk(Vec<u8>),
    Done,
    Error(String),
    Wait,
}

/// `{ value: Uint8Array, done: false }` for one chunk.
fn chunk_result<'js>(ctx: &Ctx<'js>, bytes: Vec<u8>) -> rquickjs::Result<Object<'js>> {
    let uint8 = TypedArray::<u8>::new_copy(ctx.clone(), bytes.as_slice())?;
    let result = Object::new(ctx.clone())?;
    result.set("value", uint8.into_value())?;
    result.prop("done", Property::from(false))?;
    Ok(result)
}

/// `{ value: undefined, done: true }` for a closed stream.
fn done_result<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<Object<'js>> {
    let result = Object::new(ctx.clone())?;
    result.set("value", js(ctx, rquickjs::Undefined))?;
    result.prop("done", Property::from(true))?;
    Ok(result)
}

/// Push a `{ resolve, reject, kind }` record onto the stream's
/// waiter list in the registry.
fn push_waiter<'js>(
    ctx: &Ctx<'js>,
    stream_id: u64,
    resolve: Function<'js>,
    reject: Function<'js>,
    kind: &str,
) -> rquickjs::Result<()> {
    let map = registry_map(ctx, "streamWaiters")?;
    let mut waiters: Vec<Value> = Vec::new();
    let existing = map_get(ctx, &map, stream_id)?;
    if let Some(arr) = existing.as_object().and_then(|o| o.clone().into_array()) {
        let len: usize = arr.len();
        for i in 0..len {
            if let Ok(v) = arr.get::<Value>(i) {
                waiters.push(v);
            }
        }
    }
    let record = Object::new(ctx.clone())?;
    record.set("resolve", resolve)?;
    record.set("reject", reject)?;
    record.set("kind", kind)?;
    waiters.push(Value::from(record));
    let arr = Array::new(ctx.clone())?;
    for (i, value) in waiters.into_iter().enumerate() {
        arr.set(i, value)?;
    }
    map_set(ctx, &map, stream_id, Value::from(arr))?;
    Ok(())
}

/// Take the stream's waiter list out of the registry (empty when
/// nobody is waiting).
fn take_waiters<'js>(
    ctx: &Ctx<'js>,
    stream_id: u64,
) -> rquickjs::Result<Vec<(Function<'js>, Function<'js>, String)>> {
    let map = registry_map(ctx, "streamWaiters")?;
    let existing = map_get(ctx, &map, stream_id)?;
    map_delete(ctx, &map, stream_id);
    let mut out = Vec::new();
    if let Some(arr) = existing.as_object().and_then(|o| o.clone().into_array()) {
        let len: usize = arr.len();
        for i in 0..len {
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
            let kind: String = obj.get("kind").unwrap_or_else(|_| "read".to_string());
            out.push((resolve, reject, kind));
        }
    }
    Ok(out)
}

/// Create a stream object + state, registered in the `streams` Map.
/// `fetch_id` links a `Response.body` stream to its fetch (for
/// `cancel()`); `None` for a freestanding `new ReadableStream()`.
pub fn new_stream<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    fetch_id: Option<u64>,
) -> rquickjs::Result<(Object<'js>, u64)> {
    let id = {
        let mut s = bindings.borrow_mut();
        let id = s.next_stream_id;
        s.next_stream_id += 1;
        s.stream_states.insert(
            id,
            StreamState {
                fetch_id,
                ..Default::default()
            },
        );
        id
    };
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get("readableStream")?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), STREAM_SYMBOL)?;
    obj.set(sym.as_atom(), id as f64)?;
    let streams = registry_map(ctx, "streams")?;
    map_set(ctx, &streams, id, Value::from(obj.clone()))?;
    Ok((obj, id))
}

/// One chunk arrived for `stream_id`: queue it, or hand it straight
/// to a waiting `read()`. Called from the Networking task that
/// drains `FetchStreamEvent::Chunk`.
pub fn stream_push_chunk<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
    bytes: Vec<u8>,
) -> rquickjs::Result<()> {
    // A waiting `read()` takes the chunk at once (exactly one
    // waiter per chunk, in order); otherwise it queues.
    let waiters = take_waiters(ctx, stream_id)?;
    let mut read_waiter: Option<(Function, Function, String)> = None;
    let mut rest: Vec<(Function, Function, String)> = Vec::new();
    for waiter in waiters {
        if read_waiter.is_none() && waiter.2 == "read" {
            read_waiter = Some(waiter);
        } else {
            rest.push(waiter);
        }
    }
    if !rest.is_empty() {
        let map = registry_map(ctx, "streamWaiters")?;
        let arr = Array::new(ctx.clone())?;
        for (i, (resolve, reject, kind)) in rest.into_iter().enumerate() {
            let record = Object::new(ctx.clone())?;
            record.set("resolve", resolve)?;
            record.set("reject", reject)?;
            record.set("kind", kind)?;
            arr.set(i, Value::from(record))?;
        }
        map_set(ctx, &map, stream_id, Value::from(arr))?;
    }
    match read_waiter {
        Some((resolve, _reject, _)) => {
            let result = chunk_result(ctx, bytes)?;
            let _: Value = resolve.call((Value::from(result),))?;
        }
        None => {
            if let Some(state) = bindings.borrow_mut().stream_states.get_mut(&stream_id) {
                state.queue.push_back(bytes);
            }
        }
    }
    Ok(())
}

/// The stream finished (`Done`): mark it, resolve every waiter
/// (`read` → `{ done: true }`, `closed` → `undefined`).
pub fn stream_close<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
) -> rquickjs::Result<()> {
    if let Some(state) = bindings.borrow_mut().stream_states.get_mut(&stream_id) {
        state.done = true;
    }
    for (resolve, _reject, kind) in take_waiters(ctx, stream_id)? {
        if kind == "read" {
            let result = done_result(ctx)?;
            let _: Value = resolve.call((Value::from(result),))?;
        } else {
            let _: Value = resolve.call(((),))?;
        }
    }
    Ok(())
}

/// The stream errored: mark it, drop any queued chunks (spec:
/// erroring a stream discards the queue), reject every waiter
/// with a `TypeError` carrying the message.
pub fn stream_error<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
    msg: &str,
) -> rquickjs::Result<()> {
    if let Some(state) = bindings.borrow_mut().stream_states.get_mut(&stream_id) {
        state.error = Some(msg.to_string());
        state.queue.clear();
    }
    for (_resolve, reject, _kind) in take_waiters(ctx, stream_id)? {
        let reason: Value = ctx.eval(format!("new TypeError({msg:?})"))?;
        let _: Value = reject.call((reason,))?;
    }
    Ok(())
}

/// Cancel a stream: stop its fetch through the backend (if any),
/// then close it. Idempotent — a second cancel is a no-op.
pub fn cancel_stream<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
) -> rquickjs::Result<()> {
    let (done, fetch_id) = {
        let s = bindings.borrow();
        match s.stream_states.get(&stream_id) {
            Some(state) => (state.done, state.fetch_id),
            None => return Ok(()),
        }
    };
    if done {
        return Ok(());
    }
    if let Some(fetch_id) = fetch_id
        && let Some(backend) = bindings.borrow().fetch_backend.clone()
    {
        backend.cancel_fetch(fetch_id);
    }
    let _ = ctx;
    stream_close(ctx, bindings, stream_id)
}

/// The stream object for `stream_id`, if it is still registered.
pub fn stream_object<'js>(
    ctx: &Ctx<'js>,
    stream_id: u64,
) -> rquickjs::Result<Option<Object<'js>>> {
    let map = registry_map(ctx, "streams")?;
    let value = map_get(ctx, &map, stream_id)?;
    Ok(value.as_object().cloned())
}

/// Drop a stream's registry entries (navigation / realm teardown).
/// The bytes are dropped with the state; pending waiters are
/// rejected — a navigation must not leave a `read()` hanging.
pub fn drop_stream<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    stream_id: u64,
) -> rquickjs::Result<()> {
    bindings.borrow_mut().stream_states.remove(&stream_id);
    let streams = registry_map(ctx, "streams")?;
    map_delete(ctx, &streams, stream_id);
    let _ = stream_error(ctx, bindings, stream_id, "stream dropped")?;
    Ok(())
}
