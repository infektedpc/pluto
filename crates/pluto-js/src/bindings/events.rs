//! `EventTarget`, `Event`, `CustomEvent`, and dispatch (binding item 4).
//!
//! Listener storage lives on the realm, in the registry's `listeners`
//! `Map` (key `"<target-id>:<type>"` → `Array` of `{ func, capture,
//! once }` records) — never in the DOM, so navigation drops listeners
//! with the realm. `removeEventListener` compares by JS function
//! identity (`Value` pointer equality). `dispatchEvent` from JS runs
//! synchronously (DOM spec); dispatch of an external event (a shell
//! click) arrives as a task on the user-interaction source — see
//! [`dispatch_external`], driven by `TabJs::dispatch_event`.
//!
//! Each listener callback runs with a microtask checkpoint after it
//! (the engine's pending-job queue is drained from inside the
//! dispatch, one job at a time). A throwing listener is reported
//! into [`BindingState::errors`] and dispatch continues with the
//! next listener.

use std::cell::RefCell;
use std::rc::Rc;

use rquickjs::function::{Opt, This};
use rquickjs::{Array, Coerced, Ctx, FromJs, Function, Object, Value};

use crate::bindings::{
    BindingState, WINDOW_TARGET_ID, dom_key_of_this, js, map_delete_str, map_get_str,
    map_set_str, registry_map, throw_dom, throw_type_error, to_dom_string, wrapper_id_of,
};
use crate::host::DomKey;

/// Install `EventTarget.prototype`, `Event`, and `CustomEvent` and
/// store their prototypes in the registry.
pub fn install_events<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();

    // --- EventTarget.prototype ---
    let target_proto = Object::new(ctx.clone()).map_err(|e| format!("EventTarget proto: {e}"))?;
    let (add, remove, dispatch) = make_listener_functions(ctx, bindings)?;
    target_proto
        .set("addEventListener", add)
        .map_err(|e| format!("addEventListener: {e}"))?;
    target_proto
        .set("removeEventListener", remove)
        .map_err(|e| format!("removeEventListener: {e}"))?;
    target_proto
        .set("dispatchEvent", dispatch)
        .map_err(|e| format!("dispatchEvent: {e}"))?;

    // --- Event / CustomEvent (JS constructors; the objects are pure
    // JS, the dispatch walk in Rust reads/writes their properties) ---
    ctx.eval::<(), _>(
        "globalThis.Event = function Event(type, eventInit) { \
           if (!(this instanceof Event)) throw new TypeError('Event constructor requires new'); \
           this.type = String(type); \
           this.bubbles = !!(eventInit && eventInit.bubbles); \
           this.cancelable = !!(eventInit && eventInit.cancelable); \
           this.composed = !!(eventInit && eventInit.composed); \
           this.target = null; this.currentTarget = null; this.eventPhase = 0; \
           this.defaultPrevented = false; this.cancelBubble = false; \
           this.timeStamp = performance.now(); \
         }; \
         Event.prototype.preventDefault = function () { \
           if (this.cancelable) this.defaultPrevented = true; \
         }; \
         Event.prototype.stopPropagation = function () { this.cancelBubble = true; }; \
         Event.prototype.stopImmediatePropagation = function () { \
           this.cancelBubble = true; this.__pluto_stop_immediate = true; \
         }; \
         globalThis.CustomEvent = function CustomEvent(type, eventInit) { \
           Event.call(this, type, eventInit); \
           this.detail = (eventInit && 'detail' in eventInit) ? eventInit.detail : null; \
         }; \
         CustomEvent.prototype = Object.create(Event.prototype); \
         CustomEvent.prototype.constructor = CustomEvent; \
         globalThis.KeyboardEvent = function KeyboardEvent(type, eventInit) { \
           Event.call(this, type, eventInit); \
           this.key = (eventInit && 'key' in eventInit) ? String(eventInit.key) : ''; \
           this.code = (eventInit && 'code' in eventInit) ? String(eventInit.code) : ''; \
           this.repeat = !!(eventInit && eventInit.repeat); \
         }; \
         KeyboardEvent.prototype = Object.create(Event.prototype); \
         KeyboardEvent.prototype.constructor = KeyboardEvent; \
         globalThis.MouseEvent = function MouseEvent(type, eventInit) { \
           Event.call(this, type, eventInit); \
           this.clientX = (eventInit && 'clientX' in eventInit) ? Number(eventInit.clientX) : 0; \
           this.clientY = (eventInit && 'clientY' in eventInit) ? Number(eventInit.clientY) : 0; \
           this.button = (eventInit && 'button' in eventInit) ? Number(eventInit.button) : 0; \
           this.buttons = (eventInit && 'buttons' in eventInit) ? Number(eventInit.buttons) : 0; \
         }; \
         MouseEvent.prototype = Object.create(Event.prototype); \
         MouseEvent.prototype.constructor = MouseEvent; \
         globalThis.InputEvent = function InputEvent(type, eventInit) { \
           Event.call(this, type, eventInit); \
           this.data = (eventInit && 'data' in eventInit) ? eventInit.data : null; \
           this.inputType = (eventInit && 'inputType' in eventInit) ? String(eventInit.inputType) : ''; \
         }; \
         InputEvent.prototype = Object.create(Event.prototype); \
         InputEvent.prototype.constructor = InputEvent;",
    )
    .map_err(|e| format!("install Event: {e}"))?;

    // Prototypes into the registry for the wrappers to chain from.
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    protos
        .set("eventTarget", target_proto)
        .map_err(|e| format!("prototypes eventTarget: {e}"))?;
    let event_proto: Object = ctx
        .eval("Event.prototype")
        .map_err(|e| format!("Event.prototype: {e}"))?;
    protos
        .set("event", event_proto)
        .map_err(|e| format!("prototypes event: {e}"))?;
    let custom_proto: Object = ctx
        .eval("CustomEvent.prototype")
        .map_err(|e| format!("CustomEvent.prototype: {e}"))?;
    protos
        .set("customEvent", custom_proto)
        .map_err(|e| format!("prototypes customEvent: {e}"))?;
    let _ = globals;
    Ok(())
}

/// Build the three `EventTarget` host functions (shared by the
/// prototype and by `window`, whose global object does not inherit
/// the prototype chain).
fn make_listener_functions<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(Function<'js>, Function<'js>, Function<'js>), String> {
    let b = Rc::clone(bindings);
    let add = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Value<'js>>, ty: Coerced<String>, listener: Value<'js>, options: Opt<Value<'js>>| -> rquickjs::Result<()> {
            add_event_listener(&ctx, &b, this, &ty, listener, options.0)
        },
    )
    .map_err(|e| format!("addEventListener: {e}"))?;
    let b = Rc::clone(bindings);
    let remove = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Value<'js>>, ty: Coerced<String>, listener: Value<'js>, options: Opt<Value<'js>>| -> rquickjs::Result<()> {
            remove_event_listener(&ctx, &b, this, &ty, listener, options.0)
        },
    )
    .map_err(|e| format!("removeEventListener: {e}"))?;
    let b = Rc::clone(bindings);
    let dispatch = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Value<'js>>, event: Value<'js>| -> rquickjs::Result<bool> {
            dispatch_event(&ctx, &b, this, event)
        },
    )
    .map_err(|e| format!("dispatchEvent: {e}"))?;
    Ok((add, remove, dispatch))
}

/// Install the `EventTarget` methods directly on `globalThis`
/// (`window` is an `EventTarget`, but the global object does not
/// inherit `EventTarget.prototype`).
pub fn install_window_listeners<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    let (add, remove, dispatch) = make_listener_functions(ctx, bindings)?;
    globals
        .set("addEventListener", add)
        .map_err(|e| format!("window addEventListener: {e}"))?;
    globals
        .set("removeEventListener", remove)
        .map_err(|e| format!("window removeEventListener: {e}"))?;
    globals
        .set("dispatchEvent", dispatch)
        .map_err(|e| format!("window dispatchEvent: {e}"))?;
    Ok(())
}

/// Resolve the listener-registry target id for a `this` value:
/// the window (global or `undefined`, for bare calls) or a wrapper.
fn target_id_of_this<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: &Value<'js>,
) -> rquickjs::Result<u64> {
    if this.is_undefined() || this.is_null() {
        return Ok(WINDOW_TARGET_ID);
    }
    if let Some(obj) = this.as_object() {
        let globals = ctx.globals();
        if Value::from(obj.clone()) == Value::from(globals) {
            return Ok(WINDOW_TARGET_ID);
        }
        if let Some(id) = wrapper_id_of(ctx, obj) {
            // The wrapper must still be in the cache (a stale wrapper
            // has no live key; dispatching on it is a no-op target).
            if state.borrow().wrapper_keys.contains_key(&id) {
                return Ok(id);
            }
        }
        return Err(throw_type_error(
            ctx,
            "called on an object that is not an EventTarget",
        ));
    }
    Err(throw_type_error(
        ctx,
        "called on an object that is not an EventTarget",
    ))
}

/// The registry key for one target's listeners of one type.
fn listeners_key(target: u64, ty: &str) -> String {
    format!("{target}:{ty}")
}

/// Read the listener array for a key, if any.
fn listener_array<'js>(
    ctx: &Ctx<'js>,
    key: &str,
) -> rquickjs::Result<Option<Array<'js>>> {
    let map = registry_map(ctx, "listeners")?;
    let value = map_get_str(ctx, &map, key)?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    Ok(value.as_object().and_then(|o| o.clone().into_array()))
}

/// Parse `addEventListener` options: `true` → capture; an object →
/// `{ capture, once, passive }`. M4: `passive` is stored but the
/// dispatch does not consult it (no scrolling to cancel yet).
fn parse_options(
    ctx: &Ctx<'_>,
    options: Option<Value<'_>>,
) -> rquickjs::Result<(bool, bool, bool)> {
    let Some(options) = options else {
        return Ok((false, false, false));
    };
    if let Some(b) = options.as_bool() {
        return Ok((b, false, false));
    }
    if let Some(obj) = options.as_object() {
        let capture: bool = obj.get("capture").unwrap_or(false);
        let once: bool = obj.get("once").unwrap_or(false);
        let passive: bool = obj.get("passive").unwrap_or(false);
        return Ok((capture, once, passive));
    }
    let _ = ctx;
    Ok((false, false, false))
}

fn add_event_listener<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: Value<'js>,
    ty: &str,
    listener: Value<'js>,
    options: Option<Value<'js>>,
) -> rquickjs::Result<()> {
    // A null listener is ignored per DOM (not an error).
    if listener.is_null() || listener.is_undefined() || !listener.is_function() {
        return Ok(());
    }
    let (capture, once, passive) = parse_options(ctx, options)?;
    let target = target_id_of_this(ctx, state, &this)?;
    let key = listeners_key(target, ty);
    let map = registry_map(ctx, "listeners")?;
    let mut records: Vec<Value> = Vec::new();
    if let Some(arr) = listener_array(ctx, &key)? {
        let len: usize = arr.len();
        for i in 0..len {
            if let Ok(v) = arr.get::<Value>(i) {
                records.push(v);
            }
        }
    }
    // Duplicate (same function, same capture) is ignored.
    for record in &records {
        if let Some(obj) = record.as_object() {
            let func: Value = obj.get("func")?;
            let cap: bool = obj.get("capture").unwrap_or(false);
            if func == listener && cap == capture {
                return Ok(());
            }
        }
    }
    let record = Object::new(ctx.clone())?;
    record.set("func", listener)?;
    record.set("capture", capture)?;
    record.set("once", once)?;
    record.set("passive", passive)?;
    record.set("type", ty)?;
    records.push(Value::from(record));
    let arr = Array::new(ctx.clone())?;
    for (i, value) in records.into_iter().enumerate() {
        arr.set(i, value)?;
    }
    map_set_str(ctx, &map, &key, Value::from(arr))?;
    Ok(())
}

fn remove_event_listener<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: Value<'js>,
    ty: &str,
    listener: Value<'js>,
    options: Option<Value<'js>>,
) -> rquickjs::Result<()> {
    if listener.is_null() || listener.is_undefined() || !listener.is_function() {
        return Ok(());
    }
    let (capture, _, _) = parse_options(ctx, options)?;
    let target = target_id_of_this(ctx, state, &this)?;
    let key = listeners_key(target, ty);
    let Some(arr) = listener_array(ctx, &key)? else {
        return Ok(());
    };
    let map = registry_map(ctx, "listeners")?;
    let mut kept: Vec<Value> = Vec::new();
    let len: usize = arr.len();
    for i in 0..len {
        let Ok(record) = arr.get::<Value>(i) else {
            continue;
        };
        let mut drop_it = false;
        if let Some(obj) = record.as_object() {
            let func: Value = obj.get("func").unwrap_or_else(|_| js(ctx, ()));
            let cap: bool = obj.get("capture").unwrap_or(false);
            if func == listener && cap == capture {
                drop_it = true;
            }
        }
        if !drop_it {
            kept.push(record);
        }
    }
    if kept.is_empty() {
        map_delete_str(ctx, &map, &key);
    } else {
        let out = Array::new(ctx.clone())?;
        for (i, value) in kept.into_iter().enumerate() {
            out.set(i, value)?;
        }
        map_set_str(ctx, &map, &key, Value::from(out))?;
    }
    Ok(())
}

/// The `type` of an event object, or a `TypeError` if it is not an
/// `Event`-shaped object.
fn event_type<'js>(ctx: &Ctx<'js>, event: &Value<'js>) -> rquickjs::Result<String> {
    let Some(obj) = event.as_object() else {
        return Err(throw_type_error(ctx, "dispatchEvent: argument is not an Event"));
    };
    let ty: Value = obj.get("type").unwrap_or_else(|_| js(ctx, ()));
    if ty.is_undefined() || ty.is_null() {
        return Err(throw_type_error(
            ctx,
            "dispatchEvent: event has no type",
        ));
    }
    to_dom_string(ctx, &ty)
}

/// Synchronous `dispatchEvent` (DOM spec: listeners run during the
/// call). Returns `false` when the event was canceled.
fn dispatch_event<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: Value<'js>,
    event: Value<'js>,
) -> rquickjs::Result<bool> {
    let ty = event_type(ctx, &event)?;
    let Some(event_obj) = event.as_object() else {
        return Err(throw_type_error(ctx, "dispatchEvent: argument is not an Event"));
    };
    if event_obj
        .get::<_, bool>("__pluto_dispatching")
        .unwrap_or(false)
    {
        return Err(throw_dom(
            ctx,
            "InvalidStateError",
            "dispatchEvent: the event is already being dispatched",
        ));
    }
    let target = target_id_of_this(ctx, state, &this)?;
    dispatch_to_target(ctx, state, target, event_obj, &ty)
}

/// Dispatch `event` (already validated) to `target`, walking the
/// propagation path. Shared by `dispatchEvent` and `click()`.
pub fn dispatch_to_target<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    target: u64,
    event_obj: &Object<'js>,
    ty: &str,
) -> rquickjs::Result<bool> {
    let bubbles: bool = event_obj.get("bubbles").unwrap_or(false);
    let cancelable: bool = event_obj.get("cancelable").unwrap_or(false);
    event_obj.set("target", target_object(ctx, state, target)?)?;
    event_obj.set("__pluto_dispatching", true)?;
    let path = propagation_path(ctx, state, target);
    // Phases: capture (window → parent), target, bubble (parent → window).
    let mut stop_after_target = false;
    // Capture: path[0] is the window, path[last] is the target.
    for (i, &node) in path.iter().enumerate() {
        let at_target = i + 1 == path.len();
        let phase: u32 = if at_target { 2 } else { 1 };
        event_obj.set("eventPhase", phase)?;
        event_obj.set(
            "currentTarget",
            target_object(ctx, state, node)?,
        )?;
        let capture_only = !at_target;
        invoke_listeners(ctx, state, node, event_obj, ty, capture_only, at_target)?;
        if event_obj
            .get::<_, bool>("__pluto_stop_immediate")
            .unwrap_or(false)
        {
            stop_after_target = true;
            break;
        }
        if event_obj.get::<_, bool>("cancelBubble").unwrap_or(false) {
            stop_after_target = true;
            break;
        }
        if at_target {
            break;
        }
    }
    if !stop_after_target && bubbles {
        // Bubble: ancestors from the target's parent back to the window.
        for (i, &node) in path.iter().rev().enumerate() {
            if i == 0 {
                continue; // the target already ran (both phases).
            }
            event_obj.set("eventPhase", 3)?;
            event_obj.set("currentTarget", target_object(ctx, state, node)?)?;
            invoke_listeners(ctx, state, node, event_obj, ty, false, false)?;
            if event_obj
                .get::<_, bool>("__pluto_stop_immediate")
                .unwrap_or(false)
            {
                break;
            }
            if event_obj.get::<_, bool>("cancelBubble").unwrap_or(false) {
                break;
            }
        }
    }
    event_obj.set("currentTarget", rquickjs::Null)?;
    event_obj.set("eventPhase", 0)?;
    event_obj.set("__pluto_dispatching", false)?;
    let default_prevented: bool = event_obj.get("defaultPrevented").unwrap_or(false);
    Ok(!(cancelable && default_prevented))
}

/// The JS object that is `currentTarget` for a registry target id:
/// the window (`globalThis`) or the node's wrapper.
fn target_object<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    target: u64,
) -> rquickjs::Result<Value<'js>> {
    if target == WINDOW_TARGET_ID {
        return Ok(Value::from(ctx.globals()));
    }
    let Some(key) = state.borrow().wrapper_keys.get(&target).copied() else {
        return Ok(Value::from(ctx.globals()));
    };
    match crate::bindings::dom::wrap_node(ctx, state, key) {
        Ok(obj) => Ok(Value::from(obj)),
        Err(_) => Ok(Value::from(ctx.globals())),
    }
}

/// The propagation path for `target`: window first, then ancestors
/// from the document root down to the target. The window is always
/// path[0]; the target is always path[last].
fn propagation_path(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    target: u64,
) -> Vec<u64> {
    let mut chain: Vec<u64> = vec![target];
    if target != WINDOW_TARGET_ID {
        let key = state.borrow().wrapper_keys.get(&target).copied();
        if let Some(key) = key {
            let host = state.borrow().host.clone();
            if let Some(host) = host {
                let mut cur = host.borrow().parent(key);
                // Guard against a corrupt parent chain looping.
                let mut guard = 0;
                while let Some(parent) = cur {
                    guard += 1;
                    if guard > 10_000 {
                        break;
                    }
                    // The lookup borrow must end before minting: a
                    // borrow in an `if let` scrutinee lives for the
                    // whole statement, so minting in the else branch
                    // would panic on the still-held borrow.
                    let existing = state.borrow().wrapper_ids.get(&parent).copied();
                    if let Some(id) = existing {
                        chain.push(id);
                    } else {
                        // Ancestor has no wrapper yet — give it one so
                        // listeners registered on it can run. The id
                        // is minted here (wrapping validates the key).
                        let id = mint_wrapper_id(state);
                        state.borrow_mut().wrapper_keys.insert(id, parent);
                        state.borrow_mut().wrapper_ids.insert(parent, id);
                        chain.push(id);
                    }
                    cur = host.borrow().parent(parent);
                }
            }
        }
    }
    // Chain is target → … → root; reverse to root → target, then put
    // the window first.
    chain.reverse();
    let mut path = vec![WINDOW_TARGET_ID];
    path.extend(chain);
    // The document node is the root's wrapper; the window sits above
    // it. If the target IS the document, path is [window, document].
    let _ = ctx;
    path
}

/// Mint a fresh wrapper id (and record nothing yet — the caller fills
/// in the key mapping).
fn mint_wrapper_id(state: &Rc<RefCell<BindingState>>) -> u64 {
    let mut s = state.borrow_mut();
    let id = s.next_wrapper_id;
    s.next_wrapper_id += 1;
    id
}

/// Run the listeners registered on `node` for `ty`. `capture_only`
/// selects the capture listeners (capture phase); `at_target` selects
/// both capture and bubble listeners, in registration order (target
/// phase). A microtask checkpoint runs after each listener.
fn invoke_listeners<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    node: u64,
    event_obj: &Object<'js>,
    ty: &str,
    capture_only: bool,
    at_target: bool,
) -> rquickjs::Result<()> {
    let key = listeners_key(node, ty);
    let Some(arr) = listener_array(ctx, &key)? else {
        return Ok(());
    };
    let len: usize = arr.len();
    let mut records: Vec<(Value, bool, bool)> = Vec::new();
    for i in 0..len {
        let Ok(record) = arr.get::<Value>(i) else {
            continue;
        };
        if let Some(obj) = record.as_object() {
            let func: Value = obj.get("func").unwrap_or_else(|_| js(ctx, ()));
            let capture: bool = obj.get("capture").unwrap_or(false);
            let once: bool = obj.get("once").unwrap_or(false);
            if !func.is_function() {
                continue;
            }
            if at_target || capture == capture_only {
                records.push((func, capture, once));
            }
        }
    }
    let mut ran_once: Vec<Value> = Vec::new();
    for (func, _capture, once) in records {
        if event_obj
            .get::<_, bool>("__pluto_stop_immediate")
            .unwrap_or(false)
        {
            break;
        }
        let func = match Function::from_js(ctx, func) {
            Ok(f) => f,
            Err(_) => continue,
        };
        // `this` is the currentTarget.
        let this: Value = event_obj.get("currentTarget")?;
        let result: rquickjs::Result<Value> =
            func.call((This(this), Value::from(event_obj.clone())));
        if once {
            ran_once.push(Value::from(func.clone()));
        }
        match result {
            Ok(_) => {}
            Err(_) => {
                // A throwing listener is reported and dispatch
                // continues with the next listener (DOM "invoke").
                // The pending exception carries the real message;
                // rquickjs's `Error` rendering alone is generic.
                let caught = ctx.catch();
                let msg = caught
                    .as_object()
                    .and_then(|o| o.get::<_, String>("message").ok())
                    .unwrap_or_else(|| "listener threw".to_string());
                state
                    .borrow_mut()
                    .errors
                    .push(format!("event listener threw: {msg}"));
            }
        }
        // Microtask checkpoint after each listener: drain the
        // engine's pending jobs (promise reactions the listener
        // queued) before the next listener runs.
        while ctx.execute_pending_job() {}
    }
    // `once` listeners fire at most once: drop the ones that ran.
    if !ran_once.is_empty() {
        let map = registry_map(ctx, "listeners")?;
        let Some(arr) = listener_array(ctx, &key)? else {
            return Ok(());
        };
        let mut kept: Vec<Value> = Vec::new();
        let len: usize = arr.len();
        for i in 0..len {
            let Ok(record) = arr.get::<Value>(i) else {
                continue;
            };
            let mut drop_it = false;
            if let Some(obj) = record.as_object() {
                let func: Value = obj.get("func").unwrap_or_else(|_| js(ctx, ()));
                let once: bool = obj.get("once").unwrap_or(false);
                if once && ran_once.contains(&func) {
                    drop_it = true;
                }
            }
            if !drop_it {
                kept.push(record);
            }
        }
        if kept.is_empty() {
            map_delete_str(ctx, &map, &key);
        } else {
            let out = Array::new(ctx.clone())?;
            for (i, value) in kept.into_iter().enumerate() {
                out.set(i, value)?;
            }
            map_set_str(ctx, &map, &key, Value::from(out))?;
        }
    }
    Ok(())
}

/// Dispatch an external event (a shell click, a key press) to a node:
/// builds an `Event` of `ty` and runs the dispatch walk. Called from
/// a `UserInteraction`-source Host task via `TabJs::dispatch_event`.
pub fn dispatch_external<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    target: DomKey,
    ty: &str,
    bubbles: bool,
) -> rquickjs::Result<()> {
    // The lookup borrow must end before minting: a `borrow()` in
    // the method chain lives through `unwrap_or_else`, so minting
    // inside it would panic on the still-held borrow.
    let existing = state.borrow().wrapper_ids.get(&target).copied();
    let id = match existing {
        Some(id) => id,
        None => {
            let id = mint_wrapper_id(state);
            state.borrow_mut().wrapper_keys.insert(id, target);
            state.borrow_mut().wrapper_ids.insert(target, id);
            id
        }
    };
    let event: Object = ctx.eval::<Object, _>(format!(
        "new Event({ty:?}, {{ bubbles: {bubbles}, cancelable: true }})"
    ))?;
    dispatch_to_target(ctx, state, id, &event, ty)?;
    Ok(())
}

/// The typed payload of a shell-dispatched event: which `Event`
/// subclass to construct, and its fields. `TabJs` builds one of
/// these from the shell's input event; the shell call-site wiring
/// (renderer input dispatch) is outside `pluto-js` — see
/// `TabJs::dispatch_typed_event`.
#[derive(Debug, Clone)]
pub enum TypedEventKind {
    /// `KeyboardEvent`: `key` (e.g. `"a"`, `"Enter"`) and `code`
    /// (e.g. `"KeyA"`).
    Keyboard {
        key: String,
        code: String,
    },
    /// `MouseEvent`: viewport coordinates and the button pressed.
    Mouse {
        client_x: f64,
        client_y: f64,
        button: i64,
    },
    /// `InputEvent`: the inserted text (`None` = `null`, e.g. a
    /// deletion). Listeners read the value off `event.target`
    /// (an `<input>`'s `value` is M5 form work — the event itself
    /// carries `data` here).
    Input { data: Option<String> },
}

/// Dispatch an external *typed* event to a node: builds the right
/// `Event` subclass (`KeyboardEvent` / `MouseEvent` / `InputEvent`)
/// with its fields and runs the dispatch walk. Called from a
/// `UserInteraction`-source Host task via
/// `TabJs::dispatch_typed_event`.
pub fn dispatch_typed_external<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    target: DomKey,
    ty: &str,
    bubbles: bool,
    kind: &TypedEventKind,
) -> rquickjs::Result<()> {
    let existing = state.borrow().wrapper_ids.get(&target).copied();
    let id = match existing {
        Some(id) => id,
        None => {
            let id = mint_wrapper_id(state);
            state.borrow_mut().wrapper_keys.insert(id, target);
            state.borrow_mut().wrapper_ids.insert(target, id);
            id
        }
    };
    let event: Object = match kind {
        TypedEventKind::Keyboard { key, code } => ctx.eval::<Object, _>(format!(
            "new KeyboardEvent({ty:?}, {{ bubbles: {bubbles}, cancelable: true, key: {key:?}, code: {code:?} }})"
        ))?,
        TypedEventKind::Mouse {
            client_x,
            client_y,
            button,
        } => ctx.eval::<Object, _>(format!(
            "new MouseEvent({ty:?}, {{ bubbles: {bubbles}, cancelable: true, clientX: {client_x}, clientY: {client_y}, button: {button} }})"
        ))?,
        TypedEventKind::Input { data } => {
            let data_js = match data {
                Some(d) => format!("{d:?}"),
                None => "null".to_string(),
            };
            ctx.eval::<Object, _>(format!(
                "new InputEvent({ty:?}, {{ bubbles: {bubbles}, cancelable: false, data: {data_js} }})"
            ))?
        }
    };
    dispatch_to_target(ctx, state, id, &event, ty)?;
    Ok(())
}

/// `this` validation helper shared with `dom.rs`: the `DomKey` for a
/// node wrapper, or a `TypeError`.
pub fn node_key_of_this<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    this: &Object<'js>,
) -> rquickjs::Result<DomKey> {
    dom_key_of_this(ctx, state, this)
}
