//! `MutationObserver` / `ResizeObserver` / `IntersectionObserver`
//! bindings (Stage 8, Recommendations 7–8, 10).
//!
//! The JS surface for all three observers. The pattern follows the rest
//! of the bindings: the constructors are small JS functions (like
//! `Event` in `events.rs`) that call hidden Rust host functions
//! hung off the constructor object (`MutationObserver.__pluto_*` /
//! `ResizeObserver.__pluto_*` / `IntersectionObserver.__pluto_*`);
//! the Rust side owns the ids and the registries, the callbacks live
//! in the registry's `*Observers` / `*ObserverInstances` Maps (with
//! the observer instance in the matching `*Instances` Map, so delivery
//! can pass the observer as the callback's second argument).
//!
//! Delivery:
//! * `MutationObserver` — the `Dom` queues records (see
//!   `pluto-html::observers`); [`deliver_mutation_observers`]
//!   drains them, one compound callback per observer per
//!   microtask checkpoint (called from `TabJs`'s checkpoint — see
//!   the ordering note there).
//! * `ResizeObserver` — the registry lives in [`BindingState`]
//!   (`resize_registry`); `TabJs` drives its gather/broadcast loop
//!   from the rendering step and calls
//!   [`deliver_resize_entries`] per batch.
//! * `IntersectionObserver` — the registry lives in [`BindingState`]
//!   (`intersection_registry`); the worker's rendering step gathers
//!   entries from layout-tree geometry and `TabJs` enqueues ONE host
//!   task that drains the pending queue via
//!   [`deliver_intersection_entries`] — so the callbacks run after
//!   paint, on a later turn, never inline in the step.

use std::cell::RefCell;
use std::rc::Rc;

use rquickjs::function::{Opt, This};
use rquickjs::{Array, CatchResultExt, Ctx, FromJs, Function, Object, Value};

use crate::bindings::dom::{host_of, wrap_node};
use crate::bindings::{
    BindingState, js, js_null, map_get, map_set, registry_map, throw_named, throw_type_error,
};
use crate::error::JsError;
use crate::host::DomKey;
use crate::observers::{
    IntersectionObserverEntry, ObservedSizes, ResizeBoxKind, ResizeObserverEntry, RootMargin,
};
use crate::realm::TabRealm;
use pluto_html::observers::{MutationObserverOptions, ObserverMutationRecord};

/// Install the `MutationObserver`, `ResizeObserver`, and
/// `IntersectionObserver` globals.
pub fn install_observers<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    install_mutation_observer(ctx, bindings)?;
    install_resize_observer(ctx, bindings)?;
    install_intersection_observer(ctx, bindings)?;
    Ok(())
}

// ------------------------------------------------------------------ helpers

/// Read a boolean option (`false` when absent / not a boolean).
fn opt_bool(obj: &Object, name: &str) -> bool {
    obj.get::<_, Value>(name)
        .map(|v| v.as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Read a string option.
fn opt_string(obj: &Object, name: &str) -> Option<String> {
    obj.get::<_, Value>(name)
        .ok()
        .and_then(|v| v.as_string().and_then(|s| s.to_string().ok()))
}

/// Read a string-array option (`attributeFilter`): `None` when
/// absent / not an array.
fn opt_string_array(obj: &Object, name: &str) -> Option<Vec<String>> {
    let value = obj.get::<_, Value>(name).ok()?;
    let arr: Object = value.as_object()?.clone();
    let len: u32 = arr.get("length").ok()?;
    let mut out = Vec::new();
    for i in 0..len {
        let item: Value = arr.get(i.to_string()).ok()?;
        if let Some(s) = item.as_string().and_then(|s| s.to_string().ok()) {
            out.push(s);
        }
    }
    Some(out)
}

/// The `DomKey` behind a wrapper `Value`, or a `TypeError` when the
/// value is not a node wrapper.
fn dom_key_of_value<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    value: &Value<'js>,
) -> rquickjs::Result<DomKey> {
    let Some(obj) = value.as_object() else {
        return Err(throw_type_error(ctx, "expected a Node"));
    };
    crate::bindings::dom_key_of_this(ctx, state, obj)
}

/// Wrap a node as a JS value, or `null` when the key is stale.
fn wrap_or_null<'js>(ctx: &Ctx<'js>, state: &Rc<RefCell<BindingState>>, key: DomKey) -> Value<'js> {
    wrap_node(ctx, state, key)
        .map(Value::from)
        .unwrap_or_else(|_| js_null(ctx))
}

/// One `MutationRecord` as a plain JS object.
fn mutation_record_to_js<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    rec: &ObserverMutationRecord,
) -> rquickjs::Result<Object<'js>> {
    let obj = Object::new(ctx.clone())?;
    obj.set("type", rec.kind.type_name())?;
    obj.set("target", wrap_or_null(ctx, state, rec.target))?;
    let added = Array::new(ctx.clone())?;
    for (i, node) in rec.added_nodes.iter().enumerate() {
        added.set(i, wrap_or_null(ctx, state, *node))?;
    }
    obj.set("addedNodes", added)?;
    let removed = Array::new(ctx.clone())?;
    for (i, node) in rec.removed_nodes.iter().enumerate() {
        removed.set(i, wrap_or_null(ctx, state, *node))?;
    }
    obj.set("removedNodes", removed)?;
    obj.set(
        "previousSibling",
        rec.previous_sibling
            .map(|k| wrap_or_null(ctx, state, k))
            .unwrap_or_else(|| js_null(ctx)),
    )?;
    obj.set(
        "nextSibling",
        rec.next_sibling
            .map(|k| wrap_or_null(ctx, state, k))
            .unwrap_or_else(|| js_null(ctx)),
    )?;
    obj.set(
        "attributeName",
        rec.attribute_name
            .clone()
            .map(|n| js(ctx, n))
            .unwrap_or_else(|| js_null(ctx)),
    )?;
    // `attributeNamespace` is always null in M4 (no namespaces).
    obj.set("attributeNamespace", js_null(ctx))?;
    obj.set(
        "oldValue",
        rec.old_value
            .clone()
            .map(|v| js(ctx, v))
            .unwrap_or_else(|| js_null(ctx)),
    )?;
    Ok(obj)
}

/// One `ResizeObserverEntry` as a plain JS object.
fn resize_entry_to_js<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    entry: &ResizeObserverEntry,
) -> rquickjs::Result<Object<'js>> {
    let obj = Object::new(ctx.clone())?;
    obj.set("target", wrap_or_null(ctx, state, entry.target))?;
    let s = entry.sizes;
    // Legacy `contentRect`: content box, origin at the padding
    // offset from the border-box origin.
    let rect = Object::new(ctx.clone())?;
    rect.set("x", s.content_offset.0)?;
    rect.set("y", s.content_offset.1)?;
    rect.set("width", s.content.0)?;
    rect.set("height", s.content.1)?;
    rect.set("top", s.content_offset.1)?;
    rect.set("left", s.content_offset.0)?;
    rect.set("right", s.content_offset.0 + s.content.0)?;
    rect.set("bottom", s.content_offset.1 + s.content.1)?;
    obj.set("contentRect", rect)?;
    let border = box_size_array(ctx, s.border)?;
    obj.set("borderBoxSize", border)?;
    let content = box_size_array(ctx, s.content)?;
    obj.set("contentBoxSize", content)?;
    let device = box_size_array(ctx, s.device_pixel)?;
    obj.set("devicePixelContentBoxSize", device)?;
    Ok(obj)
}

/// A one-element `ResizeObserverSize` array (`[{inlineSize,
/// blockSize}]`).
fn box_size_array<'js>(ctx: &Ctx<'js>, size: (f64, f64)) -> rquickjs::Result<Array<'js>> {
    let arr = Array::new(ctx.clone())?;
    let entry = Object::new(ctx.clone())?;
    entry.set("inlineSize", size.0)?;
    entry.set("blockSize", size.1)?;
    arr.set(0, Value::from(entry))?;
    Ok(arr)
}

/// Call an observer callback stored in a registry Map: look the
/// callback + instance up by `observer_id`, call
/// `callback(entries, observer)`. A missing callback is a silent
/// no-op (disconnected after registration). Returns the
/// callback's throw as a string, if it threw.
fn call_observer_callback<'js>(
    ctx: &Ctx<'js>,
    map_name: &str,
    instance_map_name: &str,
    observer_id: u64,
    arg: Value<'js>,
) -> Result<(), String> {
    let map = registry_map(ctx, map_name).map_err(|e| format!("{map_name} map: {e}"))?;
    let cb_val = map_get(ctx, &map, observer_id).map_err(|e| format!("{map_name} lookup: {e}"))?;
    if !cb_val.is_function() {
        return Ok(());
    }
    let func = Function::from_js(ctx, cb_val).map_err(|e| format!("callback: {e}"))?;
    let instances = registry_map(ctx, instance_map_name)
        .map_err(|e| format!("{instance_map_name} map: {e}"))?;
    let instance = map_get(ctx, &instances, observer_id)
        .map_err(|e| format!("{instance_map_name} lookup: {e}"))?;
    let result: rquickjs::Result<Value> = func.call((This(ctx.globals()), arg, instance));
    match result.catch(ctx) {
        Ok(_) => Ok(()),
        Err(caught) => Err(caught.to_string()),
    }
}

// ------------------------------------------------------------------ MutationObserver

fn install_mutation_observer<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    ctx.eval::<(), _>(
        "globalThis.MutationObserver = function MutationObserver(cb) { \
           if (!(this instanceof MutationObserver)) \
             throw new TypeError('MutationObserver constructor requires new'); \
           if (typeof cb !== 'function') \
             throw new TypeError('MutationObserver: callback is not a function'); \
           const id = MutationObserver.__pluto_new(cb, this); \
           Object.defineProperty(this, '__pluto_id', { value: id }); \
         }; \
         MutationObserver.prototype.observe = function (target, options) { \
           return MutationObserver.__pluto_observe(this.__pluto_id, target, options || {}); \
         }; \
         MutationObserver.prototype.takeRecords = function () { \
           return MutationObserver.__pluto_take_records(this.__pluto_id); \
         }; \
         MutationObserver.prototype.disconnect = function () { \
           return MutationObserver.__pluto_disconnect(this.__pluto_id); \
         };",
    )
    .map_err(|e| format!("install MutationObserver: {e}"))?;
    let ctor: Object = ctx
        .globals()
        .get("MutationObserver")
        .map_err(|e| format!("MutationObserver global: {e}"))?;

    // __pluto_new(callback, instance) -> id
    let b = Rc::clone(bindings);
    let new_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>, instance: Object<'js>| -> rquickjs::Result<f64> {
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_mutation_observer_id;
                s.next_mutation_observer_id += 1;
                id
            };
            let map = registry_map(&ctx, "mutationObservers")?;
            map_set(&ctx, &map, id, js(&ctx, cb))?;
            let instances = registry_map(&ctx, "mutationObserverInstances")?;
            map_set(&ctx, &instances, id, Value::from(instance))?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install __pluto_new mutation observer: {e}"))?;
    ctor.set("__pluto_new", new_fn)
        .map_err(|e| format!("install __pluto_new mutation observer: {e}"))?;

    // __pluto_observe(id, target, options)
    let b = Rc::clone(bindings);
    let observe_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>,
              id: f64,
              target: Value<'js>,
              options: Opt<Value<'js>>|
              -> rquickjs::Result<()> {
            let key = dom_key_of_value(&ctx, &b, &target)?;
            let opts = options
                .0
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_else(|| Object::new(ctx.clone()).unwrap());
            let parsed = MutationObserverOptions {
                child_list: opt_bool(&opts, "childList"),
                attributes: opt_bool(&opts, "attributes"),
                character_data: opt_bool(&opts, "characterData"),
                subtree: opt_bool(&opts, "subtree"),
                attribute_old_value: opt_bool(&opts, "attributeOldValue"),
                character_data_old_value: opt_bool(&opts, "characterDataOldValue"),
                attribute_filter: opt_string_array(&opts, "attributeFilter"),
            };
            if !parsed.observes_any() {
                return Err(throw_type_error(
                    &ctx,
                    "MutationObserver.observe: at least one of childList, attributes, or characterData must be true",
                ));
            }
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host
                .borrow_mut()
                .observe_mutation(id as u64, key, parsed)
            {
                return Err(throw_type_error(
                    &ctx,
                    "MutationObserver.observe: target is not in the document",
                ));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_observe: {e}"))?;
    ctor.set("__pluto_observe", observe_fn)
        .map_err(|e| format!("install __pluto_observe: {e}"))?;

    // __pluto_take_records(id) -> [MutationRecord]
    let b = Rc::clone(bindings);
    let take_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64| -> rquickjs::Result<Value<'js>> {
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let records = host.borrow_mut().take_mutation_records(id as u64);
            let arr = Array::new(ctx.clone())?;
            for (i, rec) in records.iter().enumerate() {
                let obj = mutation_record_to_js(&ctx, &b, rec)?;
                arr.set(i, Value::from(obj))?;
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("install __pluto_take_records: {e}"))?;
    ctor.set("__pluto_take_records", take_fn)
        .map_err(|e| format!("install __pluto_take_records: {e}"))?;

    // __pluto_disconnect(id)
    let b = Rc::clone(bindings);
    let disconnect_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64| -> rquickjs::Result<()> {
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut().disconnect_mutation_observer(id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_disconnect: {e}"))?;
    ctor.set("__pluto_disconnect", disconnect_fn)
        .map_err(|e| format!("install __pluto_disconnect: {e}"))?;
    Ok(())
}

/// Deliver every pending mutation observer's queued records: one
/// compound callback per observer, in pending order. Called from
/// `TabJs`'s microtask checkpoint. A throwing callback is
/// collected (not fatal): delivery continues with the next
/// observer, and the first throw is returned for the loop to
/// record as a script error.
pub(crate) fn deliver_mutation_observers(
    realm: &mut TabRealm,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), JsError> {
    let host = match bindings.borrow().host.clone() {
        Some(h) => h,
        None => return Ok(()),
    };
    let pending = host.borrow_mut().take_pending_mutation_observers();
    if pending.is_empty() {
        return Ok(());
    }
    let mut first_error: Option<String> = None;
    for id in pending {
        let records = host.borrow_mut().take_mutation_records(id);
        if records.is_empty() {
            continue;
        }
        let b = Rc::clone(bindings);
        let result = realm.with_ctx(|ctx| -> Result<(), String> {
            let arr = Array::new(ctx.clone()).map_err(|e| e.to_string())?;
            for (i, rec) in records.iter().enumerate() {
                let obj = mutation_record_to_js(&ctx, &b, rec).map_err(|e| e.to_string())?;
                arr.set(i, Value::from(obj)).map_err(|e| e.to_string())?;
            }
            call_observer_callback(
                &ctx,
                "mutationObservers",
                "mutationObserverInstances",
                id,
                Value::from(arr),
            )
        });
        if let Err(e) = result
            && first_error.is_none()
        {
            first_error = Some(e);
        }
    }
    match first_error {
        Some(e) => Err(JsError::Exception(e)),
        None => Ok(()),
    }
}

// ------------------------------------------------------------------ ResizeObserver

fn install_resize_observer<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    ctx.eval::<(), _>(
        "globalThis.ResizeObserver = function ResizeObserver(cb) { \
           if (!(this instanceof ResizeObserver)) \
             throw new TypeError('ResizeObserver constructor requires new'); \
           if (typeof cb !== 'function') \
             throw new TypeError('ResizeObserver: callback is not a function'); \
           const id = ResizeObserver.__pluto_new(cb, this); \
           Object.defineProperty(this, '__pluto_id', { value: id }); \
         }; \
         ResizeObserver.prototype.observe = function (target, options) { \
           return ResizeObserver.__pluto_observe(this.__pluto_id, target, options || {}); \
         }; \
         ResizeObserver.prototype.unobserve = function (target) { \
           return ResizeObserver.__pluto_unobserve(this.__pluto_id, target); \
         }; \
         ResizeObserver.prototype.disconnect = function () { \
           return ResizeObserver.__pluto_disconnect(this.__pluto_id); \
         };",
    )
    .map_err(|e| format!("install ResizeObserver: {e}"))?;
    let ctor: Object = ctx
        .globals()
        .get("ResizeObserver")
        .map_err(|e| format!("ResizeObserver global: {e}"))?;

    // __pluto_new(callback, instance) -> id
    let b = Rc::clone(bindings);
    let new_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>, instance: Object<'js>| -> rquickjs::Result<f64> {
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_resize_observer_id;
                s.next_resize_observer_id += 1;
                id
            };
            let map = registry_map(&ctx, "resizeObservers")?;
            map_set(&ctx, &map, id, js(&ctx, cb))?;
            let instances = registry_map(&ctx, "resizeObserverInstances")?;
            map_set(&ctx, &instances, id, Value::from(instance))?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install __pluto_new resize observer: {e}"))?;
    ctor.set("__pluto_new", new_fn)
        .map_err(|e| format!("install __pluto_new resize observer: {e}"))?;

    // __pluto_observe(id, target, options)
    let b = Rc::clone(bindings);
    let observe_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>,
              id: f64,
              target: Value<'js>,
              options: Opt<Value<'js>>|
              -> rquickjs::Result<()> {
            let key = dom_key_of_value(&ctx, &b, &target)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            // ResizeObserver only observes Elements.
            if host.borrow().node_type(key) != Some(1) {
                return Err(throw_type_error(
                    &ctx,
                    "ResizeObserver.observe: target is not an Element",
                ));
            }
            let opts = options
                .0
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_else(|| Object::new(ctx.clone()).unwrap());
            let kind = match opt_string(&opts, "box").as_deref() {
                None => ResizeBoxKind::ContentBox,
                Some(name) => ResizeBoxKind::parse(name).ok_or_else(|| {
                    throw_type_error(&ctx, "ResizeObserver.observe: invalid box option")
                })?,
            };
            b.borrow_mut().resize_registry.observe(id as u64, key, kind);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_observe resize: {e}"))?;
    ctor.set("__pluto_observe", observe_fn)
        .map_err(|e| format!("install __pluto_observe resize: {e}"))?;

    // __pluto_unobserve(id, target)
    let b = Rc::clone(bindings);
    let unobserve_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64, target: Value<'js>| -> rquickjs::Result<()> {
            let key = dom_key_of_value(&ctx, &b, &target)?;
            b.borrow_mut().resize_registry.unobserve(id as u64, key);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_unobserve: {e}"))?;
    ctor.set("__pluto_unobserve", unobserve_fn)
        .map_err(|e| format!("install __pluto_unobserve: {e}"))?;

    // __pluto_disconnect(id)
    let b = Rc::clone(bindings);
    let disconnect_fn = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, id: f64| -> rquickjs::Result<()> {
            b.borrow_mut().resize_registry.disconnect(id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_disconnect resize: {e}"))?;
    ctor.set("__pluto_disconnect", disconnect_fn)
        .map_err(|e| format!("install __pluto_disconnect resize: {e}"))?;
    Ok(())
}

/// Deliver one `ResizeObserver` batch: invoke the observer's
/// callback with `(entries, observer)`. A throwing callback is
/// returned for the loop to record as a script error.
pub(crate) fn deliver_resize_entries(
    realm: &mut TabRealm,
    bindings: &Rc<RefCell<BindingState>>,
    observer_id: u64,
    entries: &[ResizeObserverEntry],
) -> Result<(), JsError> {
    let b = Rc::clone(bindings);
    let entries: Vec<ResizeObserverEntry> = entries.to_vec();
    realm
        .with_ctx(|ctx| -> Result<(), String> {
            let arr = Array::new(ctx.clone()).map_err(|e| e.to_string())?;
            for (i, entry) in entries.iter().enumerate() {
                let obj = resize_entry_to_js(&ctx, &b, entry).map_err(|e| e.to_string())?;
                arr.set(i, Value::from(obj)).map_err(|e| e.to_string())?;
            }
            call_observer_callback(
                &ctx,
                "resizeObservers",
                "resizeObserverInstances",
                observer_id,
                Value::from(arr),
            )
        })
        .map_err(JsError::Exception)
}

/// The observed sizes of one target, for the `TabJs` step that
/// drives the registry from the host: border box from
/// `geometry()`, content box from `content_size()`.
pub(crate) fn host_observed_sizes(
    host: &Rc<RefCell<Box<dyn crate::host::DomHost>>>,
    key: DomKey,
) -> Option<ObservedSizes> {
    let mut host = host.borrow_mut();
    let geometry = host.geometry(key)?;
    let content = host
        .content_size(key)
        .unwrap_or((geometry.width, geometry.height));
    Some(ObservedSizes {
        border: (geometry.width, geometry.height),
        content,
        // No device-pixel-ratio plumbing in Pluto yet: device px
        // equal CSS px.
        device_pixel: content,
        content_offset: (0.0, 0.0),
    })
}

// ------------------------------------------------------------------ IntersectionObserver

/// One `[x, y, width, height]` rect as a `DOMRectReadOnly`-shaped
/// plain object (`x` / `y` / `width` / `height` / `top` / `left` /
/// `right` / `bottom`).
fn dom_rect<'js>(ctx: &Ctx<'js>, rect: [f64; 4]) -> rquickjs::Result<Object<'js>> {
    let obj = Object::new(ctx.clone())?;
    obj.set("x", rect[0])?;
    obj.set("y", rect[1])?;
    obj.set("width", rect[2])?;
    obj.set("height", rect[3])?;
    obj.set("top", rect[1])?;
    obj.set("left", rect[0])?;
    obj.set("right", rect[0] + rect[2])?;
    obj.set("bottom", rect[1] + rect[3])?;
    Ok(obj)
}

/// One `IntersectionObserverEntry` as a plain JS object. `time` is
/// the computation timestamp (ms since the tab started), not the
/// delivery time. `rootBounds` is always a rect in Pluto (same
/// origin only — no cross-origin null root).
fn intersection_entry_to_js<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    entry: &IntersectionObserverEntry,
) -> rquickjs::Result<Object<'js>> {
    let obj = Object::new(ctx.clone())?;
    obj.set("target", wrap_or_null(ctx, state, entry.target))?;
    obj.set("time", entry.time)?;
    obj.set("rootBounds", dom_rect(ctx, entry.root_bounds))?;
    obj.set("boundingClientRect", dom_rect(ctx, entry.bounding_client_rect))?;
    obj.set("intersectionRect", dom_rect(ctx, entry.intersection_rect))?;
    obj.set("intersectionRatio", entry.ratio)?;
    obj.set("isIntersecting", entry.is_intersecting)?;
    Ok(obj)
}

/// Read the `threshold` option: a number, or an array of numbers.
/// `None` when absent. Each value must be in 0..=1 (spec throws
/// `RangeError` — see the observe handler).
fn opt_thresholds(obj: &Object) -> Option<Vec<f64>> {
    let value = obj.get::<_, Value>("threshold").ok()?;
    if value.is_number() {
        return value.as_number().map(|n| vec![n]);
    }
    let arr = value.as_object()?;
    let len: u32 = arr.get("length").ok()?;
    let mut out = Vec::new();
    for i in 0..len {
        let item: Value = arr.get(i.to_string()).ok()?;
        out.push(item.as_number()?);
    }
    Some(out)
}

fn install_intersection_observer<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    ctx.eval::<(), _>(
        "globalThis.IntersectionObserver = function IntersectionObserver(cb, options) { \
           if (!(this instanceof IntersectionObserver)) \
             throw new TypeError('IntersectionObserver constructor requires new'); \
           if (typeof cb !== 'function') \
             throw new TypeError('IntersectionObserver: callback is not a function'); \
           const id = IntersectionObserver.__pluto_new(cb, this); \
           Object.defineProperty(this, '__pluto_id', { value: id }); \
         }; \
         IntersectionObserver.prototype.observe = function (target, options) { \
           return IntersectionObserver.__pluto_observe(this.__pluto_id, target, options || {}); \
         }; \
         IntersectionObserver.prototype.unobserve = function (target) { \
           return IntersectionObserver.__pluto_unobserve(this.__pluto_id, target); \
         }; \
         IntersectionObserver.prototype.disconnect = function () { \
           return IntersectionObserver.__pluto_disconnect(this.__pluto_id); \
         }; \
         IntersectionObserver.prototype.takeRecords = function () { \
           return IntersectionObserver.__pluto_take_records(this.__pluto_id); \
         };",
    )
    .map_err(|e| format!("install IntersectionObserver: {e}"))?;
    let ctor: Object = ctx
        .globals()
        .get("IntersectionObserver")
        .map_err(|e| format!("IntersectionObserver global: {e}"))?;

    // __pluto_new(callback, instance) -> id
    let b = Rc::clone(bindings);
    let new_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, cb: Function<'js>, instance: Object<'js>| -> rquickjs::Result<f64> {
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_intersection_observer_id;
                s.next_intersection_observer_id += 1;
                id
            };
            let map = registry_map(&ctx, "intersectionObservers")?;
            map_set(&ctx, &map, id, js(&ctx, cb))?;
            let instances = registry_map(&ctx, "intersectionObserverInstances")?;
            map_set(&ctx, &instances, id, Value::from(instance))?;
            Ok(id as f64)
        },
    )
    .map_err(|e| format!("install __pluto_new intersection observer: {e}"))?;
    ctor.set("__pluto_new", new_fn)
        .map_err(|e| format!("install __pluto_new intersection observer: {e}"))?;

    // __pluto_observe(id, target, options)
    let b = Rc::clone(bindings);
    let observe_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>,
              id: f64,
              target: Value<'js>,
              options: Opt<Value<'js>>|
              -> rquickjs::Result<()> {
            let key = dom_key_of_value(&ctx, &b, &target)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            // IntersectionObserver only observes Elements.
            if host.borrow().node_type(key) != Some(1) {
                return Err(throw_type_error(
                    &ctx,
                    "IntersectionObserver.observe: target is not an Element",
                ));
            }
            let opts = options
                .0
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_else(|| Object::new(ctx.clone()).unwrap());
            // `root`: null / undefined = the viewport; otherwise an
            // Element. A root that is not an ancestor of the target
            // is NOT an error — the geometry step reports it as
            // never intersecting (the initial entry still fires).
            let root = match opts.get::<_, Value>("root").ok() {
                Some(v) if !v.is_null() && !v.is_undefined() => {
                    let root_key = dom_key_of_value(&ctx, &b, &v)?;
                    if host.borrow().node_type(root_key) != Some(1) {
                        return Err(throw_type_error(
                            &ctx,
                            "IntersectionObserver.observe: root is not an Element",
                        ));
                    }
                    Some(root_key)
                }
                _ => None,
            };
            let root_margin = match opt_string(&opts, "rootMargin") {
                None => RootMargin::default(),
                Some(s) => RootMargin::parse(&s).ok_or_else(|| {
                    throw_type_error(&ctx, "IntersectionObserver.observe: invalid rootMargin")
                })?,
            };
            let thresholds = match opt_thresholds(&opts) {
                None => vec![0.0],
                Some(ts) => {
                    for t in &ts {
                        if !(0.0..=1.0).contains(t) {
                            return Err(throw_named(
                                &ctx,
                                "RangeError",
                                "IntersectionObserver.observe: threshold must be between 0 and 1",
                            ));
                        }
                    }
                    ts
                }
            };
            b.borrow_mut().intersection_registry.observe(
                id as u64,
                key,
                root,
                root_margin,
                thresholds,
            );
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_observe intersection: {e}"))?;
    ctor.set("__pluto_observe", observe_fn)
        .map_err(|e| format!("install __pluto_observe intersection: {e}"))?;

    // __pluto_unobserve(id, target)
    let b = Rc::clone(bindings);
    let unobserve_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64, target: Value<'js>| -> rquickjs::Result<()> {
            let key = dom_key_of_value(&ctx, &b, &target)?;
            b.borrow_mut().intersection_registry.unobserve(id as u64, key);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_unobserve intersection: {e}"))?;
    ctor.set("__pluto_unobserve", unobserve_fn)
        .map_err(|e| format!("install __pluto_unobserve intersection: {e}"))?;

    // __pluto_disconnect(id)
    let b = Rc::clone(bindings);
    let disconnect_fn = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, id: f64| -> rquickjs::Result<()> {
            b.borrow_mut().intersection_registry.disconnect(id as u64);
            Ok(())
        },
    )
    .map_err(|e| format!("install __pluto_disconnect intersection: {e}"))?;
    ctor.set("__pluto_disconnect", disconnect_fn)
        .map_err(|e| format!("install __pluto_disconnect intersection: {e}"))?;

    // __pluto_take_records(id) -> entries gathered since the last delivery
    let b = Rc::clone(bindings);
    let take_records_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64| -> rquickjs::Result<Array<'js>> {
            let entries = b.borrow_mut().intersection_registry.take_records(id as u64);
            let arr = Array::new(ctx.clone())?;
            for (i, entry) in entries.iter().enumerate() {
                let obj = intersection_entry_to_js(&ctx, &b, entry)?;
                arr.set(i, Value::from(obj))?;
            }
            Ok(arr)
        },
    )
    .map_err(|e| format!("install __pluto_take_records intersection: {e}"))?;
    ctor.set("__pluto_take_records", take_records_fn)
        .map_err(|e| format!("install __pluto_take_records intersection: {e}"))?;
    Ok(())
}

/// Deliver one `IntersectionObserver` batch: invoke the observer's
/// callback with `(entries, observer)`. A throwing callback is
/// returned for the caller to record as a script error.
pub(crate) fn deliver_intersection_entries(
    realm: &mut TabRealm,
    bindings: &Rc<RefCell<BindingState>>,
    observer_id: u64,
    entries: &[IntersectionObserverEntry],
) -> Result<(), JsError> {
    let b = Rc::clone(bindings);
    let entries: Vec<IntersectionObserverEntry> = entries.to_vec();
    realm
        .with_ctx(|ctx| -> Result<(), String> {
            let arr = Array::new(ctx.clone()).map_err(|e| e.to_string())?;
            for (i, entry) in entries.iter().enumerate() {
                let obj = intersection_entry_to_js(&ctx, &b, entry).map_err(|e| e.to_string())?;
                arr.set(i, Value::from(obj)).map_err(|e| e.to_string())?;
            }
            call_observer_callback(
                &ctx,
                "intersectionObservers",
                "intersectionObserverInstances",
                observer_id,
                Value::from(arr),
            )
        })
        .map_err(JsError::Exception)
}
