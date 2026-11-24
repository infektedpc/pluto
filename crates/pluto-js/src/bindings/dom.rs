//! `window`, `document`, `Node` / `Element`, `console` (binding
//! items 1–3, window/document half).
//!
//! Prototypes are plain JS objects chained
//! `Element → Node → EventTarget → Object.prototype` (and
//! `Document → Node → EventTarget`), created once per realm and kept
//! in the registry's `prototypes` Map. Methods are Rust host
//! functions; every method validates `this` (wrong `this` ⇒
//! `TypeError`) and its `DomKey` (stale ⇒ `null` for getters, a
//! no-op / `TypeError` for setters).
//!
//! M4 departures, flagged here (see also the README):
//! * `querySelector` with an invalid selector returns `null` / `[]`
//!   instead of throwing `SyntaxError` — the `DomHost` query methods
//!   cannot yet distinguish "invalid" from "no match".
//! * `style` supports `cssText` + `setProperty` / `getPropertyValue`
//!   / `removeProperty`; direct named access (`el.style.color = …`)
//!   is a TODO (Proxy-based named properties, later wave).
//! * `getAttribute` returns `String|null`, no namespace support.
//! * `childNodes` returns a plain JS `Array`, not a live `NodeList`.

use std::cell::RefCell;
use std::rc::Rc;

use rquickjs::function::{Opt, Rest, This};
use rquickjs::object::{Accessor, Property};
use rquickjs::{Array, Coerced, Ctx, Function, Object, Symbol, Value};

use crate::bindings::events::{dispatch_to_target, install_window_listeners};
use crate::bindings::{
    BindingState, DOMKEY_SYMBOL, dom_key_of_this, dom_key_of_this_opt, format_console_arg, js,
    js_null, map_delete,
    map_get, map_set, registry_map, throw_dom, throw_host_error, throw_named, throw_type_error,
    to_dom_string, wrapper_id_of,
};
use crate::host::{DomHost, DomKey};

/// The host document for a binding call, or a `TypeError`-style
/// internal error when no document is attached yet.
pub(crate) fn host_of(
    state: &Rc<RefCell<BindingState>>,
) -> Result<Rc<RefCell<Box<dyn DomHost>>>, String> {
    state
        .borrow()
        .host
        .clone()
        .ok_or_else(|| "no document attached to this tab".to_string())
}

/// Wrap `key` in its JS wrapper object, creating and caching it on
/// first access (`document.body === document.body`). Returns `Err`
/// when the key is stale or no document is attached.
pub fn wrap_node<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
) -> rquickjs::Result<Object<'js>> {
    let host = host_of(state).map_err(|e| rquickjs::Error::new_from_js_message("DomKey", "Object", e))?;
    if !host.borrow().validate(key) {
        return Err(rquickjs::Error::new_from_js_message(
            "DomKey",
            "Object",
            "stale DomKey",
        ));
    }
    // Cached wrapper?
    if let Some(id) = state.borrow().wrapper_ids.get(&key).copied() {
        let wrappers = registry_map(ctx, "wrappers")?;
        let existing = map_get(ctx, &wrappers, id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone());
        }
        // The JS entry died without the Rust map knowing (it cannot
        // — nothing deletes wrappers except navigation, which drops
        // the whole state); fall through and recreate it.
    }
    let node_type = host.borrow().node_type(key).unwrap_or(0);
    let proto_name = match node_type {
        9 => "document",
        1 => "element",
        3 => "text",
        _ => "node",
    };
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get(proto_name)?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let id = {
        let mut s = state.borrow_mut();
        let id = s.next_wrapper_id;
        s.next_wrapper_id += 1;
        s.wrapper_keys.insert(id, key);
        s.wrapper_ids.insert(key, id);
        id
    };
    let sym = Symbol::new_global(ctx.clone(), DOMKEY_SYMBOL)?;
    obj.set(sym.as_atom(), id as f64)?;
    let wrappers = registry_map(ctx, "wrappers")?;
    map_set(ctx, &wrappers, id, Value::from(obj.clone()))?;
    Ok(obj)
}

/// Evict a wrapper from the identity cache (stale key discovered on
/// access). The Rust maps drop the id; the JS Map entry is deleted
/// too, so a later access recreates a fresh wrapper if the slot is
/// ever reused by a different node.
pub fn evict_wrapper(ctx: &Ctx<'_>, state: &Rc<RefCell<BindingState>>, key: DomKey) {
    // The first borrow must end before the second begins: a
    // `borrow_mut()` in an `if let` scrutinee lives for the whole
    // statement, so taking both maps in one borrow panics.
    let id = state.borrow_mut().wrapper_ids.remove(&key);
    if let Some(id) = id {
        state.borrow_mut().wrapper_keys.remove(&id);
        if let Ok(wrappers) = registry_map(ctx, "wrappers") {
            map_delete(ctx, &wrappers, id);
        }
    }
}

/// The `document` wrapper, creating it on first access.
pub fn document_wrapper<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
) -> rquickjs::Result<Object<'js>> {
    let host = host_of(state).map_err(|e| rquickjs::Error::new_from_js_message("DomKey", "Object", e))?;
    let Some(root) = host.borrow().document_node() else {
        return Err(rquickjs::Error::new_from_js_message(
            "DomKey",
            "Object",
            "host has no document node",
        ));
    };
    wrap_node(ctx, state, root)
}

/// Wrap an `Option<DomKey>` as a JS value (`null` for `None`).
fn wrap_or_null<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: Option<DomKey>,
) -> rquickjs::Result<Value<'js>> {
    match key {
        Some(k) => wrap_node(ctx, state, k).map(Value::from),
        None => Ok(js_null(ctx)),
    }
}

/// Install the prototype objects into the registry. Requires
/// `EventTarget.prototype` (installed by `events::install_events`).
pub fn install_prototypes<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let event_target: Object = protos
        .get("eventTarget")
        .map_err(|e| format!("EventTarget prototype missing: {e}"))?;

    // --- Node.prototype (proto: EventTarget.prototype) ---
    let node = Object::new_proto(ctx.clone(), Some(&event_target))
        .map_err(|e| format!("Node prototype: {e}"))?;
    install_node_methods(ctx, bindings, &node)?;
    protos
        .set("node", node.clone())
        .map_err(|e| format!("prototypes node: {e}"))?;

    // --- Element.prototype (proto: Node.prototype) ---
    let element = Object::new_proto(ctx.clone(), Some(&node))
        .map_err(|e| format!("Element prototype: {e}"))?;
    install_element_methods(ctx, bindings, &element)?;
    protos
        .set("element", element)
        .map_err(|e| format!("prototypes element: {e}"))?;

    // --- Text.prototype (proto: Node.prototype) ---
    let text = Object::new_proto(ctx.clone(), Some(&node))
        .map_err(|e| format!("Text prototype: {e}"))?;
    install_text_methods(ctx, bindings, &text)?;
    protos
        .set("text", text)
        .map_err(|e| format!("prototypes text: {e}"))?;

    // --- Document.prototype (proto: Node.prototype) ---
    let document = Object::new_proto(ctx.clone(), Some(&node))
        .map_err(|e| format!("Document prototype: {e}"))?;
    install_document_methods(ctx, bindings, &document)?;
    protos
        .set("document", document)
        .map_err(|e| format!("prototypes document: {e}"))?;

    // --- DOMTokenList.prototype (classList) ---
    let token_list = Object::new(ctx.clone()).map_err(|e| format!("DOMTokenList proto: {e}"))?;
    install_token_list_methods(ctx, bindings, &token_list)?;
    protos
        .set("domTokenList", token_list)
        .map_err(|e| format!("prototypes domTokenList: {e}"))?;

    // --- CSSStyleDeclaration.prototype (style) ---
    let style = Object::new(ctx.clone()).map_err(|e| format!("CSSStyleDeclaration proto: {e}"))?;
    install_style_methods(ctx, bindings, &style)?;
    protos
        .set("styleDeclaration", style)
        .map_err(|e| format!("prototypes styleDeclaration: {e}"))?;
    Ok(())
}

/// Helper: define a method on a prototype object.
fn def_method<'js>(
    _ctx: &Ctx<'js>,
    proto: &Object<'js>,
    name: &str,
    func: Function<'js>,
) -> Result<(), String> {
    proto
        .set(name, func)
        .map_err(|e| format!("install {name}: {e}"))
}

/// Helper: define an accessor on a prototype object.
fn def_accessor<'js, G, S>(
    ctx: &Ctx<'js>,
    proto: &Object<'js>,
    name: &str,
    get: G,
    set: S,
) -> Result<(), String>
where
    G: rquickjs::function::IntoJsFunc<'js, (Ctx<'js>, rquickjs::function::This<Object<'js>>)>
        + 'js,
    S: rquickjs::function::IntoJsFunc<
            'js,
            (
                Ctx<'js>,
                rquickjs::function::This<Object<'js>>,
                Value<'js>,
            ),
        > + 'js,
{
    let _ = ctx;
    proto
        .prop(name, Accessor::new(get, set))
        .map_err(|e| format!("install accessor {name}: {e}"))
}

/// A getter-only accessor.
fn def_getter<'js, G>(
    ctx: &Ctx<'js>,
    proto: &Object<'js>,
    name: &str,
    get: G,
) -> Result<(), String>
where
    G: rquickjs::function::IntoJsFunc<'js, (Ctx<'js>, rquickjs::function::This<Object<'js>>)>
        + 'js,
{
    let _ = ctx;
    proto
        .prop(name, Accessor::from(get))
        .map_err(|e| format!("install getter {name}: {e}"))
}

// ---------------------------------------------------------------- Node

fn install_node_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    node: &Object<'js>,
) -> Result<(), String> {
    // textContent (getter + setter).
    let b = Rc::clone(bindings);
    let get_text = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        if !host.borrow().validate(key) {
            evict_wrapper(&ctx, &b, key);
            return Ok(js_null(&ctx));
        }
        Ok(match host.borrow().text_content(key) {
            Some(t) => js(&ctx, t),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_text = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow().validate(key) {
            evict_wrapper(&ctx, &b, key);
            return Err(throw_type_error(&ctx, "the node is no longer in the document"));
        }
        let text = to_dom_string(&ctx, &value)?;
        if !host.borrow_mut().set_text_content(key, &text) {
            return Err(throw_type_error(&ctx, "the node is no longer in the document"));
        }
        Ok(())
    };
    def_accessor(ctx, node, "textContent", get_text, set_text)?;

    // parentNode.
    let b = Rc::clone(bindings);
    def_getter(ctx, node, "parentNode", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        if !host.borrow().validate(key) {
            evict_wrapper(&ctx, &b, key);
            return Ok(js_null(&ctx));
        }
        wrap_or_null(&ctx, &b, host.borrow().parent(key))
    })?;

    // childNodes (a plain JS Array in M4, not a live NodeList).
    let b = Rc::clone(bindings);
    def_getter(ctx, node, "childNodes", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        if !host.borrow().validate(key) {
            evict_wrapper(&ctx, &b, key);
            return Ok(Value::from(Array::new(ctx.clone())?));
        }
        let kids = host.borrow().children(key);
        let arr = Array::new(ctx.clone())?;
        for (i, kid) in kids.into_iter().enumerate() {
            if let Ok(w) = wrap_node(&ctx, &b, kid) {
                arr.set(i, Value::from(w))?;
            }
        }
        Ok(Value::from(arr))
    })?;

    // nodeType / nodeName.
    let b = Rc::clone(bindings);
    def_getter(ctx, node, "nodeType", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().node_type(key) {
            Some(t) => js(&ctx, t as f64),
            None => js_null(&ctx),
        })
    })?;
    let b = Rc::clone(bindings);
    def_getter(ctx, node, "nodeName", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().node_name(key) {
            Some(n) => js(&ctx, n),
            None => js_null(&ctx),
        })
    })?;

    // Tree surgery. The child argument must be a wrapper object;
    // anything else is a WebIDL `TypeError` ("not a Node").
    let b = Rc::clone(bindings);
    let append = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, child: Value<'js>| -> rquickjs::Result<Value<'js>> {
            let parent = dom_key_of_this(&ctx, &b, &this)?;
            let child_key = node_arg(&ctx, &b, &child)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut()
                .append_child(parent, child_key)
                .map_err(|e| throw_host_error(&ctx, e))?;
            Ok(child)
        },
    )
    .map_err(|e| format!("appendChild: {e}"))?;
    def_method(ctx, node, "appendChild", append)?;

    let b = Rc::clone(bindings);
    let remove = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, child: Value<'js>| -> rquickjs::Result<Value<'js>> {
            let parent = dom_key_of_this(&ctx, &b, &this)?;
            let child_key = node_arg(&ctx, &b, &child)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut()
                .remove_child(parent, child_key)
                .map_err(|e| throw_host_error(&ctx, e))?;
            Ok(child)
        },
    )
    .map_err(|e| format!("removeChild: {e}"))?;
    def_method(ctx, node, "removeChild", remove)?;

    let b = Rc::clone(bindings);
    let insert = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, child: Value<'js>, sibling: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let parent = dom_key_of_this(&ctx, &b, &this)?;
            let child_key = node_arg(&ctx, &b, &child)?;
            let sibling_key = match sibling.0 {
                Some(v) if !v.is_null() && !v.is_undefined() => Some(node_arg(&ctx, &b, &v)?),
                _ => None,
            };
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut()
                .insert_before(parent, child_key, sibling_key)
                .map_err(|e| throw_host_error(&ctx, e))?;
            Ok(child)
        },
    )
    .map_err(|e| format!("insertBefore: {e}"))?;
    def_method(ctx, node, "insertBefore", insert)?;

    let b = Rc::clone(bindings);
    let replace = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, new_node: Value<'js>, old_node: Value<'js>| -> rquickjs::Result<Value<'js>> {
            let parent = dom_key_of_this(&ctx, &b, &this)?;
            let new_key = node_arg(&ctx, &b, &new_node)?;
            let old_key = node_arg(&ctx, &b, &old_node)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut()
                .replace_child(parent, old_key, new_key)
                .map_err(|e| throw_host_error(&ctx, e))?;
            Ok(old_node)
        },
    )
    .map_err(|e| format!("replaceChild: {e}"))?;
    def_method(ctx, node, "replaceChild", replace)?;
    Ok(())
}

/// The `DomKey` for a method argument that must be a `Node` wrapper.
fn node_arg<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    value: &Value<'js>,
) -> rquickjs::Result<DomKey> {
    let Some(obj) = value.as_object() else {
        return Err(throw_type_error(ctx, "argument is not a Node"));
    };
    let Some(id) = wrapper_id_of(ctx, obj) else {
        return Err(throw_type_error(ctx, "argument is not a Node"));
    };
    state
        .borrow()
        .wrapper_keys
        .get(&id)
        .copied()
        .ok_or_else(|| throw_type_error(ctx, "argument is not a Node"))
}

// ---------------------------------------------------------------- Element

fn install_element_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    element: &Object<'js>,
) -> Result<(), String> {
    // tagName (uppercase, per HTML).
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "tagName", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().tag_name(key) {
            Some(t) => js(&ctx, t.to_ascii_uppercase()),
            None => js_null(&ctx),
        })
    })?;

    // id / className: live content-attribute accessors.
    install_attr_accessor(ctx, bindings, element, "id", "id")?;
    install_attr_accessor(ctx, bindings, element, "className", "class")?;

    // Attributes. M4: `getAttribute` returns `String|null`, no
    // namespace support.
    let b = Rc::clone(bindings);
    let get_attr = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            Ok(match host.borrow().get_attribute(key, &name.to_ascii_lowercase()) {
                Some(v) => js(&ctx, v),
                None => js_null(&ctx),
            })
        },
    )
    .map_err(|e| format!("getAttribute: {e}"))?;
    def_method(ctx, element, "getAttribute", get_attr)?;

    let b = Rc::clone(bindings);
    let set_attr = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            validate_attr_name(&ctx, &name)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host.borrow_mut().set_attribute(key, &name.to_ascii_lowercase(), &value) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("setAttribute: {e}"))?;
    def_method(ctx, element, "setAttribute", set_attr)?;

    let b = Rc::clone(bindings);
    let remove_attr = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            validate_attr_name(&ctx, &name)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host.borrow_mut().remove_attribute(key, &name.to_ascii_lowercase()) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("removeAttribute: {e}"))?;
    def_method(ctx, element, "removeAttribute", remove_attr)?;

    let b = Rc::clone(bindings);
    let has_attr = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<bool> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(false),
            };
            Ok(host.borrow().has_attribute(key, &name.to_ascii_lowercase()))
        },
    )
    .map_err(|e| format!("hasAttribute: {e}"))?;
    def_method(ctx, element, "hasAttribute", has_attr)?;

    // querySelector / querySelectorAll (host does the matching with
    // the Stage-6 selector machinery; the binding never matches).
    let b = Rc::clone(bindings);
    let qs = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            wrap_or_null(&ctx, &b, host.borrow().query_selector(key, &selector))
        },
    )
    .map_err(|e| format!("querySelector: {e}"))?;
    def_method(ctx, element, "querySelector", qs)?;

    let b = Rc::clone(bindings);
    let qsa = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let keys = host.borrow().query_selector_all(key, &selector);
            let arr = Array::new(ctx.clone())?;
            for (i, k) in keys.into_iter().enumerate() {
                if let Ok(w) = wrap_node(&ctx, &b, k) {
                    arr.set(i, Value::from(w))?;
                }
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("querySelectorAll: {e}"))?;
    def_method(ctx, element, "querySelectorAll", qsa)?;

    // classList (cached per element, so `el.classList === el.classList`).
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "classList", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let id = wrapper_id_of(&ctx, &this).unwrap_or(0);
        let map = registry_map(&ctx, "classLists")?;
        let existing = map_get(&ctx, &map, id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone().into_value());
        }
        let protos = registry_map(&ctx, "prototypes")?;
        let proto: Object = protos.get("domTokenList")?;
        let list = Object::new_proto(ctx.clone(), Some(&proto))?;
        let sym = Symbol::new_global(ctx.clone(), DOMKEY_SYMBOL)?;
        list.set(sym.as_atom(), id as f64)?;
        map_set(&ctx, &map, id, Value::from(list.clone()))?;
        let _ = key;
        Ok(Value::from(list))
    })?;

    // style (cached per element, like classList).
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "style", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let id = wrapper_id_of(&ctx, &this).unwrap_or(0);
        let map = registry_map(&ctx, "styles")?;
        let existing = map_get(&ctx, &map, id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone().into_value());
        }
        let protos = registry_map(&ctx, "prototypes")?;
        let proto: Object = protos.get("styleDeclaration")?;
        let decl = Object::new_proto(ctx.clone(), Some(&proto))?;
        let sym = Symbol::new_global(ctx.clone(), DOMKEY_SYMBOL)?;
        decl.set(sym.as_atom(), id as f64)?;
        map_set(&ctx, &map, id, Value::from(decl.clone()))?;
        let _ = key;
        Ok(Value::from(decl))
    })?;

    // innerHTML (fragment parse, scripts inert — via the host).
    let b = Rc::clone(bindings);
    let get_html = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().inner_html(key) {
            Some(h) => js(&ctx, h),
            None => js(&ctx, ""),
        })
    };
    let b = Rc::clone(bindings);
    let set_html = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let html = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow_mut().set_inner_html(key, &html) {
            return Err(throw_type_error(&ctx, "the element is no longer in the document"));
        }
        Ok(())
    };
    def_accessor(ctx, element, "innerHTML", get_html, set_html)?;

    // click(): dispatch a bubbling, cancelable click on this element.
    let b = Rc::clone(bindings);
    let click = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let id = wrapper_id_of(&ctx, &this)
                .ok_or_else(|| throw_type_error(&ctx, "called on a non-Element object"))?;
            let event: Object = ctx.eval("new Event('click', { bubbles: true, cancelable: true })")?;
            let _ = key;
            dispatch_to_target(&ctx, &b, id, &event, "click")?;
            Ok(())
        },
    )
    .map_err(|e| format!("click: {e}"))?;
    def_method(ctx, element, "click", click)?;

    // Geometry reads (wave 2b, Rec 7): force a synchronous layout
    // flush through the host first, then read the box. This is the
    // ONLY sanctioned synchronous cross-thread call in the design;
    // a read/write loop therefore flushes once per iteration (layout
    // thrash, by design — the host counts forced flushes so it is
    // measurable), while batched writes flush once. A stale key
    // throws `TypeError`; no layout tree yet (or a dead layout
    // actor) yields a zero rect rather than a throw.
    let b = Rc::clone(bindings);
    let rect = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Object<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let g = host.borrow_mut().geometry(key).unwrap_or(crate::host::Geometry {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            });
            let out = Object::new(ctx.clone())?;
            out.set("x", g.x)?;
            out.set("y", g.y)?;
            out.set("width", g.width)?;
            out.set("height", g.height)?;
            out.set("top", g.y)?;
            out.set("left", g.x)?;
            out.set("right", g.x + g.width)?;
            out.set("bottom", g.y + g.height)?;
            Ok(out)
        },
    )
    .map_err(|e| format!("getBoundingClientRect: {e}"))?;
    def_method(ctx, element, "getBoundingClientRect", rect)?;

    // offsetWidth / offsetHeight: border-box size, rounded to whole
    // CSS pixels per spec (integer layout pixels).
    for (prop, is_width) in [("offsetWidth", true), ("offsetHeight", false)] {
        let b = Rc::clone(bindings);
        def_getter(
            ctx,
            element,
            prop,
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                    return Ok(0.0);
                };
                let host = match host_of(&b) {
                    Ok(h) => h,
                    Err(_) => return Ok(0.0),
                };
                Ok(match host.borrow_mut().geometry(key) {
                    Some(g) => {
                        if is_width {
                            g.width.round()
                        } else {
                            g.height.round()
                        }
                    }
                    None => 0.0,
                })
            },
        )?;
    }
    Ok(())
}

/// A live content-attribute accessor (`id`, `className`): reads the
/// attribute fresh on every access, never a cached copy.
/// `CharacterData.data` on the Text prototype (Stage 8 Rec 7):
/// the JS-visible character-data mutation — setting it queues a
/// `characterData` record (old text) for observing
/// `MutationObserver`s.
fn install_text_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    text: &Object<'js>,
) -> Result<(), String> {
    let b = Rc::clone(bindings);
    let get_data = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        Ok(match host.borrow().text_content(key) {
            Some(t) => js(&ctx, t),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_data = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        let data = to_dom_string(&ctx, &value)?;
        if !host.borrow_mut().set_character_data(key, &data) {
            return Err(throw_type_error(
                &ctx,
                "CharacterData.data: the node is not a text node",
            ));
        }
        Ok(())
    };
    def_accessor(ctx, text, "data", get_data, set_data)?;
    Ok(())
}

fn install_attr_accessor<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    element: &Object<'js>,
    prop: &str,
    attr: &str,
) -> Result<(), String> {
    let attr = attr.to_string();
    let b = Rc::clone(bindings);
    let a = attr.clone();
    let get = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().get_attribute(key, &a) {
            Some(v) => js(&ctx, v),
            None => js(&ctx, ""),
        })
    };
    let b = Rc::clone(bindings);
    let set = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let text = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow_mut().set_attribute(key, &attr, &text) {
            return Err(throw_type_error(&ctx, "the element is no longer in the document"));
        }
        Ok(())
    };
    def_accessor(ctx, element, prop, get, set)
}

/// XML `Name` sanity for attribute names (M4 subset: ASCII letters,
/// digits, `_`, `-`, `.`, `:` after a valid first character).
fn validate_attr_name(ctx: &Ctx<'_>, name: &str) -> rquickjs::Result<()> {
    let mut chars = name.chars();
    let first_ok = chars
        .next()
        .map(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
        .unwrap_or(false);
    let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'));
    if !first_ok || !rest_ok {
        return Err(throw_dom(
            ctx,
            "InvalidCharacterError",
            "the attribute name contains an invalid character",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- classList

/// Split a `class` attribute value into tokens.
fn class_tokens(class: &str) -> Vec<String> {
    class
        .split_ascii_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn install_token_list_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    list: &Object<'js>,
) -> Result<(), String> {
    // The element key behind a classList object (via its domkey id).
    let key_of = |ctx: &Ctx<'js>, b: &Rc<RefCell<BindingState>>, this: &Object<'js>| -> rquickjs::Result<DomKey> {
        dom_key_of_this(ctx, b, this)
    };
    let b = Rc::clone(bindings);
    let contains = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, token: Coerced<String>| -> rquickjs::Result<bool> {
            let key = key_of(&ctx, &b, &this)?;
            validate_token(&ctx, &token)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(false),
            };
            let class = host.borrow().get_attribute(key, "class").unwrap_or_default();
            Ok(class_tokens(&class).iter().any(|t| t == token.as_str()))
        },
    )
    .map_err(|e| format!("classList.contains: {e}"))?;
    def_method(ctx, list, "contains", contains)?;

    let b = Rc::clone(bindings);
    let add = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, tokens: Rest<Coerced<String>>| -> rquickjs::Result<()> {
            let key = key_of(&ctx, &b, &this)?;
            for t in &tokens.0 {
                validate_token(&ctx, t)?;
            }
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let class = host.borrow().get_attribute(key, "class").unwrap_or_default();
            let mut list = class_tokens(&class);
            for t in &tokens.0 {
                if !list.iter().any(|x| x == t.as_str()) {
                    list.push(t.to_string());
                }
            }
            if !host.borrow_mut().set_attribute(key, "class", &list.join(" ")) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("classList.add: {e}"))?;
    def_method(ctx, list, "add", add)?;

    let b = Rc::clone(bindings);
    let remove = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, tokens: Rest<Coerced<String>>| -> rquickjs::Result<()> {
            let key = key_of(&ctx, &b, &this)?;
            for t in &tokens.0 {
                validate_token(&ctx, t)?;
            }
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let class = host.borrow().get_attribute(key, "class").unwrap_or_default();
            let mut list = class_tokens(&class);
            for t in &tokens.0 {
                list.retain(|x| x != t.as_str());
            }
            if !host.borrow_mut().set_attribute(key, "class", &list.join(" ")) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("classList.remove: {e}"))?;
    def_method(ctx, list, "remove", remove)?;

    let b = Rc::clone(bindings);
    let toggle = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, token: Coerced<String>, force: Opt<bool>| -> rquickjs::Result<bool> {
            let key = key_of(&ctx, &b, &this)?;
            validate_token(&ctx, &token)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let class = host.borrow().get_attribute(key, "class").unwrap_or_default();
            let mut list = class_tokens(&class);
            let present = list.iter().any(|x| x == token.as_str());
            let want = force.0.unwrap_or(!present);
            if want && !present {
                list.push(token.to_string());
            } else if !want && present {
                list.retain(|x| x != token.as_str());
            }
            if !host.borrow_mut().set_attribute(key, "class", &list.join(" ")) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(want)
        },
    )
    .map_err(|e| format!("classList.toggle: {e}"))?;
    def_method(ctx, list, "toggle", toggle)?;

    // `value`: the serialized token list.
    let b = Rc::clone(bindings);
    def_getter(ctx, list, "value", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let key = key_of(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().get_attribute(key, "class").unwrap_or_default()))
    })?;
    Ok(())
}

fn validate_token(ctx: &Ctx<'_>, token: &str) -> rquickjs::Result<()> {
    if token.is_empty() {
        return Err(throw_dom(ctx, "SyntaxError", "the token is empty"));
    }
    if token.chars().any(|c| c.is_whitespace()) {
        return Err(throw_dom(
            ctx,
            "InvalidCharacterError",
            "the token contains whitespace",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- style

/// Parse a `style` attribute into `(name, value)` pairs, preserving
/// order. Values are kept verbatim (M4: no CSS parsing — the full
/// CSSOM is later; `cssText` round-trips through this split).
fn parse_style_text(css: &str) -> Vec<(String, String)> {
    css.split(';')
        .filter_map(|decl| {
            let (name, value) = decl.split_once(':')?;
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim().to_string();
            if name.is_empty() {
                None
            } else {
                Some((name, value))
            }
        })
        .collect()
}

fn serialise_style_text(decls: &[(String, String)]) -> String {
    decls
        .iter()
        .map(|(n, v)| format!("{n}: {v};"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn install_style_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    style: &Object<'js>,
) -> Result<(), String> {
    let key_of = |ctx: &Ctx<'js>, b: &Rc<RefCell<BindingState>>, this: &Object<'js>| -> rquickjs::Result<DomKey> {
        dom_key_of_this(ctx, b, this)
    };
    // cssText accessor.
    let b = Rc::clone(bindings);
    let get_css = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let key = key_of(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().style_text(key).unwrap_or_default()))
    };
    let b = Rc::clone(bindings);
    let set_css = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = key_of(&ctx, &b, &this)?;
        let css = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow_mut().set_style_text(key, &css) {
            return Err(throw_type_error(&ctx, "the element is no longer in the document"));
        }
        Ok(())
    };
    def_accessor(ctx, style, "cssText", get_css, set_css)?;

    // setProperty / getPropertyValue / removeProperty.
    let b = Rc::clone(bindings);
    let set_prop = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let key = key_of(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let css = host.borrow().style_text(key).unwrap_or_default();
            let mut decls = parse_style_text(&css);
            let name = name.to_ascii_lowercase();
            match decls.iter_mut().find(|(n, _)| *n == name) {
                Some((_, v)) => *v = value.to_string(),
                None => decls.push((name, value.to_string())),
            }
            if !host.borrow_mut().set_style_text(key, &serialise_style_text(&decls)) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("style.setProperty: {e}"))?;
    def_method(ctx, style, "setProperty", set_prop)?;

    let b = Rc::clone(bindings);
    let get_prop = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = key_of(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js(&ctx, "")),
            };
            let css = host.borrow().style_text(key).unwrap_or_default();
            let name = name.to_ascii_lowercase();
            Ok(match parse_style_text(&css).into_iter().find(|(n, _)| *n == name) {
                Some((_, v)) => js(&ctx, v),
                None => js(&ctx, ""),
            })
        },
    )
    .map_err(|e| format!("style.getPropertyValue: {e}"))?;
    def_method(ctx, style, "getPropertyValue", get_prop)?;

    let b = Rc::clone(bindings);
    let remove_prop = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = key_of(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let css = host.borrow().style_text(key).unwrap_or_default();
            let name = name.to_ascii_lowercase();
            let mut decls = parse_style_text(&css);
            let old = decls
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v.clone());
            decls.retain(|(n, _)| *n != name);
            if !host.borrow_mut().set_style_text(key, &serialise_style_text(&decls)) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            Ok(match old {
                Some(v) => js(&ctx, v),
                None => js(&ctx, ""),
            })
        },
    )
    .map_err(|e| format!("style.removeProperty: {e}"))?;
    def_method(ctx, style, "removeProperty", remove_prop)?;
    Ok(())
}

// ---------------------------------------------------------------- document

fn install_document_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    document: &Object<'js>,
) -> Result<(), String> {
    // documentElement / body / head.
    for (prop, getter) in [
        ("documentElement", 0u8),
        ("body", 1u8),
        ("head", 2u8),
    ] {
        let b = Rc::clone(bindings);
        def_getter(ctx, document, prop, move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            // `this` is the document wrapper; validate it, then read
            // the host's root element (the host knows the document).
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            let key = match getter {
                0 => host.borrow().document_element(),
                1 => host.borrow().body(),
                _ => host.borrow().head(),
            };
            wrap_or_null(&ctx, &b, key)
        })?;
    }

    // title.
    let b = Rc::clone(bindings);
    let get_title = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().title().unwrap_or_default()))
    };
    let b = Rc::clone(bindings);
    let set_title = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let text = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow_mut().set_title(&text) {
            return Err(throw_type_error(&ctx, "the document has no head"));
        }
        Ok(())
    };
    def_accessor(ctx, document, "title", get_title, set_title)?;

    // getElementById.
    let b = Rc::clone(bindings);
    let by_id = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, id: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            wrap_or_null(&ctx, &b, host.borrow().get_element_by_id(&id))
        },
    )
    .map_err(|e| format!("getElementById: {e}"))?;
    def_method(ctx, document, "getElementById", by_id)?;

    // querySelector / querySelectorAll, scoped at the document node.
    let b = Rc::clone(bindings);
    let qs = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let scope = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            wrap_or_null(&ctx, &b, host.borrow().query_selector(scope, &selector))
        },
    )
    .map_err(|e| format!("document.querySelector: {e}"))?;
    def_method(ctx, document, "querySelector", qs)?;

    let b = Rc::clone(bindings);
    let qsa = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let scope = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let keys = host.borrow().query_selector_all(scope, &selector);
            let arr = Array::new(ctx.clone())?;
            for (i, k) in keys.into_iter().enumerate() {
                if let Ok(w) = wrap_node(&ctx, &b, k) {
                    arr.set(i, Value::from(w))?;
                }
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("document.querySelectorAll: {e}"))?;
    def_method(ctx, document, "querySelectorAll", qsa)?;

    // createElement / createTextNode.
    let b = Rc::clone(bindings);
    let create_el = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, tag: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let tag = tag.to_ascii_lowercase();
            if tag.is_empty() || !tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                return Err(throw_dom(
                    &ctx,
                    "InvalidCharacterError",
                    "the tag name contains an invalid character",
                ));
            }
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let Some(key) = host.borrow_mut().create_element(&tag) else {
                return Err(throw_type_error(&ctx, "could not create element"));
            };
            wrap_node(&ctx, &b, key).map(Value::from)
        },
    )
    .map_err(|e| format!("createElement: {e}"))?;
    def_method(ctx, document, "createElement", create_el)?;

    let b = Rc::clone(bindings);
    let create_text = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, data: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let Some(key) = host.borrow_mut().create_text_node(&data) else {
                return Err(throw_type_error(&ctx, "could not create text node"));
            };
            wrap_node(&ctx, &b, key).map(Value::from)
        },
    )
    .map_err(|e| format!("createTextNode: {e}"))?;
    def_method(ctx, document, "createTextNode", create_text)?;

    // write / writeln / open / close (the parser-side insertion-point
    // stack is the parser agent's job; the binding calls the host).
    let b = Rc::clone(bindings);
    let write = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let mut text = String::new();
            for arg in &args.0 {
                text.push_str(&to_dom_string(&ctx, arg)?);
            }
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host.borrow_mut().document_write(&text) {
                return Err(throw_dom(
                    &ctx,
                    "InvalidStateError",
                    "document.write is not available in this state",
                ));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("document.write: {e}"))?;
    def_method(ctx, document, "write", write)?;

    let b = Rc::clone(bindings);
    let writeln = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let mut text = String::new();
            for arg in &args.0 {
                text.push_str(&to_dom_string(&ctx, arg)?);
            }
            text.push('\n');
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host.borrow_mut().document_write(&text) {
                return Err(throw_dom(
                    &ctx,
                    "InvalidStateError",
                    "document.writeln is not available in this state",
                ));
            }
            Ok(())
        },
    )
    .map_err(|e| format!("document.writeln: {e}"))?;
    def_method(ctx, document, "writeln", writeln)?;

    let b = Rc::clone(bindings);
    let open = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            // Ignored from a parser-inserted script (host returns
            // false) — per spec, `open()` then simply does nothing.
            let _ = host.borrow_mut().document_open();
            document_wrapper(&ctx, &b).map(Value::from)
        },
    )
    .map_err(|e| format!("document.open: {e}"))?;
    def_method(ctx, document, "open", open)?;

    let b = Rc::clone(bindings);
    let close = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let _ = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let _ = host.borrow_mut().document_close();
            Ok(())
        },
    )
    .map_err(|e| format!("document.close: {e}"))?;
    def_method(ctx, document, "close", close)?;
    Ok(())
}

// ---------------------------------------------------------------- window

/// Install `window` / `self` (both are `globalThis`), the `document`
/// accessor, `window.onerror`, and the `window` EventTarget methods.
pub fn install_window<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    // `window.window === window`, `window.self === window`.
    globals
        .set("window", Value::from(globals.clone()))
        .map_err(|e| format!("install window: {e}"))?;
    globals
        .set("self", Value::from(globals.clone()))
        .map_err(|e| format!("install self: {e}"))?;

    // `document`: an accessor that wraps the host's document node on
    // first access (the host is attached after realm creation).
    let b = Rc::clone(bindings);
    let get_document = move |ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<Value<'js>> {
        document_wrapper(&ctx, &b).map(Value::from)
    };
    globals
        .prop(
            "document",
            Accessor::from(get_document).enumerable(),
        )
        .map_err(|e| format!("install document: {e}"))?;

    // `window.onerror`: a plain writable property, initially `null`.
    // The loop reads it when reporting a script error; the page sets
    // it to a function. (M4: no accessor indirection is needed —
    // assignment is the registration.)
    globals
        .prop("onerror", Property::from(rquickjs::Null).writable().enumerable())
        .map_err(|e| format!("install onerror: {e}"))?;

    // `window` is an EventTarget: the same three methods live on the
    // global object (its prototype chain does not include
    // EventTarget.prototype — `globalThis` is the window).
    install_window_listeners(ctx, bindings)
        .map_err(|e| format!("install window listeners: {e}"))?;
    Ok(())
}

/// Install `console` (`log` / `info` / `debug` / `warn` / `error` →
/// the host log, via `tracing`, plus a captured copy in
/// [`BindingState::console_lines`] for tests).
pub fn install_console<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    let console = Object::new(ctx.clone()).map_err(|e| format!("console object: {e}"))?;
    for (name, level) in [
        ("log", "log"),
        ("info", "info"),
        ("debug", "debug"),
        ("warn", "warn"),
        ("error", "error"),
    ] {
        let b = Rc::clone(bindings);
        let level = level.to_string();
        let func = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
                let parts: Vec<String> =
                    args.0.iter().map(|v| format_console_arg(&ctx, v)).collect();
                let message = parts.join(" ");
                match level.as_str() {
                    "warn" => tracing::warn!(target: "pluto-js:console", "{message}"),
                    "error" => tracing::error!(target: "pluto-js:console", "{message}"),
                    "debug" => tracing::debug!(target: "pluto-js:console", "{message}"),
                    _ => tracing::info!(target: "pluto-js:console", "{message}"),
                }
                b.borrow_mut()
                    .console_lines
                    .push((level.clone(), message));
                Ok(())
            },
        )
        .map_err(|e| format!("console.{name}: {e}"))?;
        console
            .set(name, func)
            .map_err(|e| format!("console.{name}: {e}"))?;
    }
    globals
        .set("console", console)
        .map_err(|e| format!("install console: {e}"))?;
    Ok(())
}

/// `throw_named` re-export for submodules that build exceptions.
pub fn throw_js(ctx: &Ctx<'_>, name: &str, msg: &str) -> rquickjs::Error {
    throw_named(ctx, name, msg)
}
