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
use std::time::Instant;

use rquickjs::function::{Opt, Rest, This};
use rquickjs::object::{Accessor, Property};
use rquickjs::{Array, Coerced, Ctx, FromJs, Function, Object, Symbol, Value};
use url::Url;

use crate::bindings::events::{dispatch_to_target, install_window_listeners};
use crate::bindings::{
    BindingState, DOMKEY_SYMBOL, URL_SYMBOL, dom_key_of_this, dom_key_of_this_opt,
    format_console_arg, js, js_null, map_delete, map_get, map_set, registry, registry_map,
    throw_dom, throw_host_error, throw_named, throw_type_error, to_dom_string, wrapper_id_of,
};
use crate::host::{AnimateSpec, AnimationInfo, DomHost, DomKey, WaapiOp, WaapiStep};

/// One `URL` / `URLSearchParams` value, by id.
///
/// * URL objects store the full serialised URL (`is_url = true`).
/// * Standalone `URLSearchParams` store the serialised query
///   (`is_url = false`, no leading `?`).
/// * A URL's `searchParams` shares the URL's id, so the two stay
///   live-linked: param mutations re-serialise the URL's query.
#[derive(Debug, Clone)]
pub struct UrlEntry {
    pub is_url: bool,
    pub text: String,
}

/// Load state for an `Image()` / `<img>` node, by `DomKey`.
/// Written by the `src` setter (not loaded) and by
/// [`fire_image_event`] when the shell reports the outcome.
#[derive(Debug, Clone, Default)]
pub struct ImageLoadState {
    /// The resolved `src` URL ("" when no `src` was set).
    pub current_src: String,
    /// True once the shell reported a successful load.
    pub complete: bool,
    /// Intrinsic width / height from the decoded image (0 until loaded).
    pub width: u32,
    pub height: u32,
}

/// Local play-state + clock for one `Animation` object (M6 Phase 3,
/// Rec 11). The animation engine is the authority for sampled
/// values; this is only the JS-visible `playState` / `currentTime`
/// bookkeeping, kept in [`BindingState::animation_locals`].
#[derive(Debug, Clone)]
pub struct AnimationLocal {
    /// `"running"` | `"paused"` | `"finished"` | `"idle"`.
    pub play_state: String,
    /// When the animation started (binding-side clock).
    pub started: Instant,
    /// Duration in ms (`f32::INFINITY` for infinite / CSS).
    pub duration_ms: f32,
    /// When the current pause began (`None` while running).
    pub paused_at: Option<Instant>,
    /// Total time spent paused so far.
    pub paused_total: std::time::Duration,
    /// True for CSS animations (`currentTime` reads `null`).
    pub is_css: bool,
}

impl AnimationLocal {
    /// `currentTime` in ms, or `None` for CSS / idle animations.
    /// Clamped to `[0, duration]` for finite durations.
    pub fn current_time_ms(&self) -> Option<f64> {
        if self.is_css || self.play_state == "idle" {
            return None;
        }
        let end = self.paused_at.unwrap_or_else(Instant::now);
        let elapsed = end
            .duration_since(self.started)
            .saturating_sub(self.paused_total);
        let ms = elapsed.as_secs_f64() * 1000.0;
        if self.duration_ms.is_finite() {
            Some(ms.min(self.duration_ms as f64))
        } else {
            Some(ms)
        }
    }
}

/// `camelCase` → `kebab-case` (JS keyframe / style property names
/// arrive camelCased; the CSS side wants kebab-case).
fn camel_to_kebab(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for c in name.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// `kebab-case` → `camelCase` (for `getComputedStyle` aliases).
fn kebab_to_camel(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = false;
    for c in name.chars() {
        if c == '-' {
            upper = true;
        } else if upper {
            out.push(c.to_ascii_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

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
        // `<img>` elements get the image prototype (HTMLImageElement:
        // `src` / `complete` / `naturalWidth` / `naturalHeight`);
        // every other element gets the plain element prototype.
        1 if host.borrow().tag_name(key).as_deref() == Some("img") => "imageElement",
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
        .set("element", element.clone())
        .map_err(|e| format!("prototypes element: {e}"))?;

    // --- HTMLImageElement.prototype (proto: Element.prototype) ---
    // Only `<img>` wrappers chain here (see `wrap_node`); the
    // methods read the tag so a non-`img` can never reach them.
    let image = Object::new_proto(ctx.clone(), Some(&element))
        .map_err(|e| format!("HTMLImageElement prototype: {e}"))?;
    install_image_methods(ctx, bindings, &image)?;
    protos
        .set("imageElement", image)
        .map_err(|e| format!("prototypes imageElement: {e}"))?;
    install_image_constructor(ctx, bindings)
        .map_err(|e| format!("install Image: {e}"))?;

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
pub(crate) fn def_method<'js>(
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
pub(crate) fn def_accessor<'js, G, S>(
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
pub(crate) fn def_getter<'js, G>(
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

/// Build a JS `Array` of wrappers for `keys` (a fresh snapshot per
/// access — M4 `getElementsBy*` / `querySelectorAll` collections
/// are not live `HTMLCollection`s; see the binding notes).
pub(crate) fn elements_array<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    keys: Vec<DomKey>,
) -> rquickjs::Result<Value<'js>> {
    let arr = Array::new(ctx.clone())?;
    for (i, k) in keys.into_iter().enumerate() {
        if let Ok(w) = wrap_node(ctx, state, k) {
            arr.set(i, Value::from(w))?;
        }
    }
    Ok(Value::from(arr))
}

/// True when `key` matches `selector`: the host's
/// `query_selector_all` over the whole document, checked for
/// membership. (The binding never matches selectors itself; the
/// production host uses the Stage-6 matcher, the test host its
/// own — either way the answer comes from the host.)
fn matches_selector<'js>(
    _ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    selector: &str,
) -> bool {
    let host = match host_of(state) {
        Ok(h) => h,
        Err(_) => return false,
    };
    let Some(root) = host.borrow().document_node() else {
        return false;
    };
    host.borrow().query_selector_all(root, selector).contains(&key)
}

/// Resolve `input` against the document base URL. Returns the
/// absolute URL, or `input` unchanged when it cannot be resolved
/// (no base known / unparseable — the caller records it as-is).
fn resolve_url(state: &Rc<RefCell<BindingState>>, input: &str) -> String {
    let base = host_of(state)
        .ok()
        .and_then(|h| h.borrow().document_url())
        .unwrap_or_default();
    if base.is_empty() {
        return input.to_string();
    }
    match Url::parse(&base) {
        Ok(base_url) => match base_url.join(input) {
            Ok(joined) => joined.to_string(),
            Err(_) => input.to_string(),
        },
        Err(_) => input.to_string(),
    }
}

/// `data-foo-bar` → `fooBar` (for `dataset` keys).
fn dataset_key_from_attr(attr: &str) -> Option<String> {
    let rest = attr.strip_prefix("data-")?;
    if rest.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut cap_next = false;
    for c in rest.chars() {
        if c == '-' {
            cap_next = true;
        } else if cap_next {
            out.push(c.to_ascii_uppercase());
            cap_next = false;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// `fooBar` → `data-foo-bar` (for `dataset` reads / writes).
/// A `-` already in the name passes through, so `foo-bar` and
/// `fooBar` address the same attribute.
fn dataset_attr_from_key(prop: &str) -> String {
    let mut out = String::from("data-");
    for c in prop.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------- Animations
//
// M6 Phase 3 (Rec 11): `Element.animate` / `getAnimations` /
// `window.getComputedStyle`. Only the array-of-objects keyframe
// form is supported (the property-indexed form throws
// `TypeError`); per-step `easing` / `composite` are ignored.
// The host owns the ids and the engine; the binding owns the
// JS-visible `Animation` objects (identity via the registry's
// `animations` Map) and their local `playState` / `currentTime`.

/// Parse a duration value: a number (ms) or a `"1s"` / `"500ms"` string.
fn parse_duration_ms(ctx: &Ctx<'_>, value: &Value<'_>) -> rquickjs::Result<f32> {
    if let Some(n) = value.as_number() {
        return Ok(n as f32);
    }
    if let Some(s) = value.as_string() {
        let text = s.to_string().map_err(|e| {
            rquickjs::Error::new_from_js_message("TypeError", "duration", e.to_string())
        })?;
        let text = text.trim();
        if let Some(ms) = text.strip_suffix("ms") {
            return ms
                .trim()
                .parse::<f32>()
                .map_err(|_| throw_type_error(ctx, "invalid duration"));
        }
        if let Some(sec) = text.strip_suffix('s') {
            return sec
                .trim()
                .parse::<f32>()
                .map(|v| v * 1000.0)
                .map_err(|_| throw_type_error(ctx, "invalid duration"));
        }
    }
    Err(throw_type_error(ctx, "invalid duration"))
}

/// Parse `Element.animate` keyframes (array of objects) into
/// (offset, declarations) steps. Missing offsets are distributed
/// linearly between the surrounding explicit ones (0 at the start,
/// 1 at the end), per the Web Animations spec.
fn parse_keyframes(ctx: &Ctx<'_>, value: &Value<'_>) -> rquickjs::Result<Vec<WaapiStep>> {
    let Some(arr) = value.as_object() else {
        return Err(throw_type_error(ctx, "keyframes must be an array of objects"));
    };
    let len: u32 = arr
        .get("length")
        .map_err(|_| throw_type_error(ctx, "keyframes must be an array of objects"))?;
    let mut steps: Vec<(Option<f32>, Vec<(String, String)>)> = Vec::new();
    for i in 0..len {
        let item: Value = arr
            .get(i.to_string())
            .map_err(|_| throw_type_error(ctx, "keyframes must be an array of objects"))?;
        let Some(step_obj) = item.as_object() else {
            return Err(throw_type_error(ctx, "each keyframe must be an object"));
        };
        let offset: Option<f64> = step_obj.get("offset")?;
        let mut decls = Vec::new();
        for key in step_obj.keys::<String>().flatten() {
            if key == "offset" || key == "easing" || key == "composite" {
                continue;
            }
            let v: Coerced<String> = step_obj.get(key.as_str())?;
            decls.push((camel_to_kebab(&key), v.to_string()));
        }
        steps.push((offset.map(|o| o as f32), decls));
    }
    if steps.is_empty() {
        return Err(throw_type_error(ctx, "no keyframes"));
    }
    // Distribute missing offsets.
    let n = steps.len();
    let mut filled: Vec<f32> = vec![0.0; n];
    let mut prev_idx: Option<usize> = None;
    for i in 0..n {
        if let Some(o) = steps[i].0 {
            filled[i] = o.clamp(0.0, 1.0);
            if let Some(p) = prev_idx {
                // Fill the gap between p and i linearly.
                let gap = i - p;
                let start = filled[p];
                let end = filled[i];
                for j in (p + 1)..i {
                    filled[j] = start + (end - start) * ((j - p) as f32 / gap as f32);
                }
            }
            prev_idx = Some(i);
        }
    }
    match prev_idx {
        None => {
            // No explicit offset at all: spread evenly (a single
            // step sits at 0 — the engine holds it for the duration).
            for i in 0..n {
                filled[i] = if n == 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
            }
        }
        Some(last) => {
            for i in 0..last {
                if steps[i].0.is_none() && i == 0 {
                    filled[i] = 0.0;
                }
            }
            for i in (last + 1)..n {
                filled[i] = 1.0;
            }
        }
    }
    Ok(steps
        .into_iter()
        .enumerate()
        .map(|(i, (_, decls))| (filled[i], decls))
        .collect())
}

/// Parse the `Element.animate` options argument: a number (duration
/// in ms) or an object. Defaults: duration 0, delay 0, easing
/// `"linear"`, iterations 1, direction `"normal"`, fill `"none"`.
fn parse_animate_options(
    ctx: &Ctx<'_>,
    value: &Option<Value<'_>>,
) -> rquickjs::Result<AnimateSpec> {
    let mut spec = AnimateSpec {
        steps: Vec::new(),
        duration_ms: 0.0,
        delay_ms: 0.0,
        easing: "linear".to_string(),
        iterations: Some(1.0),
        direction: "normal".to_string(),
        fill: "none".to_string(),
    };
    let Some(value) = value else {
        return Ok(spec);
    };
    if value.is_number() {
        spec.duration_ms = parse_duration_ms(ctx, value)?;
        return Ok(spec);
    }
    let Some(obj) = value.as_object() else {
        return Err(throw_type_error(ctx, "animate options must be a number or an object"));
    };
    if let Ok(v) = obj.get::<_, Value>("duration") {
        if !v.is_undefined() && !v.is_null() {
            spec.duration_ms = parse_duration_ms(ctx, &v)?;
        }
    }
    if let Ok(v) = obj.get::<_, Value>("delay") {
        if !v.is_undefined() && !v.is_null() {
            spec.delay_ms = parse_duration_ms(ctx, &v)?;
        }
    }
    if let Ok(s) = obj.get::<_, String>("easing") {
        spec.easing = s;
    }
    if let Ok(v) = obj.get::<_, Value>("iterations") {
        if !v.is_undefined() && !v.is_null() {
            if let Some(n) = v.as_number() {
                spec.iterations = if n.is_infinite() { None } else { Some(n as f32) };
            }
        }
    }
    if let Ok(s) = obj.get::<_, String>("direction") {
        spec.direction = s;
    }
    if let Ok(s) = obj.get::<_, String>("fill") {
        spec.fill = s;
    }
    Ok(spec)
}

/// Get-or-create the JS `Animation` object for a running animation,
/// with its `playState` / `currentTime` getters and the
/// pause/play/cancel/finish methods. `anim_id` is `Some` for WAAPI
/// (host id) and `None` for CSS (identified by `name`).
#[allow(clippy::too_many_arguments)]
fn animation_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    key: DomKey,
    anim_id: Option<u64>,
    name: Option<String>,
    reg_id: u64,
    is_css: bool,
    duration_ms: f32,
) -> rquickjs::Result<Object<'js>> {
    let animations = registry_map(ctx, "animations")?;
    if let Ok(existing) = map_get(ctx, &animations, reg_id) {
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone());
        }
    }
    let obj = Object::new(ctx.clone())?;

    // playState (live from the local bookkeeping).
    let b = Rc::clone(bindings);
    let get_state = move |ctx: Ctx<'js>, This(_this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let state = b
            .borrow()
            .animation_locals
            .get(&reg_id)
            .map(|l| l.play_state.clone())
            .unwrap_or_else(|| "idle".to_string());
        Ok(js(&ctx, state))
    };
    obj.prop("playState", Accessor::from(get_state))?;

    // currentTime (ms since start, less paused time; null for CSS).
    let b = Rc::clone(bindings);
    let get_time = move |ctx: Ctx<'js>, This(_this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let ms = b
            .borrow()
            .animation_locals
            .get(&reg_id)
            .and_then(|l| l.current_time_ms());
        Ok(match ms {
            Some(v) => js(&ctx, v),
            None => js_null(&ctx),
        })
    };
    obj.prop("currentTime", Accessor::from(get_time))?;

    // pause / play / cancel / finish — each forwards to the host
    // (which queues it for the tab thread's engine) and updates
    // the local state so the getters move at once.
    for (method, op) in [
        ("pause", WaapiOp::Pause),
        ("play", WaapiOp::Play),
        ("cancel", WaapiOp::Cancel),
        ("finish", WaapiOp::Finish),
    ] {
        let b = Rc::clone(bindings);
        let name2 = name.clone();
        let f = Function::new(
            ctx.clone(),
            move |_ctx: Ctx<'js>, This(_this): This<Object<'js>>| -> rquickjs::Result<()> {
                let host = match host_of(&b) {
                    Ok(h) => h,
                    Err(_) => return Ok(()),
                };
                if let Some(id) = anim_id {
                    host.borrow_mut().waapi_control(key, id, op);
                } else if let Some(n) = &name2 {
                    host.borrow_mut().css_animation_control(key, n, op);
                }
                let now = Instant::now();
                let mut state = b.borrow_mut();
                match op {
                    WaapiOp::Pause => {
                        if let Some(l) = state.animation_locals.get_mut(&reg_id) {
                            if l.play_state == "running" {
                                l.play_state = "paused".to_string();
                                l.paused_at = Some(now);
                            }
                        }
                    }
                    WaapiOp::Play => {
                        if let Some(l) = state.animation_locals.get_mut(&reg_id) {
                            if l.play_state == "paused" {
                                if let Some(p) = l.paused_at.take() {
                                    l.paused_total += now.duration_since(p);
                                }
                                l.play_state = "running".to_string();
                            }
                        }
                        // A finished WAAPI animation is gone from
                        // the engine, so `play()` restarts it with
                        // the same spec (a fresh host id; the
                        // object keeps its identity). A finished
                        // CSS animation has no spec to replay with
                        // — `play()` leaves it finished.
                        if anim_id.is_some() {
                            let finished = state
                                .animation_locals
                                .get(&reg_id)
                                .map(|l| l.play_state == "finished")
                                .unwrap_or(false);
                            if finished {
                                if let Some(spec) = state.animation_specs.get(&reg_id).cloned() {
                                    drop(state);
                                    let new_id =
                                        host.borrow_mut().request_animate(key, spec);
                                    let mut state = b.borrow_mut();
                                    if let Some(new_id) = new_id {
                                        if let Some(old) = anim_id {
                                            state.animation_objects.remove(&(key, old));
                                        }
                                        state.animation_objects.insert((key, new_id), reg_id);
                                        if let Some(l) =
                                            state.animation_locals.get_mut(&reg_id)
                                        {
                                            l.play_state = "running".to_string();
                                            l.started = Instant::now();
                                            l.paused_at = None;
                                            l.paused_total = std::time::Duration::ZERO;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    WaapiOp::Cancel => {
                        if let Some(l) = state.animation_locals.get_mut(&reg_id) {
                            l.play_state = "idle".to_string();
                        }
                        if let Some(id) = anim_id {
                            state.animation_objects.remove(&(key, id));
                            state.animation_tombstones.insert((key, id));
                        } else if let Some(n) = &name2 {
                            state.css_animation_objects.remove(&(key, n.clone()));
                        }
                    }
                    WaapiOp::Finish => {
                        if let Some(l) = state.animation_locals.get_mut(&reg_id) {
                            l.play_state = "finished".to_string();
                            l.paused_at = None;
                        }
                    }
                }
                Ok(())
            },
        )?;
        def_method(ctx, &obj, method, f).map_err(|e| {
            rquickjs::Error::new_from_js_message("TypeError", "Animation", e)
        })?;
    }

    let _ = (is_css, duration_ms);
    map_set(ctx, &animations, reg_id, Value::from(obj.clone()))?;
    Ok(obj)
}

/// Create the `Animation` object for a freshly started animation
/// (WAAPI or CSS) and record its local state.
fn new_animation_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    key: DomKey,
    anim_id: Option<u64>,
    name: Option<String>,
    is_css: bool,
    duration_ms: f32,
    spec: Option<AnimateSpec>,
    initial_play_state: Option<&str>,
) -> rquickjs::Result<Object<'js>> {
    let reg_id = {
        let mut state = bindings.borrow_mut();
        let id = state.next_animation_id;
        state.next_animation_id += 1;
        state.animation_locals.insert(
            id,
            AnimationLocal {
                play_state: initial_play_state.unwrap_or("running").to_string(),
                started: Instant::now(),
                duration_ms,
                paused_at: None,
                paused_total: std::time::Duration::ZERO,
                is_css,
            },
        );
        if let Some(aid) = anim_id {
            state.animation_objects.insert((key, aid), id);
        } else if let Some(n) = &name {
            state.css_animation_objects.insert((key, n.clone()), id);
        }
        if let Some(spec) = spec {
            state.animation_specs.insert(id, spec);
        }
        id
    };
    animation_object(ctx, bindings, key, anim_id, name, reg_id, is_css, duration_ms)
}

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

    // dataset: a cached per-element Proxy object (see
    // `install_dataset_helpers`); reads/writes go through the
    // `data-*` attributes live.
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "dataset", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let id = wrapper_id_of(&ctx, &this).unwrap_or(0);
        let map = registry_map(&ctx, "datasets")?;
        let existing = map_get(&ctx, &map, id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone().into_value());
        }
        let reg = registry(&ctx)?;
        let factory: Function = reg.get("datasetFactory")?;
        let dataset: Value = factory.call((id as f64,))?;
        map_set(&ctx, &map, id, dataset.clone())?;
        let _ = key;
        Ok(dataset)
    })?;

    // cloneNode(deep): the host clones (attributes + optionally
    // children); the clone is detached. A stale key throws
    // `TypeError` (a method, not a getter).
    let b = Rc::clone(bindings);
    let clone = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, deep: Opt<bool>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            // Bind the clone first: the `borrow_mut()` guard must be
            // dropped before `wrap_or_null` borrows the state again.
            let cloned = host.borrow_mut().clone_node(key, deep.0.unwrap_or(false));
            wrap_or_null(&ctx, &b, cloned)
        },
    )
    .map_err(|e| format!("cloneNode: {e}"))?;
    def_method(ctx, element, "cloneNode", clone)?;

    // matches(selector): host-side selector test over the document.
    let b = Rc::clone(bindings);
    let matches = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<bool> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            Ok(matches_selector(&ctx, &b, key, &selector))
        },
    )
    .map_err(|e| format!("matches: {e}"))?;
    def_method(ctx, element, "matches", matches)?;

    // closest(selector): the nearest ancestor-or-self matching the
    // selector (element ancestors only — the document node never
    // matches an element selector).
    let b = Rc::clone(bindings);
    let closest = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, selector: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let mut cur = Some(dom_key_of_this(&ctx, &b, &this)?);
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            while let Some(key) = cur {
                if host.borrow().node_type(key) == Some(1)
                    && matches_selector(&ctx, &b, key, &selector)
                {
                    return wrap_or_null(&ctx, &b, Some(key));
                }
                cur = host.borrow().parent(key);
            }
            Ok(js_null(&ctx))
        },
    )
    .map_err(|e| format!("closest: {e}"))?;
    def_method(ctx, element, "closest", closest)?;

    // getElementsByTagName / getElementsByClassName /
    // getElementsByName: fresh snapshot arrays per access (M4: not
    // live HTMLCollections — documented in the module header).
    // Implemented through the host's `query_selector_all`, so the
    // matcher stays host-side.
    let b = Rc::clone(bindings);
    let by_tag = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, tag: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let selector = if tag.as_str() == "*" {
                "*".to_string()
            } else {
                tag.to_ascii_lowercase()
            };
            elements_array(&ctx, &b, host.borrow().query_selector_all(key, &selector))
        },
    )
    .map_err(|e| format!("getElementsByTagName: {e}"))?;
    def_method(ctx, element, "getElementsByTagName", by_tag)?;

    let b = Rc::clone(bindings);
    let by_class = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, classes: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let tokens: Vec<&str> = classes.split_ascii_whitespace().collect();
            if tokens.is_empty() {
                return Ok(Value::from(Array::new(ctx.clone())?));
            }
            let selector = tokens
                .iter()
                .map(|t| format!(".{t}"))
                .collect::<Vec<_>>()
                .join("");
            elements_array(&ctx, &b, host.borrow().query_selector_all(key, &selector))
        },
    )
    .map_err(|e| format!("getElementsByClassName: {e}"))?;
    def_method(ctx, element, "getElementsByClassName", by_class)?;

    let b = Rc::clone(bindings);
    let by_name = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let escaped = name.replace('\\', "\\\\").replace('"', "\\\"");
            let selector = format!("[name=\"{escaped}\"]");
            elements_array(&ctx, &b, host.borrow().query_selector_all(key, &selector))
        },
    )
    .map_err(|e| format!("getElementsByName: {e}"))?;
    def_method(ctx, element, "getElementsByName", by_name)?;

    // Form element APIs + validation + FormData (Stage 13 Rec 5/7/8).
    crate::bindings::forms::install_form_methods(ctx, bindings, element)?;

    // Web Animations API (M6 Phase 3, Rec 11): `Element.animate`
    // queues the animation with the host (the tab thread starts it
    // in the engine on the next tick) and returns an `Animation`
    // object; `getAnimations` returns the running animations for
    // this element (WAAPI by id, CSS by name).
    let b = Rc::clone(bindings);
    let animate = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>,
              This(this): This<Object<'js>>,
              keyframes: Value<'js>,
              options: Opt<Value<'js>>|
              -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if !host.borrow().validate(key) {
                return Err(throw_type_error(&ctx, "the element is no longer in the document"));
            }
            let steps = parse_keyframes(&ctx, &keyframes)?;
            let mut spec = parse_animate_options(&ctx, &options.0)?;
            spec.steps = steps;
            let duration_ms = spec.duration_ms;
            let anim_id = host.borrow_mut().request_animate(key, spec.clone());
            let obj = match anim_id {
                Some(id) => new_animation_object(
                    &ctx, &b, key, Some(id), None, false, duration_ms, Some(spec), None,
                )?,
                // No host: a stillborn Animation whose playState
                // is "idle" (it never runs anywhere).
                None => new_animation_object(
                    &ctx, &b, key, None, None, false, duration_ms, Some(spec), Some("idle"),
                )?,
            };
            Ok(Value::from(obj))
        },
    )
    .map_err(|e| format!("animate: {e}"))?;
    def_method(ctx, element, "animate", animate)?;

    let b = Rc::clone(bindings);
    let get_anims = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let infos: Vec<AnimationInfo> = host.borrow().get_animations(key);
            let arr = Array::new(ctx.clone())?;
            let mut i = 0usize;
            for info in infos {
                // A cancelled WAAPI animation stays reported by
                // the host until the tab thread drains the cancel
                // — the tombstone keeps `getAnimations` from
                // resurrecting it in between.
                if let Some(id) = info.id {
                    if b.borrow().animation_tombstones.contains(&(key, id)) {
                        continue;
                    }
                }
                let obj = match (info.id, info.name) {
                    (Some(id), _) => {
                        let reg_id = b.borrow().animation_objects.get(&(key, id)).copied();
                        match reg_id {
                            Some(reg_id) => {
                                // Refresh the local playState from
                                // the host's finished flag.
                                if info.finished {
                                    if let Some(l) =
                                        b.borrow_mut().animation_locals.get_mut(&reg_id)
                                    {
                                        l.play_state = "finished".to_string();
                                    }
                                }
                                animation_object(
                                    &ctx, &b, key, Some(id), None, reg_id, false, f32::INFINITY,
                                )?
                            }
                            None => new_animation_object(
                                &ctx, &b, key, Some(id), None, false, f32::INFINITY, None, None,
                            )?,
                        }
                    }
                    (None, Some(name)) => {
                        let reg_id = b
                            .borrow()
                            .css_animation_objects
                            .get(&(key, name.clone()))
                            .copied();
                        match reg_id {
                            Some(reg_id) => animation_object(
                                &ctx, &b, key, None, Some(name), reg_id, true, f32::INFINITY,
                            )?,
                            None => new_animation_object(
                                &ctx, &b, key, None, Some(name), true, f32::INFINITY, None, None,
                            )?,
                        }
                    }
                    (None, None) => continue,
                };
                arr.set(i, Value::from(obj))?;
                i += 1;
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("getAnimations: {e}"))?;
    def_method(ctx, element, "getAnimations", get_anims)?;

    Ok(())
}

/// `HTMLImageElement` methods (Stage 12 Rec 7). The prototype
/// chains off `Element.prototype`; only `<img>` wrappers reach it
/// (see `wrap_node`). `src` writes go through the host's
/// `request_image` hook — the shell decodes via the Stage 11 store
/// and reports back with `TabJs::notify_image_loaded` /
/// `notify_image_failed`, which fire `load` / `error` here.
fn install_image_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    image: &Object<'js>,
) -> Result<(), String> {
    // src (getter: the resolved URL; setter: record + ask the host
    // to fetch/decode).
    let b = Rc::clone(bindings);
    let get_src = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        if let Some(state) = b.borrow().image_states.get(&key)
            && !state.current_src.is_empty()
        {
            return Ok(js(&ctx, state.current_src.clone()));
        }
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().get_attribute(key, "src") {
            Some(v) => js(&ctx, resolve_url(&b, &v)),
            None => js(&ctx, ""),
        })
    };
    let b = Rc::clone(bindings);
    let set_src = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let src = to_dom_string(&ctx, &value)?;
        set_image_src(&ctx, &b, key, &src)
    };
    def_accessor(ctx, image, "src", get_src, set_src)?;

    // currentSrc (the resolved URL of the current request, "" if none).
    let b = Rc::clone(bindings);
    def_getter(ctx, image, "currentSrc", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        if let Some(state) = b.borrow().image_states.get(&key) {
            return Ok(js(&ctx, state.current_src.clone()));
        }
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().get_attribute(key, "src") {
            Some(v) => js(&ctx, resolve_url(&b, &v)),
            None => js(&ctx, ""),
        })
    })?;

    // complete / naturalWidth / naturalHeight (from the load state
    // the shell reports; 0 / false until a load completes).
    let b = Rc::clone(bindings);
    def_getter(ctx, image, "complete", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(false);
        };
        Ok(b.borrow()
            .image_states
            .get(&key)
            .map(|s| s.complete)
            .unwrap_or(false))
    })?;
    for (prop, is_width) in [("naturalWidth", true), ("naturalHeight", false)] {
        let b = Rc::clone(bindings);
        def_getter(
            ctx,
            image,
            prop,
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                    return Ok(0.0);
                };
                Ok(b.borrow()
                    .image_states
                    .get(&key)
                    .map(|s| if is_width { s.width } else { s.height } as f64)
                    .unwrap_or(0.0))
            },
        )?;
    }

    // width / height: reflect the content attributes (0 when absent).
    for prop in ["width", "height"] {
        install_attr_int_accessor(ctx, bindings, image, prop)?;
    }
    Ok(())
}

/// An integer content-attribute accessor (`width`, `height` on
/// `HTMLImageElement`): reads the attribute as a number, writes it
/// back as a decimal string.
fn install_attr_int_accessor<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    proto: &Object<'js>,
    prop: &str,
) -> Result<(), String> {
    let attr = prop.to_string();
    let b = Rc::clone(bindings);
    let a = attr.clone();
    let get = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(0.0);
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(0.0),
        };
        Ok(host
            .borrow()
            .get_attribute(key, &a)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0))
    };
    let b = Rc::clone(bindings);
    let set = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let n = to_dom_string(&ctx, &value)?
            .parse::<f64>()
            .unwrap_or(0.0);
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host
            .borrow_mut()
            .set_attribute(key, &attr, &format!("{}", n as u64))
        {
            return Err(throw_type_error(&ctx, "the element is no longer in the document"));
        }
        Ok(())
    };
    def_accessor(ctx, proto, prop, get, set)
}

/// Set an image's `src`: write the attribute, resolve the URL
/// against the document base, ask the host to fetch/decode it, and
/// reset the load state to "not loaded". The `load` / `error`
/// event fires later, from the shell's report
/// (`fire_image_event`).
fn set_image_src(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    src: &str,
) -> rquickjs::Result<()> {
    let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
    if !host.borrow_mut().set_attribute(key, "src", src) {
        return Err(throw_type_error(ctx, "the element is no longer in the document"));
    }
    let resolved = resolve_url(state, src);
    let base = host
        .borrow()
        .document_url()
        .unwrap_or_default();
    host.borrow_mut().request_image(&resolved, &base);
    state.borrow_mut().image_states.insert(
        key,
        ImageLoadState {
            current_src: resolved,
            complete: false,
            width: 0,
            height: 0,
        },
    );
    Ok(())
}

/// Fire a `load` or `error` event on an image element (called by
/// `TabJs::notify_image_loaded` / `notify_image_failed` when the
/// shell's Stage 11 store reports the outcome). Updates the load
/// state first, then dispatches to listeners, then calls the
/// `onload` / `onerror` property handler if one is set.
pub fn fire_image_event<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    kind: &str,
    width: u32,
    height: u32,
) -> rquickjs::Result<()> {
    {
        let mut s = state.borrow_mut();
        let entry = s.image_states.entry(key).or_default();
        if kind == "load" {
            entry.complete = true;
            entry.width = width;
            entry.height = height;
        } else {
            entry.complete = false;
        }
    }
    let Some(id) = state.borrow().wrapper_ids.get(&key).copied() else {
        return Ok(()); // Never wrapped in JS: nothing can listen.
    };
    let wrappers = registry_map(ctx, "wrappers")?;
    let wrapper = map_get(ctx, &wrappers, id)?;
    let Some(obj) = wrapper.as_object() else {
        return Ok(());
    };
    let event: Object = ctx.eval(format!(
        "new Event({kind:?}, {{ bubbles: false, cancelable: false }})"
    ))?;
    dispatch_to_target(ctx, state, id, &event, kind)?;
    let prop = if kind == "load" { "onload" } else { "onerror" };
    if let Ok(handler) = obj.get::<_, Value>(prop)
        && handler.is_function()
    {
        let func = Function::from_js(ctx, handler)?;
        let _: Value = func.call((This(Value::from(obj.clone())), Value::from(event)))?;
    }
    Ok(())
}

/// Install the `Image(width?, height?)` constructor on the global
/// object (Stage 12 Rec 7). A Rust `Function::new` is not
/// constructable, so — like `Response` / `Headers` — the
/// constructor is a JS function delegating to a Rust factory in
/// the registry; the factory returns the wrapper object, which
/// `new` then adopts.
fn install_image_constructor<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        // `Value` params, not `Opt<f64>`: rquickjs's `Opt` only
        // covers *missing* arguments — the `Image` wrapper below
        // always passes both slots, so `new Image()` arrives as
        // two explicit `undefined`s and `Opt<f64>` throws
        // "converting from js 'undefined' into type 'f64'".
        // `undefined` / `null` here mean "absent" (no attribute
        // set); anything else coerces the way `f64::from_js`
        // coerces it (so `new Image("100")` still works).
        move |ctx: Ctx<'js>, width: Value<'js>, height: Value<'js>| -> rquickjs::Result<Value<'js>> {
            let opt_number = |v: Value<'js>| -> rquickjs::Result<Option<f64>> {
                if v.is_undefined() || v.is_null() {
                    Ok(None)
                } else {
                    f64::from_js(&ctx, v).map(Some)
                }
            };
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let Some(key) = host.borrow_mut().create_element("img") else {
                return Err(throw_type_error(&ctx, "could not create image element"));
            };
            if let Some(w) = opt_number(width)? {
                host.borrow_mut()
                    .set_attribute(key, "width", &format!("{}", w as u64));
            }
            if let Some(h) = opt_number(height)? {
                host.borrow_mut()
                    .set_attribute(key, "height", &format!("{}", h as u64));
            }
            let obj = wrap_node(&ctx, &b, key).map(Value::from)?;
            if let Some(o) = obj.as_object() {
                // `onload` / `onerror` start as `null` (IDL attributes).
                o.set("onload", js_null(&ctx))?;
                o.set("onerror", js_null(&ctx))?;
            }
            Ok(obj)
        },
    )
    .map_err(|e| format!("Image factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("imageFactory", factory)
        .map_err(|e| format!("install Image factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.Image = function Image(width, height) { \
           if (!(this instanceof Image)) throw new TypeError('Image constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].imageFactory(width, height); \
         };",
    )
    .map_err(|e| format!("install Image: {e}"))?;
    let globals = ctx.globals();
    let ctor: Object = globals.get("Image").map_err(|e| e.to_string())?;
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    ctor.set(
        "prototype",
        protos
            .get::<_, Object>("imageElement")
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Image.prototype: {e}"))?;
    Ok(())
}

/// Install the `dataset` helpers: Rust functions in the registry
/// plus a JS factory that builds a `Proxy` over them. The `dataset`
/// getter on `Element.prototype` calls the factory once per
/// element and caches the Proxy (like `classList` / `style`).
/// Property reads/writes go through the `data-*` attributes live:
/// `el.dataset.fooBar` reads `data-foo-bar`, writing it sets the
/// attribute, `delete` removes it, `Object.keys` lists the
/// camelCase names.
pub fn install_dataset_helpers<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;

    // datasetGet(wrapperId, prop) -> string | undefined.
    let b = Rc::clone(bindings);
    let get = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64, prop: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let Some(key) = b.borrow().wrapper_keys.get(&(id as u64)).copied() else {
                return Ok(js_null(&ctx));
            };
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            // Absent attributes read as `undefined`, not `null`
            // (the Proxy's `get` trap distinguishes them).
            let attr = dataset_attr_from_key(&prop);
            Ok(match host.borrow().get_attribute(key, &attr) {
                Some(v) => js(&ctx, v),
                None => js(&ctx, rquickjs::Undefined),
            })
        },
    )
    .map_err(|e| format!("datasetGet: {e}"))?;
    reg.set("datasetGet", get)
        .map_err(|e| format!("install datasetGet: {e}"))?;

    // datasetSet(wrapperId, prop, value) -> bool.
    let b = Rc::clone(bindings);
    let set = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, id: f64, prop: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<bool> {
            let Some(key) = b.borrow().wrapper_keys.get(&(id as u64)).copied() else {
                return Ok(false);
            };
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(false),
            };
            let attr = dataset_attr_from_key(&prop);
            Ok(host.borrow_mut().set_attribute(key, &attr, &value))
        },
    )
    .map_err(|e| format!("datasetSet: {e}"))?;
    reg.set("datasetSet", set)
        .map_err(|e| format!("install datasetSet: {e}"))?;

    // datasetDelete(wrapperId, prop).
    let b = Rc::clone(bindings);
    let delete = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, id: f64, prop: Coerced<String>| -> rquickjs::Result<()> {
            if let Some(key) = b.borrow().wrapper_keys.get(&(id as u64)).copied()
                && let Ok(host) = host_of(&b)
            {
                let attr = dataset_attr_from_key(&prop);
                host.borrow_mut().remove_attribute(key, &attr);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("datasetDelete: {e}"))?;
    reg.set("datasetDelete", delete)
        .map_err(|e| format!("install datasetDelete: {e}"))?;

    // datasetKeys(wrapperId) -> [camelCase names].
    let b = Rc::clone(bindings);
    let keys = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, id: f64| -> rquickjs::Result<Value<'js>> {
            let arr = Array::new(ctx.clone())?;
            if let Some(key) = b.borrow().wrapper_keys.get(&(id as u64)).copied()
                && let Ok(host) = host_of(&b)
            {
                let names = host.borrow().attribute_names(key);
                let mut i = 0;
                for name in names {
                    if let Some(camel) = dataset_key_from_attr(&name) {
                        arr.set(i, Value::from(js(&ctx, camel)))?;
                        i += 1;
                    }
                }
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("datasetKeys: {e}"))?;
    reg.set("datasetKeys", keys)
        .map_err(|e| format!("install datasetKeys: {e}"))?;

    ctx.eval::<(), _>(
        "globalThis[Symbol.for('pluto.registry')].datasetFactory = function (id) { \
           const reg = globalThis[Symbol.for('pluto.registry')]; \
           return new Proxy({}, { \
             get(t, p) { \
               if (typeof p !== 'string') return undefined; \
               return reg.datasetGet(id, p); \
             }, \
             set(t, p, v) { \
               if (typeof p !== 'string') return false; \
               reg.datasetSet(id, p, String(v)); \
               return true; \
             }, \
             deleteProperty(t, p) { \
               if (typeof p === 'string') reg.datasetDelete(id, p); \
               return true; \
             }, \
             has(t, p) { \
               return typeof p === 'string' && reg.datasetGet(id, p) !== undefined; \
             }, \
             ownKeys(t) { return reg.datasetKeys(id); }, \
             getOwnPropertyDescriptor(t, p) { \
               const v = reg.datasetGet(id, p); \
               if (v === undefined) return undefined; \
               return { value: v, writable: true, enumerable: true, configurable: true }; \
             } \
           }); \
         };",
    )
    .map_err(|e| format!("install datasetFactory: {e}"))?;
    Ok(())
}
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

    // getElementsByTagName / getElementsByClassName /
    // getElementsByName, scoped at the document node (fresh
    // snapshot arrays per access — M4, not live collections).
    let b = Rc::clone(bindings);
    let doc_by_tag = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, tag: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let scope = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let selector = if tag.as_str() == "*" {
                "*".to_string()
            } else {
                tag.to_ascii_lowercase()
            };
            elements_array(&ctx, &b, host.borrow().query_selector_all(scope, &selector))
        },
    )
    .map_err(|e| format!("document.getElementsByTagName: {e}"))?;
    def_method(ctx, document, "getElementsByTagName", doc_by_tag)?;

    let b = Rc::clone(bindings);
    let doc_by_class = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, classes: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let scope = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let tokens: Vec<&str> = classes.split_ascii_whitespace().collect();
            if tokens.is_empty() {
                return Ok(Value::from(Array::new(ctx.clone())?));
            }
            let selector = tokens
                .iter()
                .map(|t| format!(".{t}"))
                .collect::<Vec<_>>()
                .join("");
            elements_array(&ctx, &b, host.borrow().query_selector_all(scope, &selector))
        },
    )
    .map_err(|e| format!("document.getElementsByClassName: {e}"))?;
    def_method(ctx, document, "getElementsByClassName", doc_by_class)?;

    let b = Rc::clone(bindings);
    let doc_by_name = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let scope = dom_key_of_this(&ctx, &b, &this)?;
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
            };
            let escaped = name.replace('\\', "\\\\").replace('"', "\\\"");
            let selector = format!("[name=\"{escaped}\"]");
            elements_array(&ctx, &b, host.borrow().query_selector_all(scope, &selector))
        },
    )
    .map_err(|e| format!("document.getElementsByName: {e}"))?;
    def_method(ctx, document, "getElementsByName", doc_by_name)?;

    // document.images: the `<img>` elements in document order
    // (fresh snapshot; each wrapper carries the image prototype).
    let b = Rc::clone(bindings);
    def_getter(ctx, document, "images", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let scope = dom_key_of_this(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        elements_array(&ctx, &b, host.borrow().query_selector_all(scope, "img"))
    })?;

    // document.forms: the `<form>` elements in document order.
    let b = Rc::clone(bindings);
    def_getter(ctx, document, "forms", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        elements_array(&ctx, &b, host.borrow().document_forms())
    })?;

    // document.activeElement: the focused element, or `<body>`
    // when nothing is focused (null when there is no body yet).
    let b = Rc::clone(bindings);
    def_getter(ctx, document, "activeElement", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        let host = host.borrow();
        match host.active_element().or_else(|| host.body()) {
            Some(key) => Ok(Value::from(wrap_node(&ctx, &b, key)?)),
            None => Ok(js_null(&ctx)),
        }
    })?;

    // document.cookie (Stage 13 Rec 10c): non-HttpOnly cookies
    // for the document origin; writes go through the jar's
    // `store_set_cookie` semantics in the host.
    let b = Rc::clone(bindings);
    let get_cookie = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().document_cookie().unwrap_or_default()))
    };
    let b = Rc::clone(bindings);
    let set_cookie = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let _ = dom_key_of_this(&ctx, &b, &this)?;
        let text = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        host.borrow_mut().set_document_cookie(&text);
        Ok(())
    };
    def_accessor(ctx, document, "cookie", get_cookie, set_cookie)?;

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

    install_location(ctx, bindings).map_err(|e| format!("install location: {e}"))?;

    // `window.getComputedStyle` (M6 Phase 3, Rec 11): a read-only
    // snapshot object whose own properties are the host's cached
    // computed values (animatable properties the engine has
    // sampled — see `DomHost::computed_styles`), under both the
    // CSS name (`background-color`) and its camelCase alias
    // (`backgroundColor`), plus `getPropertyValue(name)`. The
    // pseudo-element argument is accepted and ignored (v1: no
    // pseudo-element styles). A non-element argument throws
    // `TypeError`, per WebIDL.
    let b = Rc::clone(bindings);
    let gcs = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, el: Value<'js>, _pseudo: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let key = node_arg(&ctx, &b, &el)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let pairs = host.borrow().computed_styles(key);
            let obj = Object::new(ctx.clone())?;
            for (name, value) in &pairs {
                obj.set(name.as_str(), value.as_str())?;
                let camel = kebab_to_camel(name);
                if camel != *name {
                    obj.set(camel.as_str(), value.as_str())?;
                }
            }
            let b2 = Rc::clone(&b);
            let get_prop = Function::new(
                ctx.clone(),
                move |ctx: Ctx<'js>, This(_this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
                    let host = match host_of(&b2) {
                        Ok(h) => h,
                        Err(_) => return Ok(js(&ctx, "")),
                    };
                    Ok(match host.borrow().computed_style(key, &name.to_ascii_lowercase()) {
                        Some(v) => js(&ctx, v),
                        None => js(&ctx, ""),
                    })
                },
            )?;
            obj.set("getPropertyValue", get_prop)?;
            Ok(Value::from(obj))
        },
    )
    .map_err(|e| format!("install getComputedStyle: {e}"))?;
    globals
        .set("getComputedStyle", gcs)
        .map_err(|e| format!("install getComputedStyle: {e}"))?;

    // `window` is an EventTarget: the same three methods live on the
    // global object (its prototype chain does not include
    // EventTarget.prototype — `globalThis` is the window).
    install_window_listeners(ctx, bindings)
        .map_err(|e| format!("install window listeners: {e}"))?;
    Ok(())
}

/// Install `window.location` (Stage 12 Rec 7): a plain object whose
/// getters read the host's document URL fresh on every access.
/// `assign` / `replace` / `reload` do not navigate in the realm —
/// they record the request with the host (`request_navigation`),
/// and the shell performs the navigation (the next document's
/// `document_url` is then what `location` reads). With no host
/// URL known, `location` reads as `about:blank`.
fn install_location<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    let location = Object::new(ctx.clone()).map_err(|e| format!("location object: {e}"))?;

    // The current URL, parsed (or `about:blank` when unknown).
    let current_url = |b: &Rc<RefCell<BindingState>>| -> Url {
        let raw = host_of(b)
            .ok()
            .and_then(|h| h.borrow().document_url())
            .unwrap_or_else(|| "about:blank".to_string());
        Url::parse(&raw).unwrap_or_else(|_| Url::parse("about:blank").unwrap())
    };

    // href (getter + setter: setting `href` navigates, like assign).
    let b = Rc::clone(bindings);
    let get_href = move |ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<Value<'js>> {
        Ok(js(&ctx, current_url(&b).to_string()))
    };
    let b = Rc::clone(bindings);
    let set_href = move |ctx: Ctx<'js>, This(_this): This<Value<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        navigate_to(&ctx, &b, &to_dom_string(&ctx, &value)?, false);
        Ok(())
    };
    location
        .prop("href", Accessor::new(get_href, set_href))
        .map_err(|e| format!("location.href: {e}"))?;

    // Read-only parts.
    for (prop, part) in [
        ("protocol", 0u8),
        ("host", 1u8),
        ("hostname", 2u8),
        ("port", 3u8),
        ("pathname", 4u8),
        ("search", 5u8),
        ("hash", 6u8),
        ("origin", 7u8),
    ] {
        let b = Rc::clone(bindings);
        let get = move |ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let url = current_url(&b);
            let value = match part {
                0 => url.scheme().to_string() + ":",
                1 => url.host_str().map(|h| {
                    match url.port() {
                        Some(p) => format!("{h}:{p}"),
                        None => h.to_string(),
                    }
                }).unwrap_or_default(),
                2 => url.host_str().unwrap_or_default().to_string(),
                3 => url.port().map(|p| p.to_string()).unwrap_or_default(),
                4 => url.path().to_string(),
                5 => match url.query() {
                    Some(q) => format!("?{q}"),
                    None => String::new(),
                },
                6 => match url.fragment() {
                    Some(f) => format!("#{f}"),
                    None => String::new(),
                },
                _ => url.origin().ascii_serialization(),
            };
            Ok(js(&ctx, value))
        };
        location
            .prop(prop, Accessor::from(get))
            .map_err(|e| format!("location.{prop}: {e}"))?;
    }

    // assign(url) / replace(url): resolve against the current URL,
    // then record the navigation with the host. An unparseable URL
    // is ignored (per spec, navigation to it does nothing).
    for (name, replace) in [("assign", false), ("replace", true)] {
        let b = Rc::clone(bindings);
        let func = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, input: Coerced<String>| -> rquickjs::Result<()> {
                navigate_to(&ctx, &b, &input, replace);
                Ok(())
            },
        )
        .map_err(|e| format!("location.{name}: {e}"))?;
        location
            .set(name, func)
            .map_err(|e| format!("location.{name}: {e}"))?;
    }

    // reload(): re-request the current URL. The host records it as
    // a navigation to the same URL; the shell treats a same-URL
    // request as a reload (no new history entry is implied here —
    // the history itself is M8).
    let b = Rc::clone(bindings);
    let reload = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>| -> rquickjs::Result<()> {
            let href = current_url(&b).to_string();
            if let Ok(host) = host_of(&b) {
                host.borrow_mut().request_navigation(&href, false);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("location.reload: {e}"))?;
    location
        .set("reload", reload)
        .map_err(|e| format!("location.reload: {e}"))?;

    // toString(): the href.
    let b = Rc::clone(bindings);
    let to_string = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            Ok(js(&ctx, current_url(&b).to_string()))
        },
    )
    .map_err(|e| format!("location.toString: {e}"))?;
    location
        .set("toString", to_string)
        .map_err(|e| format!("location.toString: {e}"))?;

    globals
        .set("location", location)
        .map_err(|e| format!("install location: {e}"))?;
    Ok(())
}

/// Resolve `input` against the current location URL and record the
/// navigation with the host. Silently ignored when the URL does
/// not parse.
fn navigate_to(ctx: &Ctx<'_>, bindings: &Rc<RefCell<BindingState>>, input: &str, replace: bool) {
    let base = host_of(bindings)
        .ok()
        .and_then(|h| h.borrow().document_url())
        .unwrap_or_else(|| "about:blank".to_string());
    let resolved = Url::parse(input)
        .ok()
        .or_else(|| Url::parse(&base).ok()?.join(input).ok())
        .map(|u| u.to_string());
    if let (Some(url), Ok(host)) = (resolved, host_of(bindings)) {
        host.borrow_mut().request_navigation(&url, replace);
    }
    let _ = ctx;
}

/// Install `console` (`log` / `info` / `debug` / `warn` / `error` →
/// the host log, via `tracing`, plus a captured copy in
/// [`BindingState::console_lines`] for tests).
///
/// Stage 12 Rec 10 adds `trace`, `assert`, `count` / `countReset`,
/// `time` / `timeEnd` / `timeLog`, `dir`, `table`, and
/// `group` / `groupCollapsed` / `groupEnd` — same routing (tracing
/// + captured lines). `group` nesting indents every captured line
/// by two spaces per level (depth 0 indents nothing, so the
/// original five methods' output is unchanged outside a group).
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
                push_console_line(&b, &level, &message);
                Ok(())
            },
        )
        .map_err(|e| format!("console.{name}: {e}"))?;
        console
            .set(name, func)
            .map_err(|e| format!("console.{name}: {e}"))?;
    }

    // trace(...args): the message plus a stack capture.
    let b = Rc::clone(bindings);
    let trace = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
            let parts: Vec<String> =
                args.0.iter().map(|v| format_console_arg(&ctx, v)).collect();
            let message = parts.join(" ");
            let stack: String = ctx
                .eval::<String, _>("new Error().stack")
                .unwrap_or_default();
            let line = if stack.is_empty() {
                format!("Trace: {message}")
            } else {
                format!("Trace: {message}\n{stack}")
            };
            push_console_line(&b, "trace", &line);
            Ok(())
        },
    )
    .map_err(|e| format!("console.trace: {e}"))?;
    console.set("trace", trace).map_err(|e| format!("console.trace: {e}"))?;

    // assert(condition, ...args): logs only when false.
    let b = Rc::clone(bindings);
    let assert_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, condition: Value<'js>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
            let ok = condition.as_bool().unwrap_or(false);
            if !ok {
                let parts: Vec<String> =
                    args.0.iter().map(|v| format_console_arg(&ctx, v)).collect();
                let detail = parts.join(" ");
                let line = if detail.is_empty() {
                    "Assertion failed".to_string()
                } else {
                    format!("Assertion failed: {detail}")
                };
                push_console_line(&b, "error", &line);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("console.assert: {e}"))?;
    console.set("assert", assert_fn).map_err(|e| format!("console.assert: {e}"))?;

    // count(label?) / countReset(label?): per-label counters.
    let b = Rc::clone(bindings);
    let count = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, label: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let label = match label.0 {
                Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                _ => "default".to_string(),
            };
            let n = {
                let mut s = b.borrow_mut();
                let n = s.console_counts.get(&label).copied().unwrap_or(0) + 1;
                s.console_counts.insert(label.clone(), n);
                n
            };
            push_console_line(&b, "log", &format!("{label}: {n}"));
            Ok(())
        },
    )
    .map_err(|e| format!("console.count: {e}"))?;
    console.set("count", count).map_err(|e| format!("console.count: {e}"))?;

    let b = Rc::clone(bindings);
    let count_reset = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, label: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let label = match label.0 {
                Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                _ => "default".to_string(),
            };
            b.borrow_mut().console_counts.remove(&label);
            push_console_line(&b, "log", &format!("{label}: 0"));
            Ok(())
        },
    )
    .map_err(|e| format!("console.countReset: {e}"))?;
    console
        .set("countReset", count_reset)
        .map_err(|e| format!("console.countReset: {e}"))?;

    // time(label?) / timeEnd(label?) / timeLog(label?, ...args):
    // per-label wall-clock timers in ms.
    let b = Rc::clone(bindings);
    let time = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, label: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let label = match label.0 {
                Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                _ => "default".to_string(),
            };
            b.borrow_mut()
                .console_timers
                .insert(label, Instant::now());
            Ok(())
        },
    )
    .map_err(|e| format!("console.time: {e}"))?;
    console.set("time", time).map_err(|e| format!("console.time: {e}"))?;

    let b = Rc::clone(bindings);
    let time_end = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, label: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let label = match label.0 {
                Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                _ => "default".to_string(),
            };
            // Bind the removed timer first: the `borrow_mut()`
            // temporary would otherwise live through the `match`
            // arms and `push_console_line` would re-borrow.
            let start = b.borrow_mut().console_timers.remove(&label);
            match start {
                Some(start) => {
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    push_console_line(&b, "log", &format!("{label}: {ms:.3} ms"));
                }
                None => {
                    push_console_line(&b, "warn", &format!("Timer '{label}' does not exist"));
                }
            }
            Ok(())
        },
    )
    .map_err(|e| format!("console.timeEnd: {e}"))?;
    console.set("timeEnd", time_end).map_err(|e| format!("console.timeEnd: {e}"))?;

    let b = Rc::clone(bindings);
    let time_log = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, label: Opt<Value<'js>>, args: Rest<Value<'js>>| -> rquickjs::Result<()> {
            let label = match label.0 {
                Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                _ => "default".to_string(),
            };
            // Same temporary-lifetime trap as `timeEnd`: bind first.
            let start = b.borrow().console_timers.get(&label).copied();
            match start {
                Some(start) => {
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let extra: Vec<String> = args
                        .0
                        .iter()
                        .map(|v| format_console_arg(&ctx, v))
                        .collect();
                    let line = if extra.is_empty() {
                        format!("{label}: {ms:.3} ms")
                    } else {
                        format!("{label}: {ms:.3} ms {}", extra.join(" "))
                    };
                    push_console_line(&b, "log", &line);
                }
                None => {
                    push_console_line(&b, "warn", &format!("Timer '{label}' does not exist"));
                }
            }
            Ok(())
        },
    )
    .map_err(|e| format!("console.timeLog: {e}"))?;
    console.set("timeLog", time_log).map_err(|e| format!("console.timeLog: {e}"))?;

    // dir(value): structured inspection via JSON (2-space indent),
    // falling back to `String(value)` when it is not stringifiable.
    let b = Rc::clone(bindings);
    let dir = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, value: Value<'js>| -> rquickjs::Result<()> {
            let rendered = stringify_pretty(&ctx, &value)
                .unwrap_or_else(|| format_console_arg(&ctx, &value));
            push_console_line(&b, "log", &rendered);
            Ok(())
        },
    )
    .map_err(|e| format!("console.dir: {e}"))?;
    console.set("dir", dir).map_err(|e| format!("console.dir: {e}"))?;

    // table(data): a text table of an array / object, one row per
    // entry, columns aligned.
    let b = Rc::clone(bindings);
    let table = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, data: Value<'js>| -> rquickjs::Result<()> {
            let rendered = render_table(&ctx, &data);
            push_console_line(&b, "log", &rendered);
            Ok(())
        },
    )
    .map_err(|e| format!("console.table: {e}"))?;
    console.set("table", table).map_err(|e| format!("console.table: {e}"))?;

    // group(label?) / groupCollapsed(label?) / groupEnd(): indent
    // nesting for the captured lines.
    for (name, collapsed) in [("group", false), ("groupCollapsed", true)] {
        let b = Rc::clone(bindings);
        let group = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, label: Opt<Value<'js>>| -> rquickjs::Result<()> {
                let label = match label.0 {
                    Some(v) if !v.is_undefined() => to_dom_string(&ctx, &v)?,
                    _ => String::new(),
                };
                let shown = if collapsed {
                    format!("{label} (collapsed)")
                } else {
                    label
                };
                if !shown.is_empty() {
                    push_console_line(&b, "log", &shown);
                }
                b.borrow_mut().console_group_depth += 1;
                Ok(())
            },
        )
        .map_err(|e| format!("console.{name}: {e}"))?;
        console
            .set(name, group)
            .map_err(|e| format!("console.{name}: {e}"))?;
    }
    let b = Rc::clone(bindings);
    let group_end = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>| -> rquickjs::Result<()> {
            let mut s = b.borrow_mut();
            s.console_group_depth = s.console_group_depth.saturating_sub(1);
            Ok(())
        },
    )
    .map_err(|e| format!("console.groupEnd: {e}"))?;
    console
        .set("groupEnd", group_end)
        .map_err(|e| format!("console.groupEnd: {e}"))?;

    globals
        .set("console", console)
        .map_err(|e| format!("install console: {e}"))?;
    Ok(())
}

/// Push one captured console line, indented by the current group
/// depth, and mirror it to `tracing` (target `pluto-js:console`).
fn push_console_line(state: &Rc<RefCell<BindingState>>, level: &str, message: &str) {
    let indent = "  ".repeat(state.borrow().console_group_depth);
    let line = format!("{indent}{message}");
    match level {
        "warn" => tracing::warn!(target: "pluto-js:console", "{line}"),
        "error" => tracing::error!(target: "pluto-js:console", "{line}"),
        "debug" => tracing::debug!(target: "pluto-js:console", "{line}"),
        _ => tracing::info!(target: "pluto-js:console", "{line}"),
    }
    state
        .borrow_mut()
        .console_lines
        .push((level.to_string(), line));
}

/// `JSON.stringify(value, null, 2)`, or `None` when the value does
/// not stringify (e.g. `undefined`, functions).
fn stringify_pretty<'js>(ctx: &Ctx<'js>, value: &Value<'js>) -> Option<String> {
    let json: Object = ctx.globals().get("JSON").ok()?;
    let stringify: Function = json.get("stringify").ok()?;
    let out: Value = stringify.call((value.clone(), rquickjs::Null, 2)).ok()?;
    out.as_string()?.to_string().ok()
}

/// Render `data` as a text table: one column per key (arrays of
/// objects), `(index)` + `Values` for arrays of primitives, one
/// row per entry. Columns are padded to the widest cell.
fn render_table<'js>(ctx: &Ctx<'js>, data: &Value<'js>) -> String {
    // Collect (row label, cells) in order.
    let mut columns: Vec<String> = Vec::new();
    let mut rows: Vec<(String, Vec<String>)> = Vec::new();
    if let Some(obj) = data.as_object()
        && let Some(arr) = obj.clone().into_array()
    {
        let len: usize = arr.len();
        let mut entries: Vec<Value> = Vec::new();
        for i in 0..len {
            if let Ok(v) = arr.get::<Value>(i) {
                entries.push(v);
            }
        }
        // Columns: the union of object keys, in first-seen order;
        // a primitive entry uses a single `Values` column.
        let mut has_values = false;
        for entry in &entries {
            if let Some(o) = entry.as_object()
                && o.clone().into_array().is_none()
            {
                for key in o.keys::<String>().flatten() {
                    if !columns.contains(&key) {
                        columns.push(key);
                    }
                }
            } else {
                has_values = true;
            }
        }
        if has_values && !columns.contains(&"Values".to_string()) {
            columns.push("Values".to_string());
        }
        for (i, entry) in entries.iter().enumerate() {
            let mut cells = vec![String::new(); columns.len()];
            if let Some(o) = entry.as_object()
                && o.clone().into_array().is_none()
            {
                for (j, col) in columns.iter().enumerate() {
                    if let Ok(v) = o.get::<_, Value>(col.clone()) {
                        cells[j] = format_console_arg(ctx, &v);
                    }
                }
            } else if let Some(j) = columns.iter().position(|c| c == "Values") {
                cells[j] = format_console_arg(ctx, entry);
            }
            rows.push((i.to_string(), cells));
        }
    } else if let Some(obj) = data.as_object() {
        columns = vec!["Value".to_string()];
        for key in obj.keys::<String>().flatten() {
            if let Ok(v) = obj.get::<_, Value>(key.clone()) {
                rows.push((key, vec![format_console_arg(ctx, &v)]));
            }
        }
    } else {
        return format_console_arg(ctx, data);
    }
    // Widths: the `(index)` column plus one per data column.
    let header = std::iter::once("(index)".to_string())
        .chain(columns.iter().cloned())
        .collect::<Vec<_>>();
    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for (label, cells) in &rows {
        widths[0] = widths[0].max(label.chars().count());
        for (j, cell) in cells.iter().enumerate() {
            widths[j + 1] = widths[j + 1].max(cell.chars().count());
        }
    }
    let pad = |s: &str, w: usize| -> String {
        let mut out = s.to_string();
        while out.chars().count() < w {
            out.push(' ');
        }
        out
    };
    let mut lines = Vec::new();
    lines.push(
        header
            .iter()
            .enumerate()
            .map(|(j, h)| pad(h, widths[j]))
            .collect::<Vec<_>>()
            .join(" | "),
    );
    for (label, cells) in &rows {
        let mut parts = vec![pad(label, widths[0])];
        for (j, cell) in cells.iter().enumerate() {
            parts.push(pad(cell, widths[j + 1]));
        }
        lines.push(parts.join(" | "));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------- URL

/// The id stored on a `URL` / `URLSearchParams` object.
fn url_id_of<'js>(ctx: &Ctx<'js>, obj: &Object<'js>) -> Option<u64> {
    let sym = Symbol::new_global(ctx.clone(), URL_SYMBOL).ok()?;
    let value: Value = obj.get(sym.as_atom()).ok()?;
    value.as_number().map(|n| n as u64)
}

/// Query pairs of a `UrlEntry`: from the URL's query, or from the
/// stored serialised query for a standalone `URLSearchParams`.
fn url_entry_params(entry: &UrlEntry) -> Vec<(String, String)> {
    if entry.is_url {
        Url::parse(&entry.text)
            .map(|u| u.query_pairs().into_owned().collect())
            .unwrap_or_default()
    } else {
        url::form_urlencoded::parse(entry.text.as_bytes())
            .into_owned()
            .collect()
    }
}

/// Serialise `params` back into `entry` (re-serialising the URL's
/// query, or storing the query string for a standalone params
/// object).
fn url_entry_set_params(entry: &mut UrlEntry, params: &[(String, String)]) {
    let query = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(params)
        .finish();
    if entry.is_url {
        if let Ok(mut url) = Url::parse(&entry.text) {
            if query.is_empty() {
                url.set_query(None);
            } else {
                url.set_query(Some(&query));
            }
            entry.text = url.to_string();
        }
    } else {
        entry.text = query;
    }
}

/// Make a `URL` / `URLSearchParams` wrapper object for `id`, on
/// the named prototype.
fn make_url_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    proto_name: &str,
    id: u64,
) -> rquickjs::Result<Object<'js>> {
    let protos = registry_map(ctx, "prototypes")?;
    let proto: Object = protos.get(proto_name)?;
    let obj = Object::new_proto(ctx.clone(), Some(&proto))?;
    let sym = Symbol::new_global(ctx.clone(), URL_SYMBOL)?;
    obj.set(sym.as_atom(), id as f64)?;
    let _ = bindings;
    Ok(obj)
}

/// Mutate the parsed URL behind `id` (a no-op when the entry is
/// missing or is not a URL object).
fn with_url(
    bindings: &Rc<RefCell<BindingState>>,
    id: u64,
    f: impl FnOnce(&mut Url),
) {
    let mut s = bindings.borrow_mut();
    let Some(entry) = s.url_entries.get_mut(&id) else {
        return;
    };
    if !entry.is_url {
        return;
    }
    if let Ok(mut url) = Url::parse(&entry.text) {
        f(&mut url);
        entry.text = url.to_string();
    }
}

/// Read one component of the parsed URL behind `id`.
fn url_part(bindings: &Rc<RefCell<BindingState>>, id: u64, part: u8) -> String {
    let entry = bindings.borrow().url_entries.get(&id).cloned();
    let Some(entry) = entry else {
        return String::new();
    };
    if !entry.is_url {
        return String::new();
    }
    let Ok(url) = Url::parse(&entry.text) else {
        return String::new();
    };
    match part {
        // protocol
        0 => url.scheme().to_string() + ":",
        // host
        1 => url
            .host_str()
            .map(|h| match url.port() {
                Some(p) => format!("{h}:{p}"),
                None => h.to_string(),
            })
            .unwrap_or_default(),
        // hostname
        2 => url.host_str().unwrap_or_default().to_string(),
        // port
        3 => url.port().map(|p| p.to_string()).unwrap_or_default(),
        // pathname
        4 => url.path().to_string(),
        // search
        5 => match url.query() {
            Some(q) => format!("?{q}"),
            None => String::new(),
        },
        // hash
        6 => match url.fragment() {
            Some(f) => format!("#{f}"),
            None => String::new(),
        },
        // origin
        _ => url.origin().ascii_serialization(),
    }
}

/// Install `URL` and `URLSearchParams` (Stage 12 Rec 7), parsed
/// by the `url` crate (already a workspace dependency).
pub fn install_url<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;

    // --- URL.prototype ---
    let url_proto = Object::new(ctx.clone()).map_err(|e| format!("URL prototype: {e}"))?;
    // href (getter + setter: an invalid assignment is ignored).
    let b = Rc::clone(bindings);
    let get_href = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(id) = url_id_of(&ctx, &this) else {
            return Ok(js(&ctx, ""));
        };
        Ok(js(
            &ctx,
            b.borrow()
                .url_entries
                .get(&id)
                .map(|e| e.text.clone())
                .unwrap_or_default(),
        ))
    };
    let b = Rc::clone(bindings);
    let set_href = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let Some(id) = url_id_of(&ctx, &this) else {
            return Ok(());
        };
        let href = to_dom_string(&ctx, &value)?;
        if Url::parse(&href).is_ok() {
            if let Some(entry) = b.borrow_mut().url_entries.get_mut(&id) {
                entry.text = href;
            }
        }
        Ok(())
    };
    url_proto
        .prop("href", Accessor::new(get_href, set_href))
        .map_err(|e| format!("URL.href: {e}"))?;

    // Read-only parts + their setters (invalid values ignored).
    // `origin` is read-only.
    for (prop, part) in [
        ("protocol", 0u8),
        ("host", 1u8),
        ("hostname", 2u8),
        ("port", 3u8),
        ("pathname", 4u8),
        ("search", 5u8),
        ("hash", 6u8),
        ("origin", 7u8),
    ] {
        let b = Rc::clone(bindings);
        if part == 7 {
            let get = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                let Some(id) = url_id_of(&ctx, &this) else {
                    return Ok(js(&ctx, ""));
                };
                Ok(js(&ctx, url_part(&b, id, part)))
            };
            url_proto
                .prop(prop, Accessor::from(get))
                .map_err(|e| format!("URL.{prop}: {e}"))?;
            continue;
        }
        let get = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(js(&ctx, ""));
            };
            let b2 = Rc::clone(&b);
            Ok(js(&ctx, url_part(&b2, id, part)))
        };
        let b = Rc::clone(bindings);
        let set = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(());
            };
            let v = to_dom_string(&ctx, &value)?;
            with_url(&b, id, |url| match part {
                0 => {
                    let _ = url.set_scheme(v.trim_end_matches(':'));
                }
                1 | 2 => {
                    let _ = url.set_host(Some(&v));
                }
                3 => {
                    let _ = url.set_port(if v.is_empty() {
                        None
                    } else {
                        v.parse::<u16>().ok()
                    });
                }
                4 => url.set_path(&v),
                5 => url.set_query(if v.is_empty() {
                    None
                } else {
                    Some(v.strip_prefix('?').unwrap_or(&v))
                }),
                _ => url.set_fragment(if v.is_empty() {
                    None
                } else {
                    Some(v.strip_prefix('#').unwrap_or(&v))
                }),
            });
            Ok(())
        };
        url_proto
            .prop(prop, Accessor::new(get, set))
            .map_err(|e| format!("URL.{prop}: {e}"))?;
    }

    // searchParams (cached per URL, `===`-stable).
    let b = Rc::clone(bindings);
    let get_params = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(id) = url_id_of(&ctx, &this) else {
            return Err(throw_type_error(&ctx, "called on a non-URL object"));
        };
        let map = registry_map(&ctx, "urlSearchParams")?;
        let existing = map_get(&ctx, &map, id)?;
        if let Some(obj) = existing.as_object() {
            return Ok(obj.clone().into_value());
        }
        let obj = make_url_object(&ctx, &b, "urlSearchParams", id)?;
        map_set(&ctx, &map, id, Value::from(obj.clone()))?;
        Ok(Value::from(obj))
    };
    url_proto
        .prop("searchParams", Accessor::from(get_params))
        .map_err(|e| format!("URL.searchParams: {e}"))?;

    // toString().
    let b = Rc::clone(bindings);
    let to_string = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(js(&ctx, ""));
            };
            Ok(js(
                &ctx,
                b.borrow()
                    .url_entries
                    .get(&id)
                    .map(|e| e.text.clone())
                    .unwrap_or_default(),
            ))
        },
    )
    .map_err(|e| format!("URL.toString: {e}"))?;
    url_proto
        .set("toString", to_string)
        .map_err(|e| format!("URL.toString: {e}"))?;
    protos
        .set("url", url_proto)
        .map_err(|e| format!("prototypes url: {e}"))?;

    install_search_params_prototype(ctx, bindings)?;

    // --- URL constructor ---
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, input: Coerced<String>, base: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let base_url: Option<String> = match base.0 {
                Some(v) if !v.is_null() && !v.is_undefined() => {
                    if let Some(obj) = v.as_object()
                        && let Some(id) = url_id_of(&ctx, obj)
                    {
                        b.borrow()
                            .url_entries
                            .get(&id)
                            .map(|e| e.text.clone())
                    } else {
                        Some(to_dom_string(&ctx, &v)?)
                    }
                }
                _ => None,
            };
            let parsed = match Url::parse(&input) {
                Ok(u) => Some(u),
                Err(_) => match base_url.as_deref() {
                    Some(base) => Url::parse(base)
                        .ok()
                        .and_then(|b| b.join(&input).ok()),
                    None => None,
                },
            };
            let Some(url) = parsed else {
                return Err(throw_type_error(&ctx, "Invalid URL"));
            };
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_url_id;
                s.next_url_id += 1;
                s.url_entries.insert(
                    id,
                    UrlEntry {
                        is_url: true,
                        text: url.to_string(),
                    },
                );
                id
            };
            make_url_object(&ctx, &b, "url", id).map(Value::from)
        },
    )
    .map_err(|e| format!("URL factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("urlFactory", factory)
        .map_err(|e| format!("install URL factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.URL = function URL(input, base) { \
           if (!(this instanceof URL)) throw new TypeError('URL constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].urlFactory(input, base); \
         };",
    )
    .map_err(|e| format!("install URL: {e}"))?;
    let globals = ctx.globals();
    let url_ctor: Object = globals.get("URL").map_err(|e| e.to_string())?;
    url_ctor
        .set("prototype", protos.get::<_, Object>("url").map_err(|e| e.to_string())?)
        .map_err(|e| format!("URL.prototype: {e}"))?;

    // --- URLSearchParams constructor ---
    let b = Rc::clone(bindings);
    let params_factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, init: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let text = match init.0 {
                None => String::new(),
                Some(v) if v.is_null() || v.is_undefined() => String::new(),
                Some(v) => {
                    if let Some(obj) = v.as_object() {
                        if let Some(id) = url_id_of(&ctx, obj) {
                            // Another URLSearchParams (or a URL — its query).
                            let params = b
                                .borrow()
                                .url_entries
                                .get(&id)
                                .map(url_entry_params)
                                .unwrap_or_default();
                            url::form_urlencoded::Serializer::new(String::new())
                                .extend_pairs(&params)
                                .finish()
                        } else if let Some(arr) = obj.clone().into_array() {
                            let mut params = Vec::new();
                            let len: usize = arr.len();
                            for i in 0..len {
                                if let Ok(pair) = arr.get::<Value>(i)
                                    && let Some(pair_obj) = pair.as_object()
                                        && let Some(pair_arr) = pair_obj.clone().into_array()
                                            && pair_arr.len() >= 2
                                {
                                    let n: Value = pair_arr.get(0)?;
                                    let val: Value = pair_arr.get(1)?;
                                    params.push((
                                        to_dom_string(&ctx, &n)?,
                                        to_dom_string(&ctx, &val)?,
                                    ));
                                }
                            }
                            url::form_urlencoded::Serializer::new(String::new())
                                .extend_pairs(&params)
                                .finish()
                        } else {
                            // A record object: keys in enumeration order.
                            let mut params = Vec::new();
                            for key in obj.keys::<String>() {
                                let key = key?;
                                let val: Value = obj.get(key.clone())?;
                                if let Some(arr_obj) = val.as_object().and_then(|o| o.clone().into_array()) {
                                    let len: usize = arr_obj.len();
                                    for i in 0..len {
                                        let item: Value = arr_obj.get(i)?;
                                        params.push((key.clone(), to_dom_string(&ctx, &item)?));
                                    }
                                } else {
                                    params.push((key, to_dom_string(&ctx, &val)?));
                                }
                            }
                            url::form_urlencoded::Serializer::new(String::new())
                                .extend_pairs(&params)
                                .finish()
                        }
                    } else {
                        let s = to_dom_string(&ctx, &v)?;
                        s.strip_prefix('?').unwrap_or(&s).to_string()
                    }
                }
            };
            let id = {
                let mut s = b.borrow_mut();
                let id = s.next_url_id;
                s.next_url_id += 1;
                s.url_entries.insert(
                    id,
                    UrlEntry {
                        is_url: false,
                        text,
                    },
                );
                id
            };
            make_url_object(&ctx, &b, "urlSearchParams", id).map(Value::from)
        },
    )
    .map_err(|e| format!("URLSearchParams factory: {e}"))?;
    reg.set("urlSearchParamsFactory", params_factory)
        .map_err(|e| format!("install URLSearchParams factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.URLSearchParams = function URLSearchParams(init) { \
           if (!(this instanceof URLSearchParams)) throw new TypeError('URLSearchParams constructor requires new'); \
           return globalThis[Symbol.for('pluto.registry')].urlSearchParamsFactory(init); \
         };",
    )
    .map_err(|e| format!("install URLSearchParams: {e}"))?;
    let params_ctor: Object = globals
        .get("URLSearchParams")
        .map_err(|e| e.to_string())?;
    params_ctor
        .set(
            "prototype",
            protos
                .get::<_, Object>("urlSearchParams")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("URLSearchParams.prototype: {e}"))?;
    Ok(())
}

/// The `URLSearchParams` prototype: `get` / `getAll` / `set` /
/// `append` / `delete` / `has` / `toString` / iteration. Mutations
/// on a URL's `searchParams` re-serialise the URL's query (the
/// two share one `UrlEntry` id).
fn install_search_params_prototype<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let protos = registry_map(ctx, "prototypes").map_err(|e| format!("prototypes map: {e}"))?;
    let proto = Object::new(ctx.clone()).map_err(|e| format!("URLSearchParams prototype: {e}"))?;

    let params_of = |b: &Rc<RefCell<BindingState>>, id: u64| -> Vec<(String, String)> {
        b.borrow()
            .url_entries
            .get(&id)
            .map(url_entry_params)
            .unwrap_or_default()
    };
    let set_params = |b: &Rc<RefCell<BindingState>>, id: u64, params: &[(String, String)]| {
        if let Some(entry) = b.borrow_mut().url_entries.get_mut(&id) {
            url_entry_set_params(entry, params);
        }
    };

    // get(name) -> string | null.
    let b = Rc::clone(bindings);
    let get = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(js_null(&ctx));
            };
            Ok(match params_of(&b, id).into_iter().find(|(n, _)| *n == name.as_str()) {
                Some((_, v)) => js(&ctx, v),
                None => js_null(&ctx),
            })
        },
    )
    .map_err(|e| format!("URLSearchParams.get: {e}"))?;
    proto.set("get", get).map_err(|e| format!("URLSearchParams.get: {e}"))?;

    // getAll(name) -> [string].
    let b = Rc::clone(bindings);
    let get_all = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(Value::from(Array::new(ctx.clone())?));
            };
            let arr = Array::new(ctx.clone())?;
            let mut i = 0;
            for (n, v) in params_of(&b, id) {
                if n == name.as_str() {
                    arr.set(i, Value::from(js(&ctx, v)))?;
                    i += 1;
                }
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("URLSearchParams.getAll: {e}"))?;
    proto.set("getAll", get_all).map_err(|e| format!("URLSearchParams.getAll: {e}"))?;

    // has(name).
    let b = Rc::clone(bindings);
    let has = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<bool> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(false);
            };
            Ok(params_of(&b, id).iter().any(|(n, _)| *n == name.as_str()))
        },
    )
    .map_err(|e| format!("URLSearchParams.has: {e}"))?;
    proto.set("has", has).map_err(|e| format!("URLSearchParams.has: {e}"))?;

    // set(name, value): drop all pairs with the name, append one.
    let b = Rc::clone(bindings);
    let set = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(());
            };
            let mut params = params_of(&b, id);
            params.retain(|(n, _)| *n != name.as_str());
            params.push((name.to_string(), value.to_string()));
            set_params(&b, id, &params);
            Ok(())
        },
    )
    .map_err(|e| format!("URLSearchParams.set: {e}"))?;
    proto.set("set", set).map_err(|e| format!("URLSearchParams.set: {e}"))?;

    // append(name, value).
    let b = Rc::clone(bindings);
    let append = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>, value: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(());
            };
            let mut params = params_of(&b, id);
            params.push((name.to_string(), value.to_string()));
            set_params(&b, id, &params);
            Ok(())
        },
    )
    .map_err(|e| format!("URLSearchParams.append: {e}"))?;
    proto.set("append", append).map_err(|e| format!("URLSearchParams.append: {e}"))?;

    // delete(name).
    let b = Rc::clone(bindings);
    let delete = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, name: Coerced<String>| -> rquickjs::Result<()> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(());
            };
            let mut params = params_of(&b, id);
            params.retain(|(n, _)| *n != name.as_str());
            set_params(&b, id, &params);
            Ok(())
        },
    )
    .map_err(|e| format!("URLSearchParams.delete: {e}"))?;
    proto.set("delete", delete).map_err(|e| format!("URLSearchParams.delete: {e}"))?;

    // toString(): the serialised query (no leading `?`).
    let b = Rc::clone(bindings);
    let to_string = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(js(&ctx, ""));
            };
            let text = b
                .borrow()
                .url_entries
                .get(&id)
                .map(|e| {
                    if e.is_url {
                        Url::parse(&e.text)
                            .map(|u| u.query().unwrap_or_default().to_string())
                            .unwrap_or_default()
                    } else {
                        e.text.clone()
                    }
                })
                .unwrap_or_default();
            Ok(js(&ctx, text))
        },
    )
    .map_err(|e| format!("URLSearchParams.toString: {e}"))?;
    proto.set("toString", to_string).map_err(|e| format!("URLSearchParams.toString: {e}"))?;

    // keys() / values() / entries(): arrays, in insertion order.
    for (name, which) in [("keys", 0u8), ("values", 1u8), ("entries", 2u8)] {
        let b = Rc::clone(bindings);
        let f = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
                let Some(id) = url_id_of(&ctx, &this) else {
                    return Ok(Value::from(Array::new(ctx.clone())?));
                };
                let arr = Array::new(ctx.clone())?;
                for (i, (n, v)) in params_of(&b, id).into_iter().enumerate() {
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
        .map_err(|e| format!("URLSearchParams.{name}: {e}"))?;
        proto.set(name, f).map_err(|e| format!("URLSearchParams.{name}: {e}"))?;
    }

    // forEach(callback, thisArg?).
    let b = Rc::clone(bindings);
    let for_each = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, callback: Function<'js>, this_arg: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let Some(id) = url_id_of(&ctx, &this) else {
                return Ok(());
            };
            let this_value = this_arg.0.unwrap_or_else(|| Value::from(this.clone()));
            for (n, v) in params_of(&b, id) {
                let _: Value = callback.call((This(this_value.clone()), js(&ctx, v), js(&ctx, n), Value::from(this.clone())))?;
            }
            Ok(())
        },
    )
    .map_err(|e| format!("URLSearchParams.forEach: {e}"))?;
    proto.set("forEach", for_each).map_err(|e| format!("URLSearchParams.forEach: {e}"))?;

    protos
        .set("urlSearchParams", proto.clone())
        .map_err(|e| format!("prototypes urlSearchParams: {e}"))?;
    // `for (const [k, v] of params)`: the default iterator is `entries`.
    super::install_entries_iterator(&ctx, &proto)
        .map_err(|e| format!("URLSearchParams iterator: {e}"))?;
    Ok(())
}

/// `throw_named` re-export for submodules that build exceptions.
pub fn throw_js(ctx: &Ctx<'_>, name: &str, msg: &str) -> rquickjs::Error {
    throw_named(ctx, name, msg)
}
