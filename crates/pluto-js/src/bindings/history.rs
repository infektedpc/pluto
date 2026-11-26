//! `window.history` (Stage 16 Phase 3, Rec 8): the History API
//! core — `pushState` / `replaceState` / `back` / `forward` /
//! `go` / `length` / `state`.
//!
//! The shell owns the session history (`Vec<SessionHistoryEntry>`
//! in `pluto-browser`); the realm mirrors the current entry's
//! state (`BindingState::history_state_json`) and the length /
//! index (`history_length` / `history_index`, synced by the
//! shell's `HistorySync`). `pushState` / `replaceState` update
//! the realm state at once and notify the host, which forwards
//! `HistoryPush` / `HistoryReplace` to the shell on the next tick
//! drain; `back` / `forward` / `go` ask the host to forward
//! `HistoryGo`, and the shell navigates with
//! `CommitNavigation { traversal: true }` — the worker fires
//! `popstate` after that commit (see `TabJs::dispatch_popstate`).
//!
//! State is JSON, not structured clone: `pushState` serializes
//! with `JSON.stringify` and the realm parses it back with
//! `JSON.parse` on read / traversal. A state that is not
//! JSON-serializable (`undefined`, a function, a `Symbol`) stores
//! as `null`; a `Map` / `Date` / class instance degrades to its
//! JSON shape. Full structured clone (with its cross-realm value
//! store) is a later milestone — this keeps the shell's history
//! plain data it can persist and send over IPC.

use std::cell::RefCell;
use std::rc::Rc;

use rquickjs::function::{Opt, This};
use rquickjs::{Ctx, Function, Object, Value};

use crate::bindings::dom::host_of;
use crate::bindings::{
    BindingState, js_null, throw_dom, throw_type_error, to_dom_string,
};
use crate::host::DomHost;

/// Install `window.history` on the global object.
pub fn install_history<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    let history = Object::new(ctx.clone()).map_err(|e| format!("history object: {e}"))?;

    // `length`: the shell-mirrored entry count.
    let b = Rc::clone(bindings);
    history
        .prop(
            "length",
            rquickjs::object::Accessor::from(
                move |ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<f64> {
                    let _ = ctx;
                    Ok(b.borrow().history_length as f64)
                },
            ),
        )
        .map_err(|e| format!("history.length: {e}"))?;

    // `state`: the current entry's state, parsed back from JSON
    // (`null` when there is none).
    let b = Rc::clone(bindings);
    history
        .prop(
            "state",
            rquickjs::object::Accessor::from(
                move |ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<Value<'js>> {
                    match b.borrow().history_state_json.clone() {
                        Some(json) => {
                            let json_obj: Object = ctx.globals().get("JSON")?;
                            let parse: Function = json_obj.get("parse")?;
                            parse
                                .call::<_, Value>((json,))
                                .or_else(|_| Ok(js_null(&ctx)))
                        }
                        None => Ok(js_null(&ctx)),
                    }
                },
            ),
        )
        .map_err(|e| format!("history.state: {e}"))?;

    // `pushState(state, unusedTitle, url?)` / `replaceState(...)`.
    for (name, replace) in [("pushState", false), ("replaceState", true)] {
        let b = Rc::clone(bindings);
        let func = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>,
                  state: Value<'js>,
                  _title: Opt<Value<'js>>,
                  url: Opt<Value<'js>>|
                  -> rquickjs::Result<()> {
                history_push_replace(&ctx, &b, state, url.0, replace)
            },
        )
        .map_err(|e| format!("history.{name}: {e}"))?;
        history
            .set(name, func)
            .map_err(|e| format!("history.{name}: {e}"))?;
    }

    // `back()` / `forward()` / `go(delta)`: ask the host to move
    // the shell's index (a no-op with no host — the realm never
    // navigates itself).
    for (name, delta) in [("back", -1i32), ("forward", 1i32)] {
        let b = Rc::clone(bindings);
        let func = Function::new(
            ctx.clone(),
            move |ctx: Ctx<'js>| -> rquickjs::Result<()> {
                if let Ok(host) = host_of(&b) {
                    host.borrow_mut().request_history_traversal(delta);
                }
                let _ = ctx;
                Ok(())
            },
        )
        .map_err(|e| format!("history.{name}: {e}"))?;
        history
            .set(name, func)
            .map_err(|e| format!("history.{name}: {e}"))?;
    }
    let b = Rc::clone(bindings);
    let go_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, delta: Opt<f64>| -> rquickjs::Result<()> {
            let delta = delta.0.unwrap_or(0.0) as i32;
            // `go(0)` reloads the current entry (spec) — forward it
            // too; the shell re-commits the current entry.
            if let Ok(host) = host_of(&b) {
                host.borrow_mut().request_history_traversal(delta);
            }
            let _ = ctx;
            Ok(())
        },
    )
    .map_err(|e| format!("history.go: {e}"))?;
    history
        .set("go", go_fn)
        .map_err(|e| format!("history.go: {e}"))?;

    globals
        .set("history", history)
        .map_err(|e| format!("install history: {e}"))?;
    Ok(())
}

/// Shared `pushState` / `replaceState` body.
fn history_push_replace<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    state: Value<'js>,
    url: Option<Value<'js>>,
    replace: bool,
) -> rquickjs::Result<()> {
    let host = host_of(bindings).map_err(|e| throw_type_error(&ctx, &e))?;
    let current = host
        .borrow()
        .document_url()
        .unwrap_or_else(|| "about:blank".to_string());
    // Resolve the new URL against the current one (when given);
    // an unparseable URL throws `SecurityError` per WebIDL
    // (`pushState` with a bad URL is a `SecurityError`, not a
    // no-op — unlike `location.assign`).
    let target = match url {
        Some(value) => {
            let input = to_dom_string(ctx, &value)?;
            let base = url::Url::parse(&current).map_err(|_| {
                throw_dom(&ctx, "SecurityError", "history: current URL is not parseable")
            })?;
            base.join(&input).map_err(|_| {
                throw_dom(
                    &ctx,
                    "SecurityError",
                    "history: the URL could not be resolved",
                )
            })?
        }
        None => url::Url::parse(&current).map_err(|_| {
            throw_dom(&ctx, "SecurityError", "history: current URL is not parseable")
        })?,
    };
    // Same-origin check (spec: the URL's origin must equal the
    // document's, else `SecurityError`). Compared as parsed
    // origins (scheme + host + port), never as strings.
    let current_url = url::Url::parse(&current).map_err(|_| {
        throw_dom(&ctx, "SecurityError", "history: current URL is not parseable")
    })?;
    if target.origin() != current_url.origin() {
        return Err(throw_dom(
            &ctx,
            "SecurityError",
            "history: a cross-origin URL cannot be pushed",
        ));
    }
    let target_str = target.to_string();
    // Serialize the state: `JSON.stringify` — `undefined` (and
    // functions / symbols) stringify to `undefined`, which stores
    // as `null` (no state).
    let state_json: Option<String> = {
        let json_obj: Object = ctx.globals().get("JSON")?;
        let stringify: Function = json_obj.get("stringify")?;
        let value: Value = stringify.call((state,))?;
        value.as_string().and_then(|s| s.to_string().ok())
    };
    // Update the realm state at once: `history.state` and
    // `location.href` read the new values immediately, before the
    // shell hears about it.
    {
        let mut s = bindings.borrow_mut();
        s.history_state_json = state_json.clone();
        if replace {
            // `replaceState` keeps the entry count and index.
        } else {
            // `pushState` truncates the forward entries, then
            // pushes: length = index + 2, index += 1. The shell
            // is authoritative and re-syncs with `HistorySync`;
            // this keeps `history.length` right in the meantime.
            s.history_length = s.history_index + 2;
            s.history_index += 1;
        }
    }
    // The host updates its document URL (so `location.href` reads
    // the new URL) and records the push/replace for the tab
    // thread, which forwards it to the shell.
    {
        let mut host = host.borrow_mut();
        host.set_document_url(&target_str);
        if replace {
            host.request_history_replace(&target_str, state_json.as_deref());
        } else {
            host.request_history_push(&target_str, state_json.as_deref());
        }
    }
    Ok(())
}
