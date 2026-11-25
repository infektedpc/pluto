//! Binding tests (wave 2a): every item of §(f) 1–5 exercised
//! through `TabJs` + the [`TestHost`], plus the wrapper-identity /
//! stale-key / symbol-registry evidence the task requires.
//!
//! The tests run page JS and assert on the DOM (via the host) and on
//! JS-visible values (via `TabJs::eval_value`), so a passing test
//! means the whole crossing works — not just the Rust side.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;

use pluto_ipc::{Origin, RequestMode};

use crate::bindings::fetch::{
    FetchBackend, FetchCompletion, FetchError, FetchRequest, FetchResponse, ResponseType,
};
use crate::bindings::events::TypedEventKind;
use crate::host::DomHost;
use crate::bindings::test_host::{TestHost, TestHostLog};
use crate::engine::JsValue;
use crate::observers::ComputedIntersection;
use crate::realm::RealmConfig;
use crate::tab::TabJs;

const DOC: &str = r#"<!doctype html>
<html><head><title>Test Page</title></head><body>
<div id="app" class="root"><p id="para">Hello</p><button id="btn">Click</button></div>
<ul id="list"><li class="item">one</li><li class="item">two</li><li class="item">three</li></ul>
</body></html>"#;

fn test_tab() -> TabJs {
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(TestHost::from_html(DOC)));
    tab
}

/// Run `source` as a script task, then drive the loop to idle.
fn run(tab: &mut TabJs, source: &str) {
    tab.eval_script(source, "test.js").unwrap();
    tab.run_until_idle().unwrap();
}

fn read_str(tab: &mut TabJs, source: &str) -> String {
    match tab.eval_value(source).unwrap() {
        JsValue::String(s) => s,
        other => panic!("expected a string from `{source}`, got {other:?}"),
    }
}

fn read_num(tab: &mut TabJs, source: &str) -> f64 {
    match tab.eval_value(source).unwrap() {
        JsValue::Number(n) => n,
        other => panic!("expected a number from `{source}`, got {other:?}"),
    }
}

fn read_bool(tab: &mut TabJs, source: &str) -> bool {
    match tab.eval_value(source).unwrap() {
        JsValue::Bool(b) => b,
        other => panic!("expected a bool from `{source}`, got {other:?}"),
    }
}

/// A fake backend: records requests, answers every fetch with the
/// same canned response (synchronously, on the calling thread — the
/// loop still resolves through the Networking task queue).
struct FakeBackend {
    requests: Arc<Mutex<Vec<FetchRequest>>>,
    response: FetchResponse,
    error: Option<FetchError>,
    /// The document origin this backend fetches for, if any:
    /// `fetch_impl` reads it for the realm-side gate call
    /// (`check_fetch_allowed`), which since the Stage 9 Rec 2
    /// completion is a pass-through — cross-origin enforcement
    /// lives browser-side, not in the realm. `None` means the
    /// realm cannot know the origin.
    origin: Option<Origin>,
}

impl FakeBackend {
    fn ok(body: &str) -> (Arc<Self>, Arc<Mutex<Vec<FetchRequest>>>) {
        Self::ok_with_origin(body, None)
    }

    fn ok_with_origin(
        body: &str,
        origin: Option<Origin>,
    ) -> (Arc<Self>, Arc<Mutex<Vec<FetchRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let backend = Arc::new(Self {
            requests: Arc::clone(&requests),
            response: FetchResponse {
                status: 200,
                status_text: "OK".to_string(),
                headers: vec![
                    ("content-type".to_string(), "application/json".to_string()),
                    ("x-test".to_string(), "yes".to_string()),
                ],
                body: body.as_bytes().to_vec(),
                response_type: ResponseType::Basic,
            },
            error: None,
            origin,
        });
        (backend, requests)
    }

    fn failing(error: FetchError) -> Arc<Self> {
        Arc::new(Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            response: FetchResponse {
                status: 0,
                status_text: String::new(),
                headers: Vec::new(),
                body: Vec::new(),
                response_type: ResponseType::Opaque,
            },
            error: Some(error),
            origin: None,
        })
    }
}

impl FetchBackend for FakeBackend {
    fn document_origin(&self) -> Option<Origin> {
        self.origin.clone()
    }

    fn start_fetch(&self, request: FetchRequest, done: mpsc::Sender<FetchCompletion>) {
        let id = request.id;
        self.requests.lock().unwrap().push(request);
        let result = match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(self.response.clone()),
        };
        let _ = done.send(FetchCompletion { id, result });
    }
}

// ---------------------------------------------------------------- window

#[test]
fn window_identity_and_document() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__r = [
           window === globalThis,
           window.window === window,
           window.self === window,
           typeof document,
           document.nodeType,
           document.documentElement.tagName,
           document.nodeName,
         ].join(',');",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "true,true,true,object,9,HTML,#document"
    );
}

#[test]
fn timer_and_raf_maps_are_not_reachable_by_name() {
    // Wave-1 gap (a), fixed: the Maps live behind a symbol key.
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__r = [
           typeof __pluto_timers,
           typeof __pluto_raf,
           ('__pluto_timers' in globalThis),
         ].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "undefined,undefined,false");
    // …and the timers still work.
    run(&mut tab, "let n = 0; setTimeout(() => { n = 7; }, 0);");
    assert_eq!(read_num(&mut tab, "n"), 7.0);
}

#[test]
fn document_basics() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__r = [
           document.body.tagName,
           document.head.tagName,
           document.title,
           document.getElementById('para').textContent,
           document.getElementById('missing'),
         ].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "BODY|HEAD|Test Page|Hello|");
}

#[test]
fn document_title_is_writable() {
    let mut tab = test_tab();
    run(&mut tab, "document.title = 'New Title';");
    assert_eq!(read_str(&mut tab, "document.title"), "New Title");
}

// ---------------------------------------------------------------- document

#[test]
fn query_selector_uses_the_host_matcher() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__r = [
           document.querySelector('#para').textContent,
           document.querySelectorAll('.item').length,
           document.querySelector('div p').id,
           document.querySelector('ul > li').textContent,
           document.querySelector('li:first-child').textContent,
           document.querySelector('[id=\"btn\"]').tagName,
           document.querySelector('div, p').tagName,
           document.querySelector('.missing'),
           document.querySelectorAll('p.missing').length,
         ].join('|');",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "Hello|3|para|one|one|BUTTON|DIV||0"
    );
}

#[test]
fn element_query_selector_is_scoped() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const list = document.getElementById('list'); \
         globalThis.__r = [
           list.querySelectorAll('li').length,
           list.querySelector('li:last-child').textContent,
           list.querySelector('#para'),
         ].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "3|three|");
}

#[test]
fn create_element_and_append() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const div = document.createElement('div'); \
         div.id = 'new'; div.className = 'fresh'; \
         div.textContent = 'made'; \
         document.getElementById('app').appendChild(div); \
         globalThis.__r = document.getElementById('new').textContent;",
    );
    assert_eq!(read_str(&mut tab, "__r"), "made");
    assert_eq!(read_num(&mut tab, "document.getElementById('app').childNodes.length"), 3.0);
}

#[test]
fn create_text_node_and_parent_links() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const p = document.getElementById('para'); \
         const t = document.createTextNode('!'); \
         p.appendChild(t); \
         globalThis.__r = [p.textContent, t.parentNode === p, t.nodeType, t.nodeName].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "Hello!|true|3|#text");
}

#[test]
fn tree_surgery_insert_remove_replace() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const list = document.getElementById('list'); \
         const items = list.querySelectorAll('li'); \
         const extra = document.createElement('li'); extra.textContent = 'zero'; \
         list.insertBefore(extra, items[0]); \
         list.removeChild(items[1]); \
         const last = document.createElement('li'); last.textContent = 'LAST'; \
         list.replaceChild(last, items[2]); \
         globalThis.__r = [...list.childNodes].map(n => n.textContent).join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "zero,one,LAST");
}

#[test]
fn tree_surgery_errors_have_dom_names() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const app = document.getElementById('app'); \
         const p = document.getElementById('para'); \
         const names = []; \
         try { app.appendChild(app); } catch (e) { names.push(e.name); } \
         try { app.removeChild(document.getElementById('list')); } catch (e) { names.push(e.name); } \
         try { app.appendChild(42); } catch (e) { names.push(e.name); } \
         try { app.insertBefore(p, document.createElement('span')); } catch (e) { names.push(e.name); } \
         globalThis.__r = names.join(',');",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "HierarchyRequestError,NotFoundError,TypeError,NotFoundError"
    );
}

#[test]
fn wrong_this_throws_type_error() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const el = document.createElement('div'); \
         const proto = Object.getPrototypeOf(el); \
         let name = 'none'; \
         try { proto.appendChild.call({}, el); } catch (e) { name = e.name; } \
         globalThis.__r = name;",
    );
    assert_eq!(read_str(&mut tab, "__r"), "TypeError");
}

// ---------------------------------------------------------------- Element

#[test]
fn attributes_are_live() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const p = document.getElementById('para'); \
         p.setAttribute('data-x', '1'); \
         const a = p.getAttribute('data-x'); \
         p.id = 'renamed'; \
         const b = document.getElementById('para'); \
         const c = document.getElementById('renamed') === p; \
         p.removeAttribute('data-x'); \
         globalThis.__r = [a, b, c, p.hasAttribute('data-x'), p.getAttribute('data-x')].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "1||true|false|");
}

#[test]
fn class_list() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const p = document.getElementById('para'); \
         const cl = p.classList; \
         const same = p.classList === cl; \
         cl.add('a', 'b'); cl.remove('a'); \
         const t1 = cl.toggle('b'); const t2 = cl.toggle('b'); \
         const forced = cl.toggle('c', true); \
         globalThis.__r = [same, cl.contains('b'), t1, t2, forced, cl.value, p.className].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "true|true|false|true|true|b c|b c");
}

#[test]
fn class_list_rejects_bad_tokens() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const cl = document.getElementById('para').classList; \
         const names = []; \
         try { cl.add(''); } catch (e) { names.push(e.name); } \
         try { cl.add('a b'); } catch (e) { names.push(e.name); } \
         globalThis.__r = names.join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "SyntaxError,InvalidCharacterError");
}

#[test]
fn style_css_text_and_properties() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const p = document.getElementById('para'); \
         p.style.cssText = 'color: red; margin: 0;'; \
         p.style.setProperty('background-color', 'blue'); \
         const got = p.style.getPropertyValue('color'); \
         const removed = p.style.removeProperty('margin'); \
         globalThis.__r = [got, removed, p.style.cssText, p.getAttribute('style')].join('|');",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "red|0|color: red; background-color: blue;|color: red; background-color: blue;"
    );
}

#[test]
fn inner_html_round_trips_and_scripts_are_inert() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const app = document.getElementById('app'); \
         globalThis.__before = app.innerHTML; \
         app.innerHTML = '<span id=\"in\">inner</span><script>globalThis.__inert = 1;</script>'; \
         globalThis.__after = document.getElementById('in').textContent;",
    );
    assert!(read_str(&mut tab, "__before").contains("para"));
    assert_eq!(read_str(&mut tab, "__after"), "inner");
    assert_eq!(read_str(&mut tab, "typeof __inert"), "undefined");
}

#[test]
fn document_write_appends_markup() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "document.write('<p id=\"w\">written</p>'); \
         document.writeln('<span id=\"w2\">two</span>'); \
         globalThis.__r = [document.getElementById('w').textContent, document.getElementById('w2').textContent].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "written,two");
}

#[test]
fn click_dispatches_and_bubbles() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const order = []; \
         const btn = document.getElementById('btn'); \
         btn.addEventListener('click', (e) => order.push('btn:' + e.target.id + ':' + e.currentTarget.id)); \
         document.getElementById('app').addEventListener('click', () => order.push('app')); \
         document.addEventListener('click', () => order.push('doc')); \
         window.addEventListener('click', () => order.push('win')); \
         btn.click(); \
         globalThis.__r = order.join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "btn:btn:btn,app,doc,win");
}

#[test]
fn stop_propagation_and_prevent_default() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const order = []; \
         const btn = document.getElementById('btn'); \
         btn.addEventListener('click', (e) => { order.push('btn'); e.stopPropagation(); }); \
         document.getElementById('app').addEventListener('click', () => order.push('app')); \
         const ev = new Event('click', { bubbles: true, cancelable: true }); \
         ev.preventDefault(); \
         const notCanceled = btn.dispatchEvent(new Event('click', { bubbles: true, cancelable: true })); \
         const canceled = btn.dispatchEvent(ev); \
         globalThis.__r = [order.join(','), notCanceled, canceled, ev.defaultPrevented].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "btn,btn|true|false|true");
}

#[test]
fn capture_listeners_run_first() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const order = []; \
         const btn = document.getElementById('btn'); \
         document.getElementById('app').addEventListener('click', () => order.push('capture'), true); \
         document.getElementById('app').addEventListener('click', () => order.push('bubble'), false); \
         btn.addEventListener('click', () => order.push('target')); \
         btn.click(); \
         globalThis.__r = order.join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "capture,target,bubble");
}

#[test]
fn remove_and_once_listeners() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "let n = 0; let m = 0; \
         const btn = document.getElementById('btn'); \
         const f = () => { n++; }; \
         btn.addEventListener('click', f); \
         btn.addEventListener('click', f); \
         btn.click(); \
         btn.removeEventListener('click', f); \
         btn.click(); \
         btn.addEventListener('click', () => { m++; }, { once: true }); \
         btn.click(); btn.click(); \
         globalThis.__r = [n, m].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "1,1");
}

#[test]
fn custom_event_carries_detail() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "let got = null; \
         document.getElementById('app').addEventListener('made-up', (e) => { got = e.detail; }); \
         const ev = new CustomEvent('made-up', { bubbles: true, detail: 42 }); \
         document.getElementById('btn').dispatchEvent(ev); \
         globalThis.__r = String(got);",
    );
    assert_eq!(read_str(&mut tab, "__r"), "42");
}

#[test]
fn throwing_listener_does_not_stop_dispatch() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const order = []; \
         const btn = document.getElementById('btn'); \
         btn.addEventListener('click', () => { throw new Error('listener-boom'); }); \
         btn.addEventListener('click', () => order.push('second')); \
         btn.click(); \
         globalThis.__r = order.join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "second");
    assert!(
        tab.script_errors().iter().any(|e| e.contains("listener-boom")),
        "listener error recorded: {:?}",
        tab.script_errors()
    );
}

#[test]
fn external_dispatch_arrives_as_a_task() {
    let host = TestHost::from_html(DOC);
    let btn = host.find_tag(host.dom.root().unwrap(), "button").unwrap();
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "let n = 0; document.getElementById('btn').addEventListener('click', () => { n++; });",
    );
    tab.dispatch_event(btn, "click", true);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "n"), 1.0);
}

// ---------------------------------------------------------------- identity

#[test]
fn wrapper_identity_holds() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__r = [
           document.body === document.body,
           document.getElementById('para') === document.getElementById('para'),
           document.getElementById('para').parentNode === document.getElementById('app'),
           document.querySelector('.item') === document.querySelectorAll('.item')[0],
           document.body.parentNode === document.documentElement,
         ].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "true,true,true,true,true");
}

#[test]
fn stale_keys_read_null_and_setters_throw() {
    let mut tab = test_tab();
    // A detached node is still a live node: removal is not
    // staleness, so its getters keep working and it is no longer
    // findable by id.
    run(
        &mut tab,
        "const p = document.getElementById('para'); \
         globalThis.__p = p; \
         p.parentNode.removeChild(p); \
         globalThis.__r = [p.textContent, p.tagName, document.getElementById('para')].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "Hello|P|");
    // True staleness: swap the host for a fresh document. The old
    // wrapper's key no longer validates against the new Dom, so
    // getters read null (repeatedly — the evicted id stays stale)
    // and setters throw TypeError.
    tab.set_dom_host(Box::new(TestHost::from_html(
        "<html><head><title>Other</title></head><body><p>other</p></body></html>",
    )));
    run(
        &mut tab,
        "globalThis.__r = [__p.textContent, __p.tagName, __p.parentNode].join('|'); \
         let name = 'none'; \
         try { __p.textContent = 'x'; } catch (e) { name = e.name; } \
         globalThis.__setter = name;",
    );
    // (`join` renders `null` as `""`, so three nulls read as `"||"`.)
    assert_eq!(read_str(&mut tab, "__r"), "||");
    assert_eq!(read_str(&mut tab, "__setter"), "TypeError");
}

// ---------------------------------------------------------------- console

#[test]
fn console_goes_to_the_host_log() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "console.log('hello', 42, true); console.warn('careful'); console.error('boom');",
    );
    let lines = tab.console_lines();
    assert_eq!(
        lines,
        vec![
            ("log".to_string(), "hello 42 true".to_string()),
            ("warn".to_string(), "careful".to_string()),
            ("error".to_string(), "boom".to_string()),
        ]
    );
}

// ---------------------------------------------------------------- onerror

#[test]
fn window_onerror_can_suppress_the_report() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const seen = []; \
         window.onerror = (msg) => { seen.push(msg); return true; }; \
         globalThis.__seen = seen;",
    );
    tab.eval_script("throw new Error('suppressed-boom');", "s.js")
        .unwrap();
    tab.run_until_idle().unwrap();
    assert!(
        read_str(&mut tab, "__seen.join(',')").contains("suppressed-boom"),
        "onerror saw the error"
    );
    assert!(
        tab.script_errors().is_empty(),
        "suppressed error must not be recorded: {:?}",
        tab.script_errors()
    );
}

#[test]
fn window_onerror_returning_false_keeps_the_report() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "window.onerror = () => false;",
    );
    tab.eval_script("throw new Error('kept-boom');", "s.js").unwrap();
    tab.run_until_idle().unwrap();
    assert!(
        tab.script_errors().iter().any(|e| e.contains("kept-boom")),
        "error kept: {:?}",
        tab.script_errors()
    );
}

// ---------------------------------------------------------------- timers + dom

#[test]
fn timer_callback_mutates_the_dom() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "let n = 0; \
         setTimeout(() => { \
           n++; \
           document.getElementById('para').textContent = 'later ' + n; \
         }, 0);",
    );
    assert_eq!(
        read_str(&mut tab, "document.getElementById('para').textContent"),
        "later 1"
    );
}

#[test]
fn raf_callback_mutates_the_dom() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "let ts = 0; \
         requestAnimationFrame((t) => { \
           ts = t; \
           document.getElementById('para').textContent = 'framed'; \
         });",
    );
    assert_eq!(
        read_str(&mut tab, "document.getElementById('para').textContent"),
        "framed"
    );
    assert!(read_num(&mut tab, "ts") >= 0.0);
}

#[test]
fn microtask_runs_before_timeout_with_dom() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const order = []; \
         setTimeout(() => order.push('timeout'), 0); \
         queueMicrotask(() => order.push('micro')); \
         Promise.resolve().then(() => order.push('promise')); \
         globalThis.__order = order;",
    );
    // The microtasks drain in the checkpoint after the script task;
    // the timeout fires on a later turn.
    assert_eq!(read_str(&mut tab, "__order.join(',')"), "micro,promise,timeout");
}

// ---------------------------------------------------------------- fetch

#[test]
fn fetch_json_loader_updates_the_dom() {
    let (backend, requests) = FakeBackend::ok(r#"{"items":[1,2,3]}"#);
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('/api/items') \
           .then((r) => { globalThis.__status = r.status; globalThis.__ok = r.ok; return r.json(); }) \
           .then((d) => { document.getElementById('para').textContent = 'items:' + d.items.length; });",
    );
    assert_eq!(
        read_str(&mut tab, "document.getElementById('para').textContent"),
        "items:3"
    );
    assert_eq!(read_num(&mut tab, "__status"), 200.0);
    assert!(read_bool(&mut tab, "__ok"));
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(requests.lock().unwrap()[0].url, "/api/items");
    assert_eq!(requests.lock().unwrap()[0].method, "GET");
}

#[test]
fn fetch_response_fields_and_text() {
    let (backend, _) = FakeBackend::ok("hello body");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('/x') \
           .then((r) => { \
             globalThis.__r = [r.status, r.statusText, r.ok, r.headers.get('x-test'), r.headers.get('X-TEST'), r.headers.get('missing')].join('|'); \
             return r.text(); \
           }) \
           .then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__r"), "200|OK|true|yes|yes|");
    assert_eq!(read_str(&mut tab, "__body"), "hello body");
}

#[test]
fn fetch_rejects_without_a_backend() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "fetch('/x').catch((e) => { globalThis.__caught = e.name + ':' + e.message; });",
    );
    assert!(
        read_str(&mut tab, "__caught").contains("TypeError"),
        "caught: {}",
        read_str(&mut tab, "__caught")
    );
}

#[test]
fn fetch_backend_error_rejects() {
    let mut tab = test_tab();
    tab.set_fetch_backend(FakeBackend::failing(FetchError::Network(
        "connection refused".to_string(),
    )));
    run(
        &mut tab,
        "fetch('/x').catch((e) => { globalThis.__caught = e.message; });",
    );
    assert!(
        read_str(&mut tab, "__caught").contains("connection refused"),
        "caught: {}",
        read_str(&mut tab, "__caught")
    );
}

#[test]
fn response_constructor() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const r = new Response('hi', { status: 201, statusText: 'Created' }); \
         globalThis.__r = [r.status, r.statusText, r.ok].join('|'); \
         r.text().then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__r"), "201|Created|true");
    assert_eq!(read_str(&mut tab, "__body"), "hi");
}

#[test]
fn headers_constructor_and_methods() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const h = new Headers({ 'X-A': '1', 'x-b': '2' }); \
         h.set('X-C', '3'); \
         globalThis.__r = [h.get('x-a'), h.get('X-B'), h.has('x-c'), h.get('missing')].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "1|2|true|");
}

#[test]
fn fetch_json_parse_error_rejects() {
    let (backend, _) = FakeBackend::ok("not json {");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('/x').then((r) => r.json()).catch((e) => { globalThis.__caught = e.name; });",
    );
    assert_eq!(read_str(&mut tab, "__caught"), "SyntaxError");
}

// Stage 9 Rec 1: the realm-side gate compares parsed `Origin`s.
// The backend never sees a rejected fetch.

#[test]
fn fetch_absolute_same_origin_allowed() {
    let (backend, requests) = FakeBackend::ok_with_origin(
        "ok",
        Some(Origin::from_url("https://example.com/page")),
    );
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('https://example.com/api') \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; }) \
           .catch((e) => { globalThis.__caught = e.message; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[test]
fn fetch_absolute_same_origin_default_port_allowed() {
    // `https://example.com:443` is the same origin as
    // `https://example.com` (default-port normalization) — the
    // gate must not falsely reject it.
    let (backend, requests) = FakeBackend::ok_with_origin(
        "ok",
        Some(Origin::from_url("https://example.com")),
    );
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('https://example.com:443/api') \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[test]
fn fetch_absolute_cross_origin_passes_through_to_backend() {
    // Stage 9 Rec 2 completion: the realm-side gate is a
    // pass-through — cross-origin enforcement lives in the
    // browser-side backend + pluto-net CORS, not in the realm.
    // A default-mode (`cors`) cross-origin fetch therefore
    // reaches the backend; the backend decides (here the fake
    // answers 200, so the promise resolves).
    let (backend, requests) = FakeBackend::ok_with_origin(
        "ok",
        Some(Origin::from_url("https://example.com/page")),
    );
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('https://evil.example/x') \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    let seen = requests.lock().unwrap();
    assert_eq!(seen.len(), 1, "the backend sees the cross-origin fetch");
    assert_eq!(seen[0].url, "https://evil.example/x");
    assert_eq!(seen[0].mode, RequestMode::Cors, "default mode is cors");
}

#[test]
fn fetch_absolute_cross_origin_port_passes_through_to_backend() {
    // A different port is a different origin, even on the same
    // host — but at the realm the fetch still passes through;
    // the origin distinction is enforced browser-side.
    let (backend, requests) = FakeBackend::ok_with_origin(
        "ok",
        Some(Origin::from_url("https://example.com/")),
    );
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('https://example.com:8443/x') \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "the backend sees the cross-origin-port fetch"
    );
}

#[test]
fn fetch_relative_allowed_with_origin_tracked() {
    // Relative URLs resolve against the document base browser-side,
    // so the realm-side gate passes them through.
    let (backend, requests) = FakeBackend::ok_with_origin(
        "ok",
        Some(Origin::from_url("https://example.com/page")),
    );
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('/api/items') \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    assert_eq!(requests.lock().unwrap().len(), 1);
}

// ---------------------------------------------------------------- misc

#[test]
fn document_open_and_close() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const d = document.open(); \
         globalThis.__r = (d === document); \
         document.close();",
    );
    assert!(read_bool(&mut tab, "__r"));
}

#[test]
fn performance_now_advances() {
    let mut tab = test_tab();
    run(&mut tab, "globalThis.__t0 = performance.now();");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let t1 = read_num(&mut tab, "performance.now()");
    assert!(t1 > read_num(&mut tab, "__t0"));
}


#[test]
fn geometry_reads_force_a_flush_and_return_the_box() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const el = document.getElementById('para'); \
         const r = el.getBoundingClientRect(); \
         globalThis.__rect = r.x + ',' + r.y + ',' + r.width + ',' + r.height + ',' + r.right + ',' + r.bottom; \
         globalThis.__ow = el.offsetWidth; \
         globalThis.__oh = el.offsetHeight;",
    );
    // TestHost geometry: 0,0,100x50 for every node.
    assert_eq!(read_str(&mut tab, "__rect"), "0,0,100,50,100,50");
    assert_eq!(read_num(&mut tab, "__ow"), 100.0);
    assert_eq!(read_num(&mut tab, "__oh"), 50.0);
}

// ---------------------------------------------------------------- MutationObserver (Stage 8 Rec 7)

#[test]
fn mutation_observer_records_in_order_one_callback_per_turn() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__calls = 0; globalThis.__recs = []; \
         const app = document.getElementById('app'); \
         const obs = new MutationObserver((recs) => { \
           globalThis.__calls++; \
           globalThis.__recs = recs; \
         }); \
         obs.observe(app, { childList: true, attributes: true, characterData: true, \
           subtree: true, attributeOldValue: true, characterDataOldValue: true }); \
         const p = document.createElement('p'); \
         app.appendChild(p); \
         app.setAttribute('class', 'changed'); \
         const para = document.getElementById('para'); \
         para.childNodes[0].data = 'Bye'; \
         globalThis.__sync_calls = globalThis.__calls;",
    );
    // Nothing delivered synchronously, inside the script.
    assert_eq!(read_num(&mut tab, "__sync_calls"), 0.0);
    // One compound callback for the whole turn, with all three
    // record kinds in mutation order.
    assert_eq!(read_num(&mut tab, "__calls"), 1.0);
    assert_eq!(read_num(&mut tab, "__recs.length"), 3.0);
    assert_eq!(
        read_str(&mut tab, "__recs.map(r => r.type).join(',')"),
        "childList,attributes,characterData"
    );
    assert!(read_bool(&mut tab, "__recs[0].target === app"));
    assert_eq!(read_num(&mut tab, "__recs[0].addedNodes.length"), 1.0);
    assert!(read_bool(&mut tab, "__recs[0].addedNodes[0] === p"));
    assert_eq!(read_str(&mut tab, "__recs[1].attributeName"), "class");
    assert_eq!(read_str(&mut tab, "__recs[1].oldValue"), "root");
    assert!(read_bool(&mut tab, "__recs[2].target === para.childNodes[0]"));
    assert_eq!(read_str(&mut tab, "__recs[2].oldValue"), "Hello");
    assert_eq!(read_str(&mut tab, "para.textContent"), "Bye");
}

#[test]
fn mutation_observer_take_records_drains_and_disconnect_stops_delivery() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__calls = 0; \
         const app = document.getElementById('app'); \
         const obs = new MutationObserver(() => { globalThis.__calls++; }); \
         obs.observe(app, { childList: true }); \
         app.appendChild(document.createElement('span')); \
         globalThis.__taken = obs.takeRecords(); \
         globalThis.__taken_n = __taken.length;",
    );
    // takeRecords drained the queue synchronously: the callback
    // never fires for those records.
    assert_eq!(read_num(&mut tab, "__taken_n"), 1.0);
    assert_eq!(read_num(&mut tab, "__calls"), 0.0);
    run(
        &mut tab,
        "obs.disconnect(); \
         app.appendChild(document.createElement('em')); \
         globalThis.__after_disconnect_taken = obs.takeRecords().length;",
    );
    assert_eq!(read_num(&mut tab, "__calls"), 0.0);
    assert_eq!(read_num(&mut tab, "__after_disconnect_taken"), 0.0);
}

#[test]
fn mutation_observer_attribute_filter_and_options_validation() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const app = document.getElementById('app'); \
         globalThis.__seen = []; \
         const obs = new MutationObserver((recs) => { globalThis.__seen = recs; }); \
         obs.observe(app, { attributes: true, attributeFilter: ['id'] }); \
         app.setAttribute('class', 'nope'); \
         app.setAttribute('id', 'yes'); \
         let err = ''; \
         try { obs.observe(app, {}); } catch (e) { err = e.name; } \
         globalThis.__err = err;",
    );
    // Only the `id` change was recorded.
    assert_eq!(read_num(&mut tab, "__seen.length"), 1.0);
    assert_eq!(read_str(&mut tab, "__seen[0].attributeName"), "id");
    // Observing with no observation type throws TypeError.
    assert_eq!(read_str(&mut tab, "__err"), "TypeError");
}

// ---------------------------------------------------------------- ResizeObserver (Stage 8 Rec 8)

#[test]
fn resize_observer_fires_initial_entry_then_stays_quiet() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__entries = []; \
         const ro = new ResizeObserver((entries) => { \
           for (const e of entries) globalThis.__entries.push(e); \
         }); \
         ro.observe(document.getElementById('app')); \
         globalThis.__sync = globalThis.__entries.length;",
    );
    // No delivery until the rendering step runs.
    assert_eq!(read_num(&mut tab, "__sync"), 0.0);
    let outcome = tab.run_resize_observers();
    assert!(outcome.delivered);
    assert!(!outcome.loop_error);
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
    assert!(read_bool(
        &mut tab,
        "__entries[0].target === document.getElementById('app')"
    ));
    // TestHost geometry: 100×50 for every node.
    assert_eq!(
        read_num(&mut tab, "__entries[0].borderBoxSize[0].inlineSize"),
        100.0
    );
    assert_eq!(
        read_num(&mut tab, "__entries[0].borderBoxSize[0].blockSize"),
        50.0
    );
    assert_eq!(
        read_num(&mut tab, "__entries[0].contentBoxSize[0].inlineSize"),
        100.0
    );
    assert_eq!(read_num(&mut tab, "__entries[0].contentRect.width"), 100.0);
    assert_eq!(read_num(&mut tab, "__entries[0].contentRect.height"), 50.0);
    // A second step with unchanged sizes delivers nothing.
    let outcome2 = tab.run_resize_observers();
    assert!(!outcome2.delivered);
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
}

#[test]
fn resize_observer_unobserve_disconnect_and_bad_options() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__n = 0; \
         const ro = new ResizeObserver(() => { globalThis.__n++; }); \
         const app = document.getElementById('app'); \
         const btn = document.getElementById('btn'); \
         ro.observe(app); \
         ro.observe(btn); \
         ro.unobserve(btn); \
         let box_err = ''; \
         try { ro.observe(app, { box: 'bogus' }); } catch (e) { box_err = e.name; } \
         let el_err = ''; \
         try { ro.observe(document.createTextNode('x')); } catch (e) { el_err = e.name; } \
         globalThis.__box_err = box_err; \
         globalThis.__el_err = el_err;",
    );
    assert_eq!(read_str(&mut tab, "__box_err"), "TypeError");
    assert_eq!(read_str(&mut tab, "__el_err"), "TypeError");
    let outcome = tab.run_resize_observers();
    assert!(outcome.delivered);
    assert_eq!(read_num(&mut tab, "__n"), 1.0, "only app observed");
    run(&mut tab, "ro.disconnect();");
    let outcome2 = tab.run_resize_observers();
    assert!(!outcome2.delivered);
    assert_eq!(read_num(&mut tab, "__n"), 1.0);
}

#[test]
fn resize_observer_loop_error_when_sizes_keep_changing() {
    // A host whose geometry grows on every read: the callback can
    // never settle, so the step must stop with the resize-loop
    // error instead of looping forever.
    let mut host = TestHost::from_html(DOC);
    host.geometry_growth.set(10.0);
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "globalThis.__n = 0; \
         const ro = new ResizeObserver(() => { globalThis.__n++; }); \
         ro.observe(document.getElementById('app'));",
    );
    let outcome = tab.run_resize_observers();
    assert!(outcome.loop_error, "expected the resize-loop error");
    assert_eq!(outcome.rounds, 5);
    assert_eq!(read_num(&mut tab, "__n"), 5.0, "one callback per round");
    assert!(
        tab.script_errors()
            .iter()
            .any(|e| e.contains("ResizeObserver loop completed")),
        "loop error recorded: {:?}",
        tab.script_errors()
    );
}


// ---------------------------------------------------------------- IntersectionObserver (Stage 8 Rec 10)

/// Geometry for one observed target, as the worker step would
/// measure it: a 100×50 target fully inside a 1024×768 viewport root.
fn full_geometry() -> ComputedIntersection {
    ComputedIntersection {
        target_rect: [0.0, 0.0, 100.0, 50.0],
        root_bounds: [0.0, 0.0, 1024.0, 768.0],
        intersection_rect: [0.0, 0.0, 100.0, 50.0],
        ratio: 1.0,
        is_intersecting: true,
    }
}

#[test]
fn intersection_observer_fires_initial_entry_after_paint_then_stays_quiet() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__entries = []; \
         const io = new IntersectionObserver((entries) => { \
           for (const e of entries) globalThis.__entries.push(e); \
         }); \
         io.observe(document.getElementById('app')); \
         globalThis.__sync = globalThis.__entries.length;",
    );
    // No delivery until the rendering step runs.
    assert_eq!(read_num(&mut tab, "__sync"), 0.0);
    let targets = tab.intersection_observation_targets();
    assert_eq!(targets.len(), 1);
    let (id, target, root, _) = targets[0];
    assert!(root.is_none(), "no root option means the viewport");
    let mut geometry = HashMap::new();
    geometry.insert((id, target), full_geometry());
    tab.run_intersection_observers_with_geometry(&geometry, 42.0);
    // Still nothing delivered: the callback is a queued task, after
    // paint — it must not run inline in the step.
    assert_eq!(read_num(&mut tab, "__entries.length"), 0.0);
    // Drive the loop: the queued task runs.
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
    assert!(read_bool(
        &mut tab,
        "__entries[0].target === document.getElementById('app')"
    ));
    assert_eq!(read_num(&mut tab, "__entries[0].intersectionRatio"), 1.0);
    assert!(read_bool(&mut tab, "__entries[0].isIntersecting"));
    // `time` is the computation timestamp, not the delivery time.
    assert_eq!(read_num(&mut tab, "__entries[0].time"), 42.0);
    assert_eq!(
        read_num(&mut tab, "__entries[0].boundingClientRect.width"),
        100.0
    );
    assert_eq!(
        read_num(&mut tab, "__entries[0].boundingClientRect.height"),
        50.0
    );
    assert_eq!(read_num(&mut tab, "__entries[0].rootBounds.width"), 1024.0);
    assert_eq!(
        read_num(&mut tab, "__entries[0].intersectionRect.width"),
        100.0
    );
    // A second step with unchanged geometry delivers nothing.
    tab.run_intersection_observers_with_geometry(&geometry, 58.0);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
}

#[test]
fn intersection_observer_options_validation_take_records_and_threshold() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "globalThis.__entries = []; \
         const io = new IntersectionObserver((entries) => { \
           for (const e of entries) globalThis.__entries.push(e); \
         }); \
         const app = document.getElementById('app'); \
         let range_err = ''; \
         try { io.observe(app, { threshold: 1.5 }); } catch (e) { range_err = e.name; } \
         let margin_err = ''; \
         try { io.observe(app, { rootMargin: '10em' }); } catch (e) { margin_err = e.name; } \
         let el_err = ''; \
         try { io.observe(document.createTextNode('x')); } catch (e) { el_err = e.name; } \
         globalThis.__range_err = range_err; \
         globalThis.__margin_err = margin_err; \
         globalThis.__el_err = el_err; \
         io.observe(app, { threshold: [0.5] });",
    );
    assert_eq!(read_str(&mut tab, "__range_err"), "RangeError");
    assert_eq!(read_str(&mut tab, "__margin_err"), "TypeError");
    assert_eq!(read_str(&mut tab, "__el_err"), "TypeError");
    let targets = tab.intersection_observation_targets();
    assert_eq!(targets.len(), 1, "only the valid observe() registered");
    let (id, target, _, _) = targets[0];
    // Out of view: the initial entry fires (ratio 0, not intersecting).
    let mut out_of_view = full_geometry();
    out_of_view.intersection_rect = [0.0, 0.0, 0.0, 0.0];
    out_of_view.ratio = 0.0;
    out_of_view.is_intersecting = false;
    let mut geometry = HashMap::new();
    geometry.insert((id, target), out_of_view);
    tab.run_intersection_observers_with_geometry(&geometry, 0.0);
    // takeRecords() returns the pending entry before the task runs,
    // and the task then has nothing left to deliver. Read through
    // `eval_value` (which drains microtasks but never runs tasks):
    // a `run()` here would drive the loop and the delivery task —
    // DomManipulation outranks Script — would deliver first.
    assert_eq!(
        read_num(
            &mut tab,
            "(globalThis.__taken = io.takeRecords()).length"
        ),
        1.0
    );
    assert_eq!(read_num(&mut tab, "__taken[0].intersectionRatio"), 0.0);
    assert!(!read_bool(&mut tab, "__taken[0].isIntersecting"));
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 0.0, "taken, not delivered");
    // 25% visible: isIntersecting flips, so an entry fires even
    // though threshold 0.5 was not crossed.
    let mut quarter = full_geometry();
    quarter.intersection_rect = [0.0, 0.0, 100.0, 12.5];
    quarter.ratio = 0.25;
    geometry.insert((id, target), quarter);
    tab.run_intersection_observers_with_geometry(&geometry, 16.0);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
    assert_eq!(read_num(&mut tab, "__entries[0].intersectionRatio"), 0.25);
    // 40%: same band, still intersecting — quiet.
    let mut forty = full_geometry();
    forty.intersection_rect = [0.0, 0.0, 100.0, 20.0];
    forty.ratio = 0.4;
    geometry.insert((id, target), forty);
    tab.run_intersection_observers_with_geometry(&geometry, 32.0);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 1.0);
    // 60%: crosses 0.5 upward — fires.
    let mut sixty = full_geometry();
    sixty.intersection_rect = [0.0, 0.0, 100.0, 30.0];
    sixty.ratio = 0.6;
    geometry.insert((id, target), sixty);
    tab.run_intersection_observers_with_geometry(&geometry, 48.0);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__entries.length"), 2.0);
    assert_eq!(read_num(&mut tab, "__entries[1].intersectionRatio"), 0.6);
}

// ---------------------------------------------------------------- Stage 12 Rec 7: binding batch

/// A backend that never answers (for abort tests): records the
/// request and the ids `cancel_fetch` was called for, and sends
/// no completion.
struct NeverBackend {
    requests: Arc<Mutex<Vec<FetchRequest>>>,
    cancelled: Arc<Mutex<Vec<u64>>>,
}

impl FetchBackend for NeverBackend {
    fn start_fetch(&self, request: FetchRequest, _done: mpsc::Sender<FetchCompletion>) {
        self.requests.lock().unwrap().push(request);
    }

    fn cancel_fetch(&self, id: u64) {
        self.cancelled.lock().unwrap().push(id);
    }
}

/// A backend that answers late (for the late-completion test):
/// spawns a thread that sends the canned response after 50 ms.
struct LateBackend {
    response: FetchResponse,
}

impl FetchBackend for LateBackend {
    fn start_fetch(&self, request: FetchRequest, done: mpsc::Sender<FetchCompletion>) {
        let id = request.id;
        let response = self.response.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let _ = done.send(FetchCompletion {
                id,
                result: Ok(response),
            });
        });
    }
}

#[test]
fn location_parts_and_assign_records_navigation() {
    let log = Arc::new(Mutex::new(TestHostLog::default()));
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(TestHost::from_html_with_log(DOC, Arc::clone(&log))));
    run(
        &mut tab,
        "globalThis.__r = [location.href, location.protocol, location.host, location.hostname, location.port, location.pathname, location.search, location.hash, location.origin, String(location)].join('|'); \
         location.assign('/page?q=1#frag'); \
         location.replace('/other'); \
         location.reload();",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "https://example.com/|https:|example.com|example.com||/| | |https://example.com|https://example.com/"
            .replace("| | |", "|||")
    );
    assert_eq!(
        log.lock().unwrap().navigation_requests,
        vec![
            ("https://example.com/page?q=1#frag".to_string(), false),
            ("https://example.com/other".to_string(), true),
            ("https://example.com/".to_string(), false),
        ]
    );
}

#[test]
fn url_and_search_params() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const u = new URL('https://example.com:8080/a/b?x=1&x=2#frag'); \
         globalThis.__r = [u.href, u.origin, u.protocol, u.host, u.hostname, u.port, u.pathname, u.search, u.hash].join('|'); \
         globalThis.__p = [u.searchParams.get('x'), u.searchParams.getAll('x').join(','), u.searchParams.has('y')].join('|'); \
         u.searchParams.set('x', '9'); u.searchParams.append('y', 'z'); u.searchParams.delete('x'); \
         globalThis.__p2 = [u.search, u.searchParams.toString()].join('|'); \
         const u2 = new URL('/rel', 'https://example.com/base/'); \
         globalThis.__rel = u2.href; \
         const sp = new URLSearchParams('a=1&b=2'); \
         sp.set('b', '3'); \
         globalThis.__sp = sp.toString(); \
         let threw = ''; try { new URL('::not a url::'); } catch (e) { threw = e.name; } \
         globalThis.__threw = threw;",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "https://example.com:8080/a/b?x=1&x=2#frag|https://example.com:8080|https:|example.com:8080|example.com|8080|/a/b|?x=1&x=2|#frag"
    );
    assert_eq!(read_str(&mut tab, "__p"), "1|1,2|false");
    assert_eq!(read_str(&mut tab, "__p2"), "?y=z|y=z");
    assert_eq!(read_str(&mut tab, "__rel"), "https://example.com/rel");
    assert_eq!(read_str(&mut tab, "__sp"), "a=1&b=3");
    assert_eq!(read_str(&mut tab, "__threw"), "TypeError");
}

#[test]
fn dataset_live_read_write() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const app = document.getElementById('app'); \
         app.dataset.fooBar = 'baz'; \
         globalThis.__r = [app.getAttribute('data-foo-bar'), app.dataset.fooBar].join('|'); \
         app.setAttribute('data-user-name', 'Jo'); \
         globalThis.__r2 = app.dataset.userName; \
         delete app.dataset.fooBar; \
         globalThis.__keys = Object.keys(app.dataset).join(','); \
         globalThis.__r3 = app.hasAttribute('data-foo-bar');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "baz|baz");
    assert_eq!(read_str(&mut tab, "__r2"), "Jo");
    assert_eq!(read_str(&mut tab, "__keys"), "userName");
    assert!(!read_bool(&mut tab, "__r3"));
}

#[test]
fn clone_node_deep_and_shallow() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const app = document.getElementById('app'); \
         const deep = app.cloneNode(true); \
         const shallow = app.cloneNode(false); \
         globalThis.__r = [deep.tagName, deep.childNodes.length, deep.className, deep.textContent, deep.parentNode].join('|'); \
         globalThis.__s = [shallow.tagName, shallow.childNodes.length, shallow.className].join('|');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "DIV|2|root|HelloClick|");
    assert_eq!(read_str(&mut tab, "__s"), "DIV|0|root");
}

#[test]
fn closest_and_matches() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const para = document.getElementById('para'); \
         globalThis.__r = [para.matches('p'), para.matches('.root'), para.matches('#para')].join(','); \
         globalThis.__c = [para.closest('div') === document.getElementById('app'), para.closest('#app') === document.getElementById('app'), para.closest('ul')].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "true,false,true");
    assert_eq!(read_str(&mut tab, "__c"), "true,true,");
}

#[test]
fn get_elements_by_tag_class_name() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const input = document.createElement('input'); \
         input.setAttribute('name', 'q'); \
         document.body.appendChild(input); \
         globalThis.__r = [document.getElementsByTagName('li').length, document.getElementsByClassName('item').length, document.getElementsByName('q').length, document.getElementById('list').getElementsByTagName('li').length, document.getElementById('list').getElementsByClassName('item').length].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "3,3,1,3,3");
}

#[test]
fn image_constructor_and_document_images() {
    let log = Arc::new(Mutex::new(TestHostLog::default()));
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(TestHost::from_html_with_log(DOC, Arc::clone(&log))));
    run(
        &mut tab,
        "const img = new Image(640, 480); \
         img.src = 'https://example.com/pic.png'; \
         globalThis.__r = [img.tagName, img.width, img.height, img.complete, img.src, img instanceof Image].join('|'); \
         const docImg = document.createElement('img'); \
         docImg.setAttribute('src', '/a.png'); \
         document.body.appendChild(docImg); \
         globalThis.__n = document.images.length; \
         globalThis.__t = document.images[0].tagName;",
    );
    assert_eq!(
        read_str(&mut tab, "__r"),
        "IMG|640|480|false|https://example.com/pic.png|true"
    );
    assert_eq!(read_num(&mut tab, "__n"), 1.0);
    assert_eq!(read_str(&mut tab, "__t"), "IMG");
    assert_eq!(
        log.lock().unwrap().image_requests,
        vec![(
            "https://example.com/pic.png".to_string(),
            "https://example.com/".to_string()
        )]
    );
}

#[test]
fn image_constructor_no_args() {
    // Stage 12 wiring follow-up: `new Image()` with no arguments
    // used to throw ("converting from js 'undefined' into type
    // 'f64'") — the JS wrapper always passes both slots, and
    // rquickjs's `Opt` only covers *missing* arguments, not
    // explicit `undefined`. The factory now takes `Value`s.
    let mut tab = test_tab();
    run(
        &mut tab,
        "const img = new Image(); \
         globalThis.__r = [img.tagName, img.width, img.height, img.complete, img instanceof Image].join('|'); \
         const one = new Image(320); \
         globalThis.__s = [one.width, one.height].join(',');",
    );
    assert_eq!(read_str(&mut tab, "__r"), "IMG|0|0|false|true");
    assert_eq!(read_str(&mut tab, "__s"), "320,0");
}

#[test]
fn image_load_event_fires_on_shell_report() {
    let mut host = TestHost::from_html(DOC);
    let img_key = host.dom.create_element("img");
    host.dom.set_attribute(img_key, "id", "pic");
    let body = host.body().unwrap();
    host.dom.append_child(body, img_key);
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "const pic = document.getElementById('pic'); \
         let loaded = 0, onloaded = 0; \
         pic.addEventListener('load', () => { loaded = 1; }); \
         pic.onload = () => { onloaded = 1; }; \
         globalThis.__loaded = () => loaded; \
         globalThis.__onloaded = () => onloaded; \
         globalThis.__pic = pic;",
    );
    tab.notify_image_loaded(img_key, 640, 480);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__loaded()"), 1.0);
    assert_eq!(read_num(&mut tab, "__onloaded()"), 1.0);
    assert!(read_bool(&mut tab, "__pic.complete"));
    assert_eq!(read_num(&mut tab, "__pic.naturalWidth"), 640.0);
    assert_eq!(read_num(&mut tab, "__pic.naturalHeight"), 480.0);
}

#[test]
fn typed_keyboard_event_dispatch() {
    let host = TestHost::from_html(DOC);
    let btn = host.get_element_by_id("btn").unwrap();
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "const seen = []; \
         document.getElementById('btn').addEventListener('keydown', (e) => { seen.push([e.key, e.code, e instanceof KeyboardEvent, e instanceof Event]); }); \
         globalThis.__seen = seen;",
    );
    tab.dispatch_typed_event(
        btn,
        "keydown",
        true,
        TypedEventKind::Keyboard {
            key: "Enter".to_string(),
            code: "Enter".to_string(),
        },
    );
    tab.run_until_idle().unwrap();
    assert_eq!(read_str(&mut tab, "__seen[0].join(',')"), "Enter,Enter,true,true");
}

#[test]
fn typed_mouse_event_dispatch() {
    let host = TestHost::from_html(DOC);
    let btn = host.get_element_by_id("btn").unwrap();
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "const seen = []; \
         document.getElementById('btn').addEventListener('click', (e) => { seen.push([e.clientX, e.clientY, e.button, e instanceof MouseEvent]); }); \
         globalThis.__seen = seen;",
    );
    tab.dispatch_typed_event(
        btn,
        "click",
        true,
        TypedEventKind::Mouse {
            client_x: 10.0,
            client_y: 20.0,
            button: 0,
        },
    );
    tab.run_until_idle().unwrap();
    assert_eq!(read_str(&mut tab, "__seen[0].join(',')"), "10,20,0,true");
}

#[test]
fn typed_input_event_dispatch() {
    let host = TestHost::from_html(DOC);
    let para = host.get_element_by_id("para").unwrap();
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    run(
        &mut tab,
        "const seen = []; \
         document.getElementById('para').addEventListener('input', (e) => { seen.push([e.data, e instanceof InputEvent, e.target === document.getElementById('para')]); }); \
         globalThis.__seen = seen;",
    );
    tab.dispatch_typed_event(
        para,
        "input",
        true,
        TypedEventKind::Input {
            data: Some("x".to_string()),
        },
    );
    tab.run_until_idle().unwrap();
    assert_eq!(read_str(&mut tab, "__seen[0].join(',')"), "x,true,true");
}

// ---------------------------------------------------------------- Stage 12 Rec 8: fetch surface

#[test]
fn request_round_trip_through_fetch() {
    let (backend, requests) = FakeBackend::ok("ok");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "const req = new Request('/api', { method: 'POST', headers: { 'x-a': '1' }, body: 'hi' }); \
         globalThis.__r = [req.url, req.method, req.headers.get('x-a')].join('|'); \
         fetch(req).then((r) => r.text()).then((t) => { globalThis.__body = t; });",
    );
    assert_eq!(read_str(&mut tab, "__r"), "/api|POST|1");
    assert_eq!(read_str(&mut tab, "__body"), "ok");
    let recorded = requests.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].url, "/api");
    assert_eq!(recorded[0].method, "POST");
    assert!(recorded[0].headers.iter().any(|(n, v)| n == "x-a" && v == "1"));
    assert_eq!(recorded[0].body, Some(b"hi".to_vec()));
}

#[test]
fn abort_before_completion_rejects_with_abort_error() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let cancelled = Arc::new(Mutex::new(Vec::new()));
    let backend = Arc::new(NeverBackend {
        requests: Arc::clone(&requests),
        cancelled: Arc::clone(&cancelled),
    });
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "const c = new AbortController(); \
         let onabort = 0; c.signal.onabort = () => { onabort = 1; }; \
         globalThis.__onabort = () => onabort; \
         fetch('/slow', { signal: c.signal }) \
           .then(() => { globalThis.__resolved = 1; }) \
           .catch((e) => { globalThis.__name = e.name; }); \
         c.abort();",
    );
    assert_eq!(read_str(&mut tab, "__name"), "AbortError");
    assert_eq!(read_num(&mut tab, "__onabort()"), 1.0);
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(cancelled.lock().unwrap().len(), 1);
    assert_eq!(
        cancelled.lock().unwrap()[0],
        requests.lock().unwrap()[0].id
    );
}

#[test]
fn abort_after_completion_is_a_noop() {
    let (backend, _) = FakeBackend::ok("done");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "const c = new AbortController(); \
         fetch('/x', { signal: c.signal }) \
           .then((r) => r.text()) \
           .then((t) => { globalThis.__body = t; c.abort(); globalThis.__aborted = c.signal.aborted; });",
    );
    assert_eq!(read_str(&mut tab, "__body"), "done");
    assert!(read_bool(&mut tab, "__aborted"));
}

#[test]
fn abort_of_already_aborted_signal_rejects_immediately() {
    let (backend, requests) = FakeBackend::ok("ok");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "const c = new AbortController(); \
         c.abort(); \
         fetch('/x', { signal: c.signal }).catch((e) => { globalThis.__name = e.name; });",
    );
    assert_eq!(read_str(&mut tab, "__name"), "AbortError");
    assert_eq!(requests.lock().unwrap().len(), 0);
}

#[test]
fn late_fetch_completion_after_abort_is_ignored() {
    let backend = Arc::new(LateBackend {
        response: FetchResponse {
            status: 200,
            status_text: "OK".to_string(),
            headers: Vec::new(),
            body: b"late".to_vec(),
            response_type: ResponseType::Basic,
        },
    });
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "const c = new AbortController(); \
         globalThis.__resolved = 0; \
         fetch('/slow', { signal: c.signal }) \
           .then(() => { globalThis.__resolved = 1; }) \
           .catch((e) => { globalThis.__name = e.name; }); \
         c.abort(); \
         setTimeout(() => { globalThis.__done = 1; }, 200);",
    );
    assert_eq!(read_str(&mut tab, "__name"), "AbortError");
    assert_eq!(read_num(&mut tab, "__resolved"), 0.0);
    assert_eq!(read_num(&mut tab, "__done"), 1.0);
}

#[test]
fn headers_iteration_order_and_delete() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "const h = new Headers([['b', '2'], ['a', '1'], ['b', '3']]); \
         globalThis.__get = h.get('b'); \
         h.delete('a'); \
         globalThis.__entries = h.entries().map(([k, v]) => k + '=' + v).join(';'); \
         globalThis.__keys = h.keys().join(','); \
         const seen = []; h.forEach((v, k) => seen.push(k + ':' + v)); \
         globalThis.__foreach = seen.join(';'); \
         const iterated = []; for (const [k, v] of h) iterated.push(k + '=' + v); \
         globalThis.__iter = iterated.join(';');",
    );
    assert_eq!(read_str(&mut tab, "__get"), "2, 3");
    assert_eq!(read_str(&mut tab, "__entries"), "b=2, 3");
    assert_eq!(read_str(&mut tab, "__keys"), "b");
    assert_eq!(read_str(&mut tab, "__foreach"), "b:2, 3");
    assert_eq!(read_str(&mut tab, "__iter"), "b=2, 3");
}

#[test]
fn response_clone_blob_and_array_buffer() {
    let (backend, _) = FakeBackend::ok("hello");
    let mut tab = test_tab();
    tab.set_fetch_backend(backend);
    run(
        &mut tab,
        "fetch('/x').then((r) => { \
           globalThis.__r = r; \
           const c = r.clone(); \
           globalThis.__c = c; \
           globalThis.__meta = [c.url, c.redirected, c.status, c !== r].join('|'); \
           return Promise.all([r.text(), c.text(), r.blob(), r.arrayBuffer()]); \
         }).then(([t1, t2, blob, ab]) => { \
           globalThis.__texts = [t1, t2].join('|'); \
           globalThis.__blob = blob.size + ':' + blob.type; \
           globalThis.__ab = ab.byteLength + ':' + new Uint8Array(ab)[0]; \
         });",
    );
    assert_eq!(read_str(&mut tab, "__meta"), "/x|false|200|true");
    assert_eq!(read_str(&mut tab, "__texts"), "hello|hello");
    assert_eq!(read_str(&mut tab, "__blob"), "5:application/json");
    assert_eq!(read_str(&mut tab, "__ab"), "5:104");
}

// ---------------------------------------------------------------- Stage 12 Rec 10: console

#[test]
fn console_extended_methods() {
    let mut tab = test_tab();
    run(
        &mut tab,
        "console.trace('tmsg'); \
         console.assert(false, 'must hold'); \
         console.assert(true, 'never shown'); \
         console.count('a'); console.count('a'); console.countReset('a'); console.count('a'); \
         console.time('t'); console.timeLog('t'); console.timeEnd('t'); console.timeEnd('nope'); \
         console.dir({ a: 1 }); \
         console.table([{ a: 1, b: 2 }, { a: 3, b: 4 }]); \
         console.group('g'); console.log('inner'); console.groupEnd(); console.log('outer');",
    );
    let lines = tab.console_lines();
    let messages: Vec<&str> = lines.iter().map(|(_, m)| m.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.starts_with("Trace: tmsg")),
        "trace line missing: {messages:?}"
    );
    assert!(
        messages.contains(&"Assertion failed: must hold"),
        "assert line missing: {messages:?}"
    );
    assert!(
        !messages.iter().any(|m| m.contains("never shown")),
        "passing assert must not log: {messages:?}"
    );
    assert!(messages.contains(&"a: 1"));
    assert!(messages.contains(&"a: 2"));
    assert!(messages.contains(&"a: 0"));
    assert!(
        messages.iter().filter(|m| **m == "a: 1").count() == 2,
        "count sequence wrong: {messages:?}"
    );
    let timed: Vec<&&str> = messages.iter().filter(|m| m.starts_with("t: ")).collect();
    assert_eq!(timed.len(), 2, "timeLog + timeEnd: {messages:?}");
    assert!(timed.iter().all(|m| m.ends_with(" ms")));
    assert!(messages.contains(&"Timer 'nope' does not exist"));
    assert!(
        messages.iter().any(|m| m.contains("\"a\": 1")),
        "dir line missing: {messages:?}"
    );
    let table = messages
        .iter()
        .find(|m| m.contains("(index)"))
        .expect("table line missing");
    assert!(table.contains('a') && table.contains('b') && table.contains('4'), "{table}");
    assert!(messages.contains(&"g"));
    assert!(messages.contains(&"  inner"));
    assert!(messages.contains(&"outer"));
}

// ------------------------------------------------------------------ Stage 13
// M5 forms: value APIs, FormData, validation, focus, document.forms.

const FORM_DOC: &str = r#"<!doctype html>
<html><head><title>Forms</title></head><body>
<form id="f" action="/search" method="get">
<input id="q" name="q" value="orig" placeholder="Search" maxlength="10">
<input id="cb" type="checkbox" name="cb" value="yes" checked>
<input id="req" name="req" required>
<input id="em" name="em" type="email">
<select id="sel" name="sel"><option value="a">A</option><option value="b" selected>B</option><option value="c">C</option></select>
<textarea id="ta" name="ta">dflt</textarea>
<button id="go" type="submit" name="go" value="Go">Go</button>
</form>
</body></html>"#;

fn form_tab() -> TabJs {
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(TestHost::from_html(FORM_DOC)));
    tab
}

const FORM_SETUP: &str = "const f = document.getElementById('f'); const q = document.getElementById('q'); const cb = document.getElementById('cb'); const req = document.getElementById('req'); const em = document.getElementById('em'); const sel = document.getElementById('sel'); const ta = document.getElementById('ta');";

#[test]
fn form_value_apis_dirty_value_survives_set_attribute() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    run(&mut tab, "q.value = 'typed';");
    assert_eq!(read_str(&mut tab, "q.value"), "typed");
    run(&mut tab, "q.setAttribute('value', 'attr-change');");
    assert_eq!(read_str(&mut tab, "q.value"), "typed");
    assert_eq!(read_str(&mut tab, "q.defaultValue"), "attr-change");
    assert_eq!(read_str(&mut tab, "q.placeholder"), "Search");
    assert_eq!(read_num(&mut tab, "q.maxLength"), 10.0);
    assert_eq!(read_str(&mut tab, "q.type"), "text");
    assert_eq!(read_str(&mut tab, "q.name"), "q");
    assert!(!read_bool(&mut tab, "q.disabled"));
    assert_eq!(read_str(&mut tab, "ta.value"), "dflt");
    assert_eq!(read_str(&mut tab, "ta.type"), "textarea");
}

#[test]
fn form_checked_and_select_apis() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    assert!(read_bool(&mut tab, "cb.checked"));
    assert!(read_bool(&mut tab, "cb.defaultChecked"));
    run(&mut tab, "cb.checked = false;");
    assert!(!read_bool(&mut tab, "cb.checked"));
    assert!(read_bool(&mut tab, "cb.defaultChecked"));
    assert_eq!(read_num(&mut tab, "sel.selectedIndex"), 1.0);
    assert_eq!(read_str(&mut tab, "sel.value"), "b");
    assert_eq!(read_num(&mut tab, "sel.options.length"), 3.0);
    assert_eq!(read_str(&mut tab, "sel.options[2].textContent"), "C");
    assert!(read_bool(&mut tab, "sel.options[1].selected"));
    run(&mut tab, "sel.selectedIndex = 0;");
    assert_eq!(read_str(&mut tab, "sel.value"), "a");
    assert_eq!(read_str(&mut tab, "sel.options[0].selected ? 'y' : 'n'"), "y");
    assert_eq!(read_str(&mut tab, "q.form.id"), "f");
    assert_eq!(read_num(&mut tab, "document.forms.length"), 1.0);
    assert_eq!(read_num(&mut tab, "f.elements.length"), 7.0);
}

#[test]
fn form_selection_apis() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    run(&mut tab, "q.value = 'hello'; q.setSelectionRange(1, 3);");
    assert_eq!(read_num(&mut tab, "q.selectionStart"), 1.0);
    assert_eq!(read_num(&mut tab, "q.selectionEnd"), 3.0);
    run(&mut tab, "q.select();");
    assert_eq!(read_num(&mut tab, "q.selectionStart"), 0.0);
    assert_eq!(read_num(&mut tab, "q.selectionEnd"), 5.0);
}

#[test]
fn form_data_reads_entry_list() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    run(&mut tab, "q.value = 'typed';");
    assert_eq!(read_str(&mut tab, "new FormData(f).get('q')"), "typed");
    assert_eq!(read_str(&mut tab, "new FormData(f).get('cb')"), "yes");
    assert_eq!(read_str(&mut tab, "new FormData(f).get('sel')"), "b");
    assert_eq!(read_str(&mut tab, "new FormData(f).get('ta')"), "dflt");
    assert!(read_bool(&mut tab, "new FormData(f).has('req')"));
    run(&mut tab, "var fd = new FormData(f); fd.append('x', '1'); fd.set('q', 'z');");
    assert_eq!(read_str(&mut tab, "fd.get('x')"), "1");
    assert_eq!(read_str(&mut tab, "fd.get('q')"), "z");
    assert_eq!(read_num(&mut tab, "fd.getAll('q').length"), 1.0);
}

#[test]
fn form_validation_required_and_email() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    assert!(!read_bool(&mut tab, "req.checkValidity()"));
    assert!(read_bool(&mut tab, "req.validity.valueMissing"));
    assert!(!read_bool(&mut tab, "req.validity.valid"));
    assert_eq!(read_str(&mut tab, "req.validationMessage"), "Please fill out this field.");
    run(&mut tab, "req.value = 'x';");
    assert!(read_bool(&mut tab, "req.checkValidity()"));
    assert!(read_bool(&mut tab, "em.checkValidity()"));
    run(&mut tab, "em.value = 'not-an-email';");
    assert!(!read_bool(&mut tab, "em.checkValidity()"));
    assert!(read_bool(&mut tab, "em.validity.typeMismatch"));
    run(&mut tab, "em.value = 'a@b.com';");
    assert!(read_bool(&mut tab, "em.checkValidity()"));
    run(&mut tab, "req.setCustomValidity('nope');");
    assert!(!read_bool(&mut tab, "req.checkValidity()"));
    assert!(read_bool(&mut tab, "req.validity.customError"));
    assert_eq!(read_str(&mut tab, "req.validationMessage"), "nope");
    assert!(!read_bool(&mut tab, "f.checkValidity()"));
}

#[test]
fn form_submit_event_and_validation_gate() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    // Invalid form: requestSubmit fires `invalid`, never `submit`.
    run(
        &mut tab,
        "window.__events = []; f.addEventListener('submit', function (e) { window.__events.push('submit'); }); f.addEventListener('invalid', function (e) { window.__events.push('invalid'); }, true); req.value = ''; f.requestSubmit();",
    );
    assert_eq!(read_str(&mut tab, "window.__events.join(',')"), "invalid");
    // Valid form: submit fires (the host queue is the renderer's).
    run(
        &mut tab,
        "req.value = 'x'; em.value = 'a@b.com'; f.requestSubmit();",
    );
    assert_eq!(
        read_str(&mut tab, "window.__events.join(',')"),
        "invalid,submit"
    );
    // preventDefault on submit: the event still fires.
    run(
        &mut tab,
        "f.addEventListener('submit', function (e) { e.preventDefault(); window.__blocked = true; }); f.requestSubmit();",
    );
    assert!(read_bool(&mut tab, "window.__blocked"));
}

#[test]
fn form_reset_restores_defaults() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    run(&mut tab, "q.value = 'typed'; cb.checked = false; sel.selectedIndex = 2;");
    run(&mut tab, "f.reset();");
    assert_eq!(read_str(&mut tab, "q.value"), "orig");
    assert!(read_bool(&mut tab, "cb.checked"));
    assert_eq!(read_num(&mut tab, "sel.selectedIndex"), 1.0);
}

#[test]
fn form_focus_and_active_element() {
    let mut tab = form_tab();
    run(&mut tab, FORM_SETUP);
    assert_eq!(read_str(&mut tab, "document.activeElement.tagName"), "BODY");
    run(&mut tab, "q.focus();");
    assert_eq!(read_str(&mut tab, "document.activeElement.id"), "q");
    run(
        &mut tab,
        "window.__focus = []; q.addEventListener('blur', function () { window.__focus.push('blur'); }); sel.addEventListener('focus', function () { window.__focus.push('focus'); }); sel.focus();",
    );
    assert_eq!(read_str(&mut tab, "document.activeElement.id"), "sel");
    assert_eq!(read_str(&mut tab, "window.__focus.join(',')"), "blur,focus");
}
