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
use crate::bindings::test_host::TestHost;
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
