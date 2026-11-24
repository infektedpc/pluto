//! Binding tests (wave 2a): every item of §(f) 1–5 exercised
//! through `TabJs` + the [`TestHost`], plus the wrapper-identity /
//! stale-key / symbol-registry evidence the task requires.
//!
//! The tests run page JS and assert on the DOM (via the host) and on
//! JS-visible values (via `TabJs::eval_value`), so a passing test
//! means the whole crossing works — not just the Rust side.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;

use crate::bindings::fetch::{FetchBackend, FetchCompletion, FetchError, FetchRequest, FetchResponse};
use crate::bindings::test_host::TestHost;
use crate::engine::JsValue;
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
}

impl FakeBackend {
    fn ok(body: &str) -> (Arc<Self>, Arc<Mutex<Vec<FetchRequest>>>) {
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
            },
            error: None,
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
            },
            error: Some(error),
        })
    }
}

impl FetchBackend for FakeBackend {
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
