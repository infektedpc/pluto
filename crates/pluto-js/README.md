# pluto-js — JavaScript engine + event loop (Stage 7, wave 1)

The JS side of Pluto: the engine boundary, the per-tab event loop, one
realm per tab, error budgets, and an in-memory bytecode cache. Wave 1
builds the foundation **before any DOM binding exists** — later waves
build bindings, parser seams, and renderer integration on top of the
public API defined here, without reading the internals.

Status: wave 1 complete and wave 2a (DOM bindings) complete.
`cargo test -p pluto-js` is **70/70 green** — 29 wave-1 tests plus 41
binding tests; the engine probe is **14/16** (the 2 failures are
features the engine does not implement — see the probe verdict
below). No QuickJS type appears in this crate's public API.

## Engine pin

- **rquickjs 0.14.0** (workspace `Cargo.toml`: `rquickjs = { version =
  "0.14", features = ["futures", "macro"] }`; `Cargo.lock` pins
  rquickjs / rquickjs-core / rquickjs-macro / rquickjs-sys all to
  0.14.0). The pin was bumped from 0.9; `cargo update -p rquickjs`
  also moved `convert_case` 0.6.0 → 0.12.0 and `relative-path` 1.9.3 →
  2.0.3.
- **Bindgen-free build, verified.** The `bindgen` feature stays OFF
  (the VM's llvm18 sysroot ships libclang without C headers; see the
  NOTE in the workspace `Cargo.toml`). `cargo build -p pluto-js`
  succeeds with bindgen off — rquickjs-sys falls back to its
  pre-generated bindings. If a future pin cannot build without
  bindgen, that is a blocker, not a reason to turn bindgen on.
- The engine is QuickJS-NG (the `quickjs.c` vendored by
  rquickjs-sys 0.14.0). No JIT anywhere: adding one was explicitly
  decided against, and no JIT code (QuickJIT, TCC, or similar) is
  vendored or referenced.
- No Servo-origin code anywhere in this crate. The parser, DOM, and
  layout are Pluto's own; this crate only embeds the JS engine.

## ES-version probe verdict (Recommendation 2)

`src/probe.rs` (`probe_engine()`, run under `cargo test -p pluto-js`)
is the regression gate for engine upgrades — do not delete it. It
runs a feature checklist plus a small Test262-style subset against
the pinned engine and prints pass/fail per item
(`cargo test -p pluto-js --lib print_probe_table -- --nocapture`).

Result against rquickjs 0.14.0 / QuickJS-NG, 2026-11-24 — **14/16**:

| Item | Result | Note |
|---|---|---|
| modules | PASS | export + global side effect visible |
| top-level-await | PASS | `await Promise.resolve(42)` → 42 |
| async-generators | PASS | `for await` over an async generator → `1,2,3` |
| bigint | PASS | |
| proxy | PASS | |
| temporal | **FAIL** | `Temporal` is `undefined` — not implemented |
| array-fromAsync | PASS | ES2023 `Array.fromAsync` works |
| decorators | **FAIL** | parser rejects standard `@dec m() {}` syntax (`invalid property name`) |
| test262: promise-ordering | PASS | `sync,micro` |
| test262: async-await | PASS | |
| test262: classes | PASS | incl. `extends` + `super` |
| test262: generators | PASS | |
| test262: destructuring | PASS | incl. defaults + rest |
| test262: optional-chaining | PASS | |
| test262: nullish-coalescing | PASS | |
| test262: promise-all | PASS | |

**Verdict:** the ES2020-vs-ES2023 discrepancy is resolved with data.
The floor is **ES2020** (`PROBE_ES_FLOOR` in `src/probe.rs`): ES2020
core, modules, and top-level await all pass, and the probed ES2022 /
ES2023 features pass too (`Array.fromAsync`, classes, etc.). The
ceiling is bounded by what the engine does *not* implement: **no
Temporal and no decorators**. A future engine upgrade must re-run this
probe; a newly-failing row blocks the upgrade.

Two probe mechanics worth knowing (both verified empirically, not
assumed):

- **Async results are observed via a global, never via the eval
  promise's fulfilment value.** `Ctx::eval_promise` (JS_Eval +
  `JS_EVAL_FLAG_ASYNC`) fulfils with `{ value: <completion> }`, not
  with the completion itself — a `.then` handler attached to the
  promise receives the same wrapper object. Plain async functions and
  `await` behave correctly (`async function f() { return 42; }` fulfils
  with 42), so the wrap is specific to the async-eval entry point.
  `await_promise` therefore returns `JsValue::Undefined` on
  fulfilment (a script/module completion value is `undefined` per
  spec anyway) and the probe captures async results into
  `globalThis.__probe_v` and reads them back with a classic eval.
  Regression test: `realm::tests::top_level_await_result_is_observed_via_global`.
- **A top-level-await script does not adopt a returned promise unless
  it `await`s it.** `(async () => { ... })()` as a top-level-await
  source settles the eval promise immediately, before the IIFE runs
  — the wrapper evaluates the expression and returns, it does not
  await it. Async probe sources therefore `await` the IIFE:
  `await (async () => { ... })()`.

## Job-draining API (the report's open question — resolved)

Promise jobs land in the host microtask queue through the engine's
own job queue, drained one job at a time:

- **`Ctx::execute_pending_job(&self) -> bool`** — verified in the
  vendored source,
  `rquickjs-core-0.14.0/src/context/ctx.rs:399`. Synchronous; runs
  one pending QuickJS job, returns `true` if a job ran (even if it
  threw), `false` when the queue is empty. This is what the loop's
  microtask checkpoint uses (`realm.rs`: `drain_one_job`).
- Async twin: **`AsyncRuntime::execute_pending_job(&self) ->
  Result<bool, AsyncJobException>`**
  (`rquickjs-core-0.14.0/src/runtime/async.rs:266`), plus
  `AsyncRuntime::is_job_pending` (same file, line 256). The loop uses
  the sync `Ctx` variant (the loop is single-threaded per tab); the
  async twin is available for a future async driver.
- Promise jobs are **never** awaited ad-hoc on the tokio runtime:
  awaiting a promise as a Rust future parks the worker thread and
  nothing runs the engine's job queue while parked, so the promise
  could never settle. `await_promise` is a synchronous loop over
  `Promise::result()` + `Ctx::execute_pending_job()` instead (see
  "Watchdog" below for its limit).

## Public API surface (for wave 2)

No QuickJS type (`Ctx`, `Value`, `Function`, …) appears in any public
signature below. Engine values cross the boundary only as
`JsValue`.

| Item | File | What it is |
|---|---|---|
| `JsEngine` trait | `src/engine.rs` | The engine boundary: `eval`, `eval_module`, `compile_module`, `eval_module_bytecode`, `drain_jobs`, `memory_usage_bytes`. Implemented by `TabRealm`. |
| `JsValue` | `src/engine.rs` | Engine-agnostic value: `Undefined`, `Null`, `Bool`, `Number(f64)`, `String`, `Object`, `Function` (+ accessors). |
| `EvalKind` | `src/engine.rs` | `Classic`, `ClassicAwait`, `Module`. |
| `JsError` | `src/error.rs` | `Exception(String)`, `Interrupted`, `MemoryLimit`, `WatchdogTimeout`, `Poisoned`, `Internal(String)`; `is_script_error()` (Exception / Interrupted / MemoryLimit). |
| `DomHost` trait | `src/host.rs` | What pluto-js needs from the host: wave-1 operations (`validate` a `DomKey`, `document_element`, `body`, `get_element_by_id`, `text_content` / `set_text_content`, `get_attribute` / `set_attribute`, `request_restlye_and_layout`, `geometry` with forced synchronous layout flush, `forced_flush_count`) plus the wave-2a additive extensions (see "DOM bindings" below). All wave-2a methods have defaults, so a wave-1 host still compiles. |
| `DomHostError` | `src/host.rs` | `StaleKey`, `Hierarchy`, `NotFound`, `InvalidState` — mapped to JS `TypeError`, `HierarchyRequestError`, `NotFoundError`, `InvalidStateError` respectively. |
| `FetchBackend` trait | `src/bindings/fetch.rs` | The network backend for `fetch()`: `start_fetch(request, done)` — returns promptly, sends exactly one `FetchCompletion` on the channel from any thread; never resolves anything itself, never calls into the realm. `FetchRequest` / `FetchResponse` / `FetchError` / `FetchCompletion` are plain data (see "DOM bindings" below). |
| `bindings` module | `src/bindings/` | `mod.rs` (registry + wrapper identity + shared helpers), `dom.rs` (`window` / `document` / `Element` / `Node`), `events.rs` (`EventTarget` / `Event` / `CustomEvent`), `fetch.rs` (`fetch` / `Response` / `Headers`), `test_host.rs` (a `DomHost` over a real `pluto-html::Dom`, test-only), `tests.rs` (41 binding tests). |
| `DomKey` | `src/host.rs` | A generational slotmap key — currently `pluto_html::NodeId`. JS wrappers hold only this integer, validated on every crossing (stale key → null / no-op, never a use-after-free). |
| `Geometry` | `src/host.rs` | Layout geometry returned to JS (x, y, width, height). |
| `TabRealm` | `src/realm.rs` | One realm: `AsyncRuntime` + `AsyncContext`, owned thread id, wall-clock budget, memory limit, rejection sink. `RealmConfig` carries `memory_limit_bytes` (default 256 MiB) and `script_budget` (default 5 s). |
| `TabJs` | `src/tab.rs` | The per-tab host: `LoopState` + `TabRealm`. `new`, `navigate` (fresh realm + cleared queues + epoch bump), `eval_script` (enqueue + one turn), `eval_value` (checkpoint then eval), `run_turn`, `run_until_idle`, `script_errors`, `memory_usage_bytes`, `load_module_cached`. |
| `LoopState` | `src/event_loop.rs` | Per-source task queues, one microtask queue, timer wheel, rAF registry, epoch. |
| `TaskSource` | `src/event_loop.rs` | `Parser`, `Script`, `Timer`, `Networking`, `UserInteraction`, `DomManipulation`, `Rendering`. Priority: UserInteraction > DomManipulation > Networking > Parser > Script > Timer; Rendering is never dispatched as a task. |
| `Task` / `TaskKind` | `src/event_loop.rs` | `Script { source, url }`, `Timer { id, nesting, repeats }`, `AnimationFrame { id, timestamp_ms }`, `Host(closure)`. |
| `BytecodeCache` | `src/bytecode.rs` | In-memory, bounded-LRU bytecode cache (see below). |
| `probe_engine()` / `ProbeItem` | `src/probe.rs` | The engine feature probe; `PROBE_ES_FLOOR = "ES2020"`. |

Constraints wave 2 must keep:

- `pluto-js` does **not** depend on `pluto-layout` — layout is
  reached through `DomHost` (trait/channel) only.
- No new `pluto-ipc` message types were added in this wave.
- The stack stays hyper + rustls/ring only; single-version pins
  stand (wgpu 28, cosmic-text 0.19, taffy 0.15, libcosmic rev
  ecc5fe4).

## DOM bindings (wave 2a, Recommendation 5)

`src/bindings/` — hand-written WebIDL subset, no codegen (Rec 11).
No QuickJS type appears in any public signature; all QuickJS types
stay inside pluto-js. `pluto-js` does **not** depend on
`pluto-layout`: selector matching is reached through
`DomHost::query_selector(All)` (the production host uses
`pluto-layout::selector_matches`; the test host carries a test-only
matcher — see `test_host.rs`).

### Design (from the report, non-negotiable)

- **No `Persistent` in Rust at all.** All JS values (wrappers,
  listener functions, pending-fetch resolve/reject) live in JS-side
  `Map`s inside ONE registry object on `globalThis`, keyed by
  `Symbol.for("pluto.registry")` (reconstructed per entry).
  `BindingState` (Rust) holds only plain data: wrapper-id maps,
  pending fetches, response bodies/headers, console lines, errors.
- **Wrapper identity:** JS wrappers hold ONLY generational `DomKey`
  integers (stamped under `Symbol.for("pluto.domkey")` as a numeric
  id). `wrap_node` checks the per-realm cache first, so
  `document.body === document.body`. Every crossing validates the
  key against the live Dom; stale key ⇒ `null` for getters /
  `TypeError` for setters, never a use-after-free. An evicted id
  stays stale: getters use `dom_key_of_this_opt` (returns `None`
  for an evicted id instead of throwing), setters keep throwing.
- **Prototypes:** plain JS objects chained `Element → Node →
  EventTarget → Object.prototype`, `Document → Node → EventTarget`,
  `Text → Node`; `Event`/`CustomEvent` are JS constructors.
  `window`/`self` are `globalThis` itself; `document` is an
  enumerable accessor on `globalThis` whose getter wraps the host's
  document node lazily.
- **`this` validation:** every method validates `this` (wrong `this`
  ⇒ `TypeError`). Argument conversion is per WebIDL; departures are
  flagged in code comments ("M4: …").
- **Errors:** Rust exceptions surface as JS exceptions with the
  right name (`TypeError`, `HierarchyRequestError`, `NotFoundError`,
  `InvalidStateError`, `SyntaxError`, `InvalidCharacterError`);
  there is no `DOMException` global in QuickJS, so DOMException
  names are raised as plain `Error` with `name` set (`err.name`
  reads correctly).
- **Content attributes are live:** `id` / `className` read from the
  DOM on every access, never cached copies.
- **A throwing callback never kills the loop:** per-task isolation
  (wave 1) plus bindings report-then-continue. Per-script/callback
  errors go to the host error hook (`DomHost::report_error`) and to
  `window.onerror` first — if the handler returns `true`, the error
  is suppressed from `TabJs::script_errors` (per HTML); otherwise
  it is recorded and reported. A throwing `onerror` is recorded in
  `BindingState.errors`.

### Per-item status

1. **`window` — done.** Global object, `window.window ===
   window`, `document`, `setTimeout`/`clearTimeout`,
   `setInterval`/`clearInterval`, `queueMicrotask`,
   `requestAnimationFrame`/`cancelAnimationFrame`, `fetch`,
   `console` (log/info/debug/warn/error → host log + `tracing`),
   `onerror` handler property (plain writable property, initially
   `null`), `performance.now()`. Timers go through the wave-1 timer
   wheel; rAF through the wave-1 rAF registry.
2. **`document` — done.** `documentElement`, `body`, `head`,
   `getElementById`, `querySelector`/`querySelectorAll` (via the
   host; test host matcher covers type/universal/`#id`/`.class`/
   `[attr]` incl. `= ~= |= ^= $= *=` / `:first-child` /
   `:last-child` / `:nth-child(an+b)` / `:not(<compound>)` and
   descendant/child/adjacent/general-sibling combinators, comma
   lists), `createElement`, `createTextNode`, `title`,
   `write`/`writeln`/`open`/`close` (via `DomHost::document_write`
   / `document_open` / `document_close`; the parser-side
   insertion-point stack is the parser agent's job), `addEventListener`.
3. **`Element`/`Node` — done, with two M4 limits.** `textContent`,
   `innerHTML` (fragment parse via `DomHost::set_inner_html`,
   scripts inert; the fragment lives in its own arena and is
   deep-cloned into the host Dom — `parse_fragment` returns a new
   `Dom`, nodes cannot be adopted across arenas),
   `appendChild`/`removeChild`/`insertBefore`/`replaceChild`,
   `setAttribute`/`getAttribute`/`removeAttribute`/`hasAttribute`,
   `classList` (`add`/`remove`/`toggle`/`contains`/`value`, cached
   per element), `style` (`cssText` + `setProperty`/
   `getPropertyValue`/`removeProperty`, cached per element),
   `addEventListener`, `click()`. Limits: `style` does **not**
   support direct named access (`el.style.color = …` — a
   Proxy-based named-properties layer is a later wave); `childNodes`
   returns a plain JS `Array`, not a live `NodeList`.
4. **`EventTarget` — done.** `addEventListener` /
   `removeEventListener` / `dispatchEvent`; `Event` with `type` /
   `target` / `currentTarget` / `eventPhase` / `preventDefault` /
   `stopPropagation` / `stopImmediatePropagation`; `CustomEvent`
   with `detail`. Listeners live in the registry `listeners` Map
   (`"<target-id>:<type>"` → array of `{func, capture, once,
   passive}` records); dedup is by JS function identity + capture.
   `dispatchEvent` is synchronous (DOM spec): capture walk window →
   ancestors root→target, target phase (both, registration order),
   bubble phase if `bubbles`; each listener runs with a microtask
   checkpoint after it (`Ctx::execute_pending_job` in a loop —
   verified not to deadlock by the dispatch tests); a throwing
   listener is recorded (with its real message, via `ctx.catch()`)
   and dispatch continues. External dispatch
   (`TabJs::dispatch_event`) enqueues a `UserInteraction`-source
   Host task.
5. **`fetch` / `Response` / `Headers` — done (JS surface).**
   `fetch()` creates the promise first, stashes resolve/reject in
   the registry, stores the request in `pending_fetches`, then
   starts the backend; resolution goes through the event loop's
   networking-source task queue — never inline on a network thread.
   `Response`: `ok`, `status`, `statusText`, `headers`, `text()`,
   `json()` (JSON.parse errors reject with a real `SyntaxError`),
   plus the `Response` constructor. `Headers`: `get`/`has`/`set`/
   `append`, plus the `Headers` constructor (case-insensitive).
   Bodies are buffered in Rust (`BindingState`), not streamed.
   **Not shipped, by design:** the CORS check is a stub returning
   `Ok` with `TODO(M5)` + `TODO(M10)` (CSP `connect-src`) comments
   at the exact point, and `init.mode`/`credentials` are parsed and
   ignored with a `TODO(M5)`. **Wiring status (wave 2b):** the
   integration wave DID wire `fetch` to a real network backend
   (`PlutoFetchBackend` in pluto-browser, pluto-net + Stage-5
   `RequestResource`) — and that backend IS the M4 origin check:
   same-origin only, absolute cross-origin URLs fail before any
   network access. Full CORS preflight / `mode` handling is still
   M5; CSP `connect-src` is still M10.
   `FetchBackend` (below) is implemented by the integration wave
   with pluto-net + Stage-5 `RequestResource`.

### `DomHost` extensions (wave 2a, all additive with defaults)

```rust
pub enum DomHostError { StaleKey, Hierarchy, NotFound, InvalidState }
fn document_node(&self) -> Option<DomKey>            // default None
fn head(&self) -> Option<DomKey>                     // default None
fn parent(&self, key: DomKey) -> Option<DomKey>      // default None
fn children(&self, key: DomKey) -> Vec<DomKey>       // default empty
fn tag_name(&self, key: DomKey) -> Option<String>    // default None (stored case; binding uppercases)
fn node_type(&self, key: DomKey) -> Option<u16>      // 1 element, 3 text, 8 comment, 9 document, 10 doctype
fn node_name(&self, key: DomKey) -> Option<String>   // default None
fn title(&self) -> Option<String>                    // default None
fn set_title(&mut self, title: &str) -> bool         // default false
fn create_element(&mut self, tag: &str) -> Option<DomKey>
fn create_text_node(&mut self, data: &str) -> Option<DomKey>
fn append_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError>
fn insert_before(&mut self, parent: DomKey, child: DomKey, sibling: Option<DomKey>) -> Result<(), DomHostError>
fn remove_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError>
fn replace_child(&mut self, parent: DomKey, old: DomKey, new: DomKey) -> Result<(), DomHostError>
fn remove_attribute(&mut self, key: DomKey, name: &str) -> bool
fn has_attribute(&self, key: DomKey, name: &str) -> bool
fn query_selector(&self, scope: DomKey, selector: &str) -> Option<DomKey>
fn query_selector_all(&self, scope: DomKey, selector: &str) -> Vec<DomKey>
fn inner_html(&self, key: DomKey) -> Option<String>
fn set_inner_html(&mut self, key: DomKey, html: &str) -> bool  // fragment parse, scripts inert
fn document_write(&mut self, text: &str) -> bool
fn document_open(&mut self) -> bool
fn document_close(&mut self) -> bool
fn style_text(&self, key: DomKey) -> Option<String>  // raw `style` attribute
fn set_style_text(&mut self, key: DomKey, css: &str) -> bool
fn report_error(&self, message: &str)               // default no-op; integration wires to window.onerror / shell log
```

### `FetchBackend` trait (`src/bindings/fetch.rs`)

```rust
pub struct FetchRequest {
    pub id: u64,              // pending-fetch id, assigned by the binding; echoed in FetchCompletion
    pub url: String,          // absolute http(s)://… or same-origin path /…; anything else rejected by the binding
    pub method: String,       // uppercased, GET by default
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}
pub struct FetchResponse {
    pub status: u16,
    pub status_text: String,  // may be empty; binding fills a default (200 OK, 404 Not Found, …)
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,        // M4: bodies are buffered, not streamed
}
pub enum FetchError { Network(String), InvalidUrl(String), Cancelled }
pub struct FetchCompletion { pub id: u64, pub result: Result<FetchResponse, FetchError> }
pub trait FetchBackend: Send + Sync {
    /// Start a fetch. Return promptly (never block the worker
    /// thread) and send exactly one FetchCompletion on `done` —
    /// from any thread — when the outcome is known. The backend
    /// must never resolve anything itself and must never call
    /// into the realm.
    fn start_fetch(&self, request: FetchRequest, done: std::sync::mpsc::Sender<FetchCompletion>);
}
```

`TabJs` additions: `set_dom_host(Box<dyn DomHost>)`,
`set_fetch_backend(Arc<dyn FetchBackend>)`,
`console_lines() -> Vec<(String, String)>`,
`dispatch_event(DomKey, &str, bool)`. `run_turn` drains fetch
completions first (Networking-source Host tasks calling
`fetch::resolve_fetch`); `has_pending_work` counts in-flight
fetches (limit, documented in code: it cannot wait on an in-flight
async fetch whose completion has not arrived yet — the integration
wave must drive the loop while fetches are pending). Navigation
installs bindings on the fresh realm and re-attaches the host and
backend (fresh fetch channel per navigation).

### Test evidence (41 binding tests, all green)

- **Wrapper identity** (`wrapper_identity_holds`): `document.body
  === document.body`, `getElementById('para')` twice, `parentNode
  === getElementById('app')`, `querySelector('.item') ===
  querySelectorAll('.item')[0]`, `body.parentNode ===
  documentElement` — all `true`.
- **Stale keys** (`stale_keys_read_null_and_setters_throw`): a
  detached node is still live (removal ≠ staleness — `textContent`
  still reads, `getElementById` no longer finds it); after swapping
  the host for a fresh document, the old wrapper's getters read
  `null` repeatedly and its setter throws `TypeError`.
- **Symbol-key fix (wave-1 gap (a)) — done:**
  `timer_and_raf_maps_are_not_reachable_by_name` asserts `typeof
  __pluto_timers === 'undefined'`, `'__pluto_timers' in globalThis
  === false`, and timers still work. `tab.rs` / `realm.rs` no
  longer define `__pluto_*` properties; timers, rAF, wrappers,
  listeners, fetches, responses, headers, prototypes, classLists,
  and styles all live under `Symbol.for("pluto.registry")`.
- **Microtask gap (b) — not fixed, documented, untriggered:**
  bindings never enqueue Rust microtasks during JS execution
  (fetch goes via the task queue; `Response.text()`/`json()`
  resolve via promise jobs; dispatch checkpoints via
  `Ctx::execute_pending_job` directly), so the case cannot arise
  in this wave. Left for integration.
- **Dispatch:** `click_dispatches_and_bubbles` expects
  `btn:btn:btn,app,doc,win`; `capture_listeners_run_first`
  expects `capture,target,bubble`;
  `stop_propagation_and_prevent_default` expects the app listener
  suppressed and `dispatchEvent` returning `false` for a
  preventDefaulted cancelable event; `throwing_listener_does_not_stop_dispatch`
  records `event listener threw: listener-boom` and dispatch
  continues.
- **Fetch (fake backend):** `fetch_json_loader_updates_the_dom`
  (JSON `{"items":[1,2,3]}` → DOM `items:3`, request recorded with
  url `/api/items`, method GET), `fetch_response_fields_and_text`
  (`200|OK|true|yes|yes`, body `hello body`),
  `fetch_rejects_without_a_backend` (caught `TypeError`),
  `fetch_backend_error_rejects` (Network "connection refused"),
  `fetch_json_parse_error_rejects` (caught `SyntaxError`).

### M4 departures (in code comments too)

- Invalid `querySelector` returns `null` / `[]` instead of throwing
  `SyntaxError` (host query methods cannot yet distinguish invalid
  from no-match).
- `style` supports `cssText` + `setProperty`/`getPropertyValue`/
  `removeProperty` but not direct named access.
- `getAttribute` returns `String|null`, no namespace support.
- `childNodes` is a plain JS `Array`, not a live `NodeList`.
- DOMException names are raised as plain `Error` with `name` set
  (no `DOMException` global in QuickJS).
- `document.open()` returning false from the host is silently
  ignored per spec.
- `Response`/`Headers` constructors are JS functions delegating to
  Rust factories in the registry: a Rust `Function::new` is
  callable but not constructable in QuickJS.
- Rec 11 not built (each has a one-line `TODO(stage)` at its hook
  point): full WebIDL/codegen, workers, Shadow DOM,
  MutationObserver/ResizeObserver/IntersectionObserver, import maps,
  storage APIs, full fetch/CORS, CSP.

### Files changed (wave 2a)

`crates/pluto-js/` only (no other crate touched):

- `src/bindings/mod.rs` — `BindingState`, registry helpers,
  `wrapper_id_of` / `dom_key_of_this` / `dom_key_of_this_opt`,
  `throw_named` / `throw_type_error` / `throw_dom` /
  `throw_host_error`, `to_dom_string`, `install_bindings`,
  `call_onerror`.
- `src/bindings/dom.rs` — `wrap_node`, `evict_wrapper`,
  `document_wrapper`, prototypes, `window` / `console`,
  Node/Element/Document methods.
- `src/bindings/events.rs` — listeners, dispatch walk,
  `Event`/`CustomEvent` constructors, `dispatch_external`.
- `src/bindings/fetch.rs` — `FetchBackend` + request/response
  types, `fetch`, `Response`/`Headers` constructors + prototypes.
- `src/bindings/test_host.rs` — `TestHost` + test-only selector
  matcher.
- `src/bindings/tests.rs` — 41 binding tests.
- `src/host.rs` — additive `DomHost` extensions + `DomHostError`
  (no wave-1 signature changed).
- `src/lib.rs` — `pub mod bindings;` + re-exports.
- `src/tab.rs` — `BindingState`/host/backend fields, `set_dom_host`,
  `set_fetch_backend`, `console_lines`, `dispatch_event`, fetch
  drain in `run_turn`, `onerror` suppression in
  `record_or_propagate`, timer/rAF maps moved to the registry.
- `src/realm.rs` — map lookups via `registry_map` (no `Persistent`).
- `src/event_loop.rs` — comments only (registry map names).
- `README.md` — this section.

## Event loop (Recommendation 3)

`src/event_loop.rs` + the driver in `src/tab.rs`, built **before** any
DOM binding, proved with pure-JS ordering puzzles:

- **Per-source queues, not one FIFO.** FIFO only within a source.
  `next_task` fires due timers into the Timer queue first, then
  picks by `TaskSource::PRIORITY`.
- **One microtask queue.** The engine's pending-job queue *is* the
  microtask queue for all JS-originated microtasks (promise
  reactions, `queueMicrotask`, await continuations), drained one job
  at a time via `Ctx::execute_pending_job` from the host checkpoint.
  Rust-side microtasks live in a small separate queue drained first.
  Caveat, documented in code: a Rust microtask enqueued *during* JS
  execution would run before already-pending JS jobs (not spec
  order) — unreachable in wave 1, since no binding enqueues host
  microtasks during JS execution; wave 2 must route
  binding-originated microtasks through promise jobs.
- **Per-turn algorithm:** run one task to completion → drain
  microtasks fully → rendering opportunity check (`maybe_render`
  runs only when a frame is due AND (rAF pending OR render
  requested), with a shared frame timestamp; a microtask checkpoint
  runs after each rAF callback).
- **Timer wheel:** 5-level nesting / ≥ 4 ms clamp
  (`NESTING_CLAMP_LEVEL = 5`, `MIN_NESTED_TIMER_DELAY = 4 ms`).
  Nesting is the depth of the timer chain, not the call depth;
  intervals re-arm with nesting + 1 and the clamp applies on re-arm.
- **Epoch check at dispatch:** stale-epoch tasks (old document) are
  dropped, never run (`run_turn` compares `task.epoch` to the live
  epoch; `fire_due_timers` and `take_raf_frame` drop stale entries
  the same way).
- **Host globals** (loop infrastructure, not DOM bindings):
  `queueMicrotask` (defined in JS as `Promise.resolve().then(cb)` so
  it routes through the engine's job queue),
  `setTimeout` / `setInterval` / `clearTimeout` / `clearInterval`
  (one shared pool and id space), `requestAnimationFrame` /
  `cancelAnimationFrame`, `performance.now()`. Callbacks live in the
  realm's timer / rAF Maps — since wave 2a, inside the symbol-keyed
  registry (`Symbol.for("pluto.registry")`, see "DOM bindings"), not
  under reachable `__pluto_*` names — the Rust side stores only ids,
  never engine values, so there is no `Persistent` drop-order
  hazard. A page that guesses the symbol key can still reach the
  registry (`Symbol.for` is a shared registry); the symbol hides it
  from enumeration and name lookup, it is not a capability boundary.
- **Two bugs found by the tests, fixed in this wave:** (1) timer
  callbacks were fired with the Map method called unbound —
  extracting `Map.prototype.set`/`get`/`delete` and calling it
  without `this` throws `Map object expected`; all four call sites
  now pass the Map as `This(...)`. The same trap applies to any
  future host function that calls a JS method — always bind `this`
  explicitly. (2) `run_task` removed the callback from the Map after
  every timer firing, which killed intervals after their first
  firing (the wheel kept re-arming a callback that no longer
  existed, so `clearInterval` never ran and `run_until_idle` hit its
  30 s cap); `TaskKind::Timer` now carries `repeats`, and only
  one-shot timers consume their callback.

Tests that prove the ordering (all in `src/tab.rs`):
`ordering_timeout_vs_promise_vs_microtask_vs_raf` (expected
`sync,promise,microtask,raf,timeout`), `microtask_queueing_microtask_drains_in_order`,
`timer_nesting_clamp` (7 hops; hops 6–7 each ≥ 4 ms),
`interval_repeats_and_clears`, `clear_timeout_cancels`,
`throwing_timer_does_not_kill_the_loop`,
`throwing_script_does_not_kill_later_scripts`,
`stale_epoch_tasks_are_dropped_at_dispatch`,
`raf_callbacks_share_one_timestamp`,
`navigation_resets_realm_and_bumps_epoch`, `module_cache_hit_on_second_load`.

## One realm per tab (Recommendation 4)

`src/realm.rs` (`TabRealm`) + `src/tab.rs` (`TabJs::navigate` /
`recreate_realm`):

- One `AsyncRuntime` + `AsyncContext` per tab, created on
  navigation on the per-tab worker thread, dropped on navigation
  (fresh realm per navigation, queues cleared, epoch bumped). No
  shared JS state across tabs.
- **Owner-thread assertion:** the owning `ThreadId` is recorded at
  creation; every realm entry asserts on it (`assert_owner`, a
  `debug_assert` by design — free in release). The realm owns a
  current-thread tokio runtime used ONLY as an executor for rquickjs
  futures via `block_on`.
- **Two invariants, documented in code:** never call realm methods
  from inside a future running on the realm's own runtime
  (`block_on`-in-`block_on` panics); never call realm methods from a
  host function invoked by JS (the `with` lock is held → deadlock).
  Host closures touch only `LoopState`, never the realm.
- Rejection tracking: an unhandled rejection is reported through the
  `rejection_sink` (`mpsc::Sender<String>` in `RealmConfig`);
  `TabJs::script_errors` drains it first.

## Error budgets + watchdog (Recommendation 9, this wave's half)

Shipped in this wave:

- **Memory limit:** `Runtime::set_memory_limit` at runtime creation
  (`RealmConfig::memory_limit_bytes`, default 256 MiB). Over-limit
  allocations fail with `JsError::MemoryLimit`.
- **Interrupt handler:** `Runtime::set_interrupt_handler` with a
  wall-clock check against an armed deadline (Svix pattern):
  `arm_budget` before each call, `disarm_budget` after;
  `classify_call_result` must run *before* `disarm_budget` (the
  tripped flag is meaningless after). A `while (true) {}` spin is
  killed as `JsError::Interrupted` and the realm survives
  (test: `interrupt_handler_kills_a_spin_loop`, 100 ms budget).
- **Watchdog: shipped, with a documented limit.** `await_promise`
  drives a promise synchronously (job loop, not `into_future`);
  if the job queue empties while the promise is still pending, the
  realm is **poisoned** (`WOULD_BLOCK` sentinel) and the call
  returns `JsError::WatchdogTimeout`; later evals return
  `JsError::Poisoned`. Test: the watchdog kills a hung await and
  the realm stays poisoned afterwards. **Limit:** a module that
  top-level-awaits a *host event* (a timer, a fetch) cannot be
  evaluated through `await_promise` — it WouldBlocks immediately,
  because no host event can fire while the worker thread is driving
  the job loop. Wave 2 must drive such modules from the loop across
  turns; the seam is the poison/watchdog error type plus
  `TabJs::run_turn` (the loop, not the eval call, is where host
  events arrive).

## Bytecode cache (cheap win (b))

`src/bytecode.rs` (`BytecodeCache`): compile once per script URL,
reuse on repeat loads.

- Key = URL + SHA-256 of the source (`sha2` 0.11, added to the
  workspace deps; a changed source under the same URL recompiles).
- Bounded LRU by bytes (default 8 MiB in `TabJs`,
  `DEFAULT_BYTECODE_CAPACITY_BYTES`); oversized entries are not
  stored; eviction is bounded.
- **Modules only.** Classic scripts are not cached: rquickjs's
  bytecode API is module-shaped (`Module::declare` /
  `Module::evaluate` / `Module::write` / `Module::load`); classic
  scripts always compile. `load_classic` exists for the symmetric
  API but always compiles.
- Test evidence (`bytecode::tests`): a counting fake engine proves a
  second load of the same module reuses bytecode (compile count stays
  1), a changed source under the same URL recompiles, LRU eviction
  stays bounded, and an oversized entry is not stored.
- The integration wave wires it to fetched scripts; `TabJs` exposes
  it via `load_module_cached` / `bytecode_cache`.

## -Os / LTO for the QuickJS C build (cheap win (a)) — NOT shipped

Investigated and **deliberately not implemented**; do not fake it.

- The mechanism works: appending `[profile.dev.build-override]
  opt-level = "s"` to the workspace `Cargo.toml`, `cargo clean -p
  rquickjs-sys`, and rebuilding with `CC_ENABLE_DEBUG_OUTPUT=1`
  showed `cc` compiling `quickjs.c` with `-Os` (vs `-O0` without
  the override). rquickjs-sys's `build.rs` uses `cc::Build` with no
  explicit `.opt_level()`, so `cc` scrapes `OPT_LEVEL` from the
  build-script environment.
- Why not shipped: the same build also compiled `rquickjs-core`
  with `-C opt-level=s` (as a proc-macro dependency) and would
  apply `-Os` to *every* other build script's `cc` invocation in
  the workspace — including ring's C. Cargo has no per-crate
  `CFLAGS`; a `CFLAGS` env var is workspace-global. LTO is worse:
  it needs link-time `-flto`, which rustc does not pass, and slim
  LTO objects risk link failure — plus it would change every
  binary's link. The txiki.js measurement (-41% execution time for
  +6.7% size from `-Os` + LTO) is therefore not reachable cleanly
  from this workspace's build config. Revisit if Cargo gains a
  per-crate C-flags mechanism, or if the build moves to a wrapper
  `cc` that filters by crate (not attempted — out of scope for
  wave 1).
- Cleanup: the experiment was fully reverted (workspace `Cargo.toml`
  restored from backup, `cargo clean -p rquickjs-sys -p ring`);
  the workspace builds the C code at the default `-O0` in dev.
  Log from the experiment: `/tmp/os-probe.log` (ephemeral; not part
  of the repo).

## Files changed (wave 1)

Workspace:

- `Cargo.toml` — rquickjs 0.9 → 0.14; added `sha2 = "0.11"` to
  `[workspace.dependencies]`.
- `Cargo.lock` — rquickjs 0.14.0 (plus `convert_case` 0.12.0,
  `relative-path` 2.0.3).

`crates/pluto-js/` (the only crate touched):

- `Cargo.toml` — added `pluto-html` and `sha2` workspace deps.
- `src/lib.rs` — rewritten: module docs + re-exports. The old
  `JsRuntime` stub is gone (nothing else in the workspace
  referenced pluto-js).
- `src/error.rs` — `JsError`.
- `src/engine.rs` — `JsEngine`, `JsValue`, `EvalKind`.
- `src/host.rs` — `DomHost`, `DomKey`, `Geometry`.
- `src/bytecode.rs` — `BytecodeCache` + tests.
- `src/event_loop.rs` — `TaskSource`, `Task`, `TaskKind`,
  `Microtask`, `TimerEntry`, `RafEntry`, `LoopState`.
- `src/realm.rs` — `TabRealm`, `RealmConfig`, budgets, watchdog,
  job draining.
- `src/tab.rs` — `TabJs`, host globals, turn driver, tests.
- `src/probe.rs` — engine probe + regression tests.
- `README.md` — this file.

## Running the tests

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo test -p pluto-js            # 70 tests, all green (29 wave-1 + 41 bindings)
cargo test -p pluto-js --lib probe -- --nocapture   # probe table
cargo test                        # full workspace, no regressions
```

Formatting: the whole workspace is fmt-dirty vs local rustfmt
(pre-existing, every crate) — do not run a blanket `cargo fmt`;
hand-format only your own hunks. Pre-existing clippy warnings exist
elsewhere; don't chase them unless your change touches them.
