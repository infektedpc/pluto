# How the Pluto stack connects

## Why this stack hangs together

This is a cohesive, modern, zero-Mozilla/Google/Apple Rust browser stack —
no Servo, no Chromium, no WebKit. The pairing is not accidental:
libcosmic / iced for the shell pairs naturally with wgpu and cosmic-text,
because **iced already uses wgpu and cosmic-text under the hood**. The
browser chrome and the web content therefore share one unified rendering
and text pipeline: one `wgpu::Device`/`Queue`, one font system, one glyph
shaping path. The shader viewport in `pluto-browser` is not a second
renderer bolted on beside iced — it is a window into the same GPU context
iced is already drawing the nav bar with.

That is also why the version rule below is load-bearing rather than
cosmetic. One shared rule holds the whole stack together: **the workspace
pins one version of every shared crate** (`wgpu 28`, `cosmic-text 0.19`,
`taffy 0.15` — the versions libcosmic rev `ecc5fe4` itself uses). If any
crate drifted to a second major version, the `wgpu::Device`/`Queue` handed
out by libcosmic would be a different type from the one `pluto-paint`
sees, and the text buffers `pluto-layout` measures would not be the ones
`pluto-paint` draws. See `README.md` for why those versions, not the
older ones in early drafts.

| Layer | Crates | Integration glue to write |
| --- | --- | --- |
| 1. Shell & Presentation | libcosmic / iced + wgpu | Custom `iced::widget::Shader` (or Canvas / offscreen `wgpu::Texture`) viewport inside the libcosmic window, so web pages render into a GPU texture composited directly inside the browser tab area. Lives in `pluto-browser/src/viewport.rs`. |
| 2. Networking | hyper + rustls + tokio | `ResourceLoader` in `pluto-net`: one shared hyper client for the whole browser (explicit pool caps: 6 idle/host, 90 s idle timeout, timer set) + one shared rustls config (ALPN `[h2, http/1.1]`, shared resumption, 0-RTT off) over TLS 1.3 (rustls + webpki-roots); a scheduler (priority queue, 6 in-flight/origin, dedup, timeouts) streams `ResourceEvent`s (headers / chunks / done / error / cancelled) per `ResourceRequest`; gzip/deflate decoding, mixed-content gate, redirect audit (301/302/303/307/308, 5 hops, no https→http downgrade). No per-navigation client: the M1 fresh-loader-per-navigation defeated pooling and resumption. |
| 3. Parsing & DOM | Pluto's own WHATWG parser + slotmap | Pluto's own tokenizer + tree-construction machine (WHATWG, `pluto-html`) parses HTML5 into an arena-allocated DOM tree (`NodeId` handles instead of `Rc<RefCell<T>>` cycles); winnow remains for CSS tokenization only. Lives in `pluto-html` / `pluto-css`. |
| 4. Layout & Text | taffy + cosmic-text | `taffy::LayoutPartialTree` over the styled DOM nodes. For text leaf nodes, pass `cosmic_text::Buffer` shaping results inside Taffy's `measure_function` so line wrapping and intrinsic sizes match pixel-for-pixel. Lives in `pluto-layout`. |
| 5. GPU Rendering | wgpu + cosmic-text (swash) | Walk the laid-out Taffy tree in paint order (z-index / stacking contexts) to emit quads (backgrounds, borders, images) and glyph atlases (via cosmic-text's `SwashCache`) into a wgpu render pass. Lives in `pluto-paint`; uploaded in `viewport.rs::Primitive::prepare`, drawn in `Primitive::render`. |
| 6. Scripting | rquickjs | `rquickjs::AsyncRuntime` on the tab's event loop, exposing `NodeId`-backed host objects (`document.getElementById`, `addEventListener`, `fetch`) that mutate the DOM and trigger re-style/re-layout. Lives in `pluto-js`. |

## Data flow (vertical slice)

```
Go pressed
  -> worker: start_navigation cancels the old navigation's token,
     spawns fetch+parse; outcome returns over a channel, stale
     epochs discarded; layout runs in the loop at the current viewport
  -> pluto-net Scheduler::submit(ResourceRequest)   (ResourceEvents)
  -> pluto-html parse -> Dom arena                (NodeIds)
  -> pluto-css parse + cascade -> styled tree     (computed styles)
  -> pluto-layout Taffy tree + measure_fn         (boxes)
  -> pluto-paint DisplayList                      (quads + glyphs)
  -> viewport Primitive::prepare (upload) + render (draw)
  <- pluto-js mutates the Dom -> re-style -> re-layout -> re-paint
```

## Status

- Layer 1 shell + viewport: scaffolded, window opens, viewport is a
  blank white quad (no display list yet).
- Layers 2–6: crate stubs shaped around their glue points (see each
  crate's `src/lib.rs` header).
- Stage 4 (browser architecture): `pluto-ipc` protocol crate
  (PROTOCOL_VERSION 2 since Stage 5 — the message shape changed),
  per-tab worker thread, epoch'd double-buffered compositor holder,
  owned/serialisable validated `DisplayList`, rayon style, image
  decode on the rayon pool. Rasterisation on a worker thread is NOT
  implementable inside iced's Shader widget API — documented in
  `compositor.rs`.
- Stage 5 (networking, applied 2026-11-23): `RequestResource` v1
  frozen (destination/priority/mode/credentials/origin/top-level-site/
  epoch + streaming `ResourceEvent` reply + `CancelResource`); one
  shared hyper client + one shared rustls config; scheduler with
  priority queue, per-origin caps, dedup, timeouts; streaming-only
  bodies with byte caps (Image 50 MiB, everything else 10 MiB);
  cancellation by navigation epoch + `CancellationToken`; mixed-
  content gate; redirect audit; gzip/deflate decoding. Correction
  to the Stage 5 report: Pluto has NO dedicated tokio runtime — all
  async work runs on the iced/libcosmic executor (tokio
  multi-thread); the "stay in-process on a dedicated tokio runtime
  through M5" departure is really "stay in-process on the iced
  executor through M5; no network process". Scheduler state is
  per-`ResourceLoader` (one per worker/tab today); the shared hyper
  client is process-wide, so pooling/resumption are browser-wide
  regardless. Deferred to their milestones: in-memory partitioned
  LRU cache (M2), HSTS + preconnect + DNS TTL cache (M3), cookie jar
  (M5 — evaluate `cookie_store` first pick vs rune-cookie/kekse),
  disk cache + HTTPS-only (M5), CORS in pluto-net (M5), prefetch
  (M5-ish; prerender deferred entirely), CSP + native content
  blocker (M10), H3/DoH/0-RTT (post-M5, feature-gated, only after
  measurement).
- Stage 6 application (integration, applied 2026-11-24): Pluto's
  own WHATWG parser — the tree-construction machine + tokenizer are
  Pluto's own (`crates/pluto-html/src/tokenizer.rs` +
  `builder.rs`, vendored WHATWG `entities.json` with a build-time
  generated table), written after Joel overruled the html5ever
  0.40.1 adoption on 2026-11-24 ("write Pluto's own, do not adopt
  Servo's"). No Servo-origin code remains in the build:
  html5ever/markup5ever/tendril/web_atoms are out of the workspace
  `Cargo.toml`, pluto-html's `Cargo.toml`, and `Cargo.lock`, and
  `sink.rs` (the html5ever `TreeSink` adapter) is deleted;
  `DomEvent` moved from `pluto_html::sink` to `pluto_html` (a
  one-line import fix in `page.rs`). html5lib-tests
  tree-construction corpus: 1965/1973 (99.6%) — the 8 misses are 6
  scripted tests (need JS) + 2 `<search>`; up from 1942/1973 with
  html5ever. Pluto keeps its `Dom`, streaming driver, preload
  scanner, and everything downstream. Streaming parser (`feed` ->
  `DomEvent`s, pause-at-script, encoding-restart buffering) drives
  the worker load path; the preload scanner fires subresource fetches
  (the Stage 5 scheduler dedups them); external `<link
  rel="stylesheet">` + `@import` (serial, resolved against the
  importing sheet, inserted in place, layer preserved) with
  render-blocking (5 s total timeout — a failed or timed-out sheet is
  skipped, never fatal) and full restyle on arrival; `media=""`
  honoured as a condition evaluated at layout time; minimal session
  history in the shell (back/forward stack; address-bar navigation
  and link clicks push entries); click hit-testing against the
  retained `LayoutTree` with `<a href>` navigation (new epoch, Stage
  5 cancellation); nested scroll containers scroll via
  `set_scroll_offset` + re-collect (no re-layout); dirty bits live on
  DOM nodes (Double Dirty Bit data) and in `NodeStyle` hint/damage.
  Deferred: true network streaming (feed chunks as they arrive — the
  M2 simplification scans the whole fetched body post-fetch),
  honouring non-UTF-8 encodings (needs an encoding crate), `:visited`
  (no visited set in `ElementState` yet — that field lives in
  pluto-layout), Spineless Traversal (later swap),
  CORS (M5), CSP (M10), H3/DoH/0-RTT (post-M5). Stage 8 Rec 3/4
  are done: the renderer honours `PushClip`/`PopClip` (rect clips
  as `set_scissor_rect` runs with CPU culling, rounded clips as
  SDF slots in the clip uniform) and rasterises rounded rects +
  box shadows (SDF coverage, erf blur profile, blur capped at
  64); the display list is walked once, in paint order, into
  per-pipeline runs. Stage 8 Rec 9 is done: `InputEvent` v3
  (pointer move/down/up, wheel with position + delta mode +
  gesture phase, keys) with `:hover`/`:active`/`:focus` restyle
  via the sheet's `rules_by_pseudo_class` index, wheel chaining
  over nested scroll containers with `overscroll-behavior`, and
  Tab/Enter/Space/arrows/PageUp/PageDown/Home keyboard handling.
- Stage 7 application (JS engine + event loop, applied 2026-11-24):
  rquickjs 0.14.0 pinned (QuickJS-NG, no JIT — a JIT is a whole
  second compiler, multi-month baseline; engine swap to V8 stays the
  later path). ES probe verdict: ES2020 floor, no Temporal /
  decorators. Cheap perf win in: per-URL in-memory bytecode cache
  (`pluto-js/src/bytecode.rs`); `-Os` for the QuickJS C build
  landed in Stage 12 (per-package profile overrides scoped to
  `rquickjs-sys`, dev + release — measured ~3.4x on a fixed JS
  workload: dev median 1625.8 ms → 479.3 ms, same result). LTO
  did NOT land: Cargo forbids `lto` in a per-package profile, a
  workspace-wide `[profile.release] lto` would change Rust
  codegen for every crate, and C-level LTO is not reachable
  through Cargo without touching the vendored build script (an
  engine fork, forbidden). One realm per tab: a fresh
  `AsyncRuntime` + `AsyncContext` per navigation, owned by the tab
  worker thread, dropped on navigation; the event loop is
  host-owned (HTML §8.1.7 shape: one task to completion, drain
  microtasks, update-the-rendering only at a rendering
  opportunity), task sources prioritised UserInteraction >
  DomManipulation > Networking > Parser > Script > Timer, Rendering
  never dispatched as a task. Bindings are hand-written (no
  WebIDL/codegen in M4): `window` / `document` / `Element` /
  `EventTarget` / `fetch` / `Response` / `Headers`, timers, rAF,
  console, `document.write` / `open` / `close` (via the parser's
  `DocumentWriter`, scripts inert in `innerHTML` and in written
  fragments), geometry reads. Parser seams: parser-blocking
  scripts execute at `ScriptPaused` (external fetched first,
  stylesheet gate waited on, then parsing resumes), defer drains
  in document order after the parse, then `DOMContentLoaded`;
  async scripts run on arrival; modules dedup by URL and are
  driven by the loop (`start_module` / `poll_module` — a module
  top-level-awaiting a host event never goes through
  `await_promise`, so it cannot poison the realm). Fetch is
  promise-over-task, same-origin only: `PlutoFetchBackend`
  (pluto-browser) builds a Stage-5 `RequestResource` (destination
  Fetch, mode/credentials SameOrigin, epoch) over pluto-net,
  honours epoch cancellation + the mixed-content gate, and sends
  exactly one `FetchCompletion`; absolute cross-origin URLs fail
  before any network access, with `TODO(M5)` CORS and `TODO(M10)`
  CSP `connect-src` markers at that gate. Dirty-bit flow: a JS
  mutation marks the Double Dirty Bit flags + `render_requested`
  and returns with NO layout; the 16 ms frame tick re-lays out
  and submits only when dirty or an rAF callback is pending;
  geometry reads (`getBoundingClientRect` / `offsetWidth` /
  `offsetHeight`) force a synchronous flush via the layout actor,
  counted in a debug forced-flush counter (read/write loop: one
  flush per iteration, by design; batched writes: one flush).
  Full re-layout is the M4 fallback — dirty-subtree layout is
  Stage 8 Rec 6 (safe roots + full-layout fallback + a debug
  incremental==cold assertion), applied after Stage 7. rAF runs
  inside update-the-rendering with one shared frame timestamp;
  the cadence is a ~60 Hz timer on the tab thread (the
  shell/compositor exposes no frame signal), and a hidden tab
  stops producing frames in Stage 8 Rec 1: the shell forwards
  window focus as `SetVisible` (focus is the single-window
  proxy for visibility), the tab tick skips relayout while
  hidden, and JS timers keep running. Error isolation: a throwing script reports via
  `window.onerror` (`onerror` returning `true` suppresses the
  default report) and never blocks later scripts or the loop;
  5 s script budget + 256 MiB memory limit + interrupt handler
  (watchdog) in place. M4 done criteria met in worker tests
  (local test server): counter click increments, accordion click
  toggles, fetch-JSON loader updates the DOM, throwing script
  does not kill later scripts or the loop, navigation cancels a
  pending timer, module top-level-awaiting a fetch updates the
  DOM, `document.write` inserts at parse, `set_inner_html`
  keeps scripts inert, layout-thrash (5 flushes for a read/write
  loop, +1 for batched writes). Departures from the Stage 7
  report, each forced by the code: (1) the realm drives its
  rquickjs futures with a synchronous poll loop (`block_on_sync`)
  instead of owning a tokio runtime — a realm-owned runtime
  panics on nested `block_on` when `TabJs` is called from inside
  the tab thread's network `block_on`; (2) the tab thread runs an
  async main loop (`LocalSet` + `spawn_local` load tasks, script
  execution delegated from the load task to the main loop over
  epoch-stamped `ExecMsg`) instead of running the load inline —
  the inline design blocked the command loop, so a second
  navigation could not cancel the first mid-flight; (3)
  `ScriptHooks` methods are boxed futures and `parse_finished`
  takes/returns the `Document` by value — `async fn` trait
  methods are not dyn-compatible, and the load-task/main-loop
  split needs the document to move; (4) the frame interval uses
  `MissedTickBehavior::Delay` — `Burst` fired dozens of catch-up
  ticks after the blocking post-parse drive and starved the
  command channel for ~500 ms. Deferred, each with a one-line
  `TODO(stage)` at its hook point: full WebIDL/codegen (M5),
  workers (M5), Shadow DOM (M5), import maps (M5), storage APIs
  (M5 — landed in Stage 13: shared localStorage, `document.cookie`,
  storage events, `navigator.storage.estimate()`), full fetch/CORS (M5), CSP (M10). (IntersectionObserver
  was deferred here too — it landed as Stage 8 Rec 10, see the
  entry below.)
- Stage 8 application (rendering pipeline, applied 2026-11-24, all
  13 recommendations; original code only — concepts from the
  research report, nothing copied/ported/adapted from Chromium,
  Firefox, or WebKit): Rec 1 visibility + frame-skip gate (shell
  forwards window focus as `SetVisible`/`SetFocused`; hidden tabs
  keep JS timers but skip relayout, dirty flags stay set);
  Rec 2 pending/active display lists (submit lands pending, the
  frame boundary activates atomically; a frame built with no
  pending list while one is awaited counts a deadline miss);
  Rec 3/4 clip + raster pipelines (one `prepare` walk emitting
  ordered runs with per-run scissor + clip bind group; CPU
  culling; damage scissor; clean skip on identical list +
  viewport + scroll); Rec 5 perf instrumentation (`perf::summary`
  per navigation, `frames_hidden_skipped`, WorkerBuild/Ipc spans,
  first-paint clock started on link clicks too); Rec 6
  dirty-subtree layout (damage classification, safe-root
  predicate, incremental relayout with full-rebuild fallback and
  a debug incremental==cold assertion — plus two fixes found by
  the worker tests: a text change inside an inline span was
  dropped as "not rendered", and every full rebuild re-scanned
  the system fonts, so the retained `FontSystem` is now reused
  across rebuilds); Rec 7 MutationObserver (spec-shaped records,
  delivery at the microtask checkpoint) and Rec 8 ResizeObserver
  (gather/broadcast between layout and display-list build);
  Rec 9 input v3 (see the M2 entry above); Rec 10
  IntersectionObserver (registry + threshold-band change
  detection in pluto-js, clip-chain geometry on the layout
  actor, queued-task delivery after paint — callbacks run on a
  later turn, never inline; viewport scroll is mirrored
  worker-side so viewport-root intersections need no protocol
  change, and wheel/keyboard scrolls re-run the step on the
  retained tree; `isIntersecting` follows the spec text — true
  on any overlap — not the Chrome/Firefox quirk in the report's
  open questions; per-axis overflow clipping, html/body overflow
  propagates to the viewport and never clips); Rec 11 glyph atlas
  LRU (guillotine packer, separate mask/colour atlases, growth
  1024→2048 before eviction) and Rec 12 image cache (64 MiB
  byte cap, LRU, decode-once dedup, mip chains, animated-frame
  dirtying — small-image atlas packing NOT done, follow-up);
  Rec 13 raster-thread spike: PROVEN at the wgpu level (a worker
  thread can encode into an offscreen texture with the device /
  queue handed to `Primitive::prepare` — both are `Send + Sync`
  in wgpu 28, and libcosmic exposes no device creation), but the
  production shell wiring (stash the clones, submit from
  `render`, blit into the swapchain view) is NOT done. On-box
  checks owed after the Stage 8 pull: live fetch + first load,
  resize re-layout, page + nested-container scroll, link-click
  navigation, Back/Forward, external CSS, hover restyle, wheel
  chaining, Tab focus movement, emoji rendering, downscaled
  image draw, animated-frame advance, IntersectionObserver
  initial entry + scroll flip on a real page.

## Roadmap (Joel, 2026-11-23 — replaces the scratched vertical-slice plan)

### Milestone 1: End-to-End Static Page Pipeline ("Hello, example.com")

Goal: type a URL into libcosmic, fetch it over HTTPS, parse basic
HTML/CSS, and draw colored boxes and text on the GPU.

- **pluto-net**: async HTTP/1.1 + HTTP/2 client (hyper + rustls +
  webpki-roots) with redirect following (301/302/303/307/308, 5 hops,
  no https→http downgrade), one shared client + shared rustls config
  (pool caps, resumption, 0-RTT off), scheduler (priority queue,
  per-origin caps, dedup, timeouts), streaming-only bodies with byte
  caps, cancellation by navigation epoch + token, mixed-content gate,
  gzip/deflate decoding.
- **pluto-html & pluto-css**: winnow parsers for core HTML tags
  (`<html>`, `<head>`, `<body>`, `<div>`, `<h1>`–`<h6>`, `<p>`,
  `<span>`, `<a>`, `<style>`) into a slotmap DOM arena, plus basic CSS
  rules (`display`, `width`, `height`, `margin`, `padding`,
  `background-color`, `color`, `font-size`).
- **pluto-layout & pluto-paint**: bridge DOM nodes to taffy (block &
  flex layout) and cosmic-text (text shaping + SwashCache glyph
  rasterization), rendering solid quads and glyphs via wgpu inside
  WebViewport.
- **Done criteria**: navigating to https://example.com renders its
  centered container, heading, paragraph, and link accurately inside
  the libcosmic window.

### Milestone 2: Real-World CSS Styling, Scrolling & Link Navigation

Goal: usable for documentation, blogs, and simple multi-page sites.

- **pluto-css**: full CSS selector matching (tag, `.class`, `#id`,
  descendant/child combinators, pseudo-classes like `:hover`),
  specificity sorting, User-Agent default stylesheet, and
  `<link rel="stylesheet">` external CSS fetching.
- **pluto-layout**: CSS Grid (taffy grid mode), border-radius,
  box-shadow, overflow clipping, vertical/horizontal scroll offsets.
- **pluto-browser**: mouse wheel scrolling, hit-testing (mapping
  (x, y) clicks to DOM nodes), clickable `<a href="...">` links, and
  Back/Forward session history stacks.
- **pluto-net**: in-memory partitioned LRU cache (deferred from
  Stage 5 — build at M2, not before).
- **Done criteria**: click links, scroll smoothly at 60+ FPS, browse
  real static sites (info.cern.ch, Hacker News, Rust docs).

### Milestone 3: Rich Media, Images & Multi-Tab Shell

Goal: visual pages with raster/vector images and standard tab management.

- **pluto-net & pluto-paint**: decode PNG, JPEG, WebP, GIF asynchronously
  (image crate), upload to `wgpu::Texture` atlases, support `<img>`
  intrinsic sizing and CSS `background-image`.
- **pluto-browser**: multiple tabs (each owning its own TabState, DOM,
  layout tree, history), favicons, page `<title>` updates, loading
  progress indicator.
- **pluto-net**: HSTS store + preconnect + DNS TTL cache (deferred
  from Stage 5 — build at M3, not before). Multi-tab promotes the
  loader to a shared handle; until then scheduler state (caps/dedup)
  is per-tab.
- **Done criteria**: pages with photos, icons, and multiple open tabs
  render cleanly without blocking the UI thread during image decoding.
- **Built — Stage 11 (2026-11-25)**: all 12 Stage 11 recommendations
  applied. Image fetch → decode → display is live: `<img>` and
  `background-image` bytes reach the page (fetch through the Stage 9
  gates, decode on the shared rayon pool, never on the tab or UI
  thread); intrinsic sizing from decoded dimensions with alt text /
  broken-image placeholder on failure; srcset/sizes/`<picture>`
  selection before fetching (one candidate only); GIF + APNG driver
  with per-frame delays, loop counts, ~30 fps cap, off-screen pause;
  strict decode limits (8192 px, 64 MiB alloc cap, rejected before
  allocating). IPC decision (documented in `pluto-ipc/src/lib.rs` and
  `pluto-paint/src/lib.rs`): **decode worker-side, resources cross
  IPC out-of-band, one `ResourceReady` frame per image**
  (PROTOCOL_VERSION 4 → 5); a display list never carries decoded
  pixels. Multi-tab shell: `Vec<Tab>` in the shell (own compositor,
  history, title, favicon, renderer_tx per tab), one renderer process
  per tab with a shell-assigned `tab_id`, tab strip above the address
  bar, Ctrl+T / Ctrl+W / Ctrl+Tab, switching swaps the compositor
  with no re-fetch and no re-layout. Loader promoted to a shared
  handle (`SharedLoader`: HSTS, blocker, cookie jar, HTTP cache
  shared; CSP/referrer/caps/dedup per tab); preconnect (max 3/page)
  + DNS TTL cache (60 s positive / 5 s negative) with HSTS applied
  before warming and no preconnect to blocked tracker origins.
  Favicons (`link[rel=icon]` + `/favicon.ico` fallback, once per
  origin per session, miss remembered, 32×32 downscale, shown in the
  tab strip); `TitleChanged` coalesced onto the display-list submit;
  loading progress (`LoadProgress` bytes vs Content-Length, thin bar
  under the address bar). Deferred: dirty-subtree re-layout on image
  arrival (full rebuild, coalesced to one per tick); HTTP cache
  sharing across renderer processes (needs the M5 broker);
  preconnect socket parking in the hyper-util pool (no public API);
  full `background-position`/repeat/multiple backgrounds (cover /
  contain + no-repeat first); per-tab `history.json` persistence
  (only the active tab's history is saved); `document.images` /
  JS `Image()` constructor.

### Milestone 4: Interactive JavaScript & Live DOM Mutation (rquickjs)

Goal: client-side JS that reads and modifies the live page.

- **pluto-js**: embed `rquickjs::AsyncRuntime` with a per-tab event loop
  (`setTimeout`, `setInterval`, `queueMicrotask`, Promise resolution)
  and `<script>` / `<script src="...">` execution in document order.
- **DOM & Web API bindings**: `window`, `document` (`querySelector`,
  `getElementById`, `createElement`), `Element` (`innerHTML`,
  `textContent`, `setAttribute`, `classList`, `style`), `EventTarget`
  (`addEventListener` for click, input, keydown), `fetch()`.
- **Incremental re-layout**: mark dirty DOM subtrees on JS mutation so
  Pluto re-runs cascade, taffy layout, and wgpu display-list generation
  on the next frame.
- **Done criteria**: interactive JS widgets (counters, accordions,
  client-side `fetch()` JSON loaders, dynamic DOM apps) work in real time.
- **Built — Stage 12 (2026-11-25)**: all 12 Stage 12 recommendations
  applied (M4 leftovers — the JS engine core was Stage 7). Module
  loader (Rec 1): `set_loader` on the realm; the load task
  pre-fetches the whole import graph through the existing loader
  (same Stage 9 gates — no second fetch path), sources land in a
  per-navigation `ModuleSources` map, the loader serves from it and
  rejects a miss cleanly; relative/root-absolute/absolute specifiers
  resolve against the importing module's URL; bare specifiers and
  `data:` are rejected (import maps stay M5); dynamic `import()` with
  a string literal goes through the same loader; no loader-side
  bytecode cache (the engine dedups loaded modules by resolved name
  per realm, so a loader cache could never get a second hit).
  `-Os` for the QuickJS C build (Rec 2, see the Stage 7 entry —
  dev median 1625.8 ms → 479.3 ms on the fixed workload test).
  Layout-kind measurement (Rec 3): the layout actor reports its
  `RelayoutKind` (`ColdBuild` for a fresh tree, `FullLayout` when the
  incremental path was not attempted or failed), every submitted
  frame records it in the frame-time histogram, and the periodic
  stderr summary carries the kind counts plus the fallback rate
  (`FullLayout / (SubtreeLayout + FullLayout)`, cold builds
  excluded). Safe-root widening (Rec 4), in stages: definite-width
  auto-height blocks are safe roots now (a height change propagates
  through taffy's recursive `mark_dirty` up the ancestor chain, and
  siblings whose available space changed miss taffy's cache and are
  re-measured), and the margin-collapse guard is gone (taffy
  recomputes the collapse for every dirty ancestor); the debug
  incremental==cold assertion is the tripwire at every step. Image
  map into the incremental path (Rec 5): an image arrival no longer
  forces a full rebuild — the fresh map rides the layout job, the
  changed `<img>` nodes become extra damage, and the subtree path
  runs with the new sizes (a changed background still forces a full
  rebuild — the map is keyed by URL and cannot name its boxes); the
  full-rebuild and debug-cold paths build with the retained map,
  never an empty one. Subtree restyle (Rec 6): the dirty LCA's
  subtree only, seeded from the retained parent style and root font
  size — the merged maps are identical to the whole-document pass
  (pinned by test). rAF stays on the 16 ms timer — there is no
  shell vsync signal to drive it (Rec 11a) — and the geometry-flush
  count now routes into the perf summary (Rec 11b;
  `flushes_found_nothing_dirty` stays 0 by design — the host never
  flushes a clean tree). Engine tracking (Rec 12): no fork — the ES
  gaps (Temporal, decorators) close only via an engine upgrade;
  re-run the ES probe when a newer rquickjs crate lands; Zipp stays
  a watch-candidate; the V8 swap stays the later path. Bindings
  batch (Rec 7): `window.location` (live getters, `assign` /
  `replace` / `reload` via `DomHost::request_navigation`), `URL` /
  `URLSearchParams` (full API, `===`-stable `searchParams`),
  `dataset` (live `Proxy`, `data-foo-bar` ↔ `fooBar`),
  `cloneNode(deep)`, `matches` / `closest`, `getElementsByTagName`
  / `getElementsByClassName` / `getElementsByName` (snapshots, not
  live collections), `document.images`, `Image()` +
  `HTMLImageElement` (`src` / `currentSrc` / `complete` /
  `naturalWidth` / `naturalHeight`, `load` / `error` via
  `fire_image_event`), typed events (`KeyboardEvent` with `key` /
  `code`, `MouseEvent` with `clientX` / `clientY` / `button`,
  `InputEvent` with `data` / `inputType`) via
  `TabJs::dispatch_typed_event`. Fetch surface (Rec 8): `Request`
  (usable as `fetch()` input), `AbortController` / `AbortSignal`
  (cancels backend fetches, rejects pending promises with
  `AbortError`), `Response.url` / `.redirected` / `.blob()` /
  `.arrayBuffer()` / `.clone()`, `Headers.delete` + iteration
  (insertion order — M4 order, not spec's sorted order).
  `window.onerror` (Rec 9) now receives source/line/col (parsed
  from the first stack frame; the `error` argument stays
  `undefined` — it does not survive the realm boundary). Console
  (Rec 10) extends to `trace` / `assert` / `count` / `time` /
  `dir` / `table` / `group`, all landing in `console_lines` and
  tracing. Built — wiring follow-up (`d5ecc741`, Stage 12): the
  production host now overrides the `DomHost` methods above
  (`location` reads the real URL, `cloneNode` round-trips,
  `dataset` reads/writes live attributes, `Image().src` loads
  with onload/onerror), typed events carry clientX/Y + key, and
  the `ImageStore` notifies on JS-created image resolve/fail.
  Deferred: dirty-subtree re-layout is still not the path for
  image arrivals that change a background; per-tab history is still
  single-file.

### Milestone 5: HTML Forms, Storage & Process Isolation

Goal: log into websites, persistent state, crash-safe tab isolation.

**Built — Stage 13 (2026-11-25):** form state model in `pluto-html`
(value/checked/selectedIndex + dirty flags, seeded at parse, reset
restores defaults); production key/input dispatch with
`preventDefault()` suppression; focus model (pointer-down moves
focus first, `tabindex` ordering, disabled excluded); text editing
with caret + selection paint and UA intrinsic sizing for unstyled
controls; value/state JS APIs + `FormData` + cancelable
submit/reset events; constraint validation + `:valid`/`:invalid`/
`:required`/`:optional`/`:disabled`/`:checked`/`:read-only`;
button semantics + submission through the existing loader (GET/POST,
three encodings, Stage 9 gates); cookie jar persisted to
`cookies.json` (session cookies stripped before the first save,
expired pruned, sequences re-based); one shared localStorage store
across tabs with a global cap, debounced saves, `document.cookie`,
storage events, `navigator.storage.estimate()`; fetch moved
browser-side behind the `NetworkBroker` (renderer holds no sockets
in broker mode — proven by a no-socket seccomp test loading a real
page); fail-closed sandbox reporting (`SandboxReport` over IPC),
bridge watchdog (heartbeat + 30 s silence → kill + `RendererCrashed`),
sad tab (`about:crashed` page, Reload re-navigates, 3-crashes-in-60 s
restart policy), and the in-process fallback gated behind
`PLUTO_ALLOW_INPROCESS_FALLBACK=1`. IPC `PROTOCOL_VERSION` 7.
Deferred from the Stage 13 research: storage backends still live in
the renderer (localStorage/sessionStorage writes land in the chroot
tmpfs in broker mode — moving them behind IPC needs a sync-over-IPC
redesign of the `DomHost` storage trait); broker-mode progress is a
single end-of-fetch report; `<meta name="referrer">` policy updates
are not yet threaded over IPC.

- **Forms & input controls**: `<input>`, `<textarea>`, `<button>`,
  `<form>` submission (GET/POST), text cursor caret rendering,
  selection, keyboard focus management.
- **Web storage & cookies**: Cookie jar persistence (Set-Cookie / Cookie
  headers), localStorage, sessionStorage backed by disk storage.
  The jar is browser-side only and is evaluated at M5 (`cookie_store`
  first pick vs rune-cookie/kekse — deferred from Stage 5, do not
  build earlier). `Set-Cookie` is already stripped from the
  `ResourceEvent` headers stream, so the renderer never sees it.
- **pluto-net at M5**: CORS in pluto-net, disk cache + HTTPS-only
  mode, prefetch (prerender deferred entirely) — all deferred from
  Stage 5.
- **Multi-process sandboxing**: split pluto-browser (UI, network,
  storage) from pluto-renderer worker processes over IPC, sharing
  display lists (and, later, rendered frames for video/canvas)
  cleanly so a misbehaving page or infinite JS loop cannot freeze or
  crash the main browser window. Display lists — not rendered
  frames — are the M5 handoff (Stage 4, departure D1): they align
  Pluto with Firefox/Servo rather than Chromium, keep the browser
  process owning the one wgpu device and font system, and make the
  IPC protocol (pluto-ipc, in-process since Stage 4) the same bytes
  that cross the process boundary. Frame-sharing via DMA-BUF stays
  a post-M5 optimisation for video/canvas only. The renderer at M5
  is the same binary re-executed with `--renderer` (D3); there is
  no separate GPU process at M5 (D4); seccomp sandboxing is
  M5-adjacent, not M5 itself (D5); and the two font systems at M5
  (one per process) are unavoidable under any out-of-process
  renderer (D2).
- **Done criteria**: type into search bars, submit forms, stay logged
  into basic web apps across restarts, close hung tabs cleanly.

### Milestone 6: Modern CSS Architecture & Responsive Design

Goal: handle modern web layouts, animations, and mobile/desktop
responsive sites without layout breakage.

**Built — Stage 14 (2026-11-25):** all 12 Stage 14
recommendations applied, in three phases; suite 845 → 928
(+83 tests). IPC `PROTOCOL_VERSION` 7 → 9. Phase 1 (value +
layout model): `Time`/`Angle`/`Transform`/`Var`/`Calc`/`Clamp`
values, one 2D `Affine` type with `TransformListExt::to_affine`
(origin-resolved against the border box), `Easing`,
`Transition`/`Animation` parsed types, structured `@keyframes`
steps, `Theme` + `prefers-color-scheme` media evaluation;
`ComputedStyle` gains `z_index`/`opacity`/`transform`/
`transform_origin`/`transform_box`/`will_change_transform`/
`transitions`/`animations`/`custom_properties`; transformed
containing blocks fixed both halves together (taffy
`Position::Relative` mapping for transformed static elements +
matching safe-root predicate change); `var()`/`calc()`/`clamp()`
resolution; two-value `background-repeat`. Phase 2 (paint +
IPC): CSS 2.1 Appendix E paint order as the single source of
truth for `DisplayList::build` and hit-testing; stacking
contexts (explicit + implicit) with a real clip stack (the
single `current_clip` scalar is gone); `PushOpacity`/`PopOpacity`
(group opacity as a documented vertex-alpha approximation —
true offscreen group compositing is a future step);
`PushTransform`/`PopTransform` with CPU-side vertex application
(no new bind group — both are spoken for); `CompositingOnly`
classifier (transform/opacity-only diffs, z-index-only diffs
are `PaintOnly`, never swallowed) + out-of-band
`CompositingUpdate` IPC message, so a transform-only frame
neither rebuilds nor resubmits the full list. Phase 3
(geometry + animation engine + extras): `position: fixed`
re-anchored to the viewport via `PushFixed` (translate by
`+scroll` at paint time; the false safe-root exemption that
assumed fixed resolves against the viewport is gone);
positioned inline elements now build a real box instead of
vanishing; `position: sticky` as a scroll-derived offset via
`position_overrides` (never through taffy insets); animation
engine on the tab tick sampling a monotonic `Instant` (never
tick-counting), with `transitionend`/`animationend`,
`getComputedStyle` reflecting in-flight values, and
`Element.animate`/`getAnimations`; `prefers-color-scheme`
driven by the real system theme (`cosmic::theme::is_dark()` at
startup, live updates, `SetTheme` over IPC, restyle never
refetch); `@font-face` downloads through the existing loader
with the Stage 9 gates (never a second fetch path).
Also folded in: a `RendererProcess` poison flag — once a
renderer sends a malformed IPC message, the handle refuses all
further sends/recvs, so a frame the child queued before the
kill can never be delivered (CI run 45699261848).
Deferred from the Stage 14 research: 3D transforms /
perspective / preserve-3d; true offscreen group opacity;
fixed rounded clips stay page-space (documented
approximation); `CompositingOnly` frames skip
ResizeObserver/IntersectionObserver steps; a transform on
`<html>`/`<body>` still full-rebuilds; `getComputedStyle`
covers animatable properties only; WAAPI v1 is
array-of-objects keyframes only.

- **pluto-css & pluto-layout**: CSS Custom Properties (`var(--name)`),
  `@media` queries (viewport width/height, `prefers-color-scheme`
  synced with libcosmic's dark/light theme), `@font-face` web font
  downloading, `calc()` / `clamp()` math functions.
- **Positioning & stacking**: `position: absolute`, `fixed`, and
  `sticky`, z-index stacking contexts, CSS transform (2D/3D matrices in
  wgpu), opacity layers.
- **Animations**: CSS `transition` and `@keyframes` animations ticked
  smoothly on the wgpu frame clock.
- **Done criteria**: modern responsive layouts (Tailwind/Bootstrap sites,
  dark-mode toggles, sticky navbars, animated UI components) render and
  adapt dynamically on window resize.

### Milestone 7: Full HTML5 Spec Parser & Vector Graphics (SVG + Canvas)

Goal: eliminate parsing failures on messy real-world HTML and unlock
charts, icons, and 2D web graphics.

- **pluto-html**: full WHATWG HTML5 tree construction landed in Stage 6
  (Pluto's own tokenizer + tree builder, adoption agency, foster
  parenting, implicit tag closing, malformed markup recovery, entities);
  M7 keeps parser polish only (scripted-corpus gaps need JS, `<search>`
  special-element support).
- **Inline `<svg>` & 2D `<canvas>`**: pure-Rust vector path tessellator
  (lyon or vello / custom wgpu triangle shaders) to render inline `<svg>`
  icons/illustrations and expose the HTML5 `CanvasRenderingContext2D` API
  to rquickjs.
- **Done criteria**: pages with non-strict HTML5 markup, inline SVG icon
  sets (GitHub/Heroicons), and `<canvas>` charts or 2D browser games
  render accurately.

**Built — Stage 15 Phase 1 (2026-11-25):** namespace, layout,
fill/stroke, security — Recs 1, 2, 6, 11. The arena `Node` carries
an `ns` field (`""` = HTML namespace); `document.createElementNS`
creates namespaced nodes, `namespaceURI` is observable from JS,
and `nodeName`/`tagName` preserve case for foreign elements.
`<svg>`/`<canvas>` are replaced-element leaves (intrinsic sizing
from `width`/`height`/`viewBox` + `preserveAspectRatio`, canvas
300×150 default; children build no boxes, so SVG text cannot leak
into flow). `ComputedStyle` gains inherited `fill`/`stroke`/
`stroke-width`/`stop-color`; presentation attributes sit at
specificity 0, below every author rule. CSP inline helpers are
wired to execution: inline `<script>` and `<style>` are gated by
the document's policies (host-carried copy — the broker-mode
worker never sees the response), SVG `<script>` is inert by
namespace, not routing luck. IPC `PROTOCOL_VERSION` unchanged (9).
Painting of SVG/canvas content (paths, tessellation) is Phase 2+.

**Built — Stage 15 Phase 2 (2026-11-25):** paths, tessellation,
vector pipeline, raster-first slice, gradients — Recs 3, 4, 5, 7.
`pluto-paint::path` owns one path type (`SvgPath`) with a full
SVG path-data parser (all commands incl. arc-to-cubic) and shape
builders (rect/circle/ellipse/line/polyline/polygon); lyon 1.0
(verified in the offline registry cache — no resvg/usvg)
tessellates fills (nonzero/evenodd) and strokes (width/cap/join/
miter; dasharray renders solid, documented). `DisplayItem::
VectorPath` (flat coloured triangles, one item per shape) is the
7th wgpu pipeline — a sibling of the rect pipeline (same vertex
layout, group 0 only, so the Haswell 2-bind-group cap holds;
`PROTOCOL_VERSION` 9 → 10, variant appended last). Worker-side
CPU rasteriser (`pluto-paint::raster` + `svg`): radial / multi-stop
gradients rasterise to a coverage mask with per-pixel gradient
evaluation (per-vertex colours cannot express a radial gradient)
into an RGBA bitmap at the box's used size, re-rastered on resize;
the bitmap travels as a `Blob` via `ResourceReady` before the
submit that names it (deterministic ids, `SvgRasterCache` in the
layout actor, `ImageCache::insert_with_id` on the tab thread;
raster cap (64 MiB − 4 KiB) / 4 px so every bitmap fits one IPC
frame). Gradients are shared: CSS `linear-gradient()` /
`radial-gradient()` is a `Value::Gradient` with a computed-style
slot (one property with `url()` — setting one clears the other);
linear backgrounds are a vector quad (one strip per stop segment,
exact for any stop count — the strip is the rect polygon clipped
to the stop range, since an iso-line at `t = 0`/`1` passes exactly
through a corner and f32 rounding decided whether it survived);
radial backgrounds rasterise like SVG fills; SVG
`<linearGradient>` / `<radialGradient>` defs resolve `fill=
"url(#id)"` with `gradientTransform` and `objectBoundingBox` /
`userSpaceOnUse`. Deferred: OffscreenCanvas, full SVG `<text>`,
SMIL, filters/masks/clipPath, patterns, `<use>`/`<image>`/
`<foreignObject>`, gradient `href` inheritance, nested-`<svg>`
`viewBox`, SVG 2 geometry properties, full WAAPI on SVG
attributes, WebGL; 2D canvas (`CanvasRenderingContext2D`) is a
later M7 phase.

**Built — Stage 15 Phase 3 (2026-11-25):** 2D canvas API,
tainting policy, `<use>`/`<image>`/`<foreignObject>`, external
SVG — Recs 8, 9, 10, 12. `pluto-paint::canvas` owns the
`Canvas2d` rasteriser (bitmap + context state; paths flattened
to line segments, coverage-mask clip, per-pixel gradients,
nearest-neighbour `drawImage`, PNG `toDataURL` via the `image`
crate; raster cap 16,776,192 px, resize past it refused).
`HTMLCanvasElement` + `getContext('2d')` follow the
`imageElement` pattern: one plain-data `DomHost::canvas_call`
dispatch (host-owned bitmaps, no paint dependency in
`pluto-js`), `TestHost` implements it against the real
rasteriser via a dev-dependency. The bitmap lives worker-side:
the tab thread drains dirty canvases before each collect
(hash-compared, so an unchanged canvas is never re-sent) and
fresh bitmaps ride `ResourceReady` + `Blob` (deterministic ids
from `0x6000_0000`) before the submit that names them; scroll
re-collects carry the same payload, so canvases survive scroll.
**Tainting (Rec 9, conservative — spec-stricter):** the bitmap
carries an origin-clean flag; `drawImage` of a cross-origin
image without CORS approval taints, and `getImageData` /
`toDataURL` throw `SecurityError` when tainted, while
`putImageData` stays allowed (writes only, exposes nothing);
a canvas drawn from a tainted canvas inherits the taint. The
image pipeline records no CORS approval for `<img>`, so every
cross-origin image taints; `data:` and same-origin images are
clean (origin compared with `pluto-ipc::Origin`, the same
tainting model as `pluto-net::cors::tainting_for`). Same-
document `<use href="#id">` only (external `<use>` skipped;
cycles refused, never looped); `<svg><image>` goes through the
image pipeline and the M7 rasteriser; `<foreignObject>` gets a
real clipped viewport box. External SVG (`<img src=*.svg>`,
`<picture>` SVG `<source>`) decodes via the M7 rasteriser
after the Stage 9 fetch gates, sniffed structurally (first
element `<svg>`, never content-type trust). `PROTOCOL_VERSION`
unchanged (10). Deferred (unchanged): OffscreenCanvas, full SVG
`<text>`, SMIL, filters/masks/clipPath, patterns, SVG 2
geometry properties, full WAAPI on SVG attributes, WebGL.

### Milestone 8: Audio, Video & Real-Time Web APIs

Goal: streaming media, modern SPA routing, and live bidirectional web apps.

- **HTML5 `<audio>` & `<video>`**: pure-Rust media demuxing and decoding
  (symphonia for audio/containers + rav1d / openh264 or platform
  decoders), audio out via cpal, video frames directly to wgpu textures.
- **SPA & real-time APIs**: History API (`pushState`, `replaceState`,
  `popstate`), `MutationObserver`, `ResizeObserver`,
  `IntersectionObserver`, WebSockets (tokio-tungstenite) bridged into
  rquickjs.
- **Done criteria**: single-page applications (React/Vue/Svelte apps),
  live chat apps over WebSockets, and embedded HTML5 video/audio players
  run natively.

**Built — Stage 16 Phase 1 (2026-11-26):** media transport,
destinations, decode stack, autoplay policy — Recs 1, 2, 3, 7, 10.
`pluto-media` is the 9th crate: symphonia 0.5.5 (MP3/FLAC/Vorbis/
WAV/AAC-in-MP4), openh264 0.9.10 (H.264 only — the codec of the
overwhelming majority of real-world web video), mp4 0.14.0 (ISO
BMFF), matroska-demuxer 0.8.1 (pure Rust WebM demux). AV1 (rav1d)
is a later decoder swap, not a redesign; VP9 has no decoder in the
offline registry (only `rav1e`, an encoder) and reports
unsupported, never a fake frame; symphonia 0.5 has no Opus/ALAC.
`RendererToBrowser::VideoFrame` carries one decoded frame per
element (the shell keeps exactly one per `element_id` — a slow
shell never accumulates a backlog). `Destination` splits media
out of `Other` (`Audio` / `Video`, so `media-src` fires and media
inherits the right cap) and gains `WebSocket`. User activation is
the one handle, shared with every fresh `BindingState` (sticky
activation survives navigation); the autoplay policy defaults to
no autoplay with sound. `IntersectionObserver`'s constructor
options move to the constructor, where the spec puts them.
`PROTOCOL_VERSION` 10 → 11. Gate 1054/1054.

**Built — Stage 16 Phase 2 (2026-11-26):** media session,
audio sink, A/V clock, replaced-element leaf — Recs 4, 5, 6.
`MediaSession` is the `<video>` / `<audio>` state machine
(`pluto-js` `bindings/media.rs` drives it); the media element
is a replaced-element leaf in layout/paint with intrinsic sizing
from the decoded frame. The audio sink is cpal (null fallback
when no device exists) and the audio clock is the A/V master —
video is never scheduled by counting 16 ms tab ticks (the tick
drifts ~4% against 16.67 ms and there is no shell vsync signal).
Frames convert YUV → RGBA in the shell into the tab's live
resource table, marked dirty per blob id so the viewport
re-uploads just that texture. The phase-2 gate caught two
timing-dependent failures; the fix was deterministic clock
injection (`cfg(test)` manual clock, production untouched,
`presented > 5` threshold kept) in `session.rs` + `sink.rs`.
`PROTOCOL_VERSION` 11 → 12. Gate 1092/1092.

**Built — Stage 16 Phase 3 (2026-11-26, finisher):** History
API, same-document navigation, WebSocket transport,
`ReadableStream` + `EventSource` — Recs 8, 9, 11, 12. The phase-3
worker landed bindings without wiring and a non-compiling tree
(255 errors: a dropped `libc` dep, un-updated `CommitNavigation`
/ `DidCommit` / `WebSocketEvent` consumers); the finisher
restored the build and wired every path. The shell owns the
session history as `{url, state_json, scroll}` entries:
`pushState` / `replaceState` arrive as `HistoryPush` /
`HistoryReplace` (staged by the sync drain, forwarded by the
async drain), `history.back()` / `forward()` / `go(n)` as
`HistoryGo` (the shell moves the index, clamped, and commits
with a traversal); `CommitNavigation` carries `state_json`,
`traversal`, and the entry's `scroll` (the worker's layout
mirror and the shell's compositor must agree, or fixed / sticky
elements lay out at the wrong offset); a traversal restores the
state into the fresh realm and fires `popstate` after the commit
(`pending_popstate`, consumed in `handle_nav_outcome`);
`location.replace()` rides `nav_replace` to `DidCommit` and the
shell rewrites the current entry in place. `HistorySync`
(`{length, index}`) re-syncs the realm after every push /
replace / traversal — the shell is authoritative. A navigation
whose URL differs only in the fragment is same-document: no
fetch, no new entry — the URL updates in place, the anchor
(`NodeRect` geometry, clamped to `PageMaxScroll`) scrolls into
view, `hashchange` fires, and the entry's URL moves via a
`HistoryReplace` carrying the current state. No bfcache: Pluto
keeps none — a traversal re-fetches and rebuilds in a fresh
realm. WebSockets run on the broker (`serve_websocket`,
tokio-tungstenite, NOT the shared hyper client — no shared
client exists for the upgrade), gated by the `Destination::
WebSocket` handshake path; the shell routes `WebSocketOpen` /
`Send` / `Close` to a per-tab pump (`ws_commands`, cleaned up
by `Message::WebSocketDone`) and forwards its events as
`BrowserToRenderer::WebSocketEvent { socket_id, event }`, which
the renderer converts to `WsEvent` and delivers on the JS loop.
`ReadableStream` / `getReader` / `cancel` / `read` and
`EventSource` (retry, Last-Event-ID, CORS) ride the chunked
fetch channel (`FetchStreamEvent`). `PROTOCOL_VERSION` 12 → 13
(`CommitNavigation` fields + `WebSocketEvent.socket_id`).
Deferred: Service workers, WebRTC, MediaSource/MSE (needs Joel's
scoping decision), BroadcastChannel, `navigator.onLine`, full
WebVTT rendering, `<video>` controls UI, Fullscreen/PiP/Media
Session APIs, M5-remainder items, shell vsync signal.

### Milestone 9: Built-in Developer Tools ("Pluto DevTools")

Goal: inspect, debug, and profile websites directly inside Pluto without
relying on external browsers.

- **DOM & style inspector**: split-pane libcosmic DevTools panel showing
  the live DOM tree, computed taffy box-model overlays (margin/border/
  padding highlights in wgpu), and active CSS rules.
- **JS console & network monitor**: interactive rquickjs REPL console with
  `console.log`/`error` stack traces, plus a network waterfall showing
  pluto-net request timings, status codes, and headers.
- **Done criteria**: inspect any element on a page, toggle CSS rules live,
  view network requests, execute JS snippets in the active tab context.

**Stage 17 (M9 stages 1–2) — done 2026-11-28 (NOT the whole
milestone: inspector, matched rules / live CSS, network monitor,
dock, and structured stack frames are still open):** protocol 13
→ 14 (`DevToolsEval` / `DevToolsEvalResult` / `ConsoleLines`,
appended at the end of each enum); the console drain —
`TabJs::take_console_lines()` takes (never clones) the captured
`console` lines plus uncaught-error lines as `("error", …)`,
the tab thread forwards only new lines once per tick, and the
shell keeps the last 1,000 per tab (oldest evicted); the REPL —
`TabJs::eval_to_string()` drains microtasks first and formats
the result with `format_console_arg` on the tab thread (page
privileges only, a throw answers as the formatted exception);
a floating native console panel (hamburger menu → "DevTools")
showing the active tab's lines as untrusted plain text plus a
REPL input (`> source` echo, result line under it). Capture
runs always, so lines logged before DevTools opens are there
when it opens.

### Milestone 10: Security Hardening, Extensions & Browser Polish

Goal: prepare Pluto for daily-driver usage with strict web security,
privacy controls, and extensibility.

- **Web security model**: Same-Origin Policy (SOP), CORS preflight checks,
  Content Security Policy (CSP) headers, HTTPS-Only mode, OS-level
  process sandboxing (seccomp on Linux). CSP + the native content
  blocker are deferred from Stage 5 to M10 (gate order when they
  land: HSTS → blocker → CSP → CORS → cache → network).
- **Post-M5 (Stage 5 #16, feature-gated, only after measurement)**:
  HTTP/3, DNS-over-HTTPS, TLS 0-RTT — do not build before then.
- **Browser features**: persistent bookmarks, full-text history search,
  download manager (pluto-net streaming to disk), find-in-page (Ctrl+F),
  zoom scaling (Ctrl+ / Ctrl-).
- **Privacy & content blocking**: native Rust request/cosmetic filter
  engine (EasyList-compatible rules) inside pluto-net and pluto-css,
  built in stages over a few rounds, not all at once:
  - **Round 1 — basic (in flight, Stage 9 Rec 12)**: domain-hash
    rules first, regex last and capped; EasyList + EasyPrivacy
    defaults; `@@` exceptions, `$document` per-site disable;
    evaluated in the browser process before cache lookup; list
    updates over HTTPS with ETag, keep last-good.
  - **Round 2 — cosmetic**: element hiding driven by the lists'
    cosmetic rules, applied through pluto-css (hide matched
    elements, collapse the space they leave).
  - **Round 3 — advanced**: scriptlet injection, redirect/replace
    rules, anti-circumvention handling, per-site toggles, custom
    lists, and block stats in the UI.
  The porn-mode list (M12) is an extra list set on top of this
  engine, not a second blocker.
- **Web extensions — native Wasm only (Joel, 2026-11-28)**: no
  Chrome/Firefox WebExtension support, ever — no JSON-RPC
  serialization, no background service-worker lifecycles, no
  untrusted HTML/JS popup sandboxes. Extensions are Wasm components;
  the host is wasmtime.
  - **Why Wasm, not native libraries**: Rust has no stable ABI, so
    loading compiled .so/.dll plugins breaks across compiler
    versions. Wasm is the universal, sandboxed plugin target (the
    shape Zed and Zellij use).
  - **Runtime**: embed wasmtime (Bytecode Alliance, Rust) as the
    extension host, with fuel consumption / epoch interruption so a
    buggy extension cannot freeze a thread, plus per-extension
    memory limits.
  - **Contract (.wit)**: the browser's extension APIs are defined
    in WIT files (`wit/browser.wit`, package
    `pluto:extension@0.1.0`); `wasmtime::component::bindgen!`
    generates strongly typed Rust host traits, and extension authors
    get generated bindings in Rust, JS, Python, or C. Resource
    types give extensions opaque handles to stateful objects (tabs,
    requests) instead of raw pointers. The top-level world is
    `browser-extension`: it imports `tabs` and `ui`, and exports
    `on-load` / `on-ui-event` for the browser to call into the
    extension.
    ```wit
    package pluto:extension@0.1.0;

    interface tabs {
        record tab-info {
            id: u64,
            url: string,
            title: string,
            active: bool,
        }

        // Host functions called by the extension
        get-active: func() -> option<tab-info>;
        create: func(url: string) -> result<tab-info, string>;
    }

    interface ui {
        // Declarative UI nodes for popups/sidebars
        variant widget {
            column(list<widget>),
            row(list<widget>),
            text(string),
            button(button-props),
            text-input(input-props),
        }

        record button-props {
            id: string,
            label: string,
        }

        record input-props {
            id: string,
            placeholder: string,
            value: string,
        }

        // Extension pushes a new UI tree to the browser chrome
        render-popup: func(root: widget);
    }

    // The top-level world every extension targets
    world browser-extension {
        import tabs;
        import ui;

        // Guest functions exported by the extension for the browser to call
        export on-load: func();
        export on-ui-event: func(component-id: string, event-type: string, payload: string);
    }
    ```
  - **Host setup**: one `wasmtime::Store` per extension holding its
    `ExtensionState` (extension id, `WasiCtx`, `ResourceTable`, and
    an `mpsc::Sender<BrowserCommand>` back to the browser/UI
    thread). `bindgen!` generates the `tabs::Host` / `ui::Host`
    traits; the host implements them by sending commands (e.g.
    `CreateTab`, `UpdateExtensionUi`) over the channel, never by
    sharing mutable state.
  - **Capability-based security (WASI)**: extensions run with zero
    ambient authority — no filesystem, network, or OS access by
    default. The extension declares permissions in its
    `manifest.toml` (e.g. storage, tabs, http-request); the host
    links only the approved interfaces into that extension's
    `Store`, and builds a locked-down `WasiCtx` (no
    filesystem/network unless granted).
  - **Execution contexts**: background workers are long-lived or
    event-driven Wasm instances on a dedicated async runtime
    (tokio), talking to the browser core over typed message
    channels (`tokio::sync::mpsc`). Content scripts get DOM access
    two ways: a constrained, WIT-defined DOM/CSS selector API
    straight from Pluto's own layout/DOM tree (fast, no JS
    serialization), or Wasm injected alongside the page's rquickjs
    engine via wasm-bindgen bindings.
  - **Extension UI — declarative native widgets, not web views**:
    when the user clicks the extension icon, the host calls the
    extension's `on-popup-open`; the extension builds a `Widget`
    tree and pushes it with `ui::render-popup`; the host maps the
    tree to native libcosmic/iced widgets in `view()` and dispatches
    user events back through `on-ui-event` (refilling fuel first),
    so the extension can re-render with updated state. Native over
    web views because: instant startup (<1 ms, just an enum tree
    across the boundary, vs 20–100 ms to spin up a DOM/CSS/JS
    pipeline per popup), negligible memory (shares the main window's
    wgpu surface and font cache, vs megabytes per popup), and full
    visual cohesion (inherits theme, dark/light mode, spacing,
    typography) — at the cost of layout flexibility (only the widget
    primitives in the .wit file). Escape hatch for custom graphics:
    a `Canvas(width, height, list<u8>)` or `Svg(string)` widget
    variant pushing raw RGBA buffers or vector paths straight to a
    wgpu texture — no full web view needed.
  - **Packaging & distribution**: one signed archive holding a
    single .wasm component binary plus `manifest.toml` (or the
    manifest embedded as a custom Wasm section). Extensions compile
    to `wasm32-wasip2`, so one binary runs identically on Linux,
    macOS, and Windows.
  - **Guest SDK (`pluto-ext-sdk`)**: extension authors never touch
    raw WIT bindings. The SDK crate bundles the WIT file and runs
    `wit_bindgen::generate!` privately inside itself; per-instance
    state lives in a thread-local `RefCell` (Wasm components are
    single-threaded per instance, so zero locking overhead); a
    `#[pluto_ext::main]` procedural macro (or a
    `register_extension!` declarative macro) implements the raw WIT
    `Guest` trait and forwards calls to the developer's struct,
    which implements an Elm-inspired `Extension` trait (`on_load`,
    `on_event`, `view`, `on_ui_event`). After `on_ui_event` or
    `on_event` finishes, the SDK glue automatically calls `view()`
    and pushes the updated `Widget` tree across the boundary via
    `ui::render_popup` — authors never call render functions
    themselves. What an author writes:
    ```rust
    use pluto_ext_sdk::prelude::*;

    #[derive(Default)]
    struct WordCounter {
        count: usize,
    }

    impl Extension for WordCounter {
        fn on_load(&mut self) {
            // Subscribe to page navigation events
            events::subscribe(EventTopic::PageLoaded);
        }

        fn on_event(&mut self, event: BrowserEvent) {
            if let BrowserEvent::PageLoaded { tab_id } = event {
                if let Some(doc) = dom::document(tab_id) {
                    let text = doc.query_selector("body")
                        .map(|el| el.inner_text())
                        .unwrap_or_default();
                    self.count = text.split_whitespace().count();
                }
            }
        }

        fn view(&self) -> Widget {
            Column::new()
                .push(Text::new(format!("Page word count: {}", self.count)))
                .push(Button::new("reset", "Reset Counter"))
                .into()
        }

        fn on_ui_event(&mut self, id: &str, _kind: UiEventKind) {
            if id == "reset" {
                self.count = 0;
            }
        }
    }

    // Expands to the raw WIT `Guest` implementation + `wit_bindgen::export!`
    pluto_ext_sdk::register_extension!(WordCounter);
    ```
  - **Page / DOM interaction (`wit/dom.wit`)**: never serialize a
    whole DOM tree across the boundary — use WIT resource types.
    A resource is an opaque handle (a u32 index in the host's
    `wasmtime::component::ResourceTable`); the DOM nodes stay in
    the browser engine's memory and the extension only holds
    handles and invokes methods on them.
    ```wit
    interface dom {
        // Opaque handles to live host-side DOM objects
        resource document {
            query-selector: func(selector: string) -> option<element>;
            query-selector-all: func(selector: string) -> list<element>;
            create-element: func(tag: string) -> element;
        }

        resource element {
            tag-name: func() -> string;
            get-attribute: func(name: string) -> option<string>;
            set-attribute: func(name: string, value: string);
            inner-text: func() -> string;
            set-inner-text: func(text: string);
            append-child: func(child: borrow<element>);
            remove: func();
        }

        // Entry point gated by activeTab or host permissions
        get-document: func(tab-id: u64) -> result<document, string>;
    }
    ```
    Host-side safety: stale-handle protection — when a user
    navigates away or a node is collected by the layout engine,
    the host removes/invalidates the `ResourceTable` entry, and the
    host resource struct holds a `Weak` pointer or generational
    arena key (`slotmap::DefaultKey`), so calling a method on a
    dead node safely returns a no-op or error instead of crashing.
    Automatic cleanup — when the extension drops an `Element`
    handle, the Component Model invokes `HostElement::drop` on the
    host and frees the handle. Batched mutations — `set_inner_text`
    / `append_child` queue onto the tab's layout/style invalidation
    pipeline, so taffy layout and the wgpu render pass recalculate
    only once per frame.
  - **Extension lifecycle & storage**: extensions are event-driven
    actors, not always-on processes, so startup stays fast and idle
    memory stays low.
    - **Per-extension key-value storage (`wit/storage.wit`)**: no
      direct WASI filesystem access for settings (path traversal
      risk, cluttered directories) — a dedicated, quota-enforced
      interface backed by an embedded Rust database (redb, sled,
      or a SQLite table partitioned by `extension_id`).
      ```wit
      interface storage {
          get: func(key: string) -> result<option<list<u8>>, string>;
          set: func(key: string, value: list<u8>) -> result<_, string>;
          remove: func(key: string) -> result<_, string>;
          clear: func() -> result<_, string>;
      }
      ```
      Quota enforced in `storage::Host` (track bytes per
      `extension_id`, e.g. 5 MB cap for standard extensions)
      before writing to disk. The SDK wraps get/set with generic
      `storage::get_json<T>()` / `storage::set_json<T>()` using
      serde, so developers store typed Rust structs directly.
    - **Event subscriptions & wake-up lifecycle**: no polling
      loops. Extensions declare static wake-up triggers in
      `manifest.toml` (e.g. `events = ["page-loaded",
      "url-match:https://github.com*"]`) and can also call
      `events::subscribe(...)` at runtime. The host keeps a central
      `HashMap<EventTopic, Vec<ExtensionId>>` dispatcher; when a
      tab finishes loading or a network request fires it checks
      which extensions subscribe to that topic (and that their
      manifest permissions allow that URL/domain), then dispatches
      `bindings.call_on_event` if the extension's Store is warm.
      Hibernation: an extension with no event in 60 seconds and no
      open popup has its `wasmtime::Store` dropped to reclaim RAM;
      only its compiled `wasmtime::Component` stays cached in
      memory, and re-instantiating a pre-compiled component takes
      ~50 microseconds, so it is rebuilt on demand when a
      subscribed event fires.
  - **Packaging & manifest schema (`.plutoext`)**: distribute
    extensions as a single-file signed archive — a deterministic
    `tar.zst` (Zstandard-compressed tarball) or ZIP containing
    `manifest.toml`, `extension.wasm`, static assets (icons), and a
    detached cryptographic signature. `manifest.toml` uses strict
    serde deserialization (`#[serde(deny_unknown_fields)]`) so
    malformed manifests fail fast at install time.
    ```toml
    [extension]
    id = "org.example.word-counter" # Reverse-DNS unique identifier
    name = "Word Counter"
    version = "0.1.0"
    sdk_version = "0.1"              # Checked against host WIT compatibility
    entry = "extension.wasm"

    [permissions]
    # Coarse capabilities linked into the Wasmtime Linker
    capabilities = ["tabs", "storage", "dom"]

    # Fine-grained URL patterns (enforced by the host before dispatching events/DOM handles)
    hosts = [
        "https://*.wikipedia.org/*",
        "https://github.com*"
    ]

    # Declarative wake-up triggers (allows cold-booting hibernated extensions)
    [triggers]
    events = ["page-loaded", "toolbar-click"]
    network_filter = true
    ```
    Permission prompts & runtime enforcement, in two layers:
    install-time grants and runtime checks. Linker-level gating —
    if `"tabs"` is not in capabilities,
    `pluto::extension::tabs::add_to_linker` is never called, so the
    extension literally cannot import or invoke tab functions.
    Host-pattern guard — even with `"dom"` linked,
    `dom::get-document(tab_id)` checks the tab's current URL
    against the user-approved hosts glob list (e.g. the `globset`
    crate) before returning a `Document` handle. Install prompts
    group raw capabilities into plain-English risk tiers (e.g.
    `"dom"` + `hosts = ["<all_urls>"]` becomes "Read and modify
    data on all websites"). Cryptographic signing (ed25519-dalek):
    compute a BLAKE3 hash over the canonical concatenation of
    `manifest.toml` + `extension.wasm` + `assets/`; store
    `signature.ed25519` (64-byte signature + 32-byte publisher
    public key) inside the bundle; verify the signature at load,
    before `wasmtime::Component::from_binary`, against either a
    trusted extension-store public key or a user-pinned developer
    key (sideloaded extensions).
  - **Network interception — zero-stall pipeline**: never call into
    a Wasm sandbox synchronously on every HTTP request — two tiers.
    Tier 1, declarative rule engine (zero Wasm overhead): for
    ad-blockers and tracker blockers, extensions register static
    filter rules (EasyList/uBlock-style) via a WIT call on startup;
    the network stack (hyper + rustls) evaluates them natively in
    Rust (compiled Aho-Corasick / regex automata) in nanoseconds
    without waking Wasm. Tier 2, dynamic Wasm interception
    (`wit/network.wit`): when an extension must rewrite headers,
    redirect URLs, or inspect payloads, an async `on-request` hook
    with strict timeouts.
    ```wit
    interface network {
        record request-meta {
            id: u64,
            url: string,
            method: string,
            resource-type: string, // "document", "script", "image", "fetch"
            headers: list<tuple<string, string>>,
        }

        variant intercept-action {
            allow,
            block,
            redirect(string),
            modify-headers(list<tuple<string, string>>),
        }

        // Extension registers URL patterns it wants to dynamically inspect
        register-interceptor: func(url-patterns: list<string>) -> result<_, string>;
    }
    ```
    Preventing page-load stalls: pre-filtering — only invoke Wasm
    if the request URL matches the registered `url-patterns` and
    the extension's manifest `hosts` permissions. Strict fuel +
    wall-clock timeout — wrap `bindings.call_on_request(...)` in a
    15 ms `tokio::time::timeout` alongside a tight wasmtime fuel
    cap; if an extension hangs or exceeds 15 ms, cancel the future
    and fall back to `InterceptAction::Allow` (or a user-configured
    fail-closed mode) so browsing stays snappy.
  - **Cross-extension & web-page messaging**: Wasm components are
    completely memory-isolated from web pages and from each other,
    so all communication flows through structured,
    capability-checked message brokers on the host. Web page ↔
    extension bridge (`window.pluto`): pages are untrusted and must
    never access extension APIs directly. When a tab loads a URL
    matching an extension's `hosts` permission and the extension
    declares `page_bridge = true`, rquickjs injects a frozen global
    helper into the page context
    (`window.pluto.sendMessage("org.example.wallet",
    { type: "CONNECT_REQUEST" })`); the host serializes the JS
    object into a safe intermediate format (JSON or CBOR — never raw
    JS object references), enforces a max payload size (256 KB),
    and passes `(sender_origin, payload_bytes)` into the
    extension's exported `on-page-message` WIT function; the
    extension explicitly checks `sender_origin` before returning a
    response. Cross-extension messaging (`wit/ipc.wit`): extensions
    expose modular services to each other (e.g. a password manager
    providing credentials to a form-filler) without tight coupling.
    ```wit
    interface ipc {
        record message-envelope {
            sender-id: string,
            topic: string,
            payload: list<u8>,
        }

        // Send a request to another installed extension and await its reply
        request: func(target-ext-id: string, topic: string, payload: list<u8>) -> result<list<u8>, string>;
    }
    ```
    Allow-list security: `manifest.toml` specifies
    `accept_ipc_from = ["org.example.trusted-ext"]` (or `"*"` for
    public service extensions); the host's `ipc::Host::request`
    checks the allow-list before routing the bytes across
    `tokio::sync::oneshot` channels between the two extension
    actors.
  - **Workspace scaffolding (`pluto-extensions/`)**: a separate
    multi-crate workspace so the shared .wit contract is the single
    source of truth for host and guests.
    ```
    pluto-extensions/
    ├── Cargo.toml                  # Workspace root
    ├── wit/
    │   └── browser.wit             # Shared WIT contract
    ├── crates/
    │   ├── pluto-ext-host/         # Embedded into your browser binary (Wasmtime + ResourceTable)
    │   ├── pluto-ext-sdk/          # Published crate for extension authors (wit-bindgen + macros)
    │   └── pluto-ext-cli/          # CLI tool (`cargo install pluto-ext`)
    └── examples/
        └── word-counter/           # Sample extension compiling to `wasm32-wasip2`
            ├── Cargo.toml
            ├── manifest.toml
            └── src/lib.rs
    ```
    Root `Cargo.toml`: workspace resolver 2, members
    `crates/pluto-ext-host`, `crates/pluto-ext-sdk`,
    `crates/pluto-ext-cli`, `examples/word-counter`; workspace
    dependencies wasmtime 27.0 (`component-model`, `async`),
    wasmtime-wasi 27.0, wit-bindgen 0.36, tokio 1.41 (`full`),
    serde 1.0 (`derive`), toml 0.8, ed25519-dalek 2.1
    (`rand_core`), blake3 1.5. The SDK keeps raw
    `wit_bindgen::generate!` private and exposes the `Extension`
    trait plus the `register_extension!` macro (thread-local
    `RefCell` instance, `Guest` impl forwarding `on_load` /
    `on_ui_event` with automatic `render_popup` re-render). Example
    extension crates compile as `cdylib` so rustc outputs a Wasm
    component for `wasm32-wasip2`:
    ```toml
    [package]
    name = "word-counter"
    version = "0.1.0"
    edition = "2021"

    [lib]
    crate-type = ["cdylib"]

    [dependencies]
    pluto-ext-sdk = { path = "../../crates/pluto-ext-sdk" }
    ```
  - **Developer tooling (`pluto-ext` CLI)**: one CLI for
    scaffolding, cross-compilation, signing, and live reloading.
    `pluto-ext new <name>` generates a ready-to-build folder
    (`Cargo.toml` with `crate-type = ["cdylib"]`,
    `.cargo/config.toml` defaulting `build.target =
    "wasm32-wasip2"`, `manifest.toml`, starter `src/lib.rs`).
    `pluto-ext build --release` runs `cargo build --target
    wasm32-wasip2 --release`, validates `manifest.toml` against the
    strict schema, and strips debug symbols (`wasm-tools strip`).
    `pluto-ext pack --key developer.pem` hashes manifest +
    extension.wasm with blake3, signs with ed25519-dalek, writes
    `signature.ed25519`, and bundles deterministically into
    `dist/<id>-<version>.plutoext` (tar.zst). `pluto-ext dev`
    (hot-reload) watches `src/` and `manifest.toml` via the `notify`
    crate, recompiles on save, and sends a local IPC message over a
    Unix domain socket (`~/.pluto/dev.sock`) to the running browser;
    state-preserving reload — before swapping the
    `wasmtime::Component`, the host calls an optional
    `serialize-state: func() -> list<u8>` export on the old
    instance, instantiates the new binary, and passes the bytes into
    `restore-state: func(data: list<u8>)`, so popup UI state
    survives code edits.
  - **Security hardening & threat model**: Wasm's memory sandbox
    alone does not cover denial-of-service, spoofing, or
    side-channel vectors from untrusted extensions.
    - **Memory exhaustion** (extension allocates `Vec<u8>` in a
      loop): implement `wasmtime::ResourceLimiter` on every `Store`
      to cap linear memory (32 MB per extension) and table growth;
      `memory_growing` returning false triggers a clean Wasm trap
      instead of an OS OOM kill.
    - **CPU starvation / infinite loops** (`loop {}` inside
      `on-request` or `on-load`, hanging a Tokio worker): enable
      `config.epoch_interruption(true)` alongside
      `consume_fuel(true)`; a lightweight background timer thread
      increments the engine epoch every 10 ms, and each `Store`
      gets an epoch deadline (2 ticks = 20 ms max slice).
    - **UI spoofing / clickjacking** (a popup mimicking the
      browser's master password prompt or URL bar): render extension
      popups strictly inside a sandboxed, visually distinct popover
      container with a non-removable header bar displaying the
      extension's verified name, icon, and publisher badge; never
      allow extensions to position floating overlays outside their
      assigned viewport bounds.
    - **Timing side-channels (Spectre / cache)** (high-resolution
      timers measuring cross-origin network or memory latency):
      provide a coarsened, jittered clock in the WASI time provider
      (100 microsecond resolution + random jitter) unless the user
      explicitly grants a high-precision-timing permission.
    - **Resource handle flooding**
      (`dom::query_selector_all("*")` in a loop without dropping
      handles): cap the maximum active entries in each extension's
      `ResourceTable` (4,096 live handles); reject further handle
      allocations with `Err("Resource limit exceeded")` until the
      guest drops existing handles.
  - **Reference extension — web archiver ("Archive Vault")
    (Joel, 2026-11-28)**: a first-party reference extension for the
    native Wasm extension system above, paired with a self-hosted
    companion server. Unlike a traditional crawler that re-fetches a
    URL from scratch (missing paywalled, authenticated, or
    JS-rendered content), the extension captures the live, post-JS
    DOM directly from the browser — the exact state of the page on
    screen — runs the "fat reducer" optimizer in Wasm, and streams
    the compact snapshot to the Vault for permanent storage. It
    depends on the extensions host (pluto-ext-host + the
    `page-snapshot` WIT interface below), so it lands after the
    extension system itself. Design only — no implementation yet.
    - **Extension UI (pluto-ext-sdk, native libcosmic popover)**:
      the Archive Page toolbar icon opens a native popover with
      presets and a live size estimator. Presets (row of buttons):
      Lossless (100%) — keeps original PNG/JPEG/WebP files, full
      CSS, and raw HTML (legal/reference receipts); Balanced (75%)
      — minifies HTML/CSS, strips unused CSS rules, strips EXIF,
      re-encodes images to WebP at quality 75 (cuts page size
      ~60–70% with no visible quality loss); Ultra-Slim /
      Text-First (30%) — downscales images to 800px max width,
      quality 30, strips web fonts (`@font-face`), removes
      `<video>`/`<iframe>` embeds (thousands of articles on low
      storage). Image quality slider (10–100) fine-tunes
      compression and switches the active badge to "Custom".
      Feature toggles: strip unused CSS (selectors not matching
      the current DOM), remove tracking scripts & ads (strips
      `<script>` tags before saving, so archived pages are static,
      safe, and instant to load), embed assets inline (packs
      everything into a single standalone `.html` file with `data:`
      URIs, versus sending separate deduplicated blobs to the
      server). Action button "Save Snapshot" with a live preview
      badge (e.g. `Estimated: 4.8 MB → 620 KB`).
    - **"Fat reducer" & optimization pipeline (inside Wasm,
      `wasm32-wasip2`)**: pure-Rust passes run in memory before
      upload. (1) Live DOM snapshot (`wit/dom.wit`): ask the host
      to serialize the active tab's live DOM — dynamic JS-rendered
      content and computed input states included — along with the
      already-cached bytes of its stylesheets and images, avoiding
      duplicate network downloads; strip all `<script>` tags,
      `on*` event attributes, and `<link rel="preload">` hints so
      the archived page is 100% static and cannot phone home
      later. (2) CSS tree-shaking & minification (lightningcss):
      parse all `<style>` blocks and external `.css` files, filter
      out `@media print`, unused `@font-face` rules (if toggled),
      and selectors that match no element in the captured DOM, then
      minify whitespace and hex colors. (3) HTML minification
      (minify-html): collapse whitespace, strip HTML comments,
      remove redundant quotes/closing tags. (4) Image fat reducer
      (`image` + `webp` / `ravif`): for every `<img>` and CSS
      background-image — decode the raster bytes, resize if width
      > max_width (1600px in Balanced, 800px in Ultra-Slim),
      re-encode to WebP (fast) or AVIF (smallest) at the slider's
      quality, keep whichever is smaller
      (`compressed.len() < original.len()`), and rewrite the
      `<img src="...">` attribute to point at the optimized blob.
    - **Self-hosted Rust backend ("Archive Vault")**: a
      lightweight single-binary server (axum + tokio) for a home
      server, NAS, or VPS. Content-addressed blob store
      (cross-page deduplication): hash every image and stylesheet
      with BLAKE3, store assets on disk as
      `blobs/<blake3_prefix>/<blake3_hash>.zst` — 50 archived pages
      from the same blog share one copy of the site logo, fonts,
      or CSS framework. Metadata & full-text search: SQLite (sqlx
      or rusqlite) stores snapshot records (id, url, title,
      captured_at, preset_used, original_bytes, saved_bytes,
      html_blob_hash); Tantivy (pure-Rust Lucene alternative)
      indexes the extracted plain text of every archived page for
      millisecond search across the whole offline archive, with
      fuzzy matching and snippet highlighting. Built-in replay
      viewer: serves archived pages back at
      `http://vault.local/<snapshot_id>` with a strict
      Content-Security-Policy (`default-src 'none'; img-src 'self'
      data:; style-src 'self' 'unsafe-inline';`) — archived pages
      can never execute rogue scripts or leak requests to the live
      web.
    - **Host WIT additions (`wit/archive.wit`)**: one
      resource-reading interface on pluto-ext-host, so the
      extension can grab already-downloaded subresources from the
      browser's memory cache without re-fetching them over the
      network.
      ```wit
      interface page-snapshot {
          record captured-asset {
              url: string,
              mime-type: string,
              data: list<u8>,
          }

          record raw-snapshot {
              url: string,
              title: string,
              serialized-html: string,
              assets: list<captured-asset>,
          }

          // Grabs live DOM HTML + cached images/CSS from the active tab
          capture-tab: func(tab-id: u64) -> result<raw-snapshot, string>;
      }
      ```
    - **Staged build order** (three parts, in dependency order):
      (1) the browser host `capture-tab` implementation (serialize
      the live DOM tree, pull cached assets from memory); (2) the
      pluto-ext archiver extension (quality-slider UI,
      lightningcss/minify-html passes, image compressor); (3) the
      self-hosted axum archive server (BLAKE3 deduplicated blob
      store, SQLite schema, Tantivy full-text search).
- **Done criteria**: Pluto safely isolates cross-origin sites, blocks
  trackers natively, manages downloads/bookmarks, passes core security
  & Web Platform Test (WPT) suites, and loads a signed .wasm
  extension in a fuel-limited, permission-gated sandbox whose popup
  renders as native widgets.

### Milestone 11: Built-in WireGuard VPN (Joel, 2026-11-24)

Goal: a VPN inside the browser, no separate app — one toggle, all of
Pluto's traffic through the tunnel.

- **New crate `pluto-vpn`**: userspace WireGuard (boringtun is the
  likely shape — pure Rust, no Chromium/Firefox/WebKit code; kernel
  WireGuard via netlink is the alternative to settle at research
  time). The tunnel is a transport under pluto-net, not a second
  network stack: when the VPN is on, every request and every DNS
  lookup goes through it.
- **Keys & endpoints**: per-profile keypair, stored in the profile
  dir (0600, OS keyring when one is available). Servers and keys are
  Joel-supplied and never committed to the repo.
- **Fail closed**: if the tunnel drops, traffic stops — nothing
  leaks out direct. No kill-switch bypass, no "reconnect in the
  clear".
- **UI**: a toggle in the bottom-right corner of the loading status
  navbar (the status bar at the bottom of the window), showing
  on / off / connecting, plus a server picker.
- **Research first**: a research stage on the VPN design (boringtun
  vs kernel, kill-switch shape, DNS-leak tests, key storage) before
  application, following the usual research-then-apply pattern.
- **Done criteria**: toggle on → all traffic and DNS through the
  tunnel and the exit IP changes; toggle off → direct; tunnel
  drop → no traffic at all until it is back or the toggle is off.

### Milestone 12: Porn Mode (Joel, 2026-11-24)

Goal: the private mode — it replaces incognito/private mode
entirely; there is no separate private mode. Security and privacy
are the mode's core features, not add-ons.

- **Activation relaunches the browser**: porn mode restarts Pluto
  with the M11 VPN turned on and an ephemeral profile — no history
  saved, no cookies/cache written to disk, all storage in-memory
  and wiped on exit (depends on M5 storage work).
- **Auto-mute**: every video is muted by default the moment it
  loads (depends on the M8 media pipeline).
- **Ad blocking on, special list**: the M10 blocker runs with an
  extra porn ad-blocking list on top of the normal lists.
- **Panic key combo** (exact keys TBD by Joel): instantly swaps the
  viewport to a fake work screen — a fake spreadsheet, or a fake
  Zoom call (a video of a Zoom conference call; the video variant
  depends on M8 playback, a static mock comes first).
- **Own homepage**: in porn mode the homepage is a grid of cards
  featuring the top 20 adult sites to visit. The 20 sites are
  Joel's list (TBD by Joel — not picked here).
- **Titlebar**: while porn mode is on, the browser reports as
  "PornBrowser" in the titlebar; switching porn mode off restores
  "Pluto".
- **Research first**: a research stage once M11 lands (VPN shape,
  ephemeral-profile shape, panic-screen shape), before application.
- **Done criteria**: relaunch → VPN on, zero history/cookies/cache
  on disk after exit, videos muted, porn list active, panic combo
  shows the fake work screen, homepage is the 20-site grid,
  titlebar reads "PornBrowser" until the mode is switched off.
