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
| 2. Networking | hyper + rustls + tokio | `ResourceLoader` in `pluto-net`: fetches HTML, CSS, JS, and images over TLS 1.3 (rustls + webpki-roots), enforces CORS/redirects, and streams bytes to the parser. |
| 3. Parsing & DOM | winnow + slotmap | winnow for fast, zero-copy CSS tokenization and HTML5 state-machine parsing into an arena-allocated DOM tree (`NodeId` handles instead of `Rc<RefCell<T>>` cycles). Lives in `pluto-html` / `pluto-css`. |
| 4. Layout & Text | taffy + cosmic-text | `taffy::LayoutPartialTree` over the styled DOM nodes. For text leaf nodes, pass `cosmic_text::Buffer` shaping results inside Taffy's `measure_function` so line wrapping and intrinsic sizes match pixel-for-pixel. Lives in `pluto-layout`. |
| 5. GPU Rendering | wgpu + cosmic-text (swash) | Walk the laid-out Taffy tree in paint order (z-index / stacking contexts) to emit quads (backgrounds, borders, images) and glyph atlases (via cosmic-text's `SwashCache`) into a wgpu render pass. Lives in `pluto-paint`; uploaded in `viewport.rs::Primitive::prepare`, drawn in `Primitive::render`. |
| 6. Scripting | rquickjs | `rquickjs::AsyncRuntime` on the tab's event loop, exposing `NodeId`-backed host objects (`document.getElementById`, `addEventListener`, `fetch`) that mutate the DOM and trigger re-style/re-layout. Lives in `pluto-js`. |

## Data flow (vertical slice)

```
Go pressed
  -> pluto-net ResourceLoader::fetch(url)          (bytes)
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

## Roadmap (Joel, 2026-11-23 — replaces the scratched vertical-slice plan)

### Milestone 1: End-to-End Static Page Pipeline ("Hello, example.com")

Goal: type a URL into libcosmic, fetch it over HTTPS, parse basic
HTML/CSS, and draw colored boxes and text on the GPU.

- **pluto-net**: async HTTP/1.1 + HTTP/2 client (hyper + rustls +
  webpki-roots) with redirect following (301/302/307/308).
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
- **Done criteria**: pages with photos, icons, and multiple open tabs
  render cleanly without blocking the UI thread during image decoding.

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

### Milestone 5: HTML Forms, Storage & Process Isolation

Goal: log into websites, persistent state, crash-safe tab isolation.

- **Forms & input controls**: `<input>`, `<textarea>`, `<button>`,
  `<form>` submission (GET/POST), text cursor caret rendering,
  selection, keyboard focus management.
- **Web storage & cookies**: Cookie jar persistence (Set-Cookie / Cookie
  headers), localStorage, sessionStorage backed by disk storage.
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

- **pluto-html**: upgrade the winnow parser to full WHATWG HTML5
  tree-construction rules (adoption agency algorithm, foster parenting,
  implicit tag closing, malformed markup recovery) plus HTML entities.
- **Inline `<svg>` & 2D `<canvas>`**: pure-Rust vector path tessellator
  (lyon or vello / custom wgpu triangle shaders) to render inline `<svg>`
  icons/illustrations and expose the HTML5 `CanvasRenderingContext2D` API
  to rquickjs.
- **Done criteria**: pages with non-strict HTML5 markup, inline SVG icon
  sets (GitHub/Heroicons), and `<canvas>` charts or 2D browser games
  render accurately.

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

### Milestone 10: Security Hardening, Extensions & Browser Polish

Goal: prepare Pluto for daily-driver usage with strict web security,
privacy controls, and extensibility.

- **Web security model**: Same-Origin Policy (SOP), CORS preflight checks,
  Content Security Policy (CSP) headers, HTTPS-Only mode, OS-level
  process sandboxing (seccomp on Linux).
- **Browser features**: persistent bookmarks, full-text history search,
  download manager (pluto-net streaming to disk), find-in-page (Ctrl+F),
  zoom scaling (Ctrl+ / Ctrl-).
- **Privacy & content blocking**: native Rust request/cosmetic filter
  engine (EasyList-compatible rules) inside pluto-net and pluto-css.
- **Done criteria**: Pluto safely isolates cross-origin sites, blocks
  trackers natively, manages downloads/bookmarks, and passes core security
  & Web Platform Test (WPT) suites.
