# Pluto

Pluto is a web browser written from scratch in Rust. No Mozilla code, no Google code, no Apple code — no Servo, no Chromium, no WebKit. Every line of the engine is written originally for Pluto: its own HTML tokenizer and WHATWG tree builder, its own CSS parser and cascade, its own layout, paint, networking, JavaScript bindings, and IPC.
Goals
built in mind for the cosmic desktop.
 - **Fast.** Pages render through a wgpu display list with glyph and image atlases, style and layout run in parallel where the design allows, and the network layer pools connections per origin. The bar for "done" on real sites is 60+ FPS scrolling and clicking.
-   **Robust.** The parser recovers from malformed HTML the way the WHATWG spec says to (it passes 99.6% of the html5lib corpus); a failed fetch renders an error page, never a blank window; a renderer that dies must not take the browser with it.
-   **Rust, end to end.** Memory safety from the language, not from a sandbox bolted on afterwards — and then a sandbox anyway (see Security below).
-   **Independent.** Pluto shares no code with Google, Apple, or Mozilla projects. It builds on Rust libraries (hyper, rustls, rquickjs, wgpu, taffy, cosmic-text, libcosmic) and on web standards text — never on another browser's code.     
```sh
git clone https://github.com/infektedpc/pluto.git
cd pluto
cargo run -p pluto-browser
```

The window opens with a tab strip (one tab per page, favicon +
title, Ctrl+T / Ctrl+W / Ctrl+Tab), a nav bar (Back / Forward /
Reload / URL / Go, bookmark star, menu), a thin progress bar under
the nav bar, a status line, and a viewport that loads
https://example.com on startup —
grey page, white centered box, "Example Domain" heading, paragraph, and
link. The status line reads `Loaded https://example.com` when the page
is up. Type a URL and press Go to navigate (a bare host works:
`example.com` becomes `https://example.com`). A failed fetch renders
an error page instead of a blank window. The window title follows the
active tab's page title. Bookmarks and history persist across
restarts (`bookmarks.json`, `history.json` in the profile dir).

The window frame follows the desktop: the window manager draws it
where it provides one (KDE, Xfce, sway...), and libcosmic draws it
where it doesn't (GNOME). Only one frame is ever drawn — with the
window manager's frame, libcosmic's own header bar stays hidden.
Override it from the command line —
`--csd` forces the libcosmic frame, `--ssd` forces the window
manager's (note the `--` separating cargo's args from Pluto's):

```sh
cargo run -p pluto-browser -- --csd
```

## Security

Each tab runs its renderer in a separate process under a seccomp
sandbox (best-effort; `--no-sandbox` escapes it), so a renderer
that dies doesn't take the browser with it. Requests go through a
real Origin type with CORS, a partitioned cookie jar (SameSite Lax
by default, `__Secure-`/`__Host-` prefixes enforced), HSTS with
HTTPS-only upgrades, CSP v1, subresource integrity checks, and a
basic EasyList/EasyPrivacy blocker. Details in `ARCHITECTURE.md`.

Build notes:

- Toolchain: stable Rust 1.99+ (`cargo` on `PATH`; install via
  https://rustup.rs if it's missing). First build downloads the
  dependency tree including the libcosmic git pin, so allow a few
  minutes and a network connection.
- System libraries: the libcosmic/Wayland/X11 stack. If a build error
  names a missing system library, that's the one to install.
- No libclang needed: `pluto-js` uses rquickjs with pre-generated
  bindings (the `bindgen` feature was dropped during bootstrap).
- No gstreamer needed in M1: there is no media yet.
- Clone is plain git — one repo, no submodules:
  ```sh
  git clone https://github.com/infektedpc/pluto.git
  cd pluto
  ```

## Version notes (read before "upgrading" anything)

The spec this started from named wgpu 24, cosmic-text 0.12, and taffy 0.7,
and a bare `https://github.com` libcosmic URL. All three were corrected
during bootstrap, verified against the actual sources:

- **libcosmic** is pinned to `pop-os/libcosmic` rev
  `ecc5fe4164e1c9024b932382d24ea41f24147e75` — the pin already cached
  in this VM, so the first build does not re-fetch it. Unpinning to latest main is a one-line change in the root
  `Cargo.toml`, but it moves wgpu/cosmic-text/taffy under you.
- **wgpu 28, cosmic-text 0.19, taffy 0.15**: what that libcosmic pin's
  iced actually uses (verified against the pin's lockfile: 28.0.0 /
  0.19.0 / 0.15.1). Older majors would pull second copies of each crate,
  and the `wgpu::Device` from libcosmic would no longer be the same type
  `pluto-paint` sees.
- **Shader widget path**: `cosmic::widget::shader` and
  `cosmic::iced_wgpu::wgpu` do not exist in this pin. The real paths are
  `cosmic::iced::widget::Shader` and `cosmic::iced::wgpu` — fixed in
  `crates/pluto-browser/src/viewport.rs` and `main.rs`.
- **Window size**: this libcosmic pin has no `Settings::size()` builder
  (field is `pub(crate)`), so the window opens at the default 1024x768
  instead of 1280x800.

## Milestones

- **M1 — done.** "Hello, example.com": fetch + render through the
  worker/renderer split.
- **M2 — done.** Real-world CSS: full selectors and specificity, UA
  stylesheet, external `<link>` CSS, grid, border-radius, box-shadow,
  overflow/scroll, hit-testing, clickable links, Back/Forward history —
  plus a partitioned HTTP cache, `:visited`, persisted history, and a
  frame-time histogram for the 60 FPS bar.
- **M3 — done.** Images and the multi-tab shell: PNG/JPEG/WebP/GIF
  decode, intrinsic sizing, `background-image`, `srcset`/`<picture>`
  selection, favicons, page titles, loading progress, preconnect +
  DNS cache, animated images, strict decode limits.
- **M4 — done.** JavaScript leftovers finished: the module loader,
  `-Os` for the QuickJS C build, layout-kind measurement with
  safe-root widening, image map into the incremental path, subtree
  restyle, the binding batch (`location`, `URL`, `dataset`,
  `cloneNode`, typed events, fetch surface, `onerror`, console),
  and the production shell wiring for all of it.
- **M5 — done.** Forms, cookies + storage on disk, multi-process
  sandboxing: form state + real key/input dispatch, focus model,
  text editing with caret, the value/state JS APIs, submission +
  FormData, constraint validation, persisted cookies, one shared
  storage store across tabs, fetch behind a broker, fail-closed
  sandbox with watchdog + sad tab.
- **M6 — done.** Modern CSS: fixed/sticky/absolute positioning,
  `z-index` with Appendix E paint order, stacking contexts, opacity,
  2D transforms, transitions/`@keyframes` on a monotonic animation
  engine, `var()`/`calc()`/`clamp()`, `prefers-color-scheme`,
  `@font-face`; IPC protocol 7 → 9; CI sandbox race fixed
  (compromised-renderer poison flag).
- **M7 — done.** Inline SVG & 2D canvas: SVG namespace threaded
  downstream, replaced-element layout for `<svg>`/`<canvas>`,
  path model + lyon tessellation with a 7th wgpu pipeline,
  raster-first SVG slice, gradients in CSS and SVG, full
  CanvasRenderingContext2D with tainting policy, `<use>`/`<image>`/
  `<foreignObject>`, external `<img src=*.svg>` decode; IPC
  protocol 9 → 10.
- **M8 — done.** Audio/video + SPA/real-time APIs: video-frame
  transport measured at 1080p60 before any decoder code, new
  `pluto-media` crate (demux/decode, audio sink with the audio
  clock as A/V master, autoplay policy), `Destination::Media`/
  `WebSocket`, History API with per-entry `{url, state, scroll}`,
  same-document navigation with `hashchange`/`popstate`, WebSocket
  transport, `ReadableStream` + chunked fetch + `EventSource`,
  IntersectionObserver fix; IPC protocol 10 → 13.

Full workspace suite: 1174 tests, all passing. See `ARCHITECTURE.md`
for the full M1–M12 roadmap.
