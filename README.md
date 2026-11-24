# Pluto

A new browser: libcosmic shell + wgpu viewport, with a from-scratch engine
in Rust (`pluto-net`, `pluto-html`, `pluto-css`, `pluto-layout`,
`pluto-paint`, `pluto-js`). See `ARCHITECTURE.md` for how the layers connect.

## Run

```sh
cargo run -p pluto-browser
```

The window opens with a nav bar (Back / Forward / Reload / URL / Go), a
status line, and a viewport that loads https://example.com on startup —
grey page, white centered box, "Example Domain" heading, paragraph, and
link. The status line reads `Loaded https://example.com` when the page
is up. Type a URL and press Go to navigate (a bare host works:
`example.com` becomes `https://example.com`). A failed fetch renders
an error page instead of a blank window.

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

## Next step — Milestone 2

Full selectors and specificity, UA stylesheet, external `<link>` CSS,
grid, border-radius, box-shadow, overflow/scroll, hit-testing,
clickable links, and Back/Forward history (see `ARCHITECTURE.md` for
the full M1–M10 roadmap). Done means 60+ FPS browsing of info.cern.ch,
Hacker News, and Rust docs.
