# Pluto

A new browser: libcosmic shell + wgpu viewport, with a from-scratch engine
in Rust (`pluto-net`, `pluto-html`, `pluto-css`, `pluto-layout`,
`pluto-paint`, `pluto-js`). See `ARCHITECTURE.md` for how the layers connect.

## Run

```sh
cargo run -p pluto-browser
```

The window opens with a nav bar (Back / Forward / Reload / URL / Go) and a
blank white viewport. Pressing Go records the URL but fetches nothing yet —
the engine crates are stubs; see "Next step" below.

Build notes:

- Toolchain: stable Rust 1.99 (`~/.cargo/bin` must be on `PATH`;
  `cargo` is not on the default `PATH` in this VM).
- `pluto-js` uses rquickjs with `bindgen`, which needs libclang:
  ```sh
  export LIBCLANG_PATH="$HOME/workspace/harness-tmp/llvm18/usr/lib/llvm-18/lib"
  ```

## Version notes (read before "upgrading" anything)

The spec this started from named wgpu 24, cosmic-text 0.12, and taffy 0.7,
and a bare `https://github.com` libcosmic URL. All three were corrected
during bootstrap, verified against the actual sources:

- **libcosmic** is pinned to `pop-os/libcosmic` rev
  `ecc5fe4164e1c9024b932382d24ea41f24147e75` — the exact pin Frost uses
  and the one already cached in this VM, so the first build does not
  re-fetch it. Unpinning to latest main is a one-line change in the root
  `Cargo.toml`, but it moves wgpu/cosmic-text/taffy under you.
- **wgpu 28, cosmic-text 0.19, taffy 0.15**: what that libcosmic pin's
  iced actually uses (Frost's `Cargo.lock` proves 28.0.0 / 0.19.0 /
  0.15.1). Older majors would pull second copies of each crate, and the
  `wgpu::Device` from libcosmic would no longer be the same type
  `pluto-paint` sees.
- **Shader widget path**: `cosmic::widget::shader` and
  `cosmic::iced_wgpu::wgpu` do not exist in this pin. The real paths are
  `cosmic::iced::widget::Shader` and `cosmic::iced::wgpu` — fixed in
  `crates/pluto-browser/src/viewport.rs` and `main.rs`.
- **Window size**: this libcosmic pin has no `Settings::size()` builder
  (field is `pub(crate)`), so the window opens at the default 1024x768
  instead of 1280x800.

## Next step — Milestone 1: "Hello, example.com"

End-to-end static page pipeline (see `ARCHITECTURE.md` for the full
M1–M5 roadmap). Order:

1. `pluto-net` — async HTTP/1.1 + HTTP/2 client, redirects 301/302/307/308
2. `pluto-html` & `pluto-css` — winnow parsers (core tags, basic rules)
3. `pluto-layout` & `pluto-paint` — taffy block/flex + cosmic-text,
   quads and glyphs via wgpu in the viewport

Done means: https://example.com renders its centered container, heading,
paragraph, and link inside the libcosmic window.
