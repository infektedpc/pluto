mod compositor;
mod page;
mod renderer;
mod viewport;

use compositor::Compositor;
use cosmic::app::{Core, Task};
use cosmic::iced::{Alignment, Length};
use cosmic::{Application, Element, executor, widget};
use pluto_ipc::{BrowserToRenderer, CommandSender, RendererToBrowser};
use pluto_paint::DisplayList;
use std::sync::Arc;
use viewport::WebViewport;

pub struct PlutoBrowser {
    core: Core,
    url_input: String,
    active_url: String,
    /// The shell's compositor holder (Stage 4 #3): the last-good
    /// display list, its resource table, and the scroll offset.
    /// The worker owns the document; the shell never lays out.
    compositor: Compositor,
    status: String,
    /// Last viewport sent to the worker (CSS px, logical — the page
    /// area of the window). Starts at the default window size; the
    /// viewport widget corrects it on the first frame.
    viewport: (f32, f32),
    /// The worker's command channel, once the subscription starts
    /// it (`RendererReady`). `None` until then.
    renderer_tx: Option<CommandSender>,
    /// A navigation that arrived before the worker started (the
    /// first load), flushed on `RendererReady`.
    pending_url: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Message {
    UrlChanged(String),
    Navigate,
    Back,
    Forward,
    Reload,
    /// The worker subscription started: this is its command sender.
    RendererReady(CommandSender),
    /// One event from the worker (commit, display list, failure...).
    RendererEvent(RendererToBrowser),
    /// The viewport widget's bounds changed (window resize, or the
    /// first frame reporting the real page area).
    ViewportResized(f32, f32),
    /// A wheel event in the viewport, in pixels.
    Scrolled(f32, f32),
}

impl PlutoBrowser {
    /// Post one command to the worker, if it is running. A failed
    /// send means the worker is gone; the subscription restarts it
    /// and `RendererReady` re-syncs state.
    fn send_command(&mut self, command: BrowserToRenderer) {
        if let Some(tx) = &self.renderer_tx
            && tx.send(command).is_err()
        {
            self.renderer_tx = None;
        }
    }

    /// Start a navigation: update the address bar and status, reset
    /// the scroll, and hand the URL to the worker — or hold it until
    /// the worker starts. The old page stays visible until the new
    /// list lands: the holder keeps the last-good list.
    fn commit(&mut self, url: String) {
        self.active_url = url.clone();
        self.url_input = url.clone();
        self.compositor.reset_scroll();
        self.status = format!("Loading {url}…");
        if self.renderer_tx.is_some() {
            self.send_command(BrowserToRenderer::CommitNavigation { url });
        } else {
            self.pending_url = Some(url);
        }
    }
}

/// A bundled toolbar icon: a Cosmic symbolic SVG, tinted by the
/// theme to its icon color (white on the dark nav bar).
///
/// Bundled because libcosmic's `from_name` only searches the Cosmic
/// icon theme, which is not installed on every machine — and when
/// the lookup finds nothing it silently draws a blank button.
/// These bytes are compiled into the binary, so the icons render
/// wherever Pluto runs. See `assets/icons/README.md` for the source
/// and license.
///
/// Returns the [`Handle`], which is what `button::icon` takes —
/// the same pattern libcosmic uses for its own bundled link icon.
fn bundled_icon(bytes: &'static [u8]) -> widget::icon::Handle {
    widget::icon::from_svg_bytes(bytes).symbolic(true)
}

impl Application for PlutoBrowser {
    type Executor = executor::Default;
    type Flags = bool;
    type Message = Message;

    const APP_ID: &'static str = "org.plutobrowser.Pluto";

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(mut core: Core, use_client_decorations: Self::Flags) -> (Self, Task<Self::Message>) {
        // libcosmic draws its header bar whenever `show_headerbar` is
        // set (default true) — even when the window manager is already
        // drawing a frame, which gives the window two title bars. Show
        // it only when libcosmic is drawing the frame itself.
        core.window.show_headerbar = use_client_decorations;
        core.window.header_title = "Pluto".to_string();
        let url = page::DEFAULT_URL.to_string();
        eprintln!("[pluto] starting, loading {url}");
        // No load task here: the worker subscription (Stage 4 #2)
        // starts the pipeline, and the first navigation is flushed
        // to it as `pending_url` when it reports ready.
        let app = Self {
            core,
            url_input: url.clone(),
            active_url: url.clone(),
            compositor: Compositor::new(),
            status: format!("Loading {url}…"),
            viewport: page::DEFAULT_VIEWPORT,
            renderer_tx: None,
            pending_url: Some(url),
        };
        (app, Task::none())
    }

    /// The renderer worker subscription (Stage 4 #2): the page
    /// pipeline runs there, off the UI thread.
    fn subscription(&self) -> cosmic::iced::Subscription<Self::Message> {
        renderer::renderer_subscription()
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::UrlChanged(val) => self.url_input = val,
            Message::Navigate => {
                let url = page::normalize_url(&self.url_input);
                self.commit(url);
            }
            Message::Reload => {
                let url = self.active_url.clone();
                self.commit(url);
            }
            Message::RendererReady(tx) => {
                // The worker is up: sync its viewport first (it
                // starts at the default size), then flush the
                // pending navigation.
                self.renderer_tx = Some(tx);
                self.send_command(BrowserToRenderer::Resize {
                    width: self.viewport.0,
                    height: self.viewport.1,
                });
                if let Some(url) = self.pending_url.take() {
                    self.send_command(BrowserToRenderer::CommitNavigation { url });
                }
            }
            Message::RendererEvent(event) => match event {
                RendererToBrowser::DidCommit { url, status } => {
                    eprintln!("[pluto] page loaded: {url}");
                    self.active_url = url.clone();
                    self.url_input = url;
                    self.status = status;
                }
                RendererToBrowser::SubmitDisplayList {
                    epoch,
                    bytes,
                    resources,
                } => {
                    // The list crosses as postcard bytes even
                    // in-process — the same bytes that cross IPC at
                    // M5. Decode and validate before installing: a
                    // corrupt or out-of-bounds list never reaches
                    // the GPU.
                    match DisplayList::from_bytes(&bytes) {
                        Ok(list) => match list.validate(&resources) {
                            Ok(()) => {
                                if self.compositor.submit(
                                    epoch,
                                    Arc::new(list),
                                    Arc::new(resources),
                                ) {
                                    eprintln!(
                                        "[pluto] display list ready: epoch {}",
                                        self.compositor.epoch()
                                    );
                                }
                            }
                            Err(err) => eprintln!(
                                "[pluto] ignoring invalid display list (epoch {epoch}): {err:?}"
                            ),
                        },
                        Err(err) => eprintln!(
                            "[pluto] ignoring undecodable display list (epoch {epoch}): {err:?}"
                        ),
                    }
                }
                RendererToBrowser::LoadFailed { reason, .. } => {
                    // The worker already submitted the error page as
                    // a display list; the status line says why.
                    self.status = format!("Failed: {reason}");
                }
                RendererToBrowser::TitleChanged { .. } => {
                    // M3 (multi-tab) reads the title; the shell
                    // ignores it for now.
                }
                RendererToBrowser::RendererCrashed { reason } => {
                    eprintln!("[pluto] renderer crashed: {reason}");
                    self.status = format!("Renderer crashed: {reason}");
                    // The stale list stays visible; the next
                    // navigation recovers.
                }
                RendererToBrowser::RequestResource { .. } => {
                    // M2 (subresources) fetches these; ignored now.
                }
            },
            Message::ViewportResized(width, height) => {
                // Degenerate sizes (minimised window) carry no layout;
                // a real size arrives when the window is restored.
                if width < 1.0 || height < 1.0 {
                    return Task::none();
                }
                // Skip when nothing changed: the widget only publishes
                // on a bounds change, and the layout is deterministic
                // at a fixed size, so a repeat would be pure waste.
                if (width, height) == self.viewport {
                    return Task::none();
                }
                self.viewport = (width, height);
                self.send_command(BrowserToRenderer::Resize { width, height });
            }
            Message::Scrolled(dx, dy) => {
                // Wheel scrolling never touches layout: the holder
                // stores the offset and the renderer subtracts it.
                self.compositor.scroll_by(dx, dy);
            }
            Message::Back | Message::Forward => {
                // History lands in M2; no-op for now.
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Self::Message> {
        // NOTE: in this iced pin, `row()`/`column()` take their children
        // as an argument (no zero-arg builder + `.push()` chain).
        // Round icon buttons (ButtonClass::Icon), not the pill-shaped
        // text buttons — the glyphs render oval in a text button.
        // The icons are bundled SVGs (see `bundled_icon`): a
        // `from_name` lookup only searches the Cosmic icon theme and
        // draws a blank button where that theme is not installed.
        let nav_bar = widget::row([
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/go-previous-symbolic.svg"
            )))
            .on_press(Message::Back)
            .into(),
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/go-next-symbolic.svg"
            )))
            .on_press(Message::Forward)
            .into(),
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/view-refresh-symbolic.svg"
            )))
            .on_press(Message::Reload)
            .into(),
            widget::text_input("Enter URL...", &self.url_input)
                .on_input(Message::UrlChanged)
                .on_submit(|_| Message::Navigate)
                .width(Length::Fill)
                .into(),
            widget::button::suggested("Go")
                .on_press(Message::Navigate)
                .into(),
        ])
        .spacing(8)
        .padding(8)
        .align_y(Alignment::Center);

        // The viewport draws whatever the compositor holder
        // currently holds; the worker updates it from its thread.
        let web_view = cosmic::iced::widget::Shader::new(WebViewport {
            clear_color: [1.0, 1.0, 1.0, 1.0],
            display: Arc::clone(self.compositor.display()),
            resources: Arc::clone(self.compositor.resources()),
            scroll: self.compositor.scroll(),
        })
        .width(Length::Fill)
        .height(Length::Fill);

        // The status line sits at the bottom of the window, below the
        // page — like a browser status bar — not under the nav bar.
        let status = widget::row([widget::text(&self.status).size(12).into()]).padding(4);

        widget::column([
            nav_bar.into(),
            widget::divider::horizontal::default().into(),
            web_view.into(),
            widget::divider::horizontal::default().into(),
            status.into(),
        ])
        .into()
    }
}

fn main() -> cosmic::iced::Result {
    // NOTE: this libcosmic pin has no `Settings::size()` builder (the field
    // is pub(crate)); the default window is 1024x768. Resize after launch
    // or set it once a size builder exists upstream.
    let client_decorations = parse_decorations_arg().unwrap_or_else(use_client_decorations);
    eprintln!(
        "[pluto] window decorations: {}",
        if client_decorations {
            "client-side (libcosmic)"
        } else {
            "server-side (window manager)"
        }
    );
    let settings =
        cosmic::app::Settings::default().client_decorations(client_decorations);
    cosmic::app::run::<PlutoBrowser>(settings, client_decorations)
}

/// `--csd` / `--client-decorations` forces libcosmic to draw the frame;
/// `--ssd` / `--server-decorations` forces the window manager's frame.
/// Returns `None` when no flag is given (auto-detect per session).
/// Last flag wins.
fn parse_decorations_arg() -> Option<bool> {
    let mut choice = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--csd" | "--client-decorations" => choice = Some(true),
            "--ssd" | "--server-decorations" => choice = Some(false),
            _ => {}
        }
    }
    choice
}

/// Whether libcosmic should draw the window frame itself (client-side
/// decorations). Use the window manager's frame when it provides one;
/// only fall back to libcosmic drawing it where the WM doesn't — GNOME
/// (Mutter on Wayland offers no server-side decorations). Without this,
/// Pluto's window has no title bar or window controls at all on such
/// desktops, because the app draws no header bar of its own there.
fn use_client_decorations() -> bool {
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
    let desktop = std::env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .to_ascii_lowercase();
    match session.as_str() {
        // X11 window managers draw the frame (KWin, Mutter, Xfwm...).
        "x11" => false,
        "wayland" => {
            // GNOME is the holdout; KDE, Xfce (its Wayland session runs
            // on labwc), sway, Hyprland, COSMIC and other wlroots
            // compositors offer server-side decorations. An unknown
            // compositor falls back to libcosmic drawing.
            desktop.contains("gnome")
                || !(desktop.contains("kde")
                    || desktop.contains("xfce")
                    || desktop.contains("sway")
                    || desktop.contains("hyprland")
                    || desktop.contains("cosmic")
                    || desktop.contains("wlroots")
                    || desktop.contains("labwc")
                    || desktop.contains("wayfire"))
        }
        _ => {
            if std::env::var_os("WAYLAND_DISPLAY").is_some() {
                desktop.contains("gnome") || desktop.is_empty()
            } else {
                // X11 (DISPLAY set) or no display vars: the WM usually
                // draws the frame.
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bundled nav icons must be present, well-formed SVG, and
    /// single-color shapes — so the symbolic tint paints them wholly
    /// in the theme's icon color. A truncated or corrupt asset would
    /// otherwise fail silently at runtime: the same blank button as
    /// the theme lookup it replaced.
    #[test]
    fn bundled_nav_icons_are_valid_single_color_svgs() {
        let icons: [&[u8]; 3] = [
            include_bytes!("../assets/icons/go-previous-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/go-next-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/view-refresh-symbolic.svg") as &[u8],
        ];
        for bytes in icons {
            let text = std::str::from_utf8(bytes).expect("icon is UTF-8");
            assert!(text.starts_with("<svg"), "icon starts with <svg");
            assert!(text.contains("</svg>"), "icon closes </svg>");
            assert!(
                text.contains("viewBox=\"0 0 16 16\""),
                "icon is a 16x16 canvas"
            );
            // Every painted fill must be the one shape color. The
            // root fill="none" is not painted, and the clip-path
            // rect's white is a mask, never drawn.
            let fills: Vec<&str> = text
                .split("fill=\"")
                .skip(1)
                .map(|s| s.split('"').next().unwrap_or(""))
                .filter(|f| *f != "none" && *f != "white")
                .collect();
            assert!(!fills.is_empty(), "icon has a painted fill");
            assert!(
                fills.iter().all(|f| *f == "#232323"),
                "icon is a single-color shape: {fills:?}"
            );
            // And the handle builds without panicking.
            let _handle = bundled_icon(bytes);
        }
    }
}
