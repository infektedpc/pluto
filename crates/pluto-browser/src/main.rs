mod page;
mod viewport;

use cosmic::app::{Core, Task};
use cosmic::iced::{Alignment, Length};
use cosmic::{Application, Element, executor, widget};
use pluto_paint::DisplayList;
use std::sync::Arc;
use viewport::WebViewport;

pub struct PlutoBrowser {
    core: Core,
    url_input: String,
    active_url: String,
    display: Arc<DisplayList>,
    status: String,
}

#[derive(Debug, Clone)]
pub enum Message {
    UrlChanged(String),
    Navigate,
    Back,
    Forward,
    Reload,
    PageLoaded(Arc<DisplayList>, String),
    PageFailed(String),
}

/// Start loading `url` in the background; the result arrives as
/// `PageLoaded` / `PageFailed`. This libcosmic pin's `Task` carries
/// `Action<Message>`, so the mapper wraps the message in `Action::App`.
fn load_task(url: String) -> Task<Message> {
    Task::perform(page::load_page(url), |result| match result {
        Ok((display, final_url)) => {
            cosmic::Action::App(Message::PageLoaded(display, final_url))
        }
        Err(err) => cosmic::Action::App(Message::PageFailed(err)),
    })
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
        let app = Self {
            core,
            url_input: url.clone(),
            active_url: url.clone(),
            display: Arc::new(DisplayList::new()),
            status: format!("Loading {url}…"),
        };
        (app, load_task(url))
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::UrlChanged(val) => self.url_input = val,
            Message::Navigate => {
                let url = page::normalize_url(&self.url_input);
                self.active_url = url.clone();
                self.url_input = url.clone();
                self.display = Arc::new(DisplayList::new());
                self.status = format!("Loading {url}…");
                return load_task(url);
            }
            Message::Reload => {
                let url = self.active_url.clone();
                self.display = Arc::new(DisplayList::new());
                self.status = format!("Loading {url}…");
                return load_task(url);
            }
            Message::PageLoaded(display, final_url) => {
                eprintln!("[pluto] page loaded: {final_url}");
                self.display = display;
                self.active_url = final_url.clone();
                self.url_input = final_url.clone();
                self.status = format!("Loaded {final_url}");
            }
            Message::PageFailed(err) => {
                eprintln!("[pluto] page failed: {err}");
                // Render the failure as a page; the status line says why.
                self.display =
                    page::build_display_list(&page::error_document(&self.active_url, &err));
                self.status = format!("Failed: {err}");
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
        let nav_bar = widget::row([
            widget::button::icon(widget::icon::from_name("go-previous-symbolic"))
                .on_press(Message::Back)
                .into(),
            widget::button::icon(widget::icon::from_name("go-next-symbolic"))
                .on_press(Message::Forward)
                .into(),
            widget::button::icon(widget::icon::from_name("view-refresh-symbolic"))
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

        // NOTE: `cosmic::widget::shader` does not exist in this libcosmic pin;
        // the shader widget is `cosmic::iced::widget::Shader`.
        let web_view = cosmic::iced::widget::Shader::new(WebViewport {
            clear_color: [1.0, 1.0, 1.0, 1.0],
            display: Arc::clone(&self.display),
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
