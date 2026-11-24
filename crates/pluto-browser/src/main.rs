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
    type Flags = ();
    type Message = Message;

    const APP_ID: &'static str = "org.plutobrowser.Pluto";

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, _flags: Self::Flags) -> (Self, Task<Self::Message>) {
        let url = page::DEFAULT_URL.to_string();
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

        let status = widget::text(&self.status).size(12);

        widget::column([
            nav_bar.into(),
            widget::divider::horizontal::default().into(),
            status.into(),
            web_view.into(),
        ])
        .into()
    }
}

fn main() -> cosmic::iced::Result {
    // NOTE: this libcosmic pin has no `Settings::size()` builder (the field
    // is pub(crate)); the default window is 1024x768. Resize after launch
    // or set it once a size builder exists upstream.
    let settings = cosmic::app::Settings::default();
    cosmic::app::run::<PlutoBrowser>(settings, ())
}
