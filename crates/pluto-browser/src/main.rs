mod viewport;

use cosmic::app::{Core, Task};
use cosmic::iced::{Alignment, Length};
use cosmic::{Application, Element, executor, widget};
use viewport::WebViewport;

pub struct PlutoBrowser {
    core: Core,
    url_input: String,
    active_url: String,
}

#[derive(Debug, Clone)]
pub enum Message {
    UrlChanged(String),
    Navigate,
    Back,
    Forward,
    Reload,
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
        let app = Self {
            core,
            url_input: "https://example.com".to_string(),
            active_url: "https://example.com".to_string(),
        };
        (app, Task::none())
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::UrlChanged(val) => self.url_input = val,
            Message::Navigate => {
                self.active_url = self.url_input.clone();
                // Trigger pluto-net fetch -> pluto-html -> pluto-css -> pluto-layout
            }
            Message::Back | Message::Forward | Message::Reload => {}
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Self::Message> {
        // NOTE: in this iced pin, `row()`/`column()` take their children
        // as an argument (no zero-arg builder + `.push()` chain).
        let nav_bar = widget::row([
            widget::button::text("←")
                .on_press(Message::Back)
                .into(),
            widget::button::text("→")
                .on_press(Message::Forward)
                .into(),
            widget::button::text("↻")
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
            url: self.active_url.clone(),
            clear_color: [1.0, 1.0, 1.0, 1.0],
        })
        .width(Length::Fill)
        .height(Length::Fill);

        widget::column([
            nav_bar.into(),
            widget::divider::horizontal::default().into(),
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
