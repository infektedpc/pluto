mod compositor;
mod js_host;
mod page;
mod renderer;
mod viewport;

use compositor::Compositor;
use cosmic::app::{Core, Task};
use cosmic::iced::{Alignment, Length};
use cosmic::{Application, Element, executor, widget};
use pluto_ipc::{BrowserToRenderer, CommandSender, InputEvent, RendererToBrowser};
use pluto_paint::{perf, DisplayList};
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
    /// Whether the window is focused (Stage 8 Rec 1). Focus is
    /// the shell's proxy for visibility in a single-window shell:
    /// an unfocused window produces no frames and no display
    /// lists (the worker still runs JS timers), and the flag is
    /// re-synced to a restarted worker on `RendererReady`.
    visible: bool,
    /// The worker's command channel, once the subscription starts
    /// it (`RendererReady`). `None` until then.
    renderer_tx: Option<CommandSender>,
    /// A navigation that arrived before the worker started (the
    /// first load), flushed on `RendererReady`.
    pending_url: Option<String>,
    /// Minimal session history (R17): every committed URL, in order,
    /// plus the index of the current entry. Address-bar navigation
    /// and link clicks push entries (in `DidCommit`); Back/Forward
    /// move the index and navigate with `history_nav` set, so the
    /// commit they cause does not push a duplicate. `:visited` reads
    /// this in a later stage — the matcher still answers false until
    /// the visited set is threaded into `ElementState` (see
    /// renderer.rs).
    history: Vec<String>,
    history_index: usize,
    history_nav: bool,
    /// The modifier keys currently held, as of the last key event
    /// (Stage 8 Rec 9): pointer/wheel messages carry no modifiers
    /// of their own, so the shell tracks them here and stamps
    /// them onto every forwarded event.
    modifiers: pluto_ipc::Modifiers,
    /// Which pointer buttons are held, as a bitmask (bit 0 left,
    /// 1 middle, 2 right) — the `buttons` field on pointer moves
    /// (dragging) and on the press/release itself.
    buttons_held: u16,
    /// When the last wheel event arrived (Stage 8 Rec 9): the
    /// shell synthesises the gesture phase from the gap — a first
    /// wheel, or one after >200 ms idle, is `Began`; the rest are
    /// `Changed`. (`Ended` is synthesised worker-side by timeout.)
    last_wheel: Option<std::time::Instant>,
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
    /// A wheel event in the viewport, in widget coordinates, with
    /// the raw delta and its mode. The worker owns the whole wheel:
    /// nested containers first, the viewport remainder last
    /// (Stage 8 Rec 9) — the shell no longer scrolls the viewport
    /// blindly on wheel.
    Scrolled {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
        delta_mode: pluto_ipc::DeltaMode,
    },
    /// The window gained or lost focus (Stage 8 Rec 1): the
    /// shell's visibility signal, forwarded to the worker as
    /// `SetVisible` so a hidden tab stops producing frames. The
    /// same event also sets the worker's keyboard focus
    /// (`SetFocused`, Stage 8 Rec 9).
    WindowFocused(bool),
    /// A pointer move in the viewport, in widget coordinates.
    /// The shell adds the scroll offset before forwarding.
    PointerMoved { x: f32, y: f32 },
    /// A pointer button press in the viewport, in widget
    /// coordinates. `button`: 0 left, 1 middle, 2 right.
    PointerDown { x: f32, y: f32, button: u8 },
    /// A pointer button release in the viewport, in widget
    /// coordinates. A left release over its press target is the
    /// click — the worker synthesises it (Stage 8 Rec 9).
    PointerUp { x: f32, y: f32, button: u8 },
    /// A key press or release, delivered only when no widget
    /// captured it (the page has focus — keys typed into the URL
    /// bar are captured by the text input and never reach the
    /// page). `key` is the logical key, `code` the physical key.
    KeyEvent {
        key: String,
        code: String,
        pressed: bool,
        repeat: bool,
        modifiers: pluto_ipc::Modifiers,
    },
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
        // A navigation expects a new display list: until it
        // activates, frames presenting the old list count as
        // deadline-missed (Stage 8 Rec 2).
        self.compositor.note_expected_list();
        // Perf (Stage 8 Rec 5): the page being left gets its
        // summary line first, then the new navigation starts the
        // clock for first paint / first contentful paint.
        eprintln!("{}", perf::summary());
        if let Ok(mut counters) = perf::global().lock() {
            counters.note_navigation_start();
        }
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

/// The window-focus filter for the visibility subscription
/// (Stage 8 Rec 1): focus/unfocus map to visibility, everything
/// else is discarded. `listen_with` takes a plain function, so
/// this lives outside the `Application` impl. Focus is the
/// shell's proxy for visibility — a single-window shell has no
/// minimize event in iced, and an unfocused window is the case
/// that matters (the tab keeps its JS timers, it just stops
/// producing frames until focus returns).
fn window_visibility(
    event: cosmic::iced::event::Event,
    _status: cosmic::iced::event::Status,
    _window: cosmic::iced::window::Id,
) -> Option<Message> {
    match event {
        cosmic::iced::event::Event::Window(cosmic::iced::window::Event::Focused) => {
            Some(Message::WindowFocused(true))
        }
        cosmic::iced::event::Event::Window(cosmic::iced::window::Event::Unfocused) => {
            Some(Message::WindowFocused(false))
        }
        _ => None,
    }
}

/// Monotonic milliseconds since the shell started (Stage 8 Rec 9):
/// the timestamp stamped on every input event, for gesture and
/// velocity timing.
fn timestamp_ms() -> u64 {
    static SHELL_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    SHELL_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// The keyboard filter for the page-key subscription (Stage 8
/// Rec 9): key presses/releases reach the page only when no
/// widget captured them — the page has focus. Keys typed into
/// the URL bar are `Captured` by the text input and are dropped
/// here, so address-bar typing never reaches the page. Modifier
/// keys alone (`Shift` pressed by itself) arrive as key events
/// too, so the shell's modifier state stays current.
fn keyboard_event(
    event: cosmic::iced::event::Event,
    status: cosmic::iced::event::Status,
    _window: cosmic::iced::window::Id,
) -> Option<Message> {
    if !matches!(status, cosmic::iced::event::Status::Ignored) {
        return None;
    }
    match event {
        cosmic::iced::event::Event::Keyboard(
            cosmic::iced::keyboard::Event::KeyPressed {
                key,
                physical_key,
                modifiers,
                repeat,
                ..
            },
        ) => Some(Message::KeyEvent {
            key: key_string(&key),
            code: code_string(&physical_key),
            pressed: true,
            repeat,
            modifiers: to_modifiers(modifiers),
        }),
        cosmic::iced::event::Event::Keyboard(
            cosmic::iced::keyboard::Event::KeyReleased {
                key,
                physical_key,
                modifiers,
                ..
            },
        ) => Some(Message::KeyEvent {
            key: key_string(&key),
            code: code_string(&physical_key),
            pressed: false,
            repeat: false,
            modifiers: to_modifiers(modifiers),
        }),
        _ => None,
    }
}

/// The logical key as a string (Stage 8 Rec 9): character keys
/// give the character (locale- and modifier-aware), named keys
/// give the web key name (`Enter`, `Tab`, `ArrowDown`, `Space`,
/// `Escape`, ...).
fn key_string(key: &cosmic::iced::keyboard::Key) -> String {
    match key {
        cosmic::iced::keyboard::Key::Character(c) => c.to_string(),
        cosmic::iced::keyboard::Key::Named(named) => format!("{named:?}"),
        cosmic::iced::keyboard::Key::Unidentified => "Unidentified".to_string(),
    }
}

/// The physical key as a string (Stage 8 Rec 9): the layout-
/// independent code name (`KeyA`, `Digit1`, `Enter`, `NumpadEnter`,
/// `ArrowLeft`, `Space`, `Tab`, ...).
fn code_string(physical: &cosmic::iced::keyboard::key::Physical) -> String {
    match physical {
        cosmic::iced::keyboard::key::Physical::Code(code) => format!("{code:?}"),
        cosmic::iced::keyboard::key::Physical::Unidentified(_) => "Unidentified".to_string(),
    }
}

/// The iced modifier bitflags as the IPC `Modifiers` (Stage 8
/// Rec 9): `logo` is the platform's meta key (Cmd on macOS,
/// Super on Linux/Windows).
fn to_modifiers(m: cosmic::iced::keyboard::Modifiers) -> pluto_ipc::Modifiers {
    pluto_ipc::Modifiers {
        shift: m.shift(),
        ctrl: m.control(),
        alt: m.alt(),
        meta: m.logo(),
    }
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
            visible: true,
            renderer_tx: None,
            pending_url: Some(url),
            history: Vec::new(),
            history_index: 0,
            history_nav: false,
            modifiers: pluto_ipc::Modifiers::default(),
            buttons_held: 0,
            last_wheel: None,
        };
        (app, Task::none())
    }

    /// The renderer worker subscription (Stage 4 #2): the page
    /// pipeline runs there, off the UI thread — plus the window
    /// focus subscription (Stage 8 Rec 1), which is the shell's
    /// visibility signal for the worker, and the keyboard
    /// subscription (Stage 8 Rec 9), which forwards page keys.
    fn subscription(&self) -> cosmic::iced::Subscription<Self::Message> {
        cosmic::iced::Subscription::batch([
            renderer::renderer_subscription(),
            cosmic::iced::event::listen_with(window_visibility),
            cosmic::iced::event::listen_with(keyboard_event),
        ])
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
                // starts at the default size), then its visibility
                // (a restarted worker defaults to visible), then
                // flush the pending navigation.
                self.renderer_tx = Some(tx);
                self.send_command(BrowserToRenderer::Resize {
                    width: self.viewport.0,
                    height: self.viewport.1,
                });
                self.send_command(BrowserToRenderer::SetVisible {
                    visible: self.visible,
                });
                self.send_command(BrowserToRenderer::SetFocused {
                    focused: self.visible,
                });
                if let Some(url) = self.pending_url.take() {
                    // The first navigation expects a list too, and
                    // starts the first-paint clock (no previous
                    // page to summarise yet).
                    self.compositor.note_expected_list();
                    if let Ok(mut counters) = perf::global().lock() {
                        counters.note_navigation_start();
                    }
                    self.send_command(BrowserToRenderer::CommitNavigation { url });
                }
            }
            Message::WindowFocused(visible) => {
                // Stage 8 Rec 1 + Rec 9: forward a real visibility
                // change to the worker; a repeat of the current state
                // is not re-sent (iced may deliver focus events more
                // than once per change). The same event sets the
                // worker's keyboard focus — a single-window shell
                // has no separate focus target, so focus and
                // visibility move together.
                if visible != self.visible {
                    self.visible = visible;
                    self.send_command(BrowserToRenderer::SetVisible { visible });
                    self.send_command(BrowserToRenderer::SetFocused { focused: visible });
                }
            }
            Message::RendererEvent(event) => match event {
                RendererToBrowser::DidCommit { url, status } => {
                    eprintln!("[pluto] page loaded: {url}");
                    self.active_url = url.clone();
                    self.url_input = url.clone();
                    self.status = status;
                    // History (R17): a back/forward commit only
                    // moves the index (already done in Back/Forward);
                    // any other commit — address bar, reload, link
                    // click — pushes a new entry after truncating
                    // the forward entries, unless it is the entry
                    // we are already on (a reload).
                    if self.history_nav {
                        self.history_nav = false;
                    } else if self.history.is_empty()
                        || self.history[self.history_index] != url
                    {
                        self.history.truncate(self.history_index + 1);
                        self.history.push(url);
                        self.history_index = self.history.len() - 1;
                    }
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
                                    // Pending, not active yet: the
                                    // next frame boundary activates
                                    // it (Stage 8 Rec 2).
                                    eprintln!(
                                        "[pluto] display list pending: epoch {epoch} (active {})",
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
                RendererToBrowser::CancelResource { request_id } => {
                    // The worker cancels its own fetches by token
                    // (Stage 5 #5); the shell holds no loader, so
                    // there is nothing to cancel here. At M5, when
                    // the fetch lives browser-side behind
                    // `RequestResource`, this is where the shell
                    // cancels it on the scheduler.
                    eprintln!(
                        "[pluto] ignoring CancelResource {request_id} (no shell-side loader)"
                    );
                }
                RendererToBrowser::RendererCrashed { reason } => {
                    eprintln!("[pluto] renderer crashed: {reason}");
                    self.status = format!("Renderer crashed: {reason}");
                    // The stale list stays visible; the next
                    // navigation recovers.
                }
                RendererToBrowser::ScrollViewport { dx, dy } => {
                    // Stage 8 Rec 9: the worker owns the wheel —
                    // containers first, and only the leftover delta
                    // comes back here as the viewport scroll.
                    self.compositor.scroll_by(dx, dy);
                }
                RendererToBrowser::RequestResource(..) => {
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
                // A resize re-lays out, so a new list is expected.
                self.compositor.note_expected_list();
                self.send_command(BrowserToRenderer::Resize { width, height });
            }
            Message::Scrolled {
                x,
                y,
                dx,
                dy,
                delta_mode,
            } => {
                // Stage 8 Rec 9: the wheel goes to the worker whole
                // — it owns nested containers, and only the leftover
                // delta comes back as `ScrollViewport`. The shell
                // synthesises the gesture phase from the gap since
                // the last wheel: a first wheel, or one after
                // >200 ms idle, begins a gesture; the rest continue
                // it. (`Ended` is synthesised worker-side by
                // timeout.) Widget coords -> page coords: add the
                // viewport scroll offset.
                let now = std::time::Instant::now();
                let phase = match self.last_wheel {
                    Some(last) if now.duration_since(last) < std::time::Duration::from_millis(200) => {
                        pluto_ipc::WheelPhase::Changed
                    }
                    _ => pluto_ipc::WheelPhase::Began,
                };
                self.last_wheel = Some(now);
                let (sx, sy) = self.compositor.scroll();
                self.send_command(BrowserToRenderer::InputEvent(InputEvent::Wheel {
                    x: x + sx,
                    y: y + sy,
                    dx,
                    dy,
                    delta_mode,
                    phase,
                    modifiers: self.modifiers,
                    timestamp_ms: timestamp_ms(),
                }));
            }
            Message::PointerMoved { x, y } => {
                // Widget coords -> page coords; the worker hit-tests
                // for hover restyle (Stage 8 Rec 9) and coalesces
                // moves to the latest per frame. `button` is None
                // (a move presses nothing); `buttons` is the mask
                // of buttons currently held (for dragging).
                let (sx, sy) = self.compositor.scroll();
                self.send_command(BrowserToRenderer::InputEvent(InputEvent::Pointer {
                    kind: pluto_ipc::PointerKind::Move,
                    x: x + sx,
                    y: y + sy,
                    button: None,
                    buttons: self.buttons_held,
                    modifiers: self.modifiers,
                    timestamp_ms: timestamp_ms(),
                    pointer_id: 0,
                }));
            }
            Message::PointerDown { x, y, button } => {
                // Record the held button before forwarding, so the
                // event's own `buttons` mask includes it.
                self.buttons_held |= 1 << button;
                let (sx, sy) = self.compositor.scroll();
                self.send_command(BrowserToRenderer::InputEvent(InputEvent::Pointer {
                    kind: pluto_ipc::PointerKind::Down,
                    x: x + sx,
                    y: y + sy,
                    button: Some(button),
                    buttons: self.buttons_held,
                    modifiers: self.modifiers,
                    timestamp_ms: timestamp_ms(),
                    pointer_id: 0,
                }));
            }
            Message::PointerUp { x, y, button } => {
                // Clear the released button before forwarding, so
                // the event's own `buttons` mask excludes it.
                self.buttons_held &= !(1 << button);
                let (sx, sy) = self.compositor.scroll();
                self.send_command(BrowserToRenderer::InputEvent(InputEvent::Pointer {
                    kind: pluto_ipc::PointerKind::Up,
                    x: x + sx,
                    y: y + sy,
                    button: Some(button),
                    buttons: self.buttons_held,
                    modifiers: self.modifiers,
                    timestamp_ms: timestamp_ms(),
                    pointer_id: 0,
                }));
            }
            Message::KeyEvent {
                key,
                code,
                pressed,
                repeat,
                modifiers,
            } => {
                // The modifiers ride every key event, so the
                // shell's copy stays current for pointer/wheel
                // events, which carry no modifiers of their own.
                self.modifiers = modifiers;
                self.send_command(BrowserToRenderer::InputEvent(InputEvent::Key {
                    key,
                    code,
                    pressed,
                    repeat,
                    modifiers,
                    timestamp_ms: timestamp_ms(),
                }));
            }
            Message::Back => {
                if self.history_index > 0 {
                    self.history_index -= 1;
                    self.history_nav = true;
                    let url = self.history[self.history_index].clone();
                    self.commit(url);
                }
            }
            Message::Forward => {
                if self.history_index + 1 < self.history.len() {
                    self.history_index += 1;
                    self.history_nav = true;
                    let url = self.history[self.history_index].clone();
                    self.commit(url);
                }
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
            display: self.compositor.display(),
            resources: self.compositor.resources(),
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
