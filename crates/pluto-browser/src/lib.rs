mod atomic;
mod bookmarks;
/// The browser-side network broker (Stage 13 Rec 11) — public
/// so integration tests can drive a real broker against a
/// no-socket renderer child.
pub mod broker;
mod compositor;
mod cookies;
mod history;
mod images;
mod js_host;
mod page;
mod renderer;
/// The per-tab renderer process (Stage 9 Rec 9): public so
/// integration tests can spawn real `--renderer` children.
pub mod downloads;
pub mod renderer_process;
mod security;
mod viewport;

/// Test-only lock serialising the tests that mutate the
/// process-global `HOME` env var (the permissions / storage /
/// security persistence round-trip tests) — and the tests that
/// build a worker tab (`renderer.rs`, `WorkerHomeGuard`): a tab
/// reads the profile at `TabState::new` and writes it back on
/// shutdown, so it is a HOME-mutating test even though it never
/// touches `HOME` itself. `HOME` is process-global, so two such
/// tests running in parallel clobber each other — each `with_home`
/// helper holds this lock from its `set_var` until its
/// `restore_home` restores the old value.
/// Poisoning is tolerated: a test that panics mid-span must not
/// take the rest of the suite down with it.
#[cfg(test)]
pub(crate) static TEST_HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use bookmarks::{BookmarkStore, load_bookmarks, save_bookmarks};
use history::{load_history, save_history};
use compositor::Compositor;
use cosmic::app::{Core, Task};
use cosmic::iced::{Alignment, Length};
use cosmic::{Application, Element, executor, widget};
use pluto_ipc::{BrowserToRenderer, CommandSender, InputEvent, RendererToBrowser};
use pluto_paint::{DisplayList, Resource, ResourceId, ResourceTable, perf};
use std::collections::HashSet;
use std::sync::Arc;
use viewport::WebViewport;

/// One session-history entry (Stage 16 Phase 3, Rec 8 — finisher):
/// the committed URL, the `history.state` JSON the page pushed for
/// it (`None` = no state, `history.state` reads `null`), and the
/// viewport scroll the page was last left at. The shell owns the
/// entries; the worker mirrors the current entry's state in its
/// realm and fires `popstate` on traversal. Scroll is restored on
/// traversal / reload when the new list lands (see Tab's
/// `pending_scroll_restore`); a fresh navigation resets it.
#[derive(Debug, Clone, PartialEq)]
struct HistoryEntry {
    url: String,
    state_json: Option<String>,
    scroll: (f32, f32),
}

impl HistoryEntry {
    fn new(url: String) -> Self {
        HistoryEntry {
            url,
            state_json: None,
            scroll: (0.0, 0.0),
        }
    }
}

/// One tab (Stage 11 Rec 4): its own compositor holder, history,
/// title, favicon, renderer channel, and load state. The worker
/// owns the document; the shell never lays out. Switching tabs
/// swaps the active compositor — no re-fetch, no re-layout — and
/// closing a tab kills its renderer (see `close_tab`).
struct Tab {
    /// Shell-assigned id, echoed by the worker (never hardcoded 0).
    id: u64,
    /// The tab's compositor holder (Stage 4 #3): the last-good
    /// display list, its resource table, and the scroll offset.
    compositor: Compositor,
    /// The tab's renderer command channel, once its subscription
    /// starts it (`RendererReady`). `None` until then.
    renderer_tx: Option<CommandSender>,
    /// The tab's decoded resources, accumulated from the
    /// worker's `ResourceReady` frames (Stage 11 Rec 1): the
    /// display list validates against this table, never against
    /// the (empty) table in the submit message. Held as an `Arc`
    /// so a submit clones the pointer, not the pixels; a
    /// `ResourceReady` insert uses `Arc::make_mut` (one map copy
    /// per insert batch after a submit — the M5 broker moves
    /// resources to shared memory and removes this copy).
    resources: Arc<ResourceTable>,
    /// The latest decoded video frame per `<video>` element
    /// (Stage 16 Phase 1, Rec 1): keyed by the frame's
    /// `element_id`, holding exactly one frame per element — a
    /// new frame overwrites the old one, so a slow shell never
    /// accumulates a backlog (the drop policy lives here and in
    /// `pluto_media::FrameChannel`). No `<video>` element exists
    /// yet (Phase 2), so nothing reads this map today; the shell
    /// accepts and stores frames now so the channel is live
    /// before the first element paints. Staleness is enforced
    /// worker-side, like `ResourceReady` (the worker drops stale
    /// epochs before sending).
    video_frames: std::collections::HashMap<u64, pluto_ipc::VideoFrame>,
    /// Frame-blob ids whose RGBA pixels changed since the last
    /// presented primitive (Stage 16, M8 Phase 2, Rec 5):
    /// the shell converts each presented YUV frame into the
    /// tab's live resource table and marks its blob id here;
    /// the viewport drains the set in `prepare()` and calls
    /// `PaintRenderer::advance_image_frame` for each id, so
    /// the GPU re-uploads just that texture. Shared with the
    /// viewport by `Arc<Mutex>` — the primitive must be
    /// `Send + Sync`, so `Rc<RefCell>` cannot ride in it.
    video_dirty: Arc<std::sync::Mutex<HashSet<ResourceId>>>,
    /// Minimal session history (R17), per tab: every committed
    /// entry, in order, plus the index of the current entry.
    /// Address-bar navigation and link clicks push entries (in
    /// `DidCommit`); Back/Forward move the index and navigate
    /// with `history_nav` set, so the commit they cause does not
    /// push a duplicate. Stage 16 Phase 3, Rec 8 (finisher):
    /// entries carry the page's `history.state` JSON and the
    /// scroll the page was left at, so a traversal restores both
    /// and the worker fires `popstate` after the commit.
    /// `:visited` reads the same committed URLs worker-side — the
    /// worker's set and this history hold the same URLs, with no
    /// IPC change. Stage 10 Rec 8: `history.json` in the profile
    /// (newest 50) is saved on every push of the ACTIVE tab only
    /// — per-tab persistence is a follow-up, so a background tab's
    /// history does not survive a restart.
    history: Vec<HistoryEntry>,
    history_index: usize,
    history_nav: bool,
    /// The scroll to restore once the traversed / reloaded page's
    /// list lands (Stage 16 Phase 3, Rec 8 — finisher): applied in
    /// the `SubmitDisplayList` arm, when the new list's max-scroll
    /// is known — a plain `set_scroll` at commit time would clamp
    /// against the OLD page's bounds.
    pending_scroll_restore: Option<(f32, f32)>,
    /// WebSocket pumps for this tab (Stage 16 Phase 3, Rec 11 —
    /// finisher): `socket_id` → the broker pump's command sender.
    /// A socket's events are routed back to this tab's renderer.
    ws_commands: std::collections::HashMap<u64, tokio::sync::mpsc::Sender<crate::broker::WsCommand>>,
    /// The page title, from the worker's `TitleChanged` (Stage
    /// 11 Rec 8); the URL until the first title arrives. Shown
    /// in the tab strip and, for the active tab, the window title.
    title: String,
    /// The favicon's resource id in this tab's table (Stage 11
    /// Rec 7), if the site has one.
    favicon_id: Option<ResourceId>,
    /// The address bar's text for this tab.
    url_input: String,
    /// The committed URL for this tab.
    active_url: String,
    /// The status line's text for this tab.
    status: String,
    /// A navigation that arrived before the tab's worker started
    /// (the first load), flushed on `RendererReady`.
    pending_url: Option<String>,
    /// Document-load progress, if a load is in flight (Stage 11
    /// Rec 9): bytes received vs `Content-Length` (`None` =
    /// indeterminate). Cleared on commit.
    progress: Option<(u64, Option<u64>)>,
    /// Per-layer sandbox results from the child's
    /// `SandboxReport` (Stage 13 Rec 12a): empty until the first
    /// report arrives (in-process tabs never report).
    sandbox_layers: Vec<(String, bool)>,
    /// Sad-tab state (Stage 13 Rec 12c): set on
    /// `RendererCrashed`, cleared on the next successful
    /// navigation or Reload.
    crashed: bool,
    /// Times of recent crashes (pruned to the last 60 s): three
    /// or more makes the tab `dead` — the restart policy.
    crash_times: Vec<std::time::Instant>,
    /// The last crash's reason, for the status line.
    crash_reason: String,
    /// Restart policy: three crashes in 60 s. While dead, only
    /// Reload may navigate this tab; other navigations are
    /// blocked with a status message until the user reloads.
    dead: bool,
}

impl Tab {
    /// A fresh tab for `url` (not yet navigated — the
    /// subscription starts its renderer, and `pending_url` is
    /// flushed on `RendererReady`).
    fn new(id: u64, url: String) -> Self {
        Self {
            id,
            compositor: Compositor::new(),
            renderer_tx: None,
            resources: Arc::new(ResourceTable::new()),
            video_frames: std::collections::HashMap::new(),
            video_dirty: Arc::new(std::sync::Mutex::new(HashSet::new())),
            history: Vec::new(),
            history_index: 0,
            history_nav: false,
            pending_scroll_restore: None,
            ws_commands: std::collections::HashMap::new(),
            title: url.clone(),
            favicon_id: None,
            url_input: url.clone(),
            active_url: url.clone(),
            status: format!("Loading {url}…"),
            pending_url: Some(url),
            progress: None,
            sandbox_layers: Vec::new(),
            crashed: false,
            crash_times: Vec::new(),
            crash_reason: String::new(),
            dead: false,
        }
    }
}

pub struct PlutoBrowser {
    core: Core,
    /// The tabs, in strip order (Stage 11 Rec 4). Exactly one
    /// is active; switching swaps the compositor with no
    /// re-fetch and no re-layout.
    tabs: Vec<Tab>,
    active_tab: usize,
    /// The next shell-assigned tab id.
    next_tab_id: u64,
    /// Last viewport sent to the workers (CSS px, logical — the page
    /// area of the window). Starts at the default window size; the
    /// viewport widget corrects it on the first frame.
    viewport: (f32, f32),
    /// Whether the window is focused (Stage 8 Rec 1). Focus is
    /// the shell's proxy for visibility in a single-window shell:
    /// an unfocused window produces no frames and no display
    /// lists (the worker still runs JS timers), and the flag is
    /// re-synced to a restarted worker on `RendererReady`.
    visible: bool,
    /// Whether the OS / desktop theme is dark (M6 Phase 3, Rec
    /// 12): read from libcosmic at startup, updated by the
    /// `system_theme_*_update` hooks, and forwarded to every
    /// tab's worker as `SetTheme` — the workers restyle against
    /// it (never a refetch).
    dark_theme: bool,
    /// The persisted bookmarks (nav feature): toggled by the
    /// star, listed in the hamburger menu's Bookmarks panel.
    bookmarks: BookmarkStore,
    /// Whether the hamburger menu is open. Opening a list panel
    /// closes the menu; the menu never shows alongside a panel.
    menu_open: bool,
    /// The open Bookmarks/History list panel, if any. The lists
    /// read the real stores: bookmarks from the profile store,
    /// history from the active tab's session history.
    panel: Option<Panel>,
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
    /// The browser-side network broker (Stage 13 Rec 11): owns
    /// the one shared `ResourceLoader` (cookie jar, HTTP cache,
    /// HSTS store, blocker) and answers renderer
    /// `RequestResource` messages over IPC. The renderers hold
    /// no sockets and write no profile state of their own.
    broker: Arc<crate::broker::NetworkBroker>,
}

#[derive(Debug, Clone)]
pub enum Message {
    UrlChanged(String),
    Navigate,
    Back,
    Forward,
    Reload,
    /// The bookmark star: toggle a bookmark for the current page.
    ToggleBookmark,
    /// The hamburger button: open/close the menu.
    ToggleMenu,
    /// Menu: open the Bookmarks list panel.
    ShowBookmarks,
    /// Menu: open the History list panel.
    ShowHistory,
    /// Close the open list panel.
    ClosePanel,
    /// Navigate to a URL picked from a list panel (a bookmark or
    /// a history entry — already a committed URL).
    NavigateTo(String),
    /// Remove a bookmark from the Bookmarks panel.
    RemoveBookmark(String),
    /// Menu: open a second Pluto window (a new process).
    NewWindow,
    /// Open a new tab (Stage 11 Rec 4): the tab strip's "+" and
    /// the menu's "New Tab" item, plus Ctrl+T.
    NewTab,
    /// Close the tab at this strip index (Stage 11 Rec 4): the
    /// tab's "×" button, plus Ctrl+W. The last tab cannot close.
    CloseTab(usize),
    /// Select the tab at this strip index (Stage 11 Rec 4):
    /// clicking its tab-strip button, plus Ctrl+Tab to cycle.
    SelectTab(usize),
    /// A tab's worker subscription started: the tab id and its
    /// command sender.
    RendererReady(u64, CommandSender),
    /// One event from a tab's worker (commit, display list,
    /// failure...), tagged with the tab id.
    RendererEvent(u64, RendererToBrowser),
    /// A broker serve task finished (Stage 13 Rec 11): the
    /// events were already forwarded to the tab as they arrived;
    /// this is just the task's completion signal (no-op).
    BrokerServeDone,
    /// A WebSocket pump finished (Stage 16 Phase 3, Rec 11 —
    /// finisher): the `socket_id`'s entry is removed from its
    /// tab's pump map. The pump's events were already forwarded
    /// to the tab's renderer as they arrived.
    WebSocketDone(u64),
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
    /// The active tab (read-only).
    fn active(&self) -> &Tab {
        &self.tabs[self.active_tab]
    }

    /// The active tab (mutable).
    fn active_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active_tab]
    }

    /// The tab with this shell-assigned id, if it still exists
    /// (a `RendererReady` for a tab closed before its worker
    /// started finds nothing here and is ignored).
    fn tab_mut(&mut self, id: u64) -> Option<&mut Tab> {
        self.tabs.iter_mut().find(|tab| tab.id == id)
    }

    /// The strip index of the tab with this id, if it exists.
    fn tab_index(&self, id: u64) -> Option<usize> {
        self.tabs.iter().position(|tab| tab.id == id)
    }

    /// The window title follows the active tab (Stage 11 Rec 8):
    /// its page title once known, else its URL — "Pluto" only
    /// when the tab has neither.
    fn sync_window_title(&mut self) {
        let title = self.active().title.clone();
        self.core.window.header_title = if title.is_empty() {
            "Pluto".to_string()
        } else {
            title
        };
    }

    /// Convert one stored YUV frame to RGBA and insert it under
    /// its deterministic video blob id (Stage 16, M8 Phase 2,
    /// Rec 5), marking the id dirty so the viewport's next
    /// `prepare()` re-uploads the texture. A frame that fails
    /// conversion (zero size, short planes) is dropped — the
    /// previous frame stays on screen, the next presented frame
    /// replaces it.
    fn convert_video_frame(&mut self, tab_index: usize, element_id: u64) {
        use pluto_media::yuv420_to_rgba;
        use pluto_paint::canvas::video_blob_id;
        let Some(frame) = self.tabs[tab_index]
            .video_frames
            .get(&element_id)
            .cloned()
        else {
            return;
        };
        let Some(rgba) = yuv420_to_rgba(&frame) else {
            return;
        };
        // `element_id` IS the slotmap key's ffi value (stamped
        // by the session from the element's `DomKey`), so the
        // key round-trips through `KeyData::from_ffi`.
        let node_id =
            pluto_html::NodeId::from(slotmap::KeyData::from_ffi(element_id));
        let blob = video_blob_id(node_id);
        let tab = &mut self.tabs[tab_index];
        Arc::make_mut(&mut tab.resources).resources.insert(
            blob,
            Resource::Image {
                width: frame.width,
                height: frame.height,
                rgba,
            },
        );
        tab.video_dirty.lock().unwrap().insert(blob);
    }

    /// Convert every stored frame of a tab (called when the tab
    /// becomes active — its frames arrived as YUV while it was
    /// in the background).
    fn convert_stored_video_frames(&mut self, tab_index: usize) {
        let ids: Vec<u64> = self.tabs[tab_index]
            .video_frames
            .keys()
            .copied()
            .collect();
        for id in ids {
            self.convert_video_frame(tab_index, id);
        }
    }

    /// Open a new tab on `url` and select it (Stage 11 Rec 4).
    /// The tab's renderer starts with the subscription (its id
    /// keys it), and the pending navigation flushes on
    /// `RendererReady` — the new tab loads without disturbing
    /// the tab it came from.
    fn new_tab(&mut self, url: String) {
        let id = self.next_tab_id;
        self.next_tab_id += 1;
        self.tabs.push(Tab::new(id, url));
        self.active_tab = self.tabs.len() - 1;
        self.sync_window_title();
    }

    /// Close the tab at `index` (Stage 11 Rec 4): ask its
    /// renderer to shut down, then drop the tab. The last tab
    /// cannot close — there is always at least one. The neighbour
    /// takes over (same index, or the one before it when the
    /// last tab closed).
    fn close_tab(&mut self, index: usize) {
        if self.tabs.len() <= 1 || index >= self.tabs.len() {
            return;
        }
        let mut tab = self.tabs.remove(index);
        if let Some(tx) = tab.renderer_tx.take() {
            let _ = tx.send(BrowserToRenderer::Shutdown);
        }
        // The tab's renderer is dropped with the tab: its child
        // process is SIGKILLed on drop (see `renderer_process`),
        // so a closed tab leaks no process.
        if self.active_tab >= self.tabs.len() {
            self.active_tab = self.tabs.len() - 1;
        } else if index < self.active_tab {
            self.active_tab -= 1;
        }
        self.sync_window_title();
    }

    /// Post one command to the active tab's worker, if it is
    /// running. A failed send means the worker is gone; the
    /// subscription restarts it and `RendererReady` re-syncs state.
    fn send_command(&mut self, command: BrowserToRenderer) {
        let tab = self.active_mut();
        if let Some(tx) = &tab.renderer_tx
            && tx.send(command).is_err()
        {
            tab.renderer_tx = None;
        }
    }

    /// Post one command to one tab's worker, by id.
    fn send_command_to(&mut self, tab_id: u64, command: BrowserToRenderer) {
        if let Some(tab) = self.tab_mut(tab_id) {
            if let Some(tx) = &tab.renderer_tx
                && tx.send(command).is_err()
            {
                tab.renderer_tx = None;
            }
        }
    }
    /// Tell a tab's worker the shell's session-history length and
    /// index (Stage 16 Phase 3, Rec 8 — finisher): the worker's
    /// realm mirrors them, so `history.length` and the `popstate`
    /// guard agree with the shell after every push / replace /
    /// traversal. No-op when the tab has no renderer yet.
    fn send_history_sync(&mut self, tab_id: u64) {
        if let Some(tab) = self.tab_mut(tab_id) {
            if let Some(tx) = &tab.renderer_tx {
                let _ = tx.send(BrowserToRenderer::HistorySync {
                    length: tab.history.len(),
                    index: tab.history_index,
                });
            }
        }
    }

    /// Record a theme change and forward it to every tab's worker
    /// as `SetTheme` (M6 Phase 3, Rec 12). A repeat of the current
    /// theme is not re-sent (both hooks can fire for one switch).
    /// The workers restyle against the new theme — never a refetch.
    fn apply_theme(&mut self, dark: bool) {
        if dark == self.dark_theme {
            return;
        }
        self.dark_theme = dark;
        for tab in &mut self.tabs {
            if let Some(tx) = &tab.renderer_tx {
                let _ = tx.send(BrowserToRenderer::SetTheme { dark });
            }
        }
    }

    /// Start a navigation in the active tab: update the address
    /// bar and status, reset the scroll, and hand the URL to the
    /// worker — or hold it until the worker starts. The old page
    /// stays visible until the new list lands: the holder keeps
    /// the last-good list.
    fn commit(&mut self, url: String) {
        self.commit_tab(self.active_tab, url);
    }

    /// Start a navigation in tab `tab_index` (Stage 16 Phase 3,
    /// Rec 8 — finisher): the same commit, addressed at one tab,
    /// so a background tab's JS traversal (`HistoryGo`) navigates
    /// that tab instead of the active one.
    ///
    /// A Back/Forward commit (`history_nav`, set by
    /// `traverse_to` before calling here) is a traversal — the
    /// target entry's `history.state` JSON rides in
    /// `CommitNavigation` and the worker restores it and fires
    /// `popstate` after the commit; a reload of the current entry
    /// restores its state and its scroll too. A fresh navigation
    /// carries `None` / `false` and resets the scroll. The scroll
    /// to restore is stashed in `pending_scroll_restore` and
    /// applied when the new list lands (its max-scroll is known
    /// only then).
    fn commit_tab(&mut self, tab_index: usize, url: String) {
        // Stage 13 Rec 12c: restart policy — a dead tab (three
        // crashes in 60 s) accepts no navigation except Reload,
        // which clears `dead` before calling here.
        if self.tabs[tab_index].dead {
            self.tabs[tab_index].status =
                "Renderer crashed repeatedly — press Reload to retry".to_string();
            return;
        }
        self.tabs[tab_index].crashed = false;
        // Stage 13 Rec 11: the page being left is the referrer
        // for the next navigation's document fetch (the broker's
        // loader holds the referrer context in broker mode).
        let prev_url = self.tabs[tab_index].active_url.clone();
        self.broker.loader().set_referrer_document(&prev_url);
        // Stage 16 Phase 3, Rec 8 (finisher): what the worker
        // restores. A traversal takes the target entry's state and
        // scroll; a reload takes the current entry's; a fresh
        // navigation takes neither. The page being left keeps its
        // scroll in its entry — except on a traversal, where
        // `traverse_to` already saved it before moving the index.
        // A stale `history_nav` (a traversal whose `DidCommit`
        // never arrived) only counts when the URL is the entry
        // the index points at — anything else is fresh.
        let (state_json, traversal, restore_scroll) = {
            let tab = &mut self.tabs[tab_index];
            let is_traversal = tab.history_nav
                && !tab.history.is_empty()
                && tab.history[tab.history_index].url == url;
            if is_traversal {
                let entry = &tab.history[tab.history_index];
                (entry.state_json.clone(), true, Some(entry.scroll))
            } else {
                tab.history_nav = false;
                if !tab.history.is_empty() {
                    let idx = tab.history_index;
                    tab.history[idx].scroll = tab.compositor.scroll();
                }
                if !tab.history.is_empty() && tab.history[tab.history_index].url == url {
                    let entry = &tab.history[tab.history_index];
                    (entry.state_json.clone(), false, Some(entry.scroll))
                } else {
                    (None, false, None)
                }
            }
        };
        let has_renderer = {
            let tab = &mut self.tabs[tab_index];
            tab.active_url = url.clone();
            tab.url_input = url.clone();
            tab.title = url.clone();
            match restore_scroll {
                Some(scroll) => tab.pending_scroll_restore = Some(scroll),
                None => tab.compositor.reset_scroll(),
            }
            // A navigation expects a new display list: until it
            // activates, frames presenting the old list count as
            // deadline-missed (Stage 8 Rec 2).
            tab.compositor.note_expected_list();
            // Perf (Stage 8 Rec 5): the page being left gets its
            // summary line first, then the new navigation starts the
            // clock for first paint / first contentful paint.
            eprintln!("{}", perf::summary());
            if let Ok(mut counters) = perf::global().lock() {
                counters.note_navigation_start();
            }
            tab.status = format!("Loading {url}…");
            tab.progress = None;
            tab.renderer_tx.is_some()
        };
        if has_renderer {
            let tab_id = self.tabs[tab_index].id;
            self.send_command_to(
                tab_id,
                BrowserToRenderer::CommitNavigation {
                    url,
                    state_json,
                    traversal,
                    // A traversal / reload restores the entry's
                    // scroll; a fresh navigation starts at the top.
                    scroll: restore_scroll.unwrap_or((0.0, 0.0)),
                },
            );
        } else {
            self.tabs[tab_index].pending_url = Some(url);
        }
    }

    /// Move to the history entry at `index` in tab `tab_index`
    /// (Back/Forward, a History-panel click, or a JS
    /// `HistoryGo`): save the scroll of the entry being left,
    /// move the index, and commit with the target entry's state
    /// and scroll (a traversal). A no-op when the index is out
    /// of range, or when the index is already current.
    fn traverse_to_tab(&mut self, tab_index: usize, index: usize) {
        let url = {
            let tab = &mut self.tabs[tab_index];
            if index >= tab.history.len() || index == tab.history_index {
                return;
            }
            if !tab.history.is_empty() {
                let idx = tab.history_index;
                tab.history[idx].scroll = tab.compositor.scroll();
            }
            tab.history_index = index;
            tab.history_nav = true;
            tab.history[index].url.clone()
        };
        self.commit_tab(tab_index, url);
    }

    /// Move the ACTIVE tab to the history entry at `index`.
    fn traverse_to(&mut self, index: usize) {
        self.traverse_to_tab(self.active_tab, index);
    }

    /// The hamburger menu panel: one text button per working
    /// item (see `menu_items`). It renders in-column below the
    /// nav bar while open — the same list with the same actions
    /// a floating popover would show.
    fn menu_panel(&self) -> Element<'_, Message> {
        widget::column([
            widget::button::text("Bookmarks")
                .on_press(Message::ShowBookmarks)
                .width(Length::Fill)
                .into(),
            widget::button::text("History")
                .on_press(Message::ShowHistory)
                .width(Length::Fill)
                .into(),
            widget::button::text("New Tab")
                .on_press(Message::NewTab)
                .width(Length::Fill)
                .into(),
            widget::button::text("New Window")
                .on_press(Message::NewWindow)
                .width(Length::Fill)
                .into(),
        ])
        .spacing(4)
        .padding(8)
        .into()
    }

    /// The open Bookmarks/History list panel: a header row plus
    /// one row per entry. Bookmark rows navigate on click and
    /// carry a Remove button; history rows navigate on click
    /// (most recent first, capped — the session history itself
    /// is unbounded). An empty list says so instead of showing
    /// a blank panel.
    fn list_panel(&self, panel: Panel) -> Element<'_, Message> {
        let title = match panel {
            Panel::Bookmarks => "Bookmarks",
            Panel::History => "History",
        };
        let mut children: Vec<Element<'_, Message>> = vec![
            widget::row([
                widget::text(title).into(),
                widget::button::text("Close")
                    .on_press(Message::ClosePanel)
                    .into(),
            ])
            .spacing(8)
            .padding(4)
            .into(),
        ];
        match panel {
            Panel::Bookmarks => {
                if self.bookmarks.bookmarks().is_empty() {
                    children.push(
                        widget::text("No bookmarks yet — use the star to save this page.")
                            .into(),
                    );
                }
                for bookmark in self.bookmarks.bookmarks() {
                    let url = bookmark.url.clone();
                    children.push(
                        widget::row([
                            widget::button::text(bookmark.title.clone())
                                .on_press(Message::NavigateTo(url.clone()))
                                .width(Length::Fill)
                                .into(),
                            widget::button::text("Remove")
                                .on_press(Message::RemoveBookmark(url))
                                .into(),
                        ])
                        .spacing(8)
                        .into(),
                    );
                }
            }
            Panel::History => {
                let history = &self.active().history;
                if history.is_empty() {
                    children.push(
                        widget::text("No history yet — pages you visit will appear here.")
                            .into(),
                    );
                }
                for entry in history.iter().rev().take(HISTORY_PANEL_LIMIT) {
                    children.push(
                        widget::button::text(entry.url.clone())
                            .on_press(Message::NavigateTo(entry.url.clone()))
                            .width(Length::Fill)
                            .into(),
                    );
                }
            }
        }
        widget::column(children).spacing(4).padding(8).into()
    }
}

/// Which list panel the hamburger menu opened, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Bookmarks,
    History,
}

/// Most history entries the History panel shows (most recent
/// first): the session history is unbounded in memory, but the
/// panel is a plain list, so it caps at the newest 50.
pub const HISTORY_PANEL_LIMIT: usize = 50;

/// The hamburger menu's items, in order. The menu lists ONLY
/// items with a working backend behind them (Joel's rule): each
/// entry here names an action the shell really performs —
/// Bookmarks opens the persistent bookmark list, History opens
/// the session-history list, New Tab opens a tab in this window
/// (Stage 11 Rec 4), New Window spawns a second Pluto process.
/// Nothing else from the reference screenshot qualifies
/// (no download-prompt UI, no settings / zoom / find / sync /
/// profiles / passwords / extensions / print / save-page /
/// translate backend, and Porn Mode is M12, not built), so none
/// of it is listed. No shortcuts are shown next to the items:
/// none of the screenshot's shortcuts are wired.
pub fn menu_items() -> &'static [&'static str] {
    &["Bookmarks", "History", "New Tab", "New Window"]
}

/// The command a New Window spawns: this executable, with no
/// arguments — a bare launch opens the default start page in its
/// own window. A separate process is a real second window, not a
/// tab (New Tab opens a tab in this window instead).
fn new_window_command() -> std::io::Result<std::process::Command> {
    Ok(std::process::Command::new(std::env::current_exe()?))
}

/// Spawn a new Pluto window (see `new_window_command`).
fn spawn_new_window() -> std::io::Result<()> {
    new_window_command()?.spawn().map(|_| ())
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
        // Stage 10 Rec 8: the persisted history (newest 50) is
        // the first tab's starting session history — the History
        // panel simply sees it pre-populated. The index starts on
        // the newest entry, so Back from a fresh launch goes to
        // the previous visit.
        // Stage 13 Rec 11: the broker owns the shared loader —
        // load the persisted HSTS store and cookie jar into it
        // at startup (session cookies were stripped before the
        // last save, so nothing stale comes back).
        let broker_loader = Arc::new(pluto_net::ResourceLoader::new());
        crate::security::load_into(&broker_loader);
        crate::cookies::load_into(&broker_loader);
        let broker = Arc::new(crate::broker::NetworkBroker::new(broker_loader));
        let history = load_history();
        let history_index = history.len().saturating_sub(1);
        let mut first_tab = Tab::new(0, url.clone());
        // Persisted history stores URLs only (Stage 10 Rec 8) —
        // restored entries have no state and no scroll.
        first_tab.history = history.into_iter().map(HistoryEntry::new).collect();
        first_tab.history_index = history_index;
        let app = Self {
            core,
            tabs: vec![first_tab],
            active_tab: 0,
            next_tab_id: 1,
            viewport: page::DEFAULT_VIEWPORT,
            visible: true,
            // M6 Phase 3 (Rec 12): the real OS theme at startup
            // — libcosmic reads the system theme; the workers get
            // it on `RendererReady` and on every change after.
            dark_theme: cosmic::theme::is_dark(),
            bookmarks: load_bookmarks(),
            menu_open: false,
            panel: None,
            modifiers: pluto_ipc::Modifiers::default(),
            buttons_held: 0,
            last_wheel: None,
            broker,
        };
        (app, Task::none())
    }

    /// The OS theme mode changed (M6 Phase 3, Rec 12): libcosmic
    /// calls this when the system light/dark mode flips. The mode
    /// carries the dark flag directly — the authoritative signal
    /// for `prefers-color-scheme`.
    fn system_theme_mode_update(
        &mut self,
        _keys: &[&'static str],
        new_theme: &cosmic::cosmic_theme::ThemeMode,
    ) -> Task<Self::Message> {
        self.apply_theme(new_theme.is_dark);
        Task::none()
    }

    /// The OS theme itself changed (M6 Phase 3, Rec 12): a theme
    /// switch that keeps the mode (e.g. a new light theme) still
    /// re-reads the active theme — libcosmic installs it before
    /// calling this hook, so `is_dark()` sees the new one.
    fn system_theme_update(
        &mut self,
        _keys: &[&'static str],
        _new_theme: &cosmic::cosmic_theme::Theme,
    ) -> Task<Self::Message> {
        self.apply_theme(cosmic::theme::is_dark());
        Task::none()
    }

    /// The renderer worker subscriptions (Stage 4 #2, Stage 11
    /// Rec 4): one per tab, keyed by the shell-assigned tab id —
    /// plus the window focus subscription (Stage 8 Rec 1), which
    /// is the shell's visibility signal for the workers, and the
    /// keyboard subscription (Stage 8 Rec 9), which forwards page
    /// keys. The page pipeline runs in the workers, off the UI
    /// thread.
    fn subscription(&self) -> cosmic::iced::Subscription<Self::Message> {        let mut subscriptions: Vec<cosmic::iced::Subscription<Self::Message>> = self
            .tabs
            .iter()
            .map(|tab| renderer::renderer_subscription_for_tab(tab.id))
            .collect();
        subscriptions.push(cosmic::iced::event::listen_with(window_visibility));
        subscriptions.push(cosmic::iced::event::listen_with(keyboard_event));
        cosmic::iced::Subscription::batch(subscriptions)
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::UrlChanged(val) => self.active_mut().url_input = val,
            Message::Navigate => {
                let input = self.active().url_input.clone();
                match page::normalize_url(&input) {
                    Ok(url) => self.commit(url),
                    Err(reason) => {
                        // Stage 9 Rec 7 (T11): a disallowed scheme
                        // shows the error page, never a fetch. The
                        // shell cannot render the error page itself,
                        // so the raw input is committed and the
                        // worker-side scheme guard in `load_page`
                        // rejects it before any fetch — the renderer
                        // then renders the error page with this
                        // reason (`LoadOutcome::Failed`).
                        eprintln!("[pluto] blocked navigation: {reason}");
                        self.commit(input.trim().to_string());
                    }
                }
            }
            Message::Reload => {
                // Stage 13 Rec 12c: Reload is the one navigation
                // that may revive a dead tab — clear the restart
                // policy state before committing.
                let tab = self.active_mut();
                tab.dead = false;
                tab.crashed = false;
                let url = tab.active_url.clone();
                self.commit(url);
            }
            Message::NewTab => {
                self.menu_open = false;
                self.new_tab(page::DEFAULT_URL.to_string());
            }
            Message::CloseTab(index) => {
                self.close_tab(index);
            }
            Message::SelectTab(index) => {
                if index < self.tabs.len() && index != self.active_tab {
                    self.active_tab = index;
                    // The newly active tab's compositor is already
                    // holding its last-good list — switching shows
                    // it at once, with no re-fetch and no re-layout.
                    // Its worker already knows the window size
                    // (resizes go to every tab), so there is
                    // nothing to re-sync here. Its stored video
                    // frames convert now (they arrived as YUV
                    // while the tab was in the background).
                    self.convert_stored_video_frames(index);
                    self.sync_window_title();
                }
            }
            Message::RendererReady(tab_id, tx) => {
                // One tab's worker is up: sync its viewport first
                // (it starts at the default size), then its
                // visibility (a restarted worker defaults to
                // visible), then flush the pending navigation. A
                // ready for a closed tab is ignored.
                let viewport = self.viewport;
                let visible = self.visible;
                let pending = {
                    let Some(tab) = self.tab_mut(tab_id) else {
                        return Task::none();
                    };
                    tab.renderer_tx = Some(tx);
                    tab.pending_url.take()
                };
                self.send_command_to(
                    tab_id,
                    BrowserToRenderer::Resize {
                        width: viewport.0,
                        height: viewport.1,
                    },
                );
                self.send_command_to(tab_id, BrowserToRenderer::SetVisible { visible });
                self.send_command_to(tab_id, BrowserToRenderer::SetFocused { focused: visible });
                // M6 Phase 3 (Rec 12): a (re)started worker gets
                // the current OS theme, so its first layout
                // evaluates `prefers-color-scheme` correctly.
                let dark = self.dark_theme;
                self.send_command_to(tab_id, BrowserToRenderer::SetTheme { dark });
                if let Some(url) = pending {
                    // The first navigation expects a list too, and
                    // starts the first-paint clock (no previous
                    // page to summarise yet).
                    if let Some(tab) = self.tab_mut(tab_id) {
                        tab.compositor.note_expected_list();
                    }
                    if let Ok(mut counters) = perf::global().lock() {
                        counters.note_navigation_start();
                    }
                    self.send_command_to(
                        tab_id,
                        BrowserToRenderer::CommitNavigation {
                            url,
                            state_json: None,
                            traversal: false,
                            scroll: (0.0, 0.0),
                        },
                    );
                }
            }
            Message::WindowFocused(visible) => {
                // Stage 8 Rec 1 + Rec 9: forward a real visibility
                // change to EVERY tab's worker; a repeat of the
                // current state is not re-sent (iced may deliver
                // focus events more than once per change). The
                // same event sets each worker's keyboard focus — a
                // single-window shell has no separate focus target,
                // so focus and visibility move together.
                if visible != self.visible {
                    self.visible = visible;
                    for tab in &mut self.tabs {
                        if let Some(tx) = &tab.renderer_tx {
                            let _ = tx.send(BrowserToRenderer::SetVisible { visible });
                            let _ = tx.send(BrowserToRenderer::SetFocused { focused: visible });
                        }
                    }
                }
            }
            Message::RendererEvent(tab_id, event) => {
                let Some(tab_index) = self.tab_index(tab_id) else {
                    // An event for a closed tab: ignored.
                    return Task::none();
                };
                let is_active = tab_index == self.active_tab;
                match event {
                    RendererToBrowser::DidCommit { url, status, replace } => {
                        eprintln!("[pluto] page loaded: {url}");
                        // Stage 13 Rec 12c: the crash page commits
                        // like any page, but it must not touch the
                        // URL, title, or history — the address bar
                        // keeps the real URL, and the crash status
                        // stays until Reload.
                        if url == "about:crashed" {
                            let tab = &mut self.tabs[tab_index];
                            tab.progress = None;
                            return Task::none();
                        }
                        let tab = &mut self.tabs[tab_index];
                        tab.active_url = url.clone();
                        tab.url_input = url.clone();
                        tab.title = url.clone();
                        tab.status = status;
                        tab.progress = None;
                        tab.crashed = false;
                        tab.dead = false;
                        // History (R17): a back/forward commit only
                        // moves the index (already done in Back/Forward);
                        // a `location.replace()` commit rewrites the
                        // current entry in place (Stage 16 Phase 3,
                        // Rec 8 — finisher); any other commit — address
                        // bar, reload, link click — pushes a new entry
                        // after truncating the forward entries, unless
                        // it is the entry we are already on (a reload).
                        // A replace or push changes the length, so the
                        // worker is re-synced after either.
                        let mut history_changed = false;
                        if tab.history_nav {
                            tab.history_nav = false;
                        } else if replace {
                            if let Some(entry) = tab.history.get_mut(tab.history_index) {
                                entry.url = url.clone();
                                entry.state_json = None;
                                entry.scroll = (0.0, 0.0);
                            } else {
                                tab.history.push(HistoryEntry::new(url.clone()));
                                tab.history_index = tab.history.len() - 1;
                            }
                            history_changed = true;
                        } else if tab.history.is_empty()
                            || tab.history[tab.history_index].url != url
                        {
                            tab.history.truncate(tab.history_index + 1);
                            tab.history.push(HistoryEntry::new(url.clone()));
                            tab.history_index = tab.history.len() - 1;
                            history_changed = true;
                        }
                        if history_changed {
                            // Stage 10 Rec 8: persist on every push
                            // of the ACTIVE tab only (the profile
                            // pattern: save on every mutation, like
                            // bookmarks) — a background tab's history
                            // does not overwrite the file.
                            if is_active {
                                let urls: Vec<String> = tab
                                    .history
                                    .iter()
                                    .map(|entry| entry.url.clone())
                                    .collect();
                                save_history(&urls);
                            }
                            let tab_id = tab.id;
                            drop(tab);
                            self.send_history_sync(tab_id);
                        }
                        if is_active {
                            self.sync_window_title();
                        }
                    }
                    RendererToBrowser::SubmitDisplayList {
                        epoch,
                        bytes,
                        resources: _,
                    } => {
                        // The list crosses as postcard bytes even
                        // in-process — the same bytes that cross IPC at
                        // M5. Decode and validate before installing: a
                        // corrupt or out-of-bounds list never reaches
                        // the GPU. Stage 11 Rec 1: the submit's own
                        // `resources` field travels EMPTY — the list
                        // validates against the tab's accumulated
                        // table, built from `ResourceReady` frames.
                        match DisplayList::from_bytes(&bytes) {
                            Ok(list) => {
                                let tab = &mut self.tabs[tab_index];
                                let resources = Arc::clone(&tab.resources);
                                match list.validate(&resources) {
                                    Ok(()) => {
                                        if tab.compositor.submit(epoch, Arc::new(list), resources)
                                        {
                                            // Pending, not active yet: the
                                            // next frame boundary activates
                                            // it (Stage 8 Rec 2).
                                            eprintln!(
                                                "[pluto] display list pending: epoch {epoch} (active {})",
                                                tab.compositor.epoch()
                                            );
                                        }
                                        // Stage 16 Phase 3, Rec 8
                                        // (finisher): the traversed /
                                        // reloaded page's scroll, restored
                                        // once the new list's max-scroll
                                        // is known — the clamp here is
                                        // against the NEW page's bounds.
                                        if let Some((x, y)) =
                                            tab.pending_scroll_restore.take()
                                        {
                                            tab.compositor.set_scroll(x, y);
                                        }
                                    }
                                    Err(err) => eprintln!(
                                        "[pluto] ignoring invalid display list (epoch {epoch}): {err:?}"
                                    ),
                                }
                            }
                            Err(err) => eprintln!(
                                "[pluto] ignoring undecodable display list (epoch {epoch}): {err:?}"
                            ),
                        }
                    }
                    RendererToBrowser::CompositingUpdate { epoch, updates } => {
                        // M6 Phase 2 (Rec 9): a transform/opacity-only
                        // change — patch the submitted list in place
                        // instead of installing a new one. The
                        // compositor targets the active list, or the
                        // pending one when it has not activated yet;
                        // a newer pending list already carries the
                        // new values, so the update is stale then.
                        let tab = &mut self.tabs[tab_index];
                        let patched =
                            tab.compositor.apply_compositing_update(epoch, &updates);
                        if patched > 0 {
                            eprintln!(
                                "[pluto] compositing update: {patched} items (epoch {epoch})"
                            );
                        }
                    }
                    RendererToBrowser::ResourceReady {
                        epoch: _,
                        resource_id,
                        resource,
                    } => {
                        // One decoded resource, out-of-band (Stage
                        // 11 Rec 1): accumulate it into the tab's
                        // table, so the next submit validates
                        // against it. The `epoch` is the navigation
                        // epoch — staleness is enforced worker-side
                        // (the worker drops stale outcomes before
                        // sending); the shell inserts what arrives.
                        // `Arc::make_mut` clones the table only
                        // when a submitted list still holds it.
                        let tab = &mut self.tabs[tab_index];
                        Arc::make_mut(&mut tab.resources)
                            .resources
                            .insert(resource_id, resource);
                    }
                    RendererToBrowser::VideoFrame { epoch: _, frame } => {
                        // One decoded video frame, out-of-band
                        // (Stage 16 Phase 1, Rec 1): keep only the
                        // latest frame per element — a new frame
                        // overwrites the old one, so the shell
                        // never holds a backlog. Stage 16 Phase 2
                        // (Rec 5): the ACTIVE tab's frame is
                        // converted to RGBA at once and inserted
                        // under its deterministic blob id; a
                        // background tab's frame waits as YUV
                        // until the tab is selected (see
                        // `convert_stored_video_frames`), so an
                        // unseen tab never burns conversion time.
                        let is_active = tab_index == self.active_tab;
                        let tab = &mut self.tabs[tab_index];
                        let element_id = frame.element_id;
                        tab.video_frames.insert(element_id, frame);
                        if is_active {
                            self.convert_video_frame(tab_index, element_id);
                        }
                    }
                    RendererToBrowser::LoadProgress { received, total } => {
                        // Stage 11 Rec 9: document-load progress —
                        // the thin bar under the nav bar (see
                        // `view`) plus the status text. Cleared on
                        // the next commit.
                        let tab = &mut self.tabs[tab_index];
                        tab.progress = Some((received, total));
                    }
                    RendererToBrowser::TitleChanged { title } => {
                        // Stage 11 Rec 8: the tab's title, shown
                        // in the tab strip — and the window title,
                        // when this is the active tab.
                        let tab = &mut self.tabs[tab_index];
                        tab.title = title;
                        if is_active {
                            self.sync_window_title();
                        }
                    }
                    RendererToBrowser::FaviconChanged { resource_id } => {
                        // Stage 11 Rec 7: the tab's favicon, shown
                        // in the tab strip (`None` = default icon).
                        self.tabs[tab_index].favicon_id = resource_id;
                    }
                    RendererToBrowser::LoadFailed { reason, .. } => {
                        // The worker already submitted the error page as
                        // a display list; the status line says why.
                        let tab = &mut self.tabs[tab_index];
                        tab.status = format!("Failed: {reason}");
                        tab.progress = None;
                    }
                    RendererToBrowser::CancelResource { request_id } => {
                        // Stage 13 Rec 11: the fetch lives
                        // browser-side behind the broker — cancel it
                        // on the broker's scheduler.
                        self.broker.cancel(request_id);
                    }
                    RendererToBrowser::SandboxReport { layers } => {
                        // Stage 13 Rec 12a: record the per-layer
                        // results; warn in the status line when a
                        // load-bearing layer (user-namespace,
                        // chroot, seccomp) is OFF — the tab still
                        // loads, but the user should know the
                        // sandbox is degraded.
                        let off: Vec<&str> = layers
                            .iter()
                            .filter(|(name, ok)| {
                                !*ok && matches!(
                                    name.as_str(),
                                    "user-namespace" | "chroot" | "seccomp"
                                )
                            })
                            .map(|(name, _)| name.as_str())
                            .collect();
                        if !off.is_empty() {
                            eprintln!(
                                "[pluto] sandbox degraded on tab {}: {} OFF",
                                tab_id,
                                off.join(", ")
                            );
                            let tab = &mut self.tabs[tab_index];
                            tab.status =
                                format!("Sandbox degraded: {} OFF", off.join(", "));
                        }
                        self.tabs[tab_index].sandbox_layers = layers;
                    }
                    RendererToBrowser::Heartbeat => {
                        // Stage 13 Rec 12b: the watchdog lives in
                        // the bridge (`run_renderer_bridge`), which
                        // sees every event including heartbeats —
                        // the shell needs nothing here.
                    }
                    RendererToBrowser::RendererCrashed { reason } => {
                        // Stage 13 Rec 12c: the sad tab. NOTE the
                        // `renderer_tx` is KEPT: dropping it would
                        // close the bridge's command channel and
                        // the bridge would exit, so no later
                        // navigation could ever respawn the
                        // renderer (that was the pre-M5 behaviour —
                        // "the next navigation recovers" was a
                        // broken promise). The bridge stays alive
                        // with no renderer and respawns on the
                        // next `CommitNavigation`.
                        eprintln!("[pluto] renderer crashed: {reason}");
                        let now = std::time::Instant::now();
                        let tab = &mut self.tabs[tab_index];
                        tab.crashed = true;
                        tab.crash_reason = reason.clone();
                        tab.crash_times.retain(|t| now.duration_since(*t).as_secs() < 60);
                        tab.crash_times.push(now);
                        // Restart policy: three crashes in 60 s
                        // makes the tab dead — only Reload may
                        // navigate it until the user reloads.
                        if tab.crash_times.len() >= 3 {
                            tab.dead = true;
                            tab.status = "Renderer crashed repeatedly — press Reload to retry"
                                .to_string();
                        } else {
                            tab.status =
                                format!("Renderer crashed: {reason} — press Reload to retry");
                        }
                        tab.progress = None;
                        // Show the crash page: the bridge respawns
                        // a fresh renderer for this navigation, and
                        // the worker renders `about:crashed`
                        // locally (no fetch). The resize/visibility
                        // re-sync follows so the fresh child lays
                        // the page out at the real window size.
                        self.send_command_to(
                            tab_id,
                            BrowserToRenderer::CommitNavigation {
                                url: "about:crashed".to_string(),
                                state_json: None,
                                traversal: false,
                                scroll: (0.0, 0.0),
                            },
                        );
                        let viewport = self.viewport;
                        self.send_command_to(
                            tab_id,
                            BrowserToRenderer::Resize {
                                width: viewport.0,
                                height: viewport.1,
                            },
                        );
                        let visible = self.visible;
                        self.send_command_to(
                            tab_id,
                            BrowserToRenderer::SetVisible { visible },
                        );
                        self.send_command_to(
                            tab_id,
                            BrowserToRenderer::SetFocused { focused: visible },
                        );
                        let dark = self.dark_theme;
                        self.send_command_to(tab_id, BrowserToRenderer::SetTheme { dark });
                    }
                    RendererToBrowser::ScrollViewport { dx, dy } => {
                        // Stage 8 Rec 9: the worker owns the wheel —
                        // containers first, and only the leftover delta
                        // comes back here as the viewport scroll.
                        self.tabs[tab_index].compositor.scroll_by(dx, dy);
                    }
                    RendererToBrowser::RequestResource(request) => {
                        // Stage 13 Rec 11: serve the request on the
                        // broker and forward its events to the tab as
                        // they arrive. The future runs serve + the
                        // forward loop itself (no extra spawn): the
                        // events channel closes when the fetch ends.
                        let Some(tx) = self.tabs[tab_index].renderer_tx.clone() else {
                            return Task::none();
                        };
                        let broker = Arc::clone(&self.broker);
                        return Task::perform(
                            async move {
                                let (events_tx, mut events_rx) =
                                    tokio::sync::mpsc::channel(64);
                                let serve = broker.serve(request, events_tx);
                                tokio::pin!(serve);
                                loop {
                                    tokio::select! {
                                        _ = &mut serve => break,
                                        event = events_rx.recv() => match event {
                                            Some(event) => {
                                                if tx.send(BrowserToRenderer::ResourceEvent(event))
                                                    .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            None => break,
                                        },
                                    }
                                }
                                Message::BrokerServeDone
                            },
                            |message| cosmic::Action::App(message),
                        );
                    }
                    RendererToBrowser::HistoryPush { url, state_json } => {
                        // Stage 16 Phase 3, Rec 8 (finisher): the
                        // page's `pushState` — truncate the forward
                        // entries, append the new entry, update the
                        // address bar. The page being left keeps its
                        // scroll in its entry. The realm already
                        // updated its own length/index optimistically;
                        // `HistorySync` corrects it to the shell's.
                        let (tab_id, persist) = {
                            let tab = &mut self.tabs[tab_index];
                            if !tab.history.is_empty() {
                                let idx = tab.history_index;
                                tab.history[idx].scroll = tab.compositor.scroll();
                            }
                            tab.history.truncate(tab.history_index + 1);
                            tab.history.push(HistoryEntry {
                                url: url.clone(),
                                state_json,
                                scroll: (0.0, 0.0),
                            });
                            tab.history_index = tab.history.len() - 1;
                            tab.active_url = url.clone();
                            tab.url_input = url.clone();
                            let persist = if is_active {
                                Some(
                                    tab.history
                                        .iter()
                                        .map(|entry| entry.url.clone())
                                        .collect::<Vec<String>>(),
                                )
                            } else {
                                None
                            };
                            (tab.id, persist)
                        };
                        if let Some(urls) = persist {
                            save_history(&urls);
                        }
                        if is_active {
                            self.sync_window_title();
                        }
                        self.send_history_sync(tab_id);
                    }
                    RendererToBrowser::HistoryReplace { url, state_json } => {
                        // `replaceState`: the current entry takes the
                        // new URL and state in place — no new entry,
                        // no truncation, no navigation. Empty history
                        // (a replace before the first commit) pushes.
                        let (tab_id, persist) = {
                            let tab = &mut self.tabs[tab_index];
                            if let Some(entry) = tab.history.get_mut(tab.history_index) {
                                entry.url = url.clone();
                                entry.state_json = state_json;
                            } else {
                                tab.history.push(HistoryEntry {
                                    url: url.clone(),
                                    state_json,
                                    scroll: (0.0, 0.0),
                                });
                                tab.history_index = tab.history.len() - 1;
                            }
                            tab.active_url = url.clone();
                            tab.url_input = url.clone();
                            let persist = if is_active {
                                Some(
                                    tab.history
                                        .iter()
                                        .map(|entry| entry.url.clone())
                                        .collect::<Vec<String>>(),
                                )
                            } else {
                                None
                            };
                            (tab.id, persist)
                        };
                        if let Some(urls) = persist {
                            save_history(&urls);
                        }
                        if is_active {
                            self.sync_window_title();
                        }
                        self.send_history_sync(tab_id);
                    }
                    RendererToBrowser::HistoryGo { delta } => {
                        // `history.back()` / `forward()` / `go(n)`
                        // from JS: the shell moves the index (clamped)
                        // and navigates with a traversal commit — the
                        // worker restores the entry's state and fires
                        // `popstate`. A delta that moves nowhere is a
                        // no-op (no message back).
                        if delta != 0 {
                            let target = {
                                let tab = &self.tabs[tab_index];
                                let len = tab.history.len() as i32;
                                let idx = tab.history_index as i32 + delta;
                                idx.clamp(0, len.saturating_sub(1)) as usize
                            };
                            self.traverse_to_tab(tab_index, target);
                            let tab_id = self.tabs[tab_index].id;
                            self.send_history_sync(tab_id);
                        }
                    }
                    RendererToBrowser::WebSocketOpen {
                        socket_id,
                        url,
                        protocols,
                        origin,
                    } => {
                        // Stage 16 Phase 3, Rec 11 (finisher): the
                        // worker asked for a socket — run the pump on
                        // the broker (the renderer never opens a
                        // socket itself) and forward its events to
                        // this tab's renderer as they arrive. The
                        // pump's command sender is kept per tab so
                        // `WebSocketSend` / `WebSocketClose` reach it.
                        let (commands_tx, commands_rx) = tokio::sync::mpsc::channel(64);
                        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
                        self.tabs[tab_index].ws_commands.insert(socket_id, commands_tx);
                        let Some(tx) = self.tabs[tab_index].renderer_tx.clone() else {
                            return Task::none();
                        };
                        let broker = Arc::clone(&self.broker);
                        return Task::perform(
                            async move {
                                let serve = broker.serve_websocket(
                                    socket_id, url, protocols, origin, commands_rx, events_tx,
                                );
                                tokio::pin!(serve);
                                loop {
                                    tokio::select! {
                                        _ = &mut serve => break,
                                        event = events_rx.recv() => match event {
                                            Some(event) => {
                                                if tx.send(BrowserToRenderer::WebSocketEvent {
                                                    socket_id,
                                                    event,
                                                })
                                                .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            None => break,
                                        },
                                    }
                                }
                                Message::WebSocketDone(socket_id)
                            },
                            |message| cosmic::Action::App(message),
                        );
                    }
                    RendererToBrowser::WebSocketSend {
                        socket_id,
                        data,
                        is_binary,
                    } => {
                        // Forward to the pump's command channel; a
                        // closed or unknown socket drops the frame
                        // (the worker already treats a closed socket
                        // as an error on its next operation).
                        if let Some(commands) = self.tabs[tab_index].ws_commands.get(&socket_id)
                        {
                            let _ = commands.try_send(crate::broker::WsCommand::Send {
                                data,
                                is_binary,
                            });
                        }
                    }
                    RendererToBrowser::WebSocketClose {
                        socket_id,
                        code,
                        reason,
                    } => {
                        if let Some(commands) = self.tabs[tab_index].ws_commands.get(&socket_id)
                        {
                            let _ = commands.try_send(crate::broker::WsCommand::Close {
                                code,
                                reason,
                            });
                        }
                    }
                }
            }
            Message::BrokerServeDone => {}
            Message::WebSocketDone(socket_id) => {
                // The pump is gone: drop its command sender from
                // whichever tab held it (a closed tab's map is
                // already gone with the tab).
                for tab in &mut self.tabs {
                    tab.ws_commands.remove(&socket_id);
                }
            }
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
                // A resize re-lays out, so a new list is expected —
                // on the active tab, the one presenting frames. The
                // resize itself goes to EVERY tab's worker: they all
                // share the window size, and a background tab must
                // not keep a stale viewport for when it is selected.
                self.active_mut().compositor.note_expected_list();
                for tab in &mut self.tabs {
                    if let Some(tx) = &tab.renderer_tx {
                        let _ = tx.send(BrowserToRenderer::Resize { width, height });
                    }
                }
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
                let (sx, sy) = self.active().compositor.scroll();
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
                let (sx, sy) = self.active().compositor.scroll();
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
                let (sx, sy) = self.active().compositor.scroll();
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
                let (sx, sy) = self.active().compositor.scroll();
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
                // Stage 11 Rec 4: tab shortcuts, intercepted here
                // — never forwarded to the page. Ctrl+T opens a
                // tab, Ctrl+W closes the active tab, Ctrl+Tab
                // cycles to the next tab. Only on press (not
                // release, not auto-repeat).
                if pressed && !repeat && modifiers.ctrl {
                    match key.as_str() {
                        "t" | "T" => {
                            self.new_tab(page::DEFAULT_URL.to_string());
                            return Task::none();
                        }
                        "w" | "W" => {
                            let index = self.active_tab;
                            self.close_tab(index);
                            return Task::none();
                        }
                        "Tab" => {
                            let next = (self.active_tab + 1) % self.tabs.len();
                            if next != self.active_tab {
                                self.active_tab = next;
                                self.convert_stored_video_frames(next);
                                self.sync_window_title();
                            }
                            return Task::none();
                        }
                        _ => {}
                    }
                }
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
                let index = self.active().history_index;
                if index > 0 {
                    self.traverse_to(index - 1);
                }
            }
            Message::Forward => {
                let tab = self.active();
                let index = tab.history_index + 1;
                if index < tab.history.len() {
                    self.traverse_to(index);
                }
            }
            Message::ToggleBookmark => {
                // The star toggles a bookmark for the current
                // page and persists it at once (the profile
                // pattern: save on every mutation). The bookmark
                // title is the URL for now — real page titles
                // from `TitleChanged` are a small follow-up.
                let url = self.active().active_url.clone();
                if !url.is_empty() {
                    self.bookmarks.toggle(&url, &url);
                    save_bookmarks(&self.bookmarks);
                }
            }
            Message::ToggleMenu => {
                self.menu_open = !self.menu_open;
                if self.menu_open {
                    self.panel = None;
                }
            }
            Message::ShowBookmarks => {
                self.menu_open = false;
                self.panel = Some(Panel::Bookmarks);
            }
            Message::ShowHistory => {
                self.menu_open = false;
                self.panel = Some(Panel::History);
            }
            Message::ClosePanel => {
                self.panel = None;
            }
            Message::NavigateTo(url) => {
                self.menu_open = false;
                self.panel = None;
                // A History-panel click is a traversal to that entry
                // when it is one (state + scroll restore, `popstate`
                // after the commit); a bookmark click is a fresh
                // navigation. Repeated URLs pick the newest entry —
                // the panel cannot tell them apart either.
                if let Some(index) = self
                    .active()
                    .history
                    .iter()
                    .rposition(|entry| entry.url == url)
                {
                    self.traverse_to(index);
                } else {
                    self.commit(url);
                }
            }
            Message::RemoveBookmark(url) => {
                if self.bookmarks.remove(&url) {
                    save_bookmarks(&self.bookmarks);
                }
            }
            Message::NewWindow => {
                self.menu_open = false;
                if let Err(e) = spawn_new_window() {
                    eprintln!("[pluto] could not open a new window: {e}");
                    self.active_mut().status = format!("Could not open a new window: {e}");
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
        let active = self.active();
        let star = if self.bookmarks.is_bookmarked(&active.active_url) {
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/star-filled-symbolic.svg"
            )))
        } else {
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/star-symbolic.svg"
            )))
        };
        // The tab strip (Stage 11 Rec 4): one button per tab —
        // favicon (when the site has one) plus title — the active
        // tab's button is marked selected; every tab carries a
        // "×" close button. "+" opens a new tab. Clicking a tab
        // swaps the active compositor with no re-fetch and no
        // re-layout.
        let mut tab_row: Vec<Element<'_, Message>> = Vec::new();
        for (index, tab) in self.tabs.iter().enumerate() {
            let mut tab_children: Vec<Element<'_, Message>> = Vec::new();
            if let Some(favicon_id) = tab.favicon_id
                && let Some(pluto_paint::Resource::Image {
                    width,
                    height,
                    rgba,
                }) = tab.resources.resources.get(&favicon_id)
            {
                // A fresh handle per frame: the favicon is 32x32
                // (4 KiB), so the re-upload is negligible — but
                // the atlas never dedups it, by design of
                // `from_pixels` (a new id per handle).
                let handle = cosmic::iced::widget::image::Handle::from_rgba(
                    *width,
                    *height,
                    rgba.clone(),
                );
                tab_children.push(
                    cosmic::iced::widget::image(handle)
                        .width(Length::Fixed(16.0))
                        .height(Length::Fixed(16.0))
                        .into(),
                );
            }
            tab_children.push(widget::text(tab.title.clone()).into());
            tab_children.push(
                widget::button::text("×")
                    .on_press(Message::CloseTab(index))
                    .into(),
            );
            let tab_button = widget::button::custom(widget::row(tab_children).spacing(4))
                .selected(index == self.active_tab)
                .on_press(Message::SelectTab(index));
            tab_row.push(tab_button.into());
        }
        tab_row.push(
            widget::button::text("+")
                .on_press(Message::NewTab)
                .into(),
        );
        let tab_strip = widget::row(tab_row).spacing(4).padding(4);
        // The bookmark star (filled when the current page is
        // bookmarked) and the hamburger menu, both right after
        // Go. Round icon buttons like the other nav icons; Go
        // stays a text button.
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
            widget::text_input("Enter URL...", &active.url_input)
                .on_input(Message::UrlChanged)
                .on_submit(|_| Message::Navigate)
                .width(Length::Fill)
                .into(),
            widget::button::suggested("Go")
                .on_press(Message::Navigate)
                .into(),
            star.on_press(Message::ToggleBookmark).into(),
            widget::button::icon(bundled_icon(include_bytes!(
                "../assets/icons/open-menu-symbolic.svg"
            )))
            .on_press(Message::ToggleMenu)
            .into(),
        ])
        .spacing(8)
        .padding(8)
        .align_y(Alignment::Center);

        // The viewport draws whatever the compositor holder
        // currently holds; the worker updates it from its thread.
        // Stage 16 (M8 Phase 2, Rec 5): the viewport reads
        // the tab's LIVE resource table, not the compositor's
        // submit-time snapshot — a video frame converted after
        // the last submit must still reach `prepare()` (the
        // table is insert-only, so the live table is always a
        // superset of the submitted one; the display list itself
        // still goes through pending/activate).
        let web_view = cosmic::iced::widget::Shader::new(WebViewport {
            clear_color: [1.0, 1.0, 1.0, 1.0],
            display: active.compositor.display(),
            resources: Arc::clone(&active.resources),
            scroll: active.compositor.scroll(),
            video_dirty: Arc::clone(&active.video_dirty),
        })
        .width(Length::Fill)
        .height(Length::Fill);

        // The status line sits at the bottom of the window, below the
        // page — like a browser status bar — not under the nav bar.
        // While a load is in flight it shows the progress instead
        // of the last status text (Stage 11 Rec 9).
        let status_text = match active.progress {
            Some((received, Some(total))) => {
                format!("Loading… {received} / {total} bytes")
            }
            Some((received, None)) => format!("Loading… {received} bytes"),
            None => active.status.clone(),
        };
        let status = widget::row([widget::text(status_text).size(12).into()]).padding(4);

        // The menu and list panels render in-column below the
        // nav bar while open; the web view and status line sit
        // below them.
        let mut children: Vec<Element<'_, Self::Message>> = vec![
            tab_strip.into(),
            widget::divider::horizontal::default().into(),
            nav_bar.into(),
            widget::divider::horizontal::default().into(),
        ];
        // Stage 11 Rec 9: the thin progress bar under the nav
        // bar, while a document load is in flight. Indeterminate
        // (no Content-Length) shows as a half bar, not a
        // percentage — the status text carries the byte count.
        if let Some((received, total)) = active.progress {
            let fraction = total.map_or(0.5, |t| {
                if t == 0 {
                    1.0
                } else {
                    (received as f32 / t as f32).clamp(0.0, 1.0)
                }
            });
            children.push(
                cosmic::iced::widget::progress_bar(0.0..=1.0, fraction).into(),
            );
        }
        if self.menu_open {
            children.push(self.menu_panel());
            children.push(widget::divider::horizontal::default().into());
        }
        if let Some(panel) = self.panel {
            children.push(self.list_panel(panel));
            children.push(widget::divider::horizontal::default().into());
        }
        children.push(web_view.into());
        children.push(widget::divider::horizontal::default().into());
        children.push(status.into());
        widget::column(children).into()
    }
}

/// Run Pluto: the `--renderer` child if invoked as one, else the
/// libcosmic browser window. `main()` is a thin wrapper over this
/// so integration tests can use the crate as a library (notably to
/// spawn real `--renderer` children via `renderer_process`).
pub fn run() -> cosmic::iced::Result {
    // Stage 9 Rec 9: the `--renderer` child runs headless — no
    // window, no wgpu device, no libcosmic — so it is dispatched
    // before any GUI setup. The child only parses, lays out, runs
    // JS, and builds display lists over the inherited IPC socket.
    if let Some((fd, tab_id)) = renderer_process::parse_renderer_args() {
        std::process::exit(renderer_process::run_renderer_child(fd, tab_id));
    }
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
        let icons: [&[u8]; 6] = [
            include_bytes!("../assets/icons/go-previous-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/go-next-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/view-refresh-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/star-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/star-filled-symbolic.svg") as &[u8],
            include_bytes!("../assets/icons/open-menu-symbolic.svg") as &[u8],
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

    #[test]
    fn menu_lists_only_working_items() {
        // The exact menu contents — and the exclusion list: every
        // screenshot item whose backend does not exist. If a new
        // backend lands, this test is the place that grows the
        // menu (add the item to `menu_items` and the case here).
        // Stage 11 Rec 4: New Tab has a backend now (the
        // multi-tab shell), so it is listed, not excluded.
        assert_eq!(
            menu_items(),
            &["Bookmarks", "History", "New Tab", "New Window"]
        );
        for forbidden in [
            "Sync and save data",
            "Profiles",
            "Porn Mode",
            "Downloads",
            "Passwords",
            "Extensions and Themes",
            "Print",
            "Save Page As",
            "Find in Page",
            "Translate Page",
            "Zoom",
            "Settings",
        ] {
            assert!(
                !menu_items().contains(&forbidden),
                "menu must not list {forbidden} (no working backend)"
            );
        }
    }

    #[test]
    fn new_window_spawns_this_executable() {
        // The New Window action spawns this binary with no args
        // (a bare launch opens the start page in its own window).
        // The test checks the command, not the spawn itself —
        // spawning would open a real window.
        let cmd = new_window_command().expect("current exe resolves");
        assert_eq!(
            cmd.get_program(),
            std::env::current_exe().expect("current exe resolves")
        );
        assert_eq!(cmd.get_args().count(), 0);
    }

    #[test]
    fn history_panel_is_capped_and_most_recent_first() {
        // The panel's slice logic, over a 60-entry session
        // history: the newest 50, newest first.
        let history: Vec<HistoryEntry> = (0..60)
            .map(|i| HistoryEntry::new(format!("https://example.com/{i}")))
            .collect();
        let shown: Vec<&HistoryEntry> =
            history.iter().rev().take(HISTORY_PANEL_LIMIT).collect();
        assert_eq!(shown.len(), HISTORY_PANEL_LIMIT);
        assert_eq!(shown[0].url, "https://example.com/59");
        assert_eq!(shown[49].url, "https://example.com/10");
    }
}
