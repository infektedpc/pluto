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

use bookmarks::{
    Bookmark, BookmarkEntry, BookmarkStore, bookmark_favicon_file_name,
    delete_bookmark_favicon, load_bookmark_favicon, load_bookmarks, save_bookmark_favicon,
    save_bookmarks,
};
use history::{load_history, save_history};
use crate::js_host::storage::{JsonValue, json_escape, parse_json, write_file_atomic};
use crate::security::profile_dir;
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
    /// The tab's console lines (Stage 17, M9 stages 1–2): the
    /// `(level, message)` pairs the worker drains once per tick
    /// (`RendererToBrowser::ConsoleLines`), plus the DevTools
    /// REPL's own echo (`> source`, level `""`) and result
    /// lines (level `""`). Page output is untrusted plain text
    /// — rendered as text, never as markup, never evaluated.
    /// Capped at [`CONSOLE_LINES_CAP`] (oldest evicted); the
    /// DevTools panel shows the active tab's buffer.
    console_lines: Vec<(String, String)>,
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
            console_lines: Vec::new(),
            sandbox_layers: Vec::new(),
            crashed: false,
            crash_times: Vec::new(),
            crash_reason: String::new(),
            dead: false,
        }
    }

    /// Append console lines, evicting the oldest past
    /// [`CONSOLE_LINES_CAP`] (Stage 17, M9 stages 1–2).
    fn push_console_lines(&mut self, lines: Vec<(String, String)>) {
        self.console_lines.extend(lines);
        let excess = self.console_lines.len().saturating_sub(CONSOLE_LINES_CAP);
        if excess > 0 {
            self.console_lines.drain(..excess);
        }
    }
}

/// The bookmark dialog's draft state (follow-up UI work,
/// 2026-11-27; Firefox-style restyle, bookmarks round
/// 2026-11-28): the star opens this instead of toggling. The
/// name is editable; the favicon is clickable to pick a custom
/// image. Nothing is written to the store until Save.
#[derive(Debug, Clone)]
struct BookmarkDialog {
    /// The page URL as opened (the bookmark's identity — the
    /// editable `url` below may differ from it).
    original_url: String,
    /// The editable bookmark URL, prefilled with the page URL.
    url: String,
    /// The editable bookmark name, prefilled with the page title
    /// (or the stored title when editing).
    name: String,
    /// The editable tags field, as typed: comma-separated.
    tags: String,
    /// The editable keyword field, as typed: a single word.
    keyword: String,
    /// Whether the tags chevron dropdown is open.
    show_tags: bool,
    /// The custom favicon picked in this dialog, if any: decoded
    /// pixels for display plus the raw file bytes to persist on
    /// Save. `None` means "keep whatever favicon the bookmark
    /// already has (or the site's, for a new bookmark)".
    custom_favicon: Option<(pluto_paint::DecodedImage, Vec<u8>)>,
    /// True when the page was already bookmarked when the dialog
    /// opened — Save edits it in place and a Remove button shows.
    editing: bool,
    /// The last file-pick failure, shown in the dialog.
    error: Option<String>,
}

impl BookmarkDialog {
    /// Open the dialog for `url`: prefill the name with the
    /// stored title when the page is already bookmarked
    /// (`existing`), else with the current page `title`; tags
    /// and keyword prefill from the stored bookmark when
    /// editing, else empty.
    fn new(url: &str, title: &str, existing: Option<&Bookmark>) -> Self {
        let (name, tags, keyword, editing) = match existing {
            Some(bookmark) => (
                bookmark.title.clone(),
                bookmark.tags.join(", "),
                bookmark.keyword.clone().unwrap_or_default(),
                true,
            ),
            None => (title.to_string(), String::new(), String::new(), false),
        };
        Self {
            original_url: url.to_string(),
            url: url.to_string(),
            name,
            tags,
            keyword,
            show_tags: false,
            custom_favicon: None,
            editing,
            error: None,
        }
    }
}

/// The new-folder dialog's draft state (bookmarks round,
/// 2026-11-28): just a name, prefilled "New folder". Nothing
/// is written until Save.
#[derive(Debug, Clone)]
struct FolderDialog {
    name: String,
}

/// The settings dialog's draft state (settings round,
/// 2026-11-28): only the homepage is a draft — it is typed
/// free-form, so it validates as it is typed and only a valid
/// value is saved. Theme, search provider, tab placement, and
/// the bookmarks-bar checkbox apply and persist immediately
/// when clicked, so they read straight off the shell.
#[derive(Debug, Clone)]
struct SettingsDialog {
    /// The homepage field as typed, prefilled with the saved
    /// homepage when the dialog opens.
    homepage_draft: String,
    /// The validation error for the draft, if it is not a
    /// usable homepage (shown in the dialog; the saved
    /// homepage is untouched until the draft validates).
    homepage_error: Option<String>,
}

impl SettingsDialog {
    fn new(homepage: &str) -> Self {
        Self {
            homepage_draft: homepage.to_string(),
            homepage_error: None,
        }
    }
}

/// The DevTools console panel's draft state (Stage 17, M9
/// stages 1–2): only the REPL input is a draft — the console
/// lines themselves live per tab (see `Tab::console_lines`),
/// and the panel follows the active tab, so switching tabs
/// shows the new tab's lines with the same input draft.
#[derive(Debug, Clone, Default)]
struct DevToolsPanel {
    /// The REPL input as typed.
    repl_input: String,
}

/// Which bookmarks-bar context menu is open, if any
/// (bookmarks round, 2026-11-28): the generic bar menu, or the
/// menu for one bookmark chip / folder chip.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BarContextMenu {
    /// Right-click on empty bar space (or the pencil).
    Bar,
    /// Right-click on a bookmark chip.
    Bookmark { url: String },
    /// Right-click on a folder chip.
    Folder { id: String },
}

/// One entry in the bookmarks bar, in store order (bookmarks
/// round, 2026-11-28): a bookmark chip or a folder chip. The
/// bar reads the store, never a hardcoded list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarEntry {
    Bookmark { title: String, url: String },
    Folder { id: String, name: String },
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
    /// star, shown on the bookmarks bar and in the add/edit
    /// dialog.
    bookmarks: BookmarkStore,
    /// Decoded custom favicons, by bookmark URL (follow-up UI
    /// work, 2026-11-27): loaded once at startup from the
    /// profile dir, so the bookmarks bar and the dialog render
    /// without re-reading or re-decoding files every frame.
    bookmark_favicons: std::collections::HashMap<String, pluto_paint::DecodedImage>,
    /// The open bookmark dialog, if any (follow-up UI work,
    /// 2026-11-27): the star opens this instead of toggling.
    bookmark_dialog: Option<BookmarkDialog>,
    /// Whether the bookmarks bar is shown (bookmarks round,
    /// 2026-11-28): toggled by the hamburger menu's "Bookmarks
    /// bar" checkbox, persisted in `preferences.json`.
    show_bookmarks_bar: bool,
    /// Whether the bookmarks bar is in edit mode (bookmarks
    /// round, 2026-11-28): toggled by the pencil at the bar's
    /// far right; chips show a × to remove them, and chip clicks
    /// do not navigate.
    bookmarks_edit_mode: bool,
    /// The open bookmarks-bar context menu, if any (bookmarks
    /// round, 2026-11-28).
    bar_context_menu: Option<BarContextMenu>,
    /// The nested-MouseArea flag (bookmarks round, 2026-11-28):
    /// the bar's right-click area wraps each chip's own
    /// right-click area, and iced delivers the inner message
    /// before the outer one — so a chip right-click sets this
    /// flag, and the outer (background) handler sees it, clears
    /// it, and leaves the chip menu standing. A background
    /// right-click finds it clear and opens the generic menu.
    bar_context_chip: bool,
    /// The open folder dropdown, if any (bookmarks round,
    /// 2026-11-28): the id of the folder whose bookmarks are
    /// listed under the bar.
    open_folder: Option<String>,
    /// The open new-folder dialog, if any (bookmarks round,
    /// 2026-11-28).
    folder_dialog: Option<FolderDialog>,
    /// The theme preference (settings round, 2026-11-28):
    /// System follows the OS; Light / Dark force it. The
    /// effective value lives in `dark_theme` (forwarded to
    /// the workers); this is what the user chose.
    theme_pref: ThemePref,
    /// The default homepage (settings round, 2026-11-28):
    /// the page opened at startup, for a new tab, and for an
    /// empty address bar. A normalized URL, persisted.
    homepage: String,
    /// The default search provider (settings round,
    /// 2026-11-28): address-bar input that is a query rather
    /// than a URL goes to this provider. Persisted.
    search_provider: page::SearchProvider,
    /// The open settings dialog, if any (settings round,
    /// 2026-11-28): opened from the hamburger menu's
    /// "Settings" entry.
    settings_dialog: Option<SettingsDialog>,
    /// The open DevTools console panel, if any (Stage 17, M9
    /// stages 1–2): opened from the hamburger menu's
    /// "DevTools" entry. The panel follows the active tab.
    devtools_panel: Option<DevToolsPanel>,
    /// Whether the hamburger menu is open. Opening a list panel
    /// closes the menu; the menu never shows alongside a panel.
    menu_open: bool,
    /// The open History list panel, if any: history from the
    /// active tab's session history.
    panel: Option<Panel>,
    /// Which edge(s) the vertical strip sits on (default Left).
    tab_placement: TabPlacement,
    /// The tab whose flyout is currently shown (or collapsing),
    /// by strip index — None when no flyout is visible.
    flyout_tab: Option<usize>,
    /// The flyout's slide-out progress, 0.0 (collapsed) ..= 1.0
    /// (fully open). Animated by the hover ticker.
    flyout_progress: f32,
    /// The flyout's target progress: 1.0 while a tab is hovered,
    /// 0.0 while it collapses.
    flyout_target: f32,
    /// The vertical scroll offsets of the two tab columns (the
    /// flyout tracks the hovered tab's on-screen position).
    left_rail_scroll: f32,
    right_rail_scroll: f32,
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
    /// The bookmark star: open the bookmark dialog for the
    /// current page (follow-up UI work, 2026-11-27 — the star no
    /// longer toggles immediately; the dialog names the bookmark
    /// and picks its favicon).
    ToggleBookmark,
    /// The bookmark dialog's name field changed.
    BookmarkDialogNameChanged(String),
    /// The bookmark dialog's favicon was clicked: open the file
    /// chooser for a custom favicon image.
    BookmarkDialogPickFavicon,
    /// The favicon file chooser returned: `None` means the user
    /// cancelled; `Some(Ok(..))` is the picked file's bytes plus
    /// its decoded pixels; `Some(Err(..))` is a decode failure
    /// to show in the dialog.
    BookmarkDialogFaviconPicked(
        Option<Result<(Vec<u8>, pluto_paint::DecodedImage), String>>,
    ),
    /// The bookmark dialog's Save/Add button: persist the
    /// bookmark (title + custom favicon) and close the dialog.
    BookmarkDialogSave,
    /// The bookmark dialog's Cancel button: close without saving.
    BookmarkDialogCancel,
    /// The bookmark dialog's Remove button (editing only):
    /// delete the bookmark and its custom favicon, then close.
    BookmarkDialogRemove,
    /// The bookmark dialog's URL field changed.
    BookmarkDialogUrlChanged(String),
    /// The bookmark dialog's tags field changed.
    BookmarkDialogTagsChanged(String),
    /// The bookmark dialog's keyword field changed.
    BookmarkDialogKeywordChanged(String),
    /// The bookmark dialog's tags chevron: show/hide the
    /// existing-tags dropdown.
    BookmarkDialogToggleTags,
    /// A tag in the dialog's tags dropdown was clicked: toggle
    /// it into/out of the tags field.
    BookmarkDialogTagToggle(String),
    /// The hamburger menu's "Bookmarks bar" checkbox: show or
    /// hide the bar (persisted).
    ToggleBookmarksBar,
    /// The bookmarks bar's pencil: enter/leave edit mode.
    ToggleBookmarksEditMode,
    /// Right-click on a bookmark chip in the bar: open its
    /// context menu.
    BarContextMenuBookmark(String),
    /// Right-click on a folder chip in the bar: open its
    /// context menu.
    BarContextMenuFolder(String),
    /// Right-click on the bar's background: open the generic
    /// bar menu — unless a chip menu was just opened (see
    /// `bar_context_chip`).
    BarContextMenuBar,
    /// Context menu: add the current page as a bookmark, title
    /// as the name, no dialog.
    BarAddBookmark,
    /// Context menu: open the new-folder dialog.
    BarAddFolder,
    /// Context menu (folder chip): add the current page into
    /// that folder, title as the name.
    BarAddBookmarkHere(String),
    /// Context menu: enter bookmarks edit mode.
    BarEditBookmarks,
    /// Context menu: hide the bookmarks bar (unchecks the menu
    /// item too).
    BarHideBookmarksBar,
    /// Context menu (bookmark chip): open the bookmark dialog
    /// for that bookmark.
    BarEditBookmark(String),
    /// Remove the folder (and its bookmarks) — the × in edit
    /// mode and the folder chip's context menu.
    BarRemoveFolder(String),
    /// A folder chip was clicked: open/close its dropdown under
    /// the bar.
    ToggleFolderPopup(String),
    /// The new-folder dialog's name field changed.
    FolderDialogNameChanged(String),
    /// The new-folder dialog's Save: create the folder.
    FolderDialogSave,
    /// The new-folder dialog's Cancel: close without creating.
    FolderDialogCancel,
    /// Menu: open the settings dialog (settings round,
    /// 2026-11-28).
    ShowSettings,
    /// Menu: open the DevTools console panel (Stage 17, M9
    /// stages 1–2). The panel follows the active tab; its
    /// console lines are stored per tab (see `Tab`).
    ShowDevTools,
    /// The DevTools panel's Close button: dismiss the panel.
    /// The tabs keep their console lines — reopening shows
    /// what was captured while it was closed.
    CloseDevTools,
    /// The DevTools REPL input changed (the draft lives in the
    /// panel, not per tab — the panel follows the active tab).
    DevToolsInputChanged(String),
    /// The DevTools REPL's submit (Enter or the Run button):
    /// echo the source into the active tab's console lines and
    /// send it to that tab's worker as `DevToolsEval`.
    DevToolsSubmit,
    /// The settings dialog's Close button: close the dialog.
    /// Everything in it already applied and persisted when it
    /// changed, except an invalid homepage draft, which is
    /// discarded — the saved homepage is untouched.
    CloseSettings,
    /// The settings dialog's theme choice: apply it at once
    /// (shell chrome + workers) and persist it.
    SettingsThemeChanged(ThemePref),
    /// The settings dialog's homepage field changed: the draft
    /// validates as it is typed — a valid value is saved at
    /// once, an invalid one shows an error and saves nothing.
    SettingsHomepageChanged(String),
    /// The settings dialog's search-provider choice: apply it
    /// at once and persist it.
    SettingsSearchProviderChanged(page::SearchProvider),
    /// The hamburger button: open/close the menu.
    ToggleMenu,
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
    /// The settings dialog: put the vertical strip on an edge (or
    /// both).
    SetTabPlacement(TabPlacement),
    /// A vertical tab's chip was hovered (mouse entered): start
    /// its slide-out flyout.
    TabHovered(usize),
    /// The mouse left a vertical tab's chip: collapse its flyout.
    TabUnhovered(usize),
    /// One frame of the hover slide-out animation (16 ms ticker
    /// while a flyout is expanding or collapsing).
    HoverTick,
    /// A vertical tab column was scrolled: its new vertical
    /// offset (the flyout tracks the hovered tab's on-screen
    /// position).
    RailScrolled(TabEdge, f32),
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
    /// "Pluto ~ The far out browser ~ <tab title>", or just
    /// "Pluto ~ The far out browser" when the tab has no title yet.
    fn sync_window_title(&mut self) {
        let title = self.active().title.clone();
        self.core.window.header_title = if title.is_empty() {
            "Pluto ~ The far out browser".to_string()
        } else {
            format!("Pluto ~ The far out browser ~ {title}")
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

    /// Drop any visible flyout (mode switch, tab close, ...):
    /// strip indices shift when a tab closes, so a stale flyout
    /// would point at the wrong tab.
    fn clear_flyout(&mut self) {
        self.flyout_tab = None;
        self.flyout_progress = 0.0;
        self.flyout_target = 0.0;
    }

    /// Whether the hover slide-out is mid-animation: a flyout is
    /// shown and its progress has not reached its target yet.
    /// The ticker subscription runs only while this is true.
    fn hover_animating(&self) -> bool {
        self.flyout_tab.is_some()
            && (self.flyout_progress - self.flyout_target).abs() > f32::EPSILON
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
        if let Some(tab) = self.tab_mut(tab_id)
            && let Some(tx) = &tab.renderer_tx
                && tx.send(command).is_err()
            {
                tab.renderer_tx = None;
            }
    }
    /// Tell a tab's worker the shell's session-history length and
    /// index (Stage 16 Phase 3, Rec 8 — finisher): the worker's
    /// realm mirrors them, so `history.length` and the `popstate`
    /// guard agree with the shell after every push / replace /
    /// traversal. No-op when the tab has no renderer yet.
    fn send_history_sync(&mut self, tab_id: u64) {
        if let Some(tab) = self.tab_mut(tab_id)
            && let Some(tx) = &tab.renderer_tx {
                let _ = tx.send(BrowserToRenderer::HistorySync {
                    length: tab.history.len(),
                    index: tab.history_index,
                });
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

    /// The theme the preference means right now (settings
    /// round, 2026-11-28): a forced preference decides;
    /// System asks the OS.
    fn effective_dark(&self) -> bool {
        theme_pref_is_dark(self.theme_pref)
    }

    /// Apply the theme preference everywhere it matters
    /// (settings round, 2026-11-28): the workers get the
    /// effective dark flag via `apply_theme` (so
    /// `prefers-color-scheme` follows), and the returned task
    /// switches the shell's own libcosmic theme. For System
    /// the task restores the OS preference; for a forced theme
    /// it installs it. Callers return the task from `update`
    /// (or `init`) so libcosmic runs it.
    fn apply_theme_pref(&mut self) -> Task<Message> {
        self.apply_theme(self.effective_dark());
        match self.theme_pref {
            ThemePref::System => {
                cosmic::command::set_theme(cosmic::theme::system_preference())
            }
            ThemePref::Light => cosmic::command::set_theme(cosmic::Theme::light()),
            ThemePref::Dark => cosmic::command::set_theme(cosmic::Theme::dark()),
        }
    }

    /// Persist the shell's current preferences (settings
    /// round, 2026-11-28): every handler that changes a
    /// preference calls this instead of building a
    /// `ShellPrefs` by hand — a hand-built literal would
    /// silently reset the fields it does not mention.
    fn save_prefs(&self) {
        save_shell_prefs(&ShellPrefs {
            show_bookmarks_bar: self.show_bookmarks_bar,
            theme: self.theme_pref,
            homepage: self.homepage.clone(),
            search_provider: self.search_provider,
            tab_placement: self.tab_placement,
        });
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
    /// The hamburger menu (UI round, 2026-11-27): a compact
    /// 260px dropdown, right-aligned under the hamburger — not
    /// the half-screen column it was. Below the backend items,
    /// a divider, then the bookmarks-bar checkbox — the only view
    /// option left in the menu (tab placement moved out to the
    /// settings dialog only, 2026-11-28).
    fn menu_panel(&self) -> Element<'_, Message> {
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        for label in menu_items() {
            let message = match *label {
                "History" => Message::ShowHistory,
                "New Tab" => Message::NewTab,
                "New Window" => Message::NewWindow,
                "Settings" => Message::ShowSettings,
                "DevTools" => Message::ShowDevTools,
                // `menu_items` is the pinned backend list; a new
                // entry needs a message here too.
                _ => unreachable!("menu item {label} has no message"),
            };
            items.push(
                widget::button::text(*label)
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(message)
                    .width(Length::Fill)
                    .into(),
            );
        }
        items.push(widget::divider::horizontal::default().into());
        // The bookmarks-bar checkbox (bookmarks round,
        // 2026-11-28): the only view option in the menu — it is
        // not in `menu_items` (that list is pinned to working
        // backends). Checked = the bar is visible; clicking
        // toggles it and persists the choice.
        let bar_label = if self.show_bookmarks_bar {
            "Bookmarks bar ✓"
        } else {
            "Bookmarks bar"
        };
        items.push(
            widget::button::text(bar_label)
                .class(small_button_style(SmallButtonBase::Text, false))
                .on_press(Message::ToggleBookmarksBar)
                .width(Length::Fill)
                .into(),
        );
        let panel = widget::column(items).spacing(4).padding(8);
        let boxed = widget::container(panel)
            .width(Length::Fixed(MENU_PANEL_WIDTH))
            .style(panel_container_style);
        widget::row([
            cosmic::iced::widget::space().width(Length::Fill).into(),
            boxed.into(),
        ])
        .into()
    }

    /// The bookmarks bar, directly below the nav bar (UI round,
    /// 2026-11-27): a flat 30px strip with the theme's
    /// component background, 8px side padding, 4px gaps, 12px
    /// chip text, and a 16px favicon when one is set — the
    /// bottom divider in `view` is the strip's lower edge.
    /// The pencil at the far right (bookmarks round, 2026-11-28)
    /// toggles edit mode: every chip gets a × that removes it,
    /// and chip clicks stop navigating. Right-click anywhere on
    /// the bar — empty space, pencil, or a chip — opens the
    /// bar's context menu (see `bar_context_menu_overlay`).
    fn bookmarks_bar(&self) -> Element<'_, Message> {
        let entries = bookmarks_bar_entries(&self.bookmarks);
        let mut children: Vec<Element<'_, Message>> = Vec::new();
        if entries.is_empty() {
            children.push(
                widget::text("No bookmarks yet — use the star to save this page.")
                    .size(12)
                    .into(),
            );
        } else {
            for entry in entries {
                children.push(self.bar_chip(entry));
            }
        }
        // The pencil, always last: the edit-mode toggle, pinned
        // to the bar's far right by the fill space before it.
        children.push(cosmic::iced::widget::space().width(Length::Fill).into());
        let pencil = widget::button::custom(
            widget::icon(bundled_icon(PENCIL_ICON_SVG)).size(16),
        )
        .class(small_button_style(
            SmallButtonBase::Text,
            self.bookmarks_edit_mode,
        ))
        .on_press(Message::ToggleBookmarksEditMode)
        .into();
        children.push(pencil);
        let row = widget::row(children).spacing(4);
        // The bar's own right-click area wraps the whole row:
        // right-click on empty space (or the pencil) opens the
        // generic bar menu. Chips carry their own right-click
        // area (see `bar_chip`), which reports first and makes
        // this outer handler a no-op for that click (see
        // `bar_context_chip`).
        let area = cosmic::iced::widget::mouse_area(row)
            .on_right_press(Message::BarContextMenuBar);
        widget::container(area)
            .style(bookmarks_bar_style)
            .width(Length::Fill)
            .height(Length::Fixed(30.0))
            .padding(cosmic::iced::core::Padding {
                top: 0.0,
                right: 8.0,
                bottom: 0.0,
                left: 8.0,
            })
            .align_y(Alignment::Center)
            .into()
    }

    /// One bookmarks-bar chip (bookmarks round, 2026-11-28): a
    /// bookmark (favicon + title) or a folder (folder icon +
    /// name). In edit mode the chip is not a button — only its ×
    /// acts, so a click on the chip body does nothing. Otherwise
    /// a bookmark chip navigates and a folder chip opens its
    /// dropdown. A right-click area wraps the chip and reports
    /// its own menu before the bar's area sees the click.
    fn bar_chip(&self, entry: BarEntry) -> Element<'_, Message> {
        let (is_folder, label, url, folder_id) = match entry {
            BarEntry::Bookmark { title, url } => {
                (false, ellipsize(&title, 24), Some(url), None)
            }
            BarEntry::Folder { id, name } => (true, ellipsize(&name, 24), None, Some(id)),
        };
        let mut row_children: Vec<Element<'_, Message>> = Vec::new();
        if is_folder {
            let icon: Element<'_, Message> =
                widget::icon(bundled_icon(FOLDER_ICON_SVG)).size(16).into();
            row_children.push(icon);
        } else if let Some(image) = url
            .as_ref()
            .and_then(|url| self.bookmark_favicons.get(url))
        {
            row_children.push(Self::decoded_image_element(image.clone(), 16.0));
        }
        row_children.push(widget::text(label).size(12).into());
        let chip: Element<'_, Message> = if self.bookmarks_edit_mode {
            // Edit mode: the × removes; the chip body is inert
            // (a plain row, not a button, so nothing navigates).
            let mut remove_children = row_children;
            let message = if is_folder {
                Message::BarRemoveFolder(folder_id.clone().unwrap_or_default())
            } else {
                Message::RemoveBookmark(url.clone().unwrap_or_default())
            };
            remove_children.push(
                widget::button::text("×")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(message)
                    .into(),
            );
            widget::row(remove_children).spacing(4).into()
        } else if is_folder {
            let id = folder_id.clone().unwrap_or_default();
            widget::button::custom(widget::row(row_children).spacing(4))
                .class(small_button_style(
                    SmallButtonBase::Text,
                    self.open_folder.as_deref() == Some(id.as_str()),
                ))
                .on_press(Message::ToggleFolderPopup(id))
                .into()
        } else {
            widget::button::custom(widget::row(row_children).spacing(4))
                .class(small_button_style(SmallButtonBase::Text, false))
                .on_press(Message::NavigateTo(url.clone().unwrap_or_default()))
                .into()
        };
        // The chip's own right-click area: reports its menu
        // before the bar's area, and sets the flag that makes
        // the bar's handler a no-op for this same click.
        let right_message = if is_folder {
            Message::BarContextMenuFolder(folder_id.clone().unwrap_or_default())
        } else {
            Message::BarContextMenuBookmark(url.clone().unwrap_or_default())
        };
        cosmic::iced::widget::mouse_area(chip)
            .on_right_press(right_message)
            .into()
    }

    /// The bookmarks-bar context menu (bookmarks round,
    /// 2026-11-28): a small popup below the bar, left-aligned
    /// (a fixed position — the right-click area reports no
    /// cursor position, so the menu sits at the bar's left
    /// edge). Closes on any item click.
    fn bar_context_menu_overlay(&self) -> Option<Element<'_, Message>> {
        let menu = self.bar_context_menu.as_ref()?;
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        match menu {
            BarContextMenu::Bar => {
                items.push(
                    widget::button::text("Add bookmark")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarAddBookmark)
                        .width(Length::Fill)
                        .into(),
                );
                items.push(
                    widget::button::text("Add folder")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarAddFolder)
                        .width(Length::Fill)
                        .into(),
                );
                items.push(
                    widget::button::text("Edit bookmarks")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarEditBookmarks)
                        .width(Length::Fill)
                        .into(),
                );
                items.push(
                    widget::button::text("Hide bookmarks bar")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarHideBookmarksBar)
                        .width(Length::Fill)
                        .into(),
                );
            }
            BarContextMenu::Bookmark { url } => {
                items.push(
                    widget::button::text("Edit")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarEditBookmark(url.clone()))
                        .width(Length::Fill)
                        .into(),
                );
                items.push(
                    widget::button::text("Remove")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::RemoveBookmark(url.clone()))
                        .width(Length::Fill)
                        .into(),
                );
            }
            BarContextMenu::Folder { id } => {
                items.push(
                    widget::button::text("Add bookmark here")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarAddBookmarkHere(id.clone()))
                        .width(Length::Fill)
                        .into(),
                );
                items.push(
                    widget::button::text("Remove")
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BarRemoveFolder(id.clone()))
                        .width(Length::Fill)
                        .into(),
                );
            }
        }
        let panel = widget::column(items).spacing(4).padding(8);
        let boxed = widget::container(panel)
            .width(Length::Fixed(BAR_POPUP_WIDTH))
            .style(panel_container_style);
        Some(
            widget::column([
                cosmic::iced::widget::space()
                    .height(Length::Fixed(BAR_POPUP_TOP))
                    .into(),
                widget::row([
                    boxed.into(),
                    cosmic::iced::widget::space().width(Length::Fill).into(),
                ])
                .into(),
            ])
            .into(),
        )
    }

    /// A folder's dropdown under the bookmarks bar (bookmarks
    /// round, 2026-11-28): the folder's bookmarks, one click to
    /// navigate; an empty folder says so instead of showing a
    /// bare empty popup. Closes on any click.
    fn folder_popup_overlay(&self) -> Option<Element<'_, Message>> {
        let id = self.open_folder.as_ref()?;
        let folder = self.bookmarks.folder(id)?;
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        if folder.bookmarks.is_empty() {
            items.push(widget::text("Empty folder").size(12).into());
        } else {
            for bookmark in &folder.bookmarks {
                let mut row_children: Vec<Element<'_, Message>> = Vec::new();
                if let Some(image) = self.bookmark_favicons.get(&bookmark.url) {
                    row_children.push(Self::decoded_image_element(image.clone(), 16.0));
                }
                row_children.push(
                    widget::text(ellipsize(&bookmark.title, 24))
                        .size(12)
                        .into(),
                );
                items.push(
                    widget::button::custom(widget::row(row_children).spacing(4))
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::NavigateTo(bookmark.url.clone()))
                        .width(Length::Fill)
                        .into(),
                );
            }
        }
        let panel = widget::column(items).spacing(4).padding(8);
        let boxed = widget::container(panel)
            .width(Length::Fixed(BAR_POPUP_WIDTH))
            .style(panel_container_style);
        Some(
            widget::column([
                cosmic::iced::widget::space()
                    .height(Length::Fixed(BAR_POPUP_TOP))
                    .into(),
                widget::row([
                    boxed.into(),
                    cosmic::iced::widget::space().width(Length::Fill).into(),
                ])
                .into(),
            ])
            .into(),
        )
    }

    /// The new-folder dialog (bookmarks round, 2026-11-28): a
    /// small centered overlay asking for the folder name,
    /// prefilled "New folder". Save creates the folder at the
    /// bar's end; Cancel creates nothing.
    fn folder_dialog_overlay(&self) -> Option<Element<'_, Message>> {
        let dialog = self.folder_dialog.as_ref()?;
        let input = widget::text_input("Folder name", &dialog.name)
            .on_input(Message::FolderDialogNameChanged)
            .on_submit(|_| Message::FolderDialogSave)
            .padding(8);
        let buttons = widget::row([
            cosmic::iced::widget::space().width(Length::Fill).into(),
            widget::button::text("Cancel")
                .class(small_button_style(SmallButtonBase::Text, false))
                .on_press(Message::FolderDialogCancel)
                .into(),
            widget::button::text("Save")
                .class(small_button_style(SmallButtonBase::Suggested, false))
                .on_press(Message::FolderDialogSave)
                .into(),
        ])
        .spacing(8);
        let content = widget::column([
            widget::text("New folder").size(16).into(),
            input.into(),
            buttons.into(),
        ])
        .spacing(8)
        .padding(16);
        let boxed = widget::container(content)
            .width(Length::Fixed(FOLDER_DIALOG_WIDTH))
            .style(panel_container_style);
        Some(
            widget::container(
                widget::row([
                    cosmic::iced::widget::space().width(Length::Fill).into(),
                    boxed.into(),
                    cosmic::iced::widget::space().width(Length::Fill).into(),
                ])
                .align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .align_y(Alignment::Center)
            .into(),
        )
    }

    /// The vertical-mode content: the web view as the base layer,
    /// spanning the full width, with the floating tab columns
    /// layered on top of it (and the hover flyout above those).
    fn vertical_content<'a>(
        &'a self,
        web_view: Element<'a, Message>,
    ) -> Element<'a, Message> {
        let capacity = rail_capacity(self.viewport.1);
        let mut layers: Vec<Element<'_, Message>> = vec![web_view];
        for edge in tab_strip_edges(self.tab_placement) {
            let (start, count) =
                column_range(self.tab_placement, *edge, self.tabs.len(), capacity);
            if count > 0 {
                layers.push(self.rail_overlay(*edge, start, count));
            }
        }
        if let Some(flyout) = self.flyout_element() {
            layers.push(flyout);
        }
        cosmic::iced::widget::stack(layers)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// One floating tab column, pinned to its edge over the web
    /// view: scrollable collapsed chips for tabs
    /// `start..start + count`, with the "+" button pinned at the
    /// bottom.
    fn rail_overlay(&self, edge: TabEdge, start: usize, count: usize) -> Element<'_, Message> {
        let align = match edge {
            TabEdge::Left => Alignment::Start,
            TabEdge::Right => Alignment::End,
        };
        let mut tab_children: Vec<Element<'_, Message>> = Vec::new();
        for (i, tab) in self.tabs.iter().enumerate().skip(start).take(count) {
            tab_children.push(self.tab_chip(i, tab));
        }
        let tab_column = widget::column(tab_children)
            .spacing(TAB_GAP)
            .width(Length::Fill)
            .align_x(align);
        let scrolled = cosmic::iced::widget::scrollable(tab_column)
            .width(Length::Fill)
            .height(Length::Fill)
            .direction(cosmic::iced::widget::scrollable::Direction::Vertical(
                Default::default(),
            ))
            .on_scroll(move |viewport| {
                Message::RailScrolled(edge, viewport.absolute_offset().y)
            });
        let rail = widget::column([scrolled.into(), self.plus_chip()])
            .spacing(TAB_GAP)
            .padding(TAB_GAP)
            .align_x(align)
            .width(Length::Fixed(RAIL_WIDTH))
            .height(Length::Fill);
        match edge {
            TabEdge::Left => widget::row([
                rail.into(),
                cosmic::iced::widget::space().width(Length::Fill).into(),
            ])
            .height(Length::Fill)
            .into(),
            TabEdge::Right => widget::row([
                cosmic::iced::widget::space().width(Length::Fill).into(),
                rail.into(),
            ])
            .height(Length::Fill)
            .into(),
        }
    }

    /// One collapsed tab chip: a 32px square showing only the
    /// favicon (or the site's initial letter when it has none),
    /// floating on the page as a chip with its own background and
    /// shadow. Clicking selects the tab; hovering starts the
    /// slide-out flyout.
    fn tab_chip(&self, index: usize, tab: &Tab) -> Element<'_, Message> {
        let active = index == self.active_tab;
        let content = widget::container(self.favicon_element(tab))
            .center_x(Length::Fill)
            .center_y(Length::Fill);
        let button = widget::button::custom(content)
            .class(cosmic::theme::Button::Transparent)
            .on_press(Message::SelectTab(index))
            .width(Length::Fixed(TAB_CHIP))
            .height(Length::Fixed(TAB_CHIP));
        let chip = widget::container(button).style(move |theme| tab_chip_style(theme, active));
        cosmic::iced::widget::mouse_area(chip)
            .on_enter(Message::TabHovered(index))
            .on_exit(Message::TabUnhovered(index))
            .into()
    }

    /// The "+" new-tab chip, pinned at a column's bottom.
    fn plus_chip(&self) -> Element<'_, Message> {
        let content = widget::container(widget::text("+"))
            .center_x(Length::Fill)
            .center_y(Length::Fill);
        let button = widget::button::custom(content)
            .class(cosmic::theme::Button::Transparent)
            .on_press(Message::NewTab)
            .width(Length::Fixed(TAB_CHIP))
            .height(Length::Fixed(TAB_CHIP));
        widget::container(button)
            .style(|theme| tab_chip_style(theme, false))
            .into()
    }

    /// The hovered tab's slide-out flyout, layered above the
    /// rails: the chip elongates to 200px, revealing the centered
    /// title and the close button. None when no flyout is
    /// visible. The flyout tracks the hovered tab's on-screen
    /// position (its column, its index in the column, minus the
    /// column's scroll offset).
    fn flyout_element(&self) -> Option<Element<'_, Message>> {
        let index = self.flyout_tab?;
        if self.flyout_progress <= 0.0 || index >= self.tabs.len() {
            return None;
        }
        let capacity = rail_capacity(self.viewport.1);
        let (edge, idx_in_column) =
            flyout_column_position(self.tab_placement, index, self.tabs.len(), capacity)?;
        let tab = &self.tabs[index];
        let active = index == self.active_tab;
        let width = flyout_width(self.flyout_progress);
        let mut row_children: Vec<Element<'_, Message>> = vec![self.favicon_element(tab)];
        if flyout_shows_text(self.flyout_progress) {
            row_children.push(
                widget::text(ellipsize(&tab.title, 28))
                    .width(Length::Fill)
                    .align_x(Alignment::Center)
                    .into(),
            );
            row_children.push(
                widget::button::text("×")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::CloseTab(index))
                    .into(),
            );
        }
        let button = widget::button::custom(widget::row(row_children).spacing(TAB_GAP))
            .class(cosmic::theme::Button::Transparent)
            .on_press(Message::SelectTab(index))
            .width(Length::Fixed(width))
            .height(Length::Fixed(TAB_CHIP));
        let chip = widget::container(button).style(move |theme| tab_chip_style(theme, active));
        let hovered = cosmic::iced::widget::mouse_area(chip)
            .on_enter(Message::TabHovered(index))
            .on_exit(Message::TabUnhovered(index));
        let scroll = match edge {
            TabEdge::Left => self.left_rail_scroll,
            TabEdge::Right => self.right_rail_scroll,
        };
        let y = (TAB_GAP + idx_in_column as f32 * TAB_PITCH - scroll).max(0.0);
        let flyout_row = match edge {
            TabEdge::Left => widget::row([
                cosmic::iced::widget::space().width(Length::Fixed(TAB_GAP)).into(),
                hovered.into(),
                cosmic::iced::widget::space().width(Length::Fill).into(),
            ]),
            TabEdge::Right => widget::row([
                cosmic::iced::widget::space().width(Length::Fill).into(),
                hovered.into(),
                cosmic::iced::widget::space().width(Length::Fixed(TAB_GAP)).into(),
            ]),
        };
        Some(
            widget::column([
                cosmic::iced::widget::space().height(Length::Fixed(y)).into(),
                flyout_row.into(),
            ])
            .height(Length::Fill)
            .into(),
        )
    }

    /// Remove a bookmark and its custom favicon file, then
    /// persist (follow-up UI work, 2026-11-27): the list panel's
    /// Remove button and the dialog's Remove button share this,
    /// so a deleted bookmark never leaves an orphaned favicon
    /// file or a stale decoded icon behind.
    fn remove_bookmark(&mut self, url: &str) {
        if self.bookmarks.remove(url) {
            delete_bookmark_favicon(url);
            self.bookmark_favicons.remove(url);
            save_bookmarks(&self.bookmarks);
        }
    }

    /// Remove a folder and every bookmark inside it, with each
    /// bookmark's custom favicon file, then persist (bookmarks
    /// round, 2026-11-28). A folder's open dropdown closes too.
    fn remove_folder(&mut self, id: &str) {
        if let Some(folder) = self.bookmarks.folder(id) {
            for bookmark in &folder.bookmarks {
                delete_bookmark_favicon(&bookmark.url);
                self.bookmark_favicons.remove(&bookmark.url);
            }
        }
        if self.bookmarks.remove_folder(id) {
            if self.open_folder.as_deref() == Some(id) {
                self.open_folder = None;
            }
            save_bookmarks(&self.bookmarks);
        }
    }

    /// Close every bookmarks-bar popup (bookmarks round,
    /// 2026-11-28): the context menu, any open folder dropdown,
    /// and the new-folder dialog. Navigation, menu opens, panel
    /// opens, and dialog opens all close them — a popup never
    /// outlives the action that superseded it.
    fn close_bar_popups(&mut self) {
        self.bar_context_menu = None;
        self.open_folder = None;
        self.folder_dialog = None;
    }

    /// The active tab's site favicon as decoded pixels, if the
    /// site has one (the tab's resource table holds it as RGBA).
    fn active_site_favicon(&self) -> Option<pluto_paint::DecodedImage> {
        let tab = self.active();
        let favicon_id = tab.favicon_id?;
        // `Resource` has a single variant (`Image`), so this
        // destructure is irrefutable — no `else` needed.
        let pluto_paint::Resource::Image {
            width,
            height,
            rgba,
        } = tab.resources.resources.get(&favicon_id)?;
        Some(pluto_paint::DecodedImage {
            width: *width,
            height: *height,
            rgba: rgba.clone(),
        })
    }

    /// A decoded image as a fixed-size iced image element. The
    /// image is taken by value: the handle owns the pixels
    /// (`from_rgba` takes the bytes), so the element borrows
    /// nothing — a decoded site favicon, which only exists as a
    /// local, can be shown without lifetime gymnastics. A
    /// favicon is a few KiB, so the copy is negligible.
    fn decoded_image_element(
        image: pluto_paint::DecodedImage,
        size: f32,
    ) -> Element<'static, Message> {
        let handle = cosmic::iced::widget::image::Handle::from_rgba(
            image.width,
            image.height,
            image.rgba,
        );
        cosmic::iced::widget::image(handle)
            .width(Length::Fixed(size))
            .height(Length::Fixed(size))
            .into()
    }

    /// The bookmark dialog's favicon (follow-up UI work,
    /// 2026-11-27): the custom image picked in this dialog, else
    /// the bookmark's stored custom favicon, else the active
    /// tab's site favicon, else the host's initial letter. The
    /// caller wraps it in the clickable button. Looked up by the
    /// bookmark's original URL — the editable URL may differ
    /// from it while the dialog is open.
    fn dialog_favicon_element<'a>(
        &'a self,
        dialog: &'a BookmarkDialog,
    ) -> Element<'a, Message> {
        if let Some((decoded, _)) = &dialog.custom_favicon {
            return Self::decoded_image_element(decoded.clone(), 32.0);
        }
        if let Some(decoded) = self.bookmark_favicons.get(&dialog.original_url) {
            return Self::decoded_image_element(decoded.clone(), 32.0);
        }
        if let Some(site) = self.active_site_favicon() {
            return Self::decoded_image_element(site, 32.0);
        }
        widget::text(initial_letter(&dialog.original_url, &dialog.name).to_string())
            .size(20)
            .into()
    }

    /// The hamburger menu as a floating popup (follow-up UI work,
    /// 2026-11-27): an overlay layer above the whole shell, not
    /// an in-flow row — opening it moves nothing around it. The
    /// panel hangs below the nav bar (the top padding clears it);
    /// clicks outside the 260px panel fall through to the page,
    /// because the layer itself captures no pointer events —
    /// only the panel's buttons do.
    fn menu_overlay(&self) -> Element<'_, Message> {
        widget::container(self.menu_panel())
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(cosmic::iced::core::Padding {
                top: menu_overlay_top_offset(),
                right: 8.0,
                bottom: 0.0,
                left: 0.0,
            })
            .into()
    }

    /// The bookmark dialog as a floating popup (follow-up UI
    /// work, 2026-11-27): a centered overlay layer, like the
    /// menu — it hovers over the page without moving anything.
    /// The favicon is a button: clicking it opens the file
    /// chooser for a custom favicon image.
    /// Restyled like Firefox's "Add bookmark" dialog
    /// (bookmarks round, 2026-11-28): a star + title row, then
    /// labelled Name / URL / Tags / Keyword fields, helper text
    /// under Tags and Keyword, and Cancel + Save bottom-right
    /// (Save in the accent color). The name is selected on open
    /// (see `BookmarkDialogNameChanged`'s select-all task in
    /// `update`). The tags chevron lists every tag on every
    /// bookmark — clicking one toggles it into the tags field.
    fn bookmark_dialog_overlay<'a>(
        &'a self,
        dialog: &'a BookmarkDialog,
    ) -> Element<'a, Message> {
        let title_star: Element<'_, Message> =
            widget::icon(bundled_icon(STAR_ICON_SVG)).size(16).into();
        let favicon_button = widget::button::custom(self.dialog_favicon_element(dialog))
            .class(small_button_style(SmallButtonBase::Standard, false))
            .on_press(Message::BookmarkDialogPickFavicon);
        let name_input = widget::text_input("Name", &dialog.name)
            .id("bookmark-dialog-name".into())
            .on_input(Message::BookmarkDialogNameChanged)
            .on_submit(|_| Message::BookmarkDialogSave)
            .style(small_text_input_style())
            .width(Length::Fill);
        let url_input = widget::text_input("URL", &dialog.url)
            .on_input(Message::BookmarkDialogUrlChanged)
            .on_submit(|_| Message::BookmarkDialogSave)
            .style(small_text_input_style())
            .width(Length::Fill);
        let tags_input = widget::text_input("Separate tags with commas", &dialog.tags)
            .on_input(Message::BookmarkDialogTagsChanged)
            .on_submit(|_| Message::BookmarkDialogSave)
            .style(small_text_input_style())
            .width(Length::Fill);
        let tags_chevron = widget::button::text("▾")
            .class(small_button_style(SmallButtonBase::Text, false))
            .on_press(Message::BookmarkDialogToggleTags);
        let keyword_input = widget::text_input("Keyword", &dialog.keyword)
            .on_input(Message::BookmarkDialogKeywordChanged)
            .on_submit(|_| Message::BookmarkDialogSave)
            .style(small_text_input_style())
            .width(Length::Fill);
        let mut children: Vec<Element<'_, Message>> = vec![
            widget::row([
                title_star,
                widget::text(if dialog.editing {
                    "Edit bookmark"
                } else {
                    "Add bookmark"
                })
                .size(16)
                .into(),
            ])
            .spacing(8)
            .align_y(Alignment::Center)
            .into(),
            widget::text("Name").size(12).into(),
            widget::row([favicon_button.into(), name_input.into()])
                .spacing(8)
                .align_y(Alignment::Center)
                .into(),
            widget::text("URL").size(12).into(),
            url_input.into(),
            widget::text("Tags").size(12).into(),
            widget::row([tags_input.into(), tags_chevron.into()])
                .spacing(4)
                .align_y(Alignment::Center)
                .into(),
            widget::text("Use tags to organize and search for bookmarks from the address bar")
                .size(12)
                .into(),
            widget::text("Keyword").size(12).into(),
            keyword_input.into(),
            widget::text("Use a single keyword to open bookmarks directly from the address bar")
                .size(12)
                .into(),
        ];
        if dialog.show_tags {
            let mut tag_items: Vec<Element<'_, Message>> = Vec::new();
            for tag in self.bookmarks.all_tags() {
                let marked = if parse_tags(&dialog.tags).contains(&tag) {
                    format!("✓ {tag}")
                } else {
                    tag.clone()
                };
                tag_items.push(
                    widget::button::text(marked)
                        .class(small_button_style(SmallButtonBase::Text, false))
                        .on_press(Message::BookmarkDialogTagToggle(tag))
                        .width(Length::Fill)
                        .into(),
                );
            }
            if tag_items.is_empty() {
                tag_items.push(widget::text("No tags yet").size(12).into());
            }
            children.push(widget::column(tag_items).spacing(4).into());
        }
        if let Some(error) = &dialog.error {
            children.push(widget::text(error.clone()).size(12).into());
        }
        let mut buttons: Vec<Element<'_, Message>> = Vec::new();
        if dialog.editing {
            buttons.push(
                widget::button::text("Remove")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::BookmarkDialogRemove)
                    .into(),
            );
        }
        buttons.push(
            cosmic::iced::widget::space().width(Length::Fill).into(),
        );
        buttons.push(
            widget::button::text("Cancel")
                .class(small_button_style(SmallButtonBase::Text, false))
                .on_press(Message::BookmarkDialogCancel)
                .into(),
        );
        buttons.push(
            widget::button::text("Save")
                .class(small_button_style(SmallButtonBase::Suggested, false))
                .on_press(Message::BookmarkDialogSave)
                .into(),
        );
        children.push(widget::row(buttons).spacing(8).into());
        let panel = widget::column(children).spacing(8).padding(12);
        let boxed = widget::container(panel)
            .width(Length::Fixed(BOOKMARK_DIALOG_WIDTH))
            .style(panel_container_style);
        widget::container(boxed)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .into()
    }

    /// The settings dialog as a floating popup (settings round,
    /// 2026-11-28): a centered overlay layer, like the bookmark
    /// dialog — it hovers over the page without moving anything.
    /// Every control applies and persists at once when it
    /// changes; there is no Save, only Close. The homepage is
    /// the exception: it validates as it is typed, and an
    /// invalid draft shows an error and saves nothing (closing
    /// discards it). There is no VPN section — there is no VPN
    /// yet (that is M11), and a settings page for a VPN that
    /// does not exist would be a dead control.
    fn settings_dialog_overlay<'a>(
        &'a self,
        dialog: &'a SettingsDialog,
    ) -> Element<'a, Message> {
        // One option button: a ✓ when it is the current value.
        let option = |label: String, selected: bool, message: Message| {
            widget::button::text(label)
                .class(small_button_style(SmallButtonBase::Text, selected))
                .on_press(message)
                .into()
        };
        let mut theme_buttons: Vec<Element<'_, Message>> = Vec::new();
        for theme in ThemePref::all() {
            let label = if theme == self.theme_pref {
                format!("{} ✓", theme.label())
            } else {
                theme.label().to_string()
            };
            theme_buttons.push(option(
                label,
                theme == self.theme_pref,
                Message::SettingsThemeChanged(theme),
            ));
        }
        let mut provider_buttons: Vec<Element<'_, Message>> = Vec::new();
        for provider in page::SearchProvider::all() {
            let label = if provider == self.search_provider {
                format!("{} ✓", provider.label())
            } else {
                provider.label().to_string()
            };
            provider_buttons.push(option(
                label,
                provider == self.search_provider,
                Message::SettingsSearchProviderChanged(provider),
            ));
        }
        let mut placement_buttons: Vec<Element<'_, Message>> = Vec::new();
        for placement in [TabPlacement::Left, TabPlacement::Right, TabPlacement::Both] {
            let label = if placement == self.tab_placement {
                format!("{} ✓", placement_label(placement))
            } else {
                placement_label(placement).to_string()
            };
            placement_buttons.push(option(
                label,
                placement == self.tab_placement,
                Message::SetTabPlacement(placement),
            ));
        }
        let bar_label = if self.show_bookmarks_bar {
            "Bookmarks bar ✓"
        } else {
            "Bookmarks bar"
        };
        let mut children: Vec<Element<'_, Message>> = vec![
            widget::text("Settings").size(16).into(),
            widget::text("Theme").size(12).into(),
            widget::row(theme_buttons).spacing(8).into(),
            widget::text("Default homepage").size(12).into(),
            widget::text_input(page::DEFAULT_URL, &dialog.homepage_draft)
                .id("settings-homepage".into())
                .on_input(Message::SettingsHomepageChanged)
                .on_submit(|_| Message::CloseSettings)
                .style(small_text_input_style())
                .width(Length::Fill)
                .into(),
            widget::text(
                "The page Pluto opens at startup, for a new tab, and for an empty address bar.",
            )
            .size(12)
            .into(),
            widget::text("Default search provider").size(12).into(),
            widget::row(provider_buttons).spacing(8).into(),
            widget::text("Tabs").size(12).into(),
            widget::row(placement_buttons).spacing(8).into(),
            widget::text("Bookmarks bar").size(12).into(),
            option(
                bar_label.to_string(),
                self.show_bookmarks_bar,
                Message::ToggleBookmarksBar,
            ),
        ];
        if let Some(error) = &dialog.homepage_error {
            children.push(widget::text(error.clone()).size(12).into());
        }
        children.push(
            widget::row([
                cosmic::iced::widget::space().width(Length::Fill).into(),
                widget::button::text("Close")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::CloseSettings)
                    .into(),
            ])
            .spacing(8)
            .into(),
        );
        let panel = widget::column(children).spacing(8).padding(12);
        let boxed = widget::container(panel)
            .width(Length::Fixed(SETTINGS_DIALOG_WIDTH))
            .style(panel_container_style);
        widget::container(boxed)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .into()
    }

    /// The DevTools console panel (Stage 17, M9 stages 1–2):
    /// a floating overlay like the settings dialog — a centered
    /// panel over the page, in the `view()` stack layers. It
    /// shows the ACTIVE tab's console lines and a REPL input.
    /// Page output is rendered as untrusted plain text (the
    /// text widget — never markup, never evaluated): a page
    /// that logs `<script>…</script>` cannot inject into the
    /// panel. Lines with an empty level are the REPL's own echo
    /// (`> source`) and result lines, rendered as-is; real
    /// console lines render as `level: message` (an error's
    /// formatted stack is inside its message, so it wraps over
    /// several visual lines). The log area is a fixed-height
    /// scrollable, so a chatty page never grows the panel.
    fn devtools_panel_overlay<'a>(
        &'a self,
        panel: &'a DevToolsPanel,
    ) -> Element<'a, Message> {
        let tab = self.active();
        let mut line_elements: Vec<Element<'_, Message>> = Vec::new();
        if tab.console_lines.is_empty() {
            line_elements.push(
                widget::text("No console output yet — console.log from this page will appear here.")
                    .size(12)
                    .into(),
            );
        }
        for (level, message) in &tab.console_lines {
            let shown = if level.is_empty() {
                message.clone()
            } else {
                format!("{level}: {message}")
            };
            line_elements.push(widget::text(shown).size(12).into());
        }
        let log = cosmic::iced::widget::scrollable(
            widget::column(line_elements).spacing(2).width(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fixed(DEVTOOLS_PANEL_LOG_HEIGHT))
        .direction(cosmic::iced::widget::scrollable::Direction::Vertical(
            Default::default(),
        ));
        let children: Vec<Element<'_, Message>> = vec![
            widget::row([
                widget::text("DevTools — Console").size(16).into(),
                cosmic::iced::widget::space().width(Length::Fill).into(),
                widget::button::text("Close")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::CloseDevTools)
                    .into(),
            ])
            .spacing(8)
            .into(),
            log.into(),
            widget::row([
                widget::text_input("Evaluate in this page…", &panel.repl_input)
                    .id("devtools-repl".into())
                    .on_input(Message::DevToolsInputChanged)
                    .on_submit(|_| Message::DevToolsSubmit)
                    .style(small_text_input_style())
                    .width(Length::Fill)
                    .into(),
                widget::button::text("Run")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::DevToolsSubmit)
                    .into(),
            ])
            .spacing(8)
            .into(),
            widget::text("Runs with page privileges, like a script the page itself ran.")
                .size(12)
                .into(),
        ];
        let boxed = widget::container(widget::column(children).spacing(8).padding(12))
            .width(Length::Fixed(DEVTOOLS_PANEL_WIDTH))
            .style(panel_container_style);
        widget::container(boxed)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .into()
    }

    /// A tab's favicon as a 16px image, or the site's initial
    /// letter when it has no favicon yet (the host's initial —
    /// "example.com" → 'E' — or the title's first character).
    fn favicon_element(&self, tab: &Tab) -> Element<'_, Message> {
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
            return cosmic::iced::widget::image(handle)
                .width(Length::Fixed(16.0))
                .height(Length::Fixed(16.0))
                .into();
        }
        let letter = initial_letter(&tab.active_url, &tab.title);
        widget::text(letter.to_string()).into()
    }

    /// The open History list panel: a header row plus one row
    /// per entry. History rows navigate on click (most recent
    /// first, capped — the session history itself is unbounded).
    /// An empty list says so instead of showing a blank panel.
    /// (The Bookmarks list panel was removed with its menu entry,
    /// 2026-11-28 — bookmarks live on the bar and in the dialog.)
    fn list_panel(&self, panel: Panel) -> Element<'_, Message> {
        let title = match panel {
            Panel::History => "History",
        };
        let mut children: Vec<Element<'_, Message>> = vec![
            widget::row([
                widget::text(title).into(),
                widget::button::text("Close")
                    .class(small_button_style(SmallButtonBase::Text, false))
                    .on_press(Message::ClosePanel)
                    .into(),
            ])
            .spacing(8)
            .padding(4)
            .into(),
        ];
        match panel {
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
                            .class(small_button_style(SmallButtonBase::Text, false))
                            .on_press(Message::NavigateTo(entry.url.clone()))
                            .width(Length::Fill)
                            .into(),
                    );
                }
            }
        }
        widget::container(widget::column(children).spacing(4).padding(8))
            .width(Length::Fill)
            .style(panel_container_style)
            .into()
    }
}

/// Which list panel the hamburger menu opened, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    History,
}

/// Most history entries the History panel shows (most recent
/// first): the session history is unbounded in memory, but the
/// panel is a plain list, so it caps at the newest 50.
pub const HISTORY_PANEL_LIMIT: usize = 50;

/// The shell's corner radius (UI round, 2026-11-27): squared-off
/// controls at 4.0px — way less rounded than the theme default,
/// but not straight squares. Applied per-widget (Custom styles
/// and container styles); the theme itself is untouched, so the
/// round nav icons stay round.
pub const SHELL_RADIUS: f32 = 4.0;

/// The hamburger menu panel's fixed width (Chrome/Firefox-sized,
/// not half-screen).
pub const MENU_PANEL_WIDTH: f32 = 260.0;

/// The bookmark dialog's fixed width (follow-up UI work,
/// 2026-11-27; widened to 400 in the bookmarks round,
/// 2026-11-28, for the Firefox-style Name / URL / Tags /
/// Keyword rows).
pub const BOOKMARK_DIALOG_WIDTH: f32 = 400.0;

/// The settings dialog's width (settings round, 2026-11-28):
/// a little wider than the bookmark dialog — it holds rows of
/// option buttons (theme, provider, placement).
pub const SETTINGS_DIALOG_WIDTH: f32 = 440.0;

/// The DevTools console panel's width (Stage 17, M9 stages
/// 1–2): wider than the settings dialog — console lines and a
/// REPL input want room, and the panel floats over the page.
pub const DEVTOOLS_PANEL_WIDTH: f32 = 640.0;

/// The DevTools console panel's log area height (Stage 17, M9
/// stages 1–2): a fixed scrollable area, so the panel stays a
/// floating dialog instead of growing with a chatty page.
pub const DEVTOOLS_PANEL_LOG_HEIGHT: f32 = 320.0;

/// The per-tab console-line cap (Stage 17, M9 stages 1–2):
/// the shell keeps the last 1,000 `(level, message)` lines per
/// tab (oldest evicted), so a page that logs in a loop cannot
/// grow the shell's buffer without bound.
pub const CONSOLE_LINES_CAP: usize = 1000;

/// The bookmarks-bar popups' (context menu, folder dropdown)
/// top offset (bookmarks round, 2026-11-28): they hang just
/// below the bar — the nav bar (~48px + divider) + the 30px
/// bar + a divider. Approximate by design, like
/// `MENU_OVERLAY_TOP`.
pub const BAR_POPUP_TOP: f32 = MENU_OVERLAY_TOP + 32.0;

/// The bookmarks-bar popups' fixed width (bookmarks round,
/// 2026-11-28): a little narrower than the menu panel, sized
/// for a chip's worth of text.
pub const BAR_POPUP_WIDTH: f32 = 220.0;

/// The new-folder dialog's fixed width (bookmarks round,
/// 2026-11-28): one name field + two buttons.
pub const FOLDER_DIALOG_WIDTH: f32 = 320.0;

/// The bookmarks bar's folder glyph (bookmarks round,
/// 2026-11-28): a built-in 16×16 SVG, the same shape as the
/// other bundled icons (own drawing, filled with the theme's
/// icon color via `symbolic(true)` — no file needed, folders
/// have no favicon to load).
pub const FOLDER_ICON_SVG: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><path d="M2 4.5V4a1 1 0 0 1 1-1h3l1.5 2H13a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4.5z" fill="#232323"/></svg>"##;

/// The bookmarks bar's pencil glyph (bookmarks round,
/// 2026-11-28): a built-in 16×16 SVG, same treatment as the
/// folder glyph — the edit-mode toggle at the bar's far right.
pub const PENCIL_ICON_SVG: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><path d="M11 2.5a1.77 1.77 0 0 1 2.5 0l0 0a1.77 1.77 0 0 1 0 2.5L6 12.5 2.5 13.5 3.5 10 11 2.5z" fill="#232323"/></svg>"##;

/// The bookmark dialog's star glyph (bookmarks round,
/// 2026-11-28): a built-in 16×16 SVG, same treatment as the
/// folder glyph — the title row's star, matching Firefox's
/// dialog (the nav star is a text glyph; the dialog uses the
/// icon so it renders in both themes).
pub const STAR_ICON_SVG: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><path d="M8 1.5l2.1 4.2 4.6.7-3.3 3.2.8 4.6L8 12l-4.2 2.2.8-4.6-3.3-3.2 4.6-.7L8 1.5z" fill="#232323"/></svg>"##;

/// The floating menu overlay's top offset (follow-up UI work,
/// 2026-11-27): how far below the window top the popup hangs, so
/// it clears the hamburger button it drops from. The nav bar is
/// ~48px + a divider. Approximate by design — the panel has its
/// own background and shadow, so a few px either way is cosmetic —
/// but it must stay below the nav bar, never over the button.
pub const MENU_OVERLAY_TOP: f32 = 56.0;

/// The menu overlay's top offset: it clears the nav bar.
pub fn menu_overlay_top_offset() -> f32 {
    MENU_OVERLAY_TOP
}

/// A collapsed vertical tab chip: a 32px square showing only the
/// favicon (UI round, 2026-11-27).
pub const TAB_CHIP: f32 = 32.0;

/// A hovered vertical tab's flyout width: favicon + centered title
/// + close button.
pub const TAB_FLYOUT: f32 = 200.0;

/// The gap between tabs in a vertical column, and the rail's
/// padding on every side.
pub const TAB_GAP: f32 = 4.0;

/// A vertical tab column's width: the 32px chip plus the rail
/// padding on both sides.
pub const RAIL_WIDTH: f32 = 44.0;

/// One tab's pitch in a vertical column: chip + gap.
pub const TAB_PITCH: f32 = TAB_CHIP + TAB_GAP;

/// The default tab placement: the left edge.
pub const DEFAULT_TAB_PLACEMENT: TabPlacement = TabPlacement::Left;

/// The hover slide-out's frame interval (~60 fps).
pub const HOVER_TICK: std::time::Duration = std::time::Duration::from_millis(16);

/// One hover tick's progress step: 18 ticks ≈ 0.3 s, the mockup's
/// slide-out duration.
pub const HOVER_TICK_STEP: f32 = 1.0 / 18.0;

/// Which edge(s) the tab strip sits on (UI round, 2026-11-27):
/// one edge or both. Tabs are always vertical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TabPlacement {
    /// A single column on the left edge (the default).
    #[default]
    Left,
    /// A single column on the right edge.
    Right,
    /// Columns on both edges: the left fills first, the overflow
    /// spills to the right.
    Both,
}

/// One edge of the window a vertical tab column can sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabEdge {
    Left,
    Right,
}

/// The edges a placement mode puts a tab column on.
pub fn tab_strip_edges(mode: TabPlacement) -> &'static [TabEdge] {
    match mode {
        TabPlacement::Left => &[TabEdge::Left],
        TabPlacement::Right => &[TabEdge::Right],
        TabPlacement::Both => &[TabEdge::Left, TabEdge::Right],
    }
}

/// How many tabs fit in one vertical column of `height` px: 32px
/// chips + 4px gaps, 4px rail padding top and bottom, and the 32px
/// "+" button + one gap at the bottom. At least one — a column
/// always shows its first tab.
pub fn rail_capacity(height: f32) -> usize {
    ((height - 2.0 * TAB_GAP - TAB_CHIP - TAB_GAP) / TAB_PITCH)
        .floor()
        .max(1.0) as usize
}

/// Split the ordered tab list across the two columns in Both mode:
/// fill the left column first; only the overflow goes right.
/// Returns (left_count, right_count).
pub fn distribute_tabs(tab_count: usize, capacity: usize) -> (usize, usize) {
    let left = tab_count.min(capacity);
    (left, tab_count - left)
}

/// Which tabs a column shows: (start index, count) into the one
/// ordered tab list. Both columns read the same list — the left
/// takes the first `capacity` tabs, the right the rest; a
/// single-column mode shows the whole list on its edge.
pub fn column_range(
    placement: TabPlacement,
    edge: TabEdge,
    tab_count: usize,
    capacity: usize,
) -> (usize, usize) {
    match (placement, edge) {
        (TabPlacement::Left, TabEdge::Left) | (TabPlacement::Right, TabEdge::Right) => {
            (0, tab_count)
        }
        (TabPlacement::Both, TabEdge::Left) => {
            let (left, _) = distribute_tabs(tab_count, capacity);
            (0, left)
        }
        (TabPlacement::Both, TabEdge::Right) => {
            let (left, right) = distribute_tabs(tab_count, capacity);
            (left, right)
        }
        _ => (0, 0),
    }
}

/// Where a tab's flyout goes: (edge, index within that column).
/// None when the tab is not in a visible column.
pub fn flyout_column_position(
    placement: TabPlacement,
    index: usize,
    tab_count: usize,
    capacity: usize,
) -> Option<(TabEdge, usize)> {
    if index >= tab_count {
        return None;
    }
    match placement {
        TabPlacement::Left => Some((TabEdge::Left, index)),
        TabPlacement::Right => Some((TabEdge::Right, index)),
        TabPlacement::Both => {
            let (left, _) = distribute_tabs(tab_count, capacity);
            if index < left {
                Some((TabEdge::Left, index))
            } else {
                Some((TabEdge::Right, index - left))
            }
        }
    }
}

/// The flyout's width at `progress` (0.0 collapsed ..= 1.0 open):
/// the 32px chip elongates to the 200px flyout.
pub fn flyout_width(progress: f32) -> f32 {
    TAB_CHIP + (TAB_FLYOUT - TAB_CHIP) * progress.clamp(0.0, 1.0)
}

/// Whether the flyout shows its title and close button yet: only
/// once it is more than half open, so the text never squeezes
/// into a half-grown chip.
pub fn flyout_shows_text(progress: f32) -> bool {
    progress > 0.5
}

/// A placement entry's label.
pub fn placement_label(placement: TabPlacement) -> &'static str {
    match placement {
        TabPlacement::Left => "Tabs on left",
        TabPlacement::Right => "Tabs on right",
        TabPlacement::Both => "Tabs on both sides",
    }
}

impl TabPlacement {
    /// The persisted key for this placement.
    pub fn key(self) -> &'static str {
        match self {
            TabPlacement::Left => "left",
            TabPlacement::Right => "right",
            TabPlacement::Both => "both",
        }
    }

    /// The placement from a persisted key (`None` = unknown key).
    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "left" => Some(TabPlacement::Left),
            "right" => Some(TabPlacement::Right),
            "both" => Some(TabPlacement::Both),
            _ => None,
        }
    }
}

/// The bookmarks bar's entries, in store order: (title, url).
/// The bar reads the store, never a hardcoded list.
/// The initial letter for a URL: the host's first character
/// ("example.com" → 'E'), else the title's first character,
/// else '?'.
fn initial_letter(url: &str, title: &str) -> char {
    let host = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url);
    host.chars()
        .next()
        .or_else(|| title.chars().next())
        .unwrap_or('?')
        .to_ascii_uppercase()
}

/// Open the OS file chooser for a custom favicon image
/// (follow-up UI work, 2026-11-27): `rfd`'s async dialog
/// (portal on Linux), filtered to image files. `None` is a
/// cancel; `Some(Err(..))` is a file that would not decode
/// (wrong format, corrupt, or absurdly large — the decode
/// limits reject it before pixels are allocated).
async fn pick_favicon_file() -> Option<Result<(Vec<u8>, pluto_paint::DecodedImage), String>> {
    let picked = rfd::AsyncFileDialog::new()
        .add_filter(
            "Images",
            &["png", "jpg", "jpeg", "gif", "webp", "ico", "svg", "bmp"],
        )
        .pick_file()
        .await?;
    let bytes = picked.read().await;
    match pluto_paint::decode_image(&bytes) {
        Ok(decoded) => Some(Ok((bytes, decoded))),
        Err(e) => Some(Err(e.to_string())),
    }
}

/// The bookmarks bar's entries, in store order (follow-up UI
/// work, 2026-11-27): a bookmark chip or a folder chip per
/// entry. The bar reads the store, never a hardcoded list.
pub fn bookmarks_bar_entries(store: &BookmarkStore) -> Vec<BarEntry> {
    store
        .entries()
        .iter()
        .map(|entry| match entry {
            BookmarkEntry::Bookmark(bookmark) => BarEntry::Bookmark {
                title: bookmark.title.clone(),
                url: bookmark.url.clone(),
            },
            BookmarkEntry::Folder(folder) => BarEntry::Folder {
                id: folder.id.clone(),
                name: folder.name.clone(),
            },
        })
        .collect()
}

/// Parse the dialog's tags field: split on commas, trim each
/// tag, drop empties, dedupe preserving first-seen order
/// (bookmarks round, 2026-11-28).
pub fn parse_tags(text: &str) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    for part in text.split(',') {
        let tag = part.trim().to_string();
        if !tag.is_empty() && !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    tags
}

/// Toggle one tag in the dialog's tags field: append it when
/// absent, remove it when present, keeping the field a
/// comma-separated list (bookmarks round, 2026-11-28).
fn toggle_tag_in_text(text: &str, tag: &str) -> String {
    let mut tags = parse_tags(text);
    if let Some(pos) = tags.iter().position(|t| t == tag) {
        tags.remove(pos);
    } else {
        tags.push(tag.to_string());
    }
    tags.join(", ")
}

/// Shorten `s` to at most `max_chars` characters for a bar button
/// (bookmark titles are URLs, and a long one would push the rest
/// of the bar off-screen).
fn ellipsize(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// Which theme button a 4.0px-radius shell button is based on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmallButtonBase {
    /// The plain button (the bookmark dialog's favicon button).
    Standard,
    /// The plain text button (menu items, bookmarks bar, list panels).
    Text,
    /// The accent (suggested) button — the nav bar's Go button.
    Suggested,
}

impl SmallButtonBase {
    fn class(self) -> cosmic::theme::Button {
        match self {
            SmallButtonBase::Standard => cosmic::theme::Button::Standard,
            SmallButtonBase::Text => cosmic::theme::Button::Text,
            SmallButtonBase::Suggested => cosmic::theme::Button::Suggested,
        }
    }
}

/// A `Button::Custom` style that looks exactly like the theme's
/// `base` button but with the shell's 4.0px radius (UI round,
/// 2026-11-27). `selected` is baked in — the Custom closures do
/// not receive it.
fn small_button_style(base: SmallButtonBase, selected: bool) -> cosmic::theme::Button {
    use cosmic::iced::core::border::Radius;
    use cosmic::widget::button::Style as ButtonStyle;
    let radius = Radius::from(SHELL_RADIUS);
    cosmic::theme::Button::Custom {
        active: Box::new(move |focused, theme| {
            let mut style: ButtonStyle =
                <cosmic::Theme as cosmic::widget::button::Catalog>::active(
                    theme,
                    focused,
                    selected,
                    &base.class(),
                );
            style.border_radius = radius;
            style
        }),
        disabled: Box::new(move |theme| {
            let mut style: ButtonStyle =
                <cosmic::Theme as cosmic::widget::button::Catalog>::disabled(
                    theme,
                    &base.class(),
                );
            style.border_radius = radius;
            style
        }),
        hovered: Box::new(move |focused, theme| {
            let mut style: ButtonStyle =
                <cosmic::Theme as cosmic::widget::button::Catalog>::hovered(
                    theme,
                    focused,
                    selected,
                    &base.class(),
                );
            style.border_radius = radius;
            style
        }),
        pressed: Box::new(move |focused, theme| {
            let mut style: ButtonStyle =
                <cosmic::Theme as cosmic::widget::button::Catalog>::pressed(
                    theme,
                    focused,
                    selected,
                    &base.class(),
                );
            style.border_radius = radius;
            style
        }),
    }
}

/// A `TextInput::Custom` style that looks exactly like the theme's
/// default input but with the shell's 4.0px radius (UI round,
/// 2026-11-27): the URL bar.
fn small_text_input_style() -> cosmic::theme::TextInput {
    use cosmic::iced::core::border::Radius;
    use cosmic::widget::text_input::Appearance;
    let radius = Radius::from(SHELL_RADIUS);
    cosmic::theme::TextInput::Custom {
        active: Box::new(move |theme| {
            let mut appearance: Appearance =
                <cosmic::Theme as cosmic::widget::text_input::StyleSheet>::active(
                    theme,
                    &cosmic::theme::TextInput::Default,
                );
            appearance.border_radius = radius;
            appearance
        }),
        error: Box::new(move |theme| {
            let mut appearance: Appearance =
                <cosmic::Theme as cosmic::widget::text_input::StyleSheet>::error(
                    theme,
                    &cosmic::theme::TextInput::Default,
                );
            appearance.border_radius = radius;
            appearance
        }),
        hovered: Box::new(move |theme| {
            let mut appearance: Appearance =
                <cosmic::Theme as cosmic::widget::text_input::StyleSheet>::hovered(
                    theme,
                    &cosmic::theme::TextInput::Default,
                );
            appearance.border_radius = radius;
            appearance
        }),
        focused: Box::new(move |theme| {
            let mut appearance: Appearance =
                <cosmic::Theme as cosmic::widget::text_input::StyleSheet>::focused(
                    theme,
                    &cosmic::theme::TextInput::Default,
                );
            appearance.border_radius = radius;
            appearance
        }),
        disabled: Box::new(move |theme| {
            let mut appearance: Appearance =
                <cosmic::Theme as cosmic::widget::text_input::StyleSheet>::disabled(
                    theme,
                    &cosmic::theme::TextInput::Default,
                );
            appearance.border_radius = radius;
            appearance
        }),
    }
}

/// A floating tab chip's container style: the theme's component
/// background, a 4.0px radius, a 1px divider border (2px accent
/// when active), and a drop shadow so the chip stays readable
/// floating over any page (UI round, 2026-11-27).
fn tab_chip_style(theme: &cosmic::Theme, active: bool) -> cosmic::iced::widget::container::Style {
    use cosmic::iced::core::{Background, Border, Color, Shadow, Vector, border::Radius};
    let component = &theme.current_container().component;
    let (background, border_color, border_width) = if active {
        (
            Color::from(component.hover),
            Color::from(theme.cosmic().accent_color()),
            2.0,
        )
    } else {
        (
            Color::from(component.base),
            Color::from(component.divider),
            1.0,
        )
    };
    cosmic::iced::widget::container::Style {
        background: Some(Background::Color(background)),
        border: Border {
            radius: Radius::from(SHELL_RADIUS),
            width: border_width,
            color: border_color,
        },
        shadow: Shadow {
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.35,
            },
            offset: Vector::new(0.0, 2.0),
            blur_radius: 8.0,
        },
        ..Default::default()
    }
}

/// The bookmarks bar's container style: the theme's component
/// background as a flat 30px strip, no radius, no border, no
/// shadow (UI round, 2026-11-27) — the bar reads like the
/// mockup's dark strip; the bottom divider in `view` is its
/// lower edge.
fn bookmarks_bar_style(theme: &cosmic::Theme) -> cosmic::iced::widget::container::Style {
    use cosmic::iced::core::{Background, Border, Color, Shadow, Vector, border::Radius};
    let component = &theme.current_container().component;
    cosmic::iced::widget::container::Style {
        background: Some(Background::Color(Color::from(component.base))),
        border: Border {
            radius: Radius::from(0.0),
            width: 0.0,
            color: Color::from(component.divider),
        },
        shadow: Shadow {
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
            offset: Vector::new(0.0, 0.0),
            blur_radius: 0.0,
        },
        ..Default::default()
    }
}

/// A menu / list panel's container style: the theme's component
/// background, a 4.0px radius, a 1px divider border, and a drop
/// shadow (UI round, 2026-11-27).
fn panel_container_style(theme: &cosmic::Theme) -> cosmic::iced::widget::container::Style {
    use cosmic::iced::core::{Background, Border, Color, Shadow, Vector, border::Radius};
    let component = &theme.current_container().component;
    cosmic::iced::widget::container::Style {
        background: Some(Background::Color(Color::from(component.base))),
        border: Border {
            radius: Radius::from(SHELL_RADIUS),
            width: 1.0,
            color: Color::from(component.divider),
        },
        shadow: Shadow {
            color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.30,
            },
            offset: Vector::new(0.0, 4.0),
            blur_radius: 16.0,
        },
        ..Default::default()
    }
}


/// The hamburger menu's items, in order. The menu lists ONLY
/// items with a working backend behind them (Joel's rule): each
/// entry here names an action the shell really performs —
/// History opens the session-history list, New Tab opens a tab
/// in this window (Stage 11 Rec 4), New Window spawns a second
/// Pluto process, Settings opens the settings dialog (settings
/// round, 2026-11-28), DevTools opens the console panel (Stage
/// 17, M9 stages 1–2). Bookmarks used to open the bookmark list
/// panel here; the entry was removed 2026-11-28 (Joel) and the
/// panel with it — bookmarks live on the bookmarks bar and in
/// the add/edit dialog. Nothing else from the reference
/// screenshot qualifies (no download-prompt UI, no zoom / find /
/// sync / profiles / passwords / extensions / print / save-page /
/// translate backend, and Porn Mode is M12, not built), so none
/// of it is listed. No shortcuts are shown next to the items:
/// none of the screenshot's shortcuts are wired.
pub fn menu_items() -> &'static [&'static str] {
    &["History", "New Tab", "New Window", "Settings", "DevTools"]
}

/// The theme preference (settings round, 2026-11-28):
/// follow the OS, or force light / dark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemePref {
    /// Follow the OS theme (the default).
    #[default]
    System,
    /// Force the light theme.
    Light,
    /// Force the dark theme.
    Dark,
}

impl ThemePref {
    /// The persisted key for this preference.
    pub fn key(self) -> &'static str {
        match self {
            ThemePref::System => "system",
            ThemePref::Light => "light",
            ThemePref::Dark => "dark",
        }
    }

    /// The preference from a persisted key (`None` = unknown key).
    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "system" => Some(ThemePref::System),
            "light" => Some(ThemePref::Light),
            "dark" => Some(ThemePref::Dark),
            _ => None,
        }
    }

    /// The label the settings dialog shows.
    pub fn label(self) -> &'static str {
        match self {
            ThemePref::System => "System default",
            ThemePref::Light => "Light",
            ThemePref::Dark => "Dark",
        }
    }

    /// All preferences, in dialog order.
    pub fn all() -> [ThemePref; 3] {
        [ThemePref::Light, ThemePref::Dark, ThemePref::System]
    }
}

/// Whether the theme preference means dark right now: a
/// forced preference decides; System asks the OS.
fn theme_pref_is_dark(pref: ThemePref) -> bool {
    match pref {
        ThemePref::System => cosmic::theme::is_dark(),
        ThemePref::Light => false,
        ThemePref::Dark => true,
    }
}

/// The shell's small preferences: which parts of the shell are
/// visible and how it behaves, persisted across restarts in
/// `preferences.json` in the profile dir (bookmarks round,
/// 2026-11-28; theme / homepage / search provider / tab
/// placement added by the settings round, 2026-11-28). The file
/// is read on startup and rewritten whenever a preference
/// changes; a missing or corrupt file means defaults, and a
/// file from an older version (fewer keys) loads with defaults
/// for the missing ones — never a failed launch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellPrefs {
    show_bookmarks_bar: bool,
    theme: ThemePref,
    homepage: String,
    search_provider: page::SearchProvider,
    tab_placement: TabPlacement,
}

impl Default for ShellPrefs {
    fn default() -> Self {
        Self {
            show_bookmarks_bar: true,
            theme: ThemePref::default(),
            homepage: page::DEFAULT_URL.to_string(),
            search_provider: page::SearchProvider::default(),
            tab_placement: TabPlacement::default(),
        }
    }
}

/// The profile dir's `preferences.json` (same profile dir the
/// bookmark store and history live in). `None` when there is no
/// profile dir (no HOME) — the caller falls back to defaults.
fn shell_prefs_file() -> Option<std::path::PathBuf> {
    profile_dir().map(|dir| dir.join(PREFS_FILE))
}

/// The preferences file's name in the profile dir.
const PREFS_FILE: &str = "preferences.json";

/// Load the shell preferences (missing/corrupt → defaults).
fn load_shell_prefs() -> ShellPrefs {
    let Some(path) = shell_prefs_file() else {
        return ShellPrefs::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return ShellPrefs::default();
    };
    let Ok(parsed) = parse_json(&text) else {
        eprintln!("[pluto] corrupt preferences; using defaults");
        return ShellPrefs::default();
    };
    let mut prefs = ShellPrefs::default();
    if let Some(JsonValue::Bool(show)) = parsed.get("show_bookmarks_bar") {
        prefs.show_bookmarks_bar = *show;
    }
    if let Some(key) = parsed.get("theme").and_then(JsonValue::as_str)
        && let Some(theme) = ThemePref::from_key(key)
    {
        prefs.theme = theme;
    }
    if let Some(homepage) = parsed.get("homepage").and_then(JsonValue::as_str)
        && let Ok(normalized) = page::normalize_url(homepage)
    {
        // A hand-edited or stale homepage that no longer
        // normalizes falls back to the default, never a
        // broken startup URL.
        prefs.homepage = normalized;
    }
    if let Some(key) = parsed
        .get("search_provider")
        .and_then(JsonValue::as_str)
        && let Some(provider) = page::SearchProvider::from_key(key)
    {
        prefs.search_provider = provider;
    }
    if let Some(key) = parsed.get("tab_placement").and_then(JsonValue::as_str)
        && let Some(placement) = TabPlacement::from_key(key)
    {
        prefs.tab_placement = placement;
    }
    prefs
}

/// Save the shell preferences (a failure is logged, not fatal —
/// the in-memory value still applies for this run).
fn save_shell_prefs(prefs: &ShellPrefs) {
    let Some(dir) = profile_dir() else {
        return;
    };
    let text = format!(
        "{{\"version\":2,\"show_bookmarks_bar\":{},\"theme\":{},\"homepage\":{},\"search_provider\":{},\"tab_placement\":{}}}",
        prefs.show_bookmarks_bar,
        json_escape(prefs.theme.key()),
        json_escape(&prefs.homepage),
        json_escape(prefs.search_provider.key()),
        json_escape(prefs.tab_placement.key()),
    );
    if let Err(e) = write_file_atomic(&dir, PREFS_FILE, &text) {
        eprintln!("[pluto] failed to save preferences ({e})");
    }
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
        core.window.header_title = "Pluto ~ The far out browser".to_string();
        // Settings round (2026-11-28): the preferences load
        // once here and feed the whole shell — the homepage is
        // the first tab's URL, the theme preference decides the
        // starting dark flag and (when forced) the shell theme.
        let prefs = load_shell_prefs();
        let url = prefs.homepage.clone();
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
        // Follow-up UI work (2026-11-27): decode each stored
        // custom favicon once, at startup. A file that no longer
        // decodes (removed by hand, corrupt) is skipped with a
        // log line — the bookmark itself survives, it just shows
        // no custom icon.
        let bookmarks = load_bookmarks();
        let mut bookmark_favicons = std::collections::HashMap::new();
        for bookmark in bookmarks.bookmarks() {
            if bookmark.favicon.is_some() {
                match load_bookmark_favicon(&bookmark.url)
                    .and_then(|bytes| pluto_paint::decode_image(&bytes).ok())
                {
                    Some(decoded) => {
                        bookmark_favicons.insert(bookmark.url.clone(), decoded);
                    }
                    None => {
                        eprintln!(
                            "[pluto] custom favicon for {} unreadable; showing none",
                            bookmark.url
                        );
                    }
                }
            }
        }
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
            // M6 Phase 3 (Rec 12): the real theme at startup —
            // the preference decides, not just the OS (settings
            // round, 2026-11-28): a forced Light / Dark starts
            // dark or light; System asks the OS. The workers
            // get it on `RendererReady` and on every change
            // after.
            dark_theme: theme_pref_is_dark(prefs.theme),
            bookmarks,
            bookmark_favicons,
            bookmark_dialog: None,
            // Bookmarks round (2026-11-28): the bar's visibility
            // comes from the persisted preferences; edit mode,
            // the context menu, the folder dropdown, and the
            // folder dialog all start closed.
            show_bookmarks_bar: prefs.show_bookmarks_bar,
            bookmarks_edit_mode: false,
            bar_context_menu: None,
            bar_context_chip: false,
            open_folder: None,
            folder_dialog: None,
            // Settings round (2026-11-28): the persisted
            // preferences; the dialog itself starts closed.
            theme_pref: prefs.theme,
            homepage: prefs.homepage.clone(),
            search_provider: prefs.search_provider,
            settings_dialog: None,
            devtools_panel: None,
            menu_open: false,
            panel: None,
            tab_placement: prefs.tab_placement,
            flyout_tab: None,
            flyout_progress: 0.0,
            flyout_target: 0.0,
            left_rail_scroll: 0.0,
            right_rail_scroll: 0.0,
            modifiers: pluto_ipc::Modifiers::default(),
            buttons_held: 0,
            last_wheel: None,
            broker,
        };
        // Settings round (2026-11-28): a forced theme applies
        // at startup — the shell chrome switches before the
        // first paint, and the workers get it via `dark_theme`
        // on `RendererReady`. System needs no task: libcosmic
        // already follows the OS.
        let theme_task = match prefs.theme {
            ThemePref::System => Task::none(),
            ThemePref::Light => cosmic::command::set_theme(cosmic::Theme::light()),
            ThemePref::Dark => cosmic::command::set_theme(cosmic::Theme::dark()),
        };
        (app, theme_task)
    }

    /// The OS theme mode changed (M6 Phase 3, Rec 12): libcosmic
    /// calls this when the system light/dark mode flips. The mode
    /// carries the dark flag directly — the authoritative signal
    /// for `prefers-color-scheme`. Settings round (2026-11-28):
    /// a forced Light / Dark preference wins over the OS signal
    /// (the workers keep the forced value); System follows the
    /// OS as before.
    fn system_theme_mode_update(
        &mut self,
        _keys: &[&'static str],
        _new_theme: &cosmic::cosmic_theme::ThemeMode,
    ) -> Task<Self::Message> {
        self.apply_theme(self.effective_dark());
        Task::none()
    }

    /// The OS theme itself changed (M6 Phase 3, Rec 12): a theme
    /// switch that keeps the mode (e.g. a new light theme) still
    /// re-reads the active theme — libcosmic installs it before
    /// calling this hook, so `is_dark()` sees the new one.
    /// Settings round (2026-11-28): same preference rule as
    /// the mode hook — a forced theme is not disturbed.
    fn system_theme_update(
        &mut self,
        _keys: &[&'static str],
        _new_theme: &cosmic::cosmic_theme::Theme,
    ) -> Task<Self::Message> {
        self.apply_theme(self.effective_dark());
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
        // The hover slide-out's ticker (UI round, 2026-11-27):
        // only while a flyout is expanding or collapsing, so the
        // shell is not woken at 60 fps the rest of the time.
        if self.hover_animating() {
            subscriptions.push(
                cosmic::iced::time::every(HOVER_TICK).map(|_| Message::HoverTick),
            );
        }
        cosmic::iced::Subscription::batch(subscriptions)
    }

    fn update(&mut self, message: Self::Message) -> Task<Self::Message> {
        match message {
            Message::UrlChanged(val) => self.active_mut().url_input = val,
            Message::Navigate => {
                let input = self.active().url_input.clone();
                // Bookmarks round (2026-11-28): a bookmark
                // keyword wins over URL/search handling — typing
                // one in the address bar opens that bookmark.
                if let Some(bookmark) = self.bookmarks.find_by_keyword(&input) {
                    self.commit(bookmark.url.clone());
                    return Task::none();
                }
                // Settings round (2026-11-28): an empty address
                // bar goes to the homepage, and a query (words,
                // not a host) goes to the default search
                // provider — the browser had no search at all
                // before this, so words used to become a broken
                // `https://<words>` URL.
                if input.trim().is_empty() {
                    let homepage = self.homepage.clone();
                    self.commit(homepage);
                    return Task::none();
                }
                if page::is_search_query(&input) {
                    let url = self.search_provider.search_url(&input);
                    self.commit(url);
                    return Task::none();
                }
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
                let homepage = self.homepage.clone();
                self.new_tab(homepage);
            }
            Message::CloseTab(index) => {
                self.close_tab(index);
                // Strip indices shift on close: a stale flyout
                // would point at the wrong tab.
                self.clear_flyout();
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
                    RendererToBrowser::ConsoleLines { lines } => {
                        // Stage 17 (M9 stages 1–2): the worker's
                        // per-tick console drain — append to this
                        // tab's buffer (capped, oldest evicted).
                        // The lines are the page's own output, so
                        // they are stored as plain text and the
                        // panel renders them as plain text.
                        let tab = &mut self.tabs[tab_index];
                        tab.push_console_lines(lines);
                    }
                    RendererToBrowser::DevToolsEvalResult { result } => {
                        // Stage 17 (M9 stages 1–2): the REPL's
                        // answer for this tab — the snippet's value
                        // formatted on the tab thread (or the
                        // formatted exception). Level `""`: the
                        // panel renders it as-is, under the `>`
                        // echo the submit already stored.
                        let tab = &mut self.tabs[tab_index];
                        tab.push_console_lines(vec![(String::new(), result)]);
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
                        let dead = tab.crash_times.len() >= 3;
                        if dead {
                            tab.dead = true;
                            tab.status = "Renderer crashed repeatedly — press Reload to retry"
                                .to_string();
                        } else {
                            tab.status =
                                format!("Renderer crashed: {reason} — press Reload to retry");
                        }
                        tab.progress = None;
                        if dead {
                            // A dead tab takes no further navigation
                            // — not even the crash page. A renderer
                            // whose layout thread died but whose
                            // process survived crashes every
                            // about:crashed load the same way
                            // ("layout actor is gone"), so
                            // re-navigating here spins forever
                            // (epochs 3–121+ on google.com,
                            // 2026-11-28) instead of leaving the
                            // tab dead for Reload. The stale list
                            // stays visible with the status above.
                            return Task::none();
                        }
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
                        eprintln!("[pluto-broker] shell got RequestResource id={} url={}", request.request_id, request.url);
                        // Stage 13 Rec 11: serve the request on the
                        // broker and forward its events to the tab as
                        // they arrive. The future runs serve + the
                        // forward loop itself (no extra spawn): the
                        // events channel closes when the fetch ends.
                        let Some(tx) = self.tabs[tab_index].renderer_tx.clone() else {
                            return Task::none();
                        };
                        let broker = Arc::clone(&self.broker);
                        let request_id = request.request_id;
                        return Task::perform(
                            async move {
                                let (events_tx, mut events_rx) =
                                    tokio::sync::mpsc::channel(64);
                                let serve = broker.serve(request, events_tx);
                                tokio::pin!(serve);
                                loop {
                                    tokio::select! {
                                        _ = &mut serve => {
                                            // serve ended — drain any
                                            // event still buffered in
                                            // events_rx (try_recv only:
                                            // serve has returned, so
                                            // nothing more will ever
                                            // arrive) before breaking,
                                            // or the terminal Done —
                                            // and any trailing chunks —
                                            // is dropped unread and the
                                            // renderer's fetch waits
                                            // forever. This was the
                                            // paint stall: `forward end
                                            // reason=serve-done
                                            // buffered=3` with no
                                            // preceding `forward event
                                            // … kind=done` (confirmed
                                            // on-box 2026-11-27).
                                            while let Ok(event) = events_rx.try_recv() {
                                                let (kind, bytes) = match &event {
                                                    pluto_ipc::ResourceEvent::ResponseHeaders { .. } => ("headers", 0),
                                                    pluto_ipc::ResourceEvent::BodyChunk { bytes, .. } => ("chunk", bytes.len()),
                                                    pluto_ipc::ResourceEvent::Done { .. } => ("done", 0),
                                                    pluto_ipc::ResourceEvent::Error { .. } => ("error", 0),
                                                    pluto_ipc::ResourceEvent::Cancelled { .. } => ("cancelled", 0),
                                                };
                                                eprintln!("[pluto-worker] forward event id={} kind={} bytes={}", request_id, kind, bytes);
                                                if tx.send(BrowserToRenderer::ResourceEvent(event))
                                                    .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            eprintln!("[pluto-worker] forward end id={} reason=serve-done buffered={}", request_id, events_rx.len());
                                            break;
                                        }
                                        event = events_rx.recv() => match event {
                                            Some(event) => {
                                                let (kind, bytes) = match &event {
                                                    pluto_ipc::ResourceEvent::ResponseHeaders { .. } => ("headers", 0),
                                                    pluto_ipc::ResourceEvent::BodyChunk { bytes, .. } => ("chunk", bytes.len()),
                                                    pluto_ipc::ResourceEvent::Done { .. } => ("done", 0),
                                                    pluto_ipc::ResourceEvent::Error { .. } => ("error", 0),
                                                    pluto_ipc::ResourceEvent::Cancelled { .. } => ("cancelled", 0),
                                                };
                                                eprintln!("[pluto-worker] forward event id={} kind={} bytes={}", request_id, kind, bytes);
                                                if tx.send(BrowserToRenderer::ResourceEvent(event))
                                                    .is_err()
                                                {
                                                    eprintln!("[pluto-worker] forward end id={} reason=send-failed buffered={}", request_id, events_rx.len());
                                                    break;
                                                }
                                            }
                                            None => {
                                                eprintln!("[pluto-worker] forward end id={} reason=channel-closed buffered={}", request_id, events_rx.len());
                                                break;
                                            }
                                        },
                                    }
                                }
                                Message::BrokerServeDone
                            },
                            cosmic::Action::App,
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
                                        _ = &mut serve => {
                                            // Drain any event still
                                            // buffered in events_rx
                                            // before breaking (same
                                            // race as the document
                                            // fetch loop: a frame
                                            // buffered when the pump
                                            // ends would otherwise be
                                            // dropped unread).
                                            while let Ok(event) = events_rx.try_recv() {
                                                if tx.send(BrowserToRenderer::WebSocketEvent {
                                                    socket_id,
                                                    event,
                                                })
                                                .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            break;
                                        }
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
                            cosmic::Action::App,
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
                // Follow-up UI work (2026-11-27): the star opens
                // the bookmark dialog instead of toggling. The
                // name prefills with the page title (the URL
                // until the first TitleChanged arrives); a page
                // that is already bookmarked opens prefilled for
                // editing, with a Remove button in the dialog.
                let url = self.active().active_url.clone();
                if !url.is_empty() {
                    let title = self.active().title.clone();
                    let existing = self.bookmarks.bookmark(&url);
                    self.bookmark_dialog =
                        Some(BookmarkDialog::new(&url, &title, existing));
                    self.menu_open = false;
                    self.panel = None;
                    self.close_bar_popups();
                    // The name is selected on open (Firefox does
                    // this too): focus the name field, then select
                    // all of it, so typing replaces the prefill.
                    return cosmic::iced::widget::operation::focus::<
                        cosmic::Action<Message>,
                    >("bookmark-dialog-name")
                    .chain(cosmic::iced::widget::operation::select_all(
                        "bookmark-dialog-name",
                    ));
                }
            }
            Message::BookmarkDialogNameChanged(name) => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.name = name;
                }
            }
            Message::BookmarkDialogUrlChanged(url) => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.url = url;
                }
            }
            Message::BookmarkDialogTagsChanged(tags) => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.tags = tags;
                }
            }
            Message::BookmarkDialogKeywordChanged(keyword) => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.keyword = keyword;
                }
            }
            Message::BookmarkDialogToggleTags => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.show_tags = !dialog.show_tags;
                }
            }
            Message::BookmarkDialogTagToggle(tag) => {
                if let Some(dialog) = self.bookmark_dialog.as_mut() {
                    dialog.tags = toggle_tag_in_text(&dialog.tags, &tag);
                }
            }
            Message::BookmarkDialogPickFavicon => {
                // The chooser runs as an iced task: the OS dialog
                // is async (portal on Linux), and the decode
                // happens in the task, off the UI thread.
                return Task::perform(
                    async {
                        Message::BookmarkDialogFaviconPicked(pick_favicon_file().await)
                    },
                    cosmic::Action::App,
                );
            }
            Message::BookmarkDialogFaviconPicked(picked) => {
                let Some(dialog) = self.bookmark_dialog.as_mut() else {
                    return Task::none();
                };
                match picked {
                    // Cancelled: leave the dialog exactly as it
                    // was (keep any earlier pick and error).
                    None => {}
                    Some(Ok((bytes, decoded))) => {
                        dialog.custom_favicon = Some((decoded, bytes));
                        dialog.error = None;
                    }
                    Some(Err(reason)) => {
                        dialog.error = Some(format!(
                            "Couldn't use that image ({reason}) — try a PNG, JPEG, GIF, WebP, ICO or SVG file."
                        ));
                    }
                }
            }
            Message::BookmarkDialogSave => {
                let Some(dialog) = self.bookmark_dialog.take() else {
                    return Task::none();
                };
                // The edited URL wins, trimmed; an emptied URL
                // falls back to the original. An empty name falls
                // back to the (new) URL — a bookmark always has
                // something to show.
                let url = if dialog.url.trim().is_empty() {
                    dialog.original_url.clone()
                } else {
                    dialog.url.trim().to_string()
                };
                let title = if dialog.name.trim().is_empty() {
                    url.clone()
                } else {
                    dialog.name.trim().to_string()
                };
                let tags = parse_tags(&dialog.tags);
                let keyword = {
                    let keyword = dialog.keyword.trim().to_string();
                    if keyword.is_empty() {
                        None
                    } else {
                        Some(keyword)
                    }
                };
                // When editing and the URL moved onto one that is
                // already bookmarked elsewhere, keep the original
                // URL and save the rest in place — never silently
                // merge two bookmarks.
                let target = if dialog.editing
                    && url != dialog.original_url
                    && self.bookmarks.is_bookmarked(&url)
                {
                    dialog.original_url.clone()
                } else {
                    url.clone()
                };
                // A fresh pick replaces the stored favicon; no
                // pick keeps the bookmark's existing one
                // (editing) or none (new). A failed favicon
                // write logs and keeps the bookmark — the title
                // still saves. When the URL moved, the stored
                // favicon file moves with it (the old file is
                // deleted); a fresh pick replaces it and the old
                // file is deleted too.
                let mut favicon = self
                    .bookmarks
                    .bookmark(&dialog.original_url)
                    .and_then(|b| b.favicon.clone());
                let mut decoded = None;
                if let Some((picked, bytes)) = dialog.custom_favicon {
                    match save_bookmark_favicon(&target, &bytes) {
                        Ok(name) => {
                            favicon = Some(name);
                            decoded = Some(picked);
                        }
                        Err(e) => {
                            eprintln!("[pluto] could not save custom favicon: {e}");
                        }
                    }
                } else if target != dialog.original_url && favicon.is_some() {
                    // The favicon file is named by the URL's
                    // hash, so a moved bookmark needs its file
                    // re-saved under the new URL.
                    if let Some(bytes) = load_bookmark_favicon(&dialog.original_url) {
                        if save_bookmark_favicon(&target, &bytes).is_ok() {
                            favicon = Some(bookmark_favicon_file_name(&target));
                        }
                        delete_bookmark_favicon(&dialog.original_url);
                    }
                }
                if let Some(decoded) = decoded {
                    self.bookmark_favicons.insert(target.clone(), decoded);
                } else if target != dialog.original_url {
                    // No fresh pick: carry the decoded icon to
                    // the new URL, or drop it when there was
                    // none to carry.
                    if let Some(decoded) = self
                        .bookmark_favicons
                        .remove(&dialog.original_url)
                    {
                        self.bookmark_favicons.insert(target.clone(), decoded);
                    }
                }
                if dialog.editing {
                    if target != dialog.original_url {
                        self.bookmarks.set_url(&dialog.original_url, &target);
                    }
                    self.bookmarks.set_title(&target, &title);
                    self.bookmarks.set_favicon(&target, favicon);
                    self.bookmarks.set_tags(&target, tags);
                    self.bookmarks.set_keyword(&target, keyword);
                } else {
                    self.bookmarks
                        .add_full(&target, &title, favicon, tags, keyword);
                }
                save_bookmarks(&self.bookmarks);
            }
            Message::BookmarkDialogCancel => {
                self.bookmark_dialog = None;
            }
            Message::BookmarkDialogRemove => {
                if let Some(dialog) = self.bookmark_dialog.take() {
                    self.remove_bookmark(&dialog.original_url);
                }
            }
            Message::ToggleMenu => {
                self.menu_open = !self.menu_open;
                if self.menu_open {
                    self.panel = None;
                    self.close_bar_popups();
                }
            }
            Message::ShowHistory => {
                self.menu_open = false;
                self.panel = Some(Panel::History);
                self.close_bar_popups();
            }
            Message::ClosePanel => {
                self.panel = None;
                self.close_bar_popups();
            }
            Message::SetTabPlacement(placement) => {
                self.tab_placement = placement;
                // Settings round (2026-11-28): placement is a
                // persisted preference — the settings dialog is
                // its only writer (the menu entries were removed
                // 2026-11-28, Joel), so it cannot disagree with
                // itself.
                self.save_prefs();
                self.menu_open = false;
                self.close_bar_popups();
                self.clear_flyout();
            }
            Message::TabHovered(index) => {
                if index < self.tabs.len() {
                    self.flyout_tab = Some(index);
                    self.flyout_target = 1.0;
                }
            }
            Message::TabUnhovered(index) => {
                if self.flyout_tab == Some(index) {
                    self.flyout_target = 0.0;
                }
            }
            Message::HoverTick => {
                if self.flyout_tab.is_some() {
                    if self.flyout_progress < self.flyout_target {
                        self.flyout_progress =
                            (self.flyout_progress + HOVER_TICK_STEP).min(self.flyout_target);
                    } else if self.flyout_progress > self.flyout_target {
                        self.flyout_progress =
                            (self.flyout_progress - HOVER_TICK_STEP).max(self.flyout_target);
                    }
                    // Fully collapsed: drop the flyout (and the
                    // ticker with it — see `hover_animating`).
                    if self.flyout_target < 0.5
                        && (self.flyout_progress - self.flyout_target).abs()
                            <= f32::EPSILON
                    {
                        self.flyout_tab = None;
                    }
                }
            }
            Message::RailScrolled(edge, offset) => match edge {
                TabEdge::Left => self.left_rail_scroll = offset,
                TabEdge::Right => self.right_rail_scroll = offset,
            },
            Message::NavigateTo(url) => {
                self.menu_open = false;
                self.panel = None;
                self.close_bar_popups();
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
                self.remove_bookmark(&url);
                self.close_bar_popups();
            }
            Message::ToggleBookmarksBar => {
                self.show_bookmarks_bar = !self.show_bookmarks_bar;
                self.save_prefs();
                self.menu_open = false;
                if !self.show_bookmarks_bar {
                    // A hidden bar shows no pencil and no chips —
                    // edit mode and the folder dropdown go with it.
                    self.bookmarks_edit_mode = false;
                    self.close_bar_popups();
                }
            }
            Message::ToggleBookmarksEditMode => {
                self.bookmarks_edit_mode = !self.bookmarks_edit_mode;
                self.close_bar_popups();
            }
            Message::BarContextMenuBookmark(url) => {
                // The chip's right-click area reports before the
                // bar's (see `bar_context_chip`): this menu
                // stands, and the bar's handler for the same
                // click becomes a no-op.
                self.bar_context_menu = Some(BarContextMenu::Bookmark { url });
                self.bar_context_chip = true;
            }
            Message::BarContextMenuFolder(id) => {
                self.bar_context_menu = Some(BarContextMenu::Folder { id });
                self.bar_context_chip = true;
            }
            Message::BarContextMenuBar => {
                if self.bar_context_chip {
                    // A chip right-clicked first — its menu is
                    // already open; this background event belongs
                    // to the same click, so it does nothing.
                    self.bar_context_chip = false;
                } else {
                    self.bar_context_menu = Some(BarContextMenu::Bar);
                }
            }
            Message::BarAddBookmark => {
                self.close_bar_popups();
                let url = self.active().active_url.clone();
                if !url.is_empty() {
                    let title = self.active().title.clone();
                    if self.bookmarks.add(&url, &title) {
                        save_bookmarks(&self.bookmarks);
                    }
                }
            }
            Message::BarAddFolder => {
                self.bar_context_menu = None;
                self.folder_dialog = Some(FolderDialog {
                    name: "New folder".to_string(),
                });
            }
            Message::BarAddBookmarkHere(folder_id) => {
                self.close_bar_popups();
                let url = self.active().active_url.clone();
                if !url.is_empty() {
                    let title = self.active().title.clone();
                    if self.bookmarks.add_to_folder(&folder_id, &url, &title) {
                        save_bookmarks(&self.bookmarks);
                    }
                }
            }
            Message::BarEditBookmarks => {
                self.bookmarks_edit_mode = true;
                self.close_bar_popups();
            }
            Message::BarHideBookmarksBar => {
                self.show_bookmarks_bar = false;
                self.save_prefs();
                self.bookmarks_edit_mode = false;
                self.close_bar_popups();
            }
            Message::BarEditBookmark(url) => {
                self.close_bar_popups();
                let title = self
                    .bookmarks
                    .bookmark(&url)
                    .map(|bookmark| bookmark.title.clone())
                    .unwrap_or_else(|| url.clone());
                let existing = self.bookmarks.bookmark(&url);
                self.bookmark_dialog =
                    Some(BookmarkDialog::new(&url, &title, existing));
                return cosmic::iced::widget::operation::focus::<
                    cosmic::Action<Message>,
                >("bookmark-dialog-name")
                .chain(cosmic::iced::widget::operation::select_all(
                    "bookmark-dialog-name",
                ));
            }
            Message::BarRemoveFolder(id) => {
                self.remove_folder(&id);
                self.close_bar_popups();
            }
            Message::ToggleFolderPopup(id) => {
                self.bar_context_menu = None;
                self.open_folder = if self.open_folder.as_deref() == Some(id.as_str()) {
                    None
                } else {
                    Some(id)
                };
            }
            Message::FolderDialogNameChanged(name) => {
                if let Some(dialog) = self.folder_dialog.as_mut() {
                    dialog.name = name;
                }
            }
            Message::FolderDialogSave => {
                if let Some(dialog) = self.folder_dialog.take() {
                    self.bookmarks.add_folder(&dialog.name);
                    save_bookmarks(&self.bookmarks);
                }
            }
            Message::FolderDialogCancel => {
                self.folder_dialog = None;
            }
            Message::ShowSettings => {
                self.menu_open = false;
                self.close_bar_popups();
                let homepage = self.homepage.clone();
                self.settings_dialog = Some(SettingsDialog::new(&homepage));
            }
            Message::ShowDevTools => {
                // Stage 17 (M9 stages 1–2): the console panel,
                // opened from the hamburger menu like the settings
                // dialog. The panel follows the active tab; the
                // tabs keep their console lines while it is
                // closed, so reopening shows what was captured.
                self.menu_open = false;
                self.close_bar_popups();
                self.devtools_panel = Some(DevToolsPanel::default());
            }
            Message::CloseDevTools => {
                self.devtools_panel = None;
            }
            Message::DevToolsInputChanged(draft) => {
                if let Some(panel) = self.devtools_panel.as_mut() {
                    panel.repl_input = draft;
                }
            }
            Message::DevToolsSubmit => {
                // Stage 17 (M9 stages 1–2): the REPL submit —
                // echo the source into the active tab's console
                // lines (level `""`: rendered as-is, `> source`),
                // clear the input, and send the snippet to that
                // tab's worker. An empty snippet is a no-op (the
                // echo would be a bare `>`). A tab with no worker
                // yet still gets the echo; the worker's
                // `DevToolsEval` answer (or its "no page loaded"
                // answer) arrives as `DevToolsEvalResult`.
                let Some(panel) = self.devtools_panel.as_mut() else {
                    return Task::none();
                };
                let source = std::mem::take(&mut panel.repl_input);
                if source.trim().is_empty() {
                    return Task::none();
                }
                self.active_mut().push_console_lines(vec![(
                    String::new(),
                    format!("> {source}"),
                )]);
                self.send_command(BrowserToRenderer::DevToolsEval { source });
            }
            Message::CloseSettings => {
                self.settings_dialog = None;
            }
            Message::SettingsThemeChanged(theme) => {
                if theme != self.theme_pref {
                    self.theme_pref = theme;
                    self.save_prefs();
                    return self.apply_theme_pref();
                }
            }
            Message::SettingsHomepageChanged(draft) => {
                if let Some(dialog) = self.settings_dialog.as_mut() {
                    dialog.homepage_draft = draft;
                    // Validate as it is typed: a valid value is
                    // saved at once (normalized — a bare host
                    // gains `https://`); an invalid one shows an
                    // error and saves nothing, so the saved
                    // homepage is never a broken URL.
                    match page::normalize_url(&dialog.homepage_draft) {
                        Ok(normalized) => {
                            dialog.homepage_error = None;
                            if normalized != self.homepage {
                                self.homepage = normalized;
                                self.save_prefs();
                            }
                        }
                        Err(reason) => {
                            dialog.homepage_error = Some(reason);
                        }
                    }
                }
            }
            Message::SettingsSearchProviderChanged(provider) => {
                if provider != self.search_provider {
                    self.search_provider = provider;
                    self.save_prefs();
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
                .style(small_text_input_style())
                .width(Length::Fill)
                .into(),
            widget::button::suggested("Go")
                .class(small_button_style(SmallButtonBase::Suggested, false))
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

        // The column (UI round, 2026-11-27): the nav bar and the
        // bookmarks bar span the full window width; the tab
        // columns start underneath the bookmarks bar, flanking
        // only the web view (see `vertical_content`); the status
        // line spans the full width at the bottom.
        // Bookmarks round (2026-11-28): a hidden bar takes no
        // space at all — bar and divider both go, and the web
        // view takes the space.
        let mut children: Vec<Element<'_, Self::Message>> = vec![
            nav_bar.into(),
            widget::divider::horizontal::default().into(),
        ];
        if self.show_bookmarks_bar {
            children.push(self.bookmarks_bar());
            children.push(widget::divider::horizontal::default().into());
        }
        if let Some(panel) = self.panel {
            children.push(self.list_panel(panel));
            children.push(widget::divider::horizontal::default().into());
        }
        children.push(self.vertical_content(web_view.into()));
        children.push(widget::divider::horizontal::default().into());
        // Stage 11 Rec 9: the load progress bar — a thin (~3px)
        // line at the bottom of the window, above the status line,
        // only while a document load is in flight (moved down from
        // under the bookmarks bar, 2026-11-28). Indeterminate (no
        // Content-Length) shows as a half bar, not a percentage —
        // the status text carries the byte count. The girth is set
        // explicitly (iced's default is 30px, which rendered as the
        // fat accent bar Joel saw on-box) and the corners are square
        // (radius 0, not the theme's rounded style) so it reads as a
        // straight line, not a pill.
        if let Some((received, total)) = active.progress {
            let fraction = total.map_or(0.5, |t| {
                if t == 0 {
                    1.0
                } else {
                    (received as f32 / t as f32).clamp(0.0, 1.0)
                }
            });
            children.push(
                cosmic::iced::widget::progress_bar(0.0..=1.0, fraction)
                    .girth(Length::Fixed(3.0))
                    .class(cosmic::theme::ProgressBar::custom(|theme| {
                        // The theme's own default progress-bar style
                        // (accent fill, divider track) with only the
                        // corner radius flattened to 0 — the theme
                        // uses radius_xl, which reads as a pill.
                        // (The iced `.style()` builder can't be used
                        // here: cosmic::Theme's progress-bar class
                        // is an enum with no From<StyleFn> impl.)
                        let class = <cosmic::Theme as cosmic::iced::widget::progress_bar::Catalog>::default(
                        );
                        let mut style = <cosmic::Theme as cosmic::iced::widget::progress_bar::Catalog>::style(
                            theme, &class,
                        );
                        style.border.radius =
                            cosmic::iced::core::border::Radius::from(0.0);
                        style
                    }))
                    .into(),
            );
        }
        children.push(status.into());
        // The shell column is the base layer; the menu and the
        // bookmark dialog ride on top of it as overlay layers
        // (follow-up UI work, 2026-11-27) — opening either moves
        // nothing around it, because a stack layer never affects
        // the base layer's layout: the column lays out identically
        // with the layers present or absent.
        let base = widget::column(children);
        let mut layers: Vec<Element<'_, Message>> = vec![base.into()];
        if self.menu_open {
            layers.push(self.menu_overlay());
        }
        if let Some(dialog) = &self.bookmark_dialog {
            layers.push(self.bookmark_dialog_overlay(dialog));
        }
        // Settings round (2026-11-28): the settings dialog
        // rides on top too, above the bookmark dialog.
        if let Some(dialog) = &self.settings_dialog {
            layers.push(self.settings_dialog_overlay(dialog));
        }
        // Stage 17 (M9 stages 1–2): the DevTools console panel
        // rides on top too, above the settings dialog.
        if let Some(panel) = &self.devtools_panel {
            layers.push(self.devtools_panel_overlay(panel));
        }
        // Bookmarks round (2026-11-28): the bar's context menu,
        // a folder's dropdown, and the new-folder dialog ride
        // on top too — above the dialog, so the menu is usable
        // while the dialog is open (it never is, but the layer
        // order is the safe one).
        if let Some(menu) = self.bar_context_menu_overlay() {
            layers.push(menu);
        }
        if let Some(popup) = self.folder_popup_overlay() {
            layers.push(popup);
        }
        if let Some(folder_dialog) = self.folder_dialog_overlay() {
            layers.push(folder_dialog);
        }
        cosmic::iced::widget::stack(layers).into()
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
            &["History", "New Tab", "New Window", "Settings", "DevTools"]
        );
        for forbidden in [
            "Bookmarks",
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

    // UI round (2026-11-27): tabs are vertical only, on the
    // left edge by default. The constant pin is the contract.
    #[test]
    fn tabs_default_on_left() {
        assert_eq!(DEFAULT_TAB_PLACEMENT, TabPlacement::Left);
        assert_eq!(placement_label(TabPlacement::Left), "Tabs on left");
        assert_eq!(placement_label(TabPlacement::Right), "Tabs on right");
        assert_eq!(
            placement_label(TabPlacement::Both),
            "Tabs on both sides"
        );
    }

    #[test]
    fn tab_strip_edges_cover_each_mode() {
        assert_eq!(tab_strip_edges(TabPlacement::Left), &[TabEdge::Left]);
        assert_eq!(
            tab_strip_edges(TabPlacement::Right),
            &[TabEdge::Right]
        );
        assert_eq!(
            tab_strip_edges(TabPlacement::Both),
            &[TabEdge::Left, TabEdge::Right]
        );
    }

    #[test]
    fn rail_capacity_matches_mockup_geometry() {
        // The mockup's 648px web view holds 16 tabs; the default
        // 768px viewport holds 20; a tiny view still holds one.
        assert_eq!(rail_capacity(648.0), 16);
        assert_eq!(rail_capacity(768.0), 20);
        assert_eq!(rail_capacity(100.0), 1);
        assert_eq!(rail_capacity(0.0), 1);
    }

    #[test]
    fn distribute_tabs_fills_left_then_spills_right() {
        assert_eq!(distribute_tabs(0, 16), (0, 0));
        assert_eq!(distribute_tabs(5, 16), (5, 0));
        assert_eq!(distribute_tabs(16, 16), (16, 0));
        assert_eq!(distribute_tabs(17, 16), (16, 1));
        assert_eq!(distribute_tabs(22, 16), (16, 6));
        assert_eq!(distribute_tabs(40, 16), (16, 24));
    }

    #[test]
    fn column_ranges_share_one_ordered_list() {
        // Both mode, 22 tabs, capacity 16: the left takes the
        // first 16, the right the rest — contiguous, one list,
        // nothing lost or duplicated.
        let left = column_range(TabPlacement::Both, TabEdge::Left, 22, 16);
        let right = column_range(TabPlacement::Both, TabEdge::Right, 22, 16);
        assert_eq!(left, (0, 16));
        assert_eq!(right, (16, 6));
        assert_eq!(left.0 + left.1, right.0);
        assert_eq!(left.1 + right.1, 22);
        // Single-column modes show the whole list on their edge.
        assert_eq!(
            column_range(TabPlacement::Left, TabEdge::Left, 22, 16),
            (0, 22)
        );
        assert_eq!(
            column_range(TabPlacement::Left, TabEdge::Right, 22, 16),
            (0, 0)
        );
        assert_eq!(
            column_range(TabPlacement::Right, TabEdge::Right, 22, 16),
            (0, 22)
        );
        assert_eq!(
            column_range(TabPlacement::Right, TabEdge::Left, 22, 16),
            (0, 0)
        );
    }

    #[test]
    fn closing_a_tab_reflows_the_right_column() {
        // 17 tabs, capacity 16: 16 left, 1 right. Close one left
        // tab and the right tab moves left to fill the gap — the
        // distribution is recomputed from the list, so no tab is
        // stranded on the right.
        assert_eq!(distribute_tabs(17, 16), (16, 1));
        assert_eq!(distribute_tabs(16, 16), (16, 0));
    }

    #[test]
    fn flyout_column_position_tracks_distribution() {
        assert_eq!(
            flyout_column_position(TabPlacement::Both, 5, 22, 16),
            Some((TabEdge::Left, 5))
        );
        assert_eq!(
            flyout_column_position(TabPlacement::Both, 16, 22, 16),
            Some((TabEdge::Right, 0))
        );
        assert_eq!(
            flyout_column_position(TabPlacement::Both, 21, 22, 16),
            Some((TabEdge::Right, 5))
        );
        assert_eq!(flyout_column_position(TabPlacement::Both, 22, 22, 16), None);
        assert_eq!(
            flyout_column_position(TabPlacement::Left, 3, 22, 16),
            Some((TabEdge::Left, 3))
        );
        assert_eq!(
            flyout_column_position(TabPlacement::Right, 3, 22, 16),
            Some((TabEdge::Right, 3))
        );
    }

    #[test]
    fn flyout_width_interpolates_chip_to_flyout() {
        assert_eq!(flyout_width(0.0), TAB_CHIP);
        assert_eq!(flyout_width(0.5), 116.0);
        assert_eq!(flyout_width(1.0), TAB_FLYOUT);
        // Out-of-range progress clamps, never overshoots.
        assert_eq!(flyout_width(-0.5), TAB_CHIP);
        assert_eq!(flyout_width(1.5), TAB_FLYOUT);
    }

    #[test]
    fn flyout_shows_text_only_when_half_open() {
        assert!(!flyout_shows_text(0.0));
        assert!(!flyout_shows_text(0.5));
        assert!(flyout_shows_text(0.51));
        assert!(flyout_shows_text(1.0));
    }

    #[test]
    fn hover_animation_completes_in_eighteen_ticks() {
        // 16 ms ticks, 1/18 per tick: the slide-out takes ~0.3 s,
        // the mockup's duration.
        assert_eq!(HOVER_TICK, std::time::Duration::from_millis(16));
        let mut progress = 0.0f32;
        for _ in 0..17 {
            progress = (progress + HOVER_TICK_STEP).min(1.0);
        }
        assert!(progress < 1.0);
        progress = (progress + HOVER_TICK_STEP).min(1.0);
        assert_eq!(progress, 1.0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn menu_panel_width_is_compact() {
        // Chrome/Firefox-sized, not half-screen. The `<=` bound
        // is the contract; the allow is deliberate, as above.
        assert_eq!(MENU_PANEL_WIDTH, 260.0);
        assert!(MENU_PANEL_WIDTH <= 320.0);
    }

    #[test]
    fn shell_radius_is_small_and_nonzero() {
        assert_eq!(SHELL_RADIUS, 4.0);
    }

    #[test]
    fn bookmarks_bar_entries_reads_the_store_in_order() {
        let mut store = BookmarkStore::new();
        assert!(bookmarks_bar_entries(&store).is_empty());
        assert!(store.add("https://a.example/", "A"));
        let folder = store.add_folder("News");
        assert!(store.add("https://b.example/", "B"));
        assert_eq!(
            bookmarks_bar_entries(&store),
            vec![
                BarEntry::Bookmark {
                    title: "A".to_string(),
                    url: "https://a.example/".to_string(),
                },
                BarEntry::Folder {
                    id: folder,
                    name: "News".to_string(),
                },
                BarEntry::Bookmark {
                    title: "B".to_string(),
                    url: "https://b.example/".to_string(),
                },
            ]
        );
    }

    #[test]
    fn bookmark_dialog_prefill_new_and_editing() {
        let url = "https://example.com/";
        let fresh = BookmarkDialog::new(url, "Example Domain", None);
        assert_eq!(fresh.url, url);
        assert_eq!(fresh.original_url, url);
        assert_eq!(fresh.name, "Example Domain");
        assert!(!fresh.editing);
        assert!(fresh.tags.is_empty());
        assert!(fresh.keyword.is_empty());
        assert!(fresh.custom_favicon.is_none());
        assert!(fresh.error.is_none());
        let stored = Bookmark {
            url: url.to_string(),
            title: "My Name".to_string(),
            favicon: None,
            tags: vec!["rust".to_string(), "news".to_string()],
            keyword: Some("ex".to_string()),
        };
        let editing = BookmarkDialog::new(url, "Example Domain", Some(&stored));
        assert_eq!(editing.name, "My Name");
        assert_eq!(editing.tags, "rust, news");
        assert_eq!(editing.keyword, "ex");
        assert!(editing.editing);
    }

    #[test]
    fn initial_letter_uses_host_then_title() {
        assert_eq!(initial_letter("https://example.com/page", ""), 'E');
        assert_eq!(initial_letter("http://sub.example.org/", ""), 'S');
        assert_eq!(initial_letter("", "Hello"), 'H');
        assert_eq!(initial_letter("", ""), '?');
        assert_eq!(initial_letter("about:blank", ""), 'A');
    }

    #[test]
    fn menu_overlay_top_offset_clears_the_nav_bar() {
        assert_eq!(menu_overlay_top_offset(), MENU_OVERLAY_TOP);
        // It clears the ~48px nav bar (+ divider).
        assert_eq!(MENU_OVERLAY_TOP, 56.0);
    }

    /// A temp HOME guard, mirroring the one in `bookmarks::tests`:
    /// profile reads/writes in these tests land in a throwaway
    /// dir, never in the real profile.
    struct DialogTestHome {
        old: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    fn with_test_home(tag: &str) -> (std::path::PathBuf, DialogTestHome) {
        let dir = std::env::temp_dir().join(format!("pluto-dialog-test-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let lock = crate::TEST_HOME_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old = std::env::var_os("HOME");
        // SAFETY: HOME is only read by the persistence helpers,
        // which run on this thread while the lock is held.
        unsafe {
            std::env::set_var("HOME", &dir);
        }
        (dir, DialogTestHome { old, _lock: lock })
    }

    fn restore_test_home(home: DialogTestHome) {
        unsafe {
            match home.old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    /// A fresh shell under a temp HOME (no real profile, no real
    /// bookmarks) for the dialog state-machine tests, on a known
    /// page so the dialog prefills from the tab title.
    fn test_shell(tag: &str) -> (std::path::PathBuf, DialogTestHome, PlutoBrowser) {
        let (dir, guard) = with_test_home(tag);
        let (mut app, _) = PlutoBrowser::init(Core::default(), false);
        app.active_mut().title = "Example Domain".to_string();
        (dir, guard, app)
    }

    #[test]
    fn star_opens_dialog_prefilled_for_new_page() {
        let (dir, home, mut app) = test_shell("star-new");
        let _ = app.update(Message::ToggleBookmark);
        let dialog = app
            .bookmark_dialog
            .as_ref()
            .expect("star opens the dialog");
        assert_eq!(dialog.url, app.active().active_url);
        assert_eq!(dialog.name, "Example Domain");
        assert!(!dialog.editing);
        assert!(!app.menu_open);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_save_adds_bookmark_and_cancel_leaves_store_untouched() {
        let (dir, home, mut app) = test_shell("save-cancel");
        let url = app.active().active_url.clone();
        // Save path: rename in the dialog, then save.
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogNameChanged(
            "My Example".to_string(),
        ));
        let _ = app.update(Message::BookmarkDialogSave);
        assert!(app.bookmark_dialog.is_none());
        assert!(app.bookmarks.is_bookmarked(&url));
        assert_eq!(app.bookmarks.bookmarks()[0].title, "My Example");
        // Cancel path: the page is bookmarked now, so the dialog
        // opens in editing mode prefilled with the stored title;
        // renaming then cancelling keeps the stored title.
        let _ = app.update(Message::ToggleBookmark);
        assert!(app.bookmark_dialog.as_ref().unwrap().editing);
        assert_eq!(
            app.bookmark_dialog.as_ref().unwrap().name,
            "My Example"
        );
        let _ = app.update(Message::BookmarkDialogNameChanged(
            "Renamed".to_string(),
        ));
        let _ = app.update(Message::BookmarkDialogCancel);
        assert!(app.bookmark_dialog.is_none());
        assert_eq!(app.bookmarks.bookmarks()[0].title, "My Example");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_remove_deletes_bookmark() {
        let (dir, home, mut app) = test_shell("remove");
        let url = app.active().active_url.clone();
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogSave);
        assert!(app.bookmarks.is_bookmarked(&url));
        let _ = app.update(Message::ToggleBookmark);
        assert!(app.bookmark_dialog.as_ref().unwrap().editing);
        let _ = app.update(Message::BookmarkDialogRemove);
        assert!(app.bookmark_dialog.is_none());
        assert!(!app.bookmarks.is_bookmarked(&url));
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_custom_favicon_pick_is_saved_and_cached() {
        let (dir, home, mut app) = test_shell("favicon-pick");
        let url = app.active().active_url.clone();
        let bytes = vec![1u8, 2, 3, 4];
        let decoded = pluto_paint::DecodedImage {
            width: 1,
            height: 1,
            rgba: vec![255, 0, 0, 255],
        };
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogFaviconPicked(Some(Ok((
            bytes.clone(),
            decoded.clone(),
        )))));
        assert!(app
            .bookmark_dialog
            .as_ref()
            .unwrap()
            .custom_favicon
            .is_some());
        let _ = app.update(Message::BookmarkDialogSave);
        let bookmarks = app.bookmarks.bookmarks();
        let bookmark = bookmarks.first().expect("one bookmark saved");
        assert!(bookmark.favicon.is_some());
        assert_eq!(app.bookmark_favicons.get(&url), Some(&decoded));
        // The bytes round-trip through the profile file.
        assert_eq!(load_bookmark_favicon(&url).as_deref(), Some(&bytes[..]));
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_favicon_pick_failure_shows_error_and_keeps_dialog() {
        let (dir, home, mut app) = test_shell("favicon-error");
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogFaviconPicked(Some(Err(
            "image decode failed: unsupported format".to_string(),
        ))));
        let dialog = app.bookmark_dialog.as_ref().expect("dialog stays open");
        assert!(dialog.error.is_some());
        assert!(dialog.custom_favicon.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn bookmarks_bar_builds_with_an_empty_store() {
        let (dir, home, app) = test_shell("bar-empty");
        // The bar is always present now (an Element, not an
        // Option): building it with no bookmarks must not panic.
        // The return type is the guarantee — there is no None
        // branch left to take.
        let _ = app.bookmarks_bar();
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn menu_overlay_leaves_shell_geometry_untouched() {
        let (dir, home, mut app) = test_shell("menu-geometry");
        let viewport_before = app.viewport;
        let tabs_before = app.tabs.len();
        // The whole shell renders with the menu closed (base
        // layer only)...
        let _ = app.view();
        let _ = app.update(Message::ToggleMenu);
        assert!(app.menu_open);
        // ...and with it open (base layer + overlay layer): the
        // overlay never feeds back into the base layout, so the
        // geometry inputs are unchanged.
        assert_eq!(app.viewport, viewport_before);
        assert_eq!(app.tabs.len(), tabs_before);
        let _ = app.view();
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn parse_tags_splits_trims_and_dedupes() {
        assert_eq!(
            parse_tags("rust, news , ,rust,docs "),
            vec!["rust".to_string(), "news".to_string(), "docs".to_string()]
        );
        assert!(parse_tags("").is_empty());
        assert!(parse_tags(" , , ").is_empty());
        assert_eq!(parse_tags("solo"), vec!["solo".to_string()]);
    }

    #[test]
    fn toggle_tag_in_text_adds_and_removes() {
        assert_eq!(toggle_tag_in_text("", "rust"), "rust");
        assert_eq!(toggle_tag_in_text("rust, news", "docs"), "rust, news, docs");
        assert_eq!(toggle_tag_in_text("rust, news", "rust"), "news");
        assert_eq!(toggle_tag_in_text("rust", "rust"), "");
    }

    #[test]
    fn prefs_default_visible_and_round_trip() {
        let (dir, home, _app) = test_shell("prefs-default");
        // Nothing was saved yet in this temp HOME: the default is
        // visible.
        assert_eq!(load_shell_prefs(), ShellPrefs::default());
        assert!(ShellPrefs::default().show_bookmarks_bar);
        save_shell_prefs(&ShellPrefs {
            show_bookmarks_bar: false,
            ..ShellPrefs::default()
        });
        assert!(!load_shell_prefs().show_bookmarks_bar);
        save_shell_prefs(&ShellPrefs {
            show_bookmarks_bar: true,
            ..ShellPrefs::default()
        });
        assert!(load_shell_prefs().show_bookmarks_bar);
        // A corrupt file falls back to the default, not an error.
        std::fs::write(
            shell_prefs_file().expect("temp HOME has a profile dir"),
            "{not json",
        )
        .unwrap();
        assert!(load_shell_prefs().show_bookmarks_bar);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn devtools_opens_from_menu_and_closes() {
        // Stage 17 (M9 stages 1–2): the DevTools panel opens
        // from the hamburger menu (like the settings dialog)
        // and closes from its Close button — the panel follows
        // the active tab, and the tabs keep their console lines
        // while it is closed.
        let (dir, home, mut app) = test_shell("devtools-open");
        assert!(app.devtools_panel.is_none());
        let _ = app.update(Message::ToggleMenu);
        assert!(app.menu_open);
        let _ = app.update(Message::ShowDevTools);
        assert!(!app.menu_open);
        assert!(app.devtools_panel.is_some());
        let _ = app.update(Message::CloseDevTools);
        assert!(app.devtools_panel.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn devtools_console_lines_route_to_the_tab_and_cap() {
        // Stage 17 (M9 stages 1–2): a `ConsoleLines` drain
        // reaches the shell-side buffer of the tab it belongs
        // to, and the buffer caps at 1,000 lines with the oldest
        // evicted — a page logging in a loop cannot grow it
        // without bound.
        let (dir, home, mut app) = test_shell("devtools-cap");
        let tab_id = app.active().id;
        let _ = app.update(Message::RendererEvent(
            tab_id,
            RendererToBrowser::ConsoleLines {
                lines: vec![
                    ("log".to_string(), "first".to_string()),
                    ("warn".to_string(), "second".to_string()),
                ],
            },
        ));
        assert_eq!(
            app.active().console_lines,
            vec![
                ("log".to_string(), "first".to_string()),
                ("warn".to_string(), "second".to_string()),
            ]
        );
        let flood: Vec<(String, String)> = (0..CONSOLE_LINES_CAP + 5)
            .map(|i| ("log".to_string(), format!("line {i}")))
            .collect();
        let _ = app.update(Message::RendererEvent(
            tab_id,
            RendererToBrowser::ConsoleLines { lines: flood },
        ));
        assert_eq!(app.active().console_lines.len(), CONSOLE_LINES_CAP);
        assert_eq!(
            app.active().console_lines.first().unwrap().1,
            "line 5".to_string(),
            "the oldest lines are evicted first"
        );
        assert_eq!(
            app.active().console_lines.last().unwrap().1,
            format!("line {}", CONSOLE_LINES_CAP + 4)
        );
        // An event for a closed tab is ignored, not a panic.
        let _ = app.update(Message::RendererEvent(
            tab_id + 1000,
            RendererToBrowser::ConsoleLines {
                lines: vec![("log".to_string(), "ghost".to_string())],
            },
        ));
        assert_eq!(app.active().console_lines.len(), CONSOLE_LINES_CAP);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn devtools_repl_submit_echoes_and_clears() {
        // Stage 17 (M9 stages 1–2): submitting the REPL echoes
        // `> source` into the active tab's console lines and
        // clears the input (the worker is not running in the
        // test shell, so `send_command` is a no-op — the echo
        // and the `DevToolsEvalResult` arm are the shell-side
        // halves, tested separately). An empty submit is a
        // no-op, and the result line lands under the echo.
        let (dir, home, mut app) = test_shell("devtools-repl");
        let tab_id = app.active().id;
        let _ = app.update(Message::ShowDevTools);
        let _ = app.update(Message::DevToolsInputChanged("1 + 2".to_string()));
        let _ = app.update(Message::DevToolsSubmit);
        assert_eq!(
            app.devtools_panel
                .as_ref()
                .expect("panel stays open")
                .repl_input,
            ""
        );
        assert_eq!(
            app.active().console_lines,
            vec![(String::new(), "> 1 + 2".to_string())]
        );
        let _ = app.update(Message::RendererEvent(
            tab_id,
            RendererToBrowser::DevToolsEvalResult {
                result: "3".to_string(),
            },
        ));
        assert_eq!(
            app.active().console_lines,
            vec![
                (String::new(), "> 1 + 2".to_string()),
                (String::new(), "3".to_string()),
            ]
        );
        let _ = app.update(Message::DevToolsInputChanged("   ".to_string()));
        let _ = app.update(Message::DevToolsSubmit);
        assert_eq!(app.active().console_lines.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn settings_opens_from_menu_and_closes() {
        let (dir, home, mut app) = test_shell("settings-open");
        assert!(app.settings_dialog.is_none());
        let _ = app.update(Message::ToggleMenu);
        assert!(app.menu_open);
        let _ = app.update(Message::ShowSettings);
        assert!(!app.menu_open);
        let dialog = app.settings_dialog.as_ref().expect("settings opens");
        assert_eq!(dialog.homepage_draft, app.homepage);
        assert!(dialog.homepage_error.is_none());
        let _ = app.update(Message::CloseSettings);
        assert!(app.settings_dialog.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn settings_theme_applies_and_persists() {
        let (dir, home, mut app) = test_shell("settings-theme");
        assert_eq!(app.theme_pref, ThemePref::System);
        let _ = app.update(Message::SettingsThemeChanged(ThemePref::Dark));
        assert_eq!(app.theme_pref, ThemePref::Dark);
        assert!(app.dark_theme);
        assert_eq!(load_shell_prefs().theme, ThemePref::Dark);
        // A new shell in the same temp HOME reads it back.
        let (app2, _) = PlutoBrowser::init(Core::default(), false);
        assert_eq!(app2.theme_pref, ThemePref::Dark);
        assert!(app2.dark_theme);
        // And back to Light, then to System (which follows the OS).
        let _ = app.update(Message::SettingsThemeChanged(ThemePref::Light));
        assert!(!app.dark_theme);
        assert_eq!(load_shell_prefs().theme, ThemePref::Light);
        let _ = app.update(Message::SettingsThemeChanged(ThemePref::System));
        assert_eq!(app.theme_pref, ThemePref::System);
        assert_eq!(app.dark_theme, cosmic::theme::is_dark());
        assert_eq!(load_shell_prefs().theme, ThemePref::System);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn settings_homepage_applies_and_persists() {
        let (dir, home, mut app) = test_shell("settings-homepage");
        let _ = app.update(Message::ShowSettings);
        // A bare host gains https:// and is saved at once.
        let _ = app.update(Message::SettingsHomepageChanged(
            "example.org".to_string(),
        ));
        assert_eq!(app.homepage, "https://example.org");
        assert_eq!(load_shell_prefs().homepage, "https://example.org");
        // A new shell starts on it.
        let (app2, _) = PlutoBrowser::init(Core::default(), false);
        assert_eq!(app2.homepage, "https://example.org");
        assert_eq!(app2.active().active_url, "https://example.org");
        // A rejected scheme shows an error and saves nothing.
        let _ = app.update(Message::SettingsHomepageChanged(
            "javascript:alert(1)".to_string(),
        ));
        let dialog = app.settings_dialog.as_ref().expect("dialog stays open");
        assert!(dialog.homepage_error.is_some());
        assert_eq!(app.homepage, "https://example.org");
        assert_eq!(load_shell_prefs().homepage, "https://example.org");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn settings_tab_placement_persists() {
        let (dir, home, mut app) = test_shell("settings-placement");
        assert_eq!(app.tab_placement, TabPlacement::Left);
        let _ = app.update(Message::SetTabPlacement(TabPlacement::Right));
        assert_eq!(app.tab_placement, TabPlacement::Right);
        assert_eq!(load_shell_prefs().tab_placement, TabPlacement::Right);
        let (app2, _) = PlutoBrowser::init(Core::default(), false);
        assert_eq!(app2.tab_placement, TabPlacement::Right);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn prefs_v1_file_loads_with_defaults_for_new_fields() {
        let (dir, home, _app) = test_shell("prefs-v1");
        // Nothing has been saved yet in this temp HOME, so the
        // profile dir does not exist — create it before writing
        // the v1 file by hand.
        let path = shell_prefs_file().expect("temp HOME has a profile dir");
        std::fs::create_dir_all(path.parent().expect("prefs file has a dir"))
            .unwrap();
        std::fs::write(
            path,
            "{\"version\":1,\"show_bookmarks_bar\":false}",
        )
        .unwrap();
        let prefs = load_shell_prefs();
        assert!(!prefs.show_bookmarks_bar);
        assert_eq!(prefs.theme, ThemePref::System);
        assert_eq!(prefs.homepage, page::DEFAULT_URL);
        assert_eq!(
            prefs.search_provider,
            page::SearchProvider::DuckDuckGo
        );
        assert_eq!(prefs.tab_placement, TabPlacement::Left);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn settings_pref_keys_round_trip() {
        for theme in ThemePref::all() {
            assert_eq!(ThemePref::from_key(theme.key()), Some(theme));
        }
        assert_eq!(ThemePref::from_key("nope"), None);
        for placement in [TabPlacement::Left, TabPlacement::Right, TabPlacement::Both] {
            assert_eq!(TabPlacement::from_key(placement.key()), Some(placement));
        }
        assert_eq!(TabPlacement::from_key("nope"), None);
    }

    #[test]
    fn menu_toggle_hides_bar_and_persists() {
        let (dir, home, mut app) = test_shell("menu-bar-toggle");
        assert!(app.show_bookmarks_bar);
        let _ = app.update(Message::ToggleBookmarksBar);
        assert!(!app.show_bookmarks_bar);
        assert!(!app.menu_open);
        assert!(!load_shell_prefs().show_bookmarks_bar);
        // The new shell in the same temp HOME reads it back.
        let (mut app2, _) = PlutoBrowser::init(Core::default(), false);
        assert!(!app2.show_bookmarks_bar);
        let _ = app2.update(Message::ToggleBookmarksBar);
        assert!(app2.show_bookmarks_bar);
        assert!(load_shell_prefs().show_bookmarks_bar);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn keyword_in_address_bar_navigates_to_bookmark() {
        let (dir, home, mut app) = test_shell("keyword-nav");
        let url = "https://github.com/";
        assert!(app.bookmarks.add_full(
            url,
            "GitHub",
            None,
            Vec::new(),
            Some("gh".to_string()),
        ));
        app.active_mut().url_input = "GH".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(app.active().active_url, url);
        // A non-keyword still goes through normal URL handling.
        app.active_mut().url_input = "example.com".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(app.active().active_url, "https://example.com");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn address_bar_query_goes_to_search_provider() {
        let (dir, home, mut app) = test_shell("query-search");
        // The homepage is changed first so the empty-bar case
        // below proves the bar really went to it — through the
        // dialog, like the real flow (the homepage draft lives
        // in the dialog).
        let _ = app.update(Message::ShowSettings);
        let _ = app.update(Message::SettingsHomepageChanged(
            "example.org".to_string(),
        ));
        assert_eq!(app.homepage, "https://example.org");
        // The default provider is DuckDuckGo.
        app.active_mut().url_input = "rust browser".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(
            app.active().active_url,
            "https://duckduckgo.com/?q=rust+browser"
        );
        // Switching provider switches the search, and persists.
        let _ = app.update(Message::SettingsSearchProviderChanged(
            page::SearchProvider::Google,
        ));
        assert_eq!(
            load_shell_prefs().search_provider,
            page::SearchProvider::Google
        );
        app.active_mut().url_input = "rust browser".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(
            app.active().active_url,
            "https://www.google.com/search?q=rust+browser"
        );
        // A bare host still navigates — it is not a query.
        app.active_mut().url_input = "example.com".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(app.active().active_url, "https://example.com");
        // An empty address bar goes to the homepage.
        app.active_mut().url_input = "   ".to_string();
        let _ = app.update(Message::Navigate);
        assert_eq!(app.active().active_url, "https://example.org");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn context_menu_chip_beats_background() {
        let (dir, home, mut app) = test_shell("context-order");
        let url = app.active().active_url.clone();
        assert!(app.bookmarks.add(&url, "Example"));
        // A chip right-click opens the bookmark menu and sets
        // the flag; the background handler for the same click
        // consumes the flag and leaves the chip menu standing.
        let _ = app.update(Message::BarContextMenuBookmark(url.clone()));
        assert_eq!(
            app.bar_context_menu,
            Some(BarContextMenu::Bookmark { url: url.clone() })
        );
        assert!(app.bar_context_chip);
        let _ = app.update(Message::BarContextMenuBar);
        assert_eq!(
            app.bar_context_menu,
            Some(BarContextMenu::Bookmark { url })
        );
        assert!(!app.bar_context_chip);
        // A plain background right-click opens the generic menu.
        let _ = app.update(Message::BarContextMenuBar);
        assert_eq!(app.bar_context_menu, Some(BarContextMenu::Bar));
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn context_menu_add_bookmark_saves_current_page() {
        let (dir, home, mut app) = test_shell("context-add");
        let url = app.active().active_url.clone();
        let _ = app.update(Message::BarContextMenuBar);
        assert_eq!(app.bar_context_menu, Some(BarContextMenu::Bar));
        let _ = app.update(Message::BarAddBookmark);
        assert!(app.bar_context_menu.is_none());
        assert!(app.bookmarks.is_bookmarked(&url));
        assert_eq!(app.bookmarks.bookmarks()[0].title, "Example Domain");
        // Adding the same page again is a no-op (dedup).
        let _ = app.update(Message::BarAddBookmark);
        assert_eq!(app.bookmarks.bookmarks().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn edit_mode_x_removes_bookmark_and_folder() {
        let (dir, home, mut app) = test_shell("edit-mode-x");
        let url = app.active().active_url.clone();
        assert!(app.bookmarks.add(&url, "Example"));
        let folder = app.bookmarks.add_folder("News");
        assert!(app
            .bookmarks
            .add_to_folder(&folder, "https://news.example/", "News"));
        // The pencil enters edit mode; the × on a bookmark
        // removes it, and the × on a folder removes the folder
        // and its contents.
        let _ = app.update(Message::ToggleBookmarksEditMode);
        assert!(app.bookmarks_edit_mode);
        let _ = app.update(Message::RemoveBookmark(url.clone()));
        assert!(!app.bookmarks.is_bookmarked(&url));
        let _ = app.update(Message::BarRemoveFolder(folder.clone()));
        assert!(app.bookmarks.folder(&folder).is_none());
        assert!(!app.bookmarks.is_bookmarked("https://news.example/"));
        // Clicking the pencil again exits edit mode.
        let _ = app.update(Message::ToggleBookmarksEditMode);
        assert!(!app.bookmarks_edit_mode);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn folder_dialog_creates_folder_and_cancel_leaves_none() {
        let (dir, home, mut app) = test_shell("folder-dialog");
        let _ = app.update(Message::BarAddFolder);
        assert!(app.folder_dialog.is_some());
        assert_eq!(app.folder_dialog.as_ref().unwrap().name, "New folder");
        let _ = app.update(Message::FolderDialogCancel);
        assert!(app.folder_dialog.is_none());
        assert!(app.bookmarks.entries().is_empty());
        let _ = app.update(Message::BarAddFolder);
        let _ = app.update(Message::FolderDialogNameChanged("Read".to_string()));
        let _ = app.update(Message::FolderDialogSave);
        assert!(app.folder_dialog.is_none());
        assert_eq!(app.bookmarks.entries().len(), 1);
        assert!(matches!(
            &app.bookmarks.entries()[0],
            BookmarkEntry::Folder(folder) if folder.name == "Read"
        ));
        // The folder survives a save/load round-trip.
        let reloaded = BookmarkStore::from_json(&app.bookmarks.to_json());
        assert_eq!(reloaded.entries().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn folder_popup_open_close_and_empty_line() {
        let (dir, home, mut app) = test_shell("folder-popup");
        let folder = app.bookmarks.add_folder("Empty");
        // Clicking the folder chip opens its dropdown; clicking
        // again closes it.
        let _ = app.update(Message::ToggleFolderPopup(folder.clone()));
        assert_eq!(app.open_folder.as_deref(), Some(folder.as_str()));
        assert!(app.folder_popup_overlay().is_some());
        let _ = app.update(Message::ToggleFolderPopup(folder.clone()));
        assert!(app.open_folder.is_none());
        // Navigation closes the dropdown too.
        let _ = app.update(Message::ToggleFolderPopup(folder.clone()));
        let _ = app.update(Message::NavigateTo("https://example.com/".to_string()));
        assert!(app.open_folder.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_save_persists_tags_and_keyword() {
        let (dir, home, mut app) = test_shell("dialog-tags");
        let url = app.active().active_url.clone();
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogTagsChanged(
            "rust, news".to_string(),
        ));
        let _ = app.update(Message::BookmarkDialogKeywordChanged("ex".to_string()));
        let _ = app.update(Message::BookmarkDialogSave);
        let bookmark = app.bookmarks.bookmark(&url).expect("saved");
        assert_eq!(
            bookmark.tags,
            vec!["rust".to_string(), "news".to_string()]
        );
        assert_eq!(bookmark.keyword.as_deref(), Some("ex"));
        // Reopening the dialog prefills both, and the keyword
        // now navigates from the address bar.
        let _ = app.update(Message::ToggleBookmark);
        let dialog = app.bookmark_dialog.as_ref().unwrap();
        assert_eq!(dialog.tags, "rust, news");
        assert_eq!(dialog.keyword, "ex");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_save_with_moved_url_keeps_original_when_taken() {
        let (dir, home, mut app) = test_shell("dialog-url-move");
        let url_a = app.active().active_url.clone();
        let url_b = "https://other.example/";
        assert!(app.bookmarks.add(url_b, "Other"));
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogSave);
        assert!(app.bookmarks.is_bookmarked(&url_a));
        // Edit the first bookmark, move its URL onto the
        // second's: the original URL is kept, the rest saves.
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogUrlChanged(url_b.to_string()));
        let _ = app.update(Message::BookmarkDialogNameChanged("Renamed".to_string()));
        let _ = app.update(Message::BookmarkDialogSave);
        assert!(app.bookmarks.is_bookmarked(&url_a));
        assert!(app.bookmarks.is_bookmarked(url_b));
        assert_eq!(app.bookmarks.bookmark(&url_a).unwrap().title, "Renamed");
        assert_eq!(app.bookmarks.bookmark(url_b).unwrap().title, "Other");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn bar_context_menu_overlay_builds_for_each_target() {
        let (dir, home, mut app) = test_shell("context-overlay");
        let url = app.active().active_url.clone();
        assert!(app.bookmarks.add(&url, "Example"));
        let folder = app.bookmarks.add_folder("News");
        let _ = app.update(Message::BarContextMenuBar);
        assert!(app.bar_context_menu_overlay().is_some());
        let _ = app.update(Message::BarContextMenuBookmark(url));
        assert!(app.bar_context_menu_overlay().is_some());
        let _ = app.update(Message::BarContextMenuFolder(folder));
        assert!(app.bar_context_menu_overlay().is_some());
        // Hiding the bar closes the menu.
        let _ = app.update(Message::BarHideBookmarksBar);
        assert!(!app.show_bookmarks_bar);
        assert!(app.bar_context_menu.is_none());
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }

    #[test]
    fn dialog_tag_chevron_toggles_tags_into_field() {
        let (dir, home, mut app) = test_shell("tag-chevron");
        // A stored tag exists from another bookmark.
        assert!(app.bookmarks.add_full(
            "https://rust.example/",
            "Rust",
            None,
            vec!["rust".to_string()],
            None,
        ));
        let _ = app.update(Message::ToggleBookmark);
        let _ = app.update(Message::BookmarkDialogToggleTags);
        assert!(app.bookmark_dialog.as_ref().unwrap().show_tags);
        let _ = app.update(Message::BookmarkDialogTagToggle("rust".to_string()));
        assert_eq!(app.bookmark_dialog.as_ref().unwrap().tags, "rust");
        let _ = app.update(Message::BookmarkDialogTagToggle("rust".to_string()));
        assert_eq!(app.bookmark_dialog.as_ref().unwrap().tags, "");
        let _ = std::fs::remove_dir_all(&dir);
        restore_test_home(home);
    }
}
