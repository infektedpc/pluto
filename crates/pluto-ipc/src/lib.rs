//! Browser <-> renderer message protocol (Stage 4 #1).
//!
//! The protocol is the seam that makes "threads now, processes
//! later" real: the same message enums travel over channels
//! in-process today and over `ipc-channel`/Unix sockets at M5, when
//! the per-tab worker thread is promoted to a process. Nothing in
//! the protocol knows which transport carries it.
//!
//! Messages are versioned serde enums. [`encode_message`] prefixes
//! the payload with [`PROTOCOL_VERSION`] (a little-endian `u32`);
//! [`decode_message`] rejects bytes from another version instead of
//! misreading them. Bump `PROTOCOL_VERSION` when a message changes
//! shape, and keep the old shape decodable only if a mixed-version
//! fleet ever needs it (Pluto ships one binary, so a bump is a
//! clean break — the check exists to fail loudly, not to migrate).
//!
//! The display list itself travels inside
//! [`RendererToBrowser::SubmitDisplayList`] as postcard bytes plus
//! the [`ResourceTable`] its image/blob ids resolve against — the
//! same bytes the shell validates ([`DisplayList::validate`]) before
//! installing. The list is *not* re-serialised per hop: the worker
//! serialises once, the shell deserialises once.
//!
//! ## M3 (Stage 11) resource decision — decoded pixels cross IPC out-of-band
//!
//! IPC frames are capped at 64 MiB (`transport::MAX_FRAME_BYTES`)
//! and one decoded 4K image is ~33 MB of RGBA, so a display list
//! carrying decoded pixels can exceed the cap and fail the send at
//! submit time, far from the decode that caused it. M3 therefore
//! chose option (b) from the Stage 11 research report: decode
//! worker-side on the rayon pool, and deliver each decoded resource
//! exactly once, out-of-band, in its own
//! [`RendererToBrowser::ResourceReady`] frame — the frame carries
//! only that one resource, so any single resource under the cap
//! always fits, and two large images on one page are two frames,
//! not one over-cap frame. The display list itself carries only
//! ids; its `resources` field is sent empty by the worker, and the
//! shell validates the list against the per-tab resource table it
//! accumulated from `ResourceReady` messages. The worker refuses
//! to send a resource whose encoded frame would exceed the cap
//! (oversize → broken-image placeholder, never a failed submit).
//! The rejected alternative (a) — encoded bytes across IPC, decode
//! shell-side — would have made the shell own the `ImageCache` and
//! delayed every first paint on the shell's decode; (b) keeps the
//! existing cache ownership and the existing submit path. Shared
//! memory (the M5 design) is not needed for M3: one resource per
//! frame is enough.

use std::sync::atomic::{AtomicU64, Ordering};

use pluto_paint::{CompositedUpdate, ResourceTable};
use serde::{Deserialize, Serialize};

/// The transport layer: in-process channels and cross-process Unix
/// socketpairs carrying the versioned frames below (Stage 9 Rec 9).
pub mod transport;

/// The protocol version every encoded message carries. Bump on any
/// breaking change to the enums below.
///
/// v15 (Stage 18, M9 stage 3): the DevTools inspector channel.
/// [`BrowserToRenderer::DevToolsDomChildren`] /
/// `DevToolsInspectNode` / `DevToolsPickNode` /
/// `DevToolsHighlight` and
/// [`RendererToBrowser::DevToolsChildren`] / `DevToolsNode` /
/// `DevToolsPicked`, all appended at the end so existing
/// variants keep their postcard discriminants. The bump is the
/// shape change; v14 bytes are rejected by [`decode_message`].
///
/// v14 (Stage 17, M9 stages 1–2): the DevTools console channel.
/// [`BrowserToRenderer::DevToolsEval`] (evaluate a snippet in the
/// tab's realm) and [`RendererToBrowser::DevToolsEvalResult`] /
/// `ConsoleLines` (the formatted result, and batched console +
/// error lines drained from the tab), all appended at the end so
/// existing variants keep their postcard discriminants. The bump is
/// the shape change; v13 bytes are rejected by [`decode_message`].
///
/// v13 (Stage 16 Phase 3, Rec 8/9/11 — finisher): the history
/// and WebSocket message shapes change together.
/// [`BrowserToRenderer::CommitNavigation`] gains `state_json`,
/// `traversal`, and `scroll` (a Back/Forward traversal restores
/// the entry's state and scroll and fires `popstate` after the
/// commit; a reload restores both too; a plain navigation carries
/// `None` / `false` / `(0.0, 0.0)`), and every
/// [`WebSocketEvent`] variant gains `socket_id` (the broker's pump
/// tags its events; the envelope carries it too). The bump is the
/// shape change; v12 bytes are rejected by [`decode_message`].
///
/// v12 (Stage 16 Phase 3, Rec 8/9/11/12) adds the history and
/// WebSocket messages: [`RendererToBrowser::HistoryPush`] /
/// `HistoryReplace` / `HistoryGo` / `HistorySync` and
/// `WebSocketOpen` / `WebSocketSend` / `WebSocketClose` /
/// [`BrowserToRenderer::WebSocketEvent`], `DidCommit` gains
/// `replace`, and [`Destination::Stream`] arrives. The bump is
/// the shape change; v11 bytes are rejected by
/// [`decode_message`].
///
/// v11 (Stage 16 Phase 1, Rec 2) adds `Destination::Audio`,
/// `Destination::Video`, and `Destination::WebSocket` to the enum
/// (media was `Other` before, so `media-src` never fired and media
/// inherited the 10 MiB `Other` cap; WebSocket had no destination
/// at all), plus [`RendererToBrowser::VideoFrame`] (Rec 1). The
/// bump is the shape change; v10 bytes are rejected by
/// [`decode_message`].
///
/// v10 (Stage 15 Phase 2, M7 Rec 4) appends one display-list
/// variant at the end so existing variants keep their postcard
/// discriminants: [`DisplayItem::VectorPath`] (tessellated vector
/// triangles for inline SVG shapes and CSS linear-gradient
/// backgrounds). The bump is the shape change; v9 bytes are
/// rejected by [`decode_message`].
///
/// v9 (Stage 14 Phase 3, M6 Rec 12) adds one browser message,
/// appended at the end so existing variants keep their postcard
/// discriminants: [`BrowserToRenderer::SetTheme`] (the OS / desktop
/// theme changed — the worker restyles, never refetches). The bump
/// is the shape change; v8 bytes are rejected by
/// [`decode_message`].
///
/// v8 (Stage 14, M6) adds one renderer message, appended at the
/// end so existing variants keep their postcard discriminants:
/// [`RendererToBrowser::CompositingUpdate`] (the changed
/// transform/opacity state of a few elements, out-of-band — a
/// transform-only frame never rebuilds or resubmits the display
/// list). The bump is the shape change; v7 bytes are rejected by
/// [`decode_message`]. v5 (Stage 11, M3) adds three renderer messages, appended at
/// the end so existing variants keep their postcard discriminants:
/// [`RendererToBrowser::ResourceReady`] (one decoded resource per
/// frame, out-of-band — see the module docs), `LoadProgress`
/// (document-load progress), and `FaviconChanged` (the tab's
/// favicon resource id). The bump is the shape change; v4 bytes
/// are rejected by [`decode_message`]. v4 (Stage 9 Rec 9) moves
/// the transport cross-process: the per-tab worker thread becomes
/// a `--renderer` child process over a Unix socketpair
/// (`transport::SocketTransport`), with a version handshake before
/// the first frame. The message enums themselves are unchanged
/// from v3 — the bump is the transport break, and v3 bytes are
/// rejected by [`decode_message`]. v3 (Stage 8 Rec 9) reworked
/// `InputEvent` (pointer/wheel/keyboard with position, modifiers,
/// timestamp, pointer id, delta mode and gesture phase), adds
/// `SetFocused` and `ScrollViewport`. v2 bytes are rejected by
/// [`decode_message`] — Pluto ships one binary, so a bump is a
/// clean break, not a migration.
pub const PROTOCOL_VERSION: u32 = 15;

/// What went wrong decoding a versioned message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    #[error("message is empty")]
    Empty,
    #[error("unsupported protocol version {found} (expected {PROTOCOL_VERSION})")]
    UnsupportedVersion { found: u32 },
    #[error("message bytes do not decode: {0}")]
    Decode(String),
}

/// Encode a message as `[version:u32 LE][postcard bytes]`.
pub fn encode_message<T: Serialize>(message: &T) -> Vec<u8> {
    let mut out = PROTOCOL_VERSION.to_le_bytes().to_vec();
    out.extend_from_slice(&postcard::to_allocvec(message).expect("message serialises"));
    out
}

/// Decode bytes produced by [`encode_message`], rejecting another
/// version before touching the payload.
pub fn decode_message<T>(bytes: &[u8]) -> Result<T, ProtocolError>
where
    T: for<'de> Deserialize<'de>,
{
    if bytes.len() < 4 {
        return Err(ProtocolError::Empty);
    }
    let version = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"));
    if version != PROTOCOL_VERSION {
        return Err(ProtocolError::UnsupportedVersion { found: version });
    }
    postcard::from_bytes(&bytes[4..]).map_err(|e| ProtocolError::Decode(e.to_string()))
}

/// Messages from the browser (shell) to the renderer worker.
/// The browser owns navigation, input, and the viewport; the
/// renderer owns the document and never touches a socket, the disk,
/// or the GPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BrowserToRenderer {
    /// Start loading this URL: fetch, parse, lay out at the current
    /// viewport, and report back with `DidCommit` + `SubmitDisplayList`.
    ///
    /// Stage 16 Phase 3 (Rec 8/9): `state_json` is the session-history
    /// state to restore for this entry (the JSON the page stored with
    /// `pushState` / `replaceState` — `None` for a fresh navigation,
    /// whose `history.state` is `null`). `traversal` is true only for
    /// Back/Forward: after the commit the worker fires `popstate`
    /// with the restored state; a reload of a stateful entry carries
    /// its state with `traversal: false` and fires no `popstate`.
    /// `scroll` is the viewport scroll to restore for the entry
    /// (Stage 16 Phase 3, Rec 8 — finisher): a traversal or reload
    /// carries the entry's saved scroll, a fresh navigation
    /// `(0.0, 0.0)` — the worker's layout mirror and the shell's
    /// compositor must agree, or fixed / sticky elements lay out
    /// at the wrong offset until the next scroll event.
    CommitNavigation {
        url: String,
        state_json: Option<String>,
        traversal: bool,
        scroll: (f32, f32),
    },
    /// The layout viewport changed (window resize, first frame).
    /// Re-lay out the retained document; no refetch.
    Resize { width: f32, height: f32 },
    /// An input event for the page. Routing (hit-testing, focus)
    /// lands in M2; the vocabulary exists now so the protocol does
    /// not need a breaking change then.
    InputEvent(InputEvent),
    /// A streamed resource reply for a `RequestResource` ask.
    /// `pluto-net` sends `ResponseHeaders` once, then `BodyChunk`s
    /// as bytes arrive, then exactly one terminal event (`Done`,
    /// `Error` or `Cancelled`). The renderer never sees a socket.
    ResourceEvent(ResourceEvent),
    /// Shut the worker down. The shell sends this when the tab closes.
    Shutdown,
    /// The tab's visibility changed (Stage 8 Rec 1). A hidden tab
    /// runs no rAF callbacks, builds no display lists, and submits
    /// nothing; JS timers keep running. Appended at the end so
    /// existing variants keep their postcard discriminants — v2
    /// bytes from before this variant still decode.
    SetVisible { visible: bool },
    /// The window's focus changed (Stage 8 Rec 9, v3). The worker
    /// records it for `:focus` handling; the DOM focus itself is
    /// untouched (browsers keep the focused element when the window
    /// loses focus). Same single-window signal as `SetVisible` in
    /// practice — the shell sends both on focus change.
    SetFocused { focused: bool },
    /// The OS / desktop theme changed (M6 Phase 3, Rec 12, v9):
    /// the worker restyles against the new theme (a full rebuild
    /// — never a refetch). Appended at the end so existing variants
    /// keep their postcard discriminants.
    SetTheme { dark: bool },
    /// Session-history sync (Stage 16 Phase 3, Rec 8): the shell is
    /// authoritative for the session history, so after every
    /// mutation it tells the worker the entry count and the current
    /// index — the worker mirrors them as `history.length` and the
    /// base for `pushState` truncation. Appended at the end so
    /// existing variants keep their postcard discriminants.
    HistorySync { length: usize, index: usize },
    /// A WebSocket event for one socket (Stage 16 Phase 3, Rec 11):
    /// the browser-side pump forwards the socket's lifecycle here.
    /// Appended at the end so existing variants keep their
    /// postcard discriminants.
    WebSocketEvent {
        socket_id: u64,
        event: WebSocketEvent,
    },
    /// Evaluate a snippet in this tab's realm (Stage 17, M9
    /// stages 1–2 — the DevTools REPL): the worker runs it with
    /// page privileges only, formats the result to a string on
    /// the tab thread, and answers with
    /// [`RendererToBrowser::DevToolsEvalResult`]. No request id:
    /// the panel sends one snippet at a time and the channel is
    /// in-order, so the next result is always the answer to the
    /// last eval. Appended at the end so existing variants keep
    /// their postcard discriminants.
    DevToolsEval { source: String },
    /// Ask for one level of the DOM tree (Stage 18, M9 stage 3
    /// — the read-only inspector): the lazy tree asks for the
    /// root's children on open, then one node's children per
    /// expansion. `node: None` is the document root. `epoch`
    /// is the shell's idea of the tab's navigation epoch (see
    /// [`RendererToBrowser::DevToolsChildren`]): the worker
    /// answers against its CURRENT document either way, but a
    /// keyed (`Some`) ask whose epoch is stale gets an empty
    /// answer — a generational node id from the old document
    /// could collide with a different node in the new one — so
    /// the shell must re-ask after a navigation. Appended at
    /// the end so existing variants keep their postcard
    /// discriminants.
    DevToolsDomChildren { node: Option<u64>, epoch: u64 },
    /// Ask for one node's details (Stage 18, M9 stage 3): tag +
    /// attributes from the live DOM, computed style + box from
    /// the layout actor. A stale `epoch` gets the not-found
    /// answer (empty tag — see
    /// [`RendererToBrowser::DevToolsNode`]). Appended at the
    /// end so existing variants keep their postcard
    /// discriminants.
    DevToolsInspectNode { node: u64, epoch: u64 },
    /// Hit-test a point (Stage 18, M9 stage 3): the pick mode's
    /// click, in page coordinates (the shell already added the
    /// viewport scroll). Coordinates carry no node id, so the
    /// epoch is advisory — the worker always answers against
    /// its current tree. Appended at the end so existing
    /// variants keep their postcard discriminants.
    DevToolsPickNode { x: f32, y: f32, epoch: u64 },
    /// Set or clear the inspector's highlight (Stage 18, M9
    /// stage 3): `Some(id)` outlines that node's border box(es)
    /// in the next display list, `None` clears it. A stale
    /// `epoch` is ignored outright — a highlight for an old
    /// document's node must not land on the new one. Fire and
    /// forget: the worker re-collects + submits on its own
    /// after applying it, with no layout pass. Appended at the
    /// end so existing variants keep their postcard
    /// discriminants.
    DevToolsHighlight { node: Option<u64>, epoch: u64 },
}

/// Messages from the renderer worker to the browser (shell).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RendererToBrowser {
    /// A new display list is ready: postcard bytes of the
    /// [`DisplayList`](pluto_paint::DisplayList) plus the resource
    /// table its image/blob ids resolve against. `epoch` is the
    /// worker's monotonically increasing frame id; the shell only
    /// installs a list newer than the one it holds, so a slow worker
    /// can never resurrect a stale frame.
    ///
    /// M3 (Stage 11): the worker sends `resources` EMPTY — decoded
    /// resources travel out-of-band, once each, in
    /// [`RendererToBrowser::ResourceReady`], and the shell validates
    /// the list against the per-tab table it accumulated from those
    /// messages (see the module docs). The field stays so the wire
    /// shape keeps a table slot for tests and for a future inline
    /// small-resource path; production submits never fill it.
    SubmitDisplayList {
        epoch: u64,
        bytes: Vec<u8>,
        resources: ResourceTable,
    },
    /// Navigation committed: the URL that actually loaded (after
    /// redirects) and the status line text. Sent when the fetch
    /// lands, before the display list.
    ///
    /// Stage 16 Phase 3 (Rec 8): `replace` is true when the
    /// navigation came from `location.replace()` — the shell
    /// replaces the current history entry instead of pushing a new
    /// one (the flag used to be threaded through
    /// `request_navigation` and discarded in the worker).
    DidCommit {
        url: String,
        status: String,
        replace: bool,
    },
    /// A load failed; the worker already rendered the error page and
    /// sent it as a `SubmitDisplayList` — this carries the reason
    /// for the status line.
    LoadFailed { url: String, reason: String },
    /// The page `<title>` changed.
    TitleChanged { title: String },
    /// The worker died or gave up; the shell shows an error page.
    RendererCrashed { reason: String },
    /// The renderer needs a resource it may not fetch itself
    /// (external stylesheet, image, script — M2). The browser
    /// fetches and answers with `BrowserToRenderer::ResourceEvent`
    /// messages; `request_id` matches the answer to the ask.
    /// Frozen as v1 in Stage 5: every field of [`ResourceRequest`]
    /// is needed by a later layer (cache, cookies, CORS, CSP,
    /// preload scanner), so the shape is fixed now, before Stage 6's
    /// parser depends on it.
    RequestResource(ResourceRequest),
    /// Cancel an in-flight `RequestResource`. The browser answers
    /// with a terminal `ResourceEvent::Cancelled` for that id.
    CancelResource { request_id: u64 },
    /// The worker asks the shell to scroll the viewport (the page
    /// itself) by page px (Stage 8 Rec 9, v3): the remainder of a
    /// wheel / keyboard scroll after the nested scroll containers
    /// under the cursor took what they could. The shell owns the
    /// viewport scroll (shader uniform) and clamps it; the worker
    /// never scrolls the viewport itself.
    ScrollViewport { dx: f32, dy: f32 },
    /// One decoded resource, delivered out-of-band exactly once
    /// (M3, Stage 11 — see the module docs): the worker decoded
    /// the image on the rayon pool and sends its pixels here, in a
    /// frame of their own, BEFORE the `SubmitDisplayList` that
    /// references the id — both messages travel the same channel
    /// in order, so the shell's per-tab table holds the resource
    /// before the list that names it arrives. `epoch` is the
    /// navigation epoch the resource belongs to; a resource whose
    /// epoch is no longer current is stale and must be dropped.
    /// An animated frame advance re-sends the same id with the new
    /// frame's pixels. A resource the worker cannot send (its
    /// encoded frame would exceed the IPC cap) is never sent at
    /// all — the page shows the broken-image placeholder instead.
    ResourceReady {
        epoch: u64,
        resource_id: pluto_paint::ResourceId,
        resource: pluto_paint::Resource,
    },
    /// Document-load progress (M3, Stage 11): bytes of the
    /// document body received so far, against the `Content-Length`
    /// when the server sent one (`None` = indeterminate — the
    /// shell shows a spinner, not a percentage). Document load
    /// only; subresource progress is out of scope.
    LoadProgress {
        received: u64,
        total: Option<u64>,
    },
    /// The tab's favicon changed (M3, Stage 11): `Some(id)` names
    /// the favicon image in the tab's resource table (delivered
    /// by `ResourceReady` before this message), `None` means the
    /// site has no icon and the tab strip shows the default. Sent
    /// once per navigation (link[rel=icon] or the /favicon.ico
    /// fallback), not per frame.
    FaviconChanged {
        resource_id: Option<pluto_paint::ResourceId>,
    },
    /// Per-layer sandbox results (Stage 13 Rec 12a): sent once
    /// by the child right after the sandbox hook, before the
    /// first navigation. Each layer reports whether it landed;
    /// the shell records it per tab and warns when a load-bearing
    /// layer (user-namespace, chroot, seccomp) is OFF.
    SandboxReport { layers: Vec<(String, bool)> },
    /// Heartbeat (Stage 13 Rec 12b): the child sends one every
    /// few seconds from its tab-thread tick while alive, so the
    /// browser-side watchdog can tell a hung renderer (no
    /// heartbeat, no progress) from a merely slow one.
    Heartbeat,
    /// A transform/opacity-only change (M6, Stage 14 Rec 9):
    /// the new compositing state of the elements whose transform
    /// or opacity changed — and their descendants, whose page
    /// transforms move with them — keyed by the same ids the
    /// display list's `PushTransform`/`PushOpacity` items carry.
    /// Out-of-band like `ResourceReady`: the message carries
    /// only the changed elements, never a rebuilt list, so a
    /// transform-only frame costs a few dozen bytes per element
    /// and stays far under the 64 MiB frame cap. The shell
    /// patches its active list in place (and only when `epoch`
    /// matches the active list's epoch — a pending newer list
    /// already carries the new values).
    CompositingUpdate {
        epoch: u64,
        updates: Vec<CompositedUpdate>,
    },
    /// One decoded video frame (Stage 16 Phase 1, Rec 1): YUV420
    /// planes for one `<video>` element, sent by the renderer as
    /// soon as the decode thread produces it — NOT through
    /// `ResourceReady`/`Blob`. The M7 canvas drain (hash-compare
    /// and dirty flag, full-RGBA clone per frame) is the 498 MB/s
    /// trap for video: a video frame differs every frame, so the
    /// hash never skips. YUV420 is 1.5 bytes/px against RGBA's 4
    /// (1080p: ~3.1 MB/frame vs ~8.3 MB), and the shell uploads
    /// the planes straight into three R8 wgpu textures — no RGBA
    /// conversion on the hot path (Phase 2). Measured at 1080p60
    /// in `pluto-ipc`'s transport tests before any decoder code
    /// was written; see `VideoFrame` for the numbers.
    ///
    /// `epoch` is the navigation epoch the frame belongs to; a
    /// frame whose epoch is no longer current is stale and must
    /// be dropped. Frames are NOT queued: the renderer keeps at
    /// most one undelivered frame per element (see
    /// `pluto_media::FrameChannel`) and drops older ones — a
    /// slow shell sees the latest frame, never a backlog. The
    /// display list names a frame by `frame_id` (Phase 2 adds the
    /// `DisplayItem`); a list that names a frame the shell has
    /// not received yet paints the previous frame (or the poster
    /// when there is none).
    VideoFrame {
        epoch: u64,
        frame: VideoFrame,
    },
    /// `history.pushState()` (Stage 16 Phase 3, Rec 8): the page
    /// changed its URL + state without a navigation. The shell
    /// truncates the forward entries, appends `{ url, state_json }`,
    /// and updates the address bar — no fetch, no `popstate`. The
    /// worker already updated its own document URL and realm state;
    /// this message is the shell-side half. Appended at the end so
    /// existing variants keep their postcard discriminants.
    HistoryPush {
        url: String,
        state_json: Option<String>,
    },
    /// `history.replaceState()` / a same-document hash change
    /// (Stage 16 Phase 3, Rec 8/9): replace the current entry's URL
    /// + state in place, no new entry, no navigation.
    HistoryReplace {
        url: String,
        state_json: Option<String>,
    },
    /// `history.back()` / `forward()` / `go(n)` from script (Stage
    /// 16 Phase 3, Rec 8): the shell moves its index by `delta`
    /// (clamped) and navigates with `CommitNavigation { traversal:
    /// true }`, so the worker fires `popstate` after the commit.
    /// A `delta` that moves nowhere is a no-op (no message back).
    HistoryGo { delta: i32 },
    /// Open a WebSocket (Stage 16 Phase 3, Rec 11): the worker
    /// asks the browser-side broker to run the handshake — the
    /// renderer never opens a socket itself. `origin` is the
    /// document origin the handshake's `Origin` header carries.
    /// The broker answers with `BrowserToRenderer::WebSocketEvent`
    /// frames for this `socket_id`.
    WebSocketOpen {
        socket_id: u64,
        url: String,
        protocols: Vec<String>,
        origin: String,
    },
    /// Send one WebSocket message (Stage 16 Phase 3, Rec 11):
    /// text (`is_binary: false`, UTF-8) or binary. Client masking
    /// is applied by the browser-side pump, never by script.
    WebSocketSend {
        socket_id: u64,
        data: Vec<u8>,
        is_binary: bool,
    },
    /// Close a WebSocket (Stage 16 Phase 3, Rec 11): the
    /// browser-side pump sends the close frame and ends the pump.
    WebSocketClose {
        socket_id: u64,
        code: Option<u16>,
        reason: String,
    },
    /// The DevTools REPL's answer (Stage 17, M9 stages 1–2):
    /// the snippet's value formatted to a string on the tab
    /// thread (`format_console_arg` — a live JS value never
    /// crosses the wire), or the thrown exception formatted
    /// with its stack when the snippet threw. Appended at the
    /// end so existing variants keep their postcard
    /// discriminants.
    DevToolsEvalResult { result: String },
    /// Batched console output (Stage 17, M9 stages 1–2): the
    /// tab thread drains its captured `console` lines (and its
    /// uncaught-error lines, as `level: "error"`) once per
    /// frame tick and pushes only the new ones — a `console.log`
    /// in a loop never produces a message per line, and a
    /// chatty page never makes the shell re-read the whole
    /// capture. The shell caps its per-tab buffer (~1,000
    /// lines, oldest evicted). Appended at the end so existing
    /// variants keep their postcard discriminants.
    ConsoleLines { lines: Vec<(String, String)> },
    /// One level of the DOM tree (Stage 18, M9 stage 3): the
    /// answer to [`BrowserToRenderer::DevToolsDomChildren`].
    /// `epoch` is the worker's navigation epoch at answer time
    /// — the shell drops the answer when it is older than the
    /// epoch the shell already knows (a newer document has
    /// superseded it) and adopts a newer one. `children` is
    /// empty when the node is dead or has no children (the
    /// root ask on a tab with no document yet is empty too).
    /// Appended at the end so existing variants keep their
    /// postcard discriminants.
    DevToolsChildren {
        epoch: u64,
        node: Option<u64>,
        children: Vec<DevToolsNodeSummary>,
    },
    /// One node's details (Stage 18, M9 stage 3): the answer
    /// to [`BrowserToRenderer::DevToolsInspectNode`]. `tag` is
    /// empty when the node no longer exists in the document —
    /// the shell shows "node no longer in the document" for
    /// that, never a dead id's stale data. `geometry` is `None`
    /// when the node has no box (not laid out yet, or
    /// `display: none`); `computed` is empty then too. Borders
    /// are unmodelled in Pluto, so the box carries no border
    /// fields: the padding box IS the border box. The box is
    /// boxed: the `[String; 4]` margin/padding arrays make
    /// `DevToolsBox` ~200 bytes, and an unboxed
    /// `Option<DevToolsBox>` would push `RendererToBrowser`
    /// (and `Message::RendererEvent`) over clippy's
    /// `large_enum_variant` threshold — postcard serialises
    /// `Box<T>` exactly like `T`, so the wire is unchanged.
    /// Appended at the end so existing variants keep their
    /// postcard discriminants.
    DevToolsNode {
        epoch: u64,
        node: u64,
        tag: String,
        attrs: Vec<(String, String)>,
        computed: Vec<(String, String)>,
        geometry: Option<Box<DevToolsBox>>,
    },
    /// The pick mode's hit-test answer (Stage 18, M9 stage 3):
    /// the node under the clicked point, if any — `None` when
    /// the click hit nothing (or the ask's epoch was stale).
    /// Appended at the end so existing variants keep their
    /// postcard discriminants.
    DevToolsPicked { epoch: u64, node: Option<u64> },
}

/// One row of the inspector's lazy DOM tree (Stage 18, M9
/// stage 3): a node's identity on the wire (`id` is the
/// slotmap key's ffi value, so the shell never interprets it,
/// only echoes it back), its tag (`#text` / `#comment` /
/// `#document` for non-elements), a short text preview for
/// text nodes (whitespace-collapsed, capped), and whether it
/// has children to expand into.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevToolsNodeSummary {
    pub id: u64,
    pub tag: String,
    pub text: Option<String>,
    pub has_children: bool,
}

/// One node's border box for the inspector (Stage 18, M9
/// stage 3): page coordinates, plus the computed margin and
/// padding as CSS strings (`16px`, `5%`, `auto`) in
/// top/right/bottom/left order — percentages are never resolved
/// to fake px. No border fields: borders are unmodelled in
/// Pluto, so the padding box == the border box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevToolsBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub margin: [String; 4],
    pub padding: [String; 4],
}

/// A resource ask, frozen as v1 in Stage 5. This is the payload of
/// [`RendererToBrowser::RequestResource`] and the parameter of
/// `pluto-net`'s fetch API — one shape, so the wire format and the
/// service API can never drift apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub request_id: u64,
    /// Which tab asked (assigned by the shell, M3; echoed by the
    /// worker — never hardcoded 0 in production).
    pub tab_id: u64,
    /// The navigation epoch the ask belongs to. A reply whose epoch
    /// is no longer current is stale and must be dropped.
    pub epoch: u64,
    pub url: String,
    /// HTTP method for the ask (`GET` for navigations and classic
    /// subresources; script `fetch()` may set others — the CORS
    /// layer in `pluto-net` validates them). Stage 9 Rec 2.
    pub method: String,
    /// Extra request headers set by script (`fetch()` `init.headers`).
    /// Forbidden headers are stripped before the request is sent and
    /// CORS-unsafe names force a preflight; the list rides the wire
    /// so the browser process (never the realm) applies the policy.
    /// Stage 9 Rec 2.
    pub headers: Vec<(String, String)>,
    /// `true` when the ask came from a srcset / `<picture>` candidate
    /// (Mixed Content L2 `imageset` initiator). Imageset images are
    /// BLOCKED, never auto-upgraded, on a secure page. Threaded from
    /// the preload scanner's `from: "imageset"` label through
    /// `page.rs::subresource_request`. Stage 9 Rec 2/6.
    pub is_imageset: bool,
    pub destination: Destination,
    pub priority: Priority,
    pub mode: RequestMode,
    pub credentials: CredentialsMode,
    /// The document origin making the ask, e.g.
    /// `https://example.com` — drives the mixed-content gate and,
    /// at M5, CORS.
    pub origin: String,
    /// The top-level site the ask belongs to (the document origin
    /// for now; proper eTLD+1 partitioning is an M5 refinement).
    /// Cache keys and cookie jars are partitioned on this from day
    /// one.
    pub top_level_site: String,
    /// Honour `upgrade-insecure-requests`: when true, http
    /// subresources on an https page are upgraded to https instead
    /// of going through the mixed-content gate. Set by the
    /// document's CSP at M10; false until then.
    pub upgrade_insecure_requests: bool,
    /// The element's `integrity` attribute, verbatim (Stage 9 Rec
    /// 14, SRI): space-separated `<algorithm>-<base64 digest>`
    /// entries the fetch must check against the streamed bytes.
    /// `None` means no integrity metadata — no check, not a
    /// failure. Set by the scanner / streaming parser on
    /// subresource discoveries; navigations and `fetch()` asks
    /// leave it `None`.
    pub integrity: Option<String>,
    /// Request body bytes (Stage 13 Rec 6: form POSTs). `None`
    /// for GET/HEAD and every bodyless ask; `Some` carries the
    /// encoded form body, with `Content-Type` in `headers`.
    /// The net layer drops it (and strips the body headers) when
    /// a redirect converts the method to GET.
    pub body: Option<Vec<u8>>,
}

impl ResourceRequest {
    /// A top-level navigation ask for `url`: highest priority, the
    /// document destination, navigation mode. `origin` and
    /// `top_level_site` are the URL's own origin.
    pub fn navigate(request_id: u64, tab_id: u64, epoch: u64, url: &str) -> Self {
        let origin = origin_of(url);
        Self {
            request_id,
            tab_id,
            epoch,
            url: url.to_string(),
            method: "GET".to_string(),
            headers: Vec::new(),
            is_imageset: false,
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: origin.clone(),
            top_level_site: origin,
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        }
    }
}

/// The origin of a URL (WHATWG HTML §7.1.1): either a tuple origin
/// `(scheme, host, port)` with default-port normalization, or an
/// opaque origin — an internal value whose only meaningful operation
/// is equality, serialized as the string `"null"`.
///
/// Same origin means: the same opaque origin, or two tuple origins
/// with identical scheme, identical port, and equal hosts. Two
/// *different* opaque origins both serialize to `"null"` but are
/// **never** same origin, so opaque origins are unforgeable unique
/// identities (a fresh counter value per creation), never compared
/// by their `"null"` string.
///
/// `domain` is deliberately absent: Pluto never implements
/// `document.domain` (Stage 9 report §b — the setter relaxes the
/// same-origin check and the spec is removing it), so there is no
/// relaxed-origin state to carry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Origin {
    /// `(scheme, host, port)`; `port` is `None` when the URL has
    /// no explicit port or its explicit port is the scheme's
    /// default, so `https://example.com:443` and
    /// `https://example.com` are the same origin.
    Tuple {
        scheme: String,
        host: String,
        port: Option<u16>,
    },
    /// An opaque origin: the `u64` is a process-local identity from
    /// [`Origin::opaque`], meaningful only for equality within this
    /// process. It must never be compared against an opaque origin
    /// minted in another process (Rec 9's cross-process transport
    /// mints fresh opaque origins on receipt instead of trusting
    /// a deserialized id).
    Opaque(u64),
}

/// The default port for a scheme (WHATWG URL §4.2), or `None` when
/// the scheme has no default port.
fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        "ftp" => Some(21),
        _ => None,
    }
}

/// Counter minting fresh opaque-origin identities (see
/// [`Origin::Opaque`]).
static OPAQUE_ORIGIN_COUNTER: AtomicU64 = AtomicU64::new(1);

impl Origin {
    /// A fresh opaque origin: a unique identity no other origin —
    /// opaque or tuple — is ever same origin with, and no
    /// serialization can recreate.
    pub fn opaque() -> Self {
        Origin::Opaque(OPAQUE_ORIGIN_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Whether this is an opaque origin.
    pub fn is_opaque(&self) -> bool {
        matches!(self, Origin::Opaque(_))
    }

    /// The origin of a URL string, parsed structurally:
    ///
    /// * `http`/`https`/`ws`/`wss`/`ftp` (or any scheme with an
    ///   authority) → the tuple origin, scheme and host lowercased,
    ///   default ports normalized to `None`.
    /// * `blob:<inner>` → the inner URL's origin (parse failure →
    ///   opaque), per WHATWG URL §4.5.
    /// * Anything without an authority (`about:`, `data:`, `file:`,
    ///   an unparseable string, a relative URL) → a fresh opaque
    ///   origin. Callers that need `about:blank` / `srcdoc`
    ///   inheritance must use [`Origin::for_document`] instead,
    ///   which takes the creator origin.
    pub fn from_url(url: &str) -> Self {
        // `blob:` derives its origin from the inner URL.
        if url.len() > 5 && url[..5].eq_ignore_ascii_case("blob:") {
            return Origin::from_url(&url[5..]);
        }
        let Ok(uri) = url.parse::<http::Uri>() else {
            return Origin::opaque();
        };
        let (Some(scheme), Some(authority)) = (uri.scheme_str(), uri.authority()) else {
            return Origin::opaque();
        };
        let scheme = scheme.to_ascii_lowercase();
        let host = authority.host().to_ascii_lowercase();
        // `http::Uri` keeps the brackets on an IPv6 literal
        // (`[::1]`); strip them here — `serialize` re-adds them —
        // so the stored host is the bare literal.
        let host = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(&host)
            .to_string();
        if host.is_empty() {
            return Origin::opaque();
        }
        let port = match authority.port_u16() {
            Some(p) if Some(p) == default_port(&scheme) => None,
            p => p,
        };
        Origin::Tuple { scheme, host, port }
    }

    /// The origin of a document loaded from `url`, given the origin
    /// of the document that created it (`creator`) and whether the
    /// document is sandboxed (WHATWG HTML §7.1.1):
    ///
    /// * Sandboxed (without `allow-same-origin`) → a fresh opaque
    ///   origin, whatever the URL.
    /// * `about:blank` created by a page, and `srcdoc` without
    ///   sandbox → the creator's origin (same-origin with its
    ///   embedder); with no creator (a top-level `about:blank`) →
    ///   a fresh opaque origin.
    /// * `data:` → a fresh opaque origin per document.
    /// * Anything else → [`Origin::from_url`].
    pub fn for_document(url: &str, creator: Option<&Origin>, sandboxed: bool) -> Self {
        if sandboxed {
            return Origin::opaque();
        }
        // `about:blank` / `about:srcdoc`, ignoring any fragment.
        let without_fragment = url.split('#').next().unwrap_or(url);
        if without_fragment.eq_ignore_ascii_case("about:blank")
            || without_fragment.eq_ignore_ascii_case("about:srcdoc")
        {
            return match creator {
                Some(origin) => origin.clone(),
                None => Origin::opaque(),
            };
        }
        if url.len() >= 5 && url[..5].eq_ignore_ascii_case("data:") {
            return Origin::opaque();
        }
        Origin::from_url(url)
    }

    /// Whether `other` is the same origin as this one: the same
    /// opaque origin, or tuple origins with identical scheme, host,
    /// and port.
    pub fn is_same_origin(&self, other: &Origin) -> bool {
        self == other
    }

    /// Serialize this origin: `scheme://host` (plus `:port` when
    /// the port is non-default) for a tuple origin, `"null"` for an
    /// opaque origin. The serialization is lossy for opaque origins
    /// by design — never compare serialized origins for equality.
    pub fn serialize(&self) -> String {
        match self {
            Origin::Tuple { scheme, host, port } => {
                // An IPv6 literal needs brackets to round-trip.
                let host = if host.contains(':') {
                    format!("[{host}]")
                } else {
                    host.clone()
                };
                match port {
                    Some(p) => format!("{scheme}://{host}:{p}"),
                    None => format!("{scheme}://{host}"),
                }
            }
            Origin::Opaque(_) => "null".to_string(),
        }
    }
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.serialize())
    }
}

/// The serialized origin (`scheme://host`, `:port` when non-default,
/// `"null"` for an opaque origin) of a URL string — the serializer
/// for [`Origin`]. A URL with no authority (`about:`, `data:`,
/// unparseable) serializes to `"null"`.
pub fn origin_of(url: &str) -> String {
    Origin::from_url(url).serialize()
}

/// What kind of resource a [`RendererToBrowser::RequestResource`]
/// is asking for. The destination drives priority, byte caps, the
/// mixed-content gate, and (at M5) the CORS mode — it is not a
/// label, it is load-bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Destination {
    /// A top-level navigation. Only the browser decides what a
    /// status code means for a navigation (commit, error page,
    /// download); the renderer only asks for the bytes.
    Document,
    Stylesheet,
    Script,
    Image,
    Font,
    /// A `fetch()` call (Stage 7). CORS-gated at M5.
    Fetch,
    /// An `<audio>` element's media (Stage 16 Phase 1, Rec 2):
    /// split from `Video` so the gates can tell them apart —
    /// both map to CSP `media-src` and are upgradable mixed
    /// content, but a future `<audio>`-only rule (e.g. a
    /// background-audio policy) needs the split.
    Audio,
    /// A `<video>` element's media (Stage 16 Phase 1, Rec 2).
    /// See `Audio`.
    Video,
    /// A WebSocket handshake / connection (Stage 16 Phase 1,
    /// Rec 2). `ws://` on an https page is blockable mixed
    /// content (see `pluto_net::mixed`); `wss://` is allowed.
    /// Maps to CSP `connect-src` and the blocker's `$websocket`
    /// bit. No WebSocket transport exists yet (Rec 11) — the
    /// destination exists now so the gates are correct before
    /// the first socket opens.
    WebSocket,
    /// A streaming `fetch()` body (Stage 16 Phase 3, Rec 12):
    /// the response is delivered to JS as a `ReadableStream`,
    /// chunk by chunk, instead of buffered whole. It gets its own
    /// caps (1 GiB / 600 s — see `pluto_net::service`) so a
    /// long-lived stream (SSE, large download) is not killed by
    /// the 20 s / 10 MiB `Fetch` limits; the 10 MiB bound still
    /// applies where a body is *buffered* (`.text()` / `.json()`),
    /// enforced in the realm. Maps to CSP `connect-src` and the
    /// blocker's `xmlhttprequest` bit, like `Fetch`. Appended last
    /// so existing variants keep their postcard discriminants.
    Stream,
    /// Anything else (e.g. a future destination the parser does
    /// not know yet). Gets the most conservative caps.
    Other,
}

/// One decoded video frame, YUV420 planar (Stage 16 Phase 1,
/// Rec 1) — the unit of the video-frame channel. YUV420 is the
/// decoder's native output (openh264, rav1d) and the GPU's
/// cheapest upload shape (three R8 textures, no CPU RGBA
/// conversion), and at 1.5 bytes/px it is 2.7× smaller on the wire
/// than RGBA: a 1080p frame is 1920×1080 + 2×(960×540) =
/// 3,110,400 bytes vs 8,294,400 RGBA — ~187 MB/s at 60 fps
/// instead of ~498 MB/s. A 4K frame is 12,441,600 bytes, still
/// 5× under the 64 MiB frame cap with room for the rest of the
/// page's traffic; RGBA 4K (33 MB) would evict half the 64 MiB
/// image and GPU texture caches per frame.
///
/// The channel decision was measured, not assumed
/// (`transport::tests::video_frame_channel_measured_at_1080p60`,
/// 2026-11-26, sandbox, debug build, three runs): **YUV420 over
/// the IPC message, 168.6–219.9 fps / 500.1–652.4 MB/s at
/// 1080p standalone** (2.8–3.7× headroom over 60 fps) and
/// **59.3 fps / 176.0 MB/s under the full workspace gate's
/// parallel load** — essentially at the bar when the machine is
/// busy, which is why the test asserts the load-robust ratio
/// (YUV ≥ 20× RGBA) rather than an absolute 60 fps. The
/// socket itself does 2.9 ms/frame, and with `serde_bytes` the
/// codec is a memcpy. The RGBA-over-`ResourceReady` baseline
/// measured **0.7 fps / 5.2–5.5 MB/s** on the same harness, so
/// shared memory and direct wgpu upload were never needed: the
/// message channel wins by two orders of magnitude, and the
/// latest-only `FrameChannel` (`pluto-media`) drops instead of
/// queueing when the consumer falls behind. The same per-byte
/// serde trap also slows the existing M7
/// `Resource::Image { rgba }` path — that path is unchanged
/// here (out of scope), but `serde_bytes` on `Resource` is a
/// known follow-up.
///
/// `origin_clean` is the per-frame tainting decision (Rec 3):
/// same-origin or `data:` media is clean; cross-origin media is
/// clean ONLY when its fetch was CORS-approved (`crossorigin`
/// and a passing CORS check — the approval is recorded on the
/// fetch, unlike `<img>` today, where no approval is recorded
/// and every cross-origin image taints). A tainted frame still
/// paints — it only blocks `drawImage(video)` readback
/// (`getImageData` / `toDataURL`), exactly like a tainted canvas.
///
/// `pts_ms` / `duration_ms` are the presentation timestamps the
/// A/V clock schedules against (Rec 6); `frame_id` is a
/// per-element monotonic counter, so the shell can tell a new
/// frame from a re-send of the last one without hashing pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoFrame {
    /// Which `<video>` element this frame belongs to (the
    /// element's `DomKey` as a `u64`).
    pub element_id: u64,
    /// Per-element monotonic frame counter.
    pub frame_id: u64,
    /// Frame dimensions in px. All three planes share them
    /// (`u`/`v` are half-width, half-height: YUV420).
    pub width: u32,
    pub height: u32,
    /// Presentation timestamp, ms on the media clock.
    pub pts_ms: f64,
    /// Frame duration, ms (from the container's frame rate).
    pub duration_ms: f64,
    /// Luma plane, `width × height` bytes, row stride
    /// `y_stride`. Serialised as raw bytes (`serde_bytes` —
    /// see the workspace `Cargo.toml` note): serde's default
    /// `Vec<u8>` encoding is one call per byte, which measured
    /// 227 ms encode + 329 ms decode per 1080p frame in debug.
    #[serde(with = "serde_bytes")]
    pub y: Vec<u8>,
    /// Chroma planes, `(width/2) × (height/2)` bytes each,
    /// row stride `uv_stride`. Same `serde_bytes` treatment.
    #[serde(with = "serde_bytes")]
    pub u: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub v: Vec<u8>,
    /// Bytes per row of the luma plane (≥ `width`).
    pub y_stride: u32,
    /// Bytes per row of each chroma plane (≥ `width / 2`).
    pub uv_stride: u32,
    /// Whether this frame may be read back by script
    /// (`drawImage(video)` → `getImageData`/`toDataURL`).
    /// See the struct docs for the rule.
    pub origin_clean: bool,
}

impl VideoFrame {
    /// Bytes of YUV420 payload (all three planes), excluding
    /// the message envelope. 1080p = 3,110,400; 4K = 12,441,600.
    pub fn payload_bytes(&self) -> usize {
        self.y.len() + self.u.len() + self.v.len()
    }
}

/// Fetch priority, mirroring Chromium's five bands
/// (VeryHigh/High/Medium/Low/VeryLow). The scheduler in `pluto-net`
/// orders by (priority, discovery order); `fetchpriority` maps onto
/// this directly in Stage 6. A hint, not a guarantee — bandwidth is
/// still subject to connection limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
}

impl Priority {
    /// Whether this priority is delayable: held behind
    /// non-delayable work while the scheduler is saturated.
    /// Prefetch and below-fold images are delayable; the document
    /// and parser-blocking CSS/JS are not.
    pub fn is_delayable(self) -> bool {
        matches!(self, Priority::Low | Priority::VeryLow)
    }
}

/// The fetch mode of a request: whether it may cross origins, and
/// whether that crossing needs a CORS check. Only `Navigate` and
/// `NoCors` are used before `fetch()` arrives (Stage 7); the rest
/// exist now so the wire shape does not change then.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestMode {
    /// A top-level navigation. Same-origin policy does not apply.
    Navigate,
    /// A classic subresource load (stylesheet, script, image).
    /// Cross-origin is allowed; the response is opaque to script.
    NoCors,
    /// Same-origin only; cross-origin fails before the network.
    SameOrigin,
    /// A CORS fetch: cross-origin allowed only if the response
    /// passes the CORS check (M5).
    Cors,
}

/// Whether credentials (cookies, and later `Authorization`) travel
/// with the request. The cookie jar itself is M5; until then only
/// `SameOrigin` and `Omit` have any effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialsMode {
    Omit,
    SameOrigin,
    Include,
}

/// The streaming reply to a `RequestResource`, sent as
/// [`BrowserToRenderer::ResourceEvent`]. Exactly one terminal event
/// (`Done`, `Error` or `Cancelled`) closes each `request_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceEvent {
    /// Response headers landed. Sent once, before any chunk.
    /// `headers` are filtered — `Set-Cookie` is stripped before the
    /// renderer ever sees it (the jar, M5, is the only place that
    /// reads it). `status` is the raw HTTP status as a number.
    /// `url` is the URL the bytes actually came from, after
    /// redirects — the caller reports it as the final URL.
    ResponseHeaders {
        request_id: u64,
        status: u16,
        url: String,
        headers: Vec<(String, String)>,
        content_length: Option<u64>,
    },
    /// The next bytes of the (decoded) body.
    BodyChunk { request_id: u64, bytes: Vec<u8> },
    /// The body is complete.
    Done { request_id: u64 },
    /// The fetch failed. `code` is machine-readable
    /// (`invalid_url`, `too_many_redirects`, `insecure_redirect`,
    /// `too_large`, `timeout`, `cancelled`, `request`, `body`);
    /// `reason` is human-readable. The renderer renders an error
    /// page or a broken-resource placeholder.
    Error {
        request_id: u64,
        code: String,
        reason: String,
    },
    /// The fetch was cancelled (renderer `CancelResource`,
    /// navigation epoch bump, tab close, or timeout).
    Cancelled { request_id: u64 },
}

/// One WebSocket lifecycle event, browser → renderer (Stage 16
/// Phase 3, Rec 11). The browser-side pump owns the socket; the
/// renderer only ever sees these plain-data events, delivered over
/// the same mpsc bridge as fetch completions — rquickjs values
/// never cross threads. `Opened` is sent once, after the 101,
/// carrying the subprotocol the server selected ("" when none);
/// `Message` carries one frame's payload (text frames are UTF-8
/// bytes; `is_binary` distinguishes them); `Closed` is terminal
/// (clean close, error, or tab close — the pump sends exactly one
/// terminal event per socket); `Error` is a handshake / protocol
/// failure before `Opened` (after `Opened`, failures arrive as
/// `Closed` with `clean: false`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WebSocketEvent {
    Opened {
        socket_id: u64,
        protocol: String,
    },
    Message {
        socket_id: u64,
        data: Vec<u8>,
        is_binary: bool,
    },
    Closed {
        socket_id: u64,
        code: Option<u16>,
        reason: String,
        clean: bool,
    },
    Error {
        socket_id: u64,
        message: String,
    },
}

/// Modifier keys held during an input event (Stage 8 Rec 9, v3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

/// What the pointer did (Stage 8 Rec 9, v3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointerKind {
    Move,
    Down,
    Up,
}

/// How a wheel delta is expressed (Stage 8 Rec 9, v3). The worker
/// scales `Line` deltas (×16 px, the shell's line height) and
/// `Page` deltas (× the viewport size on that axis); `Pixel`
/// deltas pass through as-is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaMode {
    Pixel,
    Line,
    Page,
}

/// Where a wheel event sits in its gesture (Stage 8 Rec 9, v3).
/// The shell synthesises `Began`/`Changed` from idle time (iced
/// reports no wheel phases); `Ended` is synthesised worker-side by
/// timeout — the tab thread clears the gesture when no wheel has
/// arrived for ~200 ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WheelPhase {
    Began,
    Changed,
    Ended,
}

/// An input event for the page (Stage 8 Rec 9, v3). Positions are
/// page coordinates in layout px — the shell adds the viewport
/// scroll offset before forwarding, so the worker hit-tests
/// directly against its layout tree. `timestamp_ms` is monotonic
/// milliseconds since the shell started (gesture/velocity timing
/// and rAF timestamp alignment). Physical px are deliberately NOT
/// sent: the shader widget's `update` has no scale factor, and the
/// worker uses layout px only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputEvent {
    /// Pointer move / button down / button up at page coordinates.
    /// `button` is the button that changed (`None` for a move):
    /// 0 = left, 1 = middle, 2 = right. `buttons` is the bitmask
    /// of buttons held down (bit 0 = left, 1 = middle, 2 = right).
    /// `pointer_id` is 0 for the mouse, reserved for multi-pointer.
    Pointer {
        kind: PointerKind,
        x: f32,
        y: f32,
        button: Option<u8>,
        buttons: u16,
        modifiers: Modifiers,
        timestamp_ms: u64,
        pointer_id: u64,
    },
    /// Mouse wheel scroll at the cursor's page coordinates.
    Wheel {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
        delta_mode: DeltaMode,
        phase: WheelPhase,
        modifiers: Modifiers,
        timestamp_ms: u64,
    },
    /// A key press or release. `key` is the logical key
    /// ("Enter", "ArrowDown", "a"); `code` is the physical key
    /// ("Enter", "ArrowDown", "KeyA"). `repeat` is true for OS
    /// key-repeat events.
    Key {
        key: String,
        code: String,
        pressed: bool,
        repeat: bool,
        modifiers: Modifiers,
        timestamp_ms: u64,
    },
}

/// The in-process transport for the app -> worker direction: an
/// unbounded tokio mpsc channel carrying [`BrowserToRenderer`]
/// messages. Unbounded so a send from the shell's `update` never
/// blocks the UI thread; the worker drains it in order. The worker
/// -> app direction is the iced subscription the worker runs as —
/// the subscription *is* that channel in iced, so there is no
/// second queue to keep in sync.
pub type CommandSender = tokio::sync::mpsc::UnboundedSender<BrowserToRenderer>;
pub type CommandReceiver = tokio::sync::mpsc::UnboundedReceiver<BrowserToRenderer>;

/// Create the app -> worker command channel.
pub fn command_channel() -> (CommandSender, CommandReceiver) {
    tokio::sync::mpsc::unbounded_channel()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::Stylesheet;
    use pluto_html::parse as parse_html;
    use pluto_layout::{LayoutTree, Theme};
    use pluto_paint::{DisplayItem, DisplayList, Resource, ResourceId};

    fn example_list() -> DisplayList {
        let html = "<html><head><style>body { background-color: #f0f0f2; }</style></head><body><p>Hello</p></body></html>";
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("body { background-color: #f0f0f2; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        DisplayList::build(&tree)
    }

    #[test]
    fn protocol_version_is_fifteen() {
        // Bumped 5 -> 6 (Stage 13: ResourceRequest.body for form
        // POSTs), 6 -> 7 (Stage 13 Rec 12: SandboxReport +
        // Heartbeat), 7 -> 8 (Stage 14 Rec 9: CompositingUpdate),
        // 8 -> 9 (Stage 14 Phase 3, Rec 12: SetTheme), 9 -> 10
        // (Stage 15 Phase 2, M7 Rec 4: DisplayItem::VectorPath),
        // 10 -> 11 (Stage 16 Phase 1, Rec 2:
        // Destination::Audio/Video/WebSocket; Rec 1:
        // RendererToBrowser::VideoFrame), and 11 -> 12 (Stage 16
        // Phase 3, Rec 8/9/11/12: CommitNavigation carries
        // state_json + traversal, DidCommit carries replace, new
        // HistoryPush / HistoryReplace / HistoryGo / HistorySync
        // and WebSocketOpen / Send / Close / Event messages, and
        // Destination::Stream), and 12 -> 13 (Stage 16 Phase 3,
        // finisher: CommitNavigation also carries the entry's
        // scroll, and every WebSocketEvent variant carries
        // `socket_id` — the Phase 3 wire shapes, completed), and
        // 13 -> 14 (Stage 17, M9 stages 1–2: DevToolsEval /
        // DevToolsEvalResult / ConsoleLines, appended at the end
        // of each enum), and 14 -> 15 (Stage 18, M9 stage 3:
        // DevToolsDomChildren / DevToolsInspectNode /
        // DevToolsPickNode / DevToolsHighlight and
        // DevToolsChildren / DevToolsNode / DevToolsPicked,
        // appended at the end of each enum).
        assert_eq!(PROTOCOL_VERSION, 15);
    }

    #[test]
    fn devtools_messages_round_trip_through_versioned_bytes() {
        // Stage 17 (M9 stages 1–2): the REPL ask, its formatted
        // answer, and a batched console drain all survive the
        // versioned envelope — including an empty drain (a tick
        // with nothing new sends nothing, but the shape must
        // still decode) and a multi-line error batch.
        let eval = BrowserToRenderer::DevToolsEval {
            source: "document.title".to_string(),
        };
        let bytes = encode_message(&eval);
        assert_eq!(decode_message::<BrowserToRenderer>(&bytes), Ok(eval));
        let result = RendererToBrowser::DevToolsEvalResult {
            result: "Example Domain".to_string(),
        };
        let bytes = encode_message(&result);
        assert_eq!(
            decode_message::<RendererToBrowser>(&bytes),
            Ok(result)
        );
        let lines = RendererToBrowser::ConsoleLines {
            lines: vec![
                ("log".to_string(), "hello".to_string()),
                (
                    "error".to_string(),
                    "boom\n    at test.js:1:1".to_string(),
                ),
            ],
        };
        let bytes = encode_message(&lines);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(lines));
        let empty = RendererToBrowser::ConsoleLines { lines: vec![] };
        let bytes = encode_message(&empty);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(empty));
    }

    #[test]
    fn devtools_inspector_messages_round_trip_through_versioned_bytes() {
        // Stage 18 (M9 stage 3): the inspector asks and answers
        // all survive the versioned envelope — including a root
        // ask (`node: None`), a clear-highlight (`node: None`),
        // a not-found node answer (empty tag), and a node with no
        // box (`geometry: None`).
        let children = BrowserToRenderer::DevToolsDomChildren {
            node: None,
            epoch: 3,
        };
        let bytes = encode_message(&children);
        assert_eq!(decode_message::<BrowserToRenderer>(&bytes), Ok(children));
        let inspect = BrowserToRenderer::DevToolsInspectNode {
            node: 42,
            epoch: 3,
        };
        let bytes = encode_message(&inspect);
        assert_eq!(decode_message::<BrowserToRenderer>(&bytes), Ok(inspect));
        let pick = BrowserToRenderer::DevToolsPickNode {
            x: 10.0,
            y: 20.0,
            epoch: 3,
        };
        let bytes = encode_message(&pick);
        assert_eq!(decode_message::<BrowserToRenderer>(&bytes), Ok(pick));
        let highlight = BrowserToRenderer::DevToolsHighlight {
            node: None,
            epoch: 3,
        };
        let bytes = encode_message(&highlight);
        assert_eq!(
            decode_message::<BrowserToRenderer>(&bytes),
            Ok(highlight)
        );
        let answer = RendererToBrowser::DevToolsChildren {
            epoch: 3,
            node: None,
            children: vec![
                DevToolsNodeSummary {
                    id: 1,
                    tag: "html".to_string(),
                    text: None,
                    has_children: true,
                },
                DevToolsNodeSummary {
                    id: 9,
                    tag: "#text".to_string(),
                    text: Some("hello".to_string()),
                    has_children: false,
                },
            ],
        };
        let bytes = encode_message(&answer);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(answer));
        let node = RendererToBrowser::DevToolsNode {
            epoch: 3,
            node: 42,
            tag: "div".to_string(),
            attrs: vec![("id".to_string(), "a".to_string())],
            computed: vec![("display".to_string(), "block".to_string())],
            geometry: Some(Box::new(DevToolsBox {
                x: 8.0,
                y: 16.0,
                width: 600.0,
                height: 120.0,
                margin: [
                    "80px".to_string(),
                    "auto".to_string(),
                    "80px".to_string(),
                    "auto".to_string(),
                ],
                padding: [
                    "32px".to_string(),
                    "32px".to_string(),
                    "32px".to_string(),
                    "32px".to_string(),
                ],
            })),
        };
        let bytes = encode_message(&node);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(node));
        let gone = RendererToBrowser::DevToolsNode {
            epoch: 3,
            node: 42,
            tag: String::new(),
            attrs: vec![],
            computed: vec![],
            geometry: None,
        };
        let bytes = encode_message(&gone);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(gone));
        let picked = RendererToBrowser::DevToolsPicked {
            epoch: 3,
            node: Some(42),
        };
        let bytes = encode_message(&picked);
        assert_eq!(decode_message::<RendererToBrowser>(&bytes), Ok(picked));
    }

    #[test]
    fn compositing_update_round_trips_through_versioned_bytes() {
        // M6 Stage 14 (Rec 9): the out-of-band compositing
        // update survives the versioned envelope, including an
        // empty update list (a no-op frame) and several elements.
        use pluto_css::Affine;
        let msg = RendererToBrowser::CompositingUpdate {
            epoch: 42,
            updates: vec![
                CompositedUpdate {
                    id: 7,
                    matrix: Affine::translate(30.0, 0.0),
                    opacity: 1.0,
                },
                CompositedUpdate {
                    id: 9,
                    matrix: Affine::rotate(45.0),
                    opacity: 0.5,
                },
            ],
        };
        let bytes = encode_message(&msg);
        let back: RendererToBrowser =
            decode_message(&bytes).expect("message decodes");
        assert_eq!(back, msg);
        let empty = RendererToBrowser::CompositingUpdate {
            epoch: 43,
            updates: vec![],
        };
        let back: RendererToBrowser =
            decode_message(&encode_message(&empty)).expect("empty update decodes");
        assert_eq!(back, empty);
    }

    #[test]
    fn browser_messages_round_trip_through_versioned_bytes() {
        let messages = [
            BrowserToRenderer::CommitNavigation {
                url: "https://example.com".to_string(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            },
            BrowserToRenderer::Resize {
                width: 800.0,
                height: 600.0,
            },
            // Stage 8 Rec 9 (v3): pointer / wheel / key carry
            // position, modifiers, timestamp, and (for wheel)
            // delta mode + gesture phase.
            BrowserToRenderer::InputEvent(InputEvent::Pointer {
                kind: PointerKind::Move,
                x: 10.0,
                y: 20.0,
                button: None,
                buttons: 0,
                modifiers: Modifiers::default(),
                timestamp_ms: 1234,
                pointer_id: 0,
            }),
            BrowserToRenderer::InputEvent(InputEvent::Pointer {
                kind: PointerKind::Down,
                x: 10.0,
                y: 20.0,
                button: Some(0),
                buttons: 1,
                modifiers: Modifiers {
                    shift: true,
                    ..Default::default()
                },
                timestamp_ms: 1235,
                pointer_id: 0,
            }),
            BrowserToRenderer::InputEvent(InputEvent::Wheel {
                x: 10.0,
                y: 20.0,
                dx: 0.0,
                dy: 3.0,
                delta_mode: DeltaMode::Line,
                phase: WheelPhase::Began,
                modifiers: Modifiers::default(),
                timestamp_ms: 1236,
            }),
            BrowserToRenderer::InputEvent(InputEvent::Key {
                key: "Enter".to_string(),
                code: "Enter".to_string(),
                pressed: true,
                repeat: false,
                modifiers: Modifiers::default(),
                timestamp_ms: 1237,
            }),
            BrowserToRenderer::Shutdown,
            // Stage 8 Rec 1: visibility rides the same channel; a
            // hidden tab gets no rendering opportunities.
            BrowserToRenderer::SetVisible { visible: false },
            BrowserToRenderer::SetVisible { visible: true },
            // Stage 8 Rec 9 (v3): window focus rides it too.
            BrowserToRenderer::SetFocused { focused: false },
            BrowserToRenderer::SetFocused { focused: true },
        ];
        for message in messages {
            let bytes = encode_message(&message);
            assert_eq!(
                u32::from_le_bytes(bytes[..4].try_into().unwrap()),
                PROTOCOL_VERSION
            );
            let back: BrowserToRenderer = decode_message(&bytes).expect("browser message decodes");
            assert_eq!(
                postcard::to_allocvec(&back).unwrap(),
                postcard::to_allocvec(&message).unwrap()
            );
        }
    }

    #[test]
    fn renderer_messages_round_trip_through_versioned_bytes() {
        let list = example_list();
        let mut resources = ResourceTable::new();
        let id = ResourceId(1);
        resources.insert(
            id,
            Resource::Image {
                width: 1,
                height: 1,
                rgba: vec![255, 0, 0, 255],
            },
        );
        let mut list_with_image = list.clone();
        list_with_image.push(DisplayItem::Image {
            resource: id,
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 16.0,
        });
        let submit = RendererToBrowser::SubmitDisplayList {
            epoch: 42,
            bytes: list_with_image.to_bytes(),
            resources,
        };
        let bytes = encode_message(&submit);
        let back: RendererToBrowser = decode_message(&bytes).expect("renderer message decodes");
        match back {
            RendererToBrowser::SubmitDisplayList {
                epoch,
                bytes,
                resources,
            } => {
                assert_eq!(epoch, 42);
                assert_eq!(DisplayList::from_bytes(&bytes).unwrap(), list_with_image);
                assert_eq!(resources.len(), 1);
                assert!(resources.get(id).is_some());
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
        // And the other renderer messages round-trip too.
        for message in [
            RendererToBrowser::DidCommit {
                url: "https://example.com/".to_string(),
                status: "Loaded https://example.com/".to_string(),
                replace: false,
            },
            RendererToBrowser::LoadFailed {
                url: "https://example.com".to_string(),
                reason: "connection refused".to_string(),
            },
            RendererToBrowser::TitleChanged {
                title: "Example Domain".to_string(),
            },
            RendererToBrowser::RendererCrashed {
                reason: "worker task ended".to_string(),
            },
            RendererToBrowser::RequestResource(ResourceRequest {
                request_id: 7,
                tab_id: 0,
                epoch: 3,
                url: "https://example.com/style.css".to_string(),
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
                body: None,
                destination: Destination::Stylesheet,
                priority: Priority::VeryHigh,
                mode: RequestMode::NoCors,
                credentials: CredentialsMode::SameOrigin,
                origin: "https://example.com".to_string(),
                top_level_site: "https://example.com".to_string(),
                upgrade_insecure_requests: false,
                integrity: None,
            }),
            RendererToBrowser::CancelResource { request_id: 7 },
            // Stage 8 Rec 9 (v3): the worker asks the shell to
            // scroll the viewport by the wheel remainder.
            RendererToBrowser::ScrollViewport { dx: 0.0, dy: 48.0 },
            // Stage 11 (M3): one decoded resource per frame,
            // out-of-band, plus load progress and the favicon.
            RendererToBrowser::ResourceReady {
                epoch: 3,
                resource_id: ResourceId(1),
                resource: Resource::Image {
                    width: 1,
                    height: 1,
                    rgba: vec![255, 0, 0, 255],
                },
            },
            RendererToBrowser::LoadProgress {
                received: 4096,
                total: Some(16384),
            },
            RendererToBrowser::LoadProgress {
                received: 4096,
                total: None,
            },
            RendererToBrowser::FaviconChanged {
                resource_id: Some(ResourceId(2)),
            },
            RendererToBrowser::FaviconChanged { resource_id: None },
        ] {
            let bytes = encode_message(&message);
            let _: RendererToBrowser = decode_message(&bytes).expect("renderer message decodes");
        }
    }

    #[test]
    fn phase3_messages_round_trip_through_versioned_bytes() {
        // Stage 16 Phase 3 (Rec 8/9/11/12): the history and
        // WebSocket messages survive the versioned envelope,
        // including a binary WS frame and a stateful traversal.
        for message in [
            BrowserToRenderer::HistorySync {
                length: 3,
                index: 1,
            },
            BrowserToRenderer::WebSocketEvent {
                socket_id: 7,
                event: WebSocketEvent::Message {
                    socket_id: 7,
                    data: vec![0, 1, 2, 255],
                    is_binary: true,
                },
            },
            BrowserToRenderer::WebSocketEvent {
                socket_id: 7,
                event: WebSocketEvent::Closed {
                    socket_id: 7,
                    code: Some(1000),
                    reason: "done".to_string(),
                    clean: true,
                },
            },
        ] {
            let back: BrowserToRenderer =
                decode_message(&encode_message(&message)).expect("message decodes");
            assert_eq!(back, message);
        }
        for message in [
            RendererToBrowser::HistoryPush {
                url: "https://example.com/app#view".to_string(),
                state_json: Some("{\"view\":\"home\"}".to_string()),
            },
            RendererToBrowser::HistoryReplace {
                url: "https://example.com/app#other".to_string(),
                state_json: None,
            },
            RendererToBrowser::HistoryGo { delta: -2 },
            RendererToBrowser::WebSocketOpen {
                socket_id: 7,
                url: "wss://example.com/chat".to_string(),
                protocols: vec!["chat".to_string()],
                origin: "https://example.com".to_string(),
            },
            RendererToBrowser::WebSocketSend {
                socket_id: 7,
                data: b"hello".to_vec(),
                is_binary: false,
            },
            RendererToBrowser::WebSocketClose {
                socket_id: 7,
                code: Some(1000),
                reason: String::new(),
            },
        ] {
            let back: RendererToBrowser =
                decode_message(&encode_message(&message)).expect("message decodes");
            assert_eq!(back, message);
        }
    }

    #[test]
    fn resource_events_round_trip_through_versioned_bytes() {
        for message in [
            BrowserToRenderer::ResourceEvent(ResourceEvent::ResponseHeaders {
                request_id: 7,
                status: 200,
                url: "https://example.com/style.css".to_string(),
                headers: vec![("content-type".to_string(), "text/css".to_string())],
                content_length: Some(42),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::BodyChunk {
                request_id: 7,
                bytes: b"body { color: red; }".to_vec(),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Done { request_id: 7 }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Error {
                request_id: 7,
                code: "request".to_string(),
                reason: "connection refused".to_string(),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Cancelled { request_id: 7 }),
        ] {
            let bytes = encode_message(&message);
            let back: BrowserToRenderer = decode_message(&bytes).expect("resource event decodes");
            assert_eq!(
                postcard::to_allocvec(&back).unwrap(),
                postcard::to_allocvec(&message).unwrap()
            );
        }
    }

    #[test]
    fn decode_rejects_version_three_bytes() {
        // A v3 message (version prefix 3) must fail loudly, not be
        // misread as v4: the transport went cross-process in the
        // Stage 9 Rec 9 bump, and a stale renderer must be rejected
        // at the version check, never silently accepted.
        let mut v3 = encode_message(&BrowserToRenderer::Shutdown);
        v3[0] = 3;
        assert_eq!(
            decode_message::<BrowserToRenderer>(&v3),
            Err(ProtocolError::UnsupportedVersion { found: 3 })
        );
    }

    #[test]
    fn decode_rejects_wrong_version_and_garbage() {
        let good = encode_message(&BrowserToRenderer::Shutdown);
        let mut wrong_version = good.clone();
        wrong_version[0] = PROTOCOL_VERSION as u8 + 1;
        assert_eq!(
            decode_message::<BrowserToRenderer>(&wrong_version),
            Err(ProtocolError::UnsupportedVersion {
                found: PROTOCOL_VERSION + 1
            })
        );
        assert_eq!(
            decode_message::<BrowserToRenderer>(&[]),
            Err(ProtocolError::Empty)
        );
        assert_eq!(
            decode_message::<BrowserToRenderer>(&[PROTOCOL_VERSION as u8, 0]),
            Err(ProtocolError::Empty)
        );
        assert!(matches!(
            decode_message::<BrowserToRenderer>(&good[..good.len() - 1]),
            Err(ProtocolError::Decode(_))
        ));
        let mut garbage = vec![PROTOCOL_VERSION as u8, 0, 0, 0];
        garbage.extend_from_slice(b"garbage");
        assert!(matches!(
            decode_message::<BrowserToRenderer>(&garbage),
            Err(ProtocolError::Decode(_))
        ));
    }

    #[tokio::test]
    async fn command_channel_carries_browser_messages() {
        let (tx, mut rx) = command_channel();
        tx.send(BrowserToRenderer::CommitNavigation {
            url: "https://example.com".to_string(),
            state_json: None,
            traversal: false,
            scroll: (0.0, 0.0),
        })
        .expect("send succeeds while the receiver lives");
        tx.send(BrowserToRenderer::Resize {
            width: 400.0,
            height: 300.0,
        })
        .expect("send succeeds");
        match rx.recv().await.expect("first message arrives") {
            BrowserToRenderer::CommitNavigation {
                url,
                state_json,
                traversal,
                scroll,
            } => {
                assert_eq!(url, "https://example.com");
                assert_eq!(state_json, None);
                assert!(!traversal);
                assert_eq!(scroll, (0.0, 0.0));
            }
            other => panic!("expected CommitNavigation, got {other:?}"),
        }
        match rx.recv().await.expect("second message arrives") {
            BrowserToRenderer::Resize { width, height } => {
                assert_eq!((width, height), (400.0, 300.0));
            }
            other => panic!("expected Resize, got {other:?}"),
        }
        // Dropping the receiver makes further sends fail cleanly.
        drop(rx);
        assert!(tx.send(BrowserToRenderer::Shutdown).is_err());
    }

    // ------------------------------------------------------------ Origin
    //
    // Stage 9 Rec 1: the origin is a parsed type, not a string.

    #[test]
    fn origin_default_port_normalization() {
        // An explicit default port is the same origin as no port.
        assert_eq!(
            Origin::from_url("https://example.com:443/"),
            Origin::from_url("https://example.com")
        );
        assert_eq!(
            Origin::from_url("http://example.com:80/"),
            Origin::from_url("http://example.com")
        );
        assert_eq!(
            Origin::from_url("ws://example.com:80/"),
            Origin::from_url("ws://example.com")
        );
        assert_eq!(
            Origin::from_url("wss://example.com:443/"),
            Origin::from_url("wss://example.com")
        );
        assert_eq!(
            Origin::from_url("ftp://example.com:21/"),
            Origin::from_url("ftp://example.com")
        );
        // A non-default port is a different origin.
        assert_ne!(
            Origin::from_url("https://example.com:8443/"),
            Origin::from_url("https://example.com")
        );
        assert_ne!(
            Origin::from_url("https://example.com:8443/"),
            Origin::from_url("https://example.com:443/")
        );
        // Scheme and host differ, and case does not matter.
        assert_ne!(
            Origin::from_url("http://example.com/"),
            Origin::from_url("https://example.com/")
        );
        assert_ne!(
            Origin::from_url("https://example.com/"),
            Origin::from_url("https://other.example/")
        );
        assert_eq!(
            Origin::from_url("HTTPS://EXAMPLE.COM:443/"),
            Origin::from_url("https://example.com")
        );
        // The normalized form keeps no default port.
        assert_eq!(
            Origin::from_url("https://example.com:443/").serialize(),
            "https://example.com"
        );
        assert_eq!(
            Origin::from_url("https://example.com:8443/").serialize(),
            "https://example.com:8443"
        );
    }

    #[test]
    fn origin_opaque_uniqueness_and_inequality() {
        // Every fresh opaque origin is a different identity...
        let a = Origin::opaque();
        let b = Origin::opaque();
        assert_ne!(a, b);
        assert!(!a.is_same_origin(&b));
        // ...even though both serialize to "null".
        assert_eq!(a.serialize(), "null");
        assert_eq!(b.serialize(), "null");
        // And an opaque origin is never same origin with a tuple.
        assert_ne!(a, Origin::from_url("https://example.com"));
        assert!(a.is_opaque());
        assert!(!Origin::from_url("https://example.com").is_opaque());
    }

    #[test]
    fn origin_same_origin_equality() {
        let a = Origin::from_url("https://example.com/a");
        let b = Origin::from_url("https://example.com:443/b");
        assert_eq!(a, b);
        assert!(a.is_same_origin(&b));
        assert!(!a.is_same_origin(&Origin::from_url("https://example.com:8443/c")));
        assert!(!a.is_same_origin(&Origin::from_url("http://example.com/d")));
        // The same opaque origin (cloned) is same origin with itself.
        let opaque = Origin::opaque();
        assert!(opaque.is_same_origin(&opaque.clone()));
    }

    #[test]
    fn origin_serialization_round_trip() {
        for url in [
            "https://example.com",
            "https://example.com:8443",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
        ] {
            let origin = Origin::from_url(url);
            let serialized = origin.serialize();
            assert_eq!(Origin::from_url(&serialized), origin, "{url}");
            assert_eq!(origin_of(url), serialized);
        }
        // Opaque origins serialize to "null" and never parse back to
        // the same identity: a fresh opaque origin is minted instead.
        let opaque = Origin::opaque();
        assert_eq!(opaque.serialize(), "null");
        assert_ne!(Origin::from_url("null"), opaque);
        assert!(Origin::from_url("null").is_opaque());
    }

    #[test]
    fn origin_of_serializer() {
        assert_eq!(origin_of("https://example.com:443/x"), "https://example.com");
        assert_eq!(
            origin_of("https://example.com:8443/x"),
            "https://example.com:8443"
        );
        // No authority (or unparseable) serializes to "null".
        assert_eq!(origin_of("data:text/html,<p>hi</p>"), "null");
        assert_eq!(origin_of("about:blank"), "null");
        assert_eq!(origin_of("not a url"), "null");
    }

    #[test]
    fn origin_about_blank_and_srcdoc_inheritance() {
        let creator = Origin::from_url("https://example.com/app");
        // `about:blank` created by a page inherits the creator.
        assert_eq!(
            Origin::for_document("about:blank", Some(&creator), false),
            creator
        );
        // `srcdoc` without sandbox is same-origin with its embedder.
        assert_eq!(
            Origin::for_document("about:srcdoc", Some(&creator), false),
            creator
        );
        // A top-level `about:blank` with no creator is opaque...
        let blank = Origin::for_document("about:blank", None, false);
        assert!(blank.is_opaque());
        // ...and two of them are never same origin.
        assert_ne!(
            blank,
            Origin::for_document("about:blank", None, false)
        );
        // A sandboxed document is opaque even with a creator and a
        // same-origin URL.
        let sandboxed = Origin::for_document("https://example.com/", Some(&creator), true);
        assert!(sandboxed.is_opaque());
        assert_ne!(sandboxed, creator);
    }

    #[test]
    fn origin_data_url_is_opaque_per_document() {
        let creator = Origin::from_url("https://example.com/");
        let a = Origin::for_document("data:text/html,<p>a</p>", Some(&creator), false);
        let b = Origin::for_document("data:text/html,<p>b</p>", Some(&creator), false);
        assert!(a.is_opaque());
        assert!(b.is_opaque());
        assert_ne!(a, b);
        assert_ne!(a, creator);
        assert_eq!(a.serialize(), "null");
    }

    #[test]
    fn origin_blob_derives_from_inner_url() {
        assert_eq!(
            Origin::from_url("blob:https://example.com/550e8400-e29b-41d4-a716-446655440000"),
            Origin::from_url("https://example.com/")
        );
        // An unparseable inner URL is opaque.
        assert!(Origin::from_url("blob:not a url").is_opaque());
    }

    #[test]
    fn origin_round_trips_through_postcard() {
        // The type is wire-safe for Rec 9's cross-process transport
        // (tuple origins survive exactly; an opaque origin keeps its
        // process-local id — see the `Origin::Opaque` docs).
        for origin in [
            Origin::from_url("https://example.com:8443/"),
            Origin::opaque(),
        ] {
            let bytes = postcard::to_allocvec(&origin).unwrap();
            let back: Origin = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, origin);
        }
    }
}
