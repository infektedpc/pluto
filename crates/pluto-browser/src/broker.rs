//! Browser-side network broker (Stage 13 Rec 11).
//!
//! The broker owns the `ResourceLoader` in the browser process
//! and answers renderer `RequestResource` messages over IPC by
//! calling `loader.fetch_stream` and forwarding the
//! `ResourceEvent`s. The renderer never opens a socket itself in
//! broker mode — so a landed chroot no longer silently discards
//! persistence writes into a dying tmpfs root, and a renderer
//! with `socket`/`connect` removed from its seccomp allow-list
//! still loads a real HTTPS page end-to-end.
//!
//! Two halves live here:
//!
//! * [`NetworkBroker`] — browser side: one `ResourceLoader`
//!   shared by every tab, serving `RequestResource` asks and
//!   persisting the cookie jar after each fetch (a login taken
//!   in survives a crash mid-session).
//! * [`FetchSource`] / [`BrokerHandle`] — renderer side: the
//!   fetch abstraction `page.rs` and `PlutoFetchBackend` use.
//!   `Direct` keeps the old in-process behaviour (tests, the
//!   in-process fallback worker); `Broker` routes every fetch
//!   through IPC to the browser-side broker. In broker mode the
//!   renderer skips profile writes (`security` / `cookies` /
//!   blocker refresh) — the broker owns them.

use pluto_ipc::{
    CredentialsMode, Destination, RendererToBrowser, ResourceEvent, ResourceRequest,
    WebSocketEvent,
};
use pluto_net::{NetError, ResourceLoader, Response};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

// ------------------------------------------------------------------ broker

/// A browser-owned network service: one `ResourceLoader` shared
/// by every tab, answering `RequestResource` over IPC.
pub struct NetworkBroker {
    loader: Arc<ResourceLoader>,
    /// In-flight fetch cancellation tokens, by request id. A
    /// std mutex, not a tokio one: `cancel` must be callable from
    /// the shell's synchronous `update()` (no await there), and
    /// the lock is never held across an await.
    pending: std::sync::Mutex<HashMap<u64, CancellationToken>>,
    /// DevTools network log (Stage 20, M9 stage 5): per-tab
    /// request records for the waterfall. A std Mutex for the
    /// same reason as `pending` — the shell's synchronous
    /// `view()` reads it while rendering the Network tab.
    network: std::sync::Mutex<NetworkLog>,
}

impl NetworkBroker {
    pub fn new(loader: Arc<ResourceLoader>) -> Self {
        Self {
            loader,
            pending: std::sync::Mutex::new(HashMap::new()),
            network: std::sync::Mutex::new(NetworkLog::default()),
        }
    }

    /// The DevTools network records for one tab (Stage 20, M9
    /// stage 5): finished rows (oldest first, capped at
    /// [`NETWORK_RECORDS_CAP`]) plus in-flight rows, sorted by
    /// start — a snapshot for the panel to render. Cloned; the
    /// panel reads it on every frame while the Network tab is
    /// open.
    pub fn network_records(&self, tab_id: u64) -> Vec<NetworkRecord> {
        self.network.lock().unwrap().records_for(tab_id)
    }

    pub fn loader(&self) -> &Arc<ResourceLoader> {
        &self.loader
    }

    /// Serve one resource request: run the fetch on the shared
    /// loader and forward every `ResourceEvent` to `events`.
    /// The receiver closes when the fetch completes (or errors);
    /// the caller maps the channel end to the IPC stream.
    /// Afterwards the cookie jar is persisted — a `Set-Cookie`
    /// taken in during this fetch survives a crash.
    pub async fn serve(&self, request: ResourceRequest, events: mpsc::Sender<ResourceEvent>) {
        eprintln!("[pluto-broker] serve start id={} url={}", request.request_id, request.url);
        let cancel = CancellationToken::new();
        let request_id = request.request_id;
        self.pending
            .lock()
            .unwrap()
            .insert(request_id, cancel.clone());
        // Stage 20 (M9 stage 5): the record this task updates —
        // its own, or the primary's when this ask is a duplicate
        // waiter for an identical in-flight fetch (the scheduler
        // dedups globally, so the second task shares the first
        // task's row instead of opening a second one for the
        // same bytes). A Document ask always opens a fresh record
        // and clears the tab's log first: the list clears on
        // navigation, and the clear rides the document request —
        // not DidCommit — so the document row itself survives.
        let record_key = self.network.lock().unwrap().start(&request);
        let mut stream = self.loader.fetch_stream(request, cancel).await;
        while let Some(event) = stream.recv().await {
            // Record BEFORE forwarding: a forwarding failure
            // breaks the loop, and the terminal event must
            // already be in the log. Recording is a brief std
            // Mutex hold with no await — the forward path is
            // never blocked on it.
            self.network
                .lock()
                .unwrap()
                .record_event(record_key, &event);
            if events.send(event).await.is_err() {
                break;
            }
        }
        // The stream ended without a terminal event (the forward
        // broke first, so the fetch was aborted mid-flight): close
        // the record as cancelled instead of leaving it in flight
        // forever. A no-op when the record already finished, or
        // was cleared by a navigation.
        self.network.lock().unwrap().finish_orphan(record_key);
        self.pending.lock().unwrap().remove(&request_id);
        crate::cookies::save_from(&self.loader);
    }

    /// Cancel an in-flight (or queued) fetch by request id
    /// (`RendererToBrowser::CancelResource`). Synchronous by
    /// design (see the `pending` field note).
    pub fn cancel(&self, request_id: u64) {
        if let Some(token) = self.pending.lock().unwrap().remove(&request_id) {
            token.cancel();
        }
    }

    /// Serve one WebSocket (Stage 16 Phase 3, Rec 11): run the
    /// handshake on the shared loader and pump the socket — the
    /// renderer never opens a socket itself. `commands` carries
    /// the shell-forwarded `Send` / `Close` asks; socket lifecycle
    /// (open / message / close / error) goes to `events`, with
    /// exactly one terminal event. Returns when the socket ends
    /// (close frame, error, or the command channel closing — a
    /// dropped command channel means the tab went away, so the
    /// socket is closed, not left dangling).
    pub async fn serve_websocket(
        &self,
        socket_id: u64,
        url: String,
        protocols: Vec<String>,
        origin: String,
        mut commands: mpsc::Receiver<WsCommand>,
        events: mpsc::Sender<WebSocketEvent>,
    ) {
        // The handshake runs through the loader's gated path
        // (HSTS / mixed-content / blocker / CSP `connect-src` /
        // `Origin` header) with a `WebSocket` destination — the
        // same gates as `fetch()`, never a raw socket.
        let request = ResourceRequest {
            request_id: socket_id,
            tab_id: 0,
            epoch: 0,
            url: url.clone(),
            destination: Destination::WebSocket,
            priority: pluto_ipc::Priority::Medium,
            method: "GET".to_string(),
            headers: Vec::new(),
            is_imageset: false,
            mode: pluto_ipc::RequestMode::Cors,
            credentials: pluto_ipc::CredentialsMode::Include,
            origin: origin.clone(),
            top_level_site: origin,
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        };
        let cancel = CancellationToken::new();
        let connected = self
            .loader
            .connect_websocket(request, protocols, cancel.clone())
            .await;
        let (stream, protocol) = match connected {
            Ok(pair) => pair,
            Err(e) => {
                let _ = events
                    .send(WebSocketEvent::Error {
                        socket_id,
                        message: e.to_string(),
                    })
                    .await;
                return;
            }
        };
        let (mut write, mut read) = stream.split();
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;
        use tokio_tungstenite::tungstenite::protocol::CloseFrame;
        use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
        let _ = events
            .send(WebSocketEvent::Opened {
                socket_id,
                protocol,
            })
            .await;
        loop {
            tokio::select! {
                cmd = commands.recv() => {
                    let Some(cmd) = cmd else {
                        // Tab gone: close the socket and end.
                        let _ = write.send(Message::Close(None)).await;
                        break;
                    };
                    match cmd {
                        WsCommand::Send { data, is_binary } => {
                            let msg = if is_binary {
                                Message::Binary(data.into())
                            } else {
                                match String::from_utf8(data) {
                                    Ok(text) => Message::Text(text.into()),
                                    Err(_) => continue, // Not UTF-8 text: drop, stay open.
                                }
                            };
                            if write.send(msg).await.is_err() {
                                break;
                            }
                        }
                        WsCommand::Close { code, reason } => {
                            let frame = CloseFrame {
                                code: code
                                    .map(CloseCode::from)
                                    .unwrap_or(CloseCode::Normal),
                                reason: reason.into(),
                            };
                            let _ = write.send(Message::Close(Some(frame))).await;
                            let _ = write.close().await;
                            // The peer's close reply (or the socket
                            // end) delivers the terminal `Closed`
                            // event below; if the socket is already
                            // gone the read arm ends the pump.
                        }
                    }
                }
                msg = read.next() => {
                    let Some(msg) = msg else {
                        let _ = events
                            .send(WebSocketEvent::Closed {
                                socket_id,
                                code: Some(1006),
                                reason: String::new(),
                                clean: false,
                            })
                            .await;
                        break;
                    };
                    match msg {
                        Ok(Message::Text(text)) => {
                            if events
                                .send(WebSocketEvent::Message {
                                    socket_id,
                                    // `Utf8Bytes`: no `into_bytes` —
                                    // borrow the bytes and copy them.
                                    data: text.as_bytes().to_vec(),
                                    is_binary: false,
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok(Message::Binary(data)) => {
                            if events
                                .send(WebSocketEvent::Message {
                                    socket_id,
                                    data: data.into(),
                                    is_binary: true,
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok(Message::Close(frame)) => {
                            let (code, reason) = match frame {
                                Some(f) => (Some(u16::from(f.code)), f.reason.to_string()),
                                None => (Some(1005), String::new()),
                            };
                            let _ = events
                                .send(WebSocketEvent::Closed {
                                    socket_id,
                                    code,
                                    reason,
                                    clean: true,
                                })
                                .await;
                            break;
                        }
                        // Ping: tungstenite queues the Pong in its
                        // read path (RFC 6455 §5.5.2) — polling
                        // the stream is the reply. Pong / raw
                        // frames are not application messages.
                        Ok(Message::Ping(_)) => {}
                        Ok(Message::Pong(_)) => {}
                        Ok(Message::Frame(_)) => {}
                        Err(e) => {
                            let _ = events
                                .send(WebSocketEvent::Closed {
                                    socket_id,
                                    code: Some(1006),
                                    reason: e.to_string(),
                                    clean: false,
                                })
                                .await;
                            break;
                        }
                    }
                }
            }
        }
        cancel.cancel();
    }
}

// ------------------------------------------------------------------ network log

/// Cap for the DevTools network waterfall (Stage 20, M9 stage 5):
/// the last 200 finished requests per tab, oldest evicted. The
/// cap is a Joel question (see the Stage 20 note in
/// ARCHITECTURE.md) — 200 is the interim default.
pub const NETWORK_RECORDS_CAP: usize = 200;

/// What a network request ended as (Stage 20, M9 stage 5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkOutcome {
    /// Still in flight (no terminal event yet).
    InFlight,
    /// Body complete.
    Done,
    /// The fetch failed: machine-readable `code` (e.g.
    /// `blocked_by_filter`, `cors_blocked`, `mixed_content`,
    /// `timeout`) plus the human-readable `reason` — a blocked
    /// request shows *why* it never hit the network.
    Error { code: String, reason: String },
    /// Cancelled (renderer `CancelResource`, navigation epoch
    /// bump, tab close, timeout, or the broker stopping its
    /// forward when the tab went away).
    Cancelled { reason: String },
}

/// One request's DevTools network row (Stage 20, M9 stage 5):
/// everything the waterfall needs, built shell-side in
/// `NetworkBroker::serve` from the `ResourceEvent` stream —
/// no IPC, no renderer involvement. Timing is broker-side
/// `Instant`s: `started` at `serve` entry (so scheduler queue
/// time is inside the waiting phase, honestly), `ttfb` when
/// `ResponseHeaders` arrives, `ended` on the terminal event.
/// DNS / connect / TLS are NOT split out — hyper hides them,
/// and the broker cannot see them at all; the waiting phase
/// (start → TTFB) is one block. A cache hit is likewise not
/// flagged: the broker cannot tell a fresh-cache serve from a
/// fast network fetch (same event shape) — its TTFB ≈ 0 is the
/// tell. Both are honest limits of the broker-side record,
/// noted in the Stage 20 ARCHITECTURE.md note.
#[derive(Debug, Clone)]
pub struct NetworkRecord {
    /// The ask's id (unique per tab — each tab's worker numbers
    /// its own asks from 1).
    pub request_id: u64,
    pub tab_id: u64,
    pub method: String,
    pub url: String,
    /// The URL the bytes actually came from, after redirects
    /// (`ResponseHeaders.url`); `None` until headers arrive.
    pub final_url: Option<String>,
    pub destination: Destination,
    pub status: Option<u16>,
    pub content_length: Option<u64>,
    pub bytes_received: u64,
    /// Script-set request headers, SANITISED at record time:
    /// `Cookie` / `Authorization` values never reach the panel
    /// (the name is kept with the value `(redacted)` — the
    /// report's §6.5 rule, applied at the source, not hidden
    /// in the view).
    pub request_headers: Vec<(String, String)>,
    /// Response headers, sanitised the same way (`Set-Cookie`
    /// is already stripped upstream; this is defence in depth).
    pub response_headers: Vec<(String, String)>,
    pub started: Instant,
    /// Headers arrival minus `started`.
    pub ttfb: Option<Duration>,
    /// Terminal event minus `started`.
    pub ended: Option<Duration>,
    pub outcome: NetworkOutcome,
    /// The scheduler dedup key (not rendered): lets a duplicate
    /// waiter find the record it aliases.
    dedup: (String, Destination, CredentialsMode, String),
}

impl NetworkRecord {
    fn new(request: &ResourceRequest) -> Self {
        Self {
            request_id: request.request_id,
            tab_id: request.tab_id,
            method: request.method.clone(),
            url: request.url.clone(),
            final_url: None,
            destination: request.destination,
            status: None,
            content_length: None,
            bytes_received: 0,
            request_headers: sanitize_headers(&request.headers),
            response_headers: Vec::new(),
            started: Instant::now(),
            ttfb: None,
            ended: None,
            outcome: NetworkOutcome::InFlight,
            dedup: (
                request.url.clone(),
                request.destination,
                request.credentials,
                request.top_level_site.clone(),
            ),
        }
    }

    /// Total duration: to the terminal event when ended, to now
    /// while in flight (the waterfall's provisional end).
    pub fn total(&self) -> Duration {
        self.ended.unwrap_or_else(|| self.started.elapsed())
    }
}

/// Strip sensitive header values at the source (Stage 20, M9
/// stage 5 — the report's §6.5 rule): `Cookie`, `Set-Cookie`
/// and `Authorization` values never reach the panel. The header
/// NAME is kept with the value `(redacted)`, so the detail view
/// shows the header exists without leaking the secret.
/// (`Set-Cookie` is already stripped upstream before the
/// renderer sees it — this is defence in depth, so no path
/// that ever leaks one into the event stream can show it.)
/// Case-insensitive name match; everything else passes through
/// verbatim.
fn sanitize_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            if name.eq_ignore_ascii_case("cookie")
                || name.eq_ignore_ascii_case("set-cookie")
                || name.eq_ignore_ascii_case("authorization")
            {
                (name.clone(), "(redacted)".to_string())
            } else {
                (name.clone(), value.clone())
            }
        })
        .collect()
}

/// The per-tab network record store (Stage 20, M9 stage 5):
/// in-flight records by (tab, request) plus a finished ring
/// buffer per tab, capped at [`NETWORK_RECORDS_CAP`]. Lives
/// behind a std Mutex on `NetworkBroker` (same reason as
/// `pending`: the shell's synchronous `view()` reads it
/// without awaiting).
#[derive(Debug, Default)]
struct NetworkLog {
    in_flight: HashMap<(u64, u64), NetworkRecord>,
    finished: HashMap<u64, VecDeque<NetworkRecord>>,
}

impl NetworkLog {
    /// Open a record for one ask; returns the key the serving
    /// task updates for the rest of the fetch. A Document ask
    /// always opens a fresh record and clears the tab's log
    /// first (the list clears on navigation). Any other ask
    /// whose scheduler dedup key matches an in-flight record
    /// (same URL + destination + credentials + top-level site,
    /// any tab — the scheduler dedups globally) is a duplicate
    /// waiter: it aliases the first task's record instead of
    /// opening a second row for the same bytes, and the key
    /// returned is the primary's.
    fn start(&mut self, request: &ResourceRequest) -> (u64, u64) {
        let tab_id = request.tab_id;
        if request.destination == Destination::Document {
            self.clear_tab(tab_id);
        } else if let Some((&key, _)) = self.in_flight.iter().find(|(_, r)| {
            r.dedup.0 == request.url
                && r.dedup.1 == request.destination
                && r.dedup.2 == request.credentials
                && r.dedup.3 == request.top_level_site
        }) {
            return key;
        }
        let key = (tab_id, request.request_id);
        self.in_flight.insert(key, NetworkRecord::new(request));
        key
    }

    /// Fold one event into the record at `key`. Unknown keys
    /// are ignored, never created: the record finished already
    /// (the other waiter got the terminal event first), or a
    /// navigation cleared it — a late event must never
    /// resurrect a row.
    fn record_event(&mut self, key: (u64, u64), event: &ResourceEvent) {
        let Some(record) = self.in_flight.get_mut(&key) else {
            return;
        };
        let mut terminal = false;
        match event {
            ResourceEvent::ResponseHeaders {
                status,
                url,
                headers,
                content_length,
                ..
            } => {
                if record.ttfb.is_none() {
                    record.ttfb = Some(record.started.elapsed());
                }
                record.status = Some(*status);
                record.final_url = Some(url.clone());
                record.response_headers = sanitize_headers(headers);
                record.content_length = *content_length;
            }
            ResourceEvent::BodyChunk { bytes, .. } => {
                record.bytes_received += bytes.len() as u64;
            }
            ResourceEvent::Done { .. } => {
                record.ended = Some(record.started.elapsed());
                record.outcome = NetworkOutcome::Done;
                terminal = true;
            }
            ResourceEvent::Error { code, reason, .. } => {
                record.ended = Some(record.started.elapsed());
                record.outcome = NetworkOutcome::Error {
                    code: code.clone(),
                    reason: reason.clone(),
                };
                terminal = true;
            }
            ResourceEvent::Cancelled { .. } => {
                record.ended = Some(record.started.elapsed());
                record.outcome = NetworkOutcome::Cancelled {
                    reason: "cancelled".to_string(),
                };
                terminal = true;
            }
        }
        if terminal {
            self.finish(key);
        }
    }

    /// Close a record that is still in flight (the stream ended
    /// without a terminal event — the forward broke first). A
    /// no-op when the record already finished or was cleared.
    fn finish_orphan(&mut self, key: (u64, u64)) {
        let Some(mut record) = self.in_flight.remove(&key) else {
            return;
        };
        record.ended = Some(record.started.elapsed());
        record.outcome = NetworkOutcome::Cancelled {
            reason: "forwarding stopped (tab closed)".to_string(),
        };
        self.push_finished(record);
    }

    /// Move a finished record into its tab's ring buffer,
    /// evicting the oldest past the cap.
    fn finish(&mut self, key: (u64, u64)) {
        let Some(record) = self.in_flight.remove(&key) else {
            return;
        };
        self.push_finished(record);
    }

    fn push_finished(&mut self, record: NetworkRecord) {
        let finished = self.finished.entry(record.tab_id).or_default();
        finished.push_back(record);
        while finished.len() > NETWORK_RECORDS_CAP {
            finished.pop_front();
        }
    }

    /// Finished rows (oldest first) plus in-flight rows for one
    /// tab, sorted by start — the panel's snapshot.
    fn records_for(&self, tab_id: u64) -> Vec<NetworkRecord> {
        let mut out: Vec<NetworkRecord> = self
            .finished
            .get(&tab_id)
            .map(|d| d.iter().cloned().collect())
            .unwrap_or_default();
        out.extend(
            self.in_flight
                .values()
                .filter(|r| r.tab_id == tab_id)
                .cloned(),
        );
        out.sort_by_key(|r| r.started);
        out
    }

    /// Drop a tab's whole log (in-flight + finished) — the
    /// navigation clear, driven by the tab's Document ask.
    fn clear_tab(&mut self, tab_id: u64) {
        self.in_flight.retain(|(t, _), _| *t != tab_id);
        self.finished.remove(&tab_id);
    }
}

/// Commands the shell forwards to a browser-side WebSocket pump
/// (Stage 16 Phase 3, Rec 11): one message, or the close
/// handshake. The pump owns the socket; the shell only routes.
#[derive(Debug, Clone)]
pub enum WsCommand {
    Send { data: Vec<u8>, is_binary: bool },
    Close { code: Option<u16>, reason: String },
}

// ------------------------------------------------------------------ handle

/// The renderer side of the broker: sends `RequestResource` over
/// the tab thread's event channel (which the transport loop
/// forwards to the browser) and delivers inbound
/// `ResourceEvent`s to the waiting fetch by request id.
pub struct BrokerHandle {
    pending: Mutex<HashMap<u64, mpsc::Sender<ResourceEvent>>>,
    out: mpsc::Sender<RendererToBrowser>,
    next_id: std::sync::atomic::AtomicU64,
}

/// Short kind tag for a `ResourceEvent`, for the `[pluto-worker]` phase log.
fn event_kind(event: &ResourceEvent) -> &'static str {
    match event {
        ResourceEvent::ResponseHeaders { .. } => "headers",
        ResourceEvent::BodyChunk { .. } => "chunk",
        ResourceEvent::Done { .. } => "done",
        ResourceEvent::Error { .. } => "error",
        ResourceEvent::Cancelled { .. } => "cancelled",
    }
}

impl BrokerHandle {
    pub fn new(out: mpsc::Sender<RendererToBrowser>) -> Arc<Self> {
        Arc::new(Self {
            pending: Mutex::new(HashMap::new()),
            out,
            next_id: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Deliver one inbound `ResourceEvent` to its waiting fetch
    /// (no-op when the id is unknown — a late event for a
    /// cancelled fetch).
    pub async fn deliver(&self, event: ResourceEvent) {
        let request_id = match &event {
            ResourceEvent::ResponseHeaders { request_id, .. }
            | ResourceEvent::BodyChunk { request_id, .. }
            | ResourceEvent::Done { request_id }
            | ResourceEvent::Error { request_id, .. }
            | ResourceEvent::Cancelled { request_id } => *request_id,
        };
        let Some(sender) = self.pending.lock().await.get(&request_id).cloned() else {
            // [pluto-worker] phase log: an event the shell
            // forwarded but no fetch is waiting on — a dropped
            // Done here is a load that never completes.
            eprintln!(
                "[pluto-worker] broker deliver dropped id={} kind={}",
                request_id,
                event_kind(&event)
            );
            return;
        };
        if sender.send(event).await.is_err() {
            self.pending.lock().await.remove(&request_id);
        }
    }

    /// Stream one request through the broker (Stage 16 Phase 3,
    /// Rec 12): send the ask and return the event receiver,
    /// uncollapsed — headers, then chunks, then exactly one
    /// terminal event. No byte-cap enforcement here (the 10 MiB
    /// whole-response cap lives in `fetch`, not in the stream):
    /// the service's `Stream` caps (1 GiB / 600 s) bound the
    /// stream, and the realm buffers only what the page actually
    /// reads. The second return value is the broker request id,
    /// for [`BrokerHandle::cancel_stream`].
    pub async fn fetch_stream(
        self: &Arc<Self>,
        mut request: ResourceRequest,
        cancel: CancellationToken,
    ) -> (mpsc::Receiver<ResourceEvent>, u64) {
        let request_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        request.request_id = request_id;
        let (tx, rx) = mpsc::channel::<ResourceEvent>(64);
        self.pending.lock().await.insert(request_id, tx);
        let sent = self
            .out
            .send(RendererToBrowser::RequestResource(request.clone()))
            .await
            .is_ok();
        if !sent {
            self.pending.lock().await.remove(&request_id);
            return (rx, request_id);
        }
        // The caller owns cancellation: it watches `cancel` and
        // calls `cancel_stream` when it fires (the token is the
        // same one `PlutoFetchBackend` stores per fetch id, so
        // `controller.abort()` reaches the broker).
        let _ = cancel;
        (rx, request_id)
    }

    /// Cancel a streaming fetch by broker request id (Stage 16
    /// Phase 3, Rec 12): send `CancelResource` and forget the
    /// pending entry. No-op when the fetch already ended.
    pub async fn cancel_stream(&self, request_id: u64) {
        let _ = self
            .out
            .send(RendererToBrowser::CancelResource { request_id })
            .await;
        self.pending.lock().await.remove(&request_id);
    }

    /// Forget a streaming fetch's pending entry (Stage 16 Phase 3,
    /// Rec 12): the forwarding task calls this when the stream
    /// ends, so a completed stream does not leak its sender in
    /// `pending` (no more events arrive after the terminal one,
    /// so nothing else would ever remove it).
    pub async fn finish_stream(&self, request_id: u64) {
        self.pending.lock().await.remove(&request_id);
    }

    /// Fetch one request through the broker: send the ask, then
    /// assemble the `Response` from the event stream. The
    /// per-destination byte cap is enforced here (the broker
    /// streams without a cap — the renderer is the one that must
    /// not buffer unboundedly). Cancellation sends
    /// `CancelResource` and returns `NetError::Cancelled`.
    pub async fn fetch(
        self: &Arc<Self>,
        mut request: ResourceRequest,
        cancel: CancellationToken,
    ) -> Result<Response, NetError> {
        let request_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        request.request_id = request_id;
        let cap = pluto_net::service::byte_cap(request.destination);
        let (tx, mut rx) = mpsc::channel::<ResourceEvent>(64);
        self.pending.lock().await.insert(request_id, tx);
        let sent = self
            .out
            .send(RendererToBrowser::RequestResource(request.clone()))
            .await
            .is_ok();
        if !sent {
            self.pending.lock().await.remove(&request_id);
            return Err(NetError::Request("broker channel closed".to_string()));
        }
        eprintln!("[pluto-worker] broker fetch start id={} url={}", request_id, request.url);
        let mut status = http::StatusCode::OK;
        let mut headers = http::header::HeaderMap::new();
        let mut body: Vec<u8> = Vec::new();
        let mut final_url: Option<http::Uri> = None;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    eprintln!("[pluto-worker] broker fetch cancelled id={}", request_id);
                    let _ = self.out.send(RendererToBrowser::CancelResource { request_id }).await;
                    self.pending.lock().await.remove(&request_id);
                    return Err(NetError::Cancelled);
                }
                event = rx.recv() => {
                    let Some(event) = event else {
                        eprintln!("[pluto-worker] broker stream ended id={} body_bytes={}", request_id, body.len());
                        self.pending.lock().await.remove(&request_id);
                        return Err(NetError::Request("broker stream ended".to_string()));
                    };
                    match event {
                        ResourceEvent::ResponseHeaders { status: s, url, headers: h, .. } => {
                            status = http::StatusCode::from_u16(s)
                                .unwrap_or(http::StatusCode::BAD_GATEWAY);
                            final_url = url.parse().ok();
                            for (name, value) in h {
                                if let (Ok(n), Ok(v)) = (
                                    http::header::HeaderName::from_bytes(name.as_bytes()),
                                    http::header::HeaderValue::from_str(&value),
                                ) {
                                    headers.append(n, v);
                                }
                            }
                        }
                        ResourceEvent::BodyChunk { bytes, .. } => {
                            body.extend_from_slice(&bytes);
                            eprintln!("[pluto-worker] broker chunk id={} bytes={} total={}", request_id, bytes.len(), body.len());
                            if body.len() > cap {
                                let _ = self.out.send(RendererToBrowser::CancelResource { request_id }).await;
                                self.pending.lock().await.remove(&request_id);
                                return Err(NetError::TooLarge { destination: request.destination, cap });
                            }
                        }
                        ResourceEvent::Done { .. } => {
                            eprintln!("[pluto-worker] broker done id={} total_bytes={}", request_id, body.len());
                            break;
                        }
                        ResourceEvent::Error { code, reason, .. } => {
                            eprintln!("[pluto-worker] broker error id={} code={} reason={}", request_id, code, reason);
                            self.pending.lock().await.remove(&request_id);
                            return Err(NetError::Request(format!("{code}: {reason}")));
                        }
                        ResourceEvent::Cancelled { .. } => {
                            self.pending.lock().await.remove(&request_id);
                            return Err(NetError::Cancelled);
                        }
                    }
                }
            }
        }
        self.pending.lock().await.remove(&request_id);
        Ok(Response {
            status,
            headers,
            body,
            final_url: final_url.unwrap_or_else(|| {
                request
                    .url
                    .parse()
                    .unwrap_or_else(|_| http::Uri::from_static("http://127.0.0.1/"))
            }),
        })
    }
}

// ------------------------------------------------------------------ source

/// Where a renderer-side fetch goes (Stage 13 Rec 11): straight
/// at a `ResourceLoader` (`Direct`, in-process worker + tests)
/// or through IPC to the browser-side broker (`Broker`,
/// out-of-process child). `page.rs` and `PlutoFetchBackend`
/// only ever see this enum — the fetch path is one call either
/// way, so a landed chroot cannot silently lose persistence.
#[derive(Clone)]
pub enum FetchSource {
    Direct(Arc<ResourceLoader>),
    Broker(Arc<BrokerHandle>),
}

impl FetchSource {
    /// Fetch a whole response (no progress reporting).
    pub async fn fetch_with(
        &self,
        request: ResourceRequest,
        cancel: CancellationToken,
    ) -> Result<Response, NetError> {
        match self {
            FetchSource::Direct(loader) => loader.fetch_with(request, cancel).await,
            FetchSource::Broker(handle) => handle.fetch(request, cancel).await,
        }
    }

    /// Fetch with byte progress (document loads drive the shell's
    /// progress bar through this; broker mode reports the same
    /// numbers as chunks arrive).
    pub async fn fetch_with_progress(
        &self,
        request: ResourceRequest,
        cancel: CancellationToken,
        on_progress: impl Fn(u64, Option<u64>) + Send + 'static,
    ) -> Result<Response, NetError> {
        match self {
            FetchSource::Direct(loader) => {
                loader
                    .fetch_with_progress(request, cancel, on_progress)
                    .await
            }
            FetchSource::Broker(handle) => {
                // The broker streams `ResourceEvent`s without a
                // progress hook, so progress is derived from the
                // assembled body: report (0, total) on headers
                // and (received, total) per chunk is not visible
                // here — the caller gets one (len, total) report
                // at the end. Progress-bar fidelity in broker
                // mode is a follow-up, not a correctness gap:
                // the bytes are the same either way.
                let response = handle.fetch(request, cancel).await?;
                let total = response
                    .headers
                    .get(http::header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok());
                on_progress(response.body.len() as u64, total);
                Ok(response)
            }
        }
    }

    /// Warm a connection (`<link rel="preconnect">`): a no-op in
    /// broker mode — the broker's shared client pool already
    /// reuses connections, and the renderer must not open any.
    pub async fn preconnect(&self, url: &str) -> bool {
        match self {
            FetchSource::Direct(loader) => loader.preconnect(url).await,
            FetchSource::Broker(_) => false,
        }
    }

    /// DNS-only warm (`dns-prefetch`): same no-op in broker mode.
    pub async fn dns_prefetch(&self, url: &str) -> bool {
        match self {
            FetchSource::Direct(loader) => loader.dns_prefetch(url).await,
            FetchSource::Broker(_) => false,
        }
    }

    /// The current document URL the loader holds as the referrer
    /// source (`None` in broker mode — the referrer context lives
    /// in the broker; the shell sets it on commit).
    pub fn referrer_document_url(&self) -> Option<String> {
        match self {
            FetchSource::Direct(loader) => loader
                .referrer_context()
                .lock()
                .unwrap()
                .document_url
                .clone(),
            FetchSource::Broker(_) => None,
        }
    }

    /// Point the referrer context at a document URL (no-op in
    /// broker mode — the shell owns the broker's context).
    pub fn set_referrer_document(&self, url: &str) {
        if let FetchSource::Direct(loader) = self {
            loader.set_referrer_document(url);
        }
    }

    /// Apply a `<meta name="referrer">` policy (no-op in broker
    /// mode — the policy lives in the broker; threading it over
    /// IPC is a follow-up).
    pub fn set_referrer_policy(&self, policy: pluto_net::ReferrerPolicy) {
        if let FetchSource::Direct(loader) = self {
            loader.set_referrer_policy(policy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn request(
        id: u64,
        tab_id: u64,
        url: &str,
        destination: Destination,
    ) -> ResourceRequest {
        let mut req = ResourceRequest::navigate(id, tab_id, 0, url);
        req.destination = destination;
        req
    }

    fn headers_event(
        id: u64,
        status: u16,
        headers: Vec<(String, String)>,
    ) -> ResourceEvent {
        ResourceEvent::ResponseHeaders {
            request_id: id,
            status,
            url: "https://example.com/app.js".to_string(),
            headers,
            content_length: Some(5),
        }
    }

    #[test]
    fn log_records_timing_start_ttfb_end() {
        // Stage 20 (M9 stage 5): start → headers (TTFB) →
        // chunks (bytes) → Done (end + outcome), in order.
        let mut log = NetworkLog::default();
        let req = request(1, 7, "https://example.com/app.js", Destination::Script);
        let key = log.start(&req);
        assert_eq!(key, (7, 1));
        assert_eq!(log.records_for(7).len(), 1);
        assert!(matches!(
            log.records_for(7)[0].outcome,
            NetworkOutcome::InFlight
        ));
        log.record_event(key, &headers_event(1, 200, vec![]));
        let row = &log.records_for(7)[0];
        assert!(row.ttfb.is_some(), "headers set TTFB");
        assert_eq!(row.status, Some(200));
        assert_eq!(row.content_length, Some(5));
        assert!(row.ended.is_none());
        log.record_event(
            key,
            &ResourceEvent::BodyChunk {
                request_id: 1,
                bytes: b"hello".to_vec(),
            },
        );
        assert_eq!(log.records_for(7)[0].bytes_received, 5);
        log.record_event(key, &ResourceEvent::Done { request_id: 1 });
        let row = &log.records_for(7)[0];
        assert!(row.ended.is_some(), "Done sets the end");
        assert!(row.total() >= row.ttfb.unwrap(), "total covers TTFB");
        assert_eq!(row.outcome, NetworkOutcome::Done);
    }

    #[test]
    fn log_caps_finished_at_200_oldest_evicted() {
        // Stage 20 (M9 stage 5): the ring buffer caps at 200
        // finished rows per tab, oldest first.
        let mut log = NetworkLog::default();
        for i in 0..NETWORK_RECORDS_CAP + 5 {
            let req = request(
                i as u64 + 1,
                3,
                &format!("https://example.com/r{i}.js"),
                Destination::Script,
            );
            let key = log.start(&req);
            log.record_event(key, &ResourceEvent::Done { request_id: i as u64 + 1 });
        }
        let rows = log.records_for(3);
        assert_eq!(rows.len(), NETWORK_RECORDS_CAP);
        assert_eq!(
            rows.first().unwrap().url,
            "https://example.com/r5.js",
            "the oldest five are evicted first"
        );
        assert_eq!(
            rows.last().unwrap().url,
            format!("https://example.com/r{}.js", NETWORK_RECORDS_CAP + 4)
        );
    }

    #[test]
    fn log_document_request_clears_the_tab() {
        // Stage 20 (M9 stage 5): a Document ask clears the
        // tab's whole log (in-flight + finished) before opening
        // its own fresh record — the list clears on navigation,
        // and the document row itself survives.
        let mut log = NetworkLog::default();
        let sub = request(1, 9, "https://example.com/a.js", Destination::Script);
        let key = log.start(&sub);
        log.record_event(key, &ResourceEvent::Done { request_id: 1 });
        assert_eq!(log.records_for(9).len(), 1);
        let doc = request(2, 9, "https://example.com/", Destination::Document);
        let doc_key = log.start(&doc);
        let rows = log.records_for(9);
        assert_eq!(rows.len(), 1, "only the new document row remains");
        assert_eq!(rows[0].request_id, 2);
        assert!(matches!(rows[0].outcome, NetworkOutcome::InFlight));
        // The old tab's in-flight key is dead: its late events
        // are ignored, never resurrecting a row.
        log.record_event(key, &ResourceEvent::Done { request_id: 1 });
        assert_eq!(log.records_for(9).len(), 1);
        let _ = doc_key;
    }

    #[test]
    fn log_duplicate_waiter_aliases_the_primary_record() {
        // Stage 20 (M9 stage 5): two identical asks share one
        // fetch (the scheduler dedups), so the second serve task
        // aliases the first task's record — one row, not two.
        let mut log = NetworkLog::default();
        let first = request(1, 5, "https://example.com/logo.png", Destination::Image);
        let second = request(2, 5, "https://example.com/logo.png", Destination::Image);
        let key1 = log.start(&first);
        let key2 = log.start(&second);
        assert_eq!(key1, key2, "the duplicate waiter aliases the primary");
        assert_eq!(log.records_for(5).len(), 1);
        log.record_event(key1, &headers_event(1, 200, vec![]));
        log.record_event(key2, &ResourceEvent::Done { request_id: 1 });
        let rows = log.records_for(5);
        assert_eq!(rows.len(), 1, "one finished row, not two");
        assert_eq!(rows[0].outcome, NetworkOutcome::Done);
        assert_eq!(rows[0].status, Some(200));
    }

    #[test]
    fn log_sanitises_cookie_and_authorization_at_record_time() {
        // Stage 20 (M9 stage 5): sensitive header VALUES never
        // reach the panel — the names stay with `(redacted)`.
        // The strip happens at record time, not in the view.
        let mut log = NetworkLog::default();
        let mut req = request(1, 4, "https://example.com/api", Destination::Fetch);
        req.headers = vec![
            ("Cookie".to_string(), "session=abc123".to_string()),
            ("AUTHORIZATION".to_string(), "Bearer secret".to_string()),
            ("X-Custom".to_string(), "visible".to_string()),
        ];
        let key = log.start(&req);
        log.record_event(
            key,
            &headers_event(
                1,
                200,
                vec![
                    ("Set-Cookie".to_string(), "x=1".to_string()),
                    ("Content-Type".to_string(), "text/html".to_string()),
                ],
            ),
        );
        let row = &log.records_for(4)[0];
        assert_eq!(
            row.request_headers,
            vec![
                ("Cookie".to_string(), "(redacted)".to_string()),
                ("AUTHORIZATION".to_string(), "(redacted)".to_string()),
                ("X-Custom".to_string(), "visible".to_string()),
            ]
        );
        assert!(
            row.response_headers
                .iter()
                .any(|(n, v)| n == "Set-Cookie" && v == "(redacted)"),
            "response headers are sanitised too: {:?}",
            row.response_headers
        );
        assert!(
            !row.request_headers
                .iter()
                .any(|(_, v)| v.contains("abc123") || v.contains("secret")),
            "no secret value survives: {:?}",
            row.request_headers
        );
    }

    #[test]
    fn log_error_outcome_carries_code_and_reason() {
        // Stage 20 (M9 stage 5): a blocked request shows WHY
        // it never hit the network — the Error event's code and
        // reason land on the row.
        let mut log = NetworkLog::default();
        let req = request(1, 2, "https://tracker.example/x.js", Destination::Script);
        let key = log.start(&req);
        log.record_event(
            key,
            &ResourceEvent::Error {
                request_id: 1,
                code: "blocked_by_filter".to_string(),
                reason: "EasyList:42: ||tracker.example^".to_string(),
            },
        );
        let row = &log.records_for(2)[0];
        assert_eq!(
            row.outcome,
            NetworkOutcome::Error {
                code: "blocked_by_filter".to_string(),
                reason: "EasyList:42: ||tracker.example^".to_string(),
            }
        );
        assert!(row.ended.is_some());
        assert!(row.ttfb.is_none(), "no headers, no TTFB");
    }

    #[test]
    fn log_orphan_finish_marks_cancelled_and_late_events_ignored() {
        // Stage 20 (M9 stage 5): a stream that ends without a
        // terminal event closes as cancelled; a terminal event
        // for a finished record is ignored, never duplicated.
        let mut log = NetworkLog::default();
        let req = request(1, 6, "https://example.com/slow.js", Destination::Script);
        let key = log.start(&req);
        log.finish_orphan(key);
        let row = &log.records_for(6)[0];
        assert_eq!(
            row.outcome,
            NetworkOutcome::Cancelled {
                reason: "forwarding stopped (tab closed)".to_string(),
            }
        );
        log.record_event(key, &ResourceEvent::Done { request_id: 1 });
        assert_eq!(log.records_for(6).len(), 1, "no duplicate row");
        log.record_event((6, 999), &ResourceEvent::Done { request_id: 999 });
        assert_eq!(log.records_for(6).len(), 1, "unknown keys create nothing");
        log.finish_orphan(key);
        assert_eq!(log.records_for(6).len(), 1, "orphan finish is idempotent");
    }

    /// Local server for the serve-level tests: every request
    /// gets `200 OK` with a 5-byte body, then the connection
    /// closes (mirrors the `service.rs` test server).
    async fn local_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let Ok(_) = stream.read(&mut buf).await else {
                        return;
                    };
                    let _ = stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello",
                        )
                        .await;
                });
            }
        });
        format!("http://127.0.0.1:{}/", addr.port())
    }

    async fn drain(
        mut rx: tokio::sync::mpsc::Receiver<ResourceEvent>,
    ) -> Vec<ResourceEvent> {
        let mut out = Vec::new();
        while let Some(event) = rx.recv().await {
            let terminal = matches!(
                event,
                ResourceEvent::Done { .. }
                    | ResourceEvent::Error { .. }
                    | ResourceEvent::Cancelled { .. }
            );
            out.push(event);
            if terminal {
                break;
            }
        }
        out
    }

    #[tokio::test]
    async fn serve_records_a_completed_fetch() {
        // Stage 20 (M9 stage 5): a real fetch through a real
        // broker lands one finished row — status, bytes, TTFB,
        // end, Done — with no IPC involved in the recording.
        let base = local_server().await;
        let broker = NetworkBroker::new(Arc::new(ResourceLoader::new()));
        let req = request(1, 11, &base, Destination::Document);
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let serve = broker.serve(req, tx);
        let events = tokio::spawn(async move { drain(rx).await });
        serve.await;
        let events = events.await.unwrap();
        assert!(
            events.iter().any(|e| matches!(
                e,
                ResourceEvent::ResponseHeaders { status: 200, .. }
            )),
            "headers arrive: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, ResourceEvent::Done { .. })),
            "Done arrives: {events:?}"
        );
        let rows = broker.network_records(11);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.status, Some(200));
        assert_eq!(row.bytes_received, 5);
        assert!(row.ttfb.is_some(), "TTFB recorded");
        assert!(row.ended.is_some(), "end recorded");
        assert!(row.total() >= row.ttfb.unwrap());
        assert_eq!(row.outcome, NetworkOutcome::Done);
        assert_eq!(row.final_url.as_deref(), Some(base.as_str()));
    }

    #[tokio::test]
    async fn serve_records_a_failed_fetch() {
        // Stage 20 (M9 stage 5): a fetch that fails at connect
        // lands one finished row with the Error outcome — the
        // panel shows the failure, not a hole.
        let broker = NetworkBroker::new(Arc::new(ResourceLoader::new()));
        let req = request(
            1,
            12,
            "http://127.0.0.1:9/unreachable",
            Destination::Document,
        );
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let serve = broker.serve(req, tx);
        let events = tokio::spawn(async move { drain(rx).await });
        serve.await;
        let events = events.await.unwrap();
        assert!(
            events.iter().any(|e| matches!(e, ResourceEvent::Error { .. })),
            "an Error event arrives: {events:?}"
        );
        let rows = broker.network_records(12);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(matches!(row.outcome, NetworkOutcome::Error { .. }));
        assert!(row.ended.is_some());
        assert_eq!(row.status, None, "no headers, no status");
    }

    #[tokio::test]
    async fn serve_document_request_clears_the_tab_log() {
        // Stage 20 (M9 stage 5): a second Document serve for
        // the same tab replaces the log — the new navigation's
        // rows, not the old page's.
        let base = local_server().await;
        let broker = NetworkBroker::new(Arc::new(ResourceLoader::new()));
        for (id, url) in [
            (1u64, format!("{base}first")),
            (2u64, format!("{base}second")),
        ] {
            let req = request(id, 13, &url, Destination::Document);
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            let serve = broker.serve(req, tx);
            let events = tokio::spawn(async move { drain(rx).await });
            serve.await;
            let _ = events.await.unwrap();
        }
        let rows = broker.network_records(13);
        assert_eq!(rows.len(), 1, "only the second navigation's row remains");
        assert_eq!(rows[0].request_id, 2);
        assert_eq!(rows[0].url, format!("{base}second"));
    }
}
