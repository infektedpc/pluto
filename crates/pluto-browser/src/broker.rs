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

use pluto_ipc::{Destination, RendererToBrowser, ResourceEvent, ResourceRequest, WebSocketEvent};
use pluto_net::{NetError, ResourceLoader, Response};
use std::collections::HashMap;
use std::sync::Arc;
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
}

impl NetworkBroker {
    pub fn new(loader: Arc<ResourceLoader>) -> Self {
        Self {
            loader,
            pending: std::sync::Mutex::new(HashMap::new()),
        }
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
        let mut stream = self.loader.fetch_stream(request, cancel).await;
        while let Some(event) = stream.recv().await {
            if events.send(event).await.is_err() {
                break;
            }
        }
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
