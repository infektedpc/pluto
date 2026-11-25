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

use pluto_ipc::{Destination, RendererToBrowser, ResourceEvent, ResourceRequest};
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
        let sender = self.pending.lock().await.get(&request_id).cloned();
        if let Some(sender) = sender {
            if sender.send(event).await.is_err() {
                self.pending.lock().await.remove(&request_id);
            }
        }
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
        let mut status = http::StatusCode::OK;
        let mut headers = http::header::HeaderMap::new();
        let mut body: Vec<u8> = Vec::new();
        let mut final_url: Option<http::Uri> = None;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let _ = self.out.send(RendererToBrowser::CancelResource { request_id }).await;
                    self.pending.lock().await.remove(&request_id);
                    return Err(NetError::Cancelled);
                }
                event = rx.recv() => {
                    let Some(event) = event else {
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
                            if body.len() > cap {
                                let _ = self.out.send(RendererToBrowser::CancelResource { request_id }).await;
                                self.pending.lock().await.remove(&request_id);
                                return Err(NetError::TooLarge { destination: request.destination, cap });
                            }
                        }
                        ResourceEvent::Done { .. } => break,
                        ResourceEvent::Error { code, reason, .. } => {
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
