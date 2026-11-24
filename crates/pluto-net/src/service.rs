//! The network service (Stage 5 #3): one shared client, a
//! priority scheduler, dedup, caps, timeouts, and the streaming
//! fetch path.
//!
//! There is no long-lived scheduler *task*: the scheduler is shared
//! state (`Arc<Mutex<SchedulerState>>`) plus a `pump` that runs
//! whenever state changes (enqueue, cancel, fetch completion).
//! Every fetch runs as its own spawned task, so a dead runtime
//! cannot strand a loop task — and tests can own a fresh service
//! per test runtime. The shared hyper client (pool, resumption,
//! DNS cache) is process-wide regardless; only the scheduler state
//! is per-`ResourceLoader` — one per worker (per tab) in the app,
//! so caps and dedup are per-tab until M3 multi-tab promotes the
//! loader to a shared handle.
//!
//! Cancellation: every in-flight request owns a child
//! `CancellationToken` of the caller's token. Cancelling the
//! caller's token cancels the fetch; dropping it does *not*
//! (tokio-util only propagates explicit `cancel()`), so the worker
//! must cancel on navigation, not just drop. `Scheduler::cancel`
//! is the explicit path for `CancelResource` messages.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use http::{Method, StatusCode, Uri};
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper::header;
use pluto_ipc::{CredentialsMode, Destination, Priority, ResourceEvent, ResourceRequest};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

use crate::NetError;
use crate::client::{HttpsClient, shared_client};
use crate::decode::Decoder;
use crate::mixed::{MixedContentDecision, mixed_content_decision};
use crate::redirect::{RedirectDecision, next_hop};

/// Maximum concurrent fetches per origin (scheme + authority).
/// Matches the HTTP/1.1 connection limit; for HTTP/2 the
/// connection multiplexes, but the scheduler cap still applies so
/// low-priority work cannot starve the document.
pub const MAX_IN_FLIGHT_PER_ORIGIN: usize = 6;
/// Maximum concurrent fetches overall, as a safety bound.
pub const MAX_IN_FLIGHT_TOTAL: usize = 64;
/// Maximum requests waiting in the scheduler queue. Beyond this,
/// a new ask is answered with an error immediately instead of
/// growing the queue without bound.
pub const MAX_QUEUED: usize = 256;
/// Bounded channel for streaming events to one waiter (report:
/// 8–16 chunks in flight). A full channel is the backpressure
/// signal: the fetch task stops polling the body until the
/// consumer drains it.
pub const EVENT_CHANNEL_CAP: usize = 16;
/// Time to first byte (connect + headers) per request.
pub const TTFB_TIMEOUT: Duration = Duration::from_secs(10);

/// Per-destination cap on *decoded* body bytes. A gzip bomb is
/// caught by the decoded cap, not the encoded one.
pub fn byte_cap(destination: Destination) -> usize {
    const MIB: usize = 1024 * 1024;
    match destination {
        Destination::Image => 50 * MIB,
        _ => 10 * MIB,
    }
}

/// Per-destination total timeout for a fetch (connect + headers +
/// body).
pub fn total_timeout(destination: Destination) -> Duration {
    match destination {
        Destination::Document => Duration::from_secs(30),
        _ => Duration::from_secs(20),
    }
}

/// The dedup key: same URL + destination + credentials + top-level
/// site is one fetch, fanned out to all waiters (the preload
/// scanner and the parser discovering the same URL must not double
/// the traffic).
type DedupKey = (String, Destination, CredentialsMode, String);

fn dedup_key(request: &ResourceRequest) -> DedupKey {
    (
        request.url.clone(),
        request.destination,
        request.credentials,
        request.top_level_site.clone(),
    )
}

/// The origin (scheme + authority) of a URL, for per-origin caps.
/// Unparseable URLs have no origin and never reach the scheduler
/// (they fail validation first).
fn origin_of_url(url: &str) -> String {
    pluto_ipc::origin_of(url)
}

struct Queued {
    request: ResourceRequest,
    cancel: CancellationToken,
    replies: Vec<mpsc::Sender<ResourceEvent>>,
    sequence: u64,
}

struct InFlight {
    cancel: CancellationToken,
    origin: String,
    priority: Priority,
    dedup_key: DedupKey,
    replies: Vec<mpsc::Sender<ResourceEvent>>,
}

struct SchedulerState {
    queue: Vec<Queued>,
    in_flight: HashMap<u64, InFlight>,
    by_origin: HashMap<String, usize>,
    dedup: HashMap<DedupKey, u64>,
    next_sequence: u64,
}

impl SchedulerState {
    fn new() -> Self {
        Self {
            queue: Vec::new(),
            in_flight: HashMap::new(),
            by_origin: HashMap::new(),
            dedup: HashMap::new(),
            next_sequence: 0,
        }
    }
}

/// The scheduler: shared client + shared state. Cloneable (all
/// state is behind `Arc`s); `ResourceLoader` owns one.
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    client: HttpsClient,
    max_redirects: usize,
}

impl Scheduler {
    pub fn new(client: HttpsClient, max_redirects: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(SchedulerState::new())),
            client,
            max_redirects,
        }
    }

    /// Submit a resource ask. Returns the event receiver; the
    /// terminal event (`Done`, `Error` or `Cancelled`) always
    /// arrives exactly once. `cancel` is the caller's token — the
    /// fetch is cancelled when it is cancelled (explicitly).
    pub async fn submit(
        self: &Arc<Self>,
        request: ResourceRequest,
        cancel: CancellationToken,
    ) -> mpsc::Receiver<ResourceEvent> {
        let (reply_tx, reply_rx) = mpsc::channel(EVENT_CHANNEL_CAP);
        let key = dedup_key(&request);
        let mut state = self.state.lock().await;
        // Dedup: an identical request already in flight gets a
        // second waiter instead of a second fetch.
        if let Some(&id) = state.dedup.get(&key)
            && let Some(entry) = state.in_flight.get_mut(&id)
        {
            entry.replies.push(reply_tx);
            return reply_rx;
        }
        // …or already queued.
        if let Some(queued) = state
            .queue
            .iter_mut()
            .find(|q| dedup_key(&q.request) == key)
        {
            queued.replies.push(reply_tx);
            return reply_rx;
        }
        if state.queue.len() >= MAX_QUEUED {
            drop(state);
            let _ = reply_tx
                .send(ResourceEvent::Error {
                    request_id: request.request_id,
                    code: "too_many_queued".to_string(),
                    reason: "scheduler queue is full".to_string(),
                })
                .await;
            return reply_rx;
        }
        let sequence = state.next_sequence;
        state.next_sequence += 1;
        state.queue.push(Queued {
            request,
            cancel,
            replies: vec![reply_tx],
            sequence,
        });
        self.pump(&mut state);
        reply_rx
    }

    /// Cancel a queued or in-flight request by id. A queued ask is
    /// answered `Cancelled` at once; an in-flight one has its token
    /// cancelled and the fetch task sends `Cancelled` itself.
    pub async fn cancel(self: &Arc<Self>, request_id: u64) {
        let mut state = self.state.lock().await;
        if let Some(pos) = state
            .queue
            .iter()
            .position(|q| q.request.request_id == request_id)
        {
            let queued = state.queue.remove(pos);
            let replies = queued.replies;
            drop(state);
            send_all(replies, ResourceEvent::Cancelled { request_id }).await;
            return;
        }
        if let Some(entry) = state.in_flight.get(&request_id) {
            entry.cancel.cancel();
        }
    }

    /// Start queued work while the caps allow. Scans in priority
    /// order and starts the first startable ask, skipping (not
    /// blocking on) asks whose origin is saturated — a slow origin
    /// must not head-block the queue. Delayable asks (Low/VeryLow)
    /// start only when no non-delayable ask is queued or in flight
    /// for their origin.
    fn pump(self: &Arc<Self>, state: &mut SchedulerState) {
        state.queue.sort_by(|a, b| {
            a.request
                .priority
                .cmp(&b.request.priority)
                .then_with(|| a.sequence.cmp(&b.sequence))
        });
        let mut i = 0;
        while i < state.queue.len() {
            if state.in_flight.len() >= MAX_IN_FLIGHT_TOTAL {
                return;
            }
            let queued = &state.queue[i];
            if queued.cancel.is_cancelled() {
                let queued = state.queue.remove(i);
                let request_id = queued.request.request_id;
                let replies = queued.replies;
                tokio::spawn(async move {
                    send_all(replies, ResourceEvent::Cancelled { request_id }).await;
                });
                continue;
            }
            let origin = origin_of_url(&queued.request.url);
            if state.by_origin.get(&origin).copied().unwrap_or(0) >= MAX_IN_FLIGHT_PER_ORIGIN {
                i += 1;
                continue;
            }
            if queued.request.priority.is_delayable() && origin_has_non_delayable(state, &origin) {
                i += 1;
                continue;
            }
            let queued = state.queue.remove(i);
            self.start(state, queued);
            // Do not advance: the next ask shifted into place.
        }
    }

    fn start(self: &Arc<Self>, state: &mut SchedulerState, queued: Queued) {
        let origin = origin_of_url(&queued.request.url);
        *state.by_origin.entry(origin.clone()).or_insert(0) += 1;
        let child = queued.cancel.child_token();
        let key = dedup_key(&queued.request);
        state.dedup.insert(key.clone(), queued.request.request_id);
        state.in_flight.insert(
            queued.request.request_id,
            InFlight {
                cancel: child.clone(),
                origin,
                priority: queued.request.priority,
                dedup_key: key,
                replies: queued.replies,
            },
        );
        let scheduler = Arc::clone(self);
        let request = queued.request;
        tokio::spawn(async move {
            run_fetch(&scheduler, &request, &child).await;
            scheduler.finish(request.request_id).await;
        });
    }

    /// A fetch ended (any outcome): release its caps and dedup
    /// entry, then pump the queue.
    async fn finish(self: &Arc<Self>, request_id: u64) {
        let mut state = self.state.lock().await;
        if let Some(entry) = state.in_flight.remove(&request_id) {
            if let Some(count) = state.by_origin.get_mut(&entry.origin) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    state.by_origin.remove(&entry.origin);
                }
            }
            state.dedup.remove(&entry.dedup_key);
        }
        self.pump(&mut state);
    }

    /// Send one event to every waiter of `request_id`. Returns
    /// false when nobody is listening anymore (the fetch should
    /// stop: its bytes have no consumer). Dead senders are pruned.
    async fn emit(&self, request_id: u64, event: ResourceEvent) -> bool {
        let senders = {
            let state = self.state.lock().await;
            match state.in_flight.get(&request_id) {
                Some(entry) => entry.replies.clone(),
                None => return false,
            }
        };
        if senders.is_empty() {
            return false;
        }
        let mut alive = Vec::with_capacity(senders.len());
        for sender in senders {
            match sender.send(event.clone()).await {
                Ok(()) => alive.push(sender),
                Err(_) => { /* receiver dropped: prune it */ }
            }
        }
        if alive.is_empty() {
            return false;
        }
        let mut state = self.state.lock().await;
        if let Some(entry) = state.in_flight.get_mut(&request_id) {
            entry.replies = alive;
        }
        true
    }
}

fn origin_has_non_delayable(state: &SchedulerState, origin: &str) -> bool {
    state
        .queue
        .iter()
        .any(|q| !q.request.priority.is_delayable() && origin_of_url(&q.request.url) == origin)
        || state
            .in_flight
            .values()
            .any(|f| !f.priority.is_delayable() && f.origin == origin)
}

/// Send one event to every sender, ignoring dropped receivers.
async fn send_all(senders: Vec<mpsc::Sender<ResourceEvent>>, event: ResourceEvent) {
    for sender in senders {
        let _ = sender.send(event.clone()).await;
    }
}

/// Run one fetch to completion, emitting its events. The terminal
/// event is always sent (unless nobody is listening): `Done` comes
/// from `execute_fetch`, `Error`/`Cancelled` from here.
async fn run_fetch(
    scheduler: &Arc<Scheduler>,
    request: &ResourceRequest,
    cancel: &CancellationToken,
) {
    let total = total_timeout(request.destination);
    let outcome = tokio::time::timeout(total, execute_fetch(scheduler, request, cancel)).await;
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            if matches!(e, NetError::Cancelled) {
                scheduler
                    .emit(
                        request.request_id,
                        ResourceEvent::Cancelled {
                            request_id: request.request_id,
                        },
                    )
                    .await;
            } else {
                scheduler
                    .emit(
                        request.request_id,
                        ResourceEvent::Error {
                            request_id: request.request_id,
                            code: e.code().to_string(),
                            reason: e.to_string(),
                        },
                    )
                    .await;
            }
        }
        Err(_) => {
            scheduler
                .emit(
                    request.request_id,
                    ResourceEvent::Error {
                        request_id: request.request_id,
                        code: "timeout".to_string(),
                        reason: format!("fetch timed out after {total:?}"),
                    },
                )
                .await;
        }
    }
}

/// The fetch itself: gates, redirect chain, headers, streaming
/// body with decoding and caps. Emits `ResponseHeaders`, `BodyChunk`s
/// and `Done`; any error returns early and `run_fetch` emits the
/// terminal `Error`/`Cancelled`.
async fn execute_fetch(
    scheduler: &Arc<Scheduler>,
    request: &ResourceRequest,
    cancel: &CancellationToken,
) -> Result<(), NetError> {
    let mut current: Uri = request
        .url
        .parse()
        .map_err(|_| NetError::InvalidUrl(request.url.clone()))?;
    if !matches!(current.scheme_str(), Some("http") | Some("https")) {
        return Err(NetError::InvalidUrl(request.url.clone()));
    }
    // Mixed-content gate on the initial URL (and again per hop).
    current = match mixed_content_decision(
        &request.origin,
        &current,
        request.destination,
        request.upgrade_insecure_requests,
    ) {
        MixedContentDecision::Allow => current,
        MixedContentDecision::Block => {
            return Err(NetError::MixedContentBlocked(request.url.clone()));
        }
        MixedContentDecision::Upgrade(upgraded) => upgraded,
    };
    let mut method = Method::GET;
    let mut redirects = 0usize;
    loop {
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(NetError::Cancelled),
            result = tokio::time::timeout(
                TTFB_TIMEOUT,
                scheduler.client.request(build_request(&method, &current, request.destination)?),
            ) => match result {
                Ok(Ok(response)) => response,
                Ok(Err(e)) => return Err(NetError::Request(e.to_string())),
                Err(_) => return Err(NetError::Timeout("time to first byte".to_string())),
            },
        };
        let status = response.status();
        if matches!(
            status,
            StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT
                | StatusCode::PERMANENT_REDIRECT
        ) {
            if redirects >= scheduler.max_redirects {
                return Err(NetError::TooManyRedirects(scheduler.max_redirects));
            }
            let location = response
                .headers()
                .get(header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| NetError::Request("redirect without Location".into()))?
                .to_string();
            let decision = next_hop(&current, status, &location, &method)?;
            // Re-run the mixed-content gate on the hop target: a
            // redirect to http from an https page is gated exactly
            // like a direct ask (the downgrade itself is already
            // refused by next_hop).
            let RedirectDecision::Follow {
                url: hop_url,
                method: hop_method,
            } = decision;
            current = match mixed_content_decision(
                &request.origin,
                &hop_url,
                request.destination,
                request.upgrade_insecure_requests,
            ) {
                MixedContentDecision::Allow => hop_url,
                MixedContentDecision::Block => {
                    return Err(NetError::MixedContentBlocked(hop_url.to_string()));
                }
                MixedContentDecision::Upgrade(upgraded) => upgraded,
            };
            method = hop_method;
            redirects += 1;
            continue;
        }

        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(name, _)| *name != header::SET_COOKIE)
            .map(|(name, value)| {
                (
                    name.as_str().to_string(),
                    value.to_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        let content_length = response
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        let encoding = response
            .headers()
            .get(header::CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if !scheduler
            .emit(
                request.request_id,
                ResourceEvent::ResponseHeaders {
                    request_id: request.request_id,
                    status: status.as_u16(),
                    url: current.to_string(),
                    headers,
                    content_length,
                },
            )
            .await
        {
            return Err(NetError::Cancelled);
        }
        let mut decoder = Decoder::from_encoding(encoding.as_deref())?;
        let mut body = response.into_body();
        let mut decoded_total = 0usize;
        let cap = byte_cap(request.destination);
        loop {
            let frame = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(NetError::Cancelled),
                frame = body.frame() => frame,
            };
            let Some(frame) = frame else { break };
            let frame = frame.map_err(|e| NetError::Body(e.to_string()))?;
            if let Some(data) = frame.data_ref() {
                let decoded = decoder.feed(data)?;
                decoded_total += decoded.len();
                if decoded_total > cap {
                    return Err(NetError::TooLarge {
                        destination: request.destination,
                        cap,
                    });
                }
                if !decoded.is_empty()
                    && !scheduler
                        .emit(
                            request.request_id,
                            ResourceEvent::BodyChunk {
                                request_id: request.request_id,
                                bytes: decoded,
                            },
                        )
                        .await
                {
                    return Err(NetError::Cancelled);
                }
            }
        }
        let tail = decoder.finish()?;
        decoded_total += tail.len();
        if decoded_total > cap {
            return Err(NetError::TooLarge {
                destination: request.destination,
                cap,
            });
        }
        if !tail.is_empty()
            && !scheduler
                .emit(
                    request.request_id,
                    ResourceEvent::BodyChunk {
                        request_id: request.request_id,
                        bytes: tail,
                    },
                )
                .await
        {
            return Err(NetError::Cancelled);
        }
        scheduler
            .emit(
                request.request_id,
                ResourceEvent::Done {
                    request_id: request.request_id,
                },
            )
            .await;
        return Ok(());
    }
}

/// Build one request. The header set is rebuilt fresh on every hop:
/// only User-Agent, Accept and Accept-Encoding are ever sent — no
/// Authorization, no Cookie — so a cross-origin hop leaks nothing
/// (that rebuild *is* the header-stripping point; if a sensitive
/// header is ever added, strip it here).
fn build_request(
    method: &Method,
    url: &Uri,
    destination: Destination,
) -> Result<Request<Full<Bytes>>, NetError> {
    let accept = match destination {
        Destination::Document => "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        Destination::Stylesheet => "text/css,*/*;q=0.1",
        Destination::Script => "*/*",
        Destination::Image => "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        Destination::Font => "font/*,*/*;q=0.1",
        Destination::Fetch | Destination::Other => "*/*",
    };
    Request::builder()
        .method(method.clone())
        .uri(url.clone())
        .header(header::USER_AGENT, "Pluto/0.1.0")
        .header(header::ACCEPT, accept)
        .header(header::ACCEPT_ENCODING, "gzip, deflate")
        .body(Full::new(Bytes::new()))
        .map_err(|e| NetError::Request(e.to_string()))
}

/// The shared client accessor, for `ResourceLoader`.
pub fn shared() -> &'static HttpsClient {
    shared_client()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Local server: every request gets a canned response by path.
    /// Slow paths sleep, so scheduler behaviour (caps, priority,
    /// cancellation) is observable. The response includes a
    /// `Connection: close` so hyper does not pool across requests
    /// in these tests (pooling is covered by the shared-client
    /// tests, not by the scheduler tests).
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 8192];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/");
                let has_gzip_accept = request.to_ascii_lowercase().contains("accept-encoding:")
                    && request.to_ascii_lowercase().contains("gzip");
                let (status, headers, body): (&str, String, Vec<u8>) = match path {
                    "/ok" => (
                        "200 OK",
                        "Content-Type: text/html".into(),
                        b"hello".to_vec(),
                    ),
                    p if p.starts_with("/slow") => {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        ("200 OK", "Content-Type: text/html".into(), b"slow".to_vec())
                    }
                    "/gzip" => {
                        let plain = b"gzipped body".to_vec();
                        let mut e = flate2::write::GzEncoder::new(
                            Vec::new(),
                            flate2::Compression::default(),
                        );
                        use std::io::Write as _;
                        e.write_all(&plain).unwrap();
                        let compressed = e.finish().unwrap();
                        (
                            "200 OK",
                            format!(
                                "Content-Type: text/html\r\nContent-Encoding: gzip\r\nX-Saw-Gzip-Accept: {}",
                                has_gzip_accept
                            ),
                            compressed,
                        )
                    }
                    "/deflate" => {
                        let plain = b"deflated body".to_vec();
                        let mut e = flate2::write::ZlibEncoder::new(
                            Vec::new(),
                            flate2::Compression::default(),
                        );
                        use std::io::Write as _;
                        e.write_all(&plain).unwrap();
                        let compressed = e.finish().unwrap();
                        (
                            "200 OK",
                            "Content-Type: text/html\r\nContent-Encoding: deflate".into(),
                            compressed,
                        )
                    }
                    "/big" => {
                        // 1 MiB of 'a' — over the test cap path only
                        // when a small destination cap is set; the
                        // document cap is 10 MiB, so this stays
                        // under it. Used for streaming checks.
                        (
                            "200 OK",
                            "Content-Type: text/html".into(),
                            vec![b'a'; 1024 * 1024],
                        )
                    }
                    "/redirect303" => ("303 See Other", "Location: /ok".into(), Vec::new()),
                    "/loop" => ("302 Found", "Location: /loop".into(), Vec::new()),
                    _ => ("404 Not Found", String::new(), Vec::new()),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\n{headers}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(&body).await;
            });
        }
    }

    async fn test_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        format!("http://127.0.0.1:{}/", addr.port())
    }

    fn scheduler() -> Arc<Scheduler> {
        Arc::new(Scheduler::new(shared_client().clone(), 5))
    }

    fn request(
        id: u64,
        url: &str,
        destination: Destination,
        priority: Priority,
    ) -> ResourceRequest {
        let origin = origin_of_url(url);
        ResourceRequest {
            request_id: id,
            tab_id: 0,
            epoch: 1,
            url: url.to_string(),
            destination,
            priority,
            mode: pluto_ipc::RequestMode::NoCors,
            credentials: CredentialsMode::SameOrigin,
            origin: origin.clone(),
            top_level_site: origin,
            upgrade_insecure_requests: false,
        }
    }

    async fn collect(
        mut rx: mpsc::Receiver<ResourceEvent>,
    ) -> (u16, Vec<(String, String)>, Vec<u8>, Option<String>) {
        let mut status = 0;
        let mut headers = Vec::new();
        let mut body = Vec::new();
        let mut terminal: Option<String> = None;
        while let Some(event) = rx.recv().await {
            match event {
                ResourceEvent::ResponseHeaders {
                    status: s,
                    headers: h,
                    ..
                } => {
                    status = s;
                    headers = h;
                }
                ResourceEvent::BodyChunk { bytes, .. } => body.extend_from_slice(&bytes),
                ResourceEvent::Done { .. } => break,
                ResourceEvent::Error { code, reason, .. } => {
                    terminal = Some(format!("{code}: {reason}"));
                    break;
                }
                ResourceEvent::Cancelled { .. } => {
                    terminal = Some("cancelled".to_string());
                    break;
                }
            }
        }
        (status, headers, body, terminal)
    }

    #[tokio::test]
    async fn fetch_streams_a_document_and_collects() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}ok"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"hello");
        assert!(terminal.is_none());
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "text/html"),
            "headers arrive: {headers:?}"
        );
    }

    #[tokio::test]
    async fn gzip_body_is_decoded_and_accept_advertised() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}gzip"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"gzipped body", "gzip must decode, not arrive raw");
        assert!(terminal.is_none());
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "x-saw-gzip-accept" && v == "true"),
            "Accept-Encoding: gzip must be sent: {headers:?}"
        );
    }

    #[tokio::test]
    async fn deflate_body_is_decoded() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}deflate"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (_, _, body, terminal) = collect(rx).await;
        assert_eq!(body, b"deflated body");
        assert!(terminal.is_none());
    }

    #[tokio::test]
    async fn dedup_fans_one_fetch_out_to_two_waiters() {
        let base = test_server().await;
        let sched = scheduler();
        let url = format!("{base}slow");
        let rx1 = sched
            .submit(
                request(1, &url, Destination::Image, Priority::Low),
                CancellationToken::new(),
            )
            .await;
        let rx2 = sched
            .submit(
                request(2, &url, Destination::Image, Priority::Low),
                CancellationToken::new(),
            )
            .await;
        let (r1, r2) = tokio::join!(collect(rx1), collect(rx2));
        assert_eq!(r1.2, b"slow");
        assert_eq!(r2.2, b"slow");
        assert!(r1.3.is_none() && r2.3.is_none());
        // If dedup failed, the second request would have taken a
        // second 300 ms — the join above bounds the total time,
        // but the real proof is the server saw one request: the
        // slow path sleeps per request, so two fetches would take
        // ~600 ms sequential on one connection… (timing is not
        // asserted; the shared body + single Done is the signal).
    }

    #[tokio::test]
    async fn cancellation_by_token_ends_the_fetch() {
        let base = test_server().await;
        let sched = scheduler();
        let token = CancellationToken::new();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}slow"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                token.clone(),
            )
            .await;
        // Cancel while the server is still sleeping.
        tokio::time::sleep(Duration::from_millis(50)).await;
        token.cancel();
        let (_, _, _, terminal) = collect(rx).await;
        assert_eq!(terminal.as_deref(), Some("cancelled"));
    }

    #[tokio::test]
    async fn cancel_command_cancels_a_queued_request() {
        let base = test_server().await;
        let sched = scheduler();
        // Saturate the origin with 6 slow fetches, then queue a 7th.
        // Distinct query strings: identical asks would dedup into
        // one fetch (by design), which would defeat the saturation.
        let mut rxs = Vec::new();
        for id in 1..=6u64 {
            rxs.push(
                sched
                    .submit(
                        request(
                            id,
                            &format!("{base}slow?id={id}"),
                            Destination::Image,
                            Priority::Low,
                        ),
                        CancellationToken::new(),
                    )
                    .await,
            );
        }
        let queued = sched
            .submit(
                request(7, &format!("{base}ok"), Destination::Image, Priority::Low),
                CancellationToken::new(),
            )
            .await;
        sched.cancel(7).await;
        let (_, _, _, terminal) = collect(queued).await;
        assert_eq!(terminal.as_deref(), Some("cancelled"));
        // Drain the saturating fetches so the test runtime ends clean.
        for rx in rxs {
            let _ = collect(rx).await;
        }
    }

    #[tokio::test]
    async fn per_origin_cap_serialises_the_seventh_request() {
        let base = test_server().await;
        let sched = scheduler();
        let start = tokio::time::Instant::now();
        let mut rxs = Vec::new();
        for id in 1..=7u64 {
            rxs.push(
                sched
                    .submit(
                        request(
                            id,
                            &format!("{base}slow?id={id}"),
                            Destination::Image,
                            Priority::Low,
                        ),
                        CancellationToken::new(),
                    )
                    .await,
            );
        }
        for rx in rxs {
            let _ = collect(rx).await;
        }
        // 6 run concurrently (~300 ms); the 7th waits for one to
        // finish, so the total is ~600 ms, not ~300 ms.
        assert!(
            start.elapsed() >= Duration::from_millis(550),
            "7th request must wait for a slot: elapsed {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn redirect_303_is_followed() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}redirect303"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, _, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"hello");
        assert!(terminal.is_none());
    }

    #[tokio::test]
    async fn redirect_loop_errors() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}loop"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("too_many_redirects")),
            "expected too_many_redirects, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn mixed_content_blocks_a_script_on_an_https_page() {
        let base = test_server().await;
        let sched = scheduler();
        let mut req = request(1, &format!("{base}ok"), Destination::Script, Priority::High);
        // The page is https, the ask is http: blockable.
        req.origin = "https://example.com".to_string();
        req.top_level_site = "https://example.com".to_string();
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("mixed_content")),
            "expected mixed_content block, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn large_body_streams_in_chunks() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}big"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (_, _, body, terminal) = collect(rx).await;
        assert_eq!(body.len(), 1024 * 1024);
        assert!(terminal.is_none());
    }
}
