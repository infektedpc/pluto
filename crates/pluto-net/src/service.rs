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
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use http::{Method, StatusCode, Uri};
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper::header;
use pluto_ipc::{
    CredentialsMode, Destination, Origin, Priority, RequestMode, ResourceEvent, ResourceRequest,
};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::NetError;
use crate::client::{HttpsClient, shared_client};
use crate::blocker::{Blocker, profile_list_dir};
use crate::cors::{
    PreflightCache, PreflightCacheKey, ResponseTainting, classify_request_headers,
    cors_check, cors_check_is_skipped, filter_basic_response_headers,
    filter_cors_response_headers, is_forbidden_request_header, origin_header_for,
    redirect_url_has_credentials, run_preflight, strip_request_body_headers,
    strip_sensitive_headers_on_redirect, tainting_for,
};
use crate::csp::{PolicySet, directive_for_destination};
use crate::decode::Decoder;
use crate::hsts::HstsStore;
use crate::https_only::HttpsOnly;
use crate::jar::{CookieJar, url_is_secure_str};
use crate::mixed::{Initiator, MixedContentDecision, is_mixed_download, mixed_content_decision};
use crate::redirect::{RedirectDecision, next_hop};
use crate::referrer::{ReferrerContext, ReferrerPolicy, referer_for};
use crate::sri::SriVerifier;

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
///
/// Stage 9 Rec 5 / 15: the scheduler also owns the security
/// policy consulted at the very front of every fetch — the HSTS
/// store, the HTTPS-only preference + exceptions, and the
/// referrer context for the current document. They live behind
/// plain `std::sync::Mutex`es (short critical sections, never
/// held across an await) and are shared with the `ResourceLoader`
/// so `pluto-browser` can load them from the profile and save
/// them back.
///
/// Stage 9 Rec 2: the scheduler also owns the CORS preflight cache
/// (5 s default, 7200 s cap), shared the same way.
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    client: HttpsClient,
    max_redirects: usize,
    hsts: Arc<std::sync::Mutex<HstsStore>>,
    https_only: Arc<std::sync::Mutex<HttpsOnly>>,
    referrer: Arc<std::sync::Mutex<ReferrerContext>>,
    preflight: Arc<PreflightCache>,
    // Stage 9 Rec 3: the cookie jar for this loader (per-tab
    // until M3 promotes the loader to a shared handle — same
    // placement as the HSTS store above).
    jar: Arc<std::sync::Mutex<CookieJar>>,
    // Stage 9 Rec 8: the current document's CSP policy
    // container, stored when the document response arrives and
    // cleared when the next navigation starts.
    csp: Arc<std::sync::Mutex<PolicySet>>,
    // Stage 9 Rec 12: the tracker blocker (Round 1 basic),
    // loaded from the profile list dir at construction and
    // refreshed in the background by the tab thread.
    blocker: Arc<std::sync::Mutex<Blocker>>,
}

impl Scheduler {
    pub fn new(client: HttpsClient, max_redirects: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(SchedulerState::new())),
            client,
            max_redirects,
            hsts: Arc::new(std::sync::Mutex::new(HstsStore::new())),
            https_only: Arc::new(std::sync::Mutex::new(HttpsOnly::new())),
            referrer: Arc::new(std::sync::Mutex::new(ReferrerContext::new())),
            preflight: Arc::new(PreflightCache::new()),
            jar: Arc::new(std::sync::Mutex::new(CookieJar::new())),
            csp: Arc::new(std::sync::Mutex::new(PolicySet::new())),
            blocker: Arc::new(std::sync::Mutex::new(
                profile_list_dir()
                    .map(|dir| Blocker::load_from_dir(&dir))
                    .unwrap_or_default(),
            )),
        }
    }

    /// The HSTS store consulted at the front of the pipeline.
    pub fn hsts_store(&self) -> Arc<std::sync::Mutex<HstsStore>> {
        Arc::clone(&self.hsts)
    }

    /// The HTTPS-only preference + per-site exceptions.
    pub fn https_only(&self) -> Arc<std::sync::Mutex<HttpsOnly>> {
        Arc::clone(&self.https_only)
    }

    /// The referrer context for the current document.
    pub fn referrer_context(&self) -> Arc<std::sync::Mutex<ReferrerContext>> {
        Arc::clone(&self.referrer)
    }

    /// The CORS preflight cache (Stage 9 Rec 2).
    pub fn preflight_cache(&self) -> Arc<PreflightCache> {
        Arc::clone(&self.preflight)
    }

    /// The cookie jar for this loader (Stage 9 Rec 3).
    pub fn jar(&self) -> Arc<std::sync::Mutex<CookieJar>> {
        Arc::clone(&self.jar)
    }

    /// The current document's CSP policy container (Stage 9
    /// Rec 8): empty until a document response stores one.
    pub fn csp_policies(&self) -> Arc<std::sync::Mutex<PolicySet>> {
        Arc::clone(&self.csp)
    }

    /// The tracker blocker for this loader (Stage 9 Rec 12,
    /// Round 1 basic).
    pub fn blocker(&self) -> Arc<std::sync::Mutex<Blocker>> {
        Arc::clone(&self.blocker)
    }

    /// Front-of-pipeline secure upgrade (Stage 9 Rec 5): HSTS
    /// first, then HTTPS-only for hosts with no HSTS entry. Runs
    /// BEFORE the mixed-content gate and before CSP sees the URL,
    /// so an HSTS host's `http` subresource is upgraded, not
    /// blocked as mixed content. Returns true when the URL was
    /// rewritten by HTTPS-only (not HSTS) — the caller maps a
    /// later failure of a top-level navigation to
    /// [`NetError::HttpsOnlyFailed`].
    fn apply_secure_upgrade(&self, url: &mut Uri) -> bool {
        if url.scheme_str() != Some("http") {
            return false;
        }
        let now = SystemTime::now();
        if let Some(rewritten) = self.hsts.lock().unwrap().rewrite_http_uri(url, now) {
            *url = rewritten;
            return false;
        }
        if let Some(rewritten) = self.https_only.lock().unwrap().rewrite_http_uri(url) {
            *url = rewritten;
            return true;
        }
        false
    }

    /// Record an STS header from a response received over a secure
    /// transport with no TLS errors (RFC 6797 §8.1). Called for
    /// every response — including redirect responses — as soon as
    /// the headers arrive; a transport failure never reaches here,
    /// so reaching here *is* the error-free signal. Only the first
    /// STS header is processed (`HeaderMap::get` returns it).
    fn record_hsts(&self, url: &Uri, headers: &header::HeaderMap) {
        let Some(host) = url.host() else {
            return;
        };
        let value = headers
            .get(header::STRICT_TRANSPORT_SECURITY)
            .and_then(|v| v.to_str().ok());
        self.hsts
            .lock()
            .unwrap()
            .record(host, true, value, SystemTime::now());
    }

    /// A document response (not a redirect) arrived: it becomes
    /// the referrer context's document, and its `Referrer-Policy`
    /// header — or the default when it carries none — becomes the
    /// policy in effect for it. A later `<meta name="referrer">`
    /// during the parse overrides this (last one seen wins).
    fn note_document_response(&self, url: &Uri, headers: &header::HeaderMap) {
        let mut ctx = self.referrer.lock().unwrap();
        ctx.document_url = Some(url.to_string());
        ctx.policy = headers
            .get(header::REFERRER_POLICY)
            .and_then(|v| v.to_str().ok())
            .and_then(ReferrerPolicy::parse)
            .unwrap_or_default();
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

/// CSP pre-request check (Stage 9 Rec 8): the current
/// document's policy container gates the fetch by destination
/// directive (`default-src` / `script-src` / `style-src` /
/// `img-src` / `connect-src` / `font-src` / … via
/// [`directive_for_destination`]). A `false` blocks the fetch —
/// a network error, no request sent. It slots in BEFORE CORS per
/// the frozen gate order (HSTS/mixed → blocker → CSP fetch →
/// CORS → cache → network) and runs on every redirect hop too,
/// against the hop URL.
///
/// Navigations carry no fetch directive (`directive_for_destination`
/// returns `None` for `Document`) and always pass here:
/// `frame-ancestors` has no call site yet — iframes don't render,
/// so there is no framed document to check against an ancestor
/// chain. `base-uri` / `form-action` likewise have no call site
/// yet (no `<base>` handling in the parser, no forms until M5).
///
/// With no policy stored (no document response yet, or a document
/// with no CSP headers) every fetch passes — a policy container's
/// CSP list "is empty unless otherwise specified", so there is no
/// implicit policy. Do NOT invent a default `default-src 'self'`.
///
/// Deliberately NOT here, per the report:
/// * no `block-all-mixed-content` — Mixed Content L2 renders it
///   obsolete ("all mixed content is now blocked if it can't be
///   autoupgraded"); parse-and-ignore at most, never enforce;
/// * no report sending — `Violation` records are console-report
///   only in v1 (no `report-uri` / `report-to` network sends).
fn csp_pre_request_check(
    scheduler: &Scheduler,
    request: &ResourceRequest,
    url: &str,
) -> bool {
    let Some(directive) = directive_for_destination(request.destination) else {
        return true;
    };
    let policies_arc = scheduler.csp_policies();
    let policies = policies_arc.lock().unwrap();
    if policies.is_empty() {
        return true;
    }
    // `request.origin` is the document origin — the `'self'`
    // against which source expressions match.
    policies.check(directive, url, &Origin::from_url(&request.origin))
}

/// The CORS pre-request decision (Stage 9 Rec 2): runs before any
/// network traffic for the actual request.
///
/// * `SameOrigin` mode to another origin fails here — the M4
///   refusal, kept as the fallback for a request the CORS code
///   cannot classify (fail closed, never fail open).
/// * `Cors` mode to another origin: an opaque request origin can
///   never pass a CORS check, so it fails here; otherwise the
///   request is classified and, when it needs a preflight, the
///   cache is consulted first and a missing entry is fetched
///   (OPTIONS round-trip) and stored. On preflight failure the
///   whole fetch returns a network error and the preflight cache
///   entries for the request are cleared.
/// * `Navigate` / `NoCors` need no pre-request CORS decision
///   (their response handling — including the opaque filter for a
///   `no-cors` fetch() — runs after the headers arrive).
///
/// The effective request the wire can express today is method GET
/// with no script headers (`ResourceRequest` carries neither — the
/// wire shape is frozen and `PROTOCOL_VERSION` stays 3), which is
/// always simple: a preflight never fires on live traffic yet.
/// The classification + preflight path below takes the real method
/// / headers as inputs, so it starts firing the moment the wire
/// carries them; the preflight round-trip itself is proven by the
/// `cors.rs` tests against a local server.
async fn cors_pre_request(
    scheduler: &Arc<Scheduler>,
    request: &ResourceRequest,
    url: &Uri,
) -> Result<(), NetError> {
    let request_origin = Origin::from_url(&request.origin);
    let url_origin = Origin::from_url(&url.to_string());
    let same_origin = request_origin.is_same_origin(&url_origin);
    match request.mode {
        RequestMode::Navigate | RequestMode::NoCors => Ok(()),
        RequestMode::SameOrigin => {
            if same_origin {
                Ok(())
            } else {
                Err(NetError::CorsBlocked(format!(
                    "same-origin fetch to cross-origin URL {url}"
                )))
            }
        }
        RequestMode::Cors => {
            if same_origin {
                return Ok(());
            }
            if request_origin.is_opaque() {
                return Err(NetError::CorsBlocked(
                    "opaque origin cannot make a CORS request".to_string(),
                ));
            }
            // Stage 9 Rec 2: the preflight classifies the ACTUAL
            // request method + headers — a script-set `PUT`,
            // `Authorization`, or `Content-Type: application/json`
            // is what forces a preflight, so the request must
            // carry them (they ride `ResourceRequest` now).
            let classification =
                classify_request_headers(&request.method, &request.headers)
                    .map_err(NetError::CorsBlocked)?;
            if !classification.needs_preflight {
                return Ok(());
            }
            let credentials_include = request.credentials == CredentialsMode::Include;
            let key = PreflightCacheKey {
                partition_key: request.top_level_site.clone(),
                origin: request_origin.serialize(),
                url: url.to_string(),
                credentials_include,
                method: request.method.clone(),
                headers: classification.unsafe_headers.clone(),
            };
            if scheduler.preflight_cache().lookup(&key) {
                return Ok(());
            }
            match run_preflight(
                &scheduler.client,
                &request_origin,
                url,
                &request.method,
                &classification.unsafe_headers,
                request.credentials,
            )
            .await
            {
                Ok(max_age) => {
                    scheduler.preflight_cache().store(key, max_age);
                    Ok(())
                }
                Err(reason) => {
                    scheduler.preflight_cache().clear_for_request(
                        &request.top_level_site,
                        &request_origin.serialize(),
                        &url.to_string(),
                        credentials_include,
                    );
                    Err(NetError::CorsBlocked(reason))
                }
            }
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
    // Stage 9 Rec 5, front of the pipeline: HSTS / HTTPS-only
    // rewrite BEFORE the mixed-content gate (and before CSP sees
    // the URL). `https_only_rewrote` tracks whether the initial
    // URL was rewritten by HTTPS-only, so a later failure of a
    // top-level navigation maps to `HttpsOnlyFailed`.
    let mut https_only_rewrote = scheduler.apply_secure_upgrade(&mut current);
    // Stage 9 Rec 8: a navigation starts a fresh document — the
    // previous document's policy container must not govern this
    // fetch or its subresources; the new policy is stored when
    // the document response arrives (below). Subresource fetches
    // of the old document are cancelled on navigation, so the
    // only fetch that can race the clear is already dying.
    if request.destination == Destination::Document {
        *scheduler.csp_policies().lock().unwrap() = PolicySet::new();
    }
    // Stage 9 Rec 8: `upgrade-insecure-requests` from the current
    // document's policy upgrades this fetch's mixed-content gate
    // (the request flag is what the tab thread set). For a
    // navigation the stored policy was just cleared, so the old
    // document's UIR cannot leak into the new one.
    let upgrade_insecure_requests = request.upgrade_insecure_requests
        || scheduler
            .csp_policies()
            .lock()
            .unwrap()
            .upgrade_insecure_requests();
    // Mixed-content gate on the initial URL (and again per hop).
    // The initiator rides `ResourceRequest.is_imageset`, threaded
    // from the preload scanner's `from: "imageset"` label through
    // `page.rs::subresource_request` (Stage 9 Rec 2/6): an
    // imageset image BLOCKS on a secure page, never auto-upgrades.
    // The CORS-mode half is live too: `request.mode` rides the
    // wire.
    current = match mixed_content_decision(
        &request.origin,
        &current,
        request.destination,
        upgrade_insecure_requests,
        request.mode,
        if request.is_imageset {
            Initiator::Imageset
        } else {
            Initiator::Normal
        },
    ) {
        MixedContentDecision::Allow => current,
        MixedContentDecision::Block => {
            return Err(NetError::MixedContentBlocked(request.url.clone()));
        }
        MixedContentDecision::Upgrade(upgraded) => upgraded,
    };
    // Blocker (Stage 9 Rec 12, Round 1 basic): EasyList /
    // EasyPrivacy network rules, before CSP per the frozen gate
    // order. An empty blocker (no lists on disk yet) allows
    // everything — never blocks page loads. The matched rule is
    // copied out while the lock is held; the error carries the
    // list, line, and pattern for the status line.
    {
        let blocker_arc = scheduler.blocker();
        let blocker = blocker_arc.lock().unwrap();
        if let Some(rule) = blocker.is_blocked(
            &current.to_string(),
            request.destination,
            &request.top_level_site,
        ) {
            return Err(NetError::BlockedByFilter(format!(
                "{:?}:{}: {}",
                rule.source, rule.line, rule.pattern
            )));
        }
    }
    // Frozen gate order (Stage 5): HSTS/mixed → blocker → CSP fetch
    // → CORS → cache → network. The blocker gate ran above; there
    // is no cache yet (M2), so the next gate is CSP, then CORS.
    //
    // CSP pre-request check (Stage 9 Rec 8): the current
    // document's policy container gates the fetch by destination
    // directive. It slots in BEFORE CORS and runs on every
    // redirect hop too (the hop path calls it again below, with
    // the hop URL).
    if !csp_pre_request_check(scheduler, request, &current.to_string()) {
        return Err(NetError::CorsBlocked(
            "CSP pre-request check refused the fetch".to_string(),
        ));
    }
    // CORS pre-request decision (Stage 9 Rec 2): the preflight, if
    // one is needed, runs BEFORE any network traffic for the actual
    // request, and the response check + filtering run after the
    // headers arrive (below). `SameOrigin` mode to another origin
    // fails here — the M4 refusal, kept as the fallback.
    cors_pre_request(scheduler, request, &current).await?;
    let mut method = Method::GET;
    // Whether the request body has been dropped by a redirect
    // method conversion (301/302 + POST → GET, 303 + non-GET/HEAD →
    // GET). Once dropped it stays dropped, and the request-body
    // headers go with it (`build_request` strips them). Pluto sends
    // no body today, so this is vacuous until M5 forms — but the
    // tracking is written now, so the strip stays correct when a
    // body exists.
    let mut body_dropped = false;
    let mut redirects = 0usize;
    loop {
        // Whether the URL being fetched is cross-origin with the
        // request origin: on such a hop the redirect strip removes
        // `Authorization` (and `Cookie`, when it exists).
        let crossed_origin = !Origin::from_url(&request.origin)
            .is_same_origin(&Origin::from_url(&current.to_string()));
        let response = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(NetError::Cancelled),
            result = tokio::time::timeout(
                TTFB_TIMEOUT,
                scheduler.client.request(build_request(
                    &method,
                    &current,
                    request,
                    scheduler,
                    crossed_origin,
                    body_dropped,
                )?),
            ) => match result {
                Ok(Ok(response)) => response,
                Ok(Err(e)) => {
                    // Stage 9 Rec 5 / RFC 6797 §8.4: a transport
                    // failure talking to a Known HSTS Host is
                    // fatal — no click-through, no HTTP fallback.
                    if current.scheme_str() == Some("https")
                        && let Some(host) = current.host()
                        && scheduler
                            .hsts
                            .lock()
                            .unwrap()
                            .is_known_shared(host, SystemTime::now())
                    {
                        return Err(NetError::HstsTlsFailure(current.to_string()));
                    }
                    if https_only_rewrote && request.destination == Destination::Document {
                        return Err(NetError::HttpsOnlyFailed(request.url.clone()));
                    }
                    return Err(NetError::Request(e.to_string()));
                }
                Err(_) => return Err(NetError::Timeout("time to first byte".to_string())),
            },
        };
        let status = response.status();
        // HSTS recording (RFC 6797 §8.1): any response received
        // over a secure transport with no TLS errors — including
        // redirect responses — may carry a `Strict-Transport-
        // Security` header. A plain-HTTP response is ignored by
        // `record` (the `secure_transport` argument is false
        // there); here the scheme tells us which it is.
        if current.scheme_str() == Some("https") {
            scheduler.record_hsts(&current, response.headers());
        }
        // CSP policy container (Stage 9 Rec 8): a document
        // response stores its `Content-Security-Policy` (+
        // report-only) headers as the current document's policy
        // container on the scheduler; subresource fetches are
        // gated against it. Redirect responses store too — the
        // final response replaces them before any subresource of
        // the new document is fetched.
        if request.destination == Destination::Document {
            let enforce: Vec<&str> = response
                .headers()
                .get_all("content-security-policy")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .collect();
            let report_only: Vec<&str> = response
                .headers()
                .get_all("content-security-policy-report-only")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .collect();
            if !enforce.is_empty() || !report_only.is_empty() {
                *scheduler.csp_policies().lock().unwrap() =
                    PolicySet::from_headers(&enforce, &report_only);
            }
        }
        // Cookie-jar intake (Stage 9 Rec 3, Fetch §3.3.6):
        // `Set-Cookie` survives CORS failure, so the jar reads it
        // here — on EVERY response, redirects included — BEFORE
        // the CORS exposure decision and BEFORE the strip below.
        // `current` is the URL the response came from.
        {
            let set_cookies: Vec<String> = response
                .headers()
                .get_all(header::SET_COOKIE)
                .iter()
                .filter_map(|v| v.to_str().ok().map(str::to_string))
                .collect();
            if !set_cookies.is_empty() {
                scheduler.jar().lock().unwrap().store_set_cookie(
                    &current.to_string(),
                    &request.top_level_site,
                    &set_cookies,
                );
            }
        }
        // The document response (not a redirect hop) sets the
        // referrer context for the document being fetched.
        let is_redirect = matches!(
            status,
            StatusCode::MOVED_PERMANENTLY
                | StatusCode::FOUND
                | StatusCode::SEE_OTHER
                | StatusCode::TEMPORARY_REDIRECT
                | StatusCode::PERMANENT_REDIRECT
        );
        if request.destination == Destination::Document && !is_redirect {
            scheduler.note_document_response(&current, response.headers());
        }
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
            // CORS redirect traps (Stage 9 Rec 2): redirects re-run
            // the whole decision — tainting is recomputed for the
            // new URL, so a same-origin `fetch()` that 302s
            // cross-origin becomes a CORS request mid-flight (the
            // final response is CORS-checked below, against the
            // final URL). Two hard failures here:
            //
            // * `SameOrigin` mode redirected cross-origin: the M4
            //   refusal, re-applied per hop.
            // * a CORS redirect whose Location URL carries
            //   credentials (`user:pass@`): a network error when
            //   the hop origin is not same-origin — and when the
            //   tainting is already `cors`, when it carries
            //   credentials at all.
            //
            // `Authorization` stripping itself happens in
            // `build_request` (via `crossed_origin`, computed at
            // the top of the loop from the new URL).
            {
                let request_origin = Origin::from_url(&request.origin);
                let hop_origin = Origin::from_url(&hop_url.to_string());
                let hop_same_origin = request_origin.is_same_origin(&hop_origin);
                if request.mode == RequestMode::SameOrigin && !hop_same_origin {
                    return Err(NetError::CorsBlocked(format!(
                        "same-origin fetch redirected cross-origin: {current} -> {hop_url}"
                    )));
                }
                if request.mode == RequestMode::Cors
                    && redirect_url_has_credentials(&hop_url)
                {
                    let current_origin = Origin::from_url(&current.to_string());
                    let tainting_is_cors = !request_origin.is_same_origin(&current_origin);
                    if !hop_same_origin || tainting_is_cors {
                        return Err(NetError::CorsBlocked(format!(
                            "CORS redirect to a URL with credentials: {hop_url}"
                        )));
                    }
                }
            }
            // The CSP pre-request check re-runs on every redirect
            // hop, against the hop URL (a redirect to a
            // policy-forbidden host is refused, not followed).
            if !csp_pre_request_check(scheduler, request, &hop_url.to_string()) {
                return Err(NetError::CorsBlocked(
                    "CSP pre-request check refused the redirect".to_string(),
                ));
            }
            // 301/302 + POST → GET, 303 + non-GET/HEAD → GET: the
            // body is dropped, and the request-body headers go with
            // it (see `build_request`).
            if (method != Method::GET && method != Method::HEAD) && hop_method == Method::GET
            {
                body_dropped = true;
            }
            // RFC 6797 §8.3: the HSTS / HTTPS-only rewrite applies
            // whenever the UA prepares to load any `http` URI,
            // including following redirects — before the gate.
            let mut hop_url = hop_url;
            https_only_rewrote |= scheduler.apply_secure_upgrade(&mut hop_url);
            current = match mixed_content_decision(
                &request.origin,
                &hop_url,
                request.destination,
                upgrade_insecure_requests,
                request.mode,
                if request.is_imageset {
                    Initiator::Imageset
                } else {
                    Initiator::Normal
                },
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

        // Mixed-download block (Stage 9 Rec 6, L2 §5.2): a download
        // from a secure context over a non-trustworthy URL is
        // blocked, not auto-upgraded — decided here, at the
        // response-headers point, against `Content-Disposition:
        // attachment`. (A top-level navigation reaches this point
        // because navigations are never mixed content at the
        // request gate; the response is what makes it a download.)
        if is_mixed_download(&request.origin, &current, response.headers()) {
            return Err(NetError::MixedContentBlocked(format!(
                "mixed download blocked: {current}"
            )));
        }
        // CORS response decision (Stage 9 Rec 2): tainting is
        // recomputed for the FINAL url (redirects re-ran the
        // decision per hop above). The CORS check runs on the
        // response headers — skipped for 304/407 — and script sees
        // ONLY the filtered response, never the internal one. A
        // failed check is a network error. (`Set-Cookie` intake
        // already ran above, before this decision — it survives
        // CORS failure.)
        let request_origin = Origin::from_url(&request.origin);
        let tainting = tainting_for(
            request.mode,
            &request_origin,
            &current,
            request.destination,
        );
        if tainting == ResponseTainting::Opaque {
            // SRI (Stage 9 Rec 14): integrity metadata + an
            // opaque response is a network error (Fetch §4.7) —
            // the bytes cannot be verified, so the resource is
            // not used.
            if request
                .integrity
                .as_deref()
                .is_some_and(|i| !i.trim().is_empty())
            {
                return Err(NetError::SriFailed(
                    "integrity metadata present but the response is opaque (cross-origin, no CORS)"
                        .to_string(),
                ));
            }
            // Opaque: type `"opaque"`, status 0, empty headers,
            // null body — nearly indistinguishable from a network
            // error; `response.type` is the only signal (and
            // Pluto's M4 `Response` has no `type` property yet, so
            // script sees status 0 with an empty body). No bytes
            // are streamed at all.
            if !scheduler
                .emit(
                    request.request_id,
                    ResourceEvent::ResponseHeaders {
                        request_id: request.request_id,
                        status: 0,
                        url: String::new(),
                        headers: Vec::new(),
                        content_length: None,
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
        let headers: Vec<(String, String)> = match tainting {
            ResponseTainting::Cors => {
                if !cors_check_is_skipped(status)
                    && !cors_check(&request_origin, request.credentials, response.headers())
                {
                    return Err(NetError::CorsBlocked(format!(
                        "CORS check failed for {current}"
                    )));
                }
                filter_cors_response_headers(response.headers(), request.credentials)
            }
            ResponseTainting::Basic => filter_basic_response_headers(response.headers()),
            ResponseTainting::Opaque => unreachable!("opaque handled above"),
        };
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
        // SRI (Stage 9 Rec 14): when the element carried an
        // `integrity` attribute, the DECODED bytes are hashed
        // incrementally as they stream; a mismatch at the end
        // fails the fetch instead of `Done`. Malformed metadata
        // parses to no entries and is ignored, per spec (the
        // fetch proceeds unverified).
        let mut sri = request
            .integrity
            .as_deref()
            .map(SriVerifier::from_attribute)
            .filter(|v| !v.is_empty());
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
                if let Some(sri) = sri.as_mut() {
                    sri.update(&decoded);
                }
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
        if let Some(sri) = sri.as_mut() {
            sri.update(&tail);
        }
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
        // SRI (Stage 9 Rec 14): the stream is complete — verify
        // before `Done`. A mismatch fails the fetch (network
        // error); the consumer only acts on the completed body,
        // so it never uses tainted bytes.
        if let Some(sri) = sri.as_ref()
            && !sri.verify()
        {
            return Err(NetError::SriFailed(format!(
                "integrity check failed for {current}"
            )));
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

/// Build one request. The header set is rebuilt fresh on every hop
/// and then stripped for that hop: `crossed_origin` removes
/// `Authorization` (and `Cookie`, when the jar exists) the moment
/// another origin is seen, and `body_dropped` removes the
/// request-body headers when a redirect dropped the body. The
/// rebuild *is* the header-stripping point; Pluto sends no
/// `Authorization` and no `Cookie` today, so the strip calls are
/// no-ops on the current header set — they are written explicitly
/// so the point stays correct when a sensitive header is ever
/// added.
///
/// CORS-mode requests also carry the UA-appended `Origin` header
/// (see `origin_header_for` in `cors.rs`): script never sets it —
/// it is a forbidden request-header name.
fn build_request(
    method: &Method,
    url: &Uri,
    request: &ResourceRequest,
    scheduler: &Scheduler,
    crossed_origin: bool,
    body_dropped: bool,
) -> Result<Request<Full<Bytes>>, NetError> {
    let accept = match request.destination {
        Destination::Document => "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        Destination::Stylesheet => "text/css,*/*;q=0.1",
        Destination::Script => "*/*",
        Destination::Image => "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        Destination::Font => "font/*,*/*;q=0.1",
        Destination::Fetch | Destination::Other => "*/*",
    };
    let mut headers = header::HeaderMap::new();
    headers.insert(
        header::USER_AGENT,
        header::HeaderValue::from_static("Pluto/0.1.0"),
    );
    headers.insert(
        header::ACCEPT,
        accept
            .parse()
            .map_err(|e| NetError::Request(format!("bad Accept header: {e}")))?,
    );
    headers.insert(
        header::ACCEPT_ENCODING,
        header::HeaderValue::from_static("gzip, deflate"),
    );
    if let Some(referer) = referer_for_request(request, url, scheduler)
        && let Ok(value) = referer.parse::<header::HeaderValue>()
    {
        headers.insert(header::REFERER, value);
    }
    if request.mode == RequestMode::Cors {
        let request_origin = Origin::from_url(&request.origin);
        if let Ok(value) =
            origin_header_for(&request_origin, url, request.credentials).parse()
        {
            headers.insert(header::ORIGIN, value);
        }
    }
    // Stage 9 Rec 2: the script's own headers ride the request
    // (they were classified — forbidden names rejected — by the
    // CORS preflight path for `Cors` mode). Forbidden names can
    // never be sent, so a same-origin fetch — where the classifier
    // never runs — cannot smuggle `Cookie` / `Host` / `Referer`
    // / `Origin` past it. Script headers override the defaults
    // above (a script-set `Accept` wins over the destination
    // default); unparseable names/values are dropped, never
    // fatal.
    for (name, value) in &request.headers {
        if is_forbidden_request_header(&name.to_ascii_lowercase(), value) {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::from_bytes(name.as_bytes()),
            header::HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }
    // Stage 9 Rec 3: the `Cookie` header comes from the jar —
    // never from script headers (forbidden above). Credentials
    // mode gates sending entirely: `omit` sends nothing,
    // `same-origin` only to the document's own origin, `include`
    // always (SameSite rules still apply inside the jar).
    // Computed for the URL actually being fetched, so a
    // cross-origin redirect hop looks its own cookies up fresh;
    // the redirect strip below still removes it when the hop
    // crossed origins.
    {
        let credentials_allow_cookies = match request.credentials {
            CredentialsMode::Omit => false,
            CredentialsMode::SameOrigin => Origin::from_url(&request.origin)
                .is_same_origin(&Origin::from_url(&url.to_string())),
            CredentialsMode::Include => true,
        };
        if credentials_allow_cookies {
            let method_is_safe = matches!(
                *method,
                Method::GET | Method::HEAD | Method::OPTIONS | Method::TRACE
            );
            let is_top_level_navigation = request.destination == Destination::Document;
            if let Some(cookie) = scheduler.jar().lock().unwrap().cookie_header(
                &url.to_string(),
                &request.top_level_site,
                url_is_secure_str(&url.to_string()),
                method_is_safe,
                is_top_level_navigation,
            ) && let Ok(value) = header::HeaderValue::from_str(&cookie)
            {
                headers.insert(header::COOKIE, value);
            }
        }
    }
    // Redirect stripping (Stage 9 Rec 2): cross-origin hop, and
    // body-drop — see the doc comment above.
    strip_sensitive_headers_on_redirect(&mut headers, crossed_origin);
    if body_dropped {
        strip_request_body_headers(&mut headers);
    }
    let mut builder = Request::builder().method(method.clone()).uri(url.clone());
    for (name, value) in headers.iter() {
        builder = builder.header(name, value);
    }
    builder
        .body(Full::new(Bytes::new()))
        .map_err(|e| NetError::Request(e.to_string()))
}

/// The `Referer` value for one request, or `None` when the policy
/// sends nothing (Stage 9 Rec 15).
///
/// The referrer source is the loader's context document URL when
/// it belongs to this request — for a subresource, the context
/// document's origin must equal the request's `origin`; for a
/// navigation, the context document *is* the referrer (the page
/// the navigation came from). With no usable context, the
/// request's `origin` string is the source: same-origin requests
/// then send the origin, not the full document URL (the frozen
/// wire shape carries no document URL — see `referrer.rs`).
/// Known deviation: a typed-URL navigation cannot be told apart
/// from a link navigation here (both arrive as `CommitNavigation`
/// with no signal), so a typed navigation also sends the previous
/// document as its referrer; telling them apart needs an IPC
/// change, deferred.
fn referer_for_request(
    request: &ResourceRequest,
    url: &Uri,
    scheduler: &Scheduler,
) -> Option<String> {
    let request_url = Url::parse(&url.to_string()).ok()?;
    let ctx = scheduler.referrer_context().lock().unwrap().clone();
    let context_doc = ctx
        .document_url
        .as_deref()
        .and_then(|doc| Url::parse(doc).ok())
        .filter(|doc| {
            if request.destination == Destination::Document {
                true
            } else {
                pluto_ipc::origin_of(doc.as_str()) == request.origin
            }
        });
    // A navigation with no previous document sends no Referer at
    // all (a typed URL has no referrer); the `origin` fallback
    // below is only for subresources, whose `origin` *is* the
    // referring document's origin.
    let source = match context_doc {
        Some(doc) => doc,
        None if request.destination == Destination::Document => return None,
        None => Url::parse(&request.origin).ok()?,
    };
    referer_for(&source, &request_url, ctx.policy)
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
                    // Stage 9 Rec 2 (CORS): canned CORS responses.
                    // `/cors-ok` grants any origin; `/cors-none`
                    // grants none; `/cors-echo` grants exactly
                    // `http://example.com` with credentials;
                    // `/cors-star-cred` grants `*` with credentials
                    // (which must still fail for `include`);
                    // `/cors-expose` exposes one custom header.
                    "/cors-ok" => (
                        "200 OK",
                        "Content-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nX-Custom: secret"
                            .into(),
                        b"cors-body".to_vec(),
                    ),
                    "/cors-none" => (
                        "200 OK",
                        "Content-Type: text/plain".into(),
                        b"no-cors".to_vec(),
                    ),
                    "/cors-echo" => (
                        "200 OK",
                        "Content-Type: text/plain\r\nAccess-Control-Allow-Origin: http://example.com\r\nAccess-Control-Allow-Credentials: true"
                            .into(),
                        b"echo".to_vec(),
                    ),
                    "/cors-star-cred" => (
                        "200 OK",
                        "Content-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Credentials: true"
                            .into(),
                        b"star".to_vec(),
                    ),
                    "/cors-expose" => (
                        "200 OK",
                        "Content-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: X-Custom\r\nX-Custom: shown\r\nX-Hidden: no"
                            .into(),
                        b"exposed".to_vec(),
                    ),
                    // Stage 9 Rec 6: a download response.
                    "/download" => (
                        "200 OK",
                        "Content-Type: application/zip\r\nContent-Disposition: attachment; filename=\"f.zip\""
                            .into(),
                        b"zipbytes".to_vec(),
                    ),
                    // Stage 9 Rec 3 (cookie jar): sets one cookie;
                    // the intake runs before the `Set-Cookie`
                    // strip, so the jar keeps it while the
                    // renderer never sees the header.
                    "/set-cookie" => (
                        "200 OK",
                        "Content-Type: text/plain\r\nSet-Cookie: session=abc123; Path=/"
                            .into(),
                        b"set".to_vec(),
                    ),
                    // Stage 9 Rec 3: echoes the request's `Cookie:`
                    // header value back as the body (`(none)` when
                    // the request carried none).
                    "/echo-cookie" => {
                        let cookie = request
                            .lines()
                            .find(|l| l.to_ascii_lowercase().starts_with("cookie:"))
                            .map(|l| l[7..].trim().to_string())
                            .unwrap_or_else(|| "(none)".to_string());
                        ("200 OK", "Content-Type: text/plain".into(), cookie.into_bytes())
                    }
                    // Stage 9 Rec 8 (CSP): a document whose policy
                    // forbids every image (`img-src 'none'`).
                    "/csp-doc" => (
                        "200 OK",
                        "Content-Type: text/html\r\nContent-Security-Policy: img-src 'none'"
                            .into(),
                        b"doc".to_vec(),
                    ),
                    // A redirect to an explicit target: the test
                    // passes the absolute target URL as
                    // `/redir?to=<url>` so one server can redirect
                    // onto a second server (a different origin).
                    p if p.starts_with("/redir?to=") => {
                        let target = p.trim_start_matches("/redir?to=");
                        (
                            "302 Found",
                            format!("Location: {target}"),
                            Vec::new(),
                        )
                    }
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
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
            destination,
            priority,
            mode: pluto_ipc::RequestMode::NoCors,
            credentials: CredentialsMode::SameOrigin,
            origin: origin.clone(),
            top_level_site: origin,
            upgrade_insecure_requests: false,
            integrity: None,
        }
    }

    /// A request with an explicit document origin / mode /
    /// credentials — the CORS tests' requests are cross-origin
    /// with the local test server by construction (the document
    /// origin is `http://example.com`, the server is
    /// `http://127.0.0.1:<port>`).
    fn cors_request(
        id: u64,
        url: &str,
        origin: &str,
        mode: pluto_ipc::RequestMode,
        credentials: CredentialsMode,
        destination: Destination,
    ) -> ResourceRequest {
        ResourceRequest {
            request_id: id,
            tab_id: 0,
            epoch: 1,
            url: url.to_string(),
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
            destination,
            priority: Priority::Medium,
            mode,
            credentials,
            origin: origin.to_string(),
            top_level_site: origin.to_string(),
            upgrade_insecure_requests: false,
            integrity: None,
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

    // --- Stage 9 Rec 2: CORS through the live pipeline ---

    #[tokio::test]
    async fn cors_simple_cross_origin_fetch_succeeds_and_filters_headers() {
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-ok"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"cors-body");
        assert!(terminal.is_none(), "CORS fetch must succeed: {terminal:?}");
        // The filtered response keeps CORS-safelisted headers...
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "text/plain"),
            "content-type is safelisted: {headers:?}"
        );
        // ...and hides unexposed ones from script.
        assert!(
            headers.iter().all(|(k, _)| k != "x-custom"),
            "unexposed x-custom must be filtered: {headers:?}"
        );
    }

    #[tokio::test]
    async fn cors_fetch_without_allow_origin_fails() {
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-none"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("cors_blocked")),
            "expected cors_blocked, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn cors_wildcard_plus_include_credentials_fails() {
        // `Access-Control-Allow-Origin: *` with `credentials:
        // include` ALWAYS fails, even with `Allow-Credentials:
        // true` (trap 1).
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-star-cred"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::Include,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("cors_blocked")),
            "expected cors_blocked, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn cors_exact_origin_plus_include_credentials_succeeds() {
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-echo"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::Include,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (status, _, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"echo");
        assert!(terminal.is_none(), "exact-origin + include must pass: {terminal:?}");
    }

    #[tokio::test]
    async fn cors_expose_headers_filtering() {
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-expose"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"exposed");
        assert!(terminal.is_none());
        assert!(
            headers.iter().any(|(k, v)| k == "x-custom" && v == "shown"),
            "exposed x-custom must be visible: {headers:?}"
        );
        assert!(
            headers.iter().all(|(k, _)| k != "x-hidden"),
            "unexposed x-hidden must stay hidden: {headers:?}"
        );
    }

    #[tokio::test]
    async fn nocors_cross_origin_fetch_is_opaque() {
        // A `no-cors` fetch() cross-origin delivers the opaque
        // shape even when the server would grant CORS: status 0,
        // empty headers, null body — not the real response.
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}cors-ok"),
            "http://example.com",
            pluto_ipc::RequestMode::NoCors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 0);
        assert!(headers.is_empty(), "opaque headers must be empty: {headers:?}");
        assert!(body.is_empty(), "opaque body must be null/empty");
        assert!(terminal.is_none(), "opaque is a Done, not an error: {terminal:?}");
    }

    #[tokio::test]
    async fn cors_redirect_recomputes_tainting() {
        // A same-origin CORS fetch that 302s cross-origin becomes
        // a CORS request mid-flight: the final response is checked
        // against the final URL's origin.
        let target_base = test_server().await;
        let base = test_server().await;
        let sched = scheduler();
        // Redirect onto the second server's granting endpoint.
        let req = cors_request(
            1,
            &format!("{base}redir?to={target_base}cors-ok"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (status, _, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"cors-body");
        assert!(terminal.is_none(), "redirect to a granting origin must pass: {terminal:?}");
        // Redirect onto the second server's non-granting endpoint:
        // the recomputed check fails.
        let req = cors_request(
            2,
            &format!("{base}redir?to={target_base}cors-none"),
            "http://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("cors_blocked")),
            "expected cors_blocked after cross-origin redirect, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn same_origin_mode_redirect_cross_origin_fails() {
        // The initial fetch is same-origin (the document origin IS
        // the first server's origin), so the pre-request gate
        // passes — but the 302 onto a second server (a different
        // origin: different port) re-runs the decision and fails.
        let target_base = test_server().await;
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}redir?to={target_base}ok"),
            base.trim_end_matches('/'),
            pluto_ipc::RequestMode::SameOrigin,
            CredentialsMode::SameOrigin,
            Destination::Fetch,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("cors_blocked")),
            "expected cors_blocked, got {terminal:?}"
        );
    }

    // --- Stage 9 Rec 6: mixed download + CORS-mode image ---

    #[tokio::test]
    async fn mixed_download_blocked_at_response() {
        // A top-level navigation (never mixed content at the
        // request gate) whose response says `attachment` is a
        // download from a secure context over http: blocked, not
        // auto-upgraded.
        let base = test_server().await;
        let sched = scheduler();
        let req = cors_request(
            1,
            &format!("{base}download"),
            "https://example.com",
            pluto_ipc::RequestMode::Navigate,
            CredentialsMode::SameOrigin,
            Destination::Document,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("mixed_content")),
            "expected mixed_content (download) block, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn cors_mode_image_blocked_but_plain_image_upgrades() {
        // HTTPS-only would rewrite both URLs before the mixed gate
        // sees them — disable it to isolate the gate.
        let base = test_server().await;
        let sched = scheduler();
        sched.https_only().lock().unwrap().enabled = false;
        // `localhost` (a hostname, not an IP literal) so the gate
        // — not the IP rule — decides.
        let port = base
            .trim_start_matches("http://127.0.0.1:")
            .trim_end_matches('/');
        let image_url = format!("http://localhost:{port}/x.png");
        // CORS-mode image: BLOCKED, never upgraded.
        let req = cors_request(
            1,
            &image_url,
            "https://example.com",
            pluto_ipc::RequestMode::Cors,
            CredentialsMode::SameOrigin,
            Destination::Image,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal
                .as_deref()
                .is_some_and(|t| t.starts_with("mixed_content")),
            "expected mixed_content block for a CORS-mode image, got {terminal:?}"
        );
        // Plain image: UPGRADED to https — the fetch then fails at
        // TLS (the test server speaks plain HTTP), which proves the
        // gate upgraded instead of blocking: the error is a
        // transport error, not a mixed-content block.
        let req = cors_request(
            2,
            &image_url,
            "https://example.com",
            pluto_ipc::RequestMode::NoCors,
            CredentialsMode::SameOrigin,
            Destination::Image,
        );
        let rx = sched.submit(req, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        assert!(
            terminal.as_deref().is_some_and(|t| t.starts_with("request")),
            "plain image must upgrade (TLS failure), not block: got {terminal:?}"
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

    // --- Stage 9 Rec 5 / 15: front-of-pipeline upgrades + Referer ---

    fn scheduler_with_hsts(host: &str, header: &str) -> Arc<Scheduler> {
        let sched = scheduler();
        sched
            .hsts_store()
            .lock()
            .unwrap()
            .record(host, true, Some(header), SystemTime::now());
        sched
    }

    #[test]
    fn hsts_rewrite_runs_before_the_mixed_content_gate() {
        // An https page loading an http script would be BLOCKED
        // by the mixed-content gate — unless the host is a Known
        // HSTS Host, in which case the front-of-pipeline rewrite
        // upgrades it first and the gate then allows it.
        let sched = scheduler_with_hsts("example.com", "max-age=31536000");
        let mut url: Uri = "http://example.com/x.js".parse().unwrap();
        assert!(!sched.apply_secure_upgrade(&mut url));
        assert_eq!(url.to_string(), "https://example.com/x.js");
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &url,
                Destination::Script,
                false,
                pluto_ipc::RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Allow
        );
        // Without the HSTS entry the same ask is blocked.
        let plain = scheduler();
        let mut url2: Uri = "http://example.com/x.js".parse().unwrap();
        // HTTPS-only is on by default, so a bare http host WOULD
        // be rewritten by it too — disable it to isolate HSTS.
        plain.https_only().lock().unwrap().enabled = false;
        assert!(!plain.apply_secure_upgrade(&mut url2));
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &url2,
                Destination::Script,
                false,
                pluto_ipc::RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
    }

    #[test]
    fn https_only_rewrite_applies_where_hsts_does_not() {
        let sched = scheduler();
        let mut url: Uri = "http://example.com:80/x".parse().unwrap();
        assert!(sched.apply_secure_upgrade(&mut url));
        assert_eq!(url.to_string(), "https://example.com:443/x");
        // An exception bypasses it.
        sched
            .https_only()
            .lock()
            .unwrap()
            .add_exception("example.com");
        let mut url2: Uri = "http://example.com/x".parse().unwrap();
        assert!(!sched.apply_secure_upgrade(&mut url2));
        assert_eq!(url2.to_string(), "http://example.com/x");
        // IP literals never rewrite (no certificate to upgrade to).
        let mut ip: Uri = "http://127.0.0.1:8080/x".parse().unwrap();
        assert!(!sched.apply_secure_upgrade(&mut ip));
    }

    #[test]
    fn hsts_beats_https_only_for_the_same_host() {
        // Both would rewrite to the same URL; the point is the
        // HSTS path is consulted first (and the return value says
        // the rewrite was HSTS, not HTTPS-only, so a later failure
        // is not misreported as an HTTPS-only failure).
        let sched = scheduler_with_hsts("example.com", "max-age=31536000");
        let mut url: Uri = "http://example.com/x".parse().unwrap();
        assert!(!sched.apply_secure_upgrade(&mut url));
        assert_eq!(url.to_string(), "https://example.com/x");
    }

    #[test]
    fn referer_uses_the_context_document_url() {
        let sched = scheduler();
        sched
            .referrer_context()
            .lock()
            .unwrap()
            .document_url = Some("https://example.com/page?q=1".to_string());
        // Same-origin subresource: full URL.
        let req = request(1, "https://example.com/x.png", Destination::Image, Priority::Low);
        let referer = referer_for_request(
            &req,
            &"https://example.com/x.png".parse().unwrap(),
            &sched,
        );
        assert_eq!(referer.as_deref(), Some("https://example.com/page?q=1"));
        // Cross-origin subresource: origin only.
        let referer = referer_for_request(
            &req,
            &"https://cdn.example.org/x.png".parse().unwrap(),
            &sched,
        );
        assert_eq!(referer.as_deref(), Some("https://example.com"));
        // Downgrade: nothing.
        let referer = referer_for_request(
            &req,
            &"http://cdn.example.org/x.png".parse().unwrap(),
            &sched,
        );
        assert_eq!(referer, None);
    }

    #[test]
    fn referer_navigation_without_a_previous_document_is_none() {
        let sched = scheduler();
        let req = request(
            1,
            "https://example.com/",
            Destination::Document,
            Priority::VeryHigh,
        );
        assert_eq!(
            referer_for_request(&req, &"https://example.com/".parse().unwrap(), &sched),
            None
        );
    }

    #[test]
    fn referer_without_context_falls_back_to_the_origin() {
        // A subresource ask with no context (fresh loader, JS
        // fetch): the request's origin is the referrer source, so
        // same-origin sends the origin, not a full URL — the
        // documented wire-shape limitation.
        let sched = scheduler();
        let mut req = request(1, "https://example.com/x.js", Destination::Script, Priority::High);
        req.origin = "https://example.com".to_string();
        let referer = referer_for_request(
            &req,
            &"https://example.com/x.js".parse().unwrap(),
            &sched,
        );
        // The origin serialises with its (empty) path: the
        // referrer source here is the bare origin URL.
        assert_eq!(referer.as_deref(), Some("https://example.com/"));
    }

    #[test]
    fn referer_policy_header_is_recorded_from_document_responses() {
        let sched = scheduler();
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::REFERRER_POLICY,
            "strict-origin-when-cross-origin".parse().unwrap(),
        );
        sched.note_document_response(
            &"https://example.com/page".parse().unwrap(),
            &headers,
        );
        let ctx = sched.referrer_context().lock().unwrap().clone();
        assert_eq!(
            ctx.document_url.as_deref(),
            Some("https://example.com/page")
        );
        assert_eq!(
            ctx.policy,
            ReferrerPolicy::StrictOriginWhenCrossOrigin
        );
        // A document with no header resets to the default.
        let empty = header::HeaderMap::new();
        sched.note_document_response(&"https://other.com/".parse().unwrap(), &empty);
        let ctx = sched.referrer_context().lock().unwrap().clone();
        assert_eq!(ctx.policy, ReferrerPolicy::NoReferrerWhenDowngrade);
    }

    // Stage 9 Rec 14 (SRI): the `integrity` attribute rides the
    // request; matching bytes pass, tampered bytes fail the fetch
    // (network error, no `Done`).
    #[tokio::test]
    async fn sri_match_passes_and_mismatch_fails() {
        use base64::Engine as _;
        let base = test_server().await;
        let sched = scheduler();
        let sha256_b64 = |bytes: &[u8]| {
            let digest = ring::digest::digest(&ring::digest::SHA256, bytes);
            format!(
                "sha256-{}",
                base64::engine::general_purpose::STANDARD.encode(digest.as_ref())
            )
        };
        // `/ok` serves `hello`.
        let mut good = request(
            1,
            &format!("{base}ok"),
            Destination::Script,
            Priority::High,
        );
        good.integrity = Some(sha256_b64(b"hello"));
        let rx = sched.submit(good, CancellationToken::new()).await;
        let (status, _, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"hello");
        assert!(terminal.is_none(), "matching integrity passes: {terminal:?}");

        let mut bad = request(
            2,
            &format!("{base}ok"),
            Destination::Script,
            Priority::High,
        );
        bad.integrity = Some(sha256_b64(b"tampered"));
        let rx = sched.submit(bad, CancellationToken::new()).await;
        let (_, _, _, terminal) = collect(rx).await;
        let terminal = terminal.expect("mismatching integrity fails the fetch");
        assert!(
            terminal.starts_with("sri_failed:"),
            "SRI mismatch is an sri_failed network error: {terminal}"
        );
    }

    // Stage 9 Rec 3 (cookie jar): a `Set-Cookie` response stores
    // the cookie (intake before the strip), the next fetch sends
    // it back, and the renderer never sees `Set-Cookie`.
    #[tokio::test]
    async fn cookie_jar_round_trip_through_fetch() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}set-cookie"),
                    Destination::Document,
                    Priority::High,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, headers, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"set");
        assert!(terminal.is_none());
        assert!(
            !headers.iter().any(|(k, _)| k == "set-cookie"),
            "Set-Cookie is stripped before the renderer sees it: {headers:?}"
        );
        let rx = sched
            .submit(
                request(
                    2,
                    &format!("{base}echo-cookie"),
                    Destination::Fetch,
                    Priority::High,
                ),
                CancellationToken::new(),
            )
            .await;
        let (_, _, body, terminal) = collect(rx).await;
        assert!(terminal.is_none());
        assert_eq!(body, b"session=abc123", "jar sends the stored cookie back");
    }

    // Stage 9 Rec 8 (CSP): a document response stores its policy
    // container; a later subresource fetch against a forbidden
    // directive is refused before any network traffic, while an
    // ungoverned directive still fetches.
    #[tokio::test]
    async fn csp_document_policy_blocks_subresource() {
        let base = test_server().await;
        let sched = scheduler();
        let rx = sched
            .submit(
                request(
                    1,
                    &format!("{base}csp-doc"),
                    Destination::Document,
                    Priority::VeryHigh,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, _, _, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert!(terminal.is_none());
        assert!(
            !sched.csp_policies().lock().unwrap().is_empty(),
            "the document response stored its policy container"
        );
        // `img-src 'none'` refuses the image fetch.
        let rx = sched
            .submit(
                request(2, &format!("{base}ok"), Destination::Image, Priority::High),
                CancellationToken::new(),
            )
            .await;
        let (_, _, _, terminal) = collect(rx).await;
        let terminal = terminal.expect("img-src 'none' blocks the image fetch");
        assert!(
            terminal.starts_with("cors_blocked:")
                && terminal.contains("CSP pre-request check refused"),
            "CSP refusal is a cors_blocked network error: {terminal}"
        );
        // A stylesheet is not governed by `img-src` (no
        // `style-src` / `default-src` in the policy) — it fetches.
        let rx = sched
            .submit(
                request(
                    3,
                    &format!("{base}ok"),
                    Destination::Stylesheet,
                    Priority::High,
                ),
                CancellationToken::new(),
            )
            .await;
        let (status, _, body, terminal) = collect(rx).await;
        assert_eq!(status, 200);
        assert_eq!(body, b"hello");
        assert!(terminal.is_none());
    }

    // Stage 9 Rec 12 (blocker, Round 1 basic): a loaded rule
    // refuses the matching fetch before any network traffic.
    #[tokio::test]
    async fn blocker_gate_refuses_matching_fetch() {
        let base = test_server().await;
        let sched = scheduler();
        sched
            .blocker()
            .lock()
            .unwrap()
            .load_list(crate::blocker::ListSource::EasyList, "||127.0.0.1^\n");
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
        let (_, _, _, terminal) = collect(rx).await;
        let terminal = terminal.expect("the blocker refuses the matching fetch");
        assert!(
            terminal.starts_with("blocked_by_filter:"),
            "blocker refusal is a blocked_by_filter network error: {terminal}"
        );
    }

    // Stage 9 Rec 2/3 (`build_request`): the script's own headers
    // ride the request, forbidden names never do, and the `Cookie`
    // header comes from the jar — gated by credentials mode.
    #[test]
    fn build_request_forwards_script_headers_but_not_forbidden_ones() {
        let sched = scheduler();
        let url: Uri = "http://127.0.0.1:9/ok".parse().unwrap();
        let mut req = request(
            1,
            "http://127.0.0.1:9/ok",
            Destination::Fetch,
            Priority::High,
        );
        req.mode = pluto_ipc::RequestMode::Cors;
        req.headers = vec![
            ("X-Custom".to_string(), "yes".to_string()),
            ("Cookie".to_string(), "injected=1".to_string()),
            ("Host".to_string(), "evil.example.com".to_string()),
        ];
        // A cookie in the jar, for the same URL.
        sched.jar().lock().unwrap().store_set_cookie(
            "http://127.0.0.1:9/ok",
            "http://127.0.0.1:9",
            &["session=abc123; Path=/".to_string()],
        );
        let built = build_request(&Method::GET, &url, &req, &sched, false, false).unwrap();
        let headers = built.headers();
        assert_eq!(
            headers.get("x-custom").and_then(|v| v.to_str().ok()),
            Some("yes"),
            "a script-set safelisted header rides the request"
        );
        assert_eq!(
            headers.get("cookie").and_then(|v| v.to_str().ok()),
            Some("session=abc123"),
            "the jar's cookie is sent — the script's Cookie never reaches the wire"
        );
        assert!(
            headers.get("host").is_none(),
            "a script-set Host never reaches the wire"
        );
        // `omit` credentials send no cookies at all, even with a
        // jar entry for the URL.
        let mut omit = req.clone();
        omit.credentials = CredentialsMode::Omit;
        let built = build_request(&Method::GET, &url, &omit, &sched, false, false).unwrap();
        assert!(
            built.headers().get("cookie").is_none(),
            "credentials omit sends no Cookie header"
        );
    }
}
