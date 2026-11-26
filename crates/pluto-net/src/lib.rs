//! Networking (layer 2): `ResourceLoader`.
//!
//! Fetches HTML, CSS, JS, and images over TLS 1.3 (rustls +
//! webpki-roots), follows redirects (301/302/303/307/308), and
//! streams bytes to the parser. HTTP/1.1 and HTTP/2 are both
//! enabled; the server picks (ALPN `[h2, http/1.1]`).
//!
//! The shape, since Stage 5:
//! - ONE shared hyper client for the whole browser process
//!   (`client::shared_client`): explicit pool caps, one rustls
//!   config, shared resumption, 0-RTT off. A fresh client per
//!   navigation (the M1 behaviour) defeated pooling and resumption.
//! - A scheduler (`service::Scheduler`): priority queue,
//!   per-origin in-flight caps (6), dedup, timeouts. Every fetch
//!   goes through it as a `ResourceRequest` and streams back
//!   `ResourceEvent`s; `fetch()` below is just the collecting
//!   consumer of that stream.
//! - The renderer never fetches directly: it asks via
//!   `RequestResource` and the browser fetches. `ResourceLoader`
//!   is the browser-side API for that.
//!
//! Sandbox note: this VM only reaches the internet through an HTTP proxy
//! and hyper does not speak proxies, so a live TLS fetch to example.com
//! cannot be proven here -- it is compile-verified in the sandbox and
//! the live proof belongs on Joel's box. The redirect/body logic *is*
//! proven here, against a local test server (see the tests below).

pub mod cache;
pub mod client;
pub mod blocker;
pub mod cors;
pub mod csp;
pub mod decode;
pub mod dns;
pub mod hsts;
pub mod https_only;
pub mod jar;
pub mod mixed;
pub mod redirect;
pub mod referrer;
pub mod service;
pub mod sri;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use http::header::{self, HeaderMap};
use hyper::StatusCode;
use pluto_ipc::{Destination, ResourceEvent, ResourceRequest};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use service::Scheduler;

pub use blocker::Blocker;
pub use cache::{CacheStats, HttpCache, Lookup};
pub use client::{HttpsClient, shared_client};
pub use dns::{CachedResolver, DNS_NEGATIVE_TTL, DNS_POSITIVE_TTL, DnsCache};
pub use cors::PreflightCache;
pub use hsts::HstsStore;
pub use https_only::HttpsOnly;
pub use jar::CookieJar;
pub use referrer::{ReferrerContext, ReferrerPolicy};
pub use service::SharedLoader;

#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("request failed: {0}")]
    Request(String),
    #[error("failed to read response body: {0}")]
    Body(String),
    #[error("too many redirects (limit {0})")]
    TooManyRedirects(usize),
    /// The fetch was cancelled (navigation epoch bump, explicit
    /// `CancelResource`, tab close, or nobody listening anymore).
    #[error("fetch cancelled")]
    Cancelled,
    /// The decoded body passed the per-destination byte cap.
    #[error("response too large for {destination:?} (cap {cap} bytes)")]
    TooLarge {
        destination: Destination,
        cap: usize,
    },
    /// A blockable http subresource on an https page (Stage 5 #6).
    #[error("mixed content blocked: {0}")]
    MixedContentBlocked(String),
    /// A CORS failure (Stage 9 Rec 2): the preflight failed, the
    /// response failed the CORS check, or the request's mode does
    /// not permit the crossing (e.g. `SameOrigin` mode to another
    /// origin, or an opaque origin attempting a CORS fetch). Fail
    /// closed: a CORS failure is a network error, never a
    /// pass-through, and script sees only the rejection.
    #[error("CORS blocked: {0}")]
    CorsBlocked(String),
    /// A redirect downgraded https to http (Stage 5 #7).
    #[error("insecure redirect refused: {0}")]
    InsecureRedirect(String),
    /// A per-request timeout fired (time to first byte, or total).
    #[error("timed out: {0}")]
    Timeout(String),
    /// A transport (TLS/cert) failure talking to a Known HSTS Host
    /// (Stage 9 Rec 5, RFC 6797 §8.4): the connection is terminated
    /// with no click-through and no HTTP fallback. There is no
    /// click-through UI in Pluto at all, so this is simply fatal.
    #[error("TLS error on a Known HSTS Host {0}: connection terminated, no fallback to HTTP")]
    HstsTlsFailure(String),
    /// An HTTPS-only rewrite (Stage 9 Rec 5) of a top-level
    /// navigation failed: the secure version could not be reached.
    /// The browser shows its error page with this reason; the
    /// escape hatch is a per-site HTTPS-only exception.
    #[error("HTTPS-only upgrade failed for {0}: the secure version could not be reached (add a per-site exception to load it over HTTP)")]
    HttpsOnlyFailed(String),
    /// A request blocked by the tracker filter lists (Stage 9
    /// Rec 12, Round 1 basic): an EasyList / EasyPrivacy network
    /// rule matched. A network error, never a pass-through.
    #[error("blocked by filter list: {0}")]
    BlockedByFilter(String),
    /// Subresource Integrity verification failed (Stage 9 Rec 14):
    /// the fetched bytes did not match the element's `integrity`
    /// metadata. A network error — the resource is not used.
    #[error("subresource integrity check failed: {0}")]
    SriFailed(String),
}

impl NetError {
    /// Machine-readable code for the `ResourceEvent::Error` wire
    /// event, so `fetch()` can map a streamed error back to the
    /// typed variant it came from.
    pub fn code(&self) -> &'static str {
        match self {
            NetError::InvalidUrl(_) => "invalid_url",
            NetError::Request(_) => "request",
            NetError::Body(_) => "body",
            NetError::TooManyRedirects(_) => "too_many_redirects",
            NetError::Cancelled => "cancelled",
            NetError::TooLarge { .. } => "too_large",
            NetError::MixedContentBlocked(_) => "mixed_content",
            NetError::CorsBlocked(_) => "cors_blocked",
            NetError::InsecureRedirect(_) => "insecure_redirect",
            NetError::Timeout(_) => "timeout",
            NetError::HstsTlsFailure(_) => "hsts_tls_error",
            NetError::HttpsOnlyFailed(_) => "https_only_failed",
            NetError::BlockedByFilter(_) => "blocked_by_filter",
            NetError::SriFailed(_) => "sri_failed",
        }
    }

    fn from_code(code: &str, reason: &str) -> Self {
        match code {
            "invalid_url" => NetError::InvalidUrl(reason.to_string()),
            "too_many_redirects" => NetError::TooManyRedirects(5),
            "cancelled" => NetError::Cancelled,
            "too_large" => NetError::Body(reason.to_string()),
            "mixed_content" => NetError::MixedContentBlocked(reason.to_string()),
            "cors_blocked" => NetError::CorsBlocked(reason.to_string()),
            "insecure_redirect" => NetError::InsecureRedirect(reason.to_string()),
            "timeout" => NetError::Timeout(reason.to_string()),
            "hsts_tls_error" => NetError::HstsTlsFailure(reason.to_string()),
            "https_only_failed" => NetError::HttpsOnlyFailed(reason.to_string()),
            "blocked_by_filter" => NetError::BlockedByFilter(reason.to_string()),
            "sri_failed" => NetError::SriFailed(reason.to_string()),
            "body" => NetError::Body(reason.to_string()),
            _ => NetError::Request(reason.to_string()),
        }
    }
}

/// A completed fetch: status, headers, raw bytes, and the URL the bytes
/// actually came from (after redirects).
#[derive(Debug)]
pub struct Response {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub final_url: http::Uri,
}

impl Response {
    /// Body as text, replacing invalid UTF-8 (what the HTML parser gets).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    pub fn content_type(&self) -> Option<&str> {
        self.headers.get(header::CONTENT_TYPE)?.to_str().ok()
    }
}

/// The browser-side fetch API. Owns a scheduler (priority queue,
/// per-origin caps, dedup) over the one shared hyper client.
/// Cloneable: clones share the scheduler state and the client, so
/// a worker can hand its loader to spawned navigation tasks.
/// `max_redirects` is the chain budget (5), counted across the
/// whole redirect chain.
#[derive(Clone)]
pub struct ResourceLoader {
    scheduler: Arc<Scheduler>,
    pub max_redirects: usize,
}

impl Default for ResourceLoader {
    fn default() -> Self {
        Self::new()
    }
}

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

impl ResourceLoader {
    pub fn new() -> Self {
        Self::from_shared(&SharedLoader::new())
    }

    /// Build a per-tab loader sharing the stores of `shared`
    /// (Stage 11 Rec 5): HSTS, blocker, cookie jar and HTTP cache
    /// are the handle's Arcs; the rest of the scheduler state is
    /// private to this tab.
    pub fn from_shared(shared: &SharedLoader) -> Self {
        Self {
            scheduler: Arc::new(Scheduler::with_shared(shared_client().clone(), 5, shared)),
            max_redirects: 5,
        }
    }

    /// Fetch a URL as a top-level navigation (M1 API, kept):
    /// highest priority, document destination. The body is
    /// collected through the streaming path — the cap still
    /// applies, so a malicious document cannot OOM the browser.
    /// A 404 is a page, not a transport failure; only transport
    /// problems, bad URLs, and redirect loops are [`NetError`].
    pub async fn fetch(&self, url: &str) -> Result<Response, NetError> {
        let request =
            ResourceRequest::navigate(NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed), 0, 0, url);
        self.fetch_with(request, CancellationToken::new()).await
    }

    /// Fetch a full `ResourceRequest`, collecting the streamed
    /// events into one [`Response`]. `cancel` cancels the fetch
    /// (the caller's token — cancelling it cancels the in-flight
    /// request; dropping it does not).
    pub async fn fetch_with(
        &self,
        request: ResourceRequest,
        cancel: CancellationToken,
    ) -> Result<Response, NetError> {
        self.fetch_with_progress(request, cancel, |_, _| {}).await
    }

    /// Like [`fetch_with`](Self::fetch_with), but `on_progress`
    /// is called with `(received_bytes, total_bytes)` as the body
    /// streams in (Stage 11 Rec 9): `total` comes from the
    /// `Content-Length` response header when present and is `None`
    /// otherwise (indeterminate — chunked / compressed). The
    /// callback runs on the fetch task, so it must be cheap and
    /// non-blocking (the worker forwards to the shell over IPC).
    pub async fn fetch_with_progress(
        &self,
        request: ResourceRequest,
        cancel: CancellationToken,
        on_progress: impl Fn(u64, Option<u64>) + Send + 'static,
    ) -> Result<Response, NetError> {
        let final_url_hint = request.url.clone();
        let mut events = self.fetch_stream(request, cancel).await;
        let mut status = StatusCode::OK;
        let mut headers = HeaderMap::new();
        let mut body = Vec::new();
        let mut total: Option<u64> = None;
        let mut received: u64 = 0;
        let mut final_url = final_url_hint
            .parse()
            .unwrap_or_else(|_| http::Uri::from_static("/"));
        while let Some(event) = events.recv().await {
            match event {
                ResourceEvent::ResponseHeaders {
                    status: s,
                    url,
                    headers: h,
                    ..
                } => {
                    status = StatusCode::from_u16(s).unwrap_or(StatusCode::OK);
                    if let Ok(u) = url.parse() {
                        final_url = u;
                    }
                    for (name, value) in h {
                        if let (Ok(n), Ok(v)) = (
                            name.parse::<header::HeaderName>(),
                            value.parse::<header::HeaderValue>(),
                        ) {
                            if n == header::CONTENT_LENGTH {
                                total = v.to_str().ok().and_then(|s| s.parse().ok());
                            }
                            headers.insert(n, v);
                        }
                    }
                    on_progress(0, total);
                }
                ResourceEvent::BodyChunk { bytes, .. } => {
                    received += bytes.len() as u64;
                    on_progress(received, total);
                    body.extend_from_slice(&bytes);
                }
                ResourceEvent::Done { .. } => {
                    return Ok(Response {
                        status,
                        headers,
                        body,
                        final_url,
                    });
                }
                ResourceEvent::Error { code, reason, .. } => {
                    if code == "too_many_redirects" {
                        return Err(NetError::TooManyRedirects(self.max_redirects));
                    }
                    return Err(NetError::from_code(&code, &reason));
                }
                ResourceEvent::Cancelled { .. } => return Err(NetError::Cancelled),
            }
        }
        Err(NetError::Request(
            "fetch stream ended without a terminal event".to_string(),
        ))
    }

    /// Stream a resource ask: returns the event receiver. The
    /// caller reads `ResponseHeaders`, then `BodyChunk`s, then
    /// exactly one terminal event. Cancelling `cancel` cancels the
    /// fetch.
    pub async fn fetch_stream(
        &self,
        request: ResourceRequest,
        cancel: CancellationToken,
    ) -> mpsc::Receiver<ResourceEvent> {
        self.scheduler.submit(request, cancel).await
    }

    /// Cancel an in-flight or queued request by id.
    pub async fn cancel(&self, request_id: u64) {
        self.scheduler.cancel(request_id).await;
    }

    /// The shared HSTS store this loader's scheduler consults
    /// (Stage 9 Rec 5). `pluto-browser` loads it from the profile
    /// at startup and saves it back after navigations.
    pub fn hsts_store(&self) -> Arc<std::sync::Mutex<HstsStore>> {
        self.scheduler.hsts_store()
    }

    /// The current document's CSP policy container (Stage 9 Rec 8,
    /// Stage 15 Rec 11): empty until a document response stores
    /// one. The tab thread reads it to gate inline script / style
    /// execution; the fetch-side gate reads the same container.
    pub fn csp_policies(&self) -> Arc<std::sync::Mutex<crate::csp::PolicySet>> {
        self.scheduler.csp_policies()
    }

    /// The shared cookie jar this loader's scheduler consults
    /// (Stage 13 Rec 9). `pluto-browser` loads it from
    /// `cookies.json` at startup and saves it back after
    /// navigations (session cookies stripped before the save).
    pub fn cookie_jar(&self) -> Arc<std::sync::Mutex<CookieJar>> {
        self.scheduler.jar()
    }

    /// Replace this loader's HSTS store contents (profile load).
    pub fn set_hsts_store(&self, store: HstsStore) {
        *self.scheduler.hsts_store().lock().unwrap() = store;
    }

    /// The tracker blocker for this loader (Stage 9 Rec 12,
    /// Round 1 basic): loaded from the profile list dir at
    /// construction, refreshed in the background by the tab
    /// thread (see `renderer::tab_main`).
    pub fn blocker(&self) -> Arc<std::sync::Mutex<Blocker>> {
        self.scheduler.blocker()
    }

    /// The in-memory partitioned HTTP cache for this loader
    /// (Stage 10 Rec 1).
    pub fn http_cache(&self) -> Arc<std::sync::Mutex<HttpCache>> {
        self.scheduler.http_cache()
    }

    /// Cache hit / miss / store / revalidation counts.
    pub fn cache_stats(&self) -> CacheStats {
        self.scheduler.cache_stats()
    }

    /// The shared HTTPS-only preference + exception store
    /// (Stage 9 Rec 5). Default-on; see [`HttpsOnly`].
    pub fn https_only(&self) -> Arc<std::sync::Mutex<HttpsOnly>> {
        self.scheduler.https_only()
    }

    /// Replace this loader's HTTPS-only preference + exceptions
    /// (profile load).
    pub fn set_https_only(&self, mode: HttpsOnly) {
        *self.scheduler.https_only().lock().unwrap() = mode;
    }

    /// The shared referrer context (Stage 9 Rec 15): the current
    /// document URL + the policy in effect for it. `pluto-browser`
    /// updates it on navigation (document URL), on the document
    /// response (header policy), and during the parse (`<meta
    /// name="referrer">`).
    pub fn referrer_context(&self) -> Arc<std::sync::Mutex<ReferrerContext>> {
        self.scheduler.referrer_context()
    }

    /// Point the referrer context at a new document (navigation):
    /// the previous document stays the referrer for the
    /// navigation request itself; the context is re-pointed to the
    /// new document when its response headers arrive (see
    /// `service.rs`).
    pub fn set_referrer_document(&self, document_url: &str) {
        let ctx = self.scheduler.referrer_context();
        let mut ctx = ctx.lock().unwrap();
        ctx.document_url = Some(document_url.to_string());
    }

    /// Update the policy in effect for the current document
    /// (document `Referrer-Policy` header, or `<meta
    /// name="referrer">` during the parse). The document URL is
    /// left untouched.
    pub fn set_referrer_policy(&self, policy: ReferrerPolicy) {
        let ctx = self.scheduler.referrer_context();
        ctx.lock().unwrap().policy = policy;
    }

    /// Warm the connection for `url` (Stage 11 Rec 6): HSTS is
    /// applied first, then the blocker gate, then DNS (via the
    /// shared TTL cache) plus a TCP+TLS handshake through a
    /// connector that shares the TLS config (session store) and
    /// the DNS cache with the shared client. Returns `true` when
    /// a handshake completed.
    ///
    /// Honest limitation, documented here and in the report:
    /// hyper-util's legacy client exposes no API to park an
    /// externally-warmed socket in its pool, so the warmed socket
    /// is dropped after the handshake — the win is the cached DNS
    /// answer plus a primed rustls session the follow-up fetch
    /// resumes, not a reused socket. `http` origins warm TCP
    /// only (no TLS); non-http(s) URLs and blocker-blocked
    /// tracker origins are never warmed.
    pub async fn preconnect(&self, url: &str) -> bool {
        use tower_service::Service as _;
        let Ok(mut uri) = url.parse::<http::Uri>() else {
            return false;
        };
        // HSTS first: an `http` URL for a Known HSTS Host warms
        // the `https` endpoint instead.
        if uri.scheme_str() == Some("http") {
            if let Some(host) = uri.host() {
                let known = self
                    .scheduler
                    .hsts_store()
                    .lock()
                    .unwrap()
                    .is_known(host, std::time::SystemTime::now());
                if known {
                    if let Some(upgraded) = crate::hsts::upgrade_http_uri(&uri) {
                        uri = upgraded;
                    }
                }
            }
        }
        match uri.scheme_str() {
            Some("http") | Some("https") => {}
            _ => return false,
        }
        let url_str = uri.to_string();
        // Never warm a blocker-blocked tracker origin.
        if self
            .scheduler
            .blocker()
            .lock()
            .unwrap()
            .is_blocked(&url_str, Destination::Other, &url_str)
            .is_some()
        {
            return false;
        }
        let mut connector = crate::client::build_connector();
        if std::future::poll_fn(|cx| connector.poll_ready(cx))
            .await
            .is_err()
        {
            return false;
        }
        connector.call(uri).await.map(|_| ()).is_ok()
    }

    /// Resolve `url`'s host through the shared DNS TTL cache
    /// only (Stage 11 Rec 6): no TCP connection is opened, so a
    /// `<link rel="dns-prefetch">` costs one cache lookup (or one
    /// `getaddrinfo` on a miss) and nothing else. Returns `true`
    /// when the host resolves.
    pub async fn dns_prefetch(&self, url: &str) -> bool {
        use tower_service::Service as _;
        let Ok(uri) = url.parse::<http::Uri>() else {
            return false;
        };
        let Some(host) = uri.host() else {
            return false;
        };
        let Ok(name) = host.parse::<hyper_util::client::legacy::connect::dns::Name>() else {
            return false;
        };
        let mut resolver =
            CachedResolver::new(crate::client::shared_dns_cache().clone());
        let _ = std::future::poll_fn(|cx| resolver.poll_ready(cx)).await;
        resolver.call(name).await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_ipc::{CredentialsMode, Priority, RequestMode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Tiny hand-rolled HTTP/1.1 server: every connection gets one canned
    /// response based on the request path, then closes. `Connection:
    /// close` keeps hyper from reusing a connection across requests.
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
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
                let response = match path {
                    "/redirect" => {
                        "HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    }
                    "/loop" => {
                        "HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    }
                    "/final" => {
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello"
                            .to_string()
                    }
                    _ => {
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    }
                };
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    async fn test_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        format!("http://127.0.0.1:{}/", addr.port())
    }

    #[tokio::test]
    async fn follows_redirect_and_returns_body() {
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let response = loader.fetch(&format!("{base}redirect")).await.unwrap();
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.text(), "hello");
        assert_eq!(response.final_url.path(), "/final");
        assert_eq!(response.content_type(), Some("text/html"));
    }

    #[tokio::test]
    async fn too_many_redirects_errors() {
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let err = loader.fetch(&format!("{base}loop")).await.unwrap_err();
        assert!(matches!(err, NetError::TooManyRedirects(5)));
    }

    #[tokio::test]
    async fn invalid_url_errors() {
        let loader = ResourceLoader::new();
        assert!(matches!(
            loader.fetch("not a url").await.unwrap_err(),
            NetError::InvalidUrl(_)
        ));
        assert!(matches!(
            loader.fetch("ftp://example.com").await.unwrap_err(),
            NetError::InvalidUrl(_)
        ));
    }

    #[tokio::test]
    async fn http_status_is_not_an_error() {
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let response = loader.fetch(&format!("{base}nope")).await.unwrap();
        assert_eq!(response.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn fetch_stream_delivers_chunks_then_done() {
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let request = ResourceRequest {
            request_id: 1,
            tab_id: 0,
            epoch: 1,
            url: format!("{base}final"),
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: base.clone(),
            top_level_site: base.clone(),
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        };
        let mut events = loader.fetch_stream(request, CancellationToken::new()).await;
        let mut saw_headers = false;
        let mut body = Vec::new();
        while let Some(event) = events.recv().await {
            match event {
                ResourceEvent::ResponseHeaders { status, .. } => {
                    assert_eq!(status, 200);
                    saw_headers = true;
                }
                ResourceEvent::BodyChunk { bytes, .. } => body.extend_from_slice(&bytes),
                ResourceEvent::Done { .. } => break,
                other => panic!("unexpected terminal event: {other:?}"),
            }
        }
        assert!(saw_headers);
        assert_eq!(body, b"hello");
    }

    #[tokio::test]
    async fn scheduler_is_shared_across_loader_clones() {
        // Two clones of one loader share the scheduler: a dedup
        // key submitted through one is visible to the other.
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let other = loader.clone();
        let request = ResourceRequest {
            request_id: 1,
            tab_id: 0,
            epoch: 1,
            url: format!("{base}final"),
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: base.clone(),
            top_level_site: base.clone(),
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        };
        let mut rx1 = loader
            .fetch_stream(request.clone(), CancellationToken::new())
            .await;
        let mut rx2 = other.fetch_stream(request, CancellationToken::new()).await;
        let mut body1 = Vec::new();
        while let Some(event) = rx1.recv().await {
            match event {
                ResourceEvent::ResponseHeaders { .. } => {}
                ResourceEvent::BodyChunk { bytes, .. } => body1.extend_from_slice(&bytes),
                ResourceEvent::Done { .. } => break,
                other => panic!("unexpected event: {other:?}"),
            }
        }
        let mut body2 = Vec::new();
        while let Some(event) = rx2.recv().await {
            match event {
                ResourceEvent::ResponseHeaders { .. } => {}
                ResourceEvent::BodyChunk { bytes, .. } => body2.extend_from_slice(&bytes),
                ResourceEvent::Done { .. } => break,
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert_eq!(body1, b"hello");
        assert_eq!(body2, b"hello");
    }

    #[test]
    fn net_error_codes_round_trip() {
        // Every code the service emits maps back to a typed error.
        for (code, reason) in [
            ("invalid_url", "not a url"),
            ("too_many_redirects", "loop"),
            ("cancelled", ""),
            ("mixed_content", "http://x/y.js"),
            ("cors_blocked", "cross-origin fetch failed the CORS check"),
            ("insecure_redirect", "https -> http"),
            ("timeout", "slow"),
            ("hsts_tls_error", "https://example.com/"),
            ("https_only_failed", "http://example.com/"),
            ("body", "bad gzip"),
            ("request", "boom"),
            ("unknown-code", "boom"),
        ] {
            let e = NetError::from_code(code, reason);
            assert!(!e.to_string().is_empty(), "{code} must map");
        }
        assert_eq!(NetError::TooManyRedirects(5).code(), "too_many_redirects");
        assert_eq!(NetError::Cancelled.code(), "cancelled");
    }

    // ---- Stage 11 Rec 6 + 9: preconnect, dns-prefetch, progress ----

    /// Like [`test_server`], but counts accepted TCP connections —
    /// the seam that proves dns-prefetch opens none and preconnect
    /// opens exactly one.
    async fn counting_server() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::AtomicUsize;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let count2 = Arc::clone(&count);
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::spawn(async move {
                    // A preconnect handshake sends no HTTP request:
                    // just hold the socket briefly, then close.
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        (format!("http://127.0.0.1:{}/", addr.port()), count)
    }

    #[tokio::test]
    async fn fetch_with_progress_reports_bytes_vs_content_length() {
        let base = test_server().await;
        let loader = ResourceLoader::new();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<(u64, Option<u64>)>::new()));
        let seen2 = Arc::clone(&seen);
        let request = ResourceRequest {
            request_id: 1,
            tab_id: 0,
            epoch: 1,
            url: format!("{base}final"),
            method: "GET".to_string(),
            headers: Vec::new(),
            is_imageset: false,
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: base.clone(),
            top_level_site: base.clone(),
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        };
        let response = loader
            .fetch_with_progress(request, CancellationToken::new(), move |received, total| {
                seen2.lock().unwrap().push((received, total));
            })
            .await
            .unwrap();
        assert_eq!(response.body, b"hello");
        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty(), "progress was reported");
        assert_eq!(seen[0], (0, Some(5)), "headers carry Content-Length: 5");
        assert_eq!(*seen.last().unwrap(), (5, Some(5)), "all 5 bytes reported");
        assert!(seen.iter().all(|(_, t)| *t == Some(5)));
    }

    #[tokio::test]
    async fn dns_prefetch_opens_no_tcp_connection() {
        let (base, count) = counting_server().await;
        let loader = ResourceLoader::new();
        assert!(loader.dns_prefetch(&base).await, "127.0.0.1 resolves");
        // The host resolved (and the answer is cached now), but no
        // socket was opened to the server.
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "dns-prefetch must not open a TCP connection"
        );
        // A second prefetch is a cache hit.
        assert!(loader.dns_prefetch(&base).await);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn preconnect_opens_one_tcp_connection() {
        let (base, count) = counting_server().await;
        let loader = ResourceLoader::new();
        assert!(loader.preconnect(&base).await, "http preconnect warms TCP");
        // The TCP connect completes in the kernel before the
        // server task's `accept()` runs, so poll the counter
        // instead of asserting immediately.
        let mut seen = 0;
        for _ in 0..200 {
            seen = count.load(std::sync::atomic::Ordering::SeqCst);
            if seen == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            seen, 1,
            "preconnect opens exactly one TCP connection"
        );
    }

    #[tokio::test]
    async fn preconnect_refuses_non_http_and_blocked_origins() {
        let loader = ResourceLoader::new();
        assert!(!loader.preconnect("ftp://example.com").await);
        assert!(!loader.preconnect("not a url").await);
        // A blocker-blocked tracker origin is never warmed.
        loader.blocker().lock().unwrap().load_list(
            crate::blocker::ListSource::EasyList,
            "||tracker.example^\n",
        );
        assert!(!loader.preconnect("https://tracker.example/").await);
        assert!(!loader.preconnect("http://tracker.example/x.js").await);
    }
}
