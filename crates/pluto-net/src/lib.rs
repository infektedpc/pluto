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

pub mod client;
pub mod decode;
pub mod mixed;
pub mod redirect;
pub mod service;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use http::header::{self, HeaderMap};
use hyper::StatusCode;
use pluto_ipc::{Destination, ResourceEvent, ResourceRequest};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use service::Scheduler;

pub use client::{HttpsClient, shared_client};

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
    /// A redirect downgraded https to http (Stage 5 #7).
    #[error("insecure redirect refused: {0}")]
    InsecureRedirect(String),
    /// A per-request timeout fired (time to first byte, or total).
    #[error("timed out: {0}")]
    Timeout(String),
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
            NetError::InsecureRedirect(_) => "insecure_redirect",
            NetError::Timeout(_) => "timeout",
        }
    }

    fn from_code(code: &str, reason: &str) -> Self {
        match code {
            "invalid_url" => NetError::InvalidUrl(reason.to_string()),
            "too_many_redirects" => NetError::TooManyRedirects(5),
            "cancelled" => NetError::Cancelled,
            "too_large" => NetError::Body(reason.to_string()),
            "mixed_content" => NetError::MixedContentBlocked(reason.to_string()),
            "insecure_redirect" => NetError::InsecureRedirect(reason.to_string()),
            "timeout" => NetError::Timeout(reason.to_string()),
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
        Self {
            scheduler: Arc::new(Scheduler::new(shared_client().clone(), 5)),
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
        let final_url_hint = request.url.clone();
        let mut events = self.fetch_stream(request, cancel).await;
        let mut status = StatusCode::OK;
        let mut headers = HeaderMap::new();
        let mut body = Vec::new();
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
                            headers.insert(n, v);
                        }
                    }
                }
                ResourceEvent::BodyChunk { bytes, .. } => body.extend_from_slice(&bytes),
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
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: base.clone(),
            top_level_site: base.clone(),
            upgrade_insecure_requests: false,
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
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: base.clone(),
            top_level_site: base.clone(),
            upgrade_insecure_requests: false,
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
            ("insecure_redirect", "https -> http"),
            ("timeout", "slow"),
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
}
