//! Networking (layer 2): `ResourceLoader`.
//!
//! Fetches HTML, CSS, JS, and images over TLS 1.3 (rustls +
//! webpki-roots), follows redirects (301/302/307/308), and streams bytes
//! to the parser. HTTP/1.1 and HTTP/2 are both enabled; the server picks.
//!
//! One loader per tab. A fetch returns [`Response`] for *any* HTTP status
//! (a 404 is a page, not a transport failure); only transport problems,
//! bad URLs, and redirect loops are [`NetError`]. CORS is enforced by the
//! caller (M2); M1 fetches top-level navigations, which are not CORS
//! gated.
//!
//! Sandbox note: this VM only reaches the internet through an HTTP proxy
//! and hyper does not speak proxies, so a live TLS fetch to example.com
//! cannot be proven here -- it is compile-verified in the sandbox and
//! the live proof belongs on Joel's box. The redirect/body logic *is*
//! proven here, against a local test server (see the tests below).

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::{self, HeaderMap};
use hyper::{Method, Request, StatusCode, Uri};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NetError {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("request failed: {0}")]
    Request(String),
    #[error("failed to read response body: {0}")]
    Body(String),
    #[error("too many redirects (limit {0})")]
    TooManyRedirects(usize),
}

/// A completed fetch: status, headers, raw bytes, and the URL the bytes
/// actually came from (after redirects).
#[derive(Debug)]
pub struct Response {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
    pub final_url: Uri,
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

type HttpsClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

/// One loader per tab. Owns the hyper + rustls client and the redirect
/// policy.
#[derive(Debug)]
pub struct ResourceLoader {
    client: HttpsClient,
    /// Maximum redirects followed before giving up.
    pub max_redirects: usize,
}

impl Default for ResourceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceLoader {
    pub fn new() -> Self {
        // `https_or_http`: TLS where the URL says https, plain HTTP
        // where it says http (local test servers, plain-http sites).
        let https = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .build();
        let client: HttpsClient = Client::builder(TokioExecutor::new()).build(https);
        Self { client, max_redirects: 5 }
    }

    /// Fetch a URL, following redirects (301/302/307/308) up to
    /// `max_redirects`. Method is always GET for M1 navigations; the
    /// response body is fully buffered (M3 streams images to disk).
    pub async fn fetch(&self, url: &str) -> Result<Response, NetError> {
        let mut current: Uri = url
            .parse()
            .map_err(|_| NetError::InvalidUrl(url.to_string()))?;
        if !matches!(current.scheme_str(), Some("http") | Some("https")) {
            return Err(NetError::InvalidUrl(url.to_string()));
        }

        let mut redirects = 0usize;
        loop {
            let request = Request::builder()
                .method(Method::GET)
                .uri(current.clone())
                .header(header::USER_AGENT, "Pluto/0.1.0")
                .header(
                    header::ACCEPT,
                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                )
                .body(Full::new(Bytes::new()))
                .map_err(|e| NetError::Request(e.to_string()))?;

            let response = self
                .client
                .request(request)
                .await
                .map_err(|e| NetError::Request(e.to_string()))?;

            let status = response.status();
            if matches!(
                status,
                StatusCode::MOVED_PERMANENTLY
                    | StatusCode::FOUND
                    | StatusCode::TEMPORARY_REDIRECT
                    | StatusCode::PERMANENT_REDIRECT
            ) {
                if redirects >= self.max_redirects {
                    return Err(NetError::TooManyRedirects(self.max_redirects));
                }
                let location = response
                    .headers()
                    .get(header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| NetError::Request("redirect without Location".into()))?;
                current = resolve_redirect(&current, location)?;
                redirects += 1;
                continue;
            }

            let headers = response.headers().clone();
            let body = response
                .into_body()
                .collect()
                .await
                .map_err(|e| NetError::Body(e.to_string()))?
                .to_bytes()
                .to_vec();
            return Ok(Response { status, headers, body, final_url: current });
        }
    }
}

/// Resolve a `Location` value against the URL that produced it.
/// Absolute URLs pass through; absolute paths stay on the same origin;
/// relative paths resolve against the base path's directory. Fragments
/// are dropped (they are never sent to the server).
fn resolve_redirect(base: &Uri, location: &str) -> Result<Uri, NetError> {
    let location = location.split('#').next().unwrap_or("");
    if location.starts_with("http://") || location.starts_with("https://") {
        return location
            .parse()
            .map_err(|_| NetError::InvalidUrl(location.to_string()));
    }
    let scheme = base.scheme_str().unwrap_or("https");
    let authority = base
        .authority()
        .ok_or_else(|| NetError::InvalidUrl(base.to_string()))?;
    let target = if let Some(path) = location.strip_prefix('/') {
        format!("{scheme}://{authority}/{path}")
    } else {
        let base_path = base.path();
        let dir = match base_path.rfind('/') {
            Some(i) => &base_path[..=i],
            None => "/",
        };
        format!("{scheme}://{authority}{dir}{location}")
    };
    target
        .parse()
        .map_err(|_| NetError::InvalidUrl(target.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Tiny hand-rolled HTTP/1.1 server: every connection gets one canned
    /// response based on the request path, then closes. `Connection:
    /// close` keeps hyper from reusing a connection across requests.
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else { return };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else { return };
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
}
