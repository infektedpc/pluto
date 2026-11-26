//! WebSocket transport (Stage 16 Phase 3, Rec 11).
//!
//! The handshake runs here, in `pluto-net`, on the browser side —
//! never in the renderer, which has no sockets in broker mode.
//! The socket itself is owned by the browser-side pump
//! (`NetworkBroker::serve_websocket`); this module only opens it,
//! after the same frozen gate order `execute_fetch` runs
//! (HSTS/mixed → blocker → CSP), so a WebSocket cannot bypass the
//! gates in one step (the mixed-content gate used to `Allow` any
//! non-`http` scheme — `ws://` on an https page blocks here).
//!
//! Deliberately NOT the shared hyper client (`client.rs:33-36`):
//! that is a pooled legacy client built for request/response. A
//! WebSocket handshake needs an HTTP/1.1-only raw upgraded stream
//! (WS over HTTP/2 is not a thing), with the socket owned outside
//! the pool — so the handshake goes through `tokio-tungstenite`
//! directly, with Pluto's own rustls `ClientConfig` for `wss:`
//! (the same `tls_config()` the hyper client uses: ring provider,
//! webpki-roots, 0-RTT off). Client masking and ping/pong live in
//! tungstenite, not in JS. DNS goes through the OS resolver here,
//! not Pluto's DNS cache — a WebSocket is one long-lived
//! connection, so the cache buys nothing and the extra hop would
//! only add failure modes.
//!
//! Proxy note: this opens a direct TCP connection, like the hyper
//! client's connector. A deployment that reaches the internet only
//! through an HTTP proxy needs a CONNECT tunnel here first —
//! follow-up, not this phase.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use http::Uri;
use pluto_ipc::{CredentialsMode, Destination, Origin, ResourceRequest};
use tokio_tungstenite::Connector;
use tokio_util::sync::CancellationToken;

use crate::client::{CONNECT_TIMEOUT, shared_tls_config};
use crate::cors::origin_header_for;
use crate::mixed::{Initiator, MixedContentDecision, mixed_content_decision};
use crate::service::csp_pre_request_check;
use crate::{NetError, ResourceLoader};

/// The connected socket type: tungstenite over a plain or
/// rustls-wrapped TCP stream.
pub type WsStream = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

impl ResourceLoader {
    /// Open a WebSocket after the fetch gates, or fail closed.
    /// `request.destination` must be `Destination::WebSocket` and
    /// `request.origin` the document origin (serialized) — the
    /// handshake's `Origin` header is derived from it via
    /// `origin_header_for` (opaque origins send `"null"`).
    /// `protocols` ride as `Sec-WebSocket-Protocol`. `cancel`
    /// aborts the handshake (a connected socket is cancelled by
    /// closing it, not by this token).
    pub async fn connect_websocket(
        &self,
        request: ResourceRequest,
        protocols: Vec<String>,
        cancel: CancellationToken,
    ) -> Result<(WsStream, String), NetError> {
        let mut current: Uri = request
            .url
            .parse()
            .map_err(|_| NetError::InvalidUrl(request.url.clone()))?;
        if !matches!(current.scheme_str(), Some("ws") | Some("wss")) {
            return Err(NetError::InvalidUrl(request.url.clone()));
        }
        // HSTS ws → wss (the fetch path's `apply_secure_upgrade`
        // only rewrites `http`, so this is the WS half): a Known
        // HSTS Host never speaks plain `ws:`.
        if current.scheme_str() == Some("ws")
            && let Some(host) = current.host()
            && self
                .scheduler
                .hsts_store()
                .lock()
                .unwrap()
                .is_known_shared(host, SystemTime::now())
        {
            let mut parts = current.clone().into_parts();
            parts.scheme = Some("wss".parse().map_err(|_| {
                NetError::InvalidUrl(request.url.clone())
            })?);
            if let Some(authority) = &parts.authority
                && authority.port_u16() == Some(80)
            {
                let text = authority.as_str();
                let rebuilt = format!("{}:443", &text[..text.len() - 3]);
                parts.authority = Some(rebuilt.parse().map_err(|_| {
                    NetError::InvalidUrl(request.url.clone())
                })?);
            }
            current = Uri::from_parts(parts)
                .map_err(|_| NetError::InvalidUrl(request.url.clone()))?;
        }
        // Mixed content (Stage 16 Phase 1, Rec 2): `ws://` on an
        // https page BLOCKS; `wss://` is allowed. Checked before
        // the non-http early return inside the gate — this call
        // site must pass `Destination::WebSocket`, or the gate
        // cannot see the scheme.
        let upgrade_insecure_requests = request.upgrade_insecure_requests
            || self
                .scheduler
                .csp_policies()
                .lock()
                .unwrap()
                .upgrade_insecure_requests();
        match mixed_content_decision(
            &request.origin,
            &current,
            Destination::WebSocket,
            upgrade_insecure_requests,
            pluto_ipc::RequestMode::Cors,
            Initiator::Normal,
        ) {
            MixedContentDecision::Allow => {}
            MixedContentDecision::Block => {
                return Err(NetError::MixedContentBlocked(request.url.clone()));
            }
            MixedContentDecision::Upgrade(upgraded) => current = upgraded,
        }
        // Blocker (frozen order: before CSP): `$websocket` rules
        // can finally match — `destination_bit` maps this
        // destination to `T_WEBSOCKET`.
        {
            let blocker_arc = self.scheduler.blocker();
            let blocker = blocker_arc.lock().unwrap();
            if let Some(rule) = blocker.is_blocked(
                &current.to_string(),
                Destination::WebSocket,
                &request.top_level_site,
            ) {
                return Err(NetError::BlockedByFilter(format!(
                    "{:?}:{}: {}",
                    rule.source, rule.line, rule.pattern
                )));
            }
        }
        // CSP `connect-src` (same `csp_pre_request_check` the fetch
        // path runs, with the destination driving the directive).
        if !csp_pre_request_check(&self.scheduler, &request, &current.to_string()) {
            return Err(NetError::CorsBlocked(
                "CSP pre-request check refused the WebSocket".to_string(),
            ));
        }
        // The handshake request: `Origin` from the document origin
        // (never the URL's — the server checks the page, not the
        // target), plus the subprotocols. `ClientRequestBuilder`
        // is the only correct way to add custom headers: a bare
        // `Request` passed to `connect_async_*` goes through
        // `IntoClientRequest` unchanged, so the `Sec-WebSocket-Key`
        // / `Upgrade` / `Host` headers would be missing and the
        // server would reject the handshake. Tungstenite adds the
        // `Sec-WebSocket-*` headers and the client masking.
        let doc_origin = Origin::from_url(&request.origin);
        let origin_value =
            origin_header_for(&doc_origin, &current, CredentialsMode::Include);
        let mut builder = tokio_tungstenite::tungstenite::ClientRequestBuilder::new(
            current.clone(),
        )
        .with_header("Origin", origin_value);
        for protocol in &protocols {
            builder = builder.with_sub_protocol(protocol.clone());
        }
        let connector = if current.scheme_str() == Some("wss") {
            Some(Connector::Rustls(Arc::clone(shared_tls_config())))
        } else {
            Some(Connector::Plain)
        };
        let connect = tokio_tungstenite::connect_async_tls_with_config(
            builder,
            None,
            false,
            connector,
        );
        let outcome = tokio::select! {
            _ = cancel.cancelled() => return Err(NetError::Cancelled),
            outcome = tokio::time::timeout(CONNECT_TIMEOUT, connect) => match outcome {
                Ok(outcome) => outcome,
                Err(_) => {
                    return Err(NetError::Timeout(format!(
                        "websocket connect timed out after {:?}",
                        CONNECT_TIMEOUT
                    )));
                }
            },
        };
        match outcome {
            // The negotiated subprotocol rides the `Opened` event
            // (Stage 16 Phase 3, Rec 11): `ws.protocol` reads it,
            // so it must survive the handshake — "" when the server
            // selected none.
            Ok((stream, response)) => Ok((
                stream,
                response
                    .headers()
                    .get("Sec-WebSocket-Protocol")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string(),
            )),
            Err(e) => Err(NetError::Request(format!("websocket handshake: {e}"))),
        }
    }
}

/// Re-exported for tests: how long a handshake may take.
pub const WS_CONNECT_TIMEOUT: Duration = CONNECT_TIMEOUT;

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_ipc::RequestMode;

    fn ws_request(url: &str, origin: &str) -> ResourceRequest {
        ResourceRequest {
            request_id: 1,
            tab_id: 0,
            epoch: 1,
            url: url.to_string(),
            destination: Destination::WebSocket,
            priority: pluto_ipc::Priority::Medium,
            method: "GET".to_string(),
            headers: Vec::new(),
            is_imageset: false,
            mode: RequestMode::Cors,
            credentials: CredentialsMode::Include,
            origin: origin.to_string(),
            top_level_site: origin.to_string(),
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        }
    }

    /// A minimal echo server: accepts one TCP connection, runs the
    /// tungstenite server handshake, and echoes every message back
    /// until the client closes.
    async fn echo_server() -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        use futures_util::{SinkExt, StreamExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
                return;
            };
            while let Some(Ok(msg)) = ws.next().await {
                use tokio_tungstenite::tungstenite::Message;
                match msg {
                    Message::Text(_) | Message::Binary(_) => {
                        if ws.send(msg).await.is_err() {
                            break;
                        }
                    }
                    Message::Close(_) => {
                        let _ = ws.close(None).await;
                        break;
                    }
                    _ => {}
                }
            }
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn websocket_echo_round_trip() {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;
        let (addr, server) = echo_server().await;
        let loader = ResourceLoader::new();
        let request = ws_request(
            &format!("ws://127.0.0.1:{}/echo", addr.port()),
            "http://127.0.0.1",
        );
        let (mut ws, protocol) = loader
            .connect_websocket(request, Vec::new(), CancellationToken::new())
            .await
            .expect("handshake succeeds against the local echo server");
        assert_eq!(protocol, "", "no subprotocol was offered");
        ws.send(Message::Text("hello ws".into()))
            .await
            .expect("send succeeds");
        let echo = ws.next().await.expect("a reply arrives").expect("no error");
        assert_eq!(echo, Message::Text("hello ws".into()));
        ws.send(Message::Binary(vec![1, 2, 3].into()))
            .await
            .expect("binary send succeeds");
        let echo = ws.next().await.expect("a binary reply arrives").expect("no error");
        assert_eq!(echo, Message::Binary(vec![1, 2, 3].into()));
        ws.close(None).await.expect("close succeeds");
        server.abort();
    }

    #[tokio::test]
    async fn websocket_non_ws_scheme_is_rejected_before_any_socket() {
        let loader = ResourceLoader::new();
        let request = ws_request("https://127.0.0.1:1/echo", "http://127.0.0.1");
        let err = loader
            .connect_websocket(request, Vec::new(), CancellationToken::new())
            .await
            .expect_err("https: is not a websocket scheme");
        assert!(matches!(err, NetError::InvalidUrl(_)), "{err:?}");
    }

    #[tokio::test]
    async fn websocket_ws_on_https_page_is_mixed_content() {
        let loader = ResourceLoader::new();
        let request = ws_request("ws://127.0.0.1:1/echo", "https://example.com");
        let err = loader
            .connect_websocket(request, Vec::new(), CancellationToken::new())
            .await
            .expect_err("ws:// on an https page is mixed content");
        assert!(
            matches!(err, NetError::MixedContentBlocked(_)),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn websocket_wss_on_https_page_passes_the_gate() {
        // The gate allows `wss:` on an https page — the handshake
        // then fails at TCP connect (nothing listens), which proves
        // the gate let it through rather than blocking it.
        let loader = ResourceLoader::new();
        let request = ws_request("wss://127.0.0.1:1/echo", "https://example.com");
        let err = loader
            .connect_websocket(request, Vec::new(), CancellationToken::new())
            .await
            .expect_err("nothing listens on port 1");
        assert!(
            matches!(err, NetError::Request(_)),
            "gate passed, connect failed: {err:?}"
        );
    }
}
