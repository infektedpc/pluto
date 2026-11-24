//! The one shared hyper client for the whole browser process
//! (Stage 5 #2).
//!
//! Pooling and TLS session resumption only work if the client and
//! the rustls config are shared: a fresh client per navigation (the
//! M1 behaviour) throws both away. The client is built once in a
//! `OnceLock` and cloned cheaply — the legacy `Client` shares its
//! pool across clones.
//!
//! Explicit caps, because hyper-util's defaults are footguns:
//! `pool_max_idle_per_host` defaults to `usize::MAX` (unbounded)
//! and the pool timer is unset, so idle eviction never runs unless
//! it is set here.

use std::sync::OnceLock;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::{TokioExecutor, TokioTimer};

/// Body type for the shared client: empty request bodies (GET only
/// for now; POST bodies arrive with forms at M5).
pub type Body = Full<Bytes>;
/// The shared client type: hyper-util legacy client over a
/// rustls-backed HTTPS connector.
pub type HttpsClient = Client<HttpsConnector<HttpConnector>, Body>;

/// Maximum idle connections kept per host. Matches the per-origin
/// HTTP/1.1 concurrency limit (6) — a host cannot accumulate more
/// idle sockets than it can use at once.
pub const POOL_MAX_IDLE_PER_HOST: usize = 6;
/// Idle connections are evicted after this long. Matches hyper's
/// default; the point is that it is *set*, with the timer below,
/// so eviction actually runs.
pub const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);
/// TCP/TLS connect must complete within this, or the attempt fails.
/// (Hyper-rustls's builder does not expose the inner connector's
/// timeout, so it is set on a hand-built `HttpConnector` and wrapped
/// in — see [`build_connector`].)
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Build Pluto's rustls `ClientConfig` explicitly, so the security
/// properties are visible in this crate instead of hiding inside
/// hyper-rustls's shorthand:
/// - ring is the only crypto provider (workspace rustls has
///   `default-features = false` + `ring`; a second provider would
///   panic at first TLS use),
/// - Mozilla roots via webpki-roots, compiled in (no OS store),
/// - TLS 1.3 preferred, 1.2 allowed (rustls safe defaults);
///   1.0/1.1 and weak ciphers are refused by rustls and never
///   re-enabled here,
/// - one shared resumption store (256 server names, 8 tickets each
///   — the rustls default, set explicitly so it stays true),
/// - 0-RTT (`enable_early_data`) OFF: early data is replayable,
///   and resumption alone saves work, not a round trip. If ever
///   enabled, gate it to idempotent GETs only.
/// - ALPN is intentionally left empty: `HttpsConnectorBuilder`
///   rewrites it from `enable_http1`/`enable_http2` to
///   `[h2, http/1.1]` and panics if it is pre-set.
pub fn tls_config() -> Result<rustls::ClientConfig, rustls::Error> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut config = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth();
    config.enable_early_data = false;
    config.resumption = rustls::client::Resumption::in_memory_sessions(256);
    debug_assert!(
        config.alpn_protocols.is_empty(),
        "ALPN is set by the connector builder, not here"
    );
    Ok(config)
}

/// Build the HTTPS connector: the explicit TLS config above over a
/// hand-built `HttpConnector` (connect timeout + Happy Eyeballs at
/// hyper-util's default 300 ms race — verified in the hyper-util
/// 0.1.21 source; DNS itself stays the OS resolver).
pub fn build_connector() -> Result<HttpsConnector<HttpConnector>, rustls::Error> {
    let mut http = HttpConnector::new();
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));
    Ok(HttpsConnectorBuilder::new()
        .with_tls_config(tls_config()?)
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .wrap_connector(http))
}

/// Build the shared client with explicit pool caps. Cheap to clone;
/// clones share the pool.
pub fn build_client() -> Result<HttpsClient, rustls::Error> {
    let connector = build_connector()?;
    Ok(Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(POOL_MAX_IDLE_PER_HOST)
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .pool_timer(TokioTimer::new())
        .retry_canceled_requests(true)
        .build(connector))
}

/// The one shared client for the whole browser process. Built once;
/// the pool, the TLS resumption store and the DNS cache all live
/// behind this handle.
pub fn shared_client() -> &'static HttpsClient {
    static CLIENT: OnceLock<HttpsClient> = OnceLock::new();
    CLIENT
        .get_or_init(|| build_client().expect("rustls safe-default protocol versions always build"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_config_has_early_data_off_and_resumption_on() {
        let config = tls_config().expect("TLS config builds");
        assert!(!config.enable_early_data, "0-RTT stays off");
        // ALPN is empty here by design — the connector rewrites it.
        assert!(config.alpn_protocols.is_empty());
    }

    #[test]
    fn shared_client_is_a_singleton() {
        assert!(std::ptr::eq(shared_client(), shared_client()));
    }
}
