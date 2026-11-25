//! HTTPS-only mode (Stage 9 Recommendation 5).
//!
//! There is no W3C/IETF standard for HTTPS-only mode; this is a
//! user preference, not a site signal, and it must not be
//! conflated with HSTS state. It sits above HSTS in the pipeline
//! only in the sense that it applies to hosts with *no* HSTS
//! entry: every `http://` navigation and subresource request is
//! rewritten to `https://` before being sent. When the HSTS store
//! already covers a host, the HSTS rewrite (same URL shape) wins
//! and this module is never consulted for it.
//!
//! Default ON at M5 (Joel's decision), with a per-site exception
//! store as the escape hatch — do not silently break `http://`
//! intranet/test sites: a host in the exception store is fetched
//! over plain HTTP. The preference and the exceptions are
//! persisted as JSON in the browser profile by `pluto-browser`
//! (see `security.rs` there).
//!
//! Exemptions that are NOT exceptions (never rewritten, even with
//! the mode on): IP literals and `localhost` / `*.localhost`.
//! Those hosts have no certificates to upgrade to in practice,
//! and the local test servers Pluto's own test suite uses are
//! `http://127.0.0.1` — rewriting them would only fail later.
//! This matches the other browsers' localhost carve-out.
//!
//! Failure behaviour (wired in `service.rs`): when an HTTPS-only
//! rewrite of a top-level navigation fails, the navigation shows
//! the existing error page (the reason names HTTPS-only and the
//! per-site exception); subresource failures are quietly blocked
//! and logged by the existing load-path logging. There is no
//! interstitial "continue to HTTP" UI in this milestone — the
//! exception store is the escape hatch, managed through the
//! profile API until a settings UI lands (deferred, noted in
//! `pluto-browser/src/security.rs`).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::hsts::{canonical_host, host_is_ip};

/// The HTTPS-only preference + per-site exception store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpsOnly {
    /// Whether the mode is on. Default `true` (Joel's decision:
    /// default-on at M5).
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Canonical hosts (see [`canonical_host`]) exempt from the
    /// rewrite. Serialised sorted for stable profile files.
    #[serde(default)]
    pub exceptions: HashSet<String>,
}

fn default_enabled() -> bool {
    true
}

impl Default for HttpsOnly {
    fn default() -> Self {
        Self {
            enabled: true,
            exceptions: HashSet::new(),
        }
    }
}

impl HttpsOnly {
    pub fn new() -> Self {
        Self::default()
    }

    /// Is `host` exempt from HTTPS-only for a reason other than
    /// the exception store — an IP literal, `localhost`, or a
    /// `*.localhost` name?
    pub fn is_intrinsically_exempt(host: &str) -> bool {
        if host_is_ip(host) {
            return true;
        }
        let lower = host.trim_end_matches('.').to_ascii_lowercase();
        lower == "localhost" || lower.ends_with(".localhost")
    }

    /// Should an `http:` URL to `host` be rewritten? False when
    /// the mode is off, the host is intrinsically exempt, or the
    /// host is in the exception store.
    pub fn should_upgrade(&self, host: &str) -> bool {
        if !self.enabled || Self::is_intrinsically_exempt(host) {
            return false;
        }
        match canonical_host(host) {
            Some(domain) => !self.exceptions.contains(&domain),
            None => false,
        }
    }

    /// Rewrite an `http:` URI to `https:` under HTTPS-only: the
    /// same port rules as the HSTS rewrite (explicit 80 → 443,
    /// any other explicit port preserved, no port added).
    /// Returns `None` when no rewrite applies. Operates on
    /// `http::Uri` so an explicit `:80` survives to be rewritten
    /// (see `hsts::upgrade_http_uri`).
    pub fn rewrite_http_uri(&self, uri: &http::Uri) -> Option<http::Uri> {
        if uri.scheme_str() != Some("http") {
            return None;
        }
        let host = uri.host()?;
        if !self.should_upgrade(host) {
            return None;
        }
        crate::hsts::upgrade_http_uri(uri)
    }

    /// Add a per-site exception (canonicalised). Returns true
    /// when the host was newly added.
    pub fn add_exception(&mut self, host: &str) -> bool {
        match canonical_host(host) {
            Some(domain) => self.exceptions.insert(domain),
            None => false,
        }
    }

    /// Remove a per-site exception. Returns true when one was held.
    pub fn remove_exception(&mut self, host: &str) -> bool {
        match canonical_host(host) {
            Some(domain) => self.exceptions.remove(&domain),
            None => false,
        }
    }

    /// Serialise for the profile file (exceptions sorted).
    pub fn to_json(&self) -> String {
        let mut sorted = self.clone();
        let mut list: Vec<String> = sorted.exceptions.drain().collect();
        list.sort();
        serde_json::to_string(&serde_json::json!({
            "enabled": sorted.enabled,
            "exceptions": list,
        }))
        .unwrap_or_else(|_| "{\"enabled\":true,\"exceptions\":[]}".to_string())
    }

    /// Deserialise a profile file. A corrupt or empty file yields
    /// the default (enabled, no exceptions).
    pub fn from_json(text: &str) -> Self {
        serde_json::from_str(text).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_on_with_no_exceptions() {
        let mode = HttpsOnly::new();
        assert!(mode.enabled);
        assert!(mode.exceptions.is_empty());
    }

    #[test]
    fn rewrites_http_and_applies_port_rules() {
        let mode = HttpsOnly::new();
        let uri = |s: &str| s.parse::<http::Uri>().unwrap();
        assert_eq!(
            mode.rewrite_http_uri(&uri("http://example.com/x"))
                .map(|u| u.to_string()),
            Some("https://example.com/x".to_string())
        );
        assert_eq!(
            mode.rewrite_http_uri(&uri("http://example.com:80/x"))
                .map(|u| u.to_string()),
            Some("https://example.com:443/x".to_string())
        );
        assert_eq!(
            mode.rewrite_http_uri(&uri("http://example.com:8080/x"))
                .map(|u| u.to_string()),
            Some("https://example.com:8080/x".to_string())
        );
        assert_eq!(mode.rewrite_http_uri(&uri("https://example.com/x")), None);
    }

    #[test]
    fn exception_bypasses_the_rewrite() {
        let mut mode = HttpsOnly::new();
        let uri = |s: &str| s.parse::<http::Uri>().unwrap();
        assert!(mode.add_exception("example.com"));
        assert_eq!(mode.rewrite_http_uri(&uri("http://example.com/x")), None);
        // Other hosts still upgrade; subdomains are NOT excepted.
        assert!(mode.rewrite_http_uri(&uri("http://other.com/x")).is_some());
        assert!(mode
            .rewrite_http_uri(&uri("http://sub.example.com/x"))
            .is_some());
        assert!(mode.remove_exception("EXAMPLE.com"));
        assert!(mode.rewrite_http_uri(&uri("http://example.com/x")).is_some());
    }

    #[test]
    fn loopback_and_ip_hosts_are_exempt() {
        let mode = HttpsOnly::new();
        for host in [
            "127.0.0.1",
            "192.0.2.1",
            "[::1]",
            "localhost",
            "LOCALHOST",
            "app.localhost",
        ] {
            assert!(
                !mode.should_upgrade(host),
                "{host} must be exempt from HTTPS-only"
            );
            assert_eq!(
                mode.rewrite_http_uri(&format!("http://{host}/x").parse::<http::Uri>().unwrap()),
                None,
                "{host} must not rewrite"
            );
        }
    }

    #[test]
    fn disabled_mode_rewrites_nothing() {
        let mut mode = HttpsOnly::new();
        mode.enabled = false;
        assert_eq!(
            mode.rewrite_http_uri(&"http://example.com/x".parse::<http::Uri>().unwrap()),
            None
        );
    }

    #[test]
    fn json_round_trip() {
        let mut mode = HttpsOnly::new();
        mode.add_exception("example.com");
        mode.add_exception("intranet.local");
        let back = HttpsOnly::from_json(&mode.to_json());
        assert!(back.enabled);
        assert!(back.exceptions.contains("example.com"));
        assert!(back.exceptions.contains("intranet.local"));
        assert!(HttpsOnly::from_json("garbage").enabled);
        assert!(HttpsOnly::from_json("").exceptions.is_empty());
    }
}
