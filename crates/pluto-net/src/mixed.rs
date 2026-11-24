//! The mixed-content gate (Stage 5 #6): what an https page may
//! load over http.
//!
//! Pure function, no I/O — the rules are MDN's, in miniature:
//! - an http page loads anything (no mixed content on an insecure
//!   page),
//! - an https page loading an https URL is fine,
//! - an https page loading an http URL splits by destination:
//!   *blockable* (script, stylesheet, font, fetch) is blocked,
//!   *upgradable* (image — and, later, audio/video) is rewritten
//!   to https, unless the host is an IP literal (an IP has no
//!   certificate to upgrade to, so upgrading would only fail
//!   later — block instead),
//! - a top-level navigation is never mixed content (a user
//!   following a link to http is their choice); the https→http
//!   *redirect* downgrade is refused by the redirect audit instead.
//! - `upgrade-insecure-requests` (set by the document's CSP at
//!   M10; false until then) upgrades *every* http subresource,
//!   including blockable ones.

use http::Uri;
use pluto_ipc::Destination;

/// What the gate says about one subresource load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MixedContentDecision {
    /// Load it as requested.
    Allow,
    /// Do not load it at all.
    Block,
    /// Load this URL instead (same URL, https scheme).
    Upgrade(Uri),
}

/// Destinations the browser auto-upgrades to https instead of
/// blocking. Images today; audio/video join when Pluto has them.
fn is_upgradable(destination: Destination) -> bool {
    matches!(destination, Destination::Image)
}

/// Is `host` an IP literal (v4 or v6)? Upgrading an IP host is
/// pointless — there is no certificate to upgrade to.
fn host_is_ip(host: &str) -> bool {
    let host = host.strip_prefix('[').unwrap_or(host);
    let host = host.strip_suffix(']').unwrap_or(host);
    host.parse::<std::net::IpAddr>().is_ok()
}

/// Apply the gate. `page_origin` is the document origin, e.g.
/// `https://example.com`; `url` is the requested URL.
pub fn mixed_content_decision(
    page_origin: &str,
    url: &Uri,
    destination: Destination,
    upgrade_insecure_requests: bool,
) -> MixedContentDecision {
    let page_https = page_origin.starts_with("https://");
    if !page_https {
        return MixedContentDecision::Allow;
    }
    if url.scheme_str() != Some("http") {
        // https, or a non-network scheme the caller routes
        // elsewhere: not this gate's business.
        return MixedContentDecision::Allow;
    }
    if destination == Destination::Document {
        // Top-level navigations are never mixed content.
        return MixedContentDecision::Allow;
    }
    if upgrade_insecure_requests || is_upgradable(destination) {
        if let Some(host) = url.host()
            && host_is_ip(host)
        {
            return MixedContentDecision::Block;
        }
        let mut parts = http::uri::Parts::default();
        parts.scheme = Some(http::uri::Scheme::HTTPS);
        if let Some(authority) = url.authority() {
            parts.authority = Some(authority.clone());
        }
        parts.path_and_query = url.path_and_query().cloned();
        return match Uri::from_parts(parts) {
            Ok(upgraded) => MixedContentDecision::Upgrade(upgraded),
            Err(_) => MixedContentDecision::Block,
        };
    }
    MixedContentDecision::Block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uri(s: &str) -> Uri {
        s.parse().unwrap()
    }

    #[test]
    fn http_page_allows_everything() {
        assert_eq!(
            mixed_content_decision(
                "http://example.com",
                &uri("http://other.com/x.js"),
                Destination::Script,
                false
            ),
            MixedContentDecision::Allow
        );
    }

    #[test]
    fn https_page_blocks_script_but_upgrades_image() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.js"),
                Destination::Script,
                false
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.css"),
                Destination::Stylesheet,
                false
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false
            ),
            MixedContentDecision::Upgrade(uri("https://cdn.example.com/x.png"))
        );
    }

    #[test]
    fn upgrade_refused_for_ip_hosts() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://192.0.2.1/x.png"),
                Destination::Image,
                false
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://[2001:db8::1]/x.png"),
                Destination::Image,
                false
            ),
            MixedContentDecision::Block
        );
    }

    #[test]
    fn upgrade_insecure_requests_upgrades_blockable_too() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.js"),
                Destination::Script,
                true
            ),
            MixedContentDecision::Upgrade(uri("https://cdn.example.com/x.js"))
        );
    }

    #[test]
    fn document_navigation_is_never_mixed_content() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://other.com/"),
                Destination::Document,
                false
            ),
            MixedContentDecision::Allow
        );
    }

    #[test]
    fn https_url_is_always_allowed() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("https://cdn.example.com/x.js"),
                Destination::Script,
                false
            ),
            MixedContentDecision::Allow
        );
    }
}
