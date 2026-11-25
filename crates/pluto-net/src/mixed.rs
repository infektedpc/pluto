//! The mixed-content gate (Stage 5 #6, corrected Stage 9 Rec 6):
//! what an https page may load over http.
//!
//! Pure function, no I/O — the rules are Mixed Content Level 2's:
//! - an http page loads anything (no mixed content on an insecure
//!   page),
//! - an https page loading an https URL is fine,
//! - an https page loading an http URL splits by destination:
//!   *blockable* (script, stylesheet, font, fetch) is blocked,
//!   *upgradable* (plain image — and, later, audio/video) is
//!   rewritten to https, unless the host is an IP literal (an IP has
//!   no certificate to upgrade to, so upgrading would only fail
//!   later — block instead),
//! - a top-level navigation is never mixed content (a user
//!   following a link to http is their choice); the https→http
//!   *redirect* downgrade is refused by the redirect audit instead.
//! - `upgrade-insecure-requests` (set by the document's CSP at
//!   M10; false until then) upgrades *every* http subresource,
//!   including blockable ones.
//!
//! Stage 9 Rec 6 corrections (report §e): the L2 taxonomy carves
//! two kinds of image out of the upgradeable set, and they BLOCK
//! instead of upgrading:
//!
//! * CORS-mode images (`<img crossorigin>`, i.e. `mode: cors`):
//!   CORS-mode requests are force-failed, never upgraded;
//! * `imageset` images (`srcset` / `<picture>` candidates): the
//!   initiator is `imageset`, which L2 excludes from upgrading.
//!
//! Plain `img[src]` images keep auto-upgrading. Audio/video have
//! no `Destination` variant yet (the scanner maps them to
//! `Destination::Other`, which is blockable) — when audio/video
//! destinations land, they join `is_upgradable` here.
//!
//! The initiator half needs the parser/scanner to label `imageset`
//! discoveries: `pluto-html/src/scanner.rs` labels them
//! (`from: "imageset"`). The browser-side call site that builds a
//! `ResourceRequest` from a discovery (`page.rs`,
//! `subresource_request`) does NOT pass the label through —
//! `ResourceRequest` has no initiator field, the wire shape is
//! frozen, and `PROTOCOL_VERSION` stays 3 — so `service.rs` calls
//! this gate with `Initiator::Normal` today and the imageset half
//! is DEFERRED until that call site passes the label (no wire
//! change needed for that step: the label is already on the
//! discovery, it only needs threading through `subresource_request`
//! into the gate call). The CORS-mode half IS live: `ResourceRequest`
//! already carries `mode`.
//!
//! Mixed downloads (L2 §5.2) are blocked, not auto-upgraded:
//! [`is_mixed_download`] answers that at the response-headers
//! point, against `Content-Disposition: attachment` — the UA often
//! does not know a response is a download until the headers arrive.
//! The download manager itself lands later (Stage 9 Rec 13).

use http::Uri;
use http::header::HeaderMap;
use pluto_ipc::{Destination, RequestMode};

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

/// Which initiator fetched this resource (Mixed Content L2 §3.1).
/// The scanner labels `srcset` / `<picture>` image candidates
/// `imageset`; everything else is `Normal` (including plain
/// `img[src]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Initiator {
    /// A plain fetch: `img[src]`, `video[poster]`, stylesheet,
    /// script, and every other non-imageset initiator.
    Normal,
    /// A `srcset` / `<picture>` image candidate. L2 excludes
    /// imageset from auto-upgrading: it BLOCKS on a secure page.
    Imageset,
}

/// Destinations the browser auto-upgrades to https instead of
/// blocking. Plain images today; audio/video join when Pluto has
/// destinations for them (see the module docs).
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
/// `https://example.com`; `url` is the requested URL. `mode` is the
/// request's fetch mode (CORS-mode images block, never upgrade);
/// `initiator` is the L2 initiator (`Imageset` blocks, never
/// upgrades).
pub fn mixed_content_decision(
    page_origin: &str,
    url: &Uri,
    destination: Destination,
    upgrade_insecure_requests: bool,
    mode: RequestMode,
    initiator: Initiator,
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
    // `upgrade-insecure-requests` (CSP, M10) rewrites the URL to
    // https BEFORE the mixed-content gate conceptually — the URL
    // is no longer mixed content at all — so it upgrades every
    // subresource, including the two carve-outs below. That flag
    // is always false until CSP lands.
    if upgrade_insecure_requests {
        return upgrade_or_block(url);
    }
    // L2 §3.1 / §4.1 carve-outs: CORS-mode images and imageset
    // images are force-failed on a secure page — BLOCK, never
    // upgrade.
    if destination == Destination::Image
        && (mode == RequestMode::Cors || initiator == Initiator::Imageset)
    {
        return MixedContentDecision::Block;
    }
    if is_upgradable(destination) {
        return upgrade_or_block(url);
    }
    MixedContentDecision::Block
}

/// Upgrade an http URL to https in place (port left null → 443),
/// unless the host is an IP literal — then block: there is no
/// certificate to upgrade to, so upgrading would only fail later.
fn upgrade_or_block(url: &Uri) -> MixedContentDecision {
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
    match Uri::from_parts(parts) {
        Ok(upgraded) => MixedContentDecision::Upgrade(upgraded),
        Err(_) => MixedContentDecision::Block,
    }
}

/// Is this response a mixed download (Mixed Content L2 §5.2)?
/// A download from a secure context over a non-trustworthy URL is
/// BLOCKED, not auto-upgraded — decided at the response-headers
/// point, against `Content-Disposition: attachment` (the UA often
/// does not know it will be a download until the response arrives;
/// bytes may already have traversed the network before the abort,
/// which the spec calls "generally unavoidable").
///
/// `page_origin` is the document origin (`https://…` = a secure
/// context); `url` is the URL the response actually came from
/// (after redirects); `headers` are the response headers. The
/// download manager itself lands later (Stage 9 Rec 13); this is
/// only the block decision.
pub fn is_mixed_download(
    page_origin: &str,
    url: &Uri,
    headers: &HeaderMap,
) -> bool {
    if !page_origin.starts_with("https://") {
        return false;
    }
    if url.scheme_str() == Some("https") {
        return false;
    }
    if !matches!(url.scheme_str(), Some("http")) {
        return false;
    }
    headers
        .get(http::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("attachment")
        })
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
                false,
                RequestMode::NoCors,
                Initiator::Normal
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
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.css"),
                Destination::Stylesheet,
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false,
                RequestMode::NoCors,
                Initiator::Normal
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
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://[2001:db8::1]/x.png"),
                Destination::Image,
                false,
                RequestMode::NoCors,
                Initiator::Normal
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
                true,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Upgrade(uri("https://cdn.example.com/x.js"))
        );
    }

    // --- Stage 9 Rec 6: the two L2 carve-outs + mixed downloads ---

    #[test]
    fn cors_mode_image_blocks_never_upgrades() {
        // `<img crossorigin>` on an https page: force-failed, not
        // upgraded — even though a plain image would upgrade.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false,
                RequestMode::Cors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
        // The same image in `NoCors` mode upgrades.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Upgrade(uri("https://cdn.example.com/x.png"))
        );
        // CORS mode blocks only images: a CORS fetch is blockable
        // anyway, so the decision is the same either way.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.json"),
                Destination::Fetch,
                false,
                RequestMode::Cors,
                Initiator::Normal
            ),
            MixedContentDecision::Block
        );
    }

    #[test]
    fn imageset_image_blocks_never_upgrades() {
        // `srcset` / `<picture>` candidates block, never upgrade.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false,
                RequestMode::NoCors,
                Initiator::Imageset
            ),
            MixedContentDecision::Block
        );
        // The same URL as a plain `img[src]` upgrades.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.png"),
                Destination::Image,
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Upgrade(uri("https://cdn.example.com/x.png"))
        );
        // Imageset only matters for images: a script is blocked
        // either way.
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://cdn.example.com/x.js"),
                Destination::Script,
                false,
                RequestMode::NoCors,
                Initiator::Imageset
            ),
            MixedContentDecision::Block
        );
    }

    #[test]
    fn mixed_download_blocked_not_upgraded() {
        use http::header::{CONTENT_DISPOSITION, HeaderMap};
        let attachment = |value: &str| {
            let mut map = HeaderMap::new();
            map.insert(CONTENT_DISPOSITION, value.parse().unwrap());
            map
        };
        let empty = HeaderMap::new();
        // Secure context + http URL + attachment → blocked.
        assert!(is_mixed_download(
            "https://example.com",
            &uri("http://cdn.example.com/f.zip"),
            &attachment("attachment; filename=\"f.zip\"")
        ));
        // No disposition → not a download → not blocked here.
        assert!(!is_mixed_download(
            "https://example.com",
            &uri("http://cdn.example.com/f.zip"),
            &empty
        ));
        // `inline` is not a download.
        assert!(!is_mixed_download(
            "https://example.com",
            &uri("http://cdn.example.com/f.zip"),
            &attachment("inline")
        ));
        // An https URL is never a mixed download.
        assert!(!is_mixed_download(
            "https://example.com",
            &uri("https://cdn.example.com/f.zip"),
            &attachment("attachment")
        ));
        // An http page has no secure context to protect.
        assert!(!is_mixed_download(
            "http://example.com",
            &uri("http://cdn.example.com/f.zip"),
            &attachment("attachment")
        ));
    }

    #[test]
    fn document_navigation_is_never_mixed_content() {
        assert_eq!(
            mixed_content_decision(
                "https://example.com",
                &uri("http://other.com/"),
                Destination::Document,
                false,
                RequestMode::NoCors,
                Initiator::Normal
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
                false,
                RequestMode::NoCors,
                Initiator::Normal
            ),
            MixedContentDecision::Allow
        );
    }
}
