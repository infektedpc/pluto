//! Referrer policy (Stage 9 Recommendation 15).
//!
//! Pluto previously sent no `Referer` header at all (only
//! User-Agent / Accept / Accept-Encoding were ever sent). This
//! module adds it, under a policy:
//!
//! - Default: `no-referrer-when-downgrade`.
//! - Tightened option: `strict-origin-when-cross-origin`.
//! - No other policy value is implemented in v1.
//!
//! Policy sources, in the order they are applied (last one seen
//! wins for subsequent requests): the `Referrer-Policy` response
//! header on the document, then `<meta name="referrer">` elements
//! encountered while parsing it (captured in `pluto-html`,
//! `Parser::referrer_policy`). `pluto-browser` pushes the current
//! policy into the loader's [`ReferrerContext`]; `service.rs`
//! reads it when building each request.
//!
//! Rules ([`referer_for`], per the Referrer Policy spec):
//! - `no-referrer-when-downgrade`: a referrer over HTTPS to an
//!   HTTP request (a downgrade) sends nothing; a same-origin
//!   request sends the full referrer URL; a cross-origin request
//!   sends only the referrer's origin.
//! - `strict-origin-when-cross-origin`: a same-origin request
//!   sends the full referrer URL (even on a downgrade); a
//!   cross-origin request sends only the origin, and a
//!   cross-origin downgrade sends nothing.
//! - The referrer URL never carries a fragment or userinfo.
//! - A referrer that is not potentially trustworthy in the
//!   relevant sense (an `http:` referrer to an `https:` request
//!   is fine — that is an upgrade, not a downgrade) is handled by
//!   the two rules above; there is no separate trust check here.
//!
//! Known limitation (frozen wire shape): `ResourceRequest` carries
//! only the document *origin* (`scheme://authority`), not the
//! document URL, and `pluto-ipc` is out of scope for this task —
//! so when the loader has no referrer context for the current
//! document (e.g. a fresh loader, or a `fetch()` from JS), the
//! referrer source is the request's `origin` string: same-origin
//! requests then send the origin, not the full document URL.
//! The full-URL rule applies whenever the browser has pushed the
//! document URL into the context, which is the normal load path.

use url::Url;

/// The two referrer policies Pluto implements (plus the default).
/// The same two variants exist in `pluto-html` (`ReferrerPolicy`
/// there, parsed from `<meta name="referrer">`); the two enums are
/// kept in sync by hand because neither crate may depend on the
/// other — `pluto-browser` maps between them. Do not add a third
/// value to one without the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferrerPolicy {
    /// Send the full URL same-origin, only the origin
    /// cross-origin, and nothing on an https→http downgrade.
    #[default]
    NoReferrerWhenDowngrade,
    /// Send the full URL same-origin (even on a downgrade), only
    /// the origin cross-origin, and nothing on a cross-origin
    /// downgrade.
    StrictOriginWhenCrossOrigin,
}

impl ReferrerPolicy {
    /// Parse a policy token from a `Referrer-Policy` header value
    /// or a `<meta name="referrer">` content value. A header may
    /// carry a comma-separated list — the *last* recognised token
    /// wins, per spec. Unknown tokens are skipped; `None` when no
    /// token is recognised.
    pub fn parse(value: &str) -> Option<Self> {
        let mut found = None;
        for token in value.split(',') {
            let token = token.trim().to_ascii_lowercase();
            found = match token.as_str() {
                "no-referrer-when-downgrade" => Some(Self::NoReferrerWhenDowngrade),
                "strict-origin-when-cross-origin" => {
                    Some(Self::StrictOriginWhenCrossOrigin)
                }
                _ => found,
            };
        }
        found
    }

    /// The header/meta token for this policy.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoReferrerWhenDowngrade => "no-referrer-when-downgrade",
            Self::StrictOriginWhenCrossOrigin => "strict-origin-when-cross-origin",
        }
    }
}

/// Are `a` and `b` same-origin? Scheme, host (ASCII
/// case-insensitive) and effective port (default ports
/// normalised: `http` 80, `https` 443).
pub fn is_same_origin(a: &Url, b: &Url) -> bool {
    a.scheme() == b.scheme()
        && a.host_str().map(str::to_ascii_lowercase)
            == b.host_str().map(str::to_ascii_lowercase)
        && a.port_or_known_default() == b.port_or_known_default()
}

/// The origin serialization of a URL: `scheme://host[:port]`
/// (explicit non-default port kept, default port dropped).
pub fn origin_of_url(url: &Url) -> String {
    let mut out = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
    if let Some(port) = url.port() {
        out.push_str(&format!(":{port}"));
    }
    out
}

/// The `Referer` header value for a request to `request_url`
/// referred from `referrer_url` under `policy`, or `None` when
/// nothing is sent. The referrer URL is stripped of its fragment
/// and any userinfo before use.
pub fn referer_for(
    referrer_url: &Url,
    request_url: &Url,
    policy: ReferrerPolicy,
) -> Option<String> {
    let mut referrer = referrer_url.clone();
    referrer.set_fragment(None);
    let _ = referrer.set_username("");
    let _ = referrer.set_password(None);
    let downgrade =
        referrer.scheme() == "https" && request_url.scheme() != "https";
    let same_origin = is_same_origin(&referrer, request_url);
    match policy {
        ReferrerPolicy::NoReferrerWhenDowngrade => {
            if downgrade {
                return None;
            }
            if same_origin {
                Some(referrer.to_string())
            } else {
                Some(origin_of_url(&referrer))
            }
        }
        ReferrerPolicy::StrictOriginWhenCrossOrigin => {
            if same_origin {
                Some(referrer.to_string())
            } else if downgrade {
                None
            } else {
                Some(origin_of_url(&referrer))
            }
        }
    }
}

/// The referrer context for one tab's loader: the document that
/// subresource requests belong to (and that is the referrer for
/// navigations from it), plus the policy currently in effect for
/// that document. Owned by the scheduler; set by `pluto-browser`
/// (navigation start, document response headers, `<meta>` during
/// the parse). A fresh loader has no context — `service.rs` then
/// falls back to the request's `origin` field as the referrer
/// source (see the module docs for why that sends only the
/// origin same-origin).
#[derive(Debug, Clone, Default)]
pub struct ReferrerContext {
    /// The current document's URL, if known.
    pub document_url: Option<String>,
    /// The policy in effect for that document.
    pub policy: ReferrerPolicy,
}

impl ReferrerContext {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn parses_known_tokens_and_ignores_unknown() {
        assert_eq!(
            ReferrerPolicy::parse("no-referrer-when-downgrade"),
            Some(ReferrerPolicy::NoReferrerWhenDowngrade)
        );
        assert_eq!(
            ReferrerPolicy::parse("strict-origin-when-cross-origin"),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        // Unknown tokens are skipped, not fatal.
        assert_eq!(
            ReferrerPolicy::parse("no-referrer"),
            None
        );
        assert_eq!(
            ReferrerPolicy::parse("unsafe-url, strict-origin-when-cross-origin"),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        // Last recognised token in a list wins.
        assert_eq!(
            ReferrerPolicy::parse(
                "strict-origin-when-cross-origin, no-referrer-when-downgrade"
            ),
            Some(ReferrerPolicy::NoReferrerWhenDowngrade)
        );
        // Case-insensitive, whitespace-tolerant.
        assert_eq!(
            ReferrerPolicy::parse("  Strict-Origin-When-Cross-Origin "),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
    }

    #[test]
    fn default_same_origin_sends_full_url() {
        let policy = ReferrerPolicy::NoReferrerWhenDowngrade;
        assert_eq!(
            referer_for(
                &url("https://example.com/page?q=1#frag"),
                &url("https://example.com/other"),
                policy
            )
            .as_deref(),
            Some("https://example.com/page?q=1")
        );
        // Userinfo is stripped from the referrer.
        assert_eq!(
            referer_for(
                &url("https://user:pass@example.com/page"),
                &url("https://example.com/other"),
                policy
            )
            .as_deref(),
            Some("https://example.com/page")
        );
    }

    #[test]
    fn default_cross_origin_sends_origin_only() {
        let policy = ReferrerPolicy::NoReferrerWhenDowngrade;
        assert_eq!(
            referer_for(
                &url("https://example.com/page?q=1"),
                &url("https://other.com/x"),
                policy
            )
            .as_deref(),
            Some("https://example.com")
        );
        // An explicit default port is not part of the origin.
        assert_eq!(
            referer_for(
                &url("https://example.com:443/page"),
                &url("https://other.com/x"),
                policy
            )
            .as_deref(),
            Some("https://example.com")
        );
        // A non-default port is.
        assert_eq!(
            referer_for(
                &url("https://example.com:8443/page"),
                &url("https://other.com/x"),
                policy
            )
            .as_deref(),
            Some("https://example.com:8443")
        );
    }

    #[test]
    fn default_downgrade_sends_nothing() {
        let policy = ReferrerPolicy::NoReferrerWhenDowngrade;
        // https -> http, same host or not: nothing.
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("http://example.com/other"),
                policy
            ),
            None
        );
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("http://other.com/x"),
                policy
            ),
            None
        );
        // http -> https is an upgrade, not a downgrade: origin.
        assert_eq!(
            referer_for(
                &url("http://example.com/page"),
                &url("https://other.com/x"),
                policy
            )
            .as_deref(),
            Some("http://example.com")
        );
    }

    #[test]
    fn strict_origin_same_origin_sends_full_url() {
        let policy = ReferrerPolicy::StrictOriginWhenCrossOrigin;
        // Same-origin (same scheme, host, port): full URL.
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("https://example.com/other"),
                policy
            )
            .as_deref(),
            Some("https://example.com/page")
        );
        // A same-host https→http hop is NOT same-origin (the
        // scheme differs), so it is a cross-origin downgrade:
        // nothing is sent.
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("http://example.com/other"),
                policy
            ),
            None
        );
        // Cross-origin downgrade: nothing.
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("http://other.com/x"),
                policy
            ),
            None
        );
        // Cross-origin, no downgrade: origin only.
        assert_eq!(
            referer_for(
                &url("https://example.com/page"),
                &url("https://other.com/x"),
                policy
            )
            .as_deref(),
            Some("https://example.com")
        );
    }

    #[test]
    fn same_origin_normalises_default_ports() {
        assert!(is_same_origin(
            &url("https://example.com:443/a"),
            &url("https://example.com/b")
        ));
        assert!(!is_same_origin(
            &url("https://example.com:8443/a"),
            &url("https://example.com/b")
        ));
        assert!(!is_same_origin(
            &url("http://example.com/a"),
            &url("https://example.com/a")
        ));
        assert!(is_same_origin(
            &url("https://EXAMPLE.com/a"),
            &url("https://example.com/b")
        ));
    }
}
