//! The redirect audit (Stage 5 #7): what happens on each hop of
//! a redirect chain.
//!
//! Pluto already followed 301/302/307/308 up to 5 hops. The audit
//! found three gaps and fixed them:
//! - 303 See Other was not followed at all (it is now, converting
//!   to GET, per Fetch);
//! - a `Location` with a non-http(s) scheme (`ftp://…`) was mangled
//!   into a relative path instead of rejected;
//! - protocol-relative locations (`//other.com/path`) were mangled
//!   into a same-origin path instead of keeping the current scheme
//!   with the new authority.
//!
//! And one refusal added:
//!
//! - an https→http redirect (downgrade) is refused outright.
//!
//! What needs no fix, and why:
//! - Method rules: 301/302/303 convert POST→GET and drop the body;
//!   307/308 preserve method and body. Pluto only sends GET today,
//!   so the conversion is vacuous — but it is implemented generally
//!   here, so POST (M5 forms) gets it right for free.
//! - Header stripping on cross-origin hops: Pluto sends only
//!   User-Agent, Accept and Accept-Encoding — no Authorization, no
//!   Cookie — so there is nothing to leak. The fetch task rebuilds
//!   that header set fresh on every hop, which *is* the strip
//!   point; if a sensitive header is ever added, it must be
//!   stripped here.
//! - Hop budget: 5, counted across the whole chain, not per hop
//!   (enforced by the fetch task, which passes the hop count in).

use http::{Method, Uri};

use crate::NetError;

/// What one redirect hop resolves to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedirectDecision {
    /// Follow to this URL with this method.
    Follow { url: Uri, method: Method },
}

/// Resolve a `Location` value against the URL that produced it.
/// Absolute URLs pass through; protocol-relative (`//host/path`)
/// keeps the current scheme with the new authority; absolute paths
/// stay on the same origin; relative paths resolve against the base
/// path's directory. Fragments are dropped (they are never sent to
/// the server). Anything that is not an http(s) URL is rejected.
pub fn resolve_redirect(base: &Uri, location: &str) -> Result<Uri, NetError> {
    let location = location.split('#').next().unwrap_or("");
    if location.is_empty() {
        return Err(NetError::InvalidUrl(location.to_string()));
    }
    if let Some(rest) = location.strip_prefix("//") {
        // Protocol-relative: same scheme, new authority.
        let scheme = base.scheme_str().unwrap_or("https");
        let target = format!("{scheme}://{rest}");
        return target
            .parse()
            .map_err(|_| NetError::InvalidUrl(target.clone()));
    }
    if location.contains("://") {
        if location.starts_with("http://") || location.starts_with("https://") {
            return location
                .parse()
                .map_err(|_| NetError::InvalidUrl(location.to_string()));
        }
        return Err(NetError::InvalidUrl(location.to_string()));
    }
    // A Location whose first segment carries a scheme that is not
    // http(s) — `javascript:alert(1)`, `data:…`, `ftp:…` — is not a
    // relative path, it is a scheme we do not speak. Per RFC 3986 a
    // colon in the first segment *is* a scheme, so reject it rather
    // than joining it onto the base path (which would turn it into a
    // same-origin fetch of a nonsense path). A colon in a later
    // segment (`/foo:bar`, `?x:y`) is fine and falls through.
    if let Some(colon) = location.find(':') {
        let before = &location[..colon];
        if !before.is_empty()
            && before
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && before
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
            && !before.contains(['/', '?', '#'])
        {
            return Err(NetError::InvalidUrl(format!(
                "unsupported redirect scheme in {location:?}"
            )));
        }
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

/// Apply the redirect policy for one hop: resolve the `Location`,
/// refuse a scheme downgrade, and convert the method per the
/// status code. `status` must be one of 301/302/303/307/308.
pub fn next_hop(
    current: &Uri,
    status: http::StatusCode,
    location: &str,
    method: &Method,
) -> Result<RedirectDecision, NetError> {
    let target = resolve_redirect(current, location)?;
    // A redirect must never downgrade a secure context to plain
    // http (Fetch: downgrade is refused for secure contexts).
    if current.scheme_str() == Some("https") && target.scheme_str() == Some("http") {
        return Err(NetError::InsecureRedirect(format!("{current} -> {target}")));
    }
    let next_method = match status.as_u16() {
        301..=303 => Method::GET,
        307..=308 => method.clone(),
        _ => return Err(NetError::Request(format!("not a redirect: {status}"))),
    };
    Ok(RedirectDecision::Follow {
        url: target,
        method: next_method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::StatusCode;

    fn uri(s: &str) -> Uri {
        s.parse().unwrap()
    }

    #[test]
    fn resolves_absolute_relative_and_protocol_relative() {
        let base = uri("https://example.com/dir/page.html");
        assert_eq!(
            resolve_redirect(&base, "https://other.com/x").unwrap(),
            uri("https://other.com/x")
        );
        assert_eq!(
            resolve_redirect(&base, "/abs").unwrap(),
            uri("https://example.com/abs")
        );
        assert_eq!(
            resolve_redirect(&base, "rel").unwrap(),
            uri("https://example.com/dir/rel")
        );
        assert_eq!(
            resolve_redirect(&base, "//cdn.example.com/a.png").unwrap(),
            uri("https://cdn.example.com/a.png")
        );
        assert_eq!(
            resolve_redirect(&base, "/x#frag").unwrap(),
            uri("https://example.com/x")
        );
    }

    #[test]
    fn rejects_non_http_schemes() {
        let base = uri("https://example.com/");
        assert!(matches!(
            resolve_redirect(&base, "ftp://example.com/x"),
            Err(NetError::InvalidUrl(_))
        ));
        assert!(matches!(
            resolve_redirect(&base, "javascript:alert(1)"),
            Err(NetError::InvalidUrl(_))
        ));
        assert!(matches!(
            resolve_redirect(&base, ""),
            Err(NetError::InvalidUrl(_))
        ));
    }

    #[test]
    fn downgrade_is_refused() {
        let current = uri("https://example.com/a");
        assert!(matches!(
            next_hop(
                &current,
                StatusCode::MOVED_PERMANENTLY,
                "http://example.com/b",
                &Method::GET
            ),
            Err(NetError::InsecureRedirect(_))
        ));
        // Upgrade is fine, and http->http is fine.
        assert!(
            next_hop(
                &uri("http://example.com/a"),
                StatusCode::FOUND,
                "https://example.com/b",
                &Method::GET
            )
            .is_ok()
        );
    }

    #[test]
    fn method_rules_follow_fetch() {
        let current = uri("https://example.com/a");
        // 301/302/303 convert to GET…
        for status in [301u16, 302, 303] {
            let decision = next_hop(
                &current,
                StatusCode::from_u16(status).unwrap(),
                "/b",
                &Method::POST,
            )
            .unwrap();
            assert_eq!(
                decision,
                RedirectDecision::Follow {
                    url: uri("https://example.com/b"),
                    method: Method::GET,
                },
                "status {status} must convert POST to GET"
            );
        }
        // …while 307/308 preserve the method.
        for status in [307u16, 308] {
            let decision = next_hop(
                &current,
                StatusCode::from_u16(status).unwrap(),
                "/b",
                &Method::POST,
            )
            .unwrap();
            assert_eq!(
                decision,
                RedirectDecision::Follow {
                    url: uri("https://example.com/b"),
                    method: Method::POST,
                },
                "status {status} must preserve POST"
            );
        }
    }
}
