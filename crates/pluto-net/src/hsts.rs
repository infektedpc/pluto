//! The HSTS store (Stage 9 Recommendation 5, RFC 6797).
//!
//! Per Known HSTS Host: the domain name (IDNA-canonicalised via the
//! `url` crate's host handling), the expiry time (reception time +
//! `max-age`), and whether `includeSubDomains` is asserted. The
//! store is persisted as JSON in the browser profile by
//! `pluto-browser` (see `security.rs` there); this crate only owns
//! the in-memory store and its serialisation.
//!
//! Rules implemented here (RFC 6797 §6.1, §8.1–§8.4):
//! - `max-age` is REQUIRED, digits only (optionally quoted); a
//!   header without a conforming `max-age` is ignored in full.
//! - `includeSubDomains` is optional and valueless; a value on it
//!   makes the header non-conforming (ignored).
//! - Directive names are ASCII case-insensitive; each directive
//!   MUST appear at most once (a duplicate voids the header).
//! - Unrecognised directives are ignored, recognised ones still
//!   processed. Only the first `Strict-Transport-Security` header
//!   of a response is processed (the caller passes one value).
//! - Recorded ONLY from error-free HTTPS responses: the caller
//!   passes `secure_transport = true` only when the response
//!   arrived over TLS without errors. Headers on plain-HTTP
//!   responses are ignored, and `<meta http-equiv>` STS is never
//!   heeded — there is no meta path into this store at all.
//! - IP literals can never be HSTS hosts.
//! - `max-age=0` deletes the entry for that host.
//! - A fresh header updates the cached entry (sliding TTL).
//! - Matching is label-by-label, ASCII case-insensitive,
//!   right-to-left: a congruent match, or a superdomain match
//!   with asserted `includeSubDomains`.
//! - No preload list in v1 (Joel's decision).
//!
//! The rewrite itself (scheme `http` → `https`, explicit port 80
//! → 443, any other explicit port preserved, no port added when
//! none is present) is [`HstsStore::rewrite_http_url`]. It runs at
//! the very front of the fetch pipeline in `service.rs`, before
//! the mixed-content gate and before CSP ever sees the URL.
//!
//! TLS/cert errors on a Known HSTS Host are fatal with no
//! click-through: there is no click-through UI in Pluto at all,
//! and `service.rs` maps a transport failure for a Known HSTS
//! Host to [`crate::NetError::HstsTlsFailure`].

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use url::Url;

/// One Known HSTS Host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HstsEntry {
    /// IDNA-canonicalised domain (lowercase ASCII, punycode where
    /// needed), as produced by [`canonical_host`].
    pub domain: String,
    /// Unix seconds at which the entry expires.
    pub expires_at_unix: u64,
    /// Whether `includeSubDomains` was asserted.
    pub include_sub_domains: bool,
}

/// The parsed, conforming directives of one STS header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StsDirectives {
    /// `max-age` in seconds. `0` means "delete the entry".
    pub max_age: u64,
    /// Whether `includeSubDomains` was asserted.
    pub include_sub_domains: bool,
}

/// Parse one `Strict-Transport-Security` header value per RFC 6797
/// §6.1. Returns `None` when the header does not conform (no
/// `max-age`, a non-digit `max-age`, a duplicated directive, or a
/// valued `includeSubDomains`); unrecognised directives are
/// skipped. `preload` is not defined in RFC 6797 — it is ignored
/// here like any other unrecognised directive.
pub fn parse_sts_header(value: &str) -> Option<StsDirectives> {
    let mut max_age: Option<u64> = None;
    let mut include_sub_domains = false;
    let mut seen_include = false;
    for raw in value.split(';') {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }
        let (name, val) = match part.split_once('=') {
            Some((n, v)) => (n.trim(), Some(v.trim())),
            None => (part, None),
        };
        if name.eq_ignore_ascii_case("max-age") {
            if max_age.is_some() {
                return None; // duplicate directive: non-conforming
            }
            let digits = match val {
                Some(v) => {
                    // The value may be quoted per §6.1.
                    let v = v.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(v);
                    v
                }
                None => return None,
            };
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // `max-age` values in the wild fit in u64; a value that
            // overflows saturates rather than voiding the header.
            max_age = Some(digits.parse::<u64>().unwrap_or(u64::MAX));
        } else if name.eq_ignore_ascii_case("includesubdomains") {
            if seen_include {
                return None; // duplicate directive: non-conforming
            }
            seen_include = true;
            if val.is_some() {
                return None; // includeSubDomains is valueless
            }
            include_sub_domains = true;
        } else {
            // Unrecognised directive (incl. `preload`): ignore it,
            // keep processing the recognised ones.
            continue;
        }
    }
    Some(StsDirectives {
        max_age: max_age?,
        include_sub_domains,
    })
}

/// Canonicalise a host for storage/matching: lowercase ASCII,
/// IDNA-encoded via the `url` crate. Returns `None` for an empty
/// host or one that does not parse as a domain (IP literals are
/// rejected by the caller — see [`HstsStore::record`]).
pub fn canonical_host(host: &str) -> Option<String> {
    let host = host.trim().trim_end_matches('.');
    if host.is_empty() {
        return None;
    }
    let url = Url::parse(&format!("http://{host}/")).ok()?;
    match url.host()? {
        url::Host::Domain(d) => Some(d.to_string()),
        url::Host::Ipv4(_) | url::Host::Ipv6(_) => None,
    }
}

/// Rewrite an `http:` URI to `https:` with the RFC 6797 §8.3
/// port rules: explicit port 80 → 443, any other explicit port
/// preserved, no explicit port → none added. The caller has
/// already decided the host qualifies (HSTS match, HTTPS-only,
/// mixed-content auto-upgrade); this is only the mechanical
/// rewrite. Returns `None` for non-`http:` URIs.
pub fn upgrade_http_uri(uri: &http::Uri) -> Option<http::Uri> {
    if uri.scheme_str() != Some("http") {
        return None;
    }
    let mut parts = uri.clone().into_parts();
    parts.scheme = Some(http::uri::Scheme::HTTPS);
    if let Some(authority) = &parts.authority
        && authority.port_u16() == Some(80)
    {
        // The port is the authority's last `:digits` segment
        // (an IPv6 literal's brackets cannot end in `:80` without
        // it being the port), so a suffix replace is exact.
        let text = authority.as_str();
        let rebuilt = format!("{}:443", &text[..text.len() - 3]);
        parts.authority = Some(rebuilt.parse().ok()?);
    }
    http::Uri::from_parts(parts).ok()
}

/// Unix seconds for `now` (saturating at 0 before the epoch).
pub fn now_unix(now: SystemTime) -> u64 {
    now.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs()
}

/// Is `host` an IP literal (v4 or v6, with or without brackets)?
pub fn host_is_ip(host: &str) -> bool {
    let host = host.strip_prefix('[').unwrap_or(host);
    let host = host.strip_suffix(']').unwrap_or(host);
    host.parse::<IpAddr>().is_ok()
}

/// The HSTS store: Known HSTS Hosts keyed by canonical domain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HstsStore {
    /// Keyed by canonical domain (see [`canonical_host`]).
    #[serde(default)]
    pub entries: HashMap<String, HstsEntry>,
}

impl HstsStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop expired entries as of `now`.
    pub fn prune_expired(&mut self, now: SystemTime) {
        let now = now_unix(now);
        self.entries.retain(|_, e| e.expires_at_unix > now);
    }

    /// Record an STS header from a response, per RFC 6797 §8.1.
    /// `secure_transport` must be true only for a response received
    /// over a secure transport with no TLS errors/warnings; a
    /// header on a plain-HTTP response is ignored. `header_value`
    /// is the first `Strict-Transport-Security` header value, or
    /// `None` when the response carried none (no change then).
    /// `max-age=0` deletes the host's entry.
    pub fn record(
        &mut self,
        host: &str,
        secure_transport: bool,
        header_value: Option<&str>,
        now: SystemTime,
    ) {
        if !secure_transport {
            return;
        }
        let Some(value) = header_value else {
            return;
        };
        if host_is_ip(host) {
            return;
        }
        let Some(domain) = canonical_host(host) else {
            return;
        };
        let Some(directives) = parse_sts_header(value) else {
            return;
        };
        if directives.max_age == 0 {
            // Delete: only the congruent entry goes; a matched
            // superdomain entry is never touched (§8.1.1).
            self.entries.remove(&domain);
            return;
        }
        let expires_at_unix = now_unix(now).saturating_add(directives.max_age);
        // A fresh header updates the cached entry: sliding TTL.
        self.entries.insert(
            domain.clone(),
            HstsEntry {
                domain,
                expires_at_unix,
                include_sub_domains: directives.include_sub_domains,
            },
        );
    }

    /// Is `host` a Known HSTS Host as of `now`? Congruent match,
    /// or a superdomain match whose entry asserts
    /// `includeSubDomains`. Matching is label-by-label, ASCII
    /// case-insensitive, right-to-left. Expired entries do not
    /// match (and are evicted).
    pub fn is_known(&mut self, host: &str, now: SystemTime) -> bool {
        self.prune_expired(now);
        let Some(domain) = canonical_host(host) else {
            return false;
        };
        if self.entries.contains_key(&domain) {
            return true;
        }
        // Superdomain match: walk the suffixes (drop one label at
        // a time from the left); the first entry found must assert
        // includeSubDomains.
        let labels: Vec<&str> = domain.split('.').collect();
        for i in 1..labels.len() {
            let suffix = labels[i..].join(".");
            if let Some(entry) = self.entries.get(&suffix)
                && entry.include_sub_domains
            {
                return true;
            }
        }
        false
    }

    /// Non-mutating variant of [`HstsStore::is_known`] for callers
    /// that only hold a shared reference (expired entries simply
    /// do not match; eviction happens on the next mutable call).
    pub fn is_known_shared(&self, host: &str, now: SystemTime) -> bool {
        let now = now_unix(now);
        let Some(domain) = canonical_host(host) else {
            return false;
        };
        if let Some(entry) = self.entries.get(&domain)
            && entry.expires_at_unix > now
        {
            return true;
        }
        let labels: Vec<&str> = domain.split('.').collect();
        for i in 1..labels.len() {
            let suffix = labels[i..].join(".");
            if let Some(entry) = self.entries.get(&suffix)
                && entry.include_sub_domains
                && entry.expires_at_unix > now
            {
                return true;
            }
        }
        false
    }

    /// Rewrite an `http:` URI to `https:` when its host is a Known
    /// HSTS Host (RFC 6797 §8.3): explicit port 80 → 443, any
    /// other explicit port preserved, no explicit port → none
    /// added. Returns `None` when no rewrite applies (not `http:`,
    /// or the host is not known).
    ///
    /// This operates on `http::Uri`, not `url::Url`, on purpose:
    /// the `url` crate normalises an explicit default port away
    /// at parse time (`http://example.com:80/` becomes
    /// `http://example.com/`), so the `:80` → `:443` rule could
    /// never fire there. `Uri` keeps the authority verbatim.
    pub fn rewrite_http_uri(&self, uri: &http::Uri, now: SystemTime) -> Option<http::Uri> {
        if uri.scheme_str() != Some("http") {
            return None;
        }
        let host = uri.host()?;
        if !self.is_known_shared(host, now) {
            return None;
        }
        upgrade_http_uri(uri)
    }

    /// Serialise the store (expired entries pruned first) to JSON
    /// for the profile file.
    pub fn to_json(&self, now: SystemTime) -> String {
        let mut pruned = self.clone();
        pruned.prune_expired(now);
        serde_json::to_string(&pruned).unwrap_or_else(|_| "{\"entries\":{}}".to_string())
    }

    /// Deserialise a profile file. A corrupt or empty file yields
    /// an empty store (the browser logs and continues with no
    /// Known HSTS Hosts, rather than refusing to start).
    pub fn from_json(text: &str) -> Self {
        serde_json::from_str(text).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn store_with(header_host: &str, header: &str, at: u64) -> HstsStore {
        let mut s = HstsStore::new();
        s.record(header_host, true, Some(header), t(at));
        s
    }

    #[test]
    fn parses_valid_header() {
        assert_eq!(
            parse_sts_header("max-age=31536000"),
            Some(StsDirectives {
                max_age: 31536000,
                include_sub_domains: false
            })
        );
        assert_eq!(
            parse_sts_header("max-age=15768000; includeSubDomains"),
            Some(StsDirectives {
                max_age: 15768000,
                include_sub_domains: true
            })
        );
        // Quoted max-age, case-insensitive names, extra spaces.
        assert_eq!(
            parse_sts_header("Max-Age=\"63072000\"; IncludeSubDomains"),
            Some(StsDirectives {
                max_age: 63072000,
                include_sub_domains: true
            })
        );
        // Unrecognised directives (incl. preload) are ignored.
        assert_eq!(
            parse_sts_header("max-age=60; preload; foo=bar"),
            Some(StsDirectives {
                max_age: 60,
                include_sub_domains: false
            })
        );
    }

    #[test]
    fn malformed_headers_are_ignored() {
        // No max-age at all.
        assert_eq!(parse_sts_header("includeSubDomains"), None);
        assert_eq!(parse_sts_header(""), None);
        // Non-digit max-age.
        assert_eq!(parse_sts_header("max-age=abc"), None);
        assert_eq!(parse_sts_header("max-age=12x"), None);
        assert_eq!(parse_sts_header("max-age=-5"), None);
        assert_eq!(parse_sts_header("max-age="), None);
        assert_eq!(parse_sts_header("max-age"), None);
        // Duplicated directives void the header.
        assert_eq!(parse_sts_header("max-age=60; max-age=120"), None);
        assert_eq!(
            parse_sts_header("max-age=60; includeSubDomains; includeSubDomains"),
            None
        );
        // includeSubDomains with a value is non-conforming.
        assert_eq!(
            parse_sts_header("max-age=60; includeSubDomains=true"),
            None
        );
    }

    #[test]
    fn max_age_zero_deletes_the_entry() {
        let mut s = store_with("example.com", "max-age=31536000", 1000);
        assert!(s.is_known("example.com", t(1001)));
        s.record("example.com", true, Some("max-age=0"), t(1002));
        assert!(!s.is_known("example.com", t(1003)));
    }

    #[test]
    fn http_response_header_is_ignored() {
        let mut s = HstsStore::new();
        s.record("example.com", false, Some("max-age=31536000"), t(1000));
        assert!(!s.is_known("example.com", t(1001)));
    }

    #[test]
    fn ip_literal_is_never_a_known_host() {
        let mut s = HstsStore::new();
        s.record("192.0.2.1", true, Some("max-age=31536000"), t(1000));
        s.record("[2001:db8::1]", true, Some("max-age=31536000"), t(1000));
        assert!(!s.is_known("192.0.2.1", t(1001)));
        assert!(!s.is_known("[2001:db8::1]", t(1001)));
        assert!(s.entries.is_empty());
    }

    #[test]
    fn malformed_header_records_nothing() {
        let mut s = HstsStore::new();
        s.record("example.com", true, Some("max-age=abc"), t(1000));
        assert!(s.entries.is_empty());
        // And a missing header changes nothing.
        s.record("example.com", true, None, t(1000));
        assert!(s.entries.is_empty());
    }

    #[test]
    fn congruent_match_is_case_insensitive() {
        let s = store_with("Example.COM", "max-age=31536000", 1000);
        assert!(s.is_known_shared("example.com", t(1001)));
        assert!(s.is_known_shared("EXAMPLE.com", t(1001)));
        assert!(!s.is_known_shared("other.com", t(1001)));
    }

    #[test]
    fn superdomain_match_needs_include_sub_domains() {
        let with = store_with("example.com", "max-age=31536000; includeSubDomains", 1000);
        assert!(with.is_known_shared("sub.example.com", t(1001)));
        assert!(with.is_known_shared("deep.sub.example.com", t(1001)));
        // A sibling suffix is not a superdomain match.
        assert!(!with.is_known_shared("example.com.evil.com", t(1001)));
        assert!(!with.is_known_shared("notexample.com", t(1001)));

        let without = store_with("example.com", "max-age=31536000", 1000);
        assert!(without.is_known_shared("example.com", t(1001)));
        assert!(!without.is_known_shared("sub.example.com", t(1001)));
    }

    #[test]
    fn expired_entries_do_not_match() {
        let s = store_with("example.com", "max-age=60", 1000);
        assert!(s.is_known_shared("example.com", t(1059)));
        assert!(!s.is_known_shared("example.com", t(1060)));
        assert!(!s.is_known_shared("example.com", t(2000)));
    }

    #[test]
    fn fresh_header_slides_the_ttl() {
        let mut s = store_with("example.com", "max-age=60", 1000);
        // Re-record at t=1050: expiry moves to 1110, not 1060.
        s.record("example.com", true, Some("max-age=60"), t(1050));
        assert!(s.is_known_shared("example.com", t(1100)));
        assert!(!s.is_known_shared("example.com", t(1111)));
    }

    #[test]
    fn subdomain_note_does_not_touch_superdomain_entry() {
        let mut s = store_with("example.com", "max-age=31536000; includeSubDomains", 1000);
        s.record("sub.example.com", true, Some("max-age=60"), t(1000));
        // The superdomain entry keeps its own expiry and flag.
        let entry = s.entries.get("example.com").unwrap();
        assert_eq!(entry.expires_at_unix, 1000 + 31536000);
        assert!(entry.include_sub_domains);
        assert!(s.is_known_shared("other.example.com", t(2000)));
    }

    #[test]
    fn rewrite_port_rules() {
        let s = store_with("example.com", "max-age=31536000", 1000);
        let now = t(1001);
        let uri = |s: &str| s.parse::<http::Uri>().unwrap();
        // Explicit :80 -> :443.
        assert_eq!(
            s.rewrite_http_uri(&uri("http://example.com:80/x"), now)
                .map(|u| u.to_string()),
            Some("https://example.com:443/x".to_string())
        );
        // Any other explicit port is preserved.
        assert_eq!(
            s.rewrite_http_uri(&uri("http://example.com:8080/x"), now)
                .map(|u| u.to_string()),
            Some("https://example.com:8080/x".to_string())
        );
        // No explicit port: none is added.
        assert_eq!(
            s.rewrite_http_uri(&uri("http://example.com/x"), now)
                .map(|u| u.to_string()),
            Some("https://example.com/x".to_string())
        );
        // https URLs and unknown hosts are untouched.
        assert_eq!(s.rewrite_http_uri(&uri("https://example.com/x"), now), None);
        assert_eq!(s.rewrite_http_uri(&uri("http://other.com/x"), now), None);
        // Superdomain rewrite (includeSubDomains).
        let s2 = store_with("example.com", "max-age=31536000; includeSubDomains", 1000);
        assert_eq!(
            s2.rewrite_http_uri(&uri("http://sub.example.com/x"), now)
                .map(|u| u.to_string()),
            Some("https://sub.example.com/x".to_string())
        );
    }

    #[test]
    fn json_round_trip_and_pruning() {
        let mut s = store_with("example.com", "max-age=31536000; includeSubDomains", 1000);
        s.record("short.example.com", true, Some("max-age=10"), t(1000));
        let json = s.to_json(t(2000));
        let back = HstsStore::from_json(&json);
        assert!(back.is_known_shared("example.com", t(2000)));
        // The 10-second entry expired before serialisation.
        assert!(!back.entries.contains_key("short.example.com"));
        // Corrupt input yields an empty store, never a panic.
        assert!(HstsStore::from_json("not json{{").entries.is_empty());
        assert!(HstsStore::from_json("").entries.is_empty());
    }

    #[test]
    fn idna_host_canonicalises() {
        // `münchen.de` canonicalises to its punycode form; a
        // record under either spelling matches the other.
        let mut s = HstsStore::new();
        s.record("münchen.de", true, Some("max-age=60"), t(1000));
        assert_eq!(s.entries.len(), 1);
        assert!(s.is_known_shared("xn--mnchen-3ya.de", t(1001)));
        assert!(s.is_known_shared("MÜNCHEN.DE", t(1001)));
    }
}
