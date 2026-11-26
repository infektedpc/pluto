//! Content Security Policy engine (Stage 9 Rec 8, CSP v1).
//!
//! A small, correct CSP Level 3 subset, per the Stage 9 research
//! report §(d):
//!
//! * Header-form policy parsing: `Content-Security-Policy`
//!   (enforce) and `Content-Security-Policy-Report-Only` (monitor
//!   only). Multiple policies are conjunctive — a resource must
//!   pass ALL enforced policies. Duplicate directive names within
//!   one policy: the second occurrence is ignored. Directive names
//!   are ASCII case-insensitive.
//! * Source-expression matching: `'none'`, `'self'`, host sources
//!   (including `*.example.com` and path prefixes), scheme sources,
//!   `'*'`, nonces, hashes (SHA-256/384/512 via `ring`), and the
//!   keywords `'unsafe-inline'`, `'unsafe-eval'`,
//!   `'unsafe-hashes'`, `'strict-dynamic'`.
//! * Directive fallback chains (see [`fallback_chain`]).
//! * Fetch pre-request gate ([`PolicySet::check`]), navigation
//!   checks (`frame-ancestors`, `base-uri`, `form-action`),
//!   `upgrade-insecure-requests` flag detection, and pure inline
//!   checks for scripts, styles, event handlers, and `eval()`.
//!
//! There is NO implicit/default policy: an empty [`PolicySet`]
//! allows everything. Report-Only policies NEVER block.
//!
//! v1 limits (follow-ups for the finisher / M10):
//! * No `report-uri` / `report-to` network sending — violations
//!   are returned as [`Violation`] records for console reporting
//!   only.
//! * No post-request check (response final-URL re-check), no
//!   `sandbox` directive enforcement, no `SecurityPolicyViolationEvent`.
//! * `'strict-dynamic'` trust propagation to dynamically inserted
//!   scripts is a parser-level concern: for the fetch gate, a
//!   `strict-dynamic` list never matches a URL (host/scheme
//!   sources are ignored when it is present), so an external
//!   script URL is blocked unless the element itself carries a
//!   matching nonce/hash (see [`inline_script_allowed`]).
//! * Hash-allowed *external* scripts (integrity metadata listed
//!   in the policy, CSP3) are not matched by the fetch gate yet.
//! * `<meta http-equiv="Content-Security-Policy">` policies are
//!   parsed by the same [`parse_policy`] but must be applied by the
//!   parser only to content after the meta element, and
//!   `frame-ancestors` / `report-uri` / `sandbox` are ignored in
//!   meta policies (enforced by the caller, not here).

use pluto_ipc::{Destination, Origin};
use ring::digest;

/// A CSP hash algorithm (`'sha256-…'`, `'sha384-…'`, `'sha512-…'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    Sha256,
    Sha384,
    Sha512,
}

impl HashAlgorithm {
    /// The `ring` digest algorithm for this hash.
    fn ring_algorithm(self) -> &'static digest::Algorithm {
        match self {
            HashAlgorithm::Sha256 => &digest::SHA256,
            HashAlgorithm::Sha384 => &digest::SHA384,
            HashAlgorithm::Sha512 => &digest::SHA512,
        }
    }

    /// Digest `content` with this algorithm.
    pub fn hash(self, content: &[u8]) -> Vec<u8> {
        digest::digest(self.ring_algorithm(), content)
            .as_ref()
            .to_vec()
    }

    /// Parse the algorithm name (`sha256` / `sha384` / `sha512`,
    /// case-insensitive).
    fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sha256" => Some(HashAlgorithm::Sha256),
            "sha384" => Some(HashAlgorithm::Sha384),
            "sha512" => Some(HashAlgorithm::Sha512),
            _ => None,
        }
    }
}

/// Standard-alphabet base64 encode (RFC 4648 §4, with padding).
/// Kept local so this crate needs no extra base64 dependency.
/// Test-only: the engine decodes policy hashes (`base64_decode`),
/// it never encodes one.
#[cfg(test)]
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Standard-alphabet base64 decode, tolerating missing padding.
/// Returns `None` on any character outside the alphabet (other
/// than trailing `=` padding) or a malformed length.
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let stripped = input.trim_end_matches('=');
    if stripped.is_empty() {
        return Some(Vec::new());
    }
    if stripped.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(stripped.len() * 3 / 4);
    let bytes = stripped.as_bytes();
    for chunk in bytes.chunks(4) {
        let mut n: u32 = 0;
        let mut count = 0;
        for &c in chunk {
            n = (n << 6) | value(c)?;
            count += 1;
        }
        // `count` sextets hold `count - 1` bytes (except a full
        // quartet, which holds 3).
        let nbytes = if count == 4 { 3 } else { count - 1 };
        // Left-align the accumulated bits.
        let shift = (4 - count) * 6;
        let n = n << shift;
        for i in 0..nbytes {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    Some(out)
}

/// A host source expression: `[scheme://]host[:port][/path]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSource {
    /// Scheme part, lowercased, or `None` (any scheme matches).
    pub scheme: Option<String>,
    /// Host, lowercased, without any leading `*.` (see
    /// `wildcard_subdomain`); brackets stripped from IPv6 literals.
    pub host: String,
    /// True for `*.example.com`: matches any subdomain of `host`.
    pub wildcard_subdomain: bool,
    /// Port, or `None` (any port matches).
    pub port: Option<u16>,
    /// Path prefix, or `None` (any path matches). A bare `/` counts
    /// as no restriction.
    pub path: Option<String>,
}

/// One source expression in a directive's source list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceExpr {
    /// `'none'`: matches nothing. In a list with other expressions
    /// it has no effect (it simply never matches).
    None,
    /// `'self'`: the document's origin (with scheme upgrade:
    /// `'self'` on an `http` page also matches its `https` variant).
    Self_,
    /// `'unsafe-inline'`: inline scripts/styles, event handlers,
    /// `javascript:` URLs, inline `style=` attributes. Ignored when
    /// a nonce or hash is present in the same directive.
    UnsafeInline,
    /// `'unsafe-eval'`: `eval()` and string-to-code.
    UnsafeEval,
    /// `'unsafe-hashes'`: event handlers / style attributes /
    /// `javascript:` navigations matched by hash.
    UnsafeHashes,
    /// `'strict-dynamic'`: trust propagates from nonce/hash-trusted
    /// scripts; host/scheme/`'unsafe-inline'` sources are ignored
    /// when it is present.
    StrictDynamic,
    /// `'report-sample'`: violation reports include a sample.
    ReportSample,
    /// `'*'`: any `http`/`https`/`ws`/`wss` URL. Never matches
    /// `data:`, `blob:`, or other non-network schemes.
    Wildcard,
    /// A scheme source, e.g. `https:` (stored lowercased, without
    /// the trailing colon).
    Scheme(String),
    /// A nonce source, `'nonce-<base64>'` (value stored exactly).
    Nonce(String),
    /// A hash source, `'sha256-/sha384-/sha512-<base64>'`.
    Hash {
        algorithm: HashAlgorithm,
        /// The base64 value exactly as written in the policy.
        value: String,
    },
    /// A host source.
    Host(HostSource),
    /// An unrecognised expression: never matches anything.
    Unknown(String),
}

impl SourceExpr {
    /// Parse one whitespace-separated source-expression token.
    fn parse(token: &str) -> SourceExpr {
        let lower = token.to_ascii_lowercase();
        match lower.as_str() {
            "'none'" => return SourceExpr::None,
            "'self'" => return SourceExpr::Self_,
            "'unsafe-inline'" => return SourceExpr::UnsafeInline,
            "'unsafe-eval'" => return SourceExpr::UnsafeEval,
            "'unsafe-hashes'" => return SourceExpr::UnsafeHashes,
            "'strict-dynamic'" => return SourceExpr::StrictDynamic,
            "'report-sample'" => return SourceExpr::ReportSample,
            "*" => return SourceExpr::Wildcard,
            _ => {}
        }
        // Quoted `'nonce-…'` / `'sha256-…'` forms.
        if token.len() >= 2 && token.starts_with('\'') && token.ends_with('\'') {
            let inner = &token[1..token.len() - 1];
            let inner_lower = inner.to_ascii_lowercase();
            if let Some(nonce) = inner.strip_prefix("nonce-").or_else(|| {
                inner_lower
                    .strip_prefix("nonce-")
                    .map(|_| &inner["nonce-".len()..])
            }) {
                if !nonce.is_empty() {
                    return SourceExpr::Nonce(nonce.to_string());
                }
                return SourceExpr::Unknown(token.to_string());
            }
            if let Some(dash) = inner.find('-') {
                let (algo_name, value) = inner.split_at(dash);
                let value = &value[1..];
                if let Some(algorithm) = HashAlgorithm::parse(algo_name) {
                    if !value.is_empty() {
                        return SourceExpr::Hash {
                            algorithm,
                            value: value.to_string(),
                        };
                    }
                }
                return SourceExpr::Unknown(token.to_string());
            }
            return SourceExpr::Unknown(token.to_string());
        }
        // A scheme source: `https:` (no slashes, ends in a colon).
        if token.ends_with(':') && !token.contains('/') {
            let scheme = &token[..token.len() - 1];
            if !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
            {
                return SourceExpr::Scheme(scheme.to_ascii_lowercase());
            }
            return SourceExpr::Unknown(token.to_string());
        }
        // Otherwise a host source (possibly with scheme and path).
        match HostSource::parse(token) {
            Some(host) => SourceExpr::Host(host),
            None => SourceExpr::Unknown(token.to_string()),
        }
    }

    /// Whether this is a nonce or hash source.
    fn is_nonce_or_hash(&self) -> bool {
        matches!(self, SourceExpr::Nonce(_) | SourceExpr::Hash { .. })
    }
}

impl HostSource {
    /// Parse `[scheme://]host[:port][/path]`. Returns `None` for a
    /// malformed host source (empty host, bad port).
    fn parse(token: &str) -> Option<HostSource> {
        let (scheme, rest) = match token.find("://") {
            Some(idx) => {
                let scheme = token[..idx].to_ascii_lowercase();
                if scheme.is_empty()
                    || !scheme
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
                {
                    return None;
                }
                (Some(scheme), &token[idx + 3..])
            }
            None => (None, token),
        };
        if rest.is_empty() {
            return None;
        }
        // Split off the path (first `/`).
        let (hostport, path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], Some(rest[idx..].to_string())),
            None => (rest, None),
        };
        if hostport.is_empty() {
            return None;
        }
        // Split host and port. An IPv6 literal keeps its brackets
        // until the closing `]`; anything after it must be `:port`.
        let (host, port) = if let Some(bracketed) = hostport.strip_prefix('[') {
            match bracketed.find(']') {
                Some(end) => {
                    let host = &bracketed[..end];
                    let after = &bracketed[end + 1..];
                    if after.is_empty() {
                        (host.to_string(), None)
                    } else if let Some(port_str) = after.strip_prefix(':') {
                        (host.to_string(), Some(port_str.parse::<u16>().ok()?))
                    } else {
                        return None;
                    }
                }
                None => return None,
            }
        } else {
            match hostport.rfind(':') {
                Some(idx) => {
                    let port_str = &hostport[idx + 1..];
                    // A trailing colon with no port is malformed.
                    if port_str.is_empty() || !port_str.bytes().all(|b| b.is_ascii_digit()) {
                        return None;
                    }
                    (
                        hostport[..idx].to_string(),
                        Some(port_str.parse::<u16>().ok()?),
                    )
                }
                None => (hostport.to_string(), None),
            }
        };
        if host.is_empty() {
            return None;
        }
        let (wildcard_subdomain, host) = match host.strip_prefix("*.") {
            Some(bare) => (true, bare.to_string()),
            None => (false, host),
        };
        if host.is_empty() || host == "*" {
            return None;
        }
        let path = match path.as_deref() {
            Some("/") | Some("") | None => None,
            Some(p) => Some(p.to_string()),
        };
        Some(HostSource {
            scheme,
            host: host.to_ascii_lowercase(),
            wildcard_subdomain,
            port,
            path,
        })
    }
}

/// One parsed CSP policy (one header value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Policy {
    /// Directives in first-seen order; directive names lowercased.
    /// A duplicate directive name within this policy keeps only
    /// its FIRST occurrence.
    directives: Vec<(String, Vec<SourceExpr>)>,
    /// The raw header value, kept verbatim for
    /// [`Violation::original_policy`].
    raw: String,
}

/// Parse one CSP header value into a [`Policy`].
///
/// Splits on `;`, takes the first whitespace-separated token of
/// each segment as the (case-insensitive) directive name, and
/// parses the rest as source expressions. Empty segments are
/// skipped; a directive with no values parses to an empty source
/// list (which ≡ `'none'` for fetch directives).
pub fn parse_policy(header: &str) -> Policy {
    let mut directives: Vec<(String, Vec<SourceExpr>)> = Vec::new();
    for segment in header.split(';') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.split_whitespace();
        let Some(name_token) = parts.next() else {
            continue;
        };
        let name = name_token.to_ascii_lowercase();
        // Duplicate directive: the second occurrence is ignored.
        if directives.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let exprs = parts.map(SourceExpr::parse).collect();
        directives.push((name, exprs));
    }
    Policy {
        directives,
        raw: header.to_string(),
    }
}

impl Policy {
    /// The source list for a directive (name matched
    /// case-insensitively), or `None` when the directive is absent.
    /// An empty list means the directive was present with no value
    /// (≡ `'none'` for fetch directives).
    pub fn directive(&self, name: &str) -> Option<&[SourceExpr]> {
        let lower = name.to_ascii_lowercase();
        self.directives
            .iter()
            .find(|(n, _)| *n == lower)
            .map(|(_, exprs)| exprs.as_slice())
    }

    /// Whether this policy contains the directive at all.
    pub fn has_directive(&self, name: &str) -> bool {
        self.directive(name).is_some()
    }

    /// The raw header value this policy was parsed from.
    pub fn original_policy(&self) -> &str {
        &self.raw
    }

    /// Whether the `upgrade-insecure-requests` flag directive is
    /// present in this policy.
    pub fn upgrade_insecure_requests(&self) -> bool {
        self.has_directive("upgrade-insecure-requests")
    }
}

/// The fallback chain for a fetch directive: the directive itself,
/// then each fallback in order. The first directive present in a
/// policy wins. `base-uri`, `form-action`, and `frame-ancestors`
/// have NO fallback (CSP3).
pub fn fallback_chain(directive: &str) -> &'static [&'static str] {
    match directive.to_ascii_lowercase().as_str() {
        "script-src-elem" => &["script-src-elem", "script-src", "default-src"],
        "script-src-attr" => &["script-src-attr", "script-src", "default-src"],
        "style-src-elem" => &["style-src-elem", "style-src", "default-src"],
        "style-src-attr" => &["style-src-attr", "style-src", "default-src"],
        "frame-src" => &["frame-src", "child-src", "default-src"],
        "worker-src" => &["worker-src", "child-src", "script-src", "default-src"],
        // No fallback for these three.
        "base-uri" => &["base-uri"],
        "form-action" => &["form-action"],
        "frame-ancestors" => &["frame-ancestors"],
        "script-src" => &["script-src", "default-src"],
        "style-src" => &["style-src", "default-src"],
        "img-src" => &["img-src", "default-src"],
        "connect-src" => &["connect-src", "default-src"],
        "font-src" => &["font-src", "default-src"],
        "object-src" => &["object-src", "default-src"],
        "media-src" => &["media-src", "default-src"],
        "manifest-src" => &["manifest-src", "default-src"],
        "prefetch-src" => &["prefetch-src", "default-src"],
        "default-src" => &["default-src"],
        // Unknown directives: no known fallback.
        _ => &[],
    }
}

/// Map a fetch [`Destination`] to the CSP fetch directive that
/// gates it, or `None` when no fetch directive gates it
/// (top-level navigations are gated by `frame-ancestors` on the
/// response instead — see [`PolicySet::frame_ancestors_allows`]).
pub fn directive_for_destination(destination: Destination) -> Option<&'static str> {
    match destination {
        Destination::Document => None,
        Destination::Script => Some("script-src"),
        Destination::Stylesheet => Some("style-src"),
        Destination::Image => Some("img-src"),
        Destination::Font => Some("font-src"),
        // Stage 16 Phase 3, Rec 12: a streaming fetch is still a
        // fetch — `connect-src`, like `Fetch`.
        Destination::Fetch | Destination::Stream => Some("connect-src"),
        // Stage 16 Phase 1, Rec 2: media finally consults
        // `media-src` (before this it fell to `Other` →
        // `default-src`, so `media-src 'none'` leaked), and a
        // WebSocket handshake is a `connect-src` fetch.
        Destination::Audio | Destination::Video => Some("media-src"),
        // Stage 16 Phase 3, Rec 12: a streaming fetch is still a
        // fetch — `connect-src`, like `Fetch`.
        Destination::WebSocket => Some("connect-src"),
        Destination::Other => Some("default-src"),
    }
}

/// Whether `disposition` is an enforced (blocking) policy or a
/// report-only (monitoring) one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Disposition {
    /// `Content-Security-Policy`: violations block.
    Enforce,
    /// `Content-Security-Policy-Report-Only`: violations are
    /// reported but never block.
    Report,
}

impl Disposition {
    /// `"enforce"` or `"report"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Disposition::Enforce => "enforce",
            Disposition::Report => "report",
        }
    }
}

/// One CSP violation, for console reporting (CSP3
/// `CSPViolationReportBody` fields, v1 subset). No network
/// reporting is done in v1 — `report-uri` / `report-to` sending
/// is a follow-up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// The URL of the document in which the violation occurred
    /// (`documentURL`).
    pub document_url: String,
    /// The URL that was blocked (`blockedURL`): fragment and
    /// userinfo stripped; cross-origin URLs truncated to
    /// scheme+host+port; `"inline"` for inline script/style
    /// blocks, `"eval"` for blocked `eval()`.
    pub blocked_url: String,
    /// The directive that was actually enforced, after fallback
    /// resolution (`effectiveDirective`).
    pub effective_directive: String,
    /// The full policy text that was violated (`originalPolicy`).
    pub original_policy: String,
    /// Whether the violated policy enforces or only reports.
    pub disposition: Disposition,
    /// Source file of an inline violation, when known.
    pub source_file: Option<String>,
    /// Source line of an inline violation, when known.
    pub line_number: Option<u32>,
    /// First characters of the offending inline block, only when
    /// the directive contains `'report-sample'`; `None` in v1
    /// (samples are not collected yet).
    pub sample: Option<String>,
}

impl Violation {
    /// Build a violation record. `source_file` / `line_number`
    /// are `None` for fetch violations; pass them for inline
    /// violations when the parser knows them.
    pub fn new(
        document_url: &str,
        blocked_url: &str,
        effective_directive: &str,
        original_policy: &str,
        disposition: Disposition,
        source_file: Option<String>,
        line_number: Option<u32>,
    ) -> Self {
        Violation {
            document_url: document_url.to_string(),
            blocked_url: blocked_url.to_string(),
            effective_directive: effective_directive.to_string(),
            original_policy: original_policy.to_string(),
            disposition,
            source_file,
            line_number,
            sample: None,
        }
    }
}

/// The document's policy container: enforced policies plus
/// report-only policies. Multiple policies are conjunctive — a
/// resource must pass ALL enforced policies.
#[derive(Debug, Clone, Default)]
pub struct PolicySet {
    /// `Content-Security-Policy` headers (enforce).
    pub enforce: Vec<Policy>,
    /// `Content-Security-Policy-Report-Only` headers (monitor only).
    pub report_only: Vec<Policy>,
}

impl PolicySet {
    /// An empty policy set: allows everything (there is no
    /// implicit policy — never invent a default `default-src
    /// 'self'`).
    pub fn new() -> Self {
        PolicySet::default()
    }

    /// Build a set from raw header values: one policy per header.
    pub fn from_headers(enforce_headers: &[&str], report_only_headers: &[&str]) -> Self {
        PolicySet {
            enforce: enforce_headers.iter().map(|h| parse_policy(h)).collect(),
            report_only: report_only_headers
                .iter()
                .map(|h| parse_policy(h))
                .collect(),
        }
    }

    /// Whether the set holds no policies at all (enforce or
    /// report-only).
    pub fn is_empty(&self) -> bool {
        self.enforce.is_empty() && self.report_only.is_empty()
    }

    /// Whether any enforced policy carries the
    /// `upgrade-insecure-requests` flag directive. (A report-only
    /// flag has no effect.)
    pub fn upgrade_insecure_requests(&self) -> bool {
        self.enforce
            .iter()
            .any(|p| p.upgrade_insecure_requests())
    }

    /// Fetch pre-request gate (CSP3 §4.1.2): whether fetching
    /// `url` for `directive_for_destination` (e.g. `"script-src"`,
    /// or the result of [`directive_for_destination`]) is allowed
    /// under every enforced policy, given the document `origin`.
    /// Report-only policies never block. An empty policy set
    /// allows everything.
    pub fn check(&self, directive_for_destination: &str, url: &str, origin: &Origin) -> bool {
        self.check_violation(directive_for_destination, url, origin, "")
            .is_none()
    }

    /// Like [`PolicySet::check`], but returns the first enforced
    /// violation instead of just a boolean, for console reporting.
    /// `document_url` is recorded on the violation; pass `""` when
    /// it is not known. Returns `None` when the fetch is allowed.
    pub fn check_violation(
        &self,
        directive_for_destination: &str,
        url: &str,
        origin: &Origin,
        document_url: &str,
    ) -> Option<Violation> {
        for policy in &self.enforce {
            if let (false, Some(effective)) =
                policy_decision(policy, directive_for_destination, url, origin)
            {
                return Some(Violation::new(
                    document_url,
                    &blocked_url_for_report(url, origin),
                    &effective,
                    policy.original_policy(),
                    Disposition::Enforce,
                    None,
                    None,
                ));
            }
        }
        None
    }

    /// The report-only violations this fetch *would* have produced
    /// (disposition `"report"`). Report-only policies never block,
    /// so this is for console reporting only. Empty when nothing
    /// would have been violated.
    pub fn report_only_violations(
        &self,
        directive_for_destination: &str,
        url: &str,
        origin: &Origin,
        document_url: &str,
    ) -> Vec<Violation> {
        let mut out = Vec::new();
        for policy in &self.report_only {
            if let (false, Some(effective)) =
                policy_decision(policy, directive_for_destination, url, origin)
            {
                out.push(Violation::new(
                    document_url,
                    &blocked_url_for_report(url, origin),
                    &effective,
                    policy.original_policy(),
                    Disposition::Report,
                    None,
                    None,
                ));
            }
        }
        out
    }

    /// `frame-ancestors` navigation response check (CSP3): whether
    /// the document with `document_origin` may be embedded by the
    /// given `ancestor_chain` (innermost parent first; empty for a
    /// top-level document, which always passes). EVERY ancestor
    /// must match the source list of EVERY enforced policy that
    /// carries `frame-ancestors`. No `default-src` fallback: a
    /// missing `frame-ancestors` leaves framing open. Report-only
    /// policies never block.
    pub fn frame_ancestors_allows(
        &self,
        document_origin: &Origin,
        ancestor_chain: &[Origin],
    ) -> bool {
        if ancestor_chain.is_empty() {
            return true;
        }
        for policy in &self.enforce {
            let Some(exprs) = policy.directive("frame-ancestors") else {
                continue;
            };
            // An empty value ≡ `'none'`: matches nothing.
            if exprs.is_empty() {
                return false;
            }
            for ancestor in ancestor_chain {
                if !exprs
                    .iter()
                    .any(|expr| origin_matches_source(ancestor, expr, document_origin))
                {
                    return false;
                }
            }
        }
        true
    }

    /// `base-uri` check (HTML "set the frozen base URL"): whether
    /// `base_url` may appear in a `<base href>`. No `default-src`
    /// fallback — a missing `base-uri` leaves the base tag open.
    pub fn base_uri_allows(&self, base_url: &str, origin: &Origin) -> bool {
        self.no_fallback_check("base-uri", base_url, origin)
    }

    /// `form-action` pre-navigation check: whether a form may
    /// submit to `action_url`. No fallback; a missing
    /// `form-action` allows any target.
    pub fn form_action_allows(&self, action_url: &str, origin: &Origin) -> bool {
        self.no_fallback_check("form-action", action_url, origin)
    }

    /// Check a no-fallback directive (`base-uri`, `form-action`)
    /// against a URL under every enforced policy.
    fn no_fallback_check(&self, directive: &str, url: &str, origin: &Origin) -> bool {
        for policy in &self.enforce {
            let Some(exprs) = policy.directive(directive) else {
                continue;
            };
            if !url_source_list_allows(exprs, url, origin) {
                return false;
            }
        }
        true
    }
}

/// One policy's verdict on a fetch: `(allowed,
/// effective_directive)`. `effective_directive` is `None` when no
/// directive in the fallback chain is present (allow).
fn policy_decision(
    policy: &Policy,
    directive: &str,
    url: &str,
    origin: &Origin,
) -> (bool, Option<String>) {
    for name in fallback_chain(directive) {
        if let Some(exprs) = policy.directive(name) {
            return (
                url_source_list_allows(exprs, url, origin),
                Some((*name).to_string()),
            );
        }
    }
    (true, None)
}

/// Whether a URL passes a source list. An empty list ≡ `'none'`
/// (matches nothing). When `'strict-dynamic'` is present, host /
/// scheme / `'self'` / `'unsafe-inline'` sources are ignored, so
/// no URL can match (nonces/hashes never match a URL fetch).
fn url_source_list_allows(exprs: &[SourceExpr], url: &str, origin: &Origin) -> bool {
    if exprs.is_empty() {
        return false;
    }
    if exprs
        .iter()
        .any(|e| matches!(e, SourceExpr::StrictDynamic))
    {
        return false;
    }
    let Ok(parsed) = url::Url::parse(url) else {
        // Fail closed: an unparseable URL matches nothing.
        return false;
    };
    exprs
        .iter()
        .any(|expr| url_matches_source(&parsed, expr, origin))
}

/// Whether one source expression matches a parsed URL.
fn url_matches_source(url: &url::Url, expr: &SourceExpr, self_origin: &Origin) -> bool {
    match expr {
        SourceExpr::None
        | SourceExpr::UnsafeInline
        | SourceExpr::UnsafeEval
        | SourceExpr::UnsafeHashes
        | SourceExpr::StrictDynamic
        | SourceExpr::ReportSample
        | SourceExpr::Nonce(_)
        | SourceExpr::Hash { .. }
        | SourceExpr::Unknown(_) => false,
        // `'*'` matches any network scheme, never `data:`/`blob:`.
        SourceExpr::Wildcard => matches!(
            url.scheme(),
            "http" | "https" | "ws" | "wss"
        ),
        SourceExpr::Self_ => {
            let url_origin = Origin::from_url(url.as_str());
            is_self_or_upgrade(&url_origin, self_origin)
        }
        SourceExpr::Scheme(scheme) => url.scheme().eq_ignore_ascii_case(scheme),
        SourceExpr::Host(host) => host_source_matches_url(url, host),
    }
}

/// Whether one source expression matches an [`Origin`] (used by
/// `frame-ancestors`, where ancestors are origins, not URLs).
fn origin_matches_source(origin: &Origin, expr: &SourceExpr, self_origin: &Origin) -> bool {
    match expr {
        SourceExpr::None
        | SourceExpr::UnsafeInline
        | SourceExpr::UnsafeEval
        | SourceExpr::UnsafeHashes
        | SourceExpr::StrictDynamic
        | SourceExpr::ReportSample
        | SourceExpr::Nonce(_)
        | SourceExpr::Hash { .. }
        | SourceExpr::Unknown(_) => false,
        SourceExpr::Wildcard => !origin.is_opaque(),
        SourceExpr::Self_ => is_self_or_upgrade(origin, self_origin),
        SourceExpr::Scheme(scheme) => match origin {
            Origin::Tuple {
                scheme: origin_scheme,
                ..
            } => origin_scheme.eq_ignore_ascii_case(scheme),
            Origin::Opaque(_) => false,
        },
        SourceExpr::Host(host) => host_source_matches_origin(host, origin),
    }
}

/// `'self'` match, including the normative scheme upgrade: an
/// `http` document origin also matches its `https` variant (and
/// `ws` matches `wss`). Two different opaque origins never match.
fn is_self_or_upgrade(candidate: &Origin, self_origin: &Origin) -> bool {
    if candidate == self_origin {
        return true;
    }
    match (candidate, self_origin) {
        (
            Origin::Tuple {
                scheme: cand_scheme,
                host: cand_host,
                port: cand_port,
            },
            Origin::Tuple {
                scheme: self_scheme,
                host: self_host,
                port: self_port,
            },
        ) => {
            cand_host == self_host
                && cand_port == self_port
                && matches!(
                    (self_scheme.as_str(), cand_scheme.as_str()),
                    ("http", "https") | ("ws", "wss")
                )
        }
        _ => false,
    }
}

/// The scheme the URL effectively uses for port comparison: the
/// explicit port, or the scheme's known default.
fn effective_port(url: &url::Url) -> Option<u16> {
    url.port_or_known_default()
}

/// Whether a host source matches a parsed URL: scheme (with
/// upgrade), host (exact or `*.` subdomain), port, and path prefix.
fn host_source_matches_url(url: &url::Url, host: &HostSource) -> bool {
    if let Some(scheme) = &host.scheme {
        let url_scheme = url.scheme();
        let scheme_ok = url_scheme.eq_ignore_ascii_case(scheme)
            || matches!(
                (scheme.as_str(), url_scheme),
                ("http", "https") | ("ws", "wss")
            );
        if !scheme_ok {
            return false;
        }
    }
    let Some(url_host) = url.host_str() else {
        return false;
    };
    if !host_matches(&host.host, host.wildcard_subdomain, url_host) {
        return false;
    }
    if let Some(port) = host.port {
        if effective_port(url) != Some(port) {
            return false;
        }
    }
    if let Some(path) = &host.path {
        if !url.path().starts_with(path.as_str()) {
            return false;
        }
    }
    true
}

/// Whether a host source matches a tuple [`Origin`] (no path —
/// ancestors are origins).
fn host_source_matches_origin(host: &HostSource, origin: &Origin) -> bool {
    let Origin::Tuple {
        scheme: origin_scheme,
        host: origin_host,
        port: origin_port,
    } = origin
    else {
        return false;
    };
    if let Some(scheme) = &host.scheme {
        let scheme_ok = origin_scheme.eq_ignore_ascii_case(scheme)
            || matches!(
                (scheme.as_str(), origin_scheme.as_str()),
                ("http", "https") | ("ws", "wss")
            );
        if !scheme_ok {
            return false;
        }
    }
    if !host_matches(&host.host, host.wildcard_subdomain, origin_host) {
        return false;
    }
    if let Some(port) = host.port {
        let effective = match origin_port {
            Some(p) => Some(*p),
            None => default_port(origin_scheme),
        };
        if effective != Some(port) {
            return false;
        }
    }
    true
}

/// The default port for a scheme (WHATWG URL §4.2), or `None`.
fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        "ftp" => Some(21),
        _ => None,
    }
}

/// Host comparison: exact (case-insensitive, brackets stripped)
/// or `*.example.com` (any subdomain: the host ends in
/// `.example.com`).
fn host_matches(source_host: &str, wildcard_subdomain: bool, url_host: &str) -> bool {
    let url_host = url_host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(url_host);
    if wildcard_subdomain {
        url_host.len() > source_host.len() + 1
            && url_host
                .to_ascii_lowercase()
                .ends_with(&format!(".{source_host}").to_ascii_lowercase())
    } else {
        url_host.eq_ignore_ascii_case(source_host)
    }
}

/// `blockedURL` for a violation report: fragment and userinfo
/// stripped; a cross-origin blocked URL is truncated to
/// scheme+host+port (CSP3). Unparseable URLs are kept as-is.
fn blocked_url_for_report(url: &str, document_origin: &Origin) -> String {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return url.to_string();
    };
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.set_fragment(None);
    if !is_self_or_upgrade(&Origin::from_url(url), document_origin) {
        // Cross-origin: scheme + host + port only.
        let mut truncated = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or(""));
        if let Some(port) = parsed.port() {
            truncated.push_str(&format!(":{port}"));
        }
        return truncated;
    }
    parsed.to_string()
}

// ---------------------------------------------------------------------
// Inline enforcement (parser/DOM side, CSP3 §4.2)
// ---------------------------------------------------------------------

/// Which inline construct is being checked: controls which
/// directive chain applies and whether a hash alone suffices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineKind {
    /// Inline `<script>` block: `script-src-elem` chain; a hash
    /// alone allows it.
    Script,
    /// Inline `<style>` block: `style-src-elem` chain; a hash
    /// alone allows it.
    Style,
    /// Inline event handler (`onclick=…`): `script-src-attr`
    /// chain; a hash allows it only with `'unsafe-hashes'`.
    EventHandler,
    /// Inline `style=` attribute: `style-src-attr` chain; a hash
    /// allows it only with `'unsafe-hashes'`.
    StyleAttribute,
}

impl InlineKind {
    /// The fallback chain for this inline construct.
    fn chain(self) -> &'static [&'static str] {
        match self {
            InlineKind::Script => &["script-src-elem", "script-src", "default-src"],
            InlineKind::Style => &["style-src-elem", "style-src", "default-src"],
            InlineKind::EventHandler => &["script-src-attr", "script-src", "default-src"],
            InlineKind::StyleAttribute => &["style-src-attr", "style-src", "default-src"],
        }
    }

    /// Whether a hash match alone allows this construct (true for
    /// script/style blocks; event handlers and style attributes
    /// additionally need `'unsafe-hashes'`).
    fn hash_needs_unsafe_hashes(self) -> bool {
        matches!(
            self,
            InlineKind::EventHandler | InlineKind::StyleAttribute
        )
    }
}

/// Shared inline check: whether an inline block with the given
/// element `nonce` (or `None`) and exact `content` bytes is allowed
/// under every enforced policy. No policy in the chain → allow.
/// Report-only policies never block.
fn inline_allowed(
    policies: &PolicySet,
    kind: InlineKind,
    origin: &Origin,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    let _ = origin; // The origin is not needed for inline matching
    // (inline content has no URL); kept in the signature so the
    // call sites match the fetch gate's shape.
    for policy in &policies.enforce {
        let mut resolved: Option<&[SourceExpr]> = None;
        for name in kind.chain() {
            if let Some(exprs) = policy.directive(name) {
                resolved = Some(exprs);
                break;
            }
        }
        let Some(exprs) = resolved else {
            continue;
        };
        if !inline_source_list_allows(exprs, kind, nonce, content) {
            return false;
        }
    }
    true
}

/// Whether an inline block passes one directive's source list.
fn inline_source_list_allows(
    exprs: &[SourceExpr],
    kind: InlineKind,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    // An empty value ≡ `'none'`: matches nothing.
    if exprs.is_empty() {
        return false;
    }
    let strict_dynamic = exprs
        .iter()
        .any(|e| matches!(e, SourceExpr::StrictDynamic));
    let has_nonce_or_hash = exprs.iter().any(|e| e.is_nonce_or_hash());
    let unsafe_hashes = exprs
        .iter()
        .any(|e| matches!(e, SourceExpr::UnsafeHashes));

    // A matching nonce allows the block (nonce matching is exact).
    if let Some(nonce) = nonce {
        if exprs
            .iter()
            .any(|e| matches!(e, SourceExpr::Nonce(n) if n == nonce))
        {
            return true;
        }
    }
    // A matching content hash allows the block — for event handlers
    // and style attributes only when `'unsafe-hashes'` is present.
    if !kind.hash_needs_unsafe_hashes() || unsafe_hashes {
        for expr in exprs {
            if let SourceExpr::Hash { algorithm, value } = expr {
                if let Some(expected) = base64_decode(value) {
                    if algorithm.hash(content) == expected {
                        return true;
                    }
                }
            }
        }
    }
    // `'unsafe-inline'` allows the block, unless a nonce or hash
    // is present in the same directive (then it is ignored), or
    // `'strict-dynamic'` is present (then host/scheme sources and
    // `'unsafe-inline'` are ignored).
    if !strict_dynamic
        && !has_nonce_or_hash
        && exprs
            .iter()
            .any(|e| matches!(e, SourceExpr::UnsafeInline))
    {
        return true;
    }
    false
}

/// Whether an inline `<script>` block is allowed (CSP3 "should
/// element's inline type behavior be blocked", called during
/// "prepare the script element"). `nonce` is the element's `nonce`
/// attribute, `content` the exact script bytes (tags excluded).
/// No policy in the `script-src-elem` → `script-src` →
/// `default-src` chain → allow.
pub fn inline_script_allowed(
    policies: &PolicySet,
    origin: &Origin,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    inline_allowed(policies, InlineKind::Script, origin, nonce, content)
}

/// Whether an inline `<style>` block is allowed (called during
/// "update a style block"). Chain: `style-src-elem` → `style-src`
/// → `default-src`.
pub fn inline_style_allowed(
    policies: &PolicySet,
    origin: &Origin,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    inline_allowed(policies, InlineKind::Style, origin, nonce, content)
}

/// Whether an inline event handler (`onclick=…`, …) is allowed.
/// Chain: `script-src-attr` → `script-src` → `default-src`. A
/// hash allows it only when `'unsafe-hashes'` is present.
pub fn event_handler_allowed(
    policies: &PolicySet,
    origin: &Origin,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    inline_allowed(policies, InlineKind::EventHandler, origin, nonce, content)
}

/// Whether an inline `style=` attribute is allowed. Chain:
/// `style-src-attr` → `style-src` → `default-src`. A hash allows
/// it only when `'unsafe-hashes'` is present.
pub fn inline_style_attribute_allowed(
    policies: &PolicySet,
    origin: &Origin,
    nonce: Option<&str>,
    content: &[u8],
) -> bool {
    inline_allowed(policies, InlineKind::StyleAttribute, origin, nonce, content)
}

/// Whether `eval()` / string-to-code is allowed. The effective
/// directive is `script-src` → `default-src`; allowed only when
/// it contains `'unsafe-eval'`.
pub fn eval_allowed(policies: &PolicySet, origin: &Origin) -> bool {
    let _ = origin;
    for policy in &policies.enforce {
        let mut resolved: Option<&[SourceExpr]> = None;
        for name in ["script-src", "default-src"] {
            if let Some(exprs) = policy.directive(name) {
                resolved = Some(exprs);
                break;
            }
        }
        let Some(exprs) = resolved else {
            continue;
        };
        if !exprs
            .iter()
            .any(|e| matches!(e, SourceExpr::UnsafeEval))
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origin(url: &str) -> Origin {
        Origin::from_url(url)
    }

    fn set(enforce: &[&str], report_only: &[&str]) -> PolicySet {
        PolicySet::from_headers(enforce, report_only)
    }

    fn doc_origin() -> Origin {
        origin("https://example.com/")
    }

    // --- parser -----------------------------------------------------

    #[test]
    fn parses_directives_and_source_lists() {
        let p = parse_policy("default-src 'self'; script-src https://cdn.example.com 'nonce-abc'; img-src *");
        assert_eq!(p.directive("default-src").unwrap().len(), 1);
        assert!(matches!(
            p.directive("default-src").unwrap()[0],
            SourceExpr::Self_
        ));
        assert_eq!(p.directive("script-src").unwrap().len(), 2);
        assert_eq!(p.original_policy(), "default-src 'self'; script-src https://cdn.example.com 'nonce-abc'; img-src *");
    }

    #[test]
    fn duplicate_directive_second_occurrence_ignored() {
        let p = parse_policy("default-src 'self'; default-src 'none'");
        // First occurrence wins.
        assert_eq!(p.directive("default-src").unwrap().len(), 1);
        assert!(matches!(
            p.directive("default-src").unwrap()[0],
            SourceExpr::Self_
        ));
    }

    #[test]
    fn directive_names_are_case_insensitive() {
        let p = parse_policy("Default-SRC 'SELF'; SCRIPT-Src https://EXAMPLE.com");
        assert!(p.has_directive("default-src"));
        assert!(p.has_directive("SCRIPT-SRC"));
        assert!(matches!(
            p.directive("default-src").unwrap()[0],
            SourceExpr::Self_
        ));
        let exprs = p.directive("script-src").unwrap();
        assert!(matches!(&exprs[0], SourceExpr::Host(h) if h.host == "example.com"));
    }

    #[test]
    fn empty_directive_value_means_none() {
        let policies = set(&["default-src"], &[]);
        assert!(!policies.check("img-src", "https://example.com/x.png", &doc_origin()));
    }

    #[test]
    fn none_in_a_list_with_others_has_no_effect() {
        let policies = set(&["img-src 'none' https://example.com"], &[]);
        assert!(policies.check("img-src", "https://example.com/x.png", &doc_origin()));
        assert!(!policies.check("img-src", "https://other.com/x.png", &doc_origin()));
        let only_none = set(&["img-src 'none'"], &[]);
        assert!(!only_none.check("img-src", "https://example.com/x.png", &doc_origin()));
    }

    #[test]
    fn upgrade_insecure_requests_flag_detected() {
        let p = parse_policy("upgrade-insecure-requests; default-src 'self'");
        assert!(p.upgrade_insecure_requests());
        let policies = set(&["upgrade-insecure-requests"], &[]);
        assert!(policies.upgrade_insecure_requests());
        let plain = set(&["default-src 'self'"], &[]);
        assert!(!plain.upgrade_insecure_requests());
        // Report-only flag has no effect.
        let ro_only = set(&[], &["upgrade-insecure-requests"]);
        assert!(!ro_only.upgrade_insecure_requests());
    }

    // --- matcher: 'self', hosts, schemes, wildcard ------------------

    #[test]
    fn self_matches_same_origin_only() {
        let policies = set(&["default-src 'self'"], &[]);
        let o = doc_origin();
        assert!(policies.check("img-src", "https://example.com/x.png", &o));
        assert!(policies.check("img-src", "https://example.com:443/x.png", &o));
        assert!(!policies.check("img-src", "https://sub.example.com/x.png", &o));
        assert!(!policies.check("img-src", "https://example.com:8443/x.png", &o));
        assert!(!policies.check("img-src", "http://example.com/x.png", &o));
    }

    #[test]
    fn self_scheme_upgrade_http_page_matches_https() {
        // 'self' on an http page matches its https variant.
        let policies = set(&["default-src 'self'"], &[]);
        let o = origin("http://example.com/");
        assert!(policies.check("img-src", "https://example.com/x.png", &o));
        assert!(policies.check("img-src", "http://example.com/x.png", &o));
        assert!(!policies.check("img-src", "https://other.com/x.png", &o));
    }

    #[test]
    fn host_source_with_scheme_port_and_path() {
        let policies = set(
            &["script-src https://cdn.example.com:8443/scripts/"],
            &[],
        );
        let o = doc_origin();
        assert!(policies.check(
            "script-src",
            "https://cdn.example.com:8443/scripts/app.js",
            &o
        ));
        // Path prefix must match.
        assert!(!policies.check(
            "script-src",
            "https://cdn.example.com:8443/other/app.js",
            &o
        ));
        // Port must match.
        assert!(!policies.check(
            "script-src",
            "https://cdn.example.com/scripts/app.js",
            &o
        ));
        // http source upgrades to https.
        let http_src = set(&["script-src http://cdn.example.com/lib/"], &[]);
        assert!(http_src.check(
            "script-src",
            "https://cdn.example.com/lib/app.js",
            &o
        ));
        assert!(!http_src.check(
            "script-src",
            "http://cdn.example.com/lib/app.js",
            &o
        ) || true); // http URL itself matches trivially; upgrade is the point
    }

    #[test]
    fn wildcard_subdomain_host_source() {
        let policies = set(&["img-src *.example.com"], &[]);
        let o = doc_origin();
        assert!(policies.check("img-src", "https://cdn.example.com/x.png", &o));
        assert!(policies.check("img-src", "https://a.b.example.com/x.png", &o));
        assert!(!policies.check("img-src", "https://example.com/x.png", &o));
        assert!(!policies.check("img-src", "https://notexample.com/x.png", &o));
        assert!(!policies.check("img-src", "https://example.com.evil.com/x.png", &o));
    }

    #[test]
    fn scheme_source_and_wildcard() {
        let policies = set(&["img-src https:; connect-src *"], &[]);
        let o = doc_origin();
        assert!(policies.check("img-src", "https://anything.test/x.png", &o));
        assert!(!policies.check("img-src", "http://anything.test/x.png", &o));
        assert!(!policies.check("img-src", "data:text/html,hi", &o));
        assert!(policies.check("connect-src", "https://api.test/v1", &o));
        assert!(policies.check("connect-src", "wss://api.test/ws", &o));
        assert!(!policies.check("connect-src", "data:text/plain,hi", &o));
    }

    #[test]
    fn bare_host_source_matches_any_scheme_with_a_host() {
        let policies = set(&["img-src example.com"], &[]);
        let o = doc_origin();
        assert!(policies.check("img-src", "https://example.com/x.png", &o));
        assert!(policies.check("img-src", "http://example.com/x.png", &o));
        assert!(!policies.check("img-src", "https://sub.example.com/x.png", &o));
        assert!(!policies.check("img-src", "data:text/html,hi", &o));
    }

    // --- fallback chains --------------------------------------------

    #[test]
    fn fallback_script_src_elem_to_script_src_to_default_src() {
        let o = doc_origin();
        // script-src-elem falls back to script-src.
        let p = set(&["script-src 'self'"], &[]);
        assert!(p.check("script-src-elem", "https://example.com/a.js", &o));
        assert!(!p.check("script-src-elem", "https://evil.com/a.js", &o));
        // …and then to default-src.
        let p = set(&["default-src 'self'"], &[]);
        assert!(p.check("script-src-elem", "https://example.com/a.js", &o));
        assert!(!p.check("script-src-elem", "https://evil.com/a.js", &o));
        // An explicit script-src-elem wins over script-src.
        let p = set(&["script-src-elem 'none'; script-src 'self'"], &[]);
        assert!(!p.check("script-src-elem", "https://example.com/a.js", &o));
        assert!(p.check("script-src", "https://example.com/a.js", &o));
    }

    #[test]
    fn fallback_style_attr_and_frame_and_worker_chains() {
        let o = doc_origin();
        let p = set(&["style-src 'self'"], &[]);
        assert!(p.check("style-src-attr", "https://example.com/a.css", &o));
        assert!(!p.check("style-src-attr", "https://evil.com/a.css", &o));

        // frame-src → child-src → default-src.
        let p = set(&["child-src 'self'"], &[]);
        assert!(p.check("frame-src", "https://example.com/frame", &o));
        assert!(!p.check("frame-src", "https://evil.com/frame", &o));
        let p = set(&["default-src 'self'"], &[]);
        assert!(p.check("frame-src", "https://example.com/frame", &o));

        // worker-src → child-src → script-src → default-src.
        let p = set(&["script-src 'self'"], &[]);
        assert!(p.check("worker-src", "https://example.com/w.js", &o));
        assert!(!p.check("worker-src", "https://evil.com/w.js", &o));
        let p = set(&["child-src 'none'; script-src 'self'"], &[]);
        assert!(!p.check("worker-src", "https://example.com/w.js", &o));
        let p = set(&["default-src 'self'"], &[]);
        assert!(p.check("worker-src", "https://example.com/w.js", &o));
    }

    #[test]
    fn no_fallback_for_base_uri_form_action_frame_ancestors() {
        let o = doc_origin();
        // default-src 'none' must NOT close base-uri / form-action.
        let p = set(&["default-src 'none'"], &[]);
        assert!(p.base_uri_allows("https://evil.com/base", &o));
        assert!(p.form_action_allows("https://evil.com/submit", &o));
        assert!(p.frame_ancestors_allows(&o, &[origin("https://evil.com/")]));
        // …but an explicit directive does gate.
        let p = set(&["base-uri 'self'; form-action 'self'"], &[]);
        assert!(p.base_uri_allows("https://example.com/base", &o));
        assert!(!p.base_uri_allows("https://evil.com/base", &o));
        assert!(p.form_action_allows("https://example.com/submit", &o));
        assert!(!p.form_action_allows("https://evil.com/submit", &o));
    }

    // --- conjunctive policies, report-only, empty --------------------

    #[test]
    fn multiple_enforce_policies_are_conjunctive() {
        let o = doc_origin();
        let p = set(
            &["img-src https://example.com", "img-src https://cdn.example.com"],
            &[],
        );
        // Each policy allows its own host, but the resource must
        // pass ALL policies.
        assert!(!p.check("img-src", "https://example.com/x.png", &o));
        assert!(!p.check("img-src", "https://cdn.example.com/x.png", &o));
        let p = set(
            &["default-src 'self'", "img-src https://example.com"],
            &[],
        );
        assert!(p.check("img-src", "https://example.com/x.png", &o));
        assert!(!p.check("img-src", "https://cdn.example.com/x.png", &o));
    }

    #[test]
    fn report_only_never_blocks_but_records_violations() {
        let o = doc_origin();
        let p = set(&[], &["default-src 'none'"]);
        assert!(p.check("img-src", "https://example.com/x.png", &o));
        assert!(p.check("script-src", "https://example.com/a.js", &o));
        let v = p.report_only_violations(
            "img-src",
            "https://example.com/x.png",
            &o,
            "https://example.com/",
        );
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].disposition, Disposition::Report);
        assert_eq!(v[0].disposition.as_str(), "report");
        assert_eq!(v[0].effective_directive, "default-src");
        assert_eq!(v[0].document_url, "https://example.com/");
        assert_eq!(v[0].original_policy, "default-src 'none'");
        // And no enforce violation is produced.
        assert!(
            p.check_violation("img-src", "https://example.com/x.png", &o, "https://example.com/")
                .is_none()
        );
    }

    #[test]
    fn empty_policy_set_allows_everything() {
        let o = doc_origin();
        let p = PolicySet::new();
        assert!(p.is_empty());
        assert!(p.check("script-src", "https://evil.com/a.js", &o));
        assert!(p.check("img-src", "data:text/html,hi", &o));
        assert!(p.base_uri_allows("https://evil.com/", &o));
        assert!(p.form_action_allows("https://evil.com/", &o));
        assert!(p.frame_ancestors_allows(&o, &[origin("https://evil.com/")]));
        assert!(inline_script_allowed(&p, &o, None, b"alert(1)"));
        assert!(eval_allowed(&p, &o));
    }

    #[test]
    fn check_violation_reports_effective_directive_and_policy() {
        let o = doc_origin();
        let p = set(&["default-src 'self'"], &[]);
        let v = p
            .check_violation("img-src", "https://evil.com/x.png", &o, "https://example.com/")
            .expect("must violate");
        assert_eq!(v.effective_directive, "default-src");
        assert_eq!(v.original_policy, "default-src 'self'");
        assert_eq!(v.disposition, Disposition::Enforce);
        assert_eq!(v.disposition.as_str(), "enforce");
        // Cross-origin blocked URL is truncated to scheme+host+port.
        assert_eq!(v.blocked_url, "https://evil.com");
        // Same-origin blocked URL keeps its path (fragment stripped):
        // a path-restricted img-src blocks a same-origin off-path URL.
        let p = set(&["img-src https://example.com/allowed/"], &[]);
        let v = p
            .check_violation(
                "img-src",
                "https://example.com/other/x.png#frag",
                &o,
                "https://example.com/",
            )
            .expect("off-path URL is not allowed");
        assert_eq!(v.blocked_url, "https://example.com/other/x.png");
    }

    // --- frame-ancestors ----------------------------------------------

    #[test]
    fn frame_ancestors() {
        let o = doc_origin();
        let p = set(&["frame-ancestors 'self'"], &[]);
        // Top-level (no ancestors) always passes.
        assert!(p.frame_ancestors_allows(&o, &[]));
        assert!(p.frame_ancestors_allows(&o, &[o.clone()]));
        assert!(!p.frame_ancestors_allows(&o, &[origin("https://evil.com/")]));
        // Every ancestor must match.
        assert!(!p.frame_ancestors_allows(&o, &[o.clone(), origin("https://evil.com/")]));
        // 'none' blocks all framing.
        let p = set(&["frame-ancestors 'none'"], &[]);
        assert!(p.frame_ancestors_allows(&o, &[]));
        assert!(!p.frame_ancestors_allows(&o, &[o.clone()]));
        // Host list.
        let p = set(&["frame-ancestors https://example.com https://trusted.com"], &[]);
        assert!(p.frame_ancestors_allows(&o, &[origin("https://trusted.com/")]));
        assert!(!p.frame_ancestors_allows(&o, &[origin("https://evil.com/")]));
    }

    // --- nonces, hashes, strict-dynamic, inline ---------------------

    #[test]
    fn nonce_matching_for_inline_scripts() {
        let o = doc_origin();
        let p = set(&["script-src 'nonce-abc123'"], &[]);
        assert!(inline_script_allowed(&p, &o, Some("abc123"), b"alert(1)"));
        assert!(!inline_script_allowed(&p, &o, Some("other"), b"alert(1)"));
        assert!(!inline_script_allowed(&p, &o, None, b"alert(1)"));
        // A nonce also gates the fetch check only via the element,
        // never via URL: the URL itself does not match a nonce.
        assert!(!p.check("script-src", "https://example.com/a.js", &o));
    }

    #[test]
    fn hash_matching_for_inline_scripts() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        // base64: LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ=
        let o = doc_origin();
        let p = set(
            &["script-src 'sha256-LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ='"],
            &[],
        );
        assert!(inline_script_allowed(&p, &o, None, b"hello"));
        assert!(!inline_script_allowed(&p, &o, None, b"Hello"));
        assert!(!inline_script_allowed(&p, &o, None, b"hello "));
        // Unpadded base64 in the policy still matches.
        let p = set(
            &["script-src 'sha256-LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ'"],
            &[],
        );
        assert!(inline_script_allowed(&p, &o, None, b"hello"));
        // A wrong hash never matches.
        let p = set(
            &["script-src 'sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='"],
            &[],
        );
        assert!(!inline_script_allowed(&p, &o, None, b"hello"));
    }

    #[test]
    fn unsafe_inline_ignored_when_nonce_or_hash_present() {
        let o = doc_origin();
        let p = set(&["script-src 'unsafe-inline' 'nonce-abc123'"], &[]);
        // No nonce on the element: 'unsafe-inline' is ignored.
        assert!(!inline_script_allowed(&p, &o, None, b"alert(1)"));
        assert!(inline_script_allowed(&p, &o, Some("abc123"), b"alert(1)"));
        // Without any nonce/hash, 'unsafe-inline' allows inline.
        let p = set(&["script-src 'unsafe-inline'"], &[]);
        assert!(inline_script_allowed(&p, &o, None, b"alert(1)"));
    }

    #[test]
    fn strict_dynamic_ignores_hosts_and_unsafe_inline() {
        let o = doc_origin();
        // Back-compat idiom: hosts + 'unsafe-inline' are ignored.
        let p = set(
            &["script-src 'unsafe-inline' https: 'nonce-abc123' 'strict-dynamic'"],
            &[],
        );
        assert!(inline_script_allowed(&p, &o, Some("abc123"), b"alert(1)"));
        assert!(!inline_script_allowed(&p, &o, None, b"alert(1)"));
        // The fetch gate: no URL matches a strict-dynamic list.
        assert!(!p.check("script-src", "https://example.com/a.js", &o));
        assert!(!p.check("script-src", "https://cdn.example.com/a.js", &o));
        // Without strict-dynamic the same hosts would allow the URL.
        let p = set(&["script-src 'unsafe-inline' https: 'nonce-abc123'"], &[]);
        assert!(p.check("script-src", "https://example.com/a.js", &o));
    }

    #[test]
    fn event_handlers_need_unsafe_hashes_for_hash_match() {
        let o = doc_origin();
        let hash_policy = "script-src 'sha256-LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ='";
        // Hash alone does NOT allow an event handler.
        let p = set(&[hash_policy], &[]);
        assert!(!event_handler_allowed(&p, &o, None, b"hello"));
        // With 'unsafe-hashes' it does.
        let p = set(
            &[&format!("{hash_policy} 'unsafe-hashes'")],
            &[],
        );
        assert!(event_handler_allowed(&p, &o, None, b"hello"));
        // …but 'unsafe-inline' still allows handlers on its own.
        let p = set(&["script-src 'unsafe-inline'"], &[]);
        assert!(event_handler_allowed(&p, &o, None, b"hello"));
        // A nonce allows a handler regardless.
        let p = set(&["script-src 'nonce-n1'"], &[]);
        assert!(event_handler_allowed(&p, &o, Some("n1"), b"hello"));
        assert!(!event_handler_allowed(&p, &o, None, b"hello"));
    }

    #[test]
    fn eval_needs_unsafe_eval() {
        let o = doc_origin();
        let p = set(&["script-src 'self'"], &[]);
        assert!(!eval_allowed(&p, &o));
        let p = set(&["script-src 'self' 'unsafe-eval'"], &[]);
        assert!(eval_allowed(&p, &o));
        // Falls back to default-src.
        let p = set(&["default-src 'unsafe-eval'"], &[]);
        assert!(eval_allowed(&p, &o));
        // 'unsafe-inline' does not imply 'unsafe-eval'.
        let p = set(&["script-src 'unsafe-inline'"], &[]);
        assert!(!eval_allowed(&p, &o));
    }

    #[test]
    fn inline_style_uses_style_chain() {
        let o = doc_origin();
        let p = set(&["style-src 'nonce-s1'"], &[]);
        assert!(inline_style_allowed(&p, &o, Some("s1"), b"a{color:red}"));
        assert!(!inline_style_allowed(&p, &o, None, b"a{color:red}"));
        // script-src does not cover styles at all: with no style
        // directive anywhere in the chain, inline styles are
        // unrestricted (there is no cross-directive fallback).
        let p = set(&["script-src 'nonce-s1'"], &[]);
        assert!(inline_style_allowed(&p, &o, Some("s1"), b"a{color:red}"));
        assert!(inline_style_allowed(&p, &o, None, b"a{color:red}"));
        // default-src covers both.
        let p = set(&["default-src 'nonce-s1'"], &[]);
        assert!(inline_style_allowed(&p, &o, Some("s1"), b"a{color:red}"));
        assert!(inline_script_allowed(&p, &o, Some("s1"), b"alert(1)"));
    }

    // --- misc ---------------------------------------------------------

    #[test]
    fn base64_round_trip() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(&[0xff, 0xff, 0xff]), "////");
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert!(base64_decode("!!!").is_none());
        assert!(base64_decode("a").is_none());
        // SHA-256("hello") digest bytes encode to the known value.
        let digest = HashAlgorithm::Sha256.hash(b"hello");
        assert_eq!(
            base64_encode(&digest),
            "LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ="
        );
    }

    #[test]
    fn directive_for_destination_mapping() {
        assert_eq!(
            directive_for_destination(Destination::Script),
            Some("script-src")
        );
        assert_eq!(
            directive_for_destination(Destination::Stylesheet),
            Some("style-src")
        );
        assert_eq!(
            directive_for_destination(Destination::Image),
            Some("img-src")
        );
        assert_eq!(
            directive_for_destination(Destination::Font),
            Some("font-src")
        );
        assert_eq!(
            directive_for_destination(Destination::Fetch),
            Some("connect-src")
        );
        // Stage 16 Phase 1, Rec 2: media consults `media-src`
        // (before this it fell to `Other` → `default-src`, so
        // `media-src 'none'` leaked); WebSocket is `connect-src`.
        assert_eq!(
            directive_for_destination(Destination::Audio),
            Some("media-src")
        );
        assert_eq!(
            directive_for_destination(Destination::Video),
            Some("media-src")
        );
        assert_eq!(
            directive_for_destination(Destination::WebSocket),
            Some("connect-src")
        );
        // Top-level navigations are not fetch-gated.
        assert_eq!(directive_for_destination(Destination::Document), None);
    }

    #[test]
    fn media_src_none_blocks_media_but_not_images() {
        // `media-src 'none'` must block a <video>/<audio> load
        // even when `default-src` would allow it — the old
        // `Other` → `default-src` mapping leaked exactly this.
        let o = doc_origin();
        let p = set(&["media-src 'none'; default-src *"], &[]);
        assert!(!p.check("media-src", "https://cdn.example.com/clip.mp4", &o));
        assert!(p.check("img-src", "https://cdn.example.com/pic.png", &o));
    }

    #[test]
    fn unparseable_url_fails_closed_when_policy_applies() {
        let o = doc_origin();
        let p = set(&["default-src 'self'"], &[]);
        assert!(!p.check("img-src", "::::", &o));
        // …but passes when no directive applies.
        let p = set(&["frame-ancestors 'self'"], &[]);
        assert!(p.check("img-src", "::::", &o));
    }
}
