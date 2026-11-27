//! CORS — the M5 fetch gate (Stage 9 Rec 2, research report §c).
//!
//! Enforced here in `pluto-net`, never in the realm: script must
//! never receive the internal response object, only the filtered
//! response this module produces. The real check lives browser-side;
//! `check_fetch_allowed` in `pluto-js` stays a pass-through and the
//! browser-side gate in `pluto-browser/src/js_host.rs` is
//! authoritative.
//!
//! What lives here, per the Fetch Standard as the report records it:
//!
//! * the safelists (methods, request headers, response headers) and
//!   the forbidden-header list, with the 128-byte value / 1024-byte
//!   total traps that silently convert a "simple" request into a
//!   preflighted one;
//! * the preflight (built by the UA, never by script): OPTIONS,
//!   `Accept: */*`, `Access-Control-Request-Method`, and
//!   `Access-Control-Request-Headers` comma-joined with NO space
//!   after the comma, sorted, lowercase unsafe names. Credentials
//!   mode is always `same-origin` for a preflight, even when the
//!   actual request has `credentials: include`;
//! * the preflight cache: 5 s default when `Access-Control-Max-Age`
//!   is absent / unparseable / null, capped at 7200 s (Joel's
//!   decision — the Fetch Standard sets no numeric limit);
//! * the CORS check (`Access-Control-Allow-Origin`, in order), with
//!   the wildcard + credentials refusal and the byte case-sensitive
//!   `true`;
//! * tainting / filtering: what script may see of a response
//!   (`basic`, `cors`, `opaque`);
//! * the redirect helpers: credential/`Authorization` stripping on
//!   cross-origin hops and request-body-header stripping on body
//!   drop.
//!
//! Fail closed: any request the CORS code cannot classify is a
//! network error, never a pass-through. The M4 same-origin refusal
//! survives as the `RequestMode::SameOrigin` behaviour —
//! cross-origin under that mode fails before the network.
//!
//! Two deliberate limits, both forced by the frozen wire shape
//! (`ResourceRequest` carries no method and no script headers, and
//! `PROTOCOL_VERSION` stays 3 — the 3→4 bump belongs to the
//! parallel process-split worker):
//!
//! * The live pipeline can only classify the effective request the
//!   wire can express today: method GET with the UA's own headers,
//!   which is always simple — so a preflight never fires on live
//!   traffic yet. The preflight path is fully built, wired into
//!   `service.rs` at the right gate-order point, and tested here
//!   against a local server with synthetic methods / headers; it
//!   starts firing the moment the wire carries a method and
//!   headers.
//! * `fetch()` `init.mode` / `init.credentials` are parsed and
//!   discarded in `pluto-js` (`FetchRequest` carries neither), so
//!   the browser-side gate uses the `fetch()` defaults — mode
//!   `Cors`, credentials `SameOrigin`. `CredentialsMode::Include`
//!   is fully honoured by this module for any request that carries
//!   it; nothing today can ask for it through `fetch()`.
//!
//! Gate order (frozen, Stage 5): HSTS/mixed → blocker → CSP fetch →
//! CORS → cache → network. The blocker and the cache do not exist
//! yet; the CSP pre-request check is a no-op hook in `service.rs`
//! that slots in BEFORE CORS (see `csp_pre_request_check` there).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use http::header::{self, HeaderMap};
use http::{Method, StatusCode, Uri};
use http_body_util::Full;
use hyper::Request;
use bytes::Bytes;
use pluto_ipc::{CredentialsMode, Destination, Origin, RequestMode};

use crate::client::HttpsClient;

/// Default preflight-cache lifetime (Fetch §4.7): 5 s when
/// `Access-Control-Max-Age` is absent, unparseable, or null.
pub const PREFLIGHT_CACHE_DEFAULT_SECS: u64 = 5;
/// Preflight-cache cap (Joel's decision, matching Chrome's 7200 s).
/// The Fetch Standard sets no numeric limit; the cap is Pluto's.
pub const PREFLIGHT_CACHE_MAX_SECS: u64 = 7200;

/// CORS-safelisted methods: exactly these three (Fetch §2.2.1).
pub fn is_cors_safelisted_method(method: &str) -> bool {
    // Method tokens are case-sensitive; only the exact uppercase
    // forms are safelisted. (The `fetch()` binding uppercases the
    // method before it reaches here.)
    matches!(method, "GET" | "HEAD" | "POST")
}

/// Forbidden methods script may never set (Fetch §2.2.1).
/// Case-insensitive on purpose: fail closed — `connect` must not
/// slip past as a non-safelisted method that merely preflights.
pub fn is_forbidden_method(method: &str) -> bool {
    matches!(
        method.to_ascii_uppercase().as_str(),
        "CONNECT" | "TRACE" | "TRACK"
    )
}

/// CORS-safelisted request-header names (Fetch §2.2.1): exactly
/// these five. `Authorization` is in NEITHER safelist — it is the
/// only CORS non-wildcard header.
fn is_safelisted_header_name(name: &str) -> bool {
    matches!(
        name,
        "accept" | "accept-language" | "content-language" | "content-type" | "range"
    )
}

/// CORS-safelisted response-header names (Fetch §2.2.2): visible to
/// script on a CORS response with no opt-in.
pub const CORS_SAFELISTED_RESPONSE_HEADERS: [&str; 7] = [
    "cache-control",
    "content-language",
    "content-length",
    "content-type",
    "expires",
    "last-modified",
    "pragma",
];

/// Forbidden request-header names (Fetch §2.2.1): the UA owns them,
/// script may never set them. `name` must already be lowercased;
/// `value` is needed for the `X-HTTP-Method(-Override)` rule.
pub fn is_forbidden_request_header(name: &str, value: &str) -> bool {
    match name {
        "accept-charset"
        | "accept-encoding"
        | "access-control-request-headers"
        | "access-control-request-method"
        | "connection"
        | "content-length"
        | "cookie"
        | "cookie2"
        | "date"
        | "dnt"
        | "expect"
        | "host"
        | "keep-alive"
        | "origin"
        | "referer"
        | "set-cookie"
        | "te"
        | "trailer"
        | "transfer-encoding"
        | "upgrade"
        | "via" => true,
        _ => {
            if name.starts_with("proxy-") || name.starts_with("sec-") {
                return true;
            }
            // `X-HTTP-Method(-Override)` is forbidden only when its
            // value parses to a forbidden method.
            if name == "x-http-method" || name == "x-http-method-override" {
                return is_forbidden_method(value.trim());
            }
            false
        }
    }
}

/// The per-name safelist check for one header: the value must be
/// ≤ 128 bytes AND pass the per-name rule (Fetch §2.2.1).
///
/// * `Content-Type` must parse to one of exactly three essences
///   (`application/x-www-form-urlencoded`, `multipart/form-data`,
///   `text/plain`) — the essence is the value before any `;`
///   parameters, lowercased. A second `Content-Type` header is
///   handled by the caller (`classify_request_headers`): two of
///   them are never safelisted (strict MIME parsing, not the
///   forgiving extractor).
/// * `Range` must be a single range with a non-null start
///   (`bytes=-500` is NOT safelisted; `bytes=0-` is).
fn is_safelisted_header_value(name: &str, value: &str) -> bool {
    if value.len() > 128 {
        return false;
    }
    match name {
        "content-type" => {
            let essence = value.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
            matches!(
                essence.as_str(),
                "application/x-www-form-urlencoded"
                    | "multipart/form-data"
                    | "text/plain"
            )
        }
        "range" => is_single_range_with_start(value),
        _ => true,
    }
}

/// `bytes=<start>-<end?>`, exactly one range, start non-empty digits.
fn is_single_range_with_start(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("bytes=") else {
        return false;
    };
    if rest.contains(',') {
        return false;
    }
    let Some((start, _end)) = rest.split_once('-') else {
        return false;
    };
    !start.is_empty() && start.bytes().all(|b| b.is_ascii_digit())
}

/// The classification of one request's method + headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderClassification {
    /// Whether the request needs a preflight before the actual
    /// fetch (non-safelisted method, or any CORS-unsafe header).
    pub needs_preflight: bool,
    /// CORS-unsafe header names, sorted, lowercase, deduped — the
    /// exact list a preflight's `Access-Control-Request-Headers`
    /// carries (comma-joined, no space after the comma).
    pub unsafe_headers: Vec<String>,
}

/// Classify a request's method + script headers.
///
/// Fails (fail closed) when the method is forbidden
/// (`CONNECT`/`TRACE`/`TRACK`) or any header name is forbidden —
/// script may never set those, and silently dropping them would
/// hide a script that believes it sent them.
///
/// A header is CORS-unsafe when its name is not safelisted, or its
/// value fails the per-name check (incl. > 128 bytes), or there are
/// two `Content-Type` headers. If the TOTAL length of all
/// safelisted header values exceeds 1024 bytes, every safelisted
/// name is treated as unsafe.
pub fn classify_request_headers(
    method: &str,
    headers: &[(String, String)],
) -> Result<HeaderClassification, String> {
    if is_forbidden_method(method) {
        return Err(format!("CORS: forbidden method {method}"));
    }
    let mut unsafe_headers: Vec<String> = Vec::new();
    let mut safelisted_total: usize = 0;
    let mut safelisted_names: Vec<String> = Vec::new();
    let mut content_type_count = 0usize;
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        if is_forbidden_request_header(&lower, value) {
            return Err(format!("CORS: forbidden request header {name}"));
        }
        if lower == "content-type" {
            content_type_count += 1;
        }
        if is_safelisted_header_name(&lower) {
            safelisted_total += value.len();
            safelisted_names.push(lower.clone());
            if (content_type_count > 1 || !is_safelisted_header_value(&lower, value))
                && !unsafe_headers.contains(&lower) {
                    unsafe_headers.push(lower);
                }
        } else if !unsafe_headers.contains(&lower) {
            unsafe_headers.push(lower);
        }
    }
    // The 1024-byte total trap: every safelisted name is unsafe.
    if safelisted_total > 1024 {
        for name in safelisted_names {
            if !unsafe_headers.contains(&name) {
                unsafe_headers.push(name);
            }
        }
    }
    unsafe_headers.sort();
    unsafe_headers.dedup();
    Ok(HeaderClassification {
        needs_preflight: !is_cors_safelisted_method(method) || !unsafe_headers.is_empty(),
        unsafe_headers,
    })
}

/// Whether this request needs a preflight (convenience wrapper
/// over [`classify_request_headers`]).
pub fn needs_preflight(
    method: &str,
    headers: &[(String, String)],
) -> Result<bool, String> {
    Ok(classify_request_headers(method, headers)?.needs_preflight)
}

/// The `Origin` request-header value for a request (Fetch §3.2):
/// the byte-serialized request origin, forced to `"null"` when the
/// request origin is a tuple origin with scheme `https` and the URL
/// is not `https`, or when credentials mode is `same-origin` and
/// the URL is not same-origin. An opaque origin always serializes
/// to `"null"`.
///
/// Note the consequence of the second rule, stated plainly: a
/// cross-origin CORS fetch with credentials `same-origin` (the
/// `fetch()` default) sends `Origin: null`. The CORS *check* below
/// still compares the byte-serialized request origin against
/// `Access-Control-Allow-Origin` — so such a fetch passes only
/// when the server returns the literal serialized origin (or `*`),
/// not a reflection of the received `null`.
pub fn origin_header_for(
    request_origin: &Origin,
    url: &Uri,
    credentials: CredentialsMode,
) -> String {
    if request_origin.is_opaque() {
        return "null".to_string();
    }
    let url_origin = Origin::from_url(&url.to_string());
    let url_is_https = url.scheme_str() == Some("https");
    let request_is_https = matches!(
        request_origin,
        Origin::Tuple { scheme, .. } if scheme == "https"
    );
    if (request_is_https && !url_is_https)
        || (credentials == CredentialsMode::SameOrigin
            && !request_origin.is_same_origin(&url_origin))
    {
        return "null".to_string();
    }
    request_origin.serialize()
}

/// All values of one response header, split on commas, trimmed.
/// (`Access-Control-Allow-*` headers are comma lists; a header may
/// also arrive as several header lines.)
fn header_list_values(headers: &HeaderMap, name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for value in headers.get_all(name) {
        if let Ok(text) = value.to_str() {
            for item in text.split(',') {
                let item = item.trim();
                if !item.is_empty() {
                    out.push(item.to_string());
                }
            }
        }
    }
    out
}

/// The CORS check (Fetch §3.3), in order:
///
/// 1. `Access-Control-Allow-Origin` absent → failure.
/// 2. `*` with credentials mode not `include` → success.
/// 3. byte-serialized request origin ≠ value → failure (exact
///    string match — a trailing slash fails).
/// 4. credentials mode not `include` → success.
/// 5. `Access-Control-Allow-Credentials` exactly `true` (byte
///    case-sensitive — `True` fails) → success, else failure.
///
/// Wildcard + credentials `include` ALWAYS fails, even with
/// `Allow-Credentials: true` (step 3 catches it: `*` never equals
/// a serialized origin). More than one `Allow-Origin` header line
/// is a failure (fail closed).
pub fn cors_check(
    request_origin: &Origin,
    credentials: CredentialsMode,
    headers: &HeaderMap,
) -> bool {
    let values: Vec<&header::HeaderValue> =
        headers.get_all(header::ACCESS_CONTROL_ALLOW_ORIGIN).iter().collect();
    if values.len() != 1 {
        return false;
    }
    let Ok(value) = values[0].to_str() else {
        return false;
    };
    if value == "*" {
        return credentials != CredentialsMode::Include;
    }
    if value != request_origin.serialize() {
        return false;
    }
    if credentials != CredentialsMode::Include {
        return true;
    }
    headers
        .get(header::ACCESS_CONTROL_ALLOW_CREDENTIALS)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == "true")
}

/// Statuses for which the CORS check is skipped (Fetch §3.3 trap 8).
pub fn cors_check_is_skipped(status: StatusCode) -> bool {
    status == StatusCode::NOT_MODIFIED || status == StatusCode::PROXY_AUTHENTICATION_REQUIRED
}

/// What tainting a response gets, and therefore what script may see
/// of it (Fetch §3.2 / §4.2). `destination` matters for `NoCors`:
/// a classic subresource (image, stylesheet, script) still delivers
/// its bytes to the parser/renderer — script never sees them
/// through the fetch API — so only a `NoCors` *fetch()* whose
/// final URL is cross-origin is opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseTainting {
    /// Same-origin: status, URL, body; all headers except the
    /// forbidden response headers (`Set-Cookie`, `Set-Cookie2`).
    Basic,
    /// Cross-origin CORS: status, URL, body; only CORS-safelisted
    /// response headers + `Access-Control-Expose-Headers`.
    Cors,
    /// `no-cors` cross-origin fetch: type `"opaque"`, status 0,
    /// empty headers, null body. `response.type` is the only
    /// signal — and Pluto's M4 `Response` has no `type` property
    /// yet (a `pluto-js` change), so script sees status 0 with an
    /// empty body.
    Opaque,
}

pub fn tainting_for(
    mode: RequestMode,
    request_origin: &Origin,
    url: &Uri,
    destination: Destination,
) -> ResponseTainting {
    let url_origin = Origin::from_url(&url.to_string());
    let same_origin = request_origin.is_same_origin(&url_origin);
    match mode {
        RequestMode::Navigate | RequestMode::SameOrigin => ResponseTainting::Basic,
        RequestMode::Cors => {
            if same_origin {
                ResponseTainting::Basic
            } else {
                ResponseTainting::Cors
            }
        }
        RequestMode::NoCors => {
            if !same_origin && destination == Destination::Fetch {
                ResponseTainting::Opaque
            } else {
                ResponseTainting::Basic
            }
        }
    }
}

/// Filter response headers for a `cors`-tainted response: only the
/// CORS-safelisted response headers plus the
/// `Access-Control-Expose-Headers` list. With credentials mode not
/// `include`, `*` in `Expose-Headers` exposes ALL response headers;
/// with `include`, `*` exposes only a header literally named `*`
/// (i.e. nothing useful). `Set-Cookie` / `Set-Cookie2` are never
/// exposed (forbidden response headers).
pub fn filter_cors_response_headers(
    headers: &HeaderMap,
    credentials: CredentialsMode,
) -> Vec<(String, String)> {
    let exposed = header_list_values(headers, header::ACCESS_CONTROL_EXPOSE_HEADERS.as_str());
    let exposed_lower: Vec<String> = exposed.iter().map(|h| h.to_ascii_lowercase()).collect();
    let wildcard_exposes_all =
        exposed_lower.iter().any(|h| h == "*") && credentials != CredentialsMode::Include;
    let mut out = Vec::new();
    for (name, value) in headers.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if lower == "set-cookie" || lower == "set-cookie2" {
            continue;
        }
        let visible = CORS_SAFELISTED_RESPONSE_HEADERS.contains(&lower.as_str())
            || exposed_lower.iter().any(|h| h == &lower)
            || wildcard_exposes_all
            || (credentials == CredentialsMode::Include
                && exposed_lower.iter().any(|h| h == "*")
                && lower == "*");
        if visible {
            out.push((name.as_str().to_string(), value.to_str().unwrap_or("").to_string()));
        }
    }
    out
}

/// Filter response headers for a `basic`-tainted response: all
/// headers except the forbidden response headers (`Set-Cookie`,
/// `Set-Cookie2`).
pub fn filter_basic_response_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(name, _)| {
            let lower = name.as_str().to_ascii_lowercase();
            lower != "set-cookie" && lower != "set-cookie2"
        })
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or("").to_string(),
            )
        })
        .collect()
}

/// Strip the sensitive headers on a cross-origin redirect hop
/// (Fetch §3.3 / report §c): when the hop's origin is not
/// same-origin with the request origin, `Authorization` is removed
/// the moment another origin is seen — and `Cookie` with it, when
/// the jar exists (none today; the removal is written so it stays
/// correct when M5 adds both headers). Same-origin hops keep them.
pub fn strip_sensitive_headers_on_redirect(headers: &mut HeaderMap, crossed_origin: bool) {
    if crossed_origin {
        headers.remove(header::AUTHORIZATION);
        headers.remove(header::COOKIE);
    }
}

/// Delete all request-body-header names (Fetch §3.3): on a redirect
/// that drops the body (301/302 + POST → GET, 303 + non-GET/HEAD →
/// GET), the headers that described the dropped body go with it.
pub fn strip_request_body_headers(headers: &mut HeaderMap) {
    for name in [
        header::CONTENT_ENCODING,
        header::CONTENT_LANGUAGE,
        header::CONTENT_LENGTH,
        header::CONTENT_LOCATION,
        header::CONTENT_TYPE,
        header::TRANSFER_ENCODING,
    ] {
        headers.remove(name);
    }
}

/// Whether a redirect target URL carries credentials (a
/// `user:pass@` userinfo). A CORS redirect to such a URL is a
/// network error when its origin is not same-origin — and when the
/// tainting is already `cors`, when it carries credentials at all
/// (Fetch §3.3 / report §c).
pub fn redirect_url_has_credentials(url: &Uri) -> bool {
    url.authority()
        .is_some_and(|a| a.as_str().contains('@'))
}

// --------------------------------------------------------------- preflight

/// One preflight-cache entry key (Fetch §4.7): the network
/// partition key, the byte-serialized origin, the URL, the
/// credentials boolean, and the method or `*` / header name or `*`
/// the entry was cached for. Pluto stores one entry per exact
/// (method, unsafe-headers) pair — a conservative simplification
/// of the spec's per-method / per-header entries: a lookup only
/// hits for the exact request it was cached for, so Pluto may
/// preflight more often than the spec allows, never less.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreflightCacheKey {
    pub partition_key: String,
    pub origin: String,
    pub url: String,
    pub credentials_include: bool,
    pub method: String,
    pub headers: Vec<String>,
}

/// The preflight cache: successful preflight results, keyed as
/// above, expiring after their `Access-Control-Max-Age` (default
/// 5 s, capped at 7200 s). Interior-mutated so the `Scheduler` can
/// hold it behind an `Arc` like the HSTS store.
pub struct PreflightCache {
    entries: Mutex<HashMap<PreflightCacheKey, Instant>>,
}

impl PreflightCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// A live entry for this exact key, or false (miss / expired).
    /// Expired entries are pruned on access.
    pub fn lookup(&self, key: &PreflightCacheKey) -> bool {
        let mut entries = self.entries.lock().unwrap();
        match entries.get(key) {
            Some(&expiry) if expiry > Instant::now() => true,
            Some(_) => {
                entries.remove(key);
                false
            }
            None => false,
        }
    }

    /// Store a result for `max_age_secs` (clamped to
    /// [`PREFLIGHT_CACHE_MAX_SECS`]). A max-age of 0 stores
    /// nothing — the result must not be reused.
    pub fn store(&self, key: PreflightCacheKey, max_age_secs: u64) {
        if max_age_secs == 0 {
            return;
        }
        let clamped = max_age_secs.min(PREFLIGHT_CACHE_MAX_SECS);
        self.entries.lock().unwrap().insert(
            key,
            Instant::now() + Duration::from_secs(clamped),
        );
    }

    /// Clear every entry for one request (Fetch §4.7): on a
    /// preflight failure the whole fetch returns a network error
    /// and the preflight cache entries for that request are cleared.
    pub fn clear_for_request(
        &self,
        partition_key: &str,
        origin: &str,
        url: &str,
        credentials_include: bool,
    ) {
        self.entries.lock().unwrap().retain(|key, _| {
            !(key.partition_key == partition_key
                && key.origin == origin
                && key.url == url
                && key.credentials_include == credentials_include)
        });
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }
}

impl Default for PreflightCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The effective `Access-Control-Max-Age` for a preflight response:
/// the header's value when it parses as seconds, else the 5 s
/// default; clamped to the 7200 s cap.
pub fn parse_preflight_max_age(headers: &HeaderMap) -> u64 {
    headers
        .get(header::ACCESS_CONTROL_MAX_AGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(PREFLIGHT_CACHE_DEFAULT_SECS)
        .min(PREFLIGHT_CACHE_MAX_SECS)
}

/// Build the preflight request (Fetch §4.7). The UA builds it,
/// never script: method OPTIONS, URL = the actual request's URL,
/// `Accept: */*`, `Access-Control-Request-Method: <actual>`, and —
/// when unsafe headers exist — `Access-Control-Request-Headers:
/// <comma-joined with NO space after the comma, sorted, lowercase
/// unsafe names>`. Credentials mode is always `same-origin` for a
/// preflight, even when the actual request has `credentials:
/// include`.
pub fn build_preflight_request(
    request_origin: &Origin,
    url: &Uri,
    method: &str,
    unsafe_headers: &[String],
) -> Result<Request<Full<Bytes>>, String> {
    let mut builder = Request::builder()
        .method(Method::OPTIONS)
        .uri(url.clone())
        .header(header::ACCEPT, "*/*")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, method)
        .header(
            header::ORIGIN,
            origin_header_for(request_origin, url, CredentialsMode::SameOrigin),
        );
    if !unsafe_headers.is_empty() {
        let joined = unsafe_headers.join(",");
        let value: header::HeaderValue = joined
            .parse()
            .map_err(|e| format!("CORS: preflight headers do not parse: {e}"))?;
        builder = builder.header(header::ACCESS_CONTROL_REQUEST_HEADERS, value);
    }
    builder
        .body(Full::new(Bytes::new()))
        .map_err(|e| format!("CORS: preflight request does not build: {e}"))
}

/// Validate a preflight response (Fetch §4.7). Success requires
/// ALL of:
///
/// * the CORS check passes;
/// * the status is an ok status (200–299) — a 403 preflight fails
///   even with the right headers (a non-preflight CORS response
///   may have any status and still be shared if the headers match);
/// * `Access-Control-Allow-Methods` contains the actual method (or
///   the method is safelisted, or — when credentials mode is not
///   `include` — the value contains `*`);
/// * every CORS-unsafe header is listed in
///   `Access-Control-Allow-Headers` (or `*` when credentials mode
///   is not `include`) — EXCEPT `Authorization`, which `*` NEVER
///   authorizes.
///
/// Returns the effective max-age in seconds on success.
pub fn validate_preflight_response(
    status: StatusCode,
    headers: &HeaderMap,
    method: &str,
    unsafe_headers: &[String],
    credentials: CredentialsMode,
    request_origin: &Origin,
) -> Result<u64, String> {
    if !status.is_success() {
        return Err(format!("CORS: preflight status {status} is not ok (200-299)"));
    }
    if !cors_check(request_origin, credentials, headers) {
        return Err("CORS: preflight response fails the CORS check".to_string());
    }
    let allowed_methods = header_list_values(headers, header::ACCESS_CONTROL_ALLOW_METHODS.as_str());
    let method_allowed = is_cors_safelisted_method(method)
        || allowed_methods.iter().any(|m| m == method)
        || (credentials != CredentialsMode::Include
            && allowed_methods.iter().any(|m| m == "*"));
    if !method_allowed {
        return Err(format!(
            "CORS: preflight does not allow method {method}"
        ));
    }
    let allowed_headers = header_list_values(headers, header::ACCESS_CONTROL_ALLOW_HEADERS.as_str());
    let allowed_lower: Vec<String> =
        allowed_headers.iter().map(|h| h.to_ascii_lowercase()).collect();
    let wildcard = allowed_lower.iter().any(|h| h == "*");
    for unsafe_name in unsafe_headers {
        let listed = allowed_lower.iter().any(|h| h == unsafe_name);
        // `*` never authorizes `Authorization` — the single most
        // likely "works in curl, fails in browser" bug (trap 2).
        let wildcard_covers =
            wildcard && credentials != CredentialsMode::Include && unsafe_name != "authorization";
        if !listed && !wildcard_covers {
            return Err(format!(
                "CORS: preflight does not allow header {unsafe_name}"
            ));
        }
    }
    Ok(parse_preflight_max_age(headers))
}

/// Send a preflight and validate the response, returning the
/// effective max-age in seconds. The response body is meaningless
/// for a preflight and is dropped unread — reading it unbounded
/// would be a DoS vector; dropping it may close the pooled
/// connection, which is safe.
///
/// `credentials` is the ACTUAL request's credentials mode: it
/// decides whether `*` in `Allow-Methods` / `Allow-Headers`
/// authorizes anything (it does not when `include`). The preflight
/// request itself is always sent with credentials mode
/// `same-origin` (see `build_preflight_request`); the two modes
/// must not be conflated.
pub async fn run_preflight(
    client: &HttpsClient,
    request_origin: &Origin,
    url: &Uri,
    method: &str,
    unsafe_headers: &[String],
    credentials: CredentialsMode,
) -> Result<u64, String> {
    let request = build_preflight_request(request_origin, url, method, unsafe_headers)?;
    let response = tokio::time::timeout(Duration::from_secs(10), client.request(request))
        .await
        .map_err(|_| "CORS: preflight timed out".to_string())
        .and_then(|r| r.map_err(|e| format!("CORS: preflight failed: {e}")))?;
    let status = response.status();
    let headers = response.headers().clone();
    // The body is dropped without being read (see above).
    drop(response);
    validate_preflight_response(
        status,
        &headers,
        method,
        unsafe_headers,
        credentials,
        request_origin,
    )
}

/// The cookie-jar intake (Stage 9 Rec 2/3, Fetch §3.3.6) lives in
/// `service.rs`, not here: `Set-Cookie` survives CORS failure —
/// cookies from a failed CORS response are still stored when
/// credentials were included — so intake runs on EVERY response
/// (redirects included) BEFORE / independent of the CORS exposure
/// decision and BEFORE the `Set-Cookie` strip, with the raw
/// headers. The renderer still never sees `Set-Cookie`.
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn origin(url: &str) -> Origin {
        Origin::from_url(url)
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                name.parse::<header::HeaderName>().unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    // --- the CORS check: credentials matrix (§c trap 1, 6) ---

    #[test]
    fn cors_check_credentials_matrix() {
        let req = origin("https://example.com/");
        // `*` with omit / same-origin passes; with include fails.
        assert!(cors_check(
            &req,
            CredentialsMode::Omit,
            &headers(&[("access-control-allow-origin", "*")])
        ));
        assert!(cors_check(
            &req,
            CredentialsMode::SameOrigin,
            &headers(&[("access-control-allow-origin", "*")])
        ));
        assert!(!cors_check(
            &req,
            CredentialsMode::Include,
            &headers(&[("access-control-allow-origin", "*")])
        ));
        // Wildcard + include ALWAYS fails, even with
        // `Allow-Credentials: true`.
        assert!(!cors_check(
            &req,
            CredentialsMode::Include,
            &headers(&[
                ("access-control-allow-origin", "*"),
                ("access-control-allow-credentials", "true"),
            ])
        ));
        // Exact origin + include + `true` passes...
        assert!(cors_check(
            &req,
            CredentialsMode::Include,
            &headers(&[
                ("access-control-allow-origin", "https://example.com"),
                ("access-control-allow-credentials", "true"),
            ])
        ));
        // ...but `True` fails: the value is byte case-sensitive.
        assert!(!cors_check(
            &req,
            CredentialsMode::Include,
            &headers(&[
                ("access-control-allow-origin", "https://example.com"),
                ("access-control-allow-credentials", "True"),
            ])
        ));
        // Exact origin + include without the credentials header fails.
        assert!(!cors_check(
            &req,
            CredentialsMode::Include,
            &headers(&[("access-control-allow-origin", "https://example.com")])
        ));
        // Exact origin + omit passes without any credentials header.
        assert!(cors_check(
            &req,
            CredentialsMode::Omit,
            &headers(&[("access-control-allow-origin", "https://example.com")])
        ));
        // A trailing slash fails: exact string match.
        assert!(!cors_check(
            &req,
            CredentialsMode::Omit,
            &headers(&[("access-control-allow-origin", "https://example.com/")])
        ));
        // Absent `Allow-Origin` fails.
        assert!(!cors_check(&req, CredentialsMode::Omit, &headers(&[])));
        // A different origin fails.
        assert!(!cors_check(
            &req,
            CredentialsMode::Omit,
            &headers(&[("access-control-allow-origin", "https://other.example")])
        ));
    }

    #[test]
    fn cors_check_rejects_duplicate_allow_origin() {
        let req = origin("https://example.com/");
        let mut map = HeaderMap::new();
        map.append(
            header::ACCESS_CONTROL_ALLOW_ORIGIN,
            "https://example.com".parse().unwrap(),
        );
        map.append(
            header::ACCESS_CONTROL_ALLOW_ORIGIN,
            "https://example.com".parse().unwrap(),
        );
        assert!(!cors_check(&req, CredentialsMode::Omit, &map));
    }

    // --- the `Authorization` trap (§c trap 2) ---

    #[test]
    fn preflight_wildcard_allow_headers_never_authorizes_authorization() {
        let req = origin("https://example.com/");
        let ok_headers = headers(&[
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "PUT"),
            ("access-control-allow-headers", "*"),
        ]);
        // `*` covers a custom header...
        assert!(validate_preflight_response(
            StatusCode::OK,
            &ok_headers,
            "PUT",
            &["x-custom".to_string()],
            CredentialsMode::Omit,
            &req,
        )
        .is_ok());
        // ...but NEVER `Authorization`, even with credentials omit.
        assert!(validate_preflight_response(
            StatusCode::OK,
            &ok_headers,
            "PUT",
            &["authorization".to_string()],
            CredentialsMode::Omit,
            &req,
        )
        .is_err());
        // An explicit listing does authorize it.
        let explicit = headers(&[
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "PUT"),
            ("access-control-allow-headers", "X-Custom, Authorization"),
        ]);
        assert!(validate_preflight_response(
            StatusCode::OK,
            &explicit,
            "PUT",
            &["authorization".to_string()],
            CredentialsMode::Omit,
            &req,
        )
        .is_ok());
    }

    #[test]
    fn redirect_strips_authorization_and_cookie_cross_origin_only() {
        let mut map = HeaderMap::new();
        map.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        map.insert(header::COOKIE, "sid=1".parse().unwrap());
        map.insert(header::ACCEPT, "text/html".parse().unwrap());
        // Same-origin hop: untouched.
        let mut same = map.clone();
        strip_sensitive_headers_on_redirect(&mut same, false);
        assert!(same.contains_key(header::AUTHORIZATION));
        assert!(same.contains_key(header::COOKIE));
        // Cross-origin hop: both stripped, the rest kept.
        let mut crossed = map.clone();
        strip_sensitive_headers_on_redirect(&mut crossed, true);
        assert!(!crossed.contains_key(header::AUTHORIZATION));
        assert!(!crossed.contains_key(header::COOKIE));
        assert!(crossed.contains_key(header::ACCEPT));
    }

    #[test]
    fn redirect_body_drop_strips_body_headers() {
        let mut map = HeaderMap::new();
        map.insert(header::CONTENT_LENGTH, "12".parse().unwrap());
        map.insert(header::CONTENT_TYPE, "text/plain".parse().unwrap());
        map.insert(header::CONTENT_ENCODING, "gzip".parse().unwrap());
        map.insert(header::ACCEPT, "*/*".parse().unwrap());
        strip_request_body_headers(&mut map);
        assert!(!map.contains_key(header::CONTENT_LENGTH));
        assert!(!map.contains_key(header::CONTENT_TYPE));
        assert!(!map.contains_key(header::CONTENT_ENCODING));
        assert!(map.contains_key(header::ACCEPT));
    }

    #[test]
    fn redirect_url_credentials_detection() {
        assert!(redirect_url_has_credentials(
            &"https://user:pass@example.com/x".parse().unwrap()
        ));
        assert!(redirect_url_has_credentials(
            &"https://user@example.com/x".parse().unwrap()
        ));
        assert!(!redirect_url_has_credentials(
            &"https://example.com/x".parse().unwrap()
        ));
        assert!(!redirect_url_has_credentials(
            &"https://example.com/x?next=https://other.example/".parse().unwrap()
        ));
    }

    // --- safelist traps (§c trap 4, 10) ---

    #[test]
    fn dual_content_type_triggers_preflight() {
        // Two `Content-Type` headers are never safelisted (strict
        // MIME parsing, not the forgiving extractor) — even when
        // both values are individually safelisted essences.
        let classified = classify_request_headers(
            "POST",
            &[
                ("Content-Type".to_string(), "text/plain".to_string()),
                ("Content-Type".to_string(), "text/plain".to_string()),
            ],
        )
        .unwrap();
        assert!(classified.needs_preflight);
        assert_eq!(classified.unsafe_headers, vec!["content-type"]);
        // One `text/plain` Content-Type is simple.
        let classified = classify_request_headers(
            "POST",
            &[("Content-Type".to_string(), "text/plain".to_string())],
        )
        .unwrap();
        assert!(!classified.needs_preflight);
    }

    #[test]
    fn content_type_essence_rules() {
        // The three essences are safelisted, parameters included...
        for value in [
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "text/plain; charset=utf-8",
            "multipart/form-data; boundary=----x",
        ] {
            let c = classify_request_headers(
                "POST",
                &[("Content-Type".to_string(), value.to_string())],
            )
            .unwrap();
            assert!(!c.needs_preflight, "{value} must be safelisted");
        }
        // ...anything else (incl. `application/json`) preflights.
        for value in ["application/json", "text/html", "application/xml"] {
            let c = classify_request_headers(
                "POST",
                &[("Content-Type".to_string(), value.to_string())],
            )
            .unwrap();
            assert!(c.needs_preflight, "{value} must preflight");
            assert_eq!(c.unsafe_headers, vec!["content-type"]);
        }
    }

    #[test]
    fn range_single_range_rule() {
        // A single range with a non-null start is safelisted.
        for value in ["bytes=0-99", "bytes=0-", "bytes=1024-2048"] {
            let c = classify_request_headers(
                "GET",
                &[("Range".to_string(), value.to_string())],
            )
            .unwrap();
            assert!(!c.needs_preflight, "{value} must be safelisted");
        }
        // `bytes=-500` (null start), multiple ranges, and a missing
        // `bytes=` prefix are not.
        for value in ["bytes=-500", "bytes=0-99,200-299", "items=0-99"] {
            let c = classify_request_headers(
                "GET",
                &[("Range".to_string(), value.to_string())],
            )
            .unwrap();
            assert!(c.needs_preflight, "{value} must preflight");
        }
    }

    #[test]
    fn safelist_value_size_traps() {
        // One safelisted value over 128 bytes is unsafe...
        let big = "a".repeat(129);
        let c = classify_request_headers(
            "GET",
            &[("Accept".to_string(), big.clone())],
        )
        .unwrap();
        assert!(c.needs_preflight);
        assert_eq!(c.unsafe_headers, vec!["accept"]);
        // ...exactly 128 bytes is still safelisted...
        let edge = "a".repeat(128);
        let c = classify_request_headers("GET", &[("Accept".to_string(), edge)]).unwrap();
        assert!(!c.needs_preflight);
        // ...and safelisted values totalling over 1024 bytes make
        // EVERY safelisted name unsafe, even short ones. Each value
        // here is ≤ 128 bytes on its own (nine × 120 = 1080).
        let many: Vec<(String, String)> = (0..9)
            .map(|i| {
                (
                    if i % 2 == 0 {
                        "Accept".to_string()
                    } else {
                        "Accept-Language".to_string()
                    },
                    "a".repeat(120),
                )
            })
            .collect();
        let c = classify_request_headers("GET", &many).unwrap();
        assert!(c.needs_preflight);
        assert_eq!(
            c.unsafe_headers,
            vec!["accept", "accept-language"],
            "the 1024-byte total trap converts every safelisted name"
        );
        // 1024 total exactly is still simple (eight × 128).
        let exact: Vec<(String, String)> = (0..8)
            .map(|i| {
                (
                    if i % 2 == 0 {
                        "Accept".to_string()
                    } else {
                        "Accept-Language".to_string()
                    },
                    "a".repeat(128),
                )
            })
            .collect();
        let c = classify_request_headers("GET", &exact).unwrap();
        assert!(!c.needs_preflight);
    }

    #[test]
    fn forbidden_headers_and_methods_fail_closed() {
        // A forbidden method is an error, not a preflight.
        assert!(classify_request_headers("CONNECT", &[]).is_err());
        assert!(classify_request_headers("TRACE", &[]).is_err());
        assert!(classify_request_headers("TRACK", &[]).is_err());
        // A non-safelisted, non-forbidden method preflights.
        assert!(classify_request_headers("PUT", &[]).unwrap().needs_preflight);
        // Forbidden names are errors...
        for name in [
            "Origin",
            "Cookie",
            "Referer",
            "Host",
            "Content-Length",
            "Sec-Fetch-Mode",
            "Proxy-Authorization",
            "X-HTTP-Method-Override",
        ] {
            // `X-HTTP-Method-Override` is only forbidden when its
            // value parses to a forbidden method — give it one
            // (`TRACE`; `DELETE` is merely non-safelisted).
            let value = if name == "X-HTTP-Method-Override" {
                "TRACE"
            } else {
                "x"
            };
            assert!(
                classify_request_headers(name, &[(name.to_string(), value.to_string())]).is_err(),
                "{name} must be forbidden"
            );
        }
        // ...but `X-HTTP-Method-Override: GET` is allowed (and
        // unsafe — it is not a safelisted name — so it preflights).
        let c = classify_request_headers(
            "POST",
            &[(
                "X-HTTP-Method-Override".to_string(),
                "GET".to_string(),
            )],
        )
        .unwrap();
        assert!(c.needs_preflight);
        // `Authorization` is in neither safelist: it preflights and
        // is listed as unsafe.
        let c = classify_request_headers(
            "GET",
            &[("Authorization".to_string(), "Bearer x".to_string())],
        )
        .unwrap();
        assert!(c.needs_preflight);
        assert_eq!(c.unsafe_headers, vec!["authorization"]);
    }

    #[test]
    fn preflight_request_shape() {
        let req_origin = origin("https://example.com/");
        let url: Uri = "https://api.example/data".parse().unwrap();
        let request = build_preflight_request(
            &req_origin,
            &url,
            "PUT",
            &["authorization".to_string(), "x-custom".to_string()],
        )
        .unwrap();
        assert_eq!(request.method(), Method::OPTIONS);
        assert_eq!(request.uri(), &url);
        assert_eq!(
            request.headers().get(header::ACCEPT).unwrap(),
            "*/*"
        );
        assert_eq!(
            request
                .headers()
                .get(header::ACCESS_CONTROL_REQUEST_METHOD)
                .unwrap(),
            "PUT"
        );
        // Comma-joined with NO space after the comma.
        assert_eq!(
            request
                .headers()
                .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
                .unwrap(),
            "authorization,x-custom"
        );
    }

    #[test]
    fn origin_header_forcing_rules() {
        let https_origin = origin("https://example.com/");
        let http_url: Uri = "http://api.example/data".parse().unwrap();
        let https_url: Uri = "https://api.example/data".parse().unwrap();
        // https origin → http URL: forced to `null`.
        assert_eq!(
            origin_header_for(&https_origin, &http_url, CredentialsMode::Omit),
            "null"
        );
        // Cross-origin with credentials `same-origin`: `null`.
        assert_eq!(
            origin_header_for(&https_origin, &https_url, CredentialsMode::SameOrigin),
            "null"
        );
        // Cross-origin with credentials `include`: serialized.
        assert_eq!(
            origin_header_for(&https_origin, &https_url, CredentialsMode::Include),
            "https://example.com"
        );
        // Same-origin with credentials `same-origin`: serialized.
        assert_eq!(
            origin_header_for(
                &https_origin,
                &"https://example.com/data".parse().unwrap(),
                CredentialsMode::SameOrigin
            ),
            "https://example.com"
        );
        // An opaque origin always serializes to `null`.
        assert_eq!(
            origin_header_for(&Origin::opaque(), &https_url, CredentialsMode::Include),
            "null"
        );
    }

    // --- preflight validation (§c trap 7) + cache (§c trap 5) ---

    #[test]
    fn preflight_requires_ok_status() {
        let req = origin("https://example.com/");
        // A 403 preflight fails even with the right headers...
        let forbidden = headers(&[
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "PUT"),
        ]);
        assert!(validate_preflight_response(
            StatusCode::FORBIDDEN,
            &forbidden,
            "PUT",
            &[],
            CredentialsMode::Omit,
            &req,
        )
        .is_err());
        // ...while 200–299 pass.
        for status in [200u16, 204, 299] {
            assert!(
                validate_preflight_response(
                    StatusCode::from_u16(status).unwrap(),
                    &forbidden,
                    "PUT",
                    &[],
                    CredentialsMode::Omit,
                    &req,
                )
                .is_ok(),
                "preflight status {status} must pass"
            );
        }
    }

    #[test]
    fn preflight_method_rules() {
        let req = origin("https://example.com/");
        // A safelisted method passes with no `Allow-Methods` at all.
        let bare = headers(&[("access-control-allow-origin", "*")]);
        assert!(validate_preflight_response(
            StatusCode::OK,
            &bare,
            "GET",
            &[],
            CredentialsMode::Omit,
            &req,
        )
        .is_ok());
        // A listed method passes; an unlisted one fails.
        let listed = headers(&[
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "PUT, DELETE"),
        ]);
        assert!(validate_preflight_response(
            StatusCode::OK,
            &listed,
            "PUT",
            &[],
            CredentialsMode::Omit,
            &req,
        )
        .is_ok());
        assert!(validate_preflight_response(
            StatusCode::OK,
            &listed,
            "PATCH",
            &[],
            CredentialsMode::Omit,
            &req,
        )
        .is_err());
        // `*` in `Allow-Methods` covers a method only when
        // credentials mode is not `include`.
        let star = headers(&[
            ("access-control-allow-origin", "*"),
            ("access-control-allow-methods", "*"),
        ]);
        assert!(validate_preflight_response(
            StatusCode::OK,
            &star,
            "PUT",
            &[],
            CredentialsMode::Omit,
            &req,
        )
        .is_ok());
        assert!(validate_preflight_response(
            StatusCode::OK,
            &star,
            "PUT",
            &[],
            CredentialsMode::Include,
            &req,
        )
        .is_err());
    }

    #[test]
    fn preflight_max_age_default_and_cap() {
        // Absent → 5 s default; unparseable → 5 s; 0 → 0.
        assert_eq!(parse_preflight_max_age(&headers(&[])), 5);
        assert_eq!(
            parse_preflight_max_age(&headers(&[("access-control-max-age", "soon")])),
            5
        );
        assert_eq!(
            parse_preflight_max_age(&headers(&[("access-control-max-age", "0")])),
            0
        );
        assert_eq!(
            parse_preflight_max_age(&headers(&[("access-control-max-age", "60")])),
            60
        );
        // Over the cap → clamped to 7200 s.
        assert_eq!(
            parse_preflight_max_age(&headers(&[("access-control-max-age", "86400")])),
            7200
        );
    }

    #[test]
    fn preflight_cache_stores_looks_up_and_clears() {
        let cache = PreflightCache::new();
        let key = PreflightCacheKey {
            partition_key: "https://example.com".to_string(),
            origin: "https://example.com".to_string(),
            url: "https://api.example/data".to_string(),
            credentials_include: false,
            method: "PUT".to_string(),
            headers: vec!["x-custom".to_string()],
        };
        assert!(!cache.lookup(&key));
        cache.store(key.clone(), 60);
        assert!(cache.lookup(&key));
        assert_eq!(cache.len(), 1);
        // A max-age of 0 stores nothing.
        let zero = PreflightCacheKey {
            method: "DELETE".to_string(),
            ..key.clone()
        };
        cache.store(zero.clone(), 0);
        assert!(!cache.lookup(&zero));
        // Failure clears every entry for the request, whatever the
        // method / headers were.
        cache.clear_for_request(
            "https://example.com",
            "https://example.com",
            "https://api.example/data",
            false,
        );
        assert!(!cache.lookup(&key));
        assert_eq!(cache.len(), 0);
    }

    // --- filtering / tainting (§c traps 13, 9) ---

    #[test]
    fn cors_filtering_exposes_only_safelisted_plus_exposed() {
        let map = headers(&[
            ("content-type", "application/json"),
            ("content-length", "42"),
            ("x-custom", "secret"),
            ("x-exposed", "shown"),
            ("set-cookie", "sid=1"),
            ("access-control-expose-headers", "X-Exposed"),
        ]);
        let filtered = filter_cors_response_headers(&map, CredentialsMode::Omit);
        let names: Vec<&str> = filtered.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"content-type"));
        assert!(names.contains(&"content-length"));
        assert!(names.contains(&"x-exposed"));
        assert!(!names.contains(&"x-custom"), "unexposed headers stay hidden");
        assert!(!names.contains(&"set-cookie"), "Set-Cookie never leaks");
        // With credentials omit, `*` exposes everything (except
        // Set-Cookie).
        let star = headers(&[
            ("x-custom", "secret"),
            ("set-cookie", "sid=1"),
            ("access-control-expose-headers", "*"),
        ]);
        let filtered = filter_cors_response_headers(&star, CredentialsMode::Omit);
        assert!(filtered.iter().any(|(n, _)| n == "x-custom"));
        assert!(!filtered.iter().any(|(n, _)| n == "set-cookie"));
        // With credentials include, `*` exposes only a header
        // literally named `*` — i.e. nothing useful.
        let filtered = filter_cors_response_headers(&star, CredentialsMode::Include);
        assert!(!filtered.iter().any(|(n, _)| n == "x-custom"));
    }

    #[test]
    fn tainting_for_modes_and_destinations() {
        let req = origin("https://example.com/");
        let cross: Uri = "https://api.example/data".parse().unwrap();
        let same: Uri = "https://example.com/data".parse().unwrap();
        // CORS cross-origin → cors; same-origin → basic.
        assert_eq!(
            tainting_for(RequestMode::Cors, &req, &cross, Destination::Fetch),
            ResponseTainting::Cors
        );
        assert_eq!(
            tainting_for(RequestMode::Cors, &req, &same, Destination::Fetch),
            ResponseTainting::Basic
        );
        // `no-cors` fetch() cross-origin → opaque...
        assert_eq!(
            tainting_for(RequestMode::NoCors, &req, &cross, Destination::Fetch),
            ResponseTainting::Opaque
        );
        // ...but a `no-cors` classic subresource still delivers
        // its bytes to the parser/renderer.
        assert_eq!(
            tainting_for(RequestMode::NoCors, &req, &cross, Destination::Image),
            ResponseTainting::Basic
        );
        // Navigations are always basic.
        assert_eq!(
            tainting_for(RequestMode::Navigate, &req, &cross, Destination::Document),
            ResponseTainting::Basic
        );
    }

    #[test]
    fn cors_check_skipped_for_304_and_407_only() {
        assert!(cors_check_is_skipped(StatusCode::NOT_MODIFIED));
        assert!(cors_check_is_skipped(StatusCode::PROXY_AUTHENTICATION_REQUIRED));
        assert!(!cors_check_is_skipped(StatusCode::OK));
        assert!(!cors_check_is_skipped(StatusCode::FORBIDDEN));
    }

    // --- the preflight round-trip against a local server ---

    /// Local server answering OPTIONS by path: `/pre-ok` grants the
    /// preflight, `/pre-bad` lists the wrong method.
    async fn serve_preflight(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let mut lines = request.lines();
                let first = lines.next().unwrap_or("");
                let mut parts = first.split_whitespace();
                let method = parts.next().unwrap_or("");
                let path = parts.next().unwrap_or("/");
                let response = if method == "OPTIONS" && path == "/pre-ok" {
                    "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: PUT, DELETE\r\nAccess-Control-Allow-Headers: X-Custom, Authorization\r\nAccess-Control-Max-Age: 60\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                } else if method == "OPTIONS" && path == "/pre-bad" {
                    "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                } else {
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                };
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    async fn preflight_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_preflight(listener));
        format!("http://127.0.0.1:{}/", addr.port())
    }

    #[tokio::test]
    async fn preflight_round_trip_succeeds_and_fails() {
        use crate::client::shared_client;
        let base = preflight_server().await;
        let client = shared_client();
        // The request origin is a different host than the server, so
        // this is a genuine cross-origin preflight.
        let req_origin = origin("http://example.com/");
        let ok_url: Uri = format!("{base}pre-ok").parse().unwrap();
        let max_age = run_preflight(
            client,
            &req_origin,
            &ok_url,
            "PUT",
            &["authorization".to_string(), "x-custom".to_string()],
            CredentialsMode::Omit,
        )
        .await
        .expect("granted preflight must validate");
        assert_eq!(max_age, 60);
        // The wrong method is refused by the preflight.
        let bad_url: Uri = format!("{base}pre-bad").parse().unwrap();
        assert!(run_preflight(
            client,
            &req_origin,
            &bad_url,
            "PUT",
            &[],
            CredentialsMode::Omit,
        )
        .await
        .is_err());
    }
}
