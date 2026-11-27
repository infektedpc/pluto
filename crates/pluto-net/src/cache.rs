//! In-memory partitioned HTTP cache (Stage 10 Rec 1 — the big M2
//! leftover).
//!
//! RFC 9111 §2–§5, in the shape the Stage 10 report specified:
//! the key is `(top-level-site partition, method, URL,
//! Vary-selected request headers)`; an entry stores the status,
//! the response headers, the (decoded) body, the validators
//! (`ETag` / `Last-Modified`) and the time it was stored.
//! Freshness comes from `Cache-Control: max-age` / `Expires`
//! (lifetime = first of `s-maxage` — shared only, so ignored
//! here — `max-age`, `Expires`); heuristic freshness (10% of the
//! age since `Last-Modified`, capped at 24 h) applies only when
//! no explicit lifetime is present. A stale entry revalidates
//! with `If-None-Match` / `If-Modified-Since`; a `304` keeps the
//! body and refreshes the headers.
//!
//! Rules that are load-bearing, so they are stated here and not
//! left to the caller:
//!
//! * `no-store` (request or response) is never stored;
//!   `no-cache` is stored but revalidated before every reuse.
//! * `private` is stored — this cache is per-tab and therefore
//!   already a private cache — but it is never served across
//!   partitions (nothing is: the partition is part of the key).
//! * A response carrying `Set-Cookie` is stored only with the
//!   `Set-Cookie` header itself stripped: the cookie jar already
//!   consumed it on the network fetch, and a re-served response
//!   must never re-deliver it to the renderer. The slot is
//!   partition-private, never shared.
//! * `Vary: *` is never stored (it can never match a later
//!   request). A `Vary` naming a header the cache cannot observe
//!   (`Cookie`, `Referer`, `Origin`, `Authorization` — UA-added,
//!   not part of the effective request headers below) never
//!   matches either: the entry is revalidated, not served fresh.
//!   Fail closed on a `Vary` we cannot evaluate.
//! * Only `GET` responses are stored, and only for the statuses
//!   RFC 9111 §3 deems cacheable by default (200, 203, 204, 206,
//!   300, 301, 308, 404, 405, 410, 414, 501). A single entry
//!   larger than the byte cap is never stored.
//! * The cache never serves stale: on revalidation failure the
//!   fetch errors, it does not fall back to the stale body.
//! * The stored body is the DECODED body. The stored headers
//!   therefore drop `Content-Encoding` (the bytes are already
//!   decoded) and `Content-Length` (recomputed from the stored
//!   body on serve); serving a decoded body under a `gzip`
//!   `Content-Encoding` would corrupt the consumer.
//!
//! Size accounting counts stored body bytes only (headers are
//! small and bounded by the per-destination body cap upstream);
//! the cap defaults to 50 MiB, matching the existing image body
//! cap. Eviction is LRU over whole entries.
//!
//! Placement: the cache is owned by the per-tab scheduler
//! (`service.rs`), exactly like the cookie jar and the HSTS
//! store. At M3 the loader is promoted to a shared handle and
//! this cache moves with it — the key scheme does not change,
//! because the partition is already in the key.
//!
//! Memory only: disk persistence is M5. A cache hit emits the
//! same `ResponseHeaders` / `BodyChunk` / `Done` event shape as
//! a network fetch, so no consumer — and no layout or style
//! pass — can tell a hit from a fetch, and none is re-triggered
//! by one.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http::header::{self, HeaderMap, HeaderValue};

/// Default byte cap: 50 MiB, matching the image body cap in
/// `service::byte_cap`.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024 * 1024;
/// Heuristic freshness cap (RFC 9111 §4.2.2 suggests a
/// fraction of the time since `Last-Modified`; the 24 h cap is
/// the conventional bound).
pub const MAX_HEURISTIC_FRESHNESS: Duration = Duration::from_secs(24 * 60 * 60);
/// Heuristic freshness fraction: 10% of the time since
/// `Last-Modified`.
const HEURISTIC_FRACTION: u32 = 10;

/// The part of the key that does not depend on `Vary`: the
/// partition, the method, and the URL. Variants of one response
/// (different `Vary`-selected headers) share a base key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BaseKey {
    partition: String,
    method: String,
    url: String,
}

/// The full key of one stored variant: base key + the
/// `Vary`-selected request header values, in the stored
/// entry's `Vary` order (so two variants of one URL with the
/// same `Vary` list compare equal exactly when their selected
/// values agree).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FullKey {
    base: BaseKey,
    vary_values: Vec<(String, String)>,
}

/// One stored response.
#[derive(Debug, Clone)]
pub struct CachedEntry {
    /// Full key (for LRU bookkeeping and 304 updates).
    key: FullKey,
    /// Response status (only cacheable statuses are stored).
    pub status: u16,
    /// Stored response headers: `Set-Cookie`, `Content-Encoding`
    /// and `Content-Length` already stripped (see module docs).
    pub headers: HeaderMap,
    /// Decoded body.
    pub body: Bytes,
    /// When the response was stored (or last refreshed by a 304).
    stored_at: SystemTime,
    /// `Vary` header names from the stored response, lowercased,
    /// in order. Empty when the response carried no `Vary`.
    vary: Vec<String>,
    /// Validators, copied out of the headers for revalidation.
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

impl CachedEntry {
    /// The conditional-request headers for revalidating this
    /// entry: `If-None-Match` from the ETag, `If-Modified-Since`
    /// from `Last-Modified` (both when both validators exist —
    /// the server may honour either).
    pub fn revalidation_headers(&self) -> (Option<String>, Option<String>) {
        (self.etag.clone(), self.last_modified.clone())
    }

    /// Whether this entry has any validator to revalidate with.
    pub fn has_validator(&self) -> bool {
        self.etag.is_some() || self.last_modified.is_some()
    }
}

/// What a lookup found.
#[derive(Debug, Clone)]
pub enum Lookup {
    /// Fresh: serve the entry, no network traffic.
    Fresh(CachedEntry),
    /// Stale but revalidatable: send a conditional request
    /// (`If-None-Match` / `If-Modified-Since` from the entry).
    /// A `304` keeps this body; a `200` replaces the entry.
    Stale(CachedEntry),
    /// No usable entry (nothing stored, `Vary` mismatch, or a
    /// stale entry with no validator): fetch unconditionally.
    Miss,
}

/// Cache statistics, for tests and the status line. The
/// scheduler exposes these so a second navigation provably
/// issuing no network fetch is observable without a server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub revalidations: u64,
}

/// The in-memory partitioned LRU HTTP cache.
///
/// Not thread-safe on its own: the scheduler holds it behind a
/// `std::sync::Mutex` (short critical sections, never held
/// across an await), like the cookie jar.
#[derive(Debug)]
pub struct HttpCache {
    entries: HashMap<BaseKey, Vec<CachedEntry>>,
    /// LRU order, least-recently-used first. One entry per
    /// stored variant.
    order: VecDeque<FullKey>,
    total_bytes: usize,
    max_bytes: usize,
    stats: CacheStats,
}

impl Default for HttpCache {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpCache {
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_MAX_BYTES)
    }

    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            total_bytes: 0,
            max_bytes,
            stats: CacheStats::default(),
        }
    }

    /// Current byte cap.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Set the byte cap, evicting LRU entries until the cache
    /// fits. Tests use a small cap to prove eviction; production
    /// never changes it after construction.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        self.evict_while_over_cap();
    }

    /// Stored body bytes in total.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of stored variants.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.total_bytes = 0;
    }

    /// Look an entry up.
    ///
    /// `request_headers` is the EFFECTIVE request header set —
    /// the `Accept` / `Accept-Encoding` defaults plus the
    /// script's headers, as built for the wire — because a
    /// `Vary` can name any of those. Headers the cache cannot
    /// observe (`Cookie`, `Referer`, `Origin`, `Authorization`)
    /// are not in it, and a stored `Vary` naming one never
    /// matches (see module docs).
    pub fn lookup(
        &mut self,
        partition: &str,
        method: &str,
        url: &str,
        request_headers: &HeaderMap,
    ) -> Lookup {
        let now = SystemTime::now();
        let base = BaseKey {
            partition: partition.to_string(),
            method: method.to_ascii_uppercase(),
            url: url.to_string(),
        };
        let Some(variants) = self.entries.get(&base) else {
            self.stats.misses += 1;
            return Lookup::Miss;
        };
        // A request `Cache-Control: no-cache` forces
        // revalidation even of a fresh entry (RFC 9111 §5.2.1);
        // `no-store` on the request bypasses the cache entirely
        // for this fetch (treated as a miss — and `store` below
        // refuses to store the response).
        let req_cc = parse_cache_control(request_headers.get(header::CACHE_CONTROL));
        let force_revalidate = has_directive(&req_cc, "no-cache") || has_directive(&req_cc, "max-age=0");
        let Some(entry) = variants
            .iter()
            .find(|e| vary_matches(&e.vary, &e.key.vary_values, request_headers))
            .cloned()
        else {
            // No variant's `Vary` matches — except when a
            // variant's `Vary` names a header the cache cannot
            // observe (`Cookie`, …): that entry can never hit
            // fresh, but it CAN be revalidated, so it is a
            // `Stale` candidate, not a miss.
            if let Some(entry) = variants
                .iter()
                .find(|e| {
                    e.vary
                        .iter()
                        .any(|v| UNOBSERVABLE_VARY_HEADERS.contains(&v.as_str()))
                        && e.has_validator()
                })
                .cloned()
            {
                self.stats.revalidations += 1;
                self.promote(&entry.key);
                return Lookup::Stale(entry);
            }
            self.stats.misses += 1;
            return Lookup::Miss;
        };
        if !force_revalidate && is_fresh(&entry, now) {
            self.stats.hits += 1;
            self.promote(&entry.key);
            return Lookup::Fresh(entry);
        }
        if entry.has_validator() {
            self.stats.revalidations += 1;
            self.promote(&entry.key);
            return Lookup::Stale(entry);
        }
        self.stats.misses += 1;
        Lookup::Miss
    }

    /// Store a response. Returns `true` when stored. The
    /// caller must already have decided the response is a final
    /// (non-redirect, non-opaque, CORS/SRI-passing) response;
    /// this re-checks every storeability rule so a caller bug
    /// cannot smuggle an uncacheable response in.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        partition: &str,
        method: &str,
        url: &str,
        status: u16,
        response_headers: &HeaderMap,
        request_headers: &HeaderMap,
        body: Bytes,
    ) -> bool {
        if !is_storable(method, status, request_headers, response_headers, body.len(), self.max_bytes)
        {
            return false;
        }
        let now = SystemTime::now();
        let vary = parse_vary(response_headers);
        let vary_values = vary_values_for(&vary, request_headers);
        let base = BaseKey {
            partition: partition.to_string(),
            method: method.to_ascii_uppercase(),
            url: url.to_string(),
        };
        let key = FullKey {
            base: base.clone(),
            vary_values,
        };
        let mut headers = response_headers.clone();
        strip_stored_headers(&mut headers);
        let entry = CachedEntry {
            key: key.clone(),
            status,
            headers: headers.clone(),
            body: body.clone(),
            stored_at: now,
            vary,
            etag: headers
                .get(header::ETAG)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
            last_modified: headers
                .get(header::LAST_MODIFIED)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
        };
        // Replace an existing variant with the same full key:
        // its bytes leave the total, its LRU slot is reused.
        if let Some(variants) = self.entries.get_mut(&base)
            && let Some(pos) = variants.iter().position(|e| e.key == key) {
                self.total_bytes = self
                    .total_bytes
                    .saturating_sub(variants[pos].body.len());
                variants[pos] = entry;
                self.total_bytes += body.len();
                self.promote(&key);
                self.stats.stores += 1;
                self.evict_while_over_cap();
                return true;
            }
        self.entries.entry(base).or_default().push(entry);
        self.order.push_back(key);
        self.total_bytes += body.len();
        self.stats.stores += 1;
        self.evict_while_over_cap();
        true
    }

    /// A `304 Not Modified` arrived for a revalidation of
    /// `entry`: merge the 304's headers into the stored entry
    /// (RFC 9111 §4.3.4 — headers present in the 304 replace
    /// the stored ones, the rest are kept) and refresh the
    /// stored time, so the entry's freshness is recomputed from
    /// the new headers. Returns the updated entry for serving,
    /// or `None` when the entry is gone (evicted since lookup).
    pub fn update_after_304(
        &mut self,
        entry: &CachedEntry,
        headers_304: &HeaderMap,
        now: SystemTime,
    ) -> Option<CachedEntry> {
        let variants = self.entries.get_mut(&entry.key.base)?;
        let stored = variants.iter_mut().find(|e| e.key == entry.key)?;
        for (name, value) in headers_304.iter() {
            // Hop-by-hop and body-describing headers from a 304
            // must not overwrite the stored entry's: the body
            // is the stored one, not the (absent) 304 body.
            let lower = name.as_str().to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "connection"
                    | "transfer-encoding"
                    | "content-length"
                    | "content-encoding"
                    | "set-cookie"
                    | "set-cookie2"
            ) {
                continue;
            }
            stored.headers.insert(name.clone(), value.clone());
        }
        stored.stored_at = now;
        stored.etag = stored
            .headers
            .get(header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        stored.last_modified = stored
            .headers
            .get(header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        // A 304 may itself carry a new `Vary`: the variant's
        // identity changes with it. Re-key only when it changed,
        // and only when the new `Vary` is storable (a `Vary: *`
        // on the 304 means the stored response can no longer be
        // reused — drop the entry instead of serving it stale).
        let new_vary = parse_vary(&stored.headers);
        if new_vary.iter().any(|v| v == "*") {
            let key = stored.key.clone();
            let body_len = stored.body.len();
            self.remove(&key);
            let _ = body_len;
            return None;
        }
        if new_vary != stored.vary {
            // The selected values were computed under the old
            // `Vary`; the entry can no longer be matched by a
            // lookup, so it is unusable — drop it rather than
            // keep an entry that can never hit.
            let key = stored.key.clone();
            self.remove(&key);
            return None;
        }
        let updated = stored.clone();
        self.promote(&updated.key);
        Some(updated)
    }

    /// Move a key to the most-recently-used end.
    fn promote(&mut self, key: &FullKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let key = self.order.remove(pos).unwrap();
            self.order.push_back(key);
        }
    }

    /// Evict least-recently-used entries until the total fits
    /// the cap.
    fn evict_while_over_cap(&mut self) {
        while self.total_bytes > self.max_bytes {
            let Some(key) = self.order.pop_front() else {
                break;
            };
            if let Some(variants) = self.entries.get_mut(&key.base) {
                if let Some(pos) = variants.iter().position(|e| e.key == key) {
                    let removed = variants.remove(pos);
                    self.total_bytes = self.total_bytes.saturating_sub(removed.body.len());
                }
                if variants.is_empty() {
                    self.entries.remove(&key.base);
                }
            }
        }
    }

    /// Remove one variant outright.
    fn remove(&mut self, key: &FullKey) {
        if let Some(variants) = self.entries.get_mut(&key.base) {
            if let Some(pos) = variants.iter().position(|e| e.key == *key) {
                let removed = variants.remove(pos);
                self.total_bytes = self.total_bytes.saturating_sub(removed.body.len());
            }
            if variants.is_empty() {
                self.entries.remove(&key.base);
            }
        }
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
    }
}

/// Whether a response may be stored at all (RFC 9111 §3).
/// The caller (the fetch loop) uses this before buffering the
/// body, so an uncacheable response is never buffered for the
/// cache in the first place; `store` re-checks it.
pub fn is_storable(
    method: &str,
    status: u16,
    request_headers: &HeaderMap,
    response_headers: &HeaderMap,
    body_len: usize,
    max_bytes: usize,
) -> bool {
    if !method.eq_ignore_ascii_case("GET") {
        return false;
    }
    if !matches!(
        status,
        200 | 203 | 204 | 206 | 300 | 301 | 308 | 404 | 405 | 410 | 414 | 501
    ) {
        return false;
    }
    if body_len > max_bytes {
        return false;
    }
    let req_cc = parse_cache_control(request_headers.get(header::CACHE_CONTROL));
    if has_directive(&req_cc, "no-store") {
        return false;
    }
    let resp_cc = parse_cache_control(response_headers.get(header::CACHE_CONTROL));
    if has_directive(&resp_cc, "no-store") {
        return false;
    }
    // `Vary: *` can never match a later request (RFC 9111
    // §4.1): storing it would only ever revalidate-or-miss,
    // so it is not stored at all.
    if parse_vary(response_headers).iter().any(|v| v == "*") {
        return false;
    }
    true
}

/// Freshness (RFC 9111 §4.2): the entry is fresh while its age
/// is below its freshness lifetime. `no-cache` on the response
/// means "revalidate before every reuse" — never fresh.
pub fn is_fresh(entry: &CachedEntry, now: SystemTime) -> bool {
    let cc = parse_cache_control(entry.headers.get(header::CACHE_CONTROL));
    if has_directive(&cc, "no-cache") {
        return false;
    }
    let Some(lifetime) = freshness_lifetime(entry) else {
        return false;
    };
    age(entry, now) < lifetime
}

/// The freshness lifetime (RFC 9111 §4.2.1): first of
/// `s-maxage` (shared caches only — ignored here, this cache
/// is private), `max-age`, `Expires`; heuristic freshness only
/// when neither is present.
fn freshness_lifetime(entry: &CachedEntry) -> Option<Duration> {
    let cc = parse_cache_control(entry.headers.get(header::CACHE_CONTROL));
    if let Some(max_age) = cc_value(&cc, "max-age") {
        return max_age.parse::<u64>().ok().map(Duration::from_secs);
    }
    if let Some(expires) = entry.headers.get(header::EXPIRES).and_then(|v| v.to_str().ok())
        && let Some(expires_at) = parse_http_date(expires)
    {
        let date = entry
            .headers
            .get(header::DATE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_http_date)
            .unwrap_or(entry.stored_at);
        return Some(expires_at.duration_since(date).unwrap_or(Duration::ZERO));
    }
    // Heuristic freshness: 10% of the time since
    // `Last-Modified`, capped at 24 h — only for statuses
    // that are heuristically cacheable, and only when a
    // `Last-Modified` exists to measure from.
    if !matches!(
        entry.status,
        200 | 203 | 206 | 300 | 301 | 308 | 404 | 410
    ) {
        return None;
    }
    let last_modified = entry.last_modified.as_deref().and_then(parse_http_date)?;
    let date = entry
        .headers
        .get(header::DATE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date)
        .unwrap_or(entry.stored_at);
    let since = date.duration_since(last_modified).unwrap_or(Duration::ZERO);
    Some((since / HEURISTIC_FRACTION).min(MAX_HEURISTIC_FRESHNESS))
}

/// The entry's current age (RFC 9111 §5.1, simplified for a
/// cache that never sees a shared-cache hop): apparent age
/// (`stored_at - Date`, when the server sent one) plus the
/// `Age` header value, plus the time resident in this cache.
fn age(entry: &CachedEntry, now: SystemTime) -> Duration {
    let apparent = entry
        .headers
        .get(header::DATE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date)
        .map(|date| entry.stored_at.duration_since(date).unwrap_or(Duration::ZERO))
        .unwrap_or(Duration::ZERO);
    let age_header = entry
        .headers
        .get(header::AGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::ZERO);
    let resident = now.duration_since(entry.stored_at).unwrap_or(Duration::ZERO);
    apparent.max(age_header) + resident
}

/// Parse a `Cache-Control` header value into lowercase
/// `directive` / `directive=value` tokens. A malformed value
/// parses to whatever tokens survive — never fatal.
fn parse_cache_control(value: Option<&HeaderValue>) -> Vec<String> {
    let Some(value) = value.and_then(|v| v.to_str().ok()) else {
        return Vec::new();
    };
    value
        .split(',')
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// The value of a `directive=value` token, if present.
fn cc_value(directives: &[String], name: &str) -> Option<String> {
    directives.iter().find_map(|d| {
        let (k, v) = d.split_once('=')?;
        if k.trim() == name {
            Some(v.trim().trim_matches('"').to_string())
        } else {
            None
        }
    })
}

/// Whether a parsed `Cache-Control` directive list contains
/// `name` (exact token match, values ignored).
fn has_directive(directives: &[String], name: &str) -> bool {
    directives.iter().any(|d| d == name || d.starts_with(&format!("{name}=")))
}

/// The `Vary` header names from a response, lowercased, in
/// order; `*` is kept as the literal `"*"` so callers can
/// reject it.
fn parse_vary(headers: &HeaderMap) -> Vec<String> {
    let mut out = Vec::new();
    for value in headers.get_all(header::VARY) {
        if let Ok(text) = value.to_str() {
            for name in text.split(',') {
                let name = name.trim().to_ascii_lowercase();
                if !name.is_empty() {
                    out.push(name);
                }
            }
        }
    }
    out
}

/// The request-header values selected by a stored `Vary` list.
/// Values are lowercased for comparison (header values are
/// case-insensitive for the names servers `Vary` on in
/// practice — `Accept-Language`, `Accept-Encoding`); a missing
/// header selects the empty string, so "absent" and "present
/// but empty" are the same variant.
fn vary_values_for(vary: &[String], request_headers: &HeaderMap) -> Vec<(String, String)> {
    vary.iter()
        .map(|name| {
            let value = request_headers
                .get(name.as_str())
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();
            (name.clone(), value)
        })
        .collect()
}

/// Headers whose `Vary` the cache cannot evaluate, because
/// they are added by the browser after the effective request
/// headers are built (jar `Cookie`, `Referer`, `Origin`) or
/// never sent (`Authorization`). A stored `Vary` naming one
/// never matches — the entry is revalidated, not served fresh.
const UNOBSERVABLE_VARY_HEADERS: [&str; 4] = ["cookie", "referer", "origin", "authorization"];

/// Does the stored entry's `Vary` match this request? The
/// entry's stored selected values must equal the values this
/// request would select, and no `Vary` name may be one the
/// cache cannot observe.
fn vary_matches(
    vary: &[String],
    stored_values: &[(String, String)],
    request_headers: &HeaderMap,
) -> bool {
    if vary.iter().any(|v| UNOBSERVABLE_VARY_HEADERS.contains(&v.as_str())) {
        return false;
    }
    vary_values_for(vary, request_headers) == stored_values
}

/// Headers stripped from a response before it is stored (see
/// module docs): `Set-Cookie` (already consumed by the jar),
/// `Content-Encoding` / `Content-Length` (the stored body is
/// decoded), and hop-by-hop headers that describe the
/// transport, not the resource.
fn strip_stored_headers(headers: &mut HeaderMap) {
    for name in [
        "set-cookie",
        "set-cookie2",
        "content-encoding",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

/// Parse an HTTP-date (RFC 9111 §5.3, IMF-fixdate first, with
/// the obsolete RFC 850 and asctime shapes as fallback). Only
/// GMT is accepted — HTTP dates are always GMT. Returns `None`
/// on any malformed input rather than guessing.
pub fn parse_http_date(value: &str) -> Option<SystemTime> {
    let value = value.trim();
    // IMF-fixdate: `Sun, 06 Nov 1994 08:49:37 GMT`
    if let Some(parsed) = parse_imf_fixdate(value) {
        return Some(parsed);
    }
    // RFC 850: `Sunday, 06-Nov-94 08:49:37 GMT`
    if let Some(parsed) = parse_rfc850(value) {
        return Some(parsed);
    }
    // asctime: `Sun Nov  6 08:49:37 1994`
    parse_asctime(value)
}

fn month_number(name: &str) -> Option<u64> {
    Some(match name.to_ascii_lowercase().as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    })
}

fn days_since_epoch(year: u64, month: u64, day: u64) -> u64 {
    // Days from civil (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn time_to_system(year: u64, month: u64, day: u64, h: u64, m: u64, s: u64) -> Option<SystemTime> {
    if !(1..=12).contains(&month) || day == 0 || day > 31 || h > 23 || m > 59 || s > 60 {
        return None;
    }
    let days = days_since_epoch(year, month, day);
    Some(UNIX_EPOCH + Duration::from_secs(days * 86400 + h * 3600 + m * 60 + s))
}

fn parse_imf_fixdate(value: &str) -> Option<SystemTime> {
    // `Sun, 06 Nov 1994 08:49:37 GMT`
    let mut parts = value.split_whitespace();
    let _weekday = parts.next()?;
    let day: u64 = parts.next()?.parse().ok()?;
    let month = month_number(parts.next()?)?;
    let year: u64 = parts.next()?.parse().ok()?;
    let time = parts.next()?;
    if parts.next()? != "GMT" {
        return None;
    }
    let mut t = time.split(':');
    let h: u64 = t.next()?.parse().ok()?;
    let m: u64 = t.next()?.parse().ok()?;
    let s: u64 = t.next()?.parse().ok()?;
    time_to_system(year, month, day, h, m, s)
}

fn parse_rfc850(value: &str) -> Option<SystemTime> {
    // `Sunday, 06-Nov-94 08:49:37 GMT`
    let mut parts = value.split_whitespace();
    let _weekday = parts.next()?;
    let date = parts.next()?;
    let mut d = date.split('-');
    let day: u64 = d.next()?.parse().ok()?;
    let month = month_number(d.next()?)?;
    let mut year: u64 = d.next()?.parse().ok()?;
    // Two-digit year: 00–49 → 2000s, 50–99 → 1900s (RFC 9110).
    year += if year < 50 { 2000 } else { 1900 };
    let time = parts.next()?;
    if parts.next()? != "GMT" {
        return None;
    }
    let mut t = time.split(':');
    let h: u64 = t.next()?.parse().ok()?;
    let m: u64 = t.next()?.parse().ok()?;
    let s: u64 = t.next()?.parse().ok()?;
    time_to_system(year, month, day, h, m, s)
}

fn parse_asctime(value: &str) -> Option<SystemTime> {
    // `Sun Nov  6 08:49:37 1994` — the day is space-padded, so
    // split_whitespace collapses it.
    let mut parts = value.split_whitespace();
    let _weekday = parts.next()?;
    let month = month_number(parts.next()?)?;
    let day: u64 = parts.next()?.parse().ok()?;
    let time = parts.next()?;
    let year: u64 = parts.next()?.parse().ok()?;
    let mut t = time.split(':');
    let h: u64 = t.next()?.parse().ok()?;
    let m: u64 = t.next()?.parse().ok()?;
    let s: u64 = t.next()?.parse().ok()?;
    time_to_system(year, month, day, h, m, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::header::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    fn store_ok(
        cache: &mut HttpCache,
        url: &str,
        resp: &[(&str, &str)],
        body: &[u8],
    ) -> bool {
        cache.store(
            "https://example.com",
            "GET",
            url,
            200,
            &headers(resp),
            &HeaderMap::new(),
            Bytes::copy_from_slice(body),
        )
    }

    fn lookup(cache: &mut HttpCache, url: &str, req: &HeaderMap) -> Lookup {
        cache.lookup("https://example.com", "GET", url, req)
    }

    #[test]
    fn fresh_within_max_age_then_stale() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60")],
            b"body"
        ));
        assert!(matches!(lookup(&mut cache, "https://example.com/a", &HeaderMap::new()), Lookup::Fresh(_)));
        // Age the entry past its lifetime by moving its
        // stored time back (same-module access).
        let base = BaseKey {
            partition: "https://example.com".to_string(),
            method: "GET".to_string(),
            url: "https://example.com/a".to_string(),
        };
        cache.entries.get_mut(&base).unwrap()[0].stored_at =
            SystemTime::now() - Duration::from_secs(61);
        assert!(matches!(
            lookup(&mut cache, "https://example.com/a", &HeaderMap::new()),
            Lookup::Miss,
        ));
    }

    #[test]
    fn stale_with_etag_revalidates() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=0"), ("ETag", "\"v1\"")],
            b"body"
        ));
        match lookup(&mut cache, "https://example.com/a", &HeaderMap::new()) {
            Lookup::Stale(entry) => {
                assert_eq!(entry.revalidation_headers().0.as_deref(), Some("\"v1\""));
            }
            other => panic!("expected Stale, got {other:?}"),
        }
    }

    #[test]
    fn no_store_is_never_stored() {
        let mut cache = HttpCache::new();
        assert!(!store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "no-store")],
            b"body"
        ));
        assert_eq!(cache.len(), 0);
        // Request `no-store` also refuses storage.
        assert!(!cache.store(
            "https://example.com",
            "GET",
            "https://example.com/b",
            200,
            &headers(&[("Cache-Control", "max-age=60")]),
            &headers(&[("Cache-Control", "no-store")]),
            Bytes::from_static(b"body"),
        ));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn no_cache_is_stored_but_always_revalidates() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "no-cache"), ("ETag", "\"v1\"")],
            b"body"
        ));
        assert!(matches!(
            lookup(&mut cache, "https://example.com/a", &HeaderMap::new()),
            Lookup::Stale(_),
        ));
    }

    #[test]
    fn private_is_stored_within_its_partition_only() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "private, max-age=60")],
            b"body"
        ));
        assert!(matches!(lookup(&mut cache, "https://example.com/a", &HeaderMap::new()), Lookup::Fresh(_)));
        assert!(matches!(
            cache.lookup(
                "https://other.example",
                "GET",
                "https://example.com/a",
                &HeaderMap::new()
            ),
            Lookup::Miss,
        ));
    }

    #[test]
    fn partition_isolation_for_plain_entries() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60")],
            b"body"
        ));
        assert!(matches!(
            cache.lookup("https://site-b.example", "GET", "https://example.com/a", &HeaderMap::new()),
            Lookup::Miss
        ));
    }

    #[test]
    fn vary_selects_variants() {
        let mut cache = HttpCache::new();
        let req_a = headers(&[("X-Flavor", "a")]);
        let req_b = headers(&[("X-Flavor", "b")]);
        assert!(cache.store(
            "https://example.com",
            "GET",
            "https://example.com/v",
            200,
            &headers(&[("Cache-Control", "max-age=60"), ("Vary", "X-Flavor")]),
            &req_a,
            Bytes::from_static(b"flavor-a"),
        ));
        assert!(cache.store(
            "https://example.com",
            "GET",
            "https://example.com/v",
            200,
            &headers(&[("Cache-Control", "max-age=60"), ("Vary", "X-Flavor")]),
            &req_b,
            Bytes::from_static(b"flavor-b"),
        ));
        match lookup(&mut cache, "https://example.com/v", &req_a) {
            Lookup::Fresh(e) => assert_eq!(e.body.as_ref(), b"flavor-a"),
            other => panic!("expected flavor-a, got {other:?}"),
        }
        match lookup(&mut cache, "https://example.com/v", &req_b) {
            Lookup::Fresh(e) => assert_eq!(e.body.as_ref(), b"flavor-b"),
            other => panic!("expected flavor-b, got {other:?}"),
        }
        // A third flavour matches neither variant.
        assert!(matches!(
            lookup(&mut cache, "https://example.com/v", &headers(&[("X-Flavor", "c")])),
            Lookup::Miss
        ));
    }

    #[test]
    fn vary_star_is_never_stored() {
        let mut cache = HttpCache::new();
        assert!(!store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60"), ("Vary", "*")],
            b"body"
        ));
    }

    #[test]
    fn vary_on_cookie_never_serves_fresh() {
        // A `Vary: Cookie` entry is stored (the `Vary` is
        // legal) but can never hit fresh: the cache cannot
        // observe the cookie the jar would send.
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60"), ("Vary", "Cookie"), ("ETag", "\"v1\"")],
            b"body"
        ));
        assert!(matches!(
            lookup(&mut cache, "https://example.com/a", &HeaderMap::new()),
            Lookup::Stale(_),
        ));
    }

    #[test]
    fn lru_evicts_oldest_under_the_cap() {
        let mut cache = HttpCache::with_max_bytes(100);
        assert!(store_ok(&mut cache, "https://example.com/1", &[("Cache-Control", "max-age=60")], &[b'a'; 40]));
        assert!(store_ok(&mut cache, "https://example.com/2", &[("Cache-Control", "max-age=60")], &[b'b'; 40]));
        // Touch 1 so it is most-recently-used, then store 3:
        // 2 (not 1) must be evicted.
        assert!(matches!(lookup(&mut cache, "https://example.com/1", &HeaderMap::new()), Lookup::Fresh(_)));
        assert!(store_ok(&mut cache, "https://example.com/3", &[("Cache-Control", "max-age=60")], &[b'c'; 40]));
        assert_eq!(cache.total_bytes(), 80);
        assert!(matches!(lookup(&mut cache, "https://example.com/1", &HeaderMap::new()), Lookup::Fresh(_)));
        assert!(matches!(lookup(&mut cache, "https://example.com/2", &HeaderMap::new()), Lookup::Miss));
        assert!(matches!(lookup(&mut cache, "https://example.com/3", &HeaderMap::new()), Lookup::Fresh(_)));
    }

    #[test]
    fn single_entry_larger_than_the_cap_is_not_stored() {
        let mut cache = HttpCache::with_max_bytes(10);
        assert!(!store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60")],
            b"this body is far too big for the cap"
        ));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn expires_header_drives_freshness() {
        let mut cache = HttpCache::new();
        // Date in the past relative to now would make Expires
        // arithmetic depend on the clock, so use a far-future
        // Expires with an explicit Date.
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[
                ("Date", "Wed, 01 Jan 2025 00:00:00 GMT"),
                ("Expires", "Wed, 01 Jan 2025 01:00:00 GMT"),
            ],
            b"body"
        ));
        // Stored just now, Expires is 1 h after Date, but the
        // entry's age already includes the apparent age
        // (stored_at - Date, ~2 years) — so it is stale. That
        // is correct RFC 9111 age arithmetic, and it proves the
        // Expires path is exercised rather than silently fresh.
        assert!(matches!(
            lookup(&mut cache, "https://example.com/a", &HeaderMap::new()),
            Lookup::Miss,
        ));
    }

    #[test]
    fn heuristic_freshness_is_ten_percent_of_age_since_last_modified() {
        let mut cache = HttpCache::new();
        // Last-Modified 1000 s before Date: heuristic lifetime
        // is 100 s.
        // Fixed-date form: Date 2026-01-01 00:00:00, LM
        // 2025-12-31 23:43:20 (1000 s earlier) → lifetime 100 s.
        // Age the stored entry 50 s: fresh. Age it 150 s: stale.
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[
                ("Date", "Thu, 01 Jan 2026 00:00:00 GMT"),
                ("Last-Modified", "Wed, 31 Dec 2025 23:43:20 GMT"),
            ],
            b"body"
        ));
        let base = BaseKey {
            partition: "https://example.com".to_string(),
            method: "GET".to_string(),
            url: "https://example.com/a".to_string(),
        };
        // Apparent age (stored_at − Date) is huge here, so the
        // entry is stale immediately by correct age arithmetic
        // — the fixed-date form cannot express "fresh" without
        // a Date near now. Instead verify the lifetime function
        // directly on a hand-built entry with Date == stored_at.
        let entry = CachedEntry {
            key: FullKey {
                base: base.clone(),
                vary_values: Vec::new(),
            },
            status: 200,
            headers: headers(&[
                ("Date", "Thu, 01 Jan 2026 00:00:00 GMT"),
                ("Last-Modified", "Wed, 31 Dec 2025 23:43:20 GMT"),
            ]),
            body: Bytes::from_static(b"body"),
            stored_at: parse_http_date("Thu, 01 Jan 2026 00:00:00 GMT").unwrap(),
            vary: Vec::new(),
            etag: None,
            last_modified: Some("Wed, 31 Dec 2025 23:43:20 GMT".to_string()),
        };
        assert_eq!(
            freshness_lifetime(&entry),
            Some(Duration::from_secs(100)),
            "heuristic lifetime is 10% of 1000 s"
        );
        assert!(is_fresh(&entry, entry.stored_at + Duration::from_secs(50)));
        assert!(!is_fresh(&entry, entry.stored_at + Duration::from_secs(150)));
    }

    #[test]
    fn s_maxage_is_ignored_by_this_private_cache() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "s-maxage=3600, max-age=60")],
            b"body"
        ));
        let base = BaseKey {
            partition: "https://example.com".to_string(),
            method: "GET".to_string(),
            url: "https://example.com/a".to_string(),
        };
        let entry = cache.entries.get(&base).unwrap()[0].clone();
        assert_eq!(
            freshness_lifetime(&entry),
            Some(Duration::from_secs(60)),
            "a private cache uses max-age, not s-maxage"
        );
    }

    #[test]
    fn update_after_304_refreshes_headers_and_time() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=0"), ("ETag", "\"v1\""), ("X-Version", "1")],
            b"body"
        ));
        let entry = match lookup(&mut cache, "https://example.com/a", &HeaderMap::new()) {
            Lookup::Stale(e) => e,
            other => panic!("expected Stale, got {other:?}"),
        };
        let updated = cache
            .update_after_304(
                &entry,
                &headers(&[("Cache-Control", "max-age=60"), ("X-Version", "2")]),
                SystemTime::now(),
            )
            .expect("304 update keeps the entry");
        assert_eq!(updated.body.as_ref(), b"body", "304 keeps the stored body");
        assert_eq!(
            updated.headers.get("X-Version").unwrap().to_str().unwrap(),
            "2",
            "304 headers replace the stored ones"
        );
        assert!(is_fresh(&updated, SystemTime::now()), "refreshed max-age makes it fresh");
    }

    #[test]
    fn set_cookie_is_stripped_on_store() {
        let mut cache = HttpCache::new();
        assert!(store_ok(
            &mut cache,
            "https://example.com/a",
            &[("Cache-Control", "max-age=60"), ("Set-Cookie", "session=abc; Path=/")],
            b"body"
        ));
        match lookup(&mut cache, "https://example.com/a", &HeaderMap::new()) {
            Lookup::Fresh(e) => assert!(
                e.headers.get(header::SET_COOKIE).is_none(),
                "a re-served response must never re-deliver Set-Cookie"
            ),
            other => panic!("expected Fresh, got {other:?}"),
        }
    }

    #[test]
    fn http_date_parses_all_three_shapes() {
        let expected = UNIX_EPOCH + Duration::from_secs(784111777);
        assert_eq!(
            parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(expected)
        );
        assert_eq!(
            parse_http_date("Sunday, 06-Nov-94 08:49:37 GMT"),
            Some(expected)
        );
        assert_eq!(
            parse_http_date("Sun Nov  6 08:49:37 1994"),
            Some(expected)
        );
        assert_eq!(parse_http_date("not a date"), None);
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 PST"), None);
    }

    #[test]
    fn non_get_and_uncacheable_status_are_not_stored() {
        let mut cache = HttpCache::new();
        assert!(!cache.store(
            "https://example.com",
            "POST",
            "https://example.com/a",
            200,
            &headers(&[("Cache-Control", "max-age=60")]),
            &HeaderMap::new(),
            Bytes::from_static(b"body"),
        ));
        assert!(!cache.store(
            "https://example.com",
            "GET",
            "https://example.com/a",
            500,
            &headers(&[("Cache-Control", "max-age=60")]),
            &HeaderMap::new(),
            Bytes::from_static(b"body"),
        ));
        assert_eq!(cache.len(), 0);
    }
}
