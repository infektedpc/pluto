//! The partitioned cookie jar (Stage 9 Recommendation 3, RFC 6265bis + CHIPS).
//!
//! Lives in `pluto-net` (browser side) only. The renderer never sees a
//! `Set-Cookie` header and never decides `SameSite` sending: intake runs at
//! the strip point in `service.rs` ([`CookieJar::store_set_cookie`]), and the
//! `Cookie` header is attached by the browser-side send path only
//! ([`CookieJar::cookie_header`]). `HttpOnly` values are therefore only
//! ever emitted on the wire, never exposed to script or `document.cookie`.
//!
//! # Crate evaluation: `cookie_store` 0.22.1
//!
//! Evaluated 2026-11-24 from the vendored registry source (0.22.1,
//! released 2026-02-16; 0.22.0 was 2025-07-20; CI + clippy + fmt gates in
//! the repo — maintained, not abandoned). Verdict: **usable as the vetted
//! parser/validator, not as the store.**
//!
//! What it gives us (and why we do not hand-roll a `Set-Cookie` parser):
//! `Set-Cookie` parsing via the `cookie` 0.18 crate, `Domain` / `Path` /
//! default-path computation, `Max-Age`-beats-`Expires`, `HttpOnly` and
//! domain-match validation, and `serde` persistence (`serde` feature;
//! `CookieStore::{save_json, load_json}`).
//!
//! What it does **not** do (all closed by this wrapper, verified by
//! reading `cookie_store-0.22.1/src/` firsthand):
//! - `Partitioned` (CHIPS): zero matches for "partition" in the whole
//!   crate. The underlying `cookie` 0.18 crate *parses* the attribute,
//!   but `cookie_store::Cookie` drops it. Partitioning is therefore
//!   implemented here: one store per top-level site, partition key =
//!   top-level site (scheme + registrable domain, see
//!   [`partition_key`]).
//! - `SameSite`: `CookieStore::matches` ignores it entirely — no
//!   Default-is-Lax, no `None`-requires-`Secure`, no send filtering.
//!   Enforced here, both at receipt and at send.
//! - `__Secure-` / `__Host-` prefixes: no checks at all. Enforced here.
//! - Size/count limits: no 4096-byte per-cookie limit, no 50-per-domain
//!   or 3000-total caps, no eviction, no creation/last-access tracking.
//!   Enforced here, with RFC eviction priority (expired first, then
//!   non-secure, then earliest last-access).
//! - Anti-tossing (bis §5.7 step 16): absent. Enforced here.
//! - Secure-over-non-secure receipt rejection: absent (only send-side
//!   filtering). Enforced here.
//! - Public-suffix rejection: only behind the `public_suffix` feature,
//!   which pulls the `publicsuffix` 2.2.3 crate — **not present in the
//!   offline registry cache**, so the default features cannot resolve
//!   offline. The dep is therefore declared with
//!   `default-features = false, features = ["serde"]`, and public-suffix
//!   rejection is done here via the `psl` crate (2.1.260, cached, embeds
//!   the list).
//! - Identity: `CookieStore`'s map is keyed by domain string only, so a
//!   host-only cookie and a `Domain=` cookie with the same domain string
//!   would overwrite each other; the RFC treats them as distinct
//!   cookies. This jar owns its own `Vec` store with the full
//!   (name, domain, host-only, path) identity, and uses
//!   `cookie_store::Cookie` only as the parsed/validated representation.
//!
//! # Partitioning policy
//!
//! **Partition by default** (Firefox model, per the Stage 9 report Rec 3):
//! every cookie is keyed by its top-level site, so the same cookie name
//! set by the same host under two different top-level sites stays
//! separate. A `Partitioned` (CHIPS) cookie is accepted — it requires
//! `Secure` — and behaves the same as any other cookie under this
//! policy; the attribute is recorded so the distinction survives if the
//! policy ever flips to unpartitioned-by-default.
//!
//! # Same-site
//!
//! Schemeful same-site: same scheme **and** same registrable domain
//! (via the `psl` crate; IPs and `localhost` fall back to the bare host).
//! Missing `SameSite` is `Default`, which behaves as `Lax` at send time:
//! a cross-site cookie is sent only on a top-level navigation with a safe
//! method. `Strict` is same-site only. `None` is sent everywhere, but is
//! rejected at receipt unless `Secure` is set.
//!
//! # What this jar does NOT do (follow-ups, not gaps in this file)
//!
//! - Persistence to the profile is the finisher's / `pluto-browser`'s
//!   job; this type only offers [`CookieJar::to_json`] /
//!   [`CookieJar::from_json`] (serde round-trip, persistent cookies only
//!   in practice — session cookies are dropped by
//!   [`CookieJar::clear_session_cookies`] at session end).
//! - The third-party-cookie *policy* decision (partition-by-default is
//!   implemented; a user-facing toggle is a UI concern, not a jar one).

use std::collections::HashMap;

use cookie::SameSite;
use cookie_store::{Cookie, CookieDomain, RawCookie};
use serde::{Deserialize, Serialize};
use url::Url;

/// Per-cookie name + value size cap, RFC 6265bis §5.8.4 (MUST): a cookie
/// whose name + value exceeds this is ignored in full.
pub const MAX_COOKIE_SIZE: usize = 4096;
/// Per-domain cap, RFC 6265bis §5.8.4 (SHOULD): at least 50 cookies per
/// domain. Enforced per (partition, domain).
pub const MAX_COOKIES_PER_DOMAIN: usize = 50;
/// Total cap, RFC 6265bis §5.8.4 (SHOULD): at least 3000 cookies total.
/// Enforced across all partitions.
pub const MAX_COOKIES_TOTAL: usize = 3000;

/// One stored cookie: the vetted `cookie_store::Cookie` representation
/// (name, value, domain, path, expiry, flags) plus the bookkeeping RFC
/// 6265bis §5.3 requires and `cookie_store` does not keep: partition key
/// (held by the enclosing map), creation-time and last-access-time as
/// monotonic sequence numbers (ordering only — wall-clock time is never
/// needed for eviction or send sort), and the denormalised domain/path
/// used for identity and per-domain counting.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCookie {
    cookie: Cookie<'static>,
    /// Whether the `Partitioned` (CHIPS) attribute was present.
    partitioned: bool,
    /// Whether the cookie is host-only (no `Domain` attribute, or a
    /// public-suffix `Domain` attribute identical to the request host —
    /// bis §5.7 treats that as host-only rather than a suffix grant).
    host_only: bool,
    /// Stored domain string (lowercase, no leading dot), for identity
    /// and per-domain counting.
    domain: String,
    /// Stored path string, for identity and send sort.
    path: String,
    /// Creation order (kept across overwrites of the same cookie).
    created_seq: u64,
    /// Last-access order (bumped on every send that includes the cookie).
    last_access_seq: u64,
}

impl StoredCookie {
    /// Full RFC identity: (name, domain, host-only, path).
    fn same_identity(&self, name: &str, domain: &str, host_only: bool, path: &str) -> bool {
        self.cookie.name() == name
            && self.domain == domain
            && self.host_only == host_only
            && self.path == path
    }

    fn is_secure(&self) -> bool {
        self.cookie.secure().unwrap_or(false)
    }
}

/// The partitioned cookie jar.
///
/// Keyed by partition (top-level site, see [`partition_key`]); each
/// partition holds its own cookie list. `Default` is an empty jar.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CookieJar {
    partitions: HashMap<String, Vec<StoredCookie>>,
    /// Monotonic sequence for creation-time / last-access-time ordering.
    seq: u64,
}

impl CookieJar {
    /// An empty jar.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of unexpired cookies in the jar (all partitions).
    pub fn len(&self) -> usize {
        self.partitions
            .values()
            .flat_map(|p| p.iter())
            .filter(|e| !e.cookie.is_expired())
            .count()
    }

    /// Whether the jar holds no unexpired cookies.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every cookie (all partitions). Used on profile clear.
    pub fn clear(&mut self) {
        self.partitions.clear();
    }

    /// Remove session cookies (neither `Max-Age` nor `Expires`): RFC
    /// 6265bis requires them to be removed when the session ends.
    /// Persistent cookies survive.
    pub fn clear_session_cookies(&mut self) {
        for part in self.partitions.values_mut() {
            part.retain(|e| e.cookie.is_persistent());
        }
        self.partitions.retain(|_, p| !p.is_empty());
    }

    /// Serialise the jar (all partitions, including session cookies) to
    /// JSON for profile persistence. `cookie_store::Cookie` round-trips
    /// through its `Set-Cookie` string form under the `serde` feature.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Restore a jar previously produced by [`CookieJar::to_json`]
    /// (Stage 13 Rec 9): expired cookies are pruned on load — a
    /// restored jar must never serve/send stale entries — and the
    /// restored `created_seq` / `last_access_seq` values are
    /// re-based onto this process's sequence counter, so sequence
    /// numbers from a previous process cannot skew eviction
    /// priority (old entries would otherwise look ancient, or new
    /// entries would collide with restored ones).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mut jar: Self = serde_json::from_str(json)?;
        // Prune expired cookies (all partitions), then drop empty
        // partitions.
        for part in jar.partitions.values_mut() {
            part.retain(|e| !e.cookie.is_expired());
        }
        jar.partitions.retain(|_, p| !p.is_empty());
        // Re-base sequence numbers: shift every stored seq by
        // (max_stored_seq + 1) below the live counter is wrong —
        // instead, renumber in creation order so the relative
        // order survives and the live counter continues above it.
        let mut entries: Vec<&mut StoredCookie> = jar
            .partitions
            .values_mut()
            .flat_map(|p| p.iter_mut())
            .collect();
        entries.sort_by_key(|e| e.created_seq);
        let mut seq = 0u64;
        for e in entries {
            seq += 1;
            // Preserve the created-before-accessed ordering within
            // an entry: keep last_access_seq >= created_seq.
            let accessed_after = e.last_access_seq.saturating_sub(e.created_seq);
            e.created_seq = seq;
            e.last_access_seq = seq + accessed_after.min(1_000_000);
        }
        jar.seq = seq;
        Ok(jar)
    }

    /// Non-HttpOnly cookies for `url` in the `top_level_site`
    /// partition, as `name=value` pairs (Stage 13 Rec 10c:
    /// `document.cookie` reads exactly this — HttpOnly cookies are
    /// never exposed to script, but ARE sent on the wire by
    /// [`CookieJar::cookie_header`]).
    pub fn document_cookies(&mut self, url: &str, top_level_site: &str) -> Vec<(String, String)> {
        let Ok(request_url) = Url::parse(url) else {
            return Vec::new();
        };
        let Some(_host) = request_url.host_str() else {
            return Vec::new();
        };
        let key = partition_key(top_level_site);
        let Some(part) = self.partitions.get_mut(&key) else {
            return Vec::new();
        };
        part.retain(|e| !e.cookie.is_expired());
        part.iter()
            .filter(|e| {
                !e.cookie.http_only().unwrap_or(false)
                    && e.cookie.domain.matches(&request_url)
                    && e.cookie.path.matches(&request_url)
            })
            .map(|e| {
                (
                    e.cookie.name().to_string(),
                    e.cookie.value().to_string(),
                )
            })
            .collect()
    }

    /// Intake: store every `Set-Cookie` header received with the response
    /// for `url`, under the `top_level_site` partition.
    ///
    /// Runs independent of any CORS decision — `Set-Cookie` survives CORS
    /// failure (Fetch §3.3.6) — so the caller must invoke this on every
    /// response (redirects included) before stripping `Set-Cookie` from
    /// the headers the renderer sees. Unparseable URLs and headers are
    /// ignored; each header is one cookie (servers must never fold
    /// multiple `Set-Cookie`s into one comma-joined header).
    pub fn store_set_cookie(
        &mut self,
        url: &str,
        top_level_site: &str,
        set_cookie_headers: &[String],
    ) {
        let Ok(request_url) = Url::parse(url) else {
            return;
        };
        if request_url.host_str().is_none() {
            return;
        }
        let secure_context = url_is_secure(&request_url);
        let key = partition_key(top_level_site);
        for header in set_cookie_headers {
            self.store_one(&request_url, &key, header, secure_context);
        }
    }

    /// Send path: the `Cookie` header value for a request to `url` from
    /// the `top_level_site` partition, or `None` when no cookie applies.
    ///
    /// `top_level_site` is the site of the top-level document the
    /// request belongs to — the embedding page for subresources, the
    /// initiating page for a navigation. It serves both as the partition
    /// key and as the site-for-cookies for the `SameSite` comparison, so
    /// a cookie set while a host was embedded on site A is looked up
    /// under A, and rides a later top-level navigation only when that
    /// navigation is also issued from A.
    ///
    /// - `is_secure` is whether the connection is secure (HTTPS, or a
    ///   trusted host such as localhost); `Secure` cookies are withheld
    ///   otherwise.
    /// - `method_is_safe` / `is_top_level_navigation` drive the
    ///   `SameSite=Lax` / `Default` rule: a cross-site `Lax` cookie is
    ///   sent only on a top-level navigation with a safe method
    ///   (`GET`/`HEAD`/`OPTIONS`/`TRACE`).
    /// - `HttpOnly` cookies ARE included: this is the browser-side HTTP
    ///   send path, and `HttpOnly` only excludes non-HTTP APIs
    ///   (`document.cookie`), never the `Cookie` header.
    /// - Sort: longer path first, then earlier creation-time. Every
    ///   cookie in the returned list has its last-access-time bumped.
    pub fn cookie_header(
        &mut self,
        url: &str,
        top_level_site: &str,
        is_secure: bool,
        method_is_safe: bool,
        is_top_level_navigation: bool,
    ) -> Option<String> {
        let Ok(request_url) = Url::parse(url) else {
            return None;
        };
        request_url.host_str()?;
        let same_site = is_same_site(&request_url, top_level_site);
        self.seq += 1;
        let now = self.seq;
        let key = partition_key(top_level_site);
        let partition = self.partitions.get_mut(&key)?;
        partition.retain(|e| !e.cookie.is_expired());
        let mut indices: Vec<usize> = Vec::new();
        for (i, e) in partition.iter().enumerate() {
            if !e.cookie.domain.matches(&request_url) {
                continue;
            }
            if !e.cookie.path.matches(&request_url) {
                continue;
            }
            if e.is_secure() && !is_secure {
                continue;
            }
            if !same_site
                && !same_site_send_allowed(e.cookie.same_site(), method_is_safe, is_top_level_navigation)
            {
                continue;
            }
            indices.push(i);
        }
        if indices.is_empty() {
            return None;
        }
        // Longer path first, then earlier creation-time (RFC 6265bis
        // §5.8.3, SHOULD).
        indices.sort_by(|&a, &b| {
            let (pa, pb) = (&partition[a], &partition[b]);
            pb.path.len()
                .cmp(&pa.path.len())
                .then(pa.created_seq.cmp(&pb.created_seq))
        });
        let mut parts = Vec::with_capacity(indices.len());
        for &i in &indices {
            partition[i].last_access_seq = now;
            parts.push(format!(
                "{}={}",
                partition[i].cookie.name(),
                partition[i].cookie.value()
            ));
        }
        Some(parts.join("; "))
    }

    /// Store one `Set-Cookie` header value for `request_url` in the
    /// partition `key`. Every rejection below silently ignores the cookie
    /// (the RFC's "ignore the cookie entirely and abort these steps").
    fn store_one(
        &mut self,
        request_url: &Url,
        key: &str,
        header: &str,
        secure_context: bool,
    ) {
        // Vetted parse (cookie 0.18, via cookie_store's re-export). Never
        // hand-rolled: attribute splitting, quoting, and date parsing all
        // live in the crate.
        let Ok(raw) = RawCookie::parse(header) else {
            return;
        };
        let name = raw.name();
        let value = raw.value();
        // Empty name AND empty value: reject (bis §5.7 step 1).
        if name.is_empty() && value.is_empty() {
            return;
        }
        // Per-cookie cap (bis §5.8.4, MUST): name + value over 4096
        // bytes ignores the whole cookie.
        if name.len() + value.len() > MAX_COOKIE_SIZE {
            return;
        }
        // Prefix checks are case-insensitive (bis §4.1.3).
        let lower_name = name.to_ascii_lowercase();
        if name.is_empty() {
            // Nameless cookie whose value starts with a prefix: reject.
            let lower_value = value.to_ascii_lowercase();
            if lower_value.starts_with("__secure-") || lower_value.starts_with("__host-") {
                return;
            }
        }
        let secure = raw.secure().unwrap_or(false);
        if lower_name.starts_with("__secure-") && !secure {
            return;
        }
        let host_only_attr = raw.domain().is_none();
        if lower_name.starts_with("__host-")
            && (!secure || !host_only_attr || raw.path() != Some("/"))
        {
            return;
        }
        // A Secure cookie arriving over a non-secure connection is
        // rejected outright (bis §5.7).
        if secure && !secure_context {
            return;
        }
        // SameSite=None requires Secure (bis §5.7). Unrecognised
        // SameSite values parse to None in the cookie crate, i.e.
        // Default — exactly the bis rule.
        if raw.same_site() == Some(SameSite::None) && !secure {
            return;
        }
        // Partitioned (CHIPS) requires Secure.
        let partitioned = raw.partitioned().unwrap_or(false);
        if partitioned && !secure {
            return;
        }
        // Vetted RFC 6265 §5.7 validation: domain-match, default-path,
        // Max-Age-beats-Expires, HttpOnly-on-non-HTTP rejection.
        let Ok(mut parsed) = Cookie::try_from_raw_cookie(&raw, request_url) else {
            return;
        };
        // Public-suffix rejection (bis §5.7 steps 9–10): a Domain
        // attribute that IS a public suffix is rejected — except when
        // identical to the request host, in which case it is treated as
        // host-only.
        let request_host = request_url.host_str().unwrap_or("").to_ascii_lowercase();
        let mut host_only = host_only_attr;
        if let Some(domain_attr) = raw.domain() {
            let attr = domain_attr
                .trim()
                .trim_start_matches('.')
                .to_ascii_lowercase();
            if is_public_suffix(&attr) {
                if attr != request_host {
                    return;
                }
                // Public suffix identical to the request host: treated
                // as host-only (bis §5.7), not as a suffix grant.
                host_only = true;
            }
        }
        if host_only {
            // Bis: a public-suffix Domain attribute identical to the
            // request host is host-only — a Suffix domain would wrongly
            // match subdomains.
            if let Ok(ho) = CookieDomain::host_only(request_url) {
                parsed.domain = ho;
            }
        }
        let domain = String::from(&parsed.domain);
        let path: String = String::from(&parsed.path);
        // Anti-tossing (bis §5.7 step 16): a non-secure cookie may not
        // overwrite an existing secure cookie of the same name whose
        // domain domain-matches the request host and whose path is
        // path-matched by the new cookie's path. Checked before the
        // expiry branch, so a non-secure deletion cannot toss a secure
        // cookie either.
        if !secure {
            let tossed = self
                .partitions
                .get(key)
                .is_some_and(|part| {
                    part.iter().any(|e| {
                        e.cookie.name() == name
                            && e.is_secure()
                            && e.cookie.domain.matches(request_url)
                            && path_matches(&path, &e.path)
                    })
                });
            if tossed {
                return;
            }
        }
        let expired = parsed.is_expired();
        self.seq += 1;
        let now = self.seq;
        let partition = self.partitions.entry(key.to_string()).or_default();
        partition.retain(|e| !e.cookie.is_expired());
        if expired {
            // Max-Age <= 0 (or a past Expires) deletes the matching
            // cookie and stores nothing.
            partition.retain(|e| !e.same_identity(name, &domain, host_only, &path));
            return;
        }
        match partition
            .iter_mut()
            .find(|e| e.same_identity(name, &domain, host_only, &path))
        {
            Some(existing) => {
                // Overwrite keeps the old creation-time (bis §5.7).
                let created = existing.created_seq;
                *existing = StoredCookie {
                    cookie: parsed.into_owned(),
                    partitioned,
                    host_only,
                    domain,
                    path,
                    created_seq: created,
                    last_access_seq: now,
                };
            }
            None => {
                partition.push(StoredCookie {
                    cookie: parsed.into_owned(),
                    partitioned,
                    host_only,
                    domain,
                    path,
                    created_seq: now,
                    last_access_seq: now,
                });
            }
        }
        self.enforce_limits(key);
    }

    /// Enforce the 50-per-domain (within `partition_key`) and 3000-total
    /// caps with RFC eviction priority: expired first (callers purge
    /// before this runs), then non-secure, then earliest last-access.
    fn enforce_limits(&mut self, partition_key: &str) {
        if let Some(partition) = self.partitions.get_mut(partition_key) {
            loop {
                let over_domain = {
                    let mut counts: HashMap<&str, usize> = HashMap::new();
                    let mut over = None;
                    for e in partition.iter() {
                        let n = counts.entry(e.domain.as_str()).or_insert(0);
                        *n += 1;
                        if *n > MAX_COOKIES_PER_DOMAIN {
                            over = Some(e.domain.clone());
                            break;
                        }
                    }
                    over
                };
                let Some(domain) = over_domain else {
                    break;
                };
                let victim = partition
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.domain == domain)
                    .min_by_key(|(_, e)| (e.is_secure(), e.last_access_seq))
                    .map(|(i, _)| i);
                match victim {
                    Some(i) => {
                        partition.swap_remove(i);
                    }
                    None => break,
                }
            }
        }
        while self.total_len() > MAX_COOKIES_TOTAL {
            let mut best: Option<(String, usize, bool, u64)> = None;
            for (key, part) in self.partitions.iter() {
                for (i, e) in part.iter().enumerate() {
                    let cand = (e.is_secure(), e.last_access_seq);
                    let better = match &best {
                        None => true,
                        Some((_, _, s, l)) => cand < (*s, *l),
                    };
                    if better {
                        best = Some((key.clone(), i, cand.0, cand.1));
                    }
                }
            }
            let Some((key, idx, _, _)) = best else {
                break;
            };
            if let Some(part) = self.partitions.get_mut(&key) {
                part.swap_remove(idx);
            }
        }
    }

    /// Total stored entries (including expired stragglers, which lazy
    /// purges remove on the next store/send for their partition).
    fn total_len(&self) -> usize {
        self.partitions.values().map(Vec::len).sum()
    }
}

/// Whether `same_site` permits sending on a cross-site request:
/// `None` (missing) is `Default`, which behaves as `Lax`; `Lax` /
/// `Default` ride only a top-level navigation with a safe method;
/// `Strict` never rides cross-site; `None` rides everywhere (receipt
/// already enforced `Secure` for it).
fn same_site_send_allowed(
    same_site: Option<SameSite>,
    method_is_safe: bool,
    is_top_level_navigation: bool,
) -> bool {
    match same_site {
        None | Some(SameSite::Lax) => method_is_safe && is_top_level_navigation,
        Some(SameSite::Strict) => false,
        Some(SameSite::None) => true,
    }
}

/// Schemeful same-site: same scheme and same registrable domain (or the
/// same bare host when there is no registrable domain, e.g. IPs and
/// `localhost`). Unparseable top-level site fails closed (cross-site).
fn is_same_site(request_url: &Url, top_level_site: &str) -> bool {
    let Ok(top_url) = Url::parse(top_level_site) else {
        return false;
    };
    if request_url.scheme() != top_url.scheme() {
        return false;
    }
    let (Some(req_host), Some(top_host)) = (request_url.host_str(), top_url.host_str()) else {
        return false;
    };
    registrable_domain(req_host) == registrable_domain(top_host)
}

/// Registrable domain (eTLD+1) via the Public Suffix List; the bare
/// lowercased host when the PSL has no registrable domain for it.
fn registrable_domain(host: &str) -> String {
    let lower = host.to_ascii_lowercase();
    psl::domain_str(&lower).unwrap_or(&lower).to_string()
}

/// The partition key for a top-level site: `scheme://registrable-domain`
/// (so `https://sub.example.com` and `https://example.com` share one
/// partition — they are the same site). Unparseable input is used
/// verbatim as the key.
fn partition_key(top_level_site: &str) -> String {
    match Url::parse(top_level_site) {
        Ok(url) => match url.host_str() {
            Some(host) => format!("{}://{}", url.scheme(), registrable_domain(host)),
            None => top_level_site.to_string(),
        },
        Err(_) => top_level_site.to_string(),
    }
}

/// Whether `domain` (lowercase, no leading dot) is itself a public
/// suffix, i.e. the PSL's suffix for it is the whole string.
fn is_public_suffix(domain: &str) -> bool {
    matches!(psl::suffix_str(domain), Some(s) if s.eq_ignore_ascii_case(domain))
}

/// Whether a connection to `url` counts as secure for cookie purposes:
/// HTTPS, or a trusted host (`localhost` / loopback), mirroring
/// `cookie_store`'s own `is_secure`.
fn url_is_secure(url: &Url) -> bool {
    if url.scheme() == "https" {
        return true;
    }
    match url.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

/// The same rule over a plain URL string, for the service's send
/// path (`service.rs::build_request`): one definition of "secure
/// for cookies", shared by intake and send.
pub fn url_is_secure_str(url: &str) -> bool {
    Url::parse(url).map(|u| url_is_secure(&u)).unwrap_or(false)
}

/// Request-path path-matches cookie-path (RFC 6265 §5.1.4): identical,
/// or the cookie-path is a prefix and the next request-path character
/// is `/`, or the cookie-path ends in `/`.
fn path_matches(request_path: &str, cookie_path: &str) -> bool {
    cookie_path == request_path
        || (request_path.starts_with(cookie_path)
            && (cookie_path.ends_with('/')
                || request_path[cookie_path.len()..].starts_with('/')))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(jar: &mut CookieJar, url: &str, site: &str, header: &str) {
        jar.store_set_cookie(url, site, &[header.to_string()]);
    }

    /// Send over a secure connection, same-site, top-level safe GET —
    /// the permissive case.
    fn get(jar: &mut CookieJar, url: &str, site: &str) -> Option<String> {
        jar.cookie_header(url, site, url.starts_with("https"), true, true)
    }

    #[test]
    fn basic_accept_and_send() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "session=abc; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("session=abc".to_string())
        );
        // Another site's partition does not see it.
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://other.example"),
            None
        );
        // A session cookie (no Max-Age/Expires) is accepted too.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "temp=1; Path=/",
        );
        assert_eq!(jar.len(), 2);
    }

    #[test]
    fn host_only_not_sent_to_subdomain_but_domain_cookie_is() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "hostonly=1; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "wide=1; Domain=example.com; Path=/; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://sub.example.com/", "https://example.com"),
            Some("wide=1".to_string())
        );
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com")
                .unwrap()
                .split("; ")
                .count(),
            2
        );
        // Leading-dot Domain is the same cookie.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "dotted=1; Domain=.example.com; Path=/; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://sub.example.com/", "https://example.com")
                .unwrap()
                .split("; ")
                .count(),
            2
        );
    }

    #[test]
    fn path_matching_and_default_path() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "app=1; Path=/app; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/app/x", "https://example.com"),
            Some("app=1".to_string())
        );
        // /application does NOT path-match /app (next char is not '/').
        assert_eq!(
            get(
                &mut jar,
                "https://example.com/application",
                "https://example.com"
            ),
            None
        );
        assert_eq!(
            get(&mut jar, "https://example.com/other", "https://example.com"),
            None
        );
        // Default path of /a/b is /a.
        set(
            &mut jar,
            "https://example.com/a/b",
            "https://example.com",
            "def=1; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/a/x", "https://example.com"),
            Some("def=1".to_string())
        );
        assert_eq!(
            get(&mut jar, "https://example.com/b/x", "https://example.com"),
            None
        );
    }

    #[test]
    fn secure_cookie_rejected_over_http_and_not_sent_insecure() {
        let mut jar = CookieJar::new();
        // Secure over a plain-HTTP response: rejected outright.
        set(
            &mut jar,
            "http://example.com/",
            "http://example.com",
            "s=1; Secure; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "s=1; Secure; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(
            jar.cookie_header("https://example.com/", "https://example.com", true, true, true),
            Some("s=1".to_string())
        );
        // Same URL, insecure connection: withheld.
        assert_eq!(
            jar.cookie_header("http://example.com/", "https://example.com", false, true, true),
            None
        );
    }

    #[test]
    fn max_age_beats_expires_and_zero_deletes() {
        let mut jar = CookieJar::new();
        // Expires is in the past, Max-Age is live: stored.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "a=1; Expires=Wed, 09 Jun 1999 10:18:14 GMT; Max-Age=3600; Path=/",
        );
        assert_eq!(jar.len(), 1);
        // Past Expires alone: never stored.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "b=1; Expires=Wed, 09 Jun 1999 10:18:14 GMT; Path=/",
        );
        assert_eq!(jar.len(), 1);
        // Max-Age=0 deletes the live cookie.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "a=; Max-Age=0; Path=/",
        );
        assert_eq!(jar.len(), 0);
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            None
        );
    }

    #[test]
    fn size_limit_4096() {
        let mut jar = CookieJar::new();
        // name (1) + value (4095) = 4096: accepted.
        let ok_value = "v".repeat(4095);
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            &format!("a={ok_value}; Path=/; Max-Age=3600"),
        );
        assert_eq!(jar.len(), 1);
        // name (1) + value (4096) = 4097: whole cookie ignored.
        let big_value = "v".repeat(4096);
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            &format!("b={big_value}; Path=/; Max-Age=3600"),
        );
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn per_domain_limit_50() {
        let mut jar = CookieJar::new();
        for i in 0..55 {
            set(
                &mut jar,
                "https://example.com/",
                "https://example.com",
                &format!("c{i}=v; Path=/; Max-Age=3600"),
            );
        }
        assert_eq!(jar.len(), 50);
    }

    #[test]
    fn total_limit_3000() {
        let mut jar = CookieJar::new();
        for i in 0..3005 {
            set(
                &mut jar,
                &format!("https://d{i}.example/"),
                "https://top.example",
                "c=v; Path=/; Max-Age=3600",
            );
        }
        assert_eq!(jar.len(), 3000);
    }

    #[test]
    fn eviction_prefers_non_secure() {
        let mut jar = CookieJar::new();
        // 49 non-secure + 1 secure, then 2 more non-secure: the secure
        // cookie must survive the overflow, the earliest non-secure go.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "keep=1; Secure; Path=/; Max-Age=3600",
        );
        for i in 0..51 {
            set(
                &mut jar,
                "https://example.com/",
                "https://example.com",
                &format!("n{i}=v; Path=/; Max-Age=3600"),
            );
        }
        assert_eq!(jar.len(), 50);
        let header = get(&mut jar, "https://example.com/", "https://example.com").unwrap();
        assert!(header.split("; ").any(|p| p == "keep=1"), "{header}");
    }

    #[test]
    fn samesite_default_is_lax() {
        let mut jar = CookieJar::new();
        // Set while tracker.example is embedded on news.example, so the
        // cookie lives in the news.example partition.
        set(
            &mut jar,
            "https://tracker.example/",
            "https://news.example",
            "t=1; Path=/; Max-Age=3600",
        );
        // Cross-site top-level navigation, safe method: sent (Lax).
        assert_eq!(
            jar.cookie_header(
                "https://tracker.example/",
                "https://news.example",
                true,
                true,
                true
            ),
            Some("t=1".to_string())
        );
        // Cross-site subresource: withheld.
        assert_eq!(
            jar.cookie_header(
                "https://tracker.example/",
                "https://news.example",
                true,
                true,
                false
            ),
            None
        );
        // Cross-site top-level POST (unsafe method): withheld.
        assert_eq!(
            jar.cookie_header(
                "https://tracker.example/",
                "https://news.example",
                true,
                false,
                true
            ),
            None
        );
        // Same-site subresource: sent (cookie set in its own partition).
        set(
            &mut jar,
            "https://tracker.example/",
            "https://tracker.example",
            "t=1; Path=/; Max-Age=3600",
        );
        assert_eq!(
            jar.cookie_header(
                "https://tracker.example/",
                "https://tracker.example",
                true,
                true,
                false
            ),
            Some("t=1".to_string())
        );
    }

    #[test]
    fn samesite_strict_never_cross_site() {
        let mut jar = CookieJar::new();
        // Set in the other.example partition, so the cross-site send
        // below actually reaches the Strict filter (not an empty
        // partition).
        set(
            &mut jar,
            "https://example.com/",
            "https://other.example",
            "s=1; SameSite=Strict; Path=/; Max-Age=3600",
        );
        assert_eq!(
            jar.cookie_header("https://example.com/", "https://other.example", true, true, true),
            None
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "s=1; SameSite=Strict; Path=/; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("s=1".to_string())
        );
    }

    #[test]
    fn samesite_none_requires_secure() {
        let mut jar = CookieJar::new();
        // Set in the other.example partition (cross-site context).
        set(
            &mut jar,
            "https://example.com/",
            "https://other.example",
            "n=1; SameSite=None; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        set(
            &mut jar,
            "https://example.com/",
            "https://other.example",
            "n=1; SameSite=None; Secure; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        // Cross-site subresource: sent (None rides everywhere).
        assert_eq!(
            jar.cookie_header(
                "https://example.com/",
                "https://other.example",
                true,
                true,
                false
            ),
            Some("n=1".to_string())
        );
    }

    #[test]
    fn unknown_samesite_is_default_lax() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://other.example",
            "u=1; SameSite=Banana; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(
            jar.cookie_header("https://example.com/", "https://other.example", true, true, true),
            Some("u=1".to_string())
        );
        assert_eq!(
            jar.cookie_header("https://example.com/", "https://other.example", true, true, false),
            None
        );
    }

    #[test]
    fn secure_prefix_rules() {
        let mut jar = CookieJar::new();
        // __Secure- without Secure: rejected.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__Secure-id=1; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        // With Secure: accepted. Prefix match is case-insensitive.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__SECURE-id=1; Secure; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn host_prefix_rules() {
        let mut jar = CookieJar::new();
        // Missing Secure: rejected.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__Host-id=1; Path=/; Max-Age=3600",
        );
        // Domain attribute: rejected (must be host-only).
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__Host-id=1; Secure; Domain=example.com; Path=/; Max-Age=3600",
        );
        // Path other than /: rejected.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__Host-id=1; Secure; Path=/app; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        // Secure + host-only + Path=/: accepted.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "__Host-id=1; Secure; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("__Host-id=1".to_string())
        );
    }

    #[test]
    fn partitioned_keys_stay_separate() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://tracker.example/",
            "https://a.example",
            "id=aaa; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://tracker.example/",
            "https://b.example",
            "id=bbb; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 2);
        assert_eq!(
            get(&mut jar, "https://tracker.example/", "https://a.example"),
            Some("id=aaa".to_string())
        );
        assert_eq!(
            get(&mut jar, "https://tracker.example/", "https://b.example"),
            Some("id=bbb".to_string())
        );
        // Same site via a subdomain top-level: same partition.
        assert_eq!(
            get(
                &mut jar,
                "https://tracker.example/",
                "https://sub.a.example"
            ),
            Some("id=aaa".to_string())
        );
    }

    #[test]
    fn partitioned_attribute_requires_secure() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "p=1; Partitioned; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "p=1; Partitioned; Secure; SameSite=None; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn http_only_cookies_are_sent_browser_side() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "sid=secret; HttpOnly; Path=/; Max-Age=3600",
        );
        // HttpOnly excludes non-HTTP APIs (document.cookie), never the
        // browser-side Cookie header: the value rides the request.
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("sid=secret".to_string())
        );
    }

    #[test]
    fn anti_tossing() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "sid=real; Secure; Path=/; Max-Age=3600",
        );
        // A non-secure cookie may not overwrite the secure one.
        set(
            &mut jar,
            "http://example.com/",
            "https://example.com",
            "sid=evil; Path=/; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("sid=real".to_string())
        );
        // …and a non-secure Max-Age=0 may not delete it either.
        set(
            &mut jar,
            "http://example.com/",
            "https://example.com",
            "sid=; Path=/; Max-Age=0",
        );
        assert_eq!(jar.len(), 1);
    }

    #[test]
    fn public_suffix_rejected() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "x=1; Domain=com; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        set(
            &mut jar,
            "https://example.co.uk/",
            "https://example.co.uk",
            "x=1; Domain=co.uk; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
        // A real registrable domain is fine, set from a subdomain so it
        // is a true suffix cookie (rides to sibling subdomains).
        set(
            &mut jar,
            "https://sub.example.com/",
            "https://example.com",
            "x=1; Domain=example.com; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 1);
        // A public-suffix Domain identical to the request host is
        // stored host-only, not as a suffix grant: sent to the host
        // itself, never to subdomains.
        set(
            &mut jar,
            "https://co.uk/",
            "https://co.uk",
            "y=1; Domain=co.uk; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 2);
        assert_eq!(
            get(&mut jar, "https://co.uk/", "https://co.uk"),
            Some("y=1".to_string())
        );
        assert_eq!(
            get(&mut jar, "https://example.co.uk/", "https://co.uk"),
            None
        );
        // x (suffix) rides to a sibling subdomain; y (host-only) does not.
        let header = get(
            &mut jar,
            "https://other.example.com/",
            "https://example.com",
        )
        .unwrap();
        assert!(header.split("; ").any(|p| p == "x=1"), "{header}");
        assert!(!header.split("; ").any(|p| p == "y=1"), "{header}");
    }

    #[test]
    fn session_cookies_cleared_at_session_end() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "sess=1; Path=/",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "pers=1; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 2);
        jar.clear_session_cookies();
        assert_eq!(jar.len(), 1);
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("pers=1".to_string())
        );
    }

    #[test]
    fn send_sort_longer_path_first() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "short=1; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "long=1; Path=/app; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/app/x", "https://example.com"),
            Some("long=1; short=1".to_string())
        );
    }

    #[test]
    fn empty_name_and_value_rejected() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "=; Path=/; Max-Age=3600",
        );
        // Nameless cookie whose value starts with a prefix: rejected.
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "=__Secure-x; Path=/; Max-Age=3600",
        );
        assert_eq!(jar.len(), 0);
    }

    #[test]
    fn from_json_prunes_expired_and_rebases_sequences() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "keep=1; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "live=1; Path=/; Max-Age=3600",
        );
        // A cookie that expires *after* being saved: hand-edit
        // the serialised `expires` (RFC 3339) of the "live" cookie
        // to the past — the intake path would have rejected an
        // already-expired `Set-Cookie`, so the only way an expired
        // cookie reaches disk is by expiring between save and load.
        // (Editing the raw `Set-Cookie` string alone would not do:
        // `is_expired` reads the separate `expires` field.)
        let json = jar.to_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let mut edited = false;
        if let Some(parts) = value.get_mut("partitions").and_then(|p| p.as_object_mut()) {
            for (_, entries) in parts.iter_mut() {
                for entry in entries.as_array_mut().unwrap_or(&mut Vec::new()) {
                    let is_live = entry
                        .get("cookie")
                        .and_then(|c| c.get("raw_cookie"))
                        .and_then(|r| r.as_str())
                        .map(|s| s.contains("live=1"))
                        .unwrap_or(false);
                    if is_live
                        && let Some(exp) = entry
                            .get_mut("cookie")
                            .and_then(|c| c.get_mut("expires"))
                            .and_then(|e| e.get_mut("AtUtc"))
                        {
                            *exp = serde_json::Value::String(
                                "1970-01-01T00:00:00Z".to_string(),
                            );
                            edited = true;
                        }
                }
            }
        }
        assert!(edited, "live cookie expiry not found in {json}");
        let json = serde_json::to_string(&value).unwrap();
        let mut restored = CookieJar::from_json(&json).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(
            get(&mut restored, "https://example.com/", "https://example.com"),
            Some("keep=1".to_string())
        );
        // The restored jar still orders + evicts sanely: store a
        // second cookie and confirm both send.
        set(
            &mut restored,
            "https://example.com/",
            "https://example.com",
            "second=2; Path=/; Max-Age=3600",
        );
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn document_cookies_hides_http_only() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "vis=1; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "hid=1; Path=/; Max-Age=3600; HttpOnly",
        );
        // HttpOnly is still sent on the wire...
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("vis=1; hid=1".to_string())
        );
        // ...but never exposed to `document.cookie`.
        assert_eq!(
            jar.document_cookies("https://example.com/", "https://example.com"),
            vec![("vis".to_string(), "1".to_string())]
        );
    }

    #[test]
    fn serde_round_trip() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "a=1; Path=/; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://tracker.example/",
            "https://a.example",
            "id=aaa; Path=/; Max-Age=3600",
        );
        let json = jar.to_json().unwrap();
        let mut restored = CookieJar::from_json(&json).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(
            get(&mut jar, "https://example.com/", "https://example.com"),
            Some("a=1".to_string())
        );
        assert_eq!(
            get(
                &mut restored,
                "https://tracker.example/",
                "https://a.example"
            ),
            Some("id=aaa".to_string())
        );
    }

    #[test]
    fn overwrite_keeps_creation_time_for_send_sort() {
        let mut jar = CookieJar::new();
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "a=1; Path=/app; Max-Age=3600",
        );
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "b=1; Path=/; Max-Age=3600",
        );
        // Overwrite a with a new value: creation-time kept, so the
        // /app cookie still sorts before the / cookie (longer path).
        set(
            &mut jar,
            "https://example.com/",
            "https://example.com",
            "a=2; Path=/app; Max-Age=3600",
        );
        assert_eq!(
            get(&mut jar, "https://example.com/app/x", "https://example.com"),
            Some("a=2; b=1".to_string())
        );
    }
}
