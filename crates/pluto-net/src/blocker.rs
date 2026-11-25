//! EasyList-compatible network-rule tracker blocker, Round 1
//! (basic only) — Stage 9 Recommendation 12.
//!
//! Syntax essentials (Adblock Plus filter docs; see the Stage 9
//! research report §h, "Tracker blocking"): header
//! `[Adblock Plus 2.0]`; `!` comments; network rules
//! `[pattern][$options]`; `||` domain anchor; `|` start/end
//! anchor; `^` separator placeholder; `*` wildcard; `/regex/`
//! rules; `$options` (comma-separated, `~` negates); `@@`
//! exception rules (checked before blocking rules, override
//! them); `$important` (overrides exceptions); `$document`
//! per-site disable (short-circuits everything for the page).
//!
//! Round 1 is network rules ONLY:
//!
//! - NO cosmetic filtering: `##` / `#@#` / `#?#` rules (including
//!   `##+js` scriptlets) are parsed and ignored. Cosmetic
//!   filtering is Round 2 (M10, via `pluto-css`).
//! - NO scriptlets.
//!
//! Matching shape (report §h): `||domain^` rules compile into a
//! hash set keyed by domain for O(1) lookup (the lookup walks the
//! request host's label suffixes, so subdomains match); substring
//! / wildcard rules live in a second list; regex rules are
//! compiled ONCE at list load and capped at [`MAX_REGEX_RULES`]
//! (extras are ignored and logged). Per rule we store the pattern
//! kind, domain, type mask, third-party flag, domain-restriction
//! list, exception flag, and source list + line number — the last
//! two so a "why was this blocked" UI can name the rule.
//!
//! Lists: EasyList + EasyPrivacy, fetched over HTTPS from
//! easylist.to with ETag / `If-Modified-Since`, refreshed at most
//! once every [`REFRESH_INTERVAL`]; the last-good copy is kept on
//! failure and a missing/empty list means "allow everything".
//! Refresh is async and never blocks page loads: matching is a
//! synchronous read over the in-memory compiled rules, and the
//! finisher is expected to hold the `Blocker` behind a lock and
//! refresh it from a background task.
//!
//! Gate order (Stage 5, frozen): the blocker runs BEFORE cache
//! lookup, so a blocked resource is neither served from cache nor
//! cached. The finisher wires the gate into `service.rs`.
//!
//! Lists: EasyList and EasyPrivacy are by the EasyList authors,
//! licensed CC BY-SA 3.0. Attribution belongs in Settings/About
//! (the browser UI), not just here — this comment is the code-side
//! record of that obligation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http::{HeaderMap, header};
use http_body_util::{BodyExt, Full};
use hyper::Request;
use pluto_ipc::Destination;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::client::shared_client;

/// Default EasyList URL (ads).
pub const EASYLIST_URL: &str = "https://easylist.to/easylist/easylist.txt";
/// Default EasyPrivacy URL (trackers — the primary list for this
/// blocker).
pub const EASYPRIVACY_URL: &str = "https://easylist.to/easylist/easyprivacy.txt";

/// File names for the lists inside the profile dir
/// (`$HOME/.config/pluto/`, owned by `pluto-browser`).
pub const EASYLIST_FILE: &str = "easylist.txt";
pub const EASYPRIVACY_FILE: &str = "easyprivacy.txt";
/// Per-list fetch metadata (ETag / Last-Modified / fetched-at),
/// so conditional requests survive a restart.
pub const BLOCKER_META_FILE: &str = "blocker-meta.json";

/// Maximum compiled regex rules. EasyList regex rules are the
/// slowest match class and a few use look-around, which the
/// `regex` crate cannot compile at all; beyond this cap extras
/// are ignored and counted (see [`Blocker::regex_skipped`]).
pub const MAX_REGEX_RULES: usize = 512;

/// Minimum time between list fetches: refresh at most once every
/// ~4 h (report §h: "at most once every ~4–6 h"). A refresh
/// inside the window is a no-op (`SkippedRecent`).
pub const REFRESH_INTERVAL: Duration = Duration::from_secs(4 * 60 * 60);

/// A list fetch that takes longer than this is abandoned; the
/// last-good copy stays in place.
pub const REFRESH_TIMEOUT: Duration = Duration::from_secs(60);

/// Largest list body accepted (EasyList is a few MB; 32 MiB is a
/// generous bound against a compromised/misbehaving list host
/// OOMing the browser).
pub const MAX_LIST_BYTES: usize = 32 * 1024 * 1024;

// ---------------------------------------------------------------------
// Type mask bits (one per ABP resource type option).
// ---------------------------------------------------------------------

const T_SCRIPT: u32 = 1 << 0;
const T_IMAGE: u32 = 1 << 1;
const T_STYLESHEET: u32 = 1 << 2;
const T_OBJECT: u32 = 1 << 3;
const T_SUBDOCUMENT: u32 = 1 << 4;
const T_XHR: u32 = 1 << 5;
const T_FONT: u32 = 1 << 6;
const T_MEDIA: u32 = 1 << 7;
const T_WEBSOCKET: u32 = 1 << 8;
const T_PING: u32 = 1 << 9;
const T_POPUP: u32 = 1 << 10;
const T_DOCUMENT: u32 = 1 << 11;
const T_OTHER: u32 = 1 << 12;
const T_ALL: u32 = (1 << 13) - 1;

/// Map a [`Destination`] to its ABP type bit. `Fetch` is
/// `xmlhttprequest` (covers `fetch()`); `Document` is `document`
/// (top-level navigations only — subdocument/iframe loads have no
/// `Destination` yet, so `$subdocument` rules never match a Pluto
/// request in this milestone). Types with no `Destination`
/// (`media`, `object`, `websocket`, `ping`, `popup`) can only
/// appear in a rule's mask; a rule restricted to *only* those
/// types never matches.
fn destination_bit(destination: Destination) -> u32 {
    match destination {
        Destination::Document => T_DOCUMENT,
        Destination::Stylesheet => T_STYLESHEET,
        Destination::Script => T_SCRIPT,
        Destination::Image => T_IMAGE,
        Destination::Font => T_FONT,
        Destination::Fetch => T_XHR,
        Destination::Other => T_OTHER,
    }
}

/// One ABP type-option name → its bit. `xhr` is the alias for
/// `xmlhttprequest`; `3p` is the alias for `third-party`
/// (handled separately, not a type bit).
fn type_option_bit(name: &str) -> Option<u32> {
    Some(match name {
        "script" => T_SCRIPT,
        "image" => T_IMAGE,
        "stylesheet" => T_STYLESHEET,
        "object" => T_OBJECT,
        "subdocument" => T_SUBDOCUMENT,
        "xmlhttprequest" | "xhr" => T_XHR,
        "font" => T_FONT,
        "media" => T_MEDIA,
        "websocket" => T_WEBSOCKET,
        "ping" => T_PING,
        "popup" => T_POPUP,
        "document" => T_DOCUMENT,
        "other" => T_OTHER,
        _ => return None,
    })
}

/// Whether a token is a recognised `$option` name at all
/// (type, third-party, domain=, important, match-case). Used to
/// decide whether a `$suffix` is really an options list or just
/// a `$` inside the pattern (e.g. a query string).
fn is_known_option(token: &str) -> bool {
    let name = token.strip_prefix('~').unwrap_or(token);
    type_option_bit(name).is_some()
        || matches!(name, "third-party" | "3p" | "important" | "match-case")
        || name.starts_with("domain=")
}

// ---------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------

/// Which list a rule came from. Serialised into the fetch
/// metadata file, so the names are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ListSource {
    EasyList,
    EasyPrivacy,
    /// A user-supplied list (not fetched; loaded from disk or
    /// pasted in by the user). Round 1 loads it like any other
    /// list; the settings UI that manages it is later work.
    User,
}

impl ListSource {
    /// File name inside the profile dir.
    pub fn file_name(self) -> &'static str {
        match self {
            ListSource::EasyList => EASYLIST_FILE,
            ListSource::EasyPrivacy => EASYPRIVACY_FILE,
            ListSource::User => "user-filters.txt",
        }
    }

    /// Default fetch URL, if the list is fetched at all.
    pub fn url(self) -> Option<&'static str> {
        match self {
            ListSource::EasyList => Some(EASYLIST_URL),
            ListSource::EasyPrivacy => Some(EASYPRIVACY_URL),
            ListSource::User => None,
        }
    }
}

/// The rule that decided a block, for the "why was this blocked"
/// UI: the original rule text, where it came from, and whether it
/// is an exception / `$important` rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedRule {
    /// The original rule line, trimmed.
    pub pattern: String,
    /// Which list the rule came from.
    pub source: ListSource,
    /// 1-based line number in that list.
    pub line: usize,
    /// `@@` exception rule.
    pub exception: bool,
    /// `$important` rule (overrides exceptions).
    pub important: bool,
}

/// Outcome of refreshing one list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStatus {
    /// A new copy was fetched and compiled.
    Updated,
    /// The server answered 304 — the stored copy is current.
    NotModified,
    /// Inside [`REFRESH_INTERVAL`] of the last fetch; no request
    /// was made.
    SkippedRecent,
    /// The fetch failed (transport, HTTP status, oversize body,
    /// or timeout). The last-good copy stays in place.
    Failed,
}

/// One list's refresh outcome, with the failure reason (if any)
/// as a human-readable string.
#[derive(Debug, Clone)]
pub struct ListRefresh {
    pub source: ListSource,
    pub status: RefreshStatus,
    /// Set when `status == Failed`.
    pub reason: Option<String>,
}

/// Outcome of [`Blocker::refresh`], one entry per fetched list.
#[derive(Debug, Clone)]
pub struct RefreshOutcome {
    pub results: Vec<ListRefresh>,
}

// ---------------------------------------------------------------------
// Internal rule representation
// ---------------------------------------------------------------------

/// One token of a wildcard pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// A literal character.
    Lit(char),
    /// `*`: any sequence of characters (including empty).
    Star,
    /// `^`: one separator character (anything that is not a
    /// letter, digit, `_`, `-`, `.`, `%`) — or end of string.
    Sep,
}

/// How a rule's pattern matches a URL.
#[derive(Debug)]
enum PatternKind {
    /// `||domain^`: the request host is the domain or a
    /// subdomain of it. Indexed in a hash set (see
    /// [`Blocker::domain_index`]).
    Domain { domain: String },
    /// `||domain<rest>`: domain-anchored, with a wildcard tail
    /// matched against the URL remainder after the domain.
    DomainPrefix { domain: String, rest: Vec<Token> },
    /// Any other pattern: optional start/end anchors plus
    /// wildcard tokens, matched against the full URL. Implicit
    /// `*` at both ends unless anchored (bare patterns are
    /// substring matches).
    Generic { tokens: Vec<Token> },
    /// `/regex/` rule, compiled once at list load.
    Regex(regex::Regex),
}

/// A page-domain restriction from `$domain=`: the rule applies
/// only on pages whose host matches (suffix match, so a
/// restriction on `example.com` also covers
/// `sub.example.com`), unless `negated`.
#[derive(Debug, Clone)]
struct DomainRestriction {
    domain: String,
    negated: bool,
}

/// One compiled network rule.
#[derive(Debug)]
struct Rule {
    kind: PatternKind,
    /// ABP type mask, or `None` when the rule has no type
    /// options (applies to every destination).
    types: Option<u32>,
    /// `Some(true)` = `$third-party` only, `Some(false)` =
    /// `$~third-party` (first-party only), `None` = either.
    third_party: Option<bool>,
    /// `$domain=` restrictions (empty = unrestricted).
    domains: Vec<DomainRestriction>,
    /// `$match-case`: match case-sensitively. Default (absent)
    /// is case-insensitive.
    match_case: bool,
    /// The public "why was this blocked" record.
    info: MatchedRule,
}

/// Which precedence class a matching rule falls in
/// (`$important` block > `@@` exception > ordinary block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchClass {
    ImportantBlock,
    Block,
    Exception,
}

/// Per-list fetch metadata, persisted to
/// [`BLOCKER_META_FILE`] so conditional requests survive a
/// restart.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ListMeta {
    etag: Option<String>,
    last_modified: Option<String>,
    /// Seconds since the Unix epoch of the last fetch attempt
    /// that reached the server (200 or 304).
    fetched_at_secs: Option<u64>,
}

// ---------------------------------------------------------------------
// Blocker
// ---------------------------------------------------------------------

/// An EasyList-compatible network-rule blocker (Round 1).
///
/// Matching is pure and offline: [`Blocker::is_blocked`] reads
/// only the compiled in-memory rules. Lists are loaded from
/// disk ([`Blocker::load_from_dir`]) or from text
/// ([`Blocker::load_list`], the test seam), and refreshed over
/// HTTPS ([`Blocker::refresh`]). A missing/empty list means
/// "allow everything" — the blocker never fails closed on a
/// missing list, it just has nothing to match.
#[derive(Debug, Default)]
pub struct Blocker {
    /// All compiled rules, in load order.
    rules: Vec<Rule>,
    /// `||domain^` rules, keyed by their domain. Lookup walks
    /// the request host's label suffixes (`a.b.c.com`,
    /// `b.c.com`, `c.com`), so each suffix probe is O(1) and
    /// subdomains match without scanning the rule list.
    domain_index: HashMap<String, Vec<usize>>,
    /// Generic (substring/wildcard) and domain-prefixed rules,
    /// in load order.
    generic: Vec<usize>,
    /// Compiled regex rules, in load order (capped at
    /// [`MAX_REGEX_RULES`] across all lists).
    regexes: Vec<usize>,
    /// Exception rules carrying the `$document` type: a match
    /// against the top-level site disables blocking for the
    /// whole page.
    document_exceptions: Vec<usize>,
    /// Fetch metadata per list.
    meta: HashMap<ListSource, ListMeta>,
    /// Regex rules ignored because the [`MAX_REGEX_RULES`] cap
    /// was already reached, per list (so a reload does not
    /// double-count). See [`Blocker::regex_skipped`].
    regex_skipped_by_source: HashMap<ListSource, usize>,
}

impl Blocker {
    /// An empty blocker: no lists loaded, everything allowed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the persisted lists (and fetch metadata) from a
    /// profile dir. Missing or unreadable files are skipped —
    /// the blocker just starts empty for that list.
    pub fn load_from_dir(dir: &Path) -> Self {
        let mut blocker = Self::new();
        if let Ok(text) = std::fs::read_to_string(dir.join(BLOCKER_META_FILE)) {
            blocker.meta = serde_json::from_str(&text).unwrap_or_default();
        }
        for source in [ListSource::EasyList, ListSource::EasyPrivacy, ListSource::User] {
            if let Ok(text) = std::fs::read_to_string(dir.join(source.file_name())) {
                blocker.load_list(source, &text);
            }
        }
        blocker
    }

    /// Parse `text` as a filter list and REPLACE that source's
    /// rules with the result. The matching core's test seam, and
    /// what [`Blocker::refresh`] calls after a successful fetch.
    /// A malformed line is skipped, never fatal.
    pub fn load_list(&mut self, source: ListSource, text: &str) {
        self.rules.retain(|r| r.info.source != source);
        // The cap counts regex rules across ALL lists, so start
        // from the ones still compiled after this source's old
        // rules were dropped.
        let mut regex_count = self
            .rules
            .iter()
            .filter(|r| matches!(r.kind, PatternKind::Regex(_)))
            .count();
        let mut skipped_here = 0usize;
        for (idx, line) in text.lines().enumerate() {
            match parse_rule(source, idx + 1, line, regex_count) {
                ParseOutcome::Rule(rule) => {
                    if matches!(rule.kind, PatternKind::Regex(_)) {
                        regex_count += 1;
                    }
                    self.rules.push(rule);
                }
                ParseOutcome::RegexCapped => {
                    skipped_here += 1;
                }
                ParseOutcome::Skip => {}
            }
        }
        if skipped_here > 0 {
            eprintln!(
                "[pluto] blocker: {n} regex rules from {source:?} ignored \
                 (cap {cap} reached); regex rules are the slowest match class",
                n = skipped_here,
                cap = MAX_REGEX_RULES,
            );
        }
        self.regex_skipped_by_source.insert(source, skipped_here);
        self.rebuild_indexes();
    }

    /// Total compiled rules across all lists.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Compiled rules from one list.
    pub fn rule_count_for(&self, source: ListSource) -> usize {
        self.rules.iter().filter(|r| r.info.source == source).count()
    }

    /// True when no list is loaded (or all loaded lists were
    /// empty): everything is allowed.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Regex rules ignored because the [`MAX_REGEX_RULES`] cap
    /// was reached (across all lists currently loaded).
    pub fn regex_skipped(&self) -> usize {
        self.regex_skipped_by_source.values().sum()
    }

    /// When this list was last fetched from the network (200 or
    /// 304), if ever. `None` means never fetched (or loaded
    /// from disk without metadata).
    pub fn last_refresh(&self, source: ListSource) -> Option<SystemTime> {
        self.meta
            .get(&source)?
            .fetched_at_secs
            .map(|s| UNIX_EPOCH + Duration::from_secs(s))
    }

    /// Rebuild the match indexes after the rule set changed.
    fn rebuild_indexes(&mut self) {
        self.domain_index.clear();
        self.generic.clear();
        self.regexes.clear();
        self.document_exceptions.clear();
        for (idx, rule) in self.rules.iter().enumerate() {
            match &rule.kind {
                PatternKind::Domain { domain } => {
                    self.domain_index.entry(domain.clone()).or_default().push(idx);
                }
                PatternKind::DomainPrefix { .. } | PatternKind::Generic { .. } => {
                    self.generic.push(idx);
                }
                PatternKind::Regex(_) => self.regexes.push(idx),
            }
            if rule.info.exception
                && rule.types.map_or(false, |t| t & T_DOCUMENT != 0)
            {
                self.document_exceptions.push(idx);
            }
        }
    }

    /// The matching core: is this request blocked, and by which
    /// rule? Pure and offline — reads only the compiled rules.
    ///
    /// * `url` — the request URL.
    /// * `destination` — the resource type ([`Destination`]).
    /// * `top_level_site` — the page URL (document URL / origin
    ///   string). Drives `$third-party` and `$domain=`; a
    ///   `$document` exception matching it disables blocking for
    ///   the whole page.
    ///
    /// Returns the rule that blocked the request, or `None`
    /// when the request is allowed. Precedence: an `$important`
    /// blocking rule beats any exception; an `@@` exception beats
    /// an ordinary blocking rule; a `$document` exception on the
    /// page beats an ordinary blocking rule but not an
    /// `$important` one.
    pub fn is_blocked(
        &self,
        url: &str,
        destination: Destination,
        top_level_site: &str,
    ) -> Option<&MatchedRule> {
        if self.rules.is_empty() {
            return None;
        }
        let Some(host) = url_host(url) else {
            return None;
        };
        let page_host = url_host(top_level_site).unwrap_or_default();
        let third_party = is_third_party(&host, &page_host);
        let dest_bit = destination_bit(destination);

        // The request URL in both cases, for `$match-case` rules
        // (default matching is case-insensitive).
        let url_lower = url.to_lowercase();
        let host_path = strip_scheme(url);
        let host_path_lower = host_path.to_lowercase();

        // Scan in specificity order (domain-hash, then generic,
        // then regex), recording the first match of each
        // precedence class. The scan cannot return early: an
        // exception later in the list still beats an ordinary
        // block found earlier.
        let mut important_block: Option<&Rule> = None;
        let mut block: Option<&Rule> = None;
        let mut exception: Option<&Rule> = None;

        // Classify one rule against this request (or `None`
        // when it does not match), then file the first match of
        // each precedence class. Expanded inline by the macro
        // below (a closure cannot hold the matched `&Rule` past
        // its own call — E0521).
        fn classify<'r>(
            rule: &'r Rule,
            url_lower: &str,
            url: &str,
            host_path_lower: &str,
            host_path: &str,
            host: &str,
            dest_bit: u32,
            third_party: bool,
            page_host: &str,
        ) -> Option<(MatchClass, &'r Rule)> {
            if !rule_applies(rule, dest_bit, third_party, page_host) {
                return None;
            }
            if !rule_pattern_matches(rule, url_lower, url, host_path_lower, host_path, host) {
                return None;
            }
            let class = if rule.info.important && !rule.info.exception {
                MatchClass::ImportantBlock
            } else if rule.info.exception {
                MatchClass::Exception
            } else {
                MatchClass::Block
            };
            Some((class, rule))
        }

        macro_rules! consider {
            ($idx:expr) => {
                if let Some((class, rule)) = classify(
                    &self.rules[$idx],
                    &url_lower,
                    url,
                    &host_path_lower,
                    host_path,
                    &host,
                    dest_bit,
                    third_party,
                    &page_host,
                ) {
                    match class {
                        MatchClass::ImportantBlock => {
                            if important_block.is_none() {
                                important_block = Some(rule);
                            }
                        }
                        MatchClass::Exception => {
                            if exception.is_none() {
                                exception = Some(rule);
                            }
                        }
                        MatchClass::Block => {
                            if block.is_none() {
                                block = Some(rule);
                            }
                        }
                    }
                }
            };
        }

        // Domain-hash rules first: walk the host's label
        // suffixes and probe the set for each.
        let mut suffix = host.as_str();
        loop {
            if let Some(idxs) = self.domain_index.get(suffix) {
                for &idx in idxs {
                    consider!(idx);
                }
            }
            let Some(dot) = suffix.find('.') else {
                break;
            };
            suffix = &suffix[dot + 1..];
        }
        for &idx in &self.generic {
            consider!(idx);
        }
        for &idx in &self.regexes {
            consider!(idx);
        }

        if let Some(rule) = important_block {
            return Some(&rule.info);
        }
        // `$document` exception on the page: short-circuits
        // everything for the page (unless an `$important` rule
        // above already won).
        if self.page_has_document_exception(top_level_site, &page_host) {
            return None;
        }
        if exception.is_some() {
            return None;
        }
        block.map(|rule| &rule.info)
    }

    /// Does a `$document` exception match this page? The
    /// exception's pattern is matched against the top-level
    /// site URL as a document request.
    fn page_has_document_exception(&self, top_level_site: &str, page_host: &str) -> bool {
        if self.document_exceptions.is_empty() || top_level_site.is_empty() {
            return false;
        }
        let url_lower = top_level_site.to_lowercase();
        let host_path = strip_scheme(top_level_site);
        let host_path_lower = host_path.to_lowercase();
        for &idx in &self.document_exceptions {
            let rule = &self.rules[idx];
            // A `$document` exception applies to the whole page
            // regardless of which page the rule's `$domain=`
            // restriction names — the restriction is still
            // honoured (it is the page-domain check itself).
            if rule_applies(rule, T_DOCUMENT, false, page_host)
                && rule_pattern_matches(
                    rule,
                    &url_lower,
                    top_level_site,
                    &host_path_lower,
                    host_path,
                    page_host,
                )
            {
                return true;
            }
        }
        false
    }

    /// Refresh EasyList + EasyPrivacy over HTTPS, honouring the
    /// cadence ([`REFRESH_INTERVAL`]) and conditional requests
    /// (ETag / `If-Modified-Since`). Writes the new copies and
    /// the fetch metadata to `dir` (atomically, temp file +
    /// rename). On any failure the last-good compiled rules stay
    /// in place and the outcome reports `Failed`.
    ///
    /// This never blocks page loads: it is async, it only
    /// touches the network from a background task, and matching
    /// ([`Blocker::is_blocked`]) never waits on it. It takes
    /// `&mut self`, so the caller holds the `Blocker` behind a
    /// lock while a background task refreshes it.
    pub async fn refresh(&mut self, dir: &Path) -> RefreshOutcome {
        let mut results = Vec::with_capacity(2);
        for source in [ListSource::EasyList, ListSource::EasyPrivacy] {
            results.push(self.refresh_one(source, dir).await);
        }
        RefreshOutcome { results }
    }

    async fn refresh_one(&mut self, source: ListSource, dir: &Path) -> ListRefresh {
        let Some(url) = source.url() else {
            return ListRefresh {
                source,
                status: RefreshStatus::Failed,
                reason: Some("list has no fetch URL".to_string()),
            };
        };
        let meta = self.meta.entry(source).or_default();
        if let Some(fetched_at) = meta.fetched_at_secs {
            let elapsed = SystemTime::now()
                .duration_since(UNIX_EPOCH + Duration::from_secs(fetched_at))
                .unwrap_or(Duration::ZERO);
            if elapsed < REFRESH_INTERVAL {
                return ListRefresh {
                    source,
                    status: RefreshStatus::SkippedRecent,
                    reason: None,
                };
            }
        }
        let fetched = fetch_list(url, meta.etag.as_deref(), meta.last_modified.as_deref()).await;
        match fetched {
            FetchOutcome::NotModified => {
                meta.fetched_at_secs = Some(now_secs());
                self.save_meta(dir);
                ListRefresh {
                    source,
                    status: RefreshStatus::NotModified,
                    reason: None,
                }
            }
            FetchOutcome::Updated { text, etag, last_modified } => {
                self.load_list(source, &text);
                if write_file_atomic(dir, source.file_name(), &text).is_err() {
                    // The compiled rules are already live; only
                    // the on-disk copy failed to update.
                    eprintln!("[pluto] blocker: could not save {}", source.file_name());
                }
                let meta = self.meta.entry(source).or_default();
                meta.etag = etag;
                meta.last_modified = last_modified;
                meta.fetched_at_secs = Some(now_secs());
                self.save_meta(dir);
                ListRefresh {
                    source,
                    status: RefreshStatus::Updated,
                    reason: None,
                }
            }
            FetchOutcome::Failed(reason) => ListRefresh {
                source,
                status: RefreshStatus::Failed,
                reason: Some(reason),
            },
        }
    }

    fn save_meta(&self, dir: &Path) {
        let Ok(json) = serde_json::to_string(&self.meta) else {
            return;
        };
        if write_file_atomic(dir, BLOCKER_META_FILE, &json).is_err() {
            eprintln!("[pluto] blocker: could not save {BLOCKER_META_FILE}");
        }
    }
}

/// Seconds since the Unix epoch.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

/// Atomic write inside the profile dir (temp file + rename, the
/// same pattern `pluto-browser` uses for its profile files).
/// The temp name is unique per write (process id + a
/// per-process counter, created with `O_EXCL` and retried on
/// collision): two writers to the same profile dir — two Pluto
/// windows sharing `$HOME/.config/pluto`, or two test threads —
/// must never share one temp file, or their writes interleave
/// and the renamed target ends up corrupt (Stage 10 Rec 6).
fn write_file_atomic(dir: &Path, name: &str, contents: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let target = dir.join(name);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut attempt = 0;
    loop {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = dir.join(format!(".{name}.{}.{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(mut file) => {
                use std::io::Write;
                let written = file.write_all(contents.as_bytes());
                drop(file);
                let result = written.and_then(|()| std::fs::rename(&tmp, target));
                if result.is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
                return result;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // A stale temp file with this pid + counter (a pid
                // reused after a crash, or a counter wrap): take
                // the next name instead of clobbering it.
                attempt += 1;
                if attempt > 100 {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------
// Matching helpers (pure)
// ---------------------------------------------------------------------

/// The registrable domain of a host, approximated as the last
/// two labels (`sub.example.com` → `example.com`). No public
/// suffix list is bundled yet (report §b: a PSL is the M5
/// refinement for `top_level_site`); this mis-keys hosts under
/// two-label public suffixes (`example.co.uk` → `co.uk`).
/// IP literals and single-label hosts are returned whole.
fn registrable_domain(host: &str) -> &str {
    if host.contains(':') || host.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return host;
    }
    // Last two labels (`sub.example.com` → `example.com`).
    let mut labels = host.rsplit('.');
    let last = labels.next().unwrap_or("");
    let Some(second) = labels.next() else {
        return host;
    };
    if second.is_empty() {
        return host;
    }
    &host[host.len() - second.len() - 1 - last.len()..]
}

/// Whether the request host is third-party relative to the page
/// host (different registrable domains). An unparseable/empty
/// page host counts as first-party — the blocker must not break
/// pages when it lacks page context.
fn is_third_party(host: &str, page_host: &str) -> bool {
    if page_host.is_empty() {
        return false;
    }
    registrable_domain(host) != registrable_domain(page_host)
}

/// Host of a URL, lowercased. `None` when the URL does not parse
/// or has no host.
fn url_host(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()?
        .host_str()
        .map(|h| h.to_lowercase())
}

/// The URL after `scheme://` (host + path + query + fragment),
/// for `||` domain-anchored matching. URLs without `://` are
/// returned whole.
fn strip_scheme(url: &str) -> &str {
    match url.find("://") {
        Some(idx) => &url[idx + 3..],
        None => url,
    }
}

/// A separator character for `^`: anything that is not a letter,
/// digit, `_`, `-`, `.`, or `%` (ABP filter docs).
fn is_separator(c: char) -> bool {
    !(c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '%'))
}

/// Do the rule's type mask, third-party flag, and `$domain=`
/// restrictions admit this request? (Pattern matching is
/// separate, in [`rule_pattern_matches`].)
fn rule_applies(rule: &Rule, dest_bit: u32, third_party: bool, page_host: &str) -> bool {
    if let Some(types) = rule.types {
        if types & dest_bit == 0 {
            return false;
        }
    }
    if let Some(only_third_party) = rule.third_party {
        if only_third_party != third_party {
            return false;
        }
    }
    if !rule.domains.is_empty() {
        let mut has_positive = false;
        let mut positive_hit = false;
        for restriction in &rule.domains {
            let hit = host_matches_domain(page_host, &restriction.domain);
            if restriction.negated {
                if hit {
                    return false;
                }
            } else {
                has_positive = true;
                if hit {
                    positive_hit = true;
                }
            }
        }
        if has_positive && !positive_hit {
            return false;
        }
    }
    true
}

/// Does `host` equal `domain` or sit under it (`sub.domain`)?
/// Both sides are already lowercased.
fn host_matches_domain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Does the rule's pattern match this request? The caller
/// supplies the URL (and host+path remainder) in both cases so
/// `$match-case` rules can match the original case while
/// everything else matches lowercased.
fn rule_pattern_matches(
    rule: &Rule,
    url_lower: &str,
    url: &str,
    host_path_lower: &str,
    host_path: &str,
    host: &str,
) -> bool {
    let (url_text, host_path_text) = if rule.match_case {
        (url, host_path)
    } else {
        (url_lower, host_path_lower)
    };
    match &rule.kind {
        PatternKind::Domain { domain } => host_matches_domain(host, domain),
        PatternKind::DomainPrefix { domain, rest } => {
            // The host must be the domain or a subdomain of it;
            // the tail is matched against the URL remainder
            // AFTER the domain (which starts with the separator
            // the `^` consumed, or is empty at end of host).
            let Some(offset) = host_domain_offset(host, domain) else {
                return false;
            };
            let remainder = &host_path_text[offset + domain.len()..];
            let text: Vec<char> = remainder.chars().collect();
            wildcard_match(rest, &text)
        }
        PatternKind::Generic { tokens } => {
            let text: Vec<char> = url_text.chars().collect();
            wildcard_match(tokens, &text)
        }
        PatternKind::Regex(re) => re.is_match(url_text),
    }
}

/// Byte offset in `host_path` where `domain` starts, or `None`
/// when the host is not the domain or a subdomain of it.
/// `host_path` begins with the host, so the offset is the
/// domain's start inside the host prefix.
fn host_domain_offset(host: &str, domain: &str) -> Option<usize> {
    if host == domain {
        Some(0)
    } else if let Some(rest) = host.strip_suffix(&format!(".{domain}")) {
        Some(rest.len() + 1)
    } else {
        None
    }
}

/// Wildcard match over tokens: `Lit` matches one equal char,
/// `Sep` matches one separator char or end of string, `Star`
/// matches any sequence. Classic two-pointer match with star
/// backtracking; `Sep` matching end-of-string is handled by the
/// trailing-token drain (a trailing `Sep` consumes nothing at
/// end of input).
fn wildcard_match(tokens: &[Token], text: &[char]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star_p, mut star_t) = (None, None);
    while t < text.len() {
        match tokens.get(p) {
            Some(Token::Lit(c)) if *c == text[t] => {
                p += 1;
                t += 1;
            }
            Some(Token::Sep) if is_separator(text[t]) => {
                p += 1;
                t += 1;
            }
            Some(Token::Star) => {
                star_p = Some(p + 1);
                star_t = Some(t);
                p += 1;
            }
            _ => {
                // Mismatch (or pattern exhausted): rewind to
                // the last `*` and let it swallow one more char.
                match (star_p, star_t) {
                    (Some(sp), Some(st)) => {
                        p = sp;
                        t = st + 1;
                        star_t = Some(st + 1);
                    }
                    _ => return false,
                }
            }
        }
    }
    // Text exhausted: only trailing `*`s and `^`s (matching end
    // of string) may remain.
    while p < tokens.len() {
        match tokens[p] {
            Token::Star | Token::Sep => p += 1,
            Token::Lit(_) => return false,
        }
    }
    true
}

// ---------------------------------------------------------------------
// List parsing
// ---------------------------------------------------------------------

enum ParseOutcome {
    Rule(Rule),
    /// A regex rule beyond [`MAX_REGEX_RULES`]: ignored.
    RegexCapped,
    /// Comment, header, cosmetic rule, blank line, or malformed
    /// line: ignored.
    Skip,
}

/// Parse one list line into a compiled rule, if it is a network
/// rule. `regex_count` is the number of regex rules already
/// compiled across all lists (for the cap).
fn parse_rule(source: ListSource, line_no: usize, line: &str, regex_count: usize) -> ParseOutcome {
    let line = line.trim();
    if line.is_empty() || line.starts_with('!') || line.starts_with('[') {
        return ParseOutcome::Skip;
    }
    let mut rest = line;
    let mut exception = false;
    if let Some(stripped) = rest.strip_prefix("@@") {
        exception = true;
        rest = stripped;
    }

    // Split pattern from `$options`. A regex rule's options
    // follow its closing `/`; anything else splits on the last
    // `$` — but only when the suffix parses as options (a `$`
    // inside a pattern, e.g. a query string, is not a separator).
    // A line starting with `/` that is NOT a well-formed
    // `/regex/[$options]` (e.g. the substring pattern
    // `/ads/banner`) falls back to plain-pattern parsing.
    let (pattern, options) = if rest.starts_with('/') {
        match split_regex_options(rest) {
            Some((pat, opts)) => (pat, opts),
            None => split_pattern_options(rest),
        }
    } else {
        split_pattern_options(rest)
    };

    // Cosmetic rules (`##`, `#@#`, `#?#` — including `##+js`
    // scriptlets) are Round 2: parse and ignore.
    if pattern.contains("##") || pattern.contains("#@#") || pattern.contains("#?#") {
        return ParseOutcome::Skip;
    }
    if pattern.is_empty() {
        return ParseOutcome::Skip;
    }

    let opts = parse_options(options);
    let match_case = opts.match_case;

    let kind = if pattern.starts_with("||") {
        match parse_domain_rule(&pattern, match_case) {
            Some(kind) => Some(kind),
            // `*` in the domain part (e.g. `||*.example.com^`)
            // or an empty domain: fall back to a generic
            // substring match on the remainder. Slightly
            // over-broad (it can match in a path too), but it
            // preserves coverage for a rare shape.
            None => parse_generic_rule(&pattern[2..], match_case),
        }
    } else if pattern.starts_with('/') {
        // A `/regex/` rule (options already split off).
        if regex_count >= MAX_REGEX_RULES {
            return ParseOutcome::RegexCapped;
        }
        let src = &pattern[1..pattern.len().saturating_sub(1)];
        let compiled = if match_case {
            regex::Regex::new(src)
        } else {
            regex::Regex::new(&format!("(?i){src}"))
        };
        match compiled {
            Ok(re) => Some(PatternKind::Regex(re)),
            Err(_) => return ParseOutcome::Skip, // e.g. look-around: unsupported, skip
        }
    } else {
        parse_generic_rule(&pattern, match_case)
    };

    let Some(kind) = kind else {
        return ParseOutcome::Skip;
    };

    ParseOutcome::Rule(Rule {
        kind,
        types: opts.types,
        third_party: opts.third_party,
        domains: opts.domains,
        match_case,
        info: MatchedRule {
            pattern: line.to_string(),
            source,
            line: line_no,
            exception,
            important: opts.important,
        },
    })
}

/// For a `/regex/[$options]` line: the regex is between the
/// first and last `/`; options (if any) follow the last `/` after
/// a `$`. Returns `None` when there is no closing `/`.
fn split_regex_options(line: &str) -> Option<(&str, Option<&str>)> {
    let last_slash = line.rfind('/')?;
    if last_slash == 0 {
        return None;
    }
    let pattern = &line[..=last_slash];
    let after = &line[last_slash + 1..];
    let options = match after.strip_prefix('$') {
        Some(opts) => Some(opts),
        None if after.is_empty() => None,
        None => return None, // trailing junk after the regex
    };
    Some((pattern, options))
}

/// For a non-regex line: split on the last `$` when the suffix
/// parses as an options list (at least one recognised option);
/// otherwise the whole line is the pattern.
fn split_pattern_options(line: &str) -> (&str, Option<&str>) {
    let Some(dollar) = line.rfind('$') else {
        return (line, None);
    };
    let (pattern, suffix) = line.split_at(dollar);
    let options = &suffix[1..];
    if options.is_empty() {
        return (line, None);
    }
    let looks_like_options = options
        .split(',')
        .any(|token| is_known_option(token.trim()));
    if looks_like_options {
        (pattern, Some(options))
    } else {
        (line, None)
    }
}

struct ParsedOptions {
    types: Option<u32>,
    third_party: Option<bool>,
    domains: Vec<DomainRestriction>,
    important: bool,
    match_case: bool,
}

/// Parse a `$options` list (comma-separated, `~` negates).
/// Unknown options are ignored (ABP leniency — lists contain
/// options Pluto does not implement yet, e.g. `csp=`, `redirect=`).
fn parse_options(options: Option<&str>) -> ParsedOptions {
    let mut types: Option<u32> = None;
    let mut negated_types: u32 = 0;
    let mut third_party: Option<bool> = None;
    let mut domains = Vec::new();
    let mut important = false;
    let mut match_case = false;
    let Some(options) = options else {
        return ParsedOptions {
            types,
            third_party,
            domains,
            important,
            match_case,
        };
    };
    for raw in options.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let (negated, name) = match token.strip_prefix('~') {
            Some(n) => (true, n),
            None => (false, token),
        };
        if let Some(bit) = type_option_bit(name) {
            if negated {
                negated_types |= bit;
            } else {
                types = Some(types.unwrap_or(0) | bit);
            }
            continue;
        }
        match name {
            "third-party" | "3p" => third_party = Some(!negated),
            "important" => {
                if !negated {
                    important = true;
                }
            }
            "match-case" => {
                if !negated {
                    match_case = true;
                }
            }
            _ => {
                if let Some(value) = name.strip_prefix("domain=") {
                    for domain in value.split('|') {
                        let domain = domain.trim();
                        if domain.is_empty() {
                            continue;
                        }
                        let (dneg, dname) = match domain.strip_prefix('~') {
                            Some(d) => (true, d),
                            None => (false, domain),
                        };
                        domains.push(DomainRestriction {
                            domain: dname.to_lowercase(),
                            negated: dneg,
                        });
                    }
                }
                // Anything else (csp=, redirect=, badfilter, …)
                // is not implemented in Round 1: ignored.
            }
        }
    }
    // A rule with only negated types applies to every type
    // except the negated ones.
    if types.is_none() && negated_types != 0 {
        types = Some(T_ALL & !negated_types);
    } else if let Some(t) = types {
        types = Some(t & !negated_types);
    }
    ParsedOptions {
        types,
        third_party,
        domains,
        important,
        match_case,
    }
}

/// Parse a `||domain...` rule. Returns `None` when the domain
/// part is empty or malformed.
fn parse_domain_rule(pattern: &str, match_case: bool) -> Option<PatternKind> {
    let after = &pattern[2..];
    // The domain runs up to the first `/`, `^`, or `*`.
    let end = after
        .find(|c| c == '/' || c == '^' || c == '*')
        .unwrap_or(after.len());
    let domain = after[..end].to_lowercase();
    if domain.is_empty() {
        return None;
    }
    let rest = &after[end..];
    // A `*` in the domain means the rule is not a plain domain
    // rule — fall back to the generic matcher over the
    // host+path remainder.
    if domain.contains('*') {
        return None;
    }
    if rest.is_empty() || rest == "^" {
        return Some(PatternKind::Domain { domain });
    }
    let mut rest = rest;
    let mut end_anchor = false;
    if let Some(stripped) = rest.strip_suffix('|') {
        end_anchor = true;
        rest = stripped;
    }
    let mut tokens = tokenize(rest, match_case);
    if !end_anchor {
        tokens.push(Token::Star);
    }
    Some(PatternKind::DomainPrefix { domain, rest: tokens })
}

/// Parse any other pattern into a generic wildcard rule.
fn parse_generic_rule(pattern: &str, match_case: bool) -> Option<PatternKind> {
    let mut pattern = pattern;
    let mut start_anchor = false;
    let mut end_anchor = false;
    if let Some(stripped) = pattern.strip_prefix('|') {
        start_anchor = true;
        pattern = stripped;
    }
    if let Some(stripped) = pattern.strip_suffix('|') {
        end_anchor = true;
        pattern = stripped;
    }
    if pattern.is_empty() {
        return None;
    }
    let mut tokens = tokenize(pattern, match_case);
    if !start_anchor {
        tokens.insert(0, Token::Star);
    }
    if !end_anchor {
        tokens.push(Token::Star);
    }
    Some(PatternKind::Generic { tokens })
}

/// Tokenize a pattern fragment: `*` → Star, `^` → Sep, anything
/// else → Lit (lowercased unless `$match-case`).
fn tokenize(pattern: &str, match_case: bool) -> Vec<Token> {
    pattern
        .chars()
        .map(|c| match c {
            '*' => Token::Star,
            '^' => Token::Sep,
            _ => Token::Lit(if match_case { c } else { c.to_ascii_lowercase() }),
        })
        .collect()
}

// ---------------------------------------------------------------------
// List fetching
// ---------------------------------------------------------------------

enum FetchOutcome {
    NotModified,
    Updated {
        text: String,
        etag: Option<String>,
        last_modified: Option<String>,
    },
    Failed(String),
}

/// Fetch one list over HTTPS with conditional-request headers.
/// A 304 means the stored copy is current; anything else that is
/// not a 200 is a failure (the last-good copy stays in place).
async fn fetch_list(
    url: &str,
    etag: Option<&str>,
    last_modified: Option<&str>,
) -> FetchOutcome {
    let mut builder = Request::builder()
        .method("GET")
        .uri(url)
        .header(header::USER_AGENT, format!("Pluto/{}", env!("CARGO_PKG_VERSION")));
    let mut headers = HeaderMap::new();
    if let Some(etag) = etag {
        if let Ok(value) = header::HeaderValue::from_str(etag) {
            headers.insert(header::IF_NONE_MATCH, value);
        }
    }
    if let Some(last_modified) = last_modified {
        if let Ok(value) = header::HeaderValue::from_str(last_modified) {
            headers.insert(header::IF_MODIFIED_SINCE, value);
        }
    }
    for (name, value) in headers.iter() {
        builder = builder.header(name, value);
    }
    let Ok(request) = builder.body(Full::new(Bytes::new())) else {
        return FetchOutcome::Failed("could not build list request".to_string());
    };
    let response = match tokio::time::timeout(
        REFRESH_TIMEOUT,
        shared_client().request(request),
    )
    .await
    {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => return FetchOutcome::Failed(format!("list fetch failed: {e}")),
        Err(_) => {
            return FetchOutcome::Failed(format!(
                "list fetch timed out after {}s",
                REFRESH_TIMEOUT.as_secs()
            ));
        }
    };
    let status = response.status();
    if status == hyper::StatusCode::NOT_MODIFIED {
        return FetchOutcome::NotModified;
    }
    if !status.is_success() {
        return FetchOutcome::Failed(format!("list server returned {status}"));
    }
    if let Some(len) = response
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if len > MAX_LIST_BYTES {
            return FetchOutcome::Failed(format!("list body too large ({len} bytes)"));
        }
    }
    let etag = response
        .headers()
        .get(header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let last_modified = response
        .headers()
        .get(header::LAST_MODIFIED)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = match tokio::time::timeout(REFRESH_TIMEOUT, response.into_body().collect()).await {
        Ok(Ok(collected)) => collected.to_bytes(),
        Ok(Err(e)) => return FetchOutcome::Failed(format!("list body read failed: {e}")),
        Err(_) => return FetchOutcome::Failed("list body read timed out".to_string()),
    };
    if body.len() > MAX_LIST_BYTES {
        return FetchOutcome::Failed(format!("list body too large ({} bytes)", body.len()));
    }
    match String::from_utf8(body.to_vec()) {
        Ok(text) => FetchOutcome::Updated {
            text,
            etag,
            last_modified,
        },
        Err(_) => FetchOutcome::Failed("list body is not valid UTF-8".to_string()),
    }
}

/// Where fetched lists live on disk, for the finisher: the
/// profile dir (`$HOME/.config/pluto/`). Kept as a helper so
/// the browser-side wiring does not re-derive the path.
pub fn profile_list_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".config").join("pluto"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocker_with(rules: &str) -> Blocker {
        let mut b = Blocker::new();
        b.load_list(ListSource::EasyList, rules);
        b
    }

    fn blocked(
        b: &Blocker,
        url: &str,
        dest: Destination,
        page: &str,
    ) -> Option<MatchedRule> {
        b.is_blocked(url, dest, page).cloned()
    }

    const PAGE: &str = "https://news.example.com/article";

    // --- ||domain^ matching, incl. subdomains -----------------------

    #[test]
    fn domain_rule_matches_host_and_subdomains() {
        let b = blocker_with("||tracker.com^\n");
        assert!(blocked(&b, "https://tracker.com/pixel.gif", Destination::Image, PAGE).is_some());
        assert!(blocked(
            &b,
            "https://sub.tracker.com/pixel.gif",
            Destination::Image,
            PAGE
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://deep.sub.tracker.com/x.js",
            Destination::Script,
            PAGE
        )
        .is_some());
    }

    #[test]
    fn domain_rule_does_not_match_lookalike_hosts() {
        let b = blocker_with("||tracker.com^\n");
        assert!(blocked(&b, "https://nottracker.com/x.js", Destination::Script, PAGE).is_none());
        assert!(blocked(&b, "https://tracker.com.evil.com/x.js", Destination::Script, PAGE).is_none());
        assert!(blocked(&b, "https://example.com/x.js", Destination::Script, PAGE).is_none());
    }

    #[test]
    fn domain_rule_requires_separator_after_domain() {
        // `||tracker.com^` must not match a host where the
        // domain is followed by a letter (e.g. tracker.comx).
        let b = blocker_with("||tracker.com^\n");
        assert!(blocked(&b, "https://tracker.comx/x.js", Destination::Script, PAGE).is_none());
    }

    // --- wildcard / anchor / ^ handling ------------------------------

    #[test]
    fn bare_pattern_is_a_substring_match() {
        let b = blocker_with("/ads/banner\n");
        assert!(blocked(&b, "https://example.com/ads/banner.png", Destination::Image, PAGE).is_some());
        assert!(blocked(&b, "https://example.com/ads/banner", Destination::Image, PAGE).is_some());
        assert!(blocked(&b, "https://example.com/nope.png", Destination::Image, PAGE).is_none());
    }

    #[test]
    fn star_wildcard_matches_anything() {
        let b = blocker_with("*/tracker/*\n");
        assert!(blocked(&b, "https://a.com/x/tracker/y.js", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://a.com/tracker.js", Destination::Script, PAGE).is_none());
    }

    #[test]
    fn start_anchor_pins_to_url_start() {
        let b = blocker_with("|https://ads.example.com^\n");
        assert!(blocked(&b, "https://ads.example.com/x", Destination::Image, PAGE).is_some());
        assert!(blocked(&b, "https://evil.com/?u=https://ads.example.com/x", Destination::Image, PAGE).is_none());
    }

    #[test]
    fn end_anchor_pins_to_url_end() {
        let b = blocker_with("*.gif|\n");
        assert!(blocked(&b, "https://example.com/a.gif", Destination::Image, PAGE).is_some());
        assert!(blocked(&b, "https://example.com/a.gif?x=1", Destination::Image, PAGE).is_none());
    }

    #[test]
    fn separator_placeholder_matches_separator_or_end() {
        let b = blocker_with("example.com^foo\n");
        // `^` matches `/`.
        assert!(blocked(&b, "https://example.com/foo", Destination::Image, PAGE).is_some());
        // `^` matches `?` and `&`.
        assert!(blocked(&b, "https://example.com?foo", Destination::Image, PAGE).is_some());
        // `^` does not match a letter.
        assert!(blocked(&b, "https://example.comxfoo", Destination::Image, PAGE).is_none());
        // `^` matches end of string.
        let b2 = blocker_with("example.com^\n");
        assert!(blocked(&b2, "https://example.com", Destination::Image, PAGE).is_some());
    }

    #[test]
    fn domain_prefix_rule_matches_path_tail() {
        // Real EasyList shape: domain + literal path (no `^`
        // between domain and path — `||domain^/path` would be
        // dead in ABP too, since `^` already consumes the `/`).
        let b = blocker_with("||tracker.com/ads/\n");
        assert!(blocked(&b, "https://tracker.com/ads/banner.js", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://sub.tracker.com/ads/x", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://tracker.com/other/x", Destination::Script, PAGE).is_none());
        // Separator form: `^` consumes the `/` after the domain.
        let b2 = blocker_with("||tracker.com^ads/\n");
        assert!(blocked(&b2, "https://tracker.com/ads/banner.js", Destination::Script, PAGE).is_some());
        assert!(blocked(&b2, "https://tracker.com/other/x", Destination::Script, PAGE).is_none());
    }

    // --- $third-party -------------------------------------------------

    #[test]
    fn third_party_option_only_blocks_cross_site() {
        let b = blocker_with("||example.com^$third-party\n");
        // Same registrable domain as the page: first-party.
        assert!(blocked(
            &b,
            "https://cdn.example.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_none());
        // Different registrable domain: third-party.
        assert!(blocked(
            &b,
            "https://example.com/x.js",
            Destination::Script,
            "https://other.org/article"
        )
        .is_some());
    }

    #[test]
    fn negated_third_party_only_blocks_first_party() {
        let b = blocker_with("||example.com^$~third-party\n");
        assert!(blocked(
            &b,
            "https://example.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://example.com/x.js",
            Destination::Script,
            "https://other.org/article"
        )
        .is_none());
    }

    // --- $domain= restriction ----------------------------------------

    #[test]
    fn domain_option_restricts_by_page_domain() {
        let b = blocker_with("||tracker.com^$domain=news.example.com\n");
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://other.org/article"
        )
        .is_none());
        // Subdomains of the listed page domain count.
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://sub.news.example.com/article"
        )
        .is_some());
    }

    #[test]
    fn negated_domain_option_excludes_a_page() {
        let b = blocker_with("||tracker.com^$domain=~news.example.com\n");
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_none());
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://other.org/article"
        )
        .is_some());
    }

    // --- @@ exceptions, $important, $document -------------------------

    #[test]
    fn exception_overrides_block() {
        let b = blocker_with(
            "||tracker.com^\n@@||tracker.com^$image\n",
        );
        assert!(blocked(&b, "https://tracker.com/x.png", Destination::Image, PAGE).is_none());
        assert!(blocked(&b, "https://tracker.com/x.js", Destination::Script, PAGE).is_some());
    }

    #[test]
    fn exception_checked_before_block_regardless_of_order() {
        // The exception comes first in the list; the block must
        // still lose to it.
        let b = blocker_with(
            "@@||tracker.com^$image\n||tracker.com^\n",
        );
        assert!(blocked(&b, "https://tracker.com/x.png", Destination::Image, PAGE).is_none());
    }

    #[test]
    fn important_overrides_exception() {
        let b = blocker_with(
            "||tracker.com^$important\n@@||tracker.com^\n",
        );
        assert!(blocked(&b, "https://tracker.com/x.js", Destination::Script, PAGE).is_some());
    }

    #[test]
    fn document_exception_short_circuits_the_page() {
        let b = blocker_with(
            "||tracker.com^\n@@||news.example.com^$document\n",
        );
        // Every subresource on the excepted page is allowed…
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_none());
        // …but the same request on another page is blocked.
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://other.org/article"
        )
        .is_some());
    }

    #[test]
    fn important_block_beats_document_exception() {
        let b = blocker_with(
            "||tracker.com^$important\n@@||news.example.com^$document\n",
        );
        assert!(blocked(
            &b,
            "https://tracker.com/x.js",
            Destination::Script,
            "https://news.example.com/article"
        )
        .is_some());
    }

    #[test]
    fn blocking_document_rule_blocks_navigations_only() {
        let b = blocker_with("||ads.example.com^$document\n");
        assert!(blocked(
            &b,
            "https://ads.example.com/",
            Destination::Document,
            "https://other.org/"
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://ads.example.com/x.js",
            Destination::Script,
            PAGE
        )
        .is_none());
    }

    // --- regex rules ---------------------------------------------------

    #[test]
    fn regex_rule_matches() {
        let b = blocker_with("/^https:\\/\\/([^\\/]+\\.)?tracker\\.com\\//\n");
        assert!(blocked(&b, "https://tracker.com/x", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://sub.tracker.com/x", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://tracker.org/x", Destination::Script, PAGE).is_none());
    }

    #[test]
    fn regex_rule_with_options() {
        let b = blocker_with("/tracker/$script\n");
        assert!(blocked(&b, "https://example.com/tracker/x", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://example.com/tracker/x", Destination::Image, PAGE).is_none());
    }

    #[test]
    fn regex_cap_is_enforced() {
        // Zero-padded names so no capped rule's pattern is a
        // substring of an earlier rule's pattern (regex rules
        // are unanchored: `/t051/` would match a `t512` URL).
        let mut text = String::new();
        for i in 0..(MAX_REGEX_RULES + 10) {
            text.push_str(&format!("/t{i:03}/$script\n"));
        }
        let b = blocker_with(&text);
        assert_eq!(b.rule_count(), MAX_REGEX_RULES);
        assert_eq!(b.regex_skipped(), 10);
        // The first 512 still match; the 513th was ignored.
        assert!(blocked(
            &b,
            "https://example.com/t000",
            Destination::Script,
            PAGE
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://example.com/t511",
            Destination::Script,
            PAGE
        )
        .is_some());
        assert!(blocked(
            &b,
            "https://example.com/t512",
            Destination::Script,
            PAGE
        )
        .is_none());
    }

    // --- type masks -----------------------------------------------------

    #[test]
    fn type_mask_restricts_destinations() {
        let b = blocker_with("||tracker.com^$script,image\n");
        assert!(blocked(&b, "https://tracker.com/x.js", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://tracker.com/x.png", Destination::Image, PAGE).is_some());
        assert!(blocked(&b, "https://tracker.com/x.css", Destination::Stylesheet, PAGE).is_none());
        assert!(blocked(&b, "https://tracker.com/x", Destination::Document, PAGE).is_none());
    }

    #[test]
    fn negated_type_means_all_except() {
        let b = blocker_with("||tracker.com^$~script\n");
        assert!(blocked(&b, "https://tracker.com/x.js", Destination::Script, PAGE).is_none());
        assert!(blocked(&b, "https://tracker.com/x.png", Destination::Image, PAGE).is_some());
    }

    #[test]
    fn fetch_destination_maps_to_xmlhttprequest() {
        let b = blocker_with("||api.tracker.com^$xmlhttprequest\n");
        assert!(blocked(&b, "https://api.tracker.com/v1", Destination::Fetch, PAGE).is_some());
        assert!(blocked(&b, "https://api.tracker.com/v1", Destination::Script, PAGE).is_none());
        // The `xhr` alias means the same thing.
        let b2 = blocker_with("||api.tracker.com^$xhr\n");
        assert!(blocked(&b2, "https://api.tracker.com/v1", Destination::Fetch, PAGE).is_some());
    }

    #[test]
    fn no_type_options_matches_every_destination() {
        let b = blocker_with("||tracker.com^\n");
        for dest in [
            Destination::Document,
            Destination::Script,
            Destination::Image,
            Destination::Stylesheet,
            Destination::Font,
            Destination::Fetch,
            Destination::Other,
        ] {
            assert!(
                blocked(&b, "https://tracker.com/x", dest, PAGE).is_some(),
                "{dest:?} should match an untyped rule"
            );
        }
    }

    // --- list parsing ---------------------------------------------------

    #[test]
    fn parsing_ignores_comments_headers_and_cosmetics() {
        let b = blocker_with(
            "[Adblock Plus 2.0]\n\
             ! a comment\n\
             \n\
             ##.ad-banner\n\
             example.com##.sponsored\n\
             example.com##+js(abort-on-property-read, foo)\n\
             #@#.ad-allowed\n\
             ||tracker.com^\n",
        );
        assert_eq!(b.rule_count(), 1);
        assert!(blocked(&b, "https://tracker.com/x", Destination::Script, PAGE).is_some());
    }

    #[test]
    fn matched_rule_carries_source_and_line() {
        let b = blocker_with("! comment\n||tracker.com^\n@@||tracker.com^$image\n");
        let m = blocked(&b, "https://tracker.com/x.js", Destination::Script, PAGE).unwrap();
        assert_eq!(m.pattern, "||tracker.com^");
        assert_eq!(m.source, ListSource::EasyList);
        assert_eq!(m.line, 2);
        assert!(!m.exception);
        assert!(!m.important);
    }

    #[test]
    fn empty_blocker_allows_everything() {
        let b = Blocker::new();
        assert!(b.is_empty());
        assert!(b.is_blocked("https://tracker.com/x", Destination::Script, PAGE).is_none());
    }

    #[test]
    fn load_list_replaces_only_that_source() {
        let mut b = Blocker::new();
        b.load_list(ListSource::EasyList, "||a.com^\n");
        b.load_list(ListSource::EasyPrivacy, "||b.com^\n");
        assert_eq!(b.rule_count(), 2);
        b.load_list(ListSource::EasyList, "||c.com^\n");
        assert_eq!(b.rule_count(), 2);
        assert!(blocked(&b, "https://a.com/x", Destination::Script, PAGE).is_none());
        assert!(blocked(&b, "https://b.com/x", Destination::Script, PAGE).is_some());
        assert!(blocked(&b, "https://c.com/x", Destination::Script, PAGE).is_some());
    }

    #[test]
    fn unparseable_url_is_allowed() {
        let b = blocker_with("||tracker.com^\n");
        assert!(b.is_blocked("not a url", Destination::Script, PAGE).is_none());
    }

    #[test]
    fn load_from_dir_reads_persisted_lists() {
        let dir = std::env::temp_dir().join("pluto-blocker-test-load");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(EASYLIST_FILE), "||tracker.com^\n").unwrap();
        let b = Blocker::load_from_dir(&dir);
        assert_eq!(b.rule_count(), 1);
        assert!(blocked(&b, "https://tracker.com/x", Destination::Script, PAGE).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn registrable_domain_approximation() {
        assert_eq!(registrable_domain("sub.example.com"), "example.com");
        assert_eq!(registrable_domain("example.com"), "example.com");
        assert_eq!(registrable_domain("localhost"), "localhost");
        assert_eq!(registrable_domain("127.0.0.1"), "127.0.0.1");
    }

    #[test]
    fn concurrent_atomic_writes_never_corrupt() {
        // Stage 10 Rec 6, same writer as `pluto-browser`'s
        // profile files: two tracker-list refreshes (or two Pluto
        // windows) writing one list dir must each land a whole
        // file — the old fixed temp name made their writes
        // interleave (and renames fail outright when another
        // writer renamed the shared temp file first).
        let dir = std::env::temp_dir().join("pluto-blocker-test-atomic-race");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let docs: Vec<String> = (0..4)
            .map(|w| format!("||tracker{w}.com^\n{}", "x".repeat(512 + w * 997)))
            .collect();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
        let mut handles = Vec::new();
        for w in 0..4 {
            let dir = dir.clone();
            let doc = docs[w].clone();
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || -> std::io::Result<()> {
                barrier.wait();
                for _ in 0..25 {
                    write_file_atomic(&dir, "race.txt", &doc)?;
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().unwrap().expect("every concurrent write succeeds");
        }
        let text = std::fs::read_to_string(dir.join("race.txt")).unwrap();
        assert!(
            docs.contains(&text),
            "target holds exactly one writer's document, got {} bytes",
            text.len()
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "race.txt")
            .collect();
        assert!(leftovers.is_empty(), "no temp files left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
