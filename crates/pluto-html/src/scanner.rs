//! The preload scanner (R4): a scan-only mode of the SAME Pluto
//! tokenizer the main parser uses.
//!
//! While the main parser is blocked (script fetch, stylesheet wait, or
//! simply waiting for the next chunk), the scanner runs ahead over the
//! already-buffered bytes, recognises only fetchable attributes, and
//! hands URLs to the network stack so responses are already in flight
//! when the main parser reaches them. It builds no tree, matches no
//! CSS, executes nothing — so nothing it does is observable to the
//! page. It is an optimisation, not a specification: engines do not
//! scan identically, and it is not a correctness contract.
//!
//! The scanner shares two code paths with the main parser, so the two
//! cannot disagree: attribute values arrive entity-decoded from the
//! same Pluto `Tokenizer` (see `crate` docs on entities), and URL
//! resolution goes through the same [`resolve_url`] + first-`<base>`-
//! wins rule the streaming parser uses. Dedup is the Stage 5
//! scheduler's job — the scanner just emits; a speculative fetch of a
//! resource the parser later discards is wasted bandwidth, not a
//! correctness bug.
//!
//! # Blind spots (documented, not fixed here)
//!
//! * JS-injected URLs / `document.write` / DOM-injected tags —
//!   invisible until the injecting script runs.
//! * `data-src` (lazy-load libraries) — not a fetchable attribute; the
//!   scanner cannot know the library's convention.
//! * CSS `url()` — background images, `@font-face` `src` — discovered
//!   only when the CSSOM is built and the using element is styled.
//! * `@import` inside a stylesheet — the scanner reads the `<link>`,
//!   not the imported file; each import level costs a sequential
//!   round trip.
//! * `<template>` contents — not scanned (inert); the scanner tracks
//!   template depth and skips discoveries inside it.
//! * `srcset`/`sizes` — the scanner selects ONE candidate per
//!   `<img>` / `<picture>` (Stage 11 Rec 10: `w` descriptors
//!   against `sizes` + viewport, `x` against the DPR, `<source>`
//!   media/type first); it guesses the layout viewport when the
//!   caller does not pass one (`scan_with_viewport` takes the
//!   real one).
//! * `<base href>` appearing *after* resources it affects —
//!   base-resolution hazard; URLs seen before the `<base>` resolve
//!   against the document URL, and a later `<base>` does not rewrite
//!   them.
//! * Very large inline `<script>`/`<style>` blocks — the tokenizer
//!   treats their contents as raw text, so no false tag discoveries,
//!   but the scanner only sees what is buffered; a huge inline block
//!   early in the document delays everything after it.
//! * Client-rendered SPA shells — near-empty document; the scanner
//!   finds the bundle and then runs out of document.
//! * `<iframe src>` — deliberately NOT discovered (Stage 9 Rec 7,
//!   T12): iframes parse as raw text today, so the fetch would be a
//!   wasted request and a minor information leak. Re-add when
//!   iframes render.
//! * `<noscript>` contents are scanned as tags: the raw tokenizer has
//!   no scripting flag, so with scripting enabled the real parser
//!   treats noscript content as raw text while the scanner may
//!   over-fetch from it. Wasted bandwidth, not a correctness bug.

use crate::builder::{NS_HTML, NS_SVG};
use crate::tokenizer::{Token, Tokenizer};

/// What kind of resource a discovery is asking for.
///
/// This mirrors `pluto_ipc::Destination` variant-for-variant (same
/// names, same meaning) — it is a *local* mirror, not the ipc type,
/// because `pluto-html` cannot depend on `pluto-ipc`: that would
/// cycle (`pluto-html` → `pluto-ipc` → `pluto-paint` → `pluto-layout`
/// → `pluto-html`). The boundary maps this enum onto
/// `pluto_ipc::Destination` one-to-one (in `pluto-net` / the browser
/// worker, which can depend on both crates) when a discovery becomes
/// a `ResourceRequest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Destination {
    Document,
    Stylesheet,
    Script,
    Image,
    Font,
    Fetch,
    /// Stage 16 Phase 1, Rec 2: mirrors
    /// `pluto_ipc::Destination::Audio` / `Video` / `WebSocket`
    /// (see that enum's docs). `WebSocket` has no HTML producer
    /// — it exists so the mirror stays variant-for-variant.
    Audio,
    Video,
    WebSocket,
    Other,
}

/// Fetch priority, mirroring `pluto_ipc::Priority` (Chromium's five
/// bands). A hint, not a guarantee — bandwidth is still subject to
/// connection limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
}

/// One subresource the scanner (or the streaming parser) discovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredResource {
    /// Absolute URL, resolved against the document base (see
    /// [`resolve_url`]).
    pub url: String,
    pub destination: Destination,
    pub priority: Priority,
    /// Where it was found, e.g. `"img[src]"`, `"link[rel=stylesheet]"`.
    /// Debugging aid, not load-bearing — with one load-bearing
    /// exception: `"imageset"` (Stage 9 Rec 6, Mixed Content L2
    /// §3.1). A `srcset` / `<picture>` image candidate is labelled
    /// `"imageset"`, the L2 initiator name: unlike a plain
    /// `img[src]` it must BLOCK on a secure page, never
    /// auto-upgrade to https. The browser-side call site that turns
    /// a discovery into a `ResourceRequest` (`page.rs`,
    /// `subresource_request`) reads the label into
    /// `ResourceRequest.is_imageset`.
    pub from: &'static str,
    /// The element's `integrity` attribute (Stage 9 Rec 14, SRI),
    /// verbatim as written — the fetch layer parses it into hash
    /// entries and verifies the streamed bytes against them. Only
    /// `img` / `script` / `link` discoveries carry it (those are
    /// the elements the HTML spec gives integrity metadata to);
    /// every other discovery has `None`.
    pub integrity: Option<String>,
    /// Stage 12 Rec 1: the discovery is a `type="module"` script.
    /// Carried through from [`RawDiscovery::is_module`]; `false`
    /// for every other destination.
    pub is_module: bool,
}

/// Resolve `raw` against `base` (RFC 3986 join). Absolute URLs pass
/// through unchanged; unparseable bases fall back to a naive join
/// (origin + root-relative path, else base directory + relative
/// path). The same function resolves scanner discoveries and
/// streaming-parser resource events.
pub fn resolve_url(base: &str, raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return base.to_string();
    }
    if let Ok(abs) = url::Url::parse(raw) {
        return abs.to_string();
    }
    if let Ok(b) = url::Url::parse(base) {
        if let Ok(joined) = b.join(raw) {
            return joined.to_string();
        }
        return raw.to_string();
    }
    // No parseable base (e.g. `about:blank`): naive join.
    if raw.starts_with('/') {
        return raw.to_string();
    }
    match base.rfind('/') {
        Some(i) => format!("{}{}", &base[..i + 1], raw),
        None => raw.to_string(),
    }
}

/// A fetchable attribute on one element, before URL resolution:
/// `(raw url, destination, priority, from)`.
#[derive(Debug, Clone)]
pub struct RawDiscovery {
    pub url: String,
    pub destination: Destination,
    pub priority: Priority,
    pub from: &'static str,
    /// The element's `integrity` attribute, verbatim (Stage 9 Rec
    /// 14, SRI) — `Some` only for `img` / `script` / `link`
    /// elements, `None` otherwise.
    pub integrity: Option<String>,
    /// Stage 12 Rec 1: the discovery is a `type="module"` script.
    /// The loader pre-fetches a module's import graph; classic
    /// scripts never get the treatment. `false` for every other
    /// destination.
    pub is_module: bool,
}

fn attr<'a>(attrs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

/// `fetchpriority="high|low"` overrides the default priority for the
/// element; anything else (including `auto`) keeps the default.
fn fetch_priority(attrs: &[(String, String)], default: Priority) -> Priority {
    match attr(attrs, "fetchpriority") {
        Some("high") => Priority::High,
        Some("low") => Priority::Low,
        _ => default,
    }
}

/// Parse a `srcset` value into its candidate URLs (descriptors
/// discarded). Commas separate candidates; the URL is the first
/// whitespace-separated token of each candidate.
fn srcset_urls(srcset: &str) -> Vec<String> {
    parse_srcset(srcset)
        .into_iter()
        .map(|c| c.url)
        .collect()
}

/// One `srcset` candidate: its URL and its descriptor (HTML
/// Standard §4.8.4.3). `None` when the candidate carries no
/// descriptor (bare URL — valid only as the sole candidate, but
/// tolerated here as "pick me").
#[derive(Debug, Clone, PartialEq)]
pub struct SrcsetCandidate {
    pub url: String,
    pub descriptor: SrcsetDescriptor,
}

/// A `srcset` descriptor: `w` (intrinsic width in px) or `x`
/// (pixel density), or none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SrcsetDescriptor {
    Width(u32),
    Density(f32),
    None,
}

/// Parse a `srcset` value into candidates with descriptors
/// (Stage 11 Rec 10): commas separate candidates; each candidate
/// is a URL followed by an optional `Nw` or `Nx` descriptor.
/// Malformed candidates (empty URL, bad descriptor, mixed `w`
/// and `x` in one candidate) are dropped, never fatal.
pub fn parse_srcset(srcset: &str) -> Vec<SrcsetCandidate> {
    let mut out = Vec::new();
    for cand in srcset.split(',') {
        let mut parts = cand.split_whitespace();
        let Some(url) = parts.next().filter(|u| !u.is_empty()) else {
            continue;
        };
        if url.starts_with("data:") {
            continue;
        }
        let descriptor = match parts.next() {
            None => SrcsetDescriptor::None,
            Some(d) => {
                let d = d.trim_end_matches(',');
                if let Some(w) = d.strip_suffix('w')
                    && let Ok(n) = w.parse::<u32>()
                {
                    SrcsetDescriptor::Width(n)
                } else if let Some(x) = d.strip_suffix('x')
                    && let Ok(n) = x.parse::<f32>()
                    && n > 0.0
                {
                    SrcsetDescriptor::Density(n)
                } else {
                    // A bad descriptor invalidates the candidate.
                    continue;
                }
            }
        };
        out.push(SrcsetCandidate {
            url: url.to_string(),
            descriptor,
        });
    }
    out
}

/// Whether a simple media condition holds at `viewport_width`
/// (Stage 11 Rec 10 — the subset `sizes` / `<source media>`
/// need): `(max-width: Npx)` / `(min-width: Npx)` (px or em,
/// 1em = 16px), `screen` / `all` (true), `print` (false), and
/// `and` combinations. Anything else is unknown and does not
/// match — an unknown condition never selects a candidate.
pub fn media_matches(condition: &str, viewport_width: f32) -> bool {
    let condition = condition.trim();
    if condition.is_empty() {
        return true;
    }
    condition.split("and").all(|part| {
        let part = part.trim().trim_matches(|c| c == '(' || c == ')').trim();
        if part.is_empty() {
            return true;
        }
        if part.eq_ignore_ascii_case("screen") || part.eq_ignore_ascii_case("all") {
            return true;
        }
        if part.eq_ignore_ascii_case("print") {
            return false;
        }
        let Some((feature, value)) = part.split_once(':') else {
            return false;
        };
        let value = value.trim();
        let Ok(px) = media_length_px(value) else {
            return false;
        };
        match feature.trim().to_ascii_lowercase().as_str() {
            "max-width" => viewport_width <= px,
            "min-width" => viewport_width >= px,
            _ => false,
        }
    })
}

/// A media-query length in px: `Npx` or `Nem` (1em = 16px).
fn media_length_px(value: &str) -> Result<f32, ()> {
    let value = value.trim();
    if let Some(n) = value.strip_suffix("px") {
        n.trim().parse::<f32>().map_err(|_| ())
    } else if let Some(n) = value.strip_suffix("em") {
        n.trim().parse::<f32>().map(|v| v * 16.0).map_err(|_| ())
    } else {
        Err(())
    }
}

/// A `sizes` length in px: `Npx`, `Nvw` (percent of the viewport
/// width), `Nem` (16px each), or `N%` (percent of the viewport
/// width — `sizes` percentages resolve against the viewport).
fn sizes_length_px(value: &str, viewport_width: f32) -> Option<f32> {
    let value = value.trim();
    if let Some(n) = value.strip_suffix("px") {
        n.trim().parse::<f32>().ok()
    } else if let Some(n) = value.strip_suffix("vw") {
        n.trim().parse::<f32>().ok().map(|v| v / 100.0 * viewport_width)
    } else if let Some(n) = value.strip_suffix("em") {
        n.trim().parse::<f32>().ok().map(|v| v * 16.0)
    } else if let Some(n) = value.strip_suffix('%') {
        n.trim().parse::<f32>().ok().map(|v| v / 100.0 * viewport_width)
    } else {
        None
    }
}

/// Evaluate a `sizes` attribute to the source size in px (HTML
/// Standard §4.8.4.3): the first entry whose media condition
/// holds wins; the last entry (a bare length, no condition) is
/// the fallback. `None` sizes means `100vw`.
pub fn eval_sizes(sizes: Option<&str>, viewport_width: f32) -> f32 {
    let Some(sizes) = sizes.filter(|s| !s.trim().is_empty()) else {
        return viewport_width;
    };
    let entries: Vec<&str> = sizes.split(',').collect();
    for (i, entry) in entries.iter().enumerate() {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        if i + 1 == entries.len() {
            // Last entry: a bare length, no media condition.
            if let Some(px) = sizes_length_px(entry, viewport_width) {
                return px;
            }
            continue;
        }
        // `<media-condition> <length>`: the length is the last
        // whitespace-separated token.
        let mut parts = entry.rsplitn(2, |c: char| c.is_whitespace());
        let Some(length) = parts.next() else {
            continue;
        };
        let Some(condition) = parts.next() else {
            continue;
        };
        if media_matches(condition, viewport_width)
            && let Some(px) = sizes_length_px(length, viewport_width)
        {
            return px;
        }
    }
    viewport_width
}

/// Select the one `srcset` candidate to fetch (Stage 11 Rec 10):
/// `w` descriptors against the `sizes` source size (smallest
/// candidate at least as wide as the source, else the largest);
/// `x` descriptors against the device pixel ratio (smallest
/// density at least the DPR, else the largest); a bare candidate
/// is picked as-is. `None` when there is no candidate. Only the
/// selected URL is ever fetched — emitting every candidate was
/// the bandwidth bug this replaces.
pub fn select_srcset(
    candidates: &[SrcsetCandidate],
    sizes: Option<&str>,
    viewport_width: f32,
    dpr: f32,
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    // `w` descriptors win when any are present (mixed `w`/`x`
    // sets are invalid per spec; the `x` candidates are dropped).
    let widths: Vec<(u32, &str)> = candidates
        .iter()
        .filter_map(|c| match c.descriptor {
            SrcsetDescriptor::Width(w) => Some((w, c.url.as_str())),
            _ => None,
        })
        .collect();
    if !widths.is_empty() {
        let source_size = eval_sizes(sizes, viewport_width);
        let mut best: Option<(u32, &str)> = None;
        for (w, url) in &widths {
            if (*w as f32) >= source_size
                && best.is_none_or(|(bw, _)| *w < bw) {
                    best = Some((*w, url));
                }
        }
        if let Some((_, url)) = best {
            return Some(url.to_string());
        }
        // None wide enough: the largest candidate.
        return widths
            .iter()
            .max_by_key(|(w, _)| *w)
            .map(|(_, url)| url.to_string());
    }
    let densities: Vec<(f32, &str)> = candidates
        .iter()
        .filter_map(|c| match c.descriptor {
            SrcsetDescriptor::Density(x) => Some((x, c.url.as_str())),
            _ => None,
        })
        .collect();
    if !densities.is_empty() {
        let mut best: Option<(f32, &str)> = None;
        for (x, url) in &densities {
            if *x >= dpr && best.is_none_or(|(bx, _)| *x < bx) {
                best = Some((*x, url));
            }
        }
        if let Some((_, url)) = best {
            return Some(url.to_string());
        }
        return densities
            .iter()
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, url)| url.to_string());
    }
    // No descriptor at all: the (single) bare candidate.
    candidates.first().map(|c| c.url.clone())
}

/// One `<source>` inside a `<picture>` (Stage 11 Rec 10): its
/// `srcset` plus the `media` / `type` that decide whether it
/// applies. Raw attribute values, unresolved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PictureSource {
    pub srcset: Option<String>,
    pub media: Option<String>,
    pub type_: Option<String>,
}

/// Whether a `<source type>` is a format Pluto decodes (Stage 11
/// Rec 10): the `image` crate's formats, plus SVG (Stage 15, M7
/// Phase 3, Rec 12 — SVG bytes rasterise through the M7
/// rasteriser, so an SVG `<source>` is now a real candidate).
/// Parameters (`;...`) are stripped; an empty type always applies.
pub fn source_type_supported(type_: Option<&str>) -> bool {
    let Some(type_) = type_.filter(|t| !t.trim().is_empty()) else {
        return true;
    };
    let mime = type_.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    matches!(
        mime.as_str(),
        "image/png" | "image/jpeg" | "image/gif" | "image/webp" | "image/svg+xml"
    )
}

/// Select the image URL for an `<img>` (Stage 11 Rec 10, HTML
/// Standard §4.8.4): `<picture>` `<source>` media/type matching
/// first, in document order — the first matching source's srcset
/// selects; then the img's own srcset; then `img[src]`. Returns
/// the selected URL plus whether it came from a srcset / source
/// (an `imageset` initiator for the mixed-content gate) or from
/// plain `src`.
pub fn select_img_url(
    sources: &[PictureSource],
    img_srcset: Option<&str>,
    img_sizes: Option<&str>,
    img_src: Option<&str>,
    viewport_width: f32,
    dpr: f32,
) -> Option<(String, bool)> {
    for source in sources {
        let media_ok = source
            .media
            .as_deref()
            .is_none_or(|m| media_matches(m, viewport_width));
        if !media_ok || !source_type_supported(source.type_.as_deref()) {
            continue;
        }
        if let Some(srcset) = source.srcset.as_deref()
            && let Some(url) = select_srcset(
                &parse_srcset(srcset),
                img_sizes,
                viewport_width,
                dpr,
            )
        {
            return Some((url, true));
        }
    }
    if let Some(srcset) = img_srcset.filter(|s| !s.trim().is_empty())
        && let Some(url) = select_srcset(&parse_srcset(srcset), img_sizes, viewport_width, dpr)
    {
        return Some((url, true));
    }
    img_src
        .filter(|s| !s.trim().is_empty() && !s.trim_start().starts_with("data:"))
        .map(|s| (s.to_string(), false))
}

/// Whether a connection hint warms a hostname (Stage 11 Rec 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintKind {
    /// DNS + TCP + TLS warm, socket parked in the pool.
    Preconnect,
    /// DNS resolve only, no connection.
    DnsPrefetch,
}

/// One `preconnect` / `dns-prefetch` hint (Stage 11 Rec 6): a
/// `<link>` the scanner records but never fetches — the tab
/// thread acts on it (DNS-only for prefetch, full warm for
/// preconnect). Resolved against the document base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionHint {
    pub url: String,
    pub kind: HintKind,
}

/// Scan buffered HTML bytes for connection hints (`<link
/// rel="preconnect">` / `rel="dns-prefetch">`), resolved against
/// `document_url` (and any `<base href>` in the scanned prefix).
/// Hints are not fetches: the scanner never downloads them, and
/// neither does the preload path — the tab thread warms them
/// (Rec 6) after the blocker + HSTS gates.
pub fn scan_hints(bytes: &[u8], document_url: &str) -> Vec<ConnectionHint> {
    let text = String::from_utf8_lossy(bytes);
    let mut tokenizer = Tokenizer::new();
    tokenizer.feed_str(&text);
    tokenizer.set_eof();
    let mut hints = Vec::new();
    let mut base: Option<String> = None;
    while let Some(token) = tokenizer.next_token() {
        match token {
            Token::Eof => break,
            Token::StartTag(tag) => {
                if tag.name == "base" {
                    if base.is_none()
                        && let Some(href) = attr(&tag.attrs, "href")
                        && !href.trim().is_empty()
                    {
                        base = Some(resolve_url(document_url, href));
                    }
                    continue;
                }
                if tag.name != "link" {
                    continue;
                }
                let rel = attr(&tag.attrs, "rel").unwrap_or("");
                let rel_lc = rel.to_ascii_lowercase();
                let rels: Vec<&str> = rel_lc.split_whitespace().collect();
                let kind = if rels.contains(&"preconnect") {
                    HintKind::Preconnect
                } else if rels.contains(&"dns-prefetch") {
                    HintKind::DnsPrefetch
                } else {
                    continue;
                };
                if let Some(href) = attr(&tag.attrs, "href")
                    && !href.trim().is_empty()
                    && !href.trim_start().starts_with("data:")
                {
                    let b = base.clone().unwrap_or_else(|| document_url.to_string());
                    hints.push(ConnectionHint {
                        url: resolve_url(&b, href),
                        kind,
                    });
                }
            }
            _ => {}
        }
    }
    hints
}

/// Classify one element's fetchable attributes. Shared by the preload
/// scanner and the streaming parser (which classifies `ElementCreated`
/// events the same way), so the two can never disagree about what an
/// element fetches. `in_picture` / `media_tag` disambiguate `<source>`:
/// inside `<picture>` it is an image candidate, inside
/// `<video>`/`<audio>` it is media — `Destination::Video` or
/// `Destination::Audio` by the enclosing media tag (Stage 16 Phase
/// 1, Rec 2; before this it was `Destination::Other`, which capped
/// media at 10 MiB and gated it as blockable mixed content).
/// `media_tag` is `Some("video")` / `Some("audio")` for the
/// innermost enclosing media element, `None` outside media.
///
/// `viewport_width` / `dpr` drive `srcset` selection (Stage 11 Rec
/// 10): an `<img>` with a `srcset` emits exactly ONE discovery —
/// the selected candidate — never every candidate. An `<img>`
/// inside a `<picture>` emits NOTHING here: its candidates live on
/// sibling `<source>` elements, so the caller selects from them
/// ([`select_img_url`] / [`select_img_url_from_dom`]) and emits the
/// one selected URL itself.
///
/// `namespace` is the element's namespace URI (`""` = HTML; Stage 15,
/// M7 Phase 1, Rec 11): an SVG `<script src>` is never fetched — SVG
/// script is inert, so fetching it would only let a page make the
/// browser request a URL that can never execute.
///
/// Returns raw (unresolved) URLs; the caller resolves them against the
/// document base with [`resolve_url`].
pub fn classify_element(
    tag: &str,
    namespace: &str,
    attrs: &[(String, String)],
    in_picture: bool,
    media_tag: Option<&str>,
    viewport_width: f32,
    dpr: f32,
) -> Vec<RawDiscovery> {
    let mut out = Vec::new();
    let has = |name: &str| attr(attrs, name).is_some_and(|v| !v.trim().is_empty());
    // Stage 9 Rec 14 (SRI): the element's `integrity` attribute
    // rides every discovery the element makes. Only `img` /
    // `script` / `link` carry integrity metadata per the HTML
    // spec; every other element's discoveries get `None`.
    let integrity = if matches!(tag, "img" | "script" | "link") {
        attr(attrs, "integrity").map(|s| s.to_string())
    } else {
        None
    };
    match tag {
        "script" => {
            // Stage 15 (M7 Phase 1, Rec 11): only an HTML
            // `<script>` fetches. An SVG (or MathML) `<script>` is
            // inert — its `src` must never become a `Script`
            // discovery, or the page could make the browser request
            // a URL that can never execute.
            if (namespace.is_empty() || namespace == NS_HTML)
                && let Some(src) = attr(attrs, "src")
                    && !src.trim().is_empty()
                    && !src.trim_start().starts_with("data:")
                {
                    let is_module = attr(attrs, "type")
                        .is_some_and(|t| t.eq_ignore_ascii_case("module"));
                    let blocking = !has("async") && !has("defer") && !is_module;
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Script,
                        priority: fetch_priority(
                            attrs,
                            if blocking { Priority::High } else { Priority::Medium },
                        ),
                        from: "script[src]",
                        integrity: integrity.clone(),
                        is_module,
                    });
                }
        }
        "image" => {
            // Stage 15 (M7 Phase 3, Rec 10): `<svg><image>` fetches
            // like an `<img>` (same Stage 9 gates — the loader
            // treats a `Destination::Image` discovery the same way
            // regardless of which element made it). `href` is the
            // SVG2 spelling, `xlink:href` the legacy one; a
            // fragment-only href (`#id`) is not a fetch.
            if namespace == NS_SVG
                && let Some(href) = attr(attrs, "href")
                    .or_else(|| attr(attrs, "xlink:href"))
                    && !href.trim().is_empty()
                    && !href.trim_start().starts_with('#')
                    && !href.trim_start().starts_with("data:")
                {
                    out.push(RawDiscovery {
                        url: href.to_string(),
                        destination: Destination::Image,
                        priority: fetch_priority(attrs, Priority::Low),
                        from: "svg:image[href]",
                        integrity: None,
                        is_module: false,
                    });
                }
        }
        "img" => {
            // Stage 11 Rec 10: an <img> inside a <picture> emits
            // nothing here — the caller selects from the picture's
            // <source> siblings + this img (see the fn docs).
            // Outside a picture, the srcset selects exactly one
            // candidate; without a srcset the plain src is the
            // discovery.
            if in_picture {
                // Caller-selected; nothing to emit.
            } else if let Some(srcset) = attr(attrs, "srcset")
                && !srcset.trim().is_empty()
            {
                let sizes = attr(attrs, "sizes");
                if let Some(url) = select_srcset(
                    &parse_srcset(srcset),
                    sizes,
                    viewport_width,
                    dpr,
                ) {
                    out.push(RawDiscovery {
                        url,
                        destination: Destination::Image,
                        priority: fetch_priority(attrs, Priority::Low),
                        // Stage 9 Rec 6: a `srcset` candidate is
                        // an `imageset` initiator (Mixed Content
                        // L2 §3.1) — it blocks on a secure page,
                        // never auto-upgrades.
                        from: "imageset",
                        integrity: integrity.clone(),
                    is_module: false,
                    });
                } else if let Some(src) = attr(attrs, "src")
                    && !src.trim().is_empty() && !src.trim_start().starts_with("data:") {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Image,
                            priority: fetch_priority(attrs, Priority::Low),
                            from: "img[src]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
            } else if let Some(src) = attr(attrs, "src")
                && !src.trim().is_empty() && !src.trim_start().starts_with("data:") {
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Image,
                        priority: fetch_priority(attrs, Priority::Low),
                        from: "img[src]",
                        integrity: integrity.clone(),
                    is_module: false,
                    });
                }
        }
        "source" => {
            if media_tag.is_some() && !in_picture {
                if let Some(src) = attr(attrs, "src")
                    && !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: match media_tag {
                                Some("audio") => Destination::Audio,
                                _ => Destination::Video,
                            },
                            priority: Priority::Low,
                            from: "source[src]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
            } else if in_picture {
                // Stage 11 Rec 10: a `<source>` inside a
                // `<picture>` is consumed by the `<img>`'s
                // candidate selection (media/type matching, then
                // srcset selection) — it emits nothing on its own,
                // or every picture would fetch every source.
            } else {
                if let Some(src) = attr(attrs, "src")
                    && !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Image,
                            priority: fetch_priority(attrs, Priority::Low),
                            // Stage 9 Rec 6: a `<source>` inside a
                            // `<picture>` is an `imageset` initiator
                            // (L2 §3.1); outside one it is a plain
                            // image fetch.
                            from: "source[src]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
                if let Some(srcset) = attr(attrs, "srcset") {
                    for url in srcset_urls(srcset) {
                        out.push(RawDiscovery {
                            url,
                            destination: Destination::Image,
                            priority: fetch_priority(attrs, Priority::Low),
                            from: "imageset",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
                }
            }
        }
        "link" => {
            let rel = attr(attrs, "rel").unwrap_or("");
            let rel_lc = rel.to_ascii_lowercase();
            let rels: Vec<&str> = rel_lc.split_whitespace().collect();
            if let Some(href) = attr(attrs, "href")
                && !href.trim().is_empty() && !href.trim_start().starts_with("data:") {
                    if rels.contains(&"stylesheet") {
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: Destination::Stylesheet,
                            priority: fetch_priority(attrs, Priority::High),
                            from: "link[rel=stylesheet]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    } else if rels.contains(&"preload") {
                        let dest = match attr(attrs, "as") {
                            Some("script") => Destination::Script,
                            Some("style") => Destination::Stylesheet,
                            Some("image") => Destination::Image,
                            Some("font") => Destination::Font,
                            Some("fetch") => Destination::Fetch,
                            _ => Destination::Other,
                        };
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: dest,
                            priority: fetch_priority(attrs, Priority::High),
                            from: "link[rel=preload]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                        if let Some(srcset) = attr(attrs, "imagesrcset") {
                            // Stage 11 Rec 10: one selected
                            // candidate, not every candidate —
                            // `imagesizes` is this link's `sizes`.
                            let sizes = attr(attrs, "imagesizes");
                            if let Some(url) = select_srcset(
                                &parse_srcset(srcset),
                                sizes,
                                viewport_width,
                                dpr,
                            ) {
                                out.push(RawDiscovery {
                                    url,
                                    destination: Destination::Image,
                                    priority: fetch_priority(attrs, Priority::High),
                                    // Stage 9 Rec 6: a preload
                                    // `imagesrcset` candidate is an
                                    // `imageset` initiator (L2 §3.1).
                                    from: "imageset",
                                    integrity: integrity.clone(),
                                is_module: false,
                                });
                            }
                        }
                    } else if rels.contains(&"modulepreload") {
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: Destination::Script,
                            priority: fetch_priority(attrs, Priority::Medium),
                            from: "link[rel=modulepreload]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    } else if rels.contains(&"icon")
                        || rels.contains(&"apple-touch-icon")
                        || rels.contains(&"shortcut")
                    {
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: Destination::Image,
                            priority: Priority::VeryLow,
                            from: "link[rel=icon]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
                    // preconnect / dns-prefetch are not fetches; no
                    // discovery. manifest / alternate / canonical:
                    // not subresources the parser needs early.
                }
        }
        "video" => {
            // Stage 16 (M8 Phase 2, Rec 4): the element's own
            // `src` is a media discovery (preload warming — the
            // bytes land in the shared HTTP cache, and the
            // binding's authoritative fetch serves from it when
            // storable; the binding still owns selection + load,
            // so a non-storable response downloads twice — the
            // same trade-off images already make).
            if let Some(src) = attr(attrs, "src")
                && !src.trim().is_empty()
                && !src.trim_start().starts_with("data:")
            {
                out.push(RawDiscovery {
                    url: src.to_string(),
                    destination: Destination::Video,
                    priority: Priority::Low,
                    from: "video[src]",
                    integrity: integrity.clone(),
                    is_module: false,
                });
            }
            if let Some(poster) = attr(attrs, "poster")
                && !poster.trim().is_empty() && !poster.trim_start().starts_with("data:") {
                    out.push(RawDiscovery {
                        url: poster.to_string(),
                        destination: Destination::Image,
                        priority: Priority::Low,
                        from: "video[poster]",
                        integrity: integrity.clone(),
                    is_module: false,
                    });
                }
        }
        "audio" => {
            // Stage 16 (M8 Phase 2, Rec 4): as `video[src]`
            // above, for `<audio>`.
            if let Some(src) = attr(attrs, "src")
                && !src.trim().is_empty()
                && !src.trim_start().starts_with("data:")
            {
                out.push(RawDiscovery {
                    url: src.to_string(),
                    destination: Destination::Audio,
                    priority: Priority::Low,
                    from: "audio[src]",
                    integrity: integrity.clone(),
                    is_module: false,
                });
            }
        }
        // No `iframe[src]` discovery (Stage 9 Rec 7, T12): iframes
        // parse as raw text today — there is no nested browsing
        // context — so fetching the `src` is a wasted request and a
        // minor information leak (a page could make the browser fetch
        // arbitrary URLs whose bytes are never rendered). Re-add an
        // `iframe` arm here when iframes render, and only then.
        "embed" => {
            if let Some(src) = attr(attrs, "src")
                && !src.trim().is_empty() {
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Other,
                        priority: Priority::Low,
                        from: "embed[src]",
                        integrity: integrity.clone(),
                    is_module: false,
                    });
                }
        }
        "object" => {
            if let Some(data) = attr(attrs, "data")
                && !data.trim().is_empty() {
                    out.push(RawDiscovery {
                        url: data.to_string(),
                        destination: Destination::Other,
                        priority: Priority::Low,
                        from: "object[data]",
                        integrity: integrity.clone(),
                    is_module: false,
                    });
                }
        }
        "input" => {
            if attr(attrs, "type").is_some_and(|t| t.eq_ignore_ascii_case("image"))
                && let Some(src) = attr(attrs, "src")
                    && !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Image,
                            priority: Priority::Low,
                            from: "input[type=image][src]",
                            integrity: integrity.clone(),
                        is_module: false,
                        });
                    }
        }
        _ => {}
    }
    out
}

/// Select the image URL for an `<img>` node from the finished DOM
/// (Stage 11 Rec 10): the `<picture>` parent's `<source>` children
/// (media/type matching, in document order), then the img's own
/// `srcset`, then `src` — the same selection [`select_img_url`]
/// performs, read off the DOM instead of a tracked source list.
/// The streaming parser calls this when it creates an `<img>`,
/// so the parser path selects exactly one candidate too.
pub fn select_img_url_from_dom(
    dom: &crate::Dom,
    img: crate::NodeId,
    viewport_width: f32,
    dpr: f32,
) -> Option<(String, bool)> {
    let srcset = dom.attr(img, "srcset").map(str::to_string);
    let sizes = dom.attr(img, "sizes").map(str::to_string);
    let src = dom.attr(img, "src").map(str::to_string);
    let mut sources = Vec::new();
    if let Some(parent) = dom.get(img).and_then(|n| n.parent)
        && dom.tag(parent) == Some("picture")
    {
        for &child in dom.children(parent) {
            if dom.tag(child) == Some("source") {
                sources.push(PictureSource {
                    // A <source> with only `src` (no `srcset`)
                    // counts as a single bare candidate.
                    srcset: dom
                        .attr(child, "srcset")
                        .map(str::to_string)
                        .or_else(|| dom.attr(child, "src").map(str::to_string)),
                    media: dom.attr(child, "media").map(str::to_string),
                    type_: dom.attr(child, "type").map(str::to_string),
                });
            }
        }
    }
    select_img_url(
        &sources,
        srcset.as_deref(),
        sizes.as_deref(),
        src.as_deref(),
        viewport_width,
        dpr,
    )
}

/// Scan buffered HTML bytes for fetchable subresources, resolved
/// against `document_url` (and any `<base href>` in the scanned
/// prefix). Best-effort: undecodable bytes become U+FFFD (the
/// tokenizer's lossy path) and an incomplete trailing sequence is
/// ignored — the main parser re-reads the same bytes authoritatively.
///
/// The scanner emits in document order and does not dedup (the Stage
/// 5 scheduler dedups by URL + destination + epoch). It DOES select
/// (Stage 11 Rec 10): an `<img>` with a `srcset` / `<picture>`
/// emits exactly ONE discovery — the selected candidate — at the
/// default viewport (1024 px, DPR 1.0). Callers with the real
/// viewport use [`scan_with_viewport`].
pub fn scan(bytes: &[u8], document_url: &str) -> Vec<DiscoveredResource> {
    scan_with_viewport(bytes, document_url, 1024.0, 1.0)
}

/// Like [`scan`], with the layout viewport width (CSS px) and
/// device pixel ratio that `srcset` / `sizes` / `<picture>`
/// selection evaluates against. The preload path calls this with
/// the shell's real viewport, so a responsive image preloads the
/// candidate the page will actually use.
pub fn scan_with_viewport(
    bytes: &[u8],
    document_url: &str,
    viewport_width: f32,
    dpr: f32,
) -> Vec<DiscoveredResource> {
    let text = String::from_utf8_lossy(bytes);
    let mut tokenizer = Tokenizer::new();
    tokenizer.feed_str(&text);
    tokenizer.set_eof();
    let mut discoveries: Vec<DiscoveredResource> = Vec::new();
    let mut base: Option<String> = None;
    let mut template_depth: usize = 0;
    let mut picture_depth: usize = 0;
    let mut media_stack: Vec<String> = Vec::new();
    // Foreign-content tracking (Stage 15, M7 Phase 1, Rec 11): the
    // raw tokenizer has no tree, so the scanner tracks `<svg>`
    // depth itself to know when a `<script>` is foreign (inert —
    // never fetched). `<foreignObject>` re-enters HTML content,
    // so a script inside it IS an HTML script again. A
    // self-closing `<svg/>` opens nothing, so it does not count.
    // This is a heuristic pre-scan, not tree construction: the
    // streaming parser classifies from the real `Dom` namespace.
    let mut svg_depth: usize = 0;
    let mut foreign_object_depth: usize = 0;
    // The current <picture>'s <source> elements, in document
    // order (Stage 11 Rec 10): an <img> inside the picture
    // selects from these, so they are tracked, not emitted.
    let mut picture_sources: Vec<PictureSource> = Vec::new();
    let effective_base = |base: &Option<String>| -> String {
        base.clone().unwrap_or_else(|| document_url.to_string())
    };
    while let Some(token) = tokenizer.next_token() {
        match token {
            Token::Eof => break,
            Token::StartTag(tag) => {
                let name = tag.name.clone();
                match name.as_str() {
                    "template" => template_depth += 1,
                    "picture" => {
                        picture_depth += 1;
                        picture_sources.clear();
                    }
                    "video" | "audio" => media_stack.push(name.clone()),
                    "svg" if !tag.self_closing => svg_depth += 1,
                    // Tag names arrive lowercased from the tokenizer
                    // (`foreignObject` -> `foreignobject`).
                    "foreignobject" if svg_depth > 0 && !tag.self_closing => {
                        foreign_object_depth += 1
                    }
                    _ => {}
                }
                if name == "base" {
                    if base.is_none()
                        && let Some(href) = attr(&tag.attrs, "href")
                            && !href.trim().is_empty() {
                                base = Some(resolve_url(document_url, href));
                            }
                    continue;
                }
                if template_depth == 0 {
                    let in_picture = picture_depth > 0;
                    let media_tag = media_stack.last().map(String::as_str);
                    let b = effective_base(&base);
                    if name == "source" && in_picture {
                        // Tracked for the <img>'s selection, not
                        // emitted (see `classify_element`).
                        picture_sources.push(PictureSource {
                            srcset: attr(&tag.attrs, "srcset")
                                .map(str::to_string)
                                .or_else(|| attr(&tag.attrs, "src").map(str::to_string)),
                            media: attr(&tag.attrs, "media").map(str::to_string),
                            type_: attr(&tag.attrs, "type").map(str::to_string),
                        });
                        continue;
                    }
                    if name == "img" && in_picture {
                        // Stage 11 Rec 10: one selected candidate
                        // for the whole <picture> (source media /
                        // type first, then the img's own srcset,
                        // then src).
                        let integrity = attr(&tag.attrs, "integrity").map(str::to_string);
                        let selected = select_img_url(
                            &picture_sources,
                            attr(&tag.attrs, "srcset"),
                            attr(&tag.attrs, "sizes"),
                            attr(&tag.attrs, "src"),
                            viewport_width,
                            dpr,
                        );
                        if let Some((url, is_imageset)) = selected {
                            discoveries.push(DiscoveredResource {
                                url: resolve_url(&b, &url),
                                destination: Destination::Image,
                                priority: Priority::Low,
                                from: if is_imageset { "imageset" } else { "img[src]" },
                                integrity,
                                is_module: false,
                            });
                        }
                        continue;
                    }
                    for raw in classify_element(
                        &name,
                        // Foreign content (see the depth tracking
                        // above): an SVG `<script>` is inert.
                        if svg_depth > 0 && foreign_object_depth == 0 {
                            NS_SVG
                        } else {
                            ""
                        },
                        &tag.attrs,
                        in_picture,
                        media_tag,
                        viewport_width,
                        dpr,
                    ) {
                        discoveries.push(DiscoveredResource {
                            url: resolve_url(&b, &raw.url),
                            destination: raw.destination,
                            priority: raw.priority,
                            from: raw.from,
                            integrity: raw.integrity.clone(),
                            is_module: raw.is_module,
                        });
                    }
                }
            }
            Token::EndTag(tag) => match tag.name.as_str() {
                "template" => template_depth = template_depth.saturating_sub(1),
                "picture" => {
                    picture_depth = picture_depth.saturating_sub(1);
                    if picture_depth == 0 {
                        picture_sources.clear();
                    }
                }
                "video" | "audio" => {
                    media_stack.pop();
                }
                "svg" => svg_depth = svg_depth.saturating_sub(1),
                "foreignobject" => {
                    foreign_object_depth = foreign_object_depth.saturating_sub(1)
                }
                _ => {}
            },
            _ => {}
        }
    }
    discoveries
}

#[cfg(test)]
mod tests {
    use super::*;
    // NS_MATHML is used only by the tests below (the top-level
    // import was removed as unused by clippy); keep it here so the
    // lib target stays clean.
    use crate::builder::NS_MATHML;

    const DOC: &str = "https://example.com/dir/page.html";

    fn urls(found: &[DiscoveredResource]) -> Vec<&str> {
        found.iter().map(|d| d.url.as_str()).collect()
    }

    #[test]
    fn discovers_script_img_stylesheet() {
        let found = scan(
            br#"<link rel=stylesheet href="a.css"><script src="/s.js"></script><img src="i.png">"#,
            DOC,
        );
        assert_eq!(
            urls(&found),
            vec![
                "https://example.com/dir/a.css",
                "https://example.com/s.js",
                "https://example.com/dir/i.png",
            ]
        );
        assert_eq!(found[0].destination, Destination::Stylesheet);
        assert_eq!(found[0].priority, Priority::High);
        assert_eq!(found[1].destination, Destination::Script);
        assert_eq!(found[2].destination, Destination::Image);
        assert_eq!(found[2].priority, Priority::Low);
    }

    #[test]
    fn svg_script_src_is_not_discovered_but_html_script_is() {
        // Stage 15 (M7 Phase 1, Rec 11): an SVG <script> is inert
        // — its src must never become a Script discovery — while a
        // following HTML <script> still fetches.
        let found = scan(
            br#"<svg><script src="/svg.js"></script></svg><script src="/html.js"></script>"#,
            DOC,
        );
        assert_eq!(urls(&found), vec!["https://example.com/html.js"]);
        assert_eq!(found[0].destination, Destination::Script);
    }

    #[test]
    fn svg_image_href_is_discovered_as_image() {
        // Stage 15 (M7 Phase 3, Rec 10): `<svg><image href>` is
        // preloaded like an <img> (Destination::Image, same
        // gates); a fragment-only href is not a fetch, and
        // `xlink:href` works too.
        let found = scan(
            br##"<svg><image href="/a.png"/><image xlink:href="/b.png"/><image href="#frag"/></svg>"##,
            DOC,
        );
        assert_eq!(
            urls(&found),
            vec!["https://example.com/a.png", "https://example.com/b.png"]
        );
        assert!(found.iter().all(|d| d.destination == Destination::Image));
    }

    #[test]
    fn svg_script_in_foreign_object_is_discovered() {
        // Inside <foreignObject> content is HTML again, so a
        // <script> there IS an HTML script and fetches.
        let found = scan(
            br#"<svg><foreignObject><script src="/fo.js"></script></foreignObject></svg>"#,
            DOC,
        );
        assert_eq!(urls(&found), vec!["https://example.com/fo.js"]);
        assert_eq!(found[0].destination, Destination::Script);
    }

    #[test]
    fn classify_element_script_namespace_gate() {
        // Direct unit check of the classifier: same tag + attrs,
        // different namespace.
        let attrs = vec![("src".to_string(), "/s.js".to_string())];
        let svg = classify_element("script", NS_SVG, &attrs, false, None, 1024.0, 1.0);
        assert!(svg.is_empty(), "SVG script must classify to nothing");
        let html = classify_element("script", "", &attrs, false, None, 1024.0, 1.0);
        assert_eq!(html.len(), 1);
        assert_eq!(html[0].destination, Destination::Script);
        let mathml = classify_element("script", NS_MATHML, &attrs, false, None, 1024.0, 1.0);
        assert!(mathml.is_empty(), "MathML script must classify to nothing");
    }

    #[test]
    fn discovers_preload_srcset_poster() {
        // Stage 11 Rec 10: a srcset selects ONE candidate (dpr
        // 1.0 → the 1x), not every candidate.
        let found = scan(
            br#"<link rel=preload as=font href="f.woff2"><img srcset="a.png 1x, b.png 2x"><video poster="p.jpg"></video>"#,
            DOC,
        );
        assert_eq!(found[0].destination, Destination::Font);
        assert_eq!(found[0].url, "https://example.com/dir/f.woff2");
        assert_eq!(
            urls(&found[1..]),
            vec!["https://example.com/dir/a.png", "https://example.com/dir/p.jpg",]
        );
        assert_eq!(found[1].from, "imageset");
        assert_eq!(found[2].from, "video[poster]");
    }

    #[test]
    fn iframe_src_is_not_discovered() {
        // Stage 9 Rec 7 (T12): an <iframe src> must never become a
        // fetch — iframes parse as raw text today, so the bytes
        // would be fetched and discarded. Other subresources in the
        // same document still are.
        let found = scan(
            br#"<img src="i.png"><iframe src="https://evil.example.com/x"></iframe><script src="s.js"></script><link rel=stylesheet href="a.css">"#,
            DOC,
        );
        assert_eq!(
            urls(&found),
            vec![
                "https://example.com/dir/i.png",
                "https://example.com/dir/s.js",
                "https://example.com/dir/a.css",
            ]
        );
        assert!(
            found.iter().all(|d| d.from != "iframe[src]"),
            "no iframe[src] discovery: {found:?}"
        );
    }

    #[test]
    fn template_contents_are_skipped() {
        // Blind spot, by design: <template> is inert.
        let found = scan(
            br#"<template><img src="hidden.png"></template><img src="shown.png">"#,
            DOC,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/shown.png"]);
    }

    #[test]
    fn base_href_applies_after_it_appears() {
        // Base-resolution hazard: resources before the <base> resolve
        // against the document URL; later ones use the base.
        let found = scan(
            br#"<img src="a.png"><base href="https://cdn.example.com/x/"><img src="b.png">"#,
            DOC,
        );
        assert_eq!(
            urls(&found),
            vec![
                "https://example.com/dir/a.png",
                "https://cdn.example.com/x/b.png",
            ]
        );
    }

    #[test]
    fn entities_in_attribute_values_are_decoded() {
        // Attribute decoding is shared with the main parser: the
        // tokenizer decodes &amp; before the scanner ever sees it.
        let found = scan(br#"<img src="a.png?x=1&amp;y=2">"#, DOC);
        assert_eq!(urls(&found), vec!["https://example.com/dir/a.png?x=1&y=2"]);
    }

    #[test]
    fn data_urls_and_empty_src_are_ignored() {
        let found = scan(
            br#"<img src="data:image/png;base64,AAA"><img src=""><script src="s.js"></script>"#,
            DOC,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/s.js"]);
    }

    #[test]
    fn source_in_picture_is_selected_by_the_img_in_video_is_other() {
        // Stage 11 Rec 10: a <picture> <source> emits nothing on
        // its own — the <img> selects from it. A <source> in a
        // <video> is Video (Stage 16 Phase 1, Rec 2: before this
        // it was `Other`, which capped media at 10 MiB).
        let found = scan(
            br#"<picture><source srcset="p.webp"><img src="fallback.png"></picture><video><source src="v.mp4"></video><audio><source src="a.mp3"></audio>"#,
            DOC,
        );
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0].destination, Destination::Image);
        assert_eq!(found[0].url, "https://example.com/dir/p.webp");
        assert_eq!(found[0].from, "imageset");
        assert_eq!(found[1].destination, Destination::Video);
        assert_eq!(found[1].url, "https://example.com/dir/v.mp4");
        assert_eq!(found[2].destination, Destination::Audio);
        assert_eq!(found[2].url, "https://example.com/dir/a.mp3");
        // And a <picture> with no <img> selects nothing.
        let found = scan(br#"<picture><source srcset="p.webp"></picture>"#, DOC);
        assert!(found.is_empty(), "{found:?}");
    }

    #[test]
    fn media_element_own_src_is_discovered() {
        // Stage 16 (M8 Phase 2, Rec 4): a `<video>` / `<audio>`
        // element's own `src` is a media discovery (before this
        // only the poster was discovered and the `src` fetched
        // nothing at all); a `data:` src is not a fetch.
        let found = scan(
            br#"<video src="v.mp4" poster="p.png"></video><audio src="a.mp3"></audio><video src="data:video/mp4;base64,AAA"></video>"#,
            DOC,
        );
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(found[0].destination, Destination::Video);
        assert_eq!(found[0].url, "https://example.com/dir/v.mp4");
        assert_eq!(found[0].from, "video[src]");
        assert_eq!(found[1].destination, Destination::Image);
        assert_eq!(found[1].url, "https://example.com/dir/p.png");
        assert_eq!(found[1].from, "video[poster]");
        assert_eq!(found[2].destination, Destination::Audio);
        assert_eq!(found[2].url, "https://example.com/dir/a.mp3");
        assert_eq!(found[2].from, "audio[src]");
    }

    #[test]
    fn srcset_and_picture_candidates_are_labelled_imageset() {
        // Stage 9 Rec 6 (Mixed Content L2 §3.1): `srcset` /
        // `<picture>` image candidates carry the `imageset`
        // initiator label, so the mixed-content gate can block
        // them instead of auto-upgrading. Plain `img[src]` keeps
        // its own label (and keeps auto-upgrading). Stage 11 Rec
        // 10: each srcset selects ONE candidate, and a <picture>
        // with no <img> selects nothing.
        let found = scan(
            br#"<img src="plain.png"><img srcset="a.png 1x, b.png 2x"><picture><source src="p.jpg"><source srcset="q.webp"></picture><link rel="preload" as="image" href="h.png" imagesrcset="i.png 1x">"#,
            DOC,
        );
        let labels: Vec<&str> = found.iter().map(|d| d.from).collect();
        assert_eq!(
            labels,
            vec!["img[src]", "imageset", "link[rel=preload]", "imageset",],
            "srcset/picture candidates are imageset: {found:?}"
        );
        assert_eq!(
            urls(&found),
            vec![
                "https://example.com/dir/plain.png",
                "https://example.com/dir/a.png",
                "https://example.com/dir/h.png",
                "https://example.com/dir/i.png",
            ]
        );
        // And a `<source>` outside a `<picture>` is not.
        let found = scan(br#"<source src="s.jpg">"#, DOC);
        assert_eq!(found[0].from, "source[src]");
    }

    #[test]
    fn integrity_attribute_lands_on_discoveries() {
        // Stage 9 Rec 14 (SRI): the `integrity` attribute on an
        // img / script / link rides its discovery verbatim.
        let found = scan(
            br#"<script src="s.js" integrity="sha256-abc"></script><img src="i.png" integrity="sha384-def"><link rel=stylesheet href="a.css" integrity="sha512-ghi">"#,
            DOC,
        );
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].integrity.as_deref(), Some("sha256-abc"));
        assert_eq!(found[0].from, "script[src]");
        assert_eq!(found[1].integrity.as_deref(), Some("sha384-def"));
        assert_eq!(found[1].from, "img[src]");
        assert_eq!(found[2].integrity.as_deref(), Some("sha512-ghi"));
        assert_eq!(found[2].from, "link[rel=stylesheet]");
        // A srcset candidate shares its <img>'s integrity value
        // (and only the selected candidate is emitted).
        let found = scan(
            br#"<img srcset="a.png 1x, b.png 2x" integrity="sha256-xyz">"#,
            DOC,
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].url, "https://example.com/dir/a.png");
        assert_eq!(found[0].integrity.as_deref(), Some("sha256-xyz"));
        assert_eq!(found[0].from, "imageset");
    }

    #[test]
    fn integrity_absent_without_the_attribute() {
        // No `integrity` attribute: None, not an empty string.
        let found = scan(
            br#"<script src="s.js"></script><img src="i.png"><link rel=stylesheet href="a.css">"#,
            DOC,
        );
        assert_eq!(found.len(), 3);
        assert!(found.iter().all(|d| d.integrity.is_none()));
        // Elements that never carry integrity metadata per the
        // HTML spec don't get it even when the attribute is present.
        let found = scan(
            br#"<video poster="p.jpg" integrity="sha256-abc"></video><embed src="e.swf" integrity="sha256-abc"><source src="s.jpg" integrity="sha256-abc">"#,
            DOC,
        );
        assert!(!found.is_empty());
        assert!(found.iter().all(|d| d.integrity.is_none()), "{found:?}");
    }

    #[test]
    fn resolve_url_shapes() {
        assert_eq!(
            resolve_url("https://example.com/dir/page.html", "a.css"),
            "https://example.com/dir/a.css"
        );
        assert_eq!(
            resolve_url("https://example.com/dir/page.html", "/abs.css"),
            "https://example.com/abs.css"
        );
        assert_eq!(
            resolve_url("https://example.com/dir/page.html", "https://cdn.example.com/x.css"),
            "https://cdn.example.com/x.css"
        );
        assert_eq!(
            resolve_url("https://example.com/dir/page.html", "../up.css"),
            "https://example.com/up.css"
        );
    }

    // Stage 11 Rec 10: srcset / sizes / <picture> selection.

    #[test]
    fn srcset_w_selection_uses_sizes_and_viewport() {
        // sizes="(max-width: 600px) 400px, 800px": at a 500px
        // viewport the source size is 400px → the smallest
        // candidate ≥ 400px (800w); at 1024px the source is
        // 800px → the 800w candidate exactly.
        let found = scan_with_viewport(
            br#"<img srcset="small.png 200w, medium.png 800w, large.png 1600w" sizes="(max-width: 600px) 400px, 800px">"#,
            DOC,
            500.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/medium.png"]);
        let found = scan_with_viewport(
            br#"<img srcset="small.png 200w, medium.png 800w, large.png 1600w" sizes="(max-width: 600px) 400px, 800px">"#,
            DOC,
            1024.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/medium.png"]);
        // No candidate wide enough: the largest.
        let found = scan_with_viewport(
            br#"<img srcset="small.png 200w, medium.png 800w" sizes="2000px">"#,
            DOC,
            1024.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/medium.png"]);
        // No sizes: 100vw.
        let found = scan_with_viewport(
            br#"<img srcset="small.png 200w, medium.png 800w, large.png 1600w">"#,
            DOC,
            500.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/medium.png"]);
    }

    #[test]
    fn srcset_x_selection_uses_device_pixel_ratio() {
        let html = br#"<img srcset="a.png 1x, b.png 2x, c.png 3x">"#;
        // DPR 1.0 → 1x; DPR 2.0 → 2x; DPR 2.5 → 3x (smallest ≥ dpr).
        assert_eq!(
            urls(&scan_with_viewport(html, DOC, 1024.0, 1.0)),
            vec!["https://example.com/dir/a.png"]
        );
        assert_eq!(
            urls(&scan_with_viewport(html, DOC, 1024.0, 2.0)),
            vec!["https://example.com/dir/b.png"]
        );
        assert_eq!(
            urls(&scan_with_viewport(html, DOC, 1024.0, 2.5)),
            vec!["https://example.com/dir/c.png"]
        );
        // DPR past the largest: the largest.
        assert_eq!(
            urls(&scan_with_viewport(html, DOC, 1024.0, 4.0)),
            vec!["https://example.com/dir/c.png"]
        );
    }

    #[test]
    fn picture_source_media_and_type_matching() {
        // The first source whose media matches AND whose type is
        // supported wins; a WebP source is picked on this build.
        // (Stage 15, M7 Phase 3, Rec 12: an SVG source is a
        // candidate too — see the tail of this test.)
        let found = scan_with_viewport(
            br#"<picture><source media="(max-width: 600px)" srcset="small.webp" type="image/webp"><source srcset="big.png"><img src="fallback.png"></picture>"#,
            DOC,
            500.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/small.webp"]);
        // At a wide viewport the first source's media fails, so
        // the second source wins.
        let found = scan_with_viewport(
            br#"<picture><source media="(max-width: 600px)" srcset="small.webp" type="image/webp"><source srcset="big.png"><img src="fallback.png"></picture>"#,
            DOC,
            1024.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/big.png"]);
        // An unsupported type is skipped even when its media holds.
        let found = scan_with_viewport(
            br#"<picture><source srcset="vector.bmp" type="image/bmp"><img src="fallback.png"></picture>"#,
            DOC,
            1024.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/fallback.png"]);
        // Stage 15 (M7 Phase 3, Rec 12): an SVG <source> IS a
        // candidate now (the M7 rasteriser decodes it) — with no
        // media gate it wins over the img fallback.
        let found = scan_with_viewport(
            br#"<picture><source srcset="vector.svg" type="image/svg+xml"><img src="fallback.png"></picture>"#,
            DOC,
            1024.0,
            1.0,
        );
        assert_eq!(urls(&found), vec!["https://example.com/dir/vector.svg"]);
        assert!(source_type_supported(Some("image/svg+xml")));
        assert!(!source_type_supported(Some("image/bmp")));
    }

    #[test]
    fn select_img_url_from_dom_reads_picture_sources() {
        // The DOM path (streaming parser): same selection, read
        // off the finished DOM.
        let dom = crate::parse(
            r#"<picture><source media="(max-width: 600px)" srcset="small.webp"><img src="fallback.png"></picture>"#,
        );
        let img = {
            let mut found = None;
            let mut stack = vec![dom.root().unwrap()];
            while let Some(id) = stack.pop() {
                if dom.tag(id) == Some("img") {
                    found = Some(id);
                    break;
                }
                stack.extend(dom.children(id).iter().rev());
            }
            found.expect("img in DOM")
        };
        assert_eq!(
            select_img_url_from_dom(&dom, img, 500.0, 1.0),
            Some(("small.webp".to_string(), true))
        );
        assert_eq!(
            select_img_url_from_dom(&dom, img, 1024.0, 1.0),
            Some(("fallback.png".to_string(), false))
        );
    }

    // Stage 11 Rec 6: connection hints.

    #[test]
    fn scan_hints_finds_preconnect_and_dns_prefetch() {
        let hints = scan_hints(
            br#"<link rel="preconnect" href="https://cdn.example.com"><link rel="dns-prefetch" href="//fonts.example.com"><link rel="stylesheet" href="a.css">"#,
            DOC,
        );
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].kind, HintKind::Preconnect);
        assert_eq!(hints[0].url, "https://cdn.example.com/");
        assert_eq!(hints[1].kind, HintKind::DnsPrefetch);
        assert_eq!(hints[1].url, "https://fonts.example.com/");
        // No hints on a page without them — and a stylesheet is
        // not a hint.
        assert!(scan_hints(br#"<link rel="stylesheet" href="a.css">"#, DOC).is_empty());
    }
}
