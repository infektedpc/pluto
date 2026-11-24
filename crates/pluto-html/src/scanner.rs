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
//! * `srcset`/`sizes` — the scanner emits every `srcset` candidate
//!   (it must guess the layout viewport before layout); candidate
//!   selection by `sizes` is Stage 5 / scheduler work.
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
//! * `<noscript>` contents are scanned as tags: the raw tokenizer has
//!   no scripting flag, so with scripting enabled the real parser
//!   treats noscript content as raw text while the scanner may
//!   over-fetch from it. Wasted bandwidth, not a correctness bug.

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
    /// Where it was found, e.g. `"img[src]"`, `"link[rel=stylesheet]"`,
    /// `"img[srcset]"`. Debugging aid, not load-bearing.
    pub from: &'static str,
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
    srcset
        .split(',')
        .filter_map(|cand| {
            let url = cand.split_whitespace().next().unwrap_or("").trim();
            if url.is_empty() || url.starts_with("data:") {
                None
            } else {
                Some(url.to_string())
            }
        })
        .collect()
}

/// Classify one element's fetchable attributes. Shared by the preload
/// scanner and the streaming parser (which classifies `ElementCreated`
/// events the same way), so the two can never disagree about what an
/// element fetches. `in_picture` / `in_media` disambiguate `<source>`:
/// inside `<picture>` it is an image candidate, inside
/// `<video>`/`<audio>` it is media (`Destination::Other` — there is no
/// audio/video destination in the v1 protocol).
///
/// Returns raw (unresolved) URLs; the caller resolves them against the
/// document base with [`resolve_url`].
pub fn classify_element(
    tag: &str,
    attrs: &[(String, String)],
    in_picture: bool,
    in_media: bool,
) -> Vec<RawDiscovery> {
    let mut out = Vec::new();
    let has = |name: &str| attr(attrs, name).is_some_and(|v| !v.trim().is_empty());
    match tag {
        "script" => {
            if let Some(src) = attr(attrs, "src") {
                if !src.trim().is_empty() && !src.trim_start().starts_with("data:") {
                    let blocking = !has("async") && !has("defer")
                        && !matches!(attr(attrs, "type"), Some("module"));
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Script,
                        priority: fetch_priority(
                            attrs,
                            if blocking { Priority::High } else { Priority::Medium },
                        ),
                        from: "script[src]",
                    });
                }
            }
        }
        "img" => {
            if let Some(src) = attr(attrs, "src") {
                if !src.trim().is_empty() && !src.trim_start().starts_with("data:") {
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Image,
                        priority: fetch_priority(attrs, Priority::Low),
                        from: "img[src]",
                    });
                }
            }
            if let Some(srcset) = attr(attrs, "srcset") {
                for url in srcset_urls(srcset) {
                    out.push(RawDiscovery {
                        url,
                        destination: Destination::Image,
                        priority: fetch_priority(attrs, Priority::Low),
                        from: "img[srcset]",
                    });
                }
            }
        }
        "source" => {
            if in_media && !in_picture {
                if let Some(src) = attr(attrs, "src") {
                    if !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Other,
                            priority: Priority::Low,
                            from: "source[src]",
                        });
                    }
                }
            } else {
                if let Some(src) = attr(attrs, "src") {
                    if !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Image,
                            priority: fetch_priority(attrs, Priority::Low),
                            from: "source[src]",
                        });
                    }
                }
                if let Some(srcset) = attr(attrs, "srcset") {
                    for url in srcset_urls(srcset) {
                        out.push(RawDiscovery {
                            url,
                            destination: Destination::Image,
                            priority: fetch_priority(attrs, Priority::Low),
                            from: "source[srcset]",
                        });
                    }
                }
            }
        }
        "link" => {
            let rel = attr(attrs, "rel").unwrap_or("");
            let rel_lc = rel.to_ascii_lowercase();
            let rels: Vec<&str> = rel_lc.split_whitespace().collect();
            if let Some(href) = attr(attrs, "href") {
                if !href.trim().is_empty() && !href.trim_start().starts_with("data:") {
                    if rels.contains(&"stylesheet") {
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: Destination::Stylesheet,
                            priority: fetch_priority(attrs, Priority::High),
                            from: "link[rel=stylesheet]",
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
                        });
                        if let Some(srcset) = attr(attrs, "imagesrcset") {
                            for url in srcset_urls(srcset) {
                                out.push(RawDiscovery {
                                    url,
                                    destination: Destination::Image,
                                    priority: fetch_priority(attrs, Priority::High),
                                    from: "link[imagesrcset]",
                                });
                            }
                        }
                    } else if rels.contains(&"modulepreload") {
                        out.push(RawDiscovery {
                            url: href.to_string(),
                            destination: Destination::Script,
                            priority: fetch_priority(attrs, Priority::Medium),
                            from: "link[rel=modulepreload]",
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
                        });
                    }
                    // preconnect / dns-prefetch are not fetches; no
                    // discovery. manifest / alternate / canonical:
                    // not subresources the parser needs early.
                }
            }
        }
        "video" => {
            if let Some(poster) = attr(attrs, "poster") {
                if !poster.trim().is_empty() && !poster.trim_start().starts_with("data:") {
                    out.push(RawDiscovery {
                        url: poster.to_string(),
                        destination: Destination::Image,
                        priority: Priority::Low,
                        from: "video[poster]",
                    });
                }
            }
        }
        "iframe" => {
            if let Some(src) = attr(attrs, "src") {
                if !src.trim().is_empty() && !src.trim_start().starts_with("data:") {
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Other,
                        priority: Priority::Low,
                        from: "iframe[src]",
                    });
                }
            }
        }
        "embed" => {
            if let Some(src) = attr(attrs, "src") {
                if !src.trim().is_empty() {
                    out.push(RawDiscovery {
                        url: src.to_string(),
                        destination: Destination::Other,
                        priority: Priority::Low,
                        from: "embed[src]",
                    });
                }
            }
        }
        "object" => {
            if let Some(data) = attr(attrs, "data") {
                if !data.trim().is_empty() {
                    out.push(RawDiscovery {
                        url: data.to_string(),
                        destination: Destination::Other,
                        priority: Priority::Low,
                        from: "object[data]",
                    });
                }
            }
        }
        "input" => {
            if attr(attrs, "type").is_some_and(|t| t.eq_ignore_ascii_case("image")) {
                if let Some(src) = attr(attrs, "src") {
                    if !src.trim().is_empty() {
                        out.push(RawDiscovery {
                            url: src.to_string(),
                            destination: Destination::Image,
                            priority: Priority::Low,
                            from: "input[type=image][src]",
                        });
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// Scan buffered HTML bytes for fetchable subresources, resolved
/// against `document_url` (and any `<base href>` in the scanned
/// prefix). Best-effort: undecodable bytes become U+FFFD (the
/// tokenizer's lossy path) and an incomplete trailing sequence is
/// ignored — the main parser re-reads the same bytes authoritatively.
///
/// The scanner emits in document order and does not dedup (the Stage
/// 5 scheduler dedups by URL + destination + epoch).
pub fn scan(bytes: &[u8], document_url: &str) -> Vec<DiscoveredResource> {
    let text = String::from_utf8_lossy(bytes);
    let mut tokenizer = Tokenizer::new();
    tokenizer.feed_str(&text);
    tokenizer.set_eof();
    let mut discoveries: Vec<DiscoveredResource> = Vec::new();
    let mut base: Option<String> = None;
    let mut template_depth: usize = 0;
    let mut picture_depth: usize = 0;
    let mut media_depth: usize = 0;
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
                    "picture" => picture_depth += 1,
                    "video" | "audio" => media_depth += 1,
                    _ => {}
                }
                if name == "base" {
                    if base.is_none() {
                        if let Some(href) = attr(&tag.attrs, "href") {
                            if !href.trim().is_empty() {
                                base = Some(resolve_url(document_url, href));
                            }
                        }
                    }
                    continue;
                }
                if template_depth == 0 {
                    let in_picture = picture_depth > 0;
                    let in_media = media_depth > 0;
                    let b = effective_base(&base);
                    for raw in classify_element(&name, &tag.attrs, in_picture, in_media) {
                        discoveries.push(DiscoveredResource {
                            url: resolve_url(&b, &raw.url),
                            destination: raw.destination,
                            priority: raw.priority,
                            from: raw.from,
                        });
                    }
                }
            }
            Token::EndTag(tag) => match tag.name.as_str() {
                "template" => template_depth = template_depth.saturating_sub(1),
                "picture" => picture_depth = picture_depth.saturating_sub(1),
                "video" | "audio" => media_depth = media_depth.saturating_sub(1),
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
    fn discovers_preload_srcset_poster() {
        let found = scan(
            br#"<link rel=preload as=font href="f.woff2"><img srcset="a.png 1x, b.png 2x"><video poster="p.jpg"></video>"#,
            DOC,
        );
        assert_eq!(found[0].destination, Destination::Font);
        assert_eq!(found[0].url, "https://example.com/dir/f.woff2");
        assert_eq!(
            urls(&found[1..]),
            vec![
                "https://example.com/dir/a.png",
                "https://example.com/dir/b.png",
                "https://example.com/dir/p.jpg",
            ]
        );
        assert_eq!(found[3].from, "video[poster]");
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
    fn source_in_picture_is_image_in_video_is_other() {
        let found = scan(
            br#"<picture><source srcset="p.webp"></picture><video><source src="v.mp4"></video>"#,
            DOC,
        );
        assert_eq!(found[0].destination, Destination::Image);
        assert_eq!(found[1].destination, Destination::Other);
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
}
