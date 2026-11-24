//! Streaming / incremental parsing (R3): drive Pluto's own
//! `Tokenizer` + `TreeBuilder` directly, one `feed` at a time.
//!
//! The builder is pull-driven token by token, so `feed` can stop
//! at a parser-blocking `<script>` (the tokens after it stay
//! un-consumed in the tokenizer's buffer) and resume on the next
//! feed. Encoding is UTF-8 lossy as fed; a `<meta charset>` label
//! surfaces as an [`DomEvent::EncodingIndicator`] and the caller
//! re-parses from the buffered prefix if it wants to honour it
//! (the `marked`-crate pattern — Pluto buffers every fed byte in
//! [`Parser::buffered_prefix`]).
//!
//! # `document.write` (Stage 7 seam — reserved, not implemented)
//!
//! A classic script may call `document.write()`, which injects raw
//! characters at the current insertion point, ahead of the network
//! stream. That needs an insertion-point stack in the driver: bytes
//! written by a script are tokenized before the buffered network
//! bytes that follow them. M2 has no JS engine, so there is no
//! `document.write` yet; when Stage 7 lands it, the stack lives here
//! (bytes pushed in front of `input`, drained before network bytes),
//! not in the builder.
//!
//! # Script handling (R18, parser side — M2 simplification)
//!
//! A classic parser-blocking `<script>` (external or inline, no
//! `async`/`defer`, not `type="module"`) pauses the parser: `feed`
//! returns with a [`DomEvent::ScriptPaused`] and the caller fetches
//! (external) and resumes by re-feeding. Scripts are NOT executed in
//! M2 — the parser resumes after fetch, which is wrong vs spec for
//! inline scripts and is a documented simplification, not spec
//! behaviour. `defer` scripts (and `type="module"`, deferred by
//! default) are queued in document order ([`Parser::defer_scripts`])
//! to run after the parse; `async` scripts are fetched and held
//! ([`Parser::async_scripts`]).

use std::sync::{Arc, Mutex};

use crate::builder::{DomEvent, ParseOutput, ProcessOutcome, TreeBuilder};
use crate::scanner::{Destination, DiscoveredResource, Priority, classify_element, resolve_url};
use crate::tokenizer::{Token, Tokenizer};
use crate::{Dom, NodeId};

/// What a parser-blocking `<script>` pause carries. The worker fetches
/// `src` (if external), then resumes the parser by re-feeding; the
/// script itself is not executed in M2 (see module docs).
#[derive(Debug, Clone)]
pub struct ScriptPaused {
    /// The `<script>` element in the live `Dom`.
    pub script: NodeId,
    /// Resolved `src`, or `None` for an inline script.
    pub src: Option<String>,
    pub inline: bool,
    pub is_async: bool,
    pub is_defer: bool,
    pub is_module: bool,
}

/// The result of a finished streaming parse: the document plus the
/// script queues, in document order.
#[derive(Debug)]
pub struct ParsedStream {
    pub output: ParseOutput,
    /// `defer` (and `type="module"`) scripts, in document order: run
    /// after the parse, before `DOMContentLoaded` (Stage 7).
    pub defer_scripts: Vec<NodeId>,
    /// `async` scripts, in document order: fetched and held (Stage 7
    /// executes each as soon as its fetch completes).
    pub async_scripts: Vec<NodeId>,
}

/// Direct `Tokenizer` + `TreeBuilder` driver over a shared `Dom`.
///
/// The builder holds the `Dom` behind `Arc<Mutex<…>>`, so the live
/// `Dom` is readable mid-parse through [`Parser::shared_dom`].
/// Single-threaded by construction: one `Parser` lives on the
/// per-tab worker thread; other threads read the `Dom` through the
/// shared handle.
pub struct Parser {
    tokenizer: Tokenizer,
    builder: Option<TreeBuilder>,
    document_url: String,
    pending_bytes: Vec<u8>,
    buffered: Vec<u8>,
    encoding_label: Option<String>,
    defer_scripts: Vec<NodeId>,
    async_scripts: Vec<NodeId>,
    finished: bool,
}

impl Parser {
    pub fn new() -> Self {
        Self::with_document_url("")
    }

    /// Parse a document fetched from `document_url` (used as the base
    /// for relative subresource URLs).
    pub fn with_document_url(document_url: &str) -> Self {
        Self::with_opts(document_url, true)
    }

    /// Like [`Parser::with_document_url`], with scripting on/off
    /// (controls `<noscript>` parsing).
    pub fn with_opts(document_url: &str, scripting_enabled: bool) -> Self {
        Self {
            tokenizer: Tokenizer::with_scripting(scripting_enabled),
            builder: Some(TreeBuilder::new(scripting_enabled)),
            document_url: document_url.to_string(),
            pending_bytes: Vec::new(),
            buffered: Vec::new(),
            encoding_label: None,
            defer_scripts: Vec::new(),
            async_scripts: Vec::new(),
            finished: false,
        }
    }

    /// Live view of the `Dom` built so far (readable mid-parse).
    pub fn shared_dom(&self) -> Arc<Mutex<Dom>> {
        self.builder
            .as_ref()
            .map(|b| b.shared_dom())
            .unwrap_or_else(|| Arc::new(Mutex::new(Dom::new())))
    }

    /// `defer` (and `type="module"`) scripts seen so far, document order.
    pub fn defer_scripts(&self) -> &[NodeId] {
        &self.defer_scripts
    }

    /// `async` scripts seen so far, document order.
    pub fn async_scripts(&self) -> &[NodeId] {
        &self.async_scripts
    }

    /// Last encoding label surfaced, if any (`None` = no indicator
    /// fired; the parse is UTF-8 as fed).
    pub fn encoding_label(&self) -> Option<&str> {
        self.encoding_label.as_deref()
    }

    /// Every byte fed so far, in order. On an encoding indicator the
    /// caller re-parses from scratch by feeding this prefix to a fresh
    /// `Parser`.
    pub fn buffered_prefix(&self) -> &[u8] {
        &self.buffered
    }

    /// The base URL resource URLs resolve against right now: the
    /// document URL, overridden by the first `<base href>` seen so
    /// far (same rule as the preload scanner).
    fn effective_base(&self) -> String {
        match self.builder.as_ref().and_then(|b| b.base_href()) {
            Some(raw) => resolve_url(&self.document_url, &raw),
            None => self.document_url.clone(),
        }
    }

    /// Classify a completed `<script>` element from its attributes.
    fn classify_script(&self, handle: NodeId) -> ScriptPaused {
        let dom = self.shared_dom();
        let dom = dom.lock().unwrap();
        let node = dom.get(handle);
        let has_attr =
            |name: &str| node.map(|n| n.attrs.iter().any(|(n2, _)| n2 == name)).unwrap_or(false);
        let is_module = node
            .and_then(|n| n.attr("type"))
            .is_some_and(|t| t.eq_ignore_ascii_case("module"));
        let is_async = has_attr("async");
        let is_defer = has_attr("defer") && !is_async;
        let src = node.and_then(|n| n.attr("src")).and_then(|raw| {
            if raw.trim().is_empty() {
                None
            } else {
                Some(resolve_url(&self.effective_base(), raw))
            }
        });
        ScriptPaused {
            script: handle,
            src: src.clone(),
            inline: src.is_none(),
            is_async,
            is_defer: is_defer || (is_module && !is_async),
            is_module,
        }
    }

    /// Turn the builder's drained events into worker events: pass
    /// builder events through, classify every created element's
    /// fetchable attributes with the same function the preload
    /// scanner uses ([`classify_element`]), and surface a
    /// `<meta charset>` / `<meta http-equiv="content-type">` as an
    /// `EncodingIndicator`.
    fn drain_builder_events(&mut self, out: &mut Vec<DomEvent>) {
        let events = match self.builder.as_mut() {
            Some(b) => b.drain_events(),
            None => Vec::new(),
        };
        for event in events {
            if let DomEvent::ElementCreated { node, ref tag } = event {
                let attrs = {
                    let dom = self.shared_dom();
                    let dom = dom.lock().unwrap();
                    dom.get(node).map(|n| n.attrs.clone()).unwrap_or_default()
                };
                if tag == "meta" {
                    if let Some(label) = meta_charset(&attrs) {
                        self.encoding_label = Some(label.clone());
                        out.push(DomEvent::EncodingIndicator { label });
                    }
                }
                let base = self.effective_base();
                for raw in classify_element(tag, &attrs, false, false) {
                    let resolved = resolve_url(&base, &raw.url);
                    if tag == "link" && raw.destination == Destination::Stylesheet {
                        out.push(DomEvent::StylesheetDiscovered {
                            node,
                            href: resolved.clone(),
                        });
                    }
                    out.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                        url: resolved,
                        destination: raw.destination,
                        priority: raw.priority,
                        from: raw.from,
                    }));
                }
            }
            out.push(event);
        }
    }

    /// Feed one chunk of bytes. Returns the [`DomEvent`]s observed
    /// while draining the tokenizer: element creations, subtree
    /// completions, stylesheet / resource discoveries, and — at most
    /// one, last — a pause event (`ScriptPaused` or
    /// `EncodingIndicator`).
    ///
    /// * A classic parser-blocking `<script>` ends the drain: the
    ///   returned events end with `ScriptPaused`; resume by calling
    ///   `feed` again (with the next chunk, or empty to continue
    ///   draining). `defer` / `async` / `module` scripts do NOT pause
    ///   — they are queued / recorded and the drain continues.
    /// * An `EncodingIndicator` is reported as an event and the drain
    ///   continues (the caller owns the restart decision; see module
    ///   docs). The indicator may fire more than once; each firing is
    ///   an event.
    /// * Feeding after `finish()` returns no events.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<DomEvent> {
        if self.finished || self.builder.is_none() {
            return Vec::new();
        }
        self.buffered.extend_from_slice(chunk);

        // Decode, holding an incomplete UTF-8 tail for the next chunk
        // (a multi-byte character split across chunks must not become
        // U+FFFD). Interior invalid bytes decode lossy, as fed.
        let mut bytes = std::mem::take(&mut self.pending_bytes);
        bytes.extend_from_slice(chunk);
        let text: String = match std::str::from_utf8(&bytes) {
            Ok(s) => s.to_string(),
            Err(e) if e.error_len().is_none() => {
                let valid = e.valid_up_to();
                let tail = bytes[valid..].to_vec();
                let head = String::from_utf8_lossy(&bytes[..valid]).into_owned();
                self.pending_bytes = tail;
                head
            }
            Err(_) => String::from_utf8_lossy(&bytes).into_owned(),
        };
        if !text.is_empty() {
            self.tokenizer.feed_str(&text);
        }

        let mut out = Vec::new();
        loop {
            let Some(token) = self.tokenizer.next_token() else {
                // Buffer ends mid-token: wait for more input.
                self.drain_builder_events(&mut out);
                break;
            };
            if matches!(token, Token::Eof) {
                self.drain_builder_events(&mut out);
                break;
            }
            let outcome = {
                let builder = self.builder.as_mut().unwrap();
                builder.process(token, &mut self.tokenizer)
            };
            match outcome {
                ProcessOutcome::Script(handle) => {
                    self.drain_builder_events(&mut out);
                    let paused = self.classify_script(handle);
                    if paused.is_async {
                        self.async_scripts.push(handle);
                        if let Some(src) = &paused.src {
                            out.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                                url: src.clone(),
                                destination: Destination::Script,
                                priority: Priority::Medium,
                                from: "script[src]",
                            }));
                        }
                        continue;
                    }
                    if paused.is_defer {
                        self.defer_scripts.push(handle);
                        if let Some(src) = &paused.src {
                            out.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                                url: src.clone(),
                                destination: Destination::Script,
                                priority: Priority::Medium,
                                from: "script[src]",
                            }));
                        }
                        continue;
                    }
                    out.push(DomEvent::ScriptPaused(paused));
                    break;
                }
                ProcessOutcome::Continue => {}
            }
        }
        out
    }

    /// End of input: emit EOF through the tokenizer and return the
    /// finished document plus the script queues.
    pub fn finish(mut self) -> ParsedStream {
        self.finished = true;
        self.tokenizer.set_eof();
        // Drain any remaining tokens.
        loop {
            let Some(token) = self.tokenizer.next_token() else {
                break;
            };
            if matches!(token, Token::Eof) {
                break;
            }
            let outcome = {
                let builder = self.builder.as_mut().unwrap();
                builder.process(token, &mut self.tokenizer)
            };
            match outcome {
                ProcessOutcome::Script(handle) => {
                    let paused = self.classify_script(handle);
                    if paused.is_async {
                        self.async_scripts.push(handle);
                    } else if paused.is_defer {
                        self.defer_scripts.push(handle);
                    }
                }
                ProcessOutcome::Continue => {}
            }
        }
        let mut builder = self.builder.take().unwrap();
        // Fold tokenizer parse errors into the builder stats.
        builder.absorb_tokenizer_errors_pub(&self.tokenizer);
        // Feed EOF through the insertion modes (implied <body> etc.)
        // before popping the stack.
        let _ = builder.process(Token::Eof, &mut self.tokenizer);
        builder.process_eof();
        let output = builder.finish();
        ParsedStream {
            output,
            defer_scripts: std::mem::take(&mut self.defer_scripts),
            async_scripts: std::mem::take(&mut self.async_scripts),
        }
    }
}

/// Extract a charset label from a `<meta>` element's attributes:
/// `charset="..."`, or `http-equiv="content-type"` with
/// `content="...; charset=..."`.
fn meta_charset(attrs: &[(String, String)]) -> Option<String> {
    let get = |name: &str| {
        attrs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    };
    if let Some(cs) = get("charset") {
        let cs = cs.trim();
        if !cs.is_empty() {
            return Some(cs.to_string());
        }
    }
    if get("http-equiv").is_some_and(|h| h.eq_ignore_ascii_case("content-type")) {
        if let Some(content) = get("content") {
            for part in content.split(';') {
                let part = part.trim();
                if let Some(rest) = part.strip_prefix("charset=") {
                    let label = rest.trim().trim_matches('"').trim();
                    if !label.is_empty() {
                        return Some(label.to_string());
                    }
                } else if let Some(rest) =
                    part.to_ascii_lowercase().strip_prefix("charset=")
                {
                    let label = rest.trim().trim_matches('"').trim();
                    if !label.is_empty() {
                        return Some(label.to_string());
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, DomEvent};

    fn find_tag(dom: &Dom, id: NodeId, tag: &str) -> Option<NodeId> {
        for &child in dom.children(id) {
            if dom.tag(child) == Some(tag) {
                return Some(child);
            }
            if let Some(found) = find_tag(dom, child, tag) {
                return Some(found);
            }
        }
        None
    }

    fn count_tag(dom: &Dom, id: NodeId, tag: &str) -> usize {
        let mut n = 0;
        if dom.tag(id) == Some(tag) {
            n += 1;
        }
        for &child in dom.children(id) {
            n += count_tag(dom, child, tag);
        }
        n
    }

    fn text_content(dom: &Dom, id: NodeId) -> String {
        let mut s = String::new();
        if let Some(t) = dom.text(id) {
            s.push_str(t);
        }
        for &child in dom.children(id) {
            s.push_str(&text_content(dom, child));
        }
        s
    }

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style type="text/css">
    body { background-color: #f0f0f2; margin: 0; padding: 0; }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>"#;

    #[test]
    fn streaming_chunked_equals_oneshot() {
        let one = parse(EXAMPLE_COM);
        let mut p = Parser::new();
        for chunk in EXAMPLE_COM.as_bytes().chunks(7) {
            let events = p.feed(chunk);
            assert!(
                events
                    .iter()
                    .all(|e| !matches!(e, DomEvent::ScriptPaused(_))),
                "unexpected script pause: {events:?}"
            );
        }
        let done = p.finish();
        assert_eq!(done.output.dom.quirks_mode(), one.quirks_mode());
        let root = done.output.dom.root().unwrap();
        let h1 = find_tag(&done.output.dom, root, "h1").unwrap();
        assert_eq!(text_content(&done.output.dom, h1), "Example Domain");
        assert_eq!(count_tag(&done.output.dom, root, "p"), count_tag(&one, one.root().unwrap(), "p"));
    }

    #[test]
    fn live_dom_readable_mid_parse() {
        let mut p = Parser::new();
        p.feed(b"<div><p>one");
        let dom = p.shared_dom();
        let dom = dom.lock().unwrap();
        assert!(find_tag(&dom, dom.root().unwrap(), "p").is_some());
    }

    #[test]
    fn pause_at_script_and_resume() {
        let mut p = Parser::new();
        let events = p.feed(b"<p>one<script>var x = 1;</script><p>two");
        let paused = events.iter().find_map(|e| match e {
            DomEvent::ScriptPaused(p) => Some(p.clone()),
            _ => None,
        });
        let paused = paused.expect("expected a ScriptPaused event");
        assert!(paused.inline);
        assert!(!paused.is_async && !paused.is_defer);
        {
            let dom = p.shared_dom();
            let dom = dom.lock().unwrap();
            assert_eq!(count_tag(&dom, dom.root().unwrap(), "p"), 1);
            assert_eq!(dom.tag(paused.script), Some("script"));
        }
        let events = p.feed(b"");
        assert!(
            events.iter().all(|e| !matches!(e, DomEvent::ScriptPaused(_))),
            "no second pause: {events:?}"
        );
        let done = p.finish();
        let root = done.output.dom.root().unwrap();
        assert_eq!(count_tag(&done.output.dom, root, "p"), 2);
        let script = find_tag(&done.output.dom, root, "script").unwrap();
        assert_eq!(text_content(&done.output.dom, script), "var x = 1;");
    }

    #[test]
    fn defer_and_async_do_not_pause() {
        let mut p = Parser::with_document_url("https://example.com/page");
        let events = p.feed(
            b"<script defer src=\"/d.js\"></script><script async src=\"/a.js\"></script>\
              <script type=\"module\" src=\"/m.js\"></script><p>after</p>",
        );
        assert!(
            events.iter().all(|e| !matches!(e, DomEvent::ScriptPaused(_))),
            "defer/async/module must not pause: {events:?}"
        );
        let fetched: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                DomEvent::ResourceDiscovered(r) => Some(r.url.as_str()),
                _ => None,
            })
            .collect();
        assert!(fetched.contains(&"https://example.com/d.js"), "{fetched:?}");
        assert!(fetched.contains(&"https://example.com/a.js"), "{fetched:?}");
        assert!(fetched.contains(&"https://example.com/m.js"), "{fetched:?}");
        let done = p.finish();
        assert_eq!(done.defer_scripts.len(), 2, "defer + module queued");
        assert_eq!(done.async_scripts.len(), 1, "async held");
        let root = done.output.dom.root().unwrap();
        assert!(find_tag(&done.output.dom, root, "p").is_some());
    }

    #[test]
    fn external_classic_script_pauses_with_resolved_src() {
        let mut p = Parser::with_document_url("https://example.com/dir/page");
        let events = p.feed(b"<script src=\"s.js\"></script><p>two");
        let paused = events.iter().find_map(|e| match e {
            DomEvent::ScriptPaused(p) => Some(p.clone()),
            _ => None,
        });
        let paused = paused.expect("classic external script pauses");
        assert_eq!(paused.src.as_deref(), Some("https://example.com/dir/s.js"));
        assert!(!paused.inline);
        let _ = p.feed(b"");
        let done = p.finish();
        assert!(find_tag(&done.output.dom, done.output.dom.root().unwrap(), "p").is_some());
    }

    #[test]
    fn encoding_indicator_surfaces_and_prefix_is_buffered() {
        let mut p = Parser::new();
        let input = b"<meta charset=\"UTF-8\" /><meta charset=\"UTF-8\" /> other stuff";
        let events = p.feed(input);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DomEvent::EncodingIndicator { .. })),
            "expected an EncodingIndicator event: {events:?}"
        );
        assert_eq!(p.encoding_label(), Some("UTF-8"));
        assert_eq!(p.buffered_prefix(), input);
        let done = p.finish();
        let root = done.output.dom.root().unwrap();
        assert!(text_content(&done.output.dom, root).contains("other stuff"));
    }

    #[test]
    fn stylesheet_link_discovered_as_event() {
        let mut p = Parser::with_document_url("https://example.com/");
        let events = p.feed(b"<link rel=\"stylesheet\" href=\"a.css\"><p>x");
        let href = events.iter().find_map(|e| match e {
            DomEvent::StylesheetDiscovered { href, .. } => Some(href.clone()),
            _ => None,
        });
        assert_eq!(href.as_deref(), Some("https://example.com/a.css"));
        let _ = p.finish();
    }

    #[test]
    fn subtree_complete_events_fire() {
        let mut p = Parser::new();
        let events = p.feed(b"<div><p>one</p></div>");
        assert!(
            events.iter().any(|e| matches!(
                e,
                DomEvent::SubtreeComplete { .. } | DomEvent::ElementCreated { .. }
            )),
            "expected element/subtree events: {events:?}"
        );
        let _ = p.finish();
    }

    #[test]
    fn utf8_split_across_chunks_decodes_whole() {
        let mut p = Parser::new();
        p.feed("<p>caf".as_bytes());
        p.feed("\u{e9}</p>".as_bytes());
        let done = p.finish();
        let root = done.output.dom.root().unwrap();
        let p_el = find_tag(&done.output.dom, root, "p").unwrap();
        assert_eq!(text_content(&done.output.dom, p_el), "café");
    }
}
