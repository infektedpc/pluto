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
//! # `document.write` (Stage 7 — implemented in [`crate::scripts`])
//!
//! A classic script may call `document.write()`, which injects raw
//! characters at the current insertion point, ahead of the network
//! stream. The driver holds the insertion point: while paused at a
//! parser-blocking script, [`Parser::document_write`] splices the
//! markup into the tokenizer at the pause position
//! ([`Tokenizer::insert_chars_at`]) and the insertion point
//! advances past it, so sequential writes append in call order and
//! a nested script's writes land before the not-yet-consumed
//! remainder. The `open` / `close` / `write` state machine
//! (nesting-level guard, `throw-on-dynamic-markup-insertion`
//! counter, abort flag, implicit-`open()` after parsing) lives in
//! [`crate::scripts::DocumentWriter`]; the parser-side effects
//! (splice, abort) are the `Parser::document_*` methods.
//!
//! # Script handling (Stage 7)
//!
//! A classic parser-blocking `<script>` (external or inline, no
//! `async`/`defer`, not `type="module"`) pauses the parser: `feed`
//! returns with a [`DomEvent::ScriptPaused`] and the caller fetches
//! (external), waits out the pending-stylesheet gate
//! ([`crate::scripts::StylesheetGate`],
//! `ScriptPaused.stylesheets_pending`), executes, and resumes by
//! re-feeding. Scripts are still NOT executed here — execution
//! belongs to `pluto-js` / the integration wave, which drives the
//! queues below. `defer` scripts (and `type="module"`, deferred by
//! default) are queued in document order in
//! [`crate::scripts::ScriptQueues`] (module `src` URLs deduped)
//! to run after the parse, before `DOMContentLoaded`; `async`
//! scripts are queued there too and become ready in arrival
//! order via `AsyncQueue::mark_ready` as their fetches complete.
//! `defer` has no effect on inline classic scripts (they stay
//! parser-blocking). The legacy `Vec<NodeId>` views
//! ([`Parser::defer_scripts`], [`Parser::async_scripts`],
//! [`ParsedStream::defer_scripts`]) are kept for the current
//! `pluto-browser` load path; new code should use the queues.

use std::sync::{Arc, Mutex};

use crate::builder::{DomEvent, ParseOutput, ProcessOutcome, TreeBuilder};
use crate::scanner::{Destination, DiscoveredResource, Priority, classify_element, resolve_url};
use crate::scripts::{
    AsyncQueue, CloseOutcome, DeferQueue, DocumentWriter, DynamicMarkupError, OpenOutcome,
    QueuedScript, ScriptKind, ScriptQueues, StylesheetGate, WriteOutcome, classify_script,
};
use crate::tokenizer::{Token, Tokenizer};
use crate::{Dom, NodeId};

/// The referrer policy for a document (Stage 9 Rec 15) — the two
/// values Pluto implements. This mirrors
/// `pluto_net::ReferrerPolicy` (same tokens, same parse); the two
/// enums are kept in sync by hand because neither crate may depend
/// on the other — `pluto-browser` maps between them when it pushes
/// the policy into the loader. Do not add a third value to one
/// without the other.
///
/// A document with no `<meta name="referrer">` (and no
/// `Referrer-Policy` header, applied net-side) uses the default,
/// `no-referrer-when-downgrade`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferrerPolicy {
    /// Send the full URL same-origin, only the origin
    /// cross-origin, and nothing on an https→http downgrade.
    #[default]
    NoReferrerWhenDowngrade,
    /// Send the full URL same-origin (even on a downgrade), only
    /// the origin cross-origin, and nothing on a cross-origin
    /// downgrade.
    StrictOriginWhenCrossOrigin,
}

impl ReferrerPolicy {
    /// Parse a `<meta name="referrer">` content value. A
    /// comma-separated list takes the last recognised token, like
    /// the header form; unknown tokens are skipped; `None` when
    /// no token is recognised.
    pub fn parse(value: &str) -> Option<Self> {
        let mut found = None;
        for token in value.split(',') {
            let token = token.trim().to_ascii_lowercase();
            found = match token.as_str() {
                "no-referrer-when-downgrade" => Some(Self::NoReferrerWhenDowngrade),
                "strict-origin-when-cross-origin" => {
                    Some(Self::StrictOriginWhenCrossOrigin)
                }
                _ => found,
            };
        }
        found
    }

    /// The header/meta token for this policy.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoReferrerWhenDowngrade => "no-referrer-when-downgrade",
            Self::StrictOriginWhenCrossOrigin => "strict-origin-when-cross-origin",
        }
    }
}

/// What a parser-blocking `<script>` pause carries. The worker fetches
/// `src` (if external), waits for the pending-stylesheet gate to
/// clear when `stylesheets_pending` is set, executes the script
/// (in `pluto-js`), then resumes the parser by re-feeding.
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
    /// The destination [`crate::scripts::classify_script`] chose.
    /// Always [`ScriptKind::ParserBlocking`] for a pause; the
    /// other kinds never pause.
    pub kind: ScriptKind,
    /// True when a render-blocking stylesheet discovered so far
    /// has not yet loaded or errored: the script must not execute
    /// until the gate clears (see [`crate::scripts::StylesheetGate`]).
    /// A snapshot at pause time — re-check
    /// [`Parser::stylesheets_pending`] live.
    pub stylesheets_pending: bool,
}

/// The result of a finished streaming parse: the document plus the
/// script queues, in document order.
#[derive(Debug)]
pub struct ParsedStream {
    pub output: ParseOutput,
    /// `defer` (and `type="module"`) scripts, in document order: run
    /// after the parse, before `DOMContentLoaded` (Stage 7).
    /// Legacy view of [`ParsedStream::script_queues`]; kept for the
    /// current `pluto-browser` load path.
    pub defer_scripts: Vec<NodeId>,
    /// `async` scripts, in document order: fetched and held (Stage 7
    /// executes each as soon as its fetch completes). Legacy view of
    /// [`ParsedStream::script_queues`]; kept for the current
    /// `pluto-browser` load path.
    pub async_scripts: Vec<NodeId>,
    /// The script queues for this document: the defer queue (drain
    /// after the parse, before `DOMContentLoaded`) and the async
    /// queue (arrival order). New code should use these, not the
    /// legacy `Vec<NodeId>` views above.
    pub script_queues: ScriptQueues,
    /// The document's referrer policy (Stage 9 Rec 15): the last
    /// recognised `<meta name="referrer">` content value, or
    /// `None` when the document carries none (the default policy
    /// then applies).
    pub referrer_policy: Option<ReferrerPolicy>,
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
    /// Defer + async queues with module-URL dedup (Stage 7).
    /// The `Vec<NodeId>` fields above are legacy views of these,
    /// kept for the current `pluto-browser` load path.
    script_queues: ScriptQueues,
    /// The document's referrer policy (Stage 9 Rec 15): the last
    /// recognised `<meta name="referrer">` content value seen so
    /// far, or `None` when the document carries none. A `<meta
    /// http-equiv="Strict-Transport-Security">` never lands here
    /// — HSTS is never recorded from markup (RFC 6797 §8.1).
    referrer_policy: Option<ReferrerPolicy>,
    /// Render-blocking stylesheets discovered so far and not yet
    /// resolved; a parser-blocking script waits on this gate.
    stylesheet_gate: StylesheetGate,
    /// `document.write` / `open` / `close` state machine.
    writer: DocumentWriter,
    /// Char index into the tokenizer input where `document.write`
    /// text goes, set at a script pause and advanced past each
    /// write. `None` between pauses — a write then falls back to
    /// the tokenizer's current read position.
    insertion_point: Option<usize>,
    /// The layout viewport width (CSS px) `srcset` / `<picture>`
    /// selection evaluates against (Stage 11 Rec 10); the device
    /// pixel ratio alongside it. Defaults are the startup
    /// viewport (1024 px, DPR 1.0); see [`Parser::set_viewport`].
    viewport_width: f32,
    viewport_dpr: f32,
    /// Paused at a parser-blocking `<script>`, awaiting execution.
    paused: bool,
    /// The parse was aborted (`document.open()` mid-parse, or
    /// [`Parser::abort_parser`]): `feed` returns nothing and
    /// `document.write` is ignored until a new parser takes over.
    aborted: bool,
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
        let mut writer = DocumentWriter::new();
        writer.set_parsing(true);
        Self {
            tokenizer: Tokenizer::with_scripting(scripting_enabled),
            builder: Some(TreeBuilder::new(scripting_enabled)),
            document_url: document_url.to_string(),
            pending_bytes: Vec::new(),
            buffered: Vec::new(),
            encoding_label: None,
            defer_scripts: Vec::new(),
            async_scripts: Vec::new(),
            script_queues: ScriptQueues::default(),
            referrer_policy: None,
            stylesheet_gate: StylesheetGate::default(),
            writer,
            insertion_point: None,
            viewport_width: 1024.0,
            viewport_dpr: 1.0,
            paused: false,
            aborted: false,
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

    /// The layout viewport `srcset` / `<picture>` selection
    /// evaluates against (Stage 11 Rec 10): CSS px wide, at this
    /// device pixel ratio. Defaults to the startup viewport
    /// (1024 px, DPR 1.0); the load path sets the real one before
    /// the first chunk is fed, so a responsive image selects the
    /// candidate the page will actually use.
    pub fn set_viewport(&mut self, width: f32, dpr: f32) {
        self.viewport_width = width;
        self.viewport_dpr = dpr;
    }

    /// `defer` (and `type="module"`) scripts seen so far, document order.
    pub fn defer_scripts(&self) -> &[NodeId] {
        &self.defer_scripts
    }

    /// `async` scripts seen so far, document order.
    pub fn async_scripts(&self) -> &[NodeId] {
        &self.async_scripts
    }

    /// The defer + async queues (with module-URL dedup). New code
    /// should use these, not the legacy `Vec<NodeId>` views.
    pub fn script_queues(&self) -> &ScriptQueues {
        &self.script_queues
    }

    /// The defer queue alone (document order).
    pub fn defer_queue(&self) -> &DeferQueue {
        self.script_queues.defer_queue()
    }

    /// The async queue alone (arrival order once ready).
    pub fn async_queue(&self) -> &AsyncQueue {
        self.script_queues.async_queue()
    }

    /// Mutable async queue: the integration wave marks scripts ready
    /// as their fetches complete (arrival order) and executes them.
    pub fn async_queue_mut(&mut self) -> &mut AsyncQueue {
        self.script_queues.async_queue_mut()
    }

    /// The pending-stylesheet gate: true while a discovered
    /// render-blocking stylesheet has not yet loaded or errored.
    /// A parser-blocking script must not execute while this is
    /// true.
    pub fn stylesheets_pending(&self) -> bool {
        self.stylesheet_gate.is_blocked()
    }

    /// The gate itself (pending sheet URLs, for diagnostics).
    pub fn stylesheet_gate(&self) -> &StylesheetGate {
        &self.stylesheet_gate
    }

    /// A discovered stylesheet loaded: it no longer blocks
    /// parser-blocking scripts. The caller (browser) feeds this;
    /// the parser only discovers sheets, it never fetches them.
    /// Also call this for sheets the parser never discovered
    /// (preload-scanner finds, `@import` chains) if the caller
    /// tracks them in the gate — see the module docs for the
    /// known gap.
    pub fn note_stylesheet_loaded(&mut self, href: &str) {
        self.stylesheet_gate.note_loaded(href);
    }

    /// A discovered stylesheet failed (or the render-blocking
    /// deadline fired): it no longer blocks scripts — the page
    /// renders without it.
    pub fn note_stylesheet_failed(&mut self, href: &str) {
        self.stylesheet_gate.note_failed(href);
    }

    /// The `document.write` / `open` / `close` state machine.
    /// The engine must call `enter_script` / `exit_script` on it
    /// around executing a parser-inserted script, and increment
    /// the throw counter around parser-invoked custom-element
    /// constructors (see [`crate::scripts::DocumentWriter`]).
    pub fn document_writer(&mut self) -> &mut DocumentWriter {
        &mut self.writer
    }

    /// `document.write(markup)`: insert `markup` at the parser's
    /// insertion point, ahead of buffered network bytes. Only
    /// meaningful while a parser-inserted script is executing
    /// (paused) or between feeds mid-parse; after the parse has
    /// finished it returns [`WriteOutcome::ImplicitOpen`] (the
    /// caller must clear the document and drive a script-created
    /// parser — the markup is NOT appended to the finished
    /// document). After an abort it is ignored. Throws
    /// [`DynamicMarkupError`] (`InvalidStateError`) while the
    /// throw-on-dynamic-markup-insertion counter is raised.
    pub fn document_write(&mut self, markup: &str) -> Result<WriteOutcome, DynamicMarkupError> {
        let outcome = self.writer.write()?;
        if outcome == WriteOutcome::Inserted {
            let at = self
                .insertion_point
                .unwrap_or_else(|| self.tokenizer.input_pos());
            let inserted = self.tokenizer.insert_chars_at(at, markup);
            self.insertion_point = Some(at + inserted);
        }
        Ok(outcome)
    }

    /// `document.open()`: ignored while the document is still
    /// being parsed by a parser-inserted script; otherwise the
    /// document is replaced. An `Opened` result mid-parse aborts
    /// this parser — the caller clears the document and drives a
    /// fresh script-created parser. Throws
    /// [`DynamicMarkupError`] while the throw counter is raised.
    pub fn document_open(&mut self) -> Result<OpenOutcome, DynamicMarkupError> {
        let outcome = self.writer.open()?;
        if outcome == OpenOutcome::Opened {
            self.aborted = true;
        }
        Ok(outcome)
    }

    /// `document.close()`: ignored while the network parser is
    /// still running and after an abort; otherwise closes the
    /// script-created parser. Throws [`DynamicMarkupError`] while
    /// the throw counter is raised.
    pub fn document_close(&mut self) -> Result<CloseOutcome, DynamicMarkupError> {
        let outcome = self.writer.close()?;
        if outcome == CloseOutcome::Closed {
            self.finished = true;
        }
        Ok(outcome)
    }

    /// Abort the active parser: `feed` returns nothing and
    /// `document.write` / `close` are ignored until a new parser
    /// (`document.open()`) replaces it.
    pub fn abort_parser(&mut self) {
        self.aborted = true;
        self.writer.abort();
    }

    /// True after [`Parser::abort_parser`] or a mid-parse
    /// `document.open()`.
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    /// True while paused at a parser-blocking `<script>`, awaiting
    /// execution.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Last encoding label surfaced, if any (`None` = no indicator
    /// fired; the parse is UTF-8 as fed).
    pub fn encoding_label(&self) -> Option<&str> {
        self.encoding_label.as_deref()
    }

    /// The document's referrer policy from `<meta
    /// name="referrer">` seen so far (`None` = no such meta yet;
    /// the default policy applies). The browser polls this after
    /// each feed and pushes it into the loader.
    pub fn referrer_policy(&self) -> Option<ReferrerPolicy> {
        self.referrer_policy
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
    /// `defer` has no effect on inline classic scripts (they stay
    /// parser-blocking); `type="module"` is deferred by default.
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
        let is_defer = has_attr("defer");
        let src = node.and_then(|n| n.attr("src")).and_then(|raw| {
            if raw.trim().is_empty() {
                None
            } else {
                Some(resolve_url(&self.effective_base(), raw))
            }
        });
        let kind = classify_script(is_module, is_async, is_defer, src.is_some());
        ScriptPaused {
            script: handle,
            src: src.clone(),
            inline: src.is_none(),
            is_async: kind == ScriptKind::Async,
            is_defer: matches!(kind, ScriptKind::Defer | ScriptKind::Module),
            is_module,
            kind,
            stylesheets_pending: self.stylesheet_gate.is_blocked(),
        }
    }

    /// Route a completed `<script>` element: async and defer /
    /// module scripts are queued (module `src` URLs deduped — a
    /// duplicate is dropped silently) and their fetch is reported
    /// in `events`; a parser-blocking script is returned for the
    /// caller to pause on. Shared by `feed` and `finish`.
    fn handle_script(
        &mut self,
        handle: NodeId,
        events: &mut Vec<DomEvent>,
    ) -> Option<ScriptPaused> {
        let paused = self.classify_script(handle);
        // Stage 9 Rec 14 (SRI): the script's `integrity` attribute
        // rides the discovery so the fetch can verify the streamed
        // bytes against it.
        let integrity = {
            let dom = self.shared_dom();
            let dom = dom.lock().unwrap();
            dom.get(handle)
                .and_then(|n| n.attr("integrity"))
                .map(str::to_string)
        };
        if paused.is_async {
            let queued = QueuedScript::new(
                handle,
                paused.src.clone(),
                ScriptKind::Async,
                paused.is_module,
            );
            if self.script_queues.enqueue_async(queued) {
                self.async_scripts.push(handle);
                events.push(DomEvent::AsyncScriptQueued {
                    script: handle,
                    src: paused.src.clone(),
                });
                if let Some(src) = &paused.src {
                    events.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                        url: src.clone(),
                        destination: Destination::Script,
                        priority: Priority::Medium,
                        from: "script[src]",
                        integrity: integrity.clone(),
                        is_module: paused.is_module,
                    }));
                }
            }
            return None;
        }
        if paused.is_defer {
            let kind = if paused.is_module {
                ScriptKind::Module
            } else {
                ScriptKind::Defer
            };
            let queued = QueuedScript::new(handle, paused.src.clone(), kind, paused.is_module);
            if self.script_queues.enqueue_defer(queued) {
                self.defer_scripts.push(handle);
                if let Some(src) = &paused.src {
                    events.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                        url: src.clone(),
                        destination: Destination::Script,
                        priority: Priority::Medium,
                        from: "script[src]",
                        integrity: integrity.clone(),
                        is_module: paused.is_module,
                    }));
                }
            }
            return None;
        }
        Some(paused)
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
                    // Stage 9 Rec 15: <meta name="referrer">
                    // sets the document's referrer policy (last
                    // recognised one wins). A <meta http-equiv>
                    // carrying STS is deliberately NOT read here:
                    // HSTS is never recorded from markup.
                    if let Some(policy) = meta_referrer(&attrs) {
                        self.referrer_policy = Some(policy);
                    }
                }
                let base = self.effective_base();
                // Stage 11 Rec 10: an <img> inside a <picture>
                // selects ONE candidate from the DOM (its
                // <source> siblings + its own srcset/src) instead
                // of emitting every candidate — the classifier
                // cannot see siblings, so the drain does this
                // selection itself. Any other element classifies
                // with its real picture/media context (walked from
                // the DOM), not the old hardcoded false/false.
                let (in_picture, media_tag) = {
                    let dom = self.shared_dom();
                    let dom = dom.lock().unwrap();
                    let mut parent = dom.get(node).and_then(|n| n.parent);
                    let mut picture = false;
                    let mut media: Option<String> = None;
                    while let Some(p) = parent {
                        match dom.tag(p) {
                            Some("picture") => {
                                picture = true;
                                break;
                            }
                            Some(t @ ("video" | "audio")) => {
                                media = Some(t.to_string());
                                break;
                            }
                            _ => parent = dom.get(p).and_then(|n| n.parent),
                        }
                    }
                    (picture, media)
                };
                if tag == "img" && in_picture {
                    let (selected, integrity) = {
                        let dom = self.shared_dom();
                        let dom = dom.lock().unwrap();
                        let selected = crate::scanner::select_img_url_from_dom(
                            &dom,
                            node,
                            self.viewport_width,
                            self.viewport_dpr,
                        );
                        let integrity = dom
                            .attr(node, "integrity")
                            .map(str::to_string);
                        (selected, integrity)
                    };
                    if let Some((url, is_imageset)) = selected {
                        out.push(DomEvent::ResourceDiscovered(DiscoveredResource {
                            url: resolve_url(&base, &url),
                            destination: Destination::Image,
                            priority: Priority::Low,
                            from: if is_imageset { "imageset" } else { "img[src]" },
                            integrity,
                            is_module: false,
                        }));
                    }
                } else {
                    // Stage 15 (M7 Phase 1, Rec 11): classify with
                    // the element's real namespace from the `Dom` —
                    // an SVG `<script src>` must not be fetched.
                    let namespace = {
                        let dom = self.shared_dom();
                        let dom = dom.lock().unwrap();
                        dom.namespace(node).unwrap_or("").to_string()
                    };
                    for raw in classify_element(
                        tag,
                        &namespace,
                        &attrs,
                        in_picture,
                        media_tag.as_deref(),
                        self.viewport_width,
                        self.viewport_dpr,
                    ) {
                        let resolved = resolve_url(&base, &raw.url);
                        if tag == "link" && raw.destination == Destination::Stylesheet {
                            // The sheet blocks parser-blocking scripts
                            // until the browser reports it loaded/failed.
                            self.stylesheet_gate.note_discovered(&resolved);
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
                            integrity: raw.integrity.clone(),
                            is_module: raw.is_module,
                        }));
                    }
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
        if self.finished || self.builder.is_none() || self.aborted {
            return Vec::new();
        }
        // Resuming past a pause (or between chunks): the insertion
        // point is re-established at the next pause; a stale one
        // must not be reused by a later `document.write`.
        self.insertion_point = None;
        if self.paused {
            self.paused = false;
            self.writer.set_insertion_point_defined(false);
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
                    match self.handle_script(handle, &mut out) {
                        Some(paused) => {
                            // Parser-blocking: pause here. The
                            // insertion point is the tokenizer's
                            // current read position — everything
                            // after it is still network input.
                            self.paused = true;
                            self.writer.set_insertion_point_defined(true);
                            self.insertion_point = Some(self.tokenizer.input_pos());
                            out.push(DomEvent::ScriptPaused(paused));
                            break;
                        }
                        None => continue,
                    }
                }
                ProcessOutcome::Continue => {}
            }
        }
        out
    }

    /// End of input: emit EOF through the tokenizer and return the
    /// finished document plus the script queues. After this, the
    /// defer queue is drained (in order) and `DOMContentLoaded`
    /// fires; a later `document.write` takes the implicit-`open()`
    /// path.
    pub fn finish(mut self) -> ParsedStream {
        self.finished = true;
        self.writer.set_parsing(false);
        self.writer.set_insertion_point_defined(false);
        self.tokenizer.set_eof();
        // Drain any remaining tokens.
        let mut queued_events = Vec::new();
        while let Some(token) = self.tokenizer.next_token() {
            if matches!(token, Token::Eof) {
                break;
            }
            let outcome = {
                let builder = self.builder.as_mut().unwrap();
                builder.process(token, &mut self.tokenizer)
            };
            // Drain builder events here too (same as `feed`): a
            // `<meta name="referrer">` completed only by EOF must
            // still be captured into `self.referrer_policy`.
            self.drain_builder_events(&mut queued_events);
            match outcome {
                ProcessOutcome::Script(handle) => {
                    // A parser-blocking script at EOF has nothing
                    // left to block; defer/async scripts still queue.
                    let _ = self.handle_script(handle, &mut queued_events);
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
        let mut output = builder.finish();
        output.events.extend(queued_events);
        ParsedStream {
            output,
            defer_scripts: std::mem::take(&mut self.defer_scripts),
            async_scripts: std::mem::take(&mut self.async_scripts),
            script_queues: std::mem::take(&mut self.script_queues),
            referrer_policy: self.referrer_policy,
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a referrer policy from a `<meta name="referrer">`
/// element's attributes (`name="referrer"`, `content="<policy>"`).
/// Returns `None` when the element is not a referrer meta or its
/// content names no policy Pluto implements.
fn meta_referrer(attrs: &[(String, String)]) -> Option<ReferrerPolicy> {
    let get = |name: &str| {
        attrs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    };
    if get("name").is_some_and(|n| n.eq_ignore_ascii_case("referrer"))
        && let Some(content) = get("content") {
            return ReferrerPolicy::parse(content);
        }
    None
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
    if get("http-equiv").is_some_and(|h| h.eq_ignore_ascii_case("content-type"))
        && let Some(content) = get("content") {
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
    fn svg_script_never_pauses_the_parser() {
        // Stage 15 (M7 Phase 1, Rec 11): an SVG <script> is inert
        // by rule — no ScriptPaused event — unlike an HTML <script>.
        let mut p = Parser::new();
        let events = p.feed(b"<svg><script>alert(1)</script></svg><p>after");
        assert!(
            events.iter().all(|e| !matches!(e, DomEvent::ScriptPaused(_))),
            "SVG script must not pause the parser: {events:?}"
        );
        let mut p = Parser::new();
        let events = p.feed(b"<script>alert(1)</script>");
        assert!(
            events.iter().any(|e| matches!(e, DomEvent::ScriptPaused(_))),
            "HTML script must pause the parser"
        );
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

    #[test]
    fn parser_blocking_script_sees_only_dom_above_it() {
        let mut p = Parser::new();
        let events = p.feed(b"<p>one</p><script>var x = 1;</script><p>two</p>");
        let paused = events.iter().find_map(|e| match e {
            DomEvent::ScriptPaused(p) => Some(p.clone()),
            _ => None,
        });
        let paused = paused.expect("classic script pauses");
        assert_eq!(paused.kind, ScriptKind::ParserBlocking);
        assert!(!paused.stylesheets_pending, "no sheets pending");
        {
            let dom = p.shared_dom();
            let dom = dom.lock().unwrap();
            // Only the DOM above the script exists: one <p>, no
            // <p>two</p> yet — a script's querySelector must not
            // see below itself.
            assert_eq!(count_tag(&dom, dom.root().unwrap(), "p"), 1);
            assert_eq!(dom.tag(paused.script), Some("script"));
        }
        let _ = p.feed(b"");
        let done = p.finish();
        let root = done.output.dom.root().unwrap();
        assert_eq!(count_tag(&done.output.dom, root, "p"), 2);
    }

    #[test]
    fn inline_defer_script_stays_parser_blocking() {
        // `defer` has no effect on inline classic scripts (spec
        // table, HTML §4.12): they still pause the parser.
        let mut p = Parser::new();
        let events = p.feed(b"<script defer>var x = 1;</script><p>after</p>");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DomEvent::ScriptPaused(_))),
            "inline defer must pause: {events:?}"
        );
        let _ = p.feed(b"");
        let done = p.finish();
        assert!(done.script_queues.defer_queue().is_empty());
    }

    #[test]
    fn stylesheet_gate_blocks_script_pause_until_resolved() {
        let mut p = Parser::with_document_url("https://example.com/");
        let events = p.feed(b"<link rel=\"stylesheet\" href=\"a.css\"><script>var x = 1;</script>");
        let paused = events.iter().find_map(|e| match e {
            DomEvent::ScriptPaused(p) => Some(p.clone()),
            _ => None,
        });
        let paused = paused.expect("script pauses");
        assert!(
            paused.stylesheets_pending,
            "sheet a.css still pending at the pause"
        );
        assert!(p.stylesheets_pending());
        assert_eq!(
            p.stylesheet_gate().pending(),
            vec!["https://example.com/a.css"]
        );
        // A failed sheet also clears the gate (page renders
        // without it); a loaded one does the same.
        p.note_stylesheet_failed("https://example.com/a.css");
        assert!(!p.stylesheets_pending());
        let _ = p.feed(b"");
        let _ = p.finish();
    }

    #[test]
    fn defer_queue_document_order_before_dom_content_loaded() {
        let mut p = Parser::with_document_url("https://example.com/");
        let events = p.feed(
            b"<script defer src=\"/d1.js\"></script><script type=\"module\" src=\"/m.js\"></script>\
              <script defer src=\"/d2.js\"></script><p>after</p>",
        );
        assert!(
            events
                .iter()
                .all(|e| !matches!(e, DomEvent::ScriptPaused(_))),
            "defer/module must not pause: {events:?}"
        );
        let mut done = p.finish();
        // The DOMContentLoaded firing point: drain the defer queue
        // in document order, execute each, then fire the event.
        // The queue must be empty afterwards.
        let drained = done.script_queues.drain_defer();
        assert_eq!(drained.len(), 3, "defer + module + defer, document order");
        assert_eq!(drained[0].src.as_deref(), Some("https://example.com/d1.js"));
        assert_eq!(drained[1].kind, ScriptKind::Module);
        assert_eq!(drained[1].src.as_deref(), Some("https://example.com/m.js"));
        assert_eq!(drained[2].src.as_deref(), Some("https://example.com/d2.js"));
        assert!(done.script_queues.defer_queue().is_empty());
        // Legacy view agrees with the queue.
        assert_eq!(done.defer_scripts.len(), 3);
    }

    #[test]
    fn async_scripts_arrive_in_arrival_order() {
        let mut p = Parser::with_document_url("https://example.com/");
        let events = p.feed(
            b"<script async src=\"/a.js\"></script><script async src=\"/b.js\"></script><p>x</p>",
        );
        // Discovery is an event; execution order is arrival order,
        // decided by fetch completion, not by the parser.
        let queued: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DomEvent::AsyncScriptQueued { script, src } => Some((*script, src.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(queued.len(), 2);
        let mut done = p.finish();
        let queue = done.script_queues.async_queue();
        assert_eq!(queue.pending_len(), 2);
        assert!(queue.ready().is_empty());
        // b's fetch wins the race: b executes first, even though
        // a was discovered first. No order guarantee is the point.
        let (a, b) = (queued[0].0, queued[1].0);
        let mut queue = std::mem::take(done.script_queues.async_queue_mut());
        assert!(queue.mark_ready(b).is_some());
        assert!(queue.mark_ready(a).is_some());
        let ready = queue.take_ready();
        assert_eq!(ready[0].script, b, "arrival order, not document order");
        assert_eq!(ready[1].script, a);
    }

    #[test]
    fn module_scripts_deduped_by_url() {
        let mut p = Parser::with_document_url("https://example.com/");
        let _ = p.feed(
            b"<script type=\"module\" src=\"/m.js\"></script>\
              <script type=\"module\" src=\"/m.js\"></script>\
              <script type=\"module\" src=\"/other.js\"></script>",
        );
        let done = p.finish();
        // Same URL twice ⇒ one execution; a different URL queues.
        assert_eq!(done.script_queues.defer_queue().len(), 2);
        assert_eq!(done.defer_scripts.len(), 2);
        let urls: Vec<_> = done
            .script_queues
            .defer_queue()
            .scripts()
            .iter()
            .map(|s| s.src.clone().unwrap())
            .collect();
        assert_eq!(
            urls,
            vec!["https://example.com/m.js", "https://example.com/other.js"]
        );
    }

    #[test]
    fn meta_referrer_policy_is_captured() {
        use crate::ReferrerPolicy;
        // Strict policy via meta.
        let mut p = Parser::new();
        let _ = p.feed(b"<meta name=\"referrer\" content=\"strict-origin-when-cross-origin\">");
        assert_eq!(
            p.referrer_policy(),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        let done = p.finish();
        assert_eq!(
            done.referrer_policy,
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );

        // Default policy via meta; last recognised meta wins.
        let mut p = Parser::new();
        let _ = p.feed(
            b"<meta name=\"referrer\" content=\"strict-origin-when-cross-origin\">\
              <meta name=\"referrer\" content=\"no-referrer-when-downgrade\">",
        );
        assert_eq!(
            p.referrer_policy(),
            Some(ReferrerPolicy::NoReferrerWhenDowngrade)
        );
        let _ = p.finish();

        // An unrecognised value is ignored (previous policy kept).
        let mut p = Parser::new();
        let _ = p.feed(
            b"<meta name=\"referrer\" content=\"strict-origin-when-cross-origin\">\
              <meta name=\"referrer\" content=\"unsafe-url\">",
        );
        assert_eq!(
            p.referrer_policy(),
            Some(ReferrerPolicy::StrictOriginWhenCrossOrigin)
        );
        let _ = p.finish();

        // No meta at all: no policy (the default applies net-side).
        let mut p = Parser::new();
        let _ = p.feed(b"<p>hello</p>");
        assert_eq!(p.referrer_policy(), None);
        let done = p.finish();
        assert_eq!(done.referrer_policy, None);

        // A <meta http-equiv="Strict-Transport-Security"> is NOT a
        // referrer policy — and HSTS is never recorded from markup.
        let mut p = Parser::new();
        let _ = p.feed(
            b"<meta http-equiv=\"Strict-Transport-Security\" content=\"max-age=31536000\">",
        );
        assert_eq!(p.referrer_policy(), None);
        let _ = p.finish();
    }

    #[test]
    fn document_write_inserts_at_insertion_point() {
        // The whole document arrives in ONE chunk, so "<p>two</p>"
        // is already buffered in the tokenizer when the script
        // pauses. A write must land BEFORE it, not at the end.
        let mut p = Parser::new();
        let events = p.feed(b"<p>one</p><script>var x = 1;</script><p>two</p>");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DomEvent::ScriptPaused(_))),
            "expected a pause: {events:?}"
        );
        let outcome = p
            .document_write("<b>written</b>")
            .expect("write inserts mid-parse");
        assert_eq!(outcome, WriteOutcome::Inserted);
        let _ = p.feed(b"");
        let done = p.finish();
        let dom = &done.output.dom;
        let body = find_tag(dom, dom.root().unwrap(), "body").unwrap();
        let kids = dom.children(body);
        let tags: Vec<Option<&str>> = kids.iter().map(|&c| dom.tag(c)).collect();
        // The <script> element itself sits between <p>one</p> and
        // the written <b>: the write lands at the insertion point,
        // ahead of the already-buffered "<p>two</p>".
        assert_eq!(
            tags,
            vec![Some("p"), Some("script"), Some("b"), Some("p")],
            "write lands at the insertion point"
        );
        assert_eq!(text_content(dom, kids[2]), "written");
        assert_eq!(text_content(dom, kids[3]), "two");
    }

    #[test]
    fn document_write_sequential_writes_append_in_order() {
        let mut p = Parser::new();
        let _ = p.feed(b"<script>s</script><p>after</p>");
        p.document_write("a").unwrap();
        p.document_write("b").unwrap();
        let _ = p.feed(b"");
        let done = p.finish();
        let dom = &done.output.dom;
        let body = find_tag(dom, dom.root().unwrap(), "body").unwrap();
        assert_eq!(text_content(dom, body), "abafter");
    }

    #[test]
    fn document_open_ignored_during_parsing() {
        let mut p = Parser::new();
        let events = p.feed(b"<script>var x = 1;</script><p>after</p>");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DomEvent::ScriptPaused(_)))
        );
        // Simulate the engine executing the parser-inserted script:
        // nesting level > 0, insertion point defined.
        p.document_writer().enter_script();
        assert_eq!(
            p.document_open(),
            Ok(OpenOutcome::IgnoredDuringParse),
            "open() has no effect from a parser-inserted inline script"
        );
        p.document_writer().exit_script();
        // The throw counter guards open/close/write alike.
        p.document_writer().increment_throw_counter();
        assert_eq!(p.document_open(), Err(DynamicMarkupError));
        assert_eq!(p.document_write("x"), Err(DynamicMarkupError));
        p.document_writer().decrement_throw_counter();
        let _ = p.feed(b"");
        let _ = p.finish();
        // After an abort, writes are ignored; open() starts fresh.
        let mut p = Parser::new();
        let _ = p.feed(b"<p>x</p>");
        p.abort_parser();
        assert!(p.is_aborted());
        assert_eq!(p.document_write("y"), Ok(WriteOutcome::IgnoredAborted));
        assert!(
            p.feed(b"<p>more</p>").is_empty(),
            "aborted parser feeds nothing"
        );
        assert_eq!(p.document_open(), Ok(OpenOutcome::Opened));
    }
}
