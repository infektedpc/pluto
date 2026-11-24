//! Page pipeline (M1.4, extended M2 / Stage 6 application):
//! fetch -> streaming parse -> layout -> paint.
//!
//! Fetch and parse happen once per navigation; layout is cheap and
//! repeatable, so the parsed [`Document`] is retained and re-laid-out
//! whenever the viewport changes — a resize never refetches.
//!
//! The layout viewport is a single stored value owned by the shell:
//! the page area of the window (window minus chrome), reported by the
//! viewport widget. It starts at [`DEFAULT_VIEWPORT`] (the default
//! window size in this libcosmic pin) and is corrected by the first
//! bounds report. There is no fixed layout size anymore — the old
//! 1024x768 constant was the mobile 980-1024px fallback applied to a
//! desktop window, and it left the canvas stopping at 1024px on a
//! maximised window.
//!
//! Errors never crash the app: a failed fetch or layout renders as a
//! real page (an error document run through the same pipeline), and the
//! status line says what happened.
//!
//! # M2 load path (R9 / R18)
//!
//! The document body is fed to the streaming parser
//! ([`pluto_html::Parser`]) in 4 KiB chunks, and the events each feed
//! surfaces drive subresource work while parsing continues:
//!
//! * `StylesheetDiscovered { href }` — the sheet is fetched through
//!   the worker's [`ResourceLoader`] as `Destination::Stylesheet`, in
//!   document order, before the next chunk is fed. The `media`
//!   attribute is honoured as a condition (like `@media`): the sheet
//!   is still fetched, but its rules only apply when the media query
//!   holds at layout time. `@import` inside a fetched sheet resolves
//!   against the *importing sheet's* URL, is fetched serially, and is
//!   inserted in place (its rules land before the importing sheet's
//!   own rules, layer preserved).
//! * Render-blocking: the first display list is not submitted until
//!   this load returns, and it only returns once every
//!   render-blocking stylesheet has loaded or errored — or the
//!   [`RENDER_BLOCKING_TIMEOUT`] (5 s total from the navigation
//!   fetch) fires, after which the page renders with whatever sheets
//!   arrived. A failed or timed-out sheet is skipped, never fatal.
//! * `ScriptPaused` — a classic parser-blocking script: an external
//!   `src` is fetched as `Destination::Script` and its bytes held
//!   (NOT executed — no JS engine until M4/Stage 7); an inline
//!   script's text is held the same way. The parser resumes only
//!   after the fetch completes. `defer` scripts queue in document
//!   order (run after parse — Stage 7, stored only); `async`
//!   scripts are fetched and held. This is the documented M2
//!   simplification: the parser resumes after fetch, scripts are not
//!   executed — do NOT read it as spec behaviour.
//! * `EncodingIndicator` — Pluto decodes UTF-8 lossy only. A UTF-8
//!   label (e.g. `<meta charset="utf-8">`) restarts the parse once
//!   from the buffered prefix (the restart path is exercised, and
//!   already-fetched sheets/scripts are deduped, never refetched).
//!   A non-UTF-8 label is logged and the parse continues lossy —
//!   honouring it needs a real decoder (an encoding crate), which is
//!   a documented follow-up, not done here.
//! * The preload scanner runs over the fetched body before parsing
//!   and fires off fetches for discovered resources through the
//!   loader; dedup is the Stage 5 scheduler's job (same
//!   URL + destination + credentials + top-level site coalesce), so
//!   scanner and parser fetches never double-download. Scanning the
//!   whole body post-fetch is the M2 simplification: true network
//!   streaming (`fetch_stream` -> feed chunks as they arrive) is the
//!   follow-up.
//!
//! Stylesheet arrival is a full restyle (M2 rule): the merged sheet
//! is rebuilt from the document's style sources in document order —
//! UA sheet underneath (applied by layout), then inline `<style>` and
//! external sheets interleaved in document order — and the document
//! is re-laid-out from it. [`Document`] therefore carries its
//! `base_url` (the final URL after redirects), which relative URLs
//! resolve against.

use pluto_css::{MediaCondition, Stylesheet};
use pluto_html::DomEvent;
use pluto_html::{
    Dom, NodeId, Parser, resolve_url, scan as preload_scan,
};
use pluto_html::{Destination as HtmlDestination, Priority as HtmlPriority};
use pluto_ipc::{
    CredentialsMode, Destination, Priority, RequestMode, ResourceRequest, origin_of,
};
use pluto_layout::{ElementState, LayoutTree};
use pluto_net::ResourceLoader;
use pluto_paint::DisplayList;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Startup guess for the layout viewport: the default window size (no
/// `Settings::size()` in this libcosmic pin, so the window opens
/// 1024x768). The viewport widget reports its real bounds on the first
/// frame and the shell corrects this; it is never used as a layout
/// size after that.
pub const DEFAULT_VIEWPORT: (f32, f32) = (1024.0, 768.0);

/// Total render-blocking budget for external stylesheets, from the
/// navigation fetch starting. When it fires, the page renders with
/// whatever sheets arrived; a sheet still in flight is skipped (its
/// late arrival does NOT trigger a restyle in M2 — a later navigation
/// or resize re-runs the load path). 5 s, matching the rough
/// browser convention of not holding first paint hostage to CSS.
pub const RENDER_BLOCKING_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum `@import` nesting followed while loading stylesheets.
/// Cycles and pathological chains stop here instead of looping.
pub const MAX_IMPORT_DEPTH: usize = 5;

/// Bytes per streaming-parser feed: the first chunk of a real
/// document is typically ~14 KiB (TCP initcwnd); 4 KiB feeds exercise
/// the chunked path without drowning the event loop.
pub const FEED_CHUNK_BYTES: usize = 4096;

/// A parsed page: the DOM plus its merged author stylesheet, the base
/// URL relative URLs resolve against, and the scripts the parse held
/// (fetched, never executed — Stage 7 runs them). Retained across
/// resizes so layout can re-run without refetching.
#[derive(Debug, Clone)]
pub struct Document {
    dom: Dom,
    sheet: Stylesheet,
    base_url: String,
    /// `defer` (and `type="module"`) scripts, in document order: run
    /// after the parse, before `DOMContentLoaded` (Stage 7).
    pub defer_scripts: Vec<NodeId>,
    /// `async` scripts, in document order: fetched and held (Stage 7
    /// executes each as soon as its fetch completes).
    pub async_scripts: Vec<NodeId>,
    /// Scripts fetched (external) or held (inline) during the parse,
    /// in the order their pauses / discoveries fired. Stage 7 matches
    /// these to `defer_scripts` / `async_scripts` by node.
    pub held_scripts: Vec<HeldScript>,
}

impl Document {
    pub fn dom(&self) -> &Dom {
        &self.dom
    }
}

/// One script the streaming parse held: fetched bytes for an external
/// script, the element text for an inline one. Not executed in M2.
#[derive(Debug, Clone)]
pub struct HeldScript {
    /// The `<script>` element in the document.
    pub node: NodeId,
    /// Resolved `src`, or `None` for an inline script.
    pub src: Option<String>,
    /// Fetched bytes (external) or element text (inline).
    pub bytes: Vec<u8>,
}

/// One style source in document order: an inline `<style>` block, or
/// an external `<link rel="stylesheet">` whose CSS arrives later.
#[derive(Debug, Clone)]
enum StyleSource {
    Inline {
        css: String,
        media: Option<String>,
    },
    External {
        href: String,
        media: Option<String>,
    },
}

/// The page loaded at startup and when the address bar is empty.
pub const DEFAULT_URL: &str = "https://example.com";

/// Turn address-bar input into a fetchable URL. A bare host gets
/// `https://`; anything with a scheme is left alone.
pub fn normalize_url(input: &str) -> String {
    let input = input.trim();
    if input.is_empty() {
        return DEFAULT_URL.to_string();
    }
    if input.contains("://") {
        return input.to_string();
    }
    format!("https://{input}")
}

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> u64 {
    NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

/// A subresource ask against `base_url`'s origin: classic
/// (`NoCors`) mode, same-origin credentials — the same shape the
/// preload scanner's fetches use, so the Stage 5 scheduler dedups
/// scanner-discovered and parser-discovered fetches of one URL into
/// a single download (dedup key: URL + destination + credentials +
/// top-level site).
fn subresource_request(
    url: &str,
    destination: Destination,
    priority: Priority,
    base_url: &str,
) -> ResourceRequest {
    let origin = origin_of(base_url);
    ResourceRequest {
        request_id: next_request_id(),
        tab_id: 0,
        epoch: 0,
        url: url.to_string(),
        destination,
        priority,
        mode: RequestMode::NoCors,
        credentials: CredentialsMode::SameOrigin,
        origin: origin.clone(),
        top_level_site: origin,
        upgrade_insecure_requests: false,
    }
}

fn map_html_destination(d: HtmlDestination) -> Destination {
    match d {
        HtmlDestination::Document => Destination::Document,
        HtmlDestination::Stylesheet => Destination::Stylesheet,
        HtmlDestination::Script => Destination::Script,
        HtmlDestination::Image => Destination::Image,
        HtmlDestination::Font => Destination::Font,
        HtmlDestination::Fetch => Destination::Fetch,
        HtmlDestination::Other => Destination::Other,
    }
}

fn map_html_priority(p: HtmlPriority) -> Priority {
    match p {
        HtmlPriority::VeryHigh => Priority::VeryHigh,
        HtmlPriority::High => Priority::High,
        HtmlPriority::Medium => Priority::Medium,
        HtmlPriority::Low => Priority::Low,
        HtmlPriority::VeryLow => Priority::VeryLow,
    }
}

/// Fetch `url` (owned, so the future is `'static` for `Task::perform`)
/// and stream-parse the HTML. Layout happens separately in
/// [`layout_document_with_state`], at the shell's current viewport, so a resize
/// never refetches. Returns the parsed document and the final URL
/// after redirects. `loader` is the worker's loader (one per worker,
/// shared client + scheduler — Stage 5): the old fresh-loader-per-
/// navigation defeated pooling and resumption, so it is gone.
/// Cancelling `cancel` aborts the fetch (Stage 5 #5 — the worker
/// cancels the old navigation's token when a new one commits). The
/// request is a top-level navigation (VeryHigh priority, Document
/// destination) with the page's own URL as its origin, so the
/// mixed-content gate never blocks it.
///
/// The body is fed to the streaming parser in [`FEED_CHUNK_BYTES`]
/// chunks; stylesheet/script events are handled as they surface (see
/// the module docs). The old page stays visible until the new list
/// lands — this function only returns the document; submission is
/// the worker's job.
///
/// Logs each stage to stderr: the terminal is the only window into a
/// load that stalls (no logging framework is wired up yet).
pub async fn load_page_cancellable(
    loader: &ResourceLoader,
    url: String,
    cancel: CancellationToken,
) -> Result<(Document, String), String> {
    let request = ResourceRequest::navigate(next_request_id(), 0, 0, &url);
    eprintln!("[pluto] fetching {url}");
    let response = match loader.fetch_with(request, cancel.clone()).await {
        Ok(response) => response,
        Err(e) => {
            eprintln!("[pluto] fetch failed: {e}");
            return Err(e.to_string());
        }
    };
    let final_url = response.final_url.to_string();
    eprintln!(
        "[pluto] fetched {} ({} bytes), parsing",
        final_url,
        response.body.len()
    );
    // Preload scan (R4): run over the whole fetched body before
    // parsing and fire off fetches for what it finds. The scheduler
    // dedups these against the parser-driven fetches below. True
    // network streaming (feed chunks as they arrive) is the
    // follow-up — see the module docs.
    fire_preloads(loader, &response.body, &final_url, cancel.clone());
    let mut load = StreamingLoad::new(loader, cancel.clone(), final_url.clone());
    for chunk in response.body.chunks(FEED_CHUNK_BYTES) {
        if cancel.is_cancelled() {
            return Err("navigation cancelled".to_string());
        }
        load.drive(chunk).await?;
    }
    if cancel.is_cancelled() {
        return Err("navigation cancelled".to_string());
    }
    let document = load.finish_document();
    // Stage 7 reads these: the parse held the scripts (fetched,
    // never executed) and queued defer/async in document order.
    // Logged, not executed — there is no JS engine yet.
    for held in &document.held_scripts {
        eprintln!(
            "[pluto] held script node {:?} src {:?} ({} bytes)",
            held.node,
            held.src,
            held.bytes.len()
        );
    }
    eprintln!(
        "[pluto] parsed {} ({} defer, {} async scripts held)",
        final_url,
        document.defer_scripts.len(),
        document.async_scripts.len()
    );
    Ok((document, final_url))
}

/// Fire-and-forget fetches for the preload scanner's discoveries.
/// Each runs on its own task and its bytes are discarded: the point
/// is warming the scheduler / connection so the parser-driven fetch
/// that follows dedups onto an in-flight (or just-completed) request
/// instead of starting cold. `data:` URLs are never fetched (the
/// scanner already skips most of them; this is the belt).
fn fire_preloads(
    loader: &ResourceLoader,
    body: &[u8],
    base_url: &str,
    cancel: CancellationToken,
) {
    for found in preload_scan(body, base_url) {
        if found.url.trim_start().starts_with("data:") {
            continue;
        }
        let loader = loader.clone();
        let cancel = cancel.clone();
        let request = subresource_request(
            &found.url,
            map_html_destination(found.destination),
            map_html_priority(found.priority),
            base_url,
        );
        tokio::spawn(async move {
            let _ = loader.fetch_with(request, cancel).await;
        });
    }
}

/// The streaming load's mutable state: the parser, the caches that
/// survive an encoding restart (so nothing is refetched), and the
/// render-blocking deadline.
struct StreamingLoad<'a> {
    loader: &'a ResourceLoader,
    cancel: CancellationToken,
    base_url: String,
    parser: Parser,
    /// Fetched stylesheet text by absolute URL (includes `@import`
    /// targets, fetched recursively when their parent sheet loaded).
    sheet_cache: HashMap<String, String>,
    /// URLs already attempted (loaded or failed): a re-feed after an
    /// encoding restart re-discovers them, and must not refetch.
    sheet_attempted: HashSet<String>,
    /// Fetched script bytes by resolved `src`.
    script_bytes: HashMap<String, Vec<u8>>,
    /// Classic parser-blocking `<script>` nodes, in pause order.
    paused_scripts: Vec<NodeId>,
    /// Render-blocking deadline (navigation start + timeout).
    blocking_deadline: tokio::time::Instant,
    /// Whether the one encoding restart has already happened.
    restarted: bool,
    /// Whether the unsupported-encoding note has been logged once.
    encoding_logged: bool,
}

impl<'a> StreamingLoad<'a> {
    fn new(loader: &'a ResourceLoader, cancel: CancellationToken, base_url: String) -> Self {
        Self {
            loader,
            cancel,
            base_url: base_url.clone(),
            parser: Parser::with_document_url(&base_url),
            sheet_cache: HashMap::new(),
            sheet_attempted: HashSet::new(),
            script_bytes: HashMap::new(),
            paused_scripts: Vec::new(),
            blocking_deadline: tokio::time::Instant::now() + RENDER_BLOCKING_TIMEOUT,
            restarted: false,
            encoding_logged: false,
        }
    }

    /// Feed one chunk and handle every event it surfaces, resuming
    /// past script pauses (feed empty to drain) until the chunk is
    /// fully consumed. A UTF-8 encoding indicator restarts the parse
    /// from the buffered prefix (once): the parser is swapped for a
    /// fresh one and the prefix is re-fed as the new input, so the
    /// caller must not feed the chunk's bytes again. (A single loop,
    /// not `drive`/`restart` recursion — async fns cannot recurse
    /// without boxing.)
    async fn drive(&mut self, chunk: &[u8]) -> Result<(), String> {
        let mut input: Vec<u8> = chunk.to_vec();
        'outer: loop {
            let mut next: Option<Vec<u8>> = Some(std::mem::take(&mut input));
            while let Some(bytes) = next.take() {
                if self.cancel.is_cancelled() {
                    return Err("navigation cancelled".to_string());
                }
                let events = self.parser.feed(&bytes);
                let mut resume = false;
                for event in events {
                    match event {
                        DomEvent::StylesheetDiscovered { href, .. } => {
                            self.fetch_stylesheet(&href).await;
                        }
                        DomEvent::ResourceDiscovered(found) => {
                            // Scripts are fetched + held (async/defer, and
                            // classic ones the pause below also reports).
                            // Everything else the preload scanner already
                            // fired off; the scheduler dedups the rest.
                            if found.destination == HtmlDestination::Script
                                && !found.url.trim_start().starts_with("data:")
                            {
                                self.fetch_script(&found.url, Priority::Medium).await;
                            }
                        }
                        DomEvent::ScriptPaused(paused) => {
                            self.paused_scripts.push(paused.script);
                            if let Some(src) = &paused.src
                                && !src.trim_start().starts_with("data:")
                            {
                                self.fetch_script(src, Priority::High).await;
                            }
                            // Inline scripts hold nothing to fetch; the
                            // pause still ends here and parsing resumes.
                            resume = true;
                        }
                        DomEvent::EncodingIndicator { label }
                            if self.maybe_restart(&label) =>
                        {
                            input = self.parser.buffered_prefix().to_vec();
                            self.parser = Parser::with_document_url(&self.base_url);
                            continue 'outer;
                        }
                        _ => {}
                    }
                }
                if resume {
                    next = Some(Vec::new());
                }
            }
            break;
        }
        Ok(())
    }

    /// Fetch a stylesheet and, serially, its `@import` chain (each
    /// import resolved against its importing sheet's URL). Results
    /// land in `sheet_cache`; failures leave nothing behind, and the
    /// page renders without that sheet (render-blocking ends on
    /// error, per the module docs). Bounded by the render-blocking
    /// deadline: a sheet still in flight when it fires is skipped.
    /// The link's `media` attribute is NOT consulted here — it is
    /// read from the finished DOM at merge time and wraps the
    /// sheet's rules, so layout evaluates it per viewport.
    async fn fetch_stylesheet(&mut self, href: &str) {
        if href.trim_start().starts_with("data:") {
            return;
        }
        // Worklist (iterative, not recursive): imports resolve
        // against the sheet that imported them, depth-capped.
        let mut stack = vec![(href.to_string(), 0usize)];
        while let Some((url, depth)) = stack.pop() {
            if depth > MAX_IMPORT_DEPTH {
                eprintln!("[pluto] @import depth exceeded at {url}, skipping");
                continue;
            }
            if !self.sheet_attempted.insert(url.clone()) {
                continue;
            }
            let Some(text) = self
                .fetch_text(&url, Destination::Stylesheet, Priority::High)
                .await
            else {
                continue;
            };
            let sheet = Stylesheet::parse(&text);
            for import in sheet.imports.iter().rev() {
                if import.url.trim_start().starts_with("data:") {
                    continue;
                }
                stack.push((resolve_url(&url, &import.url), depth + 1));
            }
            self.sheet_cache.insert(url, text);
        }
    }

    /// Fetch one URL as text, inside the render-blocking deadline.
    /// `None` on any failure or when the deadline fires.
    async fn fetch_text(
        &self,
        url: &str,
        destination: Destination,
        priority: Priority,
    ) -> Option<String> {
        let request = subresource_request(url, destination, priority, &self.base_url);
        let fetch = self.loader.fetch_with(request, self.cancel.clone());
        match tokio::time::timeout_at(self.blocking_deadline, fetch).await {
            Ok(Ok(response)) => Some(response.text()),
            Ok(Err(e)) => {
                eprintln!("[pluto] subresource fetch failed ({url}): {e}");
                None
            }
            Err(_) => {
                eprintln!("[pluto] render-blocking timeout, skipping stylesheet {url}");
                None
            }
        }
    }

    /// Fetch a script's bytes and hold them (never execute). Deduped
    /// by `src`: a classic script's `ResourceDiscovered` and its
    /// `ScriptPaused` both ask, but only the first downloads.
    /// Script fetches are NOT render-blocking — the scheduler's own
    /// timeouts bound them, so no deadline applies here.
    async fn fetch_script(&mut self, src: &str, priority: Priority) {
        if self.script_bytes.contains_key(src) {
            return;
        }
        let request = subresource_request(src, Destination::Script, priority, &self.base_url);
        match self.loader.fetch_with(request, self.cancel.clone()).await {
            Ok(response) => {
                self.script_bytes.insert(src.to_string(), response.body);
            }
            Err(e) => {
                eprintln!("[pluto] script fetch failed ({src}): {e}");
                // Record the miss so a re-feed after an encoding
                // restart does not retry a failing URL forever.
                self.script_bytes.insert(src.to_string(), Vec::new());
            }
        }
    }

    /// Decide an encoding restart. Only a UTF-8 label restarts, and
    /// only once: the re-feed fires the indicator again, and a second
    /// restart would loop. A non-UTF-8 label is logged once and the
    /// parse continues lossy UTF-8 (documented follow-up: honouring
    /// it needs an encoding crate).
    fn maybe_restart(&mut self, label: &str) -> bool {
        let normalised = label.trim().to_ascii_lowercase();
        if normalised == "utf-8" || normalised == "utf8" {
            if !self.restarted {
                self.restarted = true;
                eprintln!("[pluto] encoding indicator ({label}): restarting parse from buffered prefix");
                return true;
            }
            return false;
        }
        if !self.encoding_logged {
            self.encoding_logged = true;
            eprintln!(
                "[pluto] encoding indicator ({label}): non-UTF-8 unsupported, continuing lossy UTF-8 (follow-up: encoding crate)"
            );
        }
        false
    }

    /// End of input: finish the parse and assemble the [`Document`]
    /// — merged sheet from the finished DOM's style sources in
    /// document order (imports expanded from the fetch cache), plus
    /// the held scripts matched to their nodes.
    fn finish_document(mut self) -> Document {
        let parser = std::mem::replace(
            &mut self.parser,
            Parser::with_document_url(&self.base_url),
        );
        let parsed = parser.finish();
        let dom = parsed.output.dom;
        let sources = collect_style_sources(&dom);
        let sheet = merge_sources(&sources, &self.sheet_cache, &self.base_url);
        let held_scripts = held_scripts(
            &dom,
            &self.base_url,
            &parsed.defer_scripts,
            &parsed.async_scripts,
            &self.paused_scripts,
            &self.script_bytes,
        );
        Document {
            dom,
            sheet,
            base_url: self.base_url,
            defer_scripts: parsed.defer_scripts,
            async_scripts: parsed.async_scripts,
            held_scripts,
        }
    }
}

/// Match held script bytes to their nodes: every paused classic
/// script plus the defer/async queues, in first-seen order. An
/// external script's bytes come from the fetch cache (empty when its
/// fetch failed); an inline script's bytes are its element text.
/// `src` is stored resolved against the document base, matching the
/// fetch cache's keys.
fn held_scripts(
    dom: &Dom,
    base_url: &str,
    defer: &[NodeId],
    async_scripts: &[NodeId],
    paused: &[NodeId],
    bytes_by_src: &HashMap<String, Vec<u8>>,
) -> Vec<HeldScript> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut out = Vec::new();
    for &node in paused.iter().chain(defer.iter()).chain(async_scripts.iter()) {
        if !seen.insert(node) {
            continue;
        }
        let src = dom
            .attr(node, "src")
            .filter(|s| !s.trim().is_empty())
            .map(|s| resolve_url(base_url, s));
        let bytes = match &src {
            Some(src) => bytes_by_src.get(src).cloned().unwrap_or_default(),
            None => script_text(dom, node).into_bytes(),
        };
        out.push(HeldScript {
            node,
            src,
            bytes,
        });
    }
    out
}

/// The text content of a `<script>` element (its inline source).
fn script_text(dom: &Dom, id: NodeId) -> String {
    let mut text = String::new();
    for &child in dom.children(id) {
        if let Some(t) = dom.text(child) {
            text.push_str(t);
        }
    }
    text
}

/// Parse HTML and collect its `<style>` rules into a [`Document`].
/// External `<link rel="stylesheet">` has no fetched CSS on this
/// path, so it contributes nothing to the merged sheet (same as the
/// old M1 behaviour). The base URL is empty: relative URLs resolve
/// against nothing. For a loaded page, use
/// [`parse_document_with_base`] or the streaming load path, which
/// fills external sheets from the fetch cache.
pub fn parse_document(html: &str) -> Document {
    parse_document_with_base(html, "")
}

/// Like [`parse_document`], with the base URL relative URLs resolve
/// against (the document's final URL after redirects).
pub fn parse_document_with_base(html: &str, base_url: &str) -> Document {
    let dom = pluto_html::parse(html);
    let sources = collect_style_sources(&dom);
    let sheet = merge_sources(&sources, &HashMap::new(), base_url);
    Document {
        dom,
        sheet,
        base_url: base_url.to_string(),
        defer_scripts: Vec::new(),
        async_scripts: Vec::new(),
        held_scripts: Vec::new(),
    }
}

/// Lay `doc` out at `viewport` (CSS px, logical — the page area of
/// the window) and build the display list, with dynamic pseudo-class
/// state (`:hover` / `:active` / `:focus`) for this pass. The canvas
/// background is painted over the whole viewport by the display list,
/// so a short page on a wide window is covered to the edge. A layout
/// failure renders as an error page at the same viewport instead of
/// an empty viewport. The returned [`LayoutTree`] is `!Send` (it owns
/// a `FontSystem`), so it never crosses an await: the renderer's
/// layout-actor thread owns it (see renderer.rs).
pub fn layout_document_with_state(
    doc: &Document,
    viewport: (f32, f32),
    state: &ElementState,
) -> (Arc<DisplayList>, LayoutTree) {
    match LayoutTree::build_with_state(&doc.dom, &doc.sheet, viewport.0, viewport.1, state) {
        Ok(tree) => (Arc::new(DisplayList::build(&tree)), tree),
        Err(e) => {
            let err_doc = parse_document(&error_document(
                "about:blank",
                &format!("layout failed: {e}"),
            ));
            let tree = LayoutTree::build_with_state(
                &err_doc.dom,
                &err_doc.sheet,
                viewport.0,
                viewport.1,
                state,
            )
            .unwrap_or_default();
            (Arc::new(DisplayList::build(&tree)), tree)
        }
    }
}

/// Walk `id` and its ancestors for the nearest `<a href>`, and
/// resolve the href against the document's base URL. The click target
/// the worker's hit-test hands back is usually the link itself, but a
/// click on a child (an `<img>` or `<span>` inside the link) must
/// still navigate — hence the ancestor walk.
pub fn find_link_href(doc: &Document, mut id: NodeId) -> Option<String> {
    loop {
        if doc.dom.tag(id) == Some("a") {
            if let Some(href) = doc.dom.attr(id, "href")
                && !href.trim().is_empty()
            {
                return Some(resolve_url(&doc.base_url, href));
            }
            return None;
        }
        let parent = doc.dom.get(id).and_then(|n| n.parent)?;
        id = parent;
    }
}

/// Collect the document's style sources in document order: inline
/// `<style>` text (with its `media` attribute, if any) and external
/// `<link rel="stylesheet">` (href + media). `data:` hrefs and links
/// without an href contribute nothing.
fn collect_style_sources(dom: &Dom) -> Vec<StyleSource> {
    let mut sources = Vec::new();
    if let Some(root) = dom.root() {
        collect_style_sources_from(dom, root, &mut sources);
    }
    sources
}

fn collect_style_sources_from(dom: &Dom, id: NodeId, out: &mut Vec<StyleSource>) {
    if dom.tag(id) == Some("style") {
        let mut css = String::new();
        for &child in dom.children(id) {
            if let Some(text) = dom.text(child) {
                css.push_str(text);
                css.push('\n');
            }
        }
        let media = dom.attr(id, "media").map(str::to_string);
        out.push(StyleSource::Inline { css, media });
        return;
    }
    if dom.tag(id) == Some("link") {
        let rel = dom.attr(id, "rel").unwrap_or("");
        let is_stylesheet = rel
            .split_whitespace()
            .any(|r| r.eq_ignore_ascii_case("stylesheet"));
        if is_stylesheet
            && let Some(href) = dom.attr(id, "href")
            && !href.trim().is_empty()
            && !href.trim_start().starts_with("data:")
        {
            let media = dom.attr(id, "media").map(str::to_string);
            out.push(StyleSource::External {
                href: href.to_string(),
                media,
            });
        }
        // A <link> has no element children worth walking, but its
        // siblings do — fall through to the child walk below.
    }
    for &child in dom.children(id) {
        collect_style_sources_from(dom, child, out);
    }
}

/// Merge style sources into one author stylesheet, in document order.
/// Each source's rules get a global `source_order` (0, 1, 2, … across
/// the merge, so document order holds across sheets — the layout
/// index offsets UA rules past nothing and author rules by this).
/// External sheets expand their `@import`s from `sheet_cache`
/// (fetched at load time, resolved against the importing sheet's URL,
/// inserted in place before the importing sheet's own rules, layer
/// preserved). An import with no cached text was never fetched
/// (failed, timed out, or over the depth cap) and is skipped.
/// A source's `media` condition wraps its rules (combined with any
/// `@media` on the rule itself), so layout evaluates it per viewport
/// instead of the loader deciding once.
fn merge_sources(
    sources: &[StyleSource],
    sheet_cache: &HashMap<String, String>,
    base_url: &str,
) -> Stylesheet {
    let mut merged = Stylesheet::default();
    let mut order = 0u32;
    for source in sources {
        match source {
            StyleSource::Inline { css, media } => {
                expand_sheet(
                    css,
                    media.as_deref(),
                    None,
                    None,
                    Some(base_url),
                    sheet_cache,
                    &mut merged,
                    &mut order,
                    0,
                );
            }
            StyleSource::External { href, media } => {
                // `href` is raw (as written in the document); the
                // fetch cache is keyed by resolved absolute URL.
                let resolved = resolve_url(base_url, href);
                if let Some(css) = sheet_cache.get(&resolved) {
                    let css = css.clone();
                    expand_sheet(
                        &css,
                        media.as_deref(),
                        None,
                        None,
                        Some(&resolved),
                        sheet_cache,
                        &mut merged,
                        &mut order,
                        0,
                    );
                }
                // No cached text: the sheet failed to load, timed out,
                // or this document never went through the load path
                // (parse_document) — the page renders without it.
            }
        }
    }
    merged
}

/// Parse one sheet's CSS into `merged`, expanding its `@import`s from
/// the cache first (in place, before its own rules). `media` /
/// `layer` / `supports` are the conditions the sheet was imported
/// under (or the `<link>` / `<style>` media); they wrap every rule
/// the sheet contributes. `sheet_url` is what the sheet's own
/// relative `@import` URLs resolve against.
#[allow(clippy::too_many_arguments)]
fn expand_sheet(
    css: &str,
    media: Option<&str>,
    layer: Option<&str>,
    supports: Option<&str>,
    sheet_url: Option<&str>,
    sheet_cache: &HashMap<String, String>,
    merged: &mut Stylesheet,
    order: &mut u32,
    depth: usize,
) {
    let sheet = Stylesheet::parse(css);
    if depth <= MAX_IMPORT_DEPTH {
        for import in &sheet.imports {
            if import.url.trim_start().starts_with("data:") {
                continue;
            }
            let base = sheet_url.unwrap_or("");
            let resolved = resolve_url(base, &import.url);
            if let Some(imported) = sheet_cache.get(&resolved) {
                let imported = imported.clone();
                expand_sheet(
                    &imported,
                    import.media.as_deref().or(media),
                    import.layer.as_deref().or(layer),
                    import.supports.as_deref().or(supports),
                    Some(&resolved),
                    sheet_cache,
                    merged,
                    order,
                    depth + 1,
                );
            }
            // Uncached import: never fetched — skipped, per merge_sources.
        }
    }
    for mut rule in sheet.rules {
        if let Some(condition) = combine_media(media, rule.media.take()) {
            rule.media = Some(condition);
        }
        if rule.layer.is_none() {
            rule.layer = layer.map(str::to_string);
        }
        if rule.supports.is_none() {
            rule.supports = supports.map(str::to_string);
        }
        rule.source_order = *order;
        *order += 1;
        merged.rules.push(rule);
    }
    for layer_name in sheet.layers {
        if !merged.layers.contains(&layer_name) {
            merged.layers.push(layer_name);
        }
    }
    merged.font_faces.extend(sheet.font_faces);
    merged.keyframes.extend(sheet.keyframes);
    merged.errors.extend(sheet.errors);
}

/// Combine an outer media condition (from `<link media>` / `<style
/// media>` / `@import`) with a rule's own `@media`: both must hold.
/// Empty conditions are no condition at all (`media=""` means "all").
fn combine_media(outer: Option<&str>, inner: Option<MediaCondition>) -> Option<MediaCondition> {
    let outer = outer.map(str::trim).filter(|s| !s.is_empty());
    match outer {
        None => inner,
        Some(outer) => {
            let outer = MediaCondition::parse(outer);
            match inner {
                None => Some(outer),
                Some(inner) => Some(MediaCondition::And(vec![outer, inner])),
            }
        }
    }
}

/// A small error page, rendered through the same pipeline as a real one.
pub fn error_document(url: &str, err: &str) -> String {
    format!(
        r#"<!doctype html>
<html>
<head>
    <title>Pluto: could not load page</title>
    <style type="text/css">
    body {{
        background-color: #ffffff;
        margin: 0;
        padding: 2em;
    }}
    h1 {{
        font-size: 24px;
    }}
    p {{
        font-size: 16px;
    }}
    </style>
</head>
<body>
    <h1>Could not load {url}</h1>
    <p>{err}</p>
    <p><a href="https://example.com">Go to example.com</a></p>
</body>
</html>"#,
        url = escape_html(url),
        err = escape_html(err),
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_paint::DisplayItem;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
    }
    a:link, a:visited {
        color: #38488f;
    }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>"#;

    /// Parse HTML and lay it out at `viewport` (test helper — the
    /// production path lays out on the layout-actor thread in
    /// renderer.rs, not here).
    fn display_list_for(html: &str, viewport: (f32, f32)) -> (Arc<DisplayList>, Document) {
        let doc = parse_document(html);
        let display = layout_for(&doc, viewport);
        (display, doc)
    }

    fn layout_for(doc: &Document, viewport: (f32, f32)) -> Arc<DisplayList> {
        layout_document_with_state(doc, viewport, &ElementState::default()).0
    }

    #[test]
    fn normalize_url_cases() {
        assert_eq!(normalize_url("example.com"), "https://example.com");
        assert_eq!(
            normalize_url("https://example.com"),
            "https://example.com"
        );
        assert_eq!(normalize_url("http://localhost:8080/"), "http://localhost:8080/");
        assert_eq!(normalize_url("  example.com  "), "https://example.com");
        assert_eq!(normalize_url(""), DEFAULT_URL);
    }

    #[test]
    fn build_display_list_example_com() {
        let (list, _doc) = display_list_for(EXAMPLE_COM, (1024.0, 768.0));
        assert!(!list.is_empty());
        // Canvas background first, then the div fill, then glyph runs.
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                height: 768.0,
                ..
            }
        ));
        assert!(list.0.iter().any(|i| matches!(
            i,
            DisplayItem::Glyphs { glyphs, .. } if !glyphs.is_empty()
        )));
    }

    /// The live example.com page as served 2026-11-23 (577 bytes):
    /// the paragraph wraps at the viewport width (its `max-width`
    /// is beyond M1, so the body is full-width). Regression for the
    /// resize fix: the canvas must cover the viewport at any size,
    /// and the text must rewrap when the viewport narrows.
    const NEW_EXAMPLE_COM: &str = "<!doctype html><html lang=en><head><meta charset=utf-8><link rel=icon href=data:,><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>Example Domain</title><style>html{color-scheme:light dark;background:light-dark(#eee,#222)}body{font:16px/1.6 system-ui,sans-serif;max-width:26em;margin:auto;padding:25vh 2em 2em;text-align:center}</style></head><body><p>This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.</p><script src=/s.js></script></body></html>";

    fn glyph_max_y(list: &DisplayList) -> f32 {
        list.0.iter().fold(0.0f32, |acc, item| match item {
            DisplayItem::Glyphs { glyphs, .. } => glyphs.iter().fold(acc, |a, g| a.max(g.y)),
            _ => acc,
        })
    }

    #[test]
    fn display_list_reflows_with_viewport_width() {
        let (wide, _doc) = display_list_for(NEW_EXAMPLE_COM, (1024.0, 768.0));
        let (narrow, _doc) = display_list_for(NEW_EXAMPLE_COM, (400.0, 768.0));
        // The canvas covers each viewport exactly — no 1024px seam
        // on a wide window, no shortfall on a narrow one.
        for (list, w) in [(&wide, 1024.0), (&narrow, 400.0)] {
            assert!(matches!(
                list.0[0],
                DisplayItem::FillRect {
                    width,
                    height: 768.0,
                    ..
                } if width == w
            ));
        }
        // Same text in both, but the narrow viewport wraps it into
        // more lines, so the lowest glyph sits lower — and each wrap
        // consumes the space at the break, so the narrow layout has
        // (slightly) fewer glyph entries, not the same count.
        let glyph_total = |list: &DisplayList| {
            list.0.iter().fold(0, |acc, item| match item {
                DisplayItem::Glyphs { glyphs, .. } => acc + glyphs.len(),
                _ => acc,
            })
        };
        // 154 at this viewport: 132 non-space glyphs (constant across
        // wraps) + 22 surviving spaces — `max-width: 26em` is honoured
        // now, so the paragraph wraps at 416px and two spaces are
        // consumed at line breaks (the old 155 predates max-width).
        // A range, not an exact wrap-dependent count: any other wrap
        // still lands near 132 non-space glyphs.
        let wide_total = glyph_total(&wide);
        assert!(
            (150..=160).contains(&wide_total),
            "wide glyph total {wide_total} outside the expected wrap range"
        );
        assert!(glyph_total(&narrow) < glyph_total(&wide));
        assert!(
            glyph_max_y(&narrow) > glyph_max_y(&wide),
            "narrow layout should wrap into more lines"
        );
    }

    #[test]
    fn layout_at_same_viewport_is_deterministic() {
        // Re-laying out at the same size must produce the identical
        // display list — so the shell's skip-if-unchanged is safe,
        // and a redundant re-layout would be invisible anyway.
        let doc = parse_document(NEW_EXAMPLE_COM);
        let first = layout_for(&doc, (1024.0, 768.0));
        let second = layout_for(&doc, (1024.0, 768.0));
        assert_eq!(first.0.len(), second.0.len());
        for (a, b) in first.0.iter().zip(second.0.iter()) {
            match (a, b) {
                (
                    DisplayItem::FillRect {
                        x: ax,
                        y: ay,
                        width: aw,
                        height: ah,
                        color: ac,
                    },
                    DisplayItem::FillRect {
                        x: bx,
                        y: by,
                        width: bw,
                        height: bh,
                        color: bc,
                    },
                ) => {
                    assert_eq!((*ax, *ay, *aw, *ah, *ac), (*bx, *by, *bw, *bh, *bc));
                }
                (
                    DisplayItem::Glyphs {
                        color: ac,
                        glyphs: ag,
                    },
                    DisplayItem::Glyphs {
                        color: bc,
                        glyphs: bg,
                    },
                ) => {
                    assert_eq!(ac, bc);
                    assert_eq!(ag.len(), bg.len());
                    for (g, h) in ag.iter().zip(bg.iter()) {
                        assert_eq!((g.x, g.y), (h.x, h.y));
                        assert_eq!(g.key, h.key);
                    }
                }
                _ => panic!("display list items differ between identical layouts"),
            }
        }
    }

    #[test]
    fn error_document_escapes() {
        let doc = error_document("https://x/?a=1&b=2", "oops <b>bold</b>");
        assert!(doc.contains("https://x/?a=1&amp;b=2"));
        assert!(doc.contains("oops &lt;b&gt;bold&lt;/b&gt;"));
        assert!(!doc.contains("<b>bold</b>"));
        // And the escaped doc still builds a display list.
        assert!(!display_list_for(&doc, (1024.0, 768.0)).0.is_empty());
    }

    /// Local server returning the example.com document, same pattern as
    /// pluto-net's tests. Proves fetch -> parse -> layout -> paint
    /// end to end without the internet (hyper has no proxy support).
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let _ = n;
                let body = EXAMPLE_COM;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    #[tokio::test]
    async fn load_page_from_local_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let (doc, final_url) =
            load_page_cancellable(&loader, url.clone(), CancellationToken::new())
                .await
                .expect("page loads");
        assert_eq!(final_url, url);
        // Layout at a non-default viewport: the viewport flows
        // through the fetch path, and the canvas covers it exactly.
        let display = layout_for(&doc, (800.0, 600.0));
        assert!(!display.is_empty());
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect {
                width: 800.0,
                height: 600.0,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn load_page_invalid_url_errors() {
        let loader = ResourceLoader::new();
        assert!(
            load_page_cancellable(
                &loader,
                "ftp://example.com/".to_string(),
                CancellationToken::new()
            )
            .await
            .is_err()
        );
    }

    /// Multi-route local server for the external-CSS / script tests:
    /// every request path is recorded in `hits` (in arrival order),
    /// so a test can prove a subresource was actually fetched.
    async fn serve_tracked(listener: TcpListener, hits: Arc<Mutex<Vec<String>>>) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let hits = Arc::clone(&hits);
            tokio::spawn(async move {
                let mut buf = [0u8; 8192];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .to_string();
                hits.lock().unwrap().push(path.clone());
                let (content_type, body) = match path.as_str() {
                    "/" => (
                        "text/html",
                        "<!doctype html><html><head><meta charset=\"utf-8\">\
                         <link rel=\"stylesheet\" href=\"/a.css\">\
                         <link rel=\"stylesheet\" href=\"/print.css\" media=\"print\">\
                         <script src=\"/s.js\"></script>\
                         <script defer src=\"/d.js\"></script></head>\
                         <body><p>Hello</p></body></html>"
                            .to_string(),
                    ),
                    "/a.css" => (
                        "text/css",
                        "@import url(\"/b.css\"); body { background-color: #112233; }".to_string(),
                    ),
                    "/b.css" => ("text/css", "p { color: #123456; }".to_string()),
                    "/print.css" => (
                        "text/css",
                        "body { background-color: #ff0000; }".to_string(),
                    ),
                    "/s.js" => ("text/javascript", "var x = 1;".to_string()),
                    "/d.js" => ("text/javascript", "var d = 1;".to_string()),
                    _ => ("text/plain", "not found".to_string()),
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    fn fill_near(color: [f32; 4], rgb: (u8, u8, u8)) -> bool {
        (color[0] - rgb.0 as f32 / 255.0).abs() < 0.01
            && (color[1] - rgb.1 as f32 / 255.0).abs() < 0.01
            && (color[2] - rgb.2 as f32 / 255.0).abs() < 0.01
    }

    /// External CSS (R9) + script fetch-and-pause (R18), end to end
    /// against a local server: the external sheet (and its `@import`)
    /// must restyle the page, the `media="print"` sheet must not, the
    /// classic script must be fetched and held (not executed — there
    /// is no JS engine yet), and the defer script must be queued.
    #[tokio::test]
    async fn load_page_external_css_and_scripts() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(Mutex::new(Vec::new()));
        tokio::spawn(serve_tracked(listener, Arc::clone(&hits)));
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let (doc, _final_url) =
            load_page_cancellable(&loader, url.clone(), CancellationToken::new())
                .await
                .expect("page loads");
        assert_eq!(doc.base_url, url);
        let hits = hits.lock().unwrap();
        // The document, both sheets, the import, and both scripts
        // were all fetched (the scanner fires some, the parser the
        // rest — dedup makes the count at most one per URL here, and
        // at least one: what matters is every URL was requested).
        for want in ["/", "/a.css", "/b.css", "/print.css", "/s.js", "/d.js"] {
            assert!(hits.contains(&want.to_string()), "expected a fetch of {want}: {hits:?}");
        }
        drop(hits);
        // The external sheet's body background is the canvas fill…
        let display = layout_for(&doc, (800.0, 600.0));
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect { color, .. } if fill_near(color, (0x11, 0x22, 0x33))
        ));
        // …the print sheet's red background did NOT win…
        assert!(!display.0.iter().any(|i| matches!(
            i,
            DisplayItem::FillRect { color, .. } if fill_near(*color, (0xff, 0x00, 0x00))
        )));
        // …and the @import'd sheet's paragraph color applies.
        assert!(display.0.iter().any(|i| matches!(
            i,
            DisplayItem::Glyphs { color, .. } if fill_near(*color, (0x12, 0x34, 0x56))
        )));
        // Scripts were fetched and held, never executed: the classic
        // script's bytes are held, the defer script is queued in
        // document order, and the page text still rendered.
        assert_eq!(doc.defer_scripts.len(), 1, "defer script queued");
        assert!(doc.async_scripts.is_empty());
        let classic = doc
            .held_scripts
            .iter()
            .find(|h| h.src.as_deref() == Some(&format!("http://127.0.0.1:{}/s.js", addr.port())))
            .expect("classic script held");
        assert_eq!(classic.bytes, b"var x = 1;");
        let deferred = doc
            .held_scripts
            .iter()
            .find(|h| h.src.as_deref() == Some(&format!("http://127.0.0.1:{}/d.js", addr.port())))
            .expect("defer script held");
        assert_eq!(deferred.bytes, b"var d = 1;");
        assert!(display.0.iter().any(|i| matches!(
            i,
            DisplayItem::Glyphs { glyphs, .. } if !glyphs.is_empty()
        )));
    }

    /// An inline classic script pauses the parser and is held (its
    /// text), and the content after it still parses — the resume
    /// path, without any network involved.
    #[tokio::test]
    async fn load_page_inline_script_pauses_and_resumes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let Ok(n) = stream.read(&mut buf).await else {
                        return;
                    };
                    let _ = n;
                    let body = "<!doctype html><html><head></head><body>\
                        <p>before</p><script>var y = 2;</script><p>after</p></body></html>";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let (doc, _) = load_page_cancellable(&loader, url, CancellationToken::new())
            .await
            .expect("page loads");
        // Both paragraphs rendered (the parser resumed past the
        // inline script), and the script text was held.
        let display = layout_for(&doc, (800.0, 600.0));
        let glyphs: usize = display.0.iter().fold(0, |acc, item| match item {
            DisplayItem::Glyphs { glyphs, .. } => acc + glyphs.len(),
            _ => acc,
        });
        assert!(glyphs >= 10, "before + after text renders: {glyphs} glyphs");
        let inline = doc
            .held_scripts
            .iter()
            .find(|h| h.src.is_none())
            .expect("inline script held");
        assert_eq!(inline.bytes, b"var y = 2;");
    }

    /// A missing external stylesheet must not fail the load: the page
    /// renders with its inline styles, and the error is only logged.
    #[tokio::test]
    async fn load_page_missing_stylesheet_renders_anyway() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
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
                    let path = request
                        .lines()
                        .next()
                        .unwrap_or("")
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/");
                    // Any stylesheet 404s; the document itself loads.
                    let (status, body) = if path == "/missing.css" {
                        ("HTTP/1.1 404 Not Found", "")
                    } else {
                        (
                            "HTTP/1.1 200 OK",
                            "<!doctype html><html><head><link rel=\"stylesheet\" href=\"/missing.css\">\
                             <style>body { background-color: #00ff00; }</style></head>\
                             <body><p>hi</p></body></html>",
                        )
                    };
                    let response = format!(
                        "{status}\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let (doc, _) = load_page_cancellable(&loader, url, CancellationToken::new())
            .await
            .expect("page loads despite the missing sheet");
        let display = layout_for(&doc, (800.0, 600.0));
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect { color, .. } if fill_near(color, (0x00, 0xff, 0x00))
        ));
    }
}
