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

use crate::images::ImageRequest;
use pluto_css::{MediaCondition, Stylesheet};
use pluto_html::DomEvent;
use pluto_html::{
    Dom, NodeId, Parser, resolve_url, scan_hints, scan_with_viewport,
};
use pluto_html::{Destination as HtmlDestination, Priority as HtmlPriority};
use pluto_ipc::{
    CredentialsMode, Destination, Origin, Priority, RequestMode, ResourceRequest, origin_of,
};
use pluto_js::ModuleSources;
use pluto_layout::{ElementState, LayoutTree, Theme};
use pluto_net::csp::{PolicySet, inline_style_allowed};
use pluto_paint::DisplayList;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

/// Sinks a load reports into while it runs (Stage 11 Rec 1/9):
/// image discoveries go to the tab thread's image pipeline
/// (fetch + decode, owned by the tab thread — the load task
/// never fetches image bytes itself), and document-fetch
/// progress goes to the shell's progress bar. Both optional:
/// `None` keeps the old behaviour (images never fetched,
/// no progress reported). The senders are `UnboundedSender`s so
/// a slow tab thread never blocks the load.
#[derive(Default)]
pub struct LoadSinks {
    pub images: Option<UnboundedSender<ImageRequest>>,
    pub progress: Option<UnboundedSender<(u64, Option<u64>)>>,
    /// Stage 12 Rec 1: the tab's shared pre-fetched module
    /// sources. The load fills it while fetching module scripts
    /// (see `StreamingLoad::fetch_module_graph`); the realm's
    /// loader reads from it. `None` when no JS is attached.
    pub module_sources: Option<ModuleSources>,
}

impl LoadSinks {
    /// No sinks: the load behaves as before Stage 11.
    #[allow(dead_code)]
    pub fn none() -> Self {
        Self::default()
    }
}

/// Maximum preconnects per page (Stage 11 Rec 6): only the
/// critical-path origins get a full DNS+TCP+TLS warm; the rest
/// get DNS-only prefetch.
pub const MAX_PRECONNECTS_PER_PAGE: usize = 3;

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

/// Maximum module import-graph depth followed while pre-fetching
/// (Stage 12 Rec 1). Module graphs run deeper than CSS `@import`
/// chains, so this is its own, larger cap; cycles and pathological
/// graphs stop here instead of looping.
pub const MAX_MODULE_IMPORT_DEPTH: usize = 16;

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

    /// The merged author stylesheet of this document.
    pub fn sheet(&self) -> &Stylesheet {
        &self.sheet
    }

    /// The base URL relative URLs resolve against (final URL after
    /// redirects).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// A copy of this document with `dom` swapped in (the wave-2b
    /// integration re-syncs the `Document` from the host's live
    /// `Dom` after scripts have mutated it).
    pub fn with_dom(&self, dom: Dom) -> Document {
        let mut doc = self.clone();
        doc.dom = dom;
        doc
    }

    /// A document from its parts (the JS host builds the document
    /// the layout actor should lay out from the live `Dom`).
    pub fn from_parts(dom: Dom, sheet: Stylesheet, base_url: &str) -> Document {
        Document {
            dom,
            sheet,
            base_url: base_url.to_string(),
            defer_scripts: Vec::new(),
            async_scripts: Vec::new(),
            held_scripts: Vec::new(),
        }
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

/// The scheme of `url` (lowercased), if it has one: the part before
/// the first `:` when it is valid scheme syntax — and not a
/// `host:port`. A colon followed only by digits up to a `/`, `?`,
/// `#`, or the end of the string is a port, not a scheme, so
/// `localhost:8080` and `example.com:8080/path` stay bare hosts
/// (Stage 9 Rec 7: without this, `localhost:8080` would read as the
/// unknown scheme `localhost` and be rejected).
fn url_scheme(url: &str) -> Option<String> {
    let (candidate, rest) = url.split_once(':')?;
    if candidate.is_empty()
        || !candidate
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic())
        || !candidate
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
    {
        return None;
    }
    let port_part = rest.split(['/', '?', '#']).next().unwrap_or("");
    if !port_part.is_empty() && port_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(candidate.to_ascii_lowercase())
}

/// Turn address-bar input into a fetchable URL. A bare host gets
/// `https://`. Only `http:` / `https:` may navigate — plus the
/// internal `about:blank`. Anything else (`javascript:`, `data:`,
/// `file:`, `ftp:`, any other scheme) is rejected (Stage 9 Rec 7,
/// T11: the address bar used to pass "anything with a scheme"
/// through, so `javascript:alert(1)`, `data:text/html,…`, and
/// `file:///etc/passwd` all reached the fetch layer); `file:` stays
/// disallowed (Joel). The caller shows the error page for the `Err`
/// instead of fetching — the worker-side guard in
/// [`load_page_cancellable_with_hooks`] enforces the same allowlist
/// again, so a rejected scheme never reaches a fetch even from a
/// caller that bypasses this function. The scheme check is
/// case-insensitive (`JAVASCRIPT:` is rejected too).
pub fn normalize_url(input: &str) -> Result<String, String> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(DEFAULT_URL.to_string());
    }
    match url_scheme(input).as_deref() {
        None => Ok(format!("https://{input}")),
        Some("http") | Some("https") => Ok(input.to_string()),
        Some("about") if input.eq_ignore_ascii_case("about:blank") => {
            Ok("about:blank".to_string())
        }
        Some(scheme) => Err(format!(
            "Pluto cannot open {scheme}: URLs — only http:, https:, and about:blank are allowed for navigation"
        )),
    }
}

/// The load-path half of the navigation scheme allowlist: `url` must
/// already be normalized (see [`normalize_url`]) — `http:` / `https:`
/// or exactly `about:blank`. Anything else fails before any fetch.
fn check_navigation_url(url: &str) -> Result<(), String> {
    match url_scheme(url.trim()).as_deref() {
        Some("http") | Some("https") => Ok(()),
        Some("about") if url.trim().eq_ignore_ascii_case("about:blank") => Ok(()),
        Some(scheme) => Err(format!(
            "Pluto cannot open {scheme}: URLs — only http:, https:, and about:blank are allowed for navigation"
        )),
        None => Err(format!(
            "Pluto cannot open {url} — only http:, https:, and about:blank are allowed for navigation"
        )),
    }
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
#[allow(clippy::too_many_arguments)]
fn subresource_request(
    url: &str,
    destination: Destination,
    priority: Priority,
    base_url: &str,
    is_imageset: bool,
    integrity: Option<String>,
    tab_id: u64,
    epoch: u64,
) -> ResourceRequest {
    let origin = origin_of(base_url);
    ResourceRequest {
        request_id: next_request_id(),
        tab_id,
        epoch,
        url: url.to_string(),
        method: "GET".to_string(),
        headers: Vec::new(),
        // Stage 9 Rec 2/6: the preload scanner's `from:
        // "imageset"` label rides the wire so the mixed-content
        // gate BLOCKS (never auto-upgrades) srcset / <picture>
        // images on a secure page.
        is_imageset,
        destination,
        priority,
        mode: RequestMode::NoCors,
        credentials: CredentialsMode::SameOrigin,
        origin: origin.clone(),
        top_level_site: origin,
        upgrade_insecure_requests: false,
        // Stage 9 Rec 14 (SRI): the element's `integrity`
        // attribute, verified over the decoded bytes in
        // `pluto-net` before `Done`.
        integrity,
        // Stage 13 Rec 6: subresources never carry a body.
        body: None,
    }
}

/// A `@font-face` fetch request (M6 Phase 3, Rec 12): the same
/// loader, the same Stage 9 gates (CSP / referrer / HTTPS-only /
/// caps) as any other subresource — `Destination::Font`, low
/// priority, `NoCors` / `SameOrigin` like the other
/// [`subresource_request`] callers. Fonts are never fetched any
/// other way.
pub fn font_resource_request(
    url: &str,
    base_url: &str,
    tab_id: u64,
    epoch: u64,
) -> ResourceRequest {
    subresource_request(
        url,
        Destination::Font,
        Priority::Low,
        base_url,
        false,
        None,
        tab_id,
        epoch,
    )
}

fn map_html_destination(d: HtmlDestination) -> Destination {
    match d {
        HtmlDestination::Document => Destination::Document,
        HtmlDestination::Stylesheet => Destination::Stylesheet,
        HtmlDestination::Script => Destination::Script,
        HtmlDestination::Image => Destination::Image,
        HtmlDestination::Font => Destination::Font,
        HtmlDestination::Fetch => Destination::Fetch,
        HtmlDestination::Audio => Destination::Audio,
        HtmlDestination::Video => Destination::Video,
        HtmlDestination::WebSocket => Destination::WebSocket,
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

/// Map the document's referrer policy (captured by `pluto-html`
/// from `<meta name="referrer">`) onto the policy `pluto-net`
/// enforces when building the `Referer` header (Stage 9 Rec 15).
/// The two enums carry the same two values by design — neither
/// crate may depend on the other, so the mapping lives here.
fn map_html_referrer_policy(
    policy: pluto_html::ReferrerPolicy,
) -> pluto_net::ReferrerPolicy {
    match policy {
        pluto_html::ReferrerPolicy::NoReferrerWhenDowngrade => {
            pluto_net::ReferrerPolicy::NoReferrerWhenDowngrade
        }
        pluto_html::ReferrerPolicy::StrictOriginWhenCrossOrigin => {
            pluto_net::ReferrerPolicy::StrictOriginWhenCrossOrigin
        }
    }
}

/// The script-execution seams `StreamingLoad` calls into (wave 2b,
/// Stage 7 Rec 6): the integration wave's runner implements this
/// trait, and the load path calls it at each seam — a parser-blocking
/// pause, an async script becoming ready, an encoding restart
/// swapping the parser, and the end of the parse (defer drain +
/// `DOMContentLoaded`). Nothing here executes JavaScript; the trait
/// is the boundary between the parser (which knows *when* a script
/// runs) and the engine (which knows *how*).
///
/// All methods are synchronous and run on the tab thread (the
/// thread that owns the parser and the `TabJs`); the runner may
/// drive the event loop from them.
pub trait ScriptHooks {
    /// A parser-blocking `<script>` paused the parse. `bytes` is
    /// its fetched source (external) or its element text (inline);
    /// the parser resumes when this returns. Async because execution
    /// is delegated to the tab thread's main loop (the load runs as
    /// a `spawn_local` task and cannot borrow the `TabJs` — see
    /// `js_host::LoadHooks`). Boxed futures, not `async fn`: the
    /// trait is used behind `Box<dyn ScriptHooks>`, and `async fn`
    /// methods are not dyn-compatible.
    fn paused_script<'a>(
        &'a mut self,
        parser: &'a std::rc::Rc<std::cell::RefCell<Parser>>,
        paused: &'a pluto_html::ScriptPaused,
        bytes: &'a [u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>;

    /// Async scripts whose fetches just completed, in arrival
    /// order. `bytes_by_src` is the load's fetched-bytes cache.
    fn async_ready<'a>(
        &'a mut self,
        parser: &'a std::rc::Rc<std::cell::RefCell<Parser>>,
        ready: Vec<pluto_html::QueuedScript>,
        bytes_by_src: &'a HashMap<String, Vec<u8>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>>;

    /// The load created the parser for this navigation (after the
    /// fetch, so it carries the final URL): the runner must attach
    /// the host to its shared `Dom` before the first chunk is fed,
    /// or scripts see an empty document and `parse_finished` would
    /// re-sync the finished document from that empty arena. The
    /// load creates the parser — not the caller — because only the
    /// load knows the final URL; the runner's own parser handle is
    /// set here, never constructed separately (a second parser
    /// would be fed nothing and its `Dom` would stay empty).
    fn parser_created(&mut self, _parser: &std::rc::Rc<std::cell::RefCell<Parser>>) {}

    /// The document response's CSP policies (Stage 15, M7 Phase 1,
    /// Rec 11): the load calls this with the policies built from
    /// the response's `content-security-policy` headers, before
    /// the first chunk is fed — the runner stashes them where the
    /// tab thread's script gate can read them. Default no-op: a
    /// runner with no execution path has nothing to gate.
    fn set_document_csp(&mut self, _policies: PolicySet) {}

    /// An encoding restart replaced the parser (and its arena) with
    /// a fresh one: the runner must re-point the host at it.
    fn parser_restarted(&mut self, parser: &std::rc::Rc<std::cell::RefCell<Parser>>);

    /// The parse finished: drain the defer queue in document order
    /// from `document`, then fire `DOMContentLoaded`, then return
    /// the document re-synced from the host's live `Dom` (JS
    /// mutations made during the parse / defer drain live there).
    /// By value in both directions so the load task can hand the
    /// document to the tab thread's main loop and get it back.
    fn parse_finished<'a>(
        &'a mut self,
        document: Document,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Document> + 'a>>;
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
/// A form-POST body for a navigation (Stage 13 Rec 6): the
/// encoded entry list plus its `Content-Type`. `None` means a
/// plain GET navigation.
#[derive(Debug, Clone)]
pub struct PostBody {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

/// Load a page, with an optional [`ScriptHooks`] runner attached:
/// parser-blocking scripts execute at their pause, async scripts on
/// arrival, defer scripts after the parse, and `DOMContentLoaded`
/// fires once the defer queue drains. `None` keeps the old
/// behaviour (scripts fetched and held, never executed).
#[allow(clippy::too_many_arguments)]
pub async fn load_page_cancellable_with_hooks(
    fetch: &crate::broker::FetchSource,
    url: String,
    cancel: CancellationToken,
    mut hooks: Option<Box<dyn ScriptHooks + '_>>,
    sinks: LoadSinks,
    tab_id: u64,
    epoch: u64,
    viewport: (f32, f32),
    post_body: Option<PostBody>,
    referrer_url: &str,
) -> Result<(Document, String), String> {
    // Stage 13 Rec 12c: the internal crash page renders locally
    // — no fetch, no gates — from the fixed crash document. It is
    // only ever reached by the shell navigating a crashed tab to
    // `about:crashed`; it is not a user-navigable URL.
    if url == "about:crashed" {
        return Ok((
            parse_document_with_base(&crash_document(), "about:crashed"),
            "about:crashed".to_string(),
        ));
    }
    // Stage 9 Rec 7 (T11): scheme allowlist at the load entry
    // point — defence in depth behind `normalize_url` (link clicks
    // and other callers reach here without it). A disallowed scheme
    // fails here, before any fetch, and the caller renders the error
    // page from this `Err` (`LoadOutcome::Failed` → `error_document`).
    if let Err(reason) = check_navigation_url(&url) {
        eprintln!("[pluto] blocked navigation: {reason}");
        return Err(reason);
    }
    let mut request = ResourceRequest::navigate(next_request_id(), tab_id, epoch, &url);
    // Stage 13 Rec 6: a form POST rides the same loader (same
    // Stage 9 gates — mixed content, blocker, HSTS, cookie jar,
    // epoch cancellation) as a GET navigation; only the method,
    // body and `Content-Type` differ.
    if let Some(post) = post_body {
        request.method = "POST".to_string();
        request.body = Some(post.bytes);
        request
            .headers
            .push(("Content-Type".to_string(), post.content_type));
    }
    eprintln!("[pluto] fetching {url}");
    // Stage 11 Rec 9: the document fetch streams its progress
    // (bytes vs Content-Length) to the shell's progress bar via
    // the sink; without a sink the callback is a no-op.
    let progress_tx = sinks.progress.clone();
    let response = match fetch
        .fetch_with_progress(request, cancel.clone(), move |received, total| {
            if let Some(tx) = &progress_tx {
                let _ = tx.send((received, total));
            }
        })
        .await
    {
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
    // Stage 15 (M7 Phase 1, Rec 11): the document's CSP, built
    // from its response headers (same extraction as the fetch-side
    // gate in `pluto-net`). It is stashed on the shared host
    // BEFORE the first chunk is fed — so a parser-blocking inline
    // script is gated under the CURRENT document's policies — and
    // carried into the streaming load for the inline-`<style>`
    // gate at `finish_document`. Replaced unconditionally: a
    // document with no CSP headers clears the previous
    // document's policies. This is the execution-time source of
    // truth in both fetch modes (in Broker mode the worker-side
    // loader never sees the document response).
    let csp = {
        let enforce: Vec<&str> = response
            .headers
            .get_all("content-security-policy")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect();
        let report_only: Vec<&str> = response
            .headers
            .get_all("content-security-policy-report-only")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect();
        PolicySet::from_headers(&enforce, &report_only)
    };
    if let Some(hooks) = hooks.as_mut() {
        hooks.set_document_csp(csp.clone());
    }
    // Stage 9 Rec 13 (downloads): decide render-vs-download
    // BEFORE parsing — a response that says "download" must
    // never render as a page (an attachment served as text/html
    // must not execute). The mixed-download block runs first
    // (the net layer already refused attachment-disposition
    // mixed downloads; this also catches an octet-stream
    // download served over http from a secure context).
    // The save itself needs the prompt UI (M10): the report
    // forbids silent auto-save, so until the prompt exists the
    // load fails with the filename — the bytes are never parsed
    // and nothing is written to disk.
    {
        use crate::downloads::{DownloadDecision, decide_with_sniff, is_mixed_download_blocked};
        let disposition = response
            .headers
            .get("content-disposition")
            .and_then(|v| v.to_str().ok());
        let content_type = response
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok());
        let nosniff = response
            .headers
            .get("x-content-type-options")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.eq_ignore_ascii_case("nosniff"));
        // The previous document is the secure context a mixed
        // download is judged against; a typed first navigation
        // has none, so the mixed check passes through to `decide`.
        // Stage 13 Rec 11: the referrer arrives as a parameter
        // — in broker mode the loader-side referrer context lives
        // in the broker, not in this process.
        let page_origin = referrer_url.to_string();
        if is_mixed_download_blocked(&page_origin, &final_url, disposition) {
            return Err(format!("mixed download blocked: {final_url}"));
        }
        let prefix_len = response.body.len().min(1445);
        match decide_with_sniff(
            disposition,
            content_type,
            &final_url,
            // `<a download>` forces download, but navigations
            // carry no download attribute (the IPC shape has no
            // such flag — threading it is a follow-up).
            false,
            &response.body[..prefix_len],
            nosniff,
        ) {
            DownloadDecision::RenderInline => {}
            DownloadDecision::BlockMixed => {
                return Err(format!("mixed download blocked: {final_url}"));
            }
            DownloadDecision::Download(prompt) => {
                let size = response
                    .headers
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown");
                return Err(format!(
                    "download: {} ({} bytes) from {} — saving needs the download prompt (M10)",
                    prompt.filename, size, prompt.origin
                ));
            }
        }
    }
    // Preload scan (R4): run over the whole fetched body before
    // parsing and fire off fetches for what it finds. The scheduler
    // dedups these against the parser-driven fetches below. True
    // network streaming (feed chunks as they arrive) is the
    // follow-up — see the module docs.
    fire_preloads(
        fetch,
        &response.body,
        &final_url,
        cancel.clone(),
        &sinks,
        tab_id,
        epoch,
        viewport,
    );
    // Stage 11 Rec 6: warm the connection hints the scanner
    // recorded (`<link rel="preconnect">` / `dns-prefetch`) on a
    // detached task — the parse must not wait on a TLS handshake.
    fire_connection_hints(fetch, &response.body, &final_url);
    // Wave 2b: the parser lives in an `Rc<RefCell>` so the script
    // runner (via the host) can reach its `DocumentWriter` while
    // the load drives it.
    let parser = std::rc::Rc::new(std::cell::RefCell::new(Parser::with_document_url(
        &final_url,
    )));
    // Stage 11 Rec 10: the parser selects srcset / <picture>
    // candidates against the real viewport, so its viewport must
    // be the tab's, not the scanner default. DPR is 1.0: the
    // worker has no HiDPI plumbing yet (the shell's scale factor
    // never reaches it) — a follow-up threads it through.
    parser.borrow_mut().set_viewport(viewport.0, 1.0);
    // The runner attaches the host to this parser's shared `Dom`
    // before the first chunk is fed (see `parser_created`).
    if let Some(hooks) = hooks.as_mut() {
        hooks.parser_created(&parser);
    }
    let mut load = StreamingLoad::new(
        fetch.clone(),
        cancel.clone(),
        final_url.clone(),
        parser,
        hooks,
        sinks,
        tab_id,
        epoch,
        csp,
    );
    for chunk in response.body.chunks(FEED_CHUNK_BYTES) {
        if cancel.is_cancelled() {
            return Err("navigation cancelled".to_string());
        }
        load.drive(chunk).await?;
    }
    if cancel.is_cancelled() {
        return Err("navigation cancelled".to_string());
    }
    let mut hooks = load.take_hooks();
    let mut document = load.finish_document();
    // Wave 2b: with hooks attached, the defer drain +
    // `DOMContentLoaded` happen here (scripts execute; the old
    // "held, never executed" log below only applies without hooks).
    if let Some(hooks) = hooks.as_mut() {
        document = hooks.parse_finished(document).await;
    }
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
/// Non-image discoveries run on their own tasks and their bytes
/// are discarded: the point is warming the scheduler / connection
/// so the parser-driven fetch that follows dedups onto an
/// in-flight (or just-completed) request instead of starting
/// cold. Image discoveries are NOT fetched here: they go to the
/// tab thread's image pipeline via the sink (fetch + decode,
/// owned by the tab thread — fetching-and-discarding them here
/// would download the bytes twice). `data:` URLs are never
/// fetched (the scanner already skips most of them; this is the
/// belt). The scan runs at the tab's real viewport (Stage 11 Rec
/// 10), so a srcset candidate is selected for the actual page
/// width, not the 1024 px default.
#[allow(clippy::too_many_arguments)]
fn fire_preloads(
    fetch: &crate::broker::FetchSource,
    body: &[u8],
    base_url: &str,
    cancel: CancellationToken,
    _sinks: &LoadSinks,
    tab_id: u64,
    epoch: u64,
    viewport: (f32, f32),
) {
    for found in scan_with_viewport(body, base_url, viewport.0, 1.0) {
        if found.url.trim_start().starts_with("data:") {
            continue;
        }
        // Stage 11 Rec 1/10: images are NOT fired here. The
        // streaming parser reports every `<img>` / `<picture>`
        // image exactly once, during the parse, through the same
        // `classify_element` selection — a scan report here would
        // be a duplicate of the parser's (the tab thread dedups
        // by URL, but the sink contract is one report per image).
        // The scan's job is the non-image preloads below.
        if found.destination == HtmlDestination::Image {
            continue;
        }
        let fetch = fetch.clone();
        let cancel = cancel.clone();
        let request = subresource_request(
            &found.url,
            map_html_destination(found.destination),
            map_html_priority(found.priority),
            base_url,
            // Stage 9 Rec 6: the scanner labels srcset /
            // <picture> candidates `"imageset"`; the label rides
            // the wire so the mixed-content gate blocks (never
            // auto-upgrades) them on a secure page.
            found.from == "imageset",
            // Stage 9 Rec 14 (SRI): the scanner reads the
            // element's `integrity` attribute. The preload scan
            // runs over the whole body before parsing starts, so
            // this request is always first — the parser's later
            // duplicate dedups to it, integrity included.
            found.integrity.clone(),
            tab_id,
            epoch,
        );
        tokio::spawn(async move {
            let _ = fetch.fetch_with(request, cancel).await;
        });
    }
}

/// Warm connection hints (Stage 11 Rec 6): the first
/// [`MAX_PRECONNECTS_PER_PAGE`] preconnect origins get a full
/// DNS+TCP+TLS warm through the shared client pool; every other
/// hint (further preconnects + all dns-prefetch) gets DNS-only.
/// Runs detached — a TLS handshake must never stall the parse —
/// on `tokio::spawn` (not `spawn_local`: tests drive this load
/// outside a `LocalSet`, and the hint task owns only `Send`
/// values). `preconnect` itself applies HSTS and refuses
/// blocker-blocked tracker origins, so this needs no extra
/// gating here.
fn fire_connection_hints(fetch: &crate::broker::FetchSource, body: &[u8], base_url: &str) {
    use pluto_html::HintKind;
    let fetch = fetch.clone();
    let hints = scan_hints(body, base_url);
    if hints.is_empty() {
        return;
    }
    tokio::task::spawn(async move {
        let mut preconnected = 0usize;
        for hint in hints {
            match hint.kind {
                HintKind::Preconnect if preconnected < MAX_PRECONNECTS_PER_PAGE => {
                    if fetch.preconnect(&hint.url).await {
                        preconnected += 1;
                    }
                }
                _ => {
                    let _ = fetch.dns_prefetch(&hint.url).await;
                }
            }
        }
    });
}

/// The streaming load's mutable state: the parser, the caches that
/// survive an encoding restart (so nothing is refetched), and the
/// render-blocking deadline.
struct StreamingLoad<'h> {
    fetch: crate::broker::FetchSource,
    cancel: CancellationToken,
    base_url: String,
    /// The parser, shared with the script runner (via the host) so
    /// a parser-blocking script's execution can raise the
    /// `DocumentWriter` nesting level and `document.write` can
    /// splice into the tokenizer.
    parser: std::rc::Rc<std::cell::RefCell<Parser>>,
    /// The script runner (wave 2b), if JS is attached to this load.
    hooks: Option<Box<dyn ScriptHooks + 'h>>,
    /// Where image discoveries and load progress are reported
    /// (Stage 11 Rec 1/9): the tab thread owns the image
    /// pipeline, so this load only reports URLs, never fetches.
    sinks: LoadSinks,
    /// The shell-assigned tab id + navigation epoch stamped on
    /// every subresource request (Stage 11 Rec 4).
    tab_id: u64,
    epoch: u64,
    /// Fetched stylesheet text by absolute URL (includes `@import`
    /// targets, fetched recursively when their parent sheet loaded).
    sheet_cache: HashMap<String, String>,
    /// URLs already attempted (loaded or failed): a re-feed after an
    /// encoding restart re-discovers them, and must not refetch.
    sheet_attempted: HashSet<String>,
    /// Fetched script bytes by resolved `src`.
    script_bytes: HashMap<String, Vec<u8>>,
    /// The document's CSP policies (Stage 15, M7 Phase 1, Rec 11),
    /// built from the document response's headers at the load site:
    /// `finish_document` gates inline `<style>` blocks against
    /// them. Carried here (not read back from the host) because
    /// the load owns the parse.
    csp: PolicySet,
    /// Module import URLs already attempted (loaded or failed),
    /// keyed by absolute URL (Stage 12 Rec 1): the same module
    /// imported from several spellings / several parents is
    /// fetched once, and circular imports terminate.
    module_attempted: HashSet<String>,
    /// Classic parser-blocking `<script>` nodes, in pause order.
    paused_scripts: Vec<NodeId>,
    /// Render-blocking deadline (navigation start + timeout).
    blocking_deadline: tokio::time::Instant,
    /// Whether the one encoding restart has already happened.
    restarted: bool,
    /// Whether the unsupported-encoding note has been logged once.
    encoding_logged: bool,
}

impl<'h> StreamingLoad<'h> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fetch: crate::broker::FetchSource,
        cancel: CancellationToken,
        base_url: String,
        parser: std::rc::Rc<std::cell::RefCell<Parser>>,
        hooks: Option<Box<dyn ScriptHooks + 'h>>,
        sinks: LoadSinks,
        tab_id: u64,
        epoch: u64,
        csp: PolicySet,
    ) -> Self {
        Self {
            fetch,
            cancel,
            base_url: base_url.clone(),
            parser,
            hooks,
            sinks,
            tab_id,
            epoch,
            sheet_cache: HashMap::new(),
            sheet_attempted: HashSet::new(),
            script_bytes: HashMap::new(),
            csp,
            module_attempted: HashSet::new(),
            paused_scripts: Vec::new(),
            blocking_deadline: tokio::time::Instant::now() + RENDER_BLOCKING_TIMEOUT,
            restarted: false,
            encoding_logged: false,
        }
    }

    /// Take the script hooks out (called before `finish_document`
    /// consumes the load, so the caller can drain the defer queue
    /// from the finished document).
    fn take_hooks(&mut self) -> Option<Box<dyn ScriptHooks + 'h>> {
        self.hooks.take()
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
                let events = self.parser.borrow_mut().feed(&bytes);
                // Stage 9 Rec 15: a `<meta name="referrer">` seen
                // in this chunk updates the loader's policy for
                // subresources discovered after it (last one seen
                // wins; the response header's policy was already
                // applied when the document response arrived).
                if let Some(policy) = self.parser.borrow().referrer_policy() {
                    self.fetch
                        .set_referrer_policy(map_html_referrer_policy(policy));
                }
                let mut resume = false;
                for event in events {
                    match event {
                        DomEvent::StylesheetDiscovered { href, .. } => {
                            self.fetch_stylesheet(&href).await;
                        }
                        DomEvent::ResourceDiscovered(found) => {
                            // Stage 11 Rec 1/10: images are
                            // reported to the tab thread's image
                            // pipeline (fetch + decode there) —
                            // never fetched here, and never fetched
                            // twice: the scheduler dedups this
                            // report against the preload scan's
                            // earlier one for the same URL.
                            if found.destination == HtmlDestination::Image
                                && !found.url.trim_start().starts_with("data:")
                            {
                                if let Some(tx) = &self.sinks.images {
                                    let _ = tx.send(ImageRequest {
                                        url: found.url.clone(),
                                        base_url: self.base_url.clone(),
                                        is_imageset: found.from == "imageset",
                                        integrity: found.integrity.clone(),
                                    });
                                }
                                continue;
                            }
                            // Scripts are fetched (async/defer, and
                            // classic ones the pause below also reports).
                            // Everything else the preload scanner already
                            // fired off; the scheduler dedups the rest.
                            // Wave 2b: an async script whose fetch just
                            // completed becomes ready and executes now
                            // (arrival order — the queue ordered it).
                            if found.destination == HtmlDestination::Script
                                && !found.url.trim_start().starts_with("data:")
                            {
                                self.fetch_script(
                                    &found.url,
                                    Priority::Medium,
                                    // Stage 9 Rec 14 (SRI): the
                                    // discovery carries the
                                    // element's `integrity`
                                    // attribute; the preload
                                    // request for the same URL
                                    // already carries it too, and
                                    // the scheduler dedups the two.
                                    found.integrity.clone(),
                                    found.is_module,
                                )
                                .await;
                                if let Some(hooks) = self.hooks.as_mut() {
                                    let ready = self
                                        .parser
                                        .borrow_mut()
                                        .async_queue_mut()
                                        .mark_ready_by_src(&found.url);
                                    if !ready.is_empty() {
                                        hooks
                                            .async_ready(
                                                &self.parser,
                                                ready,
                                                &self.script_bytes,
                                            )
                                            .await;
                                    }
                                }
                            }
                        }
                        DomEvent::ScriptPaused(paused) => {
                            self.paused_scripts.push(paused.script);
                            let mut bytes = Vec::new();
                            if let Some(src) = &paused.src
                                && !src.trim_start().starts_with("data:")
                            {
                                // Stage 9 Rec 14 (SRI): `ScriptPaused`
                                // carries no `integrity` — the preload
                                // request for the same `src` already
                                // carries it, and the scheduler dedups
                                // this fetch to it.
                                self.fetch_script(src, Priority::High, None, paused.is_module)
                                    .await;
                                bytes = self
                                    .script_bytes
                                    .get(src)
                                    .cloned()
                                    .unwrap_or_default();
                            } else if paused.src.is_none() {
                                // Inline script: read its text from the
                                // live Dom (the parser is paused at it,
                                // so the element is complete).
                                let dom = self.parser.borrow().shared_dom();
                                let dom = dom.lock().unwrap();
                                bytes = script_text(&dom, paused.script).into_bytes();
                            }
                            // Stylesheet gate (Rec 6): a
                            // parser-blocking script must wait for
                            // pending stylesheets. Sheets the parser
                            // discovered are resolved by `drive` itself
                            // (see `fetch_stylesheet`'s gate notes),
                            // so the gate is normally clear here; a
                            // sheet only the preload scanner knows
                            // about can still hold it — wait a bounded
                            // 500 ms, then execute and log (documented
                            // gap in `pluto-html/src/scripts.rs`).
                            if self.parser.borrow().stylesheets_pending() {
                                let deadline =
                                    tokio::time::Instant::now() + Duration::from_millis(500);
                                while self.parser.borrow().stylesheets_pending()
                                    && tokio::time::Instant::now() < deadline
                                {
                                    tokio::time::sleep(Duration::from_millis(10)).await;
                                }
                                if self.parser.borrow().stylesheets_pending() {
                                    eprintln!(
                                        "[pluto] stylesheet gate still pending at script pause; executing anyway"
                                    );
                                }
                            }
                            // Wave 2b: execute at the pause (or skip
                            // a failed external script — empty bytes
                            // under a `src`); without hooks the old
                            // behaviour holds (scripts fetched and
                            // held, never executed).
                            if let Some(hooks) = self.hooks.as_mut() {
                                let failed = paused.src.is_some() && bytes.is_empty();
                                if !failed {
                                    hooks
                                        .paused_script(&self.parser, &paused, &bytes)
                                        .await;
                                }
                            }
                            resume = true;
                        }
                        DomEvent::EncodingIndicator { label }
                            if self.maybe_restart(&label) =>
                        {
                            let prefix = self.parser.borrow().buffered_prefix().to_vec();
                            *self.parser.borrow_mut() =
                                Parser::with_document_url(&self.base_url);
                            if let Some(hooks) = self.hooks.as_mut() {
                                hooks.parser_restarted(&self.parser);
                            }
                            input = prefix;
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
                // Wave 2b: a failed / timed-out sheet no longer
                // blocks parser-blocking scripts.
                self.parser.borrow_mut().note_stylesheet_failed(&url);
                continue;
            };
            // Wave 2b: feed the parser's stylesheet gate — a
            // parser-blocking script waits on this.
            self.parser.borrow_mut().note_stylesheet_loaded(&url);
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
        let request = subresource_request(
            url,
            destination,
            priority,
            &self.base_url,
            false,
            None,
            self.tab_id,
            self.epoch,
        );
        let fetch = self.fetch.fetch_with(request, self.cancel.clone());
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
    /// Stage 12 Rec 1: when `is_module`, the entry module's import
    /// graph is pre-fetched into the shared module sources (same
    /// loader, same Stage 9 gates — no second fetch path), so the
    /// realm's loader can serve imports synchronously at link time.
    async fn fetch_script(
        &mut self,
        src: &str,
        priority: Priority,
        integrity: Option<String>,
        is_module: bool,
    ) {
        if self.script_bytes.contains_key(src) {
            return;
        }
        let request = subresource_request(
            src,
            Destination::Script,
            priority,
            &self.base_url,
            false,
            integrity,
            self.tab_id,
            self.epoch,
        );
        match self.fetch.fetch_with(request, self.cancel.clone()).await {
            Ok(response) => {
                let final_url = response.final_url.to_string();
                let bytes = response.body;
                self.script_bytes.insert(src.to_string(), bytes.clone());
                if is_module {
                    let source = String::from_utf8_lossy(&bytes).into_owned();
                    self.fetch_module_graph(&final_url, &source).await;
                }
            }
            Err(e) => {
                eprintln!("[pluto] script fetch failed ({src}): {e}");
                // Record the miss so a re-feed after an encoding
                // restart does not retry a failing URL forever.
                self.script_bytes.insert(src.to_string(), Vec::new());
            }
        }
    }

    /// Pre-fetch a module's import graph into the shared module
    /// sources (Stage 12 Rec 1). `entry_url` / `entry_source` are
    /// the already-fetched entry module (absolute URL). Worklist,
    /// iterative like `fetch_stylesheet`'s `@import` walk:
    /// specifiers resolve against the importing module's URL, bare
    /// specifiers are skipped (no import maps in M4 — the loader
    /// rejects them at link time), and every fetched source is
    /// scanned for its own imports. A failed import fetch is
    /// logged and skipped — the importing module rejects at link
    /// time with a loader-miss error. No-op when no module store
    /// is attached (no JS on this load).
    async fn fetch_module_graph(&mut self, entry_url: &str, entry_source: &str) {
        let Some(store) = self.sinks.module_sources.clone() else {
            return;
        };
        store.insert(entry_url, entry_source.to_string());
        // The entry itself counts as attempted, so a circular
        // import back to it terminates instead of refetching.
        self.module_attempted.insert(entry_url.to_string());
        let mut stack = vec![(entry_url.to_string(), entry_source.to_string(), 0usize)];
        while let Some((url, source, depth)) = stack.pop() {
            if depth >= MAX_MODULE_IMPORT_DEPTH {
                eprintln!(
                    "[pluto] module import depth exceeded at {url}, skipping deeper imports"
                );
                continue;
            }
            for spec in pluto_js::scan_module_imports(&source) {
                let Ok(resolved) = pluto_js::resolve_module_specifier(&url, &spec) else {
                    // Bare / data: specifier — the loader rejects
                    // it at link time with the reason; nothing to
                    // fetch.
                    continue;
                };
                if !self.module_attempted.insert(resolved.clone()) {
                    continue;
                }
                if store.contains(&resolved) {
                    continue;
                }
                let Some((final_url, text)) = self.fetch_module_source(&resolved).await else {
                    continue;
                };
                if final_url != resolved && store.contains(&final_url) {
                    // A redirect landed on a module already walked.
                    continue;
                }
                store.insert(&final_url, text.clone());
                stack.push((final_url, text, depth + 1));
            }
        }
    }

    /// Fetch one module source as text (same loader, same Stage 9
    /// gates as every other subresource). `None` on any failure.
    /// Returns the final URL (after redirects) with the source.
    async fn fetch_module_source(&self, url: &str) -> Option<(String, String)> {
        let request = subresource_request(
            url,
            Destination::Script,
            Priority::Medium,
            &self.base_url,
            false,
            None,
            self.tab_id,
            self.epoch,
        );
        match self.fetch.fetch_with(request, self.cancel.clone()).await {
            Ok(response) => {
                let final_url = response.final_url.to_string();
                let text = String::from_utf8_lossy(&response.body).into_owned();
                Some((final_url, text))
            }
            Err(e) => {
                eprintln!("[pluto] module import fetch failed ({url}): {e}");
                None
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
    fn finish_document(self) -> Document {
        let parser = std::mem::replace(
            &mut *self.parser.borrow_mut(),
            Parser::with_document_url(&self.base_url),
        );
        let parsed = parser.finish();
        let dom = parsed.output.dom;
        // Stage 15 (M7 Phase 1, Rec 11): inline `<style>` blocks
        // are gated by the document's CSP (`style-src-elem` →
        // `style-src` → `default-src`) at collection time — a
        // blocked block never reaches the merged sheet. The
        // policies are the load's own (built from the document
        // response's headers at the load site), so the gate holds
        // in both fetch modes.
        let origin = Origin::from_url(&self.base_url);
        let sources = collect_style_sources(&dom, Some((&self.csp, &origin)));
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
pub(crate) fn script_text(dom: &Dom, id: NodeId) -> String {
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
    // No policies known on this path (see `collect_style_sources`):
    // inline `<style>` blocks are allowed, as before.
    let sources = collect_style_sources(&dom, None);
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
// The convenience entry point; the renderer drives layout through
// the layout-actor thread instead, so only tests call this.
#[allow(dead_code)]
pub fn layout_document_with_state(
    doc: &Document,
    viewport: (f32, f32),
    state: &ElementState,
) -> (Arc<DisplayList>, LayoutTree) {
    match LayoutTree::build_with_state(&doc.dom, &doc.sheet, viewport.0, viewport.1, state, Theme::Light) {
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
                Theme::Light,
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
///
/// Namespace-aware (Stage 15, M7 Phase 1, Rec 1): an HTML `<a>`
/// links through `href` only; an SVG `<a>` links through `href`
/// (SVG2) or the legacy `xlink:href`. A same-spelled `<a>` in any
/// other namespace links through neither.
pub fn find_link_href(doc: &Document, mut id: NodeId) -> Option<String> {
    loop {
        if doc.dom.tag(id) == Some("a") {
            let href = match doc.dom.namespace(id) {
                Some(ns) if ns == pluto_html::NS_SVG => doc
                    .dom
                    .attr(id, "href")
                    .or_else(|| doc.dom.attr(id, "xlink:href")),
                _ => doc.dom.attr(id, "href"),
            };
            if let Some(href) = href
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
///
/// `csp` gates inline `<style>` blocks (Stage 15, M7 Phase 1,
/// Rec 11): `Some((policies, origin))` drops a block the document's
/// CSP forbids; `None` (no policies known — the `parse_document*`
/// paths) allows everything, preserving the old behaviour.
fn collect_style_sources(
    dom: &Dom,
    csp: Option<(&PolicySet, &Origin)>,
) -> Vec<StyleSource> {
    let mut sources = Vec::new();
    if let Some(root) = dom.root() {
        collect_style_sources_from(dom, root, csp, &mut sources);
    }
    sources
}

fn collect_style_sources_from(
    dom: &Dom,
    id: NodeId,
    csp: Option<(&PolicySet, &Origin)>,
    out: &mut Vec<StyleSource>,
) {
    if dom.tag(id) == Some("style") {
        let mut css = String::new();
        for &child in dom.children(id) {
            if let Some(text) = dom.text(child) {
                css.push_str(text);
                css.push('\n');
            }
        }
        // Stage 15 (M7 Phase 1, Rec 11): a CSP-blocked inline
        // `<style>` contributes nothing — it never reaches the
        // merged sheet. The nonce rides the element, the content
        // is the exact block bytes (tags excluded).
        if let Some((policies, origin)) = csp
            && !inline_style_allowed(
                policies,
                origin,
                dom.attr(id, "nonce"),
                css.as_bytes(),
            )
        {
            return;
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
        collect_style_sources_from(dom, child, csp, out);
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
    // The merge pushes rules by hand, so the pseudo-class index
    // (Stage 8 Rec 9) is rebuilt here, not at parse time.
    merged.rebuild_pseudo_class_index();
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

/// The sad-tab crash page (Stage 13 Rec 12c), rendered through
/// the same pipeline as a real page. Reached only via the
/// internal `about:crashed` URL the shell navigates a crashed
/// tab to — never fetched, never in history, never user-typed.
/// It has no links: the only way out is the shell's Reload.
pub fn crash_document() -> String {
    r#"<!doctype html>
<html>
<head>
    <title>Pluto: renderer crashed</title>
    <style type="text/css">
    body {
        background-color: #ffffff;
        margin: 0;
        padding: 2em;
    }
    h1 {
        font-size: 24px;
    }
    p {
        font-size: 16px;
    }
    </style>
</head>
<body>
    <h1>Aw, snap! The renderer crashed.</h1>
    <p>This page stopped responding. Press Reload to try again.</p>
</body>
</html>"#
        .to_string()
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
    // Test-only: the top-level import was removed by `cargo clippy --fix`
    // (unused in lib code) but the tests below need it.
    use pluto_net::ResourceLoader;
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
        assert_eq!(
            normalize_url("example.com").unwrap(),
            "https://example.com"
        );
        assert_eq!(
            normalize_url("https://example.com").unwrap(),
            "https://example.com"
        );
        assert_eq!(
            normalize_url("http://localhost:8080/").unwrap(),
            "http://localhost:8080/"
        );
        assert_eq!(
            normalize_url("  example.com  ").unwrap(),
            "https://example.com"
        );
        assert_eq!(normalize_url("").unwrap(), DEFAULT_URL);
        // A bare `host:port` is a host, not a scheme.
        assert_eq!(
            normalize_url("localhost:8080").unwrap(),
            "https://localhost:8080"
        );
        assert_eq!(
            normalize_url("example.com:8080/path").unwrap(),
            "https://example.com:8080/path"
        );
        // The internal blank page is allowed.
        assert_eq!(normalize_url("about:blank").unwrap(), "about:blank");
        assert_eq!(normalize_url("ABOUT:BLANK").unwrap(), "about:blank");
    }

    /// Stage 9 Rec 7 (T11): every non-http(s) scheme is rejected —
    /// `javascript:`, `data:`, `file:`, `ftp:`, and anything else —
    /// whatever the case, and whether or not it carries `://`.
    #[test]
    fn normalize_url_rejects_other_schemes() {
        for input in [
            "javascript:alert(1)",
            "JAVASCRIPT:alert(1)",
            "data:text/html,<h1>hi</h1>",
            "DATA:text/html,<h1>hi</h1>",
            "file:///etc/passwd",
            "FILE:///etc/passwd",
            "ftp://example.com/file.txt",
            "mailto:someone@example.com",
            "blob:https://example.com/1234",
            "view-source:https://example.com/",
            "about:config",
            "chrome://settings",
        ] {
            assert!(
                normalize_url(input).is_err(),
                "{input} must be rejected, got {:?}",
                normalize_url(input)
            );
        }
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

    /// Number of text lines in a display list, counted as distinct
    /// glyph Y values. Glyphs on one line share a Y; a rewrap adds a
    /// line, so this is the direct measure of "wraps into more lines".
    fn line_count(list: &DisplayList) -> usize {
        let mut ys: Vec<f32> = list
            .0
            .iter()
            .filter_map(|item| match item {
                DisplayItem::Glyphs { glyphs, .. } => Some(glyphs.iter().map(|g| g.y)),
                _ => None,
            })
            .flatten()
            .collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        ys.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        ys.len()
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
        // NOTE (2026-11-24): this test used to assert
        // `glyph_total(&narrow) < glyph_total(&wide)` and a 150..=160
        // range for the wide total. Both assumed the sandbox font's
        // wrap points and failed on the CI runner (ubuntu-24.04) while
        // passing in the sandbox: the runner shapes with a different
        // system font, so the text wraps at different points.
        // Reproduced in-sandbox by shaping with DejaVu Sans only
        // (FONTCONFIG_FILE pointing at a DejaVu-only config): wide and
        // narrow both wrap into 4 lines and the totals are equal
        // (153 == 153), so the strict `<` fails — and the old
        // max-glyph-Y assertion would fail too, since equal line
        // counts put the lowest glyph at the same height.
        // What is actually invariant across fonts: the glyph total is
        // (non-space glyphs, constant across wraps for a fixed font)
        // + (surviving spaces), and each line break consumes the
        // space at the break — so the narrow layout, which wraps into
        // at least as many lines as the wide one (a narrower box can
        // never fit more words per line), has at most as many glyphs,
        // with equality when both viewports wrap into the same number
        // of lines. So the font-independent statements are `<=` on
        // totals and `>=` on line counts, not the strict versions.
        let glyph_total = |list: &DisplayList| {
            list.0.iter().fold(0, |acc, item| match item {
                DisplayItem::Glyphs { glyphs, .. } => acc + glyphs.len(),
                _ => acc,
            })
        };
        let wide_total = glyph_total(&wide);
        let narrow_total = glyph_total(&narrow);
        assert!(wide_total > 0, "wide layout shaped no glyphs");
        assert!(narrow_total > 0, "narrow layout shaped no glyphs");
        assert!(
            narrow_total <= wide_total,
            "narrow layout ({narrow_total} glyphs) must not gain glyphs over wide ({wide_total})"
        );
        assert!(
            line_count(&narrow) >= line_count(&wide),
            "narrow layout must wrap into at least as many lines as wide"
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
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (doc, final_url) =
            load_page_cancellable_with_hooks(&fetch, url.clone(), CancellationToken::new(), None, LoadSinks::none(), 0, 0, DEFAULT_VIEWPORT, None, "")
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

    /// Stage 9 Rec 13 (downloads): a navigation whose response
    /// says "download" (`Content-Disposition: attachment`) is
    /// never parsed as a page — the load fails with the filename
    /// instead of rendering the bytes.
    async fn serve_attachment(listener: TcpListener) {
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
                let body = b"<html><body><script>alert(1)</script></body></html>";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Disposition: attachment; filename=\"evil.html\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.write_all(body).await;
            });
        }
    }

    #[tokio::test]
    async fn load_page_attachment_is_not_parsed_as_a_page() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_attachment(listener));
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let err = load_page_cancellable_with_hooks(&fetch, url, CancellationToken::new(), None, LoadSinks::none(), 0, 0, DEFAULT_VIEWPORT, None, "")
            .await
            .expect_err("an attachment must not load as a page");
        assert!(
            err.starts_with("download: evil.html"),
            "the load fails naming the file, never parsing it: {err}"
        );
    }

    /// Stage 11 Rec 1/9/10: a load reports image discoveries to
    /// the image sink (exactly ONE — the selected srcset
    /// candidate, never every candidate) and document-fetch
    /// progress to the progress sink (bytes vs Content-Length).
    #[tokio::test]
    async fn load_page_reports_images_and_progress_via_sinks() {
        const BODY: &str = "<!doctype html><title>Sink page</title>\
            <img src=\"/a.png\" srcset=\"/small.png 1x, /big.png 2x\" alt=\"pic\">";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let Ok(_) = stream.read(&mut buf).await else {
                        return;
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        BODY.len(),
                        BODY
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (img_tx, mut img_rx) = tokio::sync::mpsc::unbounded_channel();
        let (prog_tx, mut prog_rx) = tokio::sync::mpsc::unbounded_channel();
        let sinks = LoadSinks {
            images: Some(img_tx),
            progress: Some(prog_tx),
            module_sources: None,
        };
        let (_, _) = load_page_cancellable_with_hooks(
            &fetch,
            url.clone(),
            CancellationToken::new(),
            None,
            sinks,
            3,
            1,
            DEFAULT_VIEWPORT,
            None,
            "",
        )
        .await
        .expect("page loads");
        // Exactly one image report: the 1x candidate for DPR 1.0 —
        // the 2x candidate and the plain src are never reported.
        let req = img_rx.try_recv().expect("one image report");
        assert_eq!(req.url, format!("http://127.0.0.1:{}/small.png", addr.port()));
        assert!(req.is_imageset, "srcset selection rides the imageset label");
        assert!(img_rx.try_recv().is_err(), "no second image report");
        // Progress: (0, total) on headers, then the full body.
        let (first, total) = prog_rx.try_recv().expect("progress on headers");
        assert_eq!((first, total), (0, Some(BODY.len() as u64)));
        let mut last = (first, total);
        while let Ok(p) = prog_rx.try_recv() {
            last = p;
        }
        assert_eq!(last, (BODY.len() as u64, Some(BODY.len() as u64)));
    }

    /// Stage 13 Rec 12c: the internal crash page renders
    /// locally — no fetch, no gates — and commits as
    /// `about:crashed` so the shell can show it on a dead tab
    /// while the address bar keeps the real URL.
    #[tokio::test]
    async fn load_page_about_crashed_renders_crash_page() {
        let loader = ResourceLoader::new();
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (document, final_url) = load_page_cancellable_with_hooks(
            &fetch,
            "about:crashed".to_string(),
            CancellationToken::new(),
            None,
            LoadSinks::none(),
            0,
            0,
            DEFAULT_VIEWPORT,
            None,
            "",
        )
        .await
        .expect("about:crashed loads without a fetch");
        assert_eq!(final_url, "about:crashed");
        let dom = document.dom();
        let root = dom.root().expect("crash page has a root");
        let text = dom.descendant_text(root);
        assert!(
            text.to_lowercase().contains("renderer crashed"),
            "crash page says the renderer crashed: {text}"
        );
    }

    #[tokio::test]
    async fn load_page_invalid_url_errors() {
        let loader = ResourceLoader::new();
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        // Stage 9 Rec 7 (T11): disallowed schemes fail in the load
        // entry point, before any fetch — the caller renders the
        // error page from this `Err`.
        for url in [
            "ftp://example.com/",
            "javascript:alert(1)",
            "data:text/html,<h1>hi</h1>",
            "file:///etc/passwd",
        ] {
            assert!(
                load_page_cancellable_with_hooks(
                    &fetch,
                    url.to_string(),
                    CancellationToken::new(),
                    None,
                    LoadSinks::none(),
                    0,
                    0,
                    DEFAULT_VIEWPORT,
                    None,
                    "",
                )
                .await
                .is_err(),
                "{url} must fail before fetching"
            );
        }
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
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (doc, _final_url) =
            load_page_cancellable_with_hooks(&fetch, url.clone(), CancellationToken::new(), None, LoadSinks::none(), 0, 0, DEFAULT_VIEWPORT, None, "")
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
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (doc, _) = load_page_cancellable_with_hooks(&fetch, url, CancellationToken::new(), None, LoadSinks::none(), 0, 0, DEFAULT_VIEWPORT, None, "")
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
        let fetch = crate::broker::FetchSource::Direct(std::sync::Arc::new(loader));
        let (doc, _) = load_page_cancellable_with_hooks(&fetch, url, CancellationToken::new(), None, LoadSinks::none(), 0, 0, DEFAULT_VIEWPORT, None, "")
            .await
            .expect("page loads despite the missing sheet");
        let display = layout_for(&doc, (800.0, 600.0));
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect { color, .. } if fill_near(color, (0x00, 0xff, 0x00))
        ));
    }

    /// Walk the DOM for the first node matching `pred` (test helper).
    fn find_node(doc: &Document, pred: impl Fn(&Dom, NodeId) -> bool) -> Option<NodeId> {
        let mut stack = vec![doc.dom.root()?];
        while let Some(id) = stack.pop() {
            if pred(&doc.dom, id) {
                return Some(id);
            }
            stack.extend(doc.dom.children(id).iter().rev());
        }
        None
    }

    /// Stage 15 (M7 Phase 1, Rec 1): `find_link_href` reads an SVG
    /// `<a>`'s `href`, falling back to the legacy `xlink:href` —
    /// and an HTML `<a>` never falls back to `xlink:href`.
    #[test]
    fn find_link_href_svg_anchor_uses_xlink_href() {
        let doc = parse_document_with_base(
            "<svg><a xlink:href=\"/svg-page\"><text>link</text></a></svg>",
            "https://example.com/",
        );
        let text = find_node(&doc, |dom, id| dom.tag(id) == Some("text")).expect("svg text");
        assert_eq!(
            find_link_href(&doc, text).as_deref(),
            Some("https://example.com/svg-page")
        );
        // Plain href wins over xlink:href on an SVG anchor.
        let doc = parse_document_with_base(
            "<svg><a href=\"/plain\" xlink:href=\"/legacy\">x</a></svg>",
            "https://example.com/",
        );
        let a = find_node(&doc, |dom, id| dom.tag(id) == Some("a")).expect("svg a");
        assert_eq!(
            find_link_href(&doc, a).as_deref(),
            Some("https://example.com/plain")
        );
        // An HTML <a> with only xlink:href is not a link.
        let doc = parse_document_with_base(
            "<a xlink:href=\"/legacy\">x</a>",
            "https://example.com/",
        );
        let a = find_node(&doc, |dom, id| dom.tag(id) == Some("a")).expect("html a");
        assert_eq!(find_link_href(&doc, a), None);
    }

    /// Stage 15 (M7 Phase 1, Rec 11): `collect_style_sources`
    /// drops an inline `<style>` the document's CSP forbids, keeps
    /// one its nonce allows, and keeps everything when no policies
    /// are known.
    #[test]
    fn collect_style_sources_csp_blocks_inline_style() {
        let dom = pluto_html::parse(
            "<html><head><style>div { color: red; }</style><style nonce=\"abc\">p { color: blue; }</style></head><body></body></html>",
        );
        let origin = Origin::from_url("https://example.com/");
        let blocked = PolicySet::from_headers(&["style-src 'none'"], &[]);
        assert!(collect_style_sources(&dom, Some((&blocked, &origin))).is_empty());
        let nonce_ok = PolicySet::from_headers(&["style-src 'nonce-abc'"], &[]);
        assert_eq!(
            collect_style_sources(&dom, Some((&nonce_ok, &origin))).len(),
            1
        );
        let empty = PolicySet::from_headers(&[], &[]);
        assert_eq!(
            collect_style_sources(&dom, Some((&empty, &origin))).len(),
            2
        );
        assert_eq!(collect_style_sources(&dom, None).len(), 2);
    }
}
