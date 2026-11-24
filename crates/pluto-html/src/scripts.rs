//! Script-execution seams (Stage 7, Rec 6): the parser-side state
//! the event loop / JS engine drives — script queues, the
//! pending-stylesheet gate, and dynamic markup insertion
//! (`document.write` / `open` / `close`).
//!
//! Nothing here executes JavaScript. That belongs to `pluto-js`
//! (the integration wave): it reads a paused script's text from
//! the `Dom`, runs it, and calls back into
//! [`crate::streaming::Parser`] (`document_write`,
//! `document_open`, `document_close`, `note_stylesheet_loaded`).
//!
//! # The three queues
//!
//! Every `<script>` the parser completes is classified by
//! [`classify_script`] into exactly one destination (HTML
//! "prepare the script element"):
//!
//! * [`ScriptKind::ParserBlocking`] — classic, no `async` /
//!   `defer`, not `type="module"` (all inline classic scripts):
//!   the parser pauses (`DomEvent::ScriptPaused`); the script
//!   runs before parsing resumes, so it sees only the DOM above
//!   itself. It also waits on pending stylesheets (see below).
//! * [`ScriptKind::Defer`] — external classic with `defer`:
//!   document order, after parsing completes, before
//!   `DOMContentLoaded`. `defer` has **no effect on inline**
//!   classic scripts (they stay parser-blocking).
//! * [`ScriptKind::Async`] — classic or module with `async`:
//!   executes as soon as its fetch arrives, possibly mid-parse,
//!   with **no** order guarantee — arrival order only.
//! * [`ScriptKind::Module`] — `type="module"` without `async`:
//!   deferred by default, strict mode, deduped by URL (the same
//!   URL is fetched / executed once per document).
//!
//! * Dynamically injected scripts (`document.createElement('script')`
//!   + append) are async by default — that is a bindings-layer rule:
//!     `pluto-js` enqueues them via [`ScriptQueues::enqueue_async`],
//!     and the parser never sees them.
//!
//! # `DOMContentLoaded` firing point
//!
//! After [`crate::streaming::Parser::finish`], the integration
//! wave calls `parsed.script_queues.drain_defer()` and executes
//! the returned scripts in order; when the queue is empty (and
//! any error in one script has been reported at the per-script
//! boundary without stopping the rest), it fires
//! `DOMContentLoaded`. That drain-then-fire sequence *is* the
//! firing point — there is no parser event for it, because the
//! parser cannot know when the engine has finished running the
//! queue.
//!
//! # Pending-stylesheet gate
//!
//! A parser-blocking script can read computed style, so it must
//! wait for pending stylesheets. The render-blocking CSS state
//! itself lives in `pluto-browser` (it owns the fetch + the
//! render-blocking deadline), so the parser cannot reach it
//! directly — instead the parser tracks the gate as an explicit
//! flag fed by the caller:
//!
//! * the parser records every `<link rel=stylesheet>` it
//!   discovers in a [`StylesheetGate`] (pending until told
//!   otherwise);
//! * `ScriptPaused.stylesheets_pending` snapshots the gate at the
//!   pause, and [`crate::streaming::Parser::stylesheets_pending`]
//!   reports it live;
//! * the browser calls
//!   [`crate::streaming::Parser::note_stylesheet_loaded`] /
//!   `note_stylesheet_failed` as each sheet resolves, and only
//!   executes the paused script once the gate is clear.
//!
//! Known gap (documented, not hacked around): the parser only
//! knows about sheets *it* discovered during this parse. Sheets
//! the preload scanner found before parsing started, `@import`
//! chains inside a fetched sheet, and sheets the browser already
//! fetched are invisible to the gate — the browser must call
//! `note_stylesheet_loaded` for those too, or track its own gate
//! and treat the parser's as a lower bound. The gate also cannot
//! tell a render-blocking sheet from a non-blocking one (e.g.
//! `media` that does not match); the browser should call
//! `note_stylesheet_loaded` immediately for sheets it decides are
//! non-blocking.
//!
//! # Dynamic markup insertion
//!
//! [`DocumentWriter`] holds the `document.write` / `open` /
//! `close` state machine (HTML §8.4). The actual character
//! insertion lives in [`crate::streaming::Parser::document_write`],
//! which splices the markup into the tokenizer at the current
//! insertion point, ahead of buffered network bytes; the writer
//! itself only decides *whether* the call takes effect. The
//! insertion point advances past inserted text, so sequential
//! writes in one script append in call order; a nested script
//! (reached while parsing inserted text) gets a fresh insertion
//! point at its own pause, so its writes land before the
//! not-yet-consumed remainder of the outer insertion.
//!
//! The `open()` guard needs the script nesting level, which only
//! the engine knows: the integration wave must call
//! `DocumentWriter::enter_script` before executing a
//! parser-inserted script and `exit_script` afterwards, or
//! `open()` cannot tell a parser-inserted script from any other
//! task. Likewise the `throw-on-dynamic-markup-insertion`
//! counter: Pluto has no custom elements yet, so nothing
//! increments it today — the counter and the guard are
//! implemented, and the bindings layer must increment it around
//! custom-element constructor invocations from the parser when
//! custom elements land (until then it stays 0).

use std::collections::HashSet;

use crate::NodeId;

/// Which destination a `<script>` element was prepared for (HTML
/// "prepare the script element"). Exactly one per element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptKind {
    /// Classic script, no `async` / `defer`, not `type="module"`
    /// (includes every inline classic script): parser-blocking.
    ParserBlocking,
    /// External classic script with `defer`: document order, after
    /// parsing, before `DOMContentLoaded`.
    Defer,
    /// Classic or module script with `async`: arrival order, no
    /// order guarantee.
    Async,
    /// `type="module"` without `async`: deferred by default.
    Module,
}

/// Classify a `<script>` element from its attributes.
///
/// * `is_module` — `type` is `module` (ASCII case-insensitive).
/// * `has_async` / `has_defer` — the attributes are present.
/// * `has_src` — a non-empty `src` attribute is present.
///
/// `defer` has no effect on inline classic scripts (spec table in
/// HTML §4.12); inline modules *are* deferred by default.
pub fn classify_script(
    is_module: bool,
    has_async: bool,
    has_defer: bool,
    has_src: bool,
) -> ScriptKind {
    if is_module {
        if has_async {
            ScriptKind::Async
        } else {
            ScriptKind::Module
        }
    } else if has_async {
        ScriptKind::Async
    } else if has_defer && has_src {
        ScriptKind::Defer
    } else {
        ScriptKind::ParserBlocking
    }
}

/// One script element in a queue: the node in the live `Dom` plus
/// the routing the parser decided. The script text (inline) or
/// fetched bytes (external) are read from the `Dom` / fetch cache
/// by the consumer — the queue holds no source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedScript {
    /// The `<script>` element in the live `Dom`.
    pub script: NodeId,
    /// Resolved absolute `src`, or `None` for an inline script.
    pub src: Option<String>,
    pub kind: ScriptKind,
    /// `type="module"` (true for [`ScriptKind::Module`] and for
    /// async modules).
    pub is_module: bool,
}

impl QueuedScript {
    pub fn new(script: NodeId, src: Option<String>, kind: ScriptKind, is_module: bool) -> Self {
        Self {
            script,
            src,
            kind,
            is_module,
        }
    }
}

/// Defer queue: external classic `defer` scripts and `type="module"`
/// scripts (defer by default), in document order. Drained once,
/// after parsing completes, before `DOMContentLoaded`.
#[derive(Debug, Default, Clone)]
pub struct DeferQueue {
    scripts: Vec<QueuedScript>,
}

impl DeferQueue {
    pub fn push(&mut self, script: QueuedScript) {
        self.scripts.push(script);
    }

    pub fn scripts(&self) -> &[QueuedScript] {
        &self.scripts
    }

    pub fn len(&self) -> usize {
        self.scripts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }

    /// Drain in document order. The caller executes each script,
    /// then fires `DOMContentLoaded`.
    pub fn drain(&mut self) -> Vec<QueuedScript> {
        std::mem::take(&mut self.scripts)
    }

    pub fn clear(&mut self) {
        self.scripts.clear();
    }
}

/// Async queue: `async` scripts (classic + module). A script is
/// `pending` from discovery until its fetch completes, then
/// `ready` — in *arrival* order, which is the only order
/// guarantee async scripts have. The consumer executes ready
/// scripts as they arrive, possibly mid-parse.
#[derive(Debug, Default, Clone)]
pub struct AsyncQueue {
    pending: Vec<QueuedScript>,
    ready: Vec<QueuedScript>,
}

impl AsyncQueue {
    pub fn enqueue(&mut self, script: QueuedScript) {
        self.pending.push(script);
    }

    /// The fetch for `script` completed: move it from pending to
    /// ready (arrival order). Returns the script, or `None` if it
    /// was never enqueued (or already marked ready).
    pub fn mark_ready(&mut self, script: NodeId) -> Option<QueuedScript> {
        let pos = self.pending.iter().position(|s| s.script == script)?;
        let script = self.pending.remove(pos);
        self.ready.push(script.clone());
        Some(script)
    }

    /// The fetch for `src` completed: every pending script with
    /// that URL becomes ready, in discovery order. (Two classic
    /// async scripts may share a `src`; each still executes.)
    pub fn mark_ready_by_src(&mut self, src: &str) -> Vec<QueuedScript> {
        let mut became_ready = Vec::new();
        let mut still_pending = Vec::new();
        for script in self.pending.drain(..) {
            if script.src.as_deref() == Some(src) {
                became_ready.push(script);
            } else {
                still_pending.push(script);
            }
        }
        self.pending = still_pending;
        self.ready.extend(became_ready.iter().cloned());
        became_ready
    }

    /// Take all ready scripts, in arrival order.
    pub fn take_ready(&mut self) -> Vec<QueuedScript> {
        std::mem::take(&mut self.ready)
    }

    pub fn pending(&self) -> &[QueuedScript] {
        &self.pending
    }

    pub fn ready(&self) -> &[QueuedScript] {
        &self.ready
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.ready.is_empty()
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.ready.clear();
    }
}

/// Both script queues for one document, plus the module map used
/// to dedup by URL. A module `src` already queued — in either
/// queue — is fetched / executed once; a second `<script
/// type="module" src="…">` with the same resolved URL is dropped
/// (the enqueue methods return `false`). Classic scripts are
/// never deduped: two classic scripts with the same `src` each
/// execute.
#[derive(Debug, Default, Clone)]
pub struct ScriptQueues {
    defer: DeferQueue,
    async_queue: AsyncQueue,
    module_urls: HashSet<String>,
}

impl ScriptQueues {
    pub fn defer_queue(&self) -> &DeferQueue {
        &self.defer
    }

    pub fn defer_queue_mut(&mut self) -> &mut DeferQueue {
        &mut self.defer
    }

    pub fn async_queue(&self) -> &AsyncQueue {
        &self.async_queue
    }

    pub fn async_queue_mut(&mut self) -> &mut AsyncQueue {
        &mut self.async_queue
    }

    /// Enqueue a defer / module script. Returns `false` when a
    /// module with this `src` was already queued (deduped — the
    /// caller must not queue or fetch it again).
    pub fn enqueue_defer(&mut self, script: QueuedScript) -> bool {
        // A module URL already queued (in either queue) is
        // dropped: one module map per document.
        let duplicate_module = script.is_module
            && script
                .src
                .as_ref()
                .is_some_and(|src| !self.module_urls.insert(src.clone()));
        if duplicate_module {
            return false;
        }
        self.defer.push(script);
        true
    }

    /// Enqueue an async script. Returns `false` when an async
    /// module with this `src` was already queued (deduped).
    pub fn enqueue_async(&mut self, script: QueuedScript) -> bool {
        let duplicate_module = script.is_module
            && script
                .src
                .as_ref()
                .is_some_and(|src| !self.module_urls.insert(src.clone()));
        if duplicate_module {
            return false;
        }
        self.async_queue.enqueue(script);
        true
    }

    /// Drain the defer queue in document order — the
    /// `DOMContentLoaded` firing point (see module docs).
    pub fn drain_defer(&mut self) -> Vec<QueuedScript> {
        self.defer.drain()
    }

    pub fn is_empty(&self) -> bool {
        self.defer.is_empty() && self.async_queue.is_empty()
    }

    pub fn clear(&mut self) {
        self.defer.clear();
        self.async_queue.clear();
        self.module_urls.clear();
    }
}

/// The pending-stylesheet gate for parser-blocking scripts: the
/// set of render-blocking stylesheet URLs the parser has
/// discovered but which have not yet loaded or errored. Fed by
/// the caller (see module docs): the parser notes discoveries,
/// the browser notes resolutions.
#[derive(Debug, Default, Clone)]
pub struct StylesheetGate {
    pending: HashSet<String>,
}

impl StylesheetGate {
    /// A `<link rel=stylesheet>` was discovered: it blocks until
    /// `note_loaded` / `note_failed` is called for its URL.
    /// Idempotent — re-discovery of the same URL is a no-op.
    pub fn note_discovered(&mut self, href: &str) {
        self.pending.insert(href.to_string());
    }

    /// The sheet loaded: it no longer blocks scripts.
    pub fn note_loaded(&mut self, href: &str) {
        self.pending.remove(href);
    }

    /// The sheet failed (or the render-blocking deadline fired):
    /// it no longer blocks scripts — the page renders without it.
    pub fn note_failed(&mut self, href: &str) {
        self.pending.remove(href);
    }

    /// True while any discovered sheet is still unresolved.
    pub fn is_blocked(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Unresolved sheet URLs, sorted (deterministic for tests).
    pub fn pending(&self) -> Vec<String> {
        let mut urls: Vec<String> = self.pending.iter().cloned().collect();
        urls.sort();
        urls
    }
}

/// What `document.write` decided. The caller
/// ([`crate::streaming::Parser::document_write`]) performs the
/// `Inserted` splice; the other outcomes need no parser action
/// except what their names say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The markup was inserted at the parser's insertion point.
    Inserted,
    /// No active parser: the call follows the implicit-`open()`
    /// path — the caller must clear the document and drive a
    /// script-created parser over the written markup, not append
    /// to the finished document.
    ImplicitOpen,
    /// The active parser was aborted: the write is ignored.
    IgnoredAborted,
}

/// What `document.open` decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenOutcome {
    /// The document is replaced: the caller clears it and drives
    /// a script-created parser. (When this happens mid-parse it
    /// also aborts the in-flight parse.)
    Opened,
    /// Ignored: the document is still being parsed, a
    /// parser-inserted script is executing (nesting level > 0),
    /// and the insertion point is defined.
    IgnoredDuringParse,
}

/// What `document.close` decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseOutcome {
    /// The (script-created) parser is closed; parsing ends.
    Closed,
    /// Ignored: the document is still being parsed by the network
    /// parser, which closes itself at EOF.
    IgnoredWhileParsing,
    /// Ignored: the active parser was aborted.
    IgnoredAborted,
}

/// `open` / `close` / `write` throwing because they ran inside a
/// custom-element constructor invoked by the parser (HTML §8.4:
/// the `throw-on-dynamic-markup-insertion` counter is non-zero).
/// The exception name is `InvalidStateError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMarkupError;

impl DynamicMarkupError {
    /// The DOMException name the bindings layer must surface.
    pub fn name(&self) -> &'static str {
        "InvalidStateError"
    }
}

impl std::fmt::Display for DynamicMarkupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InvalidStateError: document.write/open/close called during parser-invoked custom element construction"
        )
    }
}

impl std::error::Error for DynamicMarkupError {}

/// The `document.write` / `open` / `close` state machine (HTML
/// §8.4, dynamic markup insertion), parser side. Owned by
/// [`crate::streaming::Parser`]; the bindings layer drives it
/// through the `Parser::document_*` methods, which also perform
/// the parser-side effects (character insertion, abort).
#[derive(Debug, Default, Clone)]
pub struct DocumentWriter {
    /// An active parser exists for this document (false after
    /// `finish()` / `close()`, true while the network parser or a
    /// script-created parser is running).
    parsing: bool,
    /// The active parser was created by `document.open()` (a
    /// script-created parser), not by navigation.
    script_created_parser: bool,
    /// How many parser-inserted scripts are currently executing.
    /// The engine maintains this via [`Self::enter_script`] /
    /// [`Self::exit_script`]; it is what makes `open()` a no-op
    /// from a parser-inserted inline script.
    script_nesting_level: usize,
    /// The `throw-on-dynamic-markup-insertion` counter: while
    /// non-zero, `open` / `close` / `write` throw
    /// `InvalidStateError`. Raised around custom-element
    /// constructor invocations from the parser. Pluto has no
    /// custom elements yet, so nothing raises it today — the
    /// counter and the guard are implemented, the raiser is a
    /// documented hook for the bindings layer.
    throw_on_dynamic_markup_insertion: usize,
    /// The `active parser was aborted` flag: while set, `write`
    /// (and `close`) are ignored.
    parser_aborted: bool,
    /// The insertion point is defined (set while the parser is
    /// paused at a parser-inserted script).
    insertion_point_defined: bool,
}

impl DocumentWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called by the parser: a parse is in progress (`true`) or
    /// has finished (`false`, after which `write` takes the
    /// implicit-`open()` path).
    pub fn set_parsing(&mut self, parsing: bool) {
        self.parsing = parsing;
    }

    pub fn parsing(&self) -> bool {
        self.parsing
    }

    pub fn set_script_created_parser(&mut self, script_created: bool) {
        self.script_created_parser = script_created;
    }

    pub fn set_insertion_point_defined(&mut self, defined: bool) {
        self.insertion_point_defined = defined;
    }

    pub fn insertion_point_defined(&self) -> bool {
        self.insertion_point_defined
    }

    /// A parser-inserted script starts executing.
    pub fn enter_script(&mut self) {
        self.script_nesting_level += 1;
    }

    /// A parser-inserted script finished executing.
    pub fn exit_script(&mut self) {
        self.script_nesting_level = self.script_nesting_level.saturating_sub(1);
    }

    pub fn script_nesting_level(&self) -> usize {
        self.script_nesting_level
    }

    /// Raise the `throw-on-dynamic-markup-insertion` counter
    /// (around a custom-element constructor invoked by the
    /// parser). Must be paired with
    /// [`Self::decrement_throw_counter`].
    pub fn increment_throw_counter(&mut self) {
        self.throw_on_dynamic_markup_insertion += 1;
    }

    pub fn decrement_throw_counter(&mut self) {
        self.throw_on_dynamic_markup_insertion =
            self.throw_on_dynamic_markup_insertion.saturating_sub(1);
    }

    pub fn throw_counter(&self) -> usize {
        self.throw_on_dynamic_markup_insertion
    }

    /// Abort the active parser: later `write` / `close` calls are
    /// ignored until a new parser (`open()`) replaces it.
    pub fn abort(&mut self) {
        self.parser_aborted = true;
    }

    pub fn parser_aborted(&self) -> bool {
        self.parser_aborted
    }

    fn throw_if_guarded(&self) -> Result<(), DynamicMarkupError> {
        if self.throw_on_dynamic_markup_insertion > 0 {
            Err(DynamicMarkupError)
        } else {
            Ok(())
        }
    }

    /// `document.open()`: no effect while the document is still
    /// being parsed by a parser-inserted script (active parser,
    /// nesting level > 0, insertion point defined); otherwise the
    /// document is replaced and a script-created parser takes
    /// over.
    pub fn open(&mut self) -> Result<OpenOutcome, DynamicMarkupError> {
        self.throw_if_guarded()?;
        if self.parsing && self.script_nesting_level > 0 && self.insertion_point_defined {
            return Ok(OpenOutcome::IgnoredDuringParse);
        }
        self.parsing = true;
        self.script_created_parser = true;
        self.parser_aborted = false;
        Ok(OpenOutcome::Opened)
    }

    /// `document.close()`: ignored while the network parser is
    /// still running (it closes itself at EOF) and after an abort;
    /// otherwise closes the script-created parser.
    pub fn close(&mut self) -> Result<CloseOutcome, DynamicMarkupError> {
        self.throw_if_guarded()?;
        if self.parser_aborted {
            return Ok(CloseOutcome::IgnoredAborted);
        }
        if self.parsing && !self.script_created_parser {
            return Ok(CloseOutcome::IgnoredWhileParsing);
        }
        self.parsing = false;
        self.script_created_parser = false;
        Ok(CloseOutcome::Closed)
    }

    /// `document.write(...)`: throws while the guard counter is
    /// raised; ignored after an abort; the implicit-`open()` path
    /// when no parser is active; inserted at the insertion point
    /// otherwise.
    pub fn write(&self) -> Result<WriteOutcome, DynamicMarkupError> {
        self.throw_if_guarded()?;
        if self.parser_aborted {
            return Ok(WriteOutcome::IgnoredAborted);
        }
        if !self.parsing {
            return Ok(WriteOutcome::ImplicitOpen);
        }
        Ok(WriteOutcome::Inserted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dom;

    /// Distinct `<script>` node ids from one arena, for queue tests.
    fn fresh_ids(n: usize) -> Vec<NodeId> {
        let mut dom = Dom::new();
        (0..n).map(|_| dom.create_element("script")).collect()
    }

    #[test]
    fn classify_script_kinds() {
        // Classic, no attrs: parser-blocking.
        assert_eq!(
            classify_script(false, false, false, false),
            ScriptKind::ParserBlocking
        );
        // External classic: parser-blocking.
        assert_eq!(
            classify_script(false, false, false, true),
            ScriptKind::ParserBlocking
        );
        // defer: external only — inline defer stays blocking.
        assert_eq!(
            classify_script(false, false, true, false),
            ScriptKind::ParserBlocking
        );
        assert_eq!(classify_script(false, false, true, true), ScriptKind::Defer);
        // async beats defer.
        assert_eq!(classify_script(false, true, true, true), ScriptKind::Async);
        assert_eq!(
            classify_script(false, true, false, false),
            ScriptKind::Async
        );
        // Module: defer by default, async wins.
        assert_eq!(
            classify_script(true, false, false, false),
            ScriptKind::Module
        );
        assert_eq!(classify_script(true, false, true, true), ScriptKind::Module);
        assert_eq!(classify_script(true, true, false, true), ScriptKind::Async);
    }

    #[test]
    fn defer_queue_keeps_document_order() {
        let mut q = ScriptQueues::default();
        let ids = fresh_ids(3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);
        assert!(q.enqueue_defer(QueuedScript::new(
            a,
            Some("https://x/a.js".into()),
            ScriptKind::Defer,
            false
        )));
        assert!(q.enqueue_defer(QueuedScript::new(
            b,
            Some("https://x/b.js".into()),
            ScriptKind::Module,
            true
        )));
        assert!(q.enqueue_defer(QueuedScript::new(c, None, ScriptKind::Module, true)));
        let drained = q.drain_defer();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].script, a);
        assert_eq!(drained[1].script, b);
        assert_eq!(drained[2].script, c);
        assert!(q.defer_queue().is_empty());
    }

    #[test]
    fn module_dedup_by_url_across_queues() {
        let mut q = ScriptQueues::default();
        let ids = fresh_ids(8);
        let (a, b, c) = (ids[0], ids[1], ids[2]);
        let url = "https://x/m.js".to_string();
        assert!(q.enqueue_defer(QueuedScript::new(
            a,
            Some(url.clone()),
            ScriptKind::Module,
            true
        )));
        // Same URL, second module element: dropped.
        assert!(!q.enqueue_defer(QueuedScript::new(
            b,
            Some(url.clone()),
            ScriptKind::Module,
            true
        )));
        // Same URL as an async module: also dropped (one module map).
        assert!(!q.enqueue_async(QueuedScript::new(c, Some(url), ScriptKind::Async, true)));
        assert_eq!(q.defer_queue().len(), 1);
        // A different URL queues fine, and classic scripts are never deduped.
        let d = ids[3];
        assert!(q.enqueue_defer(QueuedScript::new(
            d,
            Some("https://x/other.js".into()),
            ScriptKind::Module,
            true
        )));
        let e = ids[4];
        let f = ids[5];
        assert!(q.enqueue_async(QueuedScript::new(
            e,
            Some("https://x/a.js".into()),
            ScriptKind::Async,
            false
        )));
        assert!(q.enqueue_async(QueuedScript::new(
            f,
            Some("https://x/a.js".into()),
            ScriptKind::Async,
            false
        )));
        assert_eq!(q.async_queue().pending_len(), 2);
        // Inline modules have no URL: each executes.
        let g = ids[6];
        let h = ids[7];
        assert!(q.enqueue_defer(QueuedScript::new(g, None, ScriptKind::Module, true)));
        assert!(q.enqueue_defer(QueuedScript::new(h, None, ScriptKind::Module, true)));
        assert_eq!(q.defer_queue().len(), 4);
    }

    #[test]
    fn async_queue_arrival_order_only() {
        // Arrival order is the only guarantee: scripts marked
        // ready out of discovery order come out in arrival order.
        let mut q = AsyncQueue::default();
        let ids = fresh_ids(2);
        let (a, b) = (ids[0], ids[1]);
        q.enqueue(QueuedScript::new(
            a,
            Some("https://x/a.js".into()),
            ScriptKind::Async,
            false,
        ));
        q.enqueue(QueuedScript::new(
            b,
            Some("https://x/b.js".into()),
            ScriptKind::Async,
            false,
        ));
        assert_eq!(q.pending_len(), 2);
        // b's fetch wins the race: b is ready first.
        assert_eq!(q.mark_ready(b).map(|s| s.script), Some(b));
        assert_eq!(q.mark_ready(a).map(|s| s.script), Some(a));
        let ready = q.take_ready();
        assert_eq!(ready.len(), 2);
        assert_eq!(ready[0].script, b, "arrival order, not document order");
        assert_eq!(ready[1].script, a);
        assert!(q.take_ready().is_empty());
        // Unknown node: no-op.
        assert!(q.mark_ready(a).is_none());
    }

    #[test]
    fn async_mark_ready_by_src() {
        let mut q = AsyncQueue::default();
        let ids = fresh_ids(3);
        let (a, b) = (ids[0], ids[1]);
        q.enqueue(QueuedScript::new(
            a,
            Some("https://x/s.js".into()),
            ScriptKind::Async,
            false,
        ));
        q.enqueue(QueuedScript::new(
            b,
            Some("https://x/s.js".into()),
            ScriptKind::Async,
            false,
        ));
        q.enqueue(QueuedScript::new(
            ids[2],
            Some("https://x/t.js".into()),
            ScriptKind::Async,
            false,
        ));
        let ready = q.mark_ready_by_src("https://x/s.js");
        assert_eq!(
            ready.len(),
            2,
            "both classic scripts sharing a src become ready"
        );
        assert_eq!(q.pending_len(), 1);
        assert_eq!(q.take_ready().len(), 2);
    }

    #[test]
    fn stylesheet_gate_blocks_until_resolved() {
        let mut gate = StylesheetGate::default();
        assert!(!gate.is_blocked());
        gate.note_discovered("https://x/a.css");
        gate.note_discovered("https://x/b.css");
        gate.note_discovered("https://x/a.css"); // re-discovery: no-op
        assert!(gate.is_blocked());
        assert_eq!(gate.pending(), vec!["https://x/a.css", "https://x/b.css"]);
        gate.note_failed("https://x/a.css");
        assert!(gate.is_blocked());
        gate.note_loaded("https://x/b.css");
        assert!(!gate.is_blocked());
        assert!(gate.pending().is_empty());
    }

    #[test]
    fn document_writer_open_ignored_during_parse() {
        let mut w = DocumentWriter::new();
        w.set_parsing(true);
        w.set_insertion_point_defined(true);
        // No script executing yet: open() takes effect.
        w.enter_script();
        assert_eq!(w.open(), Ok(OpenOutcome::IgnoredDuringParse));
        w.exit_script();
        // After the script returns, open() works.
        assert_eq!(w.open(), Ok(OpenOutcome::Opened));
        assert!(w.parsing());
    }

    #[test]
    fn document_writer_write_paths() {
        let mut w = DocumentWriter::new();
        // No active parser: implicit open, never a silent append.
        assert_eq!(w.write(), Ok(WriteOutcome::ImplicitOpen));
        w.set_parsing(true);
        assert_eq!(w.write(), Ok(WriteOutcome::Inserted));
        // Aborted parser blocks writes.
        w.abort();
        assert_eq!(w.write(), Ok(WriteOutcome::IgnoredAborted));
        assert_eq!(w.close(), Ok(CloseOutcome::IgnoredAborted));
        // open() after abort starts fresh.
        assert_eq!(w.open(), Ok(OpenOutcome::Opened));
        assert!(!w.parser_aborted());
        assert_eq!(w.write(), Ok(WriteOutcome::Inserted));
    }

    #[test]
    fn document_writer_close_while_network_parsing() {
        let mut w = DocumentWriter::new();
        w.set_parsing(true);
        assert_eq!(w.close(), Ok(CloseOutcome::IgnoredWhileParsing));
        // Script-created parser: close ends it.
        w.set_script_created_parser(true);
        assert_eq!(w.close(), Ok(CloseOutcome::Closed));
        assert!(!w.parsing());
    }

    #[test]
    fn document_writer_throw_counter_guards_all_three() {
        let mut w = DocumentWriter::new();
        w.set_parsing(true);
        w.increment_throw_counter();
        assert_eq!(w.write(), Err(DynamicMarkupError));
        assert_eq!(w.open(), Err(DynamicMarkupError));
        assert_eq!(w.close(), Err(DynamicMarkupError));
        assert_eq!(DynamicMarkupError.name(), "InvalidStateError");
        w.decrement_throw_counter();
        assert_eq!(w.throw_counter(), 0);
        assert_eq!(w.write(), Ok(WriteOutcome::Inserted));
        assert_eq!(w.open(), Ok(OpenOutcome::Opened));
    }
}
