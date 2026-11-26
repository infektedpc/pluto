//! Wave 2b integration (Stage 7, Recommendations 7–10): the real
//! [`DomHost`] for the per-tab worker, the `fetch()` backend over
//! `pluto-net`, and the [`LoadHooks`] / [`ScriptExecutor`] pair that drives
//! the parser seams from `pluto-html`.
//!
//! This module lives in `pluto-browser` (never in `pluto-js`):
//! `pluto-js` must not depend on `pluto-layout`, and the host is the
//! piece that knows about the layout actor, the parser, and the
//! network — so it lives on the browser side of the boundary and
//! talks to layout only through the actor channel
//! ([`crate::renderer::LayoutJob`]).
//!
//! # Threading
//!
//! Everything here runs on the tab's worker thread (the persistent
//! tab thread in `renderer.rs`), which owns the `Dom`, the `Parser`
//! (while parsing), and the [`TabJs`]. The layout actor owns the
//! `LayoutTree` on its own thread; the only synchronous cross-thread
//! call is a geometry read, which blocks on a one-shot reply from
//! the actor (see [`BrowserDomHost::geometry`]).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use pluto_css::{Selector, Stylesheet};
use pluto_html::observers::{MutationObserverOptions, ObserverMutationRecord};
use pluto_html::{
    Dom, NodeId, NodeKind, Parser, QueuedScript, ScriptPaused, parse_fragment,
    resolve_url,
};
use pluto_ipc::{Destination, Origin, Priority, RequestMode, ResourceRequest};
use pluto_js::host::{AnimateSpec, AnimationInfo, WaapiOp};
use pluto_js::{
    DomHost, DomHostError, DomKey, FetchBackend, FetchCompletion, FetchError, FetchRequest,
    FetchResponse, Geometry, JsError, RealmConfig, ResponseType, TabJs,
};
use pluto_layout::{ElementState, selector_matches};
use pluto_net::ResourceLoader;
use pluto_paint::perf;
use tokio::sync::{mpsc::UnboundedSender, oneshot};
use tokio_util::sync::CancellationToken;

use crate::page::{Document, ScriptHooks, script_text};
use crate::renderer::LayoutJob;

// TODO(M5): workers (dedicated/shared/service) — each needs its own
// agent/loop/realm; the per-tab `TabJs` here is the window agent only.
// TODO(M5): Shadow DOM — no shadow-root host methods exist yet; the
// `DomHost` surface would gain `attach_shadow` / `shadow_root` here.
// (IntersectionObserver landed as Stage 8 Rec 10 — see
// `crate::renderer::run_intersection_observer_step` and
// `pluto-js::observers::IntersectionObserverRegistry`.)
// TODO(M5): import maps — module `import` specifiers resolve as
// relative/absolute URLs only in M4; an import-map layer would land
// in the module loader the runner uses.
// TODO(M5): storage APIs (`localStorage` / `IndexedDB` / cookies) —
// no storage host methods exist; they belong to a later stage, not
// to this host.

/// How long a forced geometry flush waits for the layout actor
/// before giving up (a dead actor must not hang the JS thread).
const GEOMETRY_TIMEOUT: Duration = Duration::from_secs(5);

/// The production [`DomHost`]: a [`Dom`] shared with the parser
/// (during a parse) or owned by the tab thread (after it), plus the
/// handles the host needs to reach the layout actor, the parser's
/// `document.write` state machine, and the document's stylesheet /
/// base URL for forced-flush rebuilds.
///
/// The `Dom` mutators already set the Double Dirty Bit flags
/// (`pluto-html` `note_mutation`), except while the parser's
/// `suppress_records` privilege is on — so every mutating method
/// here also calls [`Dom::mark_dirty`] explicitly. A mutation made
/// by a parser-blocking script mid-parse therefore still reaches
/// the layout pass.
pub struct BrowserDomHost {
    /// Live document. During a parse this is the parser's shared
    /// `Dom` (`Parser::shared_dom`); after the parse it stays the
    /// tab thread's live `Dom` (the finished `Document`'s arena is
    /// a clone of it at finish time, and the tab thread re-syncs
    /// from here on every re-layout).
    dom: Arc<Mutex<Dom>>,
    /// The active parser, while one exists. `document.write` /
    /// `open` / `close` drive its `DocumentWriter` through this
    /// handle; `None` after the parse (the post-parse implicit-`open`
    /// path applies instead).
    parser: Option<Rc<RefCell<Parser>>>,
    /// Layout actor channel (owned by the tab thread).
    layout_tx: mpsc::Sender<LayoutJob>,
    /// Current layout viewport (CSS px), for forced-flush rebuilds.
    viewport: (f32, f32),
    /// Dynamic pseudo-class state, for forced-flush rebuilds.
    element_state: ElementState,
    /// Merged author stylesheet of the current document (empty
    /// during a parse; set from the finished `Document`).
    sheet: Stylesheet,
    /// Document base URL (final URL after redirects).
    base_url: String,
    /// Forced synchronous layout flushes so far (debug counter —
    /// layout thrash is measurable through this).
    forced_flushes: u64,
    /// `window.onerror` / host error reports, shared with the tab
    /// thread (which logs them to the shell).
    errors: Rc<std::sync::Mutex<Vec<String>>>,
    /// Set when a mutation marks the document dirty; the tab
    /// thread's frame tick reads and clears it, then asks the loop
    /// for a rendering opportunity.
    render_requested: Rc<Cell<bool>>,
    /// A post-parse `document.write` already took the implicit-`open`
    /// path once: later writes append to `<body>` instead of
    /// clearing the document again (M4 approximation of the
    /// script-created parser stream — see `document_write`).
    implicit_open_done: bool,
    /// Navigation requested by JS (`location.assign` / `replace` /
    /// `reload`, `location.href = ...`): the tab thread drains it
    /// after each JS run and performs the navigation. Last
    /// request wins — a script that assigns twice in one turn
    /// navigates to the last URL only.
    pending_navigation: Option<(String, bool)>,
    /// Image loads requested by JS (`Image().src` / an `<img>`
    /// `src` set from JS): the tab thread drains these and routes
    /// them through the Stage 11 store, which reports back via
    /// `TabJs::notify_image_loaded` / `notify_image_failed`.
    pending_images: Vec<crate::images::ImageRequest>,
    /// `setCustomValidity` messages by node (Stage 13 Rec 8).
    custom_validity: std::collections::HashMap<DomKey, String>,
    /// Form submissions queued by JS (`form.submit()` /
    /// `requestSubmit()`): the tab thread drains these and
    /// navigates through the loader (Stage 13 Rec 6/7).
    pending_form_submits: Vec<(DomKey, Option<DomKey>)>,
    /// Focus change queued by JS (`el.focus()` / `el.blur()`):
    /// `None` = no pending change, `Some(Some(k))` = focus `k`,
    /// `Some(None)` = blur (Stage 13 Rec 3). The tab thread
    /// drains it after each JS run and moves `state.focused`.
    pending_focus: Option<Option<DomKey>>,
    /// Focused node mirror (kept in sync with the tab thread's
    /// `state.focused` via `set_focused`): `activeElement`
    /// reads this.
    focused: Option<DomKey>,
    /// Shared cookie jar for `document.cookie` (Stage 13 Rec
    /// 10c). `None` until the tab thread attaches the loader's
    /// jar with `set_cookie_jar`.
    cookie_jar: Option<Arc<std::sync::Mutex<pluto_net::CookieJar>>>,
    /// Computed-style cache for JS (M6 Phase 3, Rec 11): the tab
    /// thread merges the actor's `changed_styles` (underlying
    /// values) and the engine's in-flight samples here, so
    /// `getComputedStyle` reads current values without a layout
    /// round trip. Cleared on navigation.
    computed_styles: std::collections::HashMap<DomKey, Vec<(String, String)>>,
    /// WAAPI animation ids that finished on an element (M6 Phase
    /// 3, Rec 11): the tab thread records engine finishes here so
    /// `getAnimations` can report their `playState`. Cleared on
    /// navigation.
    waapi_finished: std::collections::HashSet<(DomKey, u64)>,
    /// WAAPI animations queued by JS (`Element.animate`): the tab
    /// thread drains these each tick and starts them in the
    /// animation engine. `(element, host anim id, spec)`.
    pending_animate: Vec<(DomKey, u64, AnimateSpec)>,
    /// WAAPI play-state controls queued by JS (`pause` / `play` /
    /// `cancel` / `finish` on an `Animation`): drained per tick.
    pending_waapi_control: Vec<(DomKey, u64, WaapiOp)>,
    /// CSS-animation play-state controls queued by JS (same ops,
    /// by animation name): drained per tick.
    pending_css_control: Vec<(DomKey, String, WaapiOp)>,
    /// Next WAAPI animation id (host-allocated, monotonic across
    /// navigations — a stale id can never address a new animation).
    next_waapi_id: u64,
    /// WAAPI animations the tab thread has started, by (element,
    /// id) — the spec is kept so `getAnimations` can report them.
    waapi_known: std::collections::HashMap<(DomKey, u64), AnimateSpec>,
    /// CSS animations currently running per element, by name —
    /// the tab thread refreshes this from the engine (starts on
    /// `process_animation_outcome`, ends on `animationend`).
    css_animations: std::collections::HashMap<DomKey, Vec<String>>,
}

impl BrowserDomHost {
    /// Create a host over `dom` (the parser's shared `Dom` at
    /// navigation start).
    pub fn new(
        dom: Arc<Mutex<Dom>>,
        layout_tx: mpsc::Sender<LayoutJob>,
        viewport: (f32, f32),
        element_state: ElementState,
        base_url: &str,
        errors: Rc<std::sync::Mutex<Vec<String>>>,
        render_requested: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            dom,
            parser: None,
            layout_tx,
            viewport,
            element_state,
            sheet: Stylesheet::default(),
            base_url: base_url.to_string(),
            forced_flushes: 0,
            errors,
            render_requested,
            implicit_open_done: false,
            pending_navigation: None,
            pending_images: Vec::new(),
            custom_validity: std::collections::HashMap::new(),
            pending_form_submits: Vec::new(),
            pending_focus: None,
            focused: None,
            cookie_jar: None,
            computed_styles: std::collections::HashMap::new(),
            waapi_finished: std::collections::HashSet::new(),
            pending_animate: Vec::new(),
            pending_waapi_control: Vec::new(),
            pending_css_control: Vec::new(),
            next_waapi_id: 1,
            waapi_known: std::collections::HashMap::new(),
            css_animations: std::collections::HashMap::new(),
        }
    }

    /// Attach / detach the active parser (the runner sets this at
    /// navigation start and clears it when the parse finishes).
    pub fn set_parser(&mut self, parser: Option<Rc<RefCell<Parser>>>) {
        self.parser = parser;
    }

    /// Re-point the live `Dom` (an encoding restart swaps the
    /// parser — and its arena — for a fresh one).
    pub fn set_dom(&mut self, dom: Arc<Mutex<Dom>>) {
        self.dom = dom;
    }

    /// Update the viewport the forced-flush rebuilds use.
    pub fn set_viewport(&mut self, viewport: (f32, f32)) {
        self.viewport = viewport;
    }

    /// Update the element state the forced-flush rebuilds use.
    pub fn set_element_state(&mut self, state: ElementState) {
        self.element_state = state;
    }

    /// Take the finished document's stylesheet / base URL (the
    /// sheet is only known once the parse has merged it).
    pub fn set_document(&mut self, document: &Document) {
        self.sheet = document.sheet().clone();
        self.base_url = document.base_url().to_string();
    }

    /// Point the base URL at a navigation's requested URL (the
    /// tab thread calls this in `start_navigation`, before the
    /// load runs): `location` reads the requested URL while the
    /// parse is still running, and `set_document` corrects it to
    /// the final URL after redirects when the parse finishes.
    pub fn set_base_url(&mut self, url: &str) {
        self.base_url = url.to_string();
    }

    /// Attach the shared cookie jar (called by the tab thread
    /// once the loader exists — Stage 13 Rec 10c).
    pub fn set_cookie_jar(&mut self, jar: Arc<std::sync::Mutex<pluto_net::CookieJar>>) {
        self.cookie_jar = Some(jar);
    }

    /// Take a JS-queued focus change for the tab thread to apply
    /// (Stage 13 Rec 3).
    pub fn take_pending_focus(&mut self) -> Option<Option<DomKey>> {
        self.pending_focus.take()
    }

    /// Mirror a tab-thread focus change (pointer-down / Tab key)
    /// into the host, so `activeElement` stays true.
    pub fn set_focused(&mut self, focused: Option<DomKey>) {
        self.focused = focused;
    }

    /// Merge computed-style pairs for one element into the JS
    /// cache (M6 Phase 3, Rec 11): the tab thread feeds the
    /// actor's `changed_styles` and the engine's in-flight
    /// samples here; same-name properties replace, others keep
    /// their last value.
    pub fn set_computed_styles(&mut self, id: DomKey, pairs: Vec<(String, String)>) {
        let entry = self.computed_styles.entry(id).or_default();
        for (prop, value) in pairs {
            if let Some((_, old)) = entry.iter_mut().find(|(p, _)| *p == prop) {
                *old = value;
            } else {
                entry.push((prop, value));
            }
        }
    }

    /// Drop the whole computed-style cache + WAAPI finish records
    /// (navigation boundary).
    pub fn clear_computed_style_cache(&mut self) {
        self.computed_styles.clear();
        self.waapi_finished.clear();
    }

    /// Record that a WAAPI animation finished on an element (the
    /// tab thread calls this from the engine outcome, so
    /// `getAnimations` can report `finished`).
    pub fn mark_waapi_finished(&mut self, id: DomKey, anim_id: u64) {
        self.waapi_finished.insert((id, anim_id));
    }

    /// Whether a WAAPI animation on an element has finished.
    pub fn waapi_is_finished(&self, id: DomKey, anim_id: u64) -> bool {
        self.waapi_finished.contains(&(id, anim_id))
    }

    /// Take the pending JS navigation request, if any (last
    /// request wins; see the field docs).
    pub fn take_pending_navigation(&mut self) -> Option<(String, bool)> {
        self.pending_navigation.take()
    }

    /// Take the pending JS image requests, if any.
    pub fn take_pending_images(&mut self) -> Vec<crate::images::ImageRequest> {
        std::mem::take(&mut self.pending_images)
    }

    /// Take the pending WAAPI animation starts, if any (M6 Phase
    /// 3, Rec 11): `(element, host anim id, spec)`.
    pub fn take_pending_animate(&mut self) -> Vec<(DomKey, u64, AnimateSpec)> {
        std::mem::take(&mut self.pending_animate)
    }

    /// Take the pending WAAPI play-state controls, if any.
    pub fn take_pending_waapi_control(&mut self) -> Vec<(DomKey, u64, WaapiOp)> {
        std::mem::take(&mut self.pending_waapi_control)
    }

    /// Take the pending CSS-animation play-state controls, if any.
    pub fn take_pending_css_control(&mut self) -> Vec<(DomKey, String, WaapiOp)> {
        std::mem::take(&mut self.pending_css_control)
    }

    /// Record that the tab thread started a WAAPI animation (so
    /// `getAnimations` reports it from then on).
    pub fn note_waapi_started(&mut self, id: DomKey, anim_id: u64, spec: AnimateSpec) {
        self.waapi_known.insert((id, anim_id), spec);
    }

    /// Record that a CSS animation started on an element (called
    /// from `process_animation_outcome` for each pending animation).
    pub fn note_css_animation_started(&mut self, id: DomKey, name: String) {
        let names = self.css_animations.entry(id).or_default();
        if !names.contains(&name) {
            names.push(name);
        }
    }

    /// Forget a WAAPI animation entirely (cancel): it drops
    /// out of `getAnimations` (and its finish record, if any).
    pub fn waapi_forget(&mut self, id: DomKey, anim_id: u64) {
        self.waapi_known.remove(&(id, anim_id));
        self.waapi_finished.remove(&(id, anim_id));
    }

    /// Forget a CSS animation by name (cancel): it drops out of
    /// `getAnimations`.
    pub fn css_forget(&mut self, id: DomKey, name: &str) {
        if let Some(names) = self.css_animations.get_mut(&id) {
            names.retain(|n| n != name);
            if names.is_empty() {
                self.css_animations.remove(&id);
            }
        }
    }

    /// Replace the running CSS animation names for an element
    /// (called after an `animationend` — the engine is the
    /// authority, so the finished name drops out here).
    pub fn set_css_animations(&mut self, id: DomKey, names: Vec<String>) {
        if names.is_empty() {
            self.css_animations.remove(&id);
        } else {
            self.css_animations.insert(id, names);
        }
    }

    /// The `<img>` nodes whose resolved `src` is `url`: the
    /// notify targets for an image outcome on that URL. Attached
    /// and detached alike — a `new Image()` element lives in the
    /// arena with no parent until it is inserted, so this scans
    /// the whole arena, not just the tree from the root. Two
    /// elements sharing one URL both report (each has its own
    /// `load` / `error`); an element whose `src` moved on since
    /// the fetch started no longer matches, so it hears nothing
    /// about the stale load.
    pub fn image_targets(&self, url: &str) -> Vec<DomKey> {
        let dom = self.dom.lock().unwrap();
        let base = self.base_url.clone();
        dom.all_node_ids()
            .into_iter()
            .filter(|&id| {
                dom.tag(id) == Some("img")
                    && dom
                        .attr(id, "src")
                        .is_some_and(|src| resolve_url(&base, src) == url)
            })
            .collect()
    }

    /// Deep- (`deep` = true) or shallow-clone `key` inside the
    /// live `Dom`, returning the detached clone's key. Same
    /// shape as the test host's `clone_within` (kind + attributes
    /// copied, children only when `deep`); `<script>` elements
    /// clone like any other element — they are inert, never
    /// executed. Unlike `clone_from` (fragment `Dom` → live
    /// `Dom`), both sides here are the one live arena.
    fn clone_within(&self, dom: &mut Dom, key: DomKey, deep: bool) -> Option<DomKey> {
        let (kind, attrs) = match dom.get(key) {
            Some(n) => (n.kind.clone(), n.attrs.clone()),
            None => return None,
        };
        let new_key = match &kind {
            NodeKind::Element { tag } => dom.create_element_with_attrs(tag, attrs),
            NodeKind::Text(data) => dom.create_text(data),
            NodeKind::Comment(data) => dom.create_comment(data),
            NodeKind::Doctype {
                name,
                public_id,
                system_id,
            } => dom.create_doctype(name, public_id, system_id),
            NodeKind::ProcessingInstruction { target, data } => {
                dom.create_pi(target, data)
            }
            NodeKind::Document => dom.create_document(),
        };
        if deep {
            let kids: Vec<NodeId> = dom.children(key).to_vec();
            for kid in kids {
                if let Some(cloned) = self.clone_within(dom, kid, true) {
                    dom.append_child(new_key, cloned);
                }
            }
        }
        Some(new_key)
    }

    /// Snapshot the live `Dom` (the tab thread re-syncs the
    /// `Document` it hands to the layout actor from this).
    pub fn dom_snapshot(&self) -> Dom {
        self.dom.lock().unwrap().clone()
    }

    /// The document base URL (final URL after redirects): an
    /// inline module's imports resolve against this (Stage 12
    /// Rec 1 — "inline.js" is not a URL).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The inline source text of a `<script>` element in the live
    /// `Dom` (its child text nodes concatenated).
    pub fn inline_script_text(&self, node: NodeId) -> String {
        let dom = self.dom.lock().unwrap();
        script_text(&dom, node)
    }

    /// Build the `Document` the layout actor should lay out right
    /// now: the live `Dom` plus the current sheet / base URL.
    pub fn current_document(&self) -> Document {
        Document::from_parts(self.dom_snapshot(), self.sheet.clone(), &self.base_url)
    }

    /// True while any node carries dirty bits (the tab thread's
    /// frame tick uses this to decide whether to re-layout).
    pub fn dom_is_dirty(&self) -> bool {
        !self.dom.lock().unwrap().dirty_set().is_empty()
    }

    /// Clear every dirty flag (called by the tab thread after a
    /// re-layout has consumed them).
    pub fn clear_dirty(&mut self) {
        self.dom.lock().unwrap().clear_dirty();
    }

    /// Mark nodes dirty on the live `Dom` (Stage 8 Rec 9): a
    /// `:hover`/`:active`/`:focus` restyle marks the changed
    /// elements, so the next re-layout re-resolves their styles
    /// (the incremental path reads the dirty set from the
    /// snapshot). The tab thread owns this host; JS mutators mark
    /// dirty through their own path.
    pub fn mark_dirty(&mut self, ids: &[NodeId]) {
        let mut dom = self.dom.lock().unwrap();
        for &id in ids {
            dom.mark_dirty(id);
        }
    }

    /// First descendant element of `id` (inclusive) with this tag.
    fn find_tag(&self, dom: &Dom, id: DomKey, tag: &str) -> Option<DomKey> {
        if dom.tag(id) == Some(tag) {
            return Some(id);
        }
        for &child in dom.children(id) {
            if let Some(found) = self.find_tag(dom, child, tag) {
                return Some(found);
            }
        }
        None
    }

    /// True when `ancestor` is an ancestor of `node` (or `node`
    /// itself).
    fn is_ancestor_or_self(&self, dom: &Dom, ancestor: DomKey, node: DomKey) -> bool {
        let mut cur = Some(node);
        while let Some(id) = cur {
            if id == ancestor {
                return true;
            }
            cur = dom.get(id).and_then(|n| n.parent);
        }
        false
    }

    /// Concatenated text of a subtree.
    fn subtree_text(&self, dom: &Dom, id: DomKey, out: &mut String) {
        if let Some(t) = dom.text(id) {
            out.push_str(t);
        }
        for &child in dom.children(id) {
            self.subtree_text(dom, child, out);
        }
    }

    /// Serialise one node (and its subtree) as HTML.
    fn serialise_node(&self, dom: &Dom, id: DomKey, out: &mut String) {
        let Some(node) = dom.get(id) else {
            return;
        };
        match &node.kind {
            NodeKind::Text(data) => out.push_str(&escape_text(data)),
            NodeKind::Comment(data) => {
                out.push_str("<!--");
                out.push_str(data);
                out.push_str("-->");
            }
            NodeKind::Element { tag } => {
                out.push('<');
                out.push_str(tag);
                for (name, value) in &node.attrs {
                    out.push(' ');
                    out.push_str(name);
                    out.push_str("=\"");
                    out.push_str(&escape_attr(value));
                    out.push('"');
                }
                if is_void(tag) {
                    out.push('>');
                } else {
                    out.push('>');
                    for &child in dom.children(id) {
                        self.serialise_node(dom, child, out);
                    }
                    out.push_str("</");
                    out.push_str(tag);
                    out.push('>');
                }
            }
            NodeKind::Document => {
                for &child in dom.children(id) {
                    self.serialise_node(dom, child, out);
                }
            }
            NodeKind::Doctype { name, .. } => {
                out.push_str("<!DOCTYPE ");
                out.push_str(name);
                out.push('>');
            }
            NodeKind::ProcessingInstruction { target, data } => {
                out.push_str("<?");
                out.push_str(target);
                out.push(' ');
                out.push_str(data);
                out.push_str("?>");
            }
        }
    }

    /// Deep-clone a node from a fragment `Dom` into the live `Dom`,
    /// returning the new key. `<script>` elements are cloned like
    /// any other element — they are inert (never executed).
    fn clone_from(&self, dom: &mut Dom, frag: &Dom, id: NodeId) -> Option<DomKey> {
        let node = frag.get(id)?;
        let new_key = match &node.kind {
            NodeKind::Element { tag } => {
                dom.create_element_with_attrs(tag, node.attrs.clone())
            }
            NodeKind::Text(data) => dom.create_text(data),
            NodeKind::Comment(data) => dom.create_comment(data),
            NodeKind::Doctype {
                name,
                public_id,
                system_id,
            } => dom.create_doctype(name, public_id, system_id),
            NodeKind::ProcessingInstruction { target, data } => {
                dom.create_pi(target, data)
            }
            NodeKind::Document => dom.create_document(),
        };
        let kids: Vec<NodeId> = frag.children(id).to_vec();
        for kid in kids {
            if let Some(cloned) = self.clone_from(dom, frag, kid) {
                dom.append_child(new_key, cloned);
            }
        }
        Some(new_key)
    }

    /// Replace (`replace = true`, `innerHTML`) or append
    /// (`replace = false`, post-parse `document.write`) `key`'s
    /// children with the fragment parsed from `html` in `key`'s
    /// context. Fragment `<script>` elements are inert.
    fn set_fragment(&self, dom: &mut Dom, key: DomKey, html: &str, replace: bool) -> bool {
        if dom.get(key).is_none() {
            return false;
        }
        let context_tag = dom.tag(key).unwrap_or("div").to_string();
        let (frag, ctx) = parse_fragment(html, &context_tag, None);
        let kids: Vec<NodeId> = frag.children(ctx).to_vec();
        if replace {
            let old: Vec<NodeId> = dom.children(key).to_vec();
            for child in old {
                dom.remove_from_parent(child);
            }
        }
        for kid in kids {
            if let Some(cloned) = self.clone_from(dom, &frag, kid) {
                dom.append_child(key, cloned);
            }
        }
        dom.mark_dirty(key);
        true
    }

    /// Split a selector list on top-level commas (commas inside
    /// `:not(...)` / attribute values do not split).
    fn parse_selector_list(raw: &str) -> Vec<Selector> {
        // The stylesheet's own parser (one invalid complex
        // selector drops the whole list, per CSS Syntax L3).
        pluto_css::parse_selectors(raw)
    }
}

fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(s: &str) -> String {
    escape_text(s).replace('"', "&quot;")
}

fn is_void(tag: &str) -> bool {
    matches!(
        tag,
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input" | "link" | "meta"
            | "source" | "track" | "wbr"
    )
}

impl DomHost for BrowserDomHost {
    fn validate(&self, key: DomKey) -> bool {
        self.dom.lock().unwrap().get(key).is_some()
    }

    fn document_node(&self) -> Option<DomKey> {
        self.dom.lock().unwrap().root()
    }

    fn document_element(&self) -> Option<DomKey> {
        let dom = self.dom.lock().unwrap();
        let root = dom.root()?;
        self.find_tag(&dom, root, "html")
    }

    fn body(&self) -> Option<DomKey> {
        let dom = self.dom.lock().unwrap();
        let html = {
            let root = dom.root()?;
            self.find_tag(&dom, root, "html")?
        };
        self.find_tag(&dom, html, "body")
    }

    fn head(&self) -> Option<DomKey> {
        let dom = self.dom.lock().unwrap();
        let html = {
            let root = dom.root()?;
            self.find_tag(&dom, root, "html")?
        };
        self.find_tag(&dom, html, "head")
    }

    fn parent(&self, key: DomKey) -> Option<DomKey> {
        self.dom.lock().unwrap().get(key).and_then(|n| n.parent)
    }

    fn children(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.lock().unwrap().children(key).to_vec()
    }

    fn tag_name(&self, key: DomKey) -> Option<String> {
        self.dom
            .lock()
            .unwrap()
            .tag(key)
            .map(|t| t.to_string())
    }

    fn node_type(&self, key: DomKey) -> Option<u16> {
        self.dom.lock().unwrap().get(key).map(|n| match &n.kind {
            NodeKind::Document => 9,
            NodeKind::Element { .. } => 1,
            NodeKind::Text(_) => 3,
            NodeKind::Comment(_) => 8,
            NodeKind::Doctype { .. } => 10,
            NodeKind::ProcessingInstruction { .. } => 7,
        })
    }

    fn node_name(&self, key: DomKey) -> Option<String> {
        self.dom.lock().unwrap().get(key).map(|n| match &n.kind {
            NodeKind::Document => "#document".to_string(),
            NodeKind::Element { tag } => tag.to_ascii_uppercase(),
            NodeKind::Text(_) => "#text".to_string(),
            NodeKind::Comment(_) => "#comment".to_string(),
            NodeKind::Doctype { name, .. } => name.clone(),
            NodeKind::ProcessingInstruction { target, .. } => target.clone(),
        })
    }

    fn get_element_by_id(&self, id: &str) -> Option<DomKey> {
        let dom = self.dom.lock().unwrap();
        let root = dom.root()?;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if dom.attr(node, "id") == Some(id) {
                return Some(node);
            }
            stack.extend(dom.children(node).iter().rev());
        }
        None
    }

    fn text_content(&self, key: DomKey) -> Option<String> {
        let dom = self.dom.lock().unwrap();
        dom.get(key)?;
        let mut out = String::new();
        self.subtree_text(&dom, key, &mut out);
        Some(out)
    }

    fn set_text_content(&mut self, key: DomKey, text: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() {
            return false;
        }
        let old: Vec<NodeId> = dom.children(key).to_vec();
        for child in old {
            dom.remove_from_parent(child);
        }
        let text_node = dom.create_text(text);
        dom.append_child(key, text_node);
        dom.mark_dirty(key);
        self.render_requested.set(true);
        true
    }

    fn set_character_data(&mut self, key: DomKey, data: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if !matches!(
            dom.get(key).map(|n| &n.kind),
            Some(NodeKind::Text(_))
        ) {
            return false;
        }
        dom.set_text(key, data);
        dom.mark_dirty(key);
        self.render_requested.set(true);
        true
    }

    fn get_attribute(&self, key: DomKey, name: &str) -> Option<String> {
        self.dom
            .lock()
            .unwrap()
            .attr(key, name)
            .map(|s| s.to_string())
    }

    fn set_attribute(&mut self, key: DomKey, name: &str, value: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() {
            return false;
        }
        dom.set_attribute(key, name, value);
        dom.mark_dirty(key);
        self.render_requested.set(true);
        true
    }

    fn remove_attribute(&mut self, key: DomKey, name: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() {
            return false;
        }
        dom.remove_attribute(key, name);
        dom.mark_dirty(key);
        self.render_requested.set(true);
        true
    }

    fn has_attribute(&self, key: DomKey, name: &str) -> bool {
        self.dom.lock().unwrap().attr(key, name).is_some()
    }

    fn title(&self) -> Option<String> {
        let dom = self.dom.lock().unwrap();
        let head = {
            let root = dom.root()?;
            let html = self.find_tag(&dom, root, "html")?;
            self.find_tag(&dom, html, "head")?
        };
        let title = self.find_tag(&dom, head, "title")?;
        let mut out = String::new();
        self.subtree_text(&dom, title, &mut out);
        Some(out)
    }

    fn set_title(&mut self, title: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        let head = {
            let Some(root) = dom.root() else {
                return false;
            };
            let Some(html) = self.find_tag(&dom, root, "html") else {
                return false;
            };
            let Some(head) = self.find_tag(&dom, html, "head") else {
                return false;
            };
            head
        };
        let title_el = match self.find_tag(&dom, head, "title") {
            Some(t) => t,
            None => {
                let t = dom.create_element("title");
                dom.append_child(head, t);
                t
            }
        };
        let old: Vec<NodeId> = dom.children(title_el).to_vec();
        for child in old {
            dom.remove_from_parent(child);
        }
        let text_node = dom.create_text(title);
        dom.append_child(title_el, text_node);
        dom.mark_dirty(title_el);
        self.render_requested.set(true);
        true
    }

    fn create_element(&mut self, tag: &str) -> Option<DomKey> {
        Some(self.dom.lock().unwrap().create_element(tag))
    }

    fn create_text_node(&mut self, data: &str) -> Option<DomKey> {
        Some(self.dom.lock().unwrap().create_text(data))
    }

    fn append_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(parent).is_none() || dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if child == parent || self.is_ancestor_or_self(&dom, child, parent) {
            return Err(DomHostError::Hierarchy);
        }
        if matches!(
            dom.get(child).map(|n| &n.kind),
            Some(NodeKind::Document) | Some(NodeKind::Doctype { .. })
        ) {
            return Err(DomHostError::Hierarchy);
        }
        dom.append_child(parent, child);
        dom.mark_dirty(parent);
        self.render_requested.set(true);
        Ok(())
    }

    fn insert_before(
        &mut self,
        parent: DomKey,
        child: DomKey,
        sibling: Option<DomKey>,
    ) -> Result<(), DomHostError> {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(parent).is_none() || dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if child == parent || self.is_ancestor_or_self(&dom, child, parent) {
            return Err(DomHostError::Hierarchy);
        }
        match sibling {
            None => {
                dom.append_child(parent, child);
                dom.mark_dirty(parent);
                self.render_requested.set(true);
                Ok(())
            }
            Some(sib) => {
                if dom.get(sib).is_none() {
                    return Err(DomHostError::StaleKey);
                }
                if !dom.children(parent).contains(&sib) {
                    return Err(DomHostError::NotFound);
                }
                dom.insert_before(parent, sib, child);
                dom.mark_dirty(parent);
                self.render_requested.set(true);
                Ok(())
            }
        }
    }

    fn remove_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(parent).is_none() || dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if dom.get(child).and_then(|n| n.parent) != Some(parent) {
            return Err(DomHostError::NotFound);
        }
        dom.remove_child(parent, child);
        dom.mark_dirty(parent);
        self.render_requested.set(true);
        Ok(())
    }

    fn replace_child(
        &mut self,
        parent: DomKey,
        old: DomKey,
        new: DomKey,
    ) -> Result<(), DomHostError> {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(parent).is_none() || dom.get(old).is_none() || dom.get(new).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if dom.get(old).and_then(|n| n.parent) != Some(parent) {
            return Err(DomHostError::NotFound);
        }
        if new == parent || self.is_ancestor_or_self(&dom, new, parent) {
            return Err(DomHostError::Hierarchy);
        }
        dom.replace_child(parent, old, new);
        dom.mark_dirty(parent);
        self.render_requested.set(true);
        Ok(())
    }

    fn request_restlye_and_layout(&mut self, key: DomKey) {
        // The `Dom` mutators already set the Double Dirty Bits; this
        // call is the explicit render request (plus a belt-and-braces
        // `mark_dirty`, which also covers mutations made while the
        // parser's `suppress_records` privilege was on).
        self.dom.lock().unwrap().mark_dirty(key);
        self.render_requested.set(true);
    }

    fn query_selector(&self, scope: DomKey, selector: &str) -> Option<DomKey> {
        self.query_selector_all(scope, selector).into_iter().next()
    }

    fn query_selector_all(&self, scope: DomKey, selector: &str) -> Vec<DomKey> {
        let dom = self.dom.lock().unwrap();
        let selectors = Self::parse_selector_list(selector);
        let mut out = Vec::new();
        if selectors.is_empty() || dom.get(scope).is_none() {
            return out;
        }
        let state = ElementState::default();
        let mut stack = vec![scope];
        let mut order: Vec<DomKey> = Vec::new();
        while let Some(node) = stack.pop() {
            order.push(node);
            stack.extend(dom.children(node).iter().rev());
        }
        for node in order {
            if !matches!(
                dom.get(node).map(|n| &n.kind),
                Some(NodeKind::Element { .. })
            ) {
                continue;
            }
            if selectors
                .iter()
                .any(|s| selector_matches(&dom, node, s, &state))
            {
                out.push(node);
            }
        }
        out
    }

    fn inner_html(&self, key: DomKey) -> Option<String> {
        let dom = self.dom.lock().unwrap();
        dom.get(key)?;
        let mut out = String::new();
        for &child in dom.children(key) {
            self.serialise_node(&dom, child, &mut out);
        }
        Some(out)
    }

    fn set_inner_html(&mut self, key: DomKey, html: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        let ok = self.set_fragment(&mut dom, key, html, true);
        if ok {
            self.render_requested.set(true);
        }
        ok
    }

    fn document_write(&mut self, text: &str) -> bool {
        // While a parser is active, the markup goes through its
        // insertion point (the parser agent's `DocumentWriter` +
        // tokenizer splice) — never appended to the finished
        // document.
        if let Some(parser) = self.parser.clone() {
            let outcome = parser.borrow_mut().document_write(text);
            return match outcome {
                Ok(pluto_html::WriteOutcome::Inserted) => {
                    self.render_requested.set(true);
                    true
                }
                // No active parser / implicit-open: fall through to
                // the post-parse path below.
                Ok(pluto_html::WriteOutcome::ImplicitOpen) => self.write_after_parse(text),
                Ok(pluto_html::WriteOutcome::IgnoredAborted) => false,
                Err(_) => false,
            };
        }
        self.write_after_parse(text)
    }

    fn document_open(&mut self) -> bool {
        if let Some(parser) = self.parser.clone() {
            return match parser.borrow_mut().document_open() {
                Ok(pluto_html::OpenOutcome::Opened) => {
                    // The document is replaced: clear it; later
                    // writes take the post-parse path.
                    self.clear_document();
                    self.implicit_open_done = true;
                    self.render_requested.set(true);
                    true
                }
                Ok(pluto_html::OpenOutcome::IgnoredDuringParse) => false,
                Err(_) => false,
            };
        }
        self.clear_document();
        self.implicit_open_done = true;
        self.render_requested.set(true);
        true
    }

    fn document_close(&mut self) -> bool {
        if let Some(parser) = self.parser.clone() {
            return match parser.borrow_mut().document_close() {
                Ok(pluto_html::CloseOutcome::Closed) => {
                    // The script-created parser is done; later
                    // writes take the post-parse path.
                    self.parser = None;
                    true
                }
                Ok(_) => false,
                Err(_) => false,
            };
        }
        true
    }

    fn style_text(&self, key: DomKey) -> Option<String> {
        self.dom
            .lock()
            .unwrap()
            .attr(key, "style")
            .map(|s| s.to_string())
    }

    fn set_style_text(&mut self, key: DomKey, css: &str) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() {
            return false;
        }
        dom.set_attribute(key, "style", css);
        dom.mark_dirty(key);
        self.render_requested.set(true);
        true
    }

    fn geometry(&mut self, key: DomKey) -> Option<Geometry> {
        if !self.validate(key) {
            return None;
        }
        // Write path is lazy (dirty bits only); a geometry read is
        // the sanctioned forced flush. If anything is dirty, the
        // actor rebuilds from the live document first (full
        // re-layout in M4 — dirty-subtree layout is Stage 8; the
        // dirty bits are populated correctly so Stage 8 can trust
        // them). Otherwise the actor answers from its retained tree.
        let dirty = self.dom_is_dirty();
        let (tx, rx) = mpsc::channel();
        if dirty {
            self.forced_flushes += 1;
            // Stage 12 Rec 11: the host count routes into the
            // process-wide perf counters too, so the perf summary
            // prints the flush count (the field existed, but
            // nothing outside perf.rs ever wrote it). The clean
            // path below never flushes — it is answered from the
            // retained tree — so `flushes_found_nothing_dirty`
            // stays 0 by design: a flush that found nothing dirty
            // cannot happen through this host.
            if let Ok(mut counters) = perf::global().lock() {
                counters.forced_flushes += 1;
            }
            let document = self.current_document();
            if self
                .layout_tx
                .send(LayoutJob::FlushGeometry {
                    document: Box::new(document),
                    viewport: self.viewport,
                    state: self.element_state.clone(),
                    id: key,
                    reply: tx,
                })
                .is_err()
            {
                return None;
            }
        } else if self
            .layout_tx
            .send(LayoutJob::Geometry { id: key, reply: tx })
            .is_err()
        {
            return None;
        }
        // The dirty bits stay set: the flush only answers the
        // geometry question — the tab thread's frame tick still
        // re-lays out and submits a fresh display list, and clears
        // the bits then.
        rx.recv_timeout(GEOMETRY_TIMEOUT).ok().flatten()
    }

    fn forced_flush_count(&self) -> u64 {
        self.forced_flushes
    }

    fn report_error(&self, message: &str) {
        if let Ok(mut errors) = self.errors.lock() {
            errors.push(message.to_string());
        }
        eprintln!("[pluto] js error: {message}");
    }

    // Stage 12 Rec 7: the binding-batch host hooks. The defaults
    // ("not present") are a missing feature here, so the
    // production host overrides all five.
    fn document_url(&self) -> Option<String> {
        if self.base_url.is_empty() {
            None
        } else {
            Some(self.base_url.clone())
        }
    }

    fn request_navigation(&mut self, url: &str, replace: bool) {
        // Recorded, not performed: the realm never navigates
        // itself — the tab thread drains this after the JS run
        // and drives `start_navigation` (which also resets the
        // realm, so a navigation requested mid-script cannot
        // re-enter the running realm).
        self.pending_navigation = Some((url.to_string(), replace));
    }

    fn attribute_names(&self, key: DomKey) -> Vec<String> {
        self.dom
            .lock()
            .unwrap()
            .get(key)
            .map(|n| n.attrs.iter().map(|(name, _)| name.clone()).collect())
            .unwrap_or_default()
    }

    fn clone_node(&mut self, key: DomKey, deep: bool) -> Option<DomKey> {
        let mut dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() {
            return None;
        }
        // The clone is detached: no dirty bits, no render
        // request — nothing lays out until it is inserted.
        self.clone_within(&mut dom, key, deep)
    }

    fn request_image(&mut self, url: &str, base_url: &str) {
        // Recorded, not fetched: the tab thread drains this and
        // routes it through the Stage 11 store (same gates as
        // the preload scanner), which reports back via
        // `TabJs::notify_image_loaded` / `notify_image_failed`.
        self.pending_images.push(crate::images::ImageRequest {
            url: url.to_string(),
            base_url: base_url.to_string(),
            is_imageset: false,
            integrity: None,
        });
    }

    // M6 Phase 3 (Rec 11): computed style + WAAPI. The tab thread
    // owns the engine; this host is the queue + the read cache.
    fn computed_style(&self, key: DomKey, property: &str) -> Option<String> {
        self.computed_styles
            .get(&key)
            .and_then(|pairs| pairs.iter().find(|(p, _)| p == property))
            .map(|(_, v)| v.clone())
    }

    fn computed_styles(&self, key: DomKey) -> Vec<(String, String)> {
        self.computed_styles.get(&key).cloned().unwrap_or_default()
    }

    fn request_animate(&mut self, key: DomKey, spec: AnimateSpec) -> Option<u64> {
        let id = self.next_waapi_id;
        self.next_waapi_id += 1;
        self.pending_animate.push((key, id, spec));
        Some(id)
    }

    fn get_animations(&self, key: DomKey) -> Vec<AnimationInfo> {
        let mut out: Vec<AnimationInfo> = self
            .waapi_known
            .keys()
            .filter(|(k, _)| *k == key)
            .map(|(_, id)| AnimationInfo {
                id: Some(*id),
                name: None,
                finished: self.waapi_finished.contains(&(key, *id)),
            })
            .collect();
        out.sort_by_key(|info| info.id);
        if let Some(names) = self.css_animations.get(&key) {
            for name in names {
                out.push(AnimationInfo {
                    id: None,
                    name: Some(name.clone()),
                    finished: false,
                });
            }
        }
        out
    }

    fn waapi_control(&mut self, key: DomKey, anim_id: u64, op: WaapiOp) {
        self.pending_waapi_control.push((key, anim_id, op));
    }

    fn css_animation_control(&mut self, key: DomKey, name: &str, op: WaapiOp) {
        self.pending_css_control.push((key, name.to_string(), op));
    }

    fn clear_animations(&mut self) {
        self.clear_computed_style_cache();
        self.pending_animate.clear();
        self.pending_waapi_control.clear();
        self.pending_css_control.clear();
        self.waapi_known.clear();
        self.css_animations.clear();
    }

    // Stage 8 Rec 7: the mutation-observer registry lives in the
    // `Dom` (see `pluto-html::observers`); this host only locks
    // the arena and delegates.
    fn observe_mutation(
        &mut self,
        observer_id: u64,
        target: DomKey,
        options: MutationObserverOptions,
    ) -> bool {
        self.dom
            .lock()
            .unwrap()
            .observe_mutation(observer_id, target, options)
    }

    fn disconnect_mutation_observer(&mut self, observer_id: u64) {
        self.dom
            .lock()
            .unwrap()
            .disconnect_mutation_observer(observer_id);
    }

    fn take_mutation_records(&mut self, observer_id: u64) -> Vec<ObserverMutationRecord> {
        self.dom
            .lock()
            .unwrap()
            .take_mutation_records(observer_id)
    }

    fn pending_mutation_observers(&self) -> Vec<u64> {
        self.dom.lock().unwrap().pending_mutation_observers()
    }

    fn take_pending_mutation_observers(&mut self) -> Vec<u64> {
        self.dom
            .lock()
            .unwrap()
            .take_pending_mutation_observers()
    }

    // Stage 8 Rec 8: the host-path content size is the border box
    // (no padding data at this layer — the worker's rendering
    // step supplies real content sizes from the layout tree).
    fn content_size(&mut self, key: DomKey) -> Option<(f64, f64)> {
        self.geometry(key).map(|g| (g.width, g.height))
    }

    fn depth(&self, key: DomKey) -> Option<usize> {
        let dom = self.dom.lock().unwrap();
        dom.get(key)?;
        let mut depth = 0;
        let mut cur = dom.get(key).and_then(|n| n.parent);
        while let Some(parent) = cur {
            depth += 1;
            cur = dom.get(parent).and_then(|n| n.parent);
        }
        Some(depth)
    }
    // ---- Stage 13 (M5 forms): delegate to the live `Dom` ----
    fn is_form_control(&self, key: DomKey) -> bool {
        self.dom.lock().unwrap().is_form_control(key)
    }

    fn input_type(&self, key: DomKey) -> Option<String> {
        let dom = self.dom.lock().unwrap();
        dom.get(key).map(|_| dom.input_type(key))
    }

    fn is_disabled(&self, key: DomKey) -> bool {
        self.dom.lock().unwrap().is_control_disabled(key)
    }

    fn form_value(&self, key: DomKey) -> Option<String> {
        self.dom.lock().unwrap().form_value(key)
    }

    fn form_default_value(&self, key: DomKey) -> Option<String> {
        self.dom.lock().unwrap().form_default_value(key)
    }

    fn form_value_dirty(&self, key: DomKey) -> bool {
        self.dom.lock().unwrap().form_value_dirty(key)
    }

    fn set_form_value(&mut self, key: DomKey, value: &str) -> bool {
        self.dom.lock().unwrap().set_form_value(key, value)
    }

    fn form_checked(&self, key: DomKey) -> Option<bool> {
        self.dom.lock().unwrap().form_checked(key)
    }

    fn set_form_checked(&mut self, key: DomKey, checked: bool) -> bool {
        self.dom.lock().unwrap().set_form_checked(key, checked)
    }

    fn form_selected_index(&self, key: DomKey) -> Option<i64> {
        self.dom.lock().unwrap().form_selected_index(key)
    }

    fn set_form_selected_index(&mut self, key: DomKey, index: i64) -> bool {
        self.dom.lock().unwrap().set_form_selected_index(key, index)
    }

    fn form_options(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.lock().unwrap().select_options(key)
    }

    fn form_owner(&self, key: DomKey) -> Option<DomKey> {
        self.dom.lock().unwrap().form_owner(key)
    }

    fn form_controls(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.lock().unwrap().form_controls(key)
    }

    fn document_forms(&self) -> Vec<DomKey> {
        let dom = self.dom.lock().unwrap();
        let Some(root) = dom.root() else {
            return Vec::new();
        };
        fn walk(dom: &Dom, id: DomKey, out: &mut Vec<DomKey>) {
            if dom.tag(id) == Some("form") {
                out.push(id);
            }
            for &c in dom.children(id) {
                walk(dom, c, out);
            }
        }
        let mut out = Vec::new();
        walk(&dom, root, &mut out);
        out
    }

    fn form_entries(&self, key: DomKey, submitter: Option<DomKey>) -> Vec<(String, String)> {
        self.dom.lock().unwrap().form_entries(key, submitter)
    }

    fn reset_form(&mut self, key: DomKey) -> bool {
        let mut dom = self.dom.lock().unwrap();
        if dom.tag(key) != Some("form") {
            return false;
        }
        dom.reset_form(key);
        true
    }

    fn form_caret(&self, key: DomKey) -> Option<usize> {
        self.dom.lock().unwrap().form_caret(key)
    }

    fn form_selection(&self, key: DomKey) -> Option<(usize, usize)> {
        self.dom.lock().unwrap().form_selection(key)
    }

    fn set_form_selection(&mut self, key: DomKey, start: usize, end: usize) -> bool {
        self.dom.lock().unwrap().set_form_selection(key, start, end)
    }

    fn insert_form_text(&mut self, key: DomKey, text: &str) -> Option<usize> {
        self.dom.lock().unwrap().insert_form_text(key, text)
    }

    fn delete_form_text(&mut self, key: DomKey, backspace: bool) -> Option<usize> {
        self.dom.lock().unwrap().delete_form_text(key, backspace)
    }

    fn custom_validity(&self, key: DomKey) -> Option<String> {
        self.custom_validity.get(&key).cloned()
    }

    fn set_custom_validity(&mut self, key: DomKey, message: &str) -> bool {
        if self.dom.lock().unwrap().get(key).is_none() {
            return false;
        }
        if message.is_empty() {
            self.custom_validity.remove(&key);
        } else {
            self.custom_validity.insert(key, message.to_string());
        }
        true
    }

    fn active_element(&self) -> Option<DomKey> {
        let dom = self.dom.lock().unwrap();
        self.focused.filter(|&k| dom.get(k).is_some())
    }

    fn take_pending_focus(&mut self) -> Option<Option<DomKey>> {
        self.pending_focus.take()
    }

    fn focus_element(&mut self, key: DomKey) -> bool {
        let dom = self.dom.lock().unwrap();
        if dom.get(key).is_none() || dom.is_control_disabled(key) {
            return false;
        }
        let focusable = dom.is_form_control(key)
            || matches!(dom.tag(key), Some("button"))
            || (dom.tag(key) == Some("a") && dom.attr(key, "href").is_some())
            || dom.attr(key, "tabindex").is_some();
        drop(dom);
        if !focusable {
            return false;
        }
        self.focused = Some(key);
        self.pending_focus = Some(Some(key));
        true
    }

    fn blur_element(&mut self) {
        self.focused = None;
        self.pending_focus = Some(None);
    }

    fn document_cookie(&self) -> Option<String> {
        let jar = self.cookie_jar.as_ref()?;
        let origin = Origin::from_url(&self.base_url);
        if origin.is_opaque() {
            return None;
        }
        let site = pluto_js::bindings::storage::top_level_site_of(&origin);
        let pairs = jar.lock().unwrap().document_cookies(&self.base_url, &site);
        Some(
            pairs
                .iter()
                .map(|(n, v)| format!("{n}={v}"))
                .collect::<Vec<_>>()
                .join("; "),
        )
    }

    fn set_document_cookie(&mut self, cookie: &str) -> bool {
        let Some(jar) = self.cookie_jar.as_ref() else {
            return false;
        };
        let origin = Origin::from_url(&self.base_url);
        if origin.is_opaque() {
            return false;
        }
        let site = pluto_js::bindings::storage::top_level_site_of(&origin);
        jar.lock()
            .unwrap()
            .store_set_cookie(&self.base_url, &site, &[cookie.to_string()]);
        true
    }

    fn request_form_submit(&mut self, form: DomKey, submitter: Option<DomKey>) -> bool {
        if self.dom.lock().unwrap().tag(form) != Some("form") {
            return false;
        }
        self.pending_form_submits.push((form, submitter));
        true
    }

    fn take_pending_form_submits(&mut self) -> Vec<(DomKey, Option<DomKey>)> {
        std::mem::take(&mut self.pending_form_submits)
    }
}

impl BrowserDomHost {
    /// Post-parse `document.write`: the implicit-`open()` path.
    /// The first write clears the document; the markup is parsed
    /// as a fragment and appended to `<body>`. M4 departure: there
    /// is no streaming script-created parser behind this — a
    /// second write appends another fragment instead of streaming
    /// into one parse, and `<script>` elements written this way
    /// are inert (they are never executed).
    fn write_after_parse(&mut self, text: &str) -> bool {
        if !self.implicit_open_done {
            self.clear_document();
            self.implicit_open_done = true;
        }
        let body = self.body();
        let Some(body) = body else {
            return false;
        };
        let mut dom = self.dom.lock().unwrap();
        let ok = self.set_fragment(&mut dom, body, text, false);
        if ok {
            self.render_requested.set(true);
        }
        ok
    }

    /// Remove every child of `<html>` (the `document.open()`
    /// replacement step). The document node itself survives, so
    /// existing `document` wrappers stay valid; element wrappers
    /// for removed nodes go stale, per the generational keys.
    fn clear_document(&mut self) {
        let mut dom = self.dom.lock().unwrap();
        let Some(html) = dom.root().and_then(|root| self.find_tag(&dom, root, "html")) else {
            return;
        };
        let kids: Vec<NodeId> = dom.children(html).to_vec();
        for kid in kids {
            dom.remove_from_parent(kid);
        }
        dom.mark_dirty(html);
    }
}

/// A shareable handle to one [`BrowserDomHost`]: the tab thread
/// keeps the `Rc` (to update viewport / sheet / parser between
/// navigations) while [`TabJs`] holds this wrapper as its
/// `Box<dyn DomHost>`. Both sides see the same host.
#[derive(Clone)]
pub struct SharedHost(pub Rc<std::sync::Mutex<BrowserDomHost>>);

impl DomHost for SharedHost {
    fn validate(&self, key: DomKey) -> bool {
        self.0.lock().unwrap().validate(key)
    }
    fn document_element(&self) -> Option<DomKey> {
        self.0.lock().unwrap().document_element()
    }
    fn body(&self) -> Option<DomKey> {
        self.0.lock().unwrap().body()
    }
    fn get_element_by_id(&self, id: &str) -> Option<DomKey> {
        self.0.lock().unwrap().get_element_by_id(id)
    }
    fn text_content(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().text_content(key)
    }
    fn set_text_content(&mut self, key: DomKey, text: &str) -> bool {
        self.0.lock().unwrap().set_text_content(key, text)
    }
    fn set_character_data(&mut self, key: DomKey, data: &str) -> bool {
        self.0.lock().unwrap().set_character_data(key, data)
    }
    fn get_attribute(&self, key: DomKey, name: &str) -> Option<String> {
        self.0.lock().unwrap().get_attribute(key, name)
    }
    fn set_attribute(&mut self, key: DomKey, name: &str, value: &str) -> bool {
        self.0.lock().unwrap().set_attribute(key, name, value)
    }
    fn request_restlye_and_layout(&mut self, key: DomKey) {
        self.0.lock().unwrap().request_restlye_and_layout(key)
    }
    fn geometry(&mut self, key: DomKey) -> Option<Geometry> {
        self.0.lock().unwrap().geometry(key)
    }
    fn forced_flush_count(&self) -> u64 {
        self.0.lock().unwrap().forced_flush_count()
    }
    fn document_node(&self) -> Option<DomKey> {
        self.0.lock().unwrap().document_node()
    }
    fn head(&self) -> Option<DomKey> {
        self.0.lock().unwrap().head()
    }
    fn parent(&self, key: DomKey) -> Option<DomKey> {
        self.0.lock().unwrap().parent(key)
    }
    fn children(&self, key: DomKey) -> Vec<DomKey> {
        self.0.lock().unwrap().children(key)
    }
    fn tag_name(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().tag_name(key)
    }
    fn node_type(&self, key: DomKey) -> Option<u16> {
        self.0.lock().unwrap().node_type(key)
    }
    fn node_name(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().node_name(key)
    }
    fn title(&self) -> Option<String> {
        self.0.lock().unwrap().title()
    }
    fn set_title(&mut self, title: &str) -> bool {
        self.0.lock().unwrap().set_title(title)
    }
    fn create_element(&mut self, tag: &str) -> Option<DomKey> {
        self.0.lock().unwrap().create_element(tag)
    }
    fn create_text_node(&mut self, data: &str) -> Option<DomKey> {
        self.0.lock().unwrap().create_text_node(data)
    }
    fn append_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        self.0.lock().unwrap().append_child(parent, child)
    }
    fn insert_before(
        &mut self,
        parent: DomKey,
        child: DomKey,
        sibling: Option<DomKey>,
    ) -> Result<(), DomHostError> {
        self.0
            .lock()
            .unwrap()
            .insert_before(parent, child, sibling)
    }
    fn remove_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        self.0.lock().unwrap().remove_child(parent, child)
    }
    fn replace_child(
        &mut self,
        parent: DomKey,
        old: DomKey,
        new: DomKey,
    ) -> Result<(), DomHostError> {
        self.0.lock().unwrap().replace_child(parent, old, new)
    }
    fn remove_attribute(&mut self, key: DomKey, name: &str) -> bool {
        self.0.lock().unwrap().remove_attribute(key, name)
    }
    fn has_attribute(&self, key: DomKey, name: &str) -> bool {
        self.0.lock().unwrap().has_attribute(key, name)
    }
    fn query_selector(&self, scope: DomKey, selector: &str) -> Option<DomKey> {
        self.0.lock().unwrap().query_selector(scope, selector)
    }
    fn query_selector_all(&self, scope: DomKey, selector: &str) -> Vec<DomKey> {
        self.0.lock().unwrap().query_selector_all(scope, selector)
    }
    fn inner_html(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().inner_html(key)
    }
    fn set_inner_html(&mut self, key: DomKey, html: &str) -> bool {
        self.0.lock().unwrap().set_inner_html(key, html)
    }
    fn document_write(&mut self, text: &str) -> bool {
        self.0.lock().unwrap().document_write(text)
    }
    fn document_open(&mut self) -> bool {
        self.0.lock().unwrap().document_open()
    }
    fn document_close(&mut self) -> bool {
        self.0.lock().unwrap().document_close()
    }
    fn style_text(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().style_text(key)
    }
    fn set_style_text(&mut self, key: DomKey, css: &str) -> bool {
        self.0.lock().unwrap().set_style_text(key, css)
    }
    fn report_error(&self, message: &str) {
        self.0.lock().unwrap().report_error(message)
    }
    fn document_url(&self) -> Option<String> {
        self.0.lock().unwrap().document_url()
    }
    fn request_navigation(&mut self, url: &str, replace: bool) {
        self.0.lock().unwrap().request_navigation(url, replace)
    }
    fn attribute_names(&self, key: DomKey) -> Vec<String> {
        self.0.lock().unwrap().attribute_names(key)
    }
    fn clone_node(&mut self, key: DomKey, deep: bool) -> Option<DomKey> {
        self.0.lock().unwrap().clone_node(key, deep)
    }
    fn request_image(&mut self, url: &str, base_url: &str) {
        self.0.lock().unwrap().request_image(url, base_url)
    }
    fn observe_mutation(
        &mut self,
        observer_id: u64,
        target: DomKey,
        options: MutationObserverOptions,
    ) -> bool {
        self.0
            .lock()
            .unwrap()
            .observe_mutation(observer_id, target, options)
    }
    fn disconnect_mutation_observer(&mut self, observer_id: u64) {
        self.0
            .lock()
            .unwrap()
            .disconnect_mutation_observer(observer_id);
    }
    fn take_mutation_records(&mut self, observer_id: u64) -> Vec<ObserverMutationRecord> {
        self.0.lock().unwrap().take_mutation_records(observer_id)
    }
    fn pending_mutation_observers(&self) -> Vec<u64> {
        self.0.lock().unwrap().pending_mutation_observers()
    }
    fn take_pending_mutation_observers(&mut self) -> Vec<u64> {
        self.0
            .lock()
            .unwrap()
            .take_pending_mutation_observers()
    }
    fn content_size(&mut self, key: DomKey) -> Option<(f64, f64)> {
        self.0.lock().unwrap().content_size(key)
    }
    fn depth(&self, key: DomKey) -> Option<usize> {
        self.0.lock().unwrap().depth(key)
    }
    // ---- Stage 13 (M5 forms): delegate to the inner host ----
    fn is_form_control(&self, key: DomKey) -> bool {
        self.0.lock().unwrap().is_form_control(key)
    }
    fn input_type(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().input_type(key)
    }
    fn is_disabled(&self, key: DomKey) -> bool {
        self.0.lock().unwrap().is_disabled(key)
    }
    fn form_value(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().form_value(key)
    }
    fn form_default_value(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().form_default_value(key)
    }
    fn form_value_dirty(&self, key: DomKey) -> bool {
        self.0.lock().unwrap().form_value_dirty(key)
    }
    fn set_form_value(&mut self, key: DomKey, value: &str) -> bool {
        self.0.lock().unwrap().set_form_value(key, value)
    }
    fn form_checked(&self, key: DomKey) -> Option<bool> {
        self.0.lock().unwrap().form_checked(key)
    }
    fn set_form_checked(&mut self, key: DomKey, checked: bool) -> bool {
        self.0.lock().unwrap().set_form_checked(key, checked)
    }
    fn form_selected_index(&self, key: DomKey) -> Option<i64> {
        self.0.lock().unwrap().form_selected_index(key)
    }
    fn set_form_selected_index(&mut self, key: DomKey, index: i64) -> bool {
        self.0.lock().unwrap().set_form_selected_index(key, index)
    }
    fn form_options(&self, key: DomKey) -> Vec<DomKey> {
        self.0.lock().unwrap().form_options(key)
    }
    fn form_owner(&self, key: DomKey) -> Option<DomKey> {
        self.0.lock().unwrap().form_owner(key)
    }
    fn form_controls(&self, key: DomKey) -> Vec<DomKey> {
        self.0.lock().unwrap().form_controls(key)
    }
    fn document_forms(&self) -> Vec<DomKey> {
        self.0.lock().unwrap().document_forms()
    }
    fn form_entries(&self, key: DomKey, submitter: Option<DomKey>) -> Vec<(String, String)> {
        self.0.lock().unwrap().form_entries(key, submitter)
    }
    fn reset_form(&mut self, key: DomKey) -> bool {
        self.0.lock().unwrap().reset_form(key)
    }
    fn form_caret(&self, key: DomKey) -> Option<usize> {
        self.0.lock().unwrap().form_caret(key)
    }
    fn form_selection(&self, key: DomKey) -> Option<(usize, usize)> {
        self.0.lock().unwrap().form_selection(key)
    }
    fn set_form_selection(&mut self, key: DomKey, start: usize, end: usize) -> bool {
        self.0.lock().unwrap().set_form_selection(key, start, end)
    }
    fn insert_form_text(&mut self, key: DomKey, text: &str) -> Option<usize> {
        self.0.lock().unwrap().insert_form_text(key, text)
    }
    fn delete_form_text(&mut self, key: DomKey, backspace: bool) -> Option<usize> {
        self.0.lock().unwrap().delete_form_text(key, backspace)
    }
    fn custom_validity(&self, key: DomKey) -> Option<String> {
        self.0.lock().unwrap().custom_validity(key)
    }
    fn set_custom_validity(&mut self, key: DomKey, message: &str) -> bool {
        self.0.lock().unwrap().set_custom_validity(key, message)
    }
    fn active_element(&self) -> Option<DomKey> {
        self.0.lock().unwrap().active_element()
    }
    fn take_pending_focus(&mut self) -> Option<Option<DomKey>> {
        self.0.lock().unwrap().take_pending_focus()
    }
    fn focus_element(&mut self, key: DomKey) -> bool {
        self.0.lock().unwrap().focus_element(key)
    }
    fn blur_element(&mut self) {
        self.0.lock().unwrap().blur_element()
    }
    fn document_cookie(&self) -> Option<String> {
        self.0.lock().unwrap().document_cookie()
    }
    fn set_document_cookie(&mut self, cookie: &str) -> bool {
        self.0.lock().unwrap().set_document_cookie(cookie)
    }
    fn request_form_submit(&mut self, form: DomKey, submitter: Option<DomKey>) -> bool {
        self.0.lock().unwrap().request_form_submit(form, submitter)
    }
    fn take_pending_form_submits(&mut self) -> Vec<(DomKey, Option<DomKey>)> {
        self.0.lock().unwrap().take_pending_form_submits()
    }
}

/// Create a fresh [`TabJs`] for the tab thread (one realm per tab,
/// fresh per navigation via `TabJs::navigate`).
pub fn new_tab_js() -> Result<TabJs, JsError> {
    TabJs::new(&RealmConfig::default())
}

// ------------------------------------------------------------------ storage
// + permissions backends (Stage 9 Rec 4 + Rec 11)
//
// The browser-side stores live in `storage.rs` / `permissions.rs`
// next to this file. They are declared here (rather than in
// `main.rs`) so this wave touches no file outside its scope:
// `crate::js_host::storage` / `crate::js_host::permissions`.
// A later tidy-up may move the two `mod` lines into `main.rs` and
// drop the `#[path]` attributes; the module paths are the only
// thing that would change.
//
// `localStorage` is shared across tabs and persisted to the
// profile; `sessionStorage` is per tab (a fresh store per
// [`TabBackends`], dropped — hence cleared — when the tab
// closes); permissions are shared and persisted, with grants
// expiring after ~90 days without a visit.

#[path = "storage.rs"]
pub mod storage;

#[path = "permissions.rs"]
pub mod permissions;

/// The per-tab storage + permissions backends: the shared,
/// profile-backed local store and permission store, plus this
/// tab's own session store. Built once per tab (in
/// `renderer.rs`, next to the `PlutoFetchBackend`); the tab
/// thread owns it for the tab's lifetime.
pub struct TabBackends {
    /// Shared `localStorage` + this tab's `sessionStorage`.
    pub storage: Arc<storage::PlutoStorageBackend>,
    /// Shared permission store.
    pub permissions: Arc<permissions::PlutoPermissionsBackend>,
}

/// Build the backends for a new tab over the process-wide
/// shared `localStorage` store (Stage 13 Rec 10a): every tab
/// gets a clone of the same `Arc<Mutex<LocalStorageStore>>`,
/// so interleaved `setItem`s from two tabs both survive.
/// `sessionStorage` stays per-tab (fresh store per tab).
pub fn new_tab_backends() -> TabBackends {
    new_tab_backends_with_shared(storage::shared_local_storage())
}

/// Build the backends for a new tab over an explicit shared
/// local store (tests and the future broker inject theirs).
pub fn new_tab_backends_with_shared(
    local: Arc<Mutex<storage::LocalStorageStore>>,
) -> TabBackends {
    let perms = Arc::new(Mutex::new(permissions::load_permissions()));
    TabBackends {
        storage: Arc::new(storage::PlutoStorageBackend::new(local)),
        permissions: Arc::new(permissions::PlutoPermissionsBackend::new(perms)),
    }
}

/// Persist the shared stores to the profile now (called after a
/// permission change lands through the backend, and — once the
/// wiring below lands — when a tab closes; `localStorage`
/// writes already save themselves on every mutation).
pub fn save_tab_backends(backends: &TabBackends) {
    backends.storage.save_local();
    backends.permissions.save();
}

// WIRING LANDED (Stage 13 — the follow-up this comment
// described is done): `TabJs::set_storage_backend` /
// `set_permissions_backend` exist in `pluto-js/src/tab.rs` and
// `TabState::tab_js` (`renderer.rs`) attaches the backends on
// realm creation; `TabJs::navigate` re-attaches them to the
// fresh realm on every navigation, so `localStorage` is shared
// across tabs and on disk, and `sessionStorage` survives
// reloads. Tab close flushes via `save_tab_backends` (which
// calls `PlutoStorageBackend::save_local`, a flush that always
// writes when dirty).

// ------------------------------------------------------------------ fetch
//
// `fetch()` over `pluto-net` (Recommendation 8): the binding builds
// the request, this backend runs it, and the loop resolves the
// promise — never the backend, never inline on a network thread.

/// The `fetch()` backend for one tab: a [`ResourceLoader`] plus the
/// document origin / epoch the CORS gate checks against.
///
/// # CORS gate (Stage 9 Rec 2, replacing the M4 same-origin gate)
///
/// Relative URLs resolve against the document base; the gate
/// compares parsed [`Origin`]s structurally (default-port
/// normalized, so `https://example.com:443` is same origin as
/// `https://example.com`), never serialized strings — two different
/// opaque origins both serialize to `"null"` but are never same
/// origin. An absolute cross-origin URL is NO LONGER rejected here:
/// it goes to `pluto-net` as a CORS fetch (`RequestMode::Cors`),
/// where the preflight / CORS check / opaque filtering run — the
/// real check lives browser-side in `pluto-net`, never in the realm
/// (`check_fetch_allowed` in `pluto-js` stays a pass-through; this
/// backend's gate stays authoritative).
///
/// `init.mode` / `init.credentials` are parsed in `pluto-js`
/// and ride `FetchRequest` (Stage 9 Rec 2): the backend enforces
/// `same-origin` mode here and threads `mode` / `credentials` /
/// `method` / `headers` to `pluto-net`, which runs the preflight
/// / CORS check / opaque filtering and the credentials rules.
/// `CredentialsMode::Include` is fully honoured by `pluto-net`
/// for any request that carries it.
///
/// Gate order at this hook point: scheme check → mixed-content
/// check → CORS (in `pluto-net`) → network. CSP `connect-src`
/// (M10) slots in BEFORE the CORS decision when it lands; the
/// no-op hook for it is `csp_pre_request_check` in `pluto-net`.
pub struct PlutoFetchBackend {
    /// Where `fetch()` goes (Stage 13 Rec 11): the direct
    /// loader in-process, the browser-side broker out-of-process.
    fetch: crate::broker::FetchSource,
    /// The tab thread's tokio runtime handle: `start_fetch` spawns
    /// the network task on it (the tab thread drives the runtime
    /// on its frame tick).
    handle: tokio::runtime::Handle,
    /// Document base URL, updated by the tab thread on navigation.
    base_url: Arc<Mutex<String>>,
    /// Document origin, from `base_url` via
    /// [`Origin::for_document`] (top-level navigation, no creator):
    /// a top-level `about:blank` or `data:` document is opaque, a
    /// normal URL is its tuple origin.
    origin: Arc<Mutex<Origin>>,
    /// Navigation epoch; a completion for a stale epoch resolves
    /// as `Cancelled` (the loop drops it anyway — belt and braces).
    epoch: Arc<AtomicU64>,
    next_request_id: AtomicU64,
    /// The shell-assigned tab id (Stage 11 Rec 4).
    tab_id: u64,
}

impl PlutoFetchBackend {
    pub fn new(
        fetch: crate::broker::FetchSource,
        handle: tokio::runtime::Handle,
        base_url: &str,
        tab_id: u64,
    ) -> Self {
        Self {
            fetch,
            handle,
            base_url: Arc::new(Mutex::new(base_url.to_string())),
            origin: Arc::new(Mutex::new(Origin::from_url(base_url))),
            epoch: Arc::new(AtomicU64::new(0)),
            next_request_id: AtomicU64::new(1),
            // Stage 11 Rec 4: the shell-assigned tab id, echoed
            // in every `fetch()` request instead of the old
            // hardcoded 0.
            tab_id,
        }
    }

    /// Re-point the gate at a new document (top-level navigation):
    /// the origin is the document origin of `url` with no creator
    /// (see [`Origin::for_document`]).
    pub fn set_base_url(&self, url: &str) {
        *self.base_url.lock().unwrap() = url.to_string();
        *self.origin.lock().unwrap() = Origin::for_document(url, None, false);
    }

    /// Re-point the gate at a new document created by another
    /// document (`creator`): an `about:blank` iframe / popup or a
    /// `srcdoc` document inherits the creator's origin, unless
    /// `sandboxed` (a sandboxed iframe always gets a fresh opaque
    /// origin). No caller yet — iframes render no nested browsing
    /// context — but the seam is here so the iframe work does not
    /// have to re-derive the rule.
    pub fn set_base_url_with_creator(
        &self,
        url: &str,
        creator: Option<&Origin>,
        sandboxed: bool,
    ) {
        *self.base_url.lock().unwrap() = url.to_string();
        *self.origin.lock().unwrap() = Origin::for_document(url, creator, sandboxed);
    }

    /// The document origin the gate checks against.
    pub fn document_origin(&self) -> Origin {
        self.origin.lock().unwrap().clone()
    }

    /// Bump the epoch (navigation): in-flight fetches for the old
    /// document complete as `Cancelled`.
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch.store(epoch, Ordering::SeqCst);
    }
}

impl FetchBackend for PlutoFetchBackend {
    /// The document origin, for the realm-side gate's early
    /// same-origin check (`check_fetch_allowed` in `pluto-js`).
    /// That check is defense in depth only — this backend's own
    /// gate above stays authoritative.
    fn document_origin(&self) -> Option<Origin> {
        Some(self.origin.lock().unwrap().clone())
    }

    fn start_fetch(&self, request: FetchRequest, done: mpsc::Sender<FetchCompletion>) {
        let id = request.id;
        let send = |result: Result<FetchResponse, FetchError>| {
            let _ = done.send(FetchCompletion { id, result });
        };
        let base = self.base_url.lock().unwrap().clone();
        let absolute = resolve_url(&base, &request.url);
        if !absolute.starts_with("http://") && !absolute.starts_with("https://") {
            send(Err(FetchError::InvalidUrl(format!(
                "fetch: unsupported URL scheme in {absolute}"
            ))));
            return;
        }
        // Mixed-content gate (Stage 5 also gates in the scheduler;
        // this is the fetch-API gate): an https page may not fetch
        // http.
        if base.starts_with("https://") && absolute.starts_with("http://") {
            send(Err(FetchError::Network(format!(
                "mixed content blocked: {absolute} on an https page"
            ))));
            return;
        }
        // Same-origin gate (M4) → CORS gate (Stage 9 Rec 2): an
        // absolute cross-origin URL is NOT rejected here anymore.
        // It goes to `pluto-net` as a CORS fetch, where the
        // preflight / CORS check / opaque filtering run. The
        // comparison below still uses parsed `Origin`s
        // (default-port normalized), never serialized strings —
        // and an opaque document origin can never pass a CORS
        // check, so an opaque document fetching cross-origin
        // fails fast here, before any network access (fail
        // closed: the CORS code cannot classify an opaque
        // origin's cross-origin fetch as anything but an error).
        // TODO(M10): CSP `connect-src` check goes here, before the
        // network, once CSP lands — it slots in BEFORE the CORS
        // decision (see the struct docs).
        let doc_origin = self.origin.lock().unwrap().clone();
        if doc_origin.is_opaque()
            && !Origin::from_url(&absolute).is_same_origin(&doc_origin)
        {
            send(Err(FetchError::InvalidUrl(format!(
                "cross-origin fetch from an opaque origin is blocked: {absolute}"
            ))));
            return;
        }
        // `mode: "same-origin"` is enforced here, before any
        // network access: a cross-origin URL with that mode is a
        // TypeError, not a CORS request (Fetch §4.2). `mode:
        // "no-cors"` and `"cors"` go to `pluto-net`, which runs the
        // preflight / CORS check / opaque filtering.
        if request.mode == RequestMode::SameOrigin
            && !Origin::from_url(&absolute).is_same_origin(&doc_origin)
        {
            send(Err(FetchError::InvalidUrl(format!(
                "same-origin fetch to a cross-origin URL is blocked: {absolute}"
            ))));
            return;
        }
        let resource = ResourceRequest {
            request_id: self.next_request_id.fetch_add(1, Ordering::Relaxed),
            tab_id: self.tab_id,
            epoch: self.epoch.load(Ordering::SeqCst),
            url: absolute,
            destination: Destination::Fetch,
            priority: Priority::Medium,
            // Stage 9 Rec 2: `init.method` / `init.headers` /
            // `init.mode` / `init.credentials` ride the wire —
            // `pluto-net` validates the method/headers (CORS
            // safelists, forbidden headers, preflight) and applies
            // the mode/credentials rules. Nothing here decides
            // CORS; this backend only enforces the same-origin
            // mode above and the opaque-origin fail-closed rule.
            method: request.method.clone(),
            headers: request.headers.clone(),
            // `fetch()` is never an imageset fetch.
            is_imageset: false,
            mode: request.mode,
            credentials: request.credentials,
            origin: doc_origin.serialize(),
            top_level_site: doc_origin.serialize(),
            upgrade_insecure_requests: false,
            integrity: None,
            body: None,
        };
        let fetch = self.fetch.clone();
        let epoch = Arc::clone(&self.epoch);
        let start_epoch = epoch.load(Ordering::SeqCst);
        // `Response.type` (Stage 9 Rec 2): the backend computes it
        // from the final URL's origin against the document origin —
        // `pluto-net` zeroed the response (status 0, no headers, no
        // body) for opaque tainting, so status 0 is the opaque
        // signal; anything else is basic (same-origin) or cors.
        let doc_origin_for_type = doc_origin.clone();
        self.handle.spawn(async move {
            let outcome = match fetch
                .fetch_with(resource, CancellationToken::new())
                .await
            {
                Ok(response) => {
                    let response_type = if response.status.as_u16() == 0 {
                        ResponseType::Opaque
                    } else if Origin::from_url(&response.final_url.to_string())
                        .is_same_origin(&doc_origin_for_type)
                    {
                        ResponseType::Basic
                    } else {
                        ResponseType::Cors
                    };
                    Ok(FetchResponse {
                        status: response.status.as_u16(),
                        status_text: String::new(),
                        headers: response
                            .headers
                            .iter()
                            .map(|(name, value)| {
                                (
                                    name.to_string(),
                                    value.to_str().unwrap_or("").to_string(),
                                )
                            })
                            .collect(),
                        body: response.body,
                        response_type,
                    })
                }
                Err(e) => Err(FetchError::Network(e.to_string())),
            };
            // Navigation epoch: a completion for the old document
            // resolves as Cancelled, never as a stale success.
            let outcome = if epoch.load(Ordering::SeqCst) != start_epoch {
                Err(FetchError::Cancelled)
            } else {
                outcome
            };
            let _ = done.send(FetchCompletion { id, result: outcome });
        });
    }
}

// ------------------------------------------------------------------ runner
//
// The parser-seam driver (Recommendation 6's integration half):
// `pluto-html` pauses, queues, and writes; this runner executes.

/// How long the load path drives the loop after parsing (defer
/// scripts, `DOMContentLoaded`, timers / fetches the scripts
/// started). Bounded on purpose: the tab thread keeps driving the
/// loop on its frame tick afterwards, so a slow fetch or a
/// repeating timer is not lost — it just settles after commit.
const POST_PARSE_DRIVE_BUDGET: Duration = Duration::from_secs(1);

/// A script-execution request from a load task to the tab thread's
/// main loop. Loads run as `spawn_local` tasks on the tab thread's
/// current-thread runtime (so a second navigation can cancel the
/// first mid-flight — the tab thread's command loop stays live), but
/// a spawned task cannot borrow the tab's `TabJs` (`spawn_local`
/// futures must be `'static`), so execution is delegated back here:
/// the load task sends the request and awaits the one-shot reply,
/// and the main loop runs the script on the `TabJs` it owns. The
/// `DocumentWriter` nesting guard is raised by the LOAD task around
/// the round trip (see `LoadHooks::paused_script`), because the
/// parser lives in the load task, not the main loop.
pub enum ExecMsg {
    /// A parser-blocking classic script: execute now, at the pause.
    Classic {
        /// Navigation epoch of the load that sent this: the main
        /// loop drops requests whose epoch is not current (a load
        /// cancelled after sending must not execute in the new
        /// document's realm).
        epoch: u64,
        source: String,
        url: String,
        reply: oneshot::Sender<()>,
    },
    /// One queued script (defer / async), in queue order.
    Queued {
        epoch: u64,
        node: NodeId,
        src: Option<String>,
        bytes: Vec<u8>,
        reply: oneshot::Sender<()>,
    },
    /// The parse finished: drain the defer queue, fire
    /// `DOMContentLoaded`, drive the loop briefly, and return the
    /// document re-synced from the host's live `Dom`.
    ParseFinished {
        epoch: u64,
        // Boxed: a `Document` is far larger than the other
        // variants' payloads (clippy `large_enum_variant`).
        document: Box<Document>,
        reply: oneshot::Sender<Document>,
    },
}

/// The load-task side of script execution: implements
/// [`ScriptHooks`] for one navigation's load task. It never touches
/// the `TabJs` itself — every execution is an [`ExecMsg`] round trip
/// to the tab thread's main loop. It DOES own the parser-facing
/// duties: the `DocumentWriter` nesting guard around each execution
/// (the parser lives in this task), and re-pointing the host at the
/// parser's shared `Dom` on creation / encoding restart.
pub struct LoadHooks {
    /// Navigation epoch of this load (stamped on every request so
    /// the main loop can drop stale ones).
    epoch: u64,
    exec_tx: UnboundedSender<ExecMsg>,
    host: Rc<std::sync::Mutex<BrowserDomHost>>,
}

impl LoadHooks {
    pub fn new(
        epoch: u64,
        exec_tx: UnboundedSender<ExecMsg>,
        host: Rc<std::sync::Mutex<BrowserDomHost>>,
    ) -> Self {
        Self {
            epoch,
            exec_tx,
            host,
        }
    }

    /// Send one execution request and await its reply. A dropped
    /// main loop (tab closing) closes the reply channel, so this
    /// never hangs — it just proceeds without the execution.
    /// (`ParseFinished` returns a `Document`, so it builds its own
    /// reply channel in `parse_finished` instead of using this.)
    async fn round_trip(&self, make: impl FnOnce(oneshot::Sender<()>) -> ExecMsg) {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.exec_tx.send(make(reply_tx)).is_err() {
            return;
        }
        let _ = reply_rx.await;
    }
}

impl ScriptHooks for LoadHooks {
    fn paused_script<'a>(
        &'a mut self,
        parser: &'a Rc<RefCell<Parser>>,
        paused: &'a ScriptPaused,
        bytes: &'a [u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move { self.paused_script_inner(parser, paused, bytes).await })
    }

    fn async_ready<'a>(
        &'a mut self,
        parser: &'a Rc<RefCell<Parser>>,
        ready: Vec<QueuedScript>,
        bytes_by_src: &'a std::collections::HashMap<String, Vec<u8>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move { self.async_ready_inner(parser, ready, bytes_by_src).await })
    }

    fn parse_finished<'a>(
        &'a mut self,
        document: Document,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Document> + 'a>> {
        Box::pin(async move { self.parse_finished_inner(document).await })
    }

    fn parser_created(&mut self, parser: &Rc<RefCell<Parser>>) {
        // The load built the parser for this navigation: the host
        // follows its shared `Dom` from the first token.
        self.host
            .lock()
            .unwrap()
            .set_dom(parser.borrow().shared_dom());
        self.host.lock().unwrap().set_parser(Some(Rc::clone(parser)));
    }

    fn parser_restarted(&mut self, parser: &Rc<RefCell<Parser>>) {
        // An encoding restart swaps the parser — and its arena —
        // for a fresh one: the host must follow it.
        self.host
            .lock()
            .unwrap()
            .set_dom(parser.borrow().shared_dom());
        self.host.lock().unwrap().set_parser(Some(Rc::clone(parser)));
    }

}

impl LoadHooks {
    async fn paused_script_inner(
        &mut self,
        parser: &Rc<RefCell<Parser>>,
        paused: &ScriptPaused,
        bytes: &[u8],
    ) {
        // Parser-blocking scripts are always classic (the parser
        // never pauses for defer / async / module scripts). The
        // `DocumentWriter` nesting level is raised around the
        // execution, so a `document.open()` from this script is
        // ignored per spec (and a `document.write` lands at the
        // insertion point).
        let url = paused
            .src
            .clone()
            .unwrap_or_else(|| "inline.js".to_string());
        let source = String::from_utf8_lossy(bytes).into_owned();
        parser.borrow_mut().document_writer().enter_script();
        let epoch = self.epoch;
        self.round_trip(|reply| ExecMsg::Classic {
            epoch,
            source,
            url,
            reply,
        })
        .await;
        parser.borrow_mut().document_writer().exit_script();
    }

    async fn async_ready_inner(
        &mut self,
        parser: &Rc<RefCell<Parser>>,
        ready: Vec<QueuedScript>,
        bytes_by_src: &std::collections::HashMap<String, Vec<u8>>,
    ) {
        // Arrival order only — the queue already ordered them. Each
        // async script also runs under the nesting guard (it
        // executes during the parse, so `document.write` from it
        // must land at the insertion point too).
        for script in ready {
            let bytes = script
                .src
                .as_ref()
                .and_then(|src| bytes_by_src.get(src))
                .cloned()
                .unwrap_or_default();
            parser.borrow_mut().document_writer().enter_script();
            let epoch = self.epoch;
            self.round_trip(|reply| ExecMsg::Queued {
                epoch,
                node: script.script,
                src: script.src,
                bytes,
                reply,
            })
            .await;
            parser.borrow_mut().document_writer().exit_script();
        }
    }

    async fn parse_finished_inner(&mut self, document: Document) -> Document {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .exec_tx
            .send(ExecMsg::ParseFinished {
                epoch: self.epoch,
                document: Box::new(document),
                reply: reply_tx,
            })
            .is_err()
        {
            // The main loop is gone; the caller cannot use the
            // document anyway. Return an empty one rather than
            // hanging on the reply.
            return Document::from_parts(Dom::new(), Stylesheet::default(), "");
        }
        match reply_rx.await {
            Ok(document) => document,
            Err(_) => Document::from_parts(Dom::new(), Stylesheet::default(), ""),
        }
    }
}

/// The main-loop side of script execution: runs scripts on the
/// `TabJs` the tab thread owns. One is created per [`ExecMsg`] (it
/// holds no state beyond the shared host handle).
pub struct ScriptExecutor {
    host: Rc<std::sync::Mutex<BrowserDomHost>>,
}

impl ScriptExecutor {
    pub fn new(host: Rc<std::sync::Mutex<BrowserDomHost>>) -> Self {
        Self { host }
    }

    /// Execute one classic script. Errors are recorded at the
    /// per-script boundary inside `eval_script`; only a
    /// realm-killing error escapes it (the tab thread reads
    /// `TabJs::realm_poisoned`).
    pub fn exec_classic(&mut self, tab: &mut TabJs, source: &str, url: &str) {
        if let Err(e) = tab.eval_script(source, url) {
            eprintln!("[pluto] script killed the realm ({url}): {e}");
        }
    }

    /// Execute one queued script (defer / async): classic scripts
    /// run now, modules run loop-driven (`start_module_async` —
    /// a module that top-level-awaits a host event is driven
    /// across turns, never through the watchdog-poisoning
    /// `await_promise` path). A failed external fetch (empty
    /// bytes under a `src`) is skipped — the script element fires
    /// an `error` event in a later wave; today it is logged.
    pub fn exec_queued(
        &mut self,
        tab: &mut TabJs,
        node: NodeId,
        src: Option<String>,
        bytes: Vec<u8>,
    ) {
        if src.is_some() && bytes.is_empty() {
            eprintln!("[pluto] skipping script {node:?} (fetch failed)");
            return;
        }
        let source = if src.is_some() {
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            // Inline text comes from the host's live `Dom` — the
            // parser's arena is swapped for a dummy by
            // `finish_document`, so the parser must not be read
            // here (defer scripts run after the parse).
            self.host.lock().unwrap().inline_script_text(node)
        };
        let is_module = {
            let dom = self.host.lock().unwrap().dom_snapshot();
            dom.attr(node, "type")
                .is_some_and(|t| t.eq_ignore_ascii_case("module"))
        };
        // An inline module's imports resolve against the document
        // URL (Stage 12 Rec 1) — "inline.js" is not a URL, so the
        // resolver could never join against it. A classic inline
        // script keeps the friendly name (it only labels errors).
        let url = match src {
            Some(src) => src,
            None if is_module => self.host.lock().unwrap().base_url().to_string(),
            None => "inline.js".to_string(),
        };
        if is_module {
            if let Err(e) = tab.start_module_async(&url, &source) {
                eprintln!("[pluto] module killed the realm ({url}): {e}");
            }
        } else {
            self.exec_classic(tab, &source, &url);
        }
    }

    /// The parse finished: drain the defer queue in document order,
    /// fire `DOMContentLoaded`, drive the loop briefly, and return
    /// the document re-synced from the host's live `Dom` (JS
    /// mutations made during the parse / defer drain live there).
    pub fn finish_page(&mut self, tab: &mut TabJs, document: Document) -> Document {
        // The parser is done: the host's `document.write` takes the
        // post-parse implicit-`open()` path from here on.
        self.host.lock().unwrap().set_parser(None);
        self.host.lock().unwrap().set_document(&document);
        // Defer scripts (classic + module), in document order,
        // before `DOMContentLoaded`. A throwing defer script is
        // recorded at its own boundary and never blocks the rest.
        let defer: Vec<NodeId> = document.defer_scripts.clone();
        for node in defer {
            let held = document
                .held_scripts
                .iter()
                .find(|h| h.node == node)
                .cloned();
            let Some(held) = held else {
                continue;
            };
            self.exec_queued(tab, node, held.src, held.bytes);
        }
        // `DOMContentLoaded`: dispatched on the document as a task,
        // so listeners registered by the defer scripts run on the
        // loop, not inline here.
        if let Some(doc) = self.host.lock().unwrap().document_node() {
            tab.dispatch_event(doc, "DOMContentLoaded", false);
        }
        // Drive the loop briefly so timers / fetches / modules the
        // scripts started settle promptly; the tab thread keeps
        // driving afterwards on its frame tick.
        if let Err(e) = tab.drive_until_idle_or_timeout(POST_PARSE_DRIVE_BUDGET) {
            eprintln!("[pluto] post-parse drive failed: {e}");
        }
        // JS mutations made during the parse / defer drain live in
        // the host's `Dom`; the `Document` handed to the layout
        // actor must be re-synced from it.
        let live = self.host.lock().unwrap().dom_snapshot();
        document.with_dom(live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{layout_thread, LayoutJob};
    use pluto_ipc::CredentialsMode;
    use pluto_js::{JsValue, RealmConfig, TabJs};
    use std::sync::mpsc;

    /// A `TabJs` over a real layout actor thread, with the host's
    /// `Dom` holding `<div id="box">start</div>`.
    fn thrash_setup() -> (TabJs, Rc<std::sync::Mutex<BrowserDomHost>>) {
        let (layout_tx, layout_rx) = mpsc::channel::<LayoutJob>();
        std::thread::spawn(move || layout_thread(layout_rx));
        let html =
            "<!doctype html><html><head></head><body><div id=\"box\">start</div></body></html>";
        let host = BrowserDomHost::new(
            Arc::new(Mutex::new(pluto_html::parse(html))),
            layout_tx,
            (1024.0, 768.0),
            ElementState::default(),
            "http://127.0.0.1/",
            Rc::new(std::sync::Mutex::new(Vec::new())),
            Rc::new(Cell::new(false)),
        );
        let host_rc = Rc::new(std::sync::Mutex::new(host));
        let mut tab = TabJs::new(&RealmConfig::default()).expect("realm");
        tab.set_dom_host(Box::new(SharedHost(Rc::clone(&host_rc))));
        (tab, host_rc)
    }

    /// M4 (Rec 7): a read/write loop flushes once per iteration BY
    /// DESIGN — every `getBoundingClientRect` after a write forces
    /// a synchronous layout — while batched writes flush exactly
    /// once, on the single read at the end.
    #[test]
    fn layout_thrash_read_write_loop_flushes_per_iteration() {
        let (mut tab, host) = thrash_setup();
        // Stage 12 Rec 11: the host count routes into the
        // process-wide perf counters too — the delta across these
        // scripts matches the host count exactly. (The global is
        // process-wide; this is the only test in this binary that
        // drives a geometry flush, so the delta is deterministic
        // here. A future flush-driving test must take its own
        // baseline the same way.)
        let perf_before = perf::global().lock().unwrap().forced_flushes;
        tab.eval_script(
            "const box = document.getElementById('box'); \
             for (let i = 0; i < 5; i++) { box.textContent = 'x' + i; box.getBoundingClientRect(); }",
            "thrash.js",
        )
        .expect("loop script runs");
        assert_eq!(
            host.lock().unwrap().forced_flush_count(),
            5,
            "one forced flush per read-after-write"
        );
        tab.eval_script(
            // `box` is the `const` the first script declared in this
            // realm; redeclaring it would throw, so reuse it.
            "for (let i = 0; i < 5; i++) { box.textContent = 'y' + i; } \
             globalThis.lastRect = box.getBoundingClientRect();",
            "thrash.js",
        )
        .expect("batched script runs");
        assert_eq!(
            host.lock().unwrap().forced_flush_count(),
            6,
            "batched writes flush once, on the single read"
        );
        // Stage 12 Rec 11: the perf global saw the same six
        // flushes the host counted.
        assert_eq!(
            perf::global().lock().unwrap().forced_flushes - perf_before,
            6,
            "perf global tracks the host's forced flushes"
        );
        // The flushed rect is real geometry, not the zero fallback:
        // the div holds text, so it has a positive width.
        let width = tab
            .eval_value("globalThis.lastRect.width")
            .expect("read the rect width")
            .as_number()
            .expect("width is a number");
        assert!(width > 0.0, "flushed rect has a positive width");
        // And the batched writes landed: the last one wins.
        let host = host.lock().unwrap();
        let key = host.get_element_by_id("box").expect("box still there");
        assert_eq!(host.text_content(key).as_deref(), Some("y4"));
    }

    // ------------------------------------------- Stage 12 Rec 7 wiring
    //
    // The binding batch (location, dataset, cloneNode, Image(),
    // typed events) is built and tested against the pluto-js test
    // host — these tests prove the same behaviour through the
    // PRODUCTION host (`BrowserDomHost` + `SharedHost`), where
    // the defaults used to answer "not present".

    /// A `TabJs` over a real layout actor thread, with the host's
    /// `Dom` holding `html` and its base URL set to `base_url`.
    fn wiring_setup(html: &str, base_url: &str) -> (TabJs, Rc<std::sync::Mutex<BrowserDomHost>>) {
        let (layout_tx, layout_rx) = mpsc::channel::<LayoutJob>();
        std::thread::spawn(move || layout_thread(layout_rx));
        let host = BrowserDomHost::new(
            Arc::new(Mutex::new(pluto_html::parse(html))),
            layout_tx,
            (1024.0, 768.0),
            ElementState::default(),
            base_url,
            Rc::new(std::sync::Mutex::new(Vec::new())),
            Rc::new(Cell::new(false)),
        );
        let host_rc = Rc::new(std::sync::Mutex::new(host));
        let mut tab = TabJs::new(&RealmConfig::default()).expect("realm");
        tab.set_dom_host(Box::new(SharedHost(Rc::clone(&host_rc))));
        (tab, host_rc)
    }

    fn read_str(tab: &mut TabJs, source: &str) -> String {
        match tab.eval_value(source).unwrap() {
            JsValue::String(s) => s,
            other => panic!("expected a string from `{source}`, got {other:?}"),
        }
    }

    fn read_num(tab: &mut TabJs, source: &str) -> f64 {
        match tab.eval_value(source).unwrap() {
            JsValue::Number(n) => n,
            other => panic!("expected a number from `{source}`, got {other:?}"),
        }
    }

    fn read_bool(tab: &mut TabJs, source: &str) -> bool {
        match tab.eval_value(source).unwrap() {
            JsValue::Bool(b) => b,
            other => panic!("expected a bool from `{source}`, got {other:?}"),
        }
    }

    const WIRING_DOC: &str = "<!doctype html><html><head></head><body>\
        <div id=\"app\" class=\"root\"><p id=\"para\">Hello</p><span>Click</span></div>\
        <button id=\"btn\">Go</button><img id=\"pic\" alt=\"x\"></body></html>";

    #[test]
    fn production_host_location_reads_the_real_url() {
        let (mut tab, host) =
            wiring_setup(WIRING_DOC, "https://example.com/page?x=1#frag");
        tab.eval_script(
            "globalThis.__href = location.href; \
             globalThis.__host = location.hostname; \
             globalThis.__path = location.pathname; \
             globalThis.__search = location.search; \
             globalThis.__proto = location.protocol;",
            "loc.js",
        )
        .expect("location script runs");
        assert_eq!(read_str(&mut tab, "__href"), "https://example.com/page?x=1#frag");
        assert_eq!(read_str(&mut tab, "__host"), "example.com");
        assert_eq!(read_str(&mut tab, "__path"), "/page");
        assert_eq!(read_str(&mut tab, "__search"), "?x=1");
        assert_eq!(read_str(&mut tab, "__proto"), "https:");
        // assign / replace / reload record a navigation with the
        // host (they do not navigate in the realm).
        tab.eval_script("location.assign('/other');", "loc.js")
            .expect("assign runs");
        assert_eq!(
            host.lock().unwrap().take_pending_navigation(),
            Some(("https://example.com/other".to_string(), false)),
            "assign resolves against the document URL, no replace"
        );
        tab.eval_script("location.replace('https://example.com/r');", "loc.js")
            .expect("replace runs");
        assert_eq!(
            host.lock().unwrap().take_pending_navigation(),
            Some(("https://example.com/r".to_string(), true)),
            "replace records the replace flag"
        );
        tab.eval_script("location.reload();", "loc.js")
            .expect("reload runs");
        assert_eq!(
            host.lock().unwrap().take_pending_navigation(),
            Some(("https://example.com/page?x=1#frag".to_string(), false)),
            "reload re-requests the current URL"
        );
        // And `location.href = ...` navigates like assign.
        tab.eval_script("location.href = '/via-href';", "loc.js")
            .expect("href set runs");
        assert_eq!(
            host.lock().unwrap().take_pending_navigation(),
            Some(("https://example.com/via-href".to_string(), false)),
            "setting href navigates like assign"
        );
    }

    #[test]
    fn production_host_clone_node_round_trips() {
        let (mut tab, _host) = wiring_setup(WIRING_DOC, "https://example.com/");
        tab.eval_script(
            "const app = document.getElementById('app'); \
             const deep = app.cloneNode(true); \
             const shallow = app.cloneNode(false); \
             globalThis.__r = [deep.tagName, deep.childNodes.length, deep.className, deep.textContent, deep.parentNode].join('|'); \
             globalThis.__s = [shallow.tagName, shallow.childNodes.length, shallow.className].join('|');",
            "clone.js",
        )
        .expect("clone script runs");
        assert_eq!(read_str(&mut tab, "__r"), "DIV|2|root|HelloClick|");
        assert_eq!(read_str(&mut tab, "__s"), "DIV|0|root");
    }

    #[test]
    fn production_host_dataset_reads_and_writes_real_attributes() {
        let (mut tab, host) = wiring_setup(WIRING_DOC, "https://example.com/");
        tab.eval_script(
            "const app = document.getElementById('app'); \
             app.dataset.fooBar = 'baz'; \
             globalThis.__r = app.dataset.fooBar; \
             app.dataset.userName = 'Jo'; \
             globalThis.__keys = Object.keys(app.dataset).join(','); \
             delete app.dataset.fooBar; \
             globalThis.__gone = app.hasAttribute('data-foo-bar');",
            "dataset.js",
        )
        .expect("dataset script runs");
        assert_eq!(read_str(&mut tab, "__r"), "baz");
        assert_eq!(read_str(&mut tab, "__keys"), "fooBar,userName");
        assert!(!read_bool(&mut tab, "__gone"), "delete removes the attribute");
        // The writes landed as real content attributes on the
        // production host's live Dom, not a test-host copy.
        let host = host.lock().unwrap();
        let app = host.get_element_by_id("app").expect("app still there");
        assert_eq!(
            host.get_attribute(app, "data-user-name").as_deref(),
            Some("Jo"),
            "dataset write is a real data-* attribute"
        );
        assert_eq!(
            host.get_attribute(app, "data-foo-bar"),
            None,
            "dataset delete removed the real attribute"
        );
        assert_eq!(
            host.attribute_names(app),
            vec!["id".to_string(), "class".to_string(), "data-user-name".to_string()],
            "attribute_names lists the live attributes in source order"
        );
    }

    #[test]
    fn production_host_image_load_event_fires_on_shell_report() {
        let (mut tab, host) = wiring_setup(WIRING_DOC, "https://example.com/page");
        tab.eval_script(
            "const pic = document.getElementById('pic'); \
             let loaded = 0, onloaded = 0, errored = 0; \
             pic.addEventListener('load', () => { loaded = 1; }); \
             pic.addEventListener('error', () => { errored = 1; }); \
             pic.onload = () => { onloaded = 1; }; \
             globalThis.__loaded = () => loaded; \
             globalThis.__onloaded = () => onloaded; \
             globalThis.__errored = () => errored; \
             globalThis.__pic = pic; \
             pic.src = '/pic.png';",
            "img.js",
        )
        .expect("image script runs");
        // The `src` set asked the host for a load (resolved
        // against the document URL), and the host knows the
        // element to report back to.
        let pending = host.lock().unwrap().take_pending_images();
        assert_eq!(pending.len(), 1, "one image requested");
        assert_eq!(pending[0].url, "https://example.com/pic.png");
        assert_eq!(pending[0].base_url, "https://example.com/page");
        let targets = host.lock().unwrap().image_targets(&pending[0].url);
        assert_eq!(targets.len(), 1, "one element has this src");
        assert_eq!(
            targets[0],
            host.lock().unwrap().get_element_by_id("pic").unwrap(),
            "the target is the document img"
        );
        tab.notify_image_loaded(targets[0], 640, 480);
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__loaded()"), 1.0);
        assert_eq!(read_num(&mut tab, "__onloaded()"), 1.0);
        assert!(read_bool(&mut tab, "__pic.complete"));
        assert_eq!(read_num(&mut tab, "__pic.naturalWidth"), 640.0);
        assert_eq!(read_num(&mut tab, "__pic.naturalHeight"), 480.0);
        // A later failure reports `error` on the same element.
        tab.eval_script("pic.src = '/gone.png';", "img.js")
            .expect("second src set runs");
        let pending = host.lock().unwrap().take_pending_images();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].url, "https://example.com/gone.png");
        let targets = host.lock().unwrap().image_targets(&pending[0].url);
        assert_eq!(targets.len(), 1);
        tab.notify_image_failed(targets[0]);
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__errored()"), 1.0);
        assert!(!read_bool(&mut tab, "__pic.complete"));
    }

    #[test]
    fn production_host_detached_image_is_found_by_url_scan() {
        let (mut tab, host) = wiring_setup(WIRING_DOC, "https://example.com/page");
        tab.eval_script(
            "const img = new Image(); \
             let loaded = 0; \
             img.addEventListener('load', () => { loaded = 1; }); \
             globalThis.__imgLoaded = () => loaded; \
             globalThis.__img = img; \
             img.src = 'https://example.com/detached.png';",
            "img.js",
        )
        .expect("detached image script runs");
        let pending = host.lock().unwrap().take_pending_images();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].url, "https://example.com/detached.png");
        // The element is detached (never inserted), so no tree
        // walk reaches it — only the arena scan finds it.
        let targets = host.lock().unwrap().image_targets(&pending[0].url);
        assert_eq!(targets.len(), 1, "detached Image() found by URL scan");
        assert_ne!(
            targets[0],
            host.lock().unwrap().get_element_by_id("pic").unwrap(),
            "the target is the detached image, not the document img"
        );
        tab.notify_image_loaded(targets[0], 320, 240);
        tab.run_until_idle().unwrap();
        assert_eq!(read_num(&mut tab, "__imgLoaded()"), 1.0);
        assert!(read_bool(&mut tab, "__img.complete"));
        assert_eq!(read_num(&mut tab, "__img.naturalWidth"), 320.0);
    }

    #[test]
    fn production_host_typed_events_reach_js_with_fields() {
        use pluto_js::bindings::events::TypedEventKind;
        let (mut tab, host) = wiring_setup(WIRING_DOC, "https://example.com/");
        let btn = host.lock().unwrap().get_element_by_id("btn").unwrap();
        let para = host.lock().unwrap().get_element_by_id("para").unwrap();
        tab.eval_script(
            "const seenKey = []; \
             const seenMouse = []; \
             const seenInput = []; \
             document.getElementById('btn').addEventListener('keydown', (e) => { seenKey.push([e.key, e.code, e instanceof KeyboardEvent, e instanceof Event]); }); \
             document.getElementById('btn').addEventListener('click', (e) => { seenMouse.push([e.clientX, e.clientY, e.button, e instanceof MouseEvent]); }); \
             document.getElementById('para').addEventListener('input', (e) => { seenInput.push([e.data, e instanceof InputEvent, e.target === document.getElementById('para')]); }); \
             globalThis.__seenKey = seenKey; \
             globalThis.__seenMouse = seenMouse; \
             globalThis.__seenInput = seenInput;",
            "typed.js",
        )
        .expect("listener script runs");
        tab.dispatch_typed_event(
            btn,
            "keydown",
            true,
            TypedEventKind::Keyboard {
                key: "Enter".to_string(),
                code: "Enter".to_string(),
            },
        );
        tab.run_until_idle().unwrap();
        assert_eq!(
            read_str(&mut tab, "__seenKey[0].join(',')"),
            "Enter,Enter,true,true"
        );
        tab.dispatch_typed_event(
            btn,
            "click",
            true,
            TypedEventKind::Mouse {
                client_x: 10.0,
                client_y: 20.0,
                button: 0,
            },
        );
        tab.run_until_idle().unwrap();
        assert_eq!(
            read_str(&mut tab, "__seenMouse[0].join(',')"),
            "10,20,0,true"
        );
        tab.dispatch_typed_event(
            para,
            "input",
            true,
            TypedEventKind::Input {
                data: Some("x".to_string()),
            },
        );
        tab.run_until_idle().unwrap();
        assert_eq!(
            read_str(&mut tab, "__seenInput[0].join(',')"),
            "x,true,true"
        );
    }

    // ------------------------------------------------------------ fetch gate
    //
    // Stage 9 Rec 1: the same-origin gate compares parsed `Origin`s.

    /// A fetch backend over a real (never used) loader: the gate
    /// rejects before any network access, so these tests need no
    /// network. The runtime is kept alive for the backend's handle.
    fn fetch_backend(base_url: &str) -> (PlutoFetchBackend, tokio::runtime::Runtime) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        let backend =
            PlutoFetchBackend::new(
            crate::broker::FetchSource::Direct(Arc::new(ResourceLoader::new())),
            rt.handle().clone(),
            base_url,
            0,
        );
        (backend, rt)
    }

    fn fetch_request(url: &str) -> FetchRequest {
        FetchRequest {
            id: 1,
            url: url.to_string(),
            method: "GET".to_string(),
            headers: Vec::new(),
            body: None,
            // Stage 9 Rec 2: the `fetch()` defaults — mode `cors`,
            // credentials `same-origin`.
            mode: RequestMode::Cors,
            credentials: CredentialsMode::SameOrigin,
        }
    }

    /// The gate's synchronous answer to one `start_fetch`, if any
    /// (rejections answer synchronously; an allowed fetch spawns a
    /// network task and answers later, so these tests only assert
    /// rejections).
    fn gate_answer(
        backend: &PlutoFetchBackend,
        url: &str,
    ) -> Option<Result<FetchResponse, FetchError>> {
        let (tx, rx) = mpsc::channel();
        backend.start_fetch(fetch_request(url), tx);
        rx.try_recv().ok().map(|c| c.result)
    }

    #[test]
    fn fetch_gate_passes_cross_origin_to_the_cors_backend() {
        // Stage 9 Rec 2: the M4 hard reject is gone — a
        // cross-origin fetch goes to `pluto-net` as a CORS fetch,
        // so there is no synchronous rejection (the network task
        // decides: preflight / CORS check / opaque filtering).
        let (backend, _rt) = fetch_backend("https://example.com/page");
        assert!(gate_answer(&backend, "https://evil.example/x").is_none());
        // A different port is a different origin — same treatment.
        assert!(gate_answer(&backend, "https://example.com:8443/x").is_none());
        // ...but the default port is the SAME origin, so the gate
        // lets it through (no synchronous answer: it goes to the
        // network task, which this test never awaits).
        assert!(gate_answer(&backend, "https://example.com:443/x").is_none());
        // Relative URLs resolve against the document base and pass
        // the gate the same way.
        assert!(gate_answer(&backend, "/api/items").is_none());
        // An opaque document fetching cross-origin fails fast (fail
        // closed — an opaque origin can never pass a CORS check).
        let (opaque, _rt2) = fetch_backend("about:blank");
        assert!(matches!(
            gate_answer(&opaque, "https://example.com/x"),
            Some(Err(FetchError::InvalidUrl(_)))
        ));
    }

    #[test]
    fn fetch_gate_same_origin_mode_rejects_cross_origin() {
        // Stage 9 Rec 2: `mode: "same-origin"` is enforced here,
        // before any network access — a cross-origin URL with
        // that mode answers synchronously with a rejection, while
        // the same URL in default `cors` mode passes through to
        // the CORS backend (see the test above).
        let (backend, _rt) = fetch_backend("https://example.com/page");
        let answer_with_mode = |url: &str, mode: RequestMode| {
            let (tx, rx) = mpsc::channel();
            let mut request = fetch_request(url);
            request.mode = mode;
            backend.start_fetch(request, tx);
            rx.try_recv().ok().map(|c| c.result)
        };
        assert!(
            matches!(
                answer_with_mode("https://evil.example/x", RequestMode::SameOrigin),
                Some(Err(FetchError::InvalidUrl(_)))
            ),
            "same-origin mode rejects a cross-origin URL synchronously"
        );
        // Same mode, same origin: no synchronous rejection (the
        // network task runs; this test never awaits it).
        assert!(
            answer_with_mode("https://example.com/x", RequestMode::SameOrigin).is_none(),
            "same-origin mode allows a same-origin URL"
        );
        // A different port is a different origin here too.
        assert!(
            matches!(
                answer_with_mode("https://example.com:8443/x", RequestMode::SameOrigin),
                Some(Err(FetchError::InvalidUrl(_)))
            ),
            "same-origin mode rejects a cross-origin port"
        );
    }

    /// Tiny local server for the CORS round-trip test: `/data` is
    /// granted to any origin, `/no-cors` to none.
    async fn serve_cors(listener: tokio::net::TcpListener) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
                let response = match path {
                    "/data" => {
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: 10\r\nConnection: close\r\n\r\nhello-cors"
                            .to_string()
                    }
                    _ => {
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 7\r\nConnection: close\r\n\r\nno-cors"
                            .to_string()
                    }
                };
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    #[test]
    fn fetch_gate_cross_origin_cors_round_trip() {
        // Stage 9 Rec 2, end to end: a cross-origin `fetch()`
        // through the real loader resolves when the server grants
        // CORS, and rejects (network error) when it does not. The
        // document base is a different origin from the server (a
        // different port), so this is a genuine cross-origin fetch.
        let (backend, rt) = fetch_backend("http://127.0.0.1:1/page");
        rt.block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(serve_cors(listener));
            let fetch = |url: String| {
                let (tx, rx) = mpsc::channel();
                backend.start_fetch(fetch_request(&url), tx);
                tokio::task::spawn_blocking(move || {
                    rx.recv_timeout(Duration::from_secs(10))
                })
            };
            // Granted: resolves with the body.
            let completion = fetch(format!("http://127.0.0.1:{port}/data"))
                .await
                .expect("spawn_blocking runs")
                .expect("a completion arrives");
            let response = completion.result.expect("CORS-granted fetch resolves");
            assert_eq!(response.status, 200);
            assert_eq!(response.body, b"hello-cors");
            // Refused: rejects with a network error.
            let completion = fetch(format!("http://127.0.0.1:{port}/no-cors"))
                .await
                .expect("spawn_blocking runs")
                .expect("a completion arrives");
            assert!(
                matches!(completion.result, Err(FetchError::Network(_))),
                "CORS-refused fetch must reject, got {:?}", 
                completion.result
            );
        });
    }

    #[test]
    fn fetch_gate_still_rejects_mixed_content_and_bad_schemes() {
        let (backend, _rt) = fetch_backend("https://example.com/page");
        assert!(matches!(
            gate_answer(&backend, "http://example.com/x"),
            Some(Err(FetchError::Network(_)))
        ));
        assert!(matches!(
            gate_answer(&backend, "data:text/html,<p>x</p>"),
            Some(Err(FetchError::InvalidUrl(_)))
        ));
    }

    #[test]
    fn fetch_backend_tracks_document_origin_on_navigation() {
        let (backend, _rt) = fetch_backend("https://example.com:443/page");
        // Default-port normalization: the stored origin has no port.
        assert_eq!(
            backend.document_origin(),
            Origin::from_url("https://example.com")
        );
        backend.set_base_url("https://other.example:8443/x");
        assert_eq!(
            backend.document_origin().serialize(),
            "https://other.example:8443"
        );
        // A top-level `about:blank` is opaque...
        backend.set_base_url("about:blank");
        let blank = backend.document_origin();
        assert!(blank.is_opaque());
        // ...and a `data:` document is a *different* opaque origin.
        backend.set_base_url("data:text/html,<p>x</p>");
        let data = backend.document_origin();
        assert!(data.is_opaque());
        assert_ne!(data, blank);
        // `about:blank` created by a page inherits the creator.
        let creator = Origin::from_url("https://example.com/");
        backend.set_base_url_with_creator("about:blank", Some(&creator), false);
        assert_eq!(backend.document_origin(), creator);
        // A sandboxed document is opaque even with a creator.
        backend.set_base_url_with_creator("about:blank", Some(&creator), true);
        assert!(backend.document_origin().is_opaque());
    }

    #[test]
    fn js_query_selector_cannot_observe_visited() {
        // Stage 10 Rec 3 (privacy): even when the host holds a
        // visited element state — the state the layout pass
        // styles with — JS selector matching must never see
        // `:visited`. The query path always matches against
        // `ElementState::default()`, so no binding, `matches()`,
        // or `querySelector` call can probe history.
        let (layout_tx, _layout_rx) = mpsc::channel::<LayoutJob>();
        let html = "<a href=\"https://example.com\">L</a><a href=\"https://example.org\">M</a>";
        let mut host = BrowserDomHost::new(
            Arc::new(Mutex::new(pluto_html::parse(html))),
            layout_tx,
            (1024.0, 768.0),
            ElementState::default(),
            "https://example.com/",
            Rc::new(std::sync::Mutex::new(Vec::new())),
            Rc::new(Cell::new(false)),
        );
        // Mark the first link visited in the host's element state.
        let snapshot = host.dom_snapshot();
        let root = snapshot.root().unwrap();
        let mut links = Vec::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if snapshot.tag(id) == Some("a") {
                links.push(id);
            }
            stack.extend(snapshot.children(id).iter().rev());
        }
        assert_eq!(links.len(), 2);
        host.set_element_state(ElementState::default().with_visited(links[0]));
        // Sanity: plain queries still work with a visited state held.
        assert_eq!(host.query_selector_all(root, "a").len(), 2);
        assert_eq!(host.query_selector(root, "a"), Some(links[0]));
        // `:visited` is invisible to JS — no match, even for the
        // link the host knows is visited.
        assert!(host.query_selector_all(root, "a:visited").is_empty());
        assert!(host.query_selector_all(root, ":visited").is_empty());
        assert_eq!(host.query_selector(root, "a:visited"), None);
        // `:not(:visited)` therefore matches every link here.
        assert_eq!(host.query_selector_all(root, "a:not(:visited)").len(), 2);
    }
}
