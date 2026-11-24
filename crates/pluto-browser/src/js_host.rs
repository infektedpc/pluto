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
use pluto_html::{
    Dom, NodeId, NodeKind, Parser, QueuedScript, ScriptPaused, parse_fragment,
    resolve_url,
};
use pluto_ipc::{CredentialsMode, Destination, Priority, RequestMode, ResourceRequest, origin_of};
use pluto_js::{
    DomHost, DomHostError, DomKey, FetchBackend, FetchCompletion, FetchError, FetchRequest,
    FetchResponse, Geometry, JsError, RealmConfig, TabJs,
};
use pluto_layout::{ElementState, selector_matches};
use pluto_net::ResourceLoader;
use tokio::sync::{mpsc::UnboundedSender, oneshot};
use tokio_util::sync::CancellationToken;

use crate::page::{Document, ScriptHooks, script_text};
use crate::renderer::LayoutJob;

// TODO(M5): workers (dedicated/shared/service) — each needs its own
// agent/loop/realm; the per-tab `TabJs` here is the window agent only.
// TODO(M5): Shadow DOM — no shadow-root host methods exist yet; the
// `DomHost` surface would gain `attach_shadow` / `shadow_root` here.
// TODO(M8): MutationObserver / ResizeObserver / IntersectionObserver —
// the observer notification queues hook into the loop's
// `DomManipulation` source and the rendering step; no observer state
// is kept here in M4.
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

    /// Snapshot the live `Dom` (the tab thread re-syncs the
    /// `Document` it hands to the layout actor from this).
    pub fn dom_snapshot(&self) -> Dom {
        self.dom.lock().unwrap().clone()
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
}

/// Create a fresh [`TabJs`] for the tab thread (one realm per tab,
/// fresh per navigation via `TabJs::navigate`).
pub fn new_tab_js() -> Result<TabJs, JsError> {
    TabJs::new(&RealmConfig::default())
}

// ------------------------------------------------------------------ fetch
//
// `fetch()` over `pluto-net` (Recommendation 8): the binding builds
// the request, this backend runs it, and the loop resolves the
// promise — never the backend, never inline on a network thread.

/// The `fetch()` backend for one tab: a [`ResourceLoader`] plus the
/// document origin / epoch the same-origin gate checks against.
///
/// # Same-origin gate (M4)
///
/// Relative URLs resolve against the document base; an absolute
/// cross-origin URL is rejected here, before any network access —
/// the promise rejects with a `TypeError` (the binding maps a
/// backend error to a rejection). This gate IS the M4 origin check
/// the bindings agent warned was a precondition for wiring `fetch`
/// to a real backend. It is not CORS:
///
/// TODO(M5): CORS — full preflight / `mode` / credentials handling
/// lands at this gate (the `RequestMode::Cors` path, opaque
/// `no-cors` responses, and the preflight cache).
/// TODO(M10): CSP — the `connect-src` check lands at this gate,
/// before the origin check (gate order when it lands: HSTS →
/// blocker → CSP → CORS → cache → network).
pub struct PlutoFetchBackend {
    loader: ResourceLoader,
    /// The tab thread's tokio runtime handle: `start_fetch` spawns
    /// the network task on it (the tab thread drives the runtime
    /// on its frame tick).
    handle: tokio::runtime::Handle,
    /// Document base URL, updated by the tab thread on navigation.
    base_url: Arc<Mutex<String>>,
    /// Document origin (`scheme://authority`), from `base_url`.
    origin: Arc<Mutex<String>>,
    /// Navigation epoch; a completion for a stale epoch resolves
    /// as `Cancelled` (the loop drops it anyway — belt and braces).
    epoch: Arc<AtomicU64>,
    next_request_id: AtomicU64,
}

impl PlutoFetchBackend {
    pub fn new(
        loader: ResourceLoader,
        handle: tokio::runtime::Handle,
        base_url: &str,
    ) -> Self {
        Self {
            loader,
            handle,
            base_url: Arc::new(Mutex::new(base_url.to_string())),
            origin: Arc::new(Mutex::new(origin_of(base_url))),
            epoch: Arc::new(AtomicU64::new(0)),
            next_request_id: AtomicU64::new(1),
        }
    }

    /// Re-point the gate at a new document (navigation).
    pub fn set_base_url(&self, url: &str) {
        *self.base_url.lock().unwrap() = url.to_string();
        *self.origin.lock().unwrap() = origin_of(url);
    }

    /// Bump the epoch (navigation): in-flight fetches for the old
    /// document complete as `Cancelled`.
    pub fn set_epoch(&self, epoch: u64) {
        self.epoch.store(epoch, Ordering::SeqCst);
    }
}

impl FetchBackend for PlutoFetchBackend {
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
        // Same-origin gate (M4): absolute cross-origin fetch MUST
        // fail, not silently succeed. TODO(M5): CORS replaces this
        // hard reject with the preflight / safelist check.
        let doc_origin = self.origin.lock().unwrap().clone();
        if origin_of(&absolute) != doc_origin {
            send(Err(FetchError::InvalidUrl(format!(
                "cross-origin fetch blocked in M4 (same-origin only; TODO(M5) CORS): {absolute}"
            ))));
            return;
        }
        // TODO(M10): CSP `connect-src` check goes here, before the
        // network, once CSP lands.
        let resource = ResourceRequest {
            request_id: self.next_request_id.fetch_add(1, Ordering::Relaxed),
            tab_id: 0,
            epoch: self.epoch.load(Ordering::SeqCst),
            url: absolute,
            destination: Destination::Fetch,
            priority: Priority::Medium,
            // Same-origin only in M4: cross-origin fails before the
            // network (see the gate above), so `SameOrigin` mode is
            // the honest request mode here.
            mode: RequestMode::SameOrigin,
            credentials: CredentialsMode::SameOrigin,
            origin: doc_origin.clone(),
            top_level_site: doc_origin,
            upgrade_insecure_requests: false,
        };
        let loader = self.loader.clone();
        let epoch = Arc::clone(&self.epoch);
        let start_epoch = epoch.load(Ordering::SeqCst);
        self.handle.spawn(async move {
            let outcome = match loader
                .fetch_with(resource, CancellationToken::new())
                .await
            {
                Ok(response) => Ok(FetchResponse {
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
                }),
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
        let url = src.unwrap_or_else(|| "inline.js".to_string());
        let is_module = {
            let dom = self.host.lock().unwrap().dom_snapshot();
            dom.attr(node, "type")
                .is_some_and(|t| t.eq_ignore_ascii_case("module"))
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
    use pluto_js::{RealmConfig, TabJs};
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
}
