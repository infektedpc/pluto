//! `TreeSink` over the slotmap [`Dom`](crate::Dom) — the html5ever
//! side of Pluto's parser.
//!
//! Design decisions (each is a spike finding — see
//! `~/workspace/pluto-spike/SPIKE.md` §2):
//!
//! * `Handle = NodeId` (`Copy + Clone`, generational slotmap key). No
//!   `Rc`/`RefCell` anywhere in the DOM; the sink holds the `Dom`
//!   behind `Arc<Mutex<…>>` because every `TreeSink` method takes
//!   `&self`. `Mutex` (not `RefCell`) keeps the sink `Send + Sync`,
//!   which is what Pluto needs for the per-tab worker thread to own
//!   the parser while another thread reads the `Dom`. The parser
//!   itself (`Tokenizer` / `TreeBuilder`, full of `Cell`/`RefCell`)
//!   stays single-threaded.
//! * `ElemName` is an *owned* struct ([`PlutoElemName`], holding
//!   interned `Namespace`/`LocalName` atoms). `TreeSink::elem_name`
//!   must return something borrowed from `&self`, and the `Dom` stores
//!   only a plain tag `String` — so the sink keeps the full `QualName`
//!   per element in a side map and clones its (cheap, refcounted)
//!   atoms on each call.
//! * `Attribute.value` is a `StrTendril` with a NonAtomic refcount —
//!   `!Send`/`!Sync`. The sink projects attribute values to owned
//!   `String` immediately and never retains an `Attribute` /
//!   `StrTendril`. (A first attempt storing full `Attribute`s failed
//!   the `Send`/`Sync` asserts.)
//! * Namespaces / attribute prefixes are not storable in Pluto's
//!   `Node.attrs: Vec<(String, String)>`. The sink keeps full names /
//!   attrs in side maps carried in [`ParseOutput`]. Foreign-content
//!   fidelity therefore depends on the side map; storing the
//!   `QualName` in the `Dom` node is a future step, not taken here.
//! * Text nodes are merged (`append` of `AppendText` extends the last
//!   text child), per the `TreeSink` contract. Pluto's old parser
//!   dropped whitespace-only text; this sink keeps everything, as the
//!   spec does.
//! * `<template>` contents live in a bare `Document` node created at
//!   `create_element` time (the `template: true` flag), held in a side
//!   map. Pluto has no fragment node kind; a `Document` node is the
//!   closest available holder.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use html5ever::interface::tree_builder::{
    ElementFlags, NodeOrText, QuirksMode as Html5everQuirksMode, TreeSink,
};
use html5ever::interface::ElemName;
use html5ever::tendril::StrTendril;
use html5ever::{Attribute, LocalName, Namespace, QualName, ns};

use crate::scanner::DiscoveredResource;
use crate::{Dom, NodeId, NodeKind, QuirksMode};

/// Events the parser observes while building the tree. The streaming
/// driver ([`crate::streaming::Parser`]) drains these from the sink
/// after each `feed` and hands them to the worker, so style/layout and
/// subresource fetch can run on the completed prefix before EOF.
/// `ScriptPaused` and `EncodingIndicator` are *not* emitted by the
/// sink — they come from the `TokenizerResult` the driver sees — but
/// they live in this enum so a feed returns one event stream.
#[derive(Debug, Clone)]
pub enum DomEvent {
    /// An element was just created (attributes complete; children
    /// not yet attached).
    ElementCreated { node: NodeId, tag: String },
    /// An element was popped from the open-element stack: its
    /// subtree is complete as far as the tree builder knows. (Best
    /// effort: adoption-agency moves can still relocate formatting
    /// elements afterwards.)
    SubtreeComplete { node: NodeId },
    /// A classic parser-blocking `<script>` just completed: the
    /// parser paused here. See [`crate::streaming::ScriptPaused`].
    ScriptPaused(crate::streaming::ScriptPaused),
    /// A `<link rel=stylesheet>` was created (render-blocking
    /// decision point for the worker).
    StylesheetDiscovered { node: NodeId, href: String },
    /// A fetchable subresource was discovered on a created element
    /// (same classification as the preload scanner —
    /// [`crate::scanner::classify_element`]).
    ResourceDiscovered(DiscoveredResource),
    /// The document indicated a different encoding (label
    /// unvalidated). html5ever does not restart decoding itself; the
    /// caller must buffer the prefix and re-parse (see
    /// [`crate::streaming::Parser`]).
    EncodingIndicator { label: String },
}

/// Owned element name: the `ElemName` implementation the sink returns.
///
/// html5ever's own examples store `&'a QualName` (borrowing a leaked
/// or arena-allocated `QualName`). Pluto's `Dom` stores only a `String`
/// tag, so borrowing is impossible without a side allocation; cloning
/// the two interned atoms per `elem_name` call is cheap and leak-free.
#[derive(Clone, PartialEq, Eq)]
pub struct PlutoElemName {
    pub ns: Namespace,
    pub local: LocalName,
}

impl fmt::Debug for PlutoElemName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ns.is_empty() {
            write!(f, "{}", self.local)
        } else {
            write!(f, "{{{}}}:{}", self.ns, self.local)
        }
    }
}

impl ElemName for PlutoElemName {
    fn ns(&self) -> &Namespace {
        &self.ns
    }
    fn local_name(&self) -> &LocalName {
        &self.local
    }
}

/// Parse-error / fidelity statistics for one parse. Parse errors are
/// never fatal (the tree builder recovers per spec); these exist so
/// tests and diagnostics can see how lossy a parse was.
#[derive(Debug, Default, Clone)]
pub struct SinkStats {
    pub parse_errors: usize,
    pub first_errors: Vec<String>,
    /// Attributes dropped because the Dom projection keeps local-name only.
    pub attrs_with_ns_or_prefix: usize,
    /// Elements created in a non-HTML namespace (fidelity via side map).
    pub foreign_elements: usize,
    pub comments: usize,
    pub doctypes: usize,
    pub processing_instructions: usize,
}

/// Everything a finished parse returns: the `Dom` plus the side maps
/// the `Dom` shape itself cannot hold (namespaces, full attributes,
/// template contents), the quirks mode, stats, and the events observed
/// during the parse.
#[derive(Debug)]
pub struct ParseOutput {
    pub dom: Dom,
    pub names: HashMap<NodeId, QualName>,
    pub full_attrs: HashMap<NodeId, Vec<(QualName, String)>>,
    pub template_contents: HashMap<NodeId, NodeId>,
    pub quirks_mode: QuirksMode,
    pub stats: SinkStats,
    pub events: Vec<DomEvent>,
}

impl ParseOutput {
    pub fn document(&self) -> NodeId {
        self.dom.root().expect("Dom::new always sets a root")
    }
}

fn map_quirks(mode: Html5everQuirksMode) -> QuirksMode {
    match mode {
        Html5everQuirksMode::NoQuirks => QuirksMode::NoQuirks,
        Html5everQuirksMode::Quirks => QuirksMode::Quirks,
        Html5everQuirksMode::LimitedQuirks => QuirksMode::LimitedQuirks,
    }
}

#[derive(Clone)]
pub struct PlutoSink {
    dom: Arc<Mutex<Dom>>,
    names: Arc<Mutex<HashMap<NodeId, QualName>>>,
    full_attrs: Arc<Mutex<HashMap<NodeId, Vec<(QualName, String)>>>>,
    template_contents: Arc<Mutex<HashMap<NodeId, NodeId>>>,
    mathml_annotation_xml_ip: Arc<Mutex<HashSet<NodeId>>>,
    quirks_mode: Arc<Mutex<Html5everQuirksMode>>,
    stats: Arc<Mutex<SinkStats>>,
    events: Arc<Mutex<Vec<DomEvent>>>,
    /// First `<base href>` wins (spec). Tracked here so the streaming
    /// driver can resolve resource URLs against it with the same rule
    /// the preload scanner uses.
    base_href: Arc<Mutex<Option<String>>>,
}

impl Default for PlutoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PlutoSink {
    pub fn new() -> Self {
        let mut dom = Dom::new();
        // Parser privilege: no mutation records / dirty bits while the
        // tree is being built (see crate docs). Cleared in finish().
        dom.set_suppress_records(true);
        Self {
            dom: Arc::new(Mutex::new(dom)),
            names: Arc::new(Mutex::new(HashMap::new())),
            full_attrs: Arc::new(Mutex::new(HashMap::new())),
            template_contents: Arc::new(Mutex::new(HashMap::new())),
            mathml_annotation_xml_ip: Arc::new(Mutex::new(HashSet::new())),
            quirks_mode: Arc::new(Mutex::new(Html5everQuirksMode::NoQuirks)),
            stats: Arc::new(Mutex::new(SinkStats::default())),
            events: Arc::new(Mutex::new(Vec::new())),
            base_href: Arc::new(Mutex::new(None)),
        }
    }

    /// Shared handle to the live `Dom`, readable while parsing is in
    /// progress (the incremental / streaming seam: the parser thread
    /// owns the `Parser`, other threads read through this `Arc`).
    pub fn shared_dom(&self) -> Arc<Mutex<Dom>> {
        Arc::clone(&self.dom)
    }

    pub fn quirks_mode(&self) -> QuirksMode {
        map_quirks(*self.quirks_mode.lock().unwrap())
    }

    pub fn stats(&self) -> SinkStats {
        self.stats.lock().unwrap().clone()
    }

    /// First `<base href>` seen so far, if any (raw value).
    pub fn base_href(&self) -> Option<String> {
        self.base_href.lock().unwrap().clone()
    }

    /// Drain the events accumulated since the last drain.
    pub fn drain_events(&self) -> Vec<DomEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    fn push_event(&self, event: DomEvent) {
        self.events.lock().unwrap().push(event);
    }

    /// Project one html5ever `Attribute` onto Pluto's `(String, String)`
    /// shape: local name + value. Namespace/prefix are counted, not kept
    /// (the full attribute lives in the `full_attrs` side map).
    fn project_attr(&self, attr: &Attribute) -> (String, String) {
        let has_ns = !attr.name.ns.is_empty() || attr.name.prefix.is_some();
        if has_ns {
            self.stats.lock().unwrap().attrs_with_ns_or_prefix += 1;
        }
        (attr.name.local.to_string(), attr.value.to_string())
    }

    fn append_impl(&self, parent: NodeId, child: NodeOrText<NodeId>) {
        let mut dom = self.dom.lock().unwrap();
        match child {
            NodeOrText::AppendNode(node) => dom.append_child(parent, node),
            NodeOrText::AppendText(text) => {
                // Merge with the previous sibling text node, per contract.
                let last = dom.children(parent).last().copied();
                match last {
                    Some(t)
                        if matches!(dom.get(t), Some(n) if matches!(n.kind, NodeKind::Text(_))) =>
                    {
                        dom.append_to_text(t, &text);
                    }
                    _ => {
                        let id = dom.create_text(&text);
                        dom.append_child(parent, id);
                    }
                }
            }
        }
    }

    fn append_before_impl(&self, sibling: NodeId, new_node: NodeOrText<NodeId>) {
        let parent = {
            let dom = self.dom.lock().unwrap();
            match dom.get(sibling).and_then(|n| n.parent) {
                Some(p) => p,
                // No parent (foster-parenting edge): fall back to append
                // to the document. Should not happen for real siblings.
                None => dom.root().unwrap(),
            }
        };
        let mut dom = self.dom.lock().unwrap();
        match new_node {
            NodeOrText::AppendNode(node) => dom.insert_before(parent, sibling, node),
            NodeOrText::AppendText(text) => {
                // Merge with the text node directly before `sibling`.
                let kids = dom.children(parent).to_vec();
                let pos = kids.iter().position(|&c| c == sibling).unwrap_or(kids.len());
                let prev = if pos > 0 { Some(kids[pos - 1]) } else { None };
                match prev {
                    Some(t)
                        if matches!(dom.get(t), Some(n) if matches!(n.kind, NodeKind::Text(_))) =>
                    {
                        dom.append_to_text(t, &text);
                    }
                    _ => {
                        let id = dom.create_text(&text);
                        dom.insert_before(parent, sibling, id);
                    }
                }
            }
        }
    }
}

impl TreeSink for PlutoSink {
    type Handle = NodeId;
    type Output = ParseOutput;
    type ElemName<'a> = PlutoElemName;

    fn finish(self) -> ParseOutput {
        // The sink may still be shared (e.g. a clone held by a streaming
        // driver), so fall back to cloning the state out of the Arcs.
        let mut dom = match Arc::try_unwrap(self.dom) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        // The parse is over: JS-driven mutations from here on record
        // and mark dirty like any other mutation.
        dom.set_suppress_records(false);
        let names = match Arc::try_unwrap(self.names) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        let full_attrs = match Arc::try_unwrap(self.full_attrs) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        let template_contents = match Arc::try_unwrap(self.template_contents) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        let events = match Arc::try_unwrap(self.events) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        let quirks = map_quirks(*self.quirks_mode.lock().unwrap());
        dom.set_quirks_mode(quirks);
        ParseOutput {
            dom,
            names,
            full_attrs,
            template_contents,
            quirks_mode: quirks,
            stats: self.stats.lock().unwrap().clone(),
            events,
        }
    }

    fn parse_error(&self, msg: Cow<'static, str>) {
        let mut stats = self.stats.lock().unwrap();
        stats.parse_errors += 1;
        if stats.first_errors.len() < 10 {
            stats.first_errors.push(msg.into_owned());
        }
    }

    fn get_document(&self) -> NodeId {
        self.dom.lock().unwrap().root().unwrap()
    }

    fn elem_name(&self, target: &NodeId) -> PlutoElemName {
        let names = self.names.lock().unwrap();
        match names.get(target) {
            Some(q) => PlutoElemName {
                ns: q.ns.clone(),
                local: q.local.clone(),
            },
            None => panic!("elem_name called on non-element node {target:?}"),
        }
    }

    fn create_element(
        &self,
        name: QualName,
        attrs: Vec<Attribute>,
        flags: ElementFlags,
    ) -> NodeId {
        let projected: Vec<(String, String)> =
            attrs.iter().map(|a| self.project_attr(a)).collect();
        let mut dom = self.dom.lock().unwrap();
        let id = dom.create_element_with_attrs(&name.local, projected.clone());
        drop(dom);
        if !name.ns.is_empty() && name.ns != ns!(html) {
            self.stats.lock().unwrap().foreign_elements += 1;
        }
        self.names.lock().unwrap().insert(id, name.clone());
        // Full attributes, with values projected to owned Strings NOW:
        // html5ever's Attribute.value is a StrTendril (NonAtomic
        // refcount, !Send/!Sync) and must never be retained in the sink.
        let full: Vec<(QualName, String)> = attrs
            .iter()
            .map(|a| (a.name.clone(), a.value.to_string()))
            .collect();
        self.full_attrs.lock().unwrap().insert(id, full);
        if flags.template {
            let contents = self.dom.lock().unwrap().create_document();
            self.template_contents.lock().unwrap().insert(id, contents);
        }
        if flags.mathml_annotation_xml_integration_point {
            self.mathml_annotation_xml_ip.lock().unwrap().insert(id);
        }
        // First <base href> wins (spec) — tracked for URL resolution.
        if &*name.local == "base" {
            let mut base = self.base_href.lock().unwrap();
            if base.is_none() {
                if let Some((_, href)) = projected.iter().find(|(n, _)| n == "href") {
                    if !href.trim().is_empty() {
                        *base = Some(href.clone());
                    }
                }
            }
        }
        self.push_event(DomEvent::ElementCreated {
            node: id,
            tag: name.local.to_string(),
        });
        id
    }

    fn create_comment(&self, text: StrTendril) -> NodeId {
        self.stats.lock().unwrap().comments += 1;
        self.dom.lock().unwrap().create_comment(&text)
    }

    fn create_pi(&self, target: StrTendril, data: StrTendril) -> NodeId {
        self.stats.lock().unwrap().processing_instructions += 1;
        self.dom.lock().unwrap().create_pi(&target, &data)
    }

    fn append(&self, parent: &NodeId, child: NodeOrText<NodeId>) {
        self.append_impl(*parent, child);
    }

    fn append_based_on_parent_node(
        &self,
        element: &NodeId,
        prev_element: &NodeId,
        child: NodeOrText<NodeId>,
    ) {
        // Foster parenting (html5ever's arena example does exactly this):
        // if the table still has a parent, insert before it; otherwise
        // append to the element that used to precede it.
        let has_parent = self
            .dom
            .lock()
            .unwrap()
            .get(*element)
            .and_then(|n| n.parent)
            .is_some();
        if has_parent {
            self.append_before_impl(*element, child);
        } else {
            self.append_impl(*prev_element, child);
        }
    }

    fn append_doctype_to_document(
        &self,
        name: StrTendril,
        public_id: StrTendril,
        system_id: StrTendril,
    ) {
        self.stats.lock().unwrap().doctypes += 1;
        let mut dom = self.dom.lock().unwrap();
        let root = dom.root().unwrap();
        let dt = dom.create_doctype(&name, &public_id, &system_id);
        dom.append_child(root, dt);
    }

    fn same_node(&self, x: &NodeId, y: &NodeId) -> bool {
        x == y
    }

    fn set_quirks_mode(&self, mode: Html5everQuirksMode) {
        *self.quirks_mode.lock().unwrap() = mode;
    }

    fn append_before_sibling(&self, sibling: &NodeId, new_node: NodeOrText<NodeId>) {
        self.append_before_impl(*sibling, new_node);
    }

    fn add_attrs_if_missing(&self, target: &NodeId, attrs: Vec<Attribute>) {
        let projected: Vec<(String, String)> =
            attrs.iter().map(|a| self.project_attr(a)).collect();
        // Keep the full-attribute side map in sync too.
        {
            let mut full = self.full_attrs.lock().unwrap();
            if let Some(existing) = full.get_mut(target) {
                let have: Vec<QualName> =
                    existing.iter().map(|(n, _)| n.clone()).collect();
                for attr in attrs {
                    if !have.contains(&attr.name) {
                        existing.push((attr.name, attr.value.to_string()));
                    }
                }
            }
        }
        self.dom
            .lock()
            .unwrap()
            .add_attrs_if_missing(*target, projected);
    }

    fn get_template_contents(&self, target: &NodeId) -> NodeId {
        *self
            .template_contents
            .lock()
            .unwrap()
            .get(target)
            .expect("get_template_contents called on non-template element")
    }

    fn is_mathml_annotation_xml_integration_point(&self, handle: &NodeId) -> bool {
        self.mathml_annotation_xml_ip
            .lock()
            .unwrap()
            .contains(handle)
    }

    fn remove_from_parent(&self, target: &NodeId) {
        self.dom.lock().unwrap().remove_from_parent(*target);
    }

    fn reparent_children(&self, node: &NodeId, new_parent: &NodeId) {
        self.dom
            .lock()
            .unwrap()
            .reparent_children(*node, *new_parent);
    }

    fn pop(&self, node: &NodeId) {
        // The tree builder popped this element from the open-element
        // stack: its subtree is complete as far as the builder knows.
        self.push_event(DomEvent::SubtreeComplete { node: *node });
    }

    // Defaults kept deliberately:
    // - mark_script_already_started: no-op (fine for a static tree).
    // - allow_declarative_shadow_roots: true (default);
    //   attach_declarative_shadow: false (default) — declarative
    //   shadow DOM is not built; M2 does not need it.
    // - maybe_clone_an_option_into_selectedcontent: default no-op —
    //   the `<selectedcontent>` clone is skipped, a known small
    //   deviation (see the html5lib-tests report: webkit02.dat).
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_sync<T: Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn sink_is_send_and_sync() {
        // The sink must stay shareable across Pluto's threads: the
        // parser thread owns the Parser, other threads read the Dom
        // through the Arc<Mutex<…>>. (The Parser/Tokenizer itself is
        // single-threaded by construction — Cell/RefCell inside
        // html5ever.)
        assert_sync::<PlutoSink>();
        assert_send::<PlutoSink>();
    }
}
