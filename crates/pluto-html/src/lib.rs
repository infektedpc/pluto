//! Parsing & DOM (layer 3): WHATWG HTML tree construction over a
//! slotmap arena DOM.
//!
//! # Pluto's own parser — no Servo-origin parser code
//!
//! The tree-construction machine itself — insertion modes, open-element
//! stack, active formatting elements, adoption agency, foster
//! parenting, implied end tags, quirks flag, raw-text/RCDATA handling —
//! is Pluto's own code in [`builder`], driven by Pluto's own
//! tokenizer in [`tokenizer`]. The DOM is the slotmap [`Dom`]; the
//! builder drives it directly through its mutators. The streaming
//! driver lives in [`streaming`] (feed / pause-at-script /
//! encoding-restart) and the preload scanner in [`scanner`].
//!
//! The DOM lives in a slotmap arena. Nodes are addressed by [`NodeId`]
//! handles, never `Rc<RefCell<T>>`, so there are no reference cycles and
//! the JS layer can hold a `NodeId` across a re-layout without keeping
//! the node itself alive. `Dom` is `Send + Sync` (there is a
//! compile-time assertion for this in `pluto-layout`'s tests — do not
//! break it: no `Rc`/`RefCell` in `Dom`, no `!Send` fields).
//!
//! # Parser privilege
//!
//! Parsing never fires mutation observers. The public mutators below
//! record a [`MutationRecord`] and mark Double Dirty Bit flags —
//! except while a parse is in progress, when the builder suppresses
//! recording (see [`Dom::set_suppress_records`]). The tree surgery is
//! the same single code path either way; only the recording wrapper is
//! skipped, per spec ("DOM mutation events must not fire for changes
//! caused by the UA parsing the document").
//!
//! # Entities
//!
//! Named character references come from the vendored WHATWG
//! `entities.json` (see `crates/pluto-html/entities.json`), generated
//! into a table at build time by `build.rs` ([`entities`]). No count
//! is asserted — published sources disagree on the total. Numeric
//! references map surrogates, `0x00`, and values `> U+10FFFF` to
//! U+FFFD (with the spec's C1 replacements for `0x80..=0x9F`).

//! # Script execution seams (Stage 7)
//!
//! The parser never executes JavaScript, but it owns the state
//! the event loop drives — see [`scripts`]:
//!
//! * [`scripts::classify_script`] routes every `<script>` to
//!   parser-blocking, defer, async, or module;
//! * [`scripts::ScriptQueues`] holds the defer queue (document
//!   order, drained after the parse, before `DOMContentLoaded`)
//!   and the async queue (arrival order), with module-URL dedup;
//! * [`scripts::StylesheetGate`] is the pending-stylesheet gate
//!   a parser-blocking script waits on — fed by the browser via
//!   [`Parser::note_stylesheet_loaded`] /
//!   [`Parser::note_stylesheet_failed`];
//! * [`scripts::DocumentWriter`] is the `document.write` /
//!   `open` / `close` state machine; the parser performs the
//!   character insertion through [`Parser::document_write`].
//!
//! The streaming [`Parser`] is the live seam: `feed` pauses at
//! parser-blocking scripts (`DomEvent::ScriptPaused`), queues
//! defer/async/module scripts, and tracks the stylesheet gate;
//! [`Parser::finish`] returns the queues in [`ParsedStream`].
//! Fragment parses ([`parse_fragment`]) create inert scripts —
//! they exist in the fragment DOM but must never execute
//! (`ParseOutput::scripts_inert`).

pub mod builder;
pub mod entities;
pub mod observers;
pub mod scanner;
pub mod scripts;
pub mod streaming;
pub mod tokenizer;

use std::collections::{HashMap, HashSet};

use builder::ParseOutput;
use slotmap::{SlotMap, new_key_type};

pub use builder::{
    DomEvent, NS_HTML, NS_MATHML, NS_SVG, NS_XLINK, NS_XML, NS_XMLNS, QName, SinkStats,
};
pub use observers::{
    MutationObserverOptions, MutationObserverRegistry, ObserverMutationRecord, ObserverRecordKind,
};
pub use scanner::{
    ConnectionHint, Destination, DiscoveredResource, HintKind, PictureSource, Priority,
    SrcsetCandidate, SrcsetDescriptor, eval_sizes, media_matches, parse_srcset, resolve_url, scan,
    scan_hints, scan_with_viewport, select_img_url, select_img_url_from_dom, select_srcset,
    source_type_supported,
};
pub use scripts::{
    AsyncQueue, CloseOutcome, DeferQueue, DocumentWriter, DynamicMarkupError, OpenOutcome,
    QueuedScript, ScriptKind, ScriptQueues, StylesheetGate, WriteOutcome, classify_script,
};
pub use streaming::{Parser, ParsedStream, ReferrerPolicy, ScriptPaused};

new_key_type! {
    /// Handle to a node in the [`Dom`] arena.
    pub struct NodeId;
}

/// Quirks mode of a document, decided by the DOCTYPE (or its absence)
/// during tree construction. Constrains selector matching (class/id
/// matching is ASCII-case-insensitive in quirks mode) and CSS box
/// behaviour later; almost nothing reads it yet in M2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuirksMode {
    /// Conforming DOCTYPE.
    #[default]
    NoQuirks,
    /// No DOCTYPE, or a non-conforming one.
    Quirks,
    /// Transitional DOCTYPEs (e.g. HTML 4.01 Transitional).
    LimitedQuirks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    Element { tag: String },
    Text(String),
    Comment(String),
    Doctype {
        name: String,
        public_id: String,
        system_id: String,
    },
    ProcessingInstruction {
        target: String,
        data: String,
    },
}

/// Double Dirty Bit flags on one node: `*_self` marks the node itself
/// dirty, `*_subtree` is the summary bit set on all ancestors of a
/// dirty node, so a traversal can skip entire clean subtrees. Style
/// and layout have separate pairs; the style/layout passes clear them
/// (see [`Dom::clear_dirty`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirtyFlags {
    pub style_self: bool,
    pub style_subtree: bool,
    pub layout_self: bool,
    pub layout_subtree: bool,
}

/// A node in the arena. Children are `NodeId`s; the parent link makes
/// upward walks (style matching, event bubbling) cheap. The node stays
/// small on purpose: auxiliary per-node data (computed style, selector
/// flags, …) belongs in a `SparseSecondaryMap` owned by the consumer,
/// not here.
///
/// `ns` is the element's namespace URI (Stage 15, M7 Phase 1, Rec 1):
/// `""` for the HTML namespace (the [`QName`] convention — HTML
/// elements carry an empty namespace), `"http://www.w3.org/2000/svg"`
/// for SVG, `"http://www.w3.org/1998/Math/MathML"` for MathML. The
/// parser sets it from tree construction (`builder::NS_SVG`, …);
/// `createElement` leaves it `""`, `createElementNS` sets it to the
/// namespace it was created with. Downstream (layout, paint, JS, the
/// preload scanner) must read this, never the tag spelling alone, to
/// tell `<svg><path>` from an unknown HTML `<path>`.
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Attributes in source order; duplicate names keep the first value
    /// (a *parse* rule — see [`Dom::set_attribute`], which replaces).
    pub attrs: Vec<(String, String)>,
    /// Namespace URI; `""` = HTML namespace.
    pub ns: String,
    pub dirty: DirtyFlags,
}

impl Node {
    /// Value of attribute `name`, or `None` if absent.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// Element tag name, or `None` for document/text nodes.
    pub fn tag(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Element { tag } => Some(tag.as_str()),
            _ => None,
        }
    }

    /// Namespace URI of an element (`""` = HTML namespace), or
    /// `None` for document/text nodes.
    pub fn namespace(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Element { .. } => Some(self.ns.as_str()),
            _ => None,
        }
    }

    /// Text content, or `None` for document/element nodes.
    pub fn text(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Text(data) => Some(data.as_str()),
            _ => None,
        }
    }

    /// Comment text, or `None` for non-comment nodes.
    pub fn comment(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Comment(data) => Some(data.as_str()),
            _ => None,
        }
    }
}

/// What changed, for one mutation. Style invalidation needs the
/// *before* state (an element that no longer matches a rule after the
/// change may still have matched it before — invalidating on the
/// post-change match alone misses it), so attribute/structure changes
/// carry a snapshot of the affected node's old class/id/attributes.
#[derive(Debug, Clone)]
pub struct MutationRecord {
    pub node: NodeId,
    pub kind: MutationKind,
    pub before: Option<BeforeSnapshot>,
}

/// The pre-mutation state of the affected node.
#[derive(Debug, Clone, Default)]
pub struct BeforeSnapshot {
    pub class: Option<String>,
    pub id: Option<String>,
    pub attrs: Vec<(String, String)>,
}

impl BeforeSnapshot {
    fn capture(dom: &Dom, node: NodeId) -> BeforeSnapshot {
        match dom.get(node) {
            Some(n) => BeforeSnapshot {
                class: n.attr("class").map(|s| s.to_string()),
                id: n.attr("id").map(|s| s.to_string()),
                attrs: n.attrs.clone(),
            },
            None => BeforeSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MutationKind {
    ChildInserted { parent: NodeId },
    ChildRemoved { parent: NodeId },
    ChildReplaced { parent: NodeId },
    ChildrenReparented { from: NodeId, to: NodeId },
    AttributeChanged { name: String },
    AttributeRemoved { name: String },
    CharacterDataChanged,
}

/// The explicit dirty set (not just bits scanned by walking the
/// tree): the nodes with `style_self` / `layout_self` set. Keeping the
/// set alongside the bits is what lets a Spineless Traversal swap in
/// later without changing every call site.
#[derive(Debug, Clone, Default)]
pub struct DirtySet {
    pub style: HashSet<NodeId>,
    pub layout: HashSet<NodeId>,
}

impl DirtySet {
    pub fn is_empty(&self) -> bool {
        self.style.is_empty() && self.layout.is_empty()
    }

    pub fn clear(&mut self) {
        self.style.clear();
        self.layout.clear();
    }
}

/// Live state of one form control (Stage 13 Rec 1, M5 forms).
///
/// The DOM is the single source of truth for control state: the
/// parser seeds the *defaults* from attributes / RCDATA text /
/// the `selected` rule, typing and script mutate the *live*
/// value, and `reset()` restores the live value from the seeded
/// defaults (never by re-parsing). The dirty flags are the
/// value/attribute split: once the user (or script, via the
/// `.value` IDL setter) has typed a value, `setAttribute('value',
/// …)` changes only the default — the typed value survives.
/// `Dom` is `Send + Sync` (see the module docs): this struct
/// holds only plain owned data, so that invariant is untouched.
#[derive(Debug, Clone, Default)]
pub struct FormControlState {
    /// Live value (input / textarea / select's selected option).
    pub value: String,
    /// Value the control had at parse time (the `value` attribute,
    /// textarea RCDATA text, or the selected option's value).
    pub default_value: String,
    /// True once the live value was set by typing or the IDL
    /// `.value` setter — after that, attribute writes only move
    /// the default.
    pub value_dirty: bool,
    /// Live checkedness (checkbox / radio).
    pub checked: bool,
    /// Checkedness at parse time (presence of `checked`).
    pub default_checked: bool,
    /// True once checkedness was set by click or the IDL
    /// `.checked` setter.
    pub checked_dirty: bool,
    /// Index into the select's options in tree order, `-1` when
    /// the select has no options.
    pub selected_index: i64,
    /// Selected index at parse time (last `selected` wins, else
    /// the first option).
    pub default_selected_index: i64,
    /// Caret offset in chars within `value` (text controls).
    pub caret: usize,
    /// Selection range within `value`, `[selection_start,
    /// selection_end)`; collapsed at the caret when equal.
    pub selection_start: usize,
    pub selection_end: usize,
}

/// The arena-allocated DOM tree for one document.
#[derive(Debug, Clone, Default)]
pub struct Dom {
    arena: SlotMap<NodeId, Node>,
    root: Option<NodeId>,
    quirks_mode: QuirksMode,
    mutations: Vec<MutationRecord>,
    dirty: DirtySet,
    /// Mutation-observer registrations + queued records (Stage 8
    /// Rec 7). The JS half drains these; the `Dom` only matches
    /// mutations against registrations and queues records.
    observers: MutationObserverRegistry,
    /// Parser privilege: while true, mutators still maintain
    /// parent/child invariants but skip mutation records and dirty
    /// bits. Set by the builder for the duration of a parse and cleared
    /// when the builder finishes; JS-driven mutations always
    /// record. This flag is internal state, not a public toggle.
    suppress_records: bool,
    /// Live form-control state, keyed by node. Seeded lazily on
    /// first access from the node's attributes / children (Stage
    /// 13 Rec 1), so controls created by the parser, by
    /// `createElement`, or by cloning all share one seeding path.
    form_states: HashMap<NodeId, FormControlState>,
}

impl Dom {
    /// Create an empty document with a single `Document` root node.
    pub fn new() -> Self {
        let mut dom = Self::default();
        let root = dom.arena.insert(Node {
            kind: NodeKind::Document,
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        });
        dom.root = Some(root);
        dom
    }

    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.arena.get(id)
    }

    /// Every live node id in the arena, including detached nodes
    /// (a `new Image()` element, or a node removed from the tree
    /// but still wrapped in JS, has no parent and no path from
    /// the root — a tree walk never reaches it).
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.arena.keys().collect()
    }

    /// Children of `id` in document order.
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.arena
            .get(id)
            .map(|n| n.children.as_slice())
            .unwrap_or(&[])
    }

    /// Tag name of an element node, or `None`.
    pub fn tag(&self, id: NodeId) -> Option<&str> {
        self.arena.get(id).and_then(Node::tag)
    }

    /// Namespace URI of an element node (`""` = HTML namespace),
    /// or `None` for non-elements. The parser fills this from tree
    /// construction; `createElement` leaves it `""`,
    /// `createElementNS` sets it (Stage 15, M7 Phase 1, Rec 1).
    pub fn namespace(&self, id: NodeId) -> Option<&str> {
        self.arena.get(id).and_then(Node::namespace)
    }

    /// True when `id` is an element in the SVG namespace.
    pub fn is_svg(&self, id: NodeId) -> bool {
        self.namespace(id)
            .is_some_and(|ns| ns == crate::builder::NS_SVG)
    }

    /// Text of a text node, or `None`.
    pub fn text(&self, id: NodeId) -> Option<&str> {
        self.arena.get(id).and_then(Node::text)
    }

    /// Attribute value on an element node, or `None`.
    pub fn attr(&self, id: NodeId, name: &str) -> Option<&str> {
        self.arena.get(id).and_then(|n| n.attr(name))
    }

    pub fn quirks_mode(&self) -> QuirksMode {
        self.quirks_mode
    }

    pub fn set_quirks_mode(&mut self, mode: QuirksMode) {
        self.quirks_mode = mode;
    }

    /// The `Doctype` child of the document root, if the document had
    /// one. (The doctype node itself is also a plain child of the
    /// root, in document order.)
    pub fn doctype(&self) -> Option<NodeId> {
        let root = self.root?;
        self.children(root).iter().copied().find(|&c| {
            matches!(self.get(c).map(|n| &n.kind), Some(NodeKind::Doctype { .. }))
        })
    }

    /// Mutation records since the last [`Dom::take_mutations`] /
    /// [`Dom::clear_mutations`]. Empty while a parse is in progress
    /// (parser privilege — see module docs).
    pub fn mutations(&self) -> &[MutationRecord] {
        &self.mutations
    }

    pub fn take_mutations(&mut self) -> Vec<MutationRecord> {
        std::mem::take(&mut self.mutations)
    }

    pub fn clear_mutations(&mut self) {
        self.mutations.clear();
    }

    // --- MutationObserver state (Stage 8 Rec 7) ---

    /// Register `observer_id` on `target` (replaces an existing
    /// registration for the same observer + target). Returns false
    /// when `target` is not a live node.
    pub fn observe_mutation(
        &mut self,
        observer_id: u64,
        target: NodeId,
        options: MutationObserverOptions,
    ) -> bool {
        if self.arena.get(target).is_none() {
            return false;
        }
        self.observers.observe(observer_id, target, options);
        true
    }

    /// Drop `observer_id` entirely: registrations and any queued
    /// records are discarded (a later `observe()` starts fresh).
    pub fn disconnect_mutation_observer(&mut self, observer_id: u64) {
        self.observers.disconnect(observer_id);
    }

    /// Drain one observer's queued records (delivery, or
    /// `takeRecords()`).
    pub fn take_mutation_records(
        &mut self,
        observer_id: u64,
    ) -> Vec<ObserverMutationRecord> {
        self.observers.take_records(observer_id)
    }

    /// Observers with undelivered records, in first-record order.
    pub fn pending_mutation_observers(&self) -> Vec<u64> {
        self.observers.pending_observers()
    }

    /// Take the pending-observer list, clearing it.
    pub fn take_pending_mutation_observers(&mut self) -> Vec<u64> {
        self.observers.take_pending_observers()
    }

    /// True when `node` is `ancestor` or a descendant of it.
    pub fn is_descendant_or_self(&self, node: NodeId, ancestor: NodeId) -> bool {
        let mut cur = Some(node);
        while let Some(id) = cur {
            if id == ancestor {
                return true;
            }
            cur = self.arena.get(id).and_then(|n| n.parent);
        }
        false
    }

    /// (previous, next) siblings of `id` in its current parent.
    fn sibling_pair(&self, id: NodeId) -> (Option<NodeId>, Option<NodeId>) {
        let Some(parent) = self.arena.get(id).and_then(|n| n.parent) else {
            return (None, None);
        };
        let kids = self.children(parent);
        let Some(pos) = kids.iter().position(|&c| c == id) else {
            return (None, None);
        };
        (
            if pos > 0 { Some(kids[pos - 1]) } else { None },
            kids.get(pos + 1).copied(),
        )
    }

    /// Match `record` against every registered observer and queue
    /// it for the ones whose options cover this mutation on this
    /// node (directly, or via `subtree` on an ancestor target).
    /// Skipped while a parse is in progress (parser privilege —
    /// parsing never fires observers). `old_value` is stripped
    /// unless the observer asked for it.
    fn notify_observers(&mut self, record: ObserverMutationRecord) {
        if self.suppress_records {
            return;
        }
        for (observer_id, target, options) in self.observers.registrations() {
            let in_scope = record.target == target
                || (options.subtree && self.is_descendant_or_self(record.target, target));
            if !in_scope {
                continue;
            }
            let wanted = match record.kind {
                ObserverRecordKind::ChildList => options.child_list,
                ObserverRecordKind::Attributes => {
                    if !options.wants_attributes() {
                        false
                    } else if let Some(filter) = &options.attribute_filter {
                        record
                            .attribute_name
                            .as_ref()
                            .is_some_and(|n| filter.iter().any(|f| f == n))
                    } else {
                        true
                    }
                }
                ObserverRecordKind::CharacterData => options.character_data,
            };
            if !wanted {
                continue;
            }
            let mut rec = record.clone();
            match rec.kind {
                ObserverRecordKind::Attributes if !options.attribute_old_value => {
                    rec.old_value = None
                }
                ObserverRecordKind::CharacterData
                    if !options.character_data_old_value =>
                {
                    rec.old_value = None
                }
                _ => {}
            }
            self.observers.queue(observer_id, rec);
        }
    }

    pub fn dirty_set(&self) -> &DirtySet {
        &self.dirty
    }

    /// Clear every dirty flag on every node and empty the dirty set.
    /// Called by the style/layout passes after they have consumed the
    /// dirty nodes.
    pub fn clear_dirty(&mut self) {
        for (_, node) in self.arena.iter_mut() {
            node.dirty = DirtyFlags::default();
        }
        self.dirty.clear();
    }

    /// Mark one node dirty (style + layout, self + subtree summary
    /// bits) without recording a mutation. The integration wave uses
    /// this when a JS mutation runs while the parser's
    /// `suppress_records` privilege is on (parser-blocking scripts
    /// mutate during a parse, and `note_mutation` silently skips
    /// them then): the dirty bits must still be set, or a mutation
    /// made mid-parse would never reach the layout pass.
    pub fn mark_dirty(&mut self, id: NodeId) {
        if let Some(n) = self.arena.get_mut(id) {
            n.dirty.style_self = true;
            n.dirty.layout_self = true;
        }
        self.dirty.style.insert(id);
        self.dirty.layout.insert(id);
        let mut cur = self.arena.get(id).and_then(|n| n.parent);
        while let Some(parent) = cur {
            if let Some(n) = self.arena.get_mut(parent) {
                n.dirty.style_subtree = true;
                n.dirty.layout_subtree = true;
            }
            cur = self.arena.get(parent).and_then(|n| n.parent);
        }
    }

    pub(crate) fn set_suppress_records(&mut self, suppress: bool) {
        self.suppress_records = suppress;
    }

    // --- node creation (allocation only; no tree mutation, no record) ---

    pub fn create_element(&mut self, tag: &str) -> NodeId {
        self.create_element_with_attrs(tag, Vec::new())
    }

    /// Like [`create_element`](Self::create_element), but the
    /// element is created in namespace `ns` — the
    /// `document.createElementNS(namespace, tag)` seam (Stage 15,
    /// M7 Phase 1, Rec 1). `""` is the HTML namespace, matching
    /// the parser's [`QName`] convention.
    pub fn create_element_ns(&mut self, ns: &str, tag: &str) -> NodeId {
        self.create_element_with_attrs_ns(tag, Vec::new(), ns)
    }

    pub fn create_element_with_attrs(
        &mut self,
        tag: &str,
        attrs: Vec<(String, String)>,
    ) -> NodeId {
        self.create_element_with_attrs_ns(tag, attrs, "")
    }

    /// Like [`create_element_with_attrs`](Self::create_element_with_attrs),
    /// with an explicit namespace (the builder's path: the parser
    /// records the namespace tree construction assigned).
    pub fn create_element_with_attrs_ns(
        &mut self,
        tag: &str,
        attrs: Vec<(String, String)>,
        ns: &str,
    ) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Element {
                tag: tag.to_string(),
            },
            parent: None,
            children: Vec::new(),
            attrs,
            ns: ns.to_string(),
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_text(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Text(data.to_string()),
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_comment(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Comment(data.to_string()),
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_doctype(&mut self, name: &str, public_id: &str, system_id: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Doctype {
                name: name.to_string(),
                public_id: public_id.to_string(),
                system_id: system_id.to_string(),
            },
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_pi(&mut self, target: &str, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::ProcessingInstruction {
                target: target.to_string(),
                data: data.to_string(),
            },
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        })
    }

    /// A bare `Document` node with no parent — used as a `<template>`
    /// contents holder. It is not the document root.
    pub fn create_document(&mut self) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Document,
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            ns: String::new(),
            dirty: DirtyFlags::default(),
        })
    }

    // --- invariant-maintaining core (one path; all mutators go through it) ---

    /// Detach `child` from its current parent, if any. Returns the old
    /// parent.
    fn detach(&mut self, child: NodeId) -> Option<NodeId> {
        let parent = self.arena.get(child).and_then(|n| n.parent);
        if let Some(parent) = parent {
            if let Some(p) = self.arena.get_mut(parent) {
                p.children.retain(|&c| c != child);
            }
            if let Some(c) = self.arena.get_mut(child) {
                c.parent = None;
            }
        }
        parent
    }

    /// Link `child` under `parent` at `pos` (`None` = append). Detaches
    /// from any previous parent first, so a node is never listed under
    /// two parents (adoption-agency moves depend on this).
    fn link_at(&mut self, parent: NodeId, child: NodeId, pos: Option<usize>) {
        self.detach(child);
        if let Some(node) = self.arena.get_mut(child) {
            node.parent = Some(parent);
        }
        if let Some(node) = self.arena.get_mut(parent) {
            match pos {
                Some(p) if p <= node.children.len() => node.children.insert(p, child),
                _ => node.children.push(child),
            }
        }
    }

    /// Record a mutation and mark Double Dirty Bit flags — unless
    /// recording is suppressed for a parse in progress. M2 marks both
    /// the style and the layout pair on every mutation (the M2
    /// invalidation rule is "full restyle on any mutation"); refining
    /// which pair a mutation kind touches is Stage 7 work.
    fn note_mutation(&mut self, node: NodeId, kind: MutationKind, before: Option<BeforeSnapshot>) {
        if self.suppress_records {
            return;
        }
        self.mutations.push(MutationRecord { node, kind, before });
        if let Some(n) = self.arena.get_mut(node) {
            n.dirty.style_self = true;
            n.dirty.layout_self = true;
        }
        self.dirty.style.insert(node);
        self.dirty.layout.insert(node);
        let mut cur = self.arena.get(node).and_then(|n| n.parent);
        while let Some(id) = cur {
            if let Some(n) = self.arena.get_mut(id) {
                n.dirty.style_subtree = true;
                n.dirty.layout_subtree = true;
            }
            cur = self.arena.get(id).and_then(|n| n.parent);
        }
    }

    // --- public mutators (the same path the parser and JS both use) ---

    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        let before = BeforeSnapshot::capture(self, child);
        let previous_sibling = self.children(parent).last().copied();
        self.link_at(parent, child, None);
        self.note_mutation(child, MutationKind::ChildInserted { parent }, Some(before));
        self.notify_observers(ObserverMutationRecord::child_list(
            parent,
            vec![child],
            vec![],
            previous_sibling,
            None,
        ));
    }

    /// Insert `child` as a child of `parent` immediately before
    /// `sibling`. If `sibling` is not a child of `parent`, appends.
    pub fn insert_before(&mut self, parent: NodeId, sibling: NodeId, child: NodeId) {
        let before = BeforeSnapshot::capture(self, child);
        let pos = self
            .arena
            .get(parent)
            .and_then(|n| n.children.iter().position(|&c| c == sibling));
        let previous_sibling = pos.and_then(|p| {
            if p > 0 {
                self.children(parent).get(p - 1).copied()
            } else {
                None
            }
        });
        self.link_at(parent, child, pos);
        self.note_mutation(child, MutationKind::ChildInserted { parent }, Some(before));
        self.notify_observers(ObserverMutationRecord::child_list(
            parent,
            vec![child],
            vec![],
            previous_sibling,
            Some(sibling),
        ));
    }

    /// Detach `child` from `parent`. (Lenient: detaches wherever the
    /// child currently lives.)
    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) {
        let before = BeforeSnapshot::capture(self, child);
        let (previous_sibling, next_sibling) = self.sibling_pair(child);
        self.detach(child);
        self.note_mutation(child, MutationKind::ChildRemoved { parent }, Some(before));
        self.notify_observers(ObserverMutationRecord::child_list(
            parent,
            vec![],
            vec![child],
            previous_sibling,
            next_sibling,
        ));
    }

    /// Detach `target` from whatever parent it has, if any.
    pub fn remove_from_parent(&mut self, target: NodeId) {
        let before = BeforeSnapshot::capture(self, target);
        let (previous_sibling, next_sibling) = self.sibling_pair(target);
        if let Some(parent) = self.detach(target) {
            self.note_mutation(target, MutationKind::ChildRemoved { parent }, Some(before));
            self.notify_observers(ObserverMutationRecord::child_list(
                parent,
                vec![],
                vec![target],
                previous_sibling,
                next_sibling,
            ));
        }
    }

    /// Replace `old` with `new` under `parent`, keeping `old`'s
    /// position among its siblings.
    pub fn replace_child(&mut self, parent: NodeId, old: NodeId, new: NodeId) {
        let before = BeforeSnapshot::capture(self, old);
        let pos = self
            .arena
            .get(parent)
            .and_then(|n| n.children.iter().position(|&c| c == old));
        let (previous_sibling, next_sibling) = self.sibling_pair(old);
        self.detach(old);
        self.link_at(parent, new, pos);
        self.note_mutation(new, MutationKind::ChildReplaced { parent }, Some(before));
        self.notify_observers(ObserverMutationRecord::child_list(
            parent,
            vec![new],
            vec![old],
            previous_sibling,
            next_sibling,
        ));
    }

    /// Move all children of `node` to `new_parent`, in order. (The
    /// bulk move the adoption agency, foster parenting, and fragment
    /// re-parenting need.)
    pub fn reparent_children(&mut self, node: NodeId, new_parent: NodeId) {
        if node == new_parent {
            return;
        }
        let kids: Vec<NodeId> = self.children(node).to_vec();
        for kid in kids {
            let (previous_sibling, next_sibling) = self.sibling_pair(kid);
            let previous_in_new = self.children(new_parent).last().copied();
            self.link_at(new_parent, kid, None);
            self.notify_observers(ObserverMutationRecord::child_list(
                node,
                vec![],
                vec![kid],
                previous_sibling,
                next_sibling,
            ));
            self.notify_observers(ObserverMutationRecord::child_list(
                new_parent,
                vec![kid],
                vec![],
                previous_in_new,
                None,
            ));
        }
        self.note_mutation(
            new_parent,
            MutationKind::ChildrenReparented { from: node, to: new_parent },
            None,
        );
    }

    /// Set an attribute, replacing any existing value. (First-
    /// duplicate-wins is a *parse* rule, enforced by the tokenizer's
    /// attribute list; the DOM API replaces.)
    pub fn set_attribute(&mut self, id: NodeId, name: &str, value: &str) {
        let before = BeforeSnapshot::capture(self, id);
        let old_value = self.attr(id, name).map(|v| v.to_string());
        if let Some(node) = self.arena.get_mut(id) {
            match node.attrs.iter_mut().find(|(n, _)| n == name) {
                Some((_, v)) => *v = value.to_string(),
                None => node.attrs.push((name.to_string(), value.to_string())),
            }
        }
        self.note_mutation(
            id,
            MutationKind::AttributeChanged { name: name.to_string() },
            Some(before),
        );
        self.notify_observers(ObserverMutationRecord::attribute(id, name, old_value));
        // Stage 13 Rec 1: an attribute write moves the control's
        // seeded default (and the live value only while clean).
        self.sync_form_default_on_attribute(id, name, true);
    }

    pub fn remove_attribute(&mut self, id: NodeId, name: &str) {
        let before = BeforeSnapshot::capture(self, id);
        let old_value = self.attr(id, name).map(|v| v.to_string());
        let mut removed = false;
        if let Some(node) = self.arena.get_mut(id) {
            let before_len = node.attrs.len();
            node.attrs.retain(|(n, _)| n != name);
            removed = node.attrs.len() != before_len;
        }
        if removed {
            self.note_mutation(
                id,
                MutationKind::AttributeRemoved { name: name.to_string() },
                Some(before),
            );
            self.notify_observers(ObserverMutationRecord::attribute(id, name, old_value));
            // Stage 13 Rec 1: removing the attribute clears the
            // seeded default (live value follows only while clean).
            self.sync_form_default_on_attribute(id, name, false);
        }
    }

    /// Add attributes that are not already present (by local name).
    /// Used by the tree builder when a second `<html>` / `<body>` tag
    /// carries new attributes.
    pub fn add_attrs_if_missing(&mut self, target: NodeId, attrs: Vec<(String, String)>) {
        let before = BeforeSnapshot::capture(self, target);
        let mut added: Vec<String> = Vec::new();
        if let Some(node) = self.arena.get_mut(target) {
            for (name, value) in attrs {
                if !node.attrs.iter().any(|(n, _)| *n == name) {
                    added.push(name.clone());
                    node.attrs.push((name, value));
                }
            }
        }
        if let Some(name) = added.first().cloned() {
            self.note_mutation(target, MutationKind::AttributeChanged { name }, Some(before));
        }
        for name in added {
            self.notify_observers(ObserverMutationRecord::attribute(target, &name, None));
        }
    }

    /// Replace a text node's data.
    pub fn set_text(&mut self, id: NodeId, data: &str) {
        let before = BeforeSnapshot::capture(self, id);
        let old_value = self.text(id).map(|v| v.to_string());
        if let Some(Node { kind: NodeKind::Text(contents), .. }) = self.arena.get_mut(id) {
            *contents = data.to_string();
        }
        self.note_mutation(id, MutationKind::CharacterDataChanged, Some(before));
        self.notify_observers(ObserverMutationRecord::character_data(id, old_value));
    }

    /// Append to an existing text node (adjacent-text merging: the
    /// builder hands the tree text in chunks).
    pub fn append_to_text(&mut self, id: NodeId, data: &str) {
        let before = BeforeSnapshot::capture(self, id);
        let old_value = self.text(id).map(|v| v.to_string());
        if let Some(Node { kind: NodeKind::Text(contents), .. }) = self.arena.get_mut(id) {
            contents.push_str(data);
        }
        self.note_mutation(id, MutationKind::CharacterDataChanged, Some(before));
        self.notify_observers(ObserverMutationRecord::character_data(id, old_value));
    }
    // ---- form-control state (Stage 13 Rec 1) ----

    /// Whether `id` is a form control carrying live state
    /// (`input`, `textarea`, `select`).
    pub fn is_form_control(&self, id: NodeId) -> bool {
        matches!(self.tag(id), Some("input" | "textarea" | "select"))
    }

    /// The `type` attribute of an `<input>`, lowercased, defaulting
    /// to `"text"` (missing or empty `type` is `text` per HTML).
    pub fn input_type(&self, id: NodeId) -> String {
        self.attr(id, "type")
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "text".to_string())
    }

    /// Whether the control is disabled: its own `disabled`
    /// attribute, or a `disabled` `<fieldset>` ancestor.
    pub fn is_control_disabled(&self, id: NodeId) -> bool {
        if self.attr(id, "disabled").is_some() {
            return true;
        }
        let mut cur = self.get(id).and_then(|n| n.parent);
        while let Some(n) = cur {
            if self.tag(n) == Some("fieldset") && self.attr(n, "disabled").is_some() {
                return true;
            }
            cur = self.get(n).and_then(|n| n.parent);
        }
        false
    }

    /// The concatenated descendant text of `id` (element text
    /// content, no markup) — the textarea RCDATA default and the
    /// `<option>` value fallback.
    pub fn descendant_text(&self, id: NodeId) -> String {
        let mut out = String::new();
        let mut stack: Vec<NodeId> = self.children(id).to_vec();
        stack.reverse();
        while let Some(n) = stack.pop() {
            match self.get(n).map(|node| &node.kind) {
                Some(NodeKind::Text(t)) => out.push_str(t),
                Some(NodeKind::Element { .. }) => {
                    let mut kids = self.children(n).to_vec();
                    kids.reverse();
                    stack.extend(kids);
                }
                _ => {}
            }
        }
        out
    }

    /// An `<option>`'s value: its `value` attribute, else its
    /// text with whitespace stripped and collapsed (HTML §4.10.9).
    pub fn option_value(&self, id: NodeId) -> String {
        if let Some(v) = self.attr(id, "value") {
            return v.to_string();
        }
        self.descendant_text(id)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A `<select>`'s `<option>` descendants in tree order
    /// (including options inside `<optgroup>`).
    pub fn select_options(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack: Vec<NodeId> = self.children(id).to_vec();
        stack.reverse();
        while let Some(n) = stack.pop() {
            match self.tag(n) {
                Some("option") => out.push(n),
                Some(_) => {
                    let mut kids = self.children(n).to_vec();
                    kids.reverse();
                    stack.extend(kids);
                }
                None => {}
            }
        }
        out
    }

    /// The seeded state for a control, computed purely from its
    /// attributes / children — the same rule the parser's
    /// last-`selected`-wins resolution uses (`builder.rs`), reused
    /// here so `reset()` restores from seeded defaults without
    /// re-parsing. `None` for non-controls / stale keys.
    fn compute_form_defaults(&self, id: NodeId) -> Option<FormControlState> {
        let tag = self.tag(id)?;
        match tag {
            "input" => {
                let ty = self.input_type(id);
                let default_value = match ty.as_str() {
                    "checkbox" | "radio" => {
                        self.attr(id, "value").unwrap_or("on").to_string()
                    }
                    _ => self.attr(id, "value").unwrap_or("").to_string(),
                };
                let default_checked = self.attr(id, "checked").is_some();
                let len = default_value.chars().count();
                Some(FormControlState {
                    value: default_value.clone(),
                    default_value,
                    value_dirty: false,
                    checked: default_checked,
                    default_checked,
                    checked_dirty: false,
                    selected_index: -1,
                    default_selected_index: -1,
                    caret: len,
                    selection_start: len,
                    selection_end: len,
                })
            }
            "textarea" => {
                let default_value = self.descendant_text(id);
                let len = default_value.chars().count();
                Some(FormControlState {
                    value: default_value.clone(),
                    default_value,
                    value_dirty: false,
                    checked: false,
                    default_checked: false,
                    checked_dirty: false,
                    selected_index: -1,
                    default_selected_index: -1,
                    caret: len,
                    selection_start: len,
                    selection_end: len,
                })
            }
            "select" => {
                let options = self.select_options(id);
                // Last `selected` wins, else the first option (the
                // builder's `update_selectedcontent` rule); no
                // options means -1.
                let default_selected_index = if options.is_empty() {
                    -1
                } else {
                    options
                        .iter()
                        .rposition(|&o| self.attr(o, "selected").is_some())
                        .map(|i| i as i64)
                        .unwrap_or(0)
                };
                let default_value = if default_selected_index >= 0 {
                    self.option_value(options[default_selected_index as usize])
                } else {
                    String::new()
                };
                Some(FormControlState {
                    value: default_value.clone(),
                    default_value,
                    value_dirty: false,
                    checked: false,
                    default_checked: false,
                    checked_dirty: false,
                    selected_index: default_selected_index,
                    default_selected_index,
                    caret: 0,
                    selection_start: 0,
                    selection_end: 0,
                })
            }
            _ => None,
        }
    }

    /// Live state for a control, seeding it from the defaults on
    /// first access. `None` for non-controls / stale keys.
    pub fn form_state_mut(&mut self, id: NodeId) -> Option<&mut FormControlState> {
        if !self.is_form_control(id) {
            return None;
        }
        if !self.form_states.contains_key(&id) {
            if let Some(seed) = self.compute_form_defaults(id) {
                self.form_states.insert(id, seed);
            }
        }
        self.form_states.get_mut(&id)
    }

    /// Live state for a control, if already seeded. Prefer the
    /// `form_*` accessors below, which fall back to the computed
    /// defaults without mutating.
    pub fn form_state(&self, id: NodeId) -> Option<&FormControlState> {
        self.form_states.get(&id)
    }

    /// Live value of a control (seeded default when never touched).
    pub fn form_value(&self, id: NodeId) -> Option<String> {
        if let Some(s) = self.form_states.get(&id) {
            return Some(s.value.clone());
        }
        self.compute_form_defaults(id).map(|s| s.value)
    }

    /// Default (parse-time) value of a control.
    pub fn form_default_value(&self, id: NodeId) -> Option<String> {
        if let Some(s) = self.form_states.get(&id) {
            return Some(s.default_value.clone());
        }
        self.compute_form_defaults(id).map(|s| s.default_value)
    }

    /// Whether the control's live value differs from its default
    /// by user/script action (the dirty-value flag).
    pub fn form_value_dirty(&self, id: NodeId) -> bool {
        self.form_states
            .get(&id)
            .map(|s| s.value_dirty)
            .unwrap_or(false)
    }

    /// Live checkedness of a checkbox / radio.
    pub fn form_checked(&self, id: NodeId) -> Option<bool> {
        if let Some(s) = self.form_states.get(&id) {
            return Some(s.checked);
        }
        self.compute_form_defaults(id).map(|s| s.checked)
    }

    /// Live selected index of a `<select>` (`-1` = no options).
    pub fn form_selected_index(&self, id: NodeId) -> Option<i64> {
        if let Some(s) = self.form_states.get(&id) {
            return Some(s.selected_index);
        }
        self.compute_form_defaults(id)
            .map(|s| s.selected_index)
    }

    /// Caret offset (chars) of a text control.
    pub fn form_caret(&self, id: NodeId) -> Option<usize> {
        if let Some(s) = self.form_states.get(&id) {
            return Some(s.caret);
        }
        self.compute_form_defaults(id).map(|s| s.caret)
    }

    /// Selection range `[start, end)` of a text control.
    pub fn form_selection(&self, id: NodeId) -> Option<(usize, usize)> {
        if let Some(s) = self.form_states.get(&id) {
            return Some((s.selection_start, s.selection_end));
        }
        self.compute_form_defaults(id)
            .map(|s| (s.selection_start, s.selection_end))
    }

    /// Set the live value (typing or the IDL `.value` setter):
    /// marks the value dirty and collapses the selection to the
    /// end of the new value.
    pub fn set_form_value(&mut self, id: NodeId, value: &str) -> bool {
        let Some(state) = self.form_state_mut(id) else {
            return false;
        };
        state.value = value.to_string();
        state.value_dirty = true;
        let len = value.chars().count();
        state.caret = len;
        state.selection_start = len;
        state.selection_end = len;
        true
    }

    /// Set the live checkedness (click or the IDL `.checked`
    /// setter): marks the checked flag dirty.
    pub fn set_form_checked(&mut self, id: NodeId, checked: bool) -> bool {
        let Some(state) = self.form_state_mut(id) else {
            return false;
        };
        state.checked = checked;
        state.checked_dirty = true;
        true
    }

    /// Set the selected index of a `<select>` (clamped to the
    /// option range; `-1` only when there are no options) and
    /// sync the live value to the selected option's value.
    pub fn set_form_selected_index(&mut self, id: NodeId, index: i64) -> bool {
        if self.tag(id) != Some("select") {
            return false;
        }
        let options = self.select_options(id);
        // `-1` deselects (the option's `selected = false` path);
        // any other negative index clamps to the first option.
        let clamped = if index < 0 || options.is_empty() {
            -1
        } else {
            index.clamp(0, options.len() as i64 - 1)
        };
        let value = if clamped >= 0 {
            self.option_value(options[clamped as usize])
        } else {
            String::new()
        };
        let Some(state) = self.form_state_mut(id) else {
            return false;
        };
        state.selected_index = clamped;
        state.value = value;
        true
    }

    /// Set the caret / selection of a text control (clamped to
    /// the value length, in chars).
    pub fn set_form_selection(&mut self, id: NodeId, start: usize, end: usize) -> bool {
        let len = self
            .form_value(id)
            .map(|v| v.chars().count())
            .unwrap_or(0);
        let Some(state) = self.form_state_mut(id) else {
            return false;
        };
        let (start, end) = (start.min(len), end.min(len));
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        state.selection_start = start;
        state.selection_end = end;
        state.caret = end;
        true
    }

    /// Insert `text` at the caret (replacing any selection) and
    /// return the new caret offset — the text-editing primitive
    /// the renderer drives from dispatched keys (Stage 13 Rec 4).
    /// Honours `maxlength` (the attribute caps the value length).
    pub fn insert_form_text(&mut self, id: NodeId, text: &str) -> Option<usize> {
        let max = self
            .attr(id, "maxlength")
            .and_then(|m| m.parse::<usize>().ok());
        let (value, start, end) = {
            let state = self.form_state_mut(id)?;
            (state.value.clone(), state.selection_start, state.selection_end)
        };
        let mut chars: Vec<char> = value.chars().collect();
        let (start, end) = (start.min(chars.len()), end.min(chars.len()));
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        chars.splice(start..end, text.chars());
        if let Some(max) = max {
            chars.truncate(max);
        }
        let new_value: String = chars.iter().collect();
        let caret = (start + text.chars().count()).min(new_value.chars().count());
        let state = self.form_state_mut(id)?;
        state.value = new_value;
        state.value_dirty = true;
        state.caret = caret;
        state.selection_start = caret;
        state.selection_end = caret;
        Some(caret)
    }

    /// Delete the selection, or the char before the caret
    /// (`backspace`) / after it (`delete`). Returns the new caret.
    pub fn delete_form_text(&mut self, id: NodeId, backspace: bool) -> Option<usize> {
        let (value, start, end) = {
            let state = self.form_state_mut(id)?;
            (
                state.value.clone(),
                state.selection_start,
                state.selection_end,
            )
        };
        let mut chars: Vec<char> = value.chars().collect();
        let caret = if start != end {
            let (s, e) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            chars.drain(s.min(chars.len())..e.min(chars.len()));
            s.min(chars.len())
        } else if backspace {
            if start == 0 {
                return Some(0);
            }
            chars.remove(start - 1);
            start - 1
        } else {
            if start >= chars.len() {
                return Some(chars.len());
            }
            chars.remove(start);
            start
        };
        let new_value: String = chars.iter().collect();
        let state = self.form_state_mut(id)?;
        state.value = new_value;
        state.value_dirty = true;
        state.caret = caret;
        state.selection_start = caret;
        state.selection_end = caret;
        Some(caret)
    }

    /// The form that owns `id`: the nearest `<form>` ancestor, or
    /// the `<form>` named by the control's `form` attribute (by
    /// `id`, document order). `None` for controls outside any form.
    pub fn form_owner(&self, id: NodeId) -> Option<NodeId> {
        if let Some(form_id) = self.attr(id, "form") {
            let form_id = form_id.to_string();
            if let Some(root) = self.root() {
                let mut stack = vec![root];
                while let Some(n) = stack.pop() {
                    if self.tag(n) == Some("form")
                        && self.attr(n, "id") == Some(form_id.as_str())
                    {
                        return Some(n);
                    }
                    stack.extend(self.children(n).iter().rev());
                }
            }
        }
        let mut cur = self.get(id).and_then(|n| n.parent);
        while let Some(n) = cur {
            if self.tag(n) == Some("form") {
                return Some(n);
            }
            cur = self.get(n).and_then(|n| n.parent);
        }
        None
    }

    /// All form controls owned by `form_id`, in document order.
    pub fn form_controls(&self, form_id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        // Pre-order walk of the form subtree.
        let mut order: Vec<NodeId> = Vec::new();
        let mut stack: Vec<NodeId> = vec![form_id];
        while let Some(n) = stack.pop() {
            order.push(n);
            let mut kids = self.children(n).to_vec();
            kids.reverse();
            stack.extend(kids);
        }
        for n in order {
            // `form.elements` includes buttons too (they carry no
            // live value state, but they are listed controls).
            let listed = self.is_form_control(n) || self.tag(n) == Some("button");
            if n != form_id && listed && self.form_owner(n) == Some(form_id) {
                out.push(n);
            }
        }
        // Controls outside the subtree whose `form` attribute names
        // this form: scan the whole document in order.
        if let Some(form_id_attr) = self.attr(form_id, "id").map(|s| s.to_string()) {
            if let Some(root) = self.root() {
                let mut all: Vec<NodeId> = Vec::new();
                let mut s = vec![root];
                while let Some(n) = s.pop() {
                    all.push(n);
                    let mut kids = self.children(n).to_vec();
                    kids.reverse();
                    s.extend(kids);
                }
                for n in all {
                    let listed = self.is_form_control(n) || self.tag(n) == Some("button");
                    if listed
                        && self.attr(n, "form") == Some(form_id_attr.as_str())
                        && !out.contains(&n)
                    {
                        out.push(n);
                    }
                }
            }
        }
        out
    }

    /// Reset every control of a form to its seeded default and
    /// clear the dirty flags (Stage 13 Rec 1): `reset()` restores
    /// from the seeded defaults, never by re-parsing.
    pub fn reset_form(&mut self, form_id: NodeId) {
        for control in self.form_controls(form_id) {
            if let Some(seed) = self.compute_form_defaults(control) {
                self.form_states.insert(control, seed);
            }
        }
    }

    /// The submission entry list for a form (Stage 13 Rec 6):
    /// `(name, value)` for successful controls only — a control
    /// needs a non-empty `name`, must not be disabled, and
    /// unchecked checkboxes / radios are excluded; `type=submit`
    /// / `image` / `button` / `reset` / `file` never submit (the
    /// submitter itself contributes its name/value when it is a
    /// submit button with a name). Selects contribute the selected
    /// option's value.
    pub fn form_entries(
        &self,
        form_id: NodeId,
        submitter: Option<NodeId>,
    ) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for control in self.form_controls(form_id) {
            let Some(name) = self.attr(control, "name") else {
                continue;
            };
            if name.is_empty() || self.is_control_disabled(control) {
                continue;
            }
            match self.tag(control) {
                Some("input") => {
                    let ty = self.input_type(control);
                    match ty.as_str() {
                        "checkbox" | "radio" => {
                            if self.form_checked(control).unwrap_or(false) {
                                out.push((
                                    name.to_string(),
                                    self.form_value(control).unwrap_or_default(),
                                ));
                            }
                        }
                        "submit" | "image" | "button" | "reset" | "file" => {
                            if Some(control) == submitter
                                && (ty == "submit" || ty == "image")
                            {
                                out.push((
                                    name.to_string(),
                                    self.form_value(control).unwrap_or_default(),
                                ));
                            }
                        }
                        _ => {
                            out.push((
                                name.to_string(),
                                self.form_value(control).unwrap_or_default(),
                            ));
                        }
                    }
                }
                Some("textarea") => {
                    out.push((
                        name.to_string(),
                        self.form_value(control).unwrap_or_default(),
                    ));
                }
                Some("select") => {
                    if let Some(idx) = self.form_selected_index(control) {
                        if idx >= 0 {
                            out.push((
                                name.to_string(),
                                self.form_value(control).unwrap_or_default(),
                            ));
                        }
                    }
                }
                Some("button") => {
                    if Some(control) == submitter {
                        let ty = self
                            .attr(control, "type")
                            .map(|t| t.to_ascii_lowercase())
                            .unwrap_or_else(|| "submit".to_string());
                        if ty == "submit" {
                            out.push((
                                name.to_string(),
                                self.attr(control, "value").unwrap_or("").to_string(),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Keep a control's seeded defaults in sync when a content
    /// attribute changes (Stage 13 Rec 1/5): `setAttribute('value',
    /// …)` moves the *default* and — only while the value is not
    /// dirty — the live value; `checked` / `selected` behave the
    /// same for their flags. Called from `set_attribute` /
    /// `remove_attribute` after the attribute write.
    fn sync_form_default_on_attribute(&mut self, id: NodeId, name: &str, present: bool) {
        if self.tag(id) == Some("option") {
            // A selected-attribute change re-resolves the owning
            // select's default + live index (last-selected-wins).
            if name == "selected" {
                let mut cur = self.get(id).and_then(|n| n.parent);
                while let Some(n) = cur {
                    if self.tag(n) == Some("select") {
                        let options = self.select_options(n);
                        let idx = if options.is_empty() {
                            -1
                        } else {
                            options
                                .iter()
                                .rposition(|&o| self.attr(o, "selected").is_some())
                                .map(|i| i as i64)
                                .unwrap_or(0)
                        };
                        let value = if idx >= 0 {
                            self.option_value(options[idx as usize])
                        } else {
                            String::new()
                        };
                        if let Some(state) = self.form_states.get_mut(&n) {
                            state.default_selected_index = idx;
                            state.selected_index = idx;
                            state.value = value;
                        } else if let Some(mut seed) = self.compute_form_defaults(n) {
                            seed.default_selected_index = idx;
                            seed.selected_index = idx;
                            seed.value = value.clone();
                            seed.default_value = value;
                            self.form_states.insert(n, seed);
                        }
                        break;
                    }
                    cur = self.get(n).and_then(|n| n.parent);
                }
            }
            return;
        }
        if !self.is_form_control(id) {
            return;
        }
        match (self.tag(id), name) {
            (Some("input"), "value") => {
                let new_default = if present {
                    self.attr(id, "value").unwrap_or("").to_string()
                } else {
                    String::new()
                };
                let _ = self.form_state_mut(id);
                if let Some(state) = self.form_states.get_mut(&id) {
                    state.default_value = new_default.clone();
                    if !state.value_dirty {
                        state.value = new_default;
                        let len = state.value.chars().count();
                        state.caret = len;
                        state.selection_start = len;
                        state.selection_end = len;
                    }
                }
            }
            (Some("input"), "checked") => {
                let _ = self.form_state_mut(id);
                if let Some(state) = self.form_states.get_mut(&id) {
                    state.default_checked = present;
                    if !state.checked_dirty {
                        state.checked = present;
                    }
                }
            }
            (Some("textarea"), "value") => {
                let new_default = if present {
                    self.attr(id, "value").unwrap_or("").to_string()
                } else {
                    self.descendant_text(id)
                };
                let _ = self.form_state_mut(id);
                if let Some(state) = self.form_states.get_mut(&id) {
                    state.default_value = new_default.clone();
                    if !state.value_dirty {
                        state.value = new_default;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Parse HTML text into a [`Dom`]. Lenient by design: this never
/// fails; malformed input is recovered per the WHATWG tree-construction
/// algorithm and parse errors are recorded in the builder stats (see
/// [`parse_detailed`]) rather than being fatal.
pub fn parse(html: &str) -> Dom {
    parse_detailed(html).dom
}

/// Like [`parse`], but also returns the side maps the `Dom` shape
/// itself cannot hold (qualified names, full namespaced attributes,
/// `<template>` contents), the quirks mode, parse-error stats, and the
/// [`DomEvent`]s observed during the parse.
pub fn parse_detailed(html: &str) -> ParseOutput {
    parse_detailed_with_opts(html, true)
}

/// Like [`parse_detailed`], with scripting on/off (controls
/// `<noscript>` parsing; the html5lib-tests corpus runs each test in
/// both modes unless restricted).
pub fn parse_detailed_with_opts(html: &str, scripting_enabled: bool) -> ParseOutput {
    builder::parse_document_with(html, scripting_enabled)
}

/// Parse an HTML fragment in the context of an element named
/// `context_tag` (namespace `context_ns`: `"svg"`, `"math"`/MathML, or
/// `None` for HTML). Returns the document plus the context element,
/// with the fragment's top-level nodes under it — the `innerHTML`
/// seam (Stage 7).
///
/// Scripts the fragment creates are **inert**: they exist as
/// `<script>` elements in the fragment DOM, but they must never
/// execute and are never routed to any script queue
/// (`ParseOutput::scripts_inert` is true for fragment parses).
/// The bindings layer's `set_inner_html` relies on this — it may
/// move the fragment's nodes into the live document, but it must
/// not execute a fragment script.
pub fn parse_fragment(
    html: &str,
    context_tag: &str,
    context_ns: Option<&str>,
) -> (Dom, NodeId) {
    let (out, ctx) = parse_fragment_detailed(html, context_tag, context_ns, true);
    (out.dom, ctx)
}

/// Like [`parse_fragment`], with scripting control and full output.
pub fn parse_fragment_detailed(
    html: &str,
    context_tag: &str,
    context_ns: Option<&str>,
    scripting_enabled: bool,
) -> (ParseOutput, NodeId) {
    builder::parse_fragment_with(html, context_tag, context_ns, scripting_enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First element descendant of `id` with the given tag, depth-first.
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

    /// All element descendants with the given tag, depth-first.
    fn find_all_tags(dom: &Dom, id: NodeId, tag: &str, out: &mut Vec<NodeId>) {
        for &child in dom.children(id) {
            if dom.tag(child) == Some(tag) {
                out.push(child);
            }
            find_all_tags(dom, child, tag, out);
        }
    }

    /// Concatenated text of a node's subtree.
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


    // ---- Stage 15 (M7 Phase 1, Rec 1): namespace threading ----

    #[test]
    fn parse_threads_svg_namespace_onto_arena_nodes() {
        let dom = parse(
            r#"<svg viewBox="0 0 10 10"><path d="M0 0h10"/><foreignObject><div>hi</div></foreignObject></svg>"#,
        );
        let root = dom.root().unwrap();
        let svg = find_tag(&dom, root, "svg").unwrap();
        assert_eq!(dom.namespace(svg), Some(NS_SVG));
        assert!(dom.is_svg(svg));
        let path = find_tag(&dom, root, "path").unwrap();
        assert_eq!(dom.namespace(path), Some(NS_SVG));
        // Inside <foreignObject> content is HTML again.
        let div = find_tag(&dom, root, "div").unwrap();
        assert_eq!(dom.namespace(div), Some(""));
        assert!(!dom.is_svg(div));
        // A plain HTML element is HTML-namespaced.
        let body = find_tag(&dom, root, "body").unwrap();
        assert_eq!(dom.namespace(body), Some(""));
    }

    #[test]
    fn parse_threads_mathml_namespace() {
        let dom = parse(r#"<math><mi>x</mi></math>"#);
        let root = dom.root().unwrap();
        let math = find_tag(&dom, root, "math").unwrap();
        assert_eq!(dom.namespace(math), Some(NS_MATHML));
        let mi = find_tag(&dom, root, "mi").unwrap();
        assert_eq!(dom.namespace(mi), Some(NS_MATHML));
    }

    #[test]
    fn create_element_ns_sets_namespace_and_create_element_does_not() {
        let mut dom = Dom::new();
        let svg = dom.create_element_ns(NS_SVG, "svg");
        assert_eq!(dom.namespace(svg), Some(NS_SVG));
        assert!(dom.is_svg(svg));
        let div = dom.create_element("div");
        assert_eq!(dom.namespace(div), Some(""));
        // Arbitrary namespaces round-trip (createElementNS accepts
        // any URI, not just the known three).
        let custom = dom.create_element_ns("https://example.com/ns", "x-thing");
        assert_eq!(dom.namespace(custom), Some("https://example.com/ns"));
    }

    #[test]
    fn form_state_seeded_from_attributes() {
        let mut dom = parse(
            r#"<form><input name="q" value="hello"><input type="checkbox" name="c" checked><textarea name="t">dflt</textarea></form>"#,
        );
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        let mut inputs = Vec::new();
        find_all_tags(&dom, form, "input", &mut inputs);
        assert_eq!(dom.form_value(inputs[0]).as_deref(), Some("hello"));
        assert_eq!(dom.form_default_value(inputs[0]).as_deref(), Some("hello"));
        assert!(!dom.form_value_dirty(inputs[0]));
        assert_eq!(dom.form_checked(inputs[1]), Some(true));
        let ta = find_tag(&dom, form, "textarea").unwrap();
        assert_eq!(dom.form_value(ta).as_deref(), Some("dflt"));
        assert_eq!(dom.form_owner(inputs[0]), Some(form));
    }

    #[test]
    fn dirty_value_survives_set_attribute_and_reset_restores() {
        let mut dom = parse(r#"<form><input name="q" value="orig"></form>"#);
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        let input = find_tag(&dom, form, "input").unwrap();
        assert!(dom.set_form_value(input, "typed"));
        assert!(dom.form_value_dirty(input));
        // setAttribute moves the default only — the typed value stands.
        dom.set_attribute(input, "value", "attr-change");
        assert_eq!(dom.form_value(input).as_deref(), Some("typed"));
        assert_eq!(dom.form_default_value(input).as_deref(), Some("attr-change"));
        // reset() restores from the (new) seeded default.
        dom.reset_form(form);
        assert_eq!(dom.form_value(input).as_deref(), Some("attr-change"));
        assert!(!dom.form_value_dirty(input));
    }

    #[test]
    fn clean_value_follows_set_attribute() {
        let mut dom = parse(r#"<form><input name="q" value="a"></form>"#);
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        let input = find_tag(&dom, form, "input").unwrap();
        dom.set_attribute(input, "value", "b");
        assert_eq!(dom.form_value(input).as_deref(), Some("b"));
        dom.remove_attribute(input, "value");
        assert_eq!(dom.form_value(input).as_deref(), Some(""));
    }

    #[test]
    fn select_last_selected_wins_and_entries_use_selected() {
        let mut dom = parse(
            r#"<form><select name="s"><option value="a">A</option><option value="b" selected>B</option><option value="c" selected>C</option></select></form>"#,
        );
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        let select = find_tag(&dom, form, "select").unwrap();
        assert_eq!(dom.form_selected_index(select), Some(2));
        assert_eq!(dom.form_value(select).as_deref(), Some("c"));
        assert_eq!(dom.form_entries(form, None), vec![("s".to_string(), "c".to_string())]);
        assert!(dom.set_form_selected_index(select, 0));
        assert_eq!(dom.form_value(select).as_deref(), Some("a"));
        dom.reset_form(form);
        assert_eq!(dom.form_selected_index(select), Some(2));
    }

    #[test]
    fn form_entries_successful_controls_only() {
        let mut dom = parse(
            r#"<form><input name="a" value="1"><input name="b" value="2" disabled><input name="c"><input type="checkbox" name="d" value="yes" checked><input type="checkbox" name="e" value="yes"><input type="submit" name="go" value="Go"></form>"#,
        );
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        // No submitter: the submit button contributes nothing.
        assert_eq!(
            dom.form_entries(form, None),
            vec![
                ("a".to_string(), "1".to_string()),
                ("c".to_string(), "".to_string()),
                ("d".to_string(), "yes".to_string()),
            ]
        );
        let mut submits = Vec::new();
        find_all_tags(&dom, form, "input", &mut submits);
        let submit = *submits.last().unwrap();
        let mut entries = dom.form_entries(form, Some(submit));
        assert_eq!(entries.pop(), Some(("go".to_string(), "Go".to_string())));
    }

    #[test]
    fn form_text_editing_insert_and_delete() {
        let mut dom = parse(r#"<form><input name="q" value="ab"></form>"#);
        let root = dom.root().unwrap();
        let form = find_tag(&dom, root, "form").unwrap();
        let input = find_tag(&dom, form, "input").unwrap();
        assert_eq!(dom.insert_form_text(input, "c"), Some(3));
        assert_eq!(dom.form_value(input).as_deref(), Some("abc"));
        assert_eq!(dom.delete_form_text(input, true), Some(2));
        assert_eq!(dom.form_value(input).as_deref(), Some("ab"));
        assert!(dom.set_form_selection(input, 1, 1));
        assert_eq!(dom.insert_form_text(input, "X"), Some(2));
        assert_eq!(dom.form_value(input).as_deref(), Some("aXb"));
    }

    fn assert_sync<T: Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn dom_is_send_and_sync() {
        // The per-tab worker thread owns the parser while other
        // threads read the Dom: Dom must stay Sync (the assertion
        // also lives in pluto-layout's tests).
        assert_sync::<Dom>();
        assert_send::<Dom>();
        assert_sync::<NodeId>();
    }

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
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

    #[test]
    fn parses_example_com_document() {
        let dom = parse(EXAMPLE_COM);
        let root = dom.root().unwrap();
        let html = find_tag(&dom, root, "html").expect("html element");
        let head = find_tag(&dom, html, "head").expect("head element");
        let body = find_tag(&dom, html, "body").expect("body element");
        assert_eq!(text_content(&dom, find_tag(&dom, head, "title").unwrap()), "Example Domain");
        let h1 = find_tag(&dom, body, "h1").expect("h1");
        assert_eq!(text_content(&dom, h1), "Example Domain");
        let link = find_tag(&dom, body, "a").expect("link");
        assert_eq!(
            dom.attr(link, "href"),
            Some("https://www.iana.org/domains/example")
        );
        assert_eq!(text_content(&dom, link), "More information...");
        // The <style> block is raw text, not parsed as elements.
        let style = find_tag(&dom, head, "style").expect("style");
        assert!(text_content(&dom, style).contains("background-color: #f0f0f2"));
        assert!(find_tag(&dom, style, "body").is_none());
        // <meta> is void: no children, and the title still follows it.
        let meta = find_tag(&dom, head, "meta").expect("meta");
        assert!(dom.children(meta).is_empty());
        // Conforming doctype -> standards mode, and the doctype node
        // itself is a child of the document.
        assert_eq!(dom.quirks_mode(), QuirksMode::NoQuirks);
        let dt = dom.doctype().expect("doctype node");
        assert!(matches!(
            dom.get(dt).map(|n| &n.kind),
            Some(NodeKind::Doctype { name, .. }) if name == "html"
        ));
    }

    #[test]
    fn void_elements_comments_and_whitespace() {
        let dom = parse("<br><img src=x><!-- a comment --><p>a</p>   <p>b</p>");
        let root = dom.root().unwrap();
        let mut ps = Vec::new();
        find_all_tags(&dom, root, "p", &mut ps);
        assert_eq!(ps.len(), 2);
        assert_eq!(text_content(&dom, ps[0]), "a");
        let br = find_tag(&dom, root, "br").expect("br");
        assert!(dom.children(br).is_empty());
        let img = find_tag(&dom, root, "img").expect("img");
        assert_eq!(dom.attr(img, "src"), Some("x"));
        // Comments are real nodes now (the old parser skipped them).
        let body = find_tag(&dom, root, "body").unwrap();
        assert!(
            dom.children(body)
                .iter()
                .any(|&c| dom.get(c).and_then(|n| n.comment()) == Some(" a comment ")),
            "comment node present under body"
        );
    }

    #[test]
    fn entities_decoded() {
        let dom = parse("<p>Fish &amp; Chips &lt;tag&gt; &#65;&#x42; &quot;q&quot; &copy;</p>");
        let root = dom.root().unwrap();
        let p = find_tag(&dom, root, "p").unwrap();
        assert_eq!(text_content(&dom, p), "Fish & Chips <tag> AB \"q\" ©");
    }

    #[test]
    fn numeric_entities_validated_to_replacement() {
        // Pluto's tokenizer (entities::numeric_char) maps
        // surrogates, NUL, and values > U+10FFFF to U+FFFD.
        let dom = parse("<p>&#xD800;&#x110000;&#0;&#65;</p>");
        let root = dom.root().unwrap();
        let p = find_tag(&dom, root, "p").unwrap();
        assert_eq!(text_content(&dom, p), "\u{fffd}\u{fffd}\u{fffd}A");
    }

    #[test]
    fn mismatched_close_ignored() {
        // SPEC TRUTH (updated from the old parser's expectation): tree
        // construction closes the <div> at `</div>`, so the second <p>
        // is a sibling of the div under <body> — not a child of the
        // document root.
        let dom = parse("<div><p>one</div><p>two</p>");
        let root = dom.root().unwrap();
        let body = find_tag(&dom, root, "body").unwrap();
        let div = find_tag(&dom, root, "div").unwrap();
        let mut ps_in_div = Vec::new();
        find_all_tags(&dom, div, "p", &mut ps_in_div);
        assert_eq!(ps_in_div.len(), 1);
        assert_eq!(text_content(&dom, ps_in_div[0]), "one");
        let mut all_ps = Vec::new();
        find_all_tags(&dom, root, "p", &mut all_ps);
        assert_eq!(all_ps.len(), 2);
        assert_eq!(text_content(&dom, all_ps[1]), "two");
        assert_eq!(
            dom.get(all_ps[1]).unwrap().parent,
            Some(body),
            "second <p> lands under <body>"
        );
    }

    #[test]
    fn attributes_parsed() {
        let dom = parse(r#"<a href='single' title="double" data-x=bare disabled>link</a>"#);
        let root = dom.root().unwrap();
        let a = find_tag(&dom, root, "a").unwrap();
        assert_eq!(dom.attr(a, "href"), Some("single"));
        assert_eq!(dom.attr(a, "title"), Some("double"));
        assert_eq!(dom.attr(a, "data-x"), Some("bare"));
        assert_eq!(dom.attr(a, "disabled"), Some(""));
        assert_eq!(dom.attr(a, "missing"), None);
    }

    #[test]
    fn stray_lt_is_text() {
        let dom = parse("<p>3 < 5</p>");
        let root = dom.root().unwrap();
        let p = find_tag(&dom, root, "p").unwrap();
        assert_eq!(text_content(&dom, p), "3 < 5");
    }

    #[test]
    fn unquoted_values_may_contain_slashes() {
        // Regression: the live example.com page carries
        // `<script src=/s.js>`. The old parser stopped unquoted values
        // at `/`, which failed the whole tag and rendered it as text.
        let dom = parse("<body><script src=/s.js></script><p>ok</p></body>");
        let root = dom.root().unwrap();
        let script = find_tag(&dom, root, "script").unwrap();
        assert_eq!(dom.attr(script, "src"), Some("/s.js"));
        // No stray text nodes from a failed tag parse.
        let body = find_tag(&dom, root, "body").unwrap();
        let texts: Vec<&str> = dom
            .children(body)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        assert!(texts.is_empty(), "unexpected text nodes: {texts:?}");

        // A trailing slash right before `>` in an UNQUOTED value is
        // part of the value, per spec (the self-closing marker is only
        // recognised between attributes, not inside a value):
        // `<img src=/a/b.png/>` means src="/a/b.png/". The old
        // hand-written parser stripped it; the spec keeps it, which
        // is spec truth — the expectation changed deliberately.
        let dom = parse("<img src=/a/b.png/><img src=x/>");
        let root = dom.root().unwrap();
        let mut imgs = Vec::new();
        find_all_tags(&dom, root, "img", &mut imgs);
        assert_eq!(imgs.len(), 2);
        assert_eq!(dom.attr(imgs[0], "src"), Some("/a/b.png/"));
        assert_eq!(dom.attr(imgs[1], "src"), Some("x/"));
    }

    #[test]
    fn misnested_formatting_adoption_agency() {
        // <b><i></b></i> — AAA pops the stack; both stay in the tree,
        // <i> inside <b> (no furthest block here, so no clone).
        let dom = parse("<b><i></b></i>");
        let root = dom.root().unwrap();
        let body = find_tag(&dom, root, "body").unwrap();
        let mut bs = Vec::new();
        find_all_tags(&dom, body, "b", &mut bs);
        let mut is = Vec::new();
        find_all_tags(&dom, body, "i", &mut is);
        assert_eq!(bs.len(), 1);
        assert_eq!(is.len(), 1);
        assert_eq!(dom.get(is[0]).unwrap().parent, Some(bs[0]));
    }

    #[test]
    fn adoption_agency_clones_across_block() {
        // <a><p></a></p> — AAA with a furthest block (<p>): the <a>
        // is cloned inside the <p>; the original <a> stays empty.
        let dom = parse("<a><p></a></p>");
        let root = dom.root().unwrap();
        let body = find_tag(&dom, root, "body").unwrap();
        let kids = dom.children(body);
        assert_eq!(kids.len(), 2, "body has <a> then <p>");
        assert_eq!(dom.tag(kids[0]), Some("a"));
        assert_eq!(dom.tag(kids[1]), Some("p"));
        assert_eq!(dom.children(kids[1]).len(), 1);
        assert_eq!(dom.tag(dom.children(kids[1])[0]), Some("a"));
    }

    #[test]
    fn table_foster_parenting() {
        // <table>B<tr>C</tr>D</table> — text that cannot live in a
        // table is foster-parented BEFORE the table, merged into one
        // text node. (The Stage 6 report writes "ABCD" for this input,
        // but the input contains only B, C, D — "BCD" is correct.)
        let dom = parse("<table>B<tr>C</tr>D</table>");
        let root = dom.root().unwrap();
        let body = find_tag(&dom, root, "body").unwrap();
        let kids = dom.children(body);
        assert_eq!(text_content(&dom, kids[0]), "BCD");
        assert_eq!(dom.tag(kids[1]), Some("table"));
        let table = kids[1];
        assert_eq!(text_content(&dom, table), "");
    }

    #[test]
    fn implied_closes_and_self_closing_non_void() {
        let dom = parse("<p>one<p>two");
        let root = dom.root().unwrap();
        let body = find_tag(&dom, root, "body").unwrap();
        let mut ps = Vec::new();
        find_all_tags(&dom, body, "p", &mut ps);
        assert_eq!(ps.len(), 2);
        assert_eq!(text_content(&dom, ps[0]), "one");
        assert_eq!(text_content(&dom, ps[1]), "two");
        assert_eq!(
            dom.get(ps[0]).unwrap().parent,
            dom.get(ps[1]).unwrap().parent,
            "implied close makes siblings"
        );

        // <span/> is a parse error and the self-closing flag is
        // IGNORED: the span stays open, text lands inside it.
        let out = parse_detailed("<div><span/>text</div>");
        assert!(out.stats.parse_errors > 0, "self-closing non-void is a parse error");
        let span = find_tag(&out.dom, out.dom.root().unwrap(), "span").unwrap();
        assert_eq!(text_content(&out.dom, span), "text");
    }

    #[test]
    fn quirks_modes() {
        assert_eq!(parse("<!DOCTYPE html><p>x").quirks_mode(), QuirksMode::NoQuirks);
        assert_eq!(parse("<p>x").quirks_mode(), QuirksMode::Quirks);
        assert_eq!(
            parse("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><p>x")
                .quirks_mode(),
            QuirksMode::LimitedQuirks
        );
    }

    #[test]
    fn parse_never_fails_and_records_errors() {
        // Malformed input is recovered, never fatal; errors recorded.
        let out = parse_detailed("<div><span/>unclosed<p>&bogus; &#xZZ;");
        assert!(out.stats.parse_errors > 0);
        assert!(find_tag(&out.dom, out.dom.root().unwrap(), "div").is_some());
    }

    #[test]
    fn fragment_parsing_reparents_under_context() {
        let (dom, ctx) = parse_fragment("<p>one<p>two", "div", None);
        assert_eq!(dom.tag(ctx), Some("div"));
        let kids = dom.children(ctx);
        assert_eq!(kids.len(), 2);
        assert_eq!(dom.tag(kids[0]), Some("p"));
        assert_eq!(text_content(&dom, kids[0]), "one");
        assert_eq!(dom.tag(kids[1]), Some("p"));
        assert_eq!(text_content(&dom, kids[1]), "two");

        // Fragment in a table context: foster parenting still applies.
        let (dom, ctx) = parse_fragment("<tr><td>cell</td></tr>", "table", None);
        assert!(find_tag(&dom, ctx, "td").is_some());
    }

    #[test]
    fn fragment_scripts_are_inert() {
        // innerHTML scripts exist in the fragment DOM but must
        // never execute: no pause, no queue event, inert flag set.
        let (out, ctx) =
            parse_fragment_detailed("<script>alert(1)</script><p>after</p>", "div", None, true);
        assert!(out.scripts_inert, "fragment parse marks its scripts inert");
        let script = find_tag(&out.dom, ctx, "script").expect("script element exists");
        assert_eq!(text_content(&out.dom, script), "alert(1)");
        assert!(find_tag(&out.dom, ctx, "p").is_some());
        assert!(
            out.events.iter().all(|e| !matches!(
                e,
                DomEvent::ScriptPaused(_) | DomEvent::AsyncScriptQueued { .. }
            )),
            "fragment scripts surface no script events: {:?}",
            out.events
        );
        // A whole-document parse is not inert.
        let doc = parse_detailed("<p>x</p>");
        assert!(!doc.scripts_inert);
    }

    #[test]
    fn mutation_apis_record_and_mark_dirty() {
        let mut dom = Dom::new();
        let root = dom.root().unwrap();
        let div = dom.create_element("div");
        dom.append_child(root, div);
        // One record for the insert; the node is style+layout dirty,
        // and the root carries the subtree summary bits.
        assert_eq!(dom.mutations().len(), 1);
        assert!(dom.get(div).unwrap().dirty.style_self);
        assert!(dom.get(div).unwrap().dirty.layout_self);
        assert!(dom.get(root).unwrap().dirty.style_subtree);
        assert!(dom.get(root).unwrap().dirty.layout_subtree);
        assert!(dom.dirty_set().style.contains(&div));
        assert!(dom.dirty_set().layout.contains(&div));

        // Attribute change: snapshot holds the BEFORE state.
        dom.set_attribute(div, "class", "red");
        dom.set_attribute(div, "class", "blue");
        let recs = dom.mutations();
        let last = recs.last().unwrap();
        assert!(matches!(last.kind, MutationKind::AttributeChanged { .. }));
        assert_eq!(last.before.as_ref().unwrap().class.as_deref(), Some("red"));

        // DOM API replaces (first-duplicate-wins is parse-only).
        assert_eq!(dom.attr(div, "class"), Some("blue"));

        // Text + structure mutations.
        let text = dom.create_text("hello");
        dom.append_child(div, text);
        dom.set_text(text, "bye");
        assert_eq!(dom.text(text), Some("bye"));
        let span = dom.create_element("span");
        dom.replace_child(div, text, span);
        assert_eq!(dom.children(div), &[span]);
        assert!(dom.get(text).unwrap().parent.is_none());
        dom.remove_child(div, span);
        assert!(dom.children(div).is_empty());

        // reparent_children moves in order.
        let a = dom.create_element("a");
        let b = dom.create_element("b");
        dom.append_child(div, a);
        dom.append_child(div, b);
        let section = dom.create_element("section");
        dom.append_child(root, section);
        dom.reparent_children(div, section);
        assert_eq!(dom.children(section), &[a, b]);
        assert!(dom.children(div).is_empty());

        // The clear pass resets bits and the set.
        dom.clear_dirty();
        assert!(dom.dirty_set().is_empty());
        assert!(!dom.get(div).unwrap().dirty.style_self);
        assert!(!dom.get(root).unwrap().dirty.style_subtree);
        // …and draining the log empties it.
        assert!(!dom.take_mutations().is_empty());
        assert!(dom.mutations().is_empty());
    }

    #[test]
    fn mutation_observer_records_in_order_and_respects_options() {
        // Stage 8 Rec 7 (DOM half): childList / attributes /
        // characterData records, in mutation order, with the
        // observer's options (subtree, attributeFilter, old values)
        // deciding what is queued.
        let mut dom = Dom::new();
        let root = dom.root().unwrap();
        let div = dom.create_element("div");
        dom.append_child(root, div);
        let text = dom.create_text("hello");
        dom.append_child(div, text);
        dom.clear_mutations();

        // Observer 1: childList + attributes on `div`, subtree on.
        let opts = MutationObserverOptions {
            child_list: true,
            attributes: true,
            character_data: true,
            subtree: true,
            attribute_old_value: true,
            character_data_old_value: true,
            attribute_filter: None,
        };
        assert!(dom.observe_mutation(1, div, opts));
        // Observer 2: attributes only, filtered to `id`, no subtree.
        let opts2 = MutationObserverOptions {
            attribute_filter: Some(vec!["id".into()]),
            ..Default::default()
        };
        assert!(dom.observe_mutation(2, div, opts2));

        let span = dom.create_element("span");
        dom.append_child(div, span);
        dom.set_attribute(div, "class", "red");
        dom.set_attribute(div, "class", "blue");
        dom.set_attribute(div, "id", "x");
        dom.set_text(text, "bye");
        dom.remove_child(div, span);

        // Observer 1 sees everything, in order, with old values.
        let recs = dom.take_mutation_records(1);
        assert_eq!(recs.len(), 6, "records: {recs:?}");
        assert_eq!(recs[0].kind, ObserverRecordKind::ChildList);
        assert_eq!(recs[0].target, div);
        assert_eq!(recs[0].added_nodes, vec![span]);
        assert!(recs[0].removed_nodes.is_empty());
        assert_eq!(recs[1].kind, ObserverRecordKind::Attributes);
        assert_eq!(recs[1].attribute_name.as_deref(), Some("class"));
        assert_eq!(recs[1].old_value.as_deref(), None);
        assert_eq!(recs[2].old_value.as_deref(), Some("red"));
        assert_eq!(recs[3].attribute_name.as_deref(), Some("id"));
        assert_eq!(recs[4].kind, ObserverRecordKind::CharacterData);
        assert_eq!(recs[4].target, text);
        assert_eq!(recs[4].old_value.as_deref(), Some("hello"));
        assert_eq!(recs[5].kind, ObserverRecordKind::ChildList);
        assert_eq!(recs[5].removed_nodes, vec![span]);
        // The span followed the text node, so that is its previous
        // sibling at removal time.
        assert_eq!(recs[5].previous_sibling, Some(text));
        // takeRecords drains: a second take is empty.
        assert!(dom.take_mutation_records(1).is_empty());
        // Observer 2 saw only the `id` attribute change.
        let recs2 = dom.take_mutation_records(2);
        assert_eq!(recs2.len(), 1, "filtered records: {recs2:?}");
        assert_eq!(recs2[0].attribute_name.as_deref(), Some("id"));
        // `takeRecords` cleared the pending list for both.
        assert!(dom.pending_mutation_observers().is_empty());

        // Disconnect stops delivery: later mutations queue nothing.
        dom.disconnect_mutation_observer(1);
        let b = dom.create_element("b");
        dom.append_child(div, b);
        assert!(dom.take_mutation_records(1).is_empty());
        assert!(dom.pending_mutation_observers().is_empty());
        // A detached node is still a live arena node, so
        // observing it succeeds (it just never matches a mutation
        // while detached).
        assert!(dom.observe_mutation(3, span, MutationObserverOptions {
            child_list: true,
            ..Default::default()
        }));
    }

    #[test]
    fn parser_mutations_do_not_record() {
        // Parser privilege: a parse leaves no mutation log and no
        // dirty bits behind (the initial tree is the baseline).
        let dom = parse("<div class=x><p>one<p>two</div>");
        assert!(dom.mutations().is_empty());
        assert!(dom.dirty_set().is_empty());
        let root = dom.root().unwrap();
        assert!(!dom.get(root).unwrap().dirty.style_subtree);
    }
}
