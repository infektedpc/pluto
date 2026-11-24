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

pub mod builder;
pub mod entities;
pub mod scanner;
pub mod streaming;
pub mod tokenizer;

use std::collections::HashSet;

use builder::ParseOutput;
use slotmap::{SlotMap, new_key_type};

pub use builder::{DomEvent, QName, SinkStats};
pub use scanner::{Destination, DiscoveredResource, Priority, resolve_url, scan};
pub use streaming::{Parser, ParsedStream, ScriptPaused};

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
#[derive(Debug, Clone)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Attributes in source order; duplicate names keep the first value
    /// (a *parse* rule — see [`Dom::set_attribute`], which replaces).
    pub attrs: Vec<(String, String)>,
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

/// The arena-allocated DOM tree for one document.
#[derive(Debug, Clone, Default)]
pub struct Dom {
    arena: SlotMap<NodeId, Node>,
    root: Option<NodeId>,
    quirks_mode: QuirksMode,
    mutations: Vec<MutationRecord>,
    dirty: DirtySet,
    /// Parser privilege: while true, mutators still maintain
    /// parent/child invariants but skip mutation records and dirty
    /// bits. Set by the builder for the duration of a parse and cleared
    /// when the builder finishes; JS-driven mutations always
    /// record. This flag is internal state, not a public toggle.
    suppress_records: bool,
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

    pub(crate) fn set_suppress_records(&mut self, suppress: bool) {
        self.suppress_records = suppress;
    }

    // --- node creation (allocation only; no tree mutation, no record) ---

    pub fn create_element(&mut self, tag: &str) -> NodeId {
        self.create_element_with_attrs(tag, Vec::new())
    }

    pub fn create_element_with_attrs(
        &mut self,
        tag: &str,
        attrs: Vec<(String, String)>,
    ) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Element {
                tag: tag.to_string(),
            },
            parent: None,
            children: Vec::new(),
            attrs,
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_text(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Text(data.to_string()),
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
            dirty: DirtyFlags::default(),
        })
    }

    pub fn create_comment(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Comment(data.to_string()),
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
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
        self.link_at(parent, child, None);
        self.note_mutation(child, MutationKind::ChildInserted { parent }, Some(before));
    }

    /// Insert `child` as a child of `parent` immediately before
    /// `sibling`. If `sibling` is not a child of `parent`, appends.
    pub fn insert_before(&mut self, parent: NodeId, sibling: NodeId, child: NodeId) {
        let before = BeforeSnapshot::capture(self, child);
        let pos = self
            .arena
            .get(parent)
            .and_then(|n| n.children.iter().position(|&c| c == sibling));
        self.link_at(parent, child, pos);
        self.note_mutation(child, MutationKind::ChildInserted { parent }, Some(before));
    }

    /// Detach `child` from `parent`. (Lenient: detaches wherever the
    /// child currently lives.)
    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) {
        let before = BeforeSnapshot::capture(self, child);
        self.detach(child);
        self.note_mutation(child, MutationKind::ChildRemoved { parent }, Some(before));
    }

    /// Detach `target` from whatever parent it has, if any.
    pub fn remove_from_parent(&mut self, target: NodeId) {
        let before = BeforeSnapshot::capture(self, target);
        if let Some(parent) = self.detach(target) {
            self.note_mutation(target, MutationKind::ChildRemoved { parent }, Some(before));
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
        self.detach(old);
        self.link_at(parent, new, pos);
        self.note_mutation(new, MutationKind::ChildReplaced { parent }, Some(before));
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
            self.link_at(new_parent, kid, None);
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
    }

    pub fn remove_attribute(&mut self, id: NodeId, name: &str) {
        let before = BeforeSnapshot::capture(self, id);
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
        }
    }

    /// Add attributes that are not already present (by local name).
    /// Used by the tree builder when a second `<html>` / `<body>` tag
    /// carries new attributes.
    pub fn add_attrs_if_missing(&mut self, target: NodeId, attrs: Vec<(String, String)>) {
        let before = BeforeSnapshot::capture(self, target);
        let mut added: Option<String> = None;
        if let Some(node) = self.arena.get_mut(target) {
            for (name, value) in attrs {
                if !node.attrs.iter().any(|(n, _)| *n == name) {
                    if added.is_none() {
                        added = Some(name.clone());
                    }
                    node.attrs.push((name, value));
                }
            }
        }
        if let Some(name) = added {
            self.note_mutation(target, MutationKind::AttributeChanged { name }, Some(before));
        }
    }

    /// Replace a text node's data.
    pub fn set_text(&mut self, id: NodeId, data: &str) {
        let before = BeforeSnapshot::capture(self, id);
        if let Some(Node { kind: NodeKind::Text(contents), .. }) = self.arena.get_mut(id) {
            *contents = data.to_string();
        }
        self.note_mutation(id, MutationKind::CharacterDataChanged, Some(before));
    }

    /// Append to an existing text node (adjacent-text merging: the
    /// builder hands the tree text in chunks).
    pub fn append_to_text(&mut self, id: NodeId, data: &str) {
        let before = BeforeSnapshot::capture(self, id);
        if let Some(Node { kind: NodeKind::Text(contents), .. }) = self.arena.get_mut(id) {
            contents.push_str(data);
        }
        self.note_mutation(id, MutationKind::CharacterDataChanged, Some(before));
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
