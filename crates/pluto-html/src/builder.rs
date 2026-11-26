//! Pluto's own WHATWG tree-construction machine, driving the
//! slotmap [`Dom`](crate::Dom) directly (no TreeSink trait — the
//! builder owns the Dom and calls its mutators).
//!
//! State: insertion mode (+ a mode stack for `<template>`), stack
//! of open elements (`<html>` is never popped), list of active
//! formatting elements (with scope markers, reconstructed when
//! text arrives after an early close), adoption agency (outer loop
//! capped at 8), foster parenting, implied end tags, quirks flag
//! from the DOCTYPE, raw-text/RCDATA handling (via the tokenizer's
//! content model), parse errors recorded and never fatal, practical
//! nesting-depth cap.
//!
//! Foreign content (SVG/MathML) is handled at the integration
//! points: elements carry a namespace in the [`ParseOutput::names`]
//! side map (the `Dom` node itself stores only the local tag name),
//! and the builder does not assume everything is HTML-namespace.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::tokenizer::{TagToken, Token, Tokenizer};
use crate::{Dom, NodeId, NodeKind, QuirksMode};

pub const NS_HTML: &str = "http://www.w3.org/1999/xhtml";
pub const NS_SVG: &str = "http://www.w3.org/2000/svg";
pub const NS_MATHML: &str = "http://www.w3.org/1998/Math/MathML";
pub const NS_XML: &str = "http://www.w3.org/XML/1998/namespace";
pub const NS_XLINK: &str = "http://www.w3.org/1999/xlink";
pub const NS_XMLNS: &str = "http://www.w3.org/2000/xmlns/";

/// Qualified name: namespace + local name + optional prefix.
/// The `Dom` stores only the local name; this lives in the
/// [`ParseOutput::names`] / `full_attrs` side maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QName {
    pub ns: String,
    pub local: String,
    pub prefix: Option<String>,
}

impl QName {
    pub fn html(local: &str) -> Self {
        QName {
            ns: String::new(),
            local: local.to_string(),
            prefix: None,
        }
    }
    pub fn new(ns: &str, local: &str) -> Self {
        QName {
            ns: ns.to_string(),
            local: local.to_string(),
            prefix: None,
        }
    }
}

/// Events the parser observes while building the tree. The streaming
/// driver ([`crate::streaming::Parser`]) drains these after each
/// feed and hands them to the worker.
#[derive(Debug, Clone)]
pub enum DomEvent {
    /// An element was just created (attributes complete; children
    /// not yet attached).
    ElementCreated { node: NodeId, tag: String },
    /// An element was popped from the open-element stack: its
    /// subtree is complete as far as the tree builder knows.
    SubtreeComplete { node: NodeId },
    /// A classic parser-blocking `<script>` just completed.
    ScriptPaused(crate::streaming::ScriptPaused),
    /// An `async` script was discovered and queued for execution
    /// as soon as its fetch arrives (arrival order, no order
    /// guarantee). Stage 7: the fetch completion is reported
    /// through the [`crate::scripts::AsyncQueue`], not an event —
    /// the parser cannot see the network.
    AsyncScriptQueued { script: NodeId, src: Option<String> },
    /// A `<link rel=stylesheet>` was created.
    StylesheetDiscovered { node: NodeId, href: String },
    /// A fetchable subresource was discovered on a created element.
    ResourceDiscovered(crate::scanner::DiscoveredResource),
    /// The document indicated a different encoding (label
    /// unvalidated).
    EncodingIndicator { label: String },
}

/// Parse-error / fidelity statistics for one parse.
#[derive(Debug, Default, Clone)]
pub struct SinkStats {
    pub parse_errors: usize,
    pub first_errors: Vec<String>,
    pub attrs_with_ns_or_prefix: usize,
    pub foreign_elements: usize,
    pub comments: usize,
    pub doctypes: usize,
    pub processing_instructions: usize,
}

/// Everything a finished parse returns: the `Dom` plus the side
/// maps the `Dom` shape itself cannot hold (namespaces, full
/// attributes, `<template>` contents), the quirks mode, stats, and
/// the events observed during the parse.
#[derive(Debug)]
pub struct ParseOutput {
    pub dom: Dom,
    pub names: HashMap<NodeId, QName>,
    pub full_attrs: HashMap<NodeId, Vec<(QName, String)>>,
    pub template_contents: HashMap<NodeId, NodeId>,
    pub quirks_mode: QuirksMode,
    pub stats: SinkStats,
    pub events: Vec<DomEvent>,
    /// True for a fragment parse ([`crate::parse_fragment`]):
    /// `<script>` elements the fragment created are inert — they
    /// must never execute. False for a whole-document parse.
    /// The bindings layer's `set_inner_html` relies on this: it
    /// may move the fragment's nodes into the live document, but
    /// it must not route a fragment script through any script
    /// queue.
    pub scripts_inert: bool,
}

impl ParseOutput {
    pub fn document(&self) -> NodeId {
        self.dom.root().expect("Dom::new always sets a root")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Initial,
    BeforeHtml,
    BeforeHead,
    InHead,
    InHeadNoscript,
    AfterHead,
    InBody,
    Text,
    InTable,
    InTableText,
    InCaption,
    InColumnGroup,
    InTableBody,
    InRow,
    InCell,
    InSelect,
    InSelectInTable,
    InTemplate,
    AfterBody,
    InFrameset,
    AfterFrameset,
    AfterAfterBody,
    AfterAfterFrameset,
}

#[derive(Debug, Clone)]
enum AfeEntry {
    Marker,
    Element(NodeId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    Default,
    ListItem,
    Button,
    Table,
}

/// Outcome of processing one token.
pub enum ProcessOutcome {
    Continue,
    /// A `<script>` element just completed (its end tag was
    /// processed in Text mode).
    Script(NodeId),
}

pub enum ModeAction {
    Done(ProcessOutcome),
    Reprocess(Token),
}

const MAX_DEPTH: usize = 512;

pub struct TreeBuilder {
    dom: Arc<Mutex<Dom>>,
    stack: Vec<NodeId>,
    afe: Vec<AfeEntry>,
    mode: Mode,
    original_mode: Mode,
    /// The insertion mode to return to when InTableText flushes
    /// its pending characters (the mode that was active when the
    /// character run started — e.g. InRow, not InTable).
    table_text_return_mode: Mode,
    template_modes: Vec<Mode>,
    template_contents_stack: Vec<NodeId>,
    html_node: Option<NodeId>,
    head_node: Option<NodeId>,
    body_node: Option<NodeId>,
    form_pointer: Option<NodeId>,
    fragment: bool,
    fragment_ctx: Option<NodeId>,
    /// Synthetic `<html>` root of a fragment parse (child of the
    /// document root; the context element is NOT on the stack).
    fragment_html: Option<NodeId>,
    /// Insertion mode a fragment resets to when its stack is back
    /// to just the synthetic html element.
    fragment_mode: Mode,
    /// True while processing a token with the in-body rules on
    /// behalf of a table mode ("anything else in table"): only
    /// then do insertions foster-parent. Without the flag,
    /// `appropriate_place` inserts under the current node even
    /// when it is table-shaped (html5lib's `insertFromTable`).
    insert_from_table: bool,
    /// Set once a clear-stack operation in a fragment has popped
    /// the context element's subtree (the reference keeps the
    /// context element on the stack, where such a clear pops it;
    /// later insertions then land outside the fragment).
    fragment_detached: bool,
    scripting: bool,
    frameset_ok: bool,
    pending_table_chars: String,
    pending_table_chars_non_ws: bool,
    quirks: QuirksMode,
    quirks_set: bool,
    parse_errors: usize,
    first_errors: Vec<String>,
    events: Vec<DomEvent>,
    names: HashMap<NodeId, QName>,
    full_attrs: HashMap<NodeId, Vec<(QName, String)>>,
    template_contents: HashMap<NodeId, NodeId>,
    base_href: Option<String>,
    skip_newline: bool,
    head_pointer: Option<NodeId>,
    /// When InHead rules are invoked from another mode (InBody /
    /// AfterHead) for a head element, the mode to return to after
    /// Text mode / a template ends.
    delegate_mode: Option<Mode>,
}

impl TreeBuilder {
    pub fn new(scripting: bool) -> Self {
        let mut dom = Dom::new();
        dom.set_suppress_records(true);
        Self {
            dom: Arc::new(Mutex::new(dom)),
            stack: Vec::new(),
            afe: Vec::new(),
            mode: Mode::Initial,
            original_mode: Mode::InBody,
            table_text_return_mode: Mode::InTable,
            template_modes: Vec::new(),
            template_contents_stack: Vec::new(),
            html_node: None,
            head_node: None,
            body_node: None,
            form_pointer: None,
            fragment: false,
            fragment_ctx: None,
            fragment_html: None,
            fragment_mode: Mode::InBody,
            fragment_detached: false,
            insert_from_table: false,
            scripting,
            frameset_ok: true,
            pending_table_chars: String::new(),
            pending_table_chars_non_ws: false,
            quirks: QuirksMode::NoQuirks,
            quirks_set: false,
            parse_errors: 0,
            first_errors: Vec::new(),
            events: Vec::new(),
            names: HashMap::new(),
            full_attrs: HashMap::new(),
            template_contents: HashMap::new(),
            base_href: None,
            skip_newline: false,
            head_pointer: None,
            delegate_mode: None,
        }
    }

    /// Fragment builder (WHATWG fragment algorithm): the stack
    /// starts as [synthetic `<html>`]; the context element is NOT
    /// on the stack — it is the *adjusted current node* while the
    /// stack holds only the synthetic html, so insertions land
    /// under it. The synthetic root's children (head/body/frameset
    /// for an `html` context) are re-parented under the context
    /// element by [`parse_fragment_with`].
    pub fn new_fragment(ctx_tag: &str, ctx_ns: Option<&str>, scripting: bool) -> (Self, NodeId) {
        let mut b = Self::new(scripting);
        b.fragment = true;
        let ns = match ctx_ns {
            Some("svg") => NS_SVG,
            Some("mathml") | Some("math") => NS_MATHML,
            _ => "",
        };
        let ctx = b.create_element_raw(ctx_tag, Vec::new(), ns);
        b.fragment_ctx = Some(ctx);
        // Synthetic html root, child of the document root.
        let root = b.dom.lock().unwrap().root().unwrap();
        let html = b.create_element_raw("html", Vec::new(), "");
        b.dom.lock().unwrap().append_child(root, html);
        b.fragment_html = Some(html);
        b.html_node = Some(html);
        b.stack.push(html);
        b.mode = match ctx_tag {
            "html" => Mode::BeforeHead,
            "head" => Mode::InHead,
            "frameset" => Mode::InFrameset,
            "table" => Mode::InTable,
            "tbody" | "thead" | "tfoot" => Mode::InTableBody,
            "tr" => Mode::InRow,
            "td" | "th" => Mode::InCell,
            "select" => Mode::InSelect,
            "template" => Mode::InTemplate,
            "caption" => Mode::InCaption,
            "colgroup" => Mode::InColumnGroup,
            _ => Mode::InBody,
        };
        b.fragment_mode = b.mode;
        (b, ctx)
    }

    pub fn shared_dom(&self) -> Arc<Mutex<Dom>> {
        Arc::clone(&self.dom)
    }

    pub fn base_href(&self) -> Option<String> {
        self.base_href.clone()
    }

    pub fn drain_events(&mut self) -> Vec<DomEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn stats(&self) -> SinkStats {
        SinkStats {
            parse_errors: self.parse_errors,
            first_errors: self.first_errors.clone(),
            attrs_with_ns_or_prefix: self
                .full_attrs
                .values()
                .flatten()
                .filter(|(q, _)| !q.ns.is_empty() || q.prefix.is_some())
                .count(),
            foreign_elements: self
                .names
                .values()
                .filter(|q| !q.ns.is_empty() && q.ns != NS_HTML)
                .count(),
            comments: 0,
            doctypes: 0,
            processing_instructions: 0,
        }
    }

    fn error(&mut self, msg: &str) {
        self.parse_errors += 1;
        if self.first_errors.len() < 10 {
            self.first_errors.push(msg.to_string());
        }
    }

    pub fn finish(self) -> ParseOutput {
        let mut dom = match Arc::try_unwrap(self.dom) {
            Ok(m) => m.into_inner().unwrap(),
            Err(a) => a.lock().unwrap().clone(),
        };
        dom.set_suppress_records(false);
        dom.set_quirks_mode(self.quirks);
        let attrs_with_ns = self
            .full_attrs
            .values()
            .flatten()
            .filter(|(q, _)| !q.ns.is_empty() || q.prefix.is_some())
            .count();
        let foreign = self
            .names
            .values()
            .filter(|q| !q.ns.is_empty() && q.ns != NS_HTML)
            .count();
        ParseOutput {
            dom,
            names: self.names,
            full_attrs: self.full_attrs,
            template_contents: self.template_contents,
            quirks_mode: self.quirks,
            stats: SinkStats {
                parse_errors: self.parse_errors,
                first_errors: self.first_errors,
                attrs_with_ns_or_prefix: attrs_with_ns,
                foreign_elements: foreign,
                comments: 0,
                doctypes: 0,
                processing_instructions: 0,
            },
            events: self.events,
            scripts_inert: self.fragment,
        }
    }

    // -- Dom access helpers (lock, do one thing, drop) ---------------------

    fn tag_of(&self, id: NodeId) -> Option<String> {
        self.dom.lock().unwrap().tag(id).map(|s| s.to_string())
    }

    fn ns_of(&self, id: NodeId) -> String {
        self.names
            .get(&id)
            .map(|q| q.ns.clone())
            .unwrap_or_default()
    }

    fn parent_of(&self, id: NodeId) -> Option<NodeId> {
        self.dom.lock().unwrap().get(id).and_then(|n| n.parent)
    }

    fn children_of(&self, id: NodeId) -> Vec<NodeId> {
        self.dom.lock().unwrap().children(id).to_vec()
    }

    fn attr_of(&self, id: NodeId, name: &str) -> Option<String> {
        self.dom
            .lock()
            .unwrap()
            .attr(id, name)
            .map(|s| s.to_string())
    }

    pub fn current_node(&self) -> Option<NodeId> {
        self.stack.last().copied()
    }

    /// Adjusted current node: in a fragment whose stack holds only
    /// the synthetic `<html>`, insertions go under the context
    /// element instead.
    fn adjusted_current_node(&self) -> Option<NodeId> {
        if self.fragment && !self.fragment_detached {
            if let Some(html) = self.fragment_html {
                if self.stack.len() == 1 && self.stack[0] == html {
                    return self.fragment_ctx;
                }
            }
        }
        self.current_node()
    }

    /// True when the stack is back to just the synthetic html
    /// element of a fragment parse.
    fn at_fragment_root(&self) -> bool {
        if self.fragment && !self.fragment_detached {
            if let Some(html) = self.fragment_html {
                return self.stack.len() == 1 && self.stack[0] == html;
            }
        }
        false
    }

    fn is_html_ns(&self, id: NodeId) -> bool {
        let ns = self.ns_of(id);
        ns.is_empty() || ns == NS_HTML
    }

    // -- element creation / insertion ---------------------------------------

    fn create_element_raw(&mut self, tag: &str, attrs: Vec<(String, String)>, ns: &str) -> NodeId {
        let id = {
            let mut dom = self.dom.lock().unwrap();
            // Stage 15 (M7 Phase 1, Rec 1): the namespace lives on
            // the arena node now, so `parse()` callers (layout,
            // paint, JS, scanner) can read it without the side maps.
            dom.create_element_with_attrs_ns(tag, attrs.clone(), ns)
        };
        let qname = if ns.is_empty() {
            QName::html(tag)
        } else {
            QName::new(ns, tag)
        };
        let full: Vec<(QName, String)> = if ns == NS_SVG || ns == NS_MATHML {
            // Foreign elements: xlink:/xml:/xmlns: attributes carry
            // their namespace and prefix (spec: adjust foreign
            // attributes).
            let t = TagToken {
                name: tag.to_string(),
                attrs: attrs.clone(),
                self_closing: false,
            };
            foreign_attrs(&t).1
        } else {
            attrs
                .iter()
                .map(|(n, v)| (QName::html(n), v.clone()))
                .collect()
        };
        self.names.insert(id, qname);
        self.full_attrs.insert(id, full);
        if tag == "base" {
            if self.base_href.is_none() {
                if let Some((_, href)) = attrs.iter().find(|(n, _)| n == "href") {
                    if !href.trim().is_empty() {
                        self.base_href = Some(href.clone());
                    }
                }
            }
        }
        self.events.push(DomEvent::ElementCreated {
            node: id,
            tag: tag.to_string(),
        });
        id
    }

    /// Create an element node (not yet inserted).
    fn create_element(&mut self, tag: &TagToken, ns: &str) -> NodeId {
        self.create_element_raw(&tag.name, tag.attrs.clone(), ns)
    }

    /// Shallow-clone an element node (same tag/attrs/namespace).
    fn clone_element(&mut self, id: NodeId) -> NodeId {
        let (tag, attrs, ns) = {
            let dom = self.dom.lock().unwrap();
            let tag = dom.tag(id).unwrap_or("").to_string();
            let attrs = dom.get(id).map(|n| n.attrs.clone()).unwrap_or_default();
            let ns = dom.namespace(id).unwrap_or("").to_string();
            (tag, attrs, ns)
        };
        let full = self.full_attrs.get(&id).cloned().unwrap_or_default();
        let qname = self.names.get(&id).cloned().unwrap_or_else(|| QName::html(&tag));
        let nid = {
            let mut dom = self.dom.lock().unwrap();
            dom.create_element_with_attrs_ns(&tag, attrs, &ns)
        };
        self.names.insert(nid, qname);
        self.full_attrs.insert(nid, full);
        self.events.push(DomEvent::ElementCreated {
            node: nid,
            tag,
        });
        nid
    }

    /// Appropriate place for insertion, honouring foster parenting
    /// and template contents. Returns (parent, before).
    fn appropriate_place(&self) -> (NodeId, Option<NodeId>) {
        let cur = match self.adjusted_current_node() {
            Some(c) => c,
            None => return (self.dom.lock().unwrap().root().unwrap(), None),
        };
        if let Some(&contents) = self.template_contents.get(&cur) {
            return (contents, None);
        }
        let tag = self.tag_of(cur).unwrap_or_default();
        // Foster parenting applies only while a table mode asked
        // for it (`insert_from_table`) and only for HTML-namespace
        // table elements: a MathML/SVG <tr> (e.g. inside <math>)
        // is not a table context — its children stay inside it.
        if self.insert_from_table
            && self.is_html_ns(cur)
            && matches!(tag.as_str(), "table" | "tbody" | "tfoot" | "thead" | "tr")
        {
            return self.foster_parent_place();
        }
        (cur, None)
    }

    fn foster_parent_place(&self) -> (NodeId, Option<NodeId>) {
        // Spec, foster parenting: if the last template element in
        // the stack is above the last table element (or there is
        // no table at all), the foster parent is that template's
        // contents, appended at the end. This is what keeps a
        // <div> after a <tr> inside template contents as a
        // sibling of the <tr>, instead of foster-parenting it
        // out to the document root.
        let last_template = self
            .stack
            .iter()
            .rposition(|&n| self.tag_of(n).as_deref() == Some("template"));
        let last_table = self
            .stack
            .iter()
            .rposition(|&n| self.tag_of(n).as_deref() == Some("table"));
        if let Some(tpos) = last_template {
            if last_table.is_none_or(|tb| tpos > tb) {
                if let Some(&contents) = self.template_contents.get(&self.stack[tpos]) {
                    return (contents, None);
                }
            }
        }
        let table = self
            .stack
            .iter()
            .rev()
            .find(|&&n| self.tag_of(n).as_deref() == Some("table"))
            .copied();
        let Some(table) = table else {
            // No table element in the stack: the foster parent is
            // the first element on the stack. In a fragment that
            // is the context element itself (the reference keeps
            // it as the stack bottom), whatever its tag — unless
            // the fragment is detached, in which case the context
            // subtree is gone and the node falls outside it.
            if self.fragment && !self.fragment_detached {
                if let Some(ctx) = self.fragment_ctx {
                    return (ctx, None);
                }
            }
            return (self.dom.lock().unwrap().root().unwrap(), None);
        };
        if let Some(parent) = self.parent_of(table) {
            (parent, Some(table))
        } else {
            let pos = self.stack.iter().position(|&n| n == table).unwrap_or(0);
            if pos > 0 {
                (self.stack[pos - 1], None)
            } else {
                (self.dom.lock().unwrap().root().unwrap(), None)
            }
        }
    }

    /// Append `child` under `parent`, redirecting into a
    /// template element's contents when `parent` is a template:
    /// the Dom itself does not redirect, so every insertion site
    /// that can target a template must do this (see
    /// `appropriate_place`). The adoption agency appends to the
    /// common ancestor / furthest block directly, so it needs
    /// this; otherwise a node ends up as a direct child of the
    /// template node, outside its contents.
    fn append_child_template_aware(&mut self, parent: NodeId, child: NodeId) {
        if let Some(&contents) = self.template_contents.get(&parent) {
            self.dom.lock().unwrap().append_child(contents, child);
        } else {
            self.dom.lock().unwrap().append_child(parent, child);
        }
    }

    /// Insert `node` at the appropriate place.
    fn insert_node(&mut self, node: NodeId) {
        let (parent, before) = self.appropriate_place();
        let mut dom = self.dom.lock().unwrap();
        match before {
            Some(sib) => dom.insert_before(parent, sib, node),
            None => dom.append_child(parent, node),
        }
    }

    /// Insert an element under the current node (bypassing foster
    /// parenting — table-structure elements belong *inside* the
    /// table, they are never foster-parented).
    fn insert_element_in_table(&mut self, tag: &TagToken, ns: &str) -> NodeId {
        let mut cur = self
            .adjusted_current_node()
            .unwrap_or_else(|| self.dom.lock().unwrap().root().unwrap());
        if let Some(&contents) = self.template_contents.get(&cur) {
            cur = contents;
        }
        let id = self.create_element(tag, ns);
        self.dom.lock().unwrap().append_child(cur, id);
        if !is_void(&tag.name) {
            self.stack.push(id);
        }
        id
    }

    /// Insert an element for a start tag at the appropriate place
    /// and push it on the stack (unless void).
    fn insert_element(&mut self, tag: &TagToken, ns: &str) -> NodeId {
        let id = self.create_element(tag, ns);
        self.insert_node(id);
        if !is_void(&tag.name) {
            self.stack.push(id);
        }
        id
    }

    /// Insert text at the appropriate place, merging with an
    /// adjacent text node.
    fn insert_text(&mut self, data: &str) {
        if data.is_empty() {
            return;
        }
        let (parent, before) = self.appropriate_place();
        self.insert_text_at(parent, before, data);
    }

    fn insert_text_at(&mut self, parent: NodeId, before: Option<NodeId>, data: &str) {
        let mut dom = self.dom.lock().unwrap();
        match before {
            Some(sib) => {
                let kids = dom.children(parent).to_vec();
                let pos = kids.iter().position(|&c| c == sib).unwrap_or(kids.len());
                let prev = if pos > 0 { Some(kids[pos - 1]) } else { None };
                match prev {
                    Some(t)
                        if matches!(dom.get(t), Some(n) if matches!(n.kind, NodeKind::Text(_))) =>
                    {
                        dom.append_to_text(t, data);
                    }
                    _ => {
                        let id = dom.create_text(data);
                        dom.insert_before(parent, sib, id);
                    }
                }
            }
            None => {
                let last = dom.children(parent).last().copied();
                match last {
                    Some(t)
                        if matches!(dom.get(t), Some(n) if matches!(n.kind, NodeKind::Text(_))) =>
                    {
                        dom.append_to_text(t, data);
                    }
                    _ => {
                        let id = dom.create_text(data);
                        dom.append_child(parent, id);
                    }
                }
            }
        }
    }

    fn insert_comment(&mut self, data: &str) {
        let (parent, before) = self.appropriate_place();
        let mut dom = self.dom.lock().unwrap();
        let id = dom.create_comment(data);
        match before {
            Some(sib) => dom.insert_before(parent, sib, id),
            None => dom.append_child(parent, id),
        }
    }

    fn insert_comment_at(&mut self, parent: NodeId, data: &str) {
        let mut dom = self.dom.lock().unwrap();
        let id = dom.create_comment(data);
        dom.append_child(parent, id);
    }

    fn pop(&mut self) -> Option<NodeId> {
        let n = self.stack.pop()?;
        self.events.push(DomEvent::SubtreeComplete { node: n });
        // A popped <option> element's children are final: refresh
        // the select's <selectedcontent> from the selected option.
        // Hooking pop() covers every close path at once — explicit
        // </option>, a following <option>/<optgroup> start tag,
        // </optgroup>, </select>, and EOF (process_eof pops the
        // stack through here).
        if self.tag_of(n).as_deref() == Some("option") {
            self.update_selectedcontent(n);
        }
        Some(n)
    }

    /// Deep-clone a node and its subtree (elements keep tag,
    /// attributes and namespace via [`Self::clone_element`]; text
    /// and comments are copied as-is).
    fn deep_clone_node(&mut self, node: NodeId) -> NodeId {
        let kind = {
            let dom = self.dom.lock().unwrap();
            dom.get(node).map(|n| n.kind.clone())
        };
        match kind {
            Some(NodeKind::Text(data)) => self.dom.lock().unwrap().create_text(&data),
            Some(NodeKind::Comment(data)) => {
                self.dom.lock().unwrap().create_comment(&data)
            }
            Some(NodeKind::Element { .. }) => {
                let nid = self.clone_element(node);
                for kid in self.children_of(node) {
                    let ckid = self.deep_clone_node(kid);
                    self.dom.lock().unwrap().append_child(nid, ckid);
                }
                nid
            }
            _ => self.dom.lock().unwrap().create_text(""),
        }
    }

    /// Descendants of `root` in document order whose tag is `tag`.
    fn descendants_with_tag(&self, root: NodeId, tag: &str) -> Vec<NodeId> {
        fn walk(b: &TreeBuilder, n: NodeId, tag: &str, out: &mut Vec<NodeId>) {
            for kid in b.children_of(n) {
                if b.tag_of(kid).as_deref() == Some(tag) {
                    out.push(kid);
                }
                walk(b, kid, tag, out);
            }
        }
        let mut out = Vec::new();
        walk(self, root, tag, &mut out);
        out
    }

    /// Refresh a select's `<selectedcontent>` element from its
    /// selected option, per the HTML spec's selectedcontent
    /// behaviour: the selectedcontent displays a clone of the
    /// selected option's children. The selected option is the
    /// last option with a `selected` attribute, else the first
    /// option in tree order. Runs when an option element closes
    /// (see [`Self::pop`]); a no-op when the option has no select
    /// ancestor or the select has no selectedcontent descendant.
    fn update_selectedcontent(&mut self, option: NodeId) {
        // Find the select ancestor of the option.
        let mut cur = self.parent_of(option);
        let mut select = None;
        while let Some(n) = cur {
            if self.tag_of(n).as_deref() == Some("select") {
                select = Some(n);
                break;
            }
            cur = self.parent_of(n);
        }
        let Some(select) = select else {
            return;
        };
        let options = self.descendants_with_tag(select, "option");
        if options.is_empty() {
            return;
        }
        let selected = options
            .iter()
            .rev()
            .find(|&&o| {
                self.dom
                    .lock()
                    .unwrap()
                    .get(o)
                    .map(|n| n.attrs.iter().any(|(a, _)| a == "selected"))
                    .unwrap_or(false)
            })
            .copied()
            .unwrap_or(options[0]);
        let Some(sc) = self
            .descendants_with_tag(select, "selectedcontent")
            .into_iter()
            .next()
        else {
            return;
        };
        // Replace the selectedcontent's children with deep clones
        // of the selected option's children.
        for kid in self.children_of(sc) {
            self.dom.lock().unwrap().remove_child(sc, kid);
        }
        for kid in self.children_of(selected) {
            let ckid = self.deep_clone_node(kid);
            self.dom.lock().unwrap().append_child(sc, ckid);
        }
    }

    /// Pop elements until `tag` (inclusive) is popped; the `<html>`
    /// element is never popped.
    fn pop_until_tag(&mut self, tag: &str) {
        loop {
            let Some(cur) = self.current_node() else { break };
            // ASCII case-insensitive: a foreign element's stored
            // tag can be camelCase ("foreignObject") while the
            // token name is lowercased ("foreignobject").
            if self
                .tag_of(cur)
                .map(|t| t.eq_ignore_ascii_case(tag))
                .unwrap_or(false)
            {
                self.pop();
                break;
            }
            if self.tag_of(cur).as_deref() == Some("html") {
                break;
            }
            self.pop();
        }
    }
    /// Pop until an HTML-namespace element with the given tag has
    /// been popped. A foreign element merely named the same (an SVG
    /// <td>, a MathML <table>) is not the element these operations
    /// mean: popping stops at it and leaves the real HTML element
    /// open (namespace-sensitivity #0: `</td>` in a cell stopped at
    /// an SVG <td> and the following text landed inside the svg).
    fn pop_until_html_tag(&mut self, tag: &str) {
        loop {
            let Some(cur) = self.current_node() else { break };
            let html = self.is_html_ns(cur);
            if html
                && self
                    .tag_of(cur)
                    .map(|t| t.eq_ignore_ascii_case(tag))
                    .unwrap_or(false)
            {
                self.pop();
                break;
            }
            self.pop();
        }
    }

    /// Pop until a node satisfying `pred` is popped. Returns true if found.
    fn pop_until(&mut self, pred: impl Fn(&str) -> bool) -> bool {
        loop {
            let Some(cur) = self.current_node() else {
                return false;
            };
            let t = self.tag_of(cur).unwrap_or_default();
            if pred(&t) {
                self.pop();
                return true;
            }
            if t == "html" {
                return false;
            }
            self.pop();
        }
    }

    // -- scope ---------------------------------------------------------------

    /// Split a character token into a leading-whitespace prefix and
    /// the remainder. Modes that treat whitespace and non-whitespace
    /// differently (before head, in head, after head, in column
    /// group, after frameset) must process the prefix by the
    /// whitespace rule and reprocess the remainder — a mixed token
    /// such as `" a "` must not be treated as a single
    /// non-whitespace token (tests19 #82, tests1 #50, domjs #36,
    /// tests6 #7).
    fn split_leading_ws(data: &str) -> (&str, &str) {
        let idx = data
            .find(|c: char| !c.is_ascii_whitespace())
            .unwrap_or(data.len());
        data.split_at(idx)
    }

    fn has_in_scope(&self, tag: &str, scope: Scope) -> bool {
        for &node in self.stack.iter().rev() {
            let t = self.tag_of(node).unwrap_or_default();
            if t == tag {
                return true;
            }
            if self.is_scope_boundary(&t, scope, node) {
                return false;
            }
        }
        false
    }

    fn has_in_scope_any(&self, tags: &[&str], scope: Scope) -> bool {
        for &node in self.stack.iter().rev() {
            let t = self.tag_of(node).unwrap_or_default();
            if tags.contains(&t.as_str()) {
                return true;
            }
            if self.is_scope_boundary(&t, scope, node) {
                return false;
            }
        }
        false
    }

    fn is_scope_boundary(&self, tag: &str, scope: Scope, node: NodeId) -> bool {
        // Spec 13.2.4.2: the table scope is ONLY
        // html/table/template — the default-scope base list does
        // NOT apply there. A td/th must not stop a table-scope
        // search, or `</tbody>` inside a cell never finds its tbody
        // (webkit02 #5: the text after it stayed in the cell
        // instead of being foster-parented before the table).
        if let Scope::Table = scope {
            return self.is_html_ns(node) && matches!(tag, "html" | "table" | "template");
        }
        // The base list (default scope, extended by list-item and
        // button scopes) is applet, caption, html, table, td, th,
        // marquee, object, select, template, MathML
        // mi/mo/mn/ms/mtext/annotation-xml, SVG
        // foreignObject/desc/title. The MathML text integration
        // points matter: a <p> outside a <math><mi> is NOT in scope
        // from inside the mi, so a <p> start tag there nests inside
        // the mi instead of closing the outer p.
        let ns = self.ns_of(node);
        let base = matches!(
            tag,
            "applet" | "caption" | "html" | "table" | "td" | "th" | "marquee" | "object"
                | "select" | "template"
        ) || (ns == NS_MATHML
            && matches!(
                tag,
                "mi" | "mo" | "mn" | "ms" | "mtext" | "annotation-xml"
            ))
            || (ns == NS_SVG
                && matches!(tag, "foreignObject" | "foreignobject" | "desc" | "title"));
        if base {
            return true;
        }
        match scope {
            Scope::Default => false,
            Scope::ListItem => matches!(tag, "ol" | "ul"),
            Scope::Button => tag == "button",
            Scope::Table => false,
        }
    }

    // -- implied end tags ------------------------------------------------------

    fn generate_implied_end_tags(&mut self, except: Option<&str>) {
        loop {
            let Some(cur) = self.current_node() else { break };
            let t = self.tag_of(cur).unwrap_or_default();
            if Some(t.as_str()) == except {
                break;
            }
            if matches!(
                t.as_str(),
                "dd" | "dt" | "li" | "optgroup" | "option" | "p" | "rb" | "rp" | "rt" | "rtc"
            ) {
                self.pop();
            } else {
                break;
            }
        }
    }

    fn generate_implied_end_tags_thoroughly(&mut self) {
        loop {
            let Some(cur) = self.current_node() else { break };
            let t = self.tag_of(cur).unwrap_or_default();
            if matches!(
                t.as_str(),
                "caption" | "colgroup" | "dd" | "dt" | "li" | "optgroup" | "option" | "p" | "rb"
                    | "rp" | "rt" | "rtc" | "tbody" | "td" | "tfoot" | "th" | "thead" | "tr"
            ) {
                self.pop();
            } else {
                break;
            }
        }
    }

    fn close_p_element(&mut self) {
        // Spec 13.2.6.4.7, "close a p element": generate implied
        // end tags except for p, then repeatedly pop until a p
        // element has been popped. The implied generation stops
        // at a formatting element sitting on top of the p (e.g.
        // `<p><b><div>`), which this loop then pops along with
        // the p — the old code only popped when p was already
        // the current node, so the div nested inside the b.
        self.generate_implied_end_tags(Some("p"));
        while let Some(cur) = self.current_node() {
            let is_p = self.tag_of(cur).as_deref() == Some("p");
            self.pop();
            if is_p {
                break;
            }
        }
    }

    fn close_p_if_in_button_scope(&mut self) {
        if self.has_in_scope("p", Scope::Button) {
            self.close_p_element();
        }
    }

    // -- active formatting elements --------------------------------------------

    fn afe_push(&mut self, node: NodeId) {
        let (tag, attrs, ns) = {
            let dom = self.dom.lock().unwrap();
            (
                dom.tag(node).unwrap_or("").to_string(),
                dom.get(node).map(|n| n.attrs.clone()).unwrap_or_default(),
                self.ns_of(node),
            )
        };
        let mut count = 0;
        for e in self.afe.iter().rev() {
            match e {
                AfeEntry::Marker => break,
                AfeEntry::Element(n) => {
                    let (t2, a2, ns2) = {
                        let dom = self.dom.lock().unwrap();
                        (
                            dom.tag(*n).unwrap_or("").to_string(),
                            dom.get(*n).map(|x| x.attrs.clone()).unwrap_or_default(),
                            self.ns_of(*n),
                        )
                    };
                    if t2 == tag && a2 == attrs && ns2 == ns {
                        count += 1;
                    }
                }
            }
        }
        if count >= 3 {
            let mut idx = None;
            for (i, e) in self.afe.iter().enumerate() {
                if let AfeEntry::Element(n) = e {
                    let (t2, a2, ns2) = {
                        let dom = self.dom.lock().unwrap();
                        (
                            dom.tag(*n).unwrap_or("").to_string(),
                            dom.get(*n).map(|x| x.attrs.clone()).unwrap_or_default(),
                            self.ns_of(*n),
                        )
                    };
                    if t2 == tag && a2 == attrs && ns2 == ns {
                        idx = Some(i);
                        break;
                    }
                }
            }
            if let Some(i) = idx {
                self.afe.remove(i);
            }
        }
        self.afe.push(AfeEntry::Element(node));
    }

    fn afe_insert_marker(&mut self) {
        self.afe.push(AfeEntry::Marker);
    }

    fn afe_remove(&mut self, node: NodeId) {
        if let Some(i) = self
            .afe
            .iter()
            .position(|e| matches!(e, AfeEntry::Element(n) if *n == node))
        {
            self.afe.remove(i);
        }
    }

    fn afe_contains(&self, node: NodeId) -> bool {
        self.afe
            .iter()
            .any(|e| matches!(e, AfeEntry::Element(n) if *n == node))
    }

    fn afe_index(&self, node: NodeId) -> Option<usize> {
        self.afe
            .iter()
            .position(|e| matches!(e, AfeEntry::Element(n) if *n == node))
    }

    /// Reconstruct the active formatting elements.
    fn reconstruct_afe(&mut self) {
        if self.afe.is_empty() {
            return;
        }
        let mut i = self.afe.len() - 1;
        match &self.afe[i] {
            AfeEntry::Marker => return,
            AfeEntry::Element(n) => {
                if self.stack.contains(n) {
                    return;
                }
            }
        }
        while i > 0 {
            match &self.afe[i - 1] {
                AfeEntry::Marker => break,
                AfeEntry::Element(n) => {
                    if self.stack.contains(n) {
                        break;
                    }
                }
            }
            i -= 1;
        }
        loop {
            let entry = match self.afe.get(i) {
                Some(AfeEntry::Element(n)) => *n,
                _ => break,
            };
            let clone = self.clone_element(entry);
            self.insert_node(clone);
            self.stack.push(clone);
            self.afe[i] = AfeEntry::Element(clone);
            i += 1;
            if i >= self.afe.len() {
                break;
            }
        }
    }

    /// Clear the active formatting elements up to (and including)
    /// the last marker.
    fn clear_afe_to_marker(&mut self) {
        while let Some(e) = self.afe.pop() {
            if matches!(e, AfeEntry::Marker) {
                break;
            }
        }
    }

    // -- adoption agency -------------------------------------------------------

    /// Spec 13.2.6.4.7, "any other end tag": walk down from the
    /// current node. A matching tag generates implied end tags
    /// (excluding itself) and pops until it is popped; a special
    /// element that does not match makes the token ignored entirely
    /// (e.g. `</span>` with a <button> on top leaves the button
    /// open, and later text lands inside it).
    fn generic_end_tag(&mut self, name: &str) {
        let mut idx = self.stack.len();
        while idx > 0 {
            idx -= 1;
            let n = self.stack[idx];
            let t = self.tag_of(n).unwrap_or_default();
            // ASCII case-insensitive: without this,
            // `</foreignObject>` never matches the camelCase
            // element and is ignored (the special-element stop
            // below fires first).
            if t.eq_ignore_ascii_case(name) {
                self.generate_implied_end_tags(Some(name));
                while self.stack.len() > idx {
                    self.pop();
                }
                break;
            }
            if is_special(&t, self.is_html_ns(n)) {
                self.error("end tag ignored: special element on top");
                break;
            }
        }
    }

    fn adoption_agency(&mut self, subject: &str) {
        // Spec step 2 (parse5's callAdoptionAgency fast path): if
        // the current node is an HTML element with the same tag name
        // and it is NOT in the list of active formatting elements,
        // pop it and return. This is what closes a <b> that Noah's
        // Ark already dropped from the list (adoption01 #15: four
        // nested <b>s, four </b>s — the fourth closes the outermost
        // b, which the Ark removed from the list on the fourth
        // insertion).
        if let Some(cur) = self.current_node() {
            if self.tag_of(cur).as_deref() == Some(subject)
                && self.is_html_ns(cur)
                && !self.afe_contains(cur)
            {
                self.pop();
                return;
            }
        }
        for _outer in 0..8 {
            let mut fmt_idx: Option<usize> = None;
            for (i, e) in self.afe.iter().enumerate().rev() {
                match e {
                    AfeEntry::Marker => break,
                    AfeEntry::Element(n) => {
                        if self.tag_of(*n).as_deref() == Some(subject) {
                            fmt_idx = Some(i);
                            break;
                        }
                    }
                }
            }
            let Some(fmt_idx) = fmt_idx else {
                // No formatting element after the last marker (a
                // marker from an applet/marquee/object can hide an
                // earlier one): fall back to the generic end-tag
                // walk, which pops until the matching element on the
                // stack (adoption02 #2: the second <nobr> must close
                // the first, which a marquee marker hid from the
                // list search).
                self.generic_end_tag(subject);
                return;
            };
            let fmt = match &self.afe[fmt_idx] {
                AfeEntry::Element(n) => *n,
                _ => unreachable!(),
            };
            let Some(fmt_stack_pos) = self.stack.iter().position(|&n| n == fmt) else {
                self.afe.remove(fmt_idx);
                return;
            };
            if !self.has_in_scope(subject, Scope::Default) {
                self.error("adoption agency: formatting element not in scope");
                return;
            }
            let furthest_pos = self.stack[fmt_stack_pos + 1..]
                .iter()
                .position(|&n| {
                    let t = self.tag_of(n).unwrap_or_default();
                    is_special(&t, self.is_html_ns(n))
                })
                .map(|p| fmt_stack_pos + 1 + p);
            let Some(furthest_pos) = furthest_pos else {
                while self.stack.len() > fmt_stack_pos {
                    self.pop();
                }
                self.afe_remove(fmt);
                return;
            };
            let furthest = self.stack[furthest_pos];
            let common_ancestor = self.stack[fmt_stack_pos - 1];
            let mut bookmark = fmt_idx;
            let mut node_idx = furthest_pos;
            let mut last = furthest;
            let mut j = 0usize;
            loop {
                j += 1;
                if node_idx == 0 {
                    break;
                }
                node_idx -= 1;
                let node = self.stack[node_idx];
                if node == fmt {
                    break;
                }
                if j > 3 && self.afe_contains(node) {
                    self.afe_remove(node);
                    if let Some(p) = self.stack.iter().position(|&n| n == node) {
                        self.stack.remove(p);
                        if p < node_idx {
                            node_idx = node_idx.saturating_sub(1);
                        }
                    }
                    continue;
                }
                if !self.afe_contains(node) {
                    if let Some(p) = self.stack.iter().position(|&n| n == node) {
                        self.stack.remove(p);
                        if p < node_idx {
                            node_idx = node_idx.saturating_sub(1);
                        }
                    }
                    continue;
                }
                let clone = self.clone_element(node);
                if let Some(ai) = self.afe_index(node) {
                    self.afe[ai] = AfeEntry::Element(clone);
                }
                if let Some(p) = self.stack.iter().position(|&n| n == node) {
                    self.stack[p] = clone;
                }
                let node = clone;
                if last == furthest {
                    bookmark = self.afe_index(node).map(|i| i + 1).unwrap_or(bookmark);
                }
                self.append_child_template_aware(node, last);
                last = node;
            }
            {
                // In a fragment, the stack bottom is the synthetic
                // html element, which stands in for the context
                // element; the reference keeps the context element
                // itself on the stack. A node the agency appends to
                // the common ancestor must therefore land under the
                // context element, not under the synthetic root —
                // otherwise it is discarded with the synthetic root
                // at the end of the fragment parse (webkit02
                // #11/#16/#17: the <aside> the agency moved out of
                // the nested <foo>s vanished).
                let ancestor = if Some(common_ancestor) == self.fragment_html {
                    self.fragment_ctx.unwrap_or(common_ancestor)
                } else {
                    common_ancestor
                };
                let ancestor_tag = self.tag_of(ancestor).unwrap_or_default();
                if matches!(
                    ancestor_tag.as_str(),
                    "table" | "tbody" | "tfoot" | "thead" | "tr"
                ) {
                    let (parent, before) = self.foster_parent_place();
                    let mut dom = self.dom.lock().unwrap();
                    match before {
                        Some(sib) => dom.insert_before(parent, sib, last),
                        None => dom.append_child(parent, last),
                    }
                } else {
                    self.append_child_template_aware(ancestor, last);
                }
            }
            let fmt_clone = self.clone_element(fmt);
            {
                let kids = self.children_of(furthest);
                let mut dom = self.dom.lock().unwrap();
                for kid in kids {
                    dom.append_child(fmt_clone, kid);
                }
            }
            // The furthest block can itself be a template element
            // (it is in the special list), so route through the
            // template-aware append.
            self.append_child_template_aware(furthest, fmt_clone);
            self.afe_remove(fmt);
            let at = bookmark.min(self.afe.len());
            self.afe.insert(at, AfeEntry::Element(fmt_clone));
            if let Some(p) = self.stack.iter().position(|&n| n == fmt) {
                self.stack.remove(p);
            }
            // Spec, adoption agency: remove the formatting
            // element from the stack and insert the clone
            // immediately below the furthest block. In the spec's
            // terms the current node is the bottommost node, so
            // "below" is toward the current node — index p + 1.
            if let Some(p) = self.stack.iter().position(|&n| n == furthest) {
                self.stack.insert(p + 1, fmt_clone);
            } else {
                self.stack.push(fmt_clone);
            }
        }
    }

    // -- misc helpers -----------------------------------------------------------

    /// Merge attributes onto an existing element (a second
    /// <html> / <body> start tag): updates the Dom AND the
    /// `full_attrs` side map the serializer reads. The Dom-only
    /// `add_attrs_if_missing` left merged attributes invisible
    /// in the output.
    fn merge_attrs(&mut self, node: NodeId, attrs: &[(String, String)]) {
        self.dom
            .lock()
            .unwrap()
            .add_attrs_if_missing(node, attrs.to_vec());
        let full = self.full_attrs.entry(node).or_default();
        for (name, value) in attrs {
            if !full
                .iter()
                .any(|(q, _)| q.ns.is_empty() && q.local == *name)
            {
                full.push((QName::html(name), value.clone()));
            }
        }
    }

    fn ensure_html_element(&mut self, tag: &TagToken) -> NodeId {
        if let Some(h) = self.html_node {
            self.merge_attrs(h, &tag.attrs);
            return h;
        }
        let root = self.dom.lock().unwrap().root().unwrap();
        let id = self.create_element(tag, "");
        self.dom.lock().unwrap().append_child(root, id);
        self.stack.push(id);
        self.html_node = Some(id);
        id
    }

    fn ensure_head_element(&mut self, tag: &TagToken) -> NodeId {
        if let Some(h) = self.head_node {
            return h;
        }
        let _ = self.ensure_html_element(&TagToken {
            name: "html".into(),
            attrs: Vec::new(),
            self_closing: false,
        });
        let id = self.create_element(tag, "");
        self.insert_node(id);
        self.stack.push(id);
        self.head_node = Some(id);
        self.head_pointer = Some(id);
        id
    }

    fn ensure_body_element(&mut self, tag: &TagToken) -> NodeId {
        if let Some(b) = self.body_node {
            self.merge_attrs(b, &tag.attrs);
            return b;
        }
        let _ = self.ensure_head_element(&TagToken {
            name: "head".into(),
            attrs: Vec::new(),
            self_closing: false,
        });
        if let Some(h) = self.head_node {
            if let Some(p) = self.stack.iter().position(|&n| n == h) {
                self.stack.remove(p);
            }
        }
        let id = self.create_element(tag, "");
        self.insert_node(id);
        self.stack.push(id);
        self.body_node = Some(id);
        self.mode = Mode::InBody;
        id
    }

    /// Reset the insertion mode appropriately for the current node
    /// (used when leaving table/select/template contexts).
    fn reset_insertion_mode(&mut self) {
        if self.at_fragment_root() {
            self.mode = self.fragment_mode;
            return;
        }
        for (i, &node) in self.stack.iter().enumerate().rev() {
            let last = i == 0;
            let tag = self.tag_of(node).unwrap_or_default();
            let mode = match tag.as_str() {
                "select" => Some(if self.select_in_table_scope() {
                    Mode::InSelectInTable
                } else {
                    Mode::InSelect
                }),
                "td" | "th" if !last => Some(Mode::InCell),
                "tr" => Some(Mode::InRow),
                "tbody" | "thead" | "tfoot" => Some(Mode::InTableBody),
                "caption" => Some(Mode::InCaption),
                "colgroup" => Some(Mode::InColumnGroup),
                "table" => Some(Mode::InTable),
                // Spec 13.2.6.4.7, reset insertion mode: a
                // template element on the stack resumes in the
                // CURRENT template insertion mode (the top of
                // template_modes), not InTemplate itself — e.g.
                // after a nested template closes inside a
                // <tr> contents, the outer template resumes in
                // InTableBody so a following <td> gets its implied
                // <tr>.
                // Only an HTML-namespace template element selects
                // the current template insertion mode; a foreign
                // element merely named "template" (e.g. an SVG
                // <template>) is not a template element and is
                // skipped by the walk.
                "template" if self.is_html_ns(node) => Some(
                    self.template_modes
                        .last()
                        .copied()
                        .unwrap_or(Mode::InTemplate),
                ),
                "body" => Some(Mode::InBody),
                "frameset" => Some(Mode::InFrameset),
                "head" if !last => Some(Mode::InHead),
                "html" => Some(if self.head_node.is_none() {
                    Mode::BeforeHead
                } else {
                    Mode::AfterHead
                }),
                _ => None,
            };
            if let Some(m) = mode {
                self.mode = m;
                return;
            }
            if last {
                self.mode = Mode::InBody;
                return;
            }
        }
        self.mode = Mode::InBody;
    }
}

/// Void elements: never pushed on the stack, never closed.
pub fn is_void(tag: &str) -> bool {
    matches!(
        tag,
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input" | "link" | "meta"
            | "param" | "source" | "track" | "wbr"
    )
}

fn is_special(tag: &str, html_ns: bool) -> bool {
    if !html_ns {
        return matches!(
            tag,
            "mi" | "mo" | "mn" | "ms" | "mtext" | "annotation-xml" | "foreignObject"
                | "foreignobject" | "desc" | "title"
        );
    }
    matches!(
        tag,
        "address" | "applet" | "area" | "article" | "aside" | "base" | "basefont" | "bgsound"
            | "blockquote" | "body" | "br" | "button" | "caption" | "center" | "col" | "colgroup"
            | "dd" | "details" | "dir" | "div" | "dl" | "dt" | "embed" | "fieldset" | "figcaption"
            | "figure" | "footer" | "form" | "frame" | "frameset" | "h1" | "h2" | "h3" | "h4"
            | "h5" | "h6" | "head" | "header" | "hgroup" | "hr" | "html" | "iframe" | "img"
            | "input" | "li" | "link" | "listing" | "main" | "marquee" | "menu" | "meta" | "nav"
            | "noembed" | "noframes" | "noscript" | "object" | "ol" | "p" | "plaintext" | "pre"
            | "script" | "search" | "section" | "select" | "source" | "style" | "summary"
            | "table" | "tbody" | "td" | "template" | "textarea" | "tfoot" | "th" | "thead"
            | "title" | "tr" | "track" | "ul" | "wbr" | "xmp"
    )
}

// ===========================================================================
// Token dispatch + insertion modes (part 2)
// ===========================================================================

impl TreeBuilder {
    /// Process one token. `tokenizer` is handed in so the builder
    /// can cancel a pending raw-text switch (foreign content).
    /// Returns `ProcessOutcome::Script` when a `<script>` element
    /// completed.
    pub fn process(&mut self, token: Token, tokenizer: &mut Tokenizer) -> ProcessOutcome {
        // In a fragment (other than an `html` context), EOF
        // stops parsing outright: the modes' "anything else" arms
        // would otherwise create implied html/head/body elements
        // under the context element (tests4 #6: a head-context
        // fragment gained a spurious <head> and <body>). The driver
        // still runs process_eof() afterwards to pop the stack and
        // clear the template modes. An `html` context is exempt:
        // there the synthetic root stands in for the context element
        // and the document-like cascade creates the head and body
        // that are reparented under it (tests_innerHTML_1 #80: an
        // empty html-context fragment yields <head> and <body>).
        if let Token::Eof = token {
            if let Some(ctx) = self.fragment_ctx {
                if self.tag_of(ctx).as_deref() != Some("html") {
                    return ProcessOutcome::Continue;
                }
            }
        }
        if let Token::StartTag(_) = &token {
            if self.stack.len() >= MAX_DEPTH {
                self.error("nesting depth cap reached");
                return ProcessOutcome::Continue;
            }
        }
        let mut token = token;
        // Foreign content (SVG/MathML) is dispatched before the
        // insertion mode, for every mode except Text (whose tokens
        // are raw/RCDATA/script text). If the foreign rules handle
        // the token, they win; otherwise the mode sees it.
        if self.mode != Mode::Text
            && matches!(
                token,
                Token::Character(_) | Token::StartTag(_) | Token::EndTag(_) | Token::Comment(_)
            )
            && self.in_foreign_content_for(&token)
        {
            if let Some(action) = self.foreign_dispatch(token.clone(), tokenizer) {
                return match action {
                    ModeAction::Done(o) => o,
                    ModeAction::Reprocess(t) => self.process(t, tokenizer),
                };
            }
        }
        if self.skip_newline {
            self.skip_newline = false;
            if let Token::Character(ref mut data) = token {
                if let Some(rest) = data.strip_prefix("\r\n") {
                    *data = rest.to_string();
                } else if let Some(rest) = data.strip_prefix('\n') {
                    *data = rest.to_string();
                }
                if data.is_empty() {
                    return ProcessOutcome::Continue;
                }
            }
        }
        loop {
            let outcome = match self.mode {
                Mode::Initial => self.mode_initial(token.clone(), tokenizer),
                Mode::BeforeHtml => self.mode_before_html(token.clone(), tokenizer),
                Mode::BeforeHead => self.mode_before_head(token.clone(), tokenizer),
                Mode::InHead => self.mode_in_head(token.clone(), tokenizer),
                Mode::InHeadNoscript => self.mode_in_head_noscript(token.clone(), tokenizer),
                Mode::AfterHead => self.mode_after_head(token.clone(), tokenizer),
                Mode::InBody => self.mode_in_body(token.clone(), tokenizer),
                Mode::Text => self.mode_text(token.clone(), tokenizer),
                Mode::InTable => self.mode_in_table(token.clone(), tokenizer),
                Mode::InTableText => self.mode_in_table_text(token.clone(), tokenizer),
                Mode::InCaption => self.mode_in_caption(token.clone(), tokenizer),
                Mode::InColumnGroup => self.mode_in_column_group(token.clone(), tokenizer),
                Mode::InTableBody => self.mode_in_table_body(token.clone(), tokenizer),
                Mode::InRow => self.mode_in_row(token.clone(), tokenizer),
                Mode::InCell => self.mode_in_cell(token.clone(), tokenizer),
                Mode::InSelect => self.mode_in_select(token.clone(), tokenizer),
                Mode::InSelectInTable => self.mode_in_select_in_table(token.clone(), tokenizer),
                Mode::InTemplate => self.mode_in_template(token.clone(), tokenizer),
                Mode::AfterBody => self.mode_after_body(token.clone(), tokenizer),
                Mode::InFrameset => self.mode_in_frameset(token.clone(), tokenizer),
                Mode::AfterFrameset => self.mode_after_frameset(token.clone(), tokenizer),
                Mode::AfterAfterBody => self.mode_after_after_body(token.clone(), tokenizer),
                Mode::AfterAfterFrameset => {
                    self.mode_after_after_frameset(token.clone(), tokenizer)
                }
            };
            match outcome {
                ModeAction::Done(o) => return o,
                ModeAction::Reprocess(t) => {
                    token = t;
                    continue;
                }
            }
        }
    }

    /// End of input: close everything still open.
    pub fn process_eof(&mut self) {
        if !self.quirks_set {
            self.quirks = QuirksMode::Quirks;
            self.quirks_set = true;
        }
        if !self.pending_table_chars.is_empty() {
            let chars = std::mem::take(&mut self.pending_table_chars);
            self.pending_table_chars_non_ws = false;
            self.insert_text(&chars);
        }
        while !self.template_modes.is_empty() {
            self.template_modes.pop();
        }
        while self.stack.len() > 1 {
            self.pop();
        }
        if !self.stack.is_empty() {
            self.pop();
        }
        self.mode = Mode::AfterAfterBody;
    }

    // -- quirks ---------------------------------------------------------------

    fn set_quirks_from_doctype(
        &mut self,
        name: &Option<String>,
        public_id: &Option<String>,
        system_id: &Option<String>,
        force_quirks: bool,
    ) {
        if self.quirks_set {
            return;
        }
        self.quirks_set = true;
        if force_quirks {
            self.quirks = QuirksMode::Quirks;
            return;
        }
        let name_ok = name.as_deref() == Some("html");
        if !name_ok {
            self.quirks = QuirksMode::Quirks;
            return;
        }
        let pub_id = public_id.clone().unwrap_or_default().to_ascii_lowercase();
        let sys_id = system_id.clone().unwrap_or_default().to_ascii_lowercase();
        if quirks_public_id(&pub_id)
            || (public_id.is_some()
                && (pub_id == "-//w3c//dtd html 4.01 frameset//en"
                    || pub_id == "-//w3c//dtd html 4.01 transitional//en")
                && system_id.is_none())
            || sys_id == "http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd"
        {
            self.quirks = QuirksMode::Quirks;
            return;
        }
        if limited_quirks_public_id(&pub_id)
            || sys_id == "http://www.w3.org/tr/html4/loose.dtd"
            || ((pub_id == "-//w3c//dtd html 4.01 frameset//en"
                || pub_id == "-//w3c//dtd html 4.01 transitional//en")
                && system_id.is_some())
        {
            self.quirks = QuirksMode::LimitedQuirks;
            return;
        }
        self.quirks = QuirksMode::NoQuirks;
    }

    // -- mode: initial ----------------------------------------------------------

    fn mode_initial(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                let root = self.dom.lock().unwrap().root().unwrap();
                self.insert_comment_at(root, &data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype {
                name,
                public_id,
                system_id,
                force_quirks,
            } => {
                self.set_quirks_from_doctype(&name, &public_id, &system_id, force_quirks);
                let root = self.dom.lock().unwrap().root().unwrap();
                let mut dom = self.dom.lock().unwrap();
                let dt = dom.create_doctype(
                    name.as_deref().unwrap_or(""),
                    public_id.as_deref().unwrap_or(""),
                    system_id.as_deref().unwrap_or(""),
                );
                dom.append_child(root, dt);
                drop(dom);
                self.mode = Mode::BeforeHtml;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => {
                self.set_quirks_from_doctype(&None, &None, &None, true);
                self.mode = Mode::BeforeHtml;
                ModeAction::Reprocess(Token::Eof)
            }
            _ => {
                self.set_quirks_from_doctype(&None, &None, &None, true);
                self.mode = Mode::BeforeHtml;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_before_html(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Doctype { .. } => {
                self.error("doctype after initial mode ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                let root = self.dom.lock().unwrap().root().unwrap();
                self.insert_comment_at(root, &data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => {
                self.ensure_html_element(tag);
                self.mode = Mode::BeforeHead;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "head" | "body" | "html" | "br") => {
                self.ensure_html_element(&TagToken {
                    name: "html".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::BeforeHead;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(_) => {
                self.error("stray end tag before html ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.ensure_html_element(&TagToken {
                    name: "html".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::BeforeHead;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_before_head(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // Whitespace and NULs are ignored before <head>
                // (a NUL must not create a body and break frameset).
                // A mixed token such as `" a "` drops its leading
                // whitespace/NUL prefix and reprocesses the rest
                // (tests19 #82: the space must not reach the body).
                let idx = s
                    .find(|c: char| !c.is_ascii_whitespace() && c != '\0')
                    .unwrap_or(s.len());
                if idx == s.len() {
                    ModeAction::Done(ProcessOutcome::Continue)
                } else if idx == 0 {
                    self.ensure_head_element(&TagToken {
                        name: "head".into(),
                        attrs: Vec::new(),
                        self_closing: false,
                    });
                    self.mode = Mode::InHead;
                    ModeAction::Reprocess(token)
                } else {
                    ModeAction::Reprocess(Token::Character(s[idx..].to_string()))
                }
            }
            Token::Comment(data) => {
                let (parent, _) = self.appropriate_place();
                self.insert_comment_at(parent, &data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in before-head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => {
                self.mode_in_body(token, _tz)
            }
            Token::StartTag(ref tag) if tag.name == "head" => {
                self.ensure_head_element(tag);
                self.mode = Mode::InHead;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "head" | "body" | "html" | "br") => {
                self.ensure_head_element(&TagToken {
                    name: "head".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::InHead;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(_) => {
                self.error("stray end tag before head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.ensure_head_element(&TagToken {
                    name: "head".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::InHead;
                ModeAction::Reprocess(token)
            }
        }
    }

    // -- mode: in head -----------------------------------------------------------

    fn mode_in_head(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // NULs are ignored in <head>; whitespace is inserted
                // into the head; anything else pops the head. A mixed
                // token splits: its leading whitespace goes into the
                // head and the remainder is reprocessed (tests1 #50).
                let d: String = s.chars().filter(|&c| c != '\0').collect();
                if d.is_empty() {
                    ModeAction::Done(ProcessOutcome::Continue)
                } else {
                    let (ws, rest) = Self::split_leading_ws(&d);
                    if !ws.is_empty() {
                        self.insert_text(ws);
                    }
                    if rest.is_empty() {
                        ModeAction::Done(ProcessOutcome::Continue)
                    } else {
                        self.pop_head();
                        self.mode = Mode::AfterHead;
                        ModeAction::Reprocess(Token::Character(rest.to_string()))
                    }
                }
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::StartTag(ref tag)
                if matches!(tag.name.as_str(), "base" | "basefont" | "bgsound" | "link" | "meta") =>
            {
                let id = self.create_element(tag, "");
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "title" => {
                self.insert_element(tag, "");
                self.original_mode = self.delegate_mode.unwrap_or(Mode::InHead);
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "noscript" => {
                if self.scripting {
                    self.insert_element(tag, "");
                    self.original_mode = self.delegate_mode.unwrap_or(Mode::InHead);
                    self.mode = Mode::Text;
                } else {
                    self.insert_element(tag, "");
                    self.mode = Mode::InHeadNoscript;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                // noframes is raw text unconditionally (unlike
                // noscript, which depends on scripting).
                self.insert_element(tag, "");
                self.original_mode = self.delegate_mode.unwrap_or(Mode::InHead);
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "style" => {
                self.insert_element(tag, "");
                self.original_mode = self.delegate_mode.unwrap_or(Mode::InHead);
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "script" => {
                self.insert_element(tag, "");
                self.original_mode = self.delegate_mode.unwrap_or(Mode::InHead);
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "template" => {
                self.insert_template(tag);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "head" => {
                self.error("second head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "head" => {
                self.pop_head();
                self.mode = Mode::AfterHead;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag)
                if matches!(tag.name.as_str(), "body" | "html" | "br") =>
            {
                self.pop_head();
                self.mode = Mode::AfterHead;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(ref tag) if tag.name == "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(_) => {
                self.error("stray end tag in head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(_) => {
                self.pop_head();
                self.mode = Mode::AfterHead;
                ModeAction::Reprocess(token)
            }
            Token::Eof => {
                self.pop_head();
                self.mode = Mode::AfterHead;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn pop_head(&mut self) {
        if let Some(h) = self.head_node {
            if let Some(p) = self.stack.iter().position(|&n| n == h) {
                while self.stack.len() > p {
                    self.pop();
                }
            }
        }
    }

    fn mode_in_head_noscript(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            // A doctype inside <noscript> (scripting off) is a parse
            // error and is ignored: the noscript element stays open,
            // so a following comment lands inside it (noscript01
            // #0). Likewise an end tag other than </noscript> /
            // </br> is ignored (noscript01 #14), and an <html>
            // start tag is processed with the in-body rules, which
            // merge its attributes without popping the noscript
            // (noscript01 #1). html5lib 1.1 InHeadNoscriptPhase.
            Token::Doctype { .. } => {
                self.error("unexpected doctype in head noscript");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::EndTag(ref tag) if tag.name == "noscript" => {
                self.pop_until_tag("noscript");
                self.mode = Mode::InHead;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                self.insert_text(s);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "basefont" | "bgsound" | "link" | "meta" | "noframes" | "style"
                ) =>
            {
                self.mode_in_head(token, tz)
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "head" | "noscript") => {
                self.error("nested head/noscript ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "br" => {
                // </br> is "anything else" in head noscript: pop the
                // noscript element and reprocess in InHead.
                self.pop_until_tag("noscript");
                self.mode = Mode::InHead;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(_) => {
                self.error("unexpected end tag in head noscript");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.error("unexpected token in head noscript");
                self.pop_until_tag("noscript");
                self.mode = Mode::InHead;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_after_head(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // NULs are ignored after </head>; whitespace is
                // inserted; anything else creates the body. A mixed
                // token splits: its leading whitespace is inserted
                // and the remainder is reprocessed.
                let d: String = s.chars().filter(|&c| c != '\0').collect();
                if d.is_empty() {
                    ModeAction::Done(ProcessOutcome::Continue)
                } else {
                    let (ws, rest) = Self::split_leading_ws(&d);
                    if !ws.is_empty() {
                        self.insert_text(ws);
                    }
                    if rest.is_empty() {
                        ModeAction::Done(ProcessOutcome::Continue)
                    } else {
                        self.ensure_body_element(&TagToken {
                            name: "body".into(),
                            attrs: Vec::new(),
                            self_closing: false,
                        });
                        // Spec 13.2.6.4.6 (after head, anything else):
                        // an implied body sets frameset-ok to "ok".
                        self.frameset_ok = true;
                        self.mode = Mode::InBody;
                        ModeAction::Reprocess(Token::Character(rest.to_string()))
                    }
                }
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype after head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::StartTag(ref tag) if tag.name == "body" => {
                self.ensure_body_element(tag);
                self.frameset_ok = false;
                self.mode = Mode::InBody;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "frameset" => {
                self.insert_element(tag, "");
                self.mode = Mode::InFrameset;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "base" | "basefont" | "bgsound" | "link" | "meta" | "noframes" | "script"
                        | "style" | "template" | "title"
                ) =>
            {
                // parse5 startTagAfterHead: push the head element
                // back on the stack, run the in-head rules (so a
                // <style>/<script> lands inside the head), then
                // remove the head element again. Without the removal
                // the head stays the current node and following
                // comments land inside it (tests3 #2).
                if let Some(h) = self.head_node {
                    self.stack.push(h);
                }
                self.delegate_mode = Some(Mode::AfterHead);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(token, tz);
                self.delegate_mode = None;
                if let Some(h) = self.head_node {
                    if let Some(p) = self.stack.iter().position(|&n| n == h) {
                        self.stack.remove(p);
                    }
                }
                if self.mode == Mode::InHead {
                    self.mode = Mode::AfterHead;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::StartTag(ref tag) if tag.name == "head" => {
                self.error("second head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag)
                if matches!(tag.name.as_str(), "body" | "html" | "br") =>
            {
                self.ensure_body_element(&TagToken {
                    name: "body".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                self.mode = Mode::InBody;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(_) => {
                self.error("stray end tag after head ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.ensure_body_element(&TagToken {
                    name: "body".into(),
                    attrs: Vec::new(),
                    self_closing: false,
                });
                // Spec 13.2.6.4.6 (after head, anything else): an
                // implied body sets the frameset-ok flag to "ok".
                self.frameset_ok = true;
                self.mode = Mode::InBody;
                ModeAction::Reprocess(token)
            }
        }
    }

    // -- template -----------------------------------------------------------------

    fn insert_template(&mut self, tag: &TagToken) {
        let id = self.create_element(tag, "");
        // A <template> is a structural element: it is inserted under
        // the current node (or into the current template's contents),
        // never foster-parented, even inside a table.
        let cur = self.adjusted_current_node();
        match cur.and_then(|c| self.template_contents.get(&c).copied()) {
            Some(contents) => {
                self.dom.lock().unwrap().append_child(contents, id);
            }
            None => {
                if let Some(cur) = cur {
                    self.dom.lock().unwrap().append_child(cur, id);
                } else {
                    self.insert_node(id);
                }
            }
        }
        self.stack.push(id);
        let contents = {
            let mut dom = self.dom.lock().unwrap();
            dom.create_document()
        };
        self.template_contents.insert(id, contents);
        self.template_contents_stack.push(contents);
        // Spec 13.2.6.4.16: opening a template pushes 'in
        // template' as the new current template insertion
        // mode. (NOT the delegating mode — that was pushed here
        // before, which made reset_insertion_mode resume in e.g.
        // InBody after a nested template closed, dropping the
        // tokens that belong to the outer template's contents.)
        self.template_modes.push(Mode::InTemplate);
        self.mode = Mode::InTemplate;
        self.afe_insert_marker();
        self.frameset_ok = false;
    }

    fn handle_template_end_tag(&mut self) {
        // "A template element" is an HTML-namespace template:
        // a foreign element merely named "template" (e.g. an SVG
        // <template>) is not one — the end tag pops until a real
        // (HTML) template has been popped, which can take a
        // foreign <template> and its subtree with it.
        if !self.stack.iter().any(|&n| {
            self.tag_of(n).as_deref() == Some("template") && self.is_html_ns(n)
        }) {
            self.error("stray </template> ignored");
            return;
        }
        self.generate_implied_end_tags_thoroughly();
        loop {
            let Some(cur) = self.current_node() else { break };
            let is_template =
                self.tag_of(cur).as_deref() == Some("template") && self.is_html_ns(cur);
            self.pop();
            if is_template {
                break;
            }
        }
        self.template_contents_stack.pop();
        self.clear_afe_to_marker();
        // The return mode comes from the stack (reset insertion
        // mode appropriately): the popped template mode was the
        // *contents* mode, not the mode to resume in.
        self.template_modes.pop();
        self.reset_insertion_mode();
    }

    // -- mode: text -----------------------------------------------------------------

    fn mode_text(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(data) => {
                self.insert_text(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref _tag) => {
                let cur = self.current_node();
                let cur_tag = cur.and_then(|c| self.tag_of(c));
                let script_node = cur;
                // Stage 15 (M7 Phase 1, Rec 11): the script-pause
                // test is namespace-aware — only an HTML `<script>`
                // executes. An SVG `<script>` is inert by rule, not
                // by routing luck (it never reaches `Mode::Text`,
                // but the pause must not fire for it either).
                let cur_ns = cur.map(|c| self.ns_of(c));
                let is_script = cur_tag.as_deref() == Some("script")
                    && cur_ns.as_deref().is_some_and(|ns| ns.is_empty() || ns == NS_HTML);
                self.pop();
                self.mode = self.original_mode;
                if is_script {
                    if let Some(n) = script_node {
                        return ModeAction::Done(ProcessOutcome::Script(n));
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => {
                // EOF inside raw/RCDATA/script text: close the element
                // and reprocess EOF in the original mode (which
                // creates the implied <body>). Unlike the end-tag
                // case, there is no script-pause to report at EOF.
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() != Some("html") {
                        self.pop();
                    }
                }
                self.mode = self.original_mode;
                ModeAction::Reprocess(Token::Eof)
            }
            _ => {
                self.error("unexpected token in text mode ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }
}

fn quirks_public_id(pub_id: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "+//silmaril//dtd html pro v0r11 19970101//",
        "-//advasoft ltd//dtd html 3.0 aswedit + extensions//",
        "-//as//dtd html 3.0 aswedit + extensions//",
        "-//ietf//dtd html 2.0 level 1//",
        "-//ietf//dtd html 2.0 level 2//",
        "-//ietf//dtd html 2.0 strict level 1//",
        "-//ietf//dtd html 2.0 strict level 2//",
        "-//ietf//dtd html 2.0 strict//",
        "-//ietf//dtd html 2.0//",
        "-//ietf//dtd html 2.1e//",
        "-//ietf//dtd html 3.0//",
        "-//ietf//dtd html 3.2 final//",
        "-//ietf//dtd html 3.2//",
        "-//ietf//dtd html 3//",
        "-//ietf//dtd html level 0//",
        "-//ietf//dtd html level 1//",
        "-//ietf//dtd html level 2//",
        "-//ietf//dtd html level 3//",
        "-//ietf//dtd html strict level 0//",
        "-//ietf//dtd html strict level 1//",
        "-//ietf//dtd html strict level 2//",
        "-//ietf//dtd html strict level 3//",
        "-//ietf//dtd html strict//",
        "-//ietf//dtd html//",
        "-//microsoft//dtd internet explorer 2.0 html strict//",
        "-//microsoft//dtd internet explorer 2.0 html//",
        "-//microsoft//dtd internet explorer 2.0 tables//",
        "-//microsoft//dtd internet explorer 3.0 html strict//",
        "-//microsoft//dtd internet explorer 3.0 html//",
        "-//microsoft//dtd internet explorer 3.0 tables//",
        "-//netscape comm. corp.//dtd html//",
        "-//netscape comm. corp.//dtd strict html//",
        "-//metrius//dtd metrius presentational//",
        "-//o'reilly and associates//dtd html 2.0//",
        "-//o'reilly and associates//dtd html extended 1.0//",
        "-//o'reilly and associates//dtd html extended relaxed 1.0//",
        "-//sq//dtd html 2.0 hotmetal + extensions//",
        "-//softquad software//dtd hotmetal pro 6.0::19990601::extensions to html 4.0//",
        "-//softquad//dtd hotmetal pro 4.0::19971010::extensions to html 4.0//",
        "-//spyglass//dtd html 2.0 extended//",
        "-//sun microsystems corp.//dtd hotjava html//",
        "-//sun microsystems corp.//dtd hotjava strict html//",
        "-//w3c//dtd html 3 1995-03-24//",
        "-//w3c//dtd html 3.2 draft//",
        "-//w3c//dtd html 3.2 final//",
        "-//w3c//dtd html 3.2//",
        "-//w3c//dtd html 3.2s draft//",
        "-//w3c//dtd html 4.0 frameset//",
        "-//w3c//dtd html 4.0 transitional//",
        "-//w3c//dtd html experimental 19960712//",
        "-//w3c//dtd html experimental 970421//",
        "-//w3c//dtd w3 html//",
        "-//w3o//dtd w3 html 3.0//",
        "-//webtechs//dtd mozilla html 2.0//",
        "-//webtechs//dtd mozilla html//",
    ];
    PREFIXES.iter().any(|p| pub_id.starts_with(p))
        || pub_id == "-//w3o//dtd w3 html strict 3.0//en//"
        || pub_id == "-/w3c/dtd html 4.0 transitional/en"
        || pub_id == "html"
}

fn limited_quirks_public_id(pub_id: &str) -> bool {
    pub_id.starts_with("-//w3c//dtd xhtml 1.0 frameset//")
        || pub_id.starts_with("-//w3c//dtd xhtml 1.0 transitional//")
}

// ===========================================================================
// InBody + foreign content (part 3)
// ===========================================================================

impl TreeBuilder {
    fn mode_in_body(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        if self.in_foreign_content_for(&token) {
            if let Some(action) = self.foreign_dispatch(token.clone(), tz) {
                return action;
            }
        }
        match token {
            Token::Character(data) => {
                self.reconstruct_afe();
                let mut d = data;
                if d.contains('\0') {
                    // NULs are dropped in body (and do not affect
                    // frameset-ok: the check below runs on the
                    // stripped text).
                    self.error("null character dropped");
                    d = d.replace('\0', "");
                }
                if d.chars().any(|c| c != ' ' && c != '\t' && c != '\n' && c != '\u{c}' && c != '\r')
                {
                    self.frameset_ok = false;
                }
                if !d.is_empty() {
                    self.insert_text(&d);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in body ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) => self.in_body_start_tag(tag.clone(), tz),
            Token::EndTag(ref tag) => self.in_body_end_tag(tag.clone(), tz),
            Token::Eof => {
                if self.stack.iter().any(|&n| self.tag_of(n).as_deref() == Some("template")) {
                    self.handle_template_end_tag_eof();
                    return ModeAction::Reprocess(Token::Eof);
                }
                while self.stack.len() > 1 {
                    self.pop();
                }
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() != Some("html") {
                        self.pop();
                    }
                }
                self.mode = Mode::AfterBody;
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    fn handle_template_end_tag_eof(&mut self) {
        // Stack-driven: a template element on the stack need not
        // have a template_modes entry (e.g. a foreign `<template>`
        // inside SVG never pushed one), so never gate on the mode
        // stack — otherwise EOF in a template loops forever.
        // "A template element" is an HTML-namespace template:
        // pop until a real one has been popped, which can take a
        // foreign <template> and its subtree with it (template.dat
        // #106). One mode entry is popped per real template, so
        // the accounting stays balanced.
        if !self.stack.iter().any(|&n| {
            self.tag_of(n).as_deref() == Some("template") && self.is_html_ns(n)
        }) {
            return;
        }
        self.template_modes.pop();
        self.template_contents_stack.pop();
        self.clear_afe_to_marker();
        loop {
            let Some(cur) = self.current_node() else { break };
            let is_template =
                self.tag_of(cur).as_deref() == Some("template") && self.is_html_ns(cur);
            self.pop();
            if is_template {
                break;
            }
        }
        self.reset_insertion_mode();
    }

    fn in_body_start_tag(&mut self, tag: TagToken, tz: &mut Tokenizer) -> ModeAction {
        let name = tag.name.as_str();
        if tag.self_closing && !is_void(name) && !matches!(name, "svg" | "math") {
            self.error("self-closing flag on non-void element ignored");
        }
        match name {
            "html" => {
                // Spec 13.2.6.4.7: a second <html> merges its
                // attributes onto the document element only when
                // no template element is open; inside template
                // contents the token is ignored outright.
                if !self.has_open_template() {
                    if let Some(h) = self.html_node {
                        self.merge_attrs(h, &tag.attrs);
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "base" | "basefont" | "bgsound" | "link" | "meta" | "noframes" | "script"
            | "style" | "template" | "title" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(Token::StartTag(tag), tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            "body" => {
                if self.body_node.is_some() {
                    self.error("second body ignored");
                    // Spec: a second <body> sets the frameset-ok
                    // flag to "not ok" (so a later <frameset> is
                    // ignored even though the body looks empty)
                    // — but only when no template element is open;
                    // inside template contents the token is ignored
                    // outright (no merge, no flag change).
                    if !self.has_open_template() {
                        self.frameset_ok = false;
                        if let Some(b) = self.body_node {
                            self.merge_attrs(b, &tag.attrs);
                        }
                    }
                    ModeAction::Done(ProcessOutcome::Continue)
                } else {
                    // No body element exists yet: the first <body>
                    // token is created by before/after head, never
                    // here. A <body> token that reaches in-body
                    // with no body element is a fragment artifact
                    // (e.g. <span><body> with a body context, where
                    // the context IS the body) — ignore it rather
                    // than conjuring a spurious second body.
                    ModeAction::Done(ProcessOutcome::Continue)
                }
            }
            "frameset" => {
                if !self.frameset_ok {
                    self.error("frameset ignored (frameset-ok is false)");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                if self.stack.len() < 2
                    || self.tag_of(self.stack[1]).as_deref() != Some("body")
                {
                    // No body element second on the stack (e.g. a
                    // fragment in a body context): ignore the token.
                    self.error("frameset ignored (no body second on stack)");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                if let Some(b) = self.body_node {
                    self.dom.lock().unwrap().remove_from_parent(b);
                    self.body_node = None;
                }
                while self.stack.len() > 1 {
                    self.pop();
                }
                self.insert_element(&tag, "");
                self.mode = Mode::InFrameset;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "address" | "article" | "aside" | "blockquote" | "center" | "details" | "dialog"
            | "dir" | "div" | "dl" | "fieldset" | "figcaption" | "figure" | "footer" | "header"
            | "hgroup" | "main" | "menu" | "nav" | "ol" | "p" | "section" | "summary" | "ul" => {
                self.close_p_if_in_button_scope();
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.close_p_if_in_button_scope();
                if let Some(cur) = self.current_node() {
                    let t = self.tag_of(cur).unwrap_or_default();
                    if matches!(t.as_str(), "h1" | "h2" | "h3" | "h4" | "h5" | "h6") {
                        self.error("nested heading: pop previous");
                        self.pop();
                    }
                }
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "pre" | "listing" => {
                self.close_p_if_in_button_scope();
                self.insert_element(&tag, "");
                self.skip_newline = true;
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "form" => {
                // Spec 13.2.6.4.7: a form start tag is ignored
                // only when the form pointer is set AND no
                // template is open. Inside a template (or a
                // template-context fragment) the form is inserted
                // but the pointer is NOT set, so nested forms are
                // kept (template.dat: "form start tags inside
                // template do not set the form element pointer").
                if self.form_pointer.is_some() && !self.in_template_context() {
                    self.error("nested form ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.close_p_if_in_button_scope();
                let id = self.insert_element(&tag, "");
                if !self.in_template_context() {
                    self.form_pointer = Some(id);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "li" | "dd" | "dt" => {
                // Spec 13.2.6.4.7, list-item start tags: walk down
                // from the current node. A stop name closes that
                // element (li closes li; dd/dt close dd or dt — so
                // a <dd> after <dd><optgroup> closes the first dd
                // through the optgroup); a special element other
                // than address/div/p stops the walk with no close.
                self.frameset_ok = false;
                let stop: &[&str] = match name {
                    "li" => &["li"],
                    _ => &["dt", "dd"],
                };
                let mut idx = self.stack.len();
                while idx > 0 {
                    idx -= 1;
                    let n = self.stack[idx];
                    let t = self.tag_of(n).unwrap_or_default();
                    if stop.contains(&t.as_str()) {
                        self.generate_implied_end_tags(Some(&t));
                        self.pop_until_tag(&t);
                        break;
                    }
                    if is_special(&t, self.is_html_ns(n))
                        && !matches!(t.as_str(), "address" | "div" | "p")
                    {
                        break;
                    }
                }
                self.close_p_if_in_button_scope();
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "plaintext" => {
                self.close_p_if_in_button_scope();
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "button" => {
                if self.has_in_scope("button", Scope::Default) {
                    self.error("nested button: close previous");
                    self.generate_implied_end_tags(None);
                    self.pop_until_tag("button");
                }
                self.reconstruct_afe();
                self.insert_element(&tag, "");
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "a" => {
                // Spec 13.2.6.4.7, <a>: only an `a` AFTER the last
                // marker counts. The old search skipped markers, so
                // an <a> in an outer scope (e.g. wrapping a table)
                // was removed when a new <a> opened inside a cell
                // — the marker between them means they are unrelated.
                let mut existing = None;
                for e in self.afe.iter().rev() {
                    match e {
                        AfeEntry::Marker => break,
                        AfeEntry::Element(n)
                            if self.tag_of(*n).as_deref() == Some("a") =>
                        {
                            existing = Some(*n);
                            break;
                        }
                        _ => {}
                    }
                }
                if let Some(old) = existing {
                    self.adoption_agency("a");
                    self.afe_remove(old);
                    if let Some(p) = self.stack.iter().position(|&n| n == old) {
                        self.stack.remove(p);
                    }
                }
                self.reconstruct_afe();
                let id = self.insert_element(&tag, "");
                self.afe_push(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "b" | "big" | "code" | "em" | "font" | "i" | "s" | "small" | "strike" | "strong"
            | "tt" | "u" => {
                self.reconstruct_afe();
                let id = self.insert_element(&tag, "");
                self.afe_push(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "nobr" => {
                self.reconstruct_afe();
                if self.has_in_scope("nobr", Scope::Default) {
                    self.adoption_agency("nobr");
                    self.reconstruct_afe();
                }
                let id = self.insert_element(&tag, "");
                self.afe_push(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "applet" | "marquee" | "object" => {
                self.reconstruct_afe();
                self.insert_element(&tag, "");
                self.afe_insert_marker();
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "table" => {
                // Spec 13.2.6.4.7: a <table> closes a p element only
                // when the document is NOT in quirks mode. In quirks
                // mode the table nests inside the open p (quirks01
                // #1/#2/#3, tests3 #23, tests20 #41).
                if self.quirks != QuirksMode::Quirks {
                    self.close_p_if_in_button_scope();
                }
                self.insert_element(&tag, "");
                self.frameset_ok = false;
                self.mode = Mode::InTable;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "area" | "br" | "embed" | "keygen" | "wbr" => {
                self.reconstruct_afe();
                let id = self.create_element(&tag, "");
                self.insert_node(id);
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "input" => {
                self.reconstruct_afe();
                let id = self.create_element(&tag, "");
                self.insert_node(id);
                let is_hidden = tag
                    .attrs
                    .iter()
                    .any(|(n, v)| n == "type" && v.eq_ignore_ascii_case("hidden"));
                if !is_hidden {
                    self.frameset_ok = false;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "param" | "source" | "track" => {
                let id = self.create_element(&tag, "");
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "hr" => {
                self.close_p_if_in_button_scope();
                let id = self.create_element(&tag, "");
                self.insert_node(id);
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "image" => {
                let mut img = tag.clone();
                img.name = "img".to_string();
                self.reconstruct_afe();
                let id = self.create_element(&img, "");
                self.insert_node(id);
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "img" => {
                self.reconstruct_afe();
                let id = self.create_element(&tag, "");
                self.insert_node(id);
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "textarea" => {
                self.insert_element(&tag, "");
                self.skip_newline = true;
                self.original_mode = Mode::InBody;
                self.mode = Mode::Text;
                self.frameset_ok = false;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "xmp" => {
                self.close_p_if_in_button_scope();
                self.reconstruct_afe();
                self.frameset_ok = false;
                self.insert_element(&tag, "");
                self.original_mode = Mode::InBody;
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "iframe" => {
                self.frameset_ok = false;
                self.insert_element(&tag, "");
                self.original_mode = Mode::InBody;
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "noembed" => {
                self.insert_element(&tag, "");
                self.original_mode = Mode::InBody;
                self.mode = Mode::Text;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "select" => {
                self.reconstruct_afe();
                self.insert_element(&tag, "");
                self.frameset_ok = false;
                // Spec 13.2.6.4.7: a select opened while a table
                // element is in table scope parses in
                // select-in-table, so a following table structure
                // token closes the select instead of being ignored
                // by the in-body rules (tests17). Keyed on scope,
                // not on the current mode: the in-table catch-all
                // delegates here with the mode temporarily switched
                // to InBody, so the mode at this point is not the
                // delegating mode.
                self.mode = if self.has_in_scope("table", Scope::Table) {
                    Mode::InSelectInTable
                } else {
                    Mode::InSelect
                };
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "optgroup" | "option" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("option") {
                        self.pop();
                    }
                }
                self.reconstruct_afe();
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "rb" | "rtc" => {
                if self.has_in_scope("ruby", Scope::Default) {
                    self.generate_implied_end_tags(None);
                }
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "rp" | "rt" => {
                if self.has_in_scope("ruby", Scope::Default) {
                    self.generate_implied_end_tags(Some("rtc"));
                }
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "caption" | "col" | "colgroup" | "frame" | "head" | "tbody" | "td" | "tfoot"
            | "th" | "thead" | "tr" => {
                self.error("table element outside table ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "math" => {
                self.reconstruct_afe();
                let mut t = tag.clone();
                adjust_mathml_attributes(&mut t);
                let id = self.create_element(&t, NS_MATHML);
                self.insert_node(id);
                if !tag.self_closing {
                    self.stack.push(id);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "svg" => {
                self.reconstruct_afe();
                let mut t = tag.clone();
                adjust_svg_attributes(&mut t);
                let id = self.create_element(&t, NS_SVG);
                self.insert_node(id);
                if !tag.self_closing {
                    self.stack.push(id);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.reconstruct_afe();
                self.insert_element(&tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    fn in_body_end_tag(&mut self, tag: TagToken, _tz: &mut Tokenizer) -> ModeAction {
        let name = tag.name.as_str();
        match name {
            "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "body" => {
                if !self.has_in_scope("body", Scope::Default) {
                    self.error("stray </body> ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.mode = Mode::AfterBody;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "html" => {
                if !self.has_in_scope("body", Scope::Default) {
                    self.error("stray </html> ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.mode = Mode::AfterBody;
                ModeAction::Reprocess(Token::EndTag(tag))
            }
            "form" => {
                // Spec (in body, `</form>`): generate implied end
                // tags, then — outside a template — REMOVE just
                // the form element from the stack; intervening
                // elements (e.g. a table) stay open. html5lib 1.1
                // does the same (remove, not pop-until). Inside
                // a template (or a template-context fragment) the
                // form is instead popped off the stack along with
                // anything above it, and the pointer is left
                // alone (it was never set inside a template).
                let in_template = self.in_template_context();
                let form = self.form_pointer;
                if !in_template {
                    self.form_pointer = None;
                }
                if (form.is_some() || in_template)
                    && self.has_in_scope("form", Scope::Default)
                {
                    self.generate_implied_end_tags(None);
                    if in_template {
                        self.pop_until_tag("form");
                    } else if let Some(f) = form {
                        if let Some(pos) = self.stack.iter().position(|&n| n == f) {
                            self.stack.remove(pos);
                        }
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "p" => {
                if !self.has_in_scope("p", Scope::Button) {
                    self.error("stray </p>: insert empty p then close");
                    let _p = self.insert_element(
                        &TagToken {
                            name: "p".into(),
                            attrs: Vec::new(),
                            self_closing: false,
                        },
                        "",
                    );
                    self.close_p_element();
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.close_p_element();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "li" | "dd" | "dt" => {
                let scope = if name == "li" {
                    Scope::ListItem
                } else {
                    Scope::Default
                };
                if !self.has_in_scope(name, scope) {
                    self.error("stray end tag ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.generate_implied_end_tags(Some(name));
                self.pop_until_tag(name);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                if !self.has_in_scope_any(
                    &["h1", "h2", "h3", "h4", "h5", "h6"],
                    Scope::Default,
                ) {
                    self.error("stray heading end tag ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.generate_implied_end_tags(None);
                self.pop_until(|t| matches!(t, "h1" | "h2" | "h3" | "h4" | "h5" | "h6"));
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "a" | "b" | "big" | "code" | "em" | "font" | "i" | "nobr" | "s" | "small"
            | "strike" | "strong" | "tt" | "u" => {
                self.adoption_agency(name);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "applet" | "marquee" | "object" => {
                if !self.has_in_scope(name, Scope::Default) {
                    self.error("stray end tag ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.generate_implied_end_tags(None);
                self.pop_until_tag(name);
                self.clear_afe_to_marker();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "br" => {
                self.error("</br> treated as <br>");
                let id = self.create_element(
                    &TagToken {
                        name: "br".into(),
                        attrs: Vec::new(),
                        self_closing: false,
                    },
                    "",
                );
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "button" => {
                if !self.has_in_scope("button", Scope::Default) {
                    self.error("stray </button> ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.generate_implied_end_tags(None);
                self.pop_until_tag("button");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "select" => {
                if !self.has_in_scope("select", Scope::Default) {
                    self.error("stray </select> ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.pop_until_tag("select");
                self.reset_insertion_mode();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            "address" | "article" | "aside" | "blockquote" | "center" | "details" | "dialog"
            | "dir" | "div" | "dl" | "fieldset" | "figcaption" | "figure" | "footer"
            | "header" | "hgroup" | "listing" | "main" | "menu" | "nav" | "ol" | "pre"
            | "section" | "summary" | "ul" => {
                // Spec 13.2.6.4.7, block end tags: if the element
                // is in scope, generate implied end tags, then pop
                // until it is popped — even over a special element
                // on top (e.g. `</div>` closes a div with a <p>
                // on top). `button` has its own arm above with the
                // same shape.
                if !self.has_in_scope(name, Scope::Default) {
                    self.error("stray block end tag ignored");
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                self.generate_implied_end_tags(None);
                self.pop_until_tag(name);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.generic_end_tag(name);
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    // -- foreign content ---------------------------------------------------------

    /// Like [`Self::in_foreign_content`], but also true for
    /// `mglyph` / `malignmark` start tags inside a MathML text
    /// integration point: spec 13.2.6.5 keeps those two foreign
    /// (they are inserted as MathML elements) while every other
    /// token in an integration point follows the in-body rules.
    fn in_foreign_content_for(&self, token: &Token) -> bool {
        // An <svg> start tag with a MathML <annotation-xml>
        // element as the adjusted current node is NOT foreign
        // content: it goes through the in-body rules, which create
        // the svg element in the SVG namespace (tests10 #51/#52/#53).
        // parse5's shouldProcessStartTagTokenInForeignContent
        // returns false for exactly this case.
        if let Token::StartTag(t) = token {
            if t.name == "svg" {
                if let Some(cur) = self.adjusted_current_node() {
                    if self.tag_of(cur).as_deref() == Some("annotation-xml")
                        && self.ns_of(cur) == NS_MATHML
                    {
                        return false;
                    }
                }
            }
        }
        if self.in_foreign_content() {
            return true;
        }
        // End tags run the foreign walk whenever the adjusted
        // current node is foreign — even when it is an HTML
        // integration point (e.g. an SVG <title>): `</svg>` must
        // still close the svg element below it, and the walk
        // either pops the match or defers to the insertion mode
        // when an HTML element blocks it. Without this, `</svg>`
        // inside an SVG title is ignored by the in-body generic
        // end-tag walk (the title is special) and the svg never
        // closes (webkit02 #20).
        if let Token::EndTag(_) = token {
            if let Some(cur) = self.adjusted_current_node() {
                if !self.is_html_ns(cur) {
                    return true;
                }
            }
        }
        if let Token::StartTag(t) = token {
            if matches!(t.name.as_str(), "mglyph" | "malignmark") {
                if let Some(cur) = self.adjusted_current_node() {
                    return self.is_mathml_text_integration_point(cur);
                }
            }
        }
        false
    }

    fn in_foreign_content(&self) -> bool {
        let Some(cur) = self.adjusted_current_node() else {
            return false;
        };
        if self.is_html_ns(cur) {
            return false;
        }
        if self.is_html_integration_point(cur) || self.is_mathml_text_integration_point(cur) {
            return false;
        }
        true
    }

    /// True when a `template` element is on the stack of open
    /// elements (parse5's `tmplCount > 0`).
    fn has_open_template(&self) -> bool {
        self.stack
            .iter()
            .any(|&n| self.tag_of(n).as_deref() == Some("template"))
    }

    /// True when the topmost `table` element on the stack is
    /// above the topmost `template` element (parse5's
    /// `_resetInsertionModeForSelect`): a select in that
    /// position parses in "in select in table", so table
    /// structure tokens close the select and reprocess in the
    /// table modes instead of being ignored.
    fn select_in_table_scope(&self) -> bool {
        let Some(table) = self
            .stack
            .iter()
            .rposition(|&n| self.tag_of(n).as_deref() == Some("table"))
        else {
            return false;
        };
        if table == 0 {
            return false;
        }
        let template = self
            .stack
            .iter()
            .rposition(|&n| self.tag_of(n).as_deref() == Some("template"));
        template.is_none_or(|t| table > t)
    }

    /// Like [`Self::has_open_template`], but also true for a
    /// fragment whose context element is a `template`: the
    /// context element is not on the stack, but the form rules
    /// (which key off "is a template open") treat it the same
    /// way — form start tags inside it do not set the form
    /// pointer, and `</form>` pops until the form is popped.
    fn in_template_context(&self) -> bool {
        if self.has_open_template() {
            return true;
        }
        self.fragment
            && self
                .fragment_ctx
                .map(|c| self.tag_of(c).as_deref() == Some("template"))
                .unwrap_or(false)
    }

    pub fn is_html_integration_point(&self, node: NodeId) -> bool {
        let ns = self.ns_of(node);
        let tag = self.tag_of(node).unwrap_or_default();
        if ns == NS_MATHML && tag == "annotation-xml" {
            if let Some(enc) = self.attr_of(node, "encoding") {
                let e = enc.to_ascii_lowercase();
                return e == "text/html" || e == "application/xhtml+xml";
            }
            return false;
        }
        if ns == NS_SVG {
            return matches!(
                tag.as_str(),
                "foreignObject" | "foreignobject" | "desc" | "title"
            );
        }
        false
    }

    pub fn is_mathml_text_integration_point(&self, node: NodeId) -> bool {
        let ns = self.ns_of(node);
        let tag = self.tag_of(node).unwrap_or_default();
        ns == NS_MATHML
            && matches!(tag.as_str(), "mi" | "mo" | "mn" | "ms" | "mtext")
    }

    /// Foreign-content dispatch. Returns `Some(action)` when the
    /// token was handled here, `None` to fall through to the normal
    /// InBody rules (breakout cases).
    fn foreign_dispatch(&mut self, token: Token, tz: &mut Tokenizer) -> Option<ModeAction> {
        match token {
            Token::Character(ref data) => {
                // In an HTML integration point or a MathML text
                // integration point, characters follow the in-body
                // rules: NULs are dropped. Elsewhere in foreign
                // content a NUL becomes U+FFFD. Either way NULs do
                // not count towards frameset-ok (html5lib's foreign
                // phase replaces a lone NUL without touching the
                // flag, via its if/elif).
                let cur = self.current_node();
                let integration = cur.map_or(false, |n| {
                    self.is_html_integration_point(n)
                        || self.is_mathml_text_integration_point(n)
                });
                let stripped: String =
                    data.chars().filter(|&c| c != '\0').collect();
                if stripped
                    .chars()
                    .any(|c| !matches!(c, ' ' | '\t' | '\n' | '\u{c}' | '\r'))
                {
                    self.frameset_ok = false;
                }
                let d = if integration {
                    stripped
                } else {
                    data.replace('\0', "\u{FFFD}")
                };
                if d.is_empty() {
                    return Some(ModeAction::Done(ProcessOutcome::Continue));
                }
                self.insert_text(&d);
                Some(ModeAction::Done(ProcessOutcome::Continue))
            }
            Token::StartTag(ref tag) => {
                let name = tag.name.as_str();
                let breakout = matches!(
                    name,
                    "b" | "big" | "blockquote" | "body" | "br" | "center" | "code" | "dd"
                        | "div" | "dl" | "dt" | "em" | "embed" | "h1" | "h2" | "h3" | "h4"
                        | "h5" | "h6" | "head" | "hr" | "i" | "img" | "li" | "listing" | "menu"
                        | "meta" | "nobr" | "ol" | "p" | "pre" | "ruby" | "s" | "small" | "span"
                        | "strong" | "strike" | "sub" | "sup" | "table" | "tt" | "u" | "ul"
                        | "var"
                ) || (name == "font"
                    && tag.attrs.iter().any(|(n, _)| {
                        matches!(n.as_str(), "color" | "face" | "size")
                    }));
                if breakout {
                    self.error("breakout start tag in foreign content");
                    while let Some(cur) = self.current_node() {
                        if self.is_html_ns(cur)
                            || self.is_html_integration_point(cur)
                            || self.is_mathml_text_integration_point(cur)
                        {
                            break;
                        }
                        self.pop();
                    }
                    return None;
                }
                // The namespace comes from the ADJUSTED current
                // node: in a fragment the stack top is the synthetic
                // html element, but insertions under a foreign
                // context element (e.g. `svg path`) inherit the
                // context's namespace.
                let cur_ns = self
                    .adjusted_current_node()
                    .map(|c| self.ns_of(c))
                    .unwrap_or_default();
                let mut t = tag.clone();
                if cur_ns == NS_SVG {
                    adjust_svg_tag_name(&mut t);
                    adjust_svg_attributes(&mut t);
                } else if cur_ns == NS_MATHML {
                    adjust_mathml_attributes(&mut t);
                }
                let (attrs, full) = foreign_attrs(&t);
                let id = {
                    let nid = self.create_element_raw(&t.name, attrs, &cur_ns);
                    self.full_attrs.insert(nid, full);
                    nid
                };
                self.insert_node(id);
                if !is_void(&t.name) && !t.self_closing {
                    self.stack.push(id);
                }
                // A foreign element never switches the tokenizer
                // to raw/RCDATA/script data, even when its name
                // collides with one of those elements (e.g. an SVG
                // <title>). The tokenizer switched on the name
                // alone; cancel it here, so the element's content
                // is tokenised as data.
                if matches!(
                    t.name.as_str(),
                    "title" | "textarea" | "style" | "xmp" | "iframe" | "noembed"
                        | "noframes" | "noscript" | "script" | "plaintext"
                ) {
                    tz.cancel_pending_content();
                }
                Some(ModeAction::Done(ProcessOutcome::Continue))
            }
            Token::EndTag(ref tag) => {
                let name = tag.name.as_str();
                if matches!(name, "br" | "p") {
                    // Breakout end tags: pop foreign elements first,
                    // then reprocess in the insertion mode.
                    while let Some(cur) = self.current_node() {
                        if self.is_html_ns(cur)
                            || self.is_html_integration_point(cur)
                            || self.is_mathml_text_integration_point(cur)
                        {
                            break;
                        }
                        self.pop();
                    }
                    return None;
                }
                // parse5's endTagInForeignContent: the topmost
                // FOREIGN element with the token's name closes, but
                // only when it sits above the topmost HTML element;
                // otherwise the token goes to the insertion mode,
                // whose own scope rules decide. An HTML element with
                // the token's name does NOT close here: `</div>` with
                // foreign elements above a div is ignored by the
                // in-body block arm (the div is out of scope behind
                // a foreignObject), so the foreign content stays open
                // (tests10 #30). The old walk matched any element and
                // popped through the foreign subtree to the div.
                let mut foreign_match: Option<usize> = None;
                let mut topmost_html: Option<usize> = None;
                for (i, &n) in self.stack.iter().enumerate().rev() {
                    if self.is_html_ns(n) {
                        if topmost_html.is_none() {
                            topmost_html = Some(i);
                        }
                    } else if foreign_match.is_none()
                        && self
                            .tag_of(n)
                            .map(|t| t.eq_ignore_ascii_case(name))
                            .unwrap_or(false)
                    {
                        foreign_match = Some(i);
                    }
                }
                match (foreign_match, topmost_html) {
                    (Some(i), Some(h)) if i > h => {
                        while self.stack.len() > i {
                            self.pop();
                        }
                        Some(ModeAction::Done(ProcessOutcome::Continue))
                    }
                    (Some(i), None) => {
                        while self.stack.len() > i {
                            self.pop();
                        }
                        Some(ModeAction::Done(ProcessOutcome::Continue))
                    }
                    _ => None,
                }
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                Some(ModeAction::Done(ProcessOutcome::Continue))
            }
            Token::Doctype { .. } => Some(ModeAction::Done(ProcessOutcome::Continue)),
            Token::Eof => None,
        }
    }
}

fn adjust_svg_tag_name(t: &mut TagToken) {
    t.name = match t.name.as_str() {
        "altglyph" => "altGlyph".to_string(),
        "altglyphdef" => "altGlyphDef".to_string(),
        "altglyphitem" => "altGlyphItem".to_string(),
        "animatecolor" => "animateColor".to_string(),
        "animatemotion" => "animateMotion".to_string(),
        "animatetransform" => "animateTransform".to_string(),
        "clippath" => "clipPath".to_string(),
        "feblend" => "feBlend".to_string(),
        "fecolormatrix" => "feColorMatrix".to_string(),
        "fecomponenttransfer" => "feComponentTransfer".to_string(),
        "fecomposite" => "feComposite".to_string(),
        "feconvolvematrix" => "feConvolveMatrix".to_string(),
        "fediffuselighting" => "feDiffuseLighting".to_string(),
        "fedisplacementmap" => "feDisplacementMap".to_string(),
        "fedistantlight" => "feDistantLight".to_string(),
        "fedropshadow" => "feDropShadow".to_string(),
        "feflood" => "feFlood".to_string(),
        "fefunca" => "feFuncA".to_string(),
        "fefuncb" => "feFuncB".to_string(),
        "fefuncg" => "feFuncG".to_string(),
        "fefuncr" => "feFuncR".to_string(),
        "fegaussianblur" => "feGaussianBlur".to_string(),
        "feimage" => "feImage".to_string(),
        "femerge" => "feMerge".to_string(),
        "femergenode" => "feMergeNode".to_string(),
        "femorphology" => "feMorphology".to_string(),
        "feoffset" => "feOffset".to_string(),
        "fepointlight" => "fePointLight".to_string(),
        "fespecularlighting" => "feSpecularLighting".to_string(),
        "fespotlight" => "feSpotLight".to_string(),
        "fetile" => "feTile".to_string(),
        "feturbulence" => "feTurbulence".to_string(),
        "foreignobject" => "foreignObject".to_string(),
        "glyphref" => "glyphRef".to_string(),
        "lineargradient" => "linearGradient".to_string(),
        "radialgradient" => "radialGradient".to_string(),
        "textpath" => "textPath".to_string(),
        _ => return,
    };
}

fn adjust_svg_attributes(t: &mut TagToken) {
    for (n, _) in t.attrs.iter_mut() {
        *n = match n.as_str() {
            "attributename" => "attributeName".to_string(),
            "attributetype" => "attributeType".to_string(),
            "basefrequency" => "baseFrequency".to_string(),
            "baseprofile" => "baseProfile".to_string(),
            "calcmode" => "calcMode".to_string(),
            "clippathunits" => "clipPathUnits".to_string(),
            "diffuseconstant" => "diffuseConstant".to_string(),
            "edgemode" => "edgeMode".to_string(),
            "filterunits" => "filterUnits".to_string(),
            "glyphref" => "glyphRef".to_string(),
            "gradienttransform" => "gradientTransform".to_string(),
            "gradientunits" => "gradientUnits".to_string(),
            "kernelmatrix" => "kernelMatrix".to_string(),
            "kernelunitlength" => "kernelUnitLength".to_string(),
            "keypoints" => "keyPoints".to_string(),
            "keysplines" => "keySplines".to_string(),
            "keytimes" => "keyTimes".to_string(),
            "lengthadjust" => "lengthAdjust".to_string(),
            "limitingconeangle" => "limitingConeAngle".to_string(),
            "markerheight" => "markerHeight".to_string(),
            "markerunits" => "markerUnits".to_string(),
            "markerwidth" => "markerWidth".to_string(),
            "maskcontentunits" => "maskContentUnits".to_string(),
            "maskunits" => "maskUnits".to_string(),
            "numoctaves" => "numOctaves".to_string(),
            "pathlength" => "pathLength".to_string(),
            "patterncontentunits" => "patternContentUnits".to_string(),
            "patterntransform" => "patternTransform".to_string(),
            "patternunits" => "patternUnits".to_string(),
            "pointsatx" => "pointsAtX".to_string(),
            "pointsaty" => "pointsAtY".to_string(),
            "pointsatz" => "pointsAtZ".to_string(),
            "preservealpha" => "preserveAlpha".to_string(),
            "preserveaspectratio" => "preserveAspectRatio".to_string(),
            "primitiveunits" => "primitiveUnits".to_string(),
            "refx" => "refX".to_string(),
            "refy" => "refY".to_string(),
            "repeatcount" => "repeatCount".to_string(),
            "repeatdur" => "repeatDur".to_string(),
            "requiredextensions" => "requiredExtensions".to_string(),
            "requiredfeatures" => "requiredFeatures".to_string(),
            "specularconstant" => "specularConstant".to_string(),
            "specularexponent" => "specularExponent".to_string(),
            "spreadmethod" => "spreadMethod".to_string(),
            "startoffset" => "startOffset".to_string(),
            "stddeviation" => "stdDeviation".to_string(),
            "stitchtiles" => "stitchTiles".to_string(),
            "surfacescale" => "surfaceScale".to_string(),
            "systemlanguage" => "systemLanguage".to_string(),
            "tablevalues" => "tableValues".to_string(),
            "targetx" => "targetX".to_string(),
            "targety" => "targetY".to_string(),
            "textlength" => "textLength".to_string(),
            "viewbox" => "viewBox".to_string(),
            "viewtarget" => "viewTarget".to_string(),
            "xchannelselector" => "xChannelSelector".to_string(),
            "ychannelselector" => "yChannelSelector".to_string(),
            "zoomandpan" => "zoomAndPan".to_string(),
            _ => continue,
        };
    }
}

fn adjust_mathml_attributes(t: &mut TagToken) {
    for (n, _) in t.attrs.iter_mut() {
        if n == "definitionurl" {
            *n = "definitionURL".to_string();
        }
    }
}

/// Split foreign attributes into plain (local, value) pairs for the
/// Dom and qualified (QName, value) pairs for the side map.
fn foreign_attrs(t: &TagToken) -> (Vec<(String, String)>, Vec<(QName, String)>) {
    let mut plain = Vec::new();
    let mut full = Vec::new();
    for (name, value) in &t.attrs {
        let (qname, local) = match name.as_str() {
            "xlink:actuate" | "xlink:arcrole" | "xlink:href" | "xlink:role" | "xlink:show"
            | "xlink:title" | "xlink:type" => (
                QName {
                    ns: NS_XLINK.to_string(),
                    local: name.trim_start_matches("xlink:").to_string(),
                    prefix: Some("xlink".to_string()),
                },
                name.clone(),
            ),
            "xml:lang" | "xml:space" => (
                QName {
                    ns: NS_XML.to_string(),
                    local: name.trim_start_matches("xml:").to_string(),
                    prefix: Some("xml".to_string()),
                },
                name.clone(),
            ),
            "xmlns" | "xmlns:xlink" => (
                QName {
                    ns: NS_XMLNS.to_string(),
                    local: name.trim_start_matches("xmlns:").to_string(),
                    prefix: if name.contains(':') {
                        Some("xmlns".to_string())
                    } else {
                        None
                    },
                },
                name.clone(),
            ),
            _ => (
                // Unadjusted attributes live in no namespace, even
                // on a foreign element (e.g. `encoding` on
                // <annotation-xml> serialises bare).
                QName {
                    ns: String::new(),
                    local: name.clone(),
                    prefix: None,
                },
                name.clone(),
            ),
        };
        plain.push((local, value.clone()));
        full.push((qname, value.clone()));
    }
    (plain, full)
}

// ===========================================================================
// Table / select / template / frameset / after-body modes (part 4)
// ===========================================================================

impl TreeBuilder {
    fn flush_pending_table_chars(&mut self) {
        if self.pending_table_chars.is_empty() {
            return;
        }
        let chars: String = std::mem::take(&mut self.pending_table_chars)
            .chars()
            .filter(|&c| c != '\0')
            .collect();
        let non_ws = chars
            .chars()
            .any(|c| !matches!(c, ' ' | '\t' | '\n' | '\u{c}' | '\r'));
        self.pending_table_chars_non_ws = false;
        if chars.is_empty() {
            return;
        }
        if non_ws {
            // Foster-parent the non-whitespace run, through the
            // in-body character rules: reconstruct the active
            // formatting elements first, with foster parenting
            // switched on, so a foster-parented <a> closed by the
            // table structure is cloned back at the foster place
            // and the text lands inside it (html5lib's InTable
            // insertText sets insertFromTable and processes the
            // characters via the in-body phase).
            let saved_flag = self.insert_from_table;
            self.insert_from_table = true;
            let before_top = self.current_node();
            self.reconstruct_afe();
            let reconstructed = self.current_node() != before_top;
            if reconstructed {
                // The text goes inside the reconstructed clone.
                let cur = self.current_node().unwrap();
                self.insert_text_at(cur, None, &chars);
            } else {
                let (parent, before) = self.foster_parent_place();
                self.insert_text_at(parent, before, &chars);
            }
            self.insert_from_table = saved_flag;
            self.frameset_ok = false;
        } else {
            // Whitespace-only pending characters stay inside the
            // table (under the current node, not foster-parented).
            let mut cur = self
                .adjusted_current_node()
                .unwrap_or_else(|| self.dom.lock().unwrap().root().unwrap());
            if let Some(&contents) = self.template_contents.get(&cur) {
                cur = contents;
            }
            self.insert_text_at(cur, None, &chars);
        }
    }

    fn mode_in_table(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref data) => {
                // Buffer characters; whitespace stays in the table,
                // anything else is foster-parented.
                let cur_tag = self
                    .current_node()
                    .and_then(|c| self.tag_of(c))
                    .unwrap_or_default();
                if matches!(cur_tag.as_str(), "table" | "tbody" | "tfoot" | "thead" | "tr") {
                    self.pending_table_chars.push_str(data);
                    if data
                        .chars()
                        .any(|c| c != ' ' && c != '\t' && c != '\n' && c != '\u{c}' && c != '\r')
                    {
                        self.pending_table_chars_non_ws = true;
                    }
                    // Remember the mode to return to when the
                    // pending characters are flushed: the mode
                    // that delegated here (InRow, InCell, ...),
                    // not InTable itself.
                    self.table_text_return_mode = self.mode;
                    self.mode = Mode::InTableText;
                    ModeAction::Done(ProcessOutcome::Continue)
                } else {
                    self.mode_in_body(token, tz)
                }
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in table ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) => match tag.name.as_str() {
                "caption" => {
                    self.flush_pending_table_chars();
                    self.clear_stack_to_table_context(&["table"]);
                    self.insert_element_in_table(tag, "");
                    self.afe_insert_marker();
                    self.mode = Mode::InCaption;
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "colgroup" => {
                    self.flush_pending_table_chars();
                    self.clear_stack_to_table_context(&["table"]);
                    self.insert_element_in_table(tag, "");
                    self.mode = Mode::InColumnGroup;
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "col" => {
                    self.flush_pending_table_chars();
                    self.clear_stack_to_table_context(&["table"]);
                    // Implied colgroup.
                    let cg = self.insert_element_in_table(
                        &TagToken {
                            name: "colgroup".into(),
                            attrs: Vec::new(),
                            self_closing: false,
                        },
                        "",
                    );
                    let _ = cg;
                    self.mode = Mode::InColumnGroup;
                    ModeAction::Reprocess(token)
                }
                "tbody" | "thead" | "tfoot" => {
                    self.flush_pending_table_chars();
                    self.clear_stack_to_table_context(&["table"]);
                    self.insert_element_in_table(tag, "");
                    self.mode = Mode::InTableBody;
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "td" | "th" | "tr" => {
                    self.flush_pending_table_chars();
                    self.clear_stack_to_table_context(&["table"]);
                    // Implied tbody.
                    let tb = self.insert_element_in_table(
                        &TagToken {
                            name: "tbody".into(),
                            attrs: Vec::new(),
                            self_closing: false,
                        },
                        "",
                    );
                    let _ = tb;
                    self.mode = Mode::InTableBody;
                    ModeAction::Reprocess(token)
                }
                "table" => {
                    self.error("nested table: close previous");
                    self.flush_pending_table_chars();
                    if self.has_in_scope("table", Scope::Table) {
                        self.pop_until_tag("table");
                        self.reset_insertion_mode();
                        return ModeAction::Reprocess(token);
                    }
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "style" | "script" | "template" => {
                    let saved = self.mode;
                    self.delegate_mode = Some(saved);
                    self.mode = Mode::InHead;
                    let out = self.mode_in_head(token, tz);
                    self.delegate_mode = None;
                    if self.mode == Mode::InHead {
                        self.mode = saved;
                    }
                    match out {
                        ModeAction::Done(o) => ModeAction::Done(o),
                        ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                    }
                }
                "input" => {
                    let is_hidden = tag
                        .attrs
                        .iter()
                        .any(|(n, v)| n == "type" && v.eq_ignore_ascii_case("hidden"));
                    if is_hidden {
                        self.flush_pending_table_chars();
                        let id = self.create_element(tag, "");
                        self.insert_node(id);
                        ModeAction::Done(ProcessOutcome::Continue)
                    } else {
                        self.flush_pending_table_chars();
                        let saved_mode = self.mode;
                        let saved_flag = self.insert_from_table;
                        self.insert_from_table = true;
                        self.mode = Mode::InBody;
                        let out = self.mode_in_body(token, tz);
                        self.insert_from_table = saved_flag;
                        if self.mode == Mode::InBody {
                            self.mode = saved_mode;
                        }
                        match out {
                            ModeAction::Done(o) => ModeAction::Done(o),
                            ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                        }
                    }
                }
                "form" => {
                    self.error("form in table ignored except pointer");
                    self.flush_pending_table_chars();
                    if self.form_pointer.is_none() {
                        // Insert under the table itself (never foster-
                        // parented), set the pointer, pop at once.
                        let id = self.insert_element_in_table(tag, "");
                        self.form_pointer = Some(id);
                        self.pop();
                    }
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                _ => {
                    self.flush_pending_table_chars();
                    // Anything else: foster-parent via InBody rules
                    // ("anything else in table" — parse error, then
                    // process in body with foster parenting).
                    self.error("unexpected token in table; foster parenting");
                    let saved_mode = self.mode;
                    let saved_flag = self.insert_from_table;
                    self.insert_from_table = true;
                    self.mode = Mode::InBody;
                    let out = self.mode_in_body(token, tz);
                    self.insert_from_table = saved_flag;
                    // Stay in table mode unless the token switched
                    // modes (e.g. select).
                    if self.mode == Mode::InBody {
                        self.mode = saved_mode;
                    }
                    match out {
                        ModeAction::Done(o) => ModeAction::Done(o),
                        ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                    }
                }
            },
            Token::EndTag(ref tag) => match tag.name.as_str() {
                "table" => {
                    if self.has_in_scope("table", Scope::Table) {
                        self.pop_until_tag("table");
                        self.reset_insertion_mode();
                    }
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "body" | "caption" | "col" | "colgroup" | "html" | "tbody" | "td" | "tfoot"
                | "th" | "thead" | "tr" => {
                    self.error("end tag in table ignored");
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                "template" => {
                    self.handle_template_end_tag();
                    ModeAction::Done(ProcessOutcome::Continue)
                }
                _ => {
                    self.flush_pending_table_chars();
                    let saved_mode = self.mode;
                    let saved_flag = self.insert_from_table;
                    self.insert_from_table = true;
                    self.mode = Mode::InBody;
                    let out = self.mode_in_body(token, tz);
                    self.insert_from_table = saved_flag;
                    if self.mode == Mode::InBody {
                        self.mode = saved_mode;
                    }
                    match out {
                        ModeAction::Done(o) => ModeAction::Done(o),
                        ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                    }
                }
            },
            Token::Eof => self.mode_in_body(token, tz),
        }
    }

    fn clear_stack_to_table_context(&mut self, contexts: &[&str]) {
        let before = self.stack.len();
        loop {
            let Some(cur) = self.current_node() else { break };
            let t = self.tag_of(cur).unwrap_or_default();
            if contexts.contains(&t.as_str()) || t == "html" || t == "template" {
                break;
            }
            self.pop();
        }
        // If the clear popped everything down to the synthetic
        // html root, the reference (which keeps the context element
        // on the stack) would have popped the context element too
        // — but only when the context's tag is not one of the stop
        // tags. A `table` context survives
        // clear_stack_to_table_context(&["table"]) (the clear
        // stops at it), so a table-context fragment is NOT
        // detached by it: the <caption>/<tbody> that follows
        // still lands under the context table. A `tr` context,
        // by contrast, is popped by the same clear, and later
        // insertions do fall outside it.
        if self.fragment
            && self.stack.len() < before
            && self.at_fragment_root()
        {
            let ctx_tag = self
                .fragment_ctx
                .and_then(|c| self.tag_of(c))
                .unwrap_or_default();
            if !(contexts.contains(&ctx_tag.as_str())
                || ctx_tag == "html"
                || ctx_tag == "template")
            {
                self.fragment_detached = true;
            }
        }
    }

    fn mode_in_table_text(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref data) => {
                self.pending_table_chars.push_str(data);
                if data
                    .chars()
                    .any(|c| c != ' ' && c != '\t' && c != '\n' && c != '\u{c}' && c != '\r')
                {
                    self.pending_table_chars_non_ws = true;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.flush_pending_table_chars();
                // Spec 13.2.6.4.11: switch back to the original
                // insertion mode (the one active when the character
                // run started), then reprocess the token.
                self.mode = self.table_text_return_mode;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_in_caption(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::EndTag(ref tag) if tag.name == "caption" => {
                if self.has_in_scope("caption", Scope::Table) {
                    self.generate_implied_end_tags(None);
                    self.pop_until_tag("caption");
                    self.clear_afe_to_marker();
                    self.mode = Mode::InTable;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "td" | "tfoot" | "th"
                        | "thead" | "tr"
                ) =>
            {
                if self.has_in_scope("caption", Scope::Table) {
                    self.generate_implied_end_tags(None);
                    self.pop_until_tag("caption");
                    self.clear_afe_to_marker();
                    self.mode = Mode::InTable;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "table" => {
                if self.has_in_scope("caption", Scope::Table) {
                    self.generate_implied_end_tags(None);
                    self.pop_until_tag("caption");
                    self.clear_afe_to_marker();
                    self.mode = Mode::InTable;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                // Caption contents are body content, except table
                // end tags handled above.
                if let Token::EndTag(ref tag) = token {
                    if matches!(
                        tag.name.as_str(),
                        "body" | "col" | "colgroup" | "html" | "tbody" | "td" | "tfoot"
                            | "th" | "thead" | "tr"
                    ) {
                        self.error("end tag in caption ignored");
                        return ModeAction::Done(ProcessOutcome::Continue);
                    }
                }
                self.mode_in_body(token, tz)
            }
        }
    }

    fn mode_in_column_group(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // Whitespace goes into the colgroup; anything else
                // pops the colgroup and reprocesses in table. A mixed
                // token splits: its leading whitespace is inserted
                // and the remainder is reprocessed (domjs #36: the
                // space stays in the colgroup, "foo" is
                // foster-parented before the table).
                let (ws, rest) = Self::split_leading_ws(s);
                if !ws.is_empty() {
                    self.insert_text(ws);
                }
                if rest.is_empty() {
                    ModeAction::Done(ProcessOutcome::Continue)
                } else {
                    // The remainder is non-whitespace: pop the
                    // colgroup and reprocess it in table — the same
                    // as the catch-all below. If the current node is
                    // not a colgroup (e.g. a <col> inside a
                    // template's contents leaves the template as the
                    // current node), the token is ignored — the
                    // catch-all returns Done, and reprocessing it
                    // here would loop forever on the same remainder
                    // (template.dat #82).
                    if self.current_node().and_then(|c| self.tag_of(c)).as_deref()
                        == Some("colgroup")
                    {
                        self.pop();
                        self.mode = Mode::InTable;
                        ModeAction::Reprocess(Token::Character(rest.to_string()))
                    } else {
                        ModeAction::Done(ProcessOutcome::Continue)
                    }
                }
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in colgroup ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::StartTag(ref tag) if tag.name == "col" => {
                let id = self.create_element(tag, "");
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "template" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(token, tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::EndTag(ref tag) if tag.name == "colgroup" => {
                if self.current_node().and_then(|c| self.tag_of(c)).as_deref() == Some("colgroup")
                {
                    self.pop();
                    self.mode = Mode::InTable;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "col" => {
                self.error("</col> ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => self.mode_in_body(token, tz),
            _ => {
                // Pop the colgroup and reprocess in table.
                if self.current_node().and_then(|c| self.tag_of(c)).as_deref()
                    == Some("colgroup")
                {
                    self.pop();
                    self.mode = Mode::InTable;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    fn mode_in_table_body(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::StartTag(ref tag) if tag.name == "tr" => {
                self.clear_stack_to_table_context(&["tbody", "tfoot", "thead", "template"]);
                self.insert_element_in_table(tag, "");
                self.mode = Mode::InRow;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "td" | "th") => {
                self.error("cell without row: implied tr");
                self.clear_stack_to_table_context(&["tbody", "tfoot", "thead", "template"]);
                let tr = self.insert_element_in_table(
                    &TagToken {
                        name: "tr".into(),
                        attrs: Vec::new(),
                        self_closing: false,
                    },
                    "",
                );
                let _ = tr;
                self.mode = Mode::InRow;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(ref tag)
                if matches!(tag.name.as_str(), "tbody" | "tfoot" | "thead") =>
            {
                if self.has_in_scope(tag.name.as_str(), Scope::Table) {
                    self.clear_stack_to_table_context(&[tag.name.as_str()]);
                    self.pop();
                    self.mode = Mode::InTable;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            // Spec 13.2.6.4.13: caption/col/colgroup/tbody/
            // tfoot/thead start tags in a table body are honoured
            // only when a tbody/thead/tfoot element is in table
            // scope (close it, switch to in table, reprocess);
            // otherwise the token is ignored outright — there is
            // no fallback to the in-table rules. This is what
            // drops a stray <caption> after a bare <tr> inside
            // template contents.
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "tfoot" | "thead"
                ) =>
            {
                if self.has_in_scope("tbody", Scope::Table)
                    || self.has_in_scope("thead", Scope::Table)
                    || self.has_in_scope("tfoot", Scope::Table)
                {
                    self.clear_stack_to_table_context(&["tbody", "tfoot", "thead", "template"]);
                    self.pop();
                    self.mode = Mode::InTable;
                    return ModeAction::Reprocess(token);
                }
                self.error("table-body context tag with no table body in scope; ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "table" => {
                if self.has_in_scope("tbody", Scope::Table)
                    || self.has_in_scope("thead", Scope::Table)
                    || self.has_in_scope("tfoot", Scope::Table)
                {
                    self.clear_stack_to_table_context(&["tbody", "thead", "tfoot"]);
                    self.pop();
                    self.mode = Mode::InTable;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "body" | "caption" | "col" | "colgroup" | "html" | "td" | "th" | "tr"
                ) =>
            {
                self.error("end tag in table body ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => self.mode_in_table(token, tz),
        }
    }

    fn mode_in_row(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::StartTag(ref tag) if matches!(tag.name.as_str(), "td" | "th") => {
                self.clear_stack_to_table_context(&["tr", "template"]);
                self.insert_element_in_table(tag, "");
                self.afe_insert_marker();
                self.mode = Mode::InCell;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "tr" => {
                if self.has_in_scope("tr", Scope::Table) {
                    self.clear_stack_to_table_context(&["tr"]);
                    self.pop();
                    self.mode = Mode::InTableBody;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "tfoot" | "thead" | "tr"
                ) =>
            {
                if self.has_in_scope("tr", Scope::Table) {
                    self.clear_stack_to_table_context(&["tr"]);
                    self.pop();
                    self.mode = Mode::InTableBody;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "table" => {
                if self.has_in_scope("tr", Scope::Table) {
                    self.clear_stack_to_table_context(&["tr"]);
                    self.pop();
                    self.mode = Mode::InTableBody;
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "tbody" | "tfoot" | "thead"
                ) =>
            {
                if self.has_in_scope(tag.name.as_str(), Scope::Table) {
                    if self.has_in_scope("tr", Scope::Table) {
                        self.clear_stack_to_table_context(&["tr"]);
                        self.pop();
                        self.mode = Mode::InTableBody;
                        return ModeAction::Reprocess(token);
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => self.mode_in_table(token, tz),
        }
    }

    fn mode_in_cell(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::EndTag(ref tag) if matches!(tag.name.as_str(), "td" | "th") => {
                if self.has_in_scope(tag.name.as_str(), Scope::Table) {
                    self.generate_implied_end_tags(None);
                    self.pop_until_html_tag(tag.name.as_str());
                    self.clear_afe_to_marker();
                    self.mode = Mode::InRow;
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "col" | "colgroup" | "tbody" | "td" | "tfoot" | "th"
                        | "thead" | "tr"
                ) =>
            {
                if self.has_in_scope("td", Scope::Table) || self.has_in_scope("th", Scope::Table)
                {
                    self.close_cell();
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "table" => {
                if self.has_in_scope("td", Scope::Table) || self.has_in_scope("th", Scope::Table)
                {
                    self.close_cell();
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag)
                if matches!(tag.name.as_str(), "tbody" | "tfoot" | "thead" | "tr") =>
            {
                if self.has_in_scope(tag.name.as_str(), Scope::Table) {
                    self.close_cell();
                    return ModeAction::Reprocess(token);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => self.mode_in_body(token, tz),
        }
    }

    fn close_cell(&mut self) {
        self.generate_implied_end_tags(None);
        // Pop up to an HTML-namespace td or th, whichever is in
        // scope. A foreign element merely named "td" (e.g. an
        // <svg><td>) is not a cell: stopping at it leaves the real
        // cell and the svg open, and following text lands inside
        // the svg instead of being foster-parented
        // (namespace-sensitivity #0).
        loop {
            let Some(cur) = self.current_node() else { break };
            let t = self.tag_of(cur).unwrap_or_default();
            let html = self.is_html_ns(cur);
            self.pop();
            if html && (t == "td" || t == "th") {
                break;
            }
        }
        self.clear_afe_to_marker();
        self.mode = Mode::InRow;
    }

    // -- select -------------------------------------------------------------

    fn mode_in_select(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref data) => {
                let d = data.replace('\0', "");
                if !d.is_empty() {
                    // Select content parses with the in-body rules
                    // in the current spec, and in-body character
                    // handling reconstructs the active formatting
                    // elements: text after an early-closed <i>/<b>
                    // lands inside a clone, not bare under the
                    // option/select (webkit02 #45).
                    self.reconstruct_afe();
                    self.insert_text(&d);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in select ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::StartTag(ref tag) if tag.name == "option" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("option") {
                        self.pop();
                    }
                }
                // Reconstruct the active formatting elements first:
                // an <option> after an early-closed <i> goes inside
                // a clone of it (webkit02 #35).
                self.reconstruct_afe();
                self.insert_element(tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "optgroup" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("option") {
                        self.pop();
                    }
                }
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("optgroup") {
                        self.pop();
                    }
                }
                self.reconstruct_afe();
                self.insert_element(tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "select" => {
                self.error("nested select treated as end tag");
                if self.has_in_scope("select", Scope::Default) {
                    self.pop_until_tag("select");
                    self.reset_insertion_mode();
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "hr" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("option") {
                        self.pop();
                    }
                }
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("optgroup") {
                        self.pop();
                    }
                }
                let id = self.create_element(tag, "");
                self.insert_node(id);
                // <selectedcontent> clone on explicit </option> is
                // handled in the end-tag branch below.
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "script" || tag.name == "template" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(token, tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::EndTag(ref tag) if tag.name == "optgroup" => {
                // Pop option then optgroup if adjacent.
                let mut i = self.stack.len();
                let mut opt_pos: Option<usize> = None;
                let mut grp_pos: Option<usize> = None;
                while i > 0 {
                    i -= 1;
                    let t = self.tag_of(self.stack[i]).unwrap_or_default();
                    if t == "option" && opt_pos.is_none() {
                        opt_pos = Some(i);
                    } else if t == "optgroup" {
                        grp_pos = Some(i);
                        break;
                    } else {
                        break;
                    }
                }
                match (opt_pos, grp_pos) {
                    (Some(_), Some(g)) => {
                        while self.stack.len() > g {
                            self.pop();
                        }
                    }
                    (None, Some(g)) => {
                        while self.stack.len() > g {
                            self.pop();
                        }
                    }
                    _ => {}
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "option" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("option") {
                        // <selectedcontent> cloning on explicit
                        // </option>: if the select has a
                        // <selectedcontent> child... (deferred: the
                        // corpus's webkit02 cases need the clone
                        // inside the option's parent select; see
                        // final report.)
                        self.pop();
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "select" => {
                if self.has_in_scope("select", Scope::Default) {
                    self.pop_until_tag("select");
                    self.reset_insertion_mode();
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            // In select, only <input> pops the select: it is
            // reprocessed outside it and becomes a sibling of the
            // select (tests7 #17, webkit02 #44). With no select in
            // scope — e.g. a select-context fragment, where the
            // context element is not on the stack — the token is
            // ignored (tests_innerHTML_1 split 76).
            // <keygen> and <textarea> do NOT pop the select: they
            // fall through to the in-body rules below, so a keygen
            // stays inside the select (tests7 #34) and a textarea
            // keeps its RCDATA text inside it (innerHTML split 78).
            Token::StartTag(ref tag) if tag.name == "input" => {
                if self.has_in_scope("select", Scope::Default) {
                    self.pop_until_tag("select");
                    self.reset_insertion_mode();
                    ModeAction::Reprocess(token)
                } else {
                    self.error("input in select with no select in scope ignored");
                    ModeAction::Done(ProcessOutcome::Continue)
                }
            }
            Token::EndTag(ref tag) if tag.name == "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "br" => {
                // In-select `</br>`: act as a `<br>` start tag.
                let id = self.create_element(
                    &TagToken {
                        name: "br".into(),
                        attrs: Vec::new(),
                        self_closing: false,
                    },
                    "",
                );
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(_) => {
                // Anything else: insert the element (in-body rules;
                // the current spec parses select content in-body).
                self.mode_in_body(token, tz)
            }
            Token::EndTag(_) => {
                // Anything else: the in-body rules. The current spec
                // parses select content with the in-body rules, so a
                // formatting end tag such as `</i>` must run the
                // adoption agency (which also maintains the active
                // formatting elements) instead of a plain pop-until
                // that leaves stale entries in the list — those
                // stale entries are later reconstructed around
                // following text (webkit02 #45).
                self.mode_in_body(token, tz)
            }
            Token::Eof => self.mode_in_body(token, tz),
        }
    }

    fn mode_in_select_in_table(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::StartTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "table" | "tbody" | "tfoot" | "thead" | "tr" | "td" | "th"
                ) =>
            {
                self.error("table token in select-in-table: close select");
                self.pop_until_tag("select");
                self.reset_insertion_mode();
                ModeAction::Reprocess(token)
            }
            Token::EndTag(ref tag)
                if matches!(
                    tag.name.as_str(),
                    "caption" | "table" | "tbody" | "tfoot" | "thead" | "tr" | "td" | "th"
                ) =>
            {
                if self.has_in_scope(tag.name.as_str(), Scope::Table) {
                    self.pop_until_tag("select");
                    self.reset_insertion_mode();
                    ModeAction::Reprocess(token)
                } else {
                    self.error("stray table end tag in select ignored");
                    ModeAction::Done(ProcessOutcome::Continue)
                }
            }
            _ => self.mode_in_select(token, tz),
        }
    }

    // -- template -------------------------------------------------------------

    fn mode_in_template(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref _data) => {
                self.mode_in_body(token, tz)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in template ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) => {
                let name = tag.name.as_str();
                // Determine the template insertion mode from the tag.
                let new_mode = match name {
                    "base" | "basefont" | "bgsound" | "link" | "meta" | "noframes" | "script"
                    | "style" | "template" | "title" => {
                        let saved = self.mode;
                        self.delegate_mode = Some(saved);
                        self.mode = Mode::InHead;
                        let out = self.mode_in_head(Token::StartTag(tag.clone()), tz);
                        self.delegate_mode = None;
                        if self.mode == Mode::InHead {
                            self.mode = saved;
                        }
                        return match out {
                            ModeAction::Done(o) => ModeAction::Done(o),
                            ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                        };
                    }
                    "caption" | "colgroup" | "tbody" | "tfoot" | "thead" => Mode::InTable,
                    "col" => Mode::InColumnGroup,
                    "tr" => Mode::InTableBody,
                    "td" | "th" => Mode::InRow,
                    _ => Mode::InBody,
                };
                // Pop the current template mode and push the new one.
                self.template_modes.pop();
                self.template_modes.push(new_mode);
                self.mode = new_mode;
                ModeAction::Reprocess(token)
            }
            Token::EndTag(ref tag) if tag.name == "template" => {
                self.handle_template_end_tag();
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(_) => {
                self.error("end tag in template ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => {
                // HTML-namespace template only: a foreign element
                // merely named "template" does not keep the parser
                // in template mode at EOF.
                if self.stack.iter().any(|&n| {
                    self.tag_of(n).as_deref() == Some("template") && self.is_html_ns(n)
                }) {
                    self.handle_template_end_tag_eof();
                    ModeAction::Reprocess(Token::Eof)
                } else {
                    self.mode_in_body(token, tz)
                }
            }
        }
    }

    // -- after body / frameset ---------------------------------------------------

    fn mode_after_body(&mut self, token: Token, tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                // Spec 13.2.6.4.17 (after body): whitespace is
                // processed with the in-body rules, so it lands
                // under the current node (an element still open
                // above the body, e.g. a div left open by the
                // adoption agency) and merges with adjacent text
                // there — not directly under the body element
                // (tricky01 #8: the newline after </body> merges
                // with the one after </pre> into "\n\n" under the
                // div). This is a subroutine call, NOT a mode
                // switch: parse5's onWhitespaceCharacter routes
                // AFTER_BODY whitespace to whitespaceCharacterInBody
                // without changing the insertion mode, so a
                // following comment is still handled by the
                // after-body rules (under the html element, webkit01
                // #26) — and after-after-body delegates its
                // whitespace here, so a mode switch would also break
                // its comments (webkit01 #27).
                self.mode_in_body(token, tz)
            }
            Token::Comment(data) => {
                if let Some(h) = self.html_node {
                    self.insert_comment_at(h, &data);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype after body ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, tz),
            Token::EndTag(ref tag) if tag.name == "html" => {
                self.mode = Mode::AfterAfterBody;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => ModeAction::Done(ProcessOutcome::Continue),
            _ => {
                self.error("unexpected token after body; back to body");
                self.mode = Mode::InBody;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_in_frameset(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // Whitespace only; non-whitespace is dropped.
                let ws: String = s.chars().filter(|c| c.is_ascii_whitespace()).collect();
                if !ws.is_empty() {
                    self.insert_text(&ws);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype in frameset ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => {
                if let Some(h) = self.html_node {
                    self.merge_attrs(h, &tag.attrs);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "frameset" => {
                self.insert_element(tag, "");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::EndTag(ref tag) if tag.name == "frameset" => {
                if let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("html") {
                        self.error("stray </frameset> ignored");
                        return ModeAction::Done(ProcessOutcome::Continue);
                    }
                    self.pop();
                    if !self
                        .stack
                        .iter()
                        .any(|&n| self.tag_of(n).as_deref() == Some("frameset"))
                    {
                        self.mode = Mode::AfterFrameset;
                    }
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "frame" => {
                let id = self.create_element(tag, "");
                self.insert_node(id);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(Token::StartTag(tag.clone()), _tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::Eof => {
                // Pop framesets.
                while let Some(cur) = self.current_node() {
                    if self.tag_of(cur).as_deref() == Some("html") {
                        break;
                    }
                    self.pop();
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            _ => {
                self.error("unexpected token in frameset ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    fn mode_after_frameset(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Character(ref s) => {
                // Whitespace is inserted into the html element;
                // non-whitespace is ignored. Every whitespace
                // character in the token is inserted, not just the
                // leading run: `" te st"` contributes both spaces
                // (tests2 #7), and `"\nfoo"` its newline (tests6 #7).
                let ws: String = s
                    .chars()
                    .filter(|c| c.is_ascii_whitespace())
                    .collect();
                if !ws.is_empty() {
                    self.insert_text(&ws);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Comment(data) => {
                self.insert_comment(&data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype after frameset ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => {
                if let Some(h) = self.html_node {
                    self.merge_attrs(h, &tag.attrs);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(Token::StartTag(tag.clone()), _tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::EndTag(ref tag) if tag.name == "html" => {
                self.mode = Mode::AfterAfterFrameset;
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Eof => ModeAction::Done(ProcessOutcome::Continue),
            _ => {
                self.error("unexpected token after frameset ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }

    fn mode_after_after_body(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Comment(data) => {
                // In a document the comment is the last child of
                // the Document object; in a fragment there is no
                // Document child to land on — the comment goes
                // under the synthetic html element instead, so it
                // survives the reparenting under the context
                // element (tests_innerHTML_1 #78).
                if self.fragment {
                    if let Some(h) = self.html_node {
                        self.insert_comment_at(h, &data);
                    }
                } else {
                    let root = self.dom.lock().unwrap().root().unwrap();
                    self.insert_comment_at(root, &data);
                }
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                self.mode_after_body(token, _tz)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, _tz),
            Token::Eof => ModeAction::Done(ProcessOutcome::Continue),
            _ => {
                self.error("unexpected token after after-body; back to body");
                self.mode = Mode::InBody;
                ModeAction::Reprocess(token)
            }
        }
    }

    fn mode_after_after_frameset(&mut self, token: Token, _tz: &mut Tokenizer) -> ModeAction {
        match token {
            Token::Comment(data) => {
                // Same fragment rule as after-after-body: the
                // comment goes under the synthetic html element.
                if self.fragment {
                    if let Some(h) = self.html_node {
                        self.insert_comment_at(h, &data);
                    }
                    return ModeAction::Done(ProcessOutcome::Continue);
                }
                let root = self.dom.lock().unwrap().root().unwrap();
                self.insert_comment_at(root, &data);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Doctype { .. } => {
                self.error("doctype ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::Character(ref s) if s.chars().all(|c| c.is_ascii_whitespace()) => {
                self.insert_text(s);
                ModeAction::Done(ProcessOutcome::Continue)
            }
            Token::StartTag(ref tag) if tag.name == "html" => self.mode_in_body(token, _tz),
            Token::StartTag(ref tag) if tag.name == "noframes" => {
                let saved = self.mode;
                self.delegate_mode = Some(saved);
                self.mode = Mode::InHead;
                let out = self.mode_in_head(Token::StartTag(tag.clone()), _tz);
                self.delegate_mode = None;
                if self.mode == Mode::InHead {
                    self.mode = saved;
                }
                match out {
                    ModeAction::Done(o) => ModeAction::Done(o),
                    ModeAction::Reprocess(t) => ModeAction::Reprocess(t),
                }
            }
            Token::Eof => ModeAction::Done(ProcessOutcome::Continue),
            _ => {
                self.error("unexpected token ignored");
                ModeAction::Done(ProcessOutcome::Continue)
            }
        }
    }
}

// ===========================================================================
// Driver: run a tokenizer + builder over a whole input
// ===========================================================================

/// Parse a whole document string through a fresh builder.
pub fn parse_document_with(html: &str, scripting: bool) -> ParseOutput {
    let mut tz = Tokenizer::with_scripting(scripting);
    tz.feed_str(html);
    tz.set_eof();
    let mut builder = TreeBuilder::new(scripting);
    run_to_eof(&mut builder, &mut tz);
    builder.finish()
}

fn run_to_eof(builder: &mut TreeBuilder, tz: &mut Tokenizer) {
    loop {
        // The builder needs to know whether the current node is
        // foreign so `<![CDATA[` is character data there.
        builder_set_cdata(builder, tz);
        let Some(token) = tz.next_token() else {
            break;
        };
        if matches!(token, Token::Eof) {
            // Feed EOF through the insertion modes (this creates
            // the implied <body> etc.) before popping the stack.
            let _ = builder.process(Token::Eof, tz);
            builder.process_eof();
            break;
        }
        match builder.process(token, tz) {
            ProcessOutcome::Continue => {}
            ProcessOutcome::Script(_) => {}
        }
        // Merge tokenizer parse errors into the builder stats.
        if tz.parse_errors > 0 {
            // Counted once at the end instead (see finish below).
        }
    }
    // Fold tokenizer errors into the builder's count.
    builder.absorb_tokenizer_errors(tz);
}

fn builder_set_cdata(builder: &TreeBuilder, tz: &mut Tokenizer) {
    // Spec 13.2.5.42: `[CDATA[` is character data whenever the
    // adjusted current node is not in the HTML namespace — that
    // includes HTML integration points (foreignObject, svg title)
    // and MathML text integration points.
    let cur = builder.adjusted_current_node();
    let foreign = match cur {
        Some(c) => {
            let ns = builder.ns_of(c);
            !ns.is_empty() && ns != NS_HTML
        }
        None => false,
    };
    tz.cdata_allowed = foreign;
}

impl TreeBuilder {
    /// Public wrapper for the streaming driver: fold the
    /// tokenizer's parse-error count into the builder stats.
    pub fn absorb_tokenizer_errors_pub(&mut self, tz: &Tokenizer) {
        self.absorb_tokenizer_errors(tz);
    }

    fn absorb_tokenizer_errors(&mut self, tz: &Tokenizer) {
        self.parse_errors += tz.parse_errors;
        for e in &tz.first_errors {
            if self.first_errors.len() < 10 {
                self.first_errors.push(e.clone());
            }
        }
    }
}

/// Parse an HTML fragment in the context of an element named
/// `context_tag` (namespace `context_ns`). Returns the output plus
/// the context element, whose children are the fragment's
/// top-level nodes.
pub fn parse_fragment_with(
    html: &str,
    context_tag: &str,
    context_ns: Option<&str>,
    scripting: bool,
) -> (ParseOutput, NodeId) {
    let mut tz = Tokenizer::with_scripting(scripting);
    // The fragment's text is tokenised in the context element's
    // content model (e.g. RCDATA for <title>/<textarea>, where a
    // NUL becomes U+FFFD rather than being dropped as in body) —
    // but only when the context element is in the HTML namespace;
    // a foreign context element (e.g. `svg title`) tokenises in
    // the data state.
    if context_ns.is_none() {
        tz.set_fragment_content_model(context_tag);
    }
    tz.feed_str(html);
    tz.set_eof();
    let (mut builder, ctx) = TreeBuilder::new_fragment(context_tag, context_ns, scripting);
    run_to_eof(&mut builder, &mut tz);
    // Move the synthetic html root's children under the context
    // element — only for an `html` context, where the synthetic
    // root stands in for the context element (head/body/frameset).
    // For any other context, nodes that landed under the synthetic
    // root are outside the fragment and are discarded with it.
    if context_tag == "html" {
        if let Some(html) = builder.fragment_html {
            let kids = builder.children_of(html);
            for kid in kids {
                builder.dom.lock().unwrap().append_child(ctx, kid);
            }
            builder.dom.lock().unwrap().remove_from_parent(html);
        }
    } else if let Some(html) = builder.fragment_html {
        builder.dom.lock().unwrap().remove_from_parent(html);
    }
    (builder.finish(), ctx)
}
