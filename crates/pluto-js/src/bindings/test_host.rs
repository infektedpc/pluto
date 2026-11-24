//! Test host for the bindings: a [`DomHost`] over a real
//! `pluto-html` [`Dom`], so binding tests exercise the whole
//! crossing (JS → host → slotmap DOM → JS) without the renderer.
//!
//! The selector matcher here is TEST-ONLY code: it parses a selector
//! string into compounds/combinators and walks the `Dom`. The
//! production host implements `query_selector` /
//! `query_selector_all` with the Stage-6 matcher
//! (`pluto-layout::selector_matches` over the parsed
//! `pluto-css::Selector`); the binding itself never matches
//! selectors, in either case.
//!
//! `set_inner_html` / `document_write` parse fragments with
//! `pluto-html::parse_fragment` and deep-clone the fragment's nodes
//! into this `Dom` (a fragment lives in its own arena, so its nodes
//! cannot be adopted directly). Fragment `<script>` elements are
//! cloned but inert — nothing here ever executes them.

use pluto_html::{Dom, NodeId, NodeKind, parse, parse_fragment};

use crate::host::{DomHost, DomHostError, DomKey, Geometry};

// ------------------------------------------------------------------ host

/// A `DomHost` backed by one `pluto-html::Dom`.
pub struct TestHost {
    /// The document. Tests may read it directly to assert DOM state.
    pub dom: Dom,
    /// `document.write` calls, in order (the text as written).
    pub writes: Vec<String>,
    /// Errors reported through [`DomHost::report_error`].
    pub errors: Vec<String>,
    /// Forced-flush counter (incremented by [`DomHost::geometry`]).
    pub flushes: u64,
}

impl TestHost {
    /// A host over a parsed document.
    pub fn from_html(html: &str) -> Self {
        Self {
            dom: parse(html),
            writes: Vec::new(),
            errors: Vec::new(),
            flushes: 0,
        }
    }

    /// First descendant element of `id` (inclusive) with this tag.
    pub fn find_tag(&self, id: DomKey, tag: &str) -> Option<DomKey> {
        if self.dom.tag(id) == Some(tag) {
            return Some(id);
        }
        for &child in self.dom.children(id) {
            if let Some(found) = self.find_tag(child, tag) {
                return Some(found);
            }
        }
        None
    }

    /// True when `ancestor` is an ancestor of `node` (or `node`
    /// itself).
    fn is_ancestor_or_self(&self, ancestor: DomKey, node: DomKey) -> bool {
        let mut cur = Some(node);
        while let Some(id) = cur {
            if id == ancestor {
                return true;
            }
            cur = self.dom.get(id).and_then(|n| n.parent);
        }
        false
    }

    /// Concatenated text of a subtree.
    fn subtree_text(&self, id: DomKey, out: &mut String) {
        if let Some(t) = self.dom.text(id) {
            out.push_str(t);
        }
        for &child in self.dom.children(id) {
            self.subtree_text(child, out);
        }
    }

    /// Serialise one node (and its subtree) as HTML.
    fn serialise_node(&self, id: DomKey, out: &mut String) {
        let Some(node) = self.dom.get(id) else {
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
                    for &child in self.dom.children(id) {
                        self.serialise_node(child, out);
                    }
                    out.push_str("</");
                    out.push_str(tag);
                    out.push('>');
                }
            }
            NodeKind::Document => {
                for &child in self.dom.children(id) {
                    self.serialise_node(child, out);
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

    /// Deep-clone a node from a fragment `Dom` into this `Dom`,
    /// returning the new key. `<script>` elements are cloned like any
    /// other element — they are inert (never executed).
    fn clone_from(&mut self, frag: &Dom, id: NodeId) -> Option<DomKey> {
        let node = frag.get(id)?;
        let new_key = match &node.kind {
            NodeKind::Element { tag } => self
                .dom
                .create_element_with_attrs(tag, node.attrs.clone()),
            NodeKind::Text(data) => self.dom.create_text(data),
            NodeKind::Comment(data) => self.dom.create_comment(data),
            NodeKind::Doctype { name, public_id, system_id } => {
                self.dom.create_doctype(name, public_id, system_id)
            }
            NodeKind::ProcessingInstruction { target, data } => {
                self.dom.create_pi(target, data)
            }
            NodeKind::Document => self.dom.create_document(),
        };
        let kids: Vec<NodeId> = frag.children(id).to_vec();
        for kid in kids {
            if let Some(cloned) = self.clone_from(frag, kid) {
                self.dom.append_child(new_key, cloned);
            }
        }
        Some(new_key)
    }

    /// Replace (`replace = true`, `innerHTML`) or append
    /// (`replace = false`, `document.write`) `key`'s children with
    /// the fragment parsed from `html` in `key`'s context.
    fn set_fragment(&mut self, key: DomKey, html: &str, replace: bool) -> bool {
        if self.dom.get(key).is_none() {
            return false;
        }
        let context_tag = self.dom.tag(key).unwrap_or("div").to_string();
        let (frag, ctx) = parse_fragment(html, &context_tag, None);
        let kids: Vec<NodeId> = frag.children(ctx).to_vec();
        if replace {
            let old: Vec<NodeId> = self.dom.children(key).to_vec();
            for child in old {
                self.dom.remove_from_parent(child);
            }
        }
        for kid in kids {
            if let Some(cloned) = self.clone_from(&frag, kid) {
                self.dom.append_child(key, cloned);
            }
        }
        true
    }
}

fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
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

impl DomHost for TestHost {
    fn validate(&self, key: DomKey) -> bool {
        self.dom.get(key).is_some()
    }

    fn document_node(&self) -> Option<DomKey> {
        self.dom.root()
    }

    fn document_element(&self) -> Option<DomKey> {
        let root = self.dom.root()?;
        self.find_tag(root, "html")
    }

    fn body(&self) -> Option<DomKey> {
        let html = self.document_element()?;
        self.find_tag(html, "body")
    }

    fn head(&self) -> Option<DomKey> {
        let html = self.document_element()?;
        self.find_tag(html, "head")
    }

    fn parent(&self, key: DomKey) -> Option<DomKey> {
        self.dom.get(key).and_then(|n| n.parent)
    }

    fn children(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.children(key).to_vec()
    }

    fn tag_name(&self, key: DomKey) -> Option<String> {
        self.dom.tag(key).map(|t| t.to_string())
    }

    fn node_type(&self, key: DomKey) -> Option<u16> {
        self.dom.get(key).map(|n| match &n.kind {
            NodeKind::Document => 9,
            NodeKind::Element { .. } => 1,
            NodeKind::Text(_) => 3,
            NodeKind::Comment(_) => 8,
            NodeKind::Doctype { .. } => 10,
            NodeKind::ProcessingInstruction { .. } => 7,
        })
    }

    fn node_name(&self, key: DomKey) -> Option<String> {
        self.dom.get(key).map(|n| match &n.kind {
            NodeKind::Document => "#document".to_string(),
            NodeKind::Element { tag } => tag.to_ascii_uppercase(),
            NodeKind::Text(_) => "#text".to_string(),
            NodeKind::Comment(_) => "#comment".to_string(),
            NodeKind::Doctype { name, .. } => name.clone(),
            NodeKind::ProcessingInstruction { target, .. } => target.clone(),
        })
    }

    fn get_element_by_id(&self, id: &str) -> Option<DomKey> {
        let root = self.dom.root()?;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if self.dom.attr(node, "id") == Some(id) {
                return Some(node);
            }
            stack.extend(self.dom.children(node).iter().rev());
        }
        None
    }

    fn text_content(&self, key: DomKey) -> Option<String> {
        self.dom.get(key)?;
        let mut out = String::new();
        self.subtree_text(key, &mut out);
        Some(out)
    }

    fn set_text_content(&mut self, key: DomKey, text: &str) -> bool {
        if self.dom.get(key).is_none() {
            return false;
        }
        let old: Vec<NodeId> = self.dom.children(key).to_vec();
        for child in old {
            self.dom.remove_from_parent(child);
        }
        let text_node = self.dom.create_text(text);
        self.dom.append_child(key, text_node);
        true
    }

    fn get_attribute(&self, key: DomKey, name: &str) -> Option<String> {
        self.dom.attr(key, name).map(|s| s.to_string())
    }

    fn set_attribute(&mut self, key: DomKey, name: &str, value: &str) -> bool {
        if self.dom.get(key).is_none() {
            return false;
        }
        self.dom.set_attribute(key, name, value);
        true
    }

    fn remove_attribute(&mut self, key: DomKey, name: &str) -> bool {
        if self.dom.get(key).is_none() {
            return false;
        }
        self.dom.remove_attribute(key, name);
        true
    }

    fn has_attribute(&self, key: DomKey, name: &str) -> bool {
        self.dom.attr(key, name).is_some()
    }

    fn title(&self) -> Option<String> {
        let head = self.head()?;
        let title = self.find_tag(head, "title")?;
        self.text_content(title)
    }

    fn set_title(&mut self, title: &str) -> bool {
        let Some(head) = self.head() else {
            return false;
        };
        let title_el = match self.find_tag(head, "title") {
            Some(t) => t,
            None => {
                let t = self.dom.create_element("title");
                self.dom.append_child(head, t);
                t
            }
        };
        self.set_text_content(title_el, title)
    }

    fn create_element(&mut self, tag: &str) -> Option<DomKey> {
        Some(self.dom.create_element(tag))
    }

    fn create_text_node(&mut self, data: &str) -> Option<DomKey> {
        Some(self.dom.create_text(data))
    }

    fn append_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        if self.dom.get(parent).is_none() || self.dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if child == parent || self.is_ancestor_or_self(child, parent) {
            return Err(DomHostError::Hierarchy);
        }
        if matches!(
            self.dom.get(child).map(|n| &n.kind),
            Some(NodeKind::Document) | Some(NodeKind::Doctype { .. })
        ) {
            return Err(DomHostError::Hierarchy);
        }
        self.dom.append_child(parent, child);
        Ok(())
    }

    fn insert_before(
        &mut self,
        parent: DomKey,
        child: DomKey,
        sibling: Option<DomKey>,
    ) -> Result<(), DomHostError> {
        if self.dom.get(parent).is_none() || self.dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if child == parent || self.is_ancestor_or_self(child, parent) {
            return Err(DomHostError::Hierarchy);
        }
        match sibling {
            None => self.append_child(parent, child),
            Some(sib) => {
                if self.dom.get(sib).is_none() {
                    return Err(DomHostError::StaleKey);
                }
                if !self.dom.children(parent).contains(&sib) {
                    return Err(DomHostError::NotFound);
                }
                self.dom.insert_before(parent, sib, child);
                Ok(())
            }
        }
    }

    fn remove_child(&mut self, parent: DomKey, child: DomKey) -> Result<(), DomHostError> {
        if self.dom.get(parent).is_none() || self.dom.get(child).is_none() {
            return Err(DomHostError::StaleKey);
        }
        if self.dom.get(child).and_then(|n| n.parent) != Some(parent) {
            return Err(DomHostError::NotFound);
        }
        self.dom.remove_child(parent, child);
        Ok(())
    }

    fn replace_child(
        &mut self,
        parent: DomKey,
        old: DomKey,
        new: DomKey,
    ) -> Result<(), DomHostError> {
        if self.dom.get(parent).is_none()
            || self.dom.get(old).is_none()
            || self.dom.get(new).is_none()
        {
            return Err(DomHostError::StaleKey);
        }
        if self.dom.get(old).and_then(|n| n.parent) != Some(parent) {
            return Err(DomHostError::NotFound);
        }
        if new == parent || self.is_ancestor_or_self(new, parent) {
            return Err(DomHostError::Hierarchy);
        }
        self.dom.replace_child(parent, old, new);
        Ok(())
    }

    fn request_restlye_and_layout(&mut self, _key: DomKey) {
        // The Dom's own mutators already mark the Double Dirty Bits;
        // nothing extra to do in the test host.
    }

    fn query_selector(&self, scope: DomKey, selector: &str) -> Option<DomKey> {
        self.query_selector_all(scope, selector).into_iter().next()
    }

    fn query_selector_all(&self, scope: DomKey, selector: &str) -> Vec<DomKey> {
        let selectors = parse_selector_list(selector);
        let mut out = Vec::new();
        if selectors.is_empty() || self.dom.get(scope).is_none() {
            return out;
        }
        let mut stack = vec![scope];
        let mut order: Vec<DomKey> = Vec::new();
        while let Some(node) = stack.pop() {
            order.push(node);
            stack.extend(self.dom.children(node).iter().rev());
        }
        for node in order {
            if !matches!(
                self.dom.get(node).map(|n| &n.kind),
                Some(NodeKind::Element { .. })
            ) {
                continue;
            }
            if selectors.iter().any(|s| self.matches(s, node)) {
                out.push(node);
            }
        }
        out
    }

    fn inner_html(&self, key: DomKey) -> Option<String> {
        self.dom.get(key)?;
        let mut out = String::new();
        for &child in self.dom.children(key) {
            self.serialise_node(child, &mut out);
        }
        Some(out)
    }

    fn set_inner_html(&mut self, key: DomKey, html: &str) -> bool {
        self.set_fragment(key, html, true)
    }

    fn document_write(&mut self, text: &str) -> bool {
        self.writes.push(text.to_string());
        // M4 test behaviour: the markup lands at the end of <body>
        // (the parser-side insertion point is the parser agent's
        // job; the binding/host seam is what is tested here).
        let Some(body) = self.body() else {
            return false;
        };
        self.set_fragment(body, text, false)
    }

    fn document_open(&mut self) -> bool {
        if let Some(html) = self.document_element() {
            let kids: Vec<NodeId> = self.dom.children(html).to_vec();
            for kid in kids {
                self.dom.remove_from_parent(kid);
            }
        }
        true
    }

    fn document_close(&mut self) -> bool {
        true
    }

    fn style_text(&self, key: DomKey) -> Option<String> {
        self.dom.attr(key, "style").map(|s| s.to_string())
    }

    fn set_style_text(&mut self, key: DomKey, css: &str) -> bool {
        if self.dom.get(key).is_none() {
            return false;
        }
        self.dom.set_attribute(key, "style", css);
        true
    }

    fn geometry(&mut self, key: DomKey) -> Option<Geometry> {
        self.dom.get(key)?;
        self.flushes += 1;
        Some(Geometry {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        })
    }

    fn forced_flush_count(&self) -> u64 {
        self.flushes
    }

    fn report_error(&self, message: &str) {
        // `&self` only — the test host cannot push without interior
        // mutability; errors are visible via `TabJs::script_errors`.
        let _ = message;
    }
}

// ------------------------------------------------------------------ matcher
//
// Test-only selector matching (see the module docs). Supports type /
// universal / `#id` / `.class` / `[attr]` / `[attr=val]` (+ `~=`,
// `|=`, `^=`, `$=`, `*=`) / `:first-child` / `:last-child` /
// `:nth-child(an+b)` / `:not(<compound>)`, with descendant, child,
// adjacent-sibling, and general-sibling combinators, and comma lists.

#[derive(Debug, Clone, PartialEq)]
enum AttrOp {
    Exists,
    Eq,
    Includes,
    Dash,
    Prefix,
    Suffix,
    Substr,
}

#[derive(Debug, Clone, PartialEq)]
enum Pseudo {
    FirstChild,
    LastChild,
    NthChild(i32, i32),
    Not(Box<Compound>),
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Compound {
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    attrs: Vec<(String, AttrOp, String)>,
    pseudos: Vec<Pseudo>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Combinator {
    Descendant,
    Child,
    NextSibling,
    SubsequentSibling,
}

#[derive(Debug, Clone, Default)]
struct ComplexSelector {
    compounds: Vec<Compound>,
    combinators: Vec<Combinator>,
}

fn parse_selector_list(text: &str) -> Vec<ComplexSelector> {
    split_top_level(text, ',')
        .into_iter()
        .filter_map(|part| parse_complex(part.trim()))
        .collect()
}

fn split_top_level(text: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in text.chars() {
        match c {
            '[' | '(' => {
                depth += 1;
                cur.push(c);
            }
            ']' | ')' => {
                depth -= 1;
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn parse_complex(text: &str) -> Option<ComplexSelector> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut sel = ComplexSelector::default();
    skip_ws(&chars, &mut i);
    if i >= chars.len() || matches!(chars[i], '>' | '+' | '~') {
        return None;
    }
    sel.compounds.push(parse_compound(&chars, &mut i)?);
    loop {
        let had_ws = skip_ws(&chars, &mut i);
        if i >= chars.len() {
            break;
        }
        let comb = match chars[i] {
            '>' => {
                i += 1;
                Combinator::Child
            }
            '+' => {
                i += 1;
                Combinator::NextSibling
            }
            '~' => {
                i += 1;
                Combinator::SubsequentSibling
            }
            _ if had_ws => Combinator::Descendant,
            _ => return None,
        };
        skip_ws(&chars, &mut i);
        sel.combinators.push(comb);
        sel.compounds.push(parse_compound(&chars, &mut i)?);
    }
    Some(sel)
}

fn skip_ws(chars: &[char], i: &mut usize) -> bool {
    let start = *i;
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
    *i > start
}

fn parse_ident(chars: &[char], i: &mut usize) -> Option<String> {
    let start = *i;
    while *i < chars.len()
        && (chars[*i].is_ascii_alphanumeric() || matches!(chars[*i], '-' | '_' | '\\'))
    {
        *i += 1;
    }
    if *i == start {
        None
    } else {
        Some(chars[start..*i].iter().collect())
    }
}

fn parse_compound(chars: &[char], i: &mut usize) -> Option<Compound> {
    let mut compound = Compound::default();
    let mut any = false;
    loop {
        if *i >= chars.len() {
            break;
        }
        match chars[*i] {
            '*' => {
                *i += 1;
                any = true;
            }
            '#' => {
                *i += 1;
                compound.id = parse_ident(chars, i);
                any = true;
            }
            '.' => {
                *i += 1;
                if let Some(class) = parse_ident(chars, i) {
                    compound.classes.push(class);
                }
                any = true;
            }
            '[' => {
                *i += 1;
                skip_ws(chars, i);
                let name = parse_ident(chars, i)?;
                skip_ws(chars, i);
                let (op, value) = if *i < chars.len() && chars[*i] == ']' {
                    (AttrOp::Exists, String::new())
                } else {
                    let op = match chars.get(*i) {
                        Some('=') => AttrOp::Eq,
                        Some('~') => AttrOp::Includes,
                        Some('|') => AttrOp::Dash,
                        Some('^') => AttrOp::Prefix,
                        Some('$') => AttrOp::Suffix,
                        Some('*') => AttrOp::Substr,
                        _ => return None,
                    };
                    *i += 1;
                    if op != AttrOp::Eq {
                        if chars.get(*i) != Some(&'=') {
                            return None;
                        }
                        *i += 1;
                    }
                    skip_ws(chars, i);
                    let value = parse_attr_value(chars, i)?;
                    (op, value)
                };
                skip_ws(chars, i);
                if chars.get(*i) != Some(&']') {
                    return None;
                }
                *i += 1;
                compound.attrs.push((name, op, value));
                any = true;
            }
            ':' => {
                *i += 1;
                let name = parse_ident(chars, i)?;
                match name.as_str() {
                    "first-child" => compound.pseudos.push(Pseudo::FirstChild),
                    "last-child" => compound.pseudos.push(Pseudo::LastChild),
                    "nth-child" => {
                        if chars.get(*i) != Some(&'(') {
                            return None;
                        }
                        *i += 1;
                        let start = *i;
                        while *i < chars.len() && chars[*i] != ')' {
                            *i += 1;
                        }
                        let arg: String = chars[start..*i].iter().collect();
                        *i += 1; // ')'
                        let (a, b) = parse_an_plus_b(arg.trim())?;
                        compound.pseudos.push(Pseudo::NthChild(a, b));
                    }
                    "not" => {
                        if chars.get(*i) != Some(&'(') {
                            return None;
                        }
                        *i += 1;
                        let inner = parse_compound(chars, i)?;
                        if chars.get(*i) != Some(&')') {
                            return None;
                        }
                        *i += 1;
                        compound.pseudos.push(Pseudo::Not(Box::new(inner)));
                    }
                    _ => return None,
                }
                any = true;
            }
            c if c.is_ascii_alphabetic() => {
                compound.tag = parse_ident(chars, i).map(|t| t.to_ascii_lowercase());
                any = true;
            }
            _ => break,
        }
    }
    if any {
        Some(compound)
    } else {
        None
    }
}

fn parse_attr_value(chars: &[char], i: &mut usize) -> Option<String> {
    if *i < chars.len() && (chars[*i] == '"' || chars[*i] == '\'') {
        let quote = chars[*i];
        *i += 1;
        let start = *i;
        while *i < chars.len() && chars[*i] != quote {
            *i += 1;
        }
        let value: String = chars[start..*i].iter().collect();
        *i += 1;
        Some(value)
    } else {
        parse_ident(chars, i)
    }
}

fn parse_an_plus_b(arg: &str) -> Option<(i32, i32)> {
    let arg = arg.replace(' ', "");
    match arg.as_str() {
        "odd" => return Some((2, 1)),
        "even" => return Some((2, 0)),
        _ => {}
    }
    if let Ok(n) = arg.parse::<i32>() {
        return Some((0, n));
    }
    let (a_part, b_part) = arg.split_once('n')?;
    let a: i32 = match a_part {
        "" | "+" => 1,
        "-" => -1,
        _ => a_part.parse().ok()?,
    };
    let b: i32 = match b_part {
        "" => 0,
        _ => b_part.parse().ok()?,
    };
    Some((a, b))
}

impl TestHost {
    fn matches(&self, sel: &ComplexSelector, node: DomKey) -> bool {
        if sel.compounds.is_empty() {
            return false;
        }
        self.match_from(sel, sel.compounds.len() - 1, node)
    }

    fn match_from(&self, sel: &ComplexSelector, idx: usize, node: DomKey) -> bool {
        if !self.match_compound(&sel.compounds[idx], node) {
            return false;
        }
        if idx == 0 {
            return true;
        }
        match sel.combinators[idx - 1] {
            Combinator::Child => match self.parent_element(node) {
                Some(p) => self.match_from(sel, idx - 1, p),
                None => false,
            },
            Combinator::Descendant => {
                let mut cur = self.parent_element(node);
                while let Some(p) = cur {
                    if self.match_from(sel, idx - 1, p) {
                        return true;
                    }
                    cur = self.parent_element(p);
                }
                false
            }
            Combinator::NextSibling => match self.prev_element_sibling(node) {
                Some(s) => self.match_from(sel, idx - 1, s),
                None => false,
            },
            Combinator::SubsequentSibling => {
                let mut cur = self.prev_element_sibling(node);
                while let Some(s) = cur {
                    if self.match_from(sel, idx - 1, s) {
                        return true;
                    }
                    cur = self.prev_element_sibling(s);
                }
                false
            }
        }
    }

    fn match_compound(&self, compound: &Compound, node: DomKey) -> bool {
        if let Some(tag) = &compound.tag
            && self.dom.tag(node) != Some(tag.as_str()) {
                return false;
            }
        if let Some(id) = &compound.id
            && self.dom.attr(node, "id") != Some(id.as_str()) {
                return false;
            }
        let classes = self
            .dom
            .attr(node, "class")
            .unwrap_or("")
            .split_ascii_whitespace()
            .collect::<Vec<_>>();
        for class in &compound.classes {
            if !classes.contains(&class.as_str()) {
                return false;
            }
        }
        for (name, op, want) in &compound.attrs {
            let got = self.dom.attr(node, name);
            let ok = match op {
                AttrOp::Exists => got.is_some(),
                AttrOp::Eq => got == Some(want.as_str()),
                AttrOp::Includes => got
                    .map(|v| v.split_ascii_whitespace().any(|t| t == want))
                    .unwrap_or(false),
                AttrOp::Dash => got
                    .map(|v| v == want || v.starts_with(&format!("{want}-")))
                    .unwrap_or(false),
                AttrOp::Prefix => got.map(|v| v.starts_with(want.as_str())).unwrap_or(false),
                AttrOp::Suffix => got.map(|v| v.ends_with(want.as_str())).unwrap_or(false),
                AttrOp::Substr => got.map(|v| v.contains(want.as_str())).unwrap_or(false),
            };
            if !ok {
                return false;
            }
        }
        for pseudo in &compound.pseudos {
            let ok = match pseudo {
                Pseudo::FirstChild => self.prev_element_sibling(node).is_none(),
                Pseudo::LastChild => self.next_element_sibling(node).is_none(),
                Pseudo::NthChild(a, b) => {
                    let n = self.element_index(node) as i32;
                    nth_matches(n, *a, *b)
                }
                Pseudo::Not(inner) => !self.match_compound(inner, node),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    fn parent_element(&self, node: DomKey) -> Option<DomKey> {
        let mut cur = self.dom.get(node).and_then(|n| n.parent);
        while let Some(id) = cur {
            if matches!(
                self.dom.get(id).map(|n| &n.kind),
                Some(NodeKind::Element { .. })
            ) {
                return Some(id);
            }
            cur = self.dom.get(id).and_then(|n| n.parent);
        }
        None
    }

    fn element_siblings(&self, node: DomKey) -> Vec<DomKey> {
        let Some(parent) = self.dom.get(node).and_then(|n| n.parent) else {
            return vec![node];
        };
        self.dom
            .children(parent)
            .iter()
            .copied()
            .filter(|&c| {
                matches!(
                    self.dom.get(c).map(|n| &n.kind),
                    Some(NodeKind::Element { .. })
                )
            })
            .collect()
    }

    fn prev_element_sibling(&self, node: DomKey) -> Option<DomKey> {
        let sibs = self.element_siblings(node);
        sibs.iter()
            .position(|&s| s == node)
            .and_then(|i| if i > 0 { Some(sibs[i - 1]) } else { None })
    }

    fn next_element_sibling(&self, node: DomKey) -> Option<DomKey> {
        let sibs = self.element_siblings(node);
        sibs.iter()
            .position(|&s| s == node)
            .and_then(|i| sibs.get(i + 1).copied())
    }

    fn element_index(&self, node: DomKey) -> usize {
        let sibs = self.element_siblings(node);
        sibs.iter().position(|&s| s == node).map(|i| i + 1).unwrap_or(0)
    }
}

fn nth_matches(n: i32, a: i32, b: i32) -> bool {
    if a == 0 {
        return n == b;
    }
    let d = n - b;
    d % a == 0 && d / a >= 0
}
