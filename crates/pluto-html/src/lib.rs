//! Parsing & DOM (layer 3): winnow HTML parser + arena DOM.
//!
//! The DOM lives in a slotmap arena. Nodes are addressed by `NodeId`
//! handles, never `Rc<RefCell<T>>`, so there are no reference cycles and
//! the JS layer can hold a `NodeId` across a re-layout without keeping
//! the node itself alive.
//!
//! M1 scope: tags `html head body div h1-h6 p span a style` (plus void
//! elements `br img meta link input hr` and anything else spelled like a
//! tag), attributes, text with entity decoding, comments and doctype
//! skipped, `style`/`script` as raw text. The parser is lenient like a
//! browser: a stray `<`, a mismatched close tag, or an unclosed element
//! never fails the parse — the input is recovered as text or ignored.
//! Full WHATWG tree construction (adoption agency, foster parenting,
//! implied closes) is M7.

use slotmap::{SlotMap, new_key_type};
use winnow::Parser;
use winnow::error::ContextError;
use winnow::stream::ContainsToken;
use winnow::token::{literal, take_while};

new_key_type! {
    /// Handle to a node in the [`Dom`] arena.
    pub struct NodeId;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    Element { tag: String },
    Text(String),
}

/// A node in the arena. Children are `NodeId`s; the parent link makes
/// upward walks (style matching, event bubbling) cheap.
#[derive(Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Attributes in source order; duplicate names keep the first value.
    pub attrs: Vec<(String, String)>,
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
}

/// The arena-allocated DOM tree for one document.
#[derive(Debug, Default)]
pub struct Dom {
    arena: SlotMap<NodeId, Node>,
    root: Option<NodeId>,
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
        })
    }

    pub fn create_text(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Text(data.to_string()),
            parent: None,
            children: Vec::new(),
            attrs: Vec::new(),
        })
    }

    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        if let Some(node) = self.arena.get_mut(child) {
            node.parent = Some(parent);
        }
        if let Some(node) = self.arena.get_mut(parent) {
            node.children.push(child);
        }
    }
}

/// `take_while` pinned to `&str` input and `ContextError`, so call sites
/// do not need a turbofish: the generic overload cannot infer its error
/// type when the result is discarded with `.ok()?`.
fn tw<'i>(
    occurrences: std::ops::RangeFrom<usize>,
    set: impl ContainsToken<char>,
) -> impl Parser<&'i str, &'i str, ContextError> {
    take_while(occurrences, set)
}

/// `literal` pinned the same way.
fn lit(literal_str: &str) -> impl Parser<&str, &str, ContextError> {
    literal::<_, _, ContextError>(literal_str)
}

/// Parse HTML text into a [`Dom`]. Lenient by design (see module docs):
/// this never fails; malformed input is recovered as text or skipped.
pub fn parse(html: &str) -> Dom {
    let mut dom = Dom::new();
    let root = dom.root().expect("Dom::new always sets a root");
    let mut input = html;
    let mut stack: Vec<NodeId> = vec![root];
    while !input.is_empty() {
        if input.starts_with("<!--") {
            skip_comment(&mut input);
            continue;
        }
        if input.starts_with("<!") {
            // doctype and other <!...> declarations
            skip_decl(&mut input);
            continue;
        }
        if input.starts_with("</") {
            if let Some(name) = parse_close_tag(&mut input) {
                close_element(&mut stack, &name, &dom);
            }
            continue;
        }
        if input.starts_with('<') {
            match parse_open_tag(&mut input) {
                Some((tag, attrs, explicit_self_closing)) => {
                    let id = dom.create_element_with_attrs(&tag, attrs);
                    let parent = *stack.last().expect("stack never empties");
                    dom.append_child(parent, id);
                    if !self_closing(&tag, explicit_self_closing) {
                        if tag == "style" || tag == "script" {
                            // Raw text until the matching close tag.
                            let close = format!("</{tag}");
                            let end = find_ci(input, &close).unwrap_or(input.len());
                            let (raw, rest) = input.split_at(end);
                            if !raw.is_empty() {
                                let text = dom.create_text(raw);
                                dom.append_child(id, text);
                            }
                            input = rest;
                            if !input.is_empty() {
                                parse_close_tag(&mut input);
                            }
                        } else {
                            stack.push(id);
                        }
                    }
                }
                None => {
                    // Stray '<': keep it as literal text and move on.
                    push_text(&mut dom, &stack, "<");
                    input = &input[1..];
                }
            }
            continue;
        }
        // Text run up to the next '<' (or end of input).
        let end = input.find('<').unwrap_or(input.len());
        let (text, rest) = input.split_at(end);
        input = rest;
        push_text(&mut dom, &stack, text);
    }
    dom
}

/// Append a text node unless it is whitespace-only.
fn push_text(dom: &mut Dom, stack: &[NodeId], raw: &str) {
    let decoded = decode_entities(raw);
    if decoded.trim().is_empty() {
        return;
    }
    let parent = *stack.last().expect("stack never empties");
    let id = dom.create_text(&decoded);
    dom.append_child(parent, id);
}

/// Pop the stack up to and including the matching open element.
/// A close tag with no matching open element is ignored.
fn close_element(stack: &mut Vec<NodeId>, name: &str, dom: &Dom) {
    for (i, id) in stack.iter().enumerate().rev() {
        if let Some(node) = dom.get(*id) {
            if let NodeKind::Element { tag } = &node.kind {
                if tag == name {
                    stack.truncate(i);
                    return;
                }
            }
        }
    }
}

fn skip_comment(input: &mut &str) {
    match input.find("-->") {
        Some(p) => *input = &input[p + 3..],
        None => *input = "",
    }
}

fn skip_decl(input: &mut &str) {
    match input.find('>') {
        Some(p) => *input = &input[p + 1..],
        None => *input = "",
    }
}

/// Parse `</name>`; always advances past the tag (malformed close tags
/// are dropped through the next `>`). Returns the tag name on success.
fn parse_close_tag(input: &mut &str) -> Option<String> {
    let mut rest = *input;
    let parsed: Option<String> = (|| {
        lit("</").parse_next(&mut rest).ok()?;
        let name: &str = tw(
            1..,
            |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_',
        )
        .parse_next(&mut rest)
        .ok()?;
        let _: &str =
            tw(0.., |c: char| c.is_whitespace()).parse_next(&mut rest).ok()?;
        if !rest.starts_with('>') {
            return None;
        }
        rest = &rest[1..];
        Some(name.to_ascii_lowercase())
    })();
    match parsed {
        Some(name) => {
            *input = rest;
            Some(name)
        }
        None => {
            match input.find('>') {
                Some(p) => *input = &input[p + 1..],
                None => *input = "",
            }
            None
        }
    }
}

/// Parse `<name attr="value" ...>` / `<name ... />` on a scratch copy of
/// the input; the caller's input only moves on success. Returns
/// `(tag, attributes, self_closing)`; `None` means "not a tag".
fn parse_open_tag(input: &mut &str) -> Option<(String, Vec<(String, String)>, bool)> {
    let mut rest = *input;
    lit("<").parse_next(&mut rest).ok()?;
    let name: &str = tw(
        1..,
        |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_',
    )
    .parse_next(&mut rest)
    .ok()?;
    if !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let tag = name.to_ascii_lowercase();
    let mut attrs: Vec<(String, String)> = Vec::new();
    loop {
        let _: &str =
            tw(0.., |c: char| c.is_whitespace()).parse_next(&mut rest).ok()?;
        if rest.starts_with("/>") {
            rest = &rest[2..];
            *input = rest;
            return Some((tag, attrs, true));
        }
        if rest.starts_with('>') {
            rest = &rest[1..];
            *input = rest;
            return Some((tag, attrs, false));
        }
        if rest.is_empty() {
            return None;
        }
        let attr_name: &str = tw(
            1..,
            |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.'),
        )
        .parse_next(&mut rest)
        .ok()?;
        let _: &str =
            tw(0.., |c: char| c.is_whitespace()).parse_next(&mut rest).ok()?;
        let mut value = String::new();
        if rest.starts_with('=') {
            rest = &rest[1..];
            let _: &str =
                tw(0.., |c: char| c.is_whitespace()).parse_next(&mut rest).ok()?;
            if let Some(q) = rest.chars().next().filter(|c| *c == '"' || *c == '\'') {
                let after = &rest[1..];
                match after.find(q) {
                    Some(end) => {
                        value = after[..end].to_string();
                        rest = &after[end + 1..];
                    }
                    None => {
                        // Unterminated quote: the rest of the tag is the value.
                        value = after.to_string();
                        rest = "";
                    }
                }
            } else {
                // Unquoted values run to whitespace or `>`; `/` is a
                // legal value character (e.g. `src=/s.js`). A trailing
                // `/` right before `>` is the self-closing marker, not
                // part of the value (`<img src=x/>` means `x`).
                let mut v: &str = tw(1.., |c: char| !c.is_whitespace() && c != '>')
                    .parse_next(&mut rest)
                    .ok()?;
                if v.ends_with('/') && rest.starts_with('>') {
                    v = &v[..v.len() - 1];
                }
                value = v.to_string();
            }
        }
        // Browsers keep the first occurrence of a duplicate attribute.
        let attr_name = attr_name.to_ascii_lowercase();
        if !attrs.iter().any(|(n, _)| *n == attr_name) {
            attrs.push((attr_name, decode_entities(&value)));
        }
    }
}

/// HTML void elements: never have children, never need a close tag.
fn is_void(tag: &str) -> bool {
    matches!(
        tag,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "source"
            | "track"
            | "wbr"
    )
}

/// Decode `&amp; &lt; &gt; &quot; &apos; &nbsp;` and numeric
/// `&#NN;` / `&#xHH;` references. Unknown entities are left as-is.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        match after.find(';') {
            Some(semi) if semi <= 10 => {
                let entity = &after[..semi];
                let decoded: Option<char> = match entity {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some('\u{a0}'),
                    _ if let Some(num) = entity.strip_prefix("#x").or_else(|| entity.strip_prefix("#X")) => {
                        u32::from_str_radix(num, 16).ok().and_then(char::from_u32)
                    }
                    _ if let Some(num) = entity.strip_prefix('#') => {
                        num.parse::<u32>().ok().and_then(char::from_u32)
                    }
                    _ => None,
                };
                match decoded {
                    Some(c) => out.push(c),
                    None => out.push_str(&rest[amp..amp + 1 + semi + 1]),
                }
                rest = &after[semi + 1..];
            }
            _ => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Case-insensitive substring search (ASCII).
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hay = haystack.as_bytes();
    let ndl = needle.as_bytes();
    if ndl.is_empty() || hay.len() < ndl.len() {
        return None;
    }
    (0..=hay.len() - ndl.len()).find(|&i| {
        hay[i..i + ndl.len()]
            .iter()
            .zip(ndl.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

/// An element is closed when it self-closes (`/>`) or is a void element.
fn self_closing(tag: &str, explicit: bool) -> bool {
    explicit || is_void(tag)
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
    }

    #[test]
    fn entities_decoded() {
        let dom = parse("<p>Fish &amp; Chips &lt;tag&gt; &#65;&#x42; &quot;q&quot;</p>");
        let root = dom.root().unwrap();
        let p = find_tag(&dom, root, "p").unwrap();
        assert_eq!(text_content(&dom, p), "Fish & Chips <tag> AB \"q\"");
    }

    #[test]
    fn mismatched_close_ignored() {
        let dom = parse("<div><p>one</div><p>two</p>");
        let root = dom.root().unwrap();
        let div = find_tag(&dom, root, "div").unwrap();
        let mut ps_in_div = Vec::new();
        find_all_tags(&dom, div, "p", &mut ps_in_div);
        assert_eq!(ps_in_div.len(), 1);
        assert_eq!(text_content(&dom, ps_in_div[0]), "one");
        // The second <p> lands under the document, not inside the closed div.
        let mut all_ps = Vec::new();
        find_all_tags(&dom, root, "p", &mut all_ps);
        assert_eq!(all_ps.len(), 2);
        assert_eq!(text_content(&dom, all_ps[1]), "two");
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

        // A trailing slash right before `>` is the self-closing
        // marker, not part of the value.
        let dom = parse("<img src=/a/b.png/><img src=x/>");
        let root = dom.root().unwrap();
        let mut imgs = Vec::new();
        find_all_tags(&dom, root, "img", &mut imgs);
        assert_eq!(imgs.len(), 2);
        assert_eq!(dom.attr(imgs[0], "src"), Some("/a/b.png"));
        assert_eq!(dom.attr(imgs[1], "src"), Some("x"));
    }
}
