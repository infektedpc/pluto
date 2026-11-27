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

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use pluto_html::observers::{MutationObserverOptions, ObserverMutationRecord};
use pluto_html::{Dom, NodeId, NodeKind, parse, parse_fragment};

use crate::host::{
    AnimateSpec, AnimationInfo, CanvasCall, CanvasGradientKindSpec, CanvasImageData,
    CanvasOutcome, CanvasPaintSpec, DomHost, DomHostError, DomKey, Geometry, WaapiOp,
};

/// Host-visible side effects the test host records (navigation +
/// image requests), shared with the test through an `Arc` so the
/// test can read them after the host has moved into the tab.
#[derive(Debug, Default)]
pub struct TestHostLog {
    /// Navigation requests from `location.assign` / `replace` /
    /// `reload`, as `(url, replace)` in order.
    pub navigation_requests: Vec<(String, bool)>,
    /// History API requests (Stage 16 Phase 3, Rec 8): `pushState`
    /// / `replaceState` as `(url, state_json)` in order.
    pub history_pushes: Vec<(String, Option<String>)>,
    /// History API requests (Stage 16 Phase 3, Rec 8):
    /// `replaceState` as `(url, state_json)` in order.
    pub history_replaces: Vec<(String, Option<String>)>,
    /// History API requests (Stage 16 Phase 3, Rec 8):
    /// `back()` / `forward()` / `go()` as `delta` in order.
    pub history_traversals: Vec<i32>,
    /// Image requests from `Image()` / `<img>.src`, as
    /// `(url, base_url)` in order.
    pub image_requests: Vec<(String, String)>,
    /// WAAPI animation starts from `Element.animate`, as
    /// `(element, host anim id, spec)` in order (M6 Phase 3,
    /// Rec 11).
    pub animate_requests: Vec<(DomKey, u64, AnimateSpec)>,
    /// WAAPI play-state controls, as `(element, anim id, op)` in
    /// order.
    pub waapi_controls: Vec<(DomKey, u64, WaapiOp)>,
    /// CSS-animation play-state controls, as `(element, name, op)`.
    pub css_controls: Vec<(DomKey, String, WaapiOp)>,
}

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
    /// Per-geometry-call width growth (test-only): `geometry()`
    /// returns width `100 + growth * (calls - 1)`, so a
    /// `ResizeObserver` step can be made to never settle (the
    /// resize-loop-error test sets this). Default 0: geometry is
    /// a fixed 100×50.
    pub geometry_growth: Cell<f64>,
    /// The document's URL (Stage 12 Rec 7: `window.location` base).
    pub document_url: String,
    /// Shared side-effect log (navigation + image requests).
    pub log: Arc<Mutex<TestHostLog>>,
    /// Custom validity messages set by `setCustomValidity`.
    pub custom_validity: std::collections::HashMap<DomKey, String>,
    /// Form submissions queued by `form.submit()` /
    /// `requestSubmit()` for the (test) renderer to drain.
    pub pending_form_submits: Vec<(DomKey, Option<DomKey>)>,
    /// The focused element (`document.activeElement`).
    pub focused: Option<DomKey>,
    /// Computed-style cache for `getComputedStyle` tests: tests
    /// set pairs here directly (M6 Phase 3, Rec 11).
    pub computed_styles: std::collections::HashMap<DomKey, Vec<(String, String)>>,
    /// Next WAAPI animation id for `request_animate`.
    pub next_waapi_id: u64,
    /// WAAPI animations started via `request_animate`, by
    /// (element, id) — `get_animations` reports these.
    pub waapi: std::collections::HashMap<(DomKey, u64), AnimateSpec>,
    /// Finished WAAPI animations, by (element, id).
    pub waapi_finished: std::collections::HashSet<(DomKey, u64)>,
    /// CSS animations "running" per element, by name — tests set
    /// these directly to exercise `getAnimations`.
    pub css_animations: std::collections::HashMap<DomKey, Vec<String>>,
    /// Play-state controls received, in order — tests assert the
    /// binding forwarded pause/cancel/finish here.
    pub waapi_controls: Vec<(DomKey, u64, WaapiOp)>,
    /// CSS play-state controls received, in order.
    pub css_controls: Vec<(DomKey, String, WaapiOp)>,
    /// 2D canvas backing stores by element (Stage 15, M7 Phase
    /// 3, Rec 8/9): the real `pluto-paint` rasteriser, so binding
    /// tests exercise the production drawing code.
    pub canvases: std::collections::HashMap<DomKey, pluto_paint::canvas::Canvas2d>,
    /// Canvas gradients by host id (allocated by
    /// `CreateGradient`).
    pub canvas_gradients: std::collections::HashMap<u64, pluto_paint::canvas::CanvasGradient>,
    /// Next canvas gradient id.
    pub next_canvas_gradient_id: u64,
    /// Decoded pixels for `<img>` elements, by element
    /// (`(width, height, rgba, origin_clean)` — set via
    /// `SetImagePixels`, as the production host is fed by the
    /// tab thread when an image loads).
    pub canvas_image_pixels:
        std::collections::HashMap<DomKey, (u32, u32, Vec<u8>, bool)>,
}

impl TestHost {
    /// A host over a parsed document.
    pub fn from_html(html: &str) -> Self {
        Self::from_html_with_log(html, Arc::new(Mutex::new(TestHostLog::default())))
    }

    /// A host over a parsed document, recording side effects into
    /// `log` (shared with the test).
    pub fn from_html_with_log(html: &str, log: Arc<Mutex<TestHostLog>>) -> Self {
        Self {
            dom: parse(html),
            writes: Vec::new(),
            errors: Vec::new(),
            flushes: 0,
            geometry_growth: Cell::new(0.0),
            document_url: "https://example.com/".to_string(),
            log,
            custom_validity: std::collections::HashMap::new(),
            pending_form_submits: Vec::new(),
            focused: None,
            computed_styles: std::collections::HashMap::new(),
            next_waapi_id: 1,
            waapi: std::collections::HashMap::new(),
            waapi_finished: std::collections::HashSet::new(),
            css_animations: std::collections::HashMap::new(),
            waapi_controls: Vec::new(),
            css_controls: Vec::new(),
            canvases: std::collections::HashMap::new(),
            canvas_gradients: std::collections::HashMap::new(),
            next_canvas_gradient_id: 1,
            canvas_image_pixels: std::collections::HashMap::new(),
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
    fn clone_from(&mut self, frag: &Dom, id: NodeId) -> Option<DomKey> {        let node = frag.get(id)?;
        let new_key = match &node.kind {
            // Stage 15 (M7 Phase 1, Rec 1): a clone keeps its
            // namespace — a cloned `<svg><path>` stays SVG.
            NodeKind::Element { tag } => self.dom.create_element_with_attrs_ns(
                tag,
                node.attrs.clone(),
                &node.ns,
            ),
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

    /// Clone a node of *this* `Dom` (Stage 12 Rec 7 `cloneNode`):
    /// same kind / tag / attributes / text; children copied only
    /// when `deep`. The clone is detached (no parent).
    fn clone_within(&mut self, id: NodeId, deep: bool) -> Option<DomKey> {
        // Clone the kind + attributes first: creating the new node
        // mutates the arena, so no borrow of the old node may live
        // across the match.
        let (kind, attrs, ns) = {
            let n = self.dom.get(id)?;
            (n.kind.clone(), n.attrs.clone(), n.ns.clone())
        };
        let new_key = match &kind {
            // Stage 15 (M7 Phase 1, Rec 1): a clone keeps its
            // namespace — a cloned `<svg><path>` stays SVG.
            NodeKind::Element { tag } => {
                self.dom.create_element_with_attrs_ns(tag, attrs, &ns)
            }
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
        if deep {
            let kids: Vec<NodeId> = self.dom.children(id).to_vec();
            for kid in kids {
                if let Some(cloned) = self.clone_within(kid, true) {
                    self.dom.append_child(new_key, cloned);
                }
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

impl TestHost {
    /// A canvas's size from its `width` / `height` attributes
    /// (300×150 default).
    fn canvas_attr_size(&self, key: DomKey) -> (u32, u32) {
        let n = |name: &str, default: u32| {
            self.dom
                .attr(key, name)
                .and_then(|v| v.trim().parse::<f64>().ok())
                .map(|v| v.max(0.0) as u32)
                .unwrap_or(default)
        };
        (n("width", 300), n("height", 150))
    }

    /// Ensure a backing store exists for `key`, sized from the
    /// attributes (re-created — state reset — when the attributes
    /// moved since the store was made, e.g. via `setAttribute`).
    /// An oversize attribute pair falls back to 300×150 (the
    /// `Resize` path refuses oversize explicitly instead).
    fn canvas_ensure(&mut self, key: DomKey) -> &mut pluto_paint::canvas::Canvas2d {
        let (w, h) = self.canvas_attr_size(key);
        let needs_new = match self.canvases.get(&key) {
            Some(c) => c.width() != w || c.height() != h,
            None => true,
        };
        if needs_new {
            let canvas = pluto_paint::canvas::Canvas2d::new(w, h).unwrap_or_else(|| {
                pluto_paint::canvas::Canvas2d::new(300, 150)
                    .expect("300x150 is under the raster cap")
            });
            self.canvases.insert(key, canvas);
        }
        self.canvases.get_mut(&key).expect("just ensured")
    }

    /// A [`CanvasPaintSpec`] as a paint (a gradient id resolves
    /// through the gradient table; an unknown id is `None` and
    /// the caller keeps the old paint).
    fn paint_spec_to_paint(
        &self,
        spec: &CanvasPaintSpec,
    ) -> Option<pluto_paint::canvas::CanvasPaint> {
        use pluto_paint::canvas::CanvasPaint;
        match spec {
            CanvasPaintSpec::Color(c) => Some(CanvasPaint::Solid(*c)),
            CanvasPaintSpec::Gradient(id) => self
                .canvas_gradients
                .get(id)
                .cloned()
                .map(CanvasPaint::Gradient),
        }
    }

    /// A paint as a [`CanvasPaintSpec`] (a gradient resolves back
    /// to its host id by value).
    fn paint_to_spec(
        &self,
        paint: &pluto_paint::canvas::CanvasPaint,
    ) -> CanvasPaintSpec {
        use pluto_paint::canvas::CanvasPaint;
        match paint {
            CanvasPaint::Solid(c) => CanvasPaintSpec::Color(*c),
            CanvasPaint::Gradient(g) => {
                let id = self
                    .canvas_gradients
                    .iter()
                    .find(|(_, v)| **v == *g)
                    .map(|(id, _)| *id)
                    .unwrap_or(0);
                CanvasPaintSpec::Gradient(id)
            }
        }
    }
}

impl DomHost for TestHost {
    fn validate(&self, key: DomKey) -> bool {
        self.dom.get(key).is_some()
    }

    // ---- Stage 13 (M5 forms): delegate to the real `Dom` ----
    fn is_form_control(&self, key: DomKey) -> bool {
        self.dom.is_form_control(key)
    }

    fn input_type(&self, key: DomKey) -> Option<String> {
        self.dom.get(key).map(|_| self.dom.input_type(key))
    }

    fn is_disabled(&self, key: DomKey) -> bool {
        self.dom.is_control_disabled(key)
    }

    fn form_value(&self, key: DomKey) -> Option<String> {
        self.dom.form_value(key)
    }

    fn form_default_value(&self, key: DomKey) -> Option<String> {
        self.dom.form_default_value(key)
    }

    fn form_value_dirty(&self, key: DomKey) -> bool {
        self.dom.form_value_dirty(key)
    }

    fn set_form_value(&mut self, key: DomKey, value: &str) -> bool {
        self.dom.set_form_value(key, value)
    }

    fn form_checked(&self, key: DomKey) -> Option<bool> {
        self.dom.form_checked(key)
    }

    fn set_form_checked(&mut self, key: DomKey, checked: bool) -> bool {
        self.dom.set_form_checked(key, checked)
    }

    fn form_selected_index(&self, key: DomKey) -> Option<i64> {
        self.dom.form_selected_index(key)
    }

    fn set_form_selected_index(&mut self, key: DomKey, index: i64) -> bool {
        self.dom.set_form_selected_index(key, index)
    }

    fn form_options(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.select_options(key)
    }

    fn form_owner(&self, key: DomKey) -> Option<DomKey> {
        self.dom.form_owner(key)
    }

    fn form_controls(&self, key: DomKey) -> Vec<DomKey> {
        self.dom.form_controls(key)
    }

    fn document_forms(&self) -> Vec<DomKey> {
        let Some(root) = self.dom.root() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            if self.dom.tag(n) == Some("form") {
                out.push(n);
            }
            let mut kids = self.dom.children(n).to_vec();
            kids.reverse();
            stack.extend(kids);
        }
        // The stack walk above is pre-order only if children are
        // pushed reversed and popped in order — re-collect in
        // document order via a recursive walk instead.
        out.clear();
        fn walk(dom: &Dom, id: DomKey, out: &mut Vec<DomKey>) {
            if dom.tag(id) == Some("form") {
                out.push(id);
            }
            for &c in dom.children(id) {
                walk(dom, c, out);
            }
        }
        walk(&self.dom, root, &mut out);
        out
    }

    fn form_entries(&self, key: DomKey, submitter: Option<DomKey>) -> Vec<(String, String)> {
        self.dom.form_entries(key, submitter)
    }

    fn reset_form(&mut self, key: DomKey) -> bool {
        if self.dom.tag(key) != Some("form") {
            return false;
        }
        self.dom.reset_form(key);
        true
    }

    fn form_caret(&self, key: DomKey) -> Option<usize> {
        self.dom.form_caret(key)
    }

    fn form_selection(&self, key: DomKey) -> Option<(usize, usize)> {
        self.dom.form_selection(key)
    }

    fn set_form_selection(&mut self, key: DomKey, start: usize, end: usize) -> bool {
        self.dom.set_form_selection(key, start, end)
    }

    fn insert_form_text(&mut self, key: DomKey, text: &str) -> Option<usize> {
        self.dom.insert_form_text(key, text)
    }

    fn delete_form_text(&mut self, key: DomKey, backspace: bool) -> Option<usize> {
        self.dom.delete_form_text(key, backspace)
    }

    fn custom_validity(&self, key: DomKey) -> Option<String> {
        self.custom_validity.get(&key).cloned()
    }

    fn set_custom_validity(&mut self, key: DomKey, message: &str) -> bool {
        if self.dom.get(key).is_none() {
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
        self.focused.filter(|&k| self.dom.get(k).is_some())
    }

    fn focus_element(&mut self, key: DomKey) -> bool {
        if self.dom.get(key).is_none() || self.dom.is_control_disabled(key) {
            return false;
        }
        let focusable = self.dom.is_form_control(key)
            || matches!(self.dom.tag(key), Some("button"))
            || (self.dom.tag(key) == Some("a")
                && self.dom.attr(key, "href").is_some())
            || self.dom.attr(key, "tabindex").is_some();
        if !focusable {
            return false;
        }
        self.focused = Some(key);
        true
    }

    fn blur_element(&mut self) {
        self.focused = None;
    }

    fn request_form_submit(&mut self, form: DomKey, submitter: Option<DomKey>) -> bool {
        if self.dom.tag(form) != Some("form") {
            return false;
        }
        self.pending_form_submits.push((form, submitter));
        true
    }

    fn take_pending_form_submits(&mut self) -> Vec<(DomKey, Option<DomKey>)> {
        std::mem::take(&mut self.pending_form_submits)
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

    fn namespace_uri(&self, key: DomKey) -> Option<String> {
        self.dom.namespace(key).map(|ns| ns.to_string())
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
            // Stage 15 (M7 Phase 1, Rec 1): nodeName is the
            // uppercase tag for HTML elements, but the qualified
            // name as-is for foreign elements (case matters in
            // SVG: `foreignObject`, not `FOREIGNOBJECT`).
            NodeKind::Element { tag } => {
                if n.ns.is_empty() {
                    tag.to_ascii_uppercase()
                } else {
                    tag.clone()
                }
            }
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

    fn set_character_data(&mut self, key: DomKey, data: &str) -> bool {
        if !matches!(
            self.dom.get(key).map(|n| &n.kind),
            Some(NodeKind::Text(_))
        ) {
            return false;
        }
        self.dom.set_text(key, data);
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

    fn create_element_ns(&mut self, namespace: &str, tag: &str) -> Option<DomKey> {
        Some(self.dom.create_element_ns(namespace, tag))
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
        let growth = self.geometry_growth.get();
        Some(Geometry {
            x: 0.0,
            y: 0.0,
            width: 100.0 + growth * (self.flushes as f64 - 1.0),
            height: 50.0,
        })
    }

    fn content_size(&mut self, key: DomKey) -> Option<(f64, f64)> {
        self.dom.get(key)?;
        let growth = self.geometry_growth.get();
        Some((
            100.0 + growth * (self.flushes as f64 - 1.0).max(0.0),
            50.0,
        ))
    }

    fn depth(&self, key: DomKey) -> Option<usize> {
        self.dom.get(key)?;
        let mut depth = 0;
        let mut cur = self.dom.get(key).and_then(|n| n.parent);
        while let Some(parent) = cur {
            depth += 1;
            cur = self.dom.get(parent).and_then(|n| n.parent);
        }
        Some(depth)
    }

    fn observe_mutation(
        &mut self,
        observer_id: u64,
        target: DomKey,
        options: MutationObserverOptions,
    ) -> bool {
        self.dom.observe_mutation(observer_id, target, options)
    }

    fn disconnect_mutation_observer(&mut self, observer_id: u64) {
        self.dom.disconnect_mutation_observer(observer_id);
    }

    fn take_mutation_records(&mut self, observer_id: u64) -> Vec<ObserverMutationRecord> {
        self.dom.take_mutation_records(observer_id)
    }

    fn pending_mutation_observers(&self) -> Vec<u64> {
        self.dom.pending_mutation_observers()
    }

    fn take_pending_mutation_observers(&mut self) -> Vec<u64> {
        self.dom.take_pending_mutation_observers()
    }

    fn forced_flush_count(&self) -> u64 {
        self.flushes
    }

    fn report_error(&self, message: &str) {
        // `&self` only — the test host cannot push without interior
        // mutability; errors are visible via `TabJs::script_errors`.
        let _ = message;
    }

    fn document_url(&self) -> Option<String> {
        Some(self.document_url.clone())
    }

    fn request_navigation(&mut self, url: &str, replace: bool) {
        self.log
            .lock()
            .unwrap()
            .navigation_requests
            .push((url.to_string(), replace));
    }

    fn set_document_url(&mut self, url: &str) {
        self.document_url = url.to_string();
    }

    fn request_history_push(&mut self, url: &str, state_json: Option<&str>) {
        self.log
            .lock()
            .unwrap()
            .history_pushes
            .push((url.to_string(), state_json.map(|s| s.to_string())));
    }

    fn request_history_replace(&mut self, url: &str, state_json: Option<&str>) {
        self.log
            .lock()
            .unwrap()
            .history_replaces
            .push((url.to_string(), state_json.map(|s| s.to_string())));
    }

    fn request_history_traversal(&mut self, delta: i32) {
        self.log
            .lock()
            .unwrap()
            .history_traversals
            .push(delta);
    }

    fn attribute_names(&self, key: DomKey) -> Vec<String> {
        self.dom
            .get(key)
            .map(|n| n.attrs.iter().map(|(name, _)| name.clone()).collect())
            .unwrap_or_default()
    }

    fn clone_node(&mut self, key: DomKey, deep: bool) -> Option<DomKey> {
        self.dom.get(key)?;
        self.clone_within(key, deep)
    }

    // M6 Phase 3 (Rec 11): computed style + WAAPI. The test
    // host applies controls synchronously (no engine): pause /
    // cancel / finish update the records `get_animations` reads,
    // so binding tests can assert the round trip.
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
        self.waapi.insert((key, id), spec.clone());
        self.log
            .lock()
            .unwrap()
            .animate_requests
            .push((key, id, spec));
        Some(id)
    }

    fn get_animations(&self, key: DomKey) -> Vec<AnimationInfo> {
        let mut out: Vec<AnimationInfo> = self
            .waapi
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
        self.waapi_controls.push((key, anim_id, op));
        self.log
            .lock()
            .unwrap()
            .waapi_controls
            .push((key, anim_id, op));
        match op {
            WaapiOp::Cancel => {
                self.waapi.remove(&(key, anim_id));
                self.waapi_finished.remove(&(key, anim_id));
            }
            WaapiOp::Finish => {
                self.waapi_finished.insert((key, anim_id));
            }
            _ => {}
        }
    }

    fn css_animation_control(&mut self, key: DomKey, name: &str, op: WaapiOp) {
        self.css_controls.push((key, name.to_string(), op));
        self.log
            .lock()
            .unwrap()
            .css_controls
            .push((key, name.to_string(), op));
        if op == WaapiOp::Cancel
            && let Some(names) = self.css_animations.get_mut(&key) {
                names.retain(|n| n != name);
            }
    }

    fn request_image(&mut self, url: &str, base_url: &str) {
        self.log
            .lock()
            .unwrap()
            .image_requests
            .push((url.to_string(), base_url.to_string()));
    }

    // --- Stage 15 (M7 Phase 3, Rec 8/9): 2D canvas ---

    fn canvas_call(&mut self, key: DomKey, call: CanvasCall) -> CanvasOutcome {
        use pluto_paint::canvas::{
            Canvas2d, CanvasGradient, CanvasGradientKind, CanvasPaint, parse_canvas_color,
        };
        use pluto_paint::DecodedImage;
        match call {
            CanvasCall::Ensure => {
                let (w, h) = {
                    let c = self.canvas_ensure(key);
                    (c.width(), c.height())
                };
                CanvasOutcome::Size(w, h)
            }
            CanvasCall::Resize(w, h) => {
                let ok = match self.canvases.get_mut(&key) {
                    Some(c) => c.resize(w, h),
                    None => match Canvas2d::new(w, h) {
                        Some(c) => {
                            self.canvases.insert(key, c);
                            true
                        }
                        None => false,
                    },
                };
                CanvasOutcome::Bool(ok)
            }
            CanvasCall::SetFill(spec) => {
                if let Some(paint) = self.paint_spec_to_paint(&spec) {
                    self.canvas_ensure(key).set_fill(paint);
                }
                CanvasOutcome::None
            }
            CanvasCall::FillStyle => {
                let paint = self.canvas_ensure(key).fill_paint().clone();
                CanvasOutcome::Paint(self.paint_to_spec(&paint))
            }
            CanvasCall::SetFillText(text) => {
                // An unparseable colour leaves the paint
                // unchanged, per spec, and the getter then reads
                // the old value back.
                if let Some(color) = parse_canvas_color(&text) {
                    self.canvas_ensure(key).set_fill(CanvasPaint::Solid(color));
                }
                let paint = self.canvas_ensure(key).fill_paint().clone();
                CanvasOutcome::Paint(self.paint_to_spec(&paint))
            }
            CanvasCall::SetStroke(spec) => {
                if let Some(paint) = self.paint_spec_to_paint(&spec) {
                    self.canvas_ensure(key).set_stroke(paint);
                }
                CanvasOutcome::None
            }
            CanvasCall::StrokeStyle => {
                let paint = self.canvas_ensure(key).stroke_paint().clone();
                CanvasOutcome::Paint(self.paint_to_spec(&paint))
            }
            CanvasCall::SetStrokeText(text) => {
                if let Some(color) = parse_canvas_color(&text) {
                    self.canvas_ensure(key)
                        .set_stroke(CanvasPaint::Solid(color));
                }
                let paint = self.canvas_ensure(key).stroke_paint().clone();
                CanvasOutcome::Paint(self.paint_to_spec(&paint))
            }
            CanvasCall::SetLineWidth(w) => {
                self.canvas_ensure(key).set_line_width(w as f32);
                CanvasOutcome::None
            }
            CanvasCall::LineWidth => {
                let w = self.canvas_ensure(key).line_width();
                CanvasOutcome::Number(w as f64)
            }
            CanvasCall::SetLineCap(cap) => {
                self.canvas_ensure(key).set_line_cap(&cap);
                CanvasOutcome::None
            }
            CanvasCall::LineCap => {
                let cap = self.canvas_ensure(key).line_cap().to_string();
                CanvasOutcome::Text(cap)
            }
            CanvasCall::SetLineJoin(join) => {
                self.canvas_ensure(key).set_line_join(&join);
                CanvasOutcome::None
            }
            CanvasCall::LineJoin => {
                let join = self.canvas_ensure(key).line_join().to_string();
                CanvasOutcome::Text(join)
            }
            CanvasCall::SetFont(font) => {
                self.canvas_ensure(key).set_font(&font);
                CanvasOutcome::None
            }
            CanvasCall::Font => {
                let font = self.canvas_ensure(key).font().to_string();
                CanvasOutcome::Text(font)
            }
            CanvasCall::SetTextAlign(align) => {
                self.canvas_ensure(key).set_text_align(&align);
                CanvasOutcome::None
            }
            CanvasCall::TextAlign => {
                let align = self.canvas_ensure(key).text_align().to_string();
                CanvasOutcome::Text(align)
            }
            CanvasCall::SetTextBaseline(baseline) => {
                self.canvas_ensure(key).set_text_baseline(&baseline);
                CanvasOutcome::None
            }
            CanvasCall::TextBaseline => {
                let baseline = self.canvas_ensure(key).text_baseline().to_string();
                CanvasOutcome::Text(baseline)
            }
            CanvasCall::FillRect(x, y, w, h) => {
                self.canvas_ensure(key)
                    .fill_rect(x as f32, y as f32, w as f32, h as f32);
                CanvasOutcome::None
            }
            CanvasCall::ClearRect(x, y, w, h) => {
                self.canvas_ensure(key)
                    .clear_rect(x as f32, y as f32, w as f32, h as f32);
                CanvasOutcome::None
            }
            CanvasCall::StrokeRect(x, y, w, h) => {
                self.canvas_ensure(key)
                    .stroke_rect(x as f32, y as f32, w as f32, h as f32);
                CanvasOutcome::None
            }
            CanvasCall::BeginPath => {
                self.canvas_ensure(key).begin_path();
                CanvasOutcome::None
            }
            CanvasCall::MoveTo(x, y) => {
                self.canvas_ensure(key).move_to(x as f32, y as f32);
                CanvasOutcome::None
            }
            CanvasCall::LineTo(x, y) => {
                self.canvas_ensure(key).line_to(x as f32, y as f32);
                CanvasOutcome::None
            }
            CanvasCall::ClosePath => {
                self.canvas_ensure(key).close_path();
                CanvasOutcome::None
            }
            CanvasCall::RectPath(x, y, w, h) => {
                self.canvas_ensure(key)
                    .rect_path(x as f32, y as f32, w as f32, h as f32);
                CanvasOutcome::None
            }
            CanvasCall::Arc(x, y, r, start, end, ccw) => {
                self.canvas_ensure(key).arc(
                    x as f32,
                    y as f32,
                    r as f32,
                    start as f32,
                    end as f32,
                    ccw,
                );
                CanvasOutcome::None
            }
            CanvasCall::QuadraticCurveTo(cpx, cpy, x, y) => {
                self.canvas_ensure(key).quadratic_curve_to(
                    cpx as f32,
                    cpy as f32,
                    x as f32,
                    y as f32,
                );
                CanvasOutcome::None
            }
            CanvasCall::BezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y) => {
                self.canvas_ensure(key).bezier_curve_to(
                    cp1x as f32,
                    cp1y as f32,
                    cp2x as f32,
                    cp2y as f32,
                    x as f32,
                    y as f32,
                );
                CanvasOutcome::None
            }
            CanvasCall::Fill(evenodd) => {
                self.canvas_ensure(key).fill(evenodd);
                CanvasOutcome::None
            }
            CanvasCall::Stroke => {
                self.canvas_ensure(key).stroke();
                CanvasOutcome::None
            }
            CanvasCall::Clip => {
                self.canvas_ensure(key).clip();
                CanvasOutcome::None
            }
            CanvasCall::Save => {
                self.canvas_ensure(key).save();
                CanvasOutcome::None
            }
            CanvasCall::Restore => {
                self.canvas_ensure(key).restore();
                CanvasOutcome::None
            }
            CanvasCall::Translate(x, y) => {
                self.canvas_ensure(key).translate(x as f32, y as f32);
                CanvasOutcome::None
            }
            CanvasCall::Scale(x, y) => {
                self.canvas_ensure(key).scale(x as f32, y as f32);
                CanvasOutcome::None
            }
            CanvasCall::Rotate(rad) => {
                self.canvas_ensure(key).rotate(rad as f32);
                CanvasOutcome::None
            }
            CanvasCall::SetTransform(m) => {
                self.canvas_ensure(key).set_transform(pluto_css::Affine {
                    a: m[0] as f32,
                    b: m[1] as f32,
                    c: m[2] as f32,
                    d: m[3] as f32,
                    e: m[4] as f32,
                    f: m[5] as f32,
                });
                CanvasOutcome::None
            }
            CanvasCall::ResetTransform => {
                self.canvas_ensure(key).reset_transform();
                CanvasOutcome::None
            }
            CanvasCall::GetTransform => {
                let m = self.canvas_ensure(key).transform();
                CanvasOutcome::Transform([
                    m.a as f64,
                    m.b as f64,
                    m.c as f64,
                    m.d as f64,
                    m.e as f64,
                    m.f as f64,
                ])
            }
            CanvasCall::CreateGradient(spec) => {
                let kind = match spec.kind {
                    CanvasGradientKindSpec::Linear(x1, y1, x2, y2) => {
                        CanvasGradientKind::Linear {
                            x1: x1 as f32,
                            y1: y1 as f32,
                            x2: x2 as f32,
                            y2: y2 as f32,
                        }
                    }
                    CanvasGradientKindSpec::Radial(x1, y1, r1, x2, y2, r2) => {
                        CanvasGradientKind::Radial {
                            x1: x1 as f32,
                            y1: y1 as f32,
                            r1: r1 as f32,
                            x2: x2 as f32,
                            y2: y2 as f32,
                            r2: r2 as f32,
                        }
                    }
                };
                let mut gradient = CanvasGradient::new(kind);
                for (offset, color) in spec.stops {
                    gradient.add_stop(offset, color);
                }
                let id = self.next_canvas_gradient_id;
                self.next_canvas_gradient_id += 1;
                self.canvas_gradients.insert(id, gradient);
                CanvasOutcome::GradientId(id)
            }
            CanvasCall::AddColorStop(id, offset, color) => {
                let parsed = parse_canvas_color(&color);
                match (parsed, self.canvas_gradients.get_mut(&id)) {
                    (Some(c), Some(g)) => {
                        g.add_stop(offset, c);
                        CanvasOutcome::Bool(true)
                    }
                    _ => CanvasOutcome::Bool(false),
                }
            }
            CanvasCall::FillText(text, x, y, _max_width) => {
                self.canvas_ensure(key)
                    .fill_text(&text, x as f32, y as f32, false);
                CanvasOutcome::None
            }
            CanvasCall::StrokeText(text, x, y, _max_width) => {
                self.canvas_ensure(key)
                    .fill_text(&text, x as f32, y as f32, true);
                CanvasOutcome::None
            }
            CanvasCall::MeasureText(text) => {
                let w = self.canvas_ensure(key).measure_text(&text);
                CanvasOutcome::Number(w as f64)
            }
            CanvasCall::DrawImageCanvas {
                src,
                sx,
                sy,
                sw,
                sh,
                dx,
                dy,
                dw,
                dh,
            } => {
                let Some(src_img) = self.canvases.get(&src).map(|c| c.bitmap()) else {
                    return CanvasOutcome::Bool(false);
                };
                let src_tainted = self
                    .canvases
                    .get(&src)
                    .is_some_and(|c| !c.origin_clean());
                let drawn = self.canvas_ensure(key).draw_image(
                    &src_img,
                    (sx as f32, sy as f32, sw as f32, sh as f32),
                    (dx as f32, dy as f32, dw as f32, dh as f32),
                );
                if drawn && src_tainted {
                    self.canvas_ensure(key).taint();
                }
                CanvasOutcome::Bool(drawn)
            }
            CanvasCall::DrawImageElement {
                src,
                sx,
                sy,
                sw,
                sh,
                dx,
                dy,
                dw,
                dh,
            } => {
                let Some((w, h, rgba, origin_clean)) =
                    self.canvas_image_pixels.get(&src).cloned()
                else {
                    return CanvasOutcome::Bool(false);
                };
                let src_img = DecodedImage {
                    width: w,
                    height: h,
                    rgba,
                };
                let drawn = self.canvas_ensure(key).draw_image(
                    &src_img,
                    (sx as f32, sy as f32, sw as f32, sh as f32),
                    (dx as f32, dy as f32, dw as f32, dh as f32),
                );
                if drawn && !origin_clean {
                    self.canvas_ensure(key).taint();
                }
                CanvasOutcome::Bool(drawn)
            }
            CanvasCall::GetImageData(x, y, w, h) => {
                let (w, h) = (w.max(0.0) as u32, h.max(0.0) as u32);
                match self
                    .canvas_ensure(key)
                    .get_image_data(x as i32, y as i32, w, h)
                {
                    Ok(img) => CanvasOutcome::ImageData(CanvasImageData {
                        width: img.width,
                        height: img.height,
                        data: img.rgba,
                    }),
                    Err(pluto_paint::canvas::CanvasError::Tainted) => CanvasOutcome::Tainted,
                    Err(_) => CanvasOutcome::ImageData(CanvasImageData {
                        width: w,
                        height: h,
                        data: vec![0u8; (w as usize) * (h as usize) * 4],
                    }),
                }
            }
            CanvasCall::PutImageData {
                width,
                height,
                data,
                dx,
                dy,
            } => {
                let mut data = data;
                data.resize((width as usize) * (height as usize) * 4, 0);
                self.canvas_ensure(key).put_image_data(
                    &DecodedImage {
                        width,
                        height,
                        rgba: data,
                    },
                    dx as i32,
                    dy as i32,
                );
                CanvasOutcome::None
            }
            CanvasCall::ToDataUrl => match self.canvas_ensure(key).to_data_url() {
                Ok(url) => CanvasOutcome::Text(url),
                Err(pluto_paint::canvas::CanvasError::Tainted) => CanvasOutcome::Tainted,
                Err(_) => CanvasOutcome::Text("data:,".to_string()),
            },
            CanvasCall::OriginClean => {
                let clean = self
                    .canvases
                    .get(&key)
                    .is_some_and(|c| c.origin_clean());
                CanvasOutcome::Bool(clean)
            }
            CanvasCall::Bitmap => {
                let bmp = self.canvases.get(&key).map(|c| {
                    let img = c.bitmap();
                    (img.width, img.height, img.rgba)
                });
                CanvasOutcome::Bitmap(bmp)
            }
            CanvasCall::SetImagePixels {
                key: src,
                width,
                height,
                data,
                origin_clean,
            } => {
                self.canvas_image_pixels
                    .insert(src, (width, height, data, origin_clean));
                CanvasOutcome::None
            }
        }
    }

    fn canvas_live_keys(&self) -> Vec<DomKey> {
        self.dom
            .all_node_ids()
            .into_iter()
            .filter(|id| self.canvases.contains_key(id))
            .collect()
    }

    fn canvas_take_dirty(&mut self) -> Vec<(DomKey, u32, u32, Vec<u8>)> {
        let keys = self.canvas_live_keys();
        let mut out = Vec::new();
        for key in keys {
            if let Some(c) = self.canvases.get_mut(&key)
                && c.dirty()
            {
                let img = c.take_bitmap();
                out.push((key, img.width, img.height, img.rgba));
            }
        }
        out
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
