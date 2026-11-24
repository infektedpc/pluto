//! Layout & text (layer 4): taffy + cosmic-text bridge.
//!
//! Style resolution (UA defaults + matched stylesheet rules, later rules
//! win), a taffy tree built from the styled DOM, and text measurement via
//! `cosmic_text::Buffer` inside taffy's `measure_function`, so line
//! wrapping and intrinsic sizes match pixel-for-pixel what `pluto-paint`
//! later draws (it re-shapes the same spans at the same widths).
//!
//! M1 scope: block and flex layout, `display width height margin padding
//! font-size color background-color`. Inline elements (`a`, `span`, ...)
//! do not get their own boxes: their text merges into the parent block's
//! text leaf as spans (per-span color/bold), so mixed content wraps as
//! one paragraph. `em` units in margin/padding/font-size resolve against
//! the element's font size here (the CSS parser keeps them `Raw`).
//! `display: none` subtrees are skipped entirely. Everything else the
//! stylesheet says that M1 does not model is ignored, never an error.
//!
//! Style resolution runs as a separate two-phase pass before the taffy
//! tree is built (Stage 4 #5): phase 1 collects every element into
//! breadth-first levels from the root element, phase 2 resolves each
//! level's styles — in parallel on rayon's shared pool, parents before
//! children, so a level only ever reads styles resolved in earlier
//! levels — into an owned `DomId -> Arc<ComputedStyle>` map. The
//! `Builder` then looks styles up from that map instead of resolving
//! them inline. The same level loop can run sequentially
//! (`resolve_document_styles(..., parallel: false)`), so the two paths
//! stay comparable and a serial fallback is real.

use cosmic_text::{Attrs, Buffer, FontSystem, Metrics, Shaping, Weight};
use pluto_css::{Stylesheet, Value};
use pluto_html::{Dom, NodeId as DomId};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use taffy::{
    AvailableSpace, Dimension, Display, FlexDirection, LayoutInput, LayoutOutput, LengthPercentage,
    LengthPercentageAuto, NodeId as TaffyId, Size, Style, TaffyError, TaffyTree,
};

/// One styled run inside a text leaf.
#[derive(Debug, Clone)]
pub struct TextSpan {
    pub text: String,
    pub color: [u8; 3],
    pub bold: bool,
}

/// The text of one taffy leaf: merged spans of a block's inline content.
#[derive(Debug, Clone)]
pub struct LeafText {
    pub spans: Vec<TextSpan>,
    pub font_size: f32,
}

/// Taffy node context: `Some` for text leaves, `None` for elements.
pub type NodeContext = Option<LeafText>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dim {
    Px(f32),
    /// Fraction of the containing block (0.0..1.0).
    Pct(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spacing {
    Px(f32),
    /// Fraction of the containing block width (0.0..1.0).
    Pct(f32),
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayKind {
    Block,
    Flex,
    Inline,
    None,
}

/// The computed style of one element: UA defaults + matched rules.
#[derive(Debug, Clone, PartialEq)]
pub struct ComputedStyle {
    pub display: DisplayKind,
    pub width: Option<Dim>,
    pub height: Option<Dim>,
    pub margin: [Spacing; 4],  // top, right, bottom, left
    pub padding: [Spacing; 4], // top, right, bottom, left
    pub font_size: f32,
    pub color: [u8; 3],
    pub background: Option<[u8; 3]>,
    pub bold: bool,
}

/// One shaped glyph, in page coordinates (origin = glyph origin, i.e.
/// the point `SwashImage.placement` is relative to).
#[derive(Debug, Clone, Copy)]
pub struct ShapedGlyph {
    pub cache_key: cosmic_text::CacheKey,
    pub x: f32,
    pub y: f32,
    pub color: [u8; 3],
}

/// One laid-out box: an element or a text leaf, in page coordinates.
#[derive(Debug, Clone)]
pub struct LayoutBox {
    pub taffy: TaffyId,
    /// `None` for anonymous text leaves (merged inline content).
    pub dom: Option<DomId>,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub style: ComputedStyle,
    pub text: Option<LeafText>,
    /// Shaped glyphs for text leaves (empty otherwise).
    pub glyphs: Vec<ShapedGlyph>,
}

#[derive(Debug, Clone)]
struct BoxMeta {
    dom: Option<DomId>,
    style: ComputedStyle,
    children: Vec<TaffyId>,
    text: Option<LeafText>,
}

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("taffy error: {0}")]
    Taffy(#[from] TaffyError),
    #[error("document has no root element")]
    NoRoot,
}

/// Owns the Taffy layout tree and the cosmic-text font system for one
/// document, plus the laid-out boxes in document order.
pub struct LayoutTree {
    pub taffy: TaffyTree<NodeContext>,
    pub font_system: FontSystem,
    pub root: Option<TaffyId>,
    pub boxes: Vec<LayoutBox>,
    pub viewport: (f32, f32),
    /// Canvas background: the html element's background, else the body's
    /// (CSS canvas rule), else white — a browser's canvas defaults to
    /// white when neither element sets one. Painted over the whole
    /// viewport by the display list, not just the element's box.
    pub canvas_background: [u8; 3],
}

impl LayoutTree {
    pub fn new() -> Self {
        Self {
            taffy: TaffyTree::new(),
            font_system: FontSystem::new(),
            root: None,
            boxes: Vec::new(),
            viewport: (0.0, 0.0),
            canvas_background: [255, 255, 255],
        }
    }

    /// Lay out `dom` with `sheet` in a viewport of the given size.
    /// Returns the tree with `boxes` in document order, positions in
    /// page coordinates (origin = viewport top-left).
    pub fn build(
        dom: &Dom,
        sheet: &Stylesheet,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Result<Self, LayoutError> {
        let mut font_system = FontSystem::new();
        // Two-phase style resolution (Stage 4 #5): every element's
        // style is resolved up front, in parallel, before any taffy
        // node is built. The builder looks styles up from this map.
        let styles = resolve_document_styles(dom, sheet, true);
        let mut builder = Builder {
            dom,
            sheet,
            styles,
            taffy: TaffyTree::new(),
            meta: HashMap::new(),
        };
        let root_dom = find_element(dom, dom.root().unwrap(), "html");
        let root = match root_dom {
            Some(html) => builder
                .build_node(html, 16.0, [0, 0, 0])
                .ok_or(LayoutError::NoRoot)?,
            None => {
                // Fragment without <html>: lay out the document's children
                // directly under a synthetic block root.
                let default_style = ua_default("body", 16.0, [0, 0, 0]);
                let children =
                    builder.build_children(dom.root().unwrap(), &default_style);
                let node = builder
                    .taffy
                    .new_with_children(to_taffy_style(&default_style), &children)?;
                builder.meta.insert(
                    node,
                    BoxMeta {
                        dom: None,
                        style: default_style,
                        children,
                        text: None,
                    },
                );
                node
            }
        };
        let mut taffy = builder.taffy;
        let meta = builder.meta;

        // Text leaves are measured by shaping them in a cosmic-text Buffer.
        let fs = &mut font_system;
        taffy.compute_layout_with_measure(
            root,
            Size {
                width: AvailableSpace::Definite(viewport_width),
                height: AvailableSpace::MaxContent,
            },
            |input, _node, ctx, _style| measure_text(fs, ctx, input),
        )?;

        let mut boxes = Vec::new();
        collect_boxes(&taffy, &meta, root, (0.0, 0.0), &mut boxes);
        for bx in boxes.iter_mut() {
            if bx.text.is_some() {
                bx.glyphs = shape_leaf(&mut font_system, bx);
            }
        }
        // Canvas background: html's, else body's (CSS canvas rule),
        // else white — the browser default when neither sets one.
        let html_bg = boxes
            .iter()
            .find(|b| b.taffy == root)
            .and_then(|b| b.style.background);
        let body_bg = root_dom
            .and_then(|r| find_element(dom, r, "body"))
            .and_then(|body_id| boxes.iter().find(|b| b.dom == Some(body_id)))
            .and_then(|b| b.style.background);
        let canvas_background = html_bg.or(body_bg).unwrap_or([255, 255, 255]);

        Ok(Self {
            taffy,
            font_system,
            root: Some(root),
            boxes,
            viewport: (viewport_width, viewport_height),
            canvas_background,
        })
    }

    /// Boxes in document order.
    pub fn boxes(&self) -> &[LayoutBox] {
        &self.boxes
    }

    /// Total content height (bottom of the lowest box).
    pub fn content_height(&self) -> f32 {
        self.boxes
            .iter()
            .map(|b| b.y + b.height)
            .fold(0.0, f32::max)
    }
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Depth-first search for the first element with this tag.
fn find_element(dom: &Dom, id: DomId, tag: &str) -> Option<DomId> {
    for &child in dom.children(id) {
        if dom.tag(child) == Some(tag) {
            return Some(child);
        }
        if let Some(found) = find_element(dom, child, tag) {
            return Some(found);
        }
    }
    None
}

struct Builder<'a> {
    dom: &'a Dom,
    sheet: &'a Stylesheet,
    /// Pre-resolved styles for every element (two-phase pass, see
    /// `resolve_document_styles`). `Arc`-shared: computed styles are
    /// immutable once resolved, the Stylo pattern.
    styles: HashMap<DomId, Arc<ComputedStyle>>,
    taffy: TaffyTree<NodeContext>,
    meta: HashMap<TaffyId, BoxMeta>,
}

impl<'a> Builder<'a> {
    /// Style for `id` from the pre-resolved map. Falls back to an
    /// inline resolve with the same inputs if the id was never
    /// collected — e.g. a node outside the `<html>` subtree — so
    /// behaviour stays identical; in practice the map covers every
    /// element the builder visits.
    fn style_for(
        &self,
        id: DomId,
        parent_font_size: f32,
        parent_color: [u8; 3],
    ) -> ComputedStyle {
        match self.styles.get(&id) {
            Some(style) => (**style).clone(),
            None => resolve_style(self.dom, id, self.sheet, parent_font_size, parent_color),
        }
    }

    /// Build a taffy node for one element (block or flex). Returns `None`
    /// for `display: none` and for inline elements (handled by the
    /// parent's inline pass).
    fn build_node(
        &mut self,
        id: DomId,
        parent_font_size: f32,
        parent_color: [u8; 3],
    ) -> Option<TaffyId> {
        let style = self.style_for(id, parent_font_size, parent_color);
        match style.display {
            DisplayKind::None | DisplayKind::Inline => None,
            _ => {
                let children = self.build_children(id, &style);
                let taffy_style = to_taffy_style(&style);
                let node = if children.is_empty() {
                    self.taffy
                        .new_leaf_with_context(taffy_style, None)
                        .ok()?
                } else {
                    self.taffy.new_with_children(taffy_style, &children).ok()?
                };
                self.meta.insert(
                    node,
                    BoxMeta {
                        dom: Some(id),
                        style,
                        children,
                        text: None,
                    },
                );
                Some(node)
            }
        }
    }

    /// Build a block container's children: block-level children become
    /// taffy nodes; text nodes and inline elements merge into text leaves.
    fn build_children(&mut self, parent: DomId, parent_style: &ComputedStyle) -> Vec<TaffyId> {
        let mut out = Vec::new();
        let mut spans: Vec<TextSpan> = Vec::new();
        for &child in self.dom.children(parent) {
            if let Some(text) = self.dom.text(child) {
                if !text.trim().is_empty() {
                    spans.push(TextSpan {
                        text: text.to_string(),
                        color: parent_style.color,
                        bold: parent_style.bold,
                    });
                }
                continue;
            }
            if self.dom.tag(child).is_none() {
                continue;
            }
            let style = self.style_for(child, parent_style.font_size, parent_style.color);
            match style.display {
                DisplayKind::None => {}
                DisplayKind::Inline => {
                    self.append_inline(child, &style, parent_style, &mut spans, &mut out);
                }
                _ => {
                    self.flush_spans(&mut spans, parent_style, &mut out);
                    if let Some(node) =
                        self.build_node(child, parent_style.font_size, parent_style.color)
                    {
                        out.push(node);
                    }
                }
            }
        }
        self.flush_spans(&mut spans, parent_style, &mut out);
        out
    }

    /// Recursively collect an inline element's text into `spans`. A
    /// block-level descendant flushes the spans and becomes a sibling
    /// block child instead (browsers split inline boxes the same way).
    fn append_inline(
        &mut self,
        id: DomId,
        style: &ComputedStyle,
        parent_style: &ComputedStyle,
        spans: &mut Vec<TextSpan>,
        out: &mut Vec<TaffyId>,
    ) {
        for &child in self.dom.children(id) {
            if let Some(text) = self.dom.text(child) {
                if !text.trim().is_empty() {
                    spans.push(TextSpan {
                        text: text.to_string(),
                        color: style.color,
                        bold: style.bold,
                    });
                }
                continue;
            }
            if self.dom.tag(child).is_none() {
                continue;
            }
            let child_style = self.style_for(child, style.font_size, style.color);
            match child_style.display {
                DisplayKind::None => {}
                DisplayKind::Inline => {
                    self.append_inline(child, &child_style, parent_style, spans, out);
                }
                _ => {
                    self.flush_spans(spans, parent_style, out);
                    if let Some(node) =
                        self.build_node(child, style.font_size, style.color)
                    {
                        out.push(node);
                    }
                }
            }
        }
    }

    /// Turn accumulated inline spans into a taffy text leaf.
    fn flush_spans(
        &mut self,
        spans: &mut Vec<TextSpan>,
        parent_style: &ComputedStyle,
        out: &mut Vec<TaffyId>,
    ) {
        if spans.is_empty() {
            return;
        }
        let leaf = LeafText {
            spans: std::mem::take(spans),
            font_size: parent_style.font_size,
        };
        let node = self
            .taffy
            .new_leaf_with_context(Style::default(), Some(leaf.clone()))
            .expect("taffy insert cannot fail");
        self.meta.insert(
            node,
            BoxMeta {
                dom: None,
                style: parent_style.clone(),
                children: Vec::new(),
                text: Some(leaf),
            },
        );
        out.push(node);
    }
}

/// Measure one text leaf by shaping it in a cosmic-text `Buffer`.
/// Called by taffy whenever it needs the leaf's size.
fn measure_text(
    font_system: &mut FontSystem,
    ctx: Option<&mut NodeContext>,
    input: LayoutInput,
) -> LayoutOutput {
    let leaf = match ctx {
        Some(c) => c.as_ref(),
        None => None,
    };
    let Some(leaf) = leaf else {
        // Non-text leaf (empty element): size from known dimensions, else 0.
        return LayoutOutput {
            size: Size {
                width: input.known_dimensions.width.unwrap_or(0.0),
                height: input.known_dimensions.height.unwrap_or(0.0),
            },
            ..LayoutOutput::HIDDEN
        };
    };
    let width_constraint = input.known_dimensions.width.or(match input.available_space.width {
        AvailableSpace::Definite(w) => Some(w),
        _ => None,
    });
    let mut buffer = Buffer::new(
        font_system,
        Metrics::new(leaf.font_size, leaf.font_size * 1.2),
    );
    set_buffer_text(&mut buffer, leaf);
    buffer.set_size(width_constraint, None);
    // `layout_runs` does NOT shape: the buffer must be shaped first
    // (the auto-shaping `layout_runs` lives on `BorrowedWithFontSystem`).
    buffer.shape_until_scroll(font_system, false);
    let mut width = 0.0f32;
    let mut height = 0.0f32;
    for run in buffer.layout_runs() {
        width = width.max(run.line_w);
        height = run.line_y + run.line_height;
    }
    LayoutOutput {
        size: Size {
            width: width_constraint.unwrap_or(width),
            height,
        },
        ..LayoutOutput::HIDDEN
    }
}

/// Fill a `Buffer` with a leaf's spans (single span: `set_text`,
/// several: `set_rich_text` so wrapping sees the whole paragraph).
fn set_buffer_text(buffer: &mut Buffer, leaf: &LeafText) {
    if leaf.spans.len() == 1 {
        let span = &leaf.spans[0];
        let attrs = Attrs::new().weight(if span.bold {
            Weight::BOLD
        } else {
            Weight::NORMAL
        });
        buffer.set_text(&span.text, &attrs, Shaping::Advanced, None);
    } else {
        let spans: Vec<(&str, Attrs)> = leaf
            .spans
            .iter()
            .map(|span| {
                (
                    span.text.as_str(),
                    Attrs::new().weight(if span.bold {
                        Weight::BOLD
                    } else {
                        Weight::NORMAL
                    }),
                )
            })
            .collect();
        buffer.set_rich_text(spans, &Attrs::new(), Shaping::Advanced, None);
    }
}

/// Shape one laid-out text box into glyphs at its final size.
/// The same inputs as `measure_text`, so positions agree with layout.
fn shape_leaf(font_system: &mut FontSystem, bx: &LayoutBox) -> Vec<ShapedGlyph> {
    let Some(leaf) = bx.text.as_ref() else {
        return Vec::new();
    };
    let mut buffer = Buffer::new(
        font_system,
        Metrics::new(leaf.font_size, leaf.font_size * 1.2),
    );
    set_buffer_text(&mut buffer, leaf);
    buffer.set_size(Some(bx.width), None);
    // See `measure_text`: `layout_runs` does not shape on its own.
    buffer.shape_until_scroll(font_system, false);

    // Byte ranges of each span in the concatenated text, for color lookup.
    let mut ranges: Vec<(std::ops::Range<usize>, [u8; 3])> = Vec::new();
    let mut off = 0;
    for span in &leaf.spans {
        ranges.push((off..off + span.text.len(), span.color));
        off += span.text.len();
    }
    let full_text: String = leaf.spans.iter().map(|s| s.text.as_str()).collect();
    // Start byte of each '\n'-separated line in the concatenated text.
    let mut line_starts = vec![0usize];
    for (i, b) in full_text.bytes().enumerate() {
        if b == b'\n' {
            line_starts.push(i + 1);
        }
    }

    let mut glyphs = Vec::new();
    for run in buffer.layout_runs() {
        // `run.text` is the whole source line in cosmic-text 0.19,
        // not the visual line's substring — do not slice or search
        // it. Each glyph carries its own byte offset within the
        // source line (`LayoutGlyph.start`), which is what locates
        // the glyph for its span color.
        let line_start = line_starts.get(run.line_i).copied().unwrap_or(0);
        for g in run.glyphs {
            let abs = line_start + g.start;
            let color = ranges
                .iter()
                .find(|(r, _)| r.contains(&abs))
                .map(|(_, c)| *c)
                .unwrap_or([0, 0, 0]);
            let physical = g.physical((bx.x, bx.y + run.line_y), 1.0);
            glyphs.push(ShapedGlyph {
                cache_key: physical.cache_key,
                x: physical.x as f32,
                y: physical.y as f32,
                color,
            });
        }
    }
    glyphs
}

/// Walk the taffy tree from `node`, collecting boxes with absolute
/// (page) coordinates.
fn collect_boxes(
    taffy: &TaffyTree<NodeContext>,
    meta: &HashMap<TaffyId, BoxMeta>,
    node: TaffyId,
    offset: (f32, f32),
    out: &mut Vec<LayoutBox>,
) {
    let Some(layout) = taffy.layout(node).ok() else {
        return;
    };
    let Some(m) = meta.get(&node) else {
        return;
    };
    let x = offset.0 + layout.location.x;
    let y = offset.1 + layout.location.y;
    out.push(LayoutBox {
        taffy: node,
        dom: m.dom,
        x,
        y,
        width: layout.size.width,
        height: layout.size.height,
        style: m.style.clone(),
        text: m.text.clone(),
        glyphs: Vec::new(),
    });
    for &child in &m.children {
        collect_boxes(taffy, meta, child, (x, y), out);
    }
}

// ---- two-phase style resolution (Stage 4 #5) ----

/// Phase 1: collect every element node of the DOM into breadth-first
/// levels from `roots` (level 0). Each entry knows its parent element
/// (`None` for a root). Only elements are collected — text nodes carry
/// no style and never pass inheritance on — but `display: none`
/// subtrees are collected too: their styles are computed and unused,
/// which keeps collection independent of style values. Which nodes get
/// boxes is unchanged; the builder still skips `display: none`.
fn collect_style_levels(
    dom: &Dom,
    roots: &[DomId],
) -> Vec<Vec<(DomId, Option<DomId>)>> {
    let mut levels: Vec<Vec<(DomId, Option<DomId>)>> = Vec::new();
    let mut frontier: Vec<(DomId, Option<DomId>)> = roots
        .iter()
        .filter(|&&id| dom.tag(id).is_some())
        .map(|&id| (id, None))
        .collect();
    while !frontier.is_empty() {
        let mut next: Vec<(DomId, Option<DomId>)> = Vec::new();
        for &(id, _) in &frontier {
            for &child in dom.children(id) {
                if dom.tag(child).is_some() {
                    next.push((child, Some(id)));
                }
            }
        }
        levels.push(std::mem::take(&mut frontier));
        frontier = next;
    }
    levels
}

/// Phase 2: resolve one style per entry of `levels`, level by level.
/// A level's parents were all resolved in earlier levels, so each
/// entry's parent font size / color is read from the already-resolved
/// map and the entries of one level are independent: with
/// `parallel` they resolve on rayon's shared global pool via
/// `par_iter` over `&Dom` / `&Stylesheet` (both `Sync`, asserted in
/// tests); without it the same loop runs sequentially, so the serial
/// path is the same code path and a real fallback, not a comment.
/// Returns an owned map from element to its (immutable, `Arc`-shared)
/// computed style.
fn resolve_styles(
    dom: &Dom,
    sheet: &Stylesheet,
    levels: &[Vec<(DomId, Option<DomId>)>],
    parallel: bool,
) -> HashMap<DomId, Arc<ComputedStyle>> {
    let mut resolved: HashMap<DomId, Arc<ComputedStyle>> = HashMap::new();
    for level in levels {
        // Parent font size / color first (serial, cheap): the parallel
        // closure then only reads `&Dom` / `&Stylesheet` plus its own
        // inputs, never the map being built.
        let inputs: Vec<(DomId, f32, [u8; 3])> = level
            .iter()
            .map(|&(id, parent)| {
                let (font_size, color) = match parent {
                    Some(p) => {
                        let parent_style = resolved
                            .get(&p)
                            .expect("parent style resolved in an earlier level");
                        (parent_style.font_size, parent_style.color)
                    }
                    None => (16.0, [0, 0, 0]),
                };
                (id, font_size, color)
            })
            .collect();
        let resolve_one = |&(id, font_size, color): &(DomId, f32, [u8; 3])| {
            (
                id,
                Arc::new(resolve_style(dom, id, sheet, font_size, color)),
            )
        };
        let resolved_level: Vec<(DomId, Arc<ComputedStyle>)> = if parallel {
            inputs.par_iter().map(resolve_one).collect()
        } else {
            inputs.iter().map(resolve_one).collect()
        };
        for (id, style) in resolved_level {
            resolved.insert(id, style);
        }
    }
    resolved
}

/// Resolve styles for the whole document: phase 1 from the `<html>`
/// element, or — for a fragment without one — from the document
/// root's element children (whose parent style is the same synthetic
/// body default the builder uses), then phase 2. `LayoutTree::build`
/// calls this with `parallel: true`.
fn resolve_document_styles(
    dom: &Dom,
    sheet: &Stylesheet,
    parallel: bool,
) -> HashMap<DomId, Arc<ComputedStyle>> {
    let root = dom.root().expect("Dom always has a root node");
    let roots: Vec<DomId> = match find_element(dom, root, "html") {
        Some(html) => vec![html],
        None => dom
            .children(root)
            .iter()
            .copied()
            .filter(|&id| dom.tag(id).is_some())
            .collect(),
    };
    let levels = collect_style_levels(dom, &roots);
    resolve_styles(dom, sheet, &levels, parallel)
}

// ---- style resolution ----

/// UA defaults for one tag, then matched stylesheet rules in order.
fn resolve_style(
    dom: &Dom,
    id: DomId,
    sheet: &Stylesheet,
    parent_font_size: f32,
    parent_color: [u8; 3],
) -> ComputedStyle {
    let tag = dom.tag(id).unwrap_or("");
    let mut style = ua_default(tag, parent_font_size, parent_color);
    let class_attr = dom.attr(id, "class").unwrap_or("");
    let classes: Vec<&str> = class_attr.split_whitespace().collect();
    for rule in &sheet.rules {
        if rule
            .selectors
            .iter()
            .any(|sel| sel.matches(tag, dom.attr(id, "id"), &classes))
        {
            for decl in &rule.declarations {
                apply_decl(&mut style, &decl.property, &decl.value, parent_font_size);
            }
        }
    }
    style
}

fn ua_default(tag: &str, parent_font_size: f32, parent_color: [u8; 3]) -> ComputedStyle {
    let display = match tag {
        "head" | "title" | "style" | "script" | "meta" | "link" | "base" => DisplayKind::None,
        "span" | "a" | "b" | "i" | "em" | "strong" | "small" | "code" => DisplayKind::Inline,
        _ => DisplayKind::Block,
    };
    let font_size = match tag {
        "h1" => 32.0,
        "h2" => 24.0,
        "h3" => 18.72,
        "h4" => 16.0,
        "h5" => 13.28,
        "h6" => 10.72,
        "small" => parent_font_size * 0.833,
        _ => parent_font_size,
    };
    // CSS default margins, in em of the element's own font size.
    let margin_em = match tag {
        "body" => [0.5, 0.5, 0.5, 0.5], // body: 8px at 16px font, as em
        "p" => [1.0, 0.0, 1.0, 0.0],
        "h1" => [0.67, 0.0, 0.67, 0.0],
        "h2" => [0.83, 0.0, 0.83, 0.0],
        "h3" => [1.0, 0.0, 1.0, 0.0],
        "h4" => [1.33, 0.0, 1.33, 0.0],
        "h5" => [1.67, 0.0, 1.67, 0.0],
        "h6" => [2.33, 0.0, 2.33, 0.0],
        _ => [0.0; 4],
    };
    // body margin is 8px flat, not 0.5em of a 32px h1 font; fix it up.
    let margin: [Spacing; 4] = if tag == "body" {
        [Spacing::Px(8.0); 4]
    } else {
        margin_em.map(|e| Spacing::Px(e * font_size))
    };
    let color = match tag {
        "a" => [0, 0, 238],
        _ => parent_color,
    };
    ComputedStyle {
        display,
        width: None,
        height: None,
        margin,
        padding: [Spacing::Px(0.0); 4],
        font_size,
        color,
        background: None,
        bold: matches!(tag, "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "b" | "strong"),
    }
}

fn apply_decl(style: &mut ComputedStyle, property: &str, value: &Value, parent_font_size: f32) {
    match property {
        "display" => {
            if let Value::Keyword(k) = value {
                style.display = match k.as_str() {
                    "block" => DisplayKind::Block,
                    "flex" => DisplayKind::Flex,
                    "inline" => DisplayKind::Inline,
                    "none" => DisplayKind::None,
                    _ => return,
                };
            }
        }
        "width" => style.width = dim_value(value),
        "height" => style.height = dim_value(value),
        "margin-top" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.margin[0] = s;
            }
        }
        "margin-right" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.margin[1] = s;
            }
        }
        "margin-bottom" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.margin[2] = s;
            }
        }
        "margin-left" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.margin[3] = s;
            }
        }
        "margin" => {
            if let Value::Raw(raw) = value {
                if let Some(sides) = parse_shorthand(raw, style.font_size) {
                    style.margin = sides;
                }
            }
        }
        "padding-top" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.padding[0] = s;
            }
        }
        "padding-right" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.padding[1] = s;
            }
        }
        "padding-bottom" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.padding[2] = s;
            }
        }
        "padding-left" => {
            if let Some(s) = spacing_value(value, style.font_size) {
                style.padding[3] = s;
            }
        }
        "padding" => {
            if let Value::Raw(raw) = value {
                if let Some(sides) = parse_shorthand(raw, style.font_size) {
                    style.padding = sides;
                }
            }
        }
        "font-size" => match value {
            Value::Length(v) => style.font_size = *v,
            Value::Percentage(p) => style.font_size = parent_font_size * p / 100.0,
            Value::Keyword(k) => {
                style.font_size = match k.as_str() {
                    "xx-small" => 9.0,
                    "x-small" => 10.0,
                    "small" => 13.0,
                    "medium" => 16.0,
                    "large" => 18.0,
                    "x-large" => 24.0,
                    "xx-large" => 32.0,
                    _ => return,
                }
            }
            Value::Raw(raw) => {
                if let Some(Spacing::Px(v)) = parse_spacing_single(raw, parent_font_size) {
                    style.font_size = v;
                }
            }
            Value::Color(..) => {}
        },
        "color" => {
            if let Value::Color(r, g, b) = value {
                style.color = [*r, *g, *b];
            }
        }
        "background-color" => {
            if let Value::Color(r, g, b) = value {
                style.background = Some([*r, *g, *b]);
            }
        }
        "font-weight" => match value {
            Value::Keyword(k) => {
                style.bold = k == "bold";
            }
            Value::Raw(raw) => {
                if let Ok(n) = raw.trim().parse::<f32>() {
                    style.bold = n >= 600.0;
                }
            }
            _ => {}
        },
        _ => {}
    }
}

fn dim_value(value: &Value) -> Option<Dim> {
    match value {
        Value::Length(v) => Some(Dim::Px(*v)),
        Value::Percentage(p) => Some(Dim::Pct(p / 100.0)),
        _ => None,
    }
}

/// A margin/padding value, resolving `em` against the font size.
fn spacing_value(value: &Value, font_size: f32) -> Option<Spacing> {
    match value {
        Value::Length(v) => Some(Spacing::Px(*v)),
        Value::Percentage(p) => Some(Spacing::Pct(p / 100.0)),
        Value::Keyword(k) if k == "auto" => Some(Spacing::Auto),
        Value::Raw(raw) => parse_spacing_single(raw, font_size),
        _ => None,
    }
}

fn parse_spacing_single(s: &str, font_size: f32) -> Option<Spacing> {
    let s = s.trim().to_ascii_lowercase();
    if s == "auto" {
        return Some(Spacing::Auto);
    }
    if let Some(n) = s.strip_suffix("px") {
        return n.trim().parse().ok().map(Spacing::Px);
    }
    if let Some(n) = s.strip_suffix('%') {
        return n.trim().parse().ok().map(|v: f32| Spacing::Pct(v / 100.0));
    }
    if let Some(n) = s.strip_suffix("em") {
        return n
            .trim()
            .parse()
            .ok()
            .map(|v: f32| Spacing::Px(v * font_size));
    }
    s.parse::<f32>().ok().map(Spacing::Px)
}

/// Expand a 1-4 token `margin`/`padding` shorthand (`Raw` form, i.e. it
/// contained an `em` the CSS parser could not expand). `auto` is kept
/// for margins; for padding it becomes 0.
fn parse_shorthand(raw: &str, font_size: f32) -> Option<[Spacing; 4]> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        vals.push(parse_spacing_single(t, font_size)?);
    }
    let (top, right, bottom, left) = match vals.len() {
        1 => (vals[0], vals[0], vals[0], vals[0]),
        2 => (vals[0], vals[1], vals[0], vals[1]),
        3 => (vals[0], vals[1], vals[2], vals[1]),
        4 => (vals[0], vals[1], vals[2], vals[3]),
        _ => unreachable!(),
    };
    Some([top, right, bottom, left])
}

fn to_taffy_style(style: &ComputedStyle) -> Style {
    let mut s = Style::default();
    s.display = match style.display {
        DisplayKind::Block | DisplayKind::Inline => Display::Block,
        DisplayKind::Flex => Display::Flex,
        DisplayKind::None => Display::None,
    };
    if let Some(w) = style.width {
        s.size.width = match w {
            Dim::Px(v) => Dimension::length(v),
            Dim::Pct(f) => Dimension::percent(f),
        };
    }
    if let Some(h) = style.height {
        s.size.height = match h {
            Dim::Px(v) => Dimension::length(v),
            Dim::Pct(f) => Dimension::percent(f),
        };
    }
    let margin = |sp: Spacing| match sp {
        Spacing::Px(v) => LengthPercentageAuto::length(v),
        Spacing::Pct(f) => LengthPercentageAuto::percent(f),
        Spacing::Auto => LengthPercentageAuto::auto(),
    };
    let padding = |sp: Spacing| match sp {
        Spacing::Px(v) => LengthPercentage::length(v),
        Spacing::Pct(f) => LengthPercentage::percent(f),
        Spacing::Auto => LengthPercentage::length(0.0),
    };
    s.margin.top = margin(style.margin[0]);
    s.margin.right = margin(style.margin[1]);
    s.margin.bottom = margin(style.margin[2]);
    s.margin.left = margin(style.margin[3]);
    s.padding.top = padding(style.padding[0]);
    s.padding.right = padding(style.padding[1]);
    s.padding.bottom = padding(style.padding[2]);
    s.padding.left = padding(style.padding[3]);
    if style.display == DisplayKind::Flex {
        s.flex_direction = FlexDirection::Row;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_html::parse as parse_html;

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
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

    fn example_tree() -> LayoutTree {
        let dom = parse_html(EXAMPLE_COM);
        let style = find_element(&dom, dom.root().unwrap(), "style").unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        let sheet = Stylesheet::parse(&css);
        LayoutTree::build(&dom, &sheet, 1024.0, 768.0).expect("layout builds")
    }

    fn box_for<'a>(tree: &'a LayoutTree, dom: &Dom, tag: &str) -> &'a LayoutBox {
        tree.boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.tag(id) == Some(tag)))
            .unwrap_or_else(|| panic!("no box for <{tag}>"))
    }

    #[test]
    fn example_com_layout() {
        let dom = parse_html(EXAMPLE_COM);
        let tree = example_tree();
        let body = box_for(&tree, &dom, "body");
        // The div's 5em top margin collapses through body/html (CSS margin
        // collapsing), so the body border box itself starts at y=80; the
        // grey canvas background still covers y=0 (see the paint test).
        assert_eq!((body.x, body.y), (0.0, 80.0));
        assert_eq!(body.width, 1024.0);
        assert_eq!(body.style.background, Some([240, 240, 242]));

        let div = box_for(&tree, &dom, "div");
        assert_eq!(div.width, 600.0);
        // width 600 + auto side margins centers the div in 1024.
        assert_eq!(div.x, 212.0);
        // 5em top margin at 16px font.
        assert_eq!(div.y, 80.0);
        assert_eq!(div.style.background, Some([253, 253, 255]));

        let h1 = box_for(&tree, &dom, "h1");
        // Inside the div: 2em (32px) padding, then the h1's own 0.67em
        // top margin (0.67 * 32 = 21.44, rounded by taffy to 21).
        assert_eq!(h1.x, 212.0 + 32.0);
        assert_eq!(h1.y, 80.0 + 32.0 + 21.0);
        assert_eq!(h1.style.font_size, 32.0);
        // The h1's text lives in its child text leaf (anonymous box),
        // not on the element box itself.
        let h1_text = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("Example Domain")))
            })
            .expect("h1 text leaf");
        assert!(!h1_text.glyphs.is_empty(), "h1 text was shaped");

        // Both paragraphs laid out below the heading, inside the div.
        let ps: Vec<&LayoutBox> = tree
            .boxes
            .iter()
            .filter(|b| b.dom.is_some_and(|id| dom.tag(id) == Some("p")))
            .collect();
        assert_eq!(ps.len(), 2);
        assert!(ps[0].y > h1.y + h1.height);
        assert!(ps[1].y > ps[0].y + ps[0].height);
        assert!(ps[0].x >= div.x + 32.0 && ps[0].x + ps[0].width <= div.x + div.width - 32.0);

        // The link text is shaped in its own text leaf. Its color is the
        // UA default link blue: the stylesheet's `a:link` rule is NOT
        // applied in M1 (pseudo-class selectors are M2), so #38488f does
        // not take effect yet.
        let link_leaf = tree
            .boxes
            .iter()
            .find(|b| {
                b.text
                    .as_ref()
                    .is_some_and(|t| t.spans.iter().any(|s| s.text.contains("More information")))
            })
            .expect("link text leaf");
        assert!(
            link_leaf.glyphs.iter().any(|g| g.color == [0, 0, 238]),
            "link glyphs carry the UA default link color"
        );

        assert!(tree.content_height() > 200.0);
        assert!(tree.content_height() < 768.0);
    }

    #[test]
    fn display_none_skips_subtree() {
        let dom = parse_html(
            r#"<div class="hide"><p>one</p></div><div><p>two</p></div>"#,
        );
        let sheet = Stylesheet::parse(".hide { display: none; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        let texts: Vec<String> = tree
            .boxes
            .iter()
            .filter_map(|b| b.text.as_ref())
            .flat_map(|t| t.spans.iter().map(|s| s.text.clone()))
            .collect();
        assert!(!texts.iter().any(|t| t.contains("one")));
        assert!(texts.iter().any(|t| t.contains("two")));
    }

    #[test]
    fn flex_row_lays_children_side_by_side() {
        let dom = parse_html(
            r#"<div class="row"><div class="a">A</div><div class="b">B</div></div>"#,
        );
        let sheet = Stylesheet::parse(
            ".row { display: flex; } .a { width: 100px; } .b { width: 100px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        let a = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("a")))
            .unwrap();
        let b = tree
            .boxes
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.attr(id, "class") == Some("b")))
            .unwrap();
        assert_eq!(a.y, b.y);
        assert_eq!(b.x, a.x + 100.0);
        assert_eq!(a.width, 100.0);
    }

    // ---- two-phase (parallel) style resolution (Stage 4 #5) ----

    /// Compile-time proof that the parallel pass may share `&Dom` /
    /// `&Stylesheet` across rayon threads. `Dom` is a slotmap arena
    /// of plain owned data (`String` / `Vec` / `NodeId`), so it is
    /// `Sync`; if a future change ever makes it non-`Sync` this test
    /// fails to compile and the parallel design must stop there —
    /// never work around it with unsafe code.
    fn assert_sync<T: Sync>() {}

    #[test]
    fn shared_style_inputs_are_sync() {
        assert_sync::<Dom>();
        assert_sync::<Stylesheet>();
        assert_sync::<ComputedStyle>();
    }

    /// The `<style>` text of a document, parsed (same extraction the
    /// layout tests use).
    fn stylesheet_from(dom: &Dom) -> Stylesheet {
        let style = find_element(dom, dom.root().unwrap(), "style").unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        Stylesheet::parse(&css)
    }

    fn assert_same_styles(
        a: &HashMap<DomId, Arc<ComputedStyle>>,
        b: &HashMap<DomId, Arc<ComputedStyle>>,
    ) {
        assert_eq!(
            a.len(),
            b.len(),
            "parallel and serial maps cover different numbers of elements"
        );
        for (id, style) in a {
            let other = b
                .get(id)
                .unwrap_or_else(|| panic!("serial map has no style for {id:?}"));
            assert_eq!(**style, **other, "computed style differs for {id:?}");
        }
    }

    /// First element descendant of `id` (or `id` itself) with this class.
    fn find_by_class(dom: &Dom, id: DomId, class: &str) -> Option<DomId> {
        if dom.attr(id, "class") == Some(class) {
            return Some(id);
        }
        for &child in dom.children(id) {
            if let Some(found) = find_by_class(dom, child, class) {
                return Some(found);
            }
        }
        None
    }

    #[test]
    fn parallel_and_serial_styles_match_example_com() {
        let dom = parse_html(EXAMPLE_COM);
        let sheet = stylesheet_from(&dom);

        // Levels are parent-before-children: every entry's parent
        // appears in an earlier level.
        let html = find_element(&dom, dom.root().unwrap(), "html").unwrap();
        let levels = collect_style_levels(&dom, &[html]);
        let mut seen: Vec<DomId> = Vec::new();
        for level in &levels {
            for &(id, parent) in level {
                if let Some(p) = parent {
                    assert!(seen.contains(&p), "parent of {id:?} not in an earlier level");
                }
            }
            seen.extend(level.iter().map(|&(id, _)| id));
        }

        let parallel = resolve_document_styles(&dom, &sheet, true);
        let serial = resolve_document_styles(&dom, &sheet, false);
        assert_same_styles(&parallel, &serial);
        // html head title meta style body div h1 p p a
        assert_eq!(parallel.len(), 11);
        // Spot-checks through the parallel map: the div keeps its
        // stylesheet width, the h1 its UA 32px font.
        let div = find_element(&dom, dom.root().unwrap(), "div").unwrap();
        assert_eq!(parallel[&div].width, Some(Dim::Px(600.0)));
        let h1 = find_element(&dom, dom.root().unwrap(), "h1").unwrap();
        assert_eq!(parallel[&h1].font_size, 32.0);
    }

    #[test]
    fn parallel_and_serial_styles_match_wide_document() {
        // 200 sibling <p> elements at one level under a wrapper, each
        // with a <span> child: wide enough that rayon's `par_iter`
        // actually splits the level across threads.
        let mut html = String::from("<html><body><div class=\"wrap\">");
        for i in 0..200 {
            html.push_str(&format!(
                "<p class=\"item\">Item {i} <span>inner {i}</span></p>"
            ));
        }
        html.push_str("</div></body></html>");
        let dom = parse_html(&html);
        let sheet = Stylesheet::parse(
            ".wrap { font-size: 20px; color: #112233; } \
             .item { font-size: 150%; margin: 1em; } \
             span { color: #ff0000; }",
        );
        let parallel = resolve_document_styles(&dom, &sheet, true);
        let serial = resolve_document_styles(&dom, &sheet, false);
        assert_same_styles(&parallel, &serial);
        // html + body + div.wrap + 200 p + 200 span
        assert_eq!(parallel.len(), 403);

        // Inheritance through the parallel map: a `.item` inherits
        // the wrap's 20px font, so 150% is 30px and `1em` is 30px.
        let root = dom.root().unwrap();
        let wrap = find_by_class(&dom, root, "wrap").unwrap();
        assert_eq!(parallel[&wrap].font_size, 20.0);
        assert_eq!(parallel[&wrap].color, [0x11, 0x22, 0x33]);
        let item = find_by_class(&dom, root, "item").unwrap();
        assert_eq!(parallel[&item].font_size, 30.0);
        assert_eq!(parallel[&item].color, [0x11, 0x22, 0x33]);
        assert_eq!(parallel[&item].margin, [Spacing::Px(30.0); 4]);
        let span = find_element(&dom, item, "span").unwrap();
        assert_eq!(parallel[&span].font_size, 30.0);
        assert_eq!(parallel[&span].color, [255, 0, 0]);
    }

    #[test]
    fn parallel_and_serial_styles_match_fragment() {
        // No <html>: roots are the document's element children.
        let dom = parse_html(
            r#"<div class="hide"><p>one</p></div><div><p>two</p></div>"#,
        );
        let sheet = Stylesheet::parse(".hide { display: none; }");
        let parallel = resolve_document_styles(&dom, &sheet, true);
        let serial = resolve_document_styles(&dom, &sheet, false);
        assert_same_styles(&parallel, &serial);
        // Both divs and both paragraphs are collected even though
        // the first subtree is `display: none` (computed, unused).
        assert_eq!(parallel.len(), 4);
        let hide = find_by_class(&dom, dom.root().unwrap(), "hide").unwrap();
        assert_eq!(parallel[&hide].display, DisplayKind::None);
    }
}
