//! Rendering-step observer helpers (Stage 8, Recommendation 8).
//!
//! The browser-side half of `ResizeObserver`: measure observed
//! targets on the layout actor's retained tree. The registry and
//! the gather/broadcast loop live in `pluto-js` (`observers`);
//! this module only answers "what size is this target, and how
//! deep is it" from the [`LayoutTree`], using the existing public
//! queries (`boxes_for_dom`, the box's computed style). No new
//! layout accessor was needed.
//!
//! Sizes come from the first box of the target (`boxes_for_dom`
//! returns element boxes in document order). The border box is
//! the box's `width` / `height`; the content box subtracts the
//! resolved padding. Pluto models no border widths, so the border
//! box and the padding box are the same rect. Percentage padding
//! resolves against the containing-block width, which the tree
//! does not retain per box — it is approximated by the box's own
//! border-box width and documented as such (the approximation is
//! exact for the common case of a full-width block whose
//! containing block is the viewport or a same-width parent).

use pluto_html::{Dom, NodeId};
use pluto_js::{ComputedIntersection, ObservedSizes, RootMargin};
use pluto_layout::{LayoutTree, OverflowKind, Spacing};

/// Resolve one padding edge to px. `containing_width` is the
/// approximation for `%` padding (see the module docs).
fn resolve_padding(edge: Spacing, containing_width: f32, viewport: (f32, f32)) -> f32 {
    match edge {
        Spacing::Px(v) => v,
        Spacing::Pct(f) => f * containing_width,
        Spacing::Vh(f) => f * viewport.1,
        Spacing::Vw(f) => f * viewport.0,
        Spacing::Auto => 0.0,
    }
}

/// The observed sizes of one target from the retained layout
/// tree, or `None` when the target has no box (unrendered —
/// `display: none`, or not in the tree — which the registry
/// treats as "never active", so no entry fires).
pub fn box_sizes(tree: &LayoutTree, id: NodeId) -> Option<ObservedSizes> {
    let boxes = tree.boxes_for_dom(id);
    let b = boxes.first()?;
    let border = (b.width as f64, b.height as f64);
    // Padding order in `ComputedStyle::padding` is
    // [top, right, bottom, left].
    let pad_top = resolve_padding(b.style.padding[0], b.width, tree.viewport);
    let pad_right = resolve_padding(b.style.padding[1], b.width, tree.viewport);
    let pad_bottom = resolve_padding(b.style.padding[2], b.width, tree.viewport);
    let pad_left = resolve_padding(b.style.padding[3], b.width, tree.viewport);
    let content = (
        (b.width as f64 - pad_left as f64 - pad_right as f64).max(0.0),
        (b.height as f64 - pad_top as f64 - pad_bottom as f64).max(0.0),
    );
    Some(ObservedSizes {
        border,
        content,
        // No device-pixel-ratio plumbing in Pluto yet: device px
        // equal CSS px.
        device_pixel: content,
        content_offset: (pad_left as f64, pad_top as f64),
    })
}

/// DOM depth of `id` (document root = 0), by walking the parent
/// chain. Stale ids count as depth 0.
pub fn dom_depth(dom: &Dom, id: NodeId) -> usize {
    let mut depth = 0;
    let mut cur = dom.get(id).and_then(|n| n.parent);
    while let Some(parent) = cur {
        depth += 1;
        cur = dom.get(parent).and_then(|n| n.parent);
    }
    depth
}

/// Sizes + depths for a batch of observed targets, in the same
/// order as `ids`. Used by the layout actor's `ResizeSizes` job.
pub fn resize_sizes_for(
    tree: &LayoutTree,
    dom: &Dom,
    ids: &[NodeId],
) -> Vec<(NodeId, Option<ObservedSizes>, usize)> {
    ids.iter()
        .map(|&id| (id, box_sizes(tree, id), dom_depth(dom, id)))
        .collect()
}

// ------------------------------------------------------------------ IntersectionObserver

/// One intersection query for the layout actor: the target, the
/// root (`None` = the viewport), and the root's `rootMargin`.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionQuery {
    pub target: NodeId,
    pub root: Option<NodeId>,
    pub root_margin: RootMargin,
}

/// Intersect two `[x, y, width, height]` rects. `None` when they do
/// not overlap (or either is degenerate).
fn intersect_rects(a: [f64; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = (a[0] + a[2]).min(b[0] + b[2]);
    let y1 = (a[1] + a[3]).min(b[1] + b[3]);
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some([x0, y0, x1 - x0, y1 - y0])
    }
}

/// The intersection geometry for one observed target, from the
/// retained layout tree, or `None` when the target (or an element
/// root) has no box — unrendered, so the registry reports no entry
/// and the initial entry still waits.
///
/// The walk (Stage 8 Rec 10, report §(e)): the target's border box
/// is clipped by the (margin-expanded) root rect and by every
/// ancestor between target and root whose `overflow` is not
/// `visible` — each such ancestor clips its contents to its own
/// box rect (Pluto models no border widths, so the border box and
/// the padding box are the same rect; see the module docs). The
/// clip is per-axis: an ancestor with `overflow-x: visible` clips
/// only vertically. A root that is not a strict ancestor of the
/// target never intersects (its initial entry still fires, with
/// ratio 0) — including the degenerate case `root == target`.
///
/// Positions already account for scroll offsets: the tree applies
/// the retained container offsets when it collects boxes, so the
/// target's box is where it is on the scrolled page, and the
/// viewport root is `[viewport_scroll, viewport size]` in page
/// coordinates.
pub fn intersection_geometry_for(
    tree: &LayoutTree,
    dom: &Dom,
    queries: &[IntersectionQuery],
    viewport_scroll: (f32, f32),
) -> Vec<Option<ComputedIntersection>> {
    queries
        .iter()
        .map(|query| {
            let target_boxes = tree.boxes_for_dom(query.target);
            let target_box = target_boxes.first()?;
            let target_rect = [
                target_box.x as f64,
                target_box.y as f64,
                target_box.width as f64,
                target_box.height as f64,
            ];
            // The root rect, and whether it is a strict ancestor of
            // the target.
            let (root_rect, is_ancestor) = match query.root {
                None => (
                    [
                        viewport_scroll.0 as f64,
                        viewport_scroll.1 as f64,
                        tree.viewport.0 as f64,
                        tree.viewport.1 as f64,
                    ],
                    true,
                ),
                Some(root_id) => {
                    let root_boxes = tree.boxes_for_dom(root_id);
                    let root_box = root_boxes.first()?;
                    let rect = [
                        root_box.x as f64,
                        root_box.y as f64,
                        root_box.width as f64,
                        root_box.height as f64,
                    ];
                    (rect, is_strict_ancestor(dom, query.target, root_id))
                }
            };
            let root_bounds = query.root_margin.expand(root_rect);
            // Clip chain: walk the ancestors from the target's
            // parent up to (excluding) the root, intersecting the
            // target rect with each clipping ancestor's box.
            let mut clipped = target_rect;
            let mut cur = dom.get(query.target).and_then(|n| n.parent);
            while let Some(ancestor) = cur {
                if Some(ancestor) == query.root {
                    break;
                }
                // html/body overflow propagates to the viewport (the
                // same rule as `scroll_chain_for`): they never clip
                // their contents in this walk.
                if dom.tag(ancestor) == Some("html") || dom.tag(ancestor) == Some("body") {
                    cur = dom.get(ancestor).and_then(|n| n.parent);
                    continue;
                }
                if let Some(abox) = tree.boxes_for_dom(ancestor).first() {
                    if abox.style.overflow_x != OverflowKind::Visible {
                        clipped = match intersect_rects(clipped, [
                            abox.x as f64,
                            clipped[1],
                            abox.width as f64,
                            clipped[3],
                        ]) {
                            Some(r) => r,
                            None => {
                                clipped = [0.0, 0.0, 0.0, 0.0];
                                break;
                            }
                        };
                    }
                    if abox.style.overflow_y != OverflowKind::Visible {
                        clipped = match intersect_rects(clipped, [
                            clipped[0],
                            abox.y as f64,
                            clipped[2],
                            abox.height as f64,
                        ]) {
                            Some(r) => r,
                            None => {
                                clipped = [0.0, 0.0, 0.0, 0.0];
                                break;
                            }
                        };
                    }
                }
                cur = dom.get(ancestor).and_then(|n| n.parent);
            }
            // The intersection with the root bounds.
            let intersection_rect = if is_ancestor {
                intersect_rects(clipped, root_bounds).unwrap_or([0.0, 0.0, 0.0, 0.0])
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            let target_area = target_rect[2] * target_rect[3];
            let ratio = if target_area > 0.0 {
                (intersection_rect[2] * intersection_rect[3] / target_area).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some(ComputedIntersection {
                target_rect,
                root_bounds,
                intersection_rect,
                ratio,
                is_intersecting: ratio > 0.0,
            })
        })
        .collect()
}

/// True when `ancestor` is a strict ancestor of `id` in `dom`.
fn is_strict_ancestor(dom: &Dom, id: NodeId, ancestor: NodeId) -> bool {
    let mut cur = dom.get(id).and_then(|n| n.parent);
    while let Some(parent) = cur {
        if parent == ancestor {
            return true;
        }
        cur = dom.get(parent).and_then(|n| n.parent);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_js::{ResizeBoxKind, ResizeObserverRegistry, RootMargin};
    use pluto_layout::{LayoutTree, Theme};

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

    fn find_id(dom: &Dom, id: NodeId, want: &str) -> Option<NodeId> {
        for &child in dom.children(id) {
            if dom.attr(child, "id") == Some(want) {
                return Some(child);
            }
            if let Some(found) = find_id(dom, child, want) {
                return Some(found);
            }
        }
        None
    }

    fn tree_for(html: &str) -> (pluto_html::Dom, LayoutTree) {
        let doc = crate::page::parse_document(html);
        let tree =
            LayoutTree::build(doc.dom(), doc.sheet(), 1024.0, 768.0, Theme::Light)
                .expect("layout builds");
        (doc.dom().clone(), tree)
    }

    const PAGE: &str = r#"<!doctype html><html><head><style>
      #box { width: 100px; height: 50px; padding: 10px 20px; background: red; }
      #tall { width: 200px; height: 80px; }
    </style></head><body><div id="box"></div><div id="tall"></div></body></html>"#;

    #[test]
    fn box_sizes_split_border_and_content_boxes() {
        let (dom, tree) = tree_for(PAGE);
        let root = dom.root().unwrap();
        let box_id = find_tag(&dom, root, "div").expect("first div");
        assert_eq!(dom.attr(box_id, "id"), Some("box"));
        let sizes = box_sizes(&tree, box_id).expect("box has sizes");
        // Pluto lays out content-box (Stage 6): `width` is the
        // content width, so the border box is content + padding
        // (10px vertical, 20px horizontal each side).
        assert_eq!(sizes.border, (140.0, 70.0));
        assert_eq!(sizes.content, (100.0, 50.0));
        assert_eq!(sizes.content_offset, (20.0, 10.0));
        // Depth: document(0) -> html(1) -> body(2) -> div(3).
        assert_eq!(dom_depth(&dom, box_id), 3);
    }

    #[test]
    fn gather_detects_a_real_layout_size_change() {
        // Stage 8 Rec 8, browser half: a ResizeObserver watching
        // a div fires its initial entry on the first layout, stays
        // quiet on an identical re-layout, and fires again when
        // the div's laid-out width changes.
        let (dom, tree) = tree_for(PAGE);
        let root = dom.root().unwrap();
        let box_id = find_tag(&dom, root, "div").expect("box div");
        let mut reg = ResizeObserverRegistry::default();
        reg.observe(1, box_id, ResizeBoxKind::BorderBox);

        let mut gather = |tree: &LayoutTree, dom: &Dom| {
            let rows = resize_sizes_for(tree, dom, &[box_id]);
            let (size, depth) = (rows[0].1, rows[0].2);
            reg.gather_round(&|k| if k == box_id { size } else { None }, &|_| depth)
        };
        let first = gather(&tree, &dom);
        assert_eq!(first.len(), 1, "initial entry fires");
        assert_eq!(first[0].1[0].sizes.border, (140.0, 70.0));
        assert!(gather(&tree, &dom).is_empty(), "identical layout: quiet");

        // Same page, wider div: a fresh layout, a new entry.
        let wider = PAGE.replace("width: 100px", "width: 200px");
        let (dom2, tree2) = tree_for(&wider);
        // NOTE: `box_id` is a key into the first Dom's arena; the
        // rebuilt document has its own arena, so look the div up
        // again there instead of reusing the stale key.
        let root2 = dom2.root().unwrap();
        let box2 = find_tag(&dom2, root2, "div").expect("box div in rebuilt doc");
        // The registry still watches the original key, so feed the
        // rebuilt sizes under it: the size source is what changed.
        let size2 = box_sizes(&tree2, box2).expect("rebuilt box has sizes");
        let batches = reg.gather_round(&|k| if k == box_id { Some(size2) } else { None }, &|_| {
            dom_depth(&dom2, box2)
        });
        assert_eq!(batches.len(), 1, "wider layout fires again");
        assert_eq!(batches[0].1[0].sizes.border, (240.0, 70.0));
        assert!(
            reg.gather_round(&|k| if k == box_id { Some(size2) } else { None }, &|_| 3)
                .is_empty()
        );
    }

    #[test]
    fn unrendered_target_has_no_sizes() {
        let (dom, tree) = tree_for(PAGE);
        let root = dom.root().unwrap();
        // A node that exists in the Dom but has no box: the text
        // node inside <body> has no element box of its own here…
        // use a detached element instead (never laid out).
        let mut dom = dom;
        let detached = dom.create_element("section");
        assert!(box_sizes(&tree, detached).is_none());
        let _ = root;
    }

    // Stage 8 Rec 10: a page whose targets sit at known page
    // coordinates (all heights explicit, no margins — nothing
    // collapses). #root (0..200, overflow hidden) holds #inner
    // (150..250, clipped to 150..200); #top 200..300; #half
    // 718..818 (half in the 768 viewport); #low 1818..1918.
    const IO_PAGE: &str = r#"<!doctype html><html><head><style>
      body{margin:0}
      #root{width:200px;height:200px;overflow:hidden}
      #inner{width:100px;height:100px}
      #top{width:100px;height:100px}
      #half{width:100px;height:100px}
      #low{width:100px;height:100px}
      #hidden{display:none}
    </style></head><body>
      <div id="root"><div style="height:150px"></div><div id="inner"></div></div>
      <div id="top"></div>
      <div style="height:418px"></div><div id="half"></div>
      <div style="height:1000px"></div><div id="low"></div>
      <div id="hidden"></div>
    </body></html>"#;

    fn query(target: NodeId, root: Option<NodeId>, margin: RootMargin) -> IntersectionQuery {
        IntersectionQuery {
            target,
            root,
            root_margin: margin,
        }
    }

    #[test]
    fn intersection_viewport_root_in_partial_and_out() {
        let (dom, tree) = tree_for(IO_PAGE);
        let root = dom.root().unwrap();
        let top = find_id(&dom, root, "top").expect("top");
        let half = find_id(&dom, root, "half").expect("half");
        let low = find_id(&dom, root, "low").expect("low");
        let rows = intersection_geometry_for(
            &tree,
            &dom,
            &[
                query(top, None, RootMargin::default()),
                query(half, None, RootMargin::default()),
                query(low, None, RootMargin::default()),
            ],
            (0.0, 0.0),
        );
        let top_geo = rows[0].as_ref().expect("top renders");
        assert_eq!(top_geo.ratio, 1.0);
        assert!(top_geo.is_intersecting);
        assert_eq!(top_geo.target_rect, [0.0, 200.0, 100.0, 100.0]);
        assert_eq!(top_geo.root_bounds, [0.0, 0.0, 1024.0, 768.0]);
        let half_geo = rows[1].as_ref().expect("half renders");
        assert!((half_geo.ratio - 0.5).abs() < 1e-9, "ratio is {}", half_geo.ratio);
        assert_eq!(half_geo.intersection_rect, [0.0, 718.0, 100.0, 50.0]);
        assert!(half_geo.is_intersecting);
        let low_geo = rows[2].as_ref().expect("low renders");
        assert_eq!(low_geo.ratio, 0.0);
        assert!(!low_geo.is_intersecting);
        assert_eq!(low_geo.intersection_rect, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn intersection_inner_scroller_clips_the_target() {
        let (dom, tree) = tree_for(IO_PAGE);
        let root = dom.root().unwrap();
        let inner = find_id(&dom, root, "inner").expect("inner");
        let rows = intersection_geometry_for(
            &tree,
            &dom,
            &[query(inner, None, RootMargin::default())],
            (0.0, 0.0),
        );
        let geo = rows[0].as_ref().expect("inner renders");
        // Inner is 150..250 in the page; the viewport alone would
        // show it fully, but #root (0..200, overflow:hidden) clips
        // it to 50px.
        assert!((geo.ratio - 0.5).abs() < 1e-9, "ratio is {}", geo.ratio);
        assert_eq!(geo.intersection_rect, [0.0, 150.0, 100.0, 50.0]);
    }

    #[test]
    fn intersection_element_root_non_ancestor_root_margin_and_unrendered() {
        let (dom, tree) = tree_for(IO_PAGE);
        let root = dom.root().unwrap();
        let inner = find_id(&dom, root, "inner").expect("inner");
        let root_el = find_id(&dom, root, "root").expect("root");
        let low = find_id(&dom, root, "low").expect("low");
        let hidden = find_id(&dom, root, "hidden").expect("hidden");
        let rows = intersection_geometry_for(
            &tree,
            &dom,
            &[
                // Element root: inner against #root -> 0.5.
                query(inner, Some(root_el), RootMargin::default()),
                // Non-ancestor root: #low is not an ancestor of
                // #inner -> never intersects (but still gets an
                // entry — the initial one — so Some, ratio 0).
                query(inner, Some(low), RootMargin::default()),
                // rootMargin 1200px pulls #low into the viewport root.
                query(
                    low,
                    None,
                    RootMargin::parse("1200px").expect("margin parses"),
                ),
                query(low, None, RootMargin::default()),
                query(hidden, None, RootMargin::default()),
            ],
            (0.0, 0.0),
        );
        assert!((rows[0].as_ref().unwrap().ratio - 0.5).abs() < 1e-9);
        assert_eq!(rows[0].as_ref().unwrap().root_bounds, [0.0, 0.0, 200.0, 200.0]);
        let non_ancestor = rows[1].as_ref().expect("non-ancestor still gets an entry");
        assert_eq!(non_ancestor.ratio, 0.0);
        assert!(!non_ancestor.is_intersecting);
        assert_eq!(rows[2].as_ref().unwrap().ratio, 1.0);
        assert_eq!(
            rows[2].as_ref().unwrap().root_bounds,
            [-1200.0, -1200.0, 3424.0, 3168.0]
        );
        assert_eq!(rows[3].as_ref().unwrap().ratio, 0.0);
        assert!(rows[4].is_none(), "display:none target has no geometry");
    }

    #[test]
    fn intersection_viewport_scroll_moves_the_root() {
        let (dom, tree) = tree_for(IO_PAGE);
        let root = dom.root().unwrap();
        let half = find_id(&dom, root, "half").expect("half");
        let low = find_id(&dom, root, "low").expect("low");
        // Scrolled 1100px down: the root is 1100..1868, so #low
        // (1818..1918) is half visible and #half (718..818) is out.
        let rows = intersection_geometry_for(
            &tree,
            &dom,
            &[
                query(low, None, RootMargin::default()),
                query(half, None, RootMargin::default()),
            ],
            (0.0, 1100.0),
        );
        assert!((rows[0].as_ref().unwrap().ratio - 0.5).abs() < 1e-9);
        assert_eq!(rows[0].as_ref().unwrap().root_bounds, [0.0, 1100.0, 1024.0, 768.0]);
        assert_eq!(rows[1].as_ref().unwrap().ratio, 0.0);
    }
}
