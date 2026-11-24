//! GPU rendering (layer 5): display list + wgpu 2D/text renderer.
//!
//! Walks the laid-out boxes in document order and emits solid quads
//! (backgrounds) and glyph runs into a [`DisplayList`]. Glyphs carry the
//! cosmic-text cache key from layout, so the renderer ([`renderer`]) only
//! needs its own `FontSystem` + `SwashCache` to rasterize them into the
//! atlas — no font state crosses the layout/paint boundary.
//!
//! M1 scope: opaque backgrounds and mask glyphs. No borders, images,
//! or color glyphs (emoji) yet.

use pluto_layout::LayoutTree;

pub mod renderer;

/// One paint operation, in paint order.
#[derive(Debug, Clone)]
pub enum DisplayItem {
    /// A solid rectangle: page/element backgrounds.
    FillRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
    },
    /// Glyphs of one color, positioned in page coordinates.
    Glyphs {
        color: [f32; 4],
        glyphs: Vec<PaintGlyph>,
    },
}

/// One glyph: cosmic-text cache key + page position of its origin.
#[derive(Debug, Clone, Copy)]
pub struct PaintGlyph {
    pub cache_key: cosmic_text::CacheKey,
    pub x: f32,
    pub y: f32,
}

/// The ordered paint operations for one frame.
#[derive(Debug, Clone, Default)]
pub struct DisplayList(pub Vec<DisplayItem>);

impl DisplayList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, item: DisplayItem) {
        self.0.push(item);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Build the paint operations for one laid-out document. The canvas
    /// background (html/body background, or white by default) paints the
    /// whole viewport first; then each box's background, then its glyphs.
    pub fn build(layout: &LayoutTree) -> Self {
        let mut list = Self::new();
        let bg = layout.canvas_background;
        list.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: layout.viewport.0,
            height: layout.viewport.1,
            color: rgba(bg),
        });
        for bx in layout.boxes() {
            if let Some(bg) = bx.style.background {
                list.push(DisplayItem::FillRect {
                    x: bx.x,
                    y: bx.y,
                    width: bx.width,
                    height: bx.height,
                    color: rgba(bg),
                });
            }
            if !bx.glyphs.is_empty() {
                // Group consecutive glyphs of the same color into one item.
                let mut current_color = bx.glyphs[0].color;
                let mut current: Vec<PaintGlyph> = Vec::new();
                for g in &bx.glyphs {
                    if g.color != current_color {
                        list.push(DisplayItem::Glyphs {
                            color: rgba(current_color),
                            glyphs: std::mem::take(&mut current),
                        });
                        current_color = g.color;
                    }
                    current.push(PaintGlyph {
                        cache_key: g.cache_key,
                        x: g.x,
                        y: g.y,
                    });
                }
                if !current.is_empty() {
                    list.push(DisplayItem::Glyphs {
                        color: rgba(current_color),
                        glyphs: current,
                    });
                }
            }
        }
        list
    }
}

fn rgba(c: [u8; 3]) -> [f32; 4] {
    [
        c[0] as f32 / 255.0,
        c[1] as f32 / 255.0,
        c[2] as f32 / 255.0,
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::Stylesheet;
    use pluto_html::parse as parse_html;
    use pluto_layout::LayoutTree;

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

    fn example_list() -> DisplayList {
        let dom = parse_html(EXAMPLE_COM);
        let style = dom
            .children(dom.root().unwrap())
            .iter()
            .find_map(|&c| {
                if dom.tag(c) == Some("html") {
                    dom.children(c)
                        .iter()
                        .find(|&&n| dom.tag(n) == Some("head"))
                        .and_then(|&head| {
                            dom.children(head)
                                .iter()
                                .find(|&&n| dom.tag(n) == Some("style"))
                                .copied()
                        })
                } else {
                    None
                }
            })
            .unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        let sheet = Stylesheet::parse(&css);
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0).expect("layout builds");
        DisplayList::build(&tree)
    }

    #[test]
    fn example_com_display_list() {
        let list = example_list();
        assert!(!list.is_empty());
        // First item: the canvas background (body #f0f0f2) over the viewport.
        match &list.0[0] {
            DisplayItem::FillRect {
                x,
                y,
                width,
                height,
                color,
            } => {
                assert_eq!((*x, *y), (0.0, 0.0));
                assert_eq!((*width, *height), (1024.0, 768.0));
                assert!((color[0] - 240.0 / 255.0).abs() < 0.001);
            }
            other => panic!("first item should be the canvas fill, got {other:?}"),
        }
        // The div's own background (#fdfdff) is painted as its own rect.
        assert!(list.0.iter().any(|item| matches!(
            item,
            DisplayItem::FillRect { width, color, .. }
            if *width == 600.0 && (color[0] - 253.0 / 255.0).abs() < 0.001
        )));
        // Text is present as glyph runs, including the link (UA default
        // blue: the stylesheet's `a:link` rule is M2, see the layout test).
        let glyph_items: Vec<&DisplayItem> = list
            .0
            .iter()
            .filter(|i| matches!(i, DisplayItem::Glyphs { .. }))
            .collect();
        assert!(!glyph_items.is_empty());
        assert!(glyph_items.iter().any(|i| match i {
            DisplayItem::Glyphs { color, glyphs } => {
                !glyphs.is_empty() && (color[2] - 238.0 / 255.0).abs() < 0.01
            }
            _ => false,
        }));
    }

    #[test]
    fn empty_document_is_white_canvas() {
        // An empty page still paints its canvas: a browser's default
        // canvas is white, not transparent.
        let dom = parse_html("<html><head></head><body></body></html>");
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        let list = DisplayList::build(&tree);
        assert_eq!(list.0.len(), 1);
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 800.0,
                height: 600.0,
                color: [1.0, 1.0, 1.0, 1.0],
                ..
            }
        ));
    }

    /// The live example.com page as served 2026-11-23 (577 bytes):
    /// unquoted attributes, a `<script src=/s.js>`, and a stylesheet
    /// whose `background: light-dark(...)` is beyond M1. Regression:
    /// the script tag used to parse-fail and render as literal text,
    /// and with no canvas fill the black paragraph text was invisible
    /// on the dark app background.
    const NEW_EXAMPLE_COM: &str = "<!doctype html><html lang=en><head><meta charset=utf-8><link rel=icon href=data:,><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>Example Domain</title><style>html{color-scheme:light dark;background:light-dark(#eee,#222)}body{font:16px/1.6 system-ui,sans-serif;max-width:26em;margin:auto;padding:25vh 2em 2em;text-align:center}</style></head><body><p>This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.</p><script src=/s.js></script></body></html>";

    #[test]
    fn new_example_com_renders() {
        let dom = parse_html(NEW_EXAMPLE_COM);
        let style = dom
            .children(dom.root().unwrap())
            .iter()
            .find_map(|&c| {
                if dom.tag(c) == Some("html") {
                    dom.children(c)
                        .iter()
                        .find(|&&n| dom.tag(n) == Some("head"))
                        .and_then(|&head| {
                            dom.children(head)
                                .iter()
                                .find(|&&n| dom.tag(n) == Some("style"))
                                .copied()
                        })
                } else {
                    None
                }
            })
            .unwrap();
        let css: String = dom
            .children(style)
            .iter()
            .filter_map(|&c| dom.text(c))
            .collect();
        let sheet = Stylesheet::parse(&css);
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0).expect("layout builds");
        let list = DisplayList::build(&tree);
        // White canvas first, then the paragraph glyphs. The script
        // tag contributes nothing: it is an element, not text.
        assert_eq!(list.0.len(), 2);
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                height: 768.0,
                color: [1.0, 1.0, 1.0, 1.0],
                ..
            }
        ));
        let glyph_total: usize = list.0.iter().fold(0, |acc, item| match item {
            DisplayItem::Glyphs { glyphs, .. } => acc + glyphs.len(),
            _ => acc,
        });
        // 155 paragraph glyphs (non-space chars). If the script tag
        // leaked into text there would be 18 more.
        assert_eq!(glyph_total, 155);
    }
}
