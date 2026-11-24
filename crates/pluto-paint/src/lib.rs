//! GPU rendering (layer 5): display list + wgpu 2D/text renderer.
//!
//! Walks the laid-out boxes in document order and emits solid quads
//! (backgrounds), glyph runs, and image/blob items into a
//! [`DisplayList`]. Glyphs carry a serialisable [`GlyphKey`] (the
//! cosmic-text cache key in plain numbers), so the renderer
//! ([`renderer`]) only needs its own `FontSystem` + `SwashCache` to
//! rasterize them into the atlas — no font state crosses the
//! layout/paint boundary.
//!
//! The display list is a closed vocabulary of owned data (Stage 4
//! #4): no borrows into the DOM/layout arena, no `Rc`, no wgpu
//! handles. It serialises with postcard, so the same bytes that cross
//! the in-process channel today cross IPC at M5. Images and
//! unrepresentable content travel by [`ResourceId`] into a
//! [`ResourceTable`] (see [`image::ImageCache`]); the list is
//! validated against the table before it reaches the renderer
//! ([`DisplayList::validate`]).
//!
//! M1 scope: opaque backgrounds and mask glyphs. Images/blobs are
//! vocabulary + renderer support for M3; no borders or color glyphs
//! (emoji) yet.

use pluto_layout::LayoutTree;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod glyph_key;
pub mod image;
pub mod renderer;

pub use glyph_key::GlyphKey;
pub use image::{DecodedImage, ImageCache, ImageError, decode_image, decode_images_parallel};

/// One paint operation, in paint order. A closed vocabulary: every
/// variant is plain owned data, so a list is serialisable and a
/// renderer in another process can draw it without pointers back
/// into the layout arena.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// A raster image by id: `<img>` / `background-image` (M3).
    /// The pixels live in the [`ResourceTable`]; the item only names
    /// them, so the same list bytes work in any process that holds
    /// the table.
    Image {
        resource: ResourceId,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    /// Blob fallback (WebRender's term): anything the vocabulary
    /// cannot express — a canvas, an SVG, a video frame — rendered
    /// as a raster image by id. The escape hatch is designed in from
    /// the start, not bolted on: producers emit a `Blob` instead of
    /// failing, and the renderer draws it exactly like an `Image`.
    Blob {
        resource: ResourceId,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    /// Push a clip rect for a scroll container: every following item
    /// paints clipped to this rect until the matching `PopClip`.
    /// (R14, M2: the vocabulary gains the clip; the shader-side
    /// scissor itself is a Stage 8 follow-up — the renderer currently
    /// ignores these items, so they are data, not a visual change.)
    PushClip {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    /// Pop the most recent `PushClip`.
    PopClip,
}

/// One glyph: serialisable cache key + page position of its origin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PaintGlyph {
    pub key: GlyphKey,
    pub x: f32,
    pub y: f32,
}

/// Handle into a [`ResourceTable`]. Id 0 is reserved as "no resource"
/// and is never handed out by [`ImageCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceId(pub u32);

/// One browser-owned resource: decoded pixels, stored once and
/// referenced by id from display items. (More kinds — fonts, shared
/// textures — arrive as variants here, not as new item types.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resource {
    /// Decoded RGBA8 pixels, row-major, `width * height * 4` bytes.
    Image {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
}

/// The resource table that travels with a display list: every
/// [`DisplayItem::Image`]/[`DisplayItem::Blob`] id must resolve here.
/// Owned by the browser (see [`ImageCache`]); at M5 it crosses IPC
/// beside the list bytes, fed once over shared memory, not per frame.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTable {
    pub resources: HashMap<ResourceId, Resource>,
}

impl ResourceTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: ResourceId, resource: Resource) {
        self.resources.insert(id, resource);
    }

    pub fn get(&self, id: ResourceId) -> Option<&Resource> {
        self.resources.get(&id)
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Bounds-check the table itself: every image's byte length must
    /// match its dimensions, and no single resource may exceed
    /// [`MAX_RESOURCE_BYTES`].
    pub fn validate(&self) -> Result<(), ValidationError> {
        for (id, resource) in &self.resources {
            match resource {
                Resource::Image {
                    width,
                    height,
                    rgba,
                } => {
                    if *width == 0 || *height == 0 {
                        return Err(ValidationError::EmptyImage { id: *id });
                    }
                    let expected = (*width as usize)
                        .checked_mul(*height as usize)
                        .and_then(|n| n.checked_mul(4))
                        .ok_or(ValidationError::ImageTooBig { id: *id })?;
                    if rgba.len() != expected {
                        return Err(ValidationError::ImageLengthMismatch {
                            id: *id,
                            expected,
                            actual: rgba.len(),
                        });
                    }
                    if rgba.len() > MAX_RESOURCE_BYTES {
                        return Err(ValidationError::ImageTooBig { id: *id });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Bounds the browser enforces on an untrusted display list (Stage 4
/// #4): a hostile renderer must not be able to make the browser
/// allocate unboundedly or draw outside the viewport. The limits are
/// generous for real pages and tight enough to bound the work.
pub const MAX_DISPLAY_ITEMS: usize = 100_000;
pub const MAX_GLYPHS_TOTAL: usize = 1_000_000;
pub const MAX_COORDINATE: f32 = 100_000.0;
pub const MAX_RESOURCE_BYTES: usize = 64 * 1024 * 1024;

/// What [`DisplayList::validate`] rejects, and why.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidationError {
    #[error("display list has {0} items, over the limit of {MAX_DISPLAY_ITEMS}")]
    TooManyItems(usize),
    #[error("display list has {0} glyphs, over the limit of {MAX_GLYPHS_TOTAL}")]
    TooManyGlyphs(usize),
    #[error("non-finite or out-of-range coordinate {0}")]
    BadCoordinate(f32),
    #[error("negative or out-of-range size {0}")]
    BadSize(f32),
    #[error("color component {0} outside 0..=1")]
    BadColor(f32),
    #[error("image/blob references missing resource {0:?}")]
    MissingResource(ResourceId),
    #[error("image resource {id:?} has zero width or height")]
    EmptyImage { id: ResourceId },
    #[error(
        "image resource {id:?} byte length {actual} does not match dimensions ({expected} expected)"
    )]
    ImageLengthMismatch {
        id: ResourceId,
        expected: usize,
        actual: usize,
    },
    #[error("image resource {id:?} exceeds {MAX_RESOURCE_BYTES} bytes")]
    ImageTooBig { id: ResourceId },
    #[error("display list bytes do not decode: {0}")]
    Decode(String),
}

/// The ordered paint operations for one frame.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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

    /// Serialise the list to bytes (postcard): the payload of a
    /// `SubmitDisplayList` message, in-process today and over IPC at
    /// M5. The resource table travels beside it, not inside it.
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("DisplayList serialises")
    }

    /// Decode bytes produced by [`DisplayList::to_bytes`]. A decode
    /// failure is a validation error, not a panic: bytes from a
    /// renderer are untrusted input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ValidationError> {
        postcard::from_bytes(bytes).map_err(|e| ValidationError::Decode(e.to_string()))
    }

    /// Validate the list against `resources` before it reaches the
    /// renderer: item/glyph counts, finite in-range coordinates and
    /// sizes, colors in 0..=1, and every image/blob id resolving in
    /// the table (whose own bytes are checked by
    /// [`ResourceTable::validate`]). Written against the in-process
    /// path from day one — this is the same code that guards the IPC
    /// path at M5.
    pub fn validate(&self, resources: &ResourceTable) -> Result<(), ValidationError> {
        if self.0.len() > MAX_DISPLAY_ITEMS {
            return Err(ValidationError::TooManyItems(self.0.len()));
        }
        resources.validate()?;
        let mut glyphs = 0usize;
        for item in &self.0 {
            match item {
                DisplayItem::FillRect {
                    x,
                    y,
                    width,
                    height,
                    color,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                    for c in color {
                        if !c.is_finite() || !(0.0..=1.0).contains(c) {
                            return Err(ValidationError::BadColor(*c));
                        }
                    }
                }
                DisplayItem::Glyphs { color, glyphs: gs } => {
                    for c in color {
                        if !c.is_finite() || !(0.0..=1.0).contains(c) {
                            return Err(ValidationError::BadColor(*c));
                        }
                    }
                    glyphs += gs.len();
                    if glyphs > MAX_GLYPHS_TOTAL {
                        return Err(ValidationError::TooManyGlyphs(glyphs));
                    }
                    for g in gs {
                        check_coord(g.x)?;
                        check_coord(g.y)?;
                    }
                }
                DisplayItem::Image {
                    resource,
                    x,
                    y,
                    width,
                    height,
                }
                | DisplayItem::Blob {
                    resource,
                    x,
                    y,
                    width,
                    height,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                    if resources.get(*resource).is_none() {
                        return Err(ValidationError::MissingResource(*resource));
                    }
                }
                DisplayItem::PushClip {
                    x,
                    y,
                    width,
                    height,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                }
                DisplayItem::PopClip => {}
            }
        }
        Ok(())
    }

    /// Build the paint operations for one laid-out document. The canvas
    /// background (html/body background, or white by default) paints the
    /// whole viewport first — it is always item 0, which the renderer
    /// relies on for the scroll-anchored canvas — then each box's
    /// background, then its glyphs. Boxes inside a scroll container
    /// carry a clip rect (`LayoutBox::clip`); when it changes, the
    /// previous clip is popped and the new one pushed, so a container's
    /// subtree paints clipped to the container's border box.
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
        let mut current_clip: Option<[f32; 4]> = None;
        for bx in layout.boxes() {
            if bx.clip != current_clip {
                if current_clip.is_some() {
                    list.push(DisplayItem::PopClip);
                }
                if let Some(c) = bx.clip {
                    list.push(DisplayItem::PushClip {
                        x: c[0],
                        y: c[1],
                        width: c[2],
                        height: c[3],
                    });
                }
                current_clip = bx.clip;
            }
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
                        // The layout glyph's cache key becomes the
                        // serialisable wire key here, at the paint
                        // boundary — layout never sees the wire form.
                        key: GlyphKey::from(g.cache_key),
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
        if current_clip.is_some() {
            list.push(DisplayItem::PopClip);
        }
        list
    }
}

fn check_coord(v: f32) -> Result<(), ValidationError> {
    if !v.is_finite() || v.abs() > MAX_COORDINATE {
        return Err(ValidationError::BadCoordinate(v));
    }
    Ok(())
}

fn check_size(v: f32) -> Result<(), ValidationError> {
    if !v.is_finite() || v < 0.0 || v > MAX_COORDINATE {
        return Err(ValidationError::BadSize(v));
    }
    Ok(())
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
        // Content-box sizing (M2): width 600 + 2×32 padding = 664.
        assert!(list.0.iter().any(|item| matches!(
            item,
            DisplayItem::FillRect { width, color, .. }
            if *width == 664.0 && (color[0] - 253.0 / 255.0).abs() < 0.001
        )));
        // Text is present as glyph runs, including the link — in the
        // author `a:link` color #38488f now (blue 143/255), not the
        // UA default blue (238/255) the old expectation assumed.
        let glyph_items: Vec<&DisplayItem> = list
            .0
            .iter()
            .filter(|i| matches!(i, DisplayItem::Glyphs { .. }))
            .collect();
        assert!(!glyph_items.is_empty());
        assert!(glyph_items.iter().any(|i| match i {
            DisplayItem::Glyphs { color, glyphs } => {
                !glyphs.is_empty() && (color[2] - 143.0 / 255.0).abs() < 0.01
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
        // 154 paragraph glyphs at this viewport (132 non-space chars,
        // constant across wraps, + 22 surviving spaces): `max-width:
        // 26em` is honoured now, so the paragraph wraps at 416px and
        // two spaces are consumed at line breaks (the old 155 predates
        // max-width). A range, not an exact wrap-dependent count: any
        // other wrap still lands near 132 non-space glyphs, while a
        // script tag leaking into text would add 18 more (out of range).
        assert!(
            (150..=160).contains(&glyph_total),
            "glyph total {glyph_total} outside the expected wrap range"
        );
    }

    fn small_list() -> DisplayList {
        let mut list = DisplayList::new();
        list.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            color: [1.0, 1.0, 1.0, 1.0],
        });
        list
    }

    #[test]
    fn display_list_round_trips_through_bytes() {
        let list = example_list();
        let bytes = list.to_bytes();
        let back = DisplayList::from_bytes(&bytes).expect("list decodes");
        assert_eq!(back, list, "postcard round-trip must be lossless");
        assert!(DisplayList::from_bytes(b"\xff\xff\xff").is_err());
        assert!(DisplayList::from_bytes(&[]).is_err());
    }

    #[test]
    fn validation_accepts_a_real_list_and_an_empty_one() {
        let resources = ResourceTable::new();
        example_list()
            .validate(&resources)
            .expect("example.com list is valid");
        DisplayList::new()
            .validate(&resources)
            .expect("an empty list is valid");
    }

    #[test]
    fn validation_rejects_bad_geometry_and_colors() {
        let resources = ResourceTable::new();
        let mut nan_rect = small_list();
        nan_rect.0[0] = DisplayItem::FillRect {
            x: f32::NAN,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            color: [1.0; 4],
        };
        assert!(matches!(
            nan_rect.validate(&resources),
            Err(ValidationError::BadCoordinate(_))
        ));
        let mut neg_size = small_list();
        neg_size.0[0] = DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: -5.0,
            height: 10.0,
            color: [1.0; 4],
        };
        assert!(matches!(
            neg_size.validate(&resources),
            Err(ValidationError::BadSize(_))
        ));
        let mut bad_color = small_list();
        bad_color.0[0] = DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            color: [1.0, 0.0, 0.0, 1.5],
        };
        assert!(matches!(
            bad_color.validate(&resources),
            Err(ValidationError::BadColor(_))
        ));
        let mut huge = small_list();
        huge.0[0] = DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: MAX_COORDINATE + 1.0,
            height: 10.0,
            color: [1.0; 4],
        };
        assert!(matches!(
            huge.validate(&resources),
            Err(ValidationError::BadSize(_))
        ));
    }

    #[test]
    fn validation_rejects_missing_resources_and_bad_tables() {
        let mut resources = ResourceTable::new();
        let id = ResourceId(7);
        // An image naming an id the table does not hold.
        let mut list = small_list();
        list.push(DisplayItem::Image {
            resource: id,
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        });
        assert_eq!(
            list.validate(&resources),
            Err(ValidationError::MissingResource(id))
        );
        // A blob naming the same missing id fails the same way.
        let mut blob_list = small_list();
        blob_list.push(DisplayItem::Blob {
            resource: id,
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        });
        assert_eq!(
            blob_list.validate(&resources),
            Err(ValidationError::MissingResource(id))
        );
        // The id present but the pixels the wrong length: the table
        // itself is invalid, so the list fails too.
        resources.insert(
            id,
            Resource::Image {
                width: 2,
                height: 2,
                rgba: vec![0u8; 15], // 16 expected
            },
        );
        assert!(matches!(
            list.validate(&resources),
            Err(ValidationError::ImageLengthMismatch { .. })
        ));
        // Correct pixels: both the image and the blob validate.
        resources.insert(
            id,
            Resource::Image {
                width: 2,
                height: 2,
                rgba: vec![0u8; 16],
            },
        );
        list.validate(&resources).expect("image resolves");
        blob_list.validate(&resources).expect("blob resolves");
        // And the table itself round-trips through bytes with the list.
        let table_bytes = postcard::to_allocvec(&resources).expect("table serialises");
        let back: ResourceTable = postcard::from_bytes(&table_bytes).expect("table deserialises");
        assert_eq!(back, resources);
    }

    #[test]
    fn validation_rejects_too_many_items() {
        let resources = ResourceTable::new();
        let mut list = DisplayList::new();
        for _ in 0..=MAX_DISPLAY_ITEMS {
            list.push(DisplayItem::FillRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                color: [0.0; 4],
            });
        }
        assert_eq!(
            list.validate(&resources),
            Err(ValidationError::TooManyItems(MAX_DISPLAY_ITEMS + 1))
        );
    }
}
