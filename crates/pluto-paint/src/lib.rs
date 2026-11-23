//! GPU rendering (layer 5): display list + wgpu 2D/text renderer.
//!
//! Walks the laid-out Taffy tree in paint order (z-index / stacking
//! contexts) and emits quads (backgrounds, borders, images) and glyph
//! runs into a [`DisplayList`]. The wgpu renderer turns the list into a
//! render pass in the paint round, using cosmic-text's `SwashCache` for
//! glyph atlases. Upload happens in the browser viewport's
//! `Primitive::prepare`; drawing happens in `Primitive::render`.

/// One paint operation, in paint order.
#[derive(Debug, Clone)]
pub enum DisplayItem {
    /// A solid rectangle: page/element backgrounds and borders.
    FillRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
    },
    /// A shaped text run. Glyphs are resolved through the `SwashCache`
    /// at render time, from the same `cosmic_text::Buffer` the layout
    /// step measured.
    Text {
        x: f32,
        y: f32,
        text: String,
        size: f32,
        color: [f32; 4],
    },
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
}
