//! Layout & text (layer 4): taffy + cosmic-text bridge.
//!
//! Implements `taffy::LayoutPartialTree` over the styled DOM nodes. For
//! text leaf nodes, `cosmic_text::Buffer` shaping results go inside
//! Taffy's `measure_function`, so line wrapping and intrinsic sizes match
//! pixel-for-pixel what `pluto-paint` later draws. The bridge lands in
//! the layout round; this stub owns the two objects it is built from --
//! the Taffy tree and the font system -- so their versions stay the
//! workspace-wide ones.

/// Owns the Taffy layout tree and the cosmic-text font system for one
/// document. The `LayoutPartialTree` impl and the `measure_function`
/// that shapes text leaves land in the layout round.
pub struct LayoutTree {
    pub taffy: taffy::TaffyTree<()>,
    pub font_system: cosmic_text::FontSystem,
}

impl LayoutTree {
    pub fn new() -> Self {
        Self {
            taffy: taffy::TaffyTree::new(),
            font_system: cosmic_text::FontSystem::new(),
        }
    }
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}
