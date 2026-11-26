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

use pluto_css::{Affine, GradientKind, MAX_BOX_SHADOW_BLUR, ResolvedCorners, ResolvedShadow};
use pluto_html::Dom;
use pluto_layout::{
    BackgroundPosition, BackgroundRepeat, BackgroundSize, Dim, LayoutBox, LayoutTree,
    SvgCanvasKind,
};

use crate::canvas::canvas_blob_id;
use crate::path::{VectorVertex, clip_colored_triangles, linear_gradient_quad};
use crate::raster::rasterize_gradient_rect;
use crate::svg::{
    MAX_RASTER_PIXELS, SvgContent, SvgRasterCache, bg_gradient_blob_id, collect_svg_content,
    gradient_fingerprint, parse_view_box, rasterize_svg_content, svg_blob_id, svg_fingerprint,
    svg_needs_raster, tessellate_svg_shape, view_box_transform,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod canvas;
pub mod glyph_key;
pub mod image;
pub mod path;
pub mod perf;
pub mod raster;
pub mod raster_spike;
pub mod svg;
pub mod renderer;

pub use glyph_key::GlyphKey;
pub use image::{
    AnimatedFrames, AnimatedImage, DecodedImage, IMAGE_CACHE_BYTE_CAP, ImageCache, ImageError,
    MAX_ANIMATED_FRAMES, MAX_ANIMATION_FPS, MAX_IMAGE_DIMENSION, MIN_FRAME_DELAY_MS,
    build_mip_chain, decode_animated, decode_image, decode_images_parallel, decode_limits,
    downscale_rgba, mip_chain_bytes, mip_level_count_for, mip_level_for_scale, resize_rgba,
};

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
    /// A solid rounded rectangle: a box background whose
    /// `border-radius` is non-zero. The fill is the rounded-rect
    /// SDF coverage in the renderer (Stage 8 Rec 4); a square box
    /// still uses [`DisplayItem::FillRect`].
    RoundedRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radii: ResolvedCorners,
        color: [f32; 4],
    },
    /// One `box-shadow` layer (Stage 8 Rec 4), resolved to used
    /// pixels. The rect is the box's border box; the renderer
    /// expands it by `spread`, translates it by the offsets, and
    /// blurs it. `blur == 0` draws as an expanded rounded rect
    /// behind the box; `blur > 0` uses the erf closed-form shader.
    /// `inset` shadows draw after the background, before content
    /// (the builder emits them in that position); outer shadows
    /// draw before the background. Layers are emitted in reverse
    /// author order so the first-listed layer paints on top.
    BoxShadow {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radii: ResolvedCorners,
        offset_x: f32,
        offset_y: f32,
        blur: f32,
        spread: f32,
        color: [f32; 4],
        inset: bool,
    },
    /// Push a clip rect for a scroll container: every following item
    /// paints clipped to this rect until the matching `PopClip`.
    /// (R14, M2: the vocabulary gains the clip; the shader-side
    /// scissor itself is a Stage 8 follow-up — the renderer currently
    /// ignores these items, so they are data, not a visual change.)
    /// Stage 8 Rec 3: the renderer honours these — rect clips via
    /// `set_scissor_rect` (batched per scissor run, items outside
    /// culled on the CPU, never a zero-size scissor) and rounded
    /// clips (`radii: Some`) via an SDF discard in the fragment
    /// shader. The clip rect is the container's padding box.
    PushClip {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radii: Option<ResolvedCorners>,
    },
    /// Pop the most recent `PushClip`.
    PopClip,
    /// The text caret of the focused editable control (Stage 13
    /// Rec 4): a 2px vertical bar at `x`, spanning the line height
    /// from `y`. The renderer draws it as a solid rect; the worker
    /// blinks it by omitting the item on alternate ~530 ms ticks.
    /// `color` is the control's text color.
    Caret {
        x: f32,
        y: f32,
        height: f32,
        color: [f32; 4],
    },
    /// The text-selection highlight of the focused editable
    /// control (Stage 13 Rec 4): one rect per selected line run
    /// (single-line controls emit one). Painted under the glyphs,
    /// above the background.
    SelectionRect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
    },
    /// Push a transform for the items that follow (M6 Phase 2,
    /// Rec 7): `matrix` maps layout coordinates to page
    /// coordinates — the box's page transform, absolute, so it
    /// already includes the ancestors and nesting never
    /// double-applies. The renderer applies it CPU-side to every
    /// vertex it emits until the matching `PopTransform`; no new
    /// bind group is needed (both are spoken for). `id` is the
    /// element's compositing key (`LayoutTree::dom_key`): a
    /// `CompositingUpdate` patches this matrix in place by id, so
    /// a transform-only change never rebuilds the list.
    PushTransform { id: u64, matrix: Affine },
    /// Pop the most recent `PushTransform`.
    PopTransform,
    /// Push a group opacity for the items that follow (M6 Phase 2,
    /// Rec 6): everything between this and the matching
    /// `PopOpacity` is one stacking-context group, composited at
    /// `opacity` as a layer. The renderer applies it by scaling
    /// vertex alpha — an explicit documented approximation of true
    /// group compositing (overlapping translucent siblings inside
    /// one group double-apply); the grouping here is already
    /// correct for a future offscreen-compositing renderer.
    /// `id` is the same compositing key as `PushTransform`.
    PushOpacity { id: u64, opacity: f32 },
    /// Pop the most recent `PushOpacity`.
    PopOpacity,
    /// Push viewport-fixed scope for the items that follow (M6
    /// Phase 3, Rec 1): everything between this and the matching
    /// `PopFixed` is painted in *viewport* coordinates, not page
    /// coordinates. The builder wraps every viewport-fixed subtree
    /// (a `position: fixed` box with no transformed ancestor, plus
    /// its descendants) in one of these. The renderer, which
    /// subtracts the viewport scroll from every page-space vertex,
    /// adds the scroll back inside this scope — the
    /// `canvas_rect` translate-by-`+scroll` pattern — so fixed
    /// content stays glued to the viewport. Appended last, so the
    /// earlier postcard discriminants never move.
    PushFixed,
    /// Pop the most recent `PushFixed`.
    PopFixed,
    /// A tessellated vector path (Stage 15, M7 Phase 2, Rec 4):
    /// flat coloured triangles in layout space, one item per
    /// shape of an inline `<svg>` (solid fills, two-stop linear
    /// gradients) or per CSS linear-gradient background.
    /// Vertices come in triples; `bounds` is the item's rect
    /// (the svg box, or the background box) for damage and CPU
    /// culling. Appended last, so the earlier postcard
    /// discriminants never move — the wire bump is
    /// `PROTOCOL_VERSION` 9→10.
    VectorPath {
        vertices: Vec<VectorVertex>,
        bounds: [f32; 4],
    },
}

/// One element's compositing state (M6 Phase 2, Rec 9): the
/// transform/opacity of a single element, keyed by the same id its
/// `PushTransform`/`PushOpacity` items carry. Sent out-of-band in
/// `RendererToBrowser::CompositingUpdate` — the message carries
/// only the changed elements, never a rebuilt list, and stays far
/// under the 64 MiB IPC frame cap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositedUpdate {
    pub id: u64,
    pub matrix: Affine,
    pub opacity: f32,
}

impl DisplayItem {
    /// The item's bounds in layout space, when it has computable
    /// ones: the rect the item paints (a shadow's bounds are the
    /// shape expanded by `spread` and inflated by `~3σ` of blur,
    /// the region its falloff can touch). `Glyphs` has no bounds
    /// here — glyph quads need font metrics the list does not
    /// carry — and `PopClip` has none. Used for the damage union
    /// and for CPU culling in the renderer.
    pub fn bounds(&self) -> Option<[f32; 4]> {
        match self {
            DisplayItem::FillRect {
                x,
                y,
                width,
                height,
                ..
            }
            | DisplayItem::RoundedRect {
                x,
                y,
                width,
                height,
                ..
            }
            | DisplayItem::Image {
                x,
                y,
                width,
                height,
                ..
            }
            | DisplayItem::Blob {
                x,
                y,
                width,
                height,
                ..
            }
            | DisplayItem::PushClip {
                x,
                y,
                width,
                height,
                ..
            }
            | DisplayItem::SelectionRect {
                x,
                y,
                width,
                height,
                ..
            } => Some([*x, *y, *width, *height]),
            DisplayItem::Caret { x, y, height, .. } => Some([*x, *y, 2.0, *height]),
            DisplayItem::VectorPath { bounds, .. } => Some(*bounds),
            DisplayItem::BoxShadow {
                x,
                y,
                width,
                height,
                offset_x,
                offset_y,
                blur,
                spread,
                inset,
                ..
            } => {
                // The shadow's reach: the shape (expanded by spread,
                // translated by the offset) plus the blur falloff.
                // An inset shadow never leaves the box.
                if *inset {
                    Some([*x, *y, *width, *height])
                } else {
                    let falloff = 3.0 * (*blur / 2.0) + 1.0;
                    let pad = *spread + falloff;
                    Some([
                        x + offset_x - pad,
                        y + offset_y - pad,
                        width + 2.0 * pad,
                        height + 2.0 * pad,
                    ])
                }
            }
            DisplayItem::Glyphs { .. }
            | DisplayItem::PopClip
            | DisplayItem::PushTransform { .. }
            | DisplayItem::PopTransform
            | DisplayItem::PushOpacity { .. }
            | DisplayItem::PopOpacity
            | DisplayItem::PushFixed
            | DisplayItem::PopFixed => None,
        }
    }

    /// The item's bounds under a page transform (M6 Phase 2,
    /// Rec 7): the axis-aligned bounding box of the item's corners
    /// mapped through `matrix`. `bounds()` is the untransformed
    /// rect, which is wrong for a rotated item — damage and CPU
    /// culling must use this. Exact for translate/scale; an
    /// over-approximation for rotate/skew (the AABB of the rotated
    /// AABB is bigger than the rotated shape), which is safe for
    /// both uses. `None` for the items `bounds()` returns `None`
    /// for.
    pub fn transformed_bounds(&self, matrix: &Affine) -> Option<[f32; 4]> {
        let [x, y, w, h] = self.bounds()?;
        let corners = [
            matrix.apply(x, y),
            matrix.apply(x + w, y),
            matrix.apply(x, y + h),
            matrix.apply(x + w, y + h),
        ];
        let min_x = corners.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
        let max_x = corners.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
        let min_y = corners.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
        let max_y = corners.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        Some([min_x, min_y, max_x - min_x, max_y - min_y])
    }
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
///
/// M3 (Stage 11): resources cross IPC out-of-band, once each, in
/// `RendererToBrowser::ResourceReady` — the worker sends
/// `SubmitDisplayList.resources` empty and the shell accumulates
/// the table per tab from those messages (see `pluto-ipc`'s module
/// docs). This type is unchanged: it is the shell-side table now,
/// filled from `ResourceReady`, not from the submit.
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
/// Cap on the vertices of one `VectorPath` item (Stage 15):
/// ~2.6 MB of vertex data, far above any real svg shape (the
/// tessellator's own cap is 65_536) and far below anything that
/// would let a hostile renderer blow the browser's memory.
pub const MAX_VECTOR_VERTICES: usize = 262_144;
pub const MAX_COORDINATE: f32 = 100_000.0;
pub const MAX_RESOURCE_BYTES: usize = 64 * 1024 * 1024;

/// What [`DisplayList::validate`] rejects, and why.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidationError {
    #[error("display list has {0} items, over the limit of {MAX_DISPLAY_ITEMS}")]
    TooManyItems(usize),
    #[error("display list has {0} glyphs, over the limit of {MAX_GLYPHS_TOTAL}")]
    TooManyGlyphs(usize),
    #[error("vector path has {0} vertices, over the limit of {MAX_VECTOR_VERTICES}")]
    TooManyVectorVertices(usize),
    #[error("vector path has {0} vertices, not a multiple of 3")]
    VectorPathNotTriangles(usize),
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
    #[error("box-shadow blur {0}px exceeds the {MAX_BOX_SHADOW_BLUR}px cap")]
    ShadowBlurTooLarge(f32),
    #[error("non-finite transform component {0}")]
    BadTransform(f32),
    #[error("opacity {0} outside 0..=1")]
    BadOpacity(f32),
    #[error("display list bytes do not decode: {0}")]
    Decode(String),
}

/// The damaged region of one display list, in layout space (Stage 8
/// Rec 5): the union of the invalidated item bounds. The worker
/// attaches it to each list; the shell skips a list whose damage is
/// empty (a clean skip, counted separately from deadline misses)
/// and records the damage-area ratio against the viewport. A full
/// rebuild invalidates everything it paints, so the damage of a
/// built list is the union of its boxes' bounds; per-item damage
/// (only the changed items) arrives with dirty-subtree layout.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DamageRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl DamageRect {
    /// The union of two damage rects.
    pub fn union(self, other: DamageRect) -> DamageRect {
        let x0 = self.x.min(other.x);
        let y0 = self.y.min(other.y);
        let x1 = (self.x + self.width).max(other.x + other.width);
        let y1 = (self.y + self.height).max(other.y + other.height);
        DamageRect {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }

    /// Union with a `[x, y, w, h]` rect.
    pub fn union_rect(self, rect: [f32; 4]) -> DamageRect {
        self.union(DamageRect {
            x: rect[0],
            y: rect[1],
            width: rect[2],
            height: rect[3],
        })
    }

    /// Area in px², for the damage-area ratio.
    pub fn area(self) -> f32 {
        (self.width * self.height).max(0.0)
    }

    /// True when the rect covers nothing.
    pub fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }
}

/// The ordered paint operations for one frame, plus the damage rect
/// (Stage 8 Rec 5) the worker computed for it. `damage: None` means
/// the list paints nothing new — the shell clean-skips it.
///
/// Wire note (Stage 8): adding the damage field (and the
/// `PushClip` radii / new item variants) changes the postcard bytes,
/// so bytes written by an older build do not decode under this one.
/// Pluto ships one binary and the list is in-process today, so no
/// runtime migration exists; `PROTOCOL_VERSION` stays 2 by the
/// coordinator's decision (the v3 bump belongs to the wave that
/// changes `InputEvent`), and this incompatibility is flagged for
/// that wave.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplayList(pub Vec<DisplayItem>, pub Option<DamageRect>);

impl DisplayList {
    pub fn new() -> Self {
        Self(Vec::new(), None)
    }

    pub fn push(&mut self, item: DisplayItem) {
        // Keep the damage honest for lists extended after build:
        // a pushed item with known bounds extends the union.
        if let Some(bounds) = item.bounds() {
            self.1 = Some(match self.1 {
                Some(damage) => damage.union_rect(bounds),
                None => DamageRect {
                    x: bounds[0],
                    y: bounds[1],
                    width: bounds[2],
                    height: bounds[3],
                },
            });
        }
        self.0.push(item);
    }

    pub fn clear(&mut self) {
        self.0.clear();
        self.1 = None;
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
                DisplayItem::RoundedRect {
                    x,
                    y,
                    width,
                    height,
                    radii,
                    color,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                    check_radii(*radii)?;
                    for c in color {
                        if !c.is_finite() || !(0.0..=1.0).contains(c) {
                            return Err(ValidationError::BadColor(*c));
                        }
                    }
                }
                DisplayItem::BoxShadow {
                    x,
                    y,
                    width,
                    height,
                    radii,
                    offset_x,
                    offset_y,
                    blur,
                    spread,
                    color,
                    ..
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                    check_radii(*radii)?;
                    check_coord(*offset_x)?;
                    check_coord(*offset_y)?;
                    if !blur.is_finite() || *blur < 0.0 || *blur > MAX_BOX_SHADOW_BLUR {
                        return Err(ValidationError::ShadowBlurTooLarge(*blur));
                    }
                    check_size(*spread)?;
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
                    radii,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*width)?;
                    check_size(*height)?;
                    if let Some(radii) = radii {
                        check_radii(*radii)?;
                    }
                }
                DisplayItem::PopClip => {}
                DisplayItem::Caret {
                    x,
                    y,
                    height,
                    color,
                } => {
                    check_coord(*x)?;
                    check_coord(*y)?;
                    check_size(*height)?;
                    for c in color {
                        if !c.is_finite() || !(0.0..=1.0).contains(c) {
                            return Err(ValidationError::BadColor(*c));
                        }
                    }
                }
                DisplayItem::SelectionRect {
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
                DisplayItem::PushTransform { matrix, .. } => {
                    // The linear part is a scale/rotation factor, not
                    // a coordinate — only finiteness is meaningful
                    // there; the translation is a page coordinate.
                    for c in [matrix.a, matrix.b, matrix.c, matrix.d] {
                        if !c.is_finite() {
                            return Err(ValidationError::BadTransform(c));
                        }
                    }
                    check_coord(matrix.e)?;
                    check_coord(matrix.f)?;
                }
                DisplayItem::PopTransform => {}
                DisplayItem::PushOpacity { opacity, .. } => {
                    if !opacity.is_finite() || !(0.0..=1.0).contains(opacity) {
                        return Err(ValidationError::BadOpacity(*opacity));
                    }
                }
                DisplayItem::PopOpacity => {}
                DisplayItem::PushFixed | DisplayItem::PopFixed => {}
                DisplayItem::VectorPath { vertices, bounds } => {
                    check_coord(bounds[0])?;
                    check_coord(bounds[1])?;
                    check_size(bounds[2])?;
                    check_size(bounds[3])?;
                    if vertices.len() > MAX_VECTOR_VERTICES {
                        return Err(ValidationError::TooManyVectorVertices(vertices.len()));
                    }
                    if !vertices.len().is_multiple_of(3) {
                        return Err(ValidationError::VectorPathNotTriangles(vertices.len()));
                    }
                    for v in vertices {
                        check_coord(v.position[0])?;
                        check_coord(v.position[1])?;
                        for c in v.color {
                            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                                return Err(ValidationError::BadColor(c));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Build the paint operations for one laid-out document. The canvas
    /// background (html/body background, or white by default) paints the
    /// whole viewport first — it is always item 0, which the renderer
    /// relies on for the scroll-anchored canvas — then the boxes in
    /// Appendix E paint order (`LayoutTree::paint_order`, M6 Phase 2
    /// Rec 4), not document order, so a higher `z-index` paints
    /// later (on top) and hit-testing agrees with what is seen.
    ///
    /// Per box, in order: the clip stack is reconciled against the
    /// box's clip chain (Rec 5 — a real stack keyed to the paint
    /// walk, so a clipped subtree that paints out of document order
    /// still clips correctly), the opacity stack against the box's
    /// translucent-ancestor chain (Rec 6 — one `PushOpacity` group
    /// per stacking context with `opacity < 1`), a `PushTransform`
    /// carrying the box's page transform when it is not the identity
    /// (Rec 7 — absolute, so descendants are never double-applied),
    /// then the box's own items: its outer box-shadows, its
    /// background (a rounded rect when the box has a
    /// `border-radius`, a plain rect otherwise), its inset
    /// box-shadows (above the background, below the content), its
    /// background image, its `<img>`, then its glyphs. Clip rects
    /// in the list are page-space AABBs of the layout-space clip
    /// under the box's page transform (exact for translate/scale,
    /// an over-approximation for rotate/skew); item coordinates
    /// stay in layout space and the renderer transforms them.
    /// The list's damage is the union of the canvas, every box's
    /// transformed bounds, and every item's transformed bounds (a
    /// shadow's falloff can leave its box), so a rotated item is
    /// never damage-culled.
    /// Build the paint operations for one laid-out document (see
    /// [`DisplayList::build_inner`]). Inline `<svg>` elements and
    /// radial CSS gradient backgrounds paint nothing here — use
    /// [`DisplayList::build_with_svg`] for those.
    pub fn build(layout: &LayoutTree) -> Self {
        Self::build_inner(layout, None, &HashMap::new(), &[], &[]).0
    }

    /// Like [`DisplayList::build`], but inline `<svg>` elements
    /// and radial CSS gradient backgrounds also paint (vector
    /// triangles or raster Blobs — see `build_inner`). Returns
    /// the list plus the new raster resources: each is a
    /// `(ResourceId, Resource)` the tab thread must insert into
    /// its image cache and announce with `ResourceReady` before
    /// submitting the list. The ids are deterministic (see
    /// [`svg_blob_id`]), so a scroll rebuild names the id the
    /// shell already holds and no entry is returned twice.
    pub fn build_with_svg(
        layout: &LayoutTree,
        dom: &Dom,
        cache: &mut SvgRasterCache,
    ) -> (Self, Vec<(ResourceId, Resource)>) {
        Self::build_inner(layout, Some((dom, cache)), &HashMap::new(), &[], &[])
    }

    /// Like [`DisplayList::build_with_svg`], plus the Phase 3
    /// producers (Stage 15, M7 Phase 3, Rec 8/10): `svg_images`
    /// resolves `<svg><image>` elements (by DOM id) to the
    /// resource ids the tab thread's image map already holds;
    /// `canvas_live` names every `<canvas>` with a bitmap (its
    /// Blob is emitted even when the bitmap is unchanged);
    /// `canvas_new` carries the fresh bitmaps (id + `Resource`)
    /// the tab thread must announce — only dirty canvases ride
    /// here, so a static canvas costs one `ResourceReady` frame,
    /// not one per tick. Canvas blob ids are deterministic
    /// ([`canvas_blob_id`]).
    pub fn build_with_svg_and_canvas(
        layout: &LayoutTree,
        dom: &Dom,
        cache: &mut SvgRasterCache,
        svg_images: &HashMap<pluto_html::NodeId, ResourceId>,
        canvas_live: &[pluto_html::NodeId],
        canvas_new: &[(pluto_html::NodeId, Resource)],
    ) -> (Self, Vec<(ResourceId, Resource)>) {
        Self::build_inner(
            layout,
            Some((dom, cache)),
            svg_images,
            canvas_live,
            canvas_new,
        )
    }

    /// The shared builder: `svg` carries the DOM + raster cache
    /// for the SVG/gradient routes (`None` in plain `build()` —
    /// radial backgrounds and inline SVGs then paint nothing,
    /// and the box keeps its background fill).
    fn build_inner(
        layout: &LayoutTree,
        mut svg: Option<(&Dom, &mut SvgRasterCache)>,
        svg_images: &HashMap<pluto_html::NodeId, ResourceId>,
        canvas_live: &[pluto_html::NodeId],
        canvas_new: &[(pluto_html::NodeId, Resource)],
    ) -> (Self, Vec<(ResourceId, Resource)>) {
        let mut list = Self::new();
        let mut new_blobs: Vec<(ResourceId, Resource)> = Vec::new();
        let bg = layout.canvas_background;
        list.0.push(DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: layout.viewport.0,
            height: layout.viewport.1,
            color: rgba(bg),
        });
        let boxes = layout.boxes();
        let order = layout.paint_order();
        let transforms = layout.page_transforms();
        let mut by_dom: HashMap<_, usize> = HashMap::new();
        for (i, bx) in boxes.iter().enumerate() {
            if let Some(dom) = bx.dom {
                by_dom.entry(dom).or_insert(i);
            }
        }
        // The effective clip chain of a box, root-first: the
        // box's own clip plus each clipping ancestor's, with
        // consecutive duplicates merged (a nested clip equal to
        // its parent's is the same region). Layout space; the
        // caller maps each entry through the box's page transform.
        let clip_chain = |bi: usize| -> Vec<([f32; 4], Option<ResolvedCorners>)> {
            let mut chain: Vec<([f32; 4], Option<ResolvedCorners>)> = Vec::new();
            let mut cur = Some(bi);
            while let Some(i) = cur {
                let bx = &boxes[i];
                if let Some(c) = bx.clip {
                    let entry = (c, bx.clip_radii);
                    if chain.last() != Some(&entry) {
                        chain.push(entry);
                    }
                }
                cur = bx.parent.and_then(|p| by_dom.get(&p).copied());
            }
            chain.reverse();
            chain
        };
        // The translucent stacking-context ancestors of a box,
        // root-first (the box itself included when it is one):
        // the groups whose `PushOpacity` must wrap the box's
        // items. Keyed by the element's compositing key; text
        // leaves have no key of their own and never contribute
        // an entry — their parent element's entry covers them.
        let opacity_chain = |bi: usize| -> Vec<(u64, f32)> {
            let mut chain: Vec<(u64, f32)> = Vec::new();
            let mut cur = Some(bi);
            while let Some(i) = cur {
                let bx = &boxes[i];
                if bx.style.opacity < 1.0 {
                    if let Some(dom) = bx.dom {
                        chain.push((LayoutTree::dom_key(dom), bx.style.opacity));
                    }
                }
                cur = bx.parent.and_then(|p| by_dom.get(&p).copied());
            }
            chain.reverse();
            chain
        };
        // The compositing key of a box: its own element's key, or
        // for a text leaf the nearest ancestor element's — the
        // leaf's page transform is the same map as its parent
        // element's, so a `CompositingUpdate` for the parent
        // patches the leaf too.
        let compositing_key = |bi: usize| -> u64 {
            let mut cur = Some(bi);
            while let Some(i) = cur {
                let bx = &boxes[i];
                if let Some(dom) = bx.dom {
                    return LayoutTree::dom_key(dom);
                }
                cur = bx.parent.and_then(|p| by_dom.get(&p).copied());
            }
            0
        };
        // The two stacks the walk maintains, reconciled per box
        // against the box's own chains (pop to the common prefix,
        // then push the rest).
        let mut clip_stack: Vec<([f32; 4], Option<ResolvedCorners>)> = Vec::new();
        let mut opacity_stack: Vec<(u64, f32)> = Vec::new();
        // The fixed stack (M6 Phase 3, Rec 1): at most one level —
        // a box is either inside a viewport-fixed subtree or it
        // is not. The chain walks the box's ancestors (a fixed
        // box's descendants are fixed too — their own
        // `is_viewport_fixed` is false, they are static).
        let fixed_chain = |bi: usize| -> Vec<bool> {
            let mut cur = Some(bi);
            while let Some(i) = cur {
                if boxes[i].is_viewport_fixed {
                    return vec![true];
                }
                cur = boxes[i].parent.and_then(|p| by_dom.get(&p).copied());
            }
            Vec::new()
        };
        let mut fixed_stack: Vec<bool> = Vec::new();
        // Damage: the canvas plus every box's transformed bounds
        // (glyph-only boxes contribute their box rects here, since
        // glyph items carry no bounds of their own); item bounds
        // are unioned in a final pass below.
        let mut damage = DamageRect {
            x: 0.0,
            y: 0.0,
            width: layout.viewport.0,
            height: layout.viewport.1,
        };
        for &bi in &order {
            let bx = &boxes[bi];
            let page_t = transforms[bi];
            // Rec 1 (M6 Phase 3): reconcile the fixed stack first
            // (outermost) — a viewport-fixed subtree paints in
            // viewport coordinates, so its items ride inside a
            // `PushFixed` scope.
            reconcile_stack(
                &mut list,
                &mut fixed_stack,
                &fixed_chain(bi),
                |_| DisplayItem::PushFixed,
                || DisplayItem::PopFixed,
            );
            let in_fixed = !fixed_stack.is_empty();
            // Rec 5: reconcile the clip stack against this box's
            // chain, each entry mapped to page space by the box's
            // own page transform.
            let page_chain: Vec<([f32; 4], Option<ResolvedCorners>)> = clip_chain(bi)
                .iter()
                .map(|(r, radii)| {
                    (
                        transform_rect(*r, &page_t),
                        radii.map(|r| scale_radii(r, &page_t)),
                    )
                })
                .collect();
            reconcile_stack(
                &mut list,
                &mut clip_stack,
                &page_chain,
                |&(r, radii)| DisplayItem::PushClip {
                    x: r[0],
                    y: r[1],
                    width: r[2],
                    height: r[3],
                    radii,
                },
                || DisplayItem::PopClip,
            );
            // Rec 6: reconcile the opacity stack against this
            // box's translucent-ancestor chain.
            reconcile_stack(
                &mut list,
                &mut opacity_stack,
                &opacity_chain(bi),
                |&(id, opacity)| DisplayItem::PushOpacity { id, opacity },
                || DisplayItem::PopOpacity,
            );
            // Rec 7: the box's page transform, pushed around its
            // own items only. The identity needs no item (the
            // renderer treats an empty stack as the identity), so
            // untransformed documents pay nothing for this.
            let has_transform = page_t != Affine::identity();
            if has_transform {
                list.0.push(DisplayItem::PushTransform {
                    id: compositing_key(bi),
                    matrix: page_t,
                });
            }
            // A box with no `dom` is a text leaf: its style is its
            // parent element's clone, so its background and
            // box-shadows are the parent's — the parent's own box
            // already paints them, and the leaf painting them
            // again would draw a second shadow around the text
            // run's rect. A leaf paints only its glyphs.
            if bx.dom.is_some() {
                // Box-shadows, resolved once per box: outer layers before
                // the background (they paint behind the box), inset
                // layers after it (above the background, below the
                // content). Each side is emitted in reverse author order
                // so the first-listed layer paints on top.
                let shadows: Vec<ResolvedShadow> = bx
                    .style
                    .box_shadow
                    .iter()
                    .map(|s| {
                        s.resolve(
                            bx.style.font_size,
                            layout.root_font_size,
                            layout.viewport,
                            bx.style.color,
                        )
                    })
                    .collect();
                for shadow in shadows.iter().rev().filter(|s| !s.inset) {
                    list.0.push(shadow_item(bx, *shadow));
                }
                if let Some(bg) = bx.style.background {
                    if bx.radii.is_zero() {
                        list.0.push(DisplayItem::FillRect {
                            x: bx.x,
                            y: bx.y,
                            width: bx.width,
                            height: bx.height,
                            color: rgba(bg),
                        });
                    } else {
                        list.0.push(DisplayItem::RoundedRect {
                            x: bx.x,
                            y: bx.y,
                            width: bx.width,
                            height: bx.height,
                            radii: bx.radii,
                            color: rgba(bg),
                        });
                    }
                }
                for shadow in shadows.iter().rev().filter(|s| s.inset) {
                    list.0.push(shadow_item(bx, *shadow));
                }
            }
            // Stage 11 Rec 1/3: images. The background image paints
            // above the background fill and below the content, clipped
            // to the box's padding box (with its border-radius, when
            // set); the `<img>` content image paints as the content.
            // The PushClip/PopClip pair is balanced inside this box's
            // iteration, so the clip stack above is unaffected. The
            // clip rect is page space — the box's padding box mapped
            // through the box's page transform — while the tiles
            // stay in layout space and the renderer transforms them.
            if let Some(resource) = bx.background_image {
                let tiles = background_tiles(bx, layout.viewport);
                if !tiles.is_empty() {
                    let clip_rect =
                        transform_rect([bx.x, bx.y, bx.width, bx.height], &page_t);
                    list.0.push(DisplayItem::PushClip {
                        x: clip_rect[0],
                        y: clip_rect[1],
                        width: clip_rect[2],
                        height: clip_rect[3],
                        radii: if bx.radii.is_zero() {
                            None
                        } else {
                            Some(scale_radii(bx.radii, &page_t))
                        },
                    });
                    for [x, y, w, h] in tiles {
                        list.0.push(DisplayItem::Image {
                            resource: ResourceId(resource),
                            x,
                            y,
                            width: w,
                            height: h,
                        });
                    }
                    list.0.push(DisplayItem::PopClip);
                }
            }
            // Stage 15 Rec 7: CSS gradient backgrounds. A linear
            // gradient is a vector quad subdivided per stop segment
            // (exact for any stop count — see
            // `path::linear_gradient_quad`); a radial gradient
            // rasterises to a Blob (per-pixel evaluation, like an
            // inline SVG's radial fill). The radial route needs
            // the DOM + raster cache, so plain `build()` paints
            // nothing for a radial background — the box keeps its
            // background fill. `background-image` and
            // `background_gradient` are mutually exclusive in the
            // cascade (setting one clears the other), and a text
            // leaf's style is its parent's clone, so this is
            // guarded by `bx.dom.is_some()` — the parent's own
            // box already paints its gradient. Rounded boxes clip
            // the gradient to the padding box, like the
            // background-image tiles above.
            if bx.dom.is_some() {
                if let Some(gradient) = &bx.style.background_gradient {
                    let needs_clip = !bx.radii.is_zero();
                    if needs_clip {
                        let clip_rect =
                            transform_rect([bx.x, bx.y, bx.width, bx.height], &page_t);
                        list.0.push(DisplayItem::PushClip {
                            x: clip_rect[0],
                            y: clip_rect[1],
                            width: clip_rect[2],
                            height: clip_rect[3],
                            radii: Some(scale_radii(bx.radii, &page_t)),
                        });
                    }
                    match &gradient.kind {
                        GradientKind::Linear { .. } => {
                            let verts = linear_gradient_quad(
                                bx.x,
                                bx.y,
                                bx.width,
                                bx.height,
                                gradient,
                            );
                            if !verts.is_empty() {
                                list.0.push(DisplayItem::VectorPath {
                                    vertices: verts,
                                    bounds: [bx.x, bx.y, bx.width, bx.height],
                                });
                            }
                        }
                        GradientKind::Radial { .. } => {
                            if let (Some(dom_id), Some((_, cache))) =
                                (bx.dom, svg.as_mut())
                            {
                                Self::emit_radial_background(
                                    &mut list,
                                    &mut new_blobs,
                                    &mut **cache,
                                    dom_id,
                                    bx,
                                    gradient,
                                );
                            }
                        }
                    }
                    if needs_clip {
                        list.0.push(DisplayItem::PopClip);
                    }
                }
            }
            if let Some(resource) = bx.image {
                list.0.push(DisplayItem::Image {
                    resource: ResourceId(resource),
                    x: bx.x,
                    y: bx.y,
                    width: bx.width,
                    height: bx.height,
                });
            }
            // Stage 15 Rec 3/5: inline `<svg>` content. The box
            // is the svg's viewport; its children paint as
            // vector triangles (solid + two-stop linear fills)
            // or as one raster Blob (radial / multi-stop
            // gradients), clipped to the box. Needs the DOM +
            // raster cache, so plain `build()` paints nothing
            // for an svg box (its background still paints).
            if bx.svg_canvas == Some(SvgCanvasKind::Svg) {
                if let (Some(svg_id), Some((dom, cache))) = (bx.dom, svg.as_mut()) {
                    Self::emit_svg(
                        &mut list,
                        &mut new_blobs,
                        layout,
                        *dom,
                        &mut **cache,
                        svg_id,
                        bx,
                        svg_images,
                    );
                }
            }
            // Stage 15 (M7 Phase 3, Rec 8): a `<canvas>` with a
            // live bitmap paints it as a Blob at the box's rect.
            // The Blob is emitted whenever the canvas is live
            // (even unchanged — the id is deterministic, so the
            // shell already holds the pixels); a fresh bitmap
            // rides `new_blobs` only when it is in `canvas_new`
            // (dirty canvases only — see `build_with_svg_and_canvas`).
            if bx.svg_canvas == Some(SvgCanvasKind::Canvas) {
                if let Some(canvas_id) = bx.dom
                    && canvas_live.contains(&canvas_id)
                {
                    let id = canvas_blob_id(canvas_id);
                    if let Some((_, resource)) =
                        canvas_new.iter().find(|(d, _)| *d == canvas_id)
                    {
                        new_blobs.push((id, resource.clone()));
                    }
                    list.0.push(DisplayItem::Blob {
                        resource: id,
                        x: bx.x,
                        y: bx.y,
                        width: bx.width,
                        height: bx.height,
                    });
                }
            }
            if !bx.glyphs.is_empty() {
                // Group consecutive glyphs of the same color into one item.
                let mut current_color = bx.glyphs[0].color;
                let mut current: Vec<PaintGlyph> = Vec::new();
                for g in &bx.glyphs {
                    if g.color != current_color {
                        list.0.push(DisplayItem::Glyphs {
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
                    list.0.push(DisplayItem::Glyphs {
                        color: rgba(current_color),
                        glyphs: current,
                    });
                }
            }
            if has_transform {
                list.0.push(DisplayItem::PopTransform);
            }
            // Damage: this box's bounds under its page transform —
            // except viewport-fixed boxes, whose bounds are in
            // viewport space: unioning them would inflate the page
            // damage (and the page's maximum scroll) by the
            // viewport scroll.
            if !in_fixed {
                damage = damage.union_rect(transform_rect(
                    [bx.x, bx.y, bx.width, bx.height],
                    &page_t,
                ));
            }
        }
        // Close whatever the walk left open (the last box's chains).
        while !clip_stack.is_empty() {
            clip_stack.pop();
            list.0.push(DisplayItem::PopClip);
        }
        while !opacity_stack.is_empty() {
            opacity_stack.pop();
            list.0.push(DisplayItem::PopOpacity);
        }
        while !fixed_stack.is_empty() {
            fixed_stack.pop();
            list.0.push(DisplayItem::PopFixed);
        }
        // Damage, second half: every item's bounds under the
        // transform active at that point in the list — a shadow's
        // falloff can leave its box, and a rotated item's bounds
        // exceed its untransformed rect. `PushClip` bounds are
        // page space already (built that way above) and are
        // covered by the box bounds, so they are skipped here.
        // Items inside a fixed scope are viewport-space and
        // skipped too (same reason as the box union above).
        let mut transform_stack: Vec<Affine> = Vec::new();
        let mut fixed_depth = 0usize;
        for item in &list.0 {
            match item {
                DisplayItem::PushFixed => {
                    fixed_depth += 1;
                }
                DisplayItem::PopFixed => {
                    fixed_depth = fixed_depth.saturating_sub(1);
                }
                DisplayItem::PushTransform { matrix, .. } => {
                    transform_stack.push(*matrix);
                }
                DisplayItem::PopTransform => {
                    transform_stack.pop();
                }
                DisplayItem::PushClip { .. } => {}
                _ => {
                    if fixed_depth > 0 {
                        continue;
                    }
                    let t = transform_stack
                        .last()
                        .copied()
                        .unwrap_or_else(Affine::identity);
                    if let Some(b) = item.transformed_bounds(&t) {
                        damage = damage.union_rect(b);
                    }
                }
            }
        }
        list.1 = Some(damage);
        (list, new_blobs)
    }

    /// Paint one inline `<svg>` box (Stage 15, M7 Phase 2): the
    /// box is the svg's viewport. Vector-first: solid shapes and
    /// two-stop linear gradients tessellate to `VectorPath`
    /// items, clipped to the box; a radial or multi-stop
    /// gradient forces the raster route (one Blob, cached by
    /// content fingerprint + bitmap size). A raster refusal
    /// (oversize bitmap, tessellation failure) falls through to
    /// the vector route, which paints the shapes it can express
    /// and skips the rest — a shape whose tessellation fails is
    /// skipped individually, never the whole svg.
    fn emit_svg(
        list: &mut DisplayList,
        new_blobs: &mut Vec<(ResourceId, Resource)>,
        layout: &LayoutTree,
        dom: &Dom,
        cache: &mut SvgRasterCache,
        svg_id: pluto_html::NodeId,
        bx: &LayoutBox,
        svg_images: &HashMap<pluto_html::NodeId, ResourceId>,
    ) {
        if bx.width <= 0.0 || bx.height <= 0.0 {
            return;
        }
        let content = collect_svg_content(
            dom,
            svg_id,
            &layout.node_styles,
            (bx.width, bx.height),
        );
        if content.shapes.is_empty() && content.images.is_empty() {
            return;
        }
        let view = view_box_transform(
            bx.width,
            bx.height,
            dom.attr(svg_id, "viewBox").and_then(parse_view_box),
            dom.attr(svg_id, "preserveAspectRatio"),
        );
        if !content.shapes.is_empty() {
            // A raster refusal (oversize bitmap, tessellation
            // failure) falls through to the vector route, which
            // paints the shapes it can express and skips the rest.
            let rastered = if svg_needs_raster(&content) {
                Self::emit_svg_raster(list, new_blobs, dom, cache, svg_id, &content, view, bx)
            } else {
                false
            };
            if !rastered {
                Self::emit_svg_vector(list, &content, view, bx);
            }
        }
        // Stage 15 (M7 Phase 3, Rec 10): `<svg><image>` elements
        // paint as `Image` items at their (viewBox-mapped) rects,
        // above the shapes — only when the tab thread resolved
        // the href to a resource id (an unfetched / broken image
        // paints nothing, like a broken `<img>`).
        for img in &content.images {
            let Some(resource) = svg_images.get(&img.dom) else {
                continue;
            };
            let (x0, y0) = view.apply(img.x, img.y);
            let (x1, y1) = view.apply(img.x + img.width, img.y + img.height);
            list.0.push(DisplayItem::Image {
                resource: *resource,
                x: bx.x + x0,
                y: bx.y + y0,
                width: (x1 - x0).abs(),
                height: (y1 - y0).abs(),
            });
        }
    }

    /// Raster route for one svg box: one Blob for the whole svg,
    /// at the box's used size. Returns true when a Blob was
    /// emitted (cache hit or fresh raster); false when the
    /// rasteriser refused (oversize/empty/tessellation failure).
    /// The Blob id is deterministic (`svg_blob_id`), so a scroll
    /// rebuild names the id the shell already holds and no
    /// `ResourceReady` is sent twice — a fresh bitmap is
    /// returned in `new_blobs` only on a cache miss.
    fn emit_svg_raster(
        list: &mut DisplayList,
        new_blobs: &mut Vec<(ResourceId, Resource)>,
        dom: &Dom,
        cache: &mut SvgRasterCache,
        svg_id: pluto_html::NodeId,
        content: &SvgContent,
        view: Affine,
        bx: &LayoutBox,
    ) -> bool {
        let w = bx.width.round().max(0.0) as u32;
        let h = bx.height.round().max(0.0) as u32;
        if w == 0 || h == 0 {
            return false;
        }
        let id = svg_blob_id(svg_id);
        let fingerprint = svg_fingerprint(dom, svg_id);
        if cache.get(svg_id, w, h, fingerprint).is_some() {
            list.0.push(DisplayItem::Blob {
                resource: id,
                x: bx.x,
                y: bx.y,
                width: bx.width,
                height: bx.height,
            });
            return true;
        }
        let Some(img) = rasterize_svg_content(content, view, w, h) else {
            return false;
        };
        let resource = Resource::Image {
            width: img.width,
            height: img.height,
            rgba: img.rgba,
        };
        cache.insert(svg_id, w, h, fingerprint, id, resource.clone());
        new_blobs.push((id, resource));
        list.0.push(DisplayItem::Blob {
            resource: id,
            x: bx.x,
            y: bx.y,
            width: bx.width,
            height: bx.height,
        });
        true
    }

    /// Vector route for one svg box: one `VectorPath` item per
    /// shape (fill + stroke tessellated together), its vertices
    /// mapped by the viewBox transform, clipped to the box
    /// (Sutherland–Hodgman, colours interpolated — the
    /// spec-correct viewport clip, including `slice` overflow),
    /// then translated to the box's page position. A shape whose
    /// tessellation fails is skipped, never the whole svg.
    fn emit_svg_vector(
        list: &mut DisplayList,
        content: &SvgContent,
        view: Affine,
        bx: &LayoutBox,
    ) {
        let clip = [0.0, 0.0, bx.width, bx.height];
        for shape in &content.shapes {
            let Ok(mut verts) = tessellate_svg_shape(shape, content.viewport) else {
                continue;
            };
            if verts.is_empty() {
                continue;
            }
            for v in &mut verts {
                let (x, y) = view.apply(v.position[0], v.position[1]);
                v.position = [x, y];
            }
            let verts = clip_colored_triangles(&verts, clip);
            if verts.is_empty() {
                continue;
            }
            let verts = verts
                .into_iter()
                .map(|mut v| {
                    v.position[0] += bx.x;
                    v.position[1] += bx.y;
                    v
                })
                .collect();
            list.0.push(DisplayItem::VectorPath {
                vertices: verts,
                bounds: [bx.x, bx.y, bx.width, bx.height],
            });
        }
    }

    /// Raster route for one radial CSS gradient background: one
    /// Blob at the box's used size, cached by element id + size
    /// + gradient fingerprint (see `emit_svg_raster` for the
    /// deterministic-id contract). Oversize bitmaps paint
    /// nothing — the box keeps its background fill.
    fn emit_radial_background(
        list: &mut DisplayList,
        new_blobs: &mut Vec<(ResourceId, Resource)>,
        cache: &mut SvgRasterCache,
        dom_id: pluto_html::NodeId,
        bx: &LayoutBox,
        gradient: &pluto_css::Gradient,
    ) {
        let w = bx.width.round().max(0.0) as u32;
        let h = bx.height.round().max(0.0) as u32;
        if w == 0
            || h == 0
            || (w as usize).checked_mul(h as usize).unwrap_or(usize::MAX) > MAX_RASTER_PIXELS
        {
            return;
        }
        let id = bg_gradient_blob_id(dom_id);
        let fingerprint = gradient_fingerprint(gradient);
        if cache.get(dom_id, w, h, fingerprint).is_some() {
            list.0.push(DisplayItem::Blob {
                resource: id,
                x: bx.x,
                y: bx.y,
                width: bx.width,
                height: bx.height,
            });
            return;
        }
        let img = rasterize_gradient_rect(gradient, w, h);
        let resource = Resource::Image {
            width: img.width,
            height: img.height,
            rgba: img.rgba,
        };
        cache.insert(dom_id, w, h, fingerprint, id, resource.clone());
        new_blobs.push((id, resource));
        list.0.push(DisplayItem::Blob {
            resource: id,
            x: bx.x,
            y: bx.y,
            width: bx.width,
            height: bx.height,
        });
    }

    /// Patch this list in place with a compositing update (M6
    /// Phase 2, Rec 9): for each `CompositedUpdate`, every
    /// `PushTransform` with the matching id takes the new matrix
    /// and every `PushOpacity` with the matching id takes the new
    /// opacity. The update carries only the changed elements, so
    /// a transform-only frame never rebuilds or resubmits the
    /// list. Returns how many items were patched (0 when the
    /// update names nothing in this list — a stale update for a
    /// rebuilt list). Damage expands to the whole viewport
    /// afterwards: the old and new positions of a moved item can
    /// be anywhere, and a transformed item's old bounds are not
    /// recoverable from the patched list, so a full repaint is
    /// the safe invalidation.
    pub fn apply_compositing_update(&mut self, updates: &[CompositedUpdate]) -> usize {
        let mut patched = 0;
        for update in updates {
            for item in self.0.iter_mut() {
                match item {
                    DisplayItem::PushTransform { id, matrix } if *id == update.id => {
                        *matrix = update.matrix;
                        patched += 1;
                    }
                    DisplayItem::PushOpacity { id, opacity } if *id == update.id => {
                        *opacity = update.opacity;
                        patched += 1;
                    }
                    _ => {}
                }
            }
        }
        if patched > 0 {
            // The old damage no longer bounds the moved content —
            // its old position is not recoverable from the patched
            // list — so invalidate the whole canvas: old damage
            // union item 0 (the canvas fill, i.e. the viewport).
            // A built list's damage already covers the canvas, so
            // this is a no-op there; it matters when the list's
            // damage was ever narrowed.
            let canvas = self.0.first().and_then(|i| i.bounds());
            self.1 = Some(match (self.1, canvas) {
                (Some(old), Some(c)) => old.union_rect(c),
                (Some(old), None) => old,
                (None, Some(c)) => DamageRect {
                    x: c[0],
                    y: c[1],
                    width: c[2],
                    height: c[3],
                },
                (None, None) => return patched,
            });
        }
        patched
    }
}

/// Reconcile a display-list stack against its target chain
/// (M6 Phase 2, Rec 5/6): pop down to the common prefix with the
/// stack, then push the rest. The stack and the target hold the
/// same kind of entry (a page-space clip, or an opacity-group
/// key), and `push`/`pop` build the list items for one entry.
fn reconcile_stack<T: PartialEq + Clone, F, G>(
    list: &mut DisplayList,
    stack: &mut Vec<T>,
    target: &[T],
    push: F,
    pop: G,
) where
    F: Fn(&T) -> DisplayItem,
    G: Fn() -> DisplayItem,
{
    let mut common = 0;
    while common < stack.len()
        && common < target.len()
        && stack[common] == target[common]
    {
        common += 1;
    }
    while stack.len() > common {
        stack.pop();
        list.0.push(pop());
    }
    for entry in &target[common..] {
        list.0.push(push(entry));
        stack.push(entry.clone());
    }
}

/// The axis-aligned bounding box of a rect's corners mapped
/// through a transform (M6 Phase 2, Rec 7): exact for
/// translate/scale, an over-approximation for rotate/skew (the
/// AABB of the rotated AABB is bigger than the rotated shape),
/// which is safe for damage and clip rects. Shared with the
/// renderer, which transforms SDF fields the same way.
pub(crate) fn transform_rect(rect: [f32; 4], matrix: &Affine) -> [f32; 4] {
    let [x, y, w, h] = rect;
    let corners = [
        matrix.apply(x, y),
        matrix.apply(x + w, y),
        matrix.apply(x, y + h),
        matrix.apply(x + w, y + h),
    ];
    let min_x = corners.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|c| c.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
    let max_y = corners
        .iter()
        .map(|c| c.1)
        .fold(f32::NEG_INFINITY, f32::max);
    [min_x, min_y, max_x - min_x, max_y - min_y]
}

/// Scale corner radii by a transform's per-axis stretch (M6
/// Phase 2, Rec 5): the x-radius scales with the x-axis stretch
/// `|(a, b)|`, the y-radius with `|(c, d)|`. Exact for
/// translate/scale; an approximation for rotate/skew. Shared with
/// the renderer, which scales SDF radii the same way.
pub(crate) fn scale_radii(radii: ResolvedCorners, matrix: &Affine) -> ResolvedCorners {
    let sx = (matrix.a * matrix.a + matrix.b * matrix.b).sqrt();
    let sy = (matrix.c * matrix.c + matrix.d * matrix.d).sqrt();
    let scale = |(rx, ry): (f32, f32)| (rx * sx, ry * sy);
    ResolvedCorners {
        top_left: scale(radii.top_left),
        top_right: scale(radii.top_right),
        bottom_right: scale(radii.bottom_right),
        bottom_left: scale(radii.bottom_left),
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

fn check_radii(radii: ResolvedCorners) -> Result<(), ValidationError> {
    for (rx, ry) in [
        radii.top_left,
        radii.top_right,
        radii.bottom_right,
        radii.bottom_left,
    ] {
        check_size(rx)?;
        check_size(ry)?;
    }
    Ok(())
}

/// One resolved shadow layer as a display item, for `box`'s border
/// box. The renderer expands the rect by `spread` and translates it
/// by the offsets; the item itself keeps the box's own rect and
/// radii.
fn shadow_item(bx: &pluto_layout::LayoutBox, shadow: ResolvedShadow) -> DisplayItem {
    DisplayItem::BoxShadow {
        x: bx.x,
        y: bx.y,
        width: bx.width,
        height: bx.height,
        radii: bx.radii,
        offset_x: shadow.offset_x,
        offset_y: shadow.offset_y,
        blur: shadow.blur,
        spread: shadow.spread,
        color: shadow.color,
        inset: shadow.inset,
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

/// Resolve one [`Dim`] against a base length (a box side, or the
/// `box - image` free space for percentage positions) and the
/// viewport.
fn resolve_dim(dim: Dim, base: f32, viewport: (f32, f32)) -> f32 {
    match dim {
        Dim::Px(v) => v,
        Dim::Pct(f) => f * base,
        Dim::Vh(v) => v / 100.0 * viewport.1,
        Dim::Vw(v) => v / 100.0 * viewport.0,
    }
}

/// Background-image tile rects for one box, in page coordinates
/// (Stage 11 Rec 3): `background-size` (cover / contain / explicit
/// lengths / intrinsic) decides the drawn size, `background-position`
/// the offset of the first tile, `background-repeat` whether and
/// how it tiles. Multiple backgrounds are staged (only the first
/// image is modelled — layout keeps one `background_image` per
/// box). The tile count is capped at 1024 so a 1px tile on a huge
/// box cannot blow the display list up; the clip in `build` hides
/// the seam either way.
fn background_tiles(bx: &LayoutBox, viewport: (f32, f32)) -> Vec<[f32; 4]> {
    const MAX_TILES: usize = 1024;
    let Some((iw, ih)) = bx.background_image_size else {
        return Vec::new();
    };
    let (iw, ih) = (iw as f32, ih as f32);
    if iw <= 0.0 || ih <= 0.0 || bx.width <= 0.0 || bx.height <= 0.0 {
        return Vec::new();
    }
    let (bw, bh) = (bx.width, bx.height);
    let (dw, dh) = match bx.style.background_size {
        BackgroundSize::Cover => {
            let s = (bw / iw).max(bh / ih);
            (iw * s, ih * s)
        }
        BackgroundSize::Contain => {
            let s = (bw / iw).min(bh / ih);
            (iw * s, ih * s)
        }
        BackgroundSize::Lengths(w, h) => {
            let w = w.map(|d| resolve_dim(d, bw, viewport));
            let h = h.map(|d| resolve_dim(d, bh, viewport));
            match (w, h) {
                (Some(w), Some(h)) => (w, h),
                (Some(w), None) => (w, ih * (w / iw)),
                (None, Some(h)) => (iw * (h / ih), h),
                (None, None) => (iw, ih),
            }
        }
        BackgroundSize::Auto => (iw, ih),
    };
    if dw <= 0.0 || dh <= 0.0 {
        return Vec::new();
    }
    // Percentage positions resolve against the free space
    // (box - image), per CSS; keywords pick an edge or the centre.
    let pos = |p: BackgroundPosition, free: f32| -> f32 {
        match p {
            BackgroundPosition::Start => 0.0,
            BackgroundPosition::Center => free / 2.0,
            BackgroundPosition::End => free,
            BackgroundPosition::Length(d) => resolve_dim(d, free, viewport),
        }
    };
    let ox = pos(bx.style.background_position.0, bw - dw);
    let oy = pos(bx.style.background_position.1, bh - dh);
    let tile = |x: f32, y: f32| [bx.x + x, bx.y + y, dw, dh];
    match bx.style.background_repeat {
        BackgroundRepeat::NoRepeat => vec![tile(ox, oy)],
        BackgroundRepeat::RepeatX => {
            let mut out = Vec::new();
            let mut k = (-ox / dw).floor() as i32;
            while out.len() < MAX_TILES {
                let x = ox + k as f32 * dw;
                if x >= bw {
                    break;
                }
                if x + dw > 0.0 {
                    out.push(tile(x, oy));
                }
                k += 1;
            }
            out
        }
        BackgroundRepeat::RepeatY => {
            let mut out = Vec::new();
            let mut k = (-oy / dh).floor() as i32;
            while out.len() < MAX_TILES {
                let y = oy + k as f32 * dh;
                if y >= bh {
                    break;
                }
                if y + dh > 0.0 {
                    out.push(tile(ox, y));
                }
                k += 1;
            }
            out
        }
        BackgroundRepeat::RepeatXY(repeat_x, repeat_y) => {
            // Two-value form (M6): each axis tiles independently
            // — `repeat no-repeat` tiles x only, `no-repeat
            // repeat` y only, `no-repeat no-repeat` one tile.
            let mut out = Vec::new();
            let mut ky = if repeat_y {
                (-oy / dh).floor() as i32
            } else {
                0
            };
            loop {
                let y = if repeat_y { oy + ky as f32 * dh } else { oy };
                if repeat_y && y >= bh {
                    break;
                }
                let mut kx = if repeat_x {
                    (-ox / dw).floor() as i32
                } else {
                    0
                };
                loop {
                    let x = if repeat_x { ox + kx as f32 * dw } else { ox };
                    if repeat_x && x >= bw {
                        break;
                    }
                    if x + dw > 0.0 && y + dh > 0.0 {
                        out.push(tile(x, y));
                    }
                    if out.len() >= MAX_TILES {
                        return out;
                    }
                    if !repeat_x {
                        break;
                    }
                    kx += 1;
                }
                if !repeat_y {
                    break;
                }
                ky += 1;
            }
            out
        }
        BackgroundRepeat::Repeat => {
            let mut out = Vec::new();
            let mut ky = (-oy / dh).floor() as i32;
            while out.len() < MAX_TILES {
                let y = oy + ky as f32 * dh;
                if y >= bh {
                    break;
                }
                let mut kx = (-ox / dw).floor() as i32;
                while out.len() < MAX_TILES {
                    let x = ox + kx as f32 * dw;
                    if x >= bw {
                        break;
                    }
                    if x + dw > 0.0 && y + dh > 0.0 {
                        out.push(tile(x, y));
                    }
                    kx += 1;
                }
                ky += 1;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::Stylesheet;
    use pluto_html::parse as parse_html;
    use pluto_layout::{ElementState, LayoutTree, Theme};

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
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0, Theme::Light).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
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
        let tree = LayoutTree::build(&dom, &sheet, 1024.0, 768.0, Theme::Light).expect("layout builds");
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

    /// Build a layout + display list from inline HTML/CSS, for
    /// the M6 Phase 2 paint tests (viewport 800x600).
    fn build_list(html: &str, css: &str) -> (pluto_html::Dom, LayoutTree, DisplayList) {
        let dom = parse_html(html);
        let sheet = Stylesheet::parse(css);
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light)
            .expect("layout builds");
        let list = DisplayList::build(&tree);
        (dom, tree, list)
    }

    /// First element with this class, depth-first (mirrors the
    /// layout tests' helper).
    fn find_by_class(
        dom: &pluto_html::Dom,
        id: pluto_html::NodeId,
        class: &str,
    ) -> Option<pluto_html::NodeId> {
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

    /// The background colours of the `FillRect` items in list
    /// order, skipping item 0 (the canvas).
    fn fill_colors(list: &DisplayList) -> Vec<[f32; 4]> {
        list.0
            .iter()
            .skip(1)
            .filter_map(|item| match item {
                DisplayItem::FillRect { color, .. } => Some(*color),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn build_paints_in_z_index_order_not_document_order() {
        // M6 Phase 2 (Rec 4): three overlapping siblings in
        // document order a, b, c with z-index 3, 1, 2 paint
        // b, c, a — the list is Appendix E order, not document
        // order.
        let (_dom, _tree, list) = build_list(
            r#"<div class="wrap"><div class="a">A</div><div class="b">B</div><div class="c">C</div></div>"#,
            ".wrap { position: relative; width: 200px; height: 200px; } \
             .a { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 3; background-color: #ff0000; } \
             .b { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 1; background-color: #00ff00; } \
             .c { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: 2; background-color: #0000ff; }",
        );
        assert_eq!(
            fill_colors(&list),
            vec![
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            "paint order is b (green), c (blue), a (red)"
        );
    }

    #[test]
    fn build_paints_negative_z_after_parent_background_before_content() {
        // M6 Phase 2 (Rec 4): a negative-z child paints after
        // the stacking-context root's own background (Appendix E
        // step 1) but before the root's in-flow content (step 3)
        // — behind the content, not behind the parent.
        let (_dom, _tree, list) = build_list(
            r#"<div class="parent">P<div class="neg"></div></div>"#,
            ".parent { position: relative; width: 200px; height: 200px; z-index: 0; background-color: #ff0000; } \
             .neg { position: absolute; left: 0; top: 0; width: 100px; height: 100px; z-index: -1; background-color: #00ff00; }",
        );
        let red = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::FillRect { color, .. } if *color == [1.0, 0.0, 0.0, 1.0]))
            .expect("parent background");
        let green = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::FillRect { color, .. } if *color == [0.0, 1.0, 0.0, 1.0]))
            .expect("negative child background");
        let glyphs = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::Glyphs { .. }))
            .expect("parent text glyphs");
        assert!(red < green && green < glyphs, "red {red} < green {green} < glyphs {glyphs}");
    }

    #[test]
    fn build_emits_push_transform_with_page_matrix() {
        // M6 Phase 2 (Rec 7/8): a rotated box's items sit
        // between a `PushTransform` carrying the box's page
        // transform (verbatim from the layout tree) and its
        // `PopTransform`, and the transformed bounds are the
        // rotated AABB (~141px for a 100px box at 45 degrees).
        let (dom, tree, list) = build_list(
            r#"<div class="spin"></div>"#,
            ".spin { position: absolute; left: 100px; top: 100px; width: 100px; height: 100px; transform: rotate(45deg); background-color: #ff0000; }",
        );
        let spin = find_by_class(&dom, dom.root().unwrap(), "spin").unwrap();
        let spin_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin))
            .unwrap();
        let page_t = tree.page_transforms()[spin_idx];
        assert_ne!(page_t, Affine::identity());
        let push = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::PushTransform { .. }))
            .expect("PushTransform");
        let pop = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::PopTransform))
            .expect("PopTransform");
        let fill = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::FillRect { color, .. } if *color == [1.0, 0.0, 0.0, 1.0]))
            .expect("spin background");
        assert!(push < fill && fill < pop, "background sits inside the transform pair");
        match &list.0[push] {
            DisplayItem::PushTransform { id, matrix } => {
                assert_eq!(*id, LayoutTree::dom_key(spin));
                assert_eq!(*matrix, page_t, "the list carries the page transform verbatim");
            }
            other => panic!("expected PushTransform, got {other:?}"),
        }
        let bounds = list.0[fill]
            .transformed_bounds(&page_t)
            .expect("fill has bounds");
        assert!((bounds[2] - 141.42).abs() < 0.1, "rotated width {}", bounds[2]);
        assert!((bounds[3] - 141.42).abs() < 0.1, "rotated height {}", bounds[3]);
        // Rec 8: `transform-origin` resolves against the border
        // box — rotating about the top-left corner (0% 0%)
        // leaves the top-left fixed.
        let (dom2, tree2, _list2) = build_list(
            r#"<div class="spin"></div>"#,
            ".spin { position: absolute; left: 100px; top: 100px; width: 100px; height: 100px; transform: rotate(45deg); transform-origin: 0% 0%; background-color: #ff0000; }",
        );
        let spin2 = find_by_class(&dom2, dom2.root().unwrap(), "spin").unwrap();
        let spin_idx2 = tree2
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin2))
            .unwrap();
        let spin_box = &tree2.boxes[spin_idx2];
        let t2 = tree2.page_transforms()[spin_idx2];
        let (px, py) = t2.apply(spin_box.x, spin_box.y);
        assert!((px - spin_box.x).abs() < 0.01 && (py - spin_box.y).abs() < 0.01,
            "top-left stays put with origin 0% 0%: ({px}, {py}) vs ({}, {})",
            spin_box.x, spin_box.y);
    }

    #[test]
    fn build_groups_opacity_with_push_pop() {
        // M6 Phase 2 (Rec 6): nested translucent stacking
        // contexts nest their `PushOpacity` groups — the outer
        // group's items wrap the inner group's, and the inner
        // background paints inside both.
        let (dom, tree, list) = build_list(
            r#"<div class="outer"><div class="inner"></div></div>"#,
            ".outer { width: 200px; height: 200px; opacity: 0.5; background-color: #ff0000; } \
             .inner { width: 100px; height: 100px; opacity: 0.5; background-color: #00ff00; }",
        );
        let root = dom.root().unwrap();
        let outer = find_by_class(&dom, root, "outer").unwrap();
        let inner = find_by_class(&dom, root, "inner").unwrap();
        let outer_key = LayoutTree::dom_key(outer);
        let inner_key = LayoutTree::dom_key(inner);
        let _ = tree;
        let pushes: Vec<(u64, f32)> = list
            .0
            .iter()
            .filter_map(|i| match i {
                DisplayItem::PushOpacity { id, opacity } => Some((*id, *opacity)),
                _ => None,
            })
            .collect();
        assert_eq!(
            pushes,
            vec![(outer_key, 0.5), (inner_key, 0.5)],
            "outer group wraps the inner group"
        );
        let pops = list
            .0
            .iter()
            .filter(|i| matches!(i, DisplayItem::PopOpacity))
            .count();
        assert_eq!(pops, 2, "both groups close");
        let red = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::FillRect { color, .. } if *color == [1.0, 0.0, 0.0, 1.0]))
            .unwrap();
        let green = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::FillRect { color, .. } if *color == [0.0, 1.0, 0.0, 1.0]))
            .unwrap();
        let first_push = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::PushOpacity { .. }))
            .unwrap();
        let last_pop = list
            .0
            .iter()
            .rposition(|i| matches!(i, DisplayItem::PopOpacity))
            .unwrap();
        assert!(
            first_push < red && red < green && green < last_pop,
            "both backgrounds sit inside the groups"
        );
    }

    #[test]
    fn apply_compositing_update_patches_and_expands_damage() {
        // M6 Phase 2 (Rec 9): a `CompositingUpdate` patches the
        // matching `PushTransform` in place (no list rebuild) and
        // expands the damage back to the canvas; an update for
        // an id the list does not carry patches nothing.
        let (dom, tree, mut list) = build_list(
            r#"<div class="spin"></div>"#,
            ".spin { width: 100px; height: 100px; transform: translateX(10px); background-color: #ff0000; }",
        );
        let spin = find_by_class(&dom, dom.root().unwrap(), "spin").unwrap();
        let key = LayoutTree::dom_key(spin);
        let (_dom2, tree2, _list2) = build_list(
            r#"<div class="spin"></div>"#,
            ".spin { width: 100px; height: 100px; transform: translateX(30px); background-color: #ff0000; }",
        );
        let spin_idx = tree
            .boxes
            .iter()
            .position(|b| b.dom == Some(spin))
            .unwrap();
        let new_matrix = tree2.page_transforms()[spin_idx];
        // Narrow the damage first, so the expansion is observable:
        // a patched list must invalidate the whole canvas, not
        // just the old (small) damage.
        list.1 = Some(DamageRect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        });
        let patched = list.apply_compositing_update(&[CompositedUpdate {
            id: key,
            matrix: new_matrix,
            opacity: 1.0,
        }]);
        assert_eq!(patched, 1, "one PushTransform patched");
        match list
            .0
            .iter()
            .find(|i| matches!(i, DisplayItem::PushTransform { .. }))
        {
            Some(DisplayItem::PushTransform { matrix, .. }) => {
                assert_eq!(*matrix, new_matrix);
            }
            other => panic!("expected a patched PushTransform, got {other:?}"),
        }
        assert_eq!(
            list.1,
            Some(DamageRect {
                x: 0.0,
                y: 0.0,
                width: 800.0,
                height: 600.0,
            }),
            "damage expands to the canvas"
        );
        // An unknown id patches nothing and leaves the list alone.
        let before = list.clone();
        assert_eq!(
            list.apply_compositing_update(&[CompositedUpdate {
                id: 0xdeadbeef,
                matrix: Affine::identity(),
                opacity: 1.0,
            }]),
            0
        );
        assert_eq!(list, before);
    }

    #[test]
    fn display_list_with_transform_and_opacity_round_trips() {
        // M6 Phase 2 (Rec 6/7): the new items survive the wire
        // (postcard discriminants appended at the end) and the
        // decoded list validates.
        let (_dom, _tree, list) = build_list(
            r#"<div class="ghost">G</div>"#,
            ".ghost { width: 100px; height: 100px; opacity: 0.5; transform: rotate(45deg); background-color: #ff0000; }",
        );
        assert!(list
            .0
            .iter()
            .any(|i| matches!(i, DisplayItem::PushTransform { .. })));
        assert!(list
            .0
            .iter()
            .any(|i| matches!(i, DisplayItem::PushOpacity { .. })));
        let back = DisplayList::from_bytes(&list.to_bytes()).expect("list decodes");
        assert_eq!(back, list, "round-trip must be lossless");
        back.validate(&ResourceTable::new())
            .expect("decoded list validates");
    }

    #[test]
    fn fixed_subtree_wraps_in_push_fixed_and_excludes_damage() {
        // M6 Phase 3 (Rec 1): a viewport-fixed subtree paints
        // inside one `PushFixed`/`PopFixed` scope (its background
        // rect sits between them), and its bounds never enter the
        // page damage — a 2000px fixed box on a short page must
        // not inflate the damage (or the page's maximum scroll)
        // past the viewport.
        let (_dom, _tree, list) = build_list(
            r#"<div class="fixed"></div><div class="content"></div>"#,
            ".fixed { position: fixed; top: 0px; left: 0px; width: 100px; height: 2000px; background-color: #ff0000; } \
             .content { height: 100px; background-color: #00ff00; }",
        );
        let push = list
            .0
            .iter()
            .position(|i| matches!(i, DisplayItem::PushFixed))
            .expect("a PushFixed opens the fixed scope");
        let pop = list
            .0
            .iter()
            .rposition(|i| matches!(i, DisplayItem::PopFixed))
            .expect("a PopFixed closes it");
        assert!(push < pop, "scope is balanced");
        let red = list
            .0
            .iter()
            .position(|i| {
                matches!(i, DisplayItem::FillRect { color, .. } if *color == [1.0, 0.0, 0.0, 1.0])
            })
            .expect("fixed background rect");
        assert!(
            push < red && red < pop,
            "fixed background paints inside the scope"
        );
        assert_eq!(
            list.1,
            Some(DamageRect {
                x: 0.0,
                y: 0.0,
                width: 800.0,
                height: 600.0,
            }),
            "damage is the viewport, not the 2000px fixed box"
        );
        // The new items survive the wire and validate.
        let back = DisplayList::from_bytes(&list.to_bytes()).expect("list decodes");
        assert_eq!(back, list, "round-trip must be lossless");
        back.validate(&ResourceTable::new())
            .expect("decoded list validates");
    }

    #[test]
    fn build_with_svg_emits_vector_path_for_linear_gradient_background() {
        // Stage 15 Rec 7: a linear-gradient background is a
        // `VectorPath` (one strip per stop segment — exact), with
        // no raster resource attached.
        let dom = parse_html("<html><body><div></div></body></html>");
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: linear-gradient(to right, red, blue); }",
        );
        let tree = LayoutTree::build_with_state(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let mut cache = SvgRasterCache::new();
        let (list, blobs) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert!(blobs.is_empty(), "linear backgrounds rasterise nothing");
        let vectors: Vec<_> = list
            .0
            .iter()
            .filter(|i| matches!(i, DisplayItem::VectorPath { .. }))
            .collect();
        assert_eq!(vectors.len(), 1, "one gradient background = one VectorPath");
        if let DisplayItem::VectorPath { vertices, bounds } = vectors[0] {
            assert!(!vertices.is_empty());
            assert_eq!(vertices.len() % 3, 0, "vertices come in triples");
            assert_eq!(bounds[2], 100.0);
            assert_eq!(bounds[3], 100.0);
        }
        list.validate(&ResourceTable::new())
            .expect("list validates with no resources");
    }

    #[test]
    fn build_with_svg_emits_blob_for_radial_gradient_background() {
        // Stage 15 Rec 7: a radial-gradient background is one
        // Blob (per-pixel evaluation) plus one fresh resource —
        // and a second build with the same cache reuses it (same
        // Blob, no new resource: the id is deterministic).
        let dom = parse_html("<html><body><div></div></body></html>");
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: radial-gradient(circle, white, black); }",
        );
        let tree = LayoutTree::build_with_state(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let mut cache = SvgRasterCache::new();
        let (list, blobs) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert_eq!(blobs.len(), 1, "one fresh raster");
        let (id, resource) = &blobs[0];
        assert!(matches!(
            resource,
            Resource::Image {
                width: 100,
                height: 100,
                ..
            }
        ));
        assert!(
            list.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if resource == id
            )),
            "the Blob names the fresh resource"
        );
        assert!(
            !list.0.iter().any(|i| matches!(i, DisplayItem::VectorPath { .. })),
            "radial backgrounds never take the vector route"
        );
        let mut table = ResourceTable::new();
        table.insert(*id, resource.clone());
        list.validate(&table).expect("list validates");
        let (list2, blobs2) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert!(blobs2.is_empty(), "cache hit returns nothing new");
        assert!(
            list2.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if resource == id
            )),
            "the second build names the same Blob id"
        );
    }

    #[test]
    fn build_with_svg_emits_vector_paths_for_inline_svg() {
        // Stage 15 Rec 3/4: a solid inline SVG is `VectorPath`
        // items (one per shape), clipped to the svg box, with no
        // raster resource attached.
        let dom = parse_html(
            "<html><body><svg width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><rect x=\"10\" y=\"10\" width=\"80\" height=\"80\" fill=\"red\"/></svg></body></html>",
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let mut cache = SvgRasterCache::new();
        let (list, blobs) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert!(blobs.is_empty(), "solid SVGs rasterise nothing");
        let vectors: Vec<_> = list
            .0
            .iter()
            .filter(|i| matches!(i, DisplayItem::VectorPath { .. }))
            .collect();
        assert_eq!(vectors.len(), 1, "one rect shape = one VectorPath");
        if let DisplayItem::VectorPath { vertices, bounds } = vectors[0] {
            assert!(!vertices.is_empty());
            assert_eq!(vertices.len() % 3, 0, "vertices come in triples");
            assert_eq!(bounds[2], 100.0, "bounds are the svg box");
            assert_eq!(bounds[3], 100.0);
        }
        // Plain build() paints nothing for the svg (no DOM/cache).
        let plain = DisplayList::build(&tree);
        assert!(
            !plain.0.iter().any(|i| matches!(i, DisplayItem::VectorPath { .. })),
            "plain build() has no vector items"
        );
        list.validate(&ResourceTable::new())
            .expect("list validates with no resources");
    }

    #[test]
    fn build_with_svg_emits_blob_for_radial_gradient_svg() {
        // Stage 15 Rec 5: an inline SVG whose fill is a radial
        // gradient is one Blob plus one fresh resource — the
        // vector route cannot express it.
        let dom = parse_html(
            "<html><body><svg width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><defs><radialGradient id=\"rg\"><stop offset=\"0\" stop-color=\"red\"/><stop offset=\"1\" stop-color=\"white\"/></radialGradient></defs><rect x=\"0\" y=\"0\" width=\"100\" height=\"100\" fill=\"url(#rg)\"/></svg></body></html>",
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let mut cache = SvgRasterCache::new();
        let (list, blobs) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert_eq!(blobs.len(), 1, "one fresh raster");
        let (id, resource) = &blobs[0];
        assert!(matches!(
            resource,
            Resource::Image {
                width: 100,
                height: 100,
                ..
            }
        ));
        assert!(
            list.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if resource == id
            )),
            "the Blob names the fresh resource"
        );
        assert!(
            !list.0.iter().any(|i| matches!(i, DisplayItem::VectorPath { .. })),
            "a radial svg is raster-only"
        );
        let mut table = ResourceTable::new();
        table.insert(*id, resource.clone());
        list.validate(&table).expect("list validates");
        let (list2, blobs2) = DisplayList::build_with_svg(&tree, &dom, &mut cache);
        assert!(blobs2.is_empty(), "cache hit returns nothing new");
        assert!(
            list2.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if resource == id
            )),
            "the second build names the same Blob id"
        );
    }

    #[test]
    fn build_with_svg_and_canvas_emits_canvas_blob_and_svg_image() {
        // Stage 15 (M7 Phase 3, Rec 8/10): a live canvas paints
        // its bitmap as a Blob (fresh pixels ride `new_blobs`
        // once), and a resolved `<svg><image>` paints as an
        // `Image` item at its rect.
        let dom = parse_html(
            "<html><body><svg width=\"100\" height=\"100\" viewBox=\"0 0 100 100\"><image href=\"photo.png\" x=\"10\" y=\"10\" width=\"20\" height=\"20\"/></svg><canvas width=\"4\" height=\"4\"></canvas></body></html>",
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            Theme::Light,
        )
        .expect("layout builds");
        let canvas_id = dom
            .all_node_ids()
            .into_iter()
            .find(|&id| dom.tag(id) == Some("canvas"))
            .expect("canvas in DOM");
        let image_id = dom
            .all_node_ids()
            .into_iter()
            .find(|&id| dom.tag(id) == Some("image"))
            .expect("svg image in DOM");
        let mut svg_images = HashMap::new();
        svg_images.insert(image_id, ResourceId(7));
        let fresh = Resource::Image {
            width: 4,
            height: 4,
            rgba: vec![255u8; 64],
        };
        let mut cache = SvgRasterCache::new();
        let (list, blobs) = DisplayList::build_with_svg_and_canvas(
            &tree,
            &dom,
            &mut cache,
            &svg_images,
            &[canvas_id],
            &[(canvas_id, fresh.clone())],
        );
        assert_eq!(blobs.len(), 1, "one fresh canvas bitmap");
        assert_eq!(blobs[0].0, crate::canvas::canvas_blob_id(canvas_id));
        assert!(
            list.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if *resource == crate::canvas::canvas_blob_id(canvas_id)
            )),
            "canvas Blob names the fresh bitmap"
        );
        assert!(
            list.0.iter().any(|i| matches!(
                i,
                DisplayItem::Image { resource, x, y, width, height }
                    if *resource == ResourceId(7) && (*x, *y, *width, *height) == (18.0, 18.0, 20.0, 20.0)
            )),
            "svg image paints at its rect (svg at 8,8 + image at 10,10)"
        );
        // A second build with the same live canvas but no fresh
        // bitmap still names the Blob, and announces nothing.
        let (list2, blobs2) = DisplayList::build_with_svg_and_canvas(
            &tree,
            &dom,
            &mut cache,
            &svg_images,
            &[canvas_id],
            &[],
        );
        assert!(blobs2.is_empty(), "unchanged canvas re-sends nothing");
        assert!(
            list2.0.iter().any(|i| matches!(
                i,
                DisplayItem::Blob { resource, .. } if *resource == crate::canvas::canvas_blob_id(canvas_id)
            )),
            "unchanged canvas still paints its Blob"
        );
    }

    #[test]
    fn validate_rejects_bad_opacity_and_transform() {
        // M6 Phase 2 (Rec 6/7): the new items are validated —
        // an opacity outside 0..=1 and a non-finite transform
        // component are both rejected, not silently rendered.
        let resources = ResourceTable::new();
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushOpacity {
            id: 1,
            opacity: 2.0,
        });
        assert!(matches!(
            list.validate(&resources),
            Err(ValidationError::BadOpacity(_))
        ));
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushTransform {
            id: 1,
            matrix: Affine {
                a: f32::NAN,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                e: 0.0,
                f: 0.0,
            },
        });
        assert!(matches!(
            list.validate(&resources),
            Err(ValidationError::BadTransform(_))
        ));
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

    #[test]
    fn rounded_box_emits_rounded_rect_and_shadows_in_order() {
        // Stage 8 Rec 4: a rounded box with an outer and an inset
        // shadow paints outer shadow -> rounded background -> inset
        // shadow, and the built list carries damage.
        let dom = parse_html(
            "<html><head></head><body><div class=\"r\">x</div></body></html>",
        );
        let sheet = Stylesheet::parse(
            ".r { width: 100px; height: 50px; background-color: #ff0000; border-radius: 8px; \
             box-shadow: 2px 3px 0 #000000, inset 0 0 4px #00ff00; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let list = DisplayList::build(&tree);
        let pos = |pred: &dyn Fn(&DisplayItem) -> bool| {
            list.0.iter().position(pred).expect("item present")
        };
        let outer = pos(&|i| matches!(i, DisplayItem::BoxShadow { inset: false, .. }));
        let rounded = pos(&|i| matches!(i, DisplayItem::RoundedRect { .. }));
        let inset = pos(&|i| matches!(i, DisplayItem::BoxShadow { inset: true, .. }));
        assert!(outer < rounded && rounded < inset, "outer -> bg -> inset");
        match &list.0[rounded] {
            DisplayItem::RoundedRect { radii, color, .. } => {
                assert_eq!(radii.top_left, (8.0, 8.0));
                assert!((color[0] - 1.0).abs() < 0.001);
            }
            other => panic!("expected a rounded rect, got {other:?}"),
        }
        match &list.0[outer] {
            DisplayItem::BoxShadow {
                offset_x,
                offset_y,
                blur,
                ..
            } => assert_eq!((*offset_x, *offset_y, *blur), (2.0, 3.0, 0.0)),
            other => panic!("expected an outer shadow, got {other:?}"),
        }
        // Damage covers the viewport (a full rebuild repaints it).
        let damage = list.1.expect("built lists carry damage");
        assert_eq!((damage.x, damage.y), (0.0, 0.0));
        assert!(damage.width >= 800.0 && damage.height >= 600.0);
        list.validate(&ResourceTable::new())
            .expect("rounded + shadow list is valid");
    }

    #[test]
    fn scroll_container_with_radius_emits_rounded_clip() {
        // Stage 8 Rec 3: the clip a rounded scroll container pushes
        // carries the container's resolved radii.
        let dom = parse_html(
            "<html><head></head><body><div class=\"scroll\"><div class=\"tall\">hello</div></div></body></html>",
        );
        let sheet = Stylesheet::parse(
            ".scroll { overflow: hidden; height: 50px; width: 200px; border-radius: 10px; } \
             .tall { height: 200px; }",
        );
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0, Theme::Light).expect("layout builds");
        let list = DisplayList::build(&tree);
        let radii = list.0.iter().find_map(|item| match item {
            DisplayItem::PushClip { radii, .. } => Some(*radii),
            _ => None,
        });
        assert!(
            matches!(radii, Some(Some(r)) if r.top_left == (10.0, 10.0)),
            "clip carries the 10px radii, got {radii:?}"
        );
        assert!(list.1.is_some(), "built lists carry damage");
        list.validate(&ResourceTable::new())
            .expect("list is valid");
    }

    #[test]
    fn damage_rect_unions_and_push_clip_carries_radii() {
        let a = DamageRect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
        };
        let b = DamageRect {
            x: 5.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
        };
        assert_eq!(
            a.union(b),
            DamageRect {
                x: 0.0,
                y: 0.0,
                width: 15.0,
                height: 15.0,
            }
        );
        assert!((a.area() - 100.0).abs() < f32::EPSILON);
        assert!(DamageRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 5.0,
        }
        .is_empty());
        let radii = ResolvedCorners {
            top_left: (8.0, 8.0),
            top_right: (8.0, 8.0),
            bottom_right: (8.0, 8.0),
            bottom_left: (8.0, 8.0),
        };
        let mut list = DisplayList::new();
        list.push(DisplayItem::PushClip {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            radii: Some(radii),
        });
        list.push(DisplayItem::PopClip);
        let back = DisplayList::from_bytes(&list.to_bytes()).expect("list decodes");
        assert_eq!(back, list);
        back.validate(&ResourceTable::new())
            .expect("clip list is valid");
    }

    #[test]
    fn validation_rejects_shadow_blur_above_the_cap() {
        // Stage 8 Rec 4: blur above 64px is a validation error, not
        // a silent clamp — the shell refuses the list.
        let resources = ResourceTable::new();
        let mut list = small_list();
        list.push(DisplayItem::BoxShadow {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            radii: ResolvedCorners::ZERO,
            offset_x: 0.0,
            offset_y: 0.0,
            blur: 100.0,
            spread: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
            inset: false,
        });
        assert_eq!(
            list.validate(&resources),
            Err(ValidationError::ShadowBlurTooLarge(100.0))
        );
        // Exactly at the cap validates.
        if let Some(DisplayItem::BoxShadow { blur, .. }) = list.0.last_mut() {
            *blur = MAX_BOX_SHADOW_BLUR;
        }
        list.validate(&resources).expect("blur at the cap is valid");
    }

    // ---- Stage 11 (M3): image items in the display list ----

    fn find_tag(dom: &pluto_html::Dom, tag: &str) -> pluto_html::NodeId {
        fn walk(
            dom: &pluto_html::Dom,
            id: pluto_html::NodeId,
            tag: &str,
        ) -> Option<pluto_html::NodeId> {
            if dom.tag(id) == Some(tag) {
                return Some(id);
            }
            dom.children(id).iter().find_map(|&c| walk(dom, c, tag))
        }
        walk(dom, dom.root().unwrap(), tag).expect("tag in dom")
    }

    fn image_items(list: &DisplayList) -> Vec<(u32, f32, f32, f32, f32)> {
        list.0
            .iter()
            .filter_map(|item| match item {
                DisplayItem::Image {
                    resource,
                    x,
                    y,
                    width,
                    height,
                } => Some((resource.0, *x, *y, *width, *height)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn display_list_emits_img_image_item() {
        use pluto_layout::{ElementState, ImageInfo, ImageMap, Theme};
        let dom = parse_html("<html><body><img src=\"a.png\"></body></html>");
        let mut images = ImageMap::default();
        images.by_node.insert(
            find_tag(&dom, "img"),
            ImageInfo {
                resource: 5,
                width: 400,
                height: 100,
            },
        );
        let sheet = Stylesheet::parse("");
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let list = DisplayList::build(&tree);
        let items = image_items(&list);
        assert_eq!(items.len(), 1, "one image item: {items:?}");
        assert_eq!(items[0].0, 5);
        assert_eq!((items[0].3, items[0].4), (400.0, 100.0));
        // The list validates against a table holding the pixels.
        let mut resources = ResourceTable::new();
        resources.insert(
            ResourceId(5),
            Resource::Image {
                width: 400,
                height: 100,
                rgba: vec![0u8; 400 * 100 * 4],
            },
        );
        list.validate(&resources).expect("image resolves");
    }

    #[test]
    fn display_list_emits_background_image_cover_and_centered() {
        use pluto_layout::{ElementState, ImageInfo, ImageMap, Theme};
        // Cover: 800x600 into a 100x100 box scales to 133.3x100.
        let dom = parse_html("<html><body><div></div></body></html>");
        let mut images = ImageMap::default();
        images.backgrounds.insert(
            "bg.png".to_string(),
            ImageInfo {
                resource: 9,
                width: 800,
                height: 600,
            },
        );
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: url(bg.png); background-size: cover; background-repeat: no-repeat; }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let list = DisplayList::build(&tree);
        let items = image_items(&list);
        assert_eq!(items.len(), 1, "one cover tile: {items:?}");
        assert_eq!(items[0].0, 9);
        assert!((items[0].3 - 800.0 / 6.0).abs() < 0.01, "cover width: {:?}", items[0]);
        assert_eq!(items[0].4, 100.0);
        // The background tile is wrapped in a PushClip/PopClip pair
        // (cover overflows the box horizontally).
        assert!(
            list.0.iter().any(|i| matches!(i, DisplayItem::PushClip { .. })),
            "background clip present"
        );
        assert!(
            list.0.iter().any(|i| matches!(i, DisplayItem::PopClip)),
            "background clip popped"
        );

        // Centred, intrinsic size: 50x50 in a 200x100 box centres
        // at (+75, +25) relative to the box.
        let dom = parse_html("<html><body><div></div></body></html>");
        let mut images = ImageMap::default();
        images.backgrounds.insert(
            "bg.png".to_string(),
            ImageInfo {
                resource: 9,
                width: 50,
                height: 50,
            },
        );
        let sheet = Stylesheet::parse(
            "div { width: 200px; height: 100px; background-image: url(bg.png); background-position: center; background-repeat: no-repeat; }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let div = tree
            .boxes()
            .iter()
            .find(|b| b.dom.is_some_and(|id| dom.tag(id) == Some("div")))
            .expect("div box");
        let list = DisplayList::build(&tree);
        let items = image_items(&list);
        assert_eq!(items.len(), 1, "one centred tile: {items:?}");
        assert!((items[0].1 - (div.x + 75.0)).abs() < 0.01, "centred x: {:?}", items[0]);
        assert!((items[0].2 - (div.y + 25.0)).abs() < 0.01, "centred y: {:?}", items[0]);
        assert_eq!((items[0].3, items[0].4), (50.0, 50.0));
    }

    #[test]
    fn display_list_tiles_repeating_background() {
        use pluto_layout::{ElementState, ImageInfo, ImageMap, Theme};
        // 30x30 tile repeated over a 100x100 box: 4x4 = 16 tiles.
        let dom = parse_html("<html><body><div></div></body></html>");
        let mut images = ImageMap::default();
        images.backgrounds.insert(
            "tile.png".to_string(),
            ImageInfo {
                resource: 4,
                width: 30,
                height: 30,
            },
        );
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: url(tile.png); }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let list = DisplayList::build(&tree);
        let items = image_items(&list);
        assert_eq!(items.len(), 16, "4x4 tiles: {items:?}");
        assert!(items.iter().all(|i| i.0 == 4));
        assert!(items.iter().all(|i| (i.3, i.4) == (30.0, 30.0)));
    }

    #[test]
    fn display_list_tiles_repeat_no_repeat_one_axis_only() {
        use pluto_layout::{ElementState, ImageInfo, ImageMap, Theme};
        // M6 Phase 1: `repeat no-repeat` tiles x only — a 100x100
        // box with a 30x30 tile gives 4 tiles in one row, not 16.
        let dom = parse_html("<html><body><div></div></body></html>");
        let mut images = ImageMap::default();
        images.backgrounds.insert(
            "tile.png".to_string(),
            ImageInfo {
                resource: 4,
                width: 30,
                height: 30,
            },
        );
        let sheet = Stylesheet::parse(
            "div { width: 100px; height: 100px; background-image: url(tile.png); background-repeat: repeat no-repeat; }",
        );
        let tree = LayoutTree::build_with_state_and_images(
            &dom,
            &sheet,
            1024.0,
            768.0,
            &ElementState::default(),
            &images,
            Theme::Light,
        )
        .expect("layout builds");
        let list = DisplayList::build(&tree);
        let items = image_items(&list);
        assert_eq!(items.len(), 4, "one row of tiles: {items:?}");
        assert!(items.iter().all(|i| (i.3, i.4) == (30.0, 30.0)));
        assert!(
            items.iter().all(|i| (i.2 - items[0].2).abs() < 0.01),
            "all tiles share one row: {items:?}"
        );
    }
}
