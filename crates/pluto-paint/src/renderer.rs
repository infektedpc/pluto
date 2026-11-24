//! wgpu renderer for a [`DisplayList`](crate::DisplayList): solid quads
//! plus glyph-atlas text.
//!
//! Five pipelines share one uniform (the viewport size, for
//! pixel-to-clip conversion) plus a clip uniform (the rounded-clip
//! slots, see [`ClipUniformData`]). Rects are opaque quads; rounded
//! rects evaluate the rounded-box SDF for coverage; box shadows use
//! the erf closed-form blur profile of that SDF (sharp branch at
//! blur 0); glyphs sample a 1024x1024 R8 atlas of cosmic-text/Swash
//! alpha masks, tinted by the text color. The atlas is shelf-packed
//! on demand in `prepare`; the same glyph rasterized once stays
//! cached across frames. Color glyphs (emoji) and subpixel-mask
//! glyphs are skipped in M1.
//!
//! Glyph atlases (Stage 8 Rec 11): two atlases, both
//! [`GlyphAtlas`] guillotine packers with generation-stamped LRU
//! eviction — the mask atlas (`R8Unorm`) and a second colour atlas
//! (`Rgba8UnormSrgb`) for colour glyphs (emoji), which used to be
//! skipped. Neither atlas ever evicts a glyph in use by the current
//! frame; both grow their logical size 1024→2048 before evicting
//! anything. The textures are allocated at the 2048 maximum up
//! front (4 MiB mask + 16 MiB colour), so growth never invalidates
//! already-uploaded glyphs or their UVs — only the packer's
//! logical bound moves. A glyph that still cannot be allocated is
//! skipped for that frame and counted, never a frame failure.
//!
//! Images (Stage 8 Rec 12): textures are uploaded with a mip chain
//! ([`crate::image::build_mip_chain`]) and drawn from the level
//! matching the draw scale; the GPU texture cache
//! ([`ImageTextureCache`]) is byte-capped at
//! [`crate::image::IMAGE_CACHE_BYTE_CAP`] (64 MiB) with LRU
//! eviction that never evicts a texture drawn this frame. Images
//! culled by the CPU cull never reach the upload path at all, so a
//! culled image is neither uploaded nor bound. An animated frame
//! advance marks its texture dirty
//! ([`PaintRenderer::advance_image_frame`]) so the next prepare
//! re-uploads just that image from the already-swapped resource
//! table.
//!
//! Clipping (Stage 8 Rec 3): rect clips become `set_scissor_rect`
//! runs — one draw call per maximal run sharing a scissor rect —
//! and rounded clips become SDF slots in the clip uniform (the
//! outermost and innermost rounded clips of the stack; deeper
//! nesting keeps those two and the rect intersection stays exact).
//! Items fully outside the effective clip or the visible page
//! region are culled on the CPU in `prepare`, never uploaded.
//! A zero-size scissor is never emitted: the run is skipped.
//!
//! The renderer owns its own `FontSystem` for `SwashCache` lookups: the
//! cache keys in the display list are font ids, and two `FontSystem`s
//! loading the same system fonts in the same order assign the same ids.
//! (If that ever proves untrue on a machine, the fix is to hand the
//! renderer's `FontSystem` to layout instead of building a second one.)

use crate::glyph_key::font_id_map;
use crate::image::{IMAGE_CACHE_BYTE_CAP, build_mip_chain, mip_chain_bytes, mip_level_for_scale};
use crate::perf::{self, SpanKind};
use crate::{
    DecodedImage, DisplayItem, DisplayList, PaintGlyph, Resource, ResourceId, ResourceTable,
};
use cosmic_text::{FontSystem, SwashCache, SwashContent};
use pluto_css::ResolvedCorners;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use wgpu::util::DeviceExt;

/// The glyph atlas textures are allocated at this size (the Rec 11
/// maximum) up front; the packer's logical size starts at
/// [`ATLAS_INITIAL_SIZE`] and grows to this on demand.
const ATLAS_TEXTURE_SIZE: u32 = 2048;
/// The packer's logical size before growth (Rec 11: 1024→2048).
const ATLAS_INITIAL_SIZE: u32 = 1024;
/// The packer's logical size after growth; never larger.
const ATLAS_MAX_SIZE: u32 = 2048;
/// `queue.write_texture` requires rows padded to this many bytes.
const ROW_ALIGNMENT: usize = 256;

/// Shared WGSL for the rounded-clip SDF and the erf helpers
/// (Stage 8 Rec 3/4), written from scratch for Pluto. The
/// rounded-box SDF is the ellipse-normalised inside test (exact on
/// the boundary, distance scaled inside/outside), and `erf_approx`
/// is the Abramowitz & Stegun 7.1.26 polynomial (|ε| < 1.5e-7) —
/// public-domain numerical mathematics, not browser code. Each
/// shader below declares its own `clip` uniform binding after this
/// snippet (WGSL is order-independent at module scope) because the
/// clip lives at a different bind-group index per pipeline; the
/// sources are combined with `format!` in `PaintRenderer::new`
/// (`concat!` only accepts literals, not consts).
const CLIP_WGSL: &str = r#"
struct ClipSlot {
    rect: vec4<f32>,
    radii_a: vec4<f32>,
    radii_b: vec4<f32>,
}

struct ClipUniform {
    count: u32,
    slots: array<ClipSlot, 2>,
}

fn rounded_box_sdf(p: vec2<f32>, half: vec2<f32>, rtl: vec2<f32>, rtr: vec2<f32>, rbr: vec2<f32>, rbl: vec2<f32>) -> f32 {
    let ap = abs(p);
    var r = rbr;
    if (p.x < 0.0) {
        if (p.y < 0.0) { r = rtl; } else { r = rbl; }
    } else {
        if (p.y < 0.0) { r = rtr; } else { r = rbr; }
    }
    let c = half - r;
    let q = ap - c;
    if (q.x <= 0.0 || q.y <= 0.0) {
        return max(ap.x - half.x, ap.y - half.y);
    }
    let s = min(r.x, r.y);
    if (s <= 0.0) {
        return max(ap.x - half.x, ap.y - half.y);
    }
    return (length(q / r) - 1.0) * s;
}

fn clip_coverage(page: vec2<f32>) -> f32 {
    var cov = 1.0;
    for (var i = 0u; i < 2u; i++) {
        if (i >= clip.count) { break; }
        let slot = clip.slots[i];
        let half = slot.rect.zw * 0.5;
        let center = slot.rect.xy + half;
        let d = rounded_box_sdf(page - center, half, slot.radii_a.xy, slot.radii_a.zw, slot.radii_b.xy, slot.radii_b.zw);
        cov = min(cov, 1.0 - smoothstep(-0.5, 0.5, d));
    }
    return cov;
}

fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}
"#;

const RECT_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct RectVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct RectOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) page: vec2<f32>,
}

@vertex
fn rect_vs(v: RectVertex) -> RectOut {
    var out: RectOut;
    // The scroll offset is a view transform, not content: the
    // canvas fill is translated the other way on the CPU (see
    // `canvas_rect`), so it stays glued to the viewport.
    let p = v.position - uniforms.scroll;
    let clip = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.color = v.color;
    out.page = v.position;
    return out;
}

@fragment
fn rect_fs(in: RectOut) -> @location(0) vec4<f32> {
    let cov = clip_coverage(in.page);
    if (cov <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, in.color.a * cov);
}
"#;

const GLYPH_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var atlas: texture_2d<f32>;
@group(1) @binding(1)
var atlas_sampler: sampler;

struct GlyphVertex {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct GlyphOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) page: vec2<f32>,
}

@vertex
fn glyph_vs(v: GlyphVertex) -> GlyphOut {
    var out: GlyphOut;
    let p = v.position - uniforms.scroll;
    let clip = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    out.page = v.position;
    return out;
}

@fragment
fn glyph_fs(in: GlyphOut) -> @location(0) vec4<f32> {
    let alpha = textureSample(atlas, atlas_sampler, in.uv).r;
    let cov = alpha * in.color.a * clip_coverage(in.page);
    if (cov <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, cov);
}
"#;

/// Colour-glyph fragment (Stage 8 Rec 11): the colour atlas holds
/// RGBA emoji, so the fragment takes the texel's own colour (the
/// vertex colour contributes only its alpha, as a tint/opacity)
/// instead of tinting an alpha mask. Same vertex layout and bind
/// groups as the mask glyph pipeline, so colour and mask glyphs
/// share one vertex buffer and differ only by pipeline + atlas.
const COLOUR_GLYPH_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var atlas: texture_2d<f32>;
@group(1) @binding(1)
var atlas_sampler: sampler;

struct GlyphVertex {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct GlyphOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) page: vec2<f32>,
}

@vertex
fn glyph_vs(v: GlyphVertex) -> GlyphOut {
    var out: GlyphOut;
    let p = v.position - uniforms.scroll;
    let clip = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    out.page = v.position;
    return out;
}

@fragment
fn glyph_fs(in: GlyphOut) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas, atlas_sampler, in.uv);
    let cov = texel.a * in.color.a * clip_coverage(in.page);
    if (cov <= 0.0) { discard; }
    return vec4<f32>(texel.rgb, cov);
}
"#;

const IMAGE_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var tex: texture_2d<f32>;
@group(1) @binding(1)
var tex_sampler: sampler;

struct ImageVertex {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct ImageOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) page: vec2<f32>,
}

@vertex
fn image_vs(v: ImageVertex) -> ImageOut {
    var out: ImageOut;
    let p = v.position - uniforms.scroll;
    let clip = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = v.uv;
    out.page = v.position;
    return out;
}

@fragment
fn image_fs(in: ImageOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, tex_sampler, in.uv);
    let a = texel.a * clip_coverage(in.page);
    if (a <= 0.0) { discard; }
    return vec4<f32>(texel.rgb, a);
}
"#;

/// Rounded-rect fill (Stage 8 Rec 4): the fragment evaluates the
/// rounded-box SDF of the rect carried per vertex and outputs
/// colour × coverage (1 px AA). The clip SDF applies on top, so a
/// rounded fill inside a rounded clip stays correct.
const ROUNDED_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct RoundedVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) rect: vec4<f32>,
    @location(3) radii_a: vec4<f32>,
    @location(4) radii_b: vec4<f32>,
}

struct RoundedOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) page: vec2<f32>,
    @location(2) rect: vec4<f32>,
    @location(3) radii_a: vec4<f32>,
    @location(4) radii_b: vec4<f32>,
}

@vertex
fn rounded_vs(v: RoundedVertex) -> RoundedOut {
    var out: RoundedOut;
    let p = v.position - uniforms.scroll;
    let ndc = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = v.color;
    out.page = v.position;
    out.rect = v.rect;
    out.radii_a = v.radii_a;
    out.radii_b = v.radii_b;
    return out;
}

@fragment
fn rounded_fs(in: RoundedOut) -> @location(0) vec4<f32> {
    let half = in.rect.zw * 0.5;
    let center = in.rect.xy + half;
    let d = rounded_box_sdf(in.page - center, half, in.radii_a.xy, in.radii_a.zw, in.radii_b.xy, in.radii_b.zw);
    let cov = (1.0 - smoothstep(-0.5, 0.5, d)) * clip_coverage(in.page);
    if (cov <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, in.color.a * cov);
}
"#;

/// Box-shadow (Stage 8 Rec 4): one erf closed-form shader for both
/// blurred and sharp shadows. The blur profile is the 1-D erf
/// profile of the rounded-box SDF — on a straight edge it reduces
/// exactly to the separable erf form (σ = blur/2), and at corners
/// the SDF's level sets round the corner for free, which is the
/// correction the separable form needs a corner loop for. `blur ==
/// 0` takes the sharp branch (the expanded rounded rect itself).
/// An inset shadow inverts the profile against the shape and clips
/// to the padding box.
const SHADOW_WGSL_BODY: &str = r#"
@group(0) @binding(1)
var<uniform> clip: ClipUniform;

struct Uniforms {
    viewport: vec2<f32>,
    scroll: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct ShadowVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) shape: vec4<f32>,
    @location(3) shape_radii_a: vec4<f32>,
    @location(4) shape_radii_b: vec4<f32>,
    @location(5) pad: vec4<f32>,
    @location(6) pad_radii_a: vec4<f32>,
    @location(7) pad_radii_b: vec4<f32>,
    @location(8) params: vec2<f32>,
}

struct ShadowOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) page: vec2<f32>,
    @location(2) shape: vec4<f32>,
    @location(3) shape_radii_a: vec4<f32>,
    @location(4) shape_radii_b: vec4<f32>,
    @location(5) pad: vec4<f32>,
    @location(6) pad_radii_a: vec4<f32>,
    @location(7) pad_radii_b: vec4<f32>,
    @location(8) params: vec2<f32>,
}

@vertex
fn shadow_vs(v: ShadowVertex) -> ShadowOut {
    var out: ShadowOut;
    let p = v.position - uniforms.scroll;
    let ndc = vec2<f32>(
        p.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - p.y / uniforms.viewport.y * 2.0,
    );
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = v.color;
    out.page = v.position;
    out.shape = v.shape;
    out.shape_radii_a = v.shape_radii_a;
    out.shape_radii_b = v.shape_radii_b;
    out.pad = v.pad;
    out.pad_radii_a = v.pad_radii_a;
    out.pad_radii_b = v.pad_radii_b;
    out.params = v.params;
    return out;
}

@fragment
fn shadow_fs(in: ShadowOut) -> @location(0) vec4<f32> {
    let shape_half = in.shape.zw * 0.5;
    let shape_center = in.shape.xy + shape_half;
    let d = rounded_box_sdf(in.page - shape_center, shape_half, in.shape_radii_a.xy, in.shape_radii_a.zw, in.shape_radii_b.xy, in.shape_radii_b.zw);
    let sigma = in.params.x;
    let inset = in.params.y > 0.5;
    var cov: f32;
    if (inset) {
        let pad_half = in.pad.zw * 0.5;
        let pad_center = in.pad.xy + pad_half;
        let d_pad = rounded_box_sdf(in.page - pad_center, pad_half, in.pad_radii_a.xy, in.pad_radii_a.zw, in.pad_radii_b.xy, in.pad_radii_b.zw);
        let pad_cov = 1.0 - smoothstep(-0.5, 0.5, d_pad);
        if (sigma < 0.01) {
            cov = pad_cov * smoothstep(-0.5, 0.5, d);
        } else {
            cov = pad_cov * 0.5 * (1.0 + erf_approx(d / (sigma * 1.41421356)));
        }
    } else {
        if (sigma < 0.01) {
            cov = 1.0 - smoothstep(-0.5, 0.5, d);
        } else {
            cov = 0.5 * (1.0 - erf_approx(d / (sigma * 1.41421356)));
        }
    }
    cov = cov * clip_coverage(in.page);
    if (cov <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, in.color.a * cov);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RectVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/// One rounded-rect fill vertex: the quad position, the colour, and
/// the rect + radii the fragment evaluates the SDF against. The
/// rect/radii repeat per vertex (they are constant per quad); the
/// shader reads them from the varyings.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoundedVertex {
    pos: [f32; 2],
    color: [f32; 4],
    rect: [f32; 4],
    radii_a: [f32; 4],
    radii_b: [f32; 4],
}

/// One box-shadow vertex: the quad position, the colour, the shadow
/// shape (already expanded by spread and translated by the offset,
/// with radii expanded the same way), the padding box the inset
/// shadow clips to (== the shape's box for outer shadows), and
/// `(sigma, inset)` — σ = blur/2, inset 1.0 for inset shadows.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowVertex {
    pos: [f32; 2],
    color: [f32; 4],
    shape: [f32; 4],
    shape_radii_a: [f32; 4],
    shape_radii_b: [f32; 4],
    pad: [f32; 4],
    pad_radii_a: [f32; 4],
    pad_radii_b: [f32; 4],
    params: [f32; 2],
}

/// One rounded clip slot in the clip uniform: the clip rect plus the
/// four corner radii as two `vec4`s (`(tl.rx, tl.ry, tr.rx, tr.ry)`
/// and `(br.rx, br.ry, bl.rx, bl.ry)`). The WGSL `ClipUniform` is
/// `count: u32` (4 bytes, then 12 bytes of padding to the array's
/// 16-byte alignment) followed by two slots — this struct mirrors
/// that layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipSlotData {
    rect: [f32; 4],
    radii_a: [f32; 4],
    radii_b: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipUniformData {
    count: u32,
    _pad: [u32; 3],
    slots: [ClipSlotData; 2],
}

impl ClipUniformData {
    /// No rounded clips: the shader's `clip_coverage` returns 1.
    fn empty() -> Self {
        Self {
            count: 0,
            _pad: [0; 3],
            slots: [ClipSlotData {
                rect: [0.0; 4],
                radii_a: [0.0; 4],
                radii_b: [0.0; 4],
            }; 2],
        }
    }
}

/// One clip-stack entry while walking the display list: a pushed
/// clip's rect plus its rounded radii, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ClipEntry {
    rect: [f32; 4],
    radii: Option<ResolvedCorners>,
}

/// The effective clip of a clip stack, lowered for the renderer
/// (Stage 8 Rec 3): the intersection of every pushed rect (the
/// scissor), plus the rounded clips the SDF can test — the outermost
/// and the innermost rounded clip. Deeper nesting keeps those two;
/// the rect intersection stays exact either way.
fn clip_stack_effective(stack: &[ClipEntry]) -> (Option<[f32; 4]>, Vec<ClipEntry>) {
    let mut rect: Option<[f32; 4]> = None;
    for entry in stack {
        rect = Some(match rect {
            Some(current) => intersect_rect(current, entry.rect).unwrap_or([0.0; 4]),
            None => entry.rect,
        });
    }
    let mut rounded: Vec<ClipEntry> = stack
        .iter()
        .copied()
        .filter(|entry| entry.radii.is_some())
        .collect();
    // Outermost + innermost only: the SDF tests at most two slots.
    if rounded.len() > 2 {
        let last = rounded.pop().expect("non-empty");
        rounded.truncate(1);
        rounded.push(last);
    }
    (rect, rounded)
}

/// Intersect two `[x, y, w, h]` rects, or `None` when they do not
/// overlap (an empty intersection means "skip the draw", never a
/// zero-size scissor).
fn intersect_rect(a: [f32; 4], b: [f32; 4]) -> Option<[f32; 4]> {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    let (w, h) = (x2 - x, y2 - y);
    if w <= 0.0 || h <= 0.0 {
        None
    } else {
        Some([x, y, w, h])
    }
}

/// The Abramowitz & Stegun 7.1.26 erf approximation (|ε| < 1.5e-7),
/// mirroring `erf_approx` in the shadow shader. Public-domain
/// numerical mathematics, implemented from the formula.
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

/// The rounded-box SDF, mirroring `rounded_box_sdf` in the shaders:
/// negative inside, positive outside, zero on the curve. `p` is
/// relative to the box centre, `half` the half-size, `radii` the
/// four corner radii. Exact on the boundary for circular and
/// elliptical corners; the distance is scaled by the smaller radius
/// inside the corner region (the shader's 1 px AA band only needs
/// the sign and the boundary right).
pub fn rounded_box_sdf(p: [f32; 2], half: [f32; 2], radii: ResolvedCorners) -> f32 {
    let ap = [p[0].abs(), p[1].abs()];
    let r = if p[0] < 0.0 {
        if p[1] < 0.0 {
            radii.top_left
        } else {
            radii.bottom_left
        }
    } else if p[1] < 0.0 {
        radii.top_right
    } else {
        radii.bottom_right
    };
    let c = [half[0] - r.0, half[1] - r.1];
    let q = [ap[0] - c[0], ap[1] - c[1]];
    if q[0] <= 0.0 || q[1] <= 0.0 {
        return (ap[0] - half[0]).max(ap[1] - half[1]);
    }
    let s = r.0.min(r.1);
    if s <= 0.0 {
        return (ap[0] - half[0]).max(ap[1] - half[1]);
    }
    (((q[0] / r.0).powi(2) + (q[1] / r.1).powi(2)).sqrt() - 1.0) * s
}

/// The shadow coverage at a page point, mirroring `shadow_fs`: the
/// 1-D erf profile of the shape SDF for outer shadows (σ = blur/2;
/// the sharp branch at blur 0), inverted against the shape and
/// clipped to the padding box for inset shadows. `shape` is the
/// offset/spread-adjusted shape rect with its radii; `pad` is the
/// padding box with its radii (unused for outer shadows).
pub fn shadow_coverage(
    page: [f32; 2],
    shape: [f32; 4],
    shape_radii: ResolvedCorners,
    sigma: f32,
    inset: bool,
    pad: [f32; 4],
    pad_radii: ResolvedCorners,
) -> f32 {
    const SQRT_2: f32 = 1.41421356;
    let smoothstep = |edge0: f32, edge1: f32, x: f32| {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let shape_half = [shape[2] * 0.5, shape[3] * 0.5];
    let shape_center = [shape[0] + shape_half[0], shape[1] + shape_half[1]];
    let d = rounded_box_sdf(
        [page[0] - shape_center[0], page[1] - shape_center[1]],
        shape_half,
        shape_radii,
    );
    if inset {
        let pad_half = [pad[2] * 0.5, pad[3] * 0.5];
        let pad_center = [pad[0] + pad_half[0], pad[1] + pad_half[1]];
        let d_pad = rounded_box_sdf(
            [page[0] - pad_center[0], page[1] - pad_center[1]],
            pad_half,
            pad_radii,
        );
        let pad_cov = 1.0 - smoothstep(-0.5, 0.5, d_pad);
        if sigma < 0.01 {
            pad_cov * smoothstep(-0.5, 0.5, d)
        } else {
            pad_cov * 0.5 * (1.0 + erf_approx(d / (sigma * SQRT_2)))
        }
    } else if sigma < 0.01 {
        1.0 - smoothstep(-0.5, 0.5, d)
    } else {
        0.5 * (1.0 - erf_approx(d / (sigma * SQRT_2)))
    }
}

/// Six vertices (two triangles) for one rounded rect, with the rect
/// + radii carried per vertex. Pure: unit-tested.
fn rounded_vertices_for(item: &DisplayItem) -> Vec<RoundedVertex> {
    let DisplayItem::RoundedRect {
        x,
        y,
        width,
        height,
        radii,
        color,
    } = item
    else {
        return Vec::new();
    };
    let (x0, y0, x1, y1) = (*x, *y, x + width, y + height);
    let rect = [*x, *y, *width, *height];
    let radii_a = [
        radii.top_left.0,
        radii.top_left.1,
        radii.top_right.0,
        radii.top_right.1,
    ];
    let radii_b = [
        radii.bottom_right.0,
        radii.bottom_right.1,
        radii.bottom_left.0,
        radii.bottom_left.1,
    ];
    let c = *color;
    let quad = |pos: [f32; 2]| RoundedVertex {
        pos,
        color: c,
        rect,
        radii_a,
        radii_b,
    };
    vec![
        quad([x0, y0]),
        quad([x1, y0]),
        quad([x0, y1]),
        quad([x1, y0]),
        quad([x1, y1]),
        quad([x0, y1]),
    ]
}

/// The quad a shadow item draws, plus the shape + padding box the
/// fragment evaluates against. Outer shadows cover the shape (the
/// box expanded by spread, translated by the offset) inflated by
/// `~3σ` of blur falloff; inset shadows cover exactly the padding
/// box. Pure: unit-tested.
fn shadow_geometry(
    item: &DisplayItem,
) -> Option<([f32; 4], [f32; 4], ResolvedCorners, [f32; 4], ResolvedCorners)> {
    let DisplayItem::BoxShadow {
        x,
        y,
        width,
        height,
        radii,
        offset_x,
        offset_y,
        blur,
        spread,
        inset,
        ..
    } = item
    else {
        return None;
    };
    let pad = [*x, *y, *width, *height];
    if *inset {
        // The hole shrinks by spread (a positive spread makes an
        // inset shadow cover more of the interior); radii shrink
        // the same way, clamped at zero.
        let shape = [
            x + spread,
            y + spread,
            (width - 2.0 * spread).max(0.0),
            (height - 2.0 * spread).max(0.0),
        ];
        let shape = [
            shape[0] + offset_x,
            shape[1] + offset_y,
            shape[2],
            shape[3],
        ];
        let shrink = |r: (f32, f32)| ((r.0 - spread).max(0.0), (r.1 - spread).max(0.0));
        let shape_radii = ResolvedCorners {
            top_left: shrink(radii.top_left),
            top_right: shrink(radii.top_right),
            bottom_right: shrink(radii.bottom_right),
            bottom_left: shrink(radii.bottom_left),
        };
        Some((pad, shape, shape_radii, pad, *radii))
    } else {
        let shape = [
            x + offset_x - spread,
            y + offset_y - spread,
            width + 2.0 * spread,
            height + 2.0 * spread,
        ];
        let shape_radii = radii.expanded_by(*spread);
        let falloff = 3.0 * (*blur / 2.0) + 1.0;
        let quad = [
            shape[0] - falloff,
            shape[1] - falloff,
            shape[2] + 2.0 * falloff,
            shape[3] + 2.0 * falloff,
        ];
        Some((quad, shape, shape_radii, pad, *radii))
    }
}

/// Six vertices (two triangles) for one shadow item. Pure:
/// unit-tested.
fn shadow_vertices_for(item: &DisplayItem) -> Vec<ShadowVertex> {
    let DisplayItem::BoxShadow { color, blur, inset, .. } = item else {
        return Vec::new();
    };
    let Some((quad, shape, shape_radii, pad, pad_radii)) = shadow_geometry(item) else {
        return Vec::new();
    };
    let (x0, y0, x1, y1) = (quad[0], quad[1], quad[0] + quad[2], quad[1] + quad[3]);
    let shape_radii_a = [
        shape_radii.top_left.0,
        shape_radii.top_left.1,
        shape_radii.top_right.0,
        shape_radii.top_right.1,
    ];
    let shape_radii_b = [
        shape_radii.bottom_right.0,
        shape_radii.bottom_right.1,
        shape_radii.bottom_left.0,
        shape_radii.bottom_left.1,
    ];
    let pad_radii_a = [
        pad_radii.top_left.0,
        pad_radii.top_left.1,
        pad_radii.top_right.0,
        pad_radii.top_right.1,
    ];
    let pad_radii_b = [
        pad_radii.bottom_right.0,
        pad_radii.bottom_right.1,
        pad_radii.bottom_left.0,
        pad_radii.bottom_left.1,
    ];
    let c = *color;
    let params = [*blur / 2.0, if *inset { 1.0 } else { 0.0 }];
    let quad_v = |pos: [f32; 2]| ShadowVertex {
        pos,
        color: c,
        shape,
        shape_radii_a,
        shape_radii_b,
        pad,
        pad_radii_a,
        pad_radii_b,
        params,
    };
    vec![
        quad_v([x0, y0]),
        quad_v([x1, y0]),
        quad_v([x0, y1]),
        quad_v([x1, y0]),
        quad_v([x1, y1]),
        quad_v([x0, y1]),
    ]
}

/// One pipeline kind a draw run uses. Runs are walked in list
/// order in `render`, so paint order across pipelines is
/// preserved (a shadow draws before its box, a glyph after the
/// rect beneath it) while maximal runs sharing a pipeline, a clip
/// bind group, and a scissor rect batch into one draw call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunKind {
    Rect,
    Glyph,
    /// Colour glyphs (emoji): same vertex layout as `Glyph`, but
    /// the colour-glyph pipeline sampling the colour atlas.
    ColourGlyph,
    Image,
    Rounded,
    Shadow,
}

/// One maximal run of the display list: same pipeline, same clip
/// bind group, same scissor rect, contiguous vertices in that
/// pipeline's buffer. `scissor` is in page space (the effective
/// rect clip intersected with the viewport and the list damage);
/// `render` converts it to target pixels. `None` means the whole
/// widget. `image_bind_group` is set for `Image` runs only (one
/// per item — each image has its own texture).
struct DrawRun {
    kind: RunKind,
    clip: usize,
    scissor: Option<[f32; 4]>,
    first_vertex: u32,
    vertex_count: u32,
    image_bind_group: Option<wgpu::BindGroup>,
}

/// Build the clip uniform for the rounded entries of an effective
/// clip (outermost + innermost, per `clip_stack_effective`).
/// The slot packing mirrors the WGSL `ClipSlot` and the
/// `rounded_vertices_for` packing: `radii_a` is (tl, tr),
/// `radii_b` is (br, bl).
fn clip_uniform_data(rounded: &[ClipEntry]) -> ClipUniformData {
    let mut data = ClipUniformData::empty();
    data.count = rounded.len() as u32;
    for (i, entry) in rounded.iter().enumerate() {
        let Some(radii) = entry.radii else {
            continue;
        };
        data.slots[i] = ClipSlotData {
            rect: entry.rect,
            radii_a: [
                radii.top_left.0,
                radii.top_left.1,
                radii.top_right.0,
                radii.top_right.1,
            ],
            radii_b: [
                radii.bottom_right.0,
                radii.bottom_right.1,
                radii.bottom_left.0,
                radii.bottom_left.1,
            ],
        };
    }
    data
}

/// Convert a page-space scissor rect to target pixels, or `None`
/// when it covers nothing of the widget (the run is skipped, never
/// drawn with a zero-size scissor). `scroll` is the page offset
/// the vertex shaders subtract; `widget` is the widget's rect in
/// target pixels; `scale` is target px per page px.
fn scissor_physical(
    rect: [f32; 4],
    scroll: (f32, f32),
    widget: (u32, u32, u32, u32),
    scale: f32,
) -> Option<(u32, u32, u32, u32)> {
    let (wx, wy, ww, wh) = (
        widget.0 as f32,
        widget.1 as f32,
        widget.2 as f32,
        widget.3 as f32,
    );
    let x0 = ((rect[0] - scroll.0) * scale + wx).floor().max(wx);
    let y0 = ((rect[1] - scroll.1) * scale + wy).floor().max(wy);
    let x1 = ((rect[0] + rect[2] - scroll.0) * scale + wx)
        .ceil()
        .min(wx + ww);
    let y1 = ((rect[1] + rect[3] - scroll.1) * scale + wy)
        .ceil()
        .min(wy + wh);
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some((
            x0 as u32,
            y0 as u32,
            (x1 - x0) as u32,
            (y1 - y0) as u32,
        ))
    }
}

/// True when `bounds` (`[x, y, w, h]`, page space) is at least
/// partly inside both the visible page region (`visible`, page
/// space, `[scroll.x, scroll.y, vw, vh]`) and the run's
/// page-space scissor (`clip`, `None` = no rect clip). A
/// zero-area bounds never counts as visible.
fn bounds_visible(bounds: [f32; 4], clip: Option<[f32; 4]>, visible: [f32; 4]) -> bool {
    if bounds[2] <= 0.0 || bounds[3] <= 0.0 {
        return false;
    }
    let mut x0 = bounds[0].max(visible[0]);
    let mut y0 = bounds[1].max(visible[1]);
    let mut x1 = (bounds[0] + bounds[2]).min(visible[0] + visible[2]);
    let mut y1 = (bounds[1] + bounds[3]).min(visible[1] + visible[3]);
    if let Some(c) = clip {
        x0 = x0.max(c[0]);
        y0 = y0.max(c[1]);
        x1 = x1.min(c[0] + c[2]);
        y1 = y1.min(c[1] + c[3]);
    }
    x1 > x0 && y1 > y0
}

/// Extend the last run when the new vertices continue it (same
/// kind, clip, and scissor, contiguous in the same buffer), or
/// start a new run. Image runs never merge: each carries its own
/// texture bind group.
fn extend_run(
    runs: &mut Vec<DrawRun>,
    kind: RunKind,
    clip: usize,
    scissor: Option<[f32; 4]>,
    first_vertex: u32,
    count: u32,
    image_bind_group: Option<wgpu::BindGroup>,
) {
    if count == 0 {
        return;
    }
    if let Some(last) = runs.last_mut()
        && last.kind == kind
        && last.clip == clip
        && last.scissor == scissor
        && last.image_bind_group.is_none()
        && image_bind_group.is_none()
        && last.first_vertex + last.vertex_count == first_vertex
    {
        last.vertex_count += count;
        return;
    }
    runs.push(DrawRun {
        kind,
        clip,
        scissor,
        first_vertex,
        vertex_count: count,
        image_bind_group,
    });
}

/// Upload a vertex buffer, or `None` when there is nothing to draw
/// (an empty buffer would still be a valid draw of zero vertices,
/// but skipping it keeps `render` from binding a buffer for a run
/// that cannot exist).
fn make_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    vertices: &[T],
) -> Option<wgpu::Buffer> {
    if vertices.is_empty() {
        None
    } else {
        Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }))
    }
}

/// One uploaded image/blob texture, cached by resource id: the
/// texture itself (kept so a dirty frame re-upload can write into
/// it) plus one bind group per mip level, each viewing only that
/// level — the draw picks the level for its scale (Rec 12).
struct CachedImage {
    texture: wgpu::Texture,
    bind_groups: Vec<wgpu::BindGroup>,
}

/// One packed rect in an atlas texture: top-left corner plus size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Which atlas a glyph lives in (Stage 8 Rec 11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtlasKind {
    Mask,
    Colour,
}

/// Which atlas a Swash glyph image belongs in: alpha masks go to
/// the R8 mask atlas, RGBA colour glyphs (emoji) to the second
/// `Rgba8UnormSrgb` atlas, subpixel masks to neither — their data
/// is RGB per pixel, not a plain alpha mask, so they are skipped
/// (and counted) rather than drawn wrong.
pub fn atlas_kind_for(content: SwashContent) -> Option<AtlasKind> {
    match content {
        SwashContent::Mask => Some(AtlasKind::Mask),
        SwashContent::Color => Some(AtlasKind::Colour),
        SwashContent::SubpixelMask => None,
    }
}

/// One atlas entry: where the glyph sits, and the generation it
/// was last used in (the LRU key).
#[derive(Debug, Clone, Copy)]
struct AtlasEntry {
    rect: AtlasRect,
    last_used: u64,
}

/// A glyph atlas (Stage 8 Rec 11): a guillotine free-rectangle
/// packer with generation-stamped LRU eviction. The texture behind
/// it is allocated at [`ATLAS_TEXTURE_SIZE`] up front; `size` is
/// the packer's *logical* bound, starting at 1024 and growing to
/// 2048 before anything is evicted. Growth only adds free strips
/// for the new area, so already-packed glyphs keep their rects and
/// their uploads. Eviction removes the least-recently-used glyph
/// that is not in use by the current frame and returns its block
/// to the free list (merged with neighbours); a glyph that cannot
/// be allocated even after evicting everything evictable is
/// skipped by the caller, never a frame failure.
///
/// Blocks carry 1 px of padding on the right and bottom so
/// filtering never bleeds a neighbour; the padding is part of the
/// block the free list tracks, so evicted blocks merge exactly.
#[derive(Debug)]
pub struct GlyphAtlas {
    size: u32,
    max_size: u32,
    free: Vec<AtlasRect>,
    map: HashMap<cosmic_text::CacheKey, AtlasEntry>,
    generation: u64,
    in_use: HashSet<cosmic_text::CacheKey>,
    evictions: u64,
    growths: u32,
}

impl GlyphAtlas {
    pub fn new() -> Self {
        Self::with_size(ATLAS_INITIAL_SIZE, ATLAS_MAX_SIZE)
    }

    /// An atlas with an explicit logical size and maximum (tests
    /// use small sizes to force growth/eviction cheaply).
    pub fn with_size(size: u32, max_size: u32) -> Self {
        let size = size.max(1).min(max_size.max(1));
        Self {
            size,
            max_size: max_size.max(size),
            free: vec![AtlasRect {
                x: 0,
                y: 0,
                w: size,
                h: size,
            }],
            map: HashMap::new(),
            generation: 0,
            in_use: HashSet::new(),
            evictions: 0,
            growths: 0,
        }
    }

    /// The packer's current logical size (1024 before growth,
    /// 2048 after).
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn contains(&self, key: &cosmic_text::CacheKey) -> bool {
        self.map.contains_key(key)
    }

    pub fn rect_for(&self, key: &cosmic_text::CacheKey) -> Option<AtlasRect> {
        self.map.get(key).map(|entry| entry.rect)
    }

    /// Glyphs evicted so far, and times the atlas grew.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn growths(&self) -> u32 {
        self.growths
    }

    /// Start a frame: the generation advances (the LRU clock) and
    /// the in-use set clears. Every glyph allocated this frame is
    /// in use until the next `begin_frame`, so eviction can never
    /// pull a glyph out from under a frame being built.
    pub fn begin_frame(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.in_use.clear();
    }

    /// Allocate (or find) `w`×`h` for `key`. A hit refreshes the
    /// glyph's generation and marks it in use. A miss packs a new
    /// block, growing 1024→2048 first when the atlas is full, and
    /// evicting least-recently-used glyphs only when growth cannot
    /// help. Returns `None` when the glyph is larger than the
    /// maximum atlas, has a zero dimension, or no evictable glyph
    /// remains.
    pub fn allocate(&mut self, key: cosmic_text::CacheKey, w: u32, h: u32) -> Option<AtlasRect> {
        if w == 0 || h == 0 || w > self.max_size || h > self.max_size {
            return None;
        }
        if let Some(entry) = self.map.get_mut(&key) {
            entry.last_used = self.generation;
            self.in_use.insert(key);
            return Some(entry.rect);
        }
        // The block carries 1 px of padding (except at the atlas
        // edge, where there is nothing to bleed into).
        let bw = if w < self.size { w + 1 } else { w };
        let bh = if h < self.size { h + 1 } else { h };
        if let Some(rect) = self.try_allocate(bw, bh) {
            let rect = AtlasRect { w, h, ..rect };
            self.map.insert(
                key,
                AtlasEntry {
                    rect,
                    last_used: self.generation,
                },
            );
            self.in_use.insert(key);
            return Some(rect);
        }
        if self.size < self.max_size {
            self.grow();
            if let Some(rect) = self.try_allocate(bw, bh) {
                let rect = AtlasRect { w, h, ..rect };
                self.map.insert(
                    key,
                    AtlasEntry {
                        rect,
                        last_used: self.generation,
                    },
                );
                self.in_use.insert(key);
                return Some(rect);
            }
        }
        // Evict least-recently-used glyphs, one at a time, until
        // the new block fits or nothing evictable remains.
        loop {
            if self.evict_one().is_none() {
                return None;
            }
            if let Some(rect) = self.try_allocate(bw, bh) {
                let rect = AtlasRect { w, h, ..rect };
                self.map.insert(
                    key,
                    AtlasEntry {
                        rect,
                        last_used: self.generation,
                    },
                );
                self.in_use.insert(key);
                return Some(rect);
            }
        }
    }

    /// Best-fit free block for a `bw`×`bh` block (smallest area
    /// that fits, then lowest, then leftmost, so packing is
    /// deterministic): carve it out of the free rect, returning
    /// the right strip and the bottom strip to the free list.
    fn try_allocate(&mut self, bw: u32, bh: u32) -> Option<AtlasRect> {
        let mut best: Option<(usize, u32)> = None; // (index, area)
        for (i, free) in self.free.iter().enumerate() {
            if free.w >= bw && free.h >= bh {
                let area = free.w * free.h;
                let better = match best {
                    None => true,
                    Some((bi, ba)) => {
                        area < ba
                            || (area == ba && (free.y, free.x) < (self.free[bi].y, self.free[bi].x))
                    }
                };
                if better {
                    best = Some((i, area));
                }
            }
        }
        let (i, _) = best?;
        let free = self.free.remove(i);
        if free.w > bw {
            self.free.push(AtlasRect {
                x: free.x + bw,
                y: free.y,
                w: free.w - bw,
                h: bh,
            });
        }
        if free.h > bh {
            self.free.push(AtlasRect {
                x: free.x,
                y: free.y + bh,
                w: free.w,
                h: free.h - bh,
            });
        }
        Some(AtlasRect {
            x: free.x,
            y: free.y,
            w: bw,
            h: bh,
        })
    }

    /// Double the logical size (1024→2048) and add the new area
    /// as free strips; packed glyphs are untouched. The strips
    /// merge with the existing free area, so a glyph as large as
    /// the new atlas can still pack into one rect.
    fn grow(&mut self) {
        let old = self.size;
        self.size = (self.size * 2).min(self.max_size);
        if self.size > old {
            self.free.push(AtlasRect {
                x: old,
                y: 0,
                w: self.size - old,
                h: old,
            });
            self.free.push(AtlasRect {
                x: 0,
                y: old,
                w: self.size,
                h: self.size - old,
            });
            self.merge_free();
            self.growths += 1;
        }
    }

    /// Evict the least-recently-used glyph that is not in use this
    /// frame, returning its block to the free list. Returns `None`
    /// when every cached glyph is in use (or the atlas is empty).
    fn evict_one(&mut self) -> Option<cosmic_text::CacheKey> {
        let mut victim: Option<(cosmic_text::CacheKey, u64)> = None;
        for (key, entry) in &self.map {
            if self.in_use.contains(key) {
                continue;
            }
            let better = match victim {
                None => true,
                Some((_, lu)) => entry.last_used < lu,
            };
            if better {
                victim = Some((*key, entry.last_used));
            }
        }
        let (key, _) = victim?;
        let entry = self.map.remove(&key).expect("victim was in the map");
        // The freed block includes the padding, so it merges
        // exactly with its neighbours.
        let bw = if entry.rect.w < self.size {
            entry.rect.w + 1
        } else {
            entry.rect.w
        };
        let bh = if entry.rect.h < self.size {
            entry.rect.h + 1
        } else {
            entry.rect.h
        };
        self.free.push(AtlasRect {
            x: entry.rect.x,
            y: entry.rect.y,
            w: bw,
            h: bh,
        });
        self.merge_free();
        self.evictions += 1;
        Some(key)
    }

    /// Merge adjacent free rects (same height, touching in x; or
    /// same width, touching in y) so evicted blocks recombine into
    /// usable space instead of fragmenting.
    fn merge_free(&mut self) {
        loop {
            let mut merged = false;
            'outer: for i in 0..self.free.len() {
                for j in (i + 1)..self.free.len() {
                    let (a, b) = (self.free[i], self.free[j]);
                    let merged_rect = if a.y == b.y && a.h == b.h && a.x + a.w == b.x {
                        Some(AtlasRect {
                            x: a.x,
                            y: a.y,
                            w: a.w + b.w,
                            h: a.h,
                        })
                    } else if b.y == a.y && b.h == a.h && b.x + b.w == a.x {
                        Some(AtlasRect {
                            x: b.x,
                            y: b.y,
                            w: b.w + a.w,
                            h: b.h,
                        })
                    } else if a.x == b.x && a.w == b.w && a.y + a.h == b.y {
                        Some(AtlasRect {
                            x: a.x,
                            y: a.y,
                            w: a.w,
                            h: a.h + b.h,
                        })
                    } else if b.x == a.x && b.w == a.w && b.y + b.h == a.y {
                        Some(AtlasRect {
                            x: b.x,
                            y: b.y,
                            w: b.w,
                            h: b.h + a.h,
                        })
                    } else {
                        None
                    };
                    if let Some(rect) = merged_rect {
                        self.free.remove(j);
                        self.free[i] = rect;
                        merged = true;
                        break 'outer;
                    }
                }
            }
            if !merged {
                break;
            }
        }
    }
}

/// One entry of the renderer's GPU image cache (Stage 8 Rec 12):
/// the mip-chain bytes it holds, how many levels were uploaded,
/// the generation it was last drawn in, whether its current frame
/// is waiting to be re-uploaded, and how many frame advances it
/// has taken. Pure bookkeeping — the wgpu handles live in the
/// renderer's separate map, so this is unit-tested without a device.
#[derive(Debug, Clone, Copy)]
struct ImageEntryMeta {
    bytes: usize,
    mip_levels: u32,
    last_used: u64,
    dirty: bool,
    frames_advanced: u32,
}

/// Byte-capped LRU bookkeeping for uploaded image textures (Stage 8
/// Rec 12): the cap is [`IMAGE_CACHE_BYTE_CAP`] (64 MiB), eviction
/// is least-recently-used and never evicts a texture drawn this
/// frame, and a frame advance dirties only its own entry.
#[derive(Debug, Default)]
struct ImageTextureCache {
    entries: HashMap<ResourceId, ImageEntryMeta>,
    bytes_used: usize,
    generation: u64,
    in_use: HashSet<ResourceId>,
    evictions: u64,
}

impl ImageTextureCache {
    fn new() -> Self {
        Self::default()
    }

    fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    fn evictions(&self) -> u64 {
        self.evictions
    }

    fn contains(&self, id: ResourceId) -> bool {
        self.entries.contains_key(&id)
    }

    fn mip_levels(&self, id: ResourceId) -> Option<u32> {
        self.entries.get(&id).map(|entry| entry.mip_levels)
    }

    /// Start a frame: the generation advances and the in-use set
    /// clears, so eviction can never pull a texture out from under
    /// a frame being built.
    fn begin_frame(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.in_use.clear();
    }

    /// Record a freshly uploaded texture (or a re-upload, which
    /// keeps the same id and replaces the byte count).
    fn register(&mut self, id: ResourceId, bytes: usize, mip_levels: u32) {
        if let Some(old) = self.entries.get(&id) {
            self.bytes_used = self.bytes_used.saturating_sub(old.bytes);
        }
        self.entries.insert(
            id,
            ImageEntryMeta {
                bytes,
                mip_levels,
                last_used: self.generation,
                dirty: false,
                frames_advanced: self.entries.get(&id).map_or(0, |old| old.frames_advanced),
            },
        );
        self.bytes_used += bytes;
        self.in_use.insert(id);
    }

    /// Mark an entry drawn this frame (refreshes its generation).
    fn touch(&mut self, id: ResourceId) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.last_used = self.generation;
        }
        self.in_use.insert(id);
    }

    /// True when this image's current frame is waiting to be
    /// re-uploaded (set by [`mark_frame_dirty`]).
    fn is_dirty(&self, id: ResourceId) -> bool {
        self.entries.get(&id).is_some_and(|entry| entry.dirty)
    }

    /// Mark an image's texture dirty so the next prepare
    /// re-uploads just that image. Returns `false` when the image
    /// has no uploaded texture yet — the next prepare uploads the
    /// current table pixels fresh, so there is nothing to dirty.
    fn mark_frame_dirty(&mut self, id: ResourceId) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        entry.dirty = true;
        entry.frames_advanced += 1;
        true
    }

    /// Clear the dirty flag after a re-upload. Returns `true`
    /// when the flag was set.
    fn clear_dirty(&mut self, id: ResourceId) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        std::mem::replace(&mut entry.dirty, false)
    }

    /// Evict least-recently-used textures that were not drawn this
    /// frame until `needed` more bytes fit under `cap`, returning
    /// the evicted ids (the caller drops their wgpu handles).
    /// Stops when nothing evictable remains, even if the cap is
    /// still exceeded — a single image larger than the cap stores
    /// alone rather than being refused.
    fn evict_for(&mut self, needed: usize, cap: usize) -> Vec<ResourceId> {
        let mut evicted = Vec::new();
        while self.bytes_used + needed > cap {
            let mut victim: Option<(ResourceId, u64)> = None;
            for (id, entry) in &self.entries {
                if self.in_use.contains(id) {
                    continue;
                }
                let better = match victim {
                    None => true,
                    Some((_, lu)) => entry.last_used < lu,
                };
                if better {
                    victim = Some((*id, entry.last_used));
                }
            }
            let Some((id, _)) = victim else {
                break;
            };
            if let Some(entry) = self.entries.remove(&id) {
                self.bytes_used = self.bytes_used.saturating_sub(entry.bytes);
            }
            self.evictions += 1;
            evicted.push(id);
        }
        evicted
    }
}
/// Six vertices (two triangles) for one solid rect. Pure: unit-tested.
fn rect_vertices_for(item: &DisplayItem) -> Vec<RectVertex> {
    let DisplayItem::FillRect {
        x,
        y,
        width,
        height,
        color,
    } = item
    else {
        return Vec::new();
    };
    let (x0, y0, x1, y1) = (*x, *y, x + width, y + height);
    let c = *color;
    vec![
        RectVertex { pos: [x0, y0], color: c },
        RectVertex { pos: [x1, y0], color: c },
        RectVertex { pos: [x0, y1], color: c },
        RectVertex { pos: [x1, y0], color: c },
        RectVertex { pos: [x1, y1], color: c },
        RectVertex { pos: [x0, y1], color: c },
    ]
}

/// Six vertices (two triangles) for one image/blob item, with UVs
/// spanning the whole texture. Pure: unit-tested. UV (0,0) is the
/// texture's first texel (top-left as uploaded), and it lands on the
/// quad's top-left, so the image draws upright.
fn image_vertices_for(item: &DisplayItem) -> Vec<ImageVertex> {
    let (x, y, width, height) = match item {
        DisplayItem::Image {
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
        } => (*x, *y, *width, *height),
        _ => return Vec::new(),
    };
    let (x0, y0, x1, y1) = (x, y, x + width, y + height);
    vec![
        ImageVertex {
            pos: [x0, y0],
            uv: [0.0, 0.0],
        },
        ImageVertex {
            pos: [x1, y0],
            uv: [1.0, 0.0],
        },
        ImageVertex {
            pos: [x0, y1],
            uv: [0.0, 1.0],
        },
        ImageVertex {
            pos: [x1, y0],
            uv: [1.0, 0.0],
        },
        ImageVertex {
            pos: [x1, y1],
            uv: [1.0, 1.0],
        },
        ImageVertex {
            pos: [x0, y1],
            uv: [0.0, 1.0],
        },
    ]
}

/// Upload RGBA8 pixels into one mip level of a texture, padding
/// rows to the 256-byte alignment `queue.write_texture` requires
/// (same pattern as `upload_glyph`, four bytes per pixel instead
/// of one).
fn upload_image_level(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 4;
    let padded = row_bytes.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
    let mut bytes = vec![0u8; padded * h];
    for row in 0..h {
        bytes[row * padded..row * padded + row_bytes]
            .copy_from_slice(&rgba[row * row_bytes..(row + 1) * row_bytes]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded as u32),
            rows_per_image: Some(h as u32),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Upload a whole mip chain (level 0 is the image, each further
/// level a downscale) into the matching levels of `texture`.
fn upload_image_mips(queue: &wgpu::Queue, texture: &wgpu::Texture, chain: &[DecodedImage]) {
    for (level, image) in chain.iter().enumerate() {
        upload_image_level(
            queue,
            texture,
            level as u32,
            image.width,
            image.height,
            &image.rgba,
        );
    }
}

/// The texture format for an uploaded image, matched to the render
/// target: an sRGB target needs an sRGB texture so sampling converts
/// to linear and the target converts back (net: the stored bytes);
/// a non-sRGB target needs a plain texture. The swapchain formats a
/// browser actually sees are the two 8-bit sRGB ones.
fn image_texture_format(target: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match target {
        wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb => {
            wgpu::TextureFormat::Rgba8UnormSrgb
        }
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// The canvas fill (item 0 of a list from [`DisplayList::build`])
/// translated by the scroll offset, so it stays glued to the
/// viewport while the content scrolls under it. The vertex shaders
/// subtract the scroll from every position; translating the canvas
/// by the same offset cancels that out for this one rect. Any other
/// item passes through unchanged. Pure: unit-tested.
fn canvas_rect(item: &DisplayItem, scroll: (f32, f32)) -> DisplayItem {
    match item {
        DisplayItem::FillRect {
            x,
            y,
            width,
            height,
            color,
        } => DisplayItem::FillRect {
            x: x + scroll.0,
            y: y + scroll.1,
            width: *width,
            height: *height,
            color: *color,
        },
        other => other.clone(),
    }
}

/// Upload one glyph's alpha mask into the atlas texture, padding rows to
/// the 256-byte alignment `queue.write_texture` requires.
fn upload_glyph(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rect: AtlasRect,
    data: &[u8],
) {
    let w = rect.w as usize;
    let h = rect.h as usize;
    let padded = w.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
    let mut bytes = vec![0u8; padded * h];
    for row in 0..h {
        bytes[row * padded..row * padded + w]
            .copy_from_slice(&data[row * w..(row + 1) * w]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x,
                y: rect.y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded as u32),
            rows_per_image: Some(h as u32),
        },
        wgpu::Extent3d {
            width: rect.w,
            height: rect.h,
            depth_or_array_layers: 1,
        },
    );
}

/// Upload one colour glyph's RGBA pixels into the colour atlas
/// texture, padding rows to the 256-byte alignment
/// `queue.write_texture` requires (same pattern as `upload_glyph`,
/// four bytes per pixel instead of one).
fn upload_colour_glyph(queue: &wgpu::Queue, texture: &wgpu::Texture, rect: AtlasRect, data: &[u8]) {
    let w = rect.w as usize;
    let h = rect.h as usize;
    let row_bytes = w * 4;
    let padded = row_bytes.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
    let mut bytes = vec![0u8; padded * h];
    for row in 0..h {
        bytes[row * padded..row * padded + row_bytes]
            .copy_from_slice(&data[row * row_bytes..(row + 1) * row_bytes]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x,
                y: rect.y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded as u32),
            rows_per_image: Some(h as u32),
        },
        wgpu::Extent3d {
            width: rect.w,
            height: rect.h,
            depth_or_array_layers: 1,
        },
    );
}

/// Top-left corner and size of one glyph's quad, in y-down page
/// coordinates. Pure: unit-tested against the `with_pixels` oracle.
///
/// `glyph.x`/`glyph.y` is the glyph origin on the baseline: layout's
/// `physical()` call already folds the run's `line_y` into it (see
/// `shape_leaf` in pluto-layout), so no `line_y` is added here —
/// adding it again would shift every glyph a line down.
/// `placement.top` is the distance from the baseline *up* to the
/// bitmap's top scanline (Swash/zeno y-up convention, same as
/// FreeType's `bitmap_top`), so it is *subtracted* in y-down page
/// space. cosmic-text's own `with_pixels` does exactly this: it
/// enumerates the bitmap's first pixel at
/// `(placement.left, -placement.top)` relative to the origin.
/// `placement.left` may be negative (overhanging glyphs like `f`/`j`)
/// and must not be clamped. (Stage 1 research: the old code added
/// `placement.top` here, displacing each glyph down by `2 * top` —
/// most for overshooting round letters like O/S.)
fn glyph_quad(glyph: &PaintGlyph, placement: cosmic_text::Placement) -> (f32, f32, f32, f32) {
    let x0 = glyph.x + placement.left as f32;
    let y0 = glyph.y - placement.top as f32;
    (
        x0,
        y0,
        x0 + placement.width as f32,
        y0 + placement.height as f32,
    )
}

pub struct PaintRenderer {
    rect_pipeline: wgpu::RenderPipeline,
    glyph_pipeline: wgpu::RenderPipeline,
    /// Colour-glyph pipeline (Stage 8 Rec 11): same vertex layout
    /// as the mask glyph pipeline, sampling the colour atlas.
    colour_glyph_pipeline: wgpu::RenderPipeline,
    image_pipeline: wgpu::RenderPipeline,
    /// Rounded-rect fill pipeline (Stage 8 Rec 4): the fragment
    /// evaluates the rounded-box SDF for coverage.
    rounded_pipeline: wgpu::RenderPipeline,
    /// Box-shadow pipeline (Stage 8 Rec 4): the erf closed-form
    /// blur profile of the shape SDF (sharp branch at blur 0).
    shadow_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    /// The shared group-0 layout (binding 0: viewport/scroll
    /// uniform, vertex; binding 1: clip uniform, fragment): one
    /// layout object, used by every pipeline. Group 0 is built
    /// per clip in `prepare` (uniform buffer + clip buffer), so
    /// no pipeline needs more than two bind groups — some devices
    /// (Haswell) cap `max_bind_groups` at 2, and the old
    /// three-group glyph layout panicked at creation there.
    uniform_clip_bind_group_layout: wgpu::BindGroupLayout,
    atlas_texture: wgpu::Texture,
    atlas_bind_group: wgpu::BindGroup,
    /// The colour atlas texture + bind group (Stage 8 Rec 11):
    /// `Rgba8UnormSrgb`, same layout as the mask atlas.
    colour_atlas_texture: wgpu::Texture,
    colour_atlas_bind_group: wgpu::BindGroup,
    atlas_bind_group_layout: wgpu::BindGroupLayout,
    atlas_sampler: wgpu::Sampler,
    /// The mask atlas packer (Stage 8 Rec 11): LRU eviction,
    /// growth 1024→2048, in-use protection per frame.
    mask_atlas: GlyphAtlas,
    /// The colour atlas packer (Stage 8 Rec 11): same policy,
    /// independent entries — evicting a mask glyph never evicts
    /// its colour namesake.
    colour_atlas: GlyphAtlas,
    font_system: FontSystem,
    /// fontdb id (as the `u64` in [`GlyphKey`]) -> `fontdb::ID`,
    /// built once from the font database: the reverse map
    /// [`GlyphKey::to_cache_key`] needs.
    font_ids: HashMap<u64, fontdb::ID>,
    swash: SwashCache,
    rect_buffer: Option<wgpu::Buffer>,
    glyph_buffer: Option<wgpu::Buffer>,
    image_buffer: Option<wgpu::Buffer>,
    rounded_buffer: Option<wgpu::Buffer>,
    shadow_buffer: Option<wgpu::Buffer>,
    /// The ordered draw runs of the last `prepare`, in list
    /// (paint) order. `render` walks these and nothing else.
    runs: Vec<DrawRun>,
    /// Clip bind groups built by the last `prepare`, indexed by
    /// `DrawRun::clip`. Rebuilt every frame (a handful at most);
    /// the buffers are small (112 bytes each).
    clip_groups: Vec<wgpu::BindGroup>,
    /// Uploaded image/blob textures, by resource id (Stage 8 Rec
    /// 12): the texture plus one bind group per mip level. The
    /// byte cap / LRU / dirty bookkeeping lives in `image_meta`.
    image_textures: HashMap<ResourceId, CachedImage>,
    /// Byte-capped LRU + frame-dirty bookkeeping for the GPU image
    /// cache (Stage 8 Rec 12).
    image_meta: ImageTextureCache,
    /// Identity of the last list `prepare` built for: (list
    /// address, viewport, scroll). A repeat `prepare` with the
    /// same identity is a clean skip — no rebuild, no re-encode.
    last_prepared: Option<(usize, (f32, f32), (f32, f32))>,
    viewport: (f32, f32),
    /// The scroll offset last written to the uniform (page
    /// coordinates): the vertex shaders subtract it, so the content
    /// scrolls without a re-layout. The canvas fill is translated
    /// the other way in `prepare` so it stays glued to the viewport.
    scroll: (f32, f32),
    /// The render target's format, for picking the image texture
    /// format (see [`image_texture_format`]).
    target_format: wgpu::TextureFormat,
    /// Glyphs skipped because the mask atlas was full (diagnostic count).
    pub atlas_full_skips: u32,
    /// Colour glyphs skipped because the colour atlas was full
    /// (Stage 8 Rec 11, diagnostic count).
    pub colour_atlas_full_skips: u32,
    /// Subpixel-mask glyphs skipped: their data is RGB per pixel,
    /// not a plain alpha mask, so they are counted, not drawn wrong.
    pub subpixel_skips: u32,
    /// Image/blob items skipped because the resource was missing or
    /// its pixels did not match its dimensions (diagnostic count).
    /// The shell validates before handing a list over, so a skip
    /// here means a bug or a race, not a normal frame.
    pub image_skips: u32,
    /// Display-list items CPU-culled in the last `prepare`
    /// (outside the clip or the viewport); also recorded in the
    /// process-wide perf counters.
    pub items_culled: u64,
}

impl PaintRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pluto viewport uniform"),
            contents: bytemuck::cast_slice(&[1.0f32, 1.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Group 0 carries the viewport/scroll uniform (binding
        // 0, vertex) AND the clip uniform (binding 1, fragment).
        // The clip used to live in its own group — group 1 for
        // rect/rounded/shadow, group 2 for glyph/image — which
        // made the glyph layout three groups and panicked at
        // creation on devices whose `max_bind_groups` is 2
        // (Haswell: "Bind group layout count 3 exceeds device bind
        // group limit 2"). Merging it here keeps every pipeline
        // at two groups or fewer. Group 0 is built per clip in
        // `prepare` (uniform buffer + clip buffer); there is no
        // single uniform bind group anymore.
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pluto uniform + clip layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pluto glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_TEXTURE_SIZE,
                height: ATLAS_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pluto glyph sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pluto atlas layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pluto atlas bind group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // The colour atlas (Stage 8 Rec 11): same size as the mask
        // atlas, `Rgba8UnormSrgb` so emoji sample as colour. The
        // texture is allocated at the maximum size up front, like
        // the mask atlas, so packer growth never invalidates it.
        let colour_atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pluto colour glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_TEXTURE_SIZE,
                height: ATLAS_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let colour_atlas_view =
            colour_atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let colour_atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pluto colour atlas bind group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&colour_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // The glyph/image layout: group 0 is uniform + clip,
        // group 1 is texture/sampler. Two groups at most: some
        // devices (Haswell) cap `max_bind_groups` at 2, and the
        // old three-group layout ([uniform, atlas, clip]) panicked
        // at creation there.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pluto pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &atlas_bind_group_layout],
            immediate_size: 0,
        });
        // The rect pipeline needs only group 0 (uniform + clip).
        // The rounded and shadow pipelines share this layout.
        let rect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pluto rect pipeline layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                immediate_size: 0,
            });
        // Each shader is the shared clip snippet (SDF + erf
        // helpers) plus its own body; `concat!` only accepts
        // literals, so the combination is a `format!`.
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto rect shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{RECT_WGSL_BODY}").into()),
        });
        let glyph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto glyph shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{GLYPH_WGSL_BODY}").into()),
        });
        let rounded_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto rounded shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{ROUNDED_WGSL_BODY}").into()),
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto shadow shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{SHADOW_WGSL_BODY}").into()),
        });
        let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pluto rect pipeline"),
            layout: Some(&rect_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rect_shader,
                entry_point: Some("rect_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RectVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &rect_shader,
                entry_point: Some("rect_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pluto glyph pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &glyph_shader,
                entry_point: Some("glyph_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GlyphVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &glyph_shader,
                entry_point: Some("glyph_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        // Colour glyphs (Stage 8 Rec 11): the same vertex layout
        // and bind-group layout as the mask glyph pipeline, with
        // the colour fragment shader sampling the colour atlas.
        let colour_glyph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto colour glyph shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{COLOUR_GLYPH_WGSL_BODY}").into()),
        });
        let colour_glyph_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pluto colour glyph pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &colour_glyph_shader,
                    entry_point: Some("glyph_vs"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GlyphVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &colour_glyph_shader,
                    entry_point: Some("glyph_fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto image shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{CLIP_WGSL}{IMAGE_WGSL_BODY}").into()),
        });
        // Images sample a texture like glyphs do, so the image
        // pipeline reuses the glyph pipeline's two-group layout
        // (uniform + clip, texture/sampler); each image texture gets its
        // own bind group built against the atlas layout.
        let image_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pluto image pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &image_shader,
                entry_point: Some("image_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ImageVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &image_shader,
                entry_point: Some("image_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Rounded-rect fills (Stage 8 Rec 4): the rect + radii
        // ride per vertex; the fragment evaluates the SDF for
        // coverage (1 px AA) and applies the clip on top.
        let rounded_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pluto rounded pipeline"),
            layout: Some(&rect_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rounded_shader,
                entry_point: Some("rounded_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RoundedVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 40,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 56,
                            shader_location: 4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &rounded_shader,
                entry_point: Some("rounded_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        // Box shadows (Stage 8 Rec 4): one erf closed-form shader
        // for blurred and sharp shadows; the shape (already
        // expanded by spread and translated by the offset) rides
        // per vertex with `(sigma, inset)`.
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pluto shadow pipeline"),
            layout: Some(&rect_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("shadow_vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ShadowVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 40,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 56,
                            shader_location: 4,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 72,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 88,
                            shader_location: 6,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 104,
                            shader_location: 7,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 120,
                            shader_location: 8,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: Some("shadow_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // The font-id map for resolving wire glyph keys: built from
        // this renderer's own FontSystem, which loads the same system
        // fonts in the same order as layout's.
        let font_system = FontSystem::new();
        let font_ids = font_id_map(&font_system);

        Self {
            rect_pipeline,
            glyph_pipeline,
            colour_glyph_pipeline,
            image_pipeline,
            rounded_pipeline,
            shadow_pipeline,
            uniform_buffer,
            uniform_clip_bind_group_layout: uniform_bind_group_layout,
            atlas_texture,
            atlas_bind_group,
            colour_atlas_texture,
            colour_atlas_bind_group,
            atlas_bind_group_layout,
            atlas_sampler,
            mask_atlas: GlyphAtlas::new(),
            colour_atlas: GlyphAtlas::new(),
            font_system,
            font_ids,
            swash: SwashCache::new(),
            rect_buffer: None,
            glyph_buffer: None,
            image_buffer: None,
            rounded_buffer: None,
            shadow_buffer: None,
            runs: Vec::new(),
            clip_groups: Vec::new(),
            image_textures: HashMap::new(),
            image_meta: ImageTextureCache::new(),
            last_prepared: None,
            viewport: (1.0, 1.0),
            scroll: (0.0, 0.0),
            target_format: format,
            atlas_full_skips: 0,
            colour_atlas_full_skips: 0,
            subpixel_skips: 0,
            image_skips: 0,
            items_culled: 0,
        }
    }

    /// Mark an animated image's texture dirty so the next
    /// `prepare` re-uploads just that image (Stage 8 Rec 12).
    /// The caller advances the frame first — in the browser cache
    /// ([`crate::ImageCache::advance_animated`]) so the resource
    /// table already holds the new frame's pixels — then calls
    /// this; `prepare` re-uploads the texture from the table and
    /// clears the flag. Returns `false` when the image has no
    /// uploaded texture yet: the next prepare uploads the current
    /// table pixels fresh, so there is nothing to dirty. Also
    /// invalidates the clean-skip identity, or a repeat `prepare`
    /// of the same list would skip the re-upload entirely.
    pub fn advance_image_frame(&mut self, id: ResourceId) -> bool {
        if self.image_meta.mark_frame_dirty(id) {
            self.last_prepared = None;
            true
        } else {
            false
        }
    }

    /// Upload the display list for the next frame: rebuild the
    /// vertex buffers, rasterize any new glyphs into the atlas,
    /// upload any new image/blob textures. `resources` is the table
    /// the list's image/blob ids resolve against — the shell
    /// validates the pair before calling, so a missing resource
    /// here is defensive only. `scroll` is the viewport's scroll
    /// offset (page coordinates, clamped >= 0 by the shell): it
    /// goes into the uniform, and the canvas fill is translated by
    /// it so it stays glued to the viewport while the content
    /// scrolls.
    ///
    /// One walk, in list order (Stage 8 Rec 3/4): the clip stack
    /// is maintained as items are visited, items fully outside
    /// the effective clip or the visible page region are culled
    /// on the CPU (never uploaded), and vertices land in
    /// per-pipeline buffers with an ordered run list (`runs`) that
    /// `render` walks — paint order across pipelines preserved.
    /// A repeat `prepare` of the same list, viewport, and scroll
    /// is a clean skip (Rec 1/5): nothing is rebuilt.
    pub fn prepare(
        &mut self,
        list: &DisplayList,
        resources: &ResourceTable,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (f32, f32),
        scroll: (f32, f32),
    ) {
        let started = Instant::now();
        // Clean skip (Stage 8 Rec 1/5): the same list, viewport,
        // and scroll as last time paints nothing new — the buffers
        // and runs from the last prepare stay valid.
        let identity = (list as *const DisplayList as usize, viewport, scroll);
        if self.last_prepared == Some(identity) {
            if let Ok(mut counters) = perf::global().lock() {
                counters.frames_clean_skipped += 1;
            }
            return;
        }
        // An empty damage rect paints nothing new either: clear
        // the runs so the old frame stops drawing, and skip.
        if let Some(damage) = list.1
            && damage.is_empty()
        {
            self.last_prepared = Some(identity);
            self.runs.clear();
            self.rect_buffer = None;
            self.glyph_buffer = None;
            self.image_buffer = None;
            self.rounded_buffer = None;
            self.shadow_buffer = None;
            if let Ok(mut counters) = perf::global().lock() {
                counters.frames_clean_skipped += 1;
            }
            return;
        }
        self.last_prepared = Some(identity);

        if viewport != self.viewport || scroll != self.scroll {
            self.viewport = viewport;
            self.scroll = scroll;
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[viewport.0, viewport.1, scroll.0, scroll.1]),
            );
        }

        // The visible page region, in page coordinates: the window
        // onto the page the viewport shows. Items are culled
        // against this (not [0,0,vw,vh] — the page scrolls).
        let visible: [f32; 4] = [scroll.0, scroll.1, viewport.0, viewport.1];
        let damage_rect: Option<[f32; 4]> =
            list.1.map(|d| [d.x, d.y, d.width, d.height]);

        let mut rect_vertices: Vec<RectVertex> = Vec::new();
        let mut glyph_vertices: Vec<GlyphVertex> = Vec::new();
        let mut image_vertices: Vec<ImageVertex> = Vec::new();
        let mut rounded_vertices: Vec<RoundedVertex> = Vec::new();
        let mut shadow_vertices: Vec<ShadowVertex> = Vec::new();
        let mut runs: Vec<DrawRun> = Vec::new();
        let mut clip_groups: Vec<wgpu::BindGroup> = Vec::new();
        let mut clip_data: Vec<ClipUniformData> = Vec::new();
        let mut clip_stack: Vec<ClipEntry> = Vec::new();
        // The current run's clip, recomputed when the stack
        // changes: the clip bind-group index, the page-space
        // scissor (already intersected with the visible region
        // and the damage), and whether the stack clips everything
        // (empty rect intersection — items are culled, never drawn
        // with a zero-size scissor).
        let mut clip_index = 0usize;
        let mut scissor: Option<[f32; 4]> = None;
        let mut clip_empty = false;
        // The empty clip (no rounded slots) is always group 0.
        let empty_data = ClipUniformData::empty();
        clip_data.push(empty_data);
        clip_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pluto uniform + clip bind group"),
            layout: &self.uniform_clip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::Buffer::as_entire_binding(
                        &device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("pluto clip uniform"),
                            contents: bytemuck::cast_slice(&[empty_data]),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        }),
                    ),
                },
            ],
        }));
        let mut culled: u64 = 0;
        let mut atlas_uploads: u64 = 0;
        let mut colour_uploads: u64 = 0;
        let mut image_uploads: u64 = 0;
        let mut glyphs_in_list = 0usize;
        let mut has_content = false;

        // New frame (Stage 8 Rec 11/12): advance the LRU
        // generations and clear the in-use sets, so eviction can
        // never pull a glyph or a texture out from under the frame
        // being built.
        self.mask_atlas.begin_frame();
        self.colour_atlas.begin_frame();
        self.image_meta.begin_frame();

        for (i, item) in list.0.iter().enumerate() {
            match item {
                DisplayItem::PushClip {
                    x,
                    y,
                    width,
                    height,
                    radii,
                } => {
                    clip_stack.push(ClipEntry {
                        rect: [*x, *y, *width, *height],
                        radii: *radii,
                    });
                    // Recompute the effective clip for the items
                    // that follow. `clip_empty` also covers a rect
                    // clip that misses the visible region entirely:
                    // the rect clip is enforced only through the
                    // scissor (the shader tests the rounded slots
                    // alone), so a scissor of `None` there would
                    // draw items the clip excludes.
                    let (rect, rounded) = clip_stack_effective(&clip_stack);
                    clip_empty = matches!(rect, Some(r) if r[2] <= 0.0 || r[3] <= 0.0)
                        || matches!(rect, Some(r) if intersect_rect(r, visible).is_none());
                    let data = clip_uniform_data(&rounded);
                    clip_index = match clip_data.iter().position(|d| *d == data) {
                        Some(idx) => idx,
                        None => {
                            let idx = clip_data.len();
                            clip_data.push(data);
                            clip_groups.push(device.create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("pluto uniform + clip bind group"),
                                    layout: &self.uniform_clip_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: self.uniform_buffer.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::Buffer::as_entire_binding(
                                                &device.create_buffer_init(
                                                    &wgpu::util::BufferInitDescriptor {
                                                        label: Some("pluto clip uniform"),
                                                        contents: bytemuck::cast_slice(&[data]),
                                                        usage: wgpu::BufferUsages::UNIFORM
                                                            | wgpu::BufferUsages::COPY_DST,
                                                    },
                                                ),
                                            ),
                                        },
                                    ],
                                },
                            ));
                            idx
                        }
                    };
                    scissor = rect.and_then(|r| {
                        intersect_rect(r, visible).and_then(|r| match damage_rect {
                            Some(d) => intersect_rect(r, d),
                            None => Some(r),
                        })
                    });
                    continue;
                }
                DisplayItem::PopClip => {
                    clip_stack.pop();
                    let (rect, rounded) = clip_stack_effective(&clip_stack);
                    clip_empty = matches!(rect, Some(r) if r[2] <= 0.0 || r[3] <= 0.0)
                        || matches!(rect, Some(r) if intersect_rect(r, visible).is_none());
                    let data = clip_uniform_data(&rounded);
                    clip_index = match clip_data.iter().position(|d| *d == data) {
                        Some(idx) => idx,
                        None => {
                            let idx = clip_data.len();
                            clip_data.push(data);
                            clip_groups.push(device.create_bind_group(
                                &wgpu::BindGroupDescriptor {
                                    label: Some("pluto uniform + clip bind group"),
                                    layout: &self.uniform_clip_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: self.uniform_buffer.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::Buffer::as_entire_binding(
                                                &device.create_buffer_init(
                                                    &wgpu::util::BufferInitDescriptor {
                                                        label: Some("pluto clip uniform"),
                                                        contents: bytemuck::cast_slice(&[data]),
                                                        usage: wgpu::BufferUsages::UNIFORM
                                                            | wgpu::BufferUsages::COPY_DST,
                                                    },
                                                ),
                                            ),
                                        },
                                    ],
                                },
                            ));
                            idx
                        }
                    };
                    scissor = rect.and_then(|r| {
                        intersect_rect(r, visible).and_then(|r| match damage_rect {
                            Some(d) => intersect_rect(r, d),
                            None => Some(r),
                        })
                    });
                    continue;
                }
                _ => {}
            }
            if clip_empty {
                // The clip stack's rect intersection is empty:
                // nothing under it can paint.
                culled += 1;
                continue;
            }
            // Item 0 is the canvas fill (DisplayList::build pushes
            // it first): translate it by the scroll so the shader's
            // subtraction leaves it covering the viewport. Every
            // other rect scrolls with the content. The canvas is
            // never culled — it is the viewport background.
            if i == 0 {
                let first = rect_vertices.len() as u32;
                let verts = rect_vertices_for(&canvas_rect(item, scroll));
                let count = verts.len() as u32;
                rect_vertices.extend(verts);
                if count > 0 {
                    extend_run(
                        &mut runs,
                        RunKind::Rect,
                        clip_index,
                        None,
                        first,
                        count,
                        None,
                    );
                }
                continue;
            }
            // CPU cull: an item fully outside the visible region
            // or the effective rect clip never uploads. Glyphs
            // have no item bounds — their quads are culled
            // per-glyph below.
            if let Some(bounds) = item.bounds()
                && !bounds_visible(bounds, scissor, visible)
            {
                culled += 1;
                continue;
            }
            match item {
                DisplayItem::FillRect { .. } => {
                    has_content = true;
                    let first = rect_vertices.len() as u32;
                    let verts = rect_vertices_for(item);
                    let count = verts.len() as u32;
                    rect_vertices.extend(verts);
                    extend_run(
                        &mut runs,
                        RunKind::Rect,
                        clip_index,
                        scissor,
                        first,
                        count,
                        None,
                    );
                }
                DisplayItem::RoundedRect { .. } => {
                    has_content = true;
                    let first = rounded_vertices.len() as u32;
                    let verts = rounded_vertices_for(item);
                    let count = verts.len() as u32;
                    rounded_vertices.extend(verts);
                    extend_run(
                        &mut runs,
                        RunKind::Rounded,
                        clip_index,
                        scissor,
                        first,
                        count,
                        None,
                    );
                }
                DisplayItem::BoxShadow { .. } => {
                    has_content = true;
                    let first = shadow_vertices.len() as u32;
                    let verts = shadow_vertices_for(item);
                    let count = verts.len() as u32;
                    shadow_vertices.extend(verts);
                    extend_run(
                        &mut runs,
                        RunKind::Shadow,
                        clip_index,
                        scissor,
                        first,
                        count,
                        None,
                    );
                }
                DisplayItem::Glyphs { color, glyphs } => {
                    has_content = true;
                    for glyph in glyphs {
                        glyphs_in_list += 1;
                        // The wire key resolves back to a cache key
                        // through this renderer's font database. An
                        // unresolvable key (a font the renderer
                        // never loaded) is skipped like a cache
                        // miss — never drawn wrong.
                        let Some(cache_key) = glyph.key.to_cache_key(&self.font_ids) else {
                            continue;
                        };
                        let Some(image) = self
                            .swash
                            .get_image(&mut self.font_system, cache_key)
                            .as_ref()
                        else {
                            continue;
                        };
                        // Colour glyphs (emoji) go to the second
                        // atlas (Rec 11); subpixel masks are
                        // skipped and counted, not drawn wrong.
                        let Some(kind) = atlas_kind_for(image.content) else {
                            self.subpixel_skips += 1;
                            continue;
                        };
                        let placement = image.placement;
                        if placement.width == 0 || placement.height == 0 {
                            continue;
                        }
                        // A rasterised glyph whose data does not
                        // match its placement is corrupt input:
                        // skip it rather than slicing out of bounds.
                        let expected_len = match kind {
                            AtlasKind::Mask => {
                                (placement.width as usize).checked_mul(placement.height as usize)
                            }
                            AtlasKind::Colour => (placement.width as usize)
                                .checked_mul(placement.height as usize)
                                .and_then(|n| n.checked_mul(4)),
                        };
                        if expected_len != Some(image.data.len()) {
                            match kind {
                                AtlasKind::Mask => self.atlas_full_skips += 1,
                                AtlasKind::Colour => self.colour_atlas_full_skips += 1,
                            }
                            continue;
                        }
                        let (x0, y0, x1, y1) = glyph_quad(glyph, placement);
                        // Per-quad cull: a glyph fully outside the
                        // visible region or the clip rect never
                        // uploads (its atlas slot stays free).
                        if !bounds_visible([x0, y0, x1 - x0, y1 - y0], scissor, visible) {
                            culled += 1;
                            continue;
                        }
                        // Allocate in the glyph's own atlas (Rec
                        // 11): growth before eviction, in-use
                        // glyphs never evicted, allocation failure
                        // skips just this glyph.
                        let is_new = match kind {
                            AtlasKind::Mask => !self.mask_atlas.contains(&cache_key),
                            AtlasKind::Colour => !self.colour_atlas.contains(&cache_key),
                        };
                        let rect = match kind {
                            AtlasKind::Mask => self.mask_atlas.allocate(
                                cache_key,
                                placement.width,
                                placement.height,
                            ),
                            AtlasKind::Colour => self.colour_atlas.allocate(
                                cache_key,
                                placement.width,
                                placement.height,
                            ),
                        };
                        let Some(rect) = rect else {
                            match kind {
                                AtlasKind::Mask => self.atlas_full_skips += 1,
                                AtlasKind::Colour => self.colour_atlas_full_skips += 1,
                            }
                            continue;
                        };
                        if is_new {
                            match kind {
                                AtlasKind::Mask => {
                                    upload_glyph(queue, &self.atlas_texture, rect, &image.data);
                                    atlas_uploads += 1;
                                }
                                AtlasKind::Colour => {
                                    upload_colour_glyph(
                                        queue,
                                        &self.colour_atlas_texture,
                                        rect,
                                        &image.data,
                                    );
                                    colour_uploads += 1;
                                }
                            }
                        }
                        // UVs are over the texture size (2048):
                        // the textures are allocated at the maximum
                        // up front, so growth never moves a UV.
                        let (u0, v0) = (
                            rect.x as f32 / ATLAS_TEXTURE_SIZE as f32,
                            rect.y as f32 / ATLAS_TEXTURE_SIZE as f32,
                        );
                        let (u1, v1) = (
                            (rect.x + rect.w) as f32 / ATLAS_TEXTURE_SIZE as f32,
                            (rect.y + rect.h) as f32 / ATLAS_TEXTURE_SIZE as f32,
                        );
                        let c = *color;
                        let first = glyph_vertices.len() as u32;
                        glyph_vertices.extend([
                            GlyphVertex {
                                pos: [x0, y0],
                                uv: [u0, v0],
                                color: c,
                            },
                            GlyphVertex {
                                pos: [x1, y0],
                                uv: [u1, v0],
                                color: c,
                            },
                            GlyphVertex {
                                pos: [x0, y1],
                                uv: [u0, v1],
                                color: c,
                            },
                            GlyphVertex {
                                pos: [x1, y0],
                                uv: [u1, v0],
                                color: c,
                            },
                            GlyphVertex {
                                pos: [x1, y1],
                                uv: [u1, v1],
                                color: c,
                            },
                            GlyphVertex {
                                pos: [x0, y1],
                                uv: [u0, v1],
                                color: c,
                            },
                        ]);
                        // One run per glyph: `extend_run` merges
                        // contiguous same-pipeline runs, and mask
                        // and colour glyphs must never merge (they
                        // draw from different atlases).
                        extend_run(
                            &mut runs,
                            match kind {
                                AtlasKind::Mask => RunKind::Glyph,
                                AtlasKind::Colour => RunKind::ColourGlyph,
                            },
                            clip_index,
                            scissor,
                            first,
                            6,
                            None,
                        );
                    }
                }
                DisplayItem::Image { resource, .. } | DisplayItem::Blob { resource, .. } => {
                    has_content = true;
                    // Resolve the id in the resource table. A
                    // missing resource or mismatched pixels is
                    // skipped and counted — the shell validates
                    // first, so this is a bug or a race, and a
                    // skipped image must never take the frame down
                    // with it. Culled images never reach this
                    // branch at all (the bounds check above): a
                    // culled image is neither uploaded nor bound.
                    let Some(Resource::Image {
                        width,
                        height,
                        rgba,
                    }) = resources.get(*resource)
                    else {
                        self.image_skips += 1;
                        continue;
                    };
                    if rgba.len() != (*width as usize) * (*height as usize) * 4 {
                        self.image_skips += 1;
                        continue;
                    }
                    // The draw scale for mip selection (Rec 12):
                    // the item's draw size over the texture's
                    // native size, smaller axis — the level must
                    // not be coarser than the tighter axis needs.
                    let (draw_width, draw_height) = match item {
                        DisplayItem::Image { width, height, .. }
                        | DisplayItem::Blob { width, height, .. } => (*width, *height),
                        _ => unreachable!("image branch"),
                    };
                    let scale = (draw_width / *width as f32).min(draw_height / *height as f32);
                    if self.image_meta.is_dirty(*resource) {
                        // Animated frame advance (Rec 12): the
                        // table already holds the new frame's
                        // pixels, so re-upload just this image's
                        // mip chain into its existing texture.
                        if let Some(entry) = self.image_textures.get(resource) {
                            let decoded = DecodedImage {
                                width: *width,
                                height: *height,
                                rgba: rgba.clone(),
                            };
                            let chain = build_mip_chain(&decoded);
                            upload_image_mips(queue, &entry.texture, &chain);
                            image_uploads += 1;
                        }
                        self.image_meta.clear_dirty(*resource);
                        self.image_meta.touch(*resource);
                    } else if !self.image_meta.contains(*resource) {
                        // First sight: build the mip chain, evict
                        // LRU textures not drawn this frame until
                        // it fits under the 64 MiB cap, then upload
                        // every level and keep one bind group per
                        // level (each viewing only that level).
                        let decoded = DecodedImage {
                            width: *width,
                            height: *height,
                            rgba: rgba.clone(),
                        };
                        let chain = build_mip_chain(&decoded);
                        let chain_bytes = mip_chain_bytes(&chain);
                        for evicted in self.image_meta.evict_for(chain_bytes, IMAGE_CACHE_BYTE_CAP)
                        {
                            self.image_textures.remove(&evicted);
                        }
                        let mip_levels = chain.len() as u32;
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("pluto image texture"),
                            size: wgpu::Extent3d {
                                width: *width,
                                height: *height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: mip_levels,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: image_texture_format(self.target_format),
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        upload_image_mips(queue, &texture, &chain);
                        image_uploads += 1;
                        let bind_groups = (0..mip_levels)
                            .map(|level| {
                                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                                    label: Some("pluto image mip view"),
                                    base_mip_level: level,
                                    mip_level_count: Some(1),
                                    ..Default::default()
                                });
                                device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("pluto image bind group"),
                                    layout: &self.atlas_bind_group_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(&view),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(
                                                &self.atlas_sampler,
                                            ),
                                        },
                                    ],
                                })
                            })
                            .collect();
                        self.image_textures.insert(
                            *resource,
                            CachedImage {
                                texture,
                                bind_groups,
                            },
                        );
                        self.image_meta.register(*resource, chain_bytes, mip_levels);
                    } else {
                        self.image_meta.touch(*resource);
                    }
                    // Pick the mip level for this draw's scale and
                    // record one draw for this item (its level's
                    // bind group + vertex range).
                    let mip_levels = self.image_meta.mip_levels(*resource).unwrap_or(1);
                    let level = mip_level_for_scale(scale, mip_levels.saturating_sub(1));
                    let Some(entry) = self.image_textures.get(resource) else {
                        self.image_skips += 1;
                        continue;
                    };
                    let Some(bind_group) = entry.bind_groups.get(level as usize) else {
                        self.image_skips += 1;
                        continue;
                    };
                    // The borrow of `self.image_textures` ends
                    // before the next iteration mutates it: clone
                    // the bind group out here.
                    let bind_group = bind_group.clone();
                    let first = image_vertices.len() as u32;
                    let verts = image_vertices_for(item);
                    let count = verts.len() as u32;
                    image_vertices.extend(verts);
                    extend_run(
                        &mut runs,
                        RunKind::Image,
                        clip_index,
                        scissor,
                        first,
                        count,
                        Some(bind_group),
                    );
                }
                DisplayItem::PushClip { .. } | DisplayItem::PopClip => {
                    unreachable!("clip items handled above")
                }
            }
        }

        self.runs = runs;
        self.clip_groups = clip_groups;
        self.items_culled = culled;
        self.rect_buffer = make_buffer(device, "pluto rect vertices", &rect_vertices);
        self.glyph_buffer = make_buffer(device, "pluto glyph vertices", &glyph_vertices);
        self.image_buffer = make_buffer(device, "pluto image vertices", &image_vertices);
        self.rounded_buffer =
            make_buffer(device, "pluto rounded vertices", &rounded_vertices);
        self.shadow_buffer = make_buffer(device, "pluto shadow vertices", &shadow_vertices);

        // Frame instrumentation (Stage 8 Rec 5, report §i).
        if let Ok(mut counters) = perf::global().lock() {
            counters.frames_produced += 1;
            counters.items_culled = culled;
            counters.atlas_uploads = atlas_uploads;
            counters.atlas_evictions =
                (self.mask_atlas.evictions() + self.colour_atlas.evictions()) as u64;
            counters.atlas_growths =
                (self.mask_atlas.growths() + self.colour_atlas.growths()) as u64;
            counters.colour_atlas_uploads = colour_uploads;
            counters.colour_glyphs_skipped = self.colour_atlas_full_skips as u64;
            counters.glyphs_skipped =
                (self.atlas_full_skips + self.colour_atlas_full_skips) as u64;
            counters.image_cache_evictions = self.image_meta.evictions();
            counters.image_cache_bytes = self.image_meta.bytes_used();
            counters.image_uploads = image_uploads;
            counters.display_list_items = list.0.len();
            counters.display_list_glyphs = glyphs_in_list;
            counters.display_list_resources = resources.len();
            counters.note_damage(list.1, viewport);
            counters.note_frame(has_content);
        }
        perf::record_span(
            SpanKind::ShellPrepare,
            started.elapsed().as_secs_f32() * 1000.0,
        );
    }

    /// Draw the prepared frame into an already-begun render pass,
    /// walking the runs in paint order (Stage 8 Rec 3/4): each run
    /// sets its scissor (page-space rect → target pixels, or the
    /// whole widget when the run has none), its pipeline, and its
    /// bind groups — the clip group at group 1 (rect/rounded/
    /// shadow) or group 2 (glyph/image). `widget` is the target
    /// rect `(x, y, w, h)` in physical pixels and `scale` the
    /// physical-per-layout-px factor (the caller derives it from
    /// its clip bounds); the viewport/scissor the caller set
    /// before the call is the outer bound, and every run scissor
    /// is inside it.
    pub fn render(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        widget: (u32, u32, u32, u32),
        scale: f32,
    ) {
        let started = Instant::now();
        for run in &self.runs {
            // The run's page-space scissor to target pixels, or
            // the whole widget when the run has none. A scissor
            // that rounds to empty skips the run — never a
            // zero-size scissor.
            let scissor = match run.scissor {
                Some(rect) => match scissor_physical(rect, self.scroll, widget, scale) {
                    Some(s) => s,
                    None => continue,
                },
                None => widget,
            };
            pass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
            let range = run.first_vertex..run.first_vertex + run.vertex_count;
            match run.kind {
                RunKind::Rect => {
                    let Some(buffer) = &self.rect_buffer else {
                        continue;
                    };
                    pass.set_pipeline(&self.rect_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
                RunKind::Glyph => {
                    let Some(buffer) = &self.glyph_buffer else {
                        continue;
                    };
                    pass.set_pipeline(&self.glyph_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_bind_group(1, &self.atlas_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
                RunKind::ColourGlyph => {
                    let Some(buffer) = &self.glyph_buffer else {
                        continue;
                    };
                    pass.set_pipeline(&self.colour_glyph_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_bind_group(1, &self.colour_atlas_bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
                RunKind::Image => {
                    let Some(buffer) = &self.image_buffer else {
                        continue;
                    };
                    let Some(bind_group) = &run.image_bind_group else {
                        continue;
                    };
                    pass.set_pipeline(&self.image_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_bind_group(1, bind_group, &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
                RunKind::Rounded => {
                    let Some(buffer) = &self.rounded_buffer else {
                        continue;
                    };
                    pass.set_pipeline(&self.rounded_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
                RunKind::Shadow => {
                    let Some(buffer) = &self.shadow_buffer else {
                        continue;
                    };
                    pass.set_pipeline(&self.shadow_pipeline);
                    pass.set_bind_group(0, &self.clip_groups[run.clip], &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(range, 0..1);
                }
            }
        }
        perf::record_span(SpanKind::ShellEncode, started.elapsed().as_secs_f32() * 1000.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DisplayItem;
    use cosmic_text::{Attrs, Buffer, Metrics, Shaping};

    /// Real cache keys, shaped from a real font (DejaVu ships in the
    /// sandbox), so the test does not guess at `CacheKey`'s fields.
    fn shaped_keys() -> Vec<cosmic_text::CacheKey> {
        let mut font_system = FontSystem::new();
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(16.0, 19.2));
        buffer.set_text(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer.set_size(None, None);
        buffer.shape_until_scroll(&mut font_system, false);
        let mut keys = Vec::new();
        for run in buffer.layout_runs() {
            for g in run.glyphs {
                keys.push(g.physical((0.0, 0.0), 1.0).cache_key);
            }
        }
        keys
    }

    #[test]
    fn rect_vertices_cover_the_rect() {
        let verts = rect_vertices_for(&DisplayItem::FillRect {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
            color: [1.0, 0.0, 0.0, 1.0],
        });
        assert_eq!(verts.len(), 6);
        let xs: Vec<f32> = verts.iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = verts.iter().map(|v| v.pos[1]).collect();
        assert!(xs.contains(&10.0) && xs.contains(&40.0));
        assert!(ys.contains(&20.0) && ys.contains(&60.0));
        assert!(verts.iter().all(|v| v.color == [1.0, 0.0, 0.0, 1.0]));
        assert!(
            rect_vertices_for(&DisplayItem::Glyphs {
                color: [0.0; 4],
                glyphs: vec![],
            })
            .is_empty()
        );
    }

    /// Distinct cache keys, derived from one shaped key by varying
    /// the glyph id (offset so they never collide with the shaped
    /// keys themselves).
    fn distinct_keys(n: usize) -> Vec<cosmic_text::CacheKey> {
        let base = shaped_keys()[0];
        (0..n)
            .map(|i| cosmic_text::CacheKey {
                glyph_id: base.glyph_id.wrapping_add(1000 + i as u16),
                ..base
            })
            .collect()
    }

    #[test]
    fn glyph_atlas_packing_never_overlaps() {
        let keys = shaped_keys();
        assert!(!keys.is_empty(), "shaping produced no glyphs");
        let mut atlas = GlyphAtlas::new();
        atlas.begin_frame();
        let mut placed = Vec::new();
        // Pack every shaped glyph (plus repeats) at mixed sizes; none may
        // overlap, none may leave the atlas, and the same key always
        // returns the same rect.
        for (i, k) in keys.iter().enumerate() {
            let (w, h) = (8 + (i as u32 % 17), 8 + (i as u32 % 11));
            let r = atlas
                .allocate(*k, w, h)
                .unwrap_or_else(|| panic!("atlas full at glyph {i}"));
            placed.push(r);
            assert_eq!(atlas.allocate(*k, w, h), Some(r));
        }
        for (i, a) in placed.iter().enumerate() {
            assert!(a.x + a.w <= ATLAS_TEXTURE_SIZE && a.y + a.h <= ATLAS_TEXTURE_SIZE);
            for b in placed.iter().skip(i + 1) {
                let overlap =
                    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
                assert!(!overlap, "atlas rects {a:?} and {b:?} overlap");
            }
        }
        // An oversized glyph (larger than the maximum atlas) never fits.
        let mut big = keys[0];
        big.glyph_id = big.glyph_id.wrapping_add(1);
        assert!(atlas.allocate(big, 3000, 3000).is_none());
        // A full-size glyph fits an empty 2048 atlas (growth first).
        let mut fresh = GlyphAtlas::new();
        assert!(fresh.allocate(big, 2048, 2048).is_some());
        assert_eq!(fresh.size(), 2048);
    }

    #[test]
    fn glyph_atlas_lru_evicts_oldest_and_never_in_use() {
        // 64x64 atlas, 8x8 glyphs: small enough to fill fast.
        let mut atlas = GlyphAtlas::with_size(64, 64);
        let keys = distinct_keys(200);
        // Fill over successive frames (one glyph per frame, so each
        // gets its own generation — the LRU order) until the first
        // eviction: the atlas was full and the oldest glyph went.
        let mut filled_until = 0usize;
        for (i, key) in keys.iter().enumerate() {
            atlas.begin_frame();
            assert!(
                atlas.allocate(*key, 8, 8).is_some(),
                "allocation {i} succeeds"
            );
            if atlas.evictions() > 0 {
                filled_until = i;
                break;
            }
        }
        assert!(
            filled_until > 10,
            "atlas filled after {filled_until} glyphs"
        );
        let oldest = keys[0];
        assert!(
            !atlas.contains(&oldest),
            "the oldest glyph was evicted first"
        );
        // The glyph allocated just before the eviction is the most
        // recent survivor; a new frame that uses it must never
        // evict it, whatever else is allocated.
        let recent = keys[filled_until - 1];
        assert!(atlas.contains(&recent), "recent glyph survives the fill");
        atlas.begin_frame();
        assert_eq!(atlas.allocate(recent, 8, 8), atlas.rect_for(&recent));
        let evictions_before = atlas.evictions();
        for key in keys.iter().skip(filled_until + 1).take(20) {
            atlas.allocate(*key, 8, 8);
            if atlas.evictions() > evictions_before {
                break;
            }
        }
        assert!(
            atlas.evictions() > evictions_before,
            "fresh allocations evicted something"
        );
        assert!(
            atlas.contains(&recent),
            "a glyph in use this frame is never evicted"
        );
    }

    #[test]
    fn glyph_atlas_grows_before_evicting() {
        // 256x256 glyphs: ~9 fit at 1024, ~49 at 2048.
        let mut atlas = GlyphAtlas::with_size(1024, 2048);
        let keys = distinct_keys(200);
        let mut grew_at = None;
        let mut first_eviction_at = None;
        for (i, key) in keys.iter().enumerate() {
            atlas.begin_frame();
            assert!(
                atlas.allocate(*key, 256, 256).is_some(),
                "allocation {i} succeeds via growth or eviction"
            );
            if grew_at.is_none() && atlas.size() == 2048 {
                grew_at = Some(i);
                assert_eq!(atlas.evictions(), 0, "nothing evicted before growth");
                assert_eq!(atlas.growths(), 1);
            }
            if first_eviction_at.is_none() && atlas.evictions() > 0 {
                first_eviction_at = Some(i);
                break;
            }
        }
        let grew = grew_at.expect("atlas grew to 2048");
        let evicted = first_eviction_at.expect("atlas eventually evicted");
        assert!(
            grew < evicted,
            "growth at glyph {grew} must precede the first eviction at glyph {evicted}"
        );
        assert_eq!(atlas.size(), 2048);
        assert_eq!(atlas.growths(), 1, "growth happens once, 1024 -> 2048");
    }

    #[test]
    fn colour_glyphs_land_in_the_second_atlas() {
        assert_eq!(atlas_kind_for(SwashContent::Mask), Some(AtlasKind::Mask));
        assert_eq!(atlas_kind_for(SwashContent::Color), Some(AtlasKind::Colour));
        assert_eq!(atlas_kind_for(SwashContent::SubpixelMask), None);
        // The two atlases are independent: the same key lives in
        // both, and evicting it from the mask atlas leaves the
        // colour atlas untouched.
        let mut mask = GlyphAtlas::with_size(64, 64);
        let mut colour = GlyphAtlas::with_size(64, 64);
        let key = shaped_keys()[0];
        mask.begin_frame();
        colour.begin_frame();
        mask.allocate(key, 16, 16).expect("mask allocates");
        let colour_rect = colour.allocate(key, 16, 16).expect("colour allocates");
        let keys = distinct_keys(200);
        for k in &keys {
            mask.begin_frame();
            mask.allocate(*k, 16, 16);
            if !mask.contains(&key) {
                break;
            }
        }
        assert!(!mask.contains(&key), "mask atlas evicted the key");
        assert!(colour.contains(&key), "colour atlas still holds it");
        assert_eq!(colour.rect_for(&key), Some(colour_rect));
    }

    #[test]
    fn image_texture_cache_evicts_lru_and_never_in_use() {
        let mut cache = ImageTextureCache::new();
        // Distinct generations: one frame per registration.
        cache.begin_frame();
        cache.register(ResourceId(1), 100, 1);
        cache.begin_frame();
        cache.register(ResourceId(2), 100, 1);
        cache.begin_frame();
        cache.register(ResourceId(3), 100, 1);
        assert_eq!(cache.bytes_used(), 300);
        // A new frame that draws image 1: it is the most recent
        // and in use, so the cap evicts 2 then 3 (LRU order).
        cache.begin_frame();
        cache.touch(ResourceId(1));
        let evicted = cache.evict_for(100, 250);
        assert_eq!(evicted, vec![ResourceId(2), ResourceId(3)]);
        assert!(cache.contains(ResourceId(1)));
        assert!(!cache.contains(ResourceId(2)));
        assert_eq!(cache.bytes_used(), 100);
        assert_eq!(cache.evictions(), 2);
        // In-use protection: 2 and 3 are drawn this frame, so only
        // 1 (not drawn) can go — even though the cap is still
        // exceeded after it.
        let mut cache = ImageTextureCache::new();
        cache.begin_frame();
        cache.register(ResourceId(1), 100, 1);
        cache.begin_frame();
        cache.register(ResourceId(2), 100, 1);
        cache.begin_frame();
        cache.register(ResourceId(3), 100, 1);
        cache.begin_frame();
        cache.touch(ResourceId(2));
        cache.touch(ResourceId(3));
        let evicted = cache.evict_for(100, 250);
        assert_eq!(evicted, vec![ResourceId(1)]);
        assert!(cache.contains(ResourceId(2)));
        assert!(cache.contains(ResourceId(3)));
        assert_eq!(cache.bytes_used(), 200);
    }

    #[test]
    fn image_texture_cache_frame_dirty_is_per_image() {
        let mut cache = ImageTextureCache::new();
        cache.begin_frame();
        cache.register(ResourceId(1), 100, 1);
        cache.register(ResourceId(2), 100, 1);
        // Advancing a frame dirties only that image.
        assert!(cache.mark_frame_dirty(ResourceId(1)));
        assert!(cache.is_dirty(ResourceId(1)));
        assert!(!cache.is_dirty(ResourceId(2)));
        // Clearing after the re-upload leaves the other untouched.
        assert!(cache.clear_dirty(ResourceId(1)));
        assert!(!cache.is_dirty(ResourceId(1)));
        assert!(!cache.clear_dirty(ResourceId(1)), "already clear");
        // An image with no uploaded texture has nothing to dirty.
        assert!(!cache.mark_frame_dirty(ResourceId(99)));
        assert!(!cache.is_dirty(ResourceId(99)));
    }

    /// Stage 1 regression: a glyph quad's top-left must be
    /// `origin + (placement.left, -placement.top)`, pinned against
    /// cosmic-text's own `with_pixels` oracle. The old code added
    /// `placement.top`, pushing each glyph down by `2 * top` — most
    /// for overshooting round letters (O/S) — and making spacing
    /// look lumpy. The string covers flat caps (H), overshooting
    /// round caps (O S C A), x-height (x o), overhanging glyphs
    /// (f j, negative `left`), and descenders (g y p).
    #[test]
    fn image_vertices_cover_the_rect() {
        let item = DisplayItem::Image {
            resource: ResourceId(3),
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        let verts = image_vertices_for(&item);
        assert_eq!(verts.len(), 6);
        let xs: Vec<f32> = verts.iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = verts.iter().map(|v| v.pos[1]).collect();
        assert!(xs.contains(&10.0) && xs.contains(&40.0));
        assert!(ys.contains(&20.0) && ys.contains(&60.0));
        // UVs span the whole texture, top-left to bottom-right.
        assert!(verts.iter().any(|v| v.uv == [0.0, 0.0]));
        assert!(verts.iter().any(|v| v.uv == [1.0, 0.0]));
        assert!(verts.iter().any(|v| v.uv == [0.0, 1.0]));
        assert!(verts.iter().any(|v| v.uv == [1.0, 1.0]));
        // The blob escape hatch draws the same quad as an image.
        let blob = DisplayItem::Blob {
            resource: ResourceId(3),
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        assert_eq!(image_vertices_for(&blob), verts);
        // Rects and glyphs are not images.
        assert!(
            image_vertices_for(&DisplayItem::FillRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                color: [0.0; 4],
            })
            .is_empty()
        );
    }

    #[test]
    fn canvas_rect_stays_glued_to_the_viewport() {
        // Scrolled down 100px, the canvas moves down 100 in page
        // space — the shader's subtraction then leaves it exactly
        // covering the viewport — while any other rect is untouched.
        let canvas = DisplayItem::FillRect {
            x: 0.0,
            y: 0.0,
            width: 800.0,
            height: 600.0,
            color: [1.0; 4],
        };
        assert_eq!(
            canvas_rect(&canvas, (0.0, 100.0)),
            DisplayItem::FillRect {
                x: 0.0,
                y: 100.0,
                width: 800.0,
                height: 600.0,
                color: [1.0; 4],
            }
        );
        assert_eq!(canvas_rect(&canvas, (0.0, 0.0)), canvas);
        // Anything that is not a rect passes through untouched
        // (the caller only applies this to item 0, the canvas).
        let image = DisplayItem::Image {
            resource: ResourceId(1),
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        assert_eq!(canvas_rect(&image, (5.0, 5.0)), image);
    }

    #[test]
    fn image_texture_format_matches_target() {
        // sRGB swapchains get sRGB textures (sampling converts to
        // linear, the target converts back); anything else gets a
        // plain texture.
        assert_eq!(
            image_texture_format(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            image_texture_format(wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            image_texture_format(wgpu::TextureFormat::Bgra8Unorm),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            image_texture_format(wgpu::TextureFormat::Rgba8Unorm),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn glyph_quad_matches_with_pixels_oracle() {
        let text = "HOSCAxo fj gyp";
        for size in [16.0f32, 24.0] {
            let mut font_system = FontSystem::new();
            let mut swash = SwashCache::new();
            let mut buffer = Buffer::new(&mut font_system, Metrics::new(size, size * 1.2));
            buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);
            buffer.set_size(None, None);
            buffer.shape_until_scroll(&mut font_system, false);
            let mut glyphs_seen = 0;
            for run in buffer.layout_runs() {
                for g in run.glyphs {
                    // Same anchor as layout: offset carries the run's
                    // baseline, so physical.y IS the baseline.
                    let physical = g.physical((0.0, run.line_y), 1.0);
                    let Some(image) = swash
                        .get_image(&mut font_system, physical.cache_key)
                        .as_ref()
                    else {
                        continue;
                    };
                    if image.content != SwashContent::Mask {
                        continue;
                    }
                    let placement = image.placement;
                    if placement.width == 0 || placement.height == 0 {
                        continue;
                    }
                    glyphs_seen += 1;
                    let glyph = PaintGlyph {
                        key: crate::GlyphKey::from(physical.cache_key),
                        x: physical.x as f32,
                        y: physical.y as f32,
                    };
                    let (x0, y0, x1, y1) = glyph_quad(&glyph, placement);
                    // Headline formula: top-left = origin + (left, -top).
                    assert_eq!(x0, glyph.x + placement.left as f32);
                    assert_eq!(y0, glyph.y - placement.top as f32);
                    assert_eq!(x1, x0 + placement.width as f32);
                    assert_eq!(y1, y0 + placement.height as f32);
                    // The quad must straddle the baseline the right
                    // way round (top at/above it, bottom at/below it,
                    // within a pixel of hinting slack).
                    assert!(y0 <= glyph.y + 1.0, "quad top above baseline");
                    assert!(y1 >= glyph.y - 1.0, "quad bottom below baseline");
                    // Oracle: with_pixels enumerates the bitmap's
                    // first pixel at (left, -top) relative to the
                    // origin — it must land exactly on the quad's
                    // top-left corner.
                    let mut first: Option<(i32, i32)> = None;
                    swash.with_pixels(
                        &mut font_system,
                        physical.cache_key,
                        cosmic_text::Color::rgb(0, 0, 0),
                        |px, py, _| {
                            if first.is_none() {
                                first = Some((px, py));
                            }
                        },
                    );
                    let (fx, fy) = first.expect("with_pixels enumerates the bitmap");
                    assert_eq!((fx, fy), (placement.left, -placement.top));
                    assert_eq!((glyph.x + fx as f32, glyph.y + fy as f32), (x0, y0));
                }
            }
            assert!(
                glyphs_seen >= 11,
                "expected most of {text:?} shaped at {size}px, saw {glyphs_seen}"
            );
        }
    }

    /// Regression test for the Haswell crash (2026-11-24): the
    /// glyph pipeline's three-group layout ([uniform, atlas, clip])
    /// panicked at creation with "Bind group layout count 3 exceeds
    /// device bind group limit 2". Every shader must parse under
    /// naga and use only bind groups 0-1, so the fix (clip merged
    /// into group 0) cannot regress silently — the sandbox has no
    /// GPU, so this is the only automated check that the layout
    /// fits a 2-group device.
    #[test]
    fn shaders_parse_and_use_at_most_two_bind_groups() {
        let sources = [
            ("rect", format!("{CLIP_WGSL}{RECT_WGSL_BODY}")),
            ("glyph", format!("{CLIP_WGSL}{GLYPH_WGSL_BODY}")),
            (
                "colour glyph",
                format!("{CLIP_WGSL}{COLOUR_GLYPH_WGSL_BODY}"),
            ),
            ("image", format!("{CLIP_WGSL}{IMAGE_WGSL_BODY}")),
            ("rounded", format!("{CLIP_WGSL}{ROUNDED_WGSL_BODY}")),
            ("shadow", format!("{CLIP_WGSL}{SHADOW_WGSL_BODY}")),
        ];
        for (name, src) in sources {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{name} shader must parse: {e:?}"));
            for (_, var) in module.global_variables.iter() {
                if let Some(binding) = var.binding {
                    assert!(
                        binding.group < 2,
                        "{name} shader uses bind group {}, above the 2-group cap",
                        binding.group
                    );
                }
            }
        }
    }
}
