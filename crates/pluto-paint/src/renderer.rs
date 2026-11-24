//! wgpu renderer for a [`DisplayList`](crate::DisplayList): solid quads
//! plus glyph-atlas text.
//!
//! Two pipelines share one uniform (the viewport size, for pixel-to-clip
//! conversion). Rects are opaque quads. Glyphs sample a 1024x1024 R8
//! atlas of cosmic-text/Swash alpha masks, tinted by the text color.
//! The atlas is shelf-packed on demand in `prepare`; the same glyph
//! rasterized once stays cached across frames. Color glyphs (emoji) and
//! subpixel-mask glyphs are skipped in M1.
//!
//! The renderer owns its own `FontSystem` for `SwashCache` lookups: the
//! cache keys in the display list are font ids, and two `FontSystem`s
//! loading the same system fonts in the same order assign the same ids.
//! (If that ever proves untrue on a machine, the fix is to hand the
//! renderer's `FontSystem` to layout instead of building a second one.)

use crate::glyph_key::font_id_map;
use crate::{DisplayItem, DisplayList, PaintGlyph, Resource, ResourceId, ResourceTable};
use cosmic_text::{FontSystem, SwashCache, SwashContent};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const ATLAS_SIZE: u32 = 1024;
/// `queue.write_texture` requires rows padded to this many bytes.
const ROW_ALIGNMENT: usize = 256;

const RECT_WGSL: &str = r#"
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
    return out;
}

@fragment
fn rect_fs(in: RectOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const GLYPH_WGSL: &str = r#"
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
    return out;
}

@fragment
fn glyph_fs(in: GlyphOut) -> @location(0) vec4<f32> {
    let alpha = textureSample(atlas, atlas_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

const IMAGE_WGSL: &str = r#"
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
    return out;
}

@fragment
fn image_fs(in: ImageOut) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, in.uv);
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

/// One textured quad to draw: its texture's bind group plus the
/// vertex range in the shared image buffer.
struct ImageDraw {
    bind_group: wgpu::BindGroup,
    first_vertex: u32,
    vertex_count: u32,
}

/// One uploaded image/blob texture, cached by resource id.
struct CachedImage {
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AtlasRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[derive(Debug)]
struct AtlasRow {
    y: u32,
    height: u32,
    x: u32,
}

/// Shelf-pack `w`x`h` into the atlas. Returns the rect, or `None` when
/// the atlas is full (the caller skips that glyph this frame).
fn pack(
    rows: &mut Vec<AtlasRow>,
    map: &mut HashMap<cosmic_text::CacheKey, AtlasRect>,
    key: cosmic_text::CacheKey,
    w: u32,
    h: u32,
) -> Option<AtlasRect> {
    if let Some(rect) = map.get(&key) {
        return Some(*rect);
    }
    // 1px padding between glyphs so filtering never bleeds neighbors.
    for row in rows.iter_mut() {
        if row.height >= h && row.x + w + 1 <= ATLAS_SIZE {
            let rect = AtlasRect {
                x: row.x,
                y: row.y,
                w,
                h,
            };
            row.x += w + 1;
            map.insert(key, rect);
            return Some(rect);
        }
    }
    let y = rows.last().map(|r| r.y + r.height + 1).unwrap_or(0);
    if y + h > ATLAS_SIZE {
        return None;
    }
    let rect = AtlasRect { x: 0, y, w, h };
    rows.push(AtlasRow {
        y,
        height: h,
        x: w + 1,
    });
    map.insert(key, rect);
    Some(rect)
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

/// Upload RGBA8 pixels into a texture, padding rows to the 256-byte
/// alignment `queue.write_texture` requires (same pattern as
/// `upload_glyph`, four bytes per pixel instead of one).
fn upload_image(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
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
            mip_level: 0,
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
    image_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    atlas_texture: wgpu::Texture,
    atlas_bind_group: wgpu::BindGroup,
    atlas_bind_group_layout: wgpu::BindGroupLayout,
    atlas_sampler: wgpu::Sampler,
    atlas_map: HashMap<cosmic_text::CacheKey, AtlasRect>,
    atlas_rows: Vec<AtlasRow>,
    font_system: FontSystem,
    /// fontdb id (as the `u64` in [`GlyphKey`]) -> `fontdb::ID`,
    /// built once from the font database: the reverse map
    /// [`GlyphKey::to_cache_key`] needs.
    font_ids: HashMap<u64, fontdb::ID>,
    swash: SwashCache,
    rect_buffer: Option<wgpu::Buffer>,
    glyph_buffer: Option<wgpu::Buffer>,
    image_buffer: Option<wgpu::Buffer>,
    /// One draw per image/blob item (its texture's bind group plus
    /// its vertex range in the shared image buffer).
    image_draws: Vec<ImageDraw>,
    /// Uploaded image/blob textures, by resource id.
    image_cache: HashMap<ResourceId, CachedImage>,
    rect_vertex_count: u32,
    glyph_vertex_count: u32,
    viewport: (f32, f32),
    /// The scroll offset last written to the uniform (page
    /// coordinates): the vertex shaders subtract it, so the content
    /// scrolls without a re-layout. The canvas fill is translated
    /// the other way in `prepare` so it stays glued to the viewport.
    scroll: (f32, f32),
    /// The render target's format, for picking the image texture
    /// format (see [`image_texture_format`]).
    target_format: wgpu::TextureFormat,
    /// Glyphs skipped because the atlas was full (diagnostic count).
    pub atlas_full_skips: u32,
    /// Image/blob items skipped because the resource was missing or
    /// its pixels did not match its dimensions (diagnostic count).
    /// The shell validates before handing a list over, so a skip
    /// here means a bug or a race, not a normal frame.
    pub image_skips: u32,
}

impl PaintRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pluto viewport uniform"),
            contents: bytemuck::cast_slice(&[1.0f32, 1.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pluto uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pluto uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pluto glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pluto pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &atlas_bind_group_layout],
            immediate_size: 0,
        });
        // The rect pipeline needs only the uniform. Built with the
        // shared two-group layout, wgpu demands bind group 1 be set at
        // draw time even though the rect shader never uses it — a
        // validation panic that only surfaced once a FillRect actually
        // drew (the white canvas default).
        let rect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pluto rect pipeline layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                immediate_size: 0,
            });
        let rect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto rect shader"),
            source: wgpu::ShaderSource::Wgsl(RECT_WGSL.into()),
        });
        let glyph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto glyph shader"),
            source: wgpu::ShaderSource::Wgsl(GLYPH_WGSL.into()),
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
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pluto image shader"),
            source: wgpu::ShaderSource::Wgsl(IMAGE_WGSL.into()),
        });
        // Images sample a texture like glyphs do, so the image
        // pipeline reuses the glyph pipeline's two-group layout
        // (uniform + texture/sampler); each image texture gets its
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

        // The font-id map for resolving wire glyph keys: built from
        // this renderer's own FontSystem, which loads the same system
        // fonts in the same order as layout's.
        let font_system = FontSystem::new();
        let font_ids = font_id_map(&font_system);

        Self {
            rect_pipeline,
            glyph_pipeline,
            image_pipeline,
            uniform_buffer,
            uniform_bind_group,
            atlas_texture,
            atlas_bind_group,
            atlas_bind_group_layout,
            atlas_sampler,
            atlas_map: HashMap::new(),
            atlas_rows: Vec::new(),
            font_system,
            font_ids,
            swash: SwashCache::new(),
            rect_buffer: None,
            glyph_buffer: None,
            image_buffer: None,
            image_draws: Vec::new(),
            image_cache: HashMap::new(),
            rect_vertex_count: 0,
            glyph_vertex_count: 0,
            viewport: (1.0, 1.0),
            scroll: (0.0, 0.0),
            target_format: format,
            atlas_full_skips: 0,
            image_skips: 0,
        }
    }

    /// Upload the display list for the next frame: rebuild the vertex
    /// buffers, rasterize any new glyphs into the atlas, upload any
    /// new image/blob textures. `resources` is the table the list's
    /// image/blob ids resolve against — the shell validates the pair
    /// before calling, so a missing resource here is defensive only.
    /// `scroll` is the viewport's scroll offset (page coordinates,
    /// clamped >= 0 by the shell): it goes into the uniform, and the
    /// canvas fill is translated by it so it stays glued to the
    /// viewport while the content scrolls.
    pub fn prepare(
        &mut self,
        list: &DisplayList,
        resources: &ResourceTable,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (f32, f32),
        scroll: (f32, f32),
    ) {
        if viewport != self.viewport || scroll != self.scroll {
            self.viewport = viewport;
            self.scroll = scroll;
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[viewport.0, viewport.1, scroll.0, scroll.1]),
            );
        }

        let mut rect_vertices: Vec<RectVertex> = Vec::new();
        for (i, item) in list.0.iter().enumerate() {
            // Item 0 is the canvas fill (DisplayList::build pushes
            // it first): translate it by the scroll so the shader's
            // subtraction leaves it covering the viewport. Every
            // other rect scrolls with the content.
            if i == 0 {
                rect_vertices.extend(rect_vertices_for(&canvas_rect(item, scroll)));
            } else {
                rect_vertices.extend(rect_vertices_for(item));
            }
        }

        let mut glyph_vertices: Vec<GlyphVertex> = Vec::new();
        for item in &list.0 {
            let DisplayItem::Glyphs { color, glyphs } = item else {
                continue;
            };
            for glyph in glyphs {
                // The wire key resolves back to a cache key through
                // this renderer's font database. An unresolvable key
                // (a font the renderer never loaded) is skipped like
                // a cache miss — never drawn wrong.
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
                if image.content != SwashContent::Mask {
                    continue; // color/subpixel glyphs are M2+
                }
                let placement = image.placement;
                if placement.width == 0 || placement.height == 0 {
                    continue;
                }
                let is_new = !self.atlas_map.contains_key(&cache_key);
                let Some(rect) = pack(
                    &mut self.atlas_rows,
                    &mut self.atlas_map,
                    cache_key,
                    placement.width,
                    placement.height,
                ) else {
                    self.atlas_full_skips += 1;
                    continue;
                };
                if is_new {
                    upload_glyph(queue, &self.atlas_texture, rect, &image.data);
                }
                let (u0, v0) = (
                    rect.x as f32 / ATLAS_SIZE as f32,
                    rect.y as f32 / ATLAS_SIZE as f32,
                );
                let (u1, v1) = (
                    (rect.x + rect.w) as f32 / ATLAS_SIZE as f32,
                    (rect.y + rect.h) as f32 / ATLAS_SIZE as f32,
                );
                let (x0, y0, x1, y1) = glyph_quad(glyph, placement);
                let c = *color;
                glyph_vertices.extend([
                    GlyphVertex { pos: [x0, y0], uv: [u0, v0], color: c },
                    GlyphVertex { pos: [x1, y0], uv: [u1, v0], color: c },
                    GlyphVertex { pos: [x0, y1], uv: [u0, v1], color: c },
                    GlyphVertex { pos: [x1, y0], uv: [u1, v0], color: c },
                    GlyphVertex { pos: [x1, y1], uv: [u1, v1], color: c },
                    GlyphVertex { pos: [x0, y1], uv: [u0, v1], color: c },
                ]);
            }
        }

        // Image/blob items: resolve the id in the resource table,
        // upload the texture on first sight, and record one draw per
        // item (its texture's bind group + vertex range). A missing
        // resource or mismatched pixels is skipped and counted — the
        // shell validates first, so this is a bug or a race, and a
        // skipped image must never take the frame down with it.
        let mut image_vertices: Vec<ImageVertex> = Vec::new();
        let mut image_draws: Vec<ImageDraw> = Vec::new();
        for item in &list.0 {
            let (resource, _, _, _, _) = match item {
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
                } => (resource, x, y, width, height),
                _ => continue,
            };
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
            if !self.image_cache.contains_key(resource) {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("pluto image texture"),
                    size: wgpu::Extent3d {
                        width: *width,
                        height: *height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: image_texture_format(self.target_format),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                upload_image(queue, &texture, *width, *height, rgba);
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pluto image bind group"),
                    layout: &self.atlas_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                        },
                    ],
                });
                self.image_cache
                    .insert(*resource, CachedImage { bind_group });
            }
            let first_vertex = image_vertices.len() as u32;
            let verts = image_vertices_for(item);
            let vertex_count = verts.len() as u32;
            image_vertices.extend(verts);
            // The borrow of `self.image_cache` ends before the next
            // iteration mutates it: clone the bind group out here.
            let bind_group = self.image_cache[resource].bind_group.clone();
            image_draws.push(ImageDraw {
                bind_group,
                first_vertex,
                vertex_count,
            });
        }

        self.rect_vertex_count = rect_vertices.len() as u32;
        self.glyph_vertex_count = glyph_vertices.len() as u32;
        self.image_draws = image_draws;
        self.rect_buffer = if rect_vertices.is_empty() {
            None
        } else {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pluto rect vertices"),
                contents: bytemuck::cast_slice(&rect_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };
        self.glyph_buffer = if glyph_vertices.is_empty() {
            None
        } else {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pluto glyph vertices"),
                contents: bytemuck::cast_slice(&glyph_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };
        self.image_buffer = if image_vertices.is_empty() {
            None
        } else {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pluto image vertices"),
                    contents: bytemuck::cast_slice(&image_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            )
        };
    }

    /// Draw the prepared frame into an already-begun render pass.
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.rect_vertex_count > 0 {
            if let Some(buffer) = &self.rect_buffer {
                pass.set_pipeline(&self.rect_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..self.rect_vertex_count, 0..1);
            }
        }
        if self.glyph_vertex_count > 0 {
            if let Some(buffer) = &self.glyph_buffer {
                pass.set_pipeline(&self.glyph_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.atlas_bind_group, &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.draw(0..self.glyph_vertex_count, 0..1);
            }
        }
        // Images draw last, one draw call per item (each has its own
        // texture). Paint order across types is already grouped by
        // pipeline — rects, then glyphs, then images — matching the
        // M1 list order for real pages, where images sit above text.
        if let Some(buffer) = &self.image_buffer {
            pass.set_pipeline(&self.image_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, buffer.slice(..));
            for draw in &self.image_draws {
                pass.set_bind_group(1, &draw.bind_group, &[]);
                pass.draw(
                    draw.first_vertex..draw.first_vertex + draw.vertex_count,
                    0..1,
                );
            }
        }
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

    #[test]
    fn atlas_packing_never_overlaps() {
        let keys = shaped_keys();
        assert!(!keys.is_empty(), "shaping produced no glyphs");
        let mut rows = Vec::new();
        let mut map = HashMap::new();
        let mut placed = Vec::new();
        // Pack every shaped glyph (plus repeats) at mixed sizes; none may
        // overlap, none may leave the atlas, and the same key always
        // returns the same rect.
        for (i, k) in keys.iter().enumerate() {
            let (w, h) = (8 + (i as u32 % 17), 8 + (i as u32 % 11));
            let r = pack(&mut rows, &mut map, *k, w, h)
                .unwrap_or_else(|| panic!("atlas full at glyph {i}"));
            placed.push(r);
            assert_eq!(pack(&mut rows, &mut map, *k, w, h), Some(r));
        }
        for (i, a) in placed.iter().enumerate() {
            assert!(a.x + a.w <= ATLAS_SIZE && a.y + a.h <= ATLAS_SIZE);
            for b in placed.iter().skip(i + 1) {
                let overlap = a.x < b.x + b.w
                    && b.x < a.x + a.w
                    && a.y < b.y + b.h
                    && b.y < a.y + a.h;
                assert!(!overlap, "atlas rects {a:?} and {b:?} overlap");
            }
        }
        // An oversized glyph (fresh key) does not fit.
        let mut big = keys[0];
        big.glyph_id = big.glyph_id.wrapping_add(1);
        assert!(pack(&mut rows, &mut map, big, 2000, 2000).is_none());
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
}
