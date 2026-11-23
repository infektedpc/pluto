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

use crate::{DisplayItem, DisplayList};
use cosmic_text::{FontSystem, SwashCache, SwashContent};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const ATLAS_SIZE: u32 = 1024;
/// `queue.write_texture` requires rows padded to this many bytes.
const ROW_ALIGNMENT: usize = 256;

const RECT_WGSL: &str = r#"
struct Uniforms {
    viewport: vec2<f32>,
    _pad: vec2<f32>,
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
    let clip = vec2<f32>(
        v.position.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - v.position.y / uniforms.viewport.y * 2.0,
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
    _pad: vec2<f32>,
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
    let clip = vec2<f32>(
        v.position.x / uniforms.viewport.x * 2.0 - 1.0,
        1.0 - v.position.y / uniforms.viewport.y * 2.0,
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

pub struct PaintRenderer {
    rect_pipeline: wgpu::RenderPipeline,
    glyph_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    atlas_texture: wgpu::Texture,
    atlas_bind_group: wgpu::BindGroup,
    atlas_map: HashMap<cosmic_text::CacheKey, AtlasRect>,
    atlas_rows: Vec<AtlasRow>,
    font_system: FontSystem,
    swash: SwashCache,
    rect_buffer: Option<wgpu::Buffer>,
    glyph_buffer: Option<wgpu::Buffer>,
    rect_vertex_count: u32,
    glyph_vertex_count: u32,
    viewport: (f32, f32),
    /// Glyphs skipped because the atlas was full (diagnostic count).
    pub atlas_full_skips: u32,
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
            layout: Some(&pipeline_layout),
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

        Self {
            rect_pipeline,
            glyph_pipeline,
            uniform_buffer,
            uniform_bind_group,
            atlas_texture,
            atlas_bind_group,
            atlas_map: HashMap::new(),
            atlas_rows: Vec::new(),
            font_system: FontSystem::new(),
            swash: SwashCache::new(),
            rect_buffer: None,
            glyph_buffer: None,
            rect_vertex_count: 0,
            glyph_vertex_count: 0,
            viewport: (1.0, 1.0),
            atlas_full_skips: 0,
        }
    }

    /// Upload the display list for the next frame: rebuild both vertex
    /// buffers, rasterize any new glyphs into the atlas.
    pub fn prepare(
        &mut self,
        list: &DisplayList,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (f32, f32),
    ) {
        if viewport != self.viewport {
            self.viewport = viewport;
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[viewport.0, viewport.1, 0.0f32, 0.0]),
            );
        }

        let mut rect_vertices: Vec<RectVertex> = Vec::new();
        for item in &list.0 {
            rect_vertices.extend(rect_vertices_for(item));
        }

        let mut glyph_vertices: Vec<GlyphVertex> = Vec::new();
        for item in &list.0 {
            let DisplayItem::Glyphs { color, glyphs } = item else {
                continue;
            };
            for glyph in glyphs {
                let Some(image) = self
                    .swash
                    .get_image(&mut self.font_system, glyph.cache_key)
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
                let is_new = !self.atlas_map.contains_key(&glyph.cache_key);
                let Some(rect) = pack(
                    &mut self.atlas_rows,
                    &mut self.atlas_map,
                    glyph.cache_key,
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
                let (x0, y0) = (
                    glyph.x + placement.left as f32,
                    glyph.y + placement.top as f32,
                );
                let (x1, y1) = (
                    x0 + placement.width as f32,
                    y0 + placement.height as f32,
                );
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

        self.rect_vertex_count = rect_vertices.len() as u32;
        self.glyph_vertex_count = glyph_vertices.len() as u32;
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
        assert!(rect_vertices_for(&DisplayItem::Glyphs {
            color: [0.0; 4],
            glyphs: vec![],
        })
        .is_empty());
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
}
