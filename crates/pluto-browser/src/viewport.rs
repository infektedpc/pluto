//! GPU viewport: the web page renders into this shader widget.
//!
//! libcosmic renders with wgpu under the hood, so `cosmic::iced::widget::Shader`
//! hands the engine the shared `wgpu::Device`, `wgpu::Queue`, and render pass
//! directly -- no CPU pixel readbacks. The `Program` produces a `Primitive`
//! per frame; the `Primitive::prepare` step is where Pluto will upload its
//! display-list vertex buffers and cosmic-text glyph atlases, and
//! `Primitive::render` is where those get drawn.
//!
//! Import notes (verified against the pinned libcosmic rev `ecc5fe4`):
//! - The shader widget is NOT `cosmic::widget::shader`; it lives at
//!   `cosmic::iced::widget::shader`.
//! - The wgpu crate is NOT `cosmic::iced_wgpu::wgpu`; it is re-exported as
//!   `cosmic::iced::wgpu`, and it is wgpu 28 -- the same version the
//!   workspace pins, so these types match the shared device exactly.

use cosmic::iced::widget::shader::{self, Pipeline, Primitive, Program};
use cosmic::iced::wgpu;
use cosmic::iced::{Rectangle, mouse};

#[derive(Debug, Clone)]
pub struct WebViewport {
    pub url: String,
    pub clear_color: [f32; 4],
}

impl<Message> Program<Message> for WebViewport {
    type State = ();
    type Primitive = WebPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        WebPrimitive {
            bounds,
            clear_color: self.clear_color,
        }
    }
}

#[derive(Debug)]
pub struct WebPrimitive {
    bounds: Rectangle,
    clear_color: [f32; 4],
}

pub struct WebRenderPipeline;

impl Pipeline for WebRenderPipeline {
    fn new(
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _format: wgpu::TextureFormat,
    ) -> Self {
        Self
    }
}

impl Primitive for WebPrimitive {
    type Pipeline = WebRenderPipeline;

    fn prepare(
        &self,
        _pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        // Upload Pluto display-list vertex buffers & cosmic-text glyph atlases here.
        // `self.clear_color` is the page background the first quad will paint.
        let _ = self.clear_color;
        let _ = self.bounds;
    }

    fn render(
        &self,
        _pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        // Load, don't clear: the target is the whole window texture and the
        // shell chrome (nav bar, divider) lives in it too. The page
        // background gets painted as a quad covering `self.bounds`.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pluto_web_viewport_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                // wgpu 28 added these fields; neither applies here
                // (2D view, no multiview).
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        // Draw Pluto page background quad + laid-out quads & text here.
    }
}
