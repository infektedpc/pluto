//! GPU viewport: the web page renders into this shader widget.
//!
//! libcosmic renders with wgpu under the hood, so `cosmic::iced::widget::Shader`
//! hands the engine the shared `wgpu::Device`, `wgpu::Queue`, and render pass
//! directly -- no CPU pixel readbacks. The `Program` produces a `Primitive`
//! per frame; `Primitive::prepare` uploads the display-list vertex buffers
//! and glyph atlas through [`PaintRenderer`], and `Primitive::render` draws
//! them into the viewport's scissored region of the window texture.
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
use pluto_paint::renderer::PaintRenderer;
use pluto_paint::DisplayList;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct WebViewport {
    pub clear_color: [f32; 4],
    /// The page to draw; empty until M1.4 wires fetch -> parse -> layout.
    pub display: Arc<DisplayList>,
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
            display: Arc::clone(&self.display),
        }
    }
}

#[derive(Debug)]
pub struct WebPrimitive {
    bounds: Rectangle,
    clear_color: [f32; 4],
    display: Arc<DisplayList>,
}

pub struct WebRenderPipeline {
    renderer: PaintRenderer,
}

impl Pipeline for WebRenderPipeline {
    fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            renderer: PaintRenderer::new(device, format),
        }
    }
}

impl Primitive for WebPrimitive {
    type Pipeline = WebRenderPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        // Upload the display-list vertex buffers and any new glyphs.
        // `self.clear_color` is the fallback page background when the
        // display list has no canvas fill yet.
        let _ = self.clear_color;
        pipeline.renderer.prepare(
            &self.display,
            device,
            queue,
            (self.bounds.width, self.bounds.height),
        );
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
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

        // The shader maps page coords into the *widget* (the uniform
        // carries the widget's size), but a render pass covers the whole
        // window texture by default. Without this, page (0,0) lands at
        // the window's top-left — under the header, nav bar and status —
        // and the top of every page is clipped away by the scissor
        // below. Point the viewport at the widget's region so NDC maps
        // to the widget, not the window.
        pass.set_viewport(
            clip_bounds.x as f32,
            clip_bounds.y as f32,
            clip_bounds.width as f32,
            clip_bounds.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pipeline.renderer.render(&mut pass);
    }
}
