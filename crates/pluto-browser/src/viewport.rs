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
use pluto_paint::{DisplayList, ResourceTable};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct WebViewport {
    pub clear_color: [f32; 4],
    /// The page to draw; empty until M1.4 wires fetch -> parse -> layout.
    pub display: Arc<DisplayList>,
    /// The resource table the display list's image/blob ids resolve
    /// against. Owned by the shell's compositor holder (Stage 4 #3);
    /// empty until M3 produces images.
    pub resources: Arc<ResourceTable>,
    /// The compositor holder's scroll offset, in page pixels. It is
    /// a view transform only: the renderer subtracts it in the
    /// vertex shader, so a wheel event never round-trips to layout.
    pub scroll: (f32, f32),
}

/// State the viewport widget keeps across frames: the last widget
/// bounds reported to the app, so a bounds change is published once.
#[derive(Debug, Default)]
pub struct ViewportState {
    last_reported: Option<(f32, f32)>,
}

impl Program<crate::Message> for WebViewport {
    type State = ViewportState;
    type Primitive = WebPrimitive;

    /// Report the widget's bounds to the app when they change, and
    /// publish wheel events as scroll deltas. The bounds are the
    /// page area (window minus chrome) — the true layout viewport.
    /// iced calls this on every event, including the
    /// `RedrawRequested` that follows a resize's relayout, so the
    /// bounds here are current (the resize event itself still sees
    /// the pre-resize layout, but it publishes nothing: the bounds
    /// are unchanged from the last report then). Publishing only on
    /// change means one message per frame during a drag-resize, and
    /// nothing at all for mouse moves with unchanged bounds. The app
    /// re-lays out from the retained document and skips the message
    /// entirely if the size is unchanged. A wheel event scrolls
    /// without touching layout at all: the shell stores the offset
    /// in the compositor holder and the renderer subtracts it in
    /// the vertex shader.
    fn update(
        &self,
        state: &mut Self::State,
        event: &cosmic::iced::Event,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Option<shader::Action<crate::Message>> {
        if let cosmic::iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) = event {
            // A line is ~16px; pixel deltas pass through as-is.
            let (dx, dy) = match delta {
                mouse::ScrollDelta::Lines { x, y } => (x * 16.0, y * 16.0),
                mouse::ScrollDelta::Pixels { x, y } => (*x, *y),
            };
            return Some(shader::Action::publish(crate::Message::Scrolled(dx, dy)));
        }
        let size = (bounds.width, bounds.height);
        if state.last_reported != Some(size) {
            state.last_reported = Some(size);
            return Some(shader::Action::publish(crate::Message::ViewportResized(
                size.0, size.1,
            )));
        }
        None
    }

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
            resources: Arc::clone(&self.resources),
            scroll: self.scroll,
        }
    }
}

#[derive(Debug)]
pub struct WebPrimitive {
    bounds: Rectangle,
    clear_color: [f32; 4],
    display: Arc<DisplayList>,
    resources: Arc<ResourceTable>,
    scroll: (f32, f32),
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
            &self.resources,
            device,
            queue,
            (self.bounds.width, self.bounds.height),
            self.scroll,
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
