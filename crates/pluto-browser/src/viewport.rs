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
use pluto_paint::{DisplayList, ResourceId, ResourceTable};
use std::collections::HashSet;
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
    /// Frame-blob ids whose RGBA pixels changed since the last
    /// presented primitive (Stage 16, M8 Phase 2, Rec 5):
    /// shared with the shell's tab (`Arc<Mutex>` — the
    /// primitive must be `Send + Sync`). Drained in
    /// `prepare()`, never copied: an id is advanced exactly
    /// once per conversion.
    pub video_dirty: Arc<std::sync::Mutex<HashSet<ResourceId>>>,
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
    /// publish pointer + wheel events. The bounds are the page area
    /// (window minus chrome) — the true layout viewport. iced calls
    /// this on every event, including the `RedrawRequested` that
    /// follows a resize's relayout, so the bounds here are current
    /// (the resize event itself still sees the pre-resize layout,
    /// but it publishes nothing: the bounds are unchanged from the
    /// last report then). Publishing only on change means one
    /// message per frame during a drag-resize, and nothing at all
    /// for mouse moves with unchanged bounds. The app re-lays out
    /// from the retained document and skips the message entirely if
    /// the size is unchanged.
    ///
    /// Pointer/wheel events publish widget coordinates; the shell
    /// adds the scroll offset and forwards page coordinates to the
    /// worker (Stage 8 Rec 9, InputEvent v3). The wheel delta is
    /// forwarded raw with its mode — the worker scales lines
    /// (×16 px) and pages (× viewport) itself. A wheel with no
    /// cursor position (cursor outside the window) is dropped: it
    /// cannot be targeted.
    fn update(
        &self,
        state: &mut Self::State,
        event: &cosmic::iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<crate::Message>> {
        let widget_pos = |pos: cosmic::iced::Point| (pos.x - bounds.x, pos.y - bounds.y);
        if let cosmic::iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) = event
            && let Some(pos) = cursor.position()
        {
            let (dx, dy, delta_mode) = match delta {
                mouse::ScrollDelta::Lines { x, y } => (*x, *y, pluto_ipc::DeltaMode::Line),
                mouse::ScrollDelta::Pixels { x, y } => (*x, *y, pluto_ipc::DeltaMode::Pixel),
            };
            let (x, y) = widget_pos(pos);
            return Some(shader::Action::publish(crate::Message::Scrolled {
                x,
                y,
                dx,
                dy,
                delta_mode,
            }));
        }
        if let cosmic::iced::Event::Mouse(mouse::Event::CursorMoved { position }) = event {
            let (x, y) = widget_pos(*position);
            return Some(shader::Action::publish(crate::Message::PointerMoved {
                x,
                y,
            }));
        }
        if let cosmic::iced::Event::Mouse(mouse::Event::ButtonPressed(button)) = event
            && let Some(pos) = cursor.position()
        {
            let (x, y) = widget_pos(pos);
            return Some(shader::Action::publish(crate::Message::PointerDown {
                x,
                y,
                button: button_number(button),
            }));
        }
        if let cosmic::iced::Event::Mouse(mouse::Event::ButtonReleased(button)) = event
            && let Some(pos) = cursor.position()
        {
            let (x, y) = widget_pos(pos);
            return Some(shader::Action::publish(crate::Message::PointerUp {
                x,
                y,
                button: button_number(button),
            }));
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
            video_dirty: Arc::clone(&self.video_dirty),
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
    video_dirty: Arc<std::sync::Mutex<HashSet<ResourceId>>>,
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
        // Stage 16 (M8 Phase 2, Rec 5): re-upload every video
        // frame whose pixels changed since the last primitive —
        // `advance_image_frame` marks the texture dirty so
        // `prepare()` below re-uploads just that texture (the
        // pre-existing gap: nothing called this for animated
        // images either — video is the first caller). The set
        // drains here, so an id is advanced once per conversion.
        for id in self.video_dirty.lock().unwrap().drain() {
            pipeline.renderer.advance_image_frame(id);
        }
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
        // The widget rect (physical px) and the physical-per-layout-px
        // scale: per-run scissors are page-space rects, converted
        // inside `render` — `bounds.width` is the widget's layout
        // width, so the ratio is the scale.
        let scale = if self.bounds.width > 0.0 {
            clip_bounds.width as f32 / self.bounds.width
        } else {
            1.0
        };
        pipeline.renderer.render(
            &mut pass,
            (
                clip_bounds.x,
                clip_bounds.y,
                clip_bounds.width,
                clip_bounds.height,
            ),
            scale,
        );
    }
}

/// The IPC button number for an iced mouse button (Stage 8 Rec 9):
/// 0 = left, 1 = middle, 2 = right, 3 = back, 4 = forward. Any
/// other button shares 5 — the worker only distinguishes the left
/// button (click synthesis); the rest get `:active` only.
fn button_number(button: &mouse::Button) -> u8 {
    match button {
        mouse::Button::Left => 0,
        mouse::Button::Middle => 1,
        mouse::Button::Right => 2,
        mouse::Button::Back => 3,
        mouse::Button::Forward => 4,
        mouse::Button::Other(_) => 5,
    }
}
