//! Stage 8 Recommendation 13 — raster-thread spike (time-boxed).
//!
//! Question: can Pluto rasterise on a worker thread — encode into an
//! offscreen texture on a worker, hand the [`wgpu::CommandBuffer`] to
//! the render thread, submit it there and blit — while the shell runs
//! inside libcosmic's iced Shader widget?
//!
//! Verdict: **the wgpu-level mechanics work; the shell wiring does
//! not exist yet.** This module proves the first half (see the test
//! `worker_offscreen_encode_submit_readback`, run under SwiftShader
//! in this sandbox) and documents the second half as the remaining
//! wall. No code in the shipping render path was touched:
//! `renderer.rs` and `pluto-browser/src/viewport.rs` are unchanged.
//!
//! Evidence (all paths are the pinned libcosmic rev `ecc5fe4`
//! checkout at `~/.cargo/git/checkouts/libcosmic-41009aea1d72760b/ecc5fe4`;
//! wgpu 28.0.0 is the crates.io release, verified in
//! `~/.cargo/registry/src/.../wgpu-28.0.0`):
//!
//! * Device creation is NOT exposed. `iced/wgpu/src/window/compositor.rs`,
//!   `Compositor::request()` creates its own `wgpu::Instance`, picks
//!   the adapter, calls `adapter.request_device(...)` and builds
//!   `Engine::new(&adapter, device, queue, ...)`. The only entry
//!   point, `Compositor::new(settings, window, shell)`, takes
//!   `iced_wgpu::Settings`, which carries only `present_mode`,
//!   `backends`, `antialiasing` and font settings — no device,
//!   adapter or instance hook, no callback.
//! * The handles are held privately. `iced/wgpu/src/engine.rs`:
//!   `Engine { pub(crate) device: wgpu::Device, pub(crate) queue:
//!   wgpu::Queue, ... }`. `iced/wgpu/src/lib.rs`: `Renderer` keeps
//!   `engine: Engine` as a private field with no accessor, and
//!   `Renderer::present()` builds its own encoder, calls
//!   `prepare`/`render`, and submits only that encoder's buffer —
//!   there is no hook to inject a foreign `CommandBuffer` into its
//!   submit. `iced/wgpu/src/offscreen.rs` (`offscreen::Pipeline`, a
//!   compute blit) is not exported from `iced_wgpu` and only serves
//!   `Renderer::screenshot`; there is no offscreen-texture hook for
//!   application code.
//! * A `Device`/`Queue` IS handed to application code every frame.
//!   `iced/wgpu/src/primitive.rs`: `Primitive::prepare(&self,
//!   pipeline, device: &wgpu::Device, queue: &wgpu::Queue, bounds,
//!   viewport)`; Pluto's `WebPrimitive::prepare`
//!   (`crates/pluto-browser/src/viewport.rs`) already receives both,
//!   as does `Pipeline::new(device, queue, format)`. Cloning those
//!   borrows is the way to obtain a `Device`/`Queue` — no upstream
//!   change is needed for the device half.
//! * Thread-safety (wgpu 28, native `send_sync` cfg — see
//!   `wgpu-28.0.0/build.rs`, on for every non-wasm target):
//!   `Device: Clone + Send + Sync` (`device.rs:19,24`), `Queue:
//!   Clone + Send + Sync` (`queue.rs:13,18`), `Texture: Clone +
//!   Send + Sync` (`texture.rs`), `CommandBuffer: Send + Sync`
//!   (`command_buffer.rs`), `CommandEncoder: Send + Sync`
//!   (`command_encoder.rs`). Note: the Stage 8 report's claim that
//!   "`CommandEncoder` is not `Send`" is stale for wgpu 28 — the
//!   source carries a `static_assertions::assert_impl_all!` for it.
//!   `assert_raster_thread_types` below pins these bounds so a wgpu
//!   upgrade that dropped them fails this crate's build, loudly.
//!
//! What the prototype proves (test, executed — see the run note at
//! the bottom of this module): a cloned `Device` moved to a worker
//! thread; the worker creates its own encoder, clears an offscreen
//! texture and finishes a `CommandBuffer`; the buffer crosses an
//! `mpsc` channel to the "render" thread; the render thread submits
//! it with `Queue::submit`; a copy-to-buffer readback confirms the
//! worker's commands actually executed on the GPU.
//!
//! What remains — the wall, deliberately NOT done in this spike
//! because it requires touching `viewport.rs`:
//!
//! 1. Stash the clones. `WebPrimitive::prepare` must `device.clone()`
//!    / `queue.clone()` into shared state (e.g. an
//!    `Arc<OnceLock<(Device, Queue)>>` owned by the pipeline) once;
//!    the spike never touches `viewport.rs`, so this step is unwired.
//! 2. Submit injection. `Renderer::present` submits only its own
//!    encoder, so the worker buffer must be submitted from
//!    `WebPrimitive::render` through the stashed `Queue` clone (same
//!    queue, so FIFO order puts the worker's work before the
//!    present's submit). `render` receives no `&Queue`, which is why
//!    the stash in (1) is mandatory.
//! 3. Blit. The offscreen texture must be sampled or copied into the
//!    swapchain view inside `render`'s pass (textured quad or
//!    `copy_texture_to_texture`). The spike target already carries
//!    `TEXTURE_BINDING | COPY_SRC` for exactly that; the blit itself
//!    is unwired.
//! 4. Lifetime / epochs. The worker must not re-encode a texture the
//!    render thread is still reading; this needs the pending/active
//!    epoch discipline of Recommendation 2 (double-buffer the
//!    offscreen target, or submit-then-wait per frame).
//! 5. Pipelines on the worker. A real raster worker needs its own
//!    render pipelines created from the cloned `Device` (same device,
//!    so the resulting texture is usable in the render thread's
//!    pass). The spike encodes a clear only, so pipeline creation on
//!    a worker is still unproven — small, but unproven.
//!
//! What would unblock the full thing without touching Pluto's shell:
//! nothing further is needed from libcosmic for the device half (the
//! `prepare` borrow suffices). A public `Renderer::device()` /
//! `Engine` accessor, or an iced API to submit extra command buffers
//! in `present`, would remove the stash-and-submit-from-`render`
//! workaround, but neither is required.
//!
//! Run note: this sandbox has no GPU and no Vulkan ICD (no
//! `/usr/share/vulkan`, no lavapipe, no EGL). The test runs against
//! the SwiftShader software Vulkan ICD that ships with the
//! Chromium checkout on this machine:
//!
//! ```text
//! VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json \
//!   cargo test -p pluto-paint --lib raster_spike
//! ```
//!
//! Without any adapter the test prints a SKIP note and passes
//! vacuously (it must not fail `cargo test` on machines without a
//! GPU); the executed verification in this sandbox used SwiftShader
//! and asserted real pixels — see the test.
//!
//! Verified 2026-11-24 ~17:54 UTC: with
//! `VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json` the
//! test ran against "SwiftShader Device (Subzero)" (Vulkan, CPU) and
//! passed — worker encode on a different thread, `CommandBuffer`
//! handoff, render-thread submit, readback `[255, 0, 0, 255]` at the
//! first and last pixels. Full `pluto-paint` suite at the same tree
//! state: 37/37 passed. Without the env var the test takes the SKIP
//! branch (`request_adapter` returns `NotFound`, no adapters) and
//! passes.

#[cfg(test)]
use std::sync::mpsc;

/// Width/height of the spike's offscreen target. 64 px keeps the
/// readback row pitch at exactly 256 bytes (`COPY_BYTES_PER_ROW_ALIGNMENT`).
pub const SPIKE_WIDTH: u32 = 64;
pub const SPIKE_HEIGHT: u32 = 64;

/// Compile-time proof that the handles a raster worker needs can
/// cross threads. If a future wgpu version drops any of these bounds,
/// this fails the build instead of failing at runtime.
pub fn assert_raster_thread_types() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
    assert_clone_send_sync::<wgpu::Device>();
    assert_clone_send_sync::<wgpu::Queue>();
    assert_clone_send_sync::<wgpu::Texture>();
    assert_send_sync::<wgpu::CommandBuffer>();
    assert_send_sync::<wgpu::CommandEncoder>();
}

/// An offscreen render target owned by the raster worker's device.
///
/// `Rgba8Unorm` (not sRGB) so a readback is byte-exact: clearing to
/// `(1.0, 0.0, 0.0, 1.0)` reads back as `[255, 0, 0, 255]` with no
/// colour-space conversion in the way. Usages cover the whole spike
/// lifecycle: render into it on the worker (`RENDER_ATTACHMENT`),
/// sample or copy from it on the render thread (`TEXTURE_BINDING`,
/// `COPY_SRC`).
#[derive(Debug)]
pub struct OffscreenTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// Create an offscreen target on `device`. The texture is `Send +
/// Sync`, so the caller may move it (or a clone of it) to a worker
/// thread; the device it was created from is the only device that may
/// ever encode into it or sample it.
pub fn create_offscreen_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> OffscreenTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pluto raster_spike offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    OffscreenTarget {
        texture,
        view,
        width,
        height,
    }
}

/// Worker-side encode: clear `target` to `color` and return the
/// finished `CommandBuffer` for the render thread to submit.
///
/// This runs entirely on the worker thread from a cloned `Device`:
/// it creates its own encoder (never the render thread's), records
/// one render pass, and finishes. No shader is involved — a clear is
/// enough to prove the encode → handoff → submit → execute chain;
/// drawing real page content would add pipelines, not new threading
/// facts (see wall item 5 in the module docs).
pub fn encode_solid_fill(
    device: &wgpu::Device,
    target: &OffscreenTarget,
    color: wgpu::Color,
) -> wgpu::CommandBuffer {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pluto raster_spike worker encoder"),
        });
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pluto raster_spike worker pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // The pass is dropped (ended) with no draw calls: the clear
        // alone is the recorded work.
        drop(pass);
    }
    encoder.finish()
}

/// Copy an offscreen target back to a CPU-readable buffer on the
/// render thread. Exists only so the test can prove the worker's
/// commands executed; production code would blit to the swapchain
/// instead (wall item 3). Test-only: the non-test build has no
/// other caller.
#[cfg(test)]
fn copy_target_to_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &OffscreenTarget,
) -> Vec<u8> {
    let size = (target.width * target.height * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pluto raster_spike readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pluto raster_spike readback encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(target.width * 4),
                rows_per_image: Some(target.height),
            },
        },
        wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let (tx, rx) = mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("raster_spike: device poll");
    rx.recv()
        .expect("raster_spike: map callback fired")
        .expect("raster_spike: buffer mapped");
    let bytes = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_types_are_thread_safe() {
        // Compile-time bounds; calling it is enough.
        assert_raster_thread_types();
    }

    /// The spike: a worker thread with a cloned `Device` encodes into
    /// an offscreen texture and hands a `CommandBuffer` to the render
    /// thread, which submits it and reads the pixels back.
    ///
    /// Needs a wgpu adapter. In this sandbox that is SwiftShader via
    /// `VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json`;
    /// with no adapter at all the test skips (prints why) instead of
    /// failing the suite on GPU-less machines.
    #[test]
    fn worker_offscreen_encode_submit_readback() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = match futures_executor::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            },
        )) {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!(
                    "raster_spike: SKIP — no wgpu adapter in this environment \
                     ({err:?}; tried all backends; in this sandbox set \
                     VK_ICD_FILENAMES=/opt/meta-chromium/vk_swiftshader_icd.json)"
                );
                return;
            }
        };
        eprintln!("raster_spike: adapter {:?}", adapter.get_info());

        let (device, queue) =
            futures_executor::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("pluto raster_spike"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                },
            ))
            .expect("raster_spike: request_device");

        // The "render thread" is this test thread. It owns the target
        // and the queue submit; the worker gets clones only.
        let target = create_offscreen_target(&device, SPIKE_WIDTH, SPIKE_HEIGHT);
        let (tx, rx) =
            mpsc::channel::<(wgpu::CommandBuffer, std::thread::ThreadId)>();
        let worker_device = device.clone();
        let worker_texture = target.texture.clone();
        let worker_view = target.view.clone();
        std::thread::spawn(move || {
            let worker_target = OffscreenTarget {
                texture: worker_texture,
                view: worker_view,
                width: SPIKE_WIDTH,
                height: SPIKE_HEIGHT,
            };
            let buffer = encode_solid_fill(
                &worker_device,
                &worker_target,
                wgpu::Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            );
            tx.send((buffer, std::thread::current().id()))
                .expect("raster_spike: worker sends buffer");
        });

        let (worker_buffer, worker_id) =
            rx.recv().expect("raster_spike: render thread receives buffer");
        assert_ne!(
            worker_id,
            std::thread::current().id(),
            "the encode really ran on a different thread"
        );

        // Render-thread submit of the worker-produced buffer.
        queue.submit([worker_buffer]);

        // Read the pixels back and prove the worker's clear executed.
        let pixels = copy_target_to_buffer(&device, &queue, &target);
        assert_eq!(pixels.len(), (SPIKE_WIDTH * SPIKE_HEIGHT * 4) as usize);
        assert_eq!(
            &pixels[0..4],
            &[255, 0, 0, 255],
            "first pixel is the worker's clear colour"
        );
        let last = ((SPIKE_WIDTH * SPIKE_HEIGHT - 1) * 4) as usize;
        assert_eq!(
            &pixels[last..last + 4],
            &[255, 0, 0, 255],
            "last pixel is the worker's clear colour"
        );
    }
}
