//! YUV420 → RGBA conversion for the shell side of the video
//! frame channel (Stage 16 Phase 2, Rec 5).
//!
//! The wire stays YUV420 (see `pluto-ipc`'s `VideoFrame` docs
//! for why); the shell converts the latest frame per element to
//! RGBA and paints it through the existing `Resource::Image` /
//! `DisplayItem::Blob` path, exactly like a canvas bitmap. The
//! conversion is BT.601 limited-range, integer-only, and runs in
//! the shell (browser process) — never in the renderer worker,
//! and never over IPC. One conversion per arriving frame, latest
//! frame only: a slow shell converts the newest frame, never a
//! backlog.
//!
//! Chroma upsampling is nearest-neighbour (each 2×2 luma block
//! shares one U/V sample — the YUV420 layout). That is a
//! deliberate quality trade: exact enough for video, cheap
//! enough to run per frame without a shader.

use pluto_ipc::VideoFrame;

/// Convert one YUV420 frame to RGBA (opaque alpha).
///
/// Returns `None` when the planes do not match the declared
/// dimensions/strides (a corrupt frame is dropped, never painted
/// half-converted). Strides may exceed the plane width (row
/// padding is skipped, not read).
pub fn yuv420_to_rgba(frame: &VideoFrame) -> Option<Vec<u8>> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let y_stride = frame.y_stride as usize;
    let uv_stride = frame.uv_stride as usize;
    if y_stride < w || uv_stride < w.div_ceil(2) {
        return None;
    }
    let uv_h = h.div_ceil(2);
    if frame.y.len() < y_stride * h || frame.u.len() < uv_stride * uv_h || frame.v.len() < uv_stride * uv_h {
        return None;
    }
    let mut rgba = vec![0u8; w * h * 4];
    for row in 0..h {
        let y_row = &frame.y[row * y_stride..row * y_stride + w];
        let uv_row = row / 2;
        let u_row = &frame.u[uv_row * uv_stride..uv_row * uv_stride + w.div_ceil(2)];
        let v_row = &frame.v[uv_row * uv_stride..uv_row * uv_stride + w.div_ceil(2)];
        let out_row = &mut rgba[row * w * 4..(row + 1) * w * 4];
        for col in 0..w {
            let y = y_row[col] as i32;
            let u = u_row[col / 2] as i32 - 128;
            let v = v_row[col / 2] as i32 - 128;
            // BT.601 limited range, integer approximation:
            //   R = 1.164*(Y-16) + 1.596*V
            //   G = 1.164*(Y-16) - 0.391*U - 0.813*V
            //   B = 1.164*(Y-16) + 2.018*U
            // Scaled by 256 and rounded (the +128 before the shift).
            let c = (y - 16) * 298;
            let r = (c + 409 * v + 128) >> 8;
            let g = (c - 100 * u - 208 * v + 128) >> 8;
            let b = (c + 516 * u + 128) >> 8;
            let px = &mut out_row[col * 4..col * 4 + 4];
            px[0] = r.clamp(0, 255) as u8;
            px[1] = g.clamp(0, 255) as u8;
            px[2] = b.clamp(0, 255) as u8;
            px[3] = 255;
        }
    }
    Some(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_2x2(y: [u8; 4], u: u8, v: u8) -> VideoFrame {
        VideoFrame {
            element_id: 1,
            frame_id: 0,
            width: 2,
            height: 2,
            pts_ms: 0.0,
            duration_ms: 16.0,
            y: y.to_vec(),
            u: vec![u],
            v: vec![v],
            y_stride: 2,
            uv_stride: 1,
            origin_clean: true,
        }
    }

    #[test]
    fn black_and_white_convert_exactly() {
        // Limited range: Y=16 is black, Y=235 is white.
        let black = yuv420_to_rgba(&frame_2x2([16, 16, 16, 16], 128, 128)).expect("converts");
        assert_eq!(&black[0..4], &[0, 0, 0, 255]);
        let white = yuv420_to_rgba(&frame_2x2([235, 235, 235, 235], 128, 128)).expect("converts");
        assert_eq!(&white[0..4], &[255, 255, 255, 255]);
        assert_eq!(black.len(), 16);
    }

    #[test]
    fn chroma_shifts_colour_in_the_right_direction() {
        // Full red in BT.601: Y≈76, U≈85, V≈255.
        let red = yuv420_to_rgba(&frame_2x2([76, 76, 76, 76], 85, 255)).expect("converts");
        assert!(red[0] > 200, "R high for red, got {}", red[0]);
        assert!(red[1] < 80, "G low for red, got {}", red[1]);
        assert!(red[2] < 80, "B low for red, got {}", red[2]);
    }

    #[test]
    fn mismatched_planes_are_rejected_not_painted() {
        let mut f = frame_2x2([16, 16, 16, 16], 128, 128);
        f.y.truncate(2);
        assert!(yuv420_to_rgba(&f).is_none());
        let mut g = frame_2x2([16, 16, 16, 16], 128, 128);
        g.y_stride = 1;
        assert!(yuv420_to_rgba(&g).is_none());
    }

    #[test]
    fn odd_dimensions_round_chroma_up() {
        let f = VideoFrame {
            element_id: 1,
            frame_id: 0,
            width: 3,
            height: 3,
            pts_ms: 0.0,
            duration_ms: 16.0,
            y: vec![235u8; 9],
            u: vec![128u8; 4],
            v: vec![128u8; 4],
            y_stride: 3,
            uv_stride: 2,
            origin_clean: true,
        };
        let rgba = yuv420_to_rgba(&f).expect("converts");
        assert_eq!(rgba.len(), 3 * 3 * 4);
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
    }
}
