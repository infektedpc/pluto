//! H.264 video decoding via openh264.
//!
//! One [`H264Decoder`] owns one video stream. Feed it Annex B
//! access units (see [`crate::demux::avcc_to_annexb`] for the
//! MP4 conversion); it hands back [`pluto_ipc::VideoFrame`]s
//! in YUV420 — the same planes the transport channel carries
//! (see `pluto-ipc`'s `VideoFrame` docs for why YUV, not
//! RGBA). The decoder may buffer a few frames before emitting
//! one (B-frame reordering is not used by the fixture, but the
//! API assumes it); call [`H264Decoder::flush`] at end of
//! stream to drain the remainder.
//!
//! The caller supplies `origin_clean` per frame (see the
//! crate-level tainting policy): same-origin or `data:` is
//! clean, cross-origin is clean only when its fetch was
//! CORS-approved.

use crate::MediaError;
use openh264::decoder::{Decoder, DecoderConfig, Flush};
use openh264::formats::YUVSource;
use pluto_ipc::VideoFrame;

/// An H.264 decoder for one video stream.
pub struct H264Decoder {
    inner: Decoder,
    next_frame_id: u64,
}

impl H264Decoder {
    /// Create a decoder. The `source` feature builds Cisco's
    /// library from the vendored C++ in `openh264-sys2`, so no
    /// download happens here; the first build compiles it.
    pub fn new() -> Result<Self, MediaError> {
        let inner = Decoder::with_api_config(
            openh264::OpenH264API::from_source(),
            DecoderConfig::new().flush_after_decode(Flush::NoFlush),
        )?;
        Ok(Self {
            inner,
            next_frame_id: 0,
        })
    }

    /// Feed one Annex B access unit (one MP4 sample, converted,
    /// with SPS/PPS prepended before the first sample and each
    /// keyframe). Returns a decoded frame when the decoder has
    /// one ready, or `Ok(None)` while it is still buffering.
    pub fn decode(
        &mut self,
        annexb: &[u8],
        element_id: u64,
        pts_ms: f64,
        duration_ms: f64,
        origin_clean: bool,
    ) -> Result<Option<VideoFrame>, MediaError> {
        let decoded = self.inner.decode(annexb)?;
        let Some(yuv) = decoded else {
            return Ok(None);
        };
        let frame_id = self.next_frame_id;
        self.next_frame_id += 1;
        Ok(Some(to_frame(
            &yuv,
            element_id,
            frame_id,
            pts_ms,
            duration_ms,
            origin_clean,
        )))
    }

    /// Drain frames the decoder still holds at end of stream.
    pub fn flush(
        &mut self,
        element_id: u64,
        pts_ms: f64,
        duration_ms: f64,
        origin_clean: bool,
    ) -> Result<Vec<VideoFrame>, MediaError> {
        let decoded = self.inner.flush_remaining()?;
        // The decoded frames borrow `self.inner`, so the
        // conversion runs on a local counter, not `&mut self`.
        let mut frame_id = self.next_frame_id;
        let frames: Vec<VideoFrame> = decoded
            .iter()
            .map(|yuv| {
                let frame = to_frame(yuv, element_id, frame_id, pts_ms, duration_ms, origin_clean);
                frame_id += 1;
                frame
            })
            .collect();
        self.next_frame_id = frame_id;
        Ok(frames)
    }
}

/// Convert one decoded YUV picture into a [`VideoFrame`].
/// A free function (not a method) so the caller's borrow of
/// the decoder can stay alive while frames are converted.
fn to_frame(
    yuv: &openh264::decoder::DecodedYUV,
    element_id: u64,
    frame_id: u64,
    pts_ms: f64,
    duration_ms: f64,
    origin_clean: bool,
) -> VideoFrame {
    let (width, height) = yuv.dimensions();
    let (y_stride, u_stride, v_stride) = yuv.strides();
    // openh264's chroma planes share a stride; `VideoFrame`
    // carries one `uv_stride` for both. If a future decoder
    // ever split them, the struct needs the second field back.
    debug_assert_eq!(u_stride, v_stride, "openh264 u/v strides match");
    VideoFrame {
        element_id,
        frame_id,
        width: width as u32,
        height: height as u32,
        pts_ms,
        duration_ms,
        y: yuv.y().to_vec(),
        u: yuv.u().to_vec(),
        v: yuv.v().to_vec(),
        y_stride: y_stride as u32,
        uv_stride: u_stride as u32,
        origin_clean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::{Mp4Demux, avcc_to_annexb};

    #[test]
    fn bbb_mp4_decodes_to_640x360_yuv_frames() {
        let bytes = crate::fixture_bytes("bbb-360-10s.mp4");
        let mut demux = Mp4Demux::open(bytes).expect("mp4 parses");
        let track = demux.video_track().expect("h264 video track");
        let prefix = demux
            .parameter_sets_annexb(track.track_id)
            .expect("sps/pps present");
        let mut decoder = H264Decoder::new().expect("decoder creates");

        let mut decoded = 0u32;
        let mut first: Option<VideoFrame> = None;
        // Decode the first 30 samples (~1 s): enough to prove
        // the real decoder works end-to-end on a real file.
        for sample_id in 1..=30 {
            let Some(sample) = demux
                .read_video_sample(track.track_id, sample_id)
                .expect("sample reads")
            else {
                continue;
            };
            let mut annexb = Vec::new();
            if sample.is_sync {
                annexb.extend_from_slice(&prefix);
            }
            avcc_to_annexb(&sample.bytes, &mut annexb).expect("sample converts");
            if let Some(frame) = decoder
                .decode(&annexb, 1, sample.pts_ms, sample.duration_ms, true)
                .expect("decode succeeds")
            {
                if first.is_none() {
                    first = Some(frame);
                }
                decoded += 1;
            }
        }
        let flushed = decoder.flush(1, 1000.0, 40.0, true).expect("flush succeeds");
        decoded += flushed.len() as u32;

        assert!(decoded > 0, "at least one real frame decoded");
        let frame = first.expect("a first frame exists");
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 360);
        assert_eq!(frame.y.len(), frame.height as usize * frame.y_stride as usize);
        assert_eq!(frame.u.len(), (frame.height as usize / 2) * frame.uv_stride as usize);
        assert_eq!(frame.v.len(), (frame.height as usize / 2) * frame.uv_stride as usize);
        assert_eq!(frame.payload_bytes(), frame.y.len() + frame.u.len() + frame.v.len());
        assert!(frame.origin_clean, "fixture is same-origin");
    }
}
