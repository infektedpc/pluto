//! Media decoding for Pluto's M8 (audio/video) milestone.
//!
//! Stage 16 Phase 1, Rec 3: the decode stack, chosen and
//! verified against real files — nothing here is a stub that
//! claims to decode.
//!
//! # The stack
//!
//! | Layer | Crate | Version | Notes |
//! |-------|-------|---------|-------|
//! | Audio decode | symphonia | 0.5.5 | MP3, FLAC, Vorbis, WAV/PCM, AAC-in-MP4 |
//! | Video decode (primary) | openh264 | 0.9.10 | H.264 only |
//! | MP4 demux | mp4 | 0.14.0 | ISO BMFF |
//! | WebM demux | matroska-demuxer | 0.8.1 | Pure Rust, no deps |
//!
//! # Decisions (and what is NOT here yet)
//!
//! * **H.264 first, AV1 later.** H.264 is the codec of the
//!   overwhelming majority of real-world web video. AV1
//!   (rav1d 1.1.0) is a large pure-Rust decoder with a C-style
//!   API that needs a dedicated integration; the transport and
//!   demux layers are codec-agnostic, so adding it later is a
//!   decoder swap, not a redesign.
//! * **No Opus, no ALAC.** symphonia 0.5 has no Opus decoder;
//!   `decode_audio` reports an unsupported codec as an error.
//! * **VP9 decode is decided: unsupported.** WebM *demux* is
//!   proven here (tracks listed, frames readable); the VP9
//!   bitstream in the fixture has no decoder in this tree —
//!   the offline registry holds no `rav1d`, `dav1d`, or
//!   `libvpx` crate (only `rav1e`, an encoder). WebM/VP9
//!   reports unsupported (`canPlayType` `""`, an `error` event
//!   and `NETWORK_NO_SOURCE`, fall through to the next
//!   `<source>`), never a fake frame. See `canplay`.
//! * **No audio sink here** — the sink lives in
//!   [`sink`](crate::sink) (cpal device, null fallback) and
//!   the A/V clock with it.
//! * **The `<video>` element** lives in `pluto-js`
//!   (`bindings/media.rs`); the state machine it drives is
//!   [`session::MediaSession`].
//!
//! # Tainting / readback policy
//!
//! The per-frame `origin_clean` on [`pluto_ipc::VideoFrame`] is
//! a deliberate decision, not inherited from the image rule
//! (M7 canvas taint is spec-stricter by design: `<img>` has
//! no CORS-approval record, so it over-taints). For video:
//!
//! * Same-origin or `data:` media — clean.
//! * Cross-origin media — clean **only when its fetch was
//!   CORS-approved**, with the approval recorded on the fetch
//!   (the fetcher must carry the flag; `pluto-net` does not
//!   record it yet — Phase 2 wiring).
//! * A tainted frame still paints; only `drawImage(video)`
//!   readback (and any pixel access) is blocked.
//!
//! The caller owns the decision: [`video::H264Decoder`] takes
//! `origin_clean` per frame, and the test fixtures are
//! same-origin by definition.

use thiserror::Error;

pub mod audio;
pub mod canplay;
pub mod demux;
pub mod frame;
pub mod session;
pub mod sink;
pub mod video;
pub mod webvtt;
pub mod yuv;

/// Read a test fixture's bytes.
///
/// Fixtures are stored base64-encoded (`tests/fixtures/<name>.b64`)
/// because the snapshot push route is text-only — binary media
/// cannot travel through it. The decoded bytes are byte-identical
/// to the original fixture files (verified by SHA-256 at encode
/// time). Test-only; never ships.
#[cfg(test)]
pub(crate) fn fixture_bytes(name: &str) -> Vec<u8> {
    let path = format!("tests/fixtures/{name}.b64");
    let encoded = std::fs::read_to_string(&path).expect("fixture present");
    decode_base64(&encoded).expect("fixture decodes")
}

/// Minimal standard-base64 decoder (test-only; avoids a new
/// dependency for one helper). Whitespace is skipped, `=` pads.
#[cfg(test)]
fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(format!("bad base64 byte {c}")),
        }
    }
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(4) {
        return Err("base64 length not a multiple of 4".to_string());
    }
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.as_chunks::<4>().0 {
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        let n = ((val(chunk[0])? as u32) << 18)
            | ((val(chunk[1])? as u32) << 12)
            | ((val(chunk[2])? as u32) << 6)
            | (val(chunk[3])? as u32);
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

pub use audio::{DecodedAudio, decode_audio};
pub use canplay::{CanPlay, can_play_type};
pub use demux::{Mp4Demux, WebmDemux, WebmTrackInfo, avcc_to_annexb};
pub use frame::FrameChannel;
pub use session::{
    AdvanceOutput, MediaEvent, MediaKind, MediaSession, NetworkState, ReadyState,
    MEDIA_ERR_DECODE, MEDIA_ERR_NETWORK, MEDIA_ERR_SRC_NOT_SUPPORTED,
};
pub use sink::{AudioSink, CpalSink, MediaClock, NullSink};
pub use video::H264Decoder;
pub use webvtt::{Cue, TextTrack, TrackKind, parse_webvtt};
pub use yuv::yuv420_to_rgba;

/// Everything that can go wrong decoding media.
#[derive(Debug, Error)]
pub enum MediaError {
    /// The container could not be parsed.
    #[error("demux error: {0}")]
    Demux(String),
    /// The codec could not be initialised or a frame failed to decode.
    #[error("decode error: {0}")]
    Decode(String),
    /// The codec or container is not one this crate decodes.
    #[error("unsupported media: {0}")]
    Unsupported(String),
}

impl From<symphonia::core::errors::Error> for MediaError {
    fn from(e: symphonia::core::errors::Error) -> Self {
        match e {
            symphonia::core::errors::Error::Unsupported(msg) => {
                MediaError::Unsupported(msg.to_string())
            }
            _ => MediaError::Decode(e.to_string()),
        }
    }
}

impl From<mp4::Error> for MediaError {
    fn from(e: mp4::Error) -> Self {
        MediaError::Demux(e.to_string())
    }
}

impl From<openh264::Error> for MediaError {
    fn from(e: openh264::Error) -> Self {
        MediaError::Decode(e.to_string())
    }
}

impl From<matroska_demuxer::DemuxError> for MediaError {
    fn from(e: matroska_demuxer::DemuxError) -> Self {
        MediaError::Demux(e.to_string())
    }
}
