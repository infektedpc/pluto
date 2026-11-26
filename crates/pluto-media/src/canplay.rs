//! `canPlayType` / codec support table (Stage 16 Phase 2,
//! Rec 4).
//!
//! One table, used by the JS `canPlayType` binding and by the
//! media session's resource selection, so the two can never
//! disagree about what plays. The answers are honest about the
//! Phase 1 decode stack:
//!
//! * Probably: MP4/H.264 video, MP3, AAC-in-MP4, WAV/PCM, FLAC.
//! * Maybe: Vorbis-in-Ogg/WebM audio (symphonia decodes it, but
//!   the container is only probed, not guaranteed).
//! * Empty (unsupported): VP9, AV1, Opus, ALAC — and anything
//!   unrecognised. An empty answer is a fact, not a shrug: the
//!   session turns it into an `error` event, never silence.
//!
//! # The VP9 decision (Phase 2, recorded here so it is not
//! re-litigated silently)
//!
//! WebM/VP9 is the codec in the `bbb-360-10s.webm` fixture and
//! the fixture demuxes (tracks listed, frames readable). There
//! is still no VP9 decoder in this tree: the offline registry
//! holds no `rav1d`, no `dav1d`, no `libvpx` / `vpx` crate at
//! all (only `rav1e`, an encoder, and `ravif`). A VP9 decoder
//! would be a new large dependency with a C-style API — a
//! dedicated integration, not a Phase 2 line item — so
//! WebM/VP9 reports unsupported: `canPlayType` returns `""`
//! and the session raises an error event and falls through to
//! the next `<source>`. Nothing in this crate pretends to
//! decode VP9.

use crate::MediaError;

/// What `canPlayType` returns: `""`, `"maybe"`, or `"probably"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanPlay {
    /// Cannot play this type.
    Empty,
    /// Might play (decoder exists; container/codec pairing unproven).
    Maybe,
    /// Confident it plays (decoder + container proven on fixtures).
    Probably,
}

impl CanPlay {
    /// The string the JS binding returns.
    pub fn as_str(self) -> &'static str {
        match self {
            CanPlay::Empty => "",
            CanPlay::Maybe => "maybe",
            CanPlay::Probably => "probably",
        }
    }
}

/// Normalise a MIME type for matching: lowercase, strip
/// parameters after `;` (the codec list is parsed separately).
fn mime_of(mime_type: &str) -> String {
    mime_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// The `codecs="..."` parameter of a MIME type, if present,
/// lowercased and split on commas.
fn codecs_of(mime_type: &str) -> Vec<String> {
    let lower = mime_type.to_ascii_lowercase();
    let Some(start) = lower.find("codecs") else {
        return Vec::new();
    };
    let rest = &lower[start..];
    let Some(eq) = rest.find('=') else {
        return Vec::new();
    };
    let after = rest[eq + 1..].trim_start_matches([' ', '"']);
    let list = after.split(';').next().unwrap_or("").trim_end_matches('"');
    list.split(',')
        .map(|c| c.trim().trim_matches('"').to_string())
        .filter(|c| !c.is_empty())
        .collect()
}

/// Whether a single codec string is one this tree decodes.
/// `None` (no codec given) is not a claim either way — the
/// caller falls back to the container-level answer.
fn codec_supported(codec: &str) -> Option<bool> {
    let c = codec.trim().to_ascii_lowercase();
    // Strip the avc1 profile suffix (`avc1.640028` → `avc1`).
    let base = c.split('.').next().unwrap_or("");
    match base {
        "avc1" | "avc3" | "mp4a" | "mp3" | "flac" | "vorbis" | "pcm" | "lpcm" | "ulaw"
        | "alaw" => Some(true),
        // mp4a.40.2 is AAC-LC (decodes); mp4a.40.5 is HE-AAC
        // (symphonia's AAC handles it as AAC); anything else in
        // the mp4a family is unknown, so stay silent.
        _ if c.starts_with("mp4a.40.") => Some(true),
        "vp9" | "vp09" | "vp8" | "vp08" | "av01" | "av1" | "opus" | "alac" => Some(false),
        _ => None,
    }
}

/// The `canPlayType` answer for a MIME type string (with an
/// optional `codecs` parameter).
pub fn can_play_type(mime_type: &str) -> CanPlay {
    let mime = mime_of(mime_type);
    let codecs = codecs_of(mime_type);
    // A named unsupported codec vetoes the container, whatever
    // the container is; a named supported codec confirms it.
    for codec in &codecs {
        match codec_supported(codec) {
            Some(false) => return CanPlay::Empty,
            Some(true) => return CanPlay::Probably,
            None => {}
        }
    }
    if !codecs.is_empty() {
        // Only unknown codecs named: no claim.
        return CanPlay::Maybe;
    }
    match mime.as_str() {
        "video/mp4" | "video/x-m4v" => CanPlay::Probably,
        "audio/mpeg" | "audio/mp3" | "audio/aac" | "audio/x-aac" | "audio/mp4"
        | "audio/wav" | "audio/x-wav" | "audio/flac" | "audio/x-flac" => CanPlay::Probably,
        "audio/ogg" | "video/ogg" | "audio/webm" => CanPlay::Maybe,
        // WebM video is VP9 (or AV1) in practice and neither
        // decodes here — see the module docs.
        "video/webm" => CanPlay::Empty,
        _ => CanPlay::Empty,
    }
}

/// Is this WebM codec id decodable here? `V_VP9` / `V_VP8` /
/// `V_AV1` are not (see the module docs); `A_OPUS` is not
/// either — symphonia 0.5 has no Opus decoder.
pub fn webm_codec_supported(codec_id: &str) -> bool {
    matches!(codec_id, "V_MPEG4/ISO/AVC" | "A_VORBIS" | "A_PCM/INT/LIT" | "A_FLAC" | "A_AAC")
}

/// The unsupported-codec error the session raises for a WebM
/// track it demuxed but cannot decode.
pub fn unsupported_codec_error(codec_id: &str) -> MediaError {
    MediaError::Unsupported(format!(
        "{codec_id} has no decoder in this tree (no VP9/Opus decoder in the offline registry); use H.264 video or MP3/AAC/Vorbis audio"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_mp4_is_probably_and_vp9_is_empty() {
        assert_eq!(can_play_type("video/mp4"), CanPlay::Probably);
        assert_eq!(
            can_play_type("video/mp4; codecs=\"avc1.640028\""),
            CanPlay::Probably
        );
        assert_eq!(can_play_type("video/webm"), CanPlay::Empty);
        assert_eq!(
            can_play_type("video/webm; codecs=\"vp9\""),
            CanPlay::Empty
        );
        assert_eq!(
            can_play_type("video/mp4; codecs=\"vp9\""),
            CanPlay::Empty,
            "a VP9 codec vetoes even an MP4 container"
        );
    }

    #[test]
    fn audio_types_match_the_decode_stack() {
        assert_eq!(can_play_type("audio/mpeg"), CanPlay::Probably);
        assert_eq!(can_play_type("audio/wav"), CanPlay::Probably);
        assert_eq!(can_play_type("audio/flac"), CanPlay::Probably);
        assert_eq!(can_play_type("audio/ogg"), CanPlay::Maybe);
        assert_eq!(can_play_type("audio/ogg; codecs=\"opus\""), CanPlay::Empty);
        assert_eq!(can_play_type("audio/ogg; codecs=\"vorbis\""), CanPlay::Probably);
        assert_eq!(can_play_type("video/avi"), CanPlay::Empty);
        assert_eq!(can_play_type("application/octet-stream"), CanPlay::Empty);
    }

    #[test]
    fn webm_codec_table_matches_the_vp9_decision() {
        assert!(!webm_codec_supported("V_VP9"));
        assert!(!webm_codec_supported("V_AV1"));
        assert!(!webm_codec_supported("A_OPUS"));
        assert!(webm_codec_supported("A_VORBIS"));
    }

    #[test]
    fn webm_fixture_codec_is_unsupported_with_a_named_reason() {
        let bytes = crate::fixture_bytes("bbb-360-10s.webm");
        let demux = crate::demux::WebmDemux::open(bytes).expect("webm parses");
        let video = demux
            .tracks()
            .into_iter()
            .find(|t| t.kind == crate::demux::WebmTrackKind::Video)
            .expect("video track");
        assert_eq!(video.codec_id, "V_VP9");
        assert!(!webm_codec_supported(&video.codec_id));
        let err = unsupported_codec_error(&video.codec_id);
        assert!(matches!(err, MediaError::Unsupported(_)));
        assert!(err.to_string().contains("V_VP9"));
    }
}
