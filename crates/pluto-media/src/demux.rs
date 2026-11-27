//! Container demuxing: MP4 (`mp4`) and WebM (`matroska-demuxer`).
//!
//! The demuxers own the bytes and hand out samples; the
//! video decoder owns the bitstream. MP4 H.264 samples are
//! stored in AVCC format (length-prefixed NAL units, no
//! parameter sets in-band) and must be converted to Annex B
//! (start codes + SPS/PPS) before openh264 will decode them —
//! see [`avcc_to_annexb`].

use crate::MediaError;
use std::io::Cursor;

// ---------------------------------------------------------------- MP4

/// An opened MP4 file, with its video (and audio) tracks listed.
pub struct Mp4Demux {
    reader: mp4::Mp4Reader<Cursor<Vec<u8>>>,
}

/// The video track of an MP4 file, as the decoder needs it.
#[derive(Debug, Clone)]
pub struct Mp4VideoTrack {
    /// Track id, for [`Mp4Demux::read_video_sample`].
    pub track_id: u32,
    /// Coded width in pixels.
    pub width: u16,
    /// Coded height in pixels.
    pub height: u16,
    /// Track timescale (ticks per second).
    pub timescale: u32,
    /// Number of samples in the track.
    pub sample_count: u32,
    /// Nominal frame rate (frames per second).
    pub frame_rate: f64,
}

/// One demuxed video sample, still in AVCC form.
#[derive(Debug)]
pub struct Mp4VideoSample {
    /// Presentation timestamp in milliseconds.
    pub pts_ms: f64,
    /// Duration in milliseconds.
    pub duration_ms: f64,
    /// True for keyframes (sync samples).
    pub is_sync: bool,
    /// AVCC bytes: length-prefixed NAL units.
    pub bytes: Vec<u8>,
}

impl Mp4Demux {
    /// Parse the header of a complete MP4 file.
    pub fn open(bytes: Vec<u8>) -> Result<Self, MediaError> {
        let size = bytes.len() as u64;
        let reader = mp4::Mp4Reader::read_header(Cursor::new(bytes), size)?;
        Ok(Self { reader })
    }

    /// The first H.264 video track, if the file has one.
    pub fn video_track(&self) -> Option<Mp4VideoTrack> {
        for track in self.reader.tracks().values() {
            if track.media_type().ok()? != mp4::MediaType::H264 {
                continue;
            }
            return Some(Mp4VideoTrack {
                track_id: track.track_id(),
                width: track.width(),
                height: track.height(),
                timescale: track.timescale(),
                sample_count: track.sample_count(),
                frame_rate: track.frame_rate(),
            });
        }
        None
    }

    /// The first audio track's id, if the file has one.
    pub fn audio_track_id(&self) -> Option<u32> {
        for track in self.reader.tracks().values() {
            if track.track_type().ok()? == mp4::TrackType::Audio {
                return Some(track.track_id());
            }
        }
        None
    }

    /// SPS/PPS as Annex B (start code + SPS + start code + PPS).
    /// MP4 stores the parameter sets in the header, not in the
    /// samples, so the decoder needs this prepended before the
    /// first sample (and before each keyframe).
    pub fn parameter_sets_annexb(&self, track_id: u32) -> Result<Vec<u8>, MediaError> {
        let track = self
            .reader
            .tracks()
            .get(&track_id)
            .ok_or_else(|| MediaError::Demux(format!("no MP4 track {track_id}")))?;
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(track.sequence_parameter_set()?);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(track.picture_parameter_set()?);
        Ok(out)
    }

    /// Read one video sample. `sample_id` is 1-based, up to
    /// `sample_count` (the `mp4` crate's convention).
    pub fn read_video_sample(
        &mut self,
        track_id: u32,
        sample_id: u32,
    ) -> Result<Option<Mp4VideoSample>, MediaError> {
        let Some(track) = self.reader.tracks().get(&track_id) else {
            return Err(MediaError::Demux(format!("no MP4 track {track_id}")));
        };
        let timescale = track.timescale() as f64;
        let Some(sample) = self.reader.read_sample(track_id, sample_id)? else {
            return Ok(None);
        };
        Ok(Some(Mp4VideoSample {
            pts_ms: sample.start_time as f64 * 1000.0 / timescale,
            duration_ms: sample.duration as f64 * 1000.0 / timescale,
            is_sync: sample.is_sync,
            bytes: sample.bytes.to_vec(),
        }))
    }
}

/// Convert one AVCC sample to Annex B.
///
/// MP4 stores each NAL unit prefixed by its length (usually
/// 4 bytes, but 1 or 2 occur); Annex B separates NAL units
/// with the `00 00 00 01` start code, which is what openh264
/// consumes. The length size is detected by trial: try 4,
/// then 2, then 1, and accept the first size whose lengths
/// parse exactly to the end of the sample. A sample that
/// parses under no size is an error, not a guess.
pub fn avcc_to_annexb(avcc: &[u8], out: &mut Vec<u8>) -> Result<(), MediaError> {
    for len_size in [4usize, 2, 1] {
        let mut tmp = Vec::with_capacity(avcc.len() + 16);
        if try_avcc(avcc, len_size, &mut tmp) {
            out.extend_from_slice(&tmp);
            return Ok(());
        }
    }
    Err(MediaError::Demux(
        "AVCC sample does not parse under any NAL length size".to_string(),
    ))
}

fn try_avcc(avcc: &[u8], len_size: usize, out: &mut Vec<u8>) -> bool {
    let mut rest = avcc;
    while !rest.is_empty() {
        if rest.len() < len_size {
            return false;
        }
        let mut n = 0usize;
        for &b in rest.iter().take(len_size) {
            n = (n << 8) | b as usize;
        }
        rest = &rest[len_size..];
        if n == 0 || rest.len() < n {
            return false;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&rest[..n]);
        rest = &rest[n..];
    }
    true
}

// ---------------------------------------------------------------- WebM

/// What kind of WebM track this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebmTrackKind {
    Video,
    Audio,
    Other,
}

/// One track of a WebM file, as the decoder needs it.
#[derive(Debug, Clone)]
pub struct WebmTrackInfo {
    /// Track number, for matching frames.
    pub track_number: u64,
    /// Video, audio, or something else.
    pub kind: WebmTrackKind,
    /// Codec id string, e.g. `"V_VP9"`, `"A_OPUS"`.
    pub codec_id: String,
    /// Coded width in pixels (video tracks).
    pub width: Option<u64>,
    /// Coded height in pixels (video tracks).
    pub height: Option<u64>,
}

/// An opened WebM file.
///
/// Phase 1 scope: demux only. The fixture's video track is
/// VP9, which has no decoder in this crate yet — `next_frame`
/// proves the frames are readable; decoding them is Phase 2.
pub struct WebmDemux {
    file: matroska_demuxer::MatroskaFile<Cursor<Vec<u8>>>,
}

impl WebmDemux {
    /// Parse the header of a complete WebM file.
    pub fn open(bytes: Vec<u8>) -> Result<Self, MediaError> {
        let file = matroska_demuxer::MatroskaFile::open(Cursor::new(bytes))?;
        Ok(Self { file })
    }

    /// All tracks in the file.
    pub fn tracks(&self) -> Vec<WebmTrackInfo> {
        self.file
            .tracks()
            .iter()
            .map(|t| {
                let kind = match t.track_type() {
                    matroska_demuxer::TrackType::Video => WebmTrackKind::Video,
                    matroska_demuxer::TrackType::Audio => WebmTrackKind::Audio,
                    _ => WebmTrackKind::Other,
                };
                let (width, height) = match t.video() {
                    Some(v) => (
                        Some(v.pixel_width().get()),
                        Some(v.pixel_height().get()),
                    ),
                    None => (None, None),
                };
                WebmTrackInfo {
                    track_number: t.track_number().get(),
                    kind,
                    codec_id: t.codec_id().to_string(),
                    width,
                    height,
                }
            })
            .collect()
    }

    /// Duration in seconds, if the file records one.
    /// `Info::duration` is in nanoseconds.
    pub fn duration_secs(&self) -> Option<f64> {
        self.file.info().duration().map(|ns| ns / 1_000_000_000.0)
    }

    /// Read the next frame from any track. Returns `Ok(None)`
    /// at end of file.
    pub fn next_frame(&mut self) -> Result<Option<matroska_demuxer::Frame>, MediaError> {
        let mut frame = matroska_demuxer::Frame::default();
        if self.file.next_frame(&mut frame)? {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_length_prefixes_become_start_codes() {
        let avcc: &[u8] = &[0, 0, 0, 2, 0x65, 0x88, 0, 0, 0, 1, 0x41];
        let mut out = Vec::new();
        avcc_to_annexb(avcc, &mut out).expect("parses");
        assert_eq!(
            out,
            vec![0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1, 0x41]
        );
    }

    #[test]
    fn avcc_garbage_is_an_error() {
        let mut out = Vec::new();
        assert!(avcc_to_annexb(&[0xff, 0xff, 0xff, 0xff, 0x01], &mut out).is_err());
    }

    #[test]
    fn mp4_fixture_has_h264_video_track() {
        let bytes = crate::fixture_bytes("bbb-360-10s.mp4");
        let demux = Mp4Demux::open(bytes).expect("mp4 parses");
        let track = demux.video_track().expect("h264 video track");
        assert_eq!(track.width, 640);
        assert_eq!(track.height, 360);
        assert!(track.sample_count > 100, "10 s of video has samples");
        assert!(track.frame_rate > 20.0 && track.frame_rate < 40.0);
    }

    #[test]
    fn mp4_fixture_samples_read_and_convert() {
        let bytes = crate::fixture_bytes("bbb-360-10s.mp4");
        let mut demux = Mp4Demux::open(bytes).expect("mp4 parses");
        let track = demux.video_track().expect("h264 video track");
        let prefix = demux
            .parameter_sets_annexb(track.track_id)
            .expect("sps/pps present");
        assert!(prefix.starts_with(&[0, 0, 0, 1]));
        let mut converted = 0u32;
        for sample_id in 1..=track.sample_count.min(10) {
            let Some(sample) = demux
                .read_video_sample(track.track_id, sample_id)
                .expect("sample reads")
            else {
                continue;
            };
            let mut annexb = Vec::new();
            avcc_to_annexb(&sample.bytes, &mut annexb).expect("sample converts");
            assert!(annexb.starts_with(&[0, 0, 0, 1]));
            converted += 1;
        }
        assert!(converted >= 5, "first samples convert, got {converted}");
    }

    #[test]
    fn webm_fixture_demuxes_to_vp9_frames() {
        let bytes = crate::fixture_bytes("bbb-360-10s.webm");
        let mut demux = WebmDemux::open(bytes).expect("webm parses");
        let tracks = demux.tracks();
        let video = tracks
            .iter()
            .find(|t| t.kind == WebmTrackKind::Video)
            .expect("vp9 video track");
        assert_eq!(video.codec_id, "V_VP9");
        assert_eq!(video.width, Some(640));
        assert_eq!(video.height, Some(360));
        let mut frames = 0u32;
        let mut video_frames = 0u32;
        while let Some(frame) = demux.next_frame().expect("frame reads") {
            frames += 1;
            if frame.track == video.track_number {
                assert!(!frame.data.is_empty());
                video_frames += 1;
            }
            if frames >= 50 {
                break;
            }
        }
        assert!(video_frames > 0, "vp9 frames readable (decode is Phase 2)");
    }
}
