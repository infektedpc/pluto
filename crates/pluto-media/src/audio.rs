//! Audio decoding via symphonia.
//!
//! Scope (honest): MP3, FLAC, Vorbis-in-Ogg, WAV/PCM and
//! AAC-in-MP4 decode to interleaved 16-bit PCM. Opus and ALAC
//! are NOT covered — symphonia 0.5 has no Opus decoder — and
//! surface as [`MediaError::Unsupported`], never silence.
//! The caller owns the bytes (the media thread already holds
//! them); no sink is created here — the output device is
//! Rec 6's dependency.

use crate::MediaError;
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio: interleaved 16-bit PCM.
#[derive(Debug)]
pub struct DecodedAudio {
    /// Codec name (e.g. `"mp3"`).
    pub codec: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Interleaved samples, `channels` per frame.
    pub samples: Vec<i16>,
}

impl DecodedAudio {
    /// Duration in seconds, from sample count and rate.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
    }
}

/// Readable name for a symphonia codec code. Unknown codes
/// fall back to symphonia's own hex Display (`0x1003`).
fn codec_name(codec: symphonia::core::codecs::CodecType) -> String {
    use symphonia::core::codecs::*;
    match codec {
        CODEC_TYPE_MP1 => "mp1".to_string(),
        CODEC_TYPE_MP2 => "mp2".to_string(),
        CODEC_TYPE_MP3 => "mp3".to_string(),
        CODEC_TYPE_AAC => "aac".to_string(),
        CODEC_TYPE_VORBIS => "vorbis".to_string(),
        CODEC_TYPE_FLAC => "flac".to_string(),
        CODEC_TYPE_OPUS => "opus".to_string(),
        CODEC_TYPE_ALAC => "alac".to_string(),
        _ => codec.to_string(),
    }
}

/// Decode a whole audio file into interleaved PCM.
///
/// `bytes` is the complete file (the fetcher holds it; a
/// streaming decoder is Rec 12's work). The format is probed
/// from the bytes, not from a file extension — a mislabelled
/// extension decodes fine and an unsupported codec errors.
pub fn decode_audio(bytes: Vec<u8>) -> Result<DecodedAudio, MediaError> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| match e {
            Error::Unsupported(msg) => MediaError::Unsupported(msg.to_string()),
            _ => MediaError::Demux(e.to_string()),
        })?;
    let mut reader = probed.format;
    let track = reader
        .default_track()
        .ok_or_else(|| MediaError::Demux("audio file has no tracks".to_string()))?;
    // symphonia's `CodecType` is a u32 newtype whose Display
    // prints the hex code (`0x1003`), not a name — map the
    // codecs this crate claims to support to readable names.
    let codec_name = codec_name(track.codec_params.codec);
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| match e {
            Error::Unsupported(msg) => MediaError::Unsupported(msg.to_string()),
            _ => MediaError::Decode(e.to_string()),
        })?;

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut samples: Vec<i16> = Vec::new();
    let mut consecutive_errors = 0u32;

    loop {
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(MediaError::from(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet in an otherwise good file skips;
            // ten bad packets in a row means the stream is dead.
            Err(Error::DecodeError(_)) => {
                consecutive_errors += 1;
                if consecutive_errors > 10 {
                    return Err(MediaError::Decode(
                        "too many consecutive corrupt audio packets".to_string(),
                    ));
                }
                continue;
            }
            Err(e) => return Err(MediaError::from(e)),
        };
        consecutive_errors = 0;
        let spec = *decoded.spec();
        sample_rate = spec.rate;
        channels = spec.channels.count() as u16;
        let mut buf = SampleBuffer::<i16>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    if sample_rate == 0 {
        return Err(MediaError::Decode(
            "audio file decoded no samples".to_string(),
        ));
    }
    Ok(DecodedAudio {
        codec: codec_name,
        sample_rate,
        channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mp3_fixture_decodes_to_44100hz_stereo_pcm() {
        let bytes = crate::fixture_bytes("sample-3s.mp3");
        let audio = decode_audio(bytes).expect("mp3 decodes");
        assert_eq!(audio.codec, "mp3");
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.channels, 2);
        assert!(!audio.samples.is_empty(), "decoded samples exist");
        // 3 s at 44.1 kHz stereo ≈ 264,600 interleaved samples.
        let secs = audio.duration_secs();
        assert!(
            (2.5..=3.5).contains(&secs),
            "duration ~3 s, got {secs:.2} s"
        );
    }

    #[test]
    fn garbage_is_an_error_not_silence() {
        let err = decode_audio(b"this is not audio at all".to_vec()).unwrap_err();
        assert!(matches!(
            err,
            MediaError::Demux(_) | MediaError::Unsupported(_)
        ));
    }
}
