//! The `HTMLMediaElement` state machine (Stage 16 Phase 2,
//! Rec 4).
//!
//! One [`MediaSession`] per element, owned by the JS bindings
//! (like `image_states`): it owns the bytes, the demuxer, the
//! decoder, the audio sink, and the clock, and it emits the
//! spec event set for the bindings to fire into the realm.
//!
//! # Resource selection
//!
//! The bindings pick candidates (`src` attribute, else
//! `<source>` children in document order); the session only
//! ever sees one candidate at a time via [`MediaSession::load`].
//! A load that fails (unsupported codec/container, corrupt
//! bytes, network failure delivered as empty bytes) reports
//! an error and the bindings move to the next candidate.
//! Unsupported is an error, never silence: VP9, Opus, and
//! unknown containers raise `MEDIA_ERR_SRC_NOT_SUPPORTED`
//! (code 4) with a named reason.
//!
//! # States
//!
//! `readyState` / `networkState` follow the HTML Standard
//! values (`HAVE_NOTHING` → `HAVE_METADATA` →
//! `HAVE_CURRENT_DATA` → `HAVE_FUTURE_DATA` →
//! `HAVE_ENOUGH_DATA`; `NETWORK_EMPTY` / `IDLE` / `LOADING` /
//! `NO_SOURCE`). The whole file is buffered before `load`
//! returns (interim architecture — Rec 12 streams), so once
//! the first frame + audio head are decoded the session may
//! claim `HAVE_ENOUGH_DATA`: nothing further can stall it.
//!
//! # Clock
//!
//! [`MediaClock`](crate::sink::MediaClock) (audio device
//! master, `Instant` fallback) is the only time source.
//! `advance()` runs on the tab tick and decodes ahead of the
//! clock; it never counts ticks.
//!
//! # Seeking
//!
//! Seeking is in-memory over the buffered whole file — no
//! HTTP Range is involved (`ResourceRequest` has no range
//! field, and none was added: the bytes are already here).
//! Video seeks to the nearest sync sample at or before the
//! target and re-creates the decoder; audio seeks through
//! symphonia (accurate where the container supports it) with
//! a restart + decode-forward fallback.
//!
//! # Interim limits (honest, not hidden)
//!
//! * Audio-only files decode fully at `load` (exact duration
//!   and instant seek); a long file pays a load-time decode.
//!   Streaming decode is Rec 12 work.
//! * MP4 video+audio decodes audio incrementally in
//!   `advance()`; after a seek the audio catches up by
//!   decode-forward when symphonia's seek fails.
//! * WebM/VP9 and WebM/Opus are unsupported (see
//!   `canplay`); a WebM file with a Vorbis audio track and no
//!   video is decoded by symphonia's Ogg reader only when the
//!   bytes probe as Ogg — a Matroska-wrapped Vorbis track
//!   reports unsupported.

use crate::audio::DecodedAudio;
use crate::canplay::{unsupported_codec_error, webm_codec_supported};
use crate::demux::{Mp4Demux, Mp4VideoTrack, WebmDemux, WebmTrackKind, avcc_to_annexb};
use crate::sink::{AudioSink, CpalSink, FrameDue, MediaClock, NullSink, frame_presentation};
use crate::video::H264Decoder;
use crate::{MediaError, TextTrack};
use pluto_ipc::VideoFrame;
use std::collections::VecDeque;
use std::sync::Arc;

/// `networkState` values (HTML Standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkState {
    Empty = 0,
    Idle = 1,
    Loading = 2,
    NoSource = 3,
}

/// `readyState` values (HTML Standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    HaveNothing = 0,
    HaveMetadata = 1,
    HaveCurrentData = 2,
    HaveFutureData = 3,
    HaveEnoughData = 4,
}

/// The media events the session emits, in order, for the
/// bindings to fire into JS. `Error` carries the message; the
/// numeric code lives on the session (`error_code()`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaEvent {
    LoadedMetadata,
    LoadedData,
    CanPlay,
    CanPlayThrough,
    Play,
    Playing,
    Pause,
    TimeUpdate,
    Seeking,
    Seeked,
    Ended,
    Error(String),
    Progress,
}

/// What kind of media this session holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    AudioOnly,
    Video,
}

/// `MediaError` codes (HTML Standard).
pub const MEDIA_ERR_ABORTED: u32 = 1;
pub const MEDIA_ERR_NETWORK: u32 = 2;
pub const MEDIA_ERR_DECODE: u32 = 3;
pub const MEDIA_ERR_SRC_NOT_SUPPORTED: u32 = 4;

/// How far ahead of the clock the decoder works (seconds).
const DECODE_AHEAD_SECS: f64 = 0.5;
/// Maximum decoded video frames held ahead of the clock.
/// 16 frames ≈ 0.5 s at 30 fps — the decode horizon — so the
/// queue never holds more than the horizon's worth.
const MAX_FRAMES_AHEAD: usize = 16;
/// Wall-time budget for one `decode_ahead` call (see its docs).
/// 64 ms — about four tab ticks — is deliberately generous:
/// catching up after a seek can mean decoding a whole GOP
/// (the BBB fixture's keyframes are ~8.3 s apart, so a seek to
/// mid-file decodes ~150 frames), and a debug build's C-only
/// openh264 needs ~25 ms/frame. The budget bounds the stall
/// per tick; the catch-up spreads over ticks and the clock
/// stays in sync because frames that arrive late are dropped,
/// never queued. Release builds decode ~10× faster, so the
/// same budget buys far more frames there.
const DECODE_BUDGET: std::time::Duration = std::time::Duration::from_millis(64);
/// Catch-up budget: when the presentation queue is empty at
/// the start of `decode_ahead` (starved — nothing to present),
/// the tick is better spent decoding than idling: a debug
/// build decodes at ~realtime, so a normal budget never closes
/// a seek deficit (decode and the clock advance at the same
/// rate). 250 ms buys ~10 frames per tick and closes a
/// typical deficit in a few ticks; it binds only while
/// starved, so steady-state ticks stay cheap.
const CATCH_UP_BUDGET: std::time::Duration = std::time::Duration::from_millis(250);
/// `timeupdate` fires at most this often (seconds).
const TIMEUPDATE_INTERVAL_SECS: f64 = 0.25;

/// One media element's playback state.
pub struct MediaSession {
    kind: MediaKind,
    /// The element this session belongs to (its `DomKey` as a
    /// `u64`) — stamped onto every decoded `VideoFrame` so the
    /// shell can key frames per element. Set by the bindings at
    /// registration; 0 until then (frames decoded before
    /// registration carry 0 and are re-keyed by the renderer
    /// from the tick's key — the tick returns (key, frame)
    /// pairs, so the frame's own id is belt-and-braces).
    element_id: u64,
    network_state: NetworkState,
    ready_state: ReadyState,
    duration: Option<f64>,
    video_size: Option<(u32, u32)>,
    paused: bool,
    ended: bool,
    seeking: bool,
    /// Media time when paused, or the seek target while seeking.
    paused_time: f64,
    volume: f32,
    muted: bool,
    loop_: bool,
    origin_clean: bool,
    error: Option<(u32, String)>,
    tracks: Vec<TextTrack>,
    events: VecDeque<MediaEvent>,
    clock: MediaClock,
    sink: Option<Box<dyn AudioSink>>,
    source: Option<Source>,
    /// Decoded video frames ahead of the clock, in pts order
    /// (at most `MAX_FRAMES_AHEAD`). This is a QUEUE, not the
    /// latest-only `FrameChannel`: decode-ahead means the
    /// newest frame is ~0.5 s in the future, so a latest-only
    /// slot would overwrite the frame that is due *now* before
    /// it is ever presented. The drop policy lives here too:
    /// when the queue is full the oldest frame is dropped
    /// (stale — the consumer gets the newest due frame, never
    /// a backlog), and `advance()` drops frames whose window
    /// has passed. The `FrameChannel` (latest-only) is the
    /// IPC-side handoff to the shell, one presented frame at a
    /// time — see the renderer.
    decoded: VecDeque<VideoFrame>,
    current_frame: Option<VideoFrame>,
    frames_presented: u64,
    frames_dropped: u64,
    last_timeupdate: f64,
    playing_fired: bool,
    /// Audio decode failure on a video file: video plays on
    /// without audio (recorded, not fatal).
    audio_error: Option<String>,
    /// True once the first frame / first audio has been decoded.
    primed: bool,
}

/// `Mp4Video` carries the demuxer, track index, and decoder, so it is
/// inherently far larger than `AudioOnly`; boxing it would churn every
/// construction and match site for no behavior gain.
#[allow(clippy::large_enum_variant)]
enum Source {
    Mp4Video {
        demux: Mp4Demux,
        track: Mp4VideoTrack,
        /// (pts_ms, is_sync) for every sample, built at load.
        index: Vec<(f64, bool)>,
        next_sample: u32,
        prefix: Vec<u8>,
        decoder: H264Decoder,
        audio: Option<AudioStream>,
        eof: bool,
    },
    AudioOnly {
        audio: AudioStream,
    },
}

/// Incremental / whole-file audio decoding via symphonia.
struct AudioStream {
    reader: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u16,
    /// Decoded PCM so far (interleaved i16).
    pcm: Vec<i16>,
    /// Media time of `pcm[0]`.
    pcm_start_time: f64,
    /// Interleaved samples already pushed to the sink.
    pushed: usize,
    eof: bool,
    /// Bytes the stream was probed from (seek fallback).
    source_bytes: Arc<Vec<u8>>,
}

impl AudioStream {
    /// Probe `bytes` with symphonia and open the first audio
    /// track (a track with a sample rate — video tracks have
    /// none). `decode_all` decodes the whole file now (audio-
    /// only files: exact duration + instant seek).
    fn open(bytes: Arc<Vec<u8>>, decode_all: bool) -> Result<Self, MediaError> {
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;
        let mss = MediaSourceStream::new(
            Box::new(std::io::Cursor::new((*bytes).clone())),
            Default::default(),
        );
        let probed = symphonia::default::get_probe()
            .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
            .map_err(|e| match e {
                symphonia::core::errors::Error::Unsupported(msg) => {
                    MediaError::Unsupported(msg.to_string())
                }
                _ => MediaError::Demux(e.to_string()),
            })?;
        let reader = probed.format;
        let track = reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.sample_rate.is_some())
            .ok_or_else(|| MediaError::Demux("no audio track in media".to_string()))?;
        let track_id = track.id;
        let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count() as u16)
            .unwrap_or(0);
        let params = track.codec_params.clone();
        let decoder = symphonia::default::get_codecs()
            .make(&params, &DecoderOptions::default())
            .map_err(|e| match e {
                symphonia::core::errors::Error::Unsupported(msg) => {
                    MediaError::Unsupported(msg.to_string())
                }
                _ => MediaError::Decode(e.to_string()),
            })?;
        let mut stream = Self {
            reader,
            decoder,
            track_id,
            sample_rate,
            channels,
            pcm: Vec::new(),
            pcm_start_time: 0.0,
            pushed: 0,
            eof: false,
            source_bytes: bytes,
        };
        if decode_all {
            stream.decode_ahead(f64::INFINITY);
        }
        // `decode_ahead` may have failed part-way (corrupt
        // packets are skipped inside); surface a total failure
        // only when nothing decoded at all.
        if decode_all && stream.pcm.is_empty() && stream.eof {
            return Err(MediaError::Decode("audio file decoded no samples".to_string()));
        }
        Ok(stream)
    }

    /// Duration of decoded PCM so far, seconds.
    fn pcm_duration(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            self.pcm.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
        }
    }

    /// Media time of the end of decoded PCM.
    fn pcm_end_time(&self) -> f64 {
        self.pcm_start_time + self.pcm_duration()
    }

    /// Decode packets until PCM reaches `target_time` or EOF.
    /// Corrupt packets are skipped (ten in a row aborts the
    /// stream — same rule as `decode_audio`).
    fn decode_ahead(&mut self, target_time: f64) {
        use symphonia::core::audio::SampleBuffer;
        use symphonia::core::errors::Error;
        let mut consecutive_errors = 0u32;
        while !self.eof && self.pcm_end_time() < target_time {
            let packet = match self.reader.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.eof = true;
                    break;
                }
                Err(_) => {
                    self.eof = true;
                    break;
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(Error::DecodeError(_)) => {
                    consecutive_errors += 1;
                    if consecutive_errors > 10 {
                        self.eof = true;
                    }
                    continue;
                }
                Err(_) => {
                    self.eof = true;
                    break;
                }
            };
            consecutive_errors = 0;
            let spec = *decoded.spec();
            self.sample_rate = spec.rate;
            self.channels = spec.channels.count() as u16;
            let mut buf = SampleBuffer::<i16>::new(decoded.capacity() as u64, spec);
            buf.copy_interleaved_ref(decoded);
            self.pcm.extend_from_slice(buf.samples());
        }
    }

    /// Seek to media time `t`: symphonia seek first; on
    /// failure, re-probe from the start and decode forward.
    /// Returns the media time PCM restarts at (≈ `t`).
    fn seek(&mut self, t: f64) -> f64 {
        use symphonia::core::formats::{SeekMode, SeekTo};
        use symphonia::core::units::Time;
        let sought = self
            .reader
            .seek(
                SeekMode::Accurate,
                SeekTo::Time {
                    time: Time::from(t),
                    track_id: Some(self.track_id),
                },
            )
            .is_ok();
        if sought {
            self.decoder.reset();
            self.pcm.clear();
            self.pcm_start_time = t;
            self.pushed = 0;
            self.eof = false;
            return t;
        }
        // Fallback: re-probe and decode forward to `t`.
        match Self::open(self.source_bytes.clone(), false) {
            Ok(fresh) => {
                let mut fresh = fresh;
                fresh.pcm_start_time = 0.0;
                fresh.decode_ahead(t);
                // Drop PCM before `t` (keep a small head so the
                // first push after the seek has something to play).
                let skip = (t * fresh.sample_rate as f64 * fresh.channels as f64) as usize;
                let skip = skip.min(fresh.pcm.len());
                fresh.pcm.drain(..skip);
                fresh.pcm_start_time = t;
                fresh.pushed = 0;
                *self = fresh;
                t
            }
            Err(_) => {
                // Unrecoverable: stay where we are.
                self.pcm_start_time + self.pcm_duration()
            }
        }
    }
}

/// What `advance()` reports to the renderer: a new frame to
/// send over IPC, and whether playback ended this tick.
/// (Metadata arrival is reported on the load path, not here —
/// the intrinsic size is known as soon as `load` returns.)
#[derive(Debug, Default)]
pub struct AdvanceOutput {
    /// A newly presentable frame (cloned), if the presented
    /// frame changed this tick.
    pub presented_frame: Option<VideoFrame>,
    /// Playback reached the end this tick.
    pub ended: bool,
}

impl MediaSession {
    pub fn new() -> Self {
        Self {
            kind: MediaKind::AudioOnly,
            element_id: 0,
            network_state: NetworkState::Empty,
            ready_state: ReadyState::HaveNothing,
            duration: None,
            video_size: None,
            paused: true,
            ended: false,
            seeking: false,
            paused_time: 0.0,
            volume: 1.0,
            muted: false,
            loop_: false,
            origin_clean: true,
            error: None,
            tracks: Vec::new(),
            events: VecDeque::new(),
            clock: MediaClock::new(),
            sink: None,
            source: None,
            decoded: VecDeque::new(),
            current_frame: None,
            frames_presented: 0,
            frames_dropped: 0,
            last_timeupdate: -1.0,
            playing_fired: false,
            audio_error: None,
            primed: false,
        }
    }

    // --- state readers (the JS bindings) ---

    pub fn kind(&self) -> MediaKind {
        self.kind
    }
    pub fn network_state(&self) -> NetworkState {
        self.network_state
    }

    /// Stamp the owning element's id onto future frames.
    pub fn set_element_id(&mut self, id: u64) {
        self.element_id = id;
    }
    pub fn ready_state(&self) -> ReadyState {
        self.ready_state
    }
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }
    pub fn video_size(&self) -> Option<(u32, u32)> {
        self.video_size
    }
    pub fn paused(&self) -> bool {
        self.paused
    }
    pub fn ended(&self) -> bool {
        self.ended
    }
    pub fn seeking(&self) -> bool {
        self.seeking
    }
    pub fn volume(&self) -> f32 {
        self.volume
    }
    pub fn muted(&self) -> bool {
        self.muted
    }
    pub fn loop_(&self) -> bool {
        self.loop_
    }
    pub fn origin_clean(&self) -> bool {
        self.origin_clean
    }
    pub fn error(&self) -> Option<(u32, String)> {
        self.error.clone()
    }
    pub fn tracks(&self) -> &[TextTrack] {
        &self.tracks
    }
    pub fn audio_error(&self) -> Option<&str> {
        self.audio_error.as_deref()
    }
    pub fn frames_presented(&self) -> u64 {
        self.frames_presented
    }
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }
    /// Whole-file buffered ranges (interim: the file arrives
    /// whole, so one range covers the duration once loaded).
    pub fn buffered(&self) -> Vec<(f64, f64)> {
        match self.duration {
            Some(d) if self.network_state != NetworkState::Empty => vec![(0.0, d)],
            _ => Vec::new(),
        }
    }

    /// No usable source: resource selection exhausted every
    /// candidate (or there were no candidates at all). The
    /// network state is `NO_SOURCE` and the error is
    /// `MEDIA_ERR_SRC_NOT_SUPPORTED` — the element fires one
    /// `error` event and stops. Called by the bindings when
    /// selection fails before any fetch, or when the last
    /// candidate's load reported unsupported.
    pub fn no_source(&mut self) {
        self.reset_for_load();
        self.network_state = NetworkState::NoSource;
        self.ready_state = ReadyState::HaveNothing;
        self.error = Some((
            MEDIA_ERR_SRC_NOT_SUPPORTED,
            "no supported media source".to_string(),
        ));
        self.events
            .push_back(MediaEvent::Error("no supported media source".to_string()));
    }

    /// True once a frame has been presented for this element
    /// (the current frame exists). Layout reads this through
    /// the bindings to mark the video box live.
    pub fn has_frame(&self) -> bool {
        self.current_frame.is_some()
    }

    /// A candidate failed but another remains: clear the
    /// failure so the element reads as loading, not errored,
    /// until selection is exhausted (spec: the `error` event
    /// fires on the element only when the last candidate
    /// fails — an intermediate failure's event belongs to the
    /// `<source>`, and is dropped here with the queued events).
    pub fn clear_failure(&mut self) {
        self.error = None;
        self.network_state = NetworkState::Loading;
        self.events.clear();
    }

    /// Test-only: switch this session to the manual clock,
    /// anchored at the current media time. From here the clock
    /// advances only via [`MediaSession::advance_manual_clock`]
    /// — never with wall time — so how many frames
    /// `advance()` presents over N ticks depends only on the
    /// decode/presentation logic, not on how fast the sandbox
    /// decodes under parallel load. Production sessions never
    /// call this: their clock stays the audio device / `Instant`
    /// fallback. `play()` / `seek()` re-anchor the manual
    /// clock at the (re)start position, exactly like the
    /// wall clock.
    #[cfg(test)]
    pub fn use_manual_clock(&mut self) {
        let t = self.current_time();
        self.clock.set_manual_now(t);
    }

    /// Test-only: advance the manual clock by `dt_secs`
    /// (one 16 ms tab tick is `0.016`). Call once per
    /// `advance()` tick, in place of sleeping for a wall
    /// tick: the session then sees exactly `dt_secs` of media
    /// time per tick, however long the decode work in between
    /// took in real time.
    #[cfg(test)]
    pub fn advance_manual_clock(&mut self, dt_secs: f64) {
        self.clock.advance_manual(dt_secs);
    }

    /// Current media time, seconds: the clock while playing,
    /// the paused/seek position otherwise.
    pub fn current_time(&self) -> f64 {
        if self.paused {
            self.paused_time
        } else {
            self.clock.now(self.sink.as_deref())
        }
    }

    /// Drain queued events, in order.
    pub fn drain_events(&mut self) -> Vec<MediaEvent> {
        self.events.drain(..).collect()
    }

    // --- loading ---

    /// Load one candidate's bytes. `mime_hint` is the
    /// candidate's `type` attribute / response Content-Type
    /// (may be empty — the container is sniffed from the
    /// bytes, not the hint). `origin_clean` is the fetch's
    /// tainting decision (same-origin / `data:` / CORS-
    /// approved cross-origin — see the crate docs).
    pub fn load(&mut self, bytes: Vec<u8>, mime_hint: &str, origin_clean: bool) {
        self.reset_for_load();
        self.origin_clean = origin_clean;
        self.network_state = NetworkState::Loading;
        self.events.push_back(MediaEvent::Progress);
        match self.load_inner(bytes, mime_hint) {
            Ok(()) => {
                self.network_state = NetworkState::Idle;
                self.events.push_back(MediaEvent::LoadedMetadata);
                // Prime the decode-ahead at load (preload=auto):
                // a paused element already holds its first frame
                // + audio head, so it reaches HAVE_ENOUGH_DATA
                // without playing. The whole file is buffered, so
                // nothing further can stall it.
                self.decode_ahead(0.0);
                if self.primed {
                    self.ready_state = ReadyState::HaveEnoughData;
                    self.events.push_back(MediaEvent::LoadedData);
                    self.events.push_back(MediaEvent::CanPlay);
                    self.events.push_back(MediaEvent::CanPlayThrough);
                }
            }
            Err((code, message)) => {
                self.network_state = NetworkState::NoSource;
                self.ready_state = ReadyState::HaveNothing;
                self.error = Some((code, message.clone()));
                self.events.push_back(MediaEvent::Error(message));
            }
        }
    }

    fn reset_for_load(&mut self) {
        self.source = None;
        self.duration = None;
        self.video_size = None;
        self.paused = true;
        self.ended = false;
        self.seeking = false;
        self.paused_time = 0.0;
        self.error = None;
        self.decoded.clear();
        self.current_frame = None;
        self.primed = false;
        self.playing_fired = false;
        self.audio_error = None;
        if let Some(sink) = self.sink.as_mut() {
            sink.pause();
            sink.reset_position();
        }
    }

    fn load_inner(&mut self, bytes: Vec<u8>, mime_hint: &str) -> Result<(), (u32, String)> {
        if bytes.is_empty() {
            return Err((
                MEDIA_ERR_NETWORK,
                "media fetch delivered no bytes".to_string(),
            ));
        }
        // Container sniff (bytes, not the hint): MP4 `ftyp`,
        // WebM EBML magic, else an audio probe.
        if is_mp4(&bytes) {
            return self.load_mp4(bytes);
        }
        if is_webm(&bytes) {
            return self.load_webm(bytes);
        }
        let _ = mime_hint;
        self.load_audio_only(Arc::new(bytes))
    }

    fn load_mp4(&mut self, bytes: Vec<u8>) -> Result<(), (u32, String)> {
        let shared = Arc::new(bytes.clone());
        let mut demux = Mp4Demux::open(bytes).map_err(|e| match e {
            MediaError::Unsupported(msg) => (MEDIA_ERR_SRC_NOT_SUPPORTED, msg),
            MediaError::Decode(msg) => (MEDIA_ERR_DECODE, msg),
            MediaError::Demux(msg) => (MEDIA_ERR_DECODE, msg),
        })?;
        let Some(track) = demux.video_track() else {
            // MP4 without video: audio-only (AAC in MP4).
            let audio = AudioStream::open(shared, true).map_err(error_of)?;
            self.duration = Some(audio.pcm_duration());
            self.kind = MediaKind::AudioOnly;
            self.source = Some(Source::AudioOnly { audio });
            self.ready_state = ReadyState::HaveMetadata;
            return Ok(());
        };
        self.kind = MediaKind::Video;
        self.video_size = Some((track.width as u32, track.height as u32));
        // Sample index (pts, sync) for exact seeking, plus the
        // duration from the last sample.
        let mut index = Vec::with_capacity(track.sample_count as usize);
        let mut duration_ms = 0.0f64;
        for sample_id in 1..=track.sample_count {
            match demux.read_video_sample(track.track_id, sample_id) {
                Ok(Some(sample)) => {
                    duration_ms = sample.pts_ms + sample.duration_ms;
                    index.push((sample.pts_ms, sample.is_sync));
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
        if index.is_empty() {
            return Err((
                MEDIA_ERR_DECODE,
                "MP4 video track has no readable samples".to_string(),
            ));
        }
        self.duration = Some(duration_ms / 1000.0);
        let prefix = demux
            .parameter_sets_annexb(track.track_id)
            .map_err(error_of)?;
        let decoder = H264Decoder::new().map_err(error_of)?;
        // Audio track (best effort): a failure here must not
        // kill the video — it is recorded, not fatal.
        let audio = match AudioStream::open(shared, false) {
            Ok(audio) => Some(audio),
            Err(e) => {
                self.audio_error = Some(e.to_string());
                None
            }
        };
        self.source = Some(Source::Mp4Video {
            demux,
            track,
            index,
            next_sample: 1,
            prefix,
            decoder,
            audio,
            eof: false,
        });
        self.ready_state = ReadyState::HaveMetadata;
        Ok(())
    }

    fn load_webm(&mut self, bytes: Vec<u8>) -> Result<(), (u32, String)> {
        let demux = WebmDemux::open(bytes).map_err(error_of)?;
        let tracks = demux.tracks();
        if let Some(video) = tracks.iter().find(|t| t.kind == WebmTrackKind::Video) {
            // The VP9 decision (see `canplay`): demuxed but not
            // decodable — an honest unsupported error, and the
            // bindings fall through to the next <source>.
            if !webm_codec_supported(&video.codec_id) {
                let e = unsupported_codec_error(&video.codec_id);
                return Err((MEDIA_ERR_SRC_NOT_SUPPORTED, e.to_string()));
            }
            return Err((
                MEDIA_ERR_SRC_NOT_SUPPORTED,
                format!("WebM video codec {} is not decodable here", video.codec_id),
            ));
        }
        if let Some(audio) = tracks.iter().find(|t| t.kind == WebmTrackKind::Audio) {
            if !webm_codec_supported(&audio.codec_id) {
                let e = unsupported_codec_error(&audio.codec_id);
                return Err((MEDIA_ERR_SRC_NOT_SUPPORTED, e.to_string()));
            }
            // A Matroska-wrapped decodable audio track (Vorbis):
            // symphonia has no Matroska reader, so this is
            // unsupported until a WebM audio path exists.
            return Err((
                MEDIA_ERR_SRC_NOT_SUPPORTED,
                format!(
                    "WebM audio ({}) is not decodable here: no Matroska audio reader",
                    audio.codec_id
                ),
            ));
        }
        Err((
            MEDIA_ERR_SRC_NOT_SUPPORTED,
            "WebM file has no audio or video track".to_string(),
        ))
    }

    fn load_audio_only(&mut self, bytes: Arc<Vec<u8>>) -> Result<(), (u32, String)> {
        let audio = AudioStream::open(bytes, true).map_err(error_of)?;
        self.duration = Some(audio.pcm_duration());
        self.kind = MediaKind::AudioOnly;
        self.source = Some(Source::AudioOnly { audio });
        self.ready_state = ReadyState::HaveMetadata;
        Ok(())
    }

    /// Add a `<track>`'s WebVTT bytes (parsed here, displayed
    /// later — cue display is deferred, see `webvtt`).
    pub fn add_track_vtt(
        &mut self,
        kind: crate::webvtt::TrackKind,
        label: String,
        language: String,
        vtt: &str,
    ) -> Result<(), String> {
        let cues = crate::webvtt::parse_webvtt(vtt)?;
        let mut track = TextTrack::new(kind, label, language);
        track.cues = cues;
        self.tracks.push(track);
        Ok(())
    }

    // --- transport controls ---

    /// Start playback. Returns `false` when there is nothing
    /// to play (no source / error): the bindings reject the
    /// `play()` promise in that case.
    pub fn play(&mut self) -> bool {
        if self.source.is_none() || self.error.is_some() {
            return false;
        }
        if self.ended {
            self.seek_inner(0.0);
            self.ended = false;
        }
        self.ensure_sink();
        let now = self.paused_time;
        if let Some(sink) = self.sink.as_mut() {
            sink.set_volume(self.volume);
            sink.set_muted(self.muted);
            self.clock.restart_at(now, Some(sink.as_mut()));
            sink.play();
        } else {
            self.clock.restart_at(now, None);
        }
        self.paused = false;
        self.playing_fired = false;
        self.events.push_back(MediaEvent::Play);
        true
    }

    pub fn pause(&mut self) {
        if self.paused {
            return;
        }
        let now = self.current_time();
        self.clock.pause_at(now);
        self.paused_time = now;
        self.paused = true;
        if let Some(sink) = self.sink.as_mut() {
            sink.pause();
        }
        self.events.push_back(MediaEvent::Pause);
    }

    /// Seek to `t` seconds (clamped to the duration when
    /// known). In-memory over the buffered bytes — no HTTP
    /// Range is involved.
    pub fn seek(&mut self, t: f64) {
        if self.source.is_none() {
            return;
        }
        let t = if t.is_nan() { 0.0 } else { t.max(0.0) };
        let t = match self.duration {
            Some(d) => t.min(d),
            None => t,
        };
        self.seeking = true;
        self.events.push_back(MediaEvent::Seeking);
        self.seek_inner(t);
        self.seeking = false;
        self.events.push_back(MediaEvent::Seeked);
        self.events.push_back(MediaEvent::TimeUpdate);
        self.last_timeupdate = t;
    }

    fn seek_inner(&mut self, t: f64) {
        let was_playing = !self.paused;
        self.paused_time = t;
        self.decoded.clear();
        self.current_frame = None;
        self.primed = false;
        match self.source.as_mut() {
            Some(Source::Mp4Video {
                index,
                next_sample,
                decoder,
                audio,
                eof,
                ..
            }) => {
                // Nearest sync sample at or before `t`.
                let target_ms = t * 1000.0;
                let mut sample_id = 1u32;
                for (i, (pts, is_sync)) in index.iter().enumerate() {
                    if *pts > target_ms {
                        break;
                    }
                    if *is_sync {
                        sample_id = (i + 1) as u32;
                    }
                }
                *next_sample = sample_id;
                *eof = false;
                // A fresh decoder: the old one holds frames from
                // before the seek and has no reset.
                if let Ok(fresh) = H264Decoder::new() {
                    *decoder = fresh;
                }
                if let Some(audio) = audio {
                    audio.seek(t);
                }
            }
            Some(Source::AudioOnly { audio }) => {
                audio.seek(t);
                // `audio.seek` restarts PCM at `t` (pushed = 0);
                // nothing further to reposition here.
            }
            None => {}
        }
        // Re-prime the decode-ahead at the new position so the
        // first frame after the seek is ready without playing.
        if !was_playing {
            self.decode_ahead(t);
        }
        if was_playing {
            if let Some(sink) = self.sink.as_mut() {
                self.clock.restart_at(t, Some(sink.as_mut()));
            } else {
                self.clock.restart_at(t, None);
            }
        } else {
            self.clock.pause_at(t);
        }
    }

    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(sink) = self.sink.as_mut() {
            sink.set_volume(self.volume);
        }
    }

    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if let Some(sink) = self.sink.as_mut() {
            sink.set_muted(muted);
        }
    }

    pub fn set_loop(&mut self, loop_: bool) {
        self.loop_ = loop_;
    }

    // --- the tick driver ---

    /// Advance playback to the wall-clock now: decode ahead
    /// of the media clock, present the due frame, push audio
    /// to the sink, and emit due events. Runs on the tab tick;
    /// the clock — never a tick count — decides what is due.
    pub fn advance(&mut self) -> AdvanceOutput {
        let mut out = AdvanceOutput::default();
        if self.paused || self.source.is_none() {
            return out;
        }
        let now = self.clock.now(self.sink.as_deref());
        let now_ms = now * 1000.0;
        // Decode ahead of the clock (video frames + audio PCM).
        self.decode_ahead(now);
        // Present the due frame from the front of the decoded
        // queue: drop frames whose window has passed (counted),
        // present the first frame whose window contains `now`,
        // leave a too-early frame queued. At most one frame is
        // presented per tick — the renderer sends it over IPC.
        while let Some(frame) = self.decoded.front() {
            let duration_ms = frame.duration_ms.max(1.0);
            match frame_presentation(frame.pts_ms, duration_ms, now_ms, duration_ms) {
                FrameDue::Present => {
                    let frame = self.decoded.pop_front().expect("front frame");
                    let changed = self
                        .current_frame
                        .as_ref()
                        .map(|f| f.frame_id != frame.frame_id)
                        .unwrap_or(true);
                    self.current_frame = Some(frame.clone());
                    self.frames_presented += 1;
                    if changed {
                        out.presented_frame = Some(frame);
                    }
                    if !self.playing_fired {
                        self.playing_fired = true;
                        self.events.push_back(MediaEvent::Playing);
                    }
                    break;
                }
                FrameDue::TooLate => {
                    self.decoded.pop_front();
                    self.frames_dropped += 1;
                }
                FrameDue::TooEarly => break,
            }
        }
        // Audio: push newly decoded PCM to the sink.
        self.push_audio(now);
        if !self.playing_fired && self.kind == MediaKind::AudioOnly && self.primed {
            self.playing_fired = true;
            self.events.push_back(MediaEvent::Playing);
        }
        // readyState progression once primed (only reachable
        // when load did not prime — e.g. a decode that only
        // becomes ready while playing).
        if self.primed && self.ready_state == ReadyState::HaveMetadata {
            self.ready_state = ReadyState::HaveCurrentData;
            self.events.push_back(MediaEvent::LoadedData);
            self.events.push_back(MediaEvent::CanPlay);
        }
        if self.primed && self.ready_state == ReadyState::HaveCurrentData {
            self.ready_state = ReadyState::HaveFutureData;
        }
        if self.primed && self.ready_state == ReadyState::HaveFutureData {
            // Buffered whole + decode keeping ahead: enough.
            self.ready_state = ReadyState::HaveEnoughData;
            self.events.push_back(MediaEvent::CanPlayThrough);
        }
        // timeupdate, at most every 250 ms of media time.
        if now - self.last_timeupdate >= TIMEUPDATE_INTERVAL_SECS {
            self.last_timeupdate = now;
            self.events.push_back(MediaEvent::TimeUpdate);
        }
        // End of stream: clock at/past duration and decoders done.
        if let Some(duration) = self.duration
            && now >= duration
            && self.decode_eof()
        {
            if self.loop_ {
                self.seek_inner(0.0);
                self.last_timeupdate = 0.0;
            } else {
                self.ended = true;
                self.paused = true;
                self.clock.pause_at(duration);
                self.paused_time = duration;
                if let Some(sink) = self.sink.as_mut() {
                    sink.pause();
                }
                self.events.push_back(MediaEvent::Ended);
                out.ended = true;
            }
        }
        out
    }

    /// Decode video samples / audio packets ahead of `now`.
    /// Work per call is bounded two ways: by the decode
    /// horizon (`now + DECODE_AHEAD_SECS`, at most
    /// `MAX_FRAMES_AHEAD` frames) AND by wall time
    /// (`DECODE_BUDGET`): a debug build's C-only openh264
    /// decodes ~25 ms/frame, so an unbounded burst would stall
    /// the tab thread for seconds per tick while the media
    /// clock runs on — every frame would arrive TooLate. The
    /// budget keeps each tick's decode bounded; the catch-up
    /// spreads over ticks, and the clock stays in sync with
    /// what was decoded because late frames are dropped,
    /// never queued.
    fn decode_ahead(&mut self, now: f64) {
        let horizon = now + DECODE_AHEAD_SECS;
        // Starved (nothing to present): spend the tick catching
        // up instead of idling — see CATCH_UP_BUDGET.
        let budget = if self.decoded.is_empty() {
            CATCH_UP_BUDGET
        } else {
            DECODE_BUDGET
        };
        let deadline = std::time::Instant::now() + budget;
        match self.source.as_mut() {
            Some(Source::Mp4Video {
                demux,
                track,
                index,
                next_sample,
                prefix,
                decoder,
                audio,
                eof,
            }) => {
                let mut decoded = 0usize;
                while !*eof && decoded < MAX_FRAMES_AHEAD {
                    // At least two decode attempts per call: the
                    // decoder buffers the first frame (openh264
                    // with NoFlush emits nothing on the first
                    // decode), so a one-attempt budget would prime
                    // nothing and a seek would never present.
                    if decoded >= 2 && std::time::Instant::now() > deadline {
                        break;
                    }
                    let sample_id = *next_sample;
                    let Some((pts_ms, _)) = index.get((sample_id - 1) as usize) else {
                        *eof = true;
                        break;
                    };
                    let pts_ms = *pts_ms;
                    if pts_ms / 1000.0 > horizon {
                        break;
                    }
                    let Ok(Some(sample)) =
                        demux.read_video_sample(track.track_id, sample_id)
                    else {
                        *eof = true;
                        break;
                    };
                    *next_sample += 1;
                    let mut annexb = Vec::new();
                    if sample.is_sync {
                        annexb.extend_from_slice(prefix);
                    }
                    if avcc_to_annexb(&sample.bytes, &mut annexb).is_err() {
                        continue;
                    }
                    match decoder.decode(
                        &annexb,
                        self.element_id,
                        sample.pts_ms,
                        sample.duration_ms,
                        self.origin_clean,
                    ) {
                        Ok(Some(frame)) => {
                            // Bounded queue: when full, the oldest
                            // frame is stale — drop it (counted)
                            // rather than growing the queue.
                            if self.decoded.len() >= MAX_FRAMES_AHEAD {
                                self.decoded.pop_front();
                                self.frames_dropped += 1;
                            }
                            self.decoded.push_back(frame);
                            self.primed = true;
                            decoded += 1;
                        }
                        Ok(None) => {
                            decoded += 1;
                        }
                        Err(_) => {
                            // A corrupt sample skips; the session
                            // does not die on one bad frame.
                            continue;
                        }
                    }
                }
                if *next_sample as usize > index.len() {
                    *eof = true;
                    // Drain the decoder's remainder. The flushed
                    // frames belong to the last samples, so they
                    // carry the last sample's pts.
                    let last_pts = index.last().map(|(pts, _)| *pts).unwrap_or(0.0);
                    if let Ok(frames) = decoder.flush(self.element_id, last_pts, 40.0, self.origin_clean) {
                        for frame in frames {
                            if self.decoded.len() >= MAX_FRAMES_AHEAD {
                                self.decoded.pop_front();
                                self.frames_dropped += 1;
                            }
                            self.decoded.push_back(frame);
                            self.primed = true;
                        }
                    }
                }
                if let Some(audio) = audio {
                    audio.decode_ahead(horizon);
                    if audio.eof && audio.pcm_duration() > 0.0 {
                        self.primed = true;
                    }
                }
            }
            Some(Source::AudioOnly { audio }) => {
                // Normally decoded fully at load; after a seek
                // the stream refills incrementally from the seek
                // point (`eof` is false again until it ends).
                if !audio.eof {
                    audio.decode_ahead(now + DECODE_AHEAD_SECS);
                }
                if !audio.pcm.is_empty() {
                    self.primed = true;
                }
            }
            None => {}
        }
    }

    /// Push decoded PCM ahead of the clock to the sink.
    fn push_audio(&mut self, now: f64) {
        let Some(sink) = self.sink.as_mut() else {
            return;
        };
        let horizon = now + DECODE_AHEAD_SECS;
        let push = |audio: &mut AudioStream, sink: &mut Box<dyn AudioSink>| {
            if audio.sample_rate == 0 || audio.channels == 0 {
                return;
            }
            let horizon_samples =
                ((horizon - audio.pcm_start_time) * audio.sample_rate as f64 * audio.channels as f64)
                    as usize;
            let end = horizon_samples.min(audio.pcm.len());
            if end > audio.pushed {
                sink.write_pcm(&audio.pcm[audio.pushed..end]);
                audio.pushed = end;
            }
        };
        match self.source.as_mut() {
            Some(Source::Mp4Video {
                audio: Some(audio), ..
            }) => push(audio, sink),
            Some(Source::AudioOnly { audio }) => push(audio, sink),
            _ => {}
        }
    }

    fn decode_eof(&self) -> bool {
        match &self.source {
            Some(Source::Mp4Video { eof, audio, .. }) => {
                *eof && audio.as_ref().map(|a| a.eof).unwrap_or(true)
            }
            Some(Source::AudioOnly { audio }) => audio.eof,
            None => true,
        }
    }

    /// Open the audio sink on first play (device when one
    /// exists, null sink otherwise — never an error).
    fn ensure_sink(&mut self) {
        if self.sink.is_some() {
            return;
        }
        let (rate, channels) = match &self.source {
            Some(Source::Mp4Video { audio, .. }) => audio
                .as_ref()
                .map(|a| (a.sample_rate, a.channels))
                .unwrap_or((0, 0)),
            Some(Source::AudioOnly { audio }) => (audio.sample_rate, audio.channels),
            None => (0, 0),
        };
        if rate == 0 || channels == 0 {
            // Video-only: no sink, the Instant clock drives it.
            return;
        }
        self.sink = match CpalSink::try_new(rate, channels) {
            Some(sink) => Some(Box::new(sink)),
            None => Some(Box::new(NullSink::new(rate, channels))),
        };
    }

    /// The whole decoded audio (audio-only sessions, tests):
    /// the PCM behind `currentTime` math.
    pub fn decoded_audio(&self) -> Option<DecodedAudio> {
        match &self.source {
            Some(Source::AudioOnly { audio }) => Some(DecodedAudio {
                codec: "audio".to_string(),
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                samples: audio.pcm.clone(),
            }),
            _ => None,
        }
    }
}

impl Default for MediaSession {
    fn default() -> Self {
        Self::new()
    }
}

fn is_mp4(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

fn is_webm(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0] == 0x1a && bytes[1] == 0x45 && bytes[2] == 0xdf && bytes[3] == 0xa3
}

fn error_of(e: MediaError) -> (u32, String) {
    match e {
        MediaError::Unsupported(msg) => (MEDIA_ERR_SRC_NOT_SUPPORTED, msg),
        MediaError::Decode(msg) => (MEDIA_ERR_DECODE, msg),
        MediaError::Demux(msg) => (MEDIA_ERR_DECODE, msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mp4_bytes() -> Vec<u8> {
        crate::fixture_bytes("bbb-360-10s.mp4")
    }

    fn mp3_bytes() -> Vec<u8> {
        crate::fixture_bytes("sample-3s.mp3")
    }

    fn webm_bytes() -> Vec<u8> {
        crate::fixture_bytes("bbb-360-10s.webm")
    }

    #[test]
    fn mp4_loads_metadata_then_decodes_frames_on_the_clock() {
        let mut s = MediaSession::new();
        s.load(mp4_bytes(), "video/mp4", true);
        assert_eq!(s.network_state(), NetworkState::Idle);
        // Load primes the decode-ahead: metadata + first frame
        // are ready without playing.
        assert_eq!(s.ready_state(), ReadyState::HaveEnoughData);
        assert_eq!(s.video_size(), Some((640, 360)));
        let duration = s.duration().expect("duration known");
        assert!((9.0..11.0).contains(&duration), "duration ~10 s, got {duration}");
        assert!(s.buffered() == vec![(0.0, duration)]);
        assert!(s.origin_clean);

        // Manual clock: each tick advances exactly 16 ms of
        // media time, however long the decode work in
        // `advance()` took in wall time (a debug build's
        // C-only openh264 needs ~25 ms/frame, and under full
        // parallel load the decode thread is descheduled, so a
        // wall clock runs ahead of decode and frames go stale
        // before they are ever presented). The threshold below
        // is unchanged — the clock change makes it
        // deterministic, not easier: a frame still only counts
        // when `frame_presentation` says it is due at the
        // media time the test drove the clock to.
        s.use_manual_clock();
        assert!(s.play());
        let mut presented = 0u32;
        for _ in 0..60 {
            let out = s.advance();
            if out.presented_frame.is_some() {
                presented += 1;
            }
            s.advance_manual_clock(0.016);
        }
        assert!(presented > 5, "frames presented on the clock, got {presented}");
        assert!(s.frames_presented() > 0);
        assert_eq!(s.ready_state(), ReadyState::HaveEnoughData);
        let t = s.current_time();
        assert!(
            (t - 0.96).abs() < 0.05,
            "manual clock advanced 60 x 16 ms, got {t:.2}",
        );
        // Events fired in spec order (metadata first).
        let events = s.drain_events();
        assert!(events.contains(&MediaEvent::LoadedMetadata));
        assert!(events.contains(&MediaEvent::Play));
        assert!(events.contains(&MediaEvent::Playing));
        assert!(events.contains(&MediaEvent::CanPlayThrough));
        assert!(events.contains(&MediaEvent::TimeUpdate));
        s.pause();
        assert!(s.paused());
        let t2 = s.current_time();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = s.advance();
        assert!((s.current_time() - t2).abs() < 0.05, "paused clock frozen");
    }

    #[test]
    fn mp4_seek_repositions_and_plays_from_there() {
        let mut s = MediaSession::new();
        s.load(mp4_bytes(), "video/mp4", true);
        // Manual clock (see the test above): the seek catch-up
        // decodes ~5 frames from the 8.33 s keyframe, and a
        // wall clock would run past them while that decode is
        // descheduled under load. The clock advances exactly
        // 16 ms per tick here, so the post-seek frames are due
        // when the test says they are.
        s.use_manual_clock();
        assert!(s.play());
        for _ in 0..10 {
            let _ = s.advance();
            s.advance_manual_clock(0.016);
        }
        // Seek to 8.5 s: the fixture's second keyframe is at
        // 8.33 s, so catch-up is ~5 frames. (A seek to 5.0 s
        // would decode ~150 frames from the keyframe at 0 s —
        // the same code path, just a longer catch-up; a debug
        // build's C-only decode makes that too slow to assert
        // in a unit test, so it is covered by the mechanism,
        // not by a second slow test.)
        s.seek(8.5);
        assert!((s.current_time() - 8.5).abs() < 0.05, "seek lands at 8.5 s");
        let events = s.drain_events();
        assert!(events.contains(&MediaEvent::Seeking));
        assert!(events.contains(&MediaEvent::Seeked));
        let mut presented = false;
        for _ in 0..60 {
            if s.advance().presented_frame.is_some() {
                presented = true;
            }
            s.advance_manual_clock(0.016);
        }
        assert!(presented, "frames present after seek");
        assert!(s.current_time() > 8.5);
    }

    #[test]
    fn mp3_loads_decodes_and_reports_duration() {
        let mut s = MediaSession::new();
        s.load(mp3_bytes(), "audio/mpeg", true);
        assert_eq!(s.kind(), MediaKind::AudioOnly);
        // Audio-only decodes fully at load, so it is primed.
        assert_eq!(s.ready_state(), ReadyState::HaveEnoughData);
        let duration = s.duration().expect("duration");
        assert!((2.5..3.5).contains(&duration), "duration ~3 s, got {duration}");
        let audio = s.decoded_audio().expect("pcm");
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.channels, 2);
        assert!(!audio.samples.is_empty());
        assert!(s.play());
        for _ in 0..10 {
            let _ = s.advance();
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        assert!(s.current_time() > 0.0);
        assert!(s.drain_events().contains(&MediaEvent::Playing));
    }

    #[test]
    fn webm_vp9_is_an_error_not_a_fake_frame() {
        let mut s = MediaSession::new();
        s.load(webm_bytes(), "video/webm", true);
        assert_eq!(s.network_state(), NetworkState::NoSource);
        assert_eq!(s.ready_state(), ReadyState::HaveNothing);
        let (code, message) = s.error().expect("error recorded");
        assert_eq!(code, MEDIA_ERR_SRC_NOT_SUPPORTED);
        assert!(message.contains("V_VP9"), "error names VP9: {message}");
        let events = s.drain_events();
        assert!(
            events.iter().any(|e| matches!(e, MediaEvent::Error(_))),
            "error event queued"
        );
        assert!(!s.play(), "nothing to play");
        assert_eq!(s.frames_presented(), 0);
    }

    #[test]
    fn garbage_is_an_error_not_silence() {
        let mut s = MediaSession::new();
        s.load(b"this is not media at all".to_vec(), "", true);
        assert_eq!(s.network_state(), NetworkState::NoSource);
        assert!(s.error().is_some());
        assert!(!s.play());
    }

    #[test]
    fn empty_load_is_a_network_error() {
        let mut s = MediaSession::new();
        s.load(Vec::new(), "video/mp4", true);
        assert_eq!(s.error().map(|(c, _)| c), Some(MEDIA_ERR_NETWORK));
    }

    #[test]
    fn volume_and_mute_reach_a_sink() {
        let mut s = MediaSession::new();
        s.load(mp3_bytes(), "audio/mpeg", true);
        s.set_volume(0.25);
        s.set_muted(true);
        assert_eq!(s.volume(), 0.25);
        assert!(s.muted());
        assert!(s.play());
        // The sink exists (null in this sandbox) and took the
        // settings — mute is sink-level, not a flag.
        assert!(s.sink.is_some());
        s.set_muted(false);
        s.pause();
    }

    #[test]
    fn end_of_stream_fires_ended_and_parks_at_duration() {
        let mut s = MediaSession::new();
        s.load(mp3_bytes(), "audio/mpeg", true);
        let duration = s.duration().unwrap();
        assert!(s.play());
        s.seek(duration - 0.1);
        let mut ended = false;
        for _ in 0..40 {
            if s.advance().ended {
                ended = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        assert!(ended, "ended fires at end of stream");
        assert!(s.ended());
        assert!(s.paused());
        assert!((s.current_time() - duration).abs() < 0.1);
        assert!(s.drain_events().contains(&MediaEvent::Ended));
    }
}
