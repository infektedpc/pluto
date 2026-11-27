//! The audio sink + the A/V clock (Stage 16 Phase 2, Rec 6).
//!
//! # The clock
//!
//! The audio device clock is the A/V master: video frames are
//! presented when their presentation timestamp is due *against
//! the audio clock*, never by counting 16 ms tab ticks (the
//! tick drifts ~4% against 16.67 ms, and there is no shell
//! vsync signal). [`MediaClock`] holds the fallback half of
//! that rule: when no device exists, it runs on a monotonic
//! `Instant` anchored at play / seek, so video still plays
//! with correct timestamps — the *decision* (which frame is
//! due at media time `now`) is identical in both modes, only
//! the source of `now` differs. [`frame_presentation`] is
//! that decision, pure and unit-tested.
//!
//! The device clock itself is samples consumed by the cpal
//! callback (`consumed_frames / sample_rate`): it advances
//! exactly as the device eats samples, freezes on pause, and
//! resets on seek. It is never wall time.
//!
//! # The sink
//!
//! [`AudioSink`] is the trait; [`NullSink`] is the always-
//! available mock (tests + no-device machines); [`CpalSink`]
//! is the real device. `CpalSink::try_new` returns `None` on
//! ANY failure — no host, no device, no config, stream build
//! or play failure — and the caller falls back to the null
//! sink + `Instant` clock. A missing audio device degrades to
//! video-only playback with correct timestamps; it never fails
//! the suite and never blocks a load. `volume` / `muted` are
//! applied in the device callback (sink-level, not a flag):
//! muted is silence on the wire, which is what M12's
//! auto-mute hooks into.
//!
//! # Resampling
//!
//! The sink requests the media's sample rate from the device
//! when the device offers it; otherwise it takes the default
//! rate and linearly resamples in [`CpalSink::write_pcm`].
//! Linear interpolation is a v1 quality trade (documented,
//! not hidden); a proper resampler is a follow-up.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the audio sink accepts and reports.
pub trait AudioSink {
    /// Queue interleaved 16-bit PCM for the device callback.
    /// The queue is bounded (see [`MAX_QUEUED_FRAMES`]): excess
    /// samples are dropped oldest-first, never grown unbounded.
    fn write_pcm(&mut self, samples: &[i16]);
    /// Output gain, 0.0–1.0 (clamped). Applied in the callback.
    fn set_volume(&mut self, volume: f32);
    /// Sink-level mute: the callback writes silence. Not a flag
    /// the session checks — the wire goes quiet.
    fn set_muted(&mut self, muted: bool);
    /// Device time: frames the callback has consumed, over the
    /// sample rate. Freezes while paused; resets on seek.
    fn position_secs(&self) -> f64;
    /// Clear the queue and zero the consumed counter (seek / restart).
    fn reset_position(&mut self);
    /// Start / stop the device callback.
    fn play(&mut self);
    fn pause(&mut self);
    /// True when a real device is behind this sink.
    fn available(&self) -> bool;
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
}

/// Maximum PCM frames the sink queue holds (~2 s at 48 kHz
/// stereo — the session only pushes ~0.5 s ahead, so this is
/// a safety bound, not a target).
pub const MAX_QUEUED_FRAMES: usize = 96_000;

/// Shared state between the tab thread (writes PCM) and the
/// cpal callback thread (reads it). The callback never blocks
/// on anything but this short mutex.
struct SharedAudioState {
    queue: Mutex<VecDeque<i16>>,
    /// f32 gain bits (AtomicU32: no float atomics on stable).
    volume_bits: AtomicU64,
    muted: AtomicBool,
    /// Frames (per channel) the callback has consumed.
    consumed_frames: AtomicU64,
    /// Frames queued per channel (== queue length / channels).
    queued_frames: AtomicU64,
}

impl SharedAudioState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            volume_bits: AtomicU64::new(1.0f32.to_bits() as u64),
            muted: AtomicBool::new(false),
            consumed_frames: AtomicU64::new(0),
            queued_frames: AtomicU64::new(0),
        })
    }

    fn volume(&self) -> f32 {
        f32::from_bits(self.volume_bits.load(Ordering::Relaxed) as u32)
    }
}

/// The always-available sink: accepts PCM, counts it, outputs
/// nothing. Its `position_secs` is virtual (frames written /
/// rate) — it is NOT a clock and the session never uses it as
/// one; the `Instant` fallback in [`MediaClock`] is the clock
/// when this sink is in use. Tests use it to prove the PCM
/// path (volume/mute/queue) without hardware.
pub struct NullSink {
    state: Arc<SharedAudioState>,
    sample_rate: u32,
    channels: u16,
    written_frames: u64,
}

impl NullSink {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            state: SharedAudioState::new(),
            sample_rate,
            channels,
            written_frames: 0,
        }
    }

    /// Frames accepted so far (test visibility).
    pub fn written_frames(&self) -> u64 {
        self.written_frames
    }
}

impl AudioSink for NullSink {
    fn write_pcm(&mut self, samples: &[i16]) {
        let frames = samples.len() / self.channels.max(1) as usize;
        self.written_frames += frames as u64;
        let mut q = self.state.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.extend(samples.iter().copied());
        let max = MAX_QUEUED_FRAMES * self.channels.max(1) as usize;
        while q.len() > max {
            q.pop_front();
        }
        self.state
            .queued_frames
            .store((q.len() / self.channels.max(1) as usize) as u64, Ordering::Relaxed);
    }

    fn set_volume(&mut self, volume: f32) {
        self.state.volume_bits.store(
            volume.clamp(0.0, 1.0).to_bits() as u64,
            Ordering::Relaxed,
        );
    }

    fn set_muted(&mut self, muted: bool) {
        self.state.muted.store(muted, Ordering::Relaxed);
    }

    fn position_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.written_frames as f64 / self.sample_rate as f64
        }
    }

    fn reset_position(&mut self) {
        self.written_frames = 0;
        self.state.consumed_frames.store(0, Ordering::Relaxed);
        self.state.queued_frames.store(0, Ordering::Relaxed);
        if let Ok(mut q) = self.state.queue.lock() {
            q.clear();
        }
    }

    fn play(&mut self) {}
    fn pause(&mut self) {}
    fn available(&self) -> bool {
        false
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
}

/// The real device sink, via cpal. Created by
/// [`CpalSink::try_new`]; `None` on any failure (no device in
/// this sandbox is the expected case — the caller degrades).
pub struct CpalSink {
    state: Arc<SharedAudioState>,
    /// The stream must be kept alive: dropping it stops audio.
    _stream: cpal::Stream,
    sample_rate: u32,
    channels: u16,
    /// Media rate the PCM arrives at; resampled to the device
    /// rate in `write_pcm` when they differ.
    media_rate: u32,
}

impl CpalSink {
    /// Open the default output device for `media_rate` /
    /// `channels` PCM. Returns `None` — never an error — when
    /// no device can be opened: the caller treats `None` as
    /// "no audio device" and plays video-only.
    pub fn try_new(media_rate: u32, channels: u16) -> Option<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        // Prefer a config at the media's own rate (no resample);
        // fall back to the device default.
        let mut config = device.default_output_config().ok()?;
        if config.sample_rate().0 != media_rate
            && let Ok(mut configs) = device.supported_output_configs() {
                for range in configs.by_ref() {
                    if range.channels() == channels
                        && range.min_sample_rate().0 <= media_rate
                        && media_rate <= range.max_sample_rate().0
                    {
                        config = range.with_sample_rate(cpal::SampleRate(media_rate));
                        break;
                    }
                }
            }
        let device_rate = config.sample_rate().0;
        let state = SharedAudioState::new();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let st = state.clone();
                let ch = channels;
                device.build_output_stream(
                    &config.config(),
                    move |out: &mut [f32], _| write_callback(&st, out, ch),
                    |e| eprintln!("[pluto] audio stream error: {e}"),
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let st = state.clone();
                let ch = channels;
                device.build_output_stream(
                    &config.config(),
                    move |out: &mut [i16], _| write_callback(&st, out, ch),
                    |e| eprintln!("[pluto] audio stream error: {e}"),
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let st = state.clone();
                let ch = channels;
                device.build_output_stream(
                    &config.config(),
                    move |out: &mut [u16], _| write_callback(&st, out, ch),
                    |e| eprintln!("[pluto] audio stream error: {e}"),
                    None,
                )
            }
            _ => return None,
        }
        .ok()?;
        // Start paused: the session calls `play()` when playback
        // actually starts, so opening a sink never makes noise.
        let _ = stream.pause();
        Some(Self {
            state,
            _stream: stream,
            sample_rate: device_rate,
            channels,
            media_rate,
        })
    }

    /// Is the device consuming? (Test/diagnostic visibility.)
    pub fn queued_frames(&self) -> u64 {
        self.state.queued_frames.load(Ordering::Relaxed)
    }
}

/// The cpal data callback, generic over the device's sample
/// format. Pulls from the shared queue, applies volume/mute,
/// counts consumed frames. Underrun (empty queue) writes
/// silence — never blocks, never panics.
fn write_callback<T>(state: &SharedAudioState, out: &mut [T], channels: u16)
where
    T: cpal::Sample + cpal::FromSample<i16>,
{
    let volume = if state.muted.load(Ordering::Relaxed) {
        0.0
    } else {
        state.volume()
    };
    let mut q = state.queue.lock().unwrap_or_else(|e| e.into_inner());
    let frames = out.len() / channels.max(1) as usize;
    for frame in 0..frames {
        for c in 0..channels as usize {
            let idx = frame * channels as usize + c;
            let sample = q.pop_front().unwrap_or(0);
            let scaled = if volume == 0.0 {
                0i16
            } else if volume == 1.0 {
                sample
            } else {
                (sample as f32 * volume) as i16
            };
            out[idx] = T::from_sample(scaled);
        }
    }
    state
        .consumed_frames
        .fetch_add(frames as u64, Ordering::Relaxed);
    state
        .queued_frames
        .store((q.len() / channels.max(1) as usize) as u64, Ordering::Relaxed);
}

/// Linear resampling of interleaved PCM (`from_rate` →
/// `to_rate`). A v1 quality trade — documented in the module
/// docs — used only when the device cannot run at the media
/// rate. Pure, so it is unit-tested.
pub fn resample_linear(
    samples: &[i16],
    channels: u16,
    from_rate: u32,
    to_rate: u32,
) -> Vec<i16> {
    if from_rate == to_rate || from_rate == 0 || to_rate == 0 || channels == 0 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let in_frames = samples.len() / ch;
    if in_frames == 0 {
        return Vec::new();
    }
    let out_frames = (in_frames as u64 * to_rate as u64 / from_rate as u64) as usize;
    let mut out = Vec::with_capacity(out_frames * ch);
    for i in 0..out_frames {
        // Source position in frames (fixed-point via f64 — the
        // counts are small; exactness is not load-bearing here).
        let pos = i as f64 * from_rate as f64 / to_rate as f64;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        let idx = idx.min(in_frames - 1);
        let next = (idx + 1).min(in_frames - 1);
        for c in 0..ch {
            let a = samples[idx * ch + c] as f64;
            let b = samples[next * ch + c] as f64;
            out.push((a + (b - a) * frac).round() as i16);
        }
    }
    out
}

impl AudioSink for CpalSink {
    fn write_pcm(&mut self, samples: &[i16]) {
        let owned;
        let samples = if self.media_rate != self.sample_rate {
            owned = resample_linear(samples, self.channels, self.media_rate, self.sample_rate);
            &owned
        } else {
            samples
        };
        let mut q = self.state.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.extend(samples.iter().copied());
        let max = MAX_QUEUED_FRAMES * self.channels.max(1) as usize;
        while q.len() > max {
            q.pop_front();
        }
        self.state
            .queued_frames
            .store((q.len() / self.channels.max(1) as usize) as u64, Ordering::Relaxed);
    }

    fn set_volume(&mut self, volume: f32) {
        self.state.volume_bits.store(
            volume.clamp(0.0, 1.0).to_bits() as u64,
            Ordering::Relaxed,
        );
    }

    fn set_muted(&mut self, muted: bool) {
        self.state.muted.store(muted, Ordering::Relaxed);
    }

    fn position_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.state.consumed_frames.load(Ordering::Relaxed) as f64 / self.sample_rate as f64
        }
    }

    fn reset_position(&mut self) {
        self.state.consumed_frames.store(0, Ordering::Relaxed);
        self.state.queued_frames.store(0, Ordering::Relaxed);
        if let Ok(mut q) = self.state.queue.lock() {
            q.clear();
        }
    }

    fn play(&mut self) {
        use cpal::traits::StreamTrait;
        let _ = self._stream.play();
    }

    fn pause(&mut self) {
        use cpal::traits::StreamTrait;
        let _ = self._stream.pause();
    }

    fn available(&self) -> bool {
        true
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
}

// ---------------------------------------------------------------- clock

/// The A/V clock: the audio device is master when one exists,
/// a monotonic `Instant` otherwise. The two modes answer the
/// same question — "what media time is it now?" — so video
/// scheduling ([`frame_presentation`]) is identical either way.
///
/// The `Instant` half is anchored, never tick-counted: `now`
/// is `anchor_media + wall_clock_since(anchor_wall)`. A 16 ms
/// tab tick that fires late moves `now` by the real elapsed
/// time, not by 16 ms.
#[derive(Debug)]
pub struct MediaClock {
    anchor_wall: Instant,
    anchor_media: f64,
    /// True while the device clock is master.
    audio_master: bool,
    /// True while paused: `now()` is exactly the anchor, and
    /// wall time does not advance it.
    paused: bool,
    /// Test-only manual time source (`None` in production,
    /// where the field does not exist): when set, `now()`
    /// returns this value and ignores both the wall clock and
    /// the device clock, advancing only when a test drives it
    /// via `advance_manual`. That makes frame-presentation
    /// counts independent of how fast the sandbox decodes
    /// under parallel load — the clock, not decode speed,
    /// decides what is due. `restart_at` / `pause_at` keep it
    /// in sync with the anchor, so play / seek / pause behave
    /// identically in both modes.
    #[cfg(test)]
    manual_now: Option<f64>,
}

impl Default for MediaClock {
    fn default() -> Self {
        Self {
            anchor_wall: Instant::now(),
            anchor_media: 0.0,
            audio_master: false,
            // A fresh clock runs (a stopwatch started at 0);
            // `pause_at` freezes it. The session manages this
            // around play/pause/seek.
            paused: false,
            #[cfg(test)]
            manual_now: None,
        }
    }
}

impl MediaClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Media time now, in seconds. While paused, exactly the
    /// anchor (wall time does not move it). With a live sink,
    /// the device position (which resets on seek/play, so it
    /// counts from the anchor) plus the anchor media time;
    /// otherwise the `Instant` anchor.
    pub fn now(&self, audio: Option<&dyn AudioSink>) -> f64 {
        // Test-only manual source: it moves only when a test
        // drives it, never with wall time or device position.
        #[cfg(test)]
        if let Some(manual) = self.manual_now {
            return manual;
        }
        if self.paused {
            return self.anchor_media;
        }
        if self.audio_master
            && let Some(sink) = audio {
                return self.anchor_media + sink.position_secs();
            }
        self.anchor_media + self.anchor_wall.elapsed().as_secs_f64()
    }

    /// (Re)start the clock at media time `t` (play / seek):
    /// the `Instant` anchor moves to now, and the device clock
    /// becomes master only when a live sink is supplied — the
    /// sink resets, so its position counts from `t` (see
    /// `now()`).
    /// Test-only: switch this clock to the manual source,
    /// anchored at media time `t`. From here `now()` returns
    /// only what `advance_manual` has driven it to.
    #[cfg(test)]
    pub fn set_manual_now(&mut self, t: f64) {
        self.manual_now = Some(t);
    }

    /// Test-only: advance the manual source by `dt` seconds.
    /// A no-op in wall time — the caller decides the pace, so
    /// a tick is exactly `dt` no matter how long the decode
    /// work between ticks took in real time. If the manual
    /// source was never set, it starts from the current media
    /// time (the wall-clock `now()` at that moment).
    #[cfg(test)]
    pub fn advance_manual(&mut self, dt: f64) {
        match self.manual_now {
            Some(manual) => self.manual_now = Some(manual + dt),
            None => {
                let now = self.anchor_media + self.anchor_wall.elapsed().as_secs_f64();
                self.manual_now = Some(now + dt);
            }
        }
    }

    pub fn restart_at(&mut self, t: f64, audio: Option<&mut dyn AudioSink>) {
        self.anchor_wall = Instant::now();
        self.anchor_media = t;
        self.paused = false;
        // Keep a manual source in sync: play / seek re-anchors
        // it at `t`, exactly like the wall anchor.
        #[cfg(test)]
        if self.manual_now.is_some() {
            self.manual_now = Some(t);
        }
        match audio {
            Some(sink) if sink.available() => {
                sink.reset_position();
                self.audio_master = true;
            }
            _ => self.audio_master = false,
        }
    }

    /// Pause: freeze the `Instant` anchor at the current media
    /// time. (The device clock freezes itself — the callback
    /// stops consuming — so nothing is needed there.)
    pub fn pause_at(&mut self, now: f64) {
        self.anchor_wall = Instant::now();
        self.anchor_media = now;
        self.audio_master = false;
        self.paused = true;
        // Keep a manual source frozen at the pause point.
        #[cfg(test)]
        if self.manual_now.is_some() {
            self.manual_now = Some(now);
        }
    }

    /// True while the device clock is master.
    pub fn is_audio_master(&self) -> bool {
        self.audio_master
    }
}

/// What to do with a frame at media time `now`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDue {
    /// Not yet: `now` is before the frame's pts.
    TooEarly,
    /// Present now: `now` is inside `[pts, pts + duration +
    /// slack)`. The slack (one frame) absorbs tick quantisation
    /// — the tab thread only checks on ticks, so a frame whose
    /// pts passed mid-tick is still the right frame to show.
    Present,
    /// Too late: `now` is past the frame's window — drop it and
    /// take the next one. A dropped frame is counted, never
    /// queued: the consumer always gets the newest frame.
    TooLate,
}

/// Decide a frame against the media clock. Pure: the tab tick
/// only decides *when to ask*, never what the answer is.
///
/// `slack_ms` is the late tolerance (one frame duration is the
/// honest value — a frame that ended one frame ago is stale).
pub fn frame_presentation(
    pts_ms: f64,
    duration_ms: f64,
    now_ms: f64,
    slack_ms: f64,
) -> FrameDue {
    if now_ms < pts_ms {
        FrameDue::TooEarly
    } else if now_ms < pts_ms + duration_ms + slack_ms {
        FrameDue::Present
    } else {
        FrameDue::TooLate
    }
}

/// How long the session may wait before polling the media
/// clock again for one element: the tab tick (16 ms) is the
/// poll, never the schedule.
pub const CLOCK_POLL: Duration = Duration::from_millis(16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_accepts_pcm_volume_and_mute_without_hardware() {
        let mut sink = NullSink::new(44_100, 2);
        assert!(!sink.available());
        sink.write_pcm(&[1000, -1000, 2000, -2000]);
        assert_eq!(sink.written_frames(), 2);
        sink.set_volume(0.5);
        sink.set_muted(true);
        sink.write_pcm(&[1000, -1000]);
        assert_eq!(sink.written_frames(), 3);
        sink.reset_position();
        assert_eq!(sink.written_frames(), 0);
        assert_eq!(sink.position_secs(), 0.0);
    }

    #[test]
    fn cpal_sink_is_none_without_a_device_in_this_sandbox() {
        // The sandbox has no audio device: `try_new` must
        // degrade to None, not panic, not hang. (On a machine
        // WITH a device this returns Some — the test only
        // asserts the no-device path degrades gracefully, and
        // accepts Some as well: either way the suite passes.)
        let sink = CpalSink::try_new(44_100, 2);
        match sink {
            None => {}
            Some(mut s) => {
                assert!(s.available());
                s.write_pcm(&[0; 4410]);
                s.play();
                s.pause();
            }
        }
    }

    #[test]
    fn clock_is_anchored_not_tick_counted() {
        let mut clock = MediaClock::new();
        let t0 = clock.now(None);
        std::thread::sleep(Duration::from_millis(30));
        let t1 = clock.now(None);
        // ~30 ms of wall time passed, not a multiple of 16.
        let elapsed_ms = (t1 - t0) * 1000.0;
        assert!(
            (20.0..60.0).contains(&elapsed_ms),
            "clock follows wall time, got {elapsed_ms} ms",
        );
        clock.pause_at(5.0);
        let p0 = clock.now(None);
        std::thread::sleep(Duration::from_millis(20));
        let p1 = clock.now(None);
        // Paused: the anchor froze (within scheduling slop).
        assert!(
            (p1 - p0).abs() < 0.01,
            "paused clock does not advance: {p0} -> {p1}"
        );
        assert!((p0 - 5.0).abs() < 0.05);
    }

    #[test]
    fn frame_presentation_decides_by_pts_not_ticks() {
        // A 40 ms frame at pts 1000.
        assert_eq!(
            frame_presentation(1000.0, 40.0, 999.0, 40.0),
            FrameDue::TooEarly
        );
        assert_eq!(
            frame_presentation(1000.0, 40.0, 1000.0, 40.0),
            FrameDue::Present
        );
        assert_eq!(
            frame_presentation(1000.0, 40.0, 1079.0, 40.0),
            FrameDue::Present,
            "inside pts+duration+slack"
        );
        assert_eq!(
            frame_presentation(1000.0, 40.0, 1080.0, 40.0),
            FrameDue::TooLate
        );
        // A 16 ms tick grid never enters the decision: a frame
        // due at 1000.5 ms is Present at 1000.5, not at 1016.
        assert_eq!(
            frame_presentation(1000.5, 40.0, 1000.5, 40.0),
            FrameDue::Present
        );
    }

    #[test]
    fn resample_linear_up_and_down() {
        // 2ch, 4 frames at 1000 Hz -> 2000 Hz doubles frames.
        let pcm: Vec<i16> = (0..8).map(|i| i * 100).collect();
        let up = resample_linear(&pcm, 2, 1000, 2000);
        assert_eq!(up.len(), 16);
        assert_eq!(up[0], 0);
        assert_eq!(up[2], 100, "half-way interpolates");
        let down = resample_linear(&pcm, 2, 2000, 1000);
        assert_eq!(down.len(), 4);
        assert_eq!(down, vec![0, 100, 400, 500]);
    }
}
