//! The latest-only frame channel — the backpressure half of
//! Rec 1.
//!
//! A video frame differs every frame, so nothing can be
//! skipped by hashing (that is the canvas drain's trick, and
//! it does not apply here). At 60 fps the consumer can fall
//! behind; the channel's answer is to *drop*: `send` replaces
//! any undelivered frame and counts it, so the consumer always
//! gets the newest frame and the queue never grows. The drop
//! count is observable, so a stalled consumer is visible in
//! the stats instead of silently lagging.

use pluto_ipc::VideoFrame;

/// Latest-only channel for decoded video frames.
#[derive(Debug, Default)]
pub struct FrameChannel {
    slot: Option<VideoFrame>,
    received: u64,
    dropped: u64,
}

impl FrameChannel {
    /// An empty channel.
    pub fn new() -> Self {
        Self::default()
    }

    /// Offer a frame. If the previous frame was never taken,
    /// it is dropped (counted) and replaced.
    pub fn send(&mut self, frame: VideoFrame) {
        self.received += 1;
        if self.slot.is_some() {
            self.dropped += 1;
        }
        self.slot = Some(frame);
    }

    /// Take the newest frame, if one is waiting.
    pub fn take(&mut self) -> Option<VideoFrame> {
        self.slot.take()
    }

    /// Look at the newest frame without taking it, so the
    /// presentation decision ([`crate::sink::frame_presentation`])
    /// can run before the frame is consumed: a too-early frame
    /// stays queued, a too-late one is taken and dropped.
    pub fn peek(&self) -> Option<&VideoFrame> {
        self.slot.as_ref()
    }

    /// Frames offered so far.
    pub fn received(&self) -> u64 {
        self.received
    }

    /// Frames dropped because the consumer never took them.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// True when a frame is waiting to be taken.
    pub fn has_frame(&self) -> bool {
        self.slot.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64) -> VideoFrame {
        VideoFrame {
            element_id: 1,
            frame_id: id,
            width: 2,
            height: 2,
            pts_ms: id as f64 * 16.0,
            duration_ms: 16.0,
            y: vec![0; 4],
            u: vec![0; 1],
            v: vec![0; 1],
            y_stride: 2,
            uv_stride: 1,
            origin_clean: true,
        }
    }

    #[test]
    fn flooding_keeps_only_the_latest_frame_and_counts_drops() {
        let mut ch = FrameChannel::new();
        for id in 0..10 {
            ch.send(frame(id));
        }
        assert_eq!(ch.received(), 10);
        assert_eq!(ch.dropped(), 9);
        let latest = ch.take().expect("a frame waits");
        assert_eq!(latest.frame_id, 9);
        assert!(ch.take().is_none());
        assert_eq!(ch.dropped(), 9, "taking does not change the count");
    }

    #[test]
    fn steady_consumer_drops_nothing() {
        let mut ch = FrameChannel::new();
        for id in 0..5 {
            ch.send(frame(id));
            assert_eq!(ch.take().unwrap().frame_id, id);
        }
        assert_eq!(ch.dropped(), 0);
    }
}
