//! User activation + autoplay policy (Stage 16 Phase 1, Rec 7).
//!
//! Built BEFORE any `<video>` element exists, so Phase 2's media
//! elements have a policy to consult from the first frame —
//! otherwise the first build would play sound on load with no
//! gate. This is also what M12's auto-mute hooks into: a video
//! that may not autoplay with sound plays muted instead.
//!
//! # Model (HTML Standard, "user activation")
//!
//! * **Transient activation** — set by a user gesture
//!   (`pointerdown`, `keydown`; see `note_gesture`) and alive for
//!   [`TRANSIENT_ACTIVATION_WINDOW`] (5 s, Chrome's value; the
//!   spec leaves the duration to the UA). It expires on its own;
//!   nothing consumes it.
//! * **Sticky activation** — set by the first gesture and never
//!   cleared for the life of the tab. It survives navigation (the
//!   new document inherits it); transient does NOT survive
//!   navigation — a new document starts with no transient
//!   activation (see `TabJs::navigate`).
//!
//! Only **trusted** gestures count: the renderer records them
//! from the `InputEvent` path (`handle_pointer_down` /
//! `handle_key`), never from JS `dispatchEvent` — a synthetic
//! `keydown` must not unlock autoplay. `navigator.userActivation`
//! exposes the same state to pages (`isActive` /
//! `hasBeenActive`).
//!
//! # Autoplay policy
//!
//! [`AutoplayPolicy::decide`]: muted autoplay is always allowed;
//! autoplay with sound needs transient OR sticky activation;
//! without either, the element plays muted instead
//! ([`AutoplayDecision::MutedOnly`]) — the M12 auto-mute
//! behaviour, available before M12. [`AutoplayDecision::Block`]
//! is reserved for an explicit block-all policy (a user pref;
//! default off) — it means "do not play at all", not "play
//! muted".

use std::time::{Duration, Instant};

/// How long one gesture keeps transient activation alive.
/// Chrome's value; the spec leaves the duration to the UA.
pub const TRANSIENT_ACTIVATION_WINDOW: Duration = Duration::from_secs(5);

/// The tab's user-activation state: one transient window plus a
/// sticky flag. Cheap to clone by sharing (`Rc<RefCell<…>>` in
/// `TabJs` / `BindingState`); the state itself holds no engine
/// values, so it survives realm recreation.
#[derive(Debug, Clone, Default)]
pub struct UserActivation {
    /// When the last gesture happened, if any. Transient
    /// activation is alive while `now < last + WINDOW`.
    last_gesture: Option<Instant>,
    /// True once any gesture has happened in this tab.
    sticky: bool,
}

impl UserActivation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a user gesture (pointerdown / keydown) at `now`:
    /// refreshes the transient window and sets sticky.
    pub fn note_gesture_at(&mut self, now: Instant) {
        self.last_gesture = Some(now);
        self.sticky = true;
    }

    /// Record a user gesture now.
    pub fn note_gesture(&mut self) {
        self.note_gesture_at(Instant::now());
    }

    /// Is transient activation alive at `now`?
    pub fn is_active_at(&self, now: Instant) -> bool {
        self.last_gesture
            .is_some_and(|t| now.duration_since(t) < TRANSIENT_ACTIVATION_WINDOW)
    }

    /// Is transient activation alive right now?
    pub fn is_active(&self) -> bool {
        self.is_active_at(Instant::now())
    }

    /// Has any gesture ever happened in this tab?
    pub fn has_been_active(&self) -> bool {
        self.sticky
    }

    /// Clear transient activation (navigation: a new document
    /// starts with no transient activation). Sticky survives.
    pub fn clear_transient(&mut self) {
        self.last_gesture = None;
    }
}

/// What the autoplay policy says about starting playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoplayDecision {
    /// Play, with sound.
    Allow,
    /// Play, but muted — the element may start only without
    /// sound (the M12 auto-mute shape).
    MutedOnly,
    /// Do not play at all.
    Block,
}

/// The tab's autoplay policy. Default: no autoplay with sound,
/// muted autoplay allowed. `block_all` is a user pref (default
/// off) that turns every autoplay attempt into `Block`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoplayPolicy {
    pub block_all: bool,
}

impl AutoplayPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide whether a media element may start playing.
    /// `muted` is the element's effective muted state (its
    /// `muted` attribute, or forced-mute from M12).
    pub fn decide(&self, muted: bool, activation: &UserActivation) -> AutoplayDecision {
        if self.block_all {
            return AutoplayDecision::Block;
        }
        if muted {
            // Muted autoplay is always allowed (spec: autoplay
            // with `muted` needs no activation).
            return AutoplayDecision::Allow;
        }
        if activation.is_active() || activation.has_been_active() {
            AutoplayDecision::Allow
        } else {
            AutoplayDecision::MutedOnly
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_activation_lives_for_the_window_then_expires() {
        let mut a = UserActivation::new();
        let t0 = Instant::now();
        assert!(!a.is_active_at(t0));
        assert!(!a.has_been_active());
        a.note_gesture_at(t0);
        assert!(a.is_active_at(t0));
        assert!(a.has_been_active());
        assert!(a.is_active_at(t0 + Duration::from_secs(4)));
        // At exactly the window edge it is gone.
        assert!(!a.is_active_at(t0 + TRANSIENT_ACTIVATION_WINDOW));
        assert!(!a.is_active_at(t0 + Duration::from_secs(60)));
        // Sticky outlives the transient window.
        assert!(a.has_been_active());
    }

    #[test]
    fn clear_transient_keeps_sticky() {
        let mut a = UserActivation::new();
        let t0 = Instant::now();
        a.note_gesture_at(t0);
        a.clear_transient();
        assert!(!a.is_active_at(t0));
        assert!(a.has_been_active());
    }

    #[test]
    fn autoplay_decision_allow_muted_block() {
        let policy = AutoplayPolicy::new();
        let fresh = UserActivation::new();
        // No activation: unmuted autoplay becomes muted-only.
        assert_eq!(policy.decide(false, &fresh), AutoplayDecision::MutedOnly);
        // Muted autoplay is allowed with no activation at all.
        assert_eq!(policy.decide(true, &fresh), AutoplayDecision::Allow);
        // A gesture unlocks sound.
        let mut active = UserActivation::new();
        active.note_gesture();
        assert_eq!(policy.decide(false, &active), AutoplayDecision::Allow);
        // Sticky activation alone (transient expired) still
        // unlocks sound — the documented Pluto rule.
        let mut stale = UserActivation::new();
        stale.note_gesture_at(Instant::now() - Duration::from_secs(60));
        assert!(!stale.is_active());
        assert_eq!(policy.decide(false, &stale), AutoplayDecision::Allow);
        // Block-all overrides everything, even muted.
        let strict = AutoplayPolicy { block_all: true };
        assert_eq!(strict.decide(true, &active), AutoplayDecision::Block);
        assert_eq!(strict.decide(false, &active), AutoplayDecision::Block);
    }
}
