//! Engine-agnostic error type for `pluto-js`.
//!
//! Nothing in this module mentions the underlying JS engine: callers match
//! on [`JsError`] without knowing which engine produced it. That is the
//! point of the trait boundary in [`crate::engine`] — an engine swap must
//! not change error handling at the call sites.

use thiserror::Error;

/// What went wrong running JS, in engine-neutral terms.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum JsError {
    /// The script threw (or a promise it returned rejected). The message is
    /// the JS exception coerced to a string (`"Error: boom"`, a bare
    /// `"boom"`, …). A throwing script never kills the loop: the caller
    /// records this and keeps going.
    #[error("JS exception: {0}")]
    Exception(String),

    /// The wall-clock script budget expired while the VM was ticking
    /// (infinite loop / runaway recursion). Raised by the interrupt
    /// handler; the realm survives and the loop continues.
    #[error("script exceeded its wall-clock budget")]
    Interrupted,

    /// The runtime hit its memory limit.
    #[error("JS runtime out of memory")]
    MemoryLimit,

    /// The wall-clock watchdog fired while a script was hung across an
    /// `await` that never resolved (no VM ticks, so the interrupt handler
    /// cannot fire). The realm is poisoned after this: its suspended
    /// async state can never be unwound safely, so the host must drop the
    /// realm and create a fresh one (see [`crate::TabJs::recreate_realm`]).
    #[error("script hung across an await that never resolved (watchdog)")]
    WatchdogTimeout,

    /// The realm was poisoned by an earlier watchdog kill and must not be
    /// entered again.
    #[error("JS realm is poisoned (earlier watchdog kill); recreate it")]
    Poisoned,

    /// Something failed inside `pluto-js` itself (not in the script):
    /// a broken host function, a failed realm creation, an engine API
    /// misuse. The message is for the log, not for the page.
    #[error("JS host error: {0}")]
    Internal(String),
}

impl JsError {
    /// True for errors that end only the current script/task. The loop
    /// records them and continues with the next task.
    pub fn is_script_error(&self) -> bool {
        matches!(
            self,
            JsError::Exception(_) | JsError::Interrupted | JsError::MemoryLimit
        )
    }
}
