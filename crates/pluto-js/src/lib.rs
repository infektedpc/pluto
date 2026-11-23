//! Scripting (layer 6): rquickjs runtime + DOM bindings.
//!
//! Runs an `rquickjs::AsyncRuntime` on the tab's event loop, exposing
//! `NodeId`-backed host objects (`document.getElementById`,
//! `addEventListener`, `fetch`) that mutate the DOM and trigger
//! re-style/re-layout. The bindings land in the js round; this stub owns
//! the runtime so the tab can already create (and drop) one.

/// The JS runtime for one tab.
pub struct JsRuntime {
    runtime: rquickjs::AsyncRuntime,
}

impl JsRuntime {
    pub fn new() -> rquickjs::Result<Self> {
        Ok(Self { runtime: rquickjs::AsyncRuntime::new()? })
    }

    /// Evaluate a script in the tab's global context. Bindings land in
    /// the js round.
    pub async fn eval(&self, _script: &str) -> rquickjs::Result<()> {
        Ok(())
    }
}
