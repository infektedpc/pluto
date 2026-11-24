//! The engine trait boundary (Stage 7, Recommendation 5's constraint).
//!
//! Every QuickJS type (`Ctx`, `Value`, `Function`, `Module`, …) lives
//! behind this trait. The public API of `pluto-js` — [`JsValue`],
//! [`JsError`], [`JsEngine`] — mentions no engine type at all, so the
//! M7 engine swap (V8 or SpiderMonkey) changes the implementation in
//! [`crate::realm`] and nothing else. Wave 2 builds bindings against
//! this surface, not against rquickjs.
//!
//! The concrete implementation is [`crate::realm::TabRealm`], which owns
//! one rquickjs `AsyncRuntime` + `AsyncContext` (one realm per tab).

use crate::error::JsError;

/// An engine-neutral JS value: the only thing that may cross the trait
/// boundary as a result. Structured values (objects, arrays, functions)
/// cross as an opaque marker in this wave; wave 2 moves structured data
/// across via JSON or via the `DomHost` trait, never by leaking engine
/// values.
#[derive(Debug, Clone, PartialEq)]
pub enum JsValue {
    /// `undefined`.
    Undefined,
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// Any JS number (all numbers are f64 on this side).
    Number(f64),
    /// A JS string.
    String(String),
    /// An object (including arrays, dates, …). No data crosses: the
    /// caller asked for a primitive and got an object instead.
    Object,
    /// A function value.
    Function,
}

impl JsValue {
    /// The value as a bool, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The value as a number, if it is one.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JsValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The value as a string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// True for `undefined`.
    pub fn is_undefined(&self) -> bool {
        matches!(self, JsValue::Undefined)
    }
}

/// How a script is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalKind {
    /// Classic script: sloppy by default, top-level `await` is a syntax
    /// error, `import`/`export` are syntax errors.
    Classic,
    /// Classic script with top-level `await` allowed (returns a promise).
    /// This is `Ctx::eval_promise` under the hood.
    ClassicAwait,
    /// ES module: strict, deferred, `import`/`export` allowed, top-level
    /// `await` allowed. Imports resolve through the module loader the
    /// host installs (wave 2); in this wave only the entry module's own
    /// source is evaluated.
    Module,
}

/// The minimal engine surface `pluto-js` needs. One implementor per
/// engine; today that is [`crate::realm::TabRealm`] (QuickJS-NG via
/// rquickjs 0.14). The trait is deliberately small: evaluation, promise
/// draining, bytecode, and memory reporting. Everything else — the event
/// loop, timers, rAF, the DOM — is host code that calls this trait.
pub trait JsEngine {
    /// Evaluate a classic script (or a top-level-await / module script
    /// per `kind`) and return its completion value. Promise results are
    /// awaited: this drains the engine's job queue until the promise
    /// settles (or the watchdog fires), so the caller never has to
    /// `.await` an engine promise ad-hoc on its own runtime.
    fn eval(&mut self, kind: EvalKind, source: &str, filename: &str) -> Result<JsValue, JsError>;

    /// Evaluate a module by URL and return its completion value
    /// (`undefined` on success). The module is compiled fresh; for the
    /// cached path use [`JsEngine::eval_module_bytecode`] with bytes from
    /// [`JsEngine::compile_module`].
    fn eval_module(&mut self, url: &str, source: &str) -> Result<JsValue, JsError>;

    /// Compile a module to bytecode bytes (QuickJS `JS_WriteObject`
    /// under the hood). The bytes are only valid for this engine version
    /// in this process — they are never written to disk.
    fn compile_module(&mut self, url: &str, source: &str) -> Result<Vec<u8>, JsError>;

    /// Evaluate a module from bytecode previously produced by
    /// [`JsEngine::compile_module`]. The caller must only pass bytes
    /// this engine produced; foreign bytes are rejected by the engine.
    fn eval_module_bytecode(&mut self, url: &str, bytecode: &[u8]) -> Result<JsValue, JsError>;

    /// Drain the engine's pending promise jobs (its internal job queue)
    /// until empty. Returns the number of jobs run. A job that throws
    /// stops the drain with [`JsError::Exception`] — the exception is
    /// also reported through the runtime's rejection tracker, so the
    /// caller does not lose it.
    ///
    /// The event loop does NOT use this for its microtask checkpoint: the
    /// checkpoint drains one job at a time so it can interleave host
    /// microtasks (see [`crate::TabJs`]). This method is the probe's and
    /// the tests' bulk drain.
    fn drain_jobs(&mut self) -> Result<usize, JsError>;

    /// Current heap use of this runtime in bytes (0 if the engine
    /// cannot report it).
    fn memory_usage_bytes(&self) -> u64;
}
