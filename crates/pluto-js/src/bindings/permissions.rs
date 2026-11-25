//! `navigator.permissions` + the deferred-device stubs
//! (Stage 9 Rec 11).
//!
//! The JS surface lives here; the grant store is the
//! [`PermissionsBackend`] trait, implemented by the integration
//! wave in `pluto-browser` (profile-persisted) and by
//! [`MemoryPermissionsBackend`] here (the default: in-memory,
//! used until the browser attaches its backend, and by tests).
//!
//! # Model (report §h, W3C Permissions)
//!
//! * `navigator.permissions.query({name})` → `Promise<
//!   PermissionStatus>` with `.state ∈ {"granted", "prompt",
//!   "denied"}` and an `onchange` event. **Query never prompts
//!   and never requests** — it only reports recorded state.
//! * Unknown permission names throw `TypeError` (not a
//!   rejection) — synchronously, from `query()` itself.
//! * Permissions are per-origin, per feature. **Deny by
//!   default** for every powerful feature. `query()` for a
//!   never-asked origin returns `prompt`.
//! * There is no generic `permissions.request()` in the spec:
//!   each feature owns its request. Pluto has no prompt UI yet,
//!   so when state is `prompt` and no UI exists, the feature
//!   call FAILS — never ship a prompt that grants nothing.
//! * Persistent grants expire after ~90 days of non-visit
//!   (Joel's decision); a visit refreshes the clock.
//! * `persistent-storage` ships with the store (needed by §f):
//!   `navigator.storage.persist()` may grant it, and
//!   `persisted()` reports it.
//!
//! # Deferred device access
//!
//! Geolocation (no location provider), camera/mic (no capture
//! pipeline until M8), notifications (no backend), MIDI, and
//! clipboard-read: `query()` returns `prompt` for these and the
//! feature call fails with `NotAllowedError` (`getUserMedia`,
//! `clipboard.readText`/`writeText` reject; `geolocation`
//! invokes the error callback). No device access is implemented
//! here — the stubs exist only so a page gets the honest failure
//! instead of a silent nothing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use pluto_ipc::Origin;
use rquickjs::function::{Opt, This};
use rquickjs::object::{Accessor, Property};
use rquickjs::{Ctx, FromJs, Function, Object, Value};

use crate::bindings::{
    BindingState, map_get, map_set, registry_map, throw_type_error, to_dom_string,
};

/// Persistent grants expire after this long without a visit to
/// the origin (Joel's decision, Stage 9 Rec 11).
pub const GRANT_TTL: Duration = Duration::from_secs(90 * 24 * 60 * 60);

/// A powerful feature, by its Permissions-registry name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermissionName {
    Geolocation,
    Notifications,
    Camera,
    Microphone,
    PersistentStorage,
    ClipboardRead,
    ClipboardWrite,
    Midi,
}

impl PermissionName {
    /// Parse a `query({name})` descriptor name. `None` means the
    /// name is unknown — the binding throws `TypeError`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "geolocation" => Some(Self::Geolocation),
            "notifications" => Some(Self::Notifications),
            "camera" => Some(Self::Camera),
            "microphone" => Some(Self::Microphone),
            "persistent-storage" => Some(Self::PersistentStorage),
            "clipboard-read" => Some(Self::ClipboardRead),
            "clipboard-write" => Some(Self::ClipboardWrite),
            "midi" => Some(Self::Midi),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Geolocation => "geolocation",
            Self::Notifications => "notifications",
            Self::Camera => "camera",
            Self::Microphone => "microphone",
            Self::PersistentStorage => "persistent-storage",
            Self::ClipboardRead => "clipboard-read",
            Self::ClipboardWrite => "clipboard-write",
            Self::Midi => "midi",
        }
    }

    /// Whether the feature's device access is deferred (report
    /// §h): no provider / pipeline / backend exists yet, so the
    /// feature call always fails even if a grant were somehow
    /// recorded. `clipboard-write` is deferred too — there is no
    /// clipboard backend at all — even though the report names
    /// only `clipboard-read` among the deferred set.
    pub fn is_device_deferred(self) -> bool {
        matches!(
            self,
            Self::Geolocation
                | Self::Notifications
                | Self::Camera
                | Self::Microphone
                | Self::ClipboardRead
                | Self::ClipboardWrite
                | Self::Midi
        )
    }
}

/// The state `query()` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionState {
    Granted,
    Prompt,
    Denied,
}

impl PermissionState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Granted => "granted",
            Self::Prompt => "prompt",
            Self::Denied => "denied",
        }
    }
}

/// The per-origin permission store behind
/// `navigator.permissions`. Implemented by the integration wave
/// with the profile-persisted store (`pluto-browser`); the
/// default [`MemoryPermissionsBackend`] is in-memory only.
///
/// Contract: `query` NEVER prompts and never requests — it only
/// reports recorded state, expiring a `granted` entry whose
/// origin has not been visited within [`GRANT_TTL`] back to
/// `prompt`. Grants for an opaque origin are never recorded and
/// always read as `prompt` (an opaque origin has no stable
/// identity to attach a grant to). Backends are `Send + Sync`.
pub trait PermissionsBackend: Send + Sync {
    /// The recorded state for (`origin`, `name`, `sysex`),
    /// expiring stale grants to `prompt`. `sysex` only matters
    /// for `midi` (the `query({name: "midi", sysex: true})`
    /// descriptor is a separate permission from plain `midi`).
    fn query(&self, origin: &Origin, name: PermissionName, sysex: bool) -> PermissionState;

    /// Record `state` for (`origin`, `name`, `sysex`) — the
    /// host/UI seam (and the test seam). A `granted` record
    /// starts the ~90-day clock now. Returns false when nothing
    /// was recorded (opaque origin).
    fn set_state(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        state: PermissionState,
    ) -> bool;

    /// Grant `persistent-storage` for `origin` (the
    /// `navigator.storage.persist()` path). Returns false when
    /// nothing was recorded (opaque origin).
    fn grant_persistent_storage(&self, origin: &Origin) -> bool;

    /// Record a visit to `origin`: refreshes the ~90-day clock
    /// on all of the origin's grants. The browser calls this on
    /// navigation; the binding never calls it.
    fn note_visit(&self, origin: &Origin);
}

/// One recorded grant: its state, when it was granted (grants
/// only), and when the origin was last visited (the expiry
/// clock).
#[derive(Debug, Clone, Copy)]
struct PermissionEntry {
    state: PermissionState,
    granted_at: Option<SystemTime>,
    last_visit: SystemTime,
}

/// The in-memory [`PermissionsBackend`]: the default until the
/// browser attaches its profile-persisted store, and the backend
/// tests use.
pub struct MemoryPermissionsBackend {
    entries: Mutex<HashMap<(String, PermissionName, bool), PermissionEntry>>,
}

impl MemoryPermissionsBackend {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Test/host seam: move an entry's clocks (e.g. to simulate
    /// a grant whose origin has not been visited for 91 days).
    /// Returns false when no entry exists.
    pub fn set_entry_times(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        granted_at: Option<SystemTime>,
        last_visit: SystemTime,
    ) -> bool {
        let mut entries = self.entries.lock().unwrap();
        let Some(entry) = entries.get_mut(&(origin.serialize(), name, sysex)) else {
            return false;
        };
        entry.granted_at = granted_at;
        entry.last_visit = last_visit;
        true
    }

    fn is_expired(entry: &PermissionEntry, now: SystemTime) -> bool {
        entry.state == PermissionState::Granted
            && now
                .duration_since(entry.last_visit)
                .map(|d| d > GRANT_TTL)
                .unwrap_or(false)
    }
}

impl Default for MemoryPermissionsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionsBackend for MemoryPermissionsBackend {
    fn query(&self, origin: &Origin, name: PermissionName, sysex: bool) -> PermissionState {
        if origin.is_opaque() {
            return PermissionState::Prompt;
        }
        let entries = self.entries.lock().unwrap();
        let Some(entry) = entries.get(&(origin.serialize(), name, sysex)) else {
            return PermissionState::Prompt;
        };
        if Self::is_expired(entry, SystemTime::now()) {
            return PermissionState::Prompt;
        }
        entry.state
    }

    fn set_state(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        state: PermissionState,
    ) -> bool {
        if origin.is_opaque() {
            return false;
        }
        let now = SystemTime::now();
        let mut entries = self.entries.lock().unwrap();
        entries.insert(
            (origin.serialize(), name, sysex),
            PermissionEntry {
                state,
                granted_at: if state == PermissionState::Granted {
                    Some(now)
                } else {
                    None
                },
                last_visit: now,
            },
        );
        true
    }

    fn grant_persistent_storage(&self, origin: &Origin) -> bool {
        self.set_state(
            origin,
            PermissionName::PersistentStorage,
            false,
            PermissionState::Granted,
        )
    }

    fn note_visit(&self, origin: &Origin) {
        if origin.is_opaque() {
            return;
        }
        let now = SystemTime::now();
        let mut entries = self.entries.lock().unwrap();
        for ((serialized, _, _), entry) in entries.iter_mut() {
            if *serialized == origin.serialize() {
                entry.last_visit = now;
            }
        }
    }
}

/// The document origin for permissions, from the attached fetch
/// backend (the realm-side source of the document origin).
/// `None` means no backend is attached yet.
fn document_origin(bindings: &Rc<RefCell<BindingState>>) -> Option<Origin> {
    bindings
        .borrow()
        .fetch_backend
        .as_ref()
        .and_then(|backend| backend.document_origin())
}

/// A `NotAllowedError` DOMException value (deferred device
/// features fail with this — report §h — never with a fake
/// success).
fn not_allowed_error<'js>(ctx: &Ctx<'js>, msg: &str) -> rquickjs::Result<Value<'js>> {
    let value: Value = ctx.eval(format!(
        "(() => {{ const e = new Error({msg:?}); e.name = 'NotAllowedError'; return e; }})()"
    ))?;
    Ok(value)
}

/// Reject a promise with a `NotAllowedError`.
fn reject_not_allowed<'js>(
    ctx: &Ctx<'js>,
    reject: &Function<'js>,
    msg: &str,
) -> rquickjs::Result<()> {
    let reason = not_allowed_error(ctx, msg)?;
    let _: Value = reject.call((reason,))?;
    Ok(())
}

/// Dispatch `onchange` to every live `PermissionStatus` for
/// (`name`, `sysex`) — called after a state change the binding
/// itself made (`navigator.storage.persist()`). The status
/// objects' `state` is a live accessor, so dispatch only has to
/// call the handlers; the value they read is already current.
/// The `permissionStatuses` registry map (id → status object)
/// and [`BindingState::permission_status_features`] (id →
/// (name, sysex)) are the bookkeeping.
fn dispatch_permission_change<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    name: PermissionName,
    sysex: bool,
) -> rquickjs::Result<()> {
    let features: Vec<(u64, (PermissionName, bool))> = bindings
        .borrow()
        .permission_status_features
        .iter()
        .map(|(id, f)| (*id, *f))
        .collect();
    let statuses = registry_map(ctx, "permissionStatuses")?;
    for (id, (feature, want_sysex)) in features {
        if feature != name || want_sysex != sysex {
            continue;
        }
        let entry = map_get(ctx, &statuses, id)?;
        let Some(obj) = entry.as_object() else {
            continue;
        };
        let handler: Value = match obj.get("onchange") {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !handler.is_function() {
            continue;
        }
        let func = Function::from_js(ctx, handler)?;
        let event = Object::new(ctx.clone())?;
        event.set("type", "change")?;
        let _: Value = func.call((This(obj.clone()), Value::from(event)))?;
    }
    Ok(())
}

/// Build one `PermissionStatus` object: a live `state` accessor
/// (it re-reads the backend on every access, so it never goes
/// stale) and a writable `onchange` property (initially `null`).
/// Registered in the `permissionStatuses` registry map under a
/// fresh id so later state changes can dispatch to it.
fn make_permission_status<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    origin: Origin,
    name: PermissionName,
    sysex: bool,
) -> rquickjs::Result<Object<'js>> {
    let obj = Object::new(ctx.clone())?;
    let backend = Arc::clone(&bindings.borrow().permissions_backend);
    let origin_for_state = origin.clone();
    let get_state = move |_ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<String> {
        Ok(backend
            .query(&origin_for_state, name, sysex)
            .as_str()
            .to_string())
    };
    obj.prop("state", Accessor::from(get_state))?;
    obj.prop("onchange", Property::from(rquickjs::Null).writable())?;
    let id = {
        let mut b = bindings.borrow_mut();
        let id = b.next_permission_status_id;
        b.next_permission_status_id += 1;
        b.permission_status_features.insert(id, (name, sysex));
        id
    };
    let statuses = registry_map(ctx, "permissionStatuses")?;
    map_set(ctx, &statuses, id, Value::from(obj.clone()))?;
    Ok(obj)
}

/// Install `navigator` on the global object: `permissions`,
/// `storage` (`persist` / `persisted`), and the deferred-device
/// stubs (`geolocation`, `mediaDevices.getUserMedia`,
/// `clipboard.readText` / `writeText`).
pub fn install_permissions<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    let navigator = Object::new(ctx.clone()).map_err(|e| format!("navigator: {e}"))?;

    install_permissions_object(ctx, bindings, &navigator)?;
    install_storage_object(ctx, bindings, &navigator)?;
    install_geolocation_stub(ctx, bindings, &navigator)?;
    install_media_devices_stub(ctx, bindings, &navigator)?;
    install_clipboard_stub(ctx, bindings, &navigator)?;

    globals
        .set("navigator", navigator)
        .map_err(|e| format!("install navigator: {e}"))?;
    Ok(())
}

fn install_permissions_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    navigator: &Object<'js>,
) -> Result<(), String> {
    let permissions = Object::new(ctx.clone()).map_err(|e| format!("permissions: {e}"))?;
    let b = Rc::clone(bindings);
    let query_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, descriptor: Value<'js>| -> rquickjs::Result<Value<'js>> {
            // Unknown names throw TypeError synchronously — NOT a
            // rejection (report §h).
            let Some(obj) = descriptor.as_object() else {
                return Err(throw_type_error(
                    &ctx,
                    "permissions.query: descriptor must be an object",
                ));
            };
            let name_value: Value = obj.get("name").map_err(|_| {
                throw_type_error(&ctx, "permissions.query: descriptor.name is required")
            })?;
            let name = to_dom_string(&ctx, &name_value)?;
            let Some(feature) = PermissionName::parse(&name) else {
                return Err(throw_type_error(
                    &ctx,
                    &format!("permissions.query: unknown permission name '{name}'"),
                ));
            };
            // `sysex` only matters for `midi`; read it whenever
            // present so `{name: "midi", sysex: true}` is a
            // distinct permission from plain `midi`.
            let sysex = obj
                .get::<_, Value>("sysex")
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let origin = document_origin(&b).unwrap_or_else(Origin::opaque);
            // Query never prompts: resolve at once with the
            // recorded state.
            let (promise, resolve, _reject) = ctx.promise()?;
            let status = make_permission_status(&ctx, &b, origin, feature, sysex)?;
            let _: Value = resolve.call((Value::from(status),))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("permissions.query: {e}"))?;
    permissions
        .set("query", query_fn)
        .map_err(|e| format!("permissions.query: {e}"))?;
    navigator
        .set("permissions", permissions)
        .map_err(|e| format!("navigator.permissions: {e}"))?;
    Ok(())
}

fn install_storage_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    navigator: &Object<'js>,
) -> Result<(), String> {
    let storage = Object::new(ctx.clone()).map_err(|e| format!("navigator.storage: {e}"))?;

    // `navigator.storage.persist()`: grant `persistent-storage`
    // for the document origin and resolve `true`; an opaque
    // origin cannot persist, so it resolves `false`. Dispatches
    // `onchange` to live `persistent-storage` statuses.
    let b = Rc::clone(bindings);
    let persist_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, _reject) = ctx.promise()?;
            let granted = match document_origin(&b) {
                Some(origin) if !origin.is_opaque() => {
                    let backend = Arc::clone(&b.borrow().permissions_backend);
                    let granted = backend.grant_persistent_storage(&origin);
                    if granted {
                        let _ = dispatch_permission_change(
                            &ctx,
                            &b,
                            PermissionName::PersistentStorage,
                            false,
                        );
                    }
                    granted
                }
                _ => false,
            };
            let _: Value = resolve.call((granted,))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("storage.persist: {e}"))?;
    storage
        .set("persist", persist_fn)
        .map_err(|e| format!("storage.persist: {e}"))?;

    // `navigator.storage.persisted()`: resolve whether
    // `persistent-storage` is currently granted.
    let b = Rc::clone(bindings);
    let persisted_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            let (promise, resolve, _reject) = ctx.promise()?;
            let persisted = match document_origin(&b) {
                Some(origin) => {
                    let backend = Arc::clone(&b.borrow().permissions_backend);
                    backend.query(&origin, PermissionName::PersistentStorage, false)
                        == PermissionState::Granted
                }
                None => false,
            };
            let _: Value = resolve.call((persisted,))?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("storage.persisted: {e}"))?;
    storage
        .set("persisted", persisted_fn)
        .map_err(|e| format!("storage.persisted: {e}"))?;

    navigator
        .set("storage", storage)
        .map_err(|e| format!("navigator.storage: {e}"))?;
    Ok(())
}

/// `navigator.geolocation` — deferred (no location provider):
/// `getCurrentPosition` / `watchPosition` invoke the error
/// callback with a `NotAllowedError` (code 1,
/// `PERMISSION_DENIED`) and never call success.
fn install_geolocation_stub<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    navigator: &Object<'js>,
) -> Result<(), String> {
    let geolocation =
        Object::new(ctx.clone()).map_err(|e| format!("navigator.geolocation: {e}"))?;
    let b = Rc::clone(bindings);
    let get_current = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, _success: Opt<Value<'js>>, error: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let _ = &b;
            if let Some(cb) = error.0
                && cb.is_function() {
                    let func = Function::from_js(&ctx, cb)?;
                    let err = Object::new(ctx.clone())?;
                    err.set("code", 1)?;
                    err.set("name", "NotAllowedError")?;
                    err.set(
                        "message",
                        "geolocation is not available in this build (no location provider)",
                    )?;
                    let _: Value = func.call((Value::from(err),))?;
                }
            Ok(())
        },
    )
    .map_err(|e| format!("geolocation.getCurrentPosition: {e}"))?;
    geolocation
        .set("getCurrentPosition", get_current)
        .map_err(|e| format!("geolocation.getCurrentPosition: {e}"))?;
    let watch = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, _success: Opt<Value<'js>>, error: Opt<Value<'js>>| -> rquickjs::Result<f64> {
            if let Some(cb) = error.0
                && cb.is_function() {
                    let func = Function::from_js(&ctx, cb)?;
                    let err = Object::new(ctx.clone())?;
                    err.set("code", 1)?;
                    err.set("name", "NotAllowedError")?;
                    err.set(
                        "message",
                        "geolocation is not available in this build (no location provider)",
                    )?;
                    let _: Value = func.call((Value::from(err),))?;
                }
            // No watch is ever registered: 0 is the (unused) id.
            Ok(0.0)
        },
    )
    .map_err(|e| format!("geolocation.watchPosition: {e}"))?;
    geolocation
        .set("watchPosition", watch)
        .map_err(|e| format!("geolocation.watchPosition: {e}"))?;
    navigator
        .set("geolocation", geolocation)
        .map_err(|e| format!("navigator.geolocation: {e}"))?;
    Ok(())
}

/// `navigator.mediaDevices.getUserMedia` — deferred (no capture
/// pipeline until M8): always rejects with `NotAllowedError`.
fn install_media_devices_stub<'js>(
    ctx: &Ctx<'js>,
    _bindings: &Rc<RefCell<BindingState>>,
    navigator: &Object<'js>,
) -> Result<(), String> {
    let media = Object::new(ctx.clone()).map_err(|e| format!("navigator.mediaDevices: {e}"))?;
    let get_user_media = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, _constraints: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, _resolve, reject) = ctx.promise()?;
            reject_not_allowed(
                &ctx,
                &reject,
                "camera/microphone capture is not available in this build (no capture pipeline until M8)",
            )?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("mediaDevices.getUserMedia: {e}"))?;
    media
        .set("getUserMedia", get_user_media)
        .map_err(|e| format!("mediaDevices.getUserMedia: {e}"))?;
    navigator
        .set("mediaDevices", media)
        .map_err(|e| format!("navigator.mediaDevices: {e}"))?;
    Ok(())
}

/// `navigator.clipboard.readText` / `writeText` — deferred (no
/// clipboard backend): always reject with `NotAllowedError`.
fn install_clipboard_stub<'js>(
    ctx: &Ctx<'js>,
    _bindings: &Rc<RefCell<BindingState>>,
    navigator: &Object<'js>,
) -> Result<(), String> {
    let clipboard = Object::new(ctx.clone()).map_err(|e| format!("navigator.clipboard: {e}"))?;
    let read_text = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<Value<'js>> {
            let (promise, _resolve, reject) = ctx.promise()?;
            reject_not_allowed(
                &ctx,
                &reject,
                "clipboard read is not available in this build (no clipboard backend)",
            )?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("clipboard.readText: {e}"))?;
    clipboard
        .set("readText", read_text)
        .map_err(|e| format!("clipboard.readText: {e}"))?;
    let write_text = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, _text: Opt<Value<'js>>| -> rquickjs::Result<Value<'js>> {
            let (promise, _resolve, reject) = ctx.promise()?;
            reject_not_allowed(
                &ctx,
                &reject,
                "clipboard write is not available in this build (no clipboard backend)",
            )?;
            Ok(promise.into_value())
        },
    )
    .map_err(|e| format!("clipboard.writeText: {e}"))?;
    clipboard
        .set("writeText", write_text)
        .map_err(|e| format!("clipboard.writeText: {e}"))?;
    navigator
        .set("clipboard", clipboard)
        .map_err(|e| format!("navigator.clipboard: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    use crate::bindings::fetch::{
        FetchBackend, FetchCompletion, FetchError, FetchRequest,
    };
    use crate::bindings::test_host::TestHost;
    use crate::engine::JsValue;
    use crate::realm::RealmConfig;
    use crate::tab::TabJs;

    const DOC: &str = "<!doctype html><html><head><title>t</title></head><body></body></html>";

    struct OriginBackend {
        origin: Option<Origin>,
    }

    impl FetchBackend for OriginBackend {
        fn document_origin(&self) -> Option<Origin> {
            self.origin.clone()
        }

        fn start_fetch(&self, request: FetchRequest, done: mpsc::Sender<FetchCompletion>) {
            let _ = done.send(FetchCompletion {
                id: request.id,
                result: Err(FetchError::Network("no network in permission tests".to_string())),
            });
        }
    }

    fn tab_with_origin(origin: Option<Origin>) -> TabJs {
        let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
        tab.set_dom_host(Box::new(TestHost::from_html(DOC)));
        tab.set_fetch_backend(Arc::new(OriginBackend { origin }));
        tab
    }

    fn run(tab: &mut TabJs, source: &str) {
        tab.eval_script(source, "test.js").unwrap();
        tab.run_until_idle().unwrap();
    }

    fn read_str(tab: &mut TabJs, source: &str) -> String {
        match tab.eval_value(source).unwrap() {
            JsValue::String(s) => s,
            other => panic!("expected a string from `{source}`, got {other:?}"),
        }
    }

    fn read_bool(tab: &mut TabJs, source: &str) -> bool {
        match tab.eval_value(source).unwrap() {
            JsValue::Bool(b) => b,
            other => panic!("expected a bool from `{source}`, got {other:?}"),
        }
    }

    fn example_origin() -> Origin {
        Origin::from_url("https://perm.example/page")
    }

    // ------------------------------------------------------------ backend

    #[test]
    fn query_states_granted_prompt_denied() {
        let backend = MemoryPermissionsBackend::new();
        let origin = example_origin();
        // Never asked: prompt.
        assert_eq!(
            backend.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Prompt
        );
        backend.set_state(
            &origin,
            PermissionName::Geolocation,
            false,
            PermissionState::Granted,
        );
        assert_eq!(
            backend.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Granted
        );
        backend.set_state(
            &origin,
            PermissionName::Geolocation,
            false,
            PermissionState::Denied,
        );
        assert_eq!(
            backend.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Denied
        );
        // `midi` with `sysex` is a separate permission from plain
        // `midi`.
        backend.set_state(
            &origin,
            PermissionName::Midi,
            true,
            PermissionState::Granted,
        );
        assert_eq!(
            backend.query(&origin, PermissionName::Midi, true),
            PermissionState::Granted
        );
        assert_eq!(
            backend.query(&origin, PermissionName::Midi, false),
            PermissionState::Prompt
        );
        // Another origin is unaffected.
        assert_eq!(
            backend.query(
                &Origin::from_url("https://other.example/"),
                PermissionName::Geolocation,
                false
            ),
            PermissionState::Prompt
        );
    }

    #[test]
    fn opaque_origin_never_records_a_grant() {
        let backend = MemoryPermissionsBackend::new();
        let opaque = Origin::opaque();
        assert!(!backend.set_state(
            &opaque,
            PermissionName::Geolocation,
            false,
            PermissionState::Granted
        ));
        assert_eq!(
            backend.query(&opaque, PermissionName::Geolocation, false),
            PermissionState::Prompt
        );
        assert!(!backend.grant_persistent_storage(&opaque));
    }

    #[test]
    fn grant_expires_after_ninety_days_without_a_visit() {
        let backend = MemoryPermissionsBackend::new();
        let origin = example_origin();
        backend.set_state(
            &origin,
            PermissionName::Notifications,
            false,
            PermissionState::Granted,
        );
        assert_eq!(
            backend.query(&origin, PermissionName::Notifications, false),
            PermissionState::Granted
        );
        // 89 days without a visit: still granted.
        let now = SystemTime::now();
        let past = now - Duration::from_secs(89 * 24 * 60 * 60);
        assert!(backend.set_entry_times(
            &origin,
            PermissionName::Notifications,
            false,
            Some(past),
            past
        ));
        assert_eq!(
            backend.query(&origin, PermissionName::Notifications, false),
            PermissionState::Granted
        );
        // 91 days: expired back to prompt.
        let older = now - Duration::from_secs(91 * 24 * 60 * 60);
        assert!(backend.set_entry_times(
            &origin,
            PermissionName::Notifications,
            false,
            Some(older),
            older
        ));
        assert_eq!(
            backend.query(&origin, PermissionName::Notifications, false),
            PermissionState::Prompt
        );
        // A visit refreshes the clock: granted again.
        backend.note_visit(&origin);
        assert_eq!(
            backend.query(&origin, PermissionName::Notifications, false),
            PermissionState::Granted
        );
        // Denied does not expire into prompt (only grants do).
        backend.set_state(
            &origin,
            PermissionName::Notifications,
            false,
            PermissionState::Denied,
        );
        assert!(backend.set_entry_times(
            &origin,
            PermissionName::Notifications,
            false,
            None,
            older
        ));
        assert_eq!(
            backend.query(&origin, PermissionName::Notifications, false),
            PermissionState::Denied
        );
    }

    // ------------------------------------------------------------ JS surface

    #[test]
    fn query_never_asked_returns_prompt() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "navigator.permissions.query({name: 'geolocation'}).then(s => { globalThis.__r = s.state; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "prompt");
        run(
            &mut tab,
            "navigator.permissions.query({name: 'midi', sysex: true}).then(s => { globalThis.__r = s.state; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "prompt");
    }

    #[test]
    fn query_unknown_name_throws_type_error_not_rejection() {
        let mut tab = tab_with_origin(Some(example_origin()));
        // The exception's `name` is read in JS (the `JsError`
        // rendering only carries the message — the established
        // pattern from the other binding tests).
        run(
            &mut tab,
            "try { navigator.permissions.query({name: 'teleportation'}); globalThis.__r = 'no-throw'; } \
             catch (e) { globalThis.__r = e.name; }",
        );
        assert_eq!(read_str(&mut tab, "__r"), "TypeError");
        // A missing descriptor throws too.
        run(
            &mut tab,
            "try { navigator.permissions.query('geolocation'); globalThis.__r = 'no-throw'; } \
             catch (e) { globalThis.__r = e.name; }",
        );
        assert_eq!(read_str(&mut tab, "__r"), "TypeError");
    }

    #[test]
    fn persist_grants_persistent_storage_and_fires_onchange() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "globalThis.__changed = 0; \
             navigator.permissions.query({name: 'persistent-storage'}).then(s => { \
               globalThis.__status = s; \
               s.onchange = () => { globalThis.__changed = 1; }; \
               globalThis.__before = s.state; \
             });",
        );
        assert_eq!(read_str(&mut tab, "__before"), "prompt");
        run(
            &mut tab,
            "navigator.storage.persist().then(v => { globalThis.__persisted = v; });",
        );
        assert!(read_bool(&mut tab, "__persisted"));
        // The live `state` accessor already reads granted, and the
        // onchange handler fired.
        assert_eq!(read_str(&mut tab, "__status.state"), "granted");
        assert_eq!(read_str(&mut tab, "String(__changed)"), "1");
        run(
            &mut tab,
            "navigator.storage.persisted().then(v => { globalThis.__r = String(v); });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "true");
    }

    #[test]
    fn persisted_is_false_before_persist() {
        let mut tab = tab_with_origin(Some(Origin::from_url(
            "https://perm-fresh.example/page",
        )));
        run(
            &mut tab,
            "navigator.storage.persisted().then(v => { globalThis.__r = String(v); });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "false");
    }

    #[test]
    fn deferred_geolocation_fails_with_not_allowed_error() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "navigator.geolocation.getCurrentPosition(\
               () => { globalThis.__r = 'success'; }, \
               (e) => { globalThis.__r = e.name + ':' + e.code; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "NotAllowedError:1");
        // ...and query() still only reports prompt for it.
        run(
            &mut tab,
            "navigator.permissions.query({name: 'geolocation'}).then(s => { globalThis.__r = s.state; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "prompt");
    }

    #[test]
    fn deferred_get_user_media_and_clipboard_reject_not_allowed() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "navigator.mediaDevices.getUserMedia({video: true}) \
               .then(() => { globalThis.__r = 'success'; }) \
               .catch(e => { globalThis.__r = e.name; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "NotAllowedError");
        run(
            &mut tab,
            "navigator.clipboard.readText() \
               .then(() => { globalThis.__r = 'success'; }) \
               .catch(e => { globalThis.__r = e.name; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "NotAllowedError");
        run(
            &mut tab,
            "navigator.clipboard.writeText('x') \
               .then(() => { globalThis.__r = 'success'; }) \
               .catch(e => { globalThis.__r = e.name; });",
        );
        assert_eq!(read_str(&mut tab, "__r"), "NotAllowedError");
    }
}
