//! The permission store, browser side (Stage 9 Rec 11).
//!
//! The JS surface is in `pluto-js` (`bindings::permissions`);
//! this module is the profile-persisted store it talks to
//! through the [`PermissionsBackend`] trait:
//! `origin × feature × state × expiry`, in
//! `permissions.json` under the profile dir (serde-free JSON via
//! the helpers in [`storage`](crate::js_host::storage),
//! written temp-file + rename, the
//! [`security`](crate::security) pattern).
//!
//! Rules (report §h):
//!
//! * Deny by default; `query()` for a never-asked origin
//!   returns `prompt`; query never prompts and never requests.
//! * Persistent grants expire after ~90 days of non-visit
//!   ([`GRANT_TTL`], Joel's decision) — a visit refreshes the
//!   clock. Only `granted` entries expire; `denied` is sticky.
//! * Grants for an opaque origin are never recorded and always
//!   read as `prompt`.
//! * `persistent-storage` ships with the store (needed by §f);
//!   `navigator.storage.persist()` grants it through
//!   [`PermissionsBackend::grant_persistent_storage`].
//! * Device access is deferred — the store records states, but
//!   no prompt UI and no device backend exist yet, so a `prompt`
//!   state makes the feature call fail in the bindings rather
//!   than pretend to prompt.
//!
//! Wiring: [`PlutoPermissionsBackend`] is the per-tab backend
//! over the shared store. The tab thread builds one per tab via
//! [`crate::js_host::new_tab_backends`]; the `TabJs` setter that
//! attaches it is the follow-up hunk documented there.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pluto_ipc::Origin;
use pluto_js::bindings::permissions::{
    PermissionsBackend, PermissionName, PermissionState, GRANT_TTL as JS_GRANT_TTL,
};

use crate::js_host::storage::{
    JsonValue, json_escape, parse_json, write_file_atomic,
};
use crate::security::profile_dir;

/// File name for the persisted permission store inside the
/// profile dir.
pub const PERMISSIONS_FILE: &str = "permissions.json";

/// Persistent grants expire after this long without a visit to
/// the origin (Joel's decision — same value as the `pluto-js`
/// default, kept in sync by the `grant_ttl_matches_js` test).
pub const GRANT_TTL: Duration = JS_GRANT_TTL;

/// Seconds since the Unix epoch, for the profile file.
fn secs_since_epoch(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn from_secs(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

/// One recorded grant.
#[derive(Debug, Clone, Copy)]
struct PermissionEntry {
    state: PermissionState,
    granted_at: Option<SystemTime>,
    last_visit: SystemTime,
}

impl PermissionEntry {
    fn is_expired(&self, now: SystemTime) -> bool {
        self.state == PermissionState::Granted
            && now
                .duration_since(self.last_visit)
                .map(|d| d > GRANT_TTL)
                .unwrap_or(false)
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"state\":{},\"granted_at\":{},\"last_visit\":{}}}",
            json_escape(self.state.as_str()),
            match self.granted_at {
                Some(t) => secs_since_epoch(t).to_string(),
                None => "null".to_string(),
            },
            secs_since_epoch(self.last_visit),
        )
    }

    fn from_json(value: &JsonValue) -> Option<Self> {
        let state = match value.get("state").and_then(|v| v.as_str()) {
            Some("granted") => PermissionState::Granted,
            Some("denied") => PermissionState::Denied,
            Some("prompt") => PermissionState::Prompt,
            _ => return None,
        };
        let granted_at = match value.get("granted_at") {
            Some(JsonValue::Null) | None => None,
            Some(v) => v.as_f64().map(|n| from_secs(n as u64)),
        };
        let last_visit = value
            .get("last_visit")
            .and_then(|v| v.as_f64())
            .map(|n| from_secs(n as u64))?;
        Some(Self {
            state,
            granted_at,
            last_visit,
        })
    }
}

/// The profile-persisted permission store: `origin × feature ×
/// sysex → entry`. (`sysex` only matters for `midi` — the
/// `query({name: "midi", sysex: true})` descriptor is a separate
/// permission from plain `midi`.)
#[derive(Debug, Default)]
pub struct PermissionStore {
    entries: HashMap<(String, PermissionName, bool), PermissionEntry>,
}

impl PermissionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded state for (`origin`, `name`, `sysex`) at
    /// `now`, expiring a stale grant back to `prompt`. A
    /// never-asked origin reads as `prompt`; an opaque origin
    /// always reads as `prompt`.
    pub fn query_at(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        now: SystemTime,
    ) -> PermissionState {
        if origin.is_opaque() {
            return PermissionState::Prompt;
        }
        let Some(entry) = self.entries.get(&(origin.serialize(), name, sysex)) else {
            return PermissionState::Prompt;
        };
        if entry.is_expired(now) {
            return PermissionState::Prompt;
        }
        entry.state
    }

    /// The recorded state right now.
    pub fn query(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
    ) -> PermissionState {
        self.query_at(origin, name, sysex, SystemTime::now())
    }

    /// Record `state` for (`origin`, `name`, `sysex`) at `now`.
    /// A `granted` record starts the ~90-day clock now. Returns
    /// false when nothing was recorded (opaque origin).
    pub fn set_state_at(
        &mut self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        state: PermissionState,
        now: SystemTime,
    ) -> bool {
        if origin.is_opaque() {
            return false;
        }
        self.entries.insert(
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

    /// Record `state` right now.
    pub fn set_state(
        &mut self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        state: PermissionState,
    ) -> bool {
        self.set_state_at(origin, name, sysex, state, SystemTime::now())
    }

    /// Grant `persistent-storage` for `origin` at `now` (the
    /// `navigator.storage.persist()` path).
    pub fn grant_persistent_storage_at(&mut self, origin: &Origin, now: SystemTime) -> bool {
        self.set_state_at(
            origin,
            PermissionName::PersistentStorage,
            false,
            PermissionState::Granted,
            now,
        )
    }

    /// Record a visit to `origin` at `now`: refreshes the
    /// ~90-day clock on all of the origin's grants. The browser
    /// calls this on navigation.
    pub fn note_visit_at(&mut self, origin: &Origin, now: SystemTime) {
        if origin.is_opaque() {
            return;
        }
        let serialized = origin.serialize();
        for ((held, _, _), entry) in self.entries.iter_mut() {
            if *held == serialized {
                entry.last_visit = now;
            }
        }
    }

    /// Record a visit right now.
    pub fn note_visit(&mut self, origin: &Origin) {
        self.note_visit_at(origin, SystemTime::now())
    }

    /// Drop expired grants (housekeeping; `query_at` already
    /// treats them as `prompt` without this).
    pub fn prune_expired(&mut self, now: SystemTime) {
        self.entries.retain(|_, entry| !entry.is_expired(now));
    }

    /// Serialise the whole store (for the profile file).
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();
        for ((origin, name, sysex), entry) in &self.entries {
            parts.push(format!(
                "{{\"origin\":{},\"name\":{},\"sysex\":{},\"entry\":{}}}",
                json_escape(origin),
                json_escape(name.as_str()),
                sysex,
                entry.to_json(),
            ));
        }
        format!("{{\"version\":1,\"entries\":[{}]}}", parts.join(","))
    }

    /// Deserialise a profile file (corrupt input → empty store,
    /// logged).
    pub fn from_json(text: &str) -> Self {
        let parsed = match parse_json(text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[pluto] corrupt permissions profile ({e}); starting empty");
                return Self::new();
            }
        };
        let mut store = Self::new();
        let Some(entries) = parsed.get("entries").and_then(|v| v.as_array()) else {
            eprintln!("[pluto] permissions profile has no entries; starting empty");
            return store;
        };
        for item in entries {
            let (Some(origin), Some(name_s), Some(sysex), Some(entry_v)) = (
                item.get("origin").and_then(|v| v.as_str()),
                item.get("name").and_then(|v| v.as_str()),
                item.get("sysex"),
                item.get("entry"),
            ) else {
                continue;
            };
            let Some(name) = PermissionName::parse(name_s) else {
                continue;
            };
            let sysex = matches!(sysex, JsonValue::Bool(true));
            let Some(entry) = PermissionEntry::from_json(entry_v) else {
                continue;
            };
            store
                .entries
                .insert((origin.to_string(), name, sysex), entry);
        }
        store
    }
}

/// Load the persisted permission store. Missing or corrupt files
/// fall back to an empty store (logged, not fatal).
pub fn load_permissions() -> PermissionStore {
    let Some(dir) = profile_dir() else {
        eprintln!("[pluto] no HOME: permissions persistence disabled");
        return PermissionStore::new();
    };
    match std::fs::read_to_string(dir.join(PERMISSIONS_FILE)) {
        Ok(text) => PermissionStore::from_json(&text),
        Err(_) => {
            eprintln!(
                "[pluto] no persisted permissions yet ({})",
                dir.display()
            );
            PermissionStore::new()
        }
    }
}

/// Save the permission store to the profile. Failures are
/// logged, never fatal.
pub fn save_permissions(store: &PermissionStore) {
    let Some(dir) = profile_dir() else {
        return;
    };
    if let Err(e) = write_file_atomic(&dir, PERMISSIONS_FILE, &store.to_json()) {
        eprintln!("[pluto] could not save permissions: {e}");
    }
}

// ------------------------------------------------------------------ backend

/// The per-tab permissions backend over the shared,
/// profile-persisted store. Implements the `pluto-js`
/// [`PermissionsBackend`] trait, so the JS bindings call it
/// through the same interface as the in-memory default.
pub struct PlutoPermissionsBackend {
    store: Arc<Mutex<PermissionStore>>,
}

impl PlutoPermissionsBackend {
    /// A backend over the shared store (loaded once per tab
    /// creation from the profile).
    pub fn new(store: Arc<Mutex<PermissionStore>>) -> Self {
        Self { store }
    }

    /// The shared store (for saving after a change).
    pub fn store(&self) -> Arc<Mutex<PermissionStore>> {
        Arc::clone(&self.store)
    }

    /// Persist the shared store to the profile now.
    pub fn save(&self) {
        save_permissions(&self.store.lock().unwrap());
    }
}

impl PermissionsBackend for PlutoPermissionsBackend {
    fn query(&self, origin: &Origin, name: PermissionName, sysex: bool) -> PermissionState {
        self.store.lock().unwrap().query(origin, name, sysex)
    }

    fn set_state(
        &self,
        origin: &Origin,
        name: PermissionName,
        sysex: bool,
        state: PermissionState,
    ) -> bool {
        let recorded = self
            .store
            .lock()
            .unwrap()
            .set_state(origin, name, sysex, state);
        if recorded {
            self.save();
        }
        recorded
    }

    fn grant_persistent_storage(&self, origin: &Origin) -> bool {
        let mut store = self.store.lock().unwrap();
        let granted = store.grant_persistent_storage_at(origin, SystemTime::now());
        drop(store);
        if granted {
            self.save();
        }
        granted
    }

    fn note_visit(&self, origin: &Origin) {
        let mut store = self.store.lock().unwrap();
        store.note_visit(origin);
        drop(store);
        self.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pluto-permissions-test-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Guard holding the shared HOME lock for the span between
    /// `with_home` and `restore_home` (see `TEST_HOME_LOCK`).
    struct HomeGuard {
        old: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    fn with_home(dir: &PathBuf) -> HomeGuard {
        // Serialise with every other HOME-mutating test in the
        // crate: `HOME` is process-global, and a parallel test
        // swapping it mid-span made `load_permissions` read the
        // wrong home (flake: Granted read back as Prompt).
        let lock = crate::TEST_HOME_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old = std::env::var_os("HOME");
        // SAFETY: HOME is only read by the persistence helpers,
        // which run on this thread while the lock is held.
        unsafe {
            std::env::set_var("HOME", dir);
        }
        HomeGuard { old, _lock: lock }
    }

    fn restore_home(guard: HomeGuard) {
        unsafe {
            match guard.old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        // `guard._lock` drops here, after HOME is restored.
    }

    fn example_origin() -> Origin {
        Origin::from_url("https://perm.example/page")
    }

    #[test]
    fn grant_ttl_matches_js() {
        // The browser store and the `pluto-js` default must agree
        // on ~90 days, or a grant would expire at different times
        // depending on which backend a tab has attached.
        assert_eq!(GRANT_TTL, JS_GRANT_TTL);
        assert_eq!(GRANT_TTL, Duration::from_secs(90 * 24 * 60 * 60));
    }

    #[test]
    fn query_states_granted_prompt_denied() {
        let mut store = PermissionStore::new();
        let origin = example_origin();
        assert_eq!(
            store.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Prompt,
            "never-asked origin reads as prompt"
        );
        store.set_state(
            &origin,
            PermissionName::Geolocation,
            false,
            PermissionState::Granted,
        );
        assert_eq!(
            store.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Granted
        );
        store.set_state(
            &origin,
            PermissionName::Geolocation,
            false,
            PermissionState::Denied,
        );
        assert_eq!(
            store.query(&origin, PermissionName::Geolocation, false),
            PermissionState::Denied
        );
        // `midi` + `sysex` is a separate permission.
        store.set_state(
            &origin,
            PermissionName::Midi,
            true,
            PermissionState::Granted,
        );
        assert_eq!(
            store.query(&origin, PermissionName::Midi, true),
            PermissionState::Granted
        );
        assert_eq!(
            store.query(&origin, PermissionName::Midi, false),
            PermissionState::Prompt
        );
        // Another origin is unaffected, and every feature starts
        // at prompt (deny by default).
        let other = Origin::from_url("https://other.example/");
        for name in [
            PermissionName::Geolocation,
            PermissionName::Notifications,
            PermissionName::Camera,
            PermissionName::Microphone,
            PermissionName::PersistentStorage,
            PermissionName::ClipboardRead,
            PermissionName::ClipboardWrite,
            PermissionName::Midi,
        ] {
            assert_eq!(store.query(&other, name, false), PermissionState::Prompt);
        }
    }

    #[test]
    fn grant_expires_after_ninety_days_without_a_visit() {
        let mut store = PermissionStore::new();
        let origin = example_origin();
        let granted_at = SystemTime::now();
        store.set_state_at(
            &origin,
            PermissionName::Notifications,
            false,
            PermissionState::Granted,
            granted_at,
        );
        // 89 days later: still granted.
        assert_eq!(
            store.query_at(
                &origin,
                PermissionName::Notifications,
                false,
                granted_at + Duration::from_secs(89 * 24 * 60 * 60)
            ),
            PermissionState::Granted
        );
        // 91 days later: expired back to prompt.
        assert_eq!(
            store.query_at(
                &origin,
                PermissionName::Notifications,
                false,
                granted_at + Duration::from_secs(91 * 24 * 60 * 60)
            ),
            PermissionState::Prompt
        );
        // A visit at day 90 refreshes the clock: granted again
        // at day 91.
        store.note_visit_at(
            &origin,
            granted_at + Duration::from_secs(90 * 24 * 60 * 60),
        );
        assert_eq!(
            store.query_at(
                &origin,
                PermissionName::Notifications,
                false,
                granted_at + Duration::from_secs(91 * 24 * 60 * 60)
            ),
            PermissionState::Granted
        );
        // `prune_expired` drops the stale grant entirely.
        store.note_visit_at(
            &origin,
            granted_at + Duration::from_secs(90 * 24 * 60 * 60),
        );
        store.prune_expired(granted_at + Duration::from_secs(200 * 24 * 60 * 60));
        assert_eq!(
            store.query(&origin, PermissionName::Notifications, false),
            PermissionState::Prompt
        );
        // Denied does not expire into prompt.
        store.set_state_at(
            &origin,
            PermissionName::Notifications,
            false,
            PermissionState::Denied,
            granted_at,
        );
        assert_eq!(
            store.query_at(
                &origin,
                PermissionName::Notifications,
                false,
                granted_at + Duration::from_secs(365 * 24 * 60 * 60)
            ),
            PermissionState::Denied
        );
    }

    #[test]
    fn opaque_origin_never_records_a_grant() {
        let mut store = PermissionStore::new();
        let opaque = Origin::opaque();
        assert!(!store.set_state(
            &opaque,
            PermissionName::Geolocation,
            false,
            PermissionState::Granted
        ));
        assert_eq!(
            store.query(&opaque, PermissionName::Geolocation, false),
            PermissionState::Prompt
        );
        assert!(!store.grant_persistent_storage_at(&opaque, SystemTime::now()));
    }

    #[test]
    fn persistent_storage_grant_and_round_trip() {
        let home = temp_home("round-trip");
        let old = with_home(&home);
        let origin = example_origin();
        {
            let mut store = load_permissions();
            assert!(store.grant_persistent_storage_at(&origin, SystemTime::now()));
            store.set_state(
                &Origin::from_url("https://denied.example/"),
                PermissionName::Camera,
                false,
                PermissionState::Denied,
            );
            save_permissions(&store);
        }
        let loaded = load_permissions();
        assert_eq!(
            loaded.query(&origin, PermissionName::PersistentStorage, false),
            PermissionState::Granted
        );
        assert_eq!(
            loaded.query(
                &Origin::from_url("https://denied.example/"),
                PermissionName::Camera,
                false
            ),
            PermissionState::Denied
        );
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn corrupt_permissions_load_empty() {
        let home = temp_home("corrupt");
        let old = with_home(&home);
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(PERMISSIONS_FILE), "{{{{").unwrap();
        let loaded = load_permissions();
        assert_eq!(
            loaded.query(
                &example_origin(),
                PermissionName::Geolocation,
                false
            ),
            PermissionState::Prompt
        );
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn backend_trait_delegates_to_store() {
        // The `PermissionsBackend` impl the bindings call: query
        // reports, `grant_persistent_storage` grants, and an
        // opaque origin stays prompt.
        let backend = PlutoPermissionsBackend::new(Arc::new(Mutex::new(
            PermissionStore::new(),
        )));
        let origin = example_origin();
        assert_eq!(
            backend.query(&origin, PermissionName::PersistentStorage, false),
            PermissionState::Prompt
        );
        assert!(backend.grant_persistent_storage(&origin));
        assert_eq!(
            backend.query(&origin, PermissionName::PersistentStorage, false),
            PermissionState::Granted
        );
        let opaque = Origin::opaque();
        assert!(!backend.grant_persistent_storage(&opaque));
        assert_eq!(
            backend.query(&opaque, PermissionName::Geolocation, false),
            PermissionState::Prompt
        );
    }
}
