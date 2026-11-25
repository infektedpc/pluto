//! `localStorage` / `sessionStorage` (Stage 9 Rec 4).
//!
//! The JS surface lives here; the partitioned store is the
//! [`StorageBackend`] trait, implemented by the integration wave in
//! `pluto-browser` (disk-backed `localStorage`, per-tab
//! `sessionStorage`) and by [`MemoryStorageBackend`] here (the
//! default: an in-memory store used until the browser attaches its
//! backends, and by tests).
//!
//! # Partitioning (report §f)
//!
//! Both storages are keyed by ONE shared key shape,
//! [`StorageKey`] = `(top-level site, origin)`, where the site is
//! `scheme + host` of the top-level origin. There is NO public
//! suffix list in the tree — do NOT hand-roll a PSL — so the site
//! is the document origin's host as the approximation for now
//! (the same approximation the Wave 1 code uses for
//! `top_level_site`); a PSL is the future refinement that turns
//! the host into a registrable domain. `https://widget.example`
//! embedded on `shop.example` gets a different bucket than the
//! same widget on `news.example`.
//!
//! # Quotas (report §f, WHATWG Storage §4.1)
//!
//! 5 MiB (5 × 2²⁰ bytes) per storage, per key. The size of a
//! bucket is the sum of its keys' and values' UTF-8 byte lengths.
//! A `setItem` that would push a bucket over quota throws
//! `QuotaExceededError`; an update that shrinks a value frees the
//! difference.
//!
//! # Scoping
//!
//! `localStorage` is shared across all tabs of that origin and
//! persists (in the browser's backend); `sessionStorage` is per
//! top-level browsing context (per tab), survives reloads and
//! same-tab navigation, is cleared when the tab closes, and stays
//! memory-only. Both throw `SecurityError` in an opaque origin
//! (e.g. a sandboxed iframe without `allow-same-origin`, or a
//! `data:` document) — the check runs against the document origin
//! the fetch backend tracks, before any backend call.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use pluto_ipc::Origin;
use rquickjs::function::{This};
use rquickjs::object::Accessor;
use rquickjs::{Coerced, Ctx, Function, Object, Value};

use crate::bindings::{
    BindingState, js, js_null, throw_dom, to_dom_string,
};

/// Bytes of storage per [`StorageKey`], per storage kind
/// (WHATWG Storage §4.1: `localStorage` and `sessionStorage` are
/// each 5 MiB).
pub const STORAGE_QUOTA_BYTES: usize = 5 * 1024 * 1024;

/// The one shared partition key for storage (and, per report §f,
/// for cookies, cache, and the blocker): `(top-level site,
/// origin)`.
///
/// The site is `scheme + host` of the top-level origin — the
/// host is the site approximation until a public-suffix list
/// exists (see [`top_level_site_of`]). The port is ignored: a
/// site ignores port by definition, so
/// `https://example.com:8443` and `https://example.com` share a
/// site (but are different origins, hence different buckets when
/// the frame origin differs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StorageKey {
    /// `scheme://host` of the top-level origin, or `"null"` for
    /// an opaque top-level origin.
    pub top_level_site: String,
    /// The frame/document origin.
    pub origin: Origin,
}

impl StorageKey {
    pub fn new(origin: &Origin, top_level_site: &str) -> Self {
        Self {
            top_level_site: top_level_site.to_string(),
            origin: origin.clone(),
        }
    }

    /// The key for a top-level document: the site is the
    /// document origin's own site.
    pub fn for_document(origin: &Origin) -> Self {
        Self {
            top_level_site: top_level_site_of(origin),
            origin: origin.clone(),
        }
    }
}

/// The top-level site of an origin: `scheme://host` for a tuple
/// origin, `"null"` for an opaque origin.
///
/// NO public-suffix list exists in the tree, so the host stands
/// in for the registrable domain: `https://sub.example.com` and
/// `https://example.com` are different sites here, where a PSL
/// would call them the same site. That is the documented
/// approximation (report §f), not a bug to fix by hand-rolling a
/// suffix list.
pub fn top_level_site_of(origin: &Origin) -> String {
    match origin {
        Origin::Tuple { scheme, host, .. } => {
            let host = if host.contains(':') {
                format!("[{host}]")
            } else {
                host.clone()
            };
            format!("{scheme}://{host}")
        }
        Origin::Opaque(_) => "null".to_string(),
    }
}

/// Why a storage operation failed, so the binding can raise the
/// right DOMException name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// The write would push the bucket over
    /// [`STORAGE_QUOTA_BYTES`].
    QuotaExceeded,
    /// The document origin is opaque: storage is unavailable.
    SecurityError,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::QuotaExceeded => {
                write!(f, "storage quota exceeded (5 MiB per origin)")
            }
            StorageError::SecurityError => {
                write!(f, "storage is unavailable for an opaque origin")
            }
        }
    }
}

/// Which web storage area an operation addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageKind {
    Local,
    Session,
}

/// The partitioned store behind `localStorage` /
/// `sessionStorage`. Implemented by the integration wave with the
/// profile-backed store (`pluto-browser`); the default
/// [`MemoryStorageBackend`] is in-memory only.
///
/// Contract: every method is keyed by the full [`StorageKey`]
/// (the backend never derives a site itself — the binding passes
/// the document origin and its top-level site), and `set_item`
/// enforces the 5 MiB quota per key, returning
/// [`StorageError::QuotaExceeded`] instead of partially writing.
/// Backends are `Send + Sync`: the browser's `localStorage`
/// store is shared across tab threads.
pub trait StorageBackend: Send + Sync {
    fn get_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
    ) -> Result<Option<String>, StorageError>;

    fn set_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
        value: &str,
    ) -> Result<(), StorageError>;

    fn remove_item(&self, kind: StorageKind, key: &StorageKey, name: &str);

    fn clear(&self, kind: StorageKind, key: &StorageKey);

    /// The name of the `index`-th key in insertion order, or
    /// `None` when `index` is out of range.
    fn key(&self, kind: StorageKind, key: &StorageKey, index: usize) -> Option<String>;

    fn len(&self, kind: StorageKind, key: &StorageKey) -> usize;

    /// Cross-tab `storage` events this backend has not delivered
    /// yet (Stage 13 Rec 10d): mutations from *other* tabs, in
    /// order. Defaults to none (the in-memory backend is
    /// process-shared with no fan-out — every tab already sees
    /// the same buckets, so there is nothing to deliver).
    fn drain_storage_events(&self) -> Vec<StorageEventRecord> {
        Vec::new()
    }

    /// `(used, quota)` bytes for `navigator.storage.estimate()`
    /// (Stage 13 Rec 10d). Defaults to the in-memory accounting.
    fn usage_bytes(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// One cross-tab `storage` event (Stage 13 Rec 10d): the mutated
/// key + old/new values for one origin. `key: None` means the
/// whole bucket was cleared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageEventRecord {
    /// The origin whose bucket changed (serialised URL origin).
    pub origin: String,
    /// The mutated key (`None` = whole-bucket `clear()`).
    pub key: Option<String>,
    /// The previous value (`None` = key absent).
    pub old_value: Option<String>,
    /// The new value (`None` = key removed / bucket cleared).
    pub new_value: Option<String>,
}

/// One storage bucket: entries in insertion order (the `key(n)`
/// order), with the quota accounted in UTF-8 bytes.
#[derive(Debug, Default)]
struct StorageBucket {
    entries: Vec<(String, String)>,
}

impl StorageBucket {
    fn byte_len(&self) -> usize {
        self.entries
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .sum()
    }

    fn get(&self, name: &str) -> Option<String> {
        self.entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    }

    fn set(&mut self, name: &str, value: &str) -> Result<(), StorageError> {
        let new_size = match self.entries.iter().find(|(k, _)| k == name) {
            Some((_, old)) => self.byte_len() - old.len() + value.len(),
            None => self.byte_len() + name.len() + value.len(),
        };
        if new_size > STORAGE_QUOTA_BYTES {
            return Err(StorageError::QuotaExceeded);
        }
        match self.entries.iter_mut().find(|(k, _)| k == name) {
            Some((_, v)) => *v = value.to_string(),
            None => self.entries.push((name.to_string(), value.to_string())),
        }
        Ok(())
    }

    fn remove(&mut self, name: &str) {
        self.entries.retain(|(k, _)| k != name);
    }

    fn key(&self, index: usize) -> Option<String> {
        self.entries.get(index).map(|(k, _)| k.clone())
    }
}

/// The in-memory [`StorageBackend`]: the default until the
/// browser attaches its profile-backed backends, and the backend
/// tests use.
///
/// `localStorage` buckets live in a process-wide shared map (the
/// in-process stand-in for the cross-tab store: `localStorage`
/// is shared across tabs and survives navigation, which
/// recreates the realm — a per-realm map would lose it), while
/// `sessionStorage` buckets are per instance (per realm, i.e.
/// per document load). The browser's backends replace both halves
/// when wired: disk-backed local storage shared across tabs, and
/// a per-tab session store dropped when the tab closes.
pub struct MemoryStorageBackend {
    local: Arc<Mutex<HashMap<StorageKey, StorageBucket>>>,
    session: Arc<Mutex<HashMap<StorageKey, StorageBucket>>>,
}

/// The process-wide shared `localStorage` map (see
/// [`MemoryStorageBackend`]).
static SHARED_LOCAL: OnceLock<Arc<Mutex<HashMap<StorageKey, StorageBucket>>>> =
    OnceLock::new();

fn shared_local() -> Arc<Mutex<HashMap<StorageKey, StorageBucket>>> {
    Arc::clone(SHARED_LOCAL.get_or_init(|| Arc::new(Mutex::new(HashMap::new()))))
}

impl MemoryStorageBackend {
    /// A backend with a fresh local map and a fresh session map
    /// (fully isolated — for unit tests of one backend).
    pub fn new() -> Self {
        Self {
            local: Arc::new(Mutex::new(HashMap::new())),
            session: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The default for a realm: `localStorage` shared
    /// process-wide (cross-tab, survives navigation),
    /// `sessionStorage` fresh per realm.
    pub fn default_for_realm() -> Self {
        Self {
            local: shared_local(),
            session: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn map(&self, kind: StorageKind) -> &Arc<Mutex<HashMap<StorageKey, StorageBucket>>> {
        match kind {
            StorageKind::Local => &self.local,
            StorageKind::Session => &self.session,
        }
    }
}

impl Default for MemoryStorageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for MemoryStorageBackend {
    fn get_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
    ) -> Result<Option<String>, StorageError> {
        if key.origin.is_opaque() {
            return Err(StorageError::SecurityError);
        }
        let map = self.map(kind).lock().unwrap();
        Ok(map.get(key).and_then(|b| b.get(name)))
    }

    fn set_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        if key.origin.is_opaque() {
            return Err(StorageError::SecurityError);
        }
        let mut map = self.map(kind).lock().unwrap();
        map.entry(key.clone()).or_default().set(name, value)
    }

    fn remove_item(&self, kind: StorageKind, key: &StorageKey, name: &str) {
        if key.origin.is_opaque() {
            return;
        }
        let mut map = self.map(kind).lock().unwrap();
        if let Some(bucket) = map.get_mut(key) {
            bucket.remove(name);
        }
    }

    fn clear(&self, kind: StorageKind, key: &StorageKey) {
        if key.origin.is_opaque() {
            return;
        }
        let mut map = self.map(kind).lock().unwrap();
        if let Some(bucket) = map.get_mut(key) {
            bucket.entries.clear();
        }
    }

    fn key(&self, kind: StorageKind, key: &StorageKey, index: usize) -> Option<String> {
        if key.origin.is_opaque() {
            return None;
        }
        let map = self.map(kind).lock().unwrap();
        map.get(key).and_then(|b| b.key(index))
    }

    fn len(&self, kind: StorageKind, key: &StorageKey) -> usize {
        if key.origin.is_opaque() {
            return 0;
        }
        let map = self.map(kind).lock().unwrap();
        map.get(key).map(|b| b.entries.len()).unwrap_or(0)
    }
}

/// Map a [`StorageError`] to the DOMException it raises
/// (`QuotaExceededError` / `SecurityError`).
fn throw_storage_error(ctx: &Ctx<'_>, err: StorageError) -> rquickjs::Error {
    match err {
        StorageError::QuotaExceeded => throw_dom(
            ctx,
            "QuotaExceededError",
            "storage quota exceeded (5 MiB per origin)",
        ),
        StorageError::SecurityError => throw_dom(
            ctx,
            "SecurityError",
            "storage is unavailable for an opaque origin",
        ),
    }
}

/// The document origin for storage, from the attached fetch
/// backend (the realm-side source of the document origin — the
/// same origin the fetch gate checks). `None` means no backend
/// is attached yet: storage is unavailable. An opaque origin
/// means storage throws `SecurityError` (report §f).
fn storage_key(
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<StorageKey, StorageError> {
    let origin = bindings
        .borrow()
        .fetch_backend
        .as_ref()
        .and_then(|backend| backend.document_origin())
        .ok_or(StorageError::SecurityError)?;
    if origin.is_opaque() {
        return Err(StorageError::SecurityError);
    }
    Ok(StorageKey::for_document(&origin))
}

/// Install `localStorage` / `sessionStorage` on the global
/// object (`window.localStorage`, same object every access).
pub fn install_storage<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let globals = ctx.globals();
    for (name, kind) in [
        ("localStorage", StorageKind::Local),
        ("sessionStorage", StorageKind::Session),
    ] {
        let obj = make_storage_object(ctx, bindings, kind)?;
        globals
            .set(name, obj)
            .map_err(|e| format!("install {name}: {e}"))?;
    }
    Ok(())
}

/// Build one `Storage` object: `getItem` / `setItem` /
/// `removeItem` / `clear` / `key` / `length`.
fn make_storage_object<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    kind: StorageKind,
) -> Result<Object<'js>, String> {
    let obj = Object::new(ctx.clone()).map_err(|e| format!("Storage object: {e}"))?;

    // `length`: a live accessor over the backend.
    let b = Rc::clone(bindings);
    let get_length = move |_ctx: Ctx<'js>, This(_this): This<Value<'js>>| -> rquickjs::Result<f64> {
        let key = match storage_key(&b) {
            Ok(key) => key,
            Err(_) => return Ok(0.0),
        };
        Ok(b.borrow().storage_backend.len(kind, &key) as f64)
    };
    obj.prop("length", Accessor::from(get_length))
        .map_err(|e| format!("Storage.length: {e}"))?;

    // `getItem(name)`: `null` when absent (or when storage is
    // unavailable — no, unavailable throws; see below).
    let b = Rc::clone(bindings);
    let get_item = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, name: Coerced<String>| -> rquickjs::Result<Value<'js>> {
            let key = storage_key(&b).map_err(|e| throw_storage_error(&ctx, e))?;
            let backend = Arc::clone(&b.borrow().storage_backend);
            match backend.get_item(kind, &key, &name) {
                Ok(Some(value)) => Ok(js(&ctx, value)),
                Ok(None) => Ok(js_null(&ctx)),
                Err(e) => Err(throw_storage_error(&ctx, e)),
            }
        },
    )
    .map_err(|e| format!("Storage.getItem: {e}"))?;
    obj.set("getItem", get_item)
        .map_err(|e| format!("Storage.getItem: {e}"))?;

    // `setItem(name, value)`: WebIDL `DOMString` coercion for
    // both arguments; quota / opaque-origin throws.
    let b = Rc::clone(bindings);
    let set_item = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, name: Value<'js>, value: Value<'js>| -> rquickjs::Result<()> {
            let name = to_dom_string(&ctx, &name)?;
            let value = to_dom_string(&ctx, &value)?;
            let key = storage_key(&b).map_err(|e| throw_storage_error(&ctx, e))?;
            let backend = Arc::clone(&b.borrow().storage_backend);
            backend
                .set_item(kind, &key, &name, &value)
                .map_err(|e| throw_storage_error(&ctx, e))
        },
    )
    .map_err(|e| format!("Storage.setItem: {e}"))?;
    obj.set("setItem", set_item)
        .map_err(|e| format!("Storage.setItem: {e}"))?;

    // `removeItem(name)`.
    let b = Rc::clone(bindings);
    let remove_item = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, name: Coerced<String>| -> rquickjs::Result<()> {
            let key = storage_key(&b).map_err(|e| throw_storage_error(&ctx, e))?;
            let backend = Arc::clone(&b.borrow().storage_backend);
            backend.remove_item(kind, &key, &name);
            Ok(())
        },
    )
    .map_err(|e| format!("Storage.removeItem: {e}"))?;
    obj.set("removeItem", remove_item)
        .map_err(|e| format!("Storage.removeItem: {e}"))?;

    // `clear()`.
    let b = Rc::clone(bindings);
    let clear = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>| -> rquickjs::Result<()> {
            let key = storage_key(&b).map_err(|e| throw_storage_error(&ctx, e))?;
            let backend = Arc::clone(&b.borrow().storage_backend);
            backend.clear(kind, &key);
            Ok(())
        },
    )
    .map_err(|e| format!("Storage.clear: {e}"))?;
    obj.set("clear", clear)
        .map_err(|e| format!("Storage.clear: {e}"))?;

    // `key(index)`: `null` out of range; a non-integer index
    // coerces per WebIDL `unsigned long` (NaN / negative → 0 is
    // the `ToUint32` outcome — QuickJS's `Coerced<f64>` plus a
    // clamp below keeps this honest enough for M5).
    let b = Rc::clone(bindings);
    let key_fn = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, index: Coerced<f64>| -> rquickjs::Result<Value<'js>> {
            let key = storage_key(&b).map_err(|e| throw_storage_error(&ctx, e))?;
            let backend = Arc::clone(&b.borrow().storage_backend);
            let index = if index.is_nan() || *index < 0.0 {
                0usize
            } else {
                *index as usize
            };
            match backend.key(kind, &key, index) {
                Some(name) => Ok(js(&ctx, name)),
                None => Ok(js_null(&ctx)),
            }
        },
    )
    .map_err(|e| format!("Storage.key: {e}"))?;
    obj.set("key", key_fn)
        .map_err(|e| format!("Storage.key: {e}"))?;

    // `Storage` objects are not constructable by pages (M5: no
    // `Storage` constructor is installed — `localStorage` /
    // `sessionStorage` are the only instances).
    Ok(obj)
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

    /// A fetch backend that only carries a document origin (the
    /// storage bindings read the origin from it; no fetch ever
    /// runs in these tests).
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
                result: Err(FetchError::Network("no network in storage tests".to_string())),
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

    fn read_num(tab: &mut TabJs, source: &str) -> f64 {
        match tab.eval_value(source).unwrap() {
            JsValue::Number(n) => n,
            other => panic!("expected a number from `{source}`, got {other:?}"),
        }
    }

    fn example_origin() -> Origin {
        Origin::from_url("https://storage-a.example/page")
    }

    // ------------------------------------------------------------ backend

    #[test]
    fn partition_keys_differ_by_top_level_site() {
        // Same widget origin, different top-level sites: different
        // buckets (report §f).
        let backend = MemoryStorageBackend::new();
        let widget = Origin::from_url("https://widget.example/embed");
        let shop = StorageKey::new(&widget, "https://shop.example");
        let news = StorageKey::new(&widget, "https://news.example");
        backend
            .set_item(StorageKind::Local, &shop, "cart", "shop-cart")
            .unwrap();
        backend
            .set_item(StorageKind::Local, &news, "cart", "news-cart")
            .unwrap();
        assert_eq!(
            backend
                .get_item(StorageKind::Local, &shop, "cart")
                .unwrap()
                .as_deref(),
            Some("shop-cart")
        );
        assert_eq!(
            backend
                .get_item(StorageKind::Local, &news, "cart")
                .unwrap()
                .as_deref(),
            Some("news-cart")
        );
        // Same top-level site, different frame origins: different
        // buckets too.
        let other = StorageKey::new(
            &Origin::from_url("https://other-widget.example/embed"),
            "https://shop.example",
        );
        assert_eq!(
            backend
                .get_item(StorageKind::Local, &other, "cart")
                .unwrap(),
            None
        );
    }

    #[test]
    fn top_level_site_ignores_port_and_is_host_approximation() {
        // A site ignores port (report §b/§f).
        assert_eq!(
            top_level_site_of(&Origin::from_url("https://example.com:8443/x")),
            "https://example.com"
        );
        assert_eq!(
            top_level_site_of(&Origin::from_url("https://example.com/x")),
            "https://example.com"
        );
        // No PSL: a subdomain is a different site here (the
        // documented approximation, not a PSL computation).
        assert_ne!(
            top_level_site_of(&Origin::from_url("https://sub.example.com/x")),
            top_level_site_of(&Origin::from_url("https://example.com/x"))
        );
        assert_eq!(top_level_site_of(&Origin::opaque()), "null");
    }

    #[test]
    fn quota_enforced_at_five_mib() {
        let backend = MemoryStorageBackend::new();
        let key = StorageKey::for_document(&example_origin());
        // Exactly quota: 1-byte key + (quota - 1)-byte value.
        let value = "x".repeat(STORAGE_QUOTA_BYTES - 1);
        backend
            .set_item(StorageKind::Local, &key, "k", &value)
            .unwrap();
        // One more byte, anywhere, exceeds it.
        assert_eq!(
            backend.set_item(StorageKind::Local, &key, "k2", "x"),
            Err(StorageError::QuotaExceeded)
        );
        // Updating the existing key to a smaller value frees room.
        backend
            .set_item(StorageKind::Local, &key, "k", "small")
            .unwrap();
        backend
            .set_item(StorageKind::Local, &key, "k2", "x")
            .unwrap();
        // A single value larger than the whole quota never fits.
        let huge = "y".repeat(STORAGE_QUOTA_BYTES + 1);
        assert_eq!(
            backend.set_item(StorageKind::Local, &key, "big", &huge),
            Err(StorageError::QuotaExceeded)
        );
        // A failed write leaves the old value untouched.
        assert_eq!(
            backend
                .get_item(StorageKind::Local, &key, "k")
                .unwrap()
                .as_deref(),
            Some("small")
        );
    }

    #[test]
    fn opaque_origin_is_security_error_at_backend() {
        let backend = MemoryStorageBackend::new();
        let key = StorageKey::for_document(&Origin::opaque());
        assert_eq!(
            backend.set_item(StorageKind::Local, &key, "k", "v"),
            Err(StorageError::SecurityError)
        );
        assert_eq!(
            backend.get_item(StorageKind::Local, &key, "k"),
            Err(StorageError::SecurityError)
        );
        assert_eq!(backend.len(StorageKind::Local, &key), 0);
        assert_eq!(backend.key(StorageKind::Local, &key, 0), None);
    }

    #[test]
    fn key_order_is_insertion_order() {
        let backend = MemoryStorageBackend::new();
        let key = StorageKey::for_document(&example_origin());
        backend
            .set_item(StorageKind::Local, &key, "b", "2")
            .unwrap();
        backend
            .set_item(StorageKind::Local, &key, "a", "1")
            .unwrap();
        backend
            .set_item(StorageKind::Local, &key, "c", "3")
            .unwrap();
        assert_eq!(backend.len(StorageKind::Local, &key), 3);
        assert_eq!(
            backend.key(StorageKind::Local, &key, 0).as_deref(),
            Some("b")
        );
        assert_eq!(
            backend.key(StorageKind::Local, &key, 1).as_deref(),
            Some("a")
        );
        // Updating a key keeps its position.
        backend
            .set_item(StorageKind::Local, &key, "b", "20")
            .unwrap();
        assert_eq!(
            backend.key(StorageKind::Local, &key, 0).as_deref(),
            Some("b")
        );
        assert_eq!(backend.key(StorageKind::Local, &key, 3), None);
        backend.remove_item(StorageKind::Local, &key, "a");
        assert_eq!(backend.len(StorageKind::Local, &key), 2);
        backend.clear(StorageKind::Local, &key);
        assert_eq!(backend.len(StorageKind::Local, &key), 0);
    }

    #[test]
    fn session_and_local_buckets_are_independent() {
        let backend = MemoryStorageBackend::new();
        let key = StorageKey::for_document(&example_origin());
        backend
            .set_item(StorageKind::Local, &key, "k", "local")
            .unwrap();
        backend
            .set_item(StorageKind::Session, &key, "k", "session")
            .unwrap();
        assert_eq!(
            backend
                .get_item(StorageKind::Local, &key, "k")
                .unwrap()
                .as_deref(),
            Some("local")
        );
        assert_eq!(
            backend
                .get_item(StorageKind::Session, &key, "k")
                .unwrap()
                .as_deref(),
            Some("session")
        );
    }

    // ------------------------------------------------------------ JS surface

    #[test]
    fn local_storage_round_trip_through_js() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "localStorage.setItem('name', 'pluto'); \
             localStorage.setItem('n', 42); \
             globalThis.__r = [ \
               localStorage.getItem('name'), \
               localStorage.getItem('n'), \
               String(localStorage.getItem('missing')), \
               localStorage.length, \
               localStorage.key(0), \
               localStorage.key(1), \
               String(localStorage.key(9)), \
             ].join('|');",
        );
        assert_eq!(read_str(&mut tab, "__r"), "pluto|42|null|2|name|n|null");
        run(&mut tab, "localStorage.removeItem('n'); globalThis.__r = localStorage.length;");
        assert_eq!(read_num(&mut tab, "__r"), 1.0);
        run(&mut tab, "localStorage.clear(); globalThis.__r = localStorage.length;");
        assert_eq!(read_num(&mut tab, "__r"), 0.0);
    }

    #[test]
    fn session_storage_round_trip_through_js() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "sessionStorage.setItem('tab', 'one'); \
             globalThis.__r = [sessionStorage.getItem('tab'), sessionStorage.length].join('|');",
        );
        assert_eq!(read_str(&mut tab, "__r"), "one|1");
        // The two storages do not share entries.
        run(
            &mut tab,
            "globalThis.__r = String(localStorage.getItem('tab'));",
        );
        assert_eq!(read_str(&mut tab, "__r"), "null");
    }

    #[test]
    fn local_storage_is_shared_across_realms() {
        // `localStorage` is shared across tabs / navigations: the
        // default backend's local map is process-wide, so a value
        // set before a navigation is still there after it.
        let origin = Origin::from_url("https://storage-shared.example/page");
        let mut tab = tab_with_origin(Some(origin));
        run(&mut tab, "localStorage.setItem('shared', 'yes');");
        tab.navigate().unwrap();
        run(
            &mut tab,
            "globalThis.__r = String(localStorage.getItem('shared'));",
        );
        assert_eq!(read_str(&mut tab, "__r"), "yes");
        // A different origin does not see it.
        let mut other = tab_with_origin(Some(Origin::from_url(
            "https://storage-other.example/page",
        )));
        run(
            &mut other,
            "globalThis.__r = String(localStorage.getItem('shared'));",
        );
        assert_eq!(read_str(&mut other, "__r"), "null");
    }

    #[test]
    fn storage_throws_security_error_in_opaque_origin() {
        let mut tab = tab_with_origin(Some(Origin::opaque()));
        // The exception's `name` is read in JS (the `JsError`
        // rendering only carries the message — the established
        // pattern from the other binding tests).
        run(
            &mut tab,
            "try { localStorage.setItem('k', 'v'); globalThis.__r = 'no-throw'; } \
             catch (e) { globalThis.__r = e.name; }",
        );
        assert_eq!(read_str(&mut tab, "__r"), "SecurityError");
        run(
            &mut tab,
            "try { sessionStorage.getItem('k'); globalThis.__r = 'no-throw'; } \
             catch (e) { globalThis.__r = e.name; }",
        );
        assert_eq!(read_str(&mut tab, "__r"), "SecurityError");
    }

    #[test]
    fn storage_throws_quota_exceeded_error_through_js() {
        let mut tab = tab_with_origin(Some(Origin::from_url(
            "https://storage-quota.example/page",
        )));
        run(
            &mut tab,
            "localStorage.setItem('k', 'x'.repeat(5 * 1024 * 1024 - 1));",
        );
        run(
            &mut tab,
            "try { localStorage.setItem('k2', 'x'); globalThis.__r = 'no-throw'; } \
             catch (e) { globalThis.__r = e.name; }",
        );
        assert_eq!(read_str(&mut tab, "__r"), "QuotaExceededError");
        // The failed write left the bucket intact.
        run(
            &mut tab,
            "globalThis.__r = localStorage.length;",
        );
        assert_eq!(read_num(&mut tab, "__r"), 1.0);
    }

    #[test]
    fn storage_objects_are_singletons() {
        let mut tab = tab_with_origin(Some(example_origin()));
        run(
            &mut tab,
            "globalThis.__r = String(localStorage === window.localStorage && sessionStorage === window.sessionStorage);",
        );
        assert_eq!(read_str(&mut tab, "__r"), "true");
    }
}
