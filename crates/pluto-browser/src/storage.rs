//! Partitioned web storage, browser side (Stage 9 Rec 4).
//!
//! The JS surface is in `pluto-js` (`bindings::storage`); this
//! module is the profile-backed store it talks to through the
//! [`StorageBackend`] trait:
//!
//! * [`LocalStorageStore`] — one bucket per [`StorageKey`]
//!   (`top-level site`, `origin`), shared across all tabs of
//!   that origin, persisted to the profile as `local-storage.json`
//!   (serde-free JSON — `pluto-browser` has no serde dependency
//!   of its own — written temp-file + rename, the
//!   [`security`](crate::security) pattern).
//! * [`SessionStorageStore`] — the same buckets, per top-level
//!   browsing context (per tab), memory-only, dropped (hence
//!   cleared) when the tab closes.
//!
//! Quotas: 5 MiB (5 × 2²⁰ bytes) per bucket, counted as the UTF-8
//! byte length of keys + values; a write that would exceed it
//! returns [`StorageError::QuotaExceeded`] and writes nothing.
//! An opaque document origin (sandboxed iframe without
//! `allow-same-origin`, `data:` document) returns
//! [`StorageError::SecurityError`] before any bucket is touched.
//!
//! The top-level site is `scheme + host` of the top-level origin
//! — the host stands in for the registrable domain because there
//! is NO public-suffix list in the tree (do not hand-roll one);
//! a PSL is the future refinement. See
//! [`pluto_js::bindings::storage::top_level_site_of`].
//!
//! Wiring: [`PlutoStorageBackend`] is the per-tab backend
//! (shared local store + its own session store). The tab thread
//! builds one per tab via [`crate::js_host::new_tab_backends`];
//! the `TabJs` setter that attaches it is the follow-up hunk
//! documented there.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pluto_ipc::Origin;
use pluto_js::bindings::storage::{
    StorageBackend, StorageError, StorageKey, StorageKind, STORAGE_QUOTA_BYTES,
};

use crate::security::profile_dir;

/// File name for the persisted `localStorage` store inside the
/// profile dir.
pub const LOCAL_STORAGE_FILE: &str = "local-storage.json";

// ------------------------------------------------------------------ JSON
//
// Minimal JSON value + parser, only what the profile files need
// (objects / arrays / strings / numbers / booleans / null).
// `pluto-browser` names no serde dependency, so the profile
// stores serialize by hand; the parser is deliberately strict
// about structure and lenient about nothing else — a corrupt
// file loads as an empty store (logged, never fatal).

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    Obj(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// The value of `key` in an object, if this is one.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Obj(pairs) => pairs
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The number, if this is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// The array, if this is one.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Arr(items) => Some(items),
            _ => None,
        }
    }
}

/// Escape a string as a JSON string literal (including the
/// quotes).
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct JsonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    input: &'a str,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            input,
        }
    }

    fn err(&self, msg: &str) -> String {
        format!("JSON parse error at byte {}: {msg}", self.pos)
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.peek() {
            Some(got) if got == c => {
                self.pos += 1;
                Ok(())
            }
            other => Err(self.err(&format!("expected '{c}', got {other:?}"))),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(JsonValue::Str(self.parse_string()?)),
            Some('t') => self.parse_lit("true", JsonValue::Bool(true)),
            Some('f') => self.parse_lit("false", JsonValue::Bool(false)),
            Some('n') => self.parse_lit("null", JsonValue::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(self.err(&format!("unexpected character {other:?}"))),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: JsonValue) -> Result<JsonValue, String> {
        for expected in lit.chars() {
            match self.chars.get(self.pos) {
                Some(&got) if got == expected => self.pos += 1,
                _ => return Err(self.err(&format!("expected literal '{lit}'"))),
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        while self.pos < self.chars.len()
            && matches!(
                self.chars[self.pos],
                '-' | '+' | '.' | 'e' | 'E' | '0'..='9'
            )
        {
            self.pos += 1;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map(JsonValue::Num)
            .map_err(|_| self.err(&format!("bad number '{text}'")))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            let Some(c) = self.chars.get(self.pos).copied() else {
                return Err(self.err("unterminated string"));
            };
            self.pos += 1;
            match c {
                '"' => break,
                '\\' => {
                    let esc = self.chars.get(self.pos).copied().ok_or_else(|| {
                        self.err("unterminated escape")
                    })?;
                    self.pos += 1;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000C}'),
                        'u' => {
                            let mut code: u32 = 0;
                            for _ in 0..4 {
                                let h = self.chars.get(self.pos).copied().ok_or_else(|| {
                                    self.err("short \\u escape")
                                })?;
                                self.pos += 1;
                                code = code * 16
                                    + h.to_digit(16)
                                        .ok_or_else(|| self.err("bad \\u escape"))?;
                            }
                            // Surrogate pairs: a high surrogate is
                            // followed by `\uDC00`–`\uDFFF`.
                            if (0xD800..0xDC00).contains(&code)
                                && self.chars.get(self.pos) == Some(&'\\')
                                && self.chars.get(self.pos + 1) == Some(&'u')
                            {
                                let save = self.pos;
                                self.pos += 2;
                                let mut low: u32 = 0;
                                let mut ok = true;
                                for _ in 0..4 {
                                    match self.chars.get(self.pos).copied() {
                                        Some(h) => {
                                            self.pos += 1;
                                            match h.to_digit(16) {
                                                Some(d) => low = low * 16 + d,
                                                None => ok = false,
                                            }
                                        }
                                        None => ok = false,
                                    }
                                }
                                if ok && (0xDC00..0xE000).contains(&low) {
                                    let c = 0x10000
                                        + ((code - 0xD800) << 10)
                                        + (low - 0xDC00);
                                    out.push(
                                        char::from_u32(c).unwrap_or('\u{FFFD}'),
                                    );
                                } else {
                                    self.pos = save;
                                    out.push('\u{FFFD}');
                                }
                            } else {
                                out.push(
                                    char::from_u32(code).unwrap_or('\u{FFFD}'),
                                );
                            }
                        }
                        other => {
                            return Err(self.err(&format!(
                                "bad escape '\\{other}'"
                            )))
                        }
                    }
                }
                c => out.push(c),
            }
        }
        Ok(out)
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(JsonValue::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(self.err(&format!(
                        "expected ',' or ']' in array, got {other:?}"
                    )))
                }
            }
        }
        Ok(JsonValue::Arr(items))
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect('{')?;
        let mut pairs = Vec::new();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(JsonValue::Obj(pairs));
        }
        loop {
            if self.peek() != Some('"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.expect(':')?;
            let value = self.parse_value()?;
            pairs.push((key, value));
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(self.err(&format!(
                        "expected ',' or '}}' in object, got {other:?}"
                    )))
                }
            }
        }
        Ok(JsonValue::Obj(pairs))
    }
}

/// Parse a whole JSON document. Trailing garbage is an error.
pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = JsonParser::new(input);
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.chars.len() {
        return Err(parser.err("trailing characters after JSON value"));
    }
    let _ = parser.input;
    Ok(value)
}

/// Write a file atomically (temp-file + rename), creating the
/// profile dir if needed — the [`security`](crate::security)
/// pattern, duplicated here because that module's helper is
/// private.
pub fn write_file_atomic(dir: &PathBuf, name: &str, contents: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let target = dir.join(name);
    let tmp = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, target)?;
    Ok(())
}

// ------------------------------------------------------------------ bucket

/// One storage bucket: entries in insertion order (the `key(n)`
/// order), with the quota accounted in UTF-8 bytes. Mirrors the
/// `pluto-js` bucket semantics exactly (same quota rule, same
/// order); the two implementations are separate because the
/// browser store is the persistent one and the `pluto-js` one is
/// the in-memory default.
#[derive(Debug, Default, Clone)]
struct Bucket {
    entries: Vec<(String, String)>,
}

impl Bucket {
    fn byte_len(&self) -> usize {
        self.entries.iter().map(|(k, v)| k.len() + v.len()).sum()
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

    fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .entries
            .iter()
            .map(|(k, v)| format!("[{},{}]", json_escape(k), json_escape(v)))
            .collect();
        format!("[{}]", parts.join(","))
    }

    fn from_json(value: &JsonValue) -> Option<Self> {
        let mut bucket = Bucket::default();
        for pair in value.as_array()? {
            let pair = pair.as_array()?;
            if pair.len() != 2 {
                return None;
            }
            bucket.entries.push((
                pair[0].as_str()?.to_string(),
                pair[1].as_str()?.to_string(),
            ));
        }
        Some(bucket)
    }
}

fn check_origin(origin: &Origin) -> Result<(), StorageError> {
    if origin.is_opaque() {
        return Err(StorageError::SecurityError);
    }
    Ok(())
}

fn buckets_to_json(buckets: &HashMap<StorageKey, Bucket>) -> String {
    let mut parts = Vec::new();
    for (key, bucket) in buckets {
        parts.push(format!(
            "{{\"top_level_site\":{},\"origin\":{},\"entries\":{}}}",
            json_escape(&key.top_level_site),
            json_escape(&key.origin.serialize()),
            bucket.to_json(),
        ));
    }
    format!("{{\"version\":1,\"buckets\":[{}]}}", parts.join(","))
}

fn buckets_from_json(text: &str) -> HashMap<StorageKey, Bucket> {
    let parsed = match parse_json(text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[pluto] corrupt local-storage profile ({e}); starting empty");
            return HashMap::new();
        }
    };
    let mut out = HashMap::new();
    let Some(buckets) = parsed.get("buckets").and_then(|b| b.as_array()) else {
        eprintln!("[pluto] local-storage profile has no buckets; starting empty");
        return out;
    };
    for item in buckets {
        let (Some(site), Some(origin_s), Some(entries)) = (
            item.get("top_level_site").and_then(|v| v.as_str()),
            item.get("origin").and_then(|v| v.as_str()),
            item.get("entries"),
        ) else {
            continue;
        };
        let origin = Origin::from_url(origin_s);
        if origin.is_opaque() {
            continue; // Opaque origins never persist a bucket.
        }
        let Some(bucket) = Bucket::from_json(entries) else {
            continue;
        };
        out.insert(
            StorageKey::new(&origin, site),
            bucket,
        );
    }
    out
}

// ------------------------------------------------------------------ local

/// The profile-backed `localStorage` store: one bucket per
/// partition key, shared across all tabs of that origin.
#[derive(Debug, Default)]
pub struct LocalStorageStore {
    buckets: HashMap<StorageKey, Bucket>,
}

impl LocalStorageStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_item(
        &self,
        key: &StorageKey,
        name: &str,
    ) -> Result<Option<String>, StorageError> {
        check_origin(&key.origin)?;
        Ok(self.buckets.get(key).and_then(|b| b.get(name)))
    }

    pub fn set_item(
        &mut self,
        key: &StorageKey,
        name: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        check_origin(&key.origin)?;
        self.buckets.entry(key.clone()).or_default().set(name, value)
    }

    pub fn remove_item(&mut self, key: &StorageKey, name: &str) {
        if check_origin(&key.origin).is_err() {
            return;
        }
        if let Some(bucket) = self.buckets.get_mut(key) {
            bucket.remove(name);
        }
    }

    pub fn clear(&mut self, key: &StorageKey) {
        if check_origin(&key.origin).is_err() {
            return;
        }
        if let Some(bucket) = self.buckets.get_mut(key) {
            bucket.entries.clear();
        }
    }

    pub fn key(&self, key: &StorageKey, index: usize) -> Option<String> {
        if check_origin(&key.origin).is_err() {
            return None;
        }
        self.buckets.get(key).and_then(|b| b.key(index))
    }

    pub fn len(&self, key: &StorageKey) -> usize {
        if check_origin(&key.origin).is_err() {
            return 0;
        }
        self.buckets.get(key).map(|b| b.entries.len()).unwrap_or(0)
    }

    /// Serialise the whole store (for the profile file).
    pub fn to_json(&self) -> String {
        buckets_to_json(&self.buckets)
    }

    /// Deserialise a profile file (corrupt input → empty store,
    /// logged).
    pub fn from_json(text: &str) -> Self {
        Self {
            buckets: buckets_from_json(text),
        }
    }
}

/// Load the persisted `localStorage` store. Missing or corrupt
/// files fall back to an empty store (logged, not fatal).
pub fn load_local_storage() -> LocalStorageStore {
    let Some(dir) = profile_dir() else {
        eprintln!("[pluto] no HOME: localStorage persistence disabled");
        return LocalStorageStore::new();
    };
    match std::fs::read_to_string(dir.join(LOCAL_STORAGE_FILE)) {
        Ok(text) => LocalStorageStore::from_json(&text),
        Err(_) => {
            eprintln!(
                "[pluto] no persisted localStorage yet ({})",
                dir.display()
            );
            LocalStorageStore::new()
        }
    }
}

/// Save the `localStorage` store to the profile. Failures are
/// logged, never fatal.
pub fn save_local_storage(store: &LocalStorageStore) {
    let Some(dir) = profile_dir() else {
        return;
    };
    if let Err(e) = write_file_atomic(&dir, LOCAL_STORAGE_FILE, &store.to_json()) {
        eprintln!("[pluto] could not save localStorage: {e}");
    }
}

// ------------------------------------------------------------------ session

/// The per-tab `sessionStorage` store: the same buckets, but
/// memory-only and owned by one tab — dropping the owning
/// [`PlutoStorageBackend`] (on tab close) clears it.
#[derive(Debug, Default)]
pub struct SessionStorageStore {
    buckets: HashMap<StorageKey, Bucket>,
}

impl SessionStorageStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every bucket (the tab-close clear; also what dropping
    /// the store does implicitly).
    pub fn clear_all(&mut self) {
        self.buckets.clear();
    }

    pub fn get_item(
        &self,
        key: &StorageKey,
        name: &str,
    ) -> Result<Option<String>, StorageError> {
        check_origin(&key.origin)?;
        Ok(self.buckets.get(key).and_then(|b| b.get(name)))
    }

    pub fn set_item(
        &mut self,
        key: &StorageKey,
        name: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        check_origin(&key.origin)?;
        self.buckets.entry(key.clone()).or_default().set(name, value)
    }

    pub fn remove_item(&mut self, key: &StorageKey, name: &str) {
        if check_origin(&key.origin).is_err() {
            return;
        }
        if let Some(bucket) = self.buckets.get_mut(key) {
            bucket.remove(name);
        }
    }

    pub fn clear(&mut self, key: &StorageKey) {
        if check_origin(&key.origin).is_err() {
            return;
        }
        if let Some(bucket) = self.buckets.get_mut(key) {
            bucket.entries.clear();
        }
    }

    pub fn key(&self, key: &StorageKey, index: usize) -> Option<String> {
        if check_origin(&key.origin).is_err() {
            return None;
        }
        self.buckets.get(key).and_then(|b| b.key(index))
    }

    pub fn len(&self, key: &StorageKey) -> usize {
        if check_origin(&key.origin).is_err() {
            return 0;
        }
        self.buckets.get(key).map(|b| b.entries.len()).unwrap_or(0)
    }
}

// ------------------------------------------------------------------ backend

/// The per-tab storage backend for one realm: the shared,
/// profile-backed `localStorage` store plus this tab's own
/// `sessionStorage` store. Implements the `pluto-js`
/// [`StorageBackend`] trait, so the JS bindings call it through
/// the same interface as the in-memory default.
pub struct PlutoStorageBackend {
    local: Arc<Mutex<LocalStorageStore>>,
    session: Arc<Mutex<SessionStorageStore>>,
}

impl PlutoStorageBackend {
    /// A backend over the shared local store (loaded once per
    /// tab creation from the profile) with a fresh session store.
    pub fn new(local: Arc<Mutex<LocalStorageStore>>) -> Self {
        Self {
            local,
            session: Arc::new(Mutex::new(SessionStorageStore::new())),
        }
    }

    /// The shared local store (for saving after a write).
    pub fn local_store(&self) -> Arc<Mutex<LocalStorageStore>> {
        Arc::clone(&self.local)
    }

    /// Persist the shared local store to the profile now.
    pub fn save_local(&self) {
        save_local_storage(&self.local.lock().unwrap());
    }
}

impl StorageBackend for PlutoStorageBackend {
    fn get_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
    ) -> Result<Option<String>, StorageError> {
        match kind {
            StorageKind::Local => self.local.lock().unwrap().get_item(key, name),
            StorageKind::Session => self.session.lock().unwrap().get_item(key, name),
        }
    }

    fn set_item(
        &self,
        kind: StorageKind,
        key: &StorageKey,
        name: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        let result = match kind {
            StorageKind::Local => self.local.lock().unwrap().set_item(key, name, value),
            StorageKind::Session => {
                self.session.lock().unwrap().set_item(key, name, value)
            }
        };
        // `localStorage` persists to disk: a successful write is
        // saved at once (the files are small and writes are
        // infrequent relative to reads).
        if result.is_ok() && kind == StorageKind::Local {
            self.save_local();
        }
        result
    }

    fn remove_item(&self, kind: StorageKind, key: &StorageKey, name: &str) {
        match kind {
            StorageKind::Local => {
                self.local.lock().unwrap().remove_item(key, name);
                self.save_local();
            }
            StorageKind::Session => self.session.lock().unwrap().remove_item(key, name),
        }
    }

    fn clear(&self, kind: StorageKind, key: &StorageKey) {
        match kind {
            StorageKind::Local => {
                self.local.lock().unwrap().clear(key);
                self.save_local();
            }
            StorageKind::Session => self.session.lock().unwrap().clear(key),
        }
    }

    fn key(&self, kind: StorageKind, key: &StorageKey, index: usize) -> Option<String> {
        match kind {
            StorageKind::Local => self.local.lock().unwrap().key(key, index),
            StorageKind::Session => self.session.lock().unwrap().key(key, index),
        }
    }

    fn len(&self, kind: StorageKind, key: &StorageKey) -> usize {
        match kind {
            StorageKind::Local => self.local.lock().unwrap().len(key),
            StorageKind::Session => self.session.lock().unwrap().len(key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_js::bindings::storage::top_level_site_of;

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pluto-storage-test-{tag}"));
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
        // swapping it mid-span made the store read the wrong home.
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

    fn widget_key(site: &str) -> StorageKey {
        StorageKey::new(
            &Origin::from_url("https://widget.example/embed"),
            site,
        )
    }

    #[test]
    fn partition_keys_differ_by_top_level_site() {
        let mut store = LocalStorageStore::new();
        let shop = widget_key("https://shop.example");
        let news = widget_key("https://news.example");
        store.set_item(&shop, "cart", "shop-cart").unwrap();
        store.set_item(&news, "cart", "news-cart").unwrap();
        assert_eq!(
            store.get_item(&shop, "cart").unwrap().as_deref(),
            Some("shop-cart")
        );
        assert_eq!(
            store.get_item(&news, "cart").unwrap().as_deref(),
            Some("news-cart")
        );
        // The same widget at top level gets its own bucket.
        let top = StorageKey::for_document(&Origin::from_url("https://widget.example/"));
        assert_eq!(store.get_item(&top, "cart").unwrap(), None);
        // Site ignores port: :8443 and no port share the site,
        // hence the same key shape (different frame origins still
        // differ).
        let a = StorageKey::new(
            &Origin::from_url("https://widget.example:8443/embed"),
            &top_level_site_of(&Origin::from_url("https://shop.example:8443/")),
        );
        let b = StorageKey::new(
            &Origin::from_url("https://widget.example:8443/embed"),
            &top_level_site_of(&Origin::from_url("https://shop.example/")),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn five_mib_quota_enforced() {
        let mut store = LocalStorageStore::new();
        let key = StorageKey::for_document(&Origin::from_url("https://quota.example/"));
        let value = "x".repeat(STORAGE_QUOTA_BYTES - 1);
        store.set_item(&key, "k", &value).unwrap();
        assert_eq!(
            store.set_item(&key, "k2", "x"),
            Err(StorageError::QuotaExceeded)
        );
        store.set_item(&key, "k", "small").unwrap();
        store.set_item(&key, "k2", "x").unwrap();
        // The failed write left the old value untouched.
        assert_eq!(
            store.get_item(&key, "k").unwrap().as_deref(),
            Some("small")
        );
    }

    #[test]
    fn opaque_origin_is_security_error() {
        let mut store = LocalStorageStore::new();
        let key = StorageKey::for_document(&Origin::opaque());
        assert_eq!(
            store.set_item(&key, "k", "v"),
            Err(StorageError::SecurityError)
        );
        assert_eq!(
            store.get_item(&key, "k"),
            Err(StorageError::SecurityError)
        );
        assert_eq!(store.len(&key), 0);
        let mut session = SessionStorageStore::new();
        assert_eq!(
            session.set_item(&key, "k", "v"),
            Err(StorageError::SecurityError)
        );
    }

    #[test]
    fn session_storage_is_per_tab_and_cleared_on_close() {
        let key = StorageKey::for_document(&Origin::from_url("https://tab.example/"));
        // Two tabs: independent session buckets.
        let mut tab_a = SessionStorageStore::new();
        let mut tab_b = SessionStorageStore::new();
        tab_a.set_item(&key, "tab", "a").unwrap();
        tab_b.set_item(&key, "tab", "b").unwrap();
        assert_eq!(
            tab_a.get_item(&key, "tab").unwrap().as_deref(),
            Some("a")
        );
        assert_eq!(
            tab_b.get_item(&key, "tab").unwrap().as_deref(),
            Some("b")
        );
        // Closing tab A (dropping / clearing its store) does not
        // touch tab B — and a fresh store starts empty.
        tab_a.clear_all();
        assert_eq!(tab_a.get_item(&key, "tab").unwrap(), None);
        assert_eq!(
            tab_b.get_item(&key, "tab").unwrap().as_deref(),
            Some("b")
        );
        drop(tab_a);
        let fresh = SessionStorageStore::new();
        assert_eq!(fresh.get_item(&key, "tab").unwrap(), None);
    }

    #[test]
    fn local_storage_persists_across_loads() {
        let home = temp_home("round-trip");
        let old = with_home(&home);
        let key = StorageKey::for_document(&Origin::from_url("https://persist.example/"));
        let shop = widget_key("https://shop.example");
        {
            let mut store = load_local_storage();
            store.set_item(&key, "name", "pluto \"quoted\" \\ backslash\nnewline").unwrap();
            store.set_item(&shop, "cart", "3 items").unwrap();
            save_local_storage(&store);
        }
        let loaded = load_local_storage();
        assert_eq!(
            loaded.get_item(&key, "name").unwrap().as_deref(),
            Some("pluto \"quoted\" \\ backslash\nnewline")
        );
        assert_eq!(
            loaded.get_item(&shop, "cart").unwrap().as_deref(),
            Some("3 items")
        );
        // The other site's bucket stayed separate on disk too.
        let news = widget_key("https://news.example");
        assert_eq!(loaded.get_item(&news, "cart").unwrap(), None);
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn corrupt_local_storage_loads_empty() {
        let home = temp_home("corrupt");
        let old = with_home(&home);
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(LOCAL_STORAGE_FILE), "not json{{").unwrap();
        let loaded = load_local_storage();
        let key = StorageKey::for_document(&Origin::from_url("https://example.com/"));
        assert_eq!(loaded.len(&key), 0);
        // And a store that round-trips keeps insertion order for
        // `key(n)`.
        let mut store = LocalStorageStore::new();
        store.set_item(&key, "b", "2").unwrap();
        store.set_item(&key, "a", "1").unwrap();
        assert_eq!(store.key(&key, 0).as_deref(), Some("b"));
        assert_eq!(store.key(&key, 1).as_deref(), Some("a"));
        assert_eq!(store.key(&key, 2), None);
        let reloaded = LocalStorageStore::from_json(&store.to_json());
        assert_eq!(reloaded.key(&key, 0).as_deref(), Some("b"));
        assert_eq!(reloaded.key(&key, 1).as_deref(), Some("a"));
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn json_helpers_round_trip() {
        for (value, text) in [
            (JsonValue::Str("a\"b\\c\nd".to_string()), "\"a\\\"b\\\\c\\nd\""),
            (JsonValue::Num(42.0), "42"),
            (JsonValue::Bool(true), "true"),
            (JsonValue::Null, "null"),
        ] {
            let parsed = parse_json(text).unwrap();
            assert_eq!(parsed, value);
        }
        let doc = "{\"a\":[1,\"x\",{\"b\":null}],\"c\":\"\\u00e9\"}";
        let parsed = parse_json(doc).unwrap();
        assert_eq!(
            parsed
                .get("c")
                .and_then(|v| v.as_str()),
            Some("é")
        );
        assert!(parse_json("{\"a\":}").is_err());
        assert!(parse_json("[1,2] trailing").is_err());
        assert!(parse_json("").is_err());
    }
}
