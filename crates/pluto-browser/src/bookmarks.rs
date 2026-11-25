//! Persistent bookmarks (nav feature, 2026-11-24).
//!
//! The bookmark star in the nav bar needs a real backend, so this
//! is one: a minimal store of bookmarked URLs, persisted to the
//! profile as `bookmarks.json` next to the other profile files
//! (`$HOME/.config/pluto/`, see [`crate::security::profile_dir`]).
//!
//! The format follows the existing profile pattern (serde-free
//! JSON via [`crate::js_host::storage`], temp-file + rename
//! write): `{"version":1,"bookmarks":[{"url":…,"title":…}]}`.
//! A bookmark's title is the page URL for now — the shell has no
//! page-title plumbing yet (`TitleChanged` is ignored until M3),
//! so there is no title to store. A missing or corrupt file loads
//! as an empty store (logged, never fatal), like the other
//! profile stores.

use crate::js_host::storage::{json_escape, parse_json, write_file_atomic};
use crate::security::profile_dir;

/// File name for the persisted bookmark store inside the profile
/// dir.
pub const BOOKMARKS_FILE: &str = "bookmarks.json";

/// One bookmark: the committed URL, plus the title shown in the
/// bookmarks list (currently the URL itself — see the module
/// docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub url: String,
    pub title: String,
}

impl Bookmark {
    fn new(url: &str, title: &str) -> Self {
        Self {
            url: url.to_string(),
            title: title.to_string(),
        }
    }
}

/// The bookmark store: bookmarks in the order they were added.
/// Deduplicated by URL — one bookmark per URL, at most.
#[derive(Debug, Default, Clone)]
pub struct BookmarkStore {
    bookmarks: Vec<Bookmark>,
}

impl BookmarkStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// All bookmarks, in the order they were added.
    pub fn bookmarks(&self) -> &[Bookmark] {
        &self.bookmarks
    }

    /// Whether `url` is bookmarked.
    pub fn is_bookmarked(&self, url: &str) -> bool {
        self.bookmarks.iter().any(|b| b.url == url)
    }

    /// Add a bookmark. Returns false when nothing was added: an
    /// empty URL, or a URL that is already bookmarked (the
    /// existing entry is kept, in its original position).
    pub fn add(&mut self, url: &str, title: &str) -> bool {
        if url.is_empty() || self.is_bookmarked(url) {
            return false;
        }
        self.bookmarks.push(Bookmark::new(url, title));
        true
    }

    /// Remove the bookmark for `url`. Returns false when there
    /// was none.
    pub fn remove(&mut self, url: &str) -> bool {
        let before = self.bookmarks.len();
        self.bookmarks.retain(|b| b.url != url);
        self.bookmarks.len() != before
    }

    /// Toggle the bookmark for `url`: add it when absent, remove
    /// it when present. Returns true when the URL is bookmarked
    /// after the toggle (i.e. the star should now be filled).
    pub fn toggle(&mut self, url: &str, title: &str) -> bool {
        if self.is_bookmarked(url) {
            self.remove(url);
            false
        } else {
            self.add(url, title);
            self.is_bookmarked(url)
        }
    }

    /// Serialise the whole store (for the profile file).
    pub fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .bookmarks
            .iter()
            .map(|b| {
                format!(
                    "{{\"url\":{},\"title\":{}}}",
                    json_escape(&b.url),
                    json_escape(&b.title),
                )
            })
            .collect();
        format!("{{\"version\":1,\"bookmarks\":[{}]}}", parts.join(","))
    }

    /// Deserialise a profile file (corrupt input → empty store,
    /// logged). Entries without a usable URL are skipped.
    pub fn from_json(text: &str) -> Self {
        let parsed = match parse_json(text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[pluto] corrupt bookmarks profile ({e}); starting empty");
                return Self::new();
            }
        };
        let mut store = Self::new();
        let Some(items) = parsed.get("bookmarks").and_then(|v| v.as_array()) else {
            eprintln!("[pluto] bookmarks profile has no bookmarks; starting empty");
            return store;
        };
        for item in items {
            let Some(url) = item.get("url").and_then(|v| v.as_str()) else {
                continue;
            };
            if url.is_empty() {
                continue;
            }
            let title = item
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(url);
            store.bookmarks.push(Bookmark::new(url, title));
        }
        store
    }
}

/// Load the persisted bookmark store. Missing or corrupt files
/// fall back to an empty store (logged, not fatal).
pub fn load_bookmarks() -> BookmarkStore {
    let Some(dir) = profile_dir() else {
        eprintln!("[pluto] no HOME: bookmarks persistence disabled");
        return BookmarkStore::new();
    };
    match std::fs::read_to_string(dir.join(BOOKMARKS_FILE)) {
        Ok(text) => BookmarkStore::from_json(&text),
        Err(_) => {
            eprintln!(
                "[pluto] no persisted bookmarks yet ({})",
                dir.display()
            );
            BookmarkStore::new()
        }
    }
}

/// Save the bookmark store to the profile. Failures are logged,
/// never fatal.
pub fn save_bookmarks(store: &BookmarkStore) {
    let Some(dir) = profile_dir() else {
        return;
    };
    if let Err(e) = write_file_atomic(&dir, BOOKMARKS_FILE, &store.to_json()) {
        eprintln!("[pluto] could not save bookmarks: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pluto-bookmarks-test-{tag}"));
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

    #[test]
    fn toggle_round_trip() {
        let mut store = BookmarkStore::new();
        let url = "https://example.com/";
        assert!(!store.is_bookmarked(url));
        // First toggle: added, star filled.
        assert!(store.toggle(url, url));
        assert!(store.is_bookmarked(url));
        assert_eq!(store.bookmarks().len(), 1);
        // Second toggle: removed, star empty.
        assert!(!store.toggle(url, url));
        assert!(!store.is_bookmarked(url));
        assert!(store.bookmarks().is_empty());
        // Toggling twice more lands in the same states.
        assert!(store.toggle(url, url));
        assert!(!store.toggle(url, url));
    }

    #[test]
    fn add_is_deduped_and_empty_rejected() {
        let mut store = BookmarkStore::new();
        assert!(store.add("https://a.example/", "https://a.example/"));
        assert!(!store.add("https://a.example/", "https://a.example/"));
        assert_eq!(store.bookmarks().len(), 1);
        // The duplicate keeps its original position.
        assert!(store.add("https://b.example/", "https://b.example/"));
        assert!(!store.add("https://a.example/", "other title"));
        assert_eq!(store.bookmarks()[0].url, "https://a.example/");
        assert_eq!(store.bookmarks()[0].title, "https://a.example/");
        assert!(!store.add("", ""));
        assert!(!store.remove("https://missing.example/"));
        assert!(store.remove("https://a.example/"));
        assert_eq!(store.bookmarks().len(), 1);
    }

    #[test]
    fn bookmarks_persist_across_loads() {
        let home = temp_home("round-trip");
        let old = with_home(&home);
        {
            let mut store = load_bookmarks();
            store.add("https://example.com/", "https://example.com/");
            store.add(
                "https://quoted.example/a?b=\"c\"",
                "https://quoted.example/a?b=\"c\"",
            );
            save_bookmarks(&store);
        }
        let loaded = load_bookmarks();
        assert_eq!(loaded.bookmarks().len(), 2);
        assert!(loaded.is_bookmarked("https://example.com/"));
        assert!(loaded.is_bookmarked("https://quoted.example/a?b=\"c\""));
        // Insertion order survives the round-trip.
        assert_eq!(loaded.bookmarks()[0].url, "https://example.com/");
        // Removing one and saving persists the removal.
        let mut reloaded = loaded;
        assert!(reloaded.remove("https://example.com/"));
        save_bookmarks(&reloaded);
        let loaded2 = load_bookmarks();
        assert_eq!(loaded2.bookmarks().len(), 1);
        assert!(!loaded2.is_bookmarked("https://example.com/"));
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn corrupt_bookmarks_load_empty() {
        let home = temp_home("corrupt");
        let old = with_home(&home);
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(BOOKMARKS_FILE), "not json{{").unwrap();
        let loaded = load_bookmarks();
        assert!(loaded.bookmarks().is_empty());
        // A document with entries missing a URL skips them.
        let partial = BookmarkStore::from_json(
            "{\"version\":1,\"bookmarks\":[{\"title\":\"no url\"},{\"url\":\"https://ok.example/\"}]}",
        );
        assert_eq!(partial.bookmarks().len(), 1);
        assert_eq!(partial.bookmarks()[0].url, "https://ok.example/");
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }
}
