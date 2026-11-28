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
//!
//! Follow-up UI work (2026-11-27): a bookmark may carry a custom
//! favicon, picked in the bookmark dialog. The image bytes live
//! OUTSIDE `bookmarks.json` — one file per bookmark under
//! `bookmark-favicons/` in the profile dir, named by
//! [`bookmark_favicon_file_name`] — and `bookmarks.json` carries
//! only the file name, as an optional `"favicon"` key. Old files
//! without the key load as favicon-less bookmarks, so the format
//! stays backward compatible at version 1.

use crate::atomic::write_file_atomic_bytes;
use crate::js_host::storage::{json_escape, parse_json, write_file_atomic};
use crate::security::profile_dir;

/// File name for the persisted bookmark store inside the profile
/// dir.
pub const BOOKMARKS_FILE: &str = "bookmarks.json";

/// Subdirectory of the profile dir holding custom favicon files,
/// one per bookmarked URL (see [`bookmark_favicon_file_name`]).
pub const BOOKMARK_FAVICON_DIR: &str = "bookmark-favicons";

/// The custom favicon's file name for `url`: the first 16 hex
/// characters of the URL's SHA-256. Deterministic, so the file
/// can be found again from the URL alone (no path stored per
/// bookmark), and unique per bookmark because the store holds
/// one bookmark per URL.
pub fn bookmark_favicon_file_name(url: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, url.as_bytes());
    digest.as_ref()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Persist `bytes` as the custom favicon for `url` (atomic
/// write, like the rest of the profile). Returns the stored file
/// name on success.
pub fn save_bookmark_favicon(url: &str, bytes: &[u8]) -> std::io::Result<String> {
    let Some(dir) = profile_dir() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no HOME: no profile dir for the custom favicon",
        ));
    };
    let name = bookmark_favicon_file_name(url);
    write_file_atomic_bytes(&dir.join(BOOKMARK_FAVICON_DIR), &name, bytes)?;
    Ok(name)
}

/// Read back the custom favicon bytes for `url`, if one was saved.
pub fn load_bookmark_favicon(url: &str) -> Option<Vec<u8>> {
    let dir = profile_dir()?;
    std::fs::read(
        dir.join(BOOKMARK_FAVICON_DIR)
            .join(bookmark_favicon_file_name(url)),
    )
    .ok()
}

/// Delete the custom favicon file for `url`, if any. Best-effort:
/// a missing file is not an error (the bookmark may never have
/// had one).
pub fn delete_bookmark_favicon(url: &str) {
    if let Some(dir) = profile_dir() {
        let _ = std::fs::remove_file(
            dir.join(BOOKMARK_FAVICON_DIR)
                .join(bookmark_favicon_file_name(url)),
        );
    }
}

/// One bookmark: the committed URL, plus the title shown in the
/// bookmarks list (currently the URL itself — see the module
/// docs), plus the custom favicon's file name in
/// [`BOOKMARK_FAVICON_DIR`], if the user picked one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub url: String,
    pub title: String,
    pub favicon: Option<String>,
}

impl Bookmark {
    fn new_with_favicon(url: &str, title: &str, favicon: Option<String>) -> Self {
        Self {
            url: url.to_string(),
            title: title.to_string(),
            favicon,
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
        self.add_with_favicon(url, title, None)
    }

    /// Add a bookmark with a custom favicon file name (from
    /// [`save_bookmark_favicon`]). Same dedup/empty rules as
    /// [`add`](Self::add).
    pub fn add_with_favicon(
        &mut self,
        url: &str,
        title: &str,
        favicon: Option<String>,
    ) -> bool {
        if url.is_empty() || self.is_bookmarked(url) {
            return false;
        }
        self.bookmarks
            .push(Bookmark::new_with_favicon(url, title, favicon));
        true
    }

    /// Change the title of the bookmark for `url`, in place
    /// (position kept). Returns false when there is no such
    /// bookmark.
    pub fn set_title(&mut self, url: &str, title: &str) -> bool {
        let Some(bookmark) = self.bookmarks.iter_mut().find(|b| b.url == url) else {
            return false;
        };
        bookmark.title = title.to_string();
        true
    }

    /// Change the custom favicon of the bookmark for `url`
    /// (`Some(file)` to set one, `None` to clear it), in place.
    /// Returns false when there is no such bookmark.
    pub fn set_favicon(&mut self, url: &str, favicon: Option<String>) -> bool {
        let Some(bookmark) = self.bookmarks.iter_mut().find(|b| b.url == url) else {
            return false;
        };
        bookmark.favicon = favicon;
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

    /// Serialise the whole store (for the profile file). A
    /// bookmark with a custom favicon carries its file name as an
    /// optional `"favicon"` key; bookmarks without one omit it,
    /// so old readers (and old files) keep working.
    pub fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .bookmarks
            .iter()
            .map(|b| {
                let mut fields = vec![
                    format!("\"url\":{}", json_escape(&b.url)),
                    format!("\"title\":{}", json_escape(&b.title)),
                ];
                if let Some(favicon) = &b.favicon {
                    fields.push(format!("\"favicon\":{}", json_escape(favicon)));
                }
                format!("{{{}}}", fields.join(","))
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
            let favicon = item
                .get("favicon")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            store
                .bookmarks
                .push(Bookmark::new_with_favicon(url, title, favicon));
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

    #[test]
    fn favicon_file_name_is_deterministic_and_url_scoped() {
        let name = bookmark_favicon_file_name("https://example.com/");
        // 8 bytes of SHA-256 as hex.
        assert_eq!(name.len(), 16);
        assert!(name.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(name, bookmark_favicon_file_name("https://example.com/"));
        assert_ne!(name, bookmark_favicon_file_name("https://other.example/"));
        assert_ne!(
            name,
            bookmark_favicon_file_name("https://example.com/other")
        );
    }

    #[test]
    fn favicon_json_round_trip_and_old_files_stay_compatible() {
        let mut store = BookmarkStore::new();
        assert!(store.add_with_favicon(
            "https://a.example/",
            "A",
            Some("deadbeef01234567".to_string()),
        ));
        assert!(store.add("https://b.example/", "B"));
        let json = store.to_json();
        assert!(json.contains("\"favicon\":\"deadbeef01234567\""));
        let loaded = BookmarkStore::from_json(&json);
        assert_eq!(loaded.bookmarks().len(), 2);
        assert_eq!(
            loaded.bookmarks()[0].favicon.as_deref(),
            Some("deadbeef01234567")
        );
        assert_eq!(loaded.bookmarks()[1].favicon, None);
        // An old file with no favicon key at all loads as
        // favicon-less (version 1, backward compatible).
        let old = BookmarkStore::from_json(
            "{\"version\":1,\"bookmarks\":[{\"url\":\"https://old.example/\",\"title\":\"Old\"}]}",
        );
        assert_eq!(old.bookmarks().len(), 1);
        assert_eq!(old.bookmarks()[0].favicon, None);
    }

    #[test]
    fn set_title_and_favicon_keep_position() {
        let mut store = BookmarkStore::new();
        store.add("https://a.example/", "A");
        store.add("https://b.example/", "B");
        assert!(store.set_title("https://b.example/", "Bee"));
        assert!(store.set_favicon("https://b.example/", Some("cafef00d".to_string())));
        assert_eq!(store.bookmarks()[1].title, "Bee");
        assert_eq!(
            store.bookmarks()[1].favicon.as_deref(),
            Some("cafef00d")
        );
        // Clearing the favicon and unknown URLs.
        assert!(store.set_favicon("https://b.example/", None));
        assert_eq!(store.bookmarks()[1].favicon, None);
        assert!(!store.set_title("https://missing.example/", "X"));
        assert!(!store.set_favicon("https://missing.example/", None));
        // Order untouched by the edits.
        assert_eq!(store.bookmarks()[0].url, "https://a.example/");
        assert_eq!(store.bookmarks()[1].url, "https://b.example/");
    }

    #[test]
    fn custom_favicon_file_round_trip() {
        let home = temp_home("favicon");
        let old = with_home(&home);
        let url = "https://example.com/";
        let bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes";
        assert_eq!(load_bookmark_favicon(url), None);
        let name = save_bookmark_favicon(url, bytes).unwrap();
        assert_eq!(name, bookmark_favicon_file_name(url));
        assert_eq!(load_bookmark_favicon(url).as_deref(), Some(&bytes[..]));
        delete_bookmark_favicon(url);
        assert_eq!(load_bookmark_favicon(url), None);
        // Deleting again is not an error.
        delete_bookmark_favicon(url);
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }
}
