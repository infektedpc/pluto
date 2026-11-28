//! Persistent bookmarks (nav feature, 2026-11-24).
//!
//! The bookmark star in the nav bar needs a real backend, so this
//! is one: a minimal store of bookmarked URLs, persisted to the
//! profile as `bookmarks.json` next to the other profile files
//! (`$HOME/.config/pluto/`, see [`crate::security::profile_dir`]).
//!
//! The format follows the existing profile pattern (serde-free
//! JSON via [`crate::js_host::storage`], temp-file + rename
//! write): version 2, `{"version":2,"entries":[…]}`, where an
//! entry is `{"type":"bookmark","url":…,"title":…}` (with
//! optional `"favicon"`, `"tags"`, `"keyword"` keys) or
//! `{"type":"folder","id":…,"name":…,"bookmarks":[…]}`.
//! Version 1 files (`{"version":1,"bookmarks":[…]}`) load as
//! top-level bookmarks, so old profiles keep working. A missing
//! or corrupt file loads as an empty store (logged, never
//! fatal), like the other profile stores.
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
use crate::js_host::storage::{JsonValue, json_escape, parse_json, write_file_atomic};
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
/// bookmarks list, plus the custom favicon's file name in
/// [`BOOKMARK_FAVICON_DIR`], if the user picked one — plus the
/// tags and keyword from the Firefox-style dialog (bookmarks
/// round, 2026-11-28): tags organise and search, a keyword opens
/// the bookmark straight from the address bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub url: String,
    pub title: String,
    pub favicon: Option<String>,
    /// Free-form tags, in the order the user typed them.
    pub tags: Vec<String>,
    /// A single keyword that opens this bookmark from the
    /// address bar (`None` = no keyword).
    pub keyword: Option<String>,
}

impl Bookmark {
    fn new_full(
        url: &str,
        title: &str,
        favicon: Option<String>,
        tags: Vec<String>,
        keyword: Option<String>,
    ) -> Self {
        Self {
            url: url.to_string(),
            title: title.to_string(),
            favicon,
            tags,
            keyword,
        }
    }
}

/// One bookmark folder: a named group of bookmarks, shown as a
/// chip in the bookmarks bar (bookmarks round, 2026-11-28).
/// Folders are top-level only — no nesting — and hold plain
/// bookmarks, never other folders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookmarkFolder {
    /// Stable id (`folder-1`, `folder-2`, …), unique in the
    /// store; the bar's open-folder state keys on it.
    pub id: String,
    pub name: String,
    pub bookmarks: Vec<Bookmark>,
}

/// One entry in the bookmarks bar's ordered list: a bookmark or
/// a folder, in the order they were added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookmarkEntry {
    Bookmark(Bookmark),
    Folder(BookmarkFolder),
}

/// The bookmark store: the bar's ordered entries (bookmarks
/// and folders, in the order they were added). Deduplicated by
/// URL — one bookmark per URL, at most, anywhere in the store
/// (a bookmark inside a folder still counts).
#[derive(Debug, Clone)]
pub struct BookmarkStore {
    entries: Vec<BookmarkEntry>,
    next_folder_id: u64,
}

impl Default for BookmarkStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BookmarkStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_folder_id: 1,
        }
    }

    /// The bar's entries, in the order they were added.
    pub fn entries(&self) -> &[BookmarkEntry] {
        &self.entries
    }

    /// All bookmarks, in bar order — top-level ones plus the
    /// contents of every folder. (A flat view: the list panel
    /// and the keyword search read this, not the entries.)
    pub fn bookmarks(&self) -> Vec<&Bookmark> {
        let mut out = Vec::new();
        for entry in &self.entries {
            match entry {
                BookmarkEntry::Bookmark(bookmark) => out.push(bookmark),
                BookmarkEntry::Folder(folder) => out.extend(folder.bookmarks.iter()),
            }
        }
        out
    }

    /// The bookmark for `url`, wherever it lives (top-level or
    /// inside a folder).
    pub fn bookmark(&self, url: &str) -> Option<&Bookmark> {
        for entry in &self.entries {
            match entry {
                BookmarkEntry::Bookmark(bookmark) if bookmark.url == url => {
                    return Some(bookmark);
                }
                BookmarkEntry::Folder(folder) => {
                    if let Some(bookmark) =
                        folder.bookmarks.iter().find(|b| b.url == url)
                    {
                        return Some(bookmark);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The bookmark for `url`, mutable — the setters go through
    /// this so an edit lands wherever the bookmark lives.
    fn bookmark_mut(&mut self, url: &str) -> Option<&mut Bookmark> {
        for entry in &mut self.entries {
            match entry {
                BookmarkEntry::Bookmark(bookmark) if bookmark.url == url => {
                    return Some(bookmark);
                }
                BookmarkEntry::Folder(folder) => {
                    if let Some(bookmark) =
                        folder.bookmarks.iter_mut().find(|b| b.url == url)
                    {
                        return Some(bookmark);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Whether `url` is bookmarked, anywhere in the store.
    pub fn is_bookmarked(&self, url: &str) -> bool {
        self.bookmark(url).is_some()
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
        self.add_full(url, title, favicon, Vec::new(), None)
    }

    /// Add a bookmark with everything the dialog holds (favicon,
    /// tags, keyword). Same dedup/empty rules as [`add`](Self::add).
    pub fn add_full(
        &mut self,
        url: &str,
        title: &str,
        favicon: Option<String>,
        tags: Vec<String>,
        keyword: Option<String>,
    ) -> bool {
        if url.is_empty() || self.is_bookmarked(url) {
            return false;
        }
        self.entries.push(BookmarkEntry::Bookmark(
            Bookmark::new_full(url, title, favicon, tags, keyword),
        ));
        true
    }

    /// Change the title of the bookmark for `url`, in place
    /// (position kept). Returns false when there is no such
    /// bookmark.
    pub fn set_title(&mut self, url: &str, title: &str) -> bool {
        let Some(bookmark) = self.bookmark_mut(url) else {
            return false;
        };
        bookmark.title = title.to_string();
        true
    }

    /// Change the custom favicon of the bookmark for `url`
    /// (`Some(file)` to set one, `None` to clear it), in place.
    /// Returns false when there is no such bookmark.
    pub fn set_favicon(&mut self, url: &str, favicon: Option<String>) -> bool {
        let Some(bookmark) = self.bookmark_mut(url) else {
            return false;
        };
        bookmark.favicon = favicon;
        true
    }

    /// Change the tags of the bookmark for `url`, in place.
    /// Returns false when there is no such bookmark.
    pub fn set_tags(&mut self, url: &str, tags: Vec<String>) -> bool {
        let Some(bookmark) = self.bookmark_mut(url) else {
            return false;
        };
        bookmark.tags = tags;
        true
    }

    /// Change the keyword of the bookmark for `url`, in place
    /// (an empty keyword clears it). Returns false when there is
    /// no such bookmark.
    pub fn set_keyword(&mut self, url: &str, keyword: Option<String>) -> bool {
        let Some(bookmark) = self.bookmark_mut(url) else {
            return false;
        };
        bookmark.keyword = keyword.filter(|k| !k.trim().is_empty());
        true
    }

    /// Change the URL of the bookmark for `old`, in place
    /// (position kept). Returns false when there is no such
    /// bookmark, when `new` is empty, or when `new` is already
    /// bookmarked under a different entry.
    pub fn set_url(&mut self, old: &str, new: &str) -> bool {
        if new.is_empty() || (new != old && self.is_bookmarked(new)) {
            return false;
        }
        let Some(bookmark) = self.bookmark_mut(old) else {
            return false;
        };
        bookmark.url = new.to_string();
        true
    }

    /// Remove the bookmark for `url`, wherever it lives
    /// (top-level or inside a folder). Returns false when there
    /// was none. An emptied folder stays — it shows "Empty
    /// folder" in its popup.
    pub fn remove(&mut self, url: &str) -> bool {
        let before = self.bookmarks().len();
        for entry in &mut self.entries {
            if let BookmarkEntry::Folder(folder) = entry {
                folder.bookmarks.retain(|b| b.url != url);
            }
        }
        self.entries
            .retain(|e| !matches!(e, BookmarkEntry::Bookmark(b) if b.url == url));
        self.bookmarks().len() != before
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

    /// Add a folder with `name` (empty names become "New
    /// folder"). Returns the new folder's id.
    pub fn add_folder(&mut self, name: &str) -> String {
        let id = format!("folder-{}", self.next_folder_id);
        self.next_folder_id += 1;
        let name = if name.trim().is_empty() {
            "New folder".to_string()
        } else {
            name.trim().to_string()
        };
        self.entries.push(BookmarkEntry::Folder(BookmarkFolder {
            id: id.clone(),
            name,
            bookmarks: Vec::new(),
        }));
        id
    }

    /// The folder with `id`, if any.
    pub fn folder(&self, id: &str) -> Option<&BookmarkFolder> {
        self.entries.iter().find_map(|entry| match entry {
            BookmarkEntry::Folder(folder) if folder.id == id => Some(folder),
            _ => None,
        })
    }

    /// Remove the folder with `id` and every bookmark inside
    /// it. Returns false when there was no such folder.
    pub fn remove_folder(&mut self, id: &str) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|e| !matches!(e, BookmarkEntry::Folder(f) if f.id == id));
        self.entries.len() != before
    }

    /// Add a bookmark into the folder with `folder_id`. Same
    /// dedup/empty rules as [`add`](Self::add), and false when
    /// the folder does not exist.
    pub fn add_to_folder(&mut self, folder_id: &str, url: &str, title: &str) -> bool {
        self.add_to_folder_full(folder_id, url, title, None, Vec::new(), None)
    }

    /// Add a bookmark into a folder with everything the dialog
    /// holds (favicon, tags, keyword).
    pub fn add_to_folder_full(
        &mut self,
        folder_id: &str,
        url: &str,
        title: &str,
        favicon: Option<String>,
        tags: Vec<String>,
        keyword: Option<String>,
    ) -> bool {
        if url.is_empty() || self.is_bookmarked(url) {
            return false;
        }
        let Some(folder) = self.entries.iter_mut().find_map(|entry| match entry {
            BookmarkEntry::Folder(folder) if folder.id == folder_id => Some(folder),
            _ => None,
        }) else {
            return false;
        };
        folder
            .bookmarks
            .push(Bookmark::new_full(url, title, favicon, tags, keyword));
        true
    }

    /// The bookmark whose keyword is `input` (the whole input,
    /// trimmed, case-insensitive — a keyword is typed from
    /// memory, so case must not matter). Empty input never
    /// matches; a bookmark with no keyword never matches. First
    /// match wins.
    pub fn find_by_keyword(&self, input: &str) -> Option<&Bookmark> {
        let input = input.trim();
        if input.is_empty() {
            return None;
        }
        self.bookmarks().into_iter().find(|bookmark| {
            bookmark
                .keyword
                .as_deref()
                .is_some_and(|keyword| keyword.eq_ignore_ascii_case(input))
        })
    }

    /// Every tag on every bookmark, deduplicated and sorted —
    /// the tag chevron in the bookmark dialog lists these.
    pub fn all_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .bookmarks()
            .iter()
            .flat_map(|bookmark| bookmark.tags.iter().cloned())
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// Serialise the whole store (for the profile file).
    /// Optional keys are omitted when empty — a bookmark
    /// without a custom favicon, tags, or keyword carries no
    /// `"favicon"` / `"tags"` / `"keyword"` key — so old readers
    /// (and old files) keep working.
    pub fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .entries
            .iter()
            .map(|entry| match entry {
                BookmarkEntry::Bookmark(bookmark) => bookmark_json(bookmark),
                BookmarkEntry::Folder(folder) => {
                    let inner: Vec<String> =
                        folder.bookmarks.iter().map(bookmark_json).collect();
                    format!(
                        "{{\"type\":\"folder\",\"id\":{},\"name\":{},\"bookmarks\":[{}]}}",
                        json_escape(&folder.id),
                        json_escape(&folder.name),
                        inner.join(",")
                    )
                }
            })
            .collect();
        format!("{{\"version\":2,\"entries\":[{}]}}", parts.join(","))
    }

    /// Deserialise a profile file (corrupt input → empty store,
    /// logged). Version 2 reads the `"entries"` array
    /// (bookmarks and folders); version 1 reads the flat
    /// `"bookmarks"` array as top-level bookmarks. Entries
    /// without a usable URL are skipped, as are duplicate URLs
    /// and duplicate folder ids (first one wins).
    pub fn from_json(text: &str) -> Self {
        let parsed = match parse_json(text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[pluto] corrupt bookmarks profile ({e}); starting empty");
                return Self::new();
            }
        };
        let mut store = Self::new();
        if let Some(items) = parsed.get("entries").and_then(|v| v.as_array()) {
            for item in items {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("folder") => {
                        let Some(id) = item.get("id").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        if id.is_empty() || store.folder(id).is_some() {
                            continue;
                        }
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("New folder");
                        let mut folder = BookmarkFolder {
                            id: id.to_string(),
                            name: name.to_string(),
                            bookmarks: Vec::new(),
                        };
                        if let Some(inner) =
                            item.get("bookmarks").and_then(|v| v.as_array())
                        {
                            for child in inner {
                                if let Some(bookmark) = bookmark_from_json(child)
                                    && !folder
                                        .bookmarks
                                        .iter()
                                        .any(|b| b.url == bookmark.url)
                                    && !store.is_bookmarked(&bookmark.url)
                                {
                                    folder.bookmarks.push(bookmark);
                                }
                            }
                        }
                        store.entries.push(BookmarkEntry::Folder(folder));
                    }
                    // "bookmark", or a missing type: a plain
                    // bookmark object.
                    _ => {
                        if let Some(bookmark) = bookmark_from_json(item)
                            && !store.is_bookmarked(&bookmark.url)
                        {
                            store.entries.push(BookmarkEntry::Bookmark(bookmark));
                        }
                    }
                }
            }
            store.refresh_next_folder_id();
            return store;
        }
        let Some(items) = parsed.get("bookmarks").and_then(|v| v.as_array()) else {
            eprintln!("[pluto] bookmarks profile has no bookmarks; starting empty");
            return store;
        };
        for item in items {
            if let Some(bookmark) = bookmark_from_json(item)
                && !store.is_bookmarked(&bookmark.url)
            {
                store.entries.push(BookmarkEntry::Bookmark(bookmark));
            }
        }
        store
    }

    /// After loading, point the folder-id counter past the
    /// highest `folder-N` id in the store, so a new folder never
    /// reuses an id.
    fn refresh_next_folder_id(&mut self) {
        let mut max = 0u64;
        for entry in &self.entries {
            if let BookmarkEntry::Folder(folder) = entry
                && let Some(number) = folder
                    .id
                    .strip_prefix("folder-")
                    .and_then(|suffix| suffix.parse::<u64>().ok())
            {
                max = max.max(number);
            }
        }
        self.next_folder_id = max + 1;
    }
}

/// One bookmark object as a JSON object literal (shared by
/// top-level bookmarks, folder contents, and folder entries).
fn bookmark_json(bookmark: &Bookmark) -> String {
    let mut fields = vec![
        "\"type\":\"bookmark\"".to_string(),
        format!("\"url\":{}", json_escape(&bookmark.url)),
        format!("\"title\":{}", json_escape(&bookmark.title)),
    ];
    if let Some(favicon) = &bookmark.favicon {
        fields.push(format!("\"favicon\":{}", json_escape(favicon)));
    }
    if !bookmark.tags.is_empty() {
        let tags: Vec<String> = bookmark.tags.iter().map(|tag| json_escape(tag)).collect();
        fields.push(format!("\"tags\":[{}]", tags.join(",")));
    }
    if let Some(keyword) = &bookmark.keyword {
        fields.push(format!("\"keyword\":{}", json_escape(keyword)));
    }
    format!("{{{}}}", fields.join(","))
}

/// One bookmark object from a parsed JSON value: a usable URL is
/// required; the title defaults to the URL, and the favicon,
/// tags, and keyword keys are optional (missing = none). Tags
/// are trimmed, emptied, and deduped, preserving first-seen
/// order; an empty keyword is none.
fn bookmark_from_json(item: &JsonValue) -> Option<Bookmark> {
    let url = item.get("url").and_then(|v| v.as_str())?;
    if url.is_empty() {
        return None;
    }
    let title = item
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or(url);
    let favicon = item
        .get("favicon")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let mut tags: Vec<String> = Vec::new();
    if let Some(items) = item.get("tags").and_then(|v| v.as_array()) {
        for value in items {
            if let Some(tag) = value.as_str() {
                let tag = tag.trim().to_string();
                if !tag.is_empty() && !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
        }
    }
    let keyword = item
        .get("keyword")
        .and_then(|v| v.as_str())
        .map(|keyword| keyword.trim().to_string())
        .filter(|keyword| !keyword.is_empty());
    Some(Bookmark {
        url: url.to_string(),
        title: title.to_string(),
        favicon,
        tags,
        keyword,
    })
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

    #[test]
    fn folders_hold_bookmarks_and_remove_takes_contents() {
        let mut store = BookmarkStore::new();
        assert!(store.add("https://top.example/", "Top"));
        let id = store.add_folder("News");
        assert!(id.starts_with("folder-"));
        assert!(store.add_to_folder(&id, "https://news.example/", "News Site"));
        // A bookmark inside a folder still counts as bookmarked,
        // and the bar order is insertion order.
        assert!(store.is_bookmarked("https://news.example/"));
        assert_eq!(store.bookmarks().len(), 2);
        assert_eq!(store.entries().len(), 2);
        assert!(matches!(
            &store.entries()[1],
            BookmarkEntry::Folder(folder) if folder.name == "News"
        ));
        // Removing by URL finds it inside the folder; the folder
        // stays, now empty.
        assert!(store.remove("https://news.example/"));
        assert!(!store.is_bookmarked("https://news.example/"));
        assert!(store.folder(&id).unwrap().bookmarks.is_empty());
        // Removing the folder takes its contents with it.
        assert!(store.add_to_folder(&id, "https://news2.example/", "News 2"));
        assert!(store.remove_folder(&id));
        assert!(!store.is_bookmarked("https://news2.example/"));
        assert!(store.folder(&id).is_none());
        assert!(!store.remove_folder(&id));
        // Unknown folders and duplicate URLs are rejected.
        assert!(!store.add_to_folder("folder-999", "https://x.example/", "X"));
        let dup_folder = store.add_folder("");
        assert!(!store.add_to_folder(&dup_folder, "https://top.example/", "Dup"));
        // An empty folder name becomes "New folder".
        let empty_id = store.add_folder("   ");
        assert_eq!(store.folder(&empty_id).unwrap().name, "New folder");
    }

    #[test]
    fn folders_and_tags_and_keywords_round_trip_and_v1_loads() {
        let mut store = BookmarkStore::new();
        assert!(store.add_full(
            "https://a.example/",
            "A",
            Some("deadbeef01234567".to_string()),
            vec!["rust".to_string(), "news".to_string()],
            Some("ra".to_string()),
        ));
        let folder = store.add_folder("Read");
        assert!(store.add_to_folder_full(
            &folder,
            "https://b.example/",
            "B",
            None,
            vec!["rust".to_string()],
            None,
        ));
        let json = store.to_json();
        assert!(json.contains("\"version\":2"));
        assert!(json.contains("\"type\":\"folder\""));
        assert!(json.contains("\"tags\":[\"rust\",\"news\"]"));
        assert!(json.contains("\"keyword\":\"ra\""));
        let loaded = BookmarkStore::from_json(&json);
        assert_eq!(loaded.bookmarks().len(), 2);
        assert_eq!(
            loaded.bookmarks()[0].tags,
            vec!["rust".to_string(), "news".to_string()]
        );
        assert_eq!(loaded.bookmarks()[0].keyword.as_deref(), Some("ra"));
        assert_eq!(
            loaded.bookmarks()[0].favicon.as_deref(),
            Some("deadbeef01234567")
        );
        assert_eq!(loaded.bookmarks()[1].keyword, None);
        assert_eq!(
            loaded.bookmarks()[1].tags,
            vec!["rust".to_string()]
        );
        let loaded_folder = loaded.folder(&folder).expect("folder survives");
        assert_eq!(loaded_folder.name, "Read");
        assert_eq!(loaded_folder.bookmarks.len(), 1);
        // A new folder after a load never reuses an id.
        let mut reloaded = loaded;
        let next = reloaded.add_folder("More");
        assert_ne!(next, folder);
        // A version-1 flat file loads as top-level bookmarks,
        // with no tags or keyword.
        let old = BookmarkStore::from_json(
            "{\"version\":1,\"bookmarks\":[{\"url\":\"https://old.example/\",\"title\":\"Old\",\"favicon\":\"abc\"}]}",
        );
        assert_eq!(old.bookmarks().len(), 1);
        assert_eq!(old.bookmarks()[0].title, "Old");
        assert_eq!(old.bookmarks()[0].favicon.as_deref(), Some("abc"));
        assert!(old.bookmarks()[0].tags.is_empty());
        assert_eq!(old.bookmarks()[0].keyword, None);
        assert!(old.entries().iter().all(|e| matches!(
            e,
            BookmarkEntry::Bookmark(_)
        )));
    }

    #[test]
    fn keyword_lookup_is_case_insensitive_and_exact() {
        let mut store = BookmarkStore::new();
        assert!(store.add_full(
            "https://github.com/",
            "GitHub",
            None,
            Vec::new(),
            Some("gh".to_string()),
        ));
        assert_eq!(
            store.find_by_keyword("gh").map(|b| b.url.as_str()),
            Some("https://github.com/")
        );
        assert_eq!(
            store.find_by_keyword("GH").map(|b| b.url.as_str()),
            Some("https://github.com/")
        );
        assert_eq!(
            store.find_by_keyword("  gh  ").map(|b| b.url.as_str()),
            Some("https://github.com/")
        );
        // Partial matches, other words, and empty input never match.
        assert!(store.find_by_keyword("g").is_none());
        assert!(store.find_by_keyword("ghx").is_none());
        assert!(store.find_by_keyword("github").is_none());
        assert!(store.find_by_keyword("").is_none());
        assert!(store.find_by_keyword("   ").is_none());
        // A bookmark without a keyword never matches, and an
        // empty keyword set on one clears it.
        assert!(store.add("https://plain.example/", "Plain"));
        assert!(store.find_by_keyword("plain").is_none());
        assert!(store.set_keyword("https://github.com/", Some("hub".to_string())));
        assert_eq!(
            store.find_by_keyword("hub").map(|b| b.url.as_str()),
            Some("https://github.com/")
        );
        assert!(store.find_by_keyword("gh").is_none());
        assert!(store.set_keyword("https://github.com/", Some("   ".to_string())));
        assert_eq!(store.bookmark("https://github.com/").unwrap().keyword, None);
    }

    #[test]
    fn all_tags_dedupes_and_sorts() {
        let mut store = BookmarkStore::new();
        assert!(store.add_full(
            "https://a.example/",
            "A",
            None,
            vec!["rust".to_string(), "news".to_string()],
            None,
        ));
        let folder = store.add_folder("F");
        assert!(store.add_to_folder_full(
            &folder,
            "https://b.example/",
            "B",
            None,
            vec!["news".to_string(), "docs".to_string()],
            None,
        ));
        assert_eq!(
            store.all_tags(),
            vec!["docs".to_string(), "news".to_string(), "rust".to_string()]
        );
        // Setting tags replaces them wholesale.
        assert!(store.set_tags(
            "https://a.example/",
            vec!["docs".to_string()]
        ));
        assert_eq!(
            store.all_tags(),
            vec!["docs".to_string(), "news".to_string()]
        );
    }

    #[test]
    fn set_url_moves_bookmark_and_refuses_taken_urls() {
        let mut store = BookmarkStore::new();
        assert!(store.add("https://a.example/", "A"));
        assert!(store.add("https://b.example/", "B"));
        assert!(store.set_url("https://a.example/", "https://a2.example/"));
        assert!(store.is_bookmarked("https://a2.example/"));
        assert!(!store.is_bookmarked("https://a.example/"));
        assert_eq!(store.bookmarks()[0].url, "https://a2.example/");
        // Taken, empty, and unknown URLs are refused.
        assert!(!store.set_url("https://a2.example/", "https://b.example/"));
        assert!(!store.set_url("https://a2.example/", ""));
        assert!(!store.set_url("https://missing.example/", "https://x.example/"));
        assert_eq!(store.bookmarks()[0].url, "https://a2.example/");
    }
}
