//! Persistent session history (Stage 10 Rec 8).
//!
//! The shell's session history (`history: Vec<String>` in
//! `PlutoBrowser`, pushed on every committed navigation) vanished
//! when Pluto closed. This persists it to the profile as
//! `history.json` next to `bookmarks.json` (`$HOME/.config/pluto/`,
//! see [`crate::security::profile_dir`]), following the
//! `BookmarkStore` pattern exactly: the same profile dir, the
//! same temp-file + rename write via
//! [`crate::js_host::storage::write_file_atomic`], the same
//! serde-free JSON, and the same corrupt-file behaviour (empty
//! history, logged, never fatal).
//!
//! The format is `{"version":1,"history":[url,…]}` — the
//! committed URLs in order, oldest first, exactly the shell's
//! in-memory order. Only the newest
//! [`crate::HISTORY_PANEL_LIMIT`] entries are persisted (the
//! same 50 the History panel shows): the in-memory list is
//! unbounded, but the panel caps at the newest 50, so anything
//! older would never be seen again anyway.
//!
//! This also makes Rec 3's `:visited` durable across restarts:
//! the worker seeds its `visited_urls` set from the loaded
//! history in `TabState::new` (see `renderer.rs`).

use crate::HISTORY_PANEL_LIMIT;
use crate::js_host::storage::{json_escape, parse_json, write_file_atomic};
use crate::security::profile_dir;

/// File name for the persisted history inside the profile dir.
pub const HISTORY_FILE: &str = "history.json";

/// Serialise the history for the profile file: the newest
/// [`HISTORY_PANEL_LIMIT`] entries, oldest first.
pub fn history_to_json(history: &[String]) -> String {
    let start = history.len().saturating_sub(HISTORY_PANEL_LIMIT);
    let parts: Vec<String> = history[start..]
        .iter()
        .map(|url| json_escape(url))
        .collect();
    format!("{{\"version\":1,\"history\":[{}]}}", parts.join(","))
}

/// Deserialise a profile file (corrupt input → empty history,
/// logged). Entries that are not strings are skipped.
pub fn history_from_json(text: &str) -> Vec<String> {
    let parsed = match parse_json(text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[pluto] corrupt history profile ({e}); starting empty");
            return Vec::new();
        }
    };
    let Some(items) = parsed.get("history").and_then(|v| v.as_array()) else {
        eprintln!("[pluto] history profile has no history; starting empty");
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .filter(|url| !url.is_empty())
        .collect()
}

/// Load the persisted history. Missing or corrupt files fall
/// back to an empty history (logged, not fatal).
pub fn load_history() -> Vec<String> {
    let Some(dir) = profile_dir() else {
        eprintln!("[pluto] no HOME: history persistence disabled");
        return Vec::new();
    };
    match std::fs::read_to_string(dir.join(HISTORY_FILE)) {
        Ok(text) => history_from_json(&text),
        Err(_) => {
            eprintln!("[pluto] no persisted history yet ({})", dir.display());
            Vec::new()
        }
    }
}

/// Save the history to the profile (newest
/// [`HISTORY_PANEL_LIMIT`] entries — see [`history_to_json`]).
/// Failures are logged, never fatal.
pub fn save_history(history: &[String]) {
    let Some(dir) = profile_dir() else {
        return;
    };
    if let Err(e) = write_file_atomic(&dir, HISTORY_FILE, &history_to_json(history)) {
        eprintln!("[pluto] could not save history: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pluto-history-test-{tag}"));
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
    fn history_round_trip_preserves_order() {
        let home = temp_home("round-trip");
        let old = with_home(&home);
        let history = vec![
            "https://example.com/".to_string(),
            "https://quoted.example/a?b=\"c\"".to_string(),
            "https://example.com/page#frag".to_string(),
        ];
        save_history(&history);
        let loaded = load_history();
        assert_eq!(loaded, history, "entries survive in order");
        // A later save overwrites: removing an entry persists.
        save_history(&history[..2]);
        let loaded2 = load_history();
        assert_eq!(loaded2.len(), 2);
        assert_eq!(loaded2[1], "https://quoted.example/a?b=\"c\"");
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn corrupt_history_loads_empty() {
        let home = temp_home("corrupt");
        let old = with_home(&home);
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(HISTORY_FILE), "not json{{").unwrap();
        let loaded = load_history();
        assert!(loaded.is_empty());
        // A document with a non-string entry skips it; a
        // document with no history key loads empty.
        let partial = history_from_json(
            "{\"version\":1,\"history\":[42,\"https://ok.example/\",null]}",
        );
        assert_eq!(partial, vec!["https://ok.example/".to_string()]);
        assert!(history_from_json("{\"version\":1}").is_empty());
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn persist_cap_keeps_newest_fifty() {
        let home = temp_home("cap");
        let old = with_home(&home);
        let history: Vec<String> = (0..60)
            .map(|i| format!("https://example.com/{i}"))
            .collect();
        save_history(&history);
        let loaded = load_history();
        assert_eq!(loaded.len(), HISTORY_PANEL_LIMIT);
        assert_eq!(loaded[0], "https://example.com/10");
        assert_eq!(loaded[HISTORY_PANEL_LIMIT - 1], "https://example.com/59");
        // The in-memory list stays uncapped — only the
        // persisted copy is trimmed.
        assert_eq!(history.len(), 60);
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }
}
