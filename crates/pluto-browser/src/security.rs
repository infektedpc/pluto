//! Browser-profile security state (Stage 9 Recommendation 5).
//!
//! Pluto had no profile/state persistence pattern before this
//! module — nothing on disk was ever read back — so this sets the
//! precedent: serde JSON files under the profile dir,
//! `$HOME/.config/pluto/`:
//!
//! - `hsts.json` — the HSTS store ([`pluto_net::HstsStore`]).
//! - `https-only.json` — the HTTPS-only preference + per-site
//!   exception store ([`pluto_net::HttpsOnly`]).
//!
//! The HSTS store and the HTTPS-only preference live in
//! `pluto-net` (the fetch pipeline consults them); this module
//! only owns their life on disk and the hand-off to a
//! [`pluto_net::ResourceLoader`]: [`load_into`] at startup,
//! [`save_from`] after each navigation completes (HSTS entries
//! are recorded during fetches, so saving after the navigation
//! captures them; the files are small and the write is a
//! temp-file + rename).
//!
//! HTTPS-only failure behaviour: a top-level navigation whose
//! HTTPS-only upgrade fails shows the existing error page
//! (`page::error_document`, via the normal `LoadOutcome::Failed`
//! path) with a reason naming HTTPS-only and the per-site
//! exception; subresource failures are quietly blocked and logged
//! by the load path, as before. There is no interstitial
//! "continue to HTTP" UI in this milestone — the escape hatch is
//! the exception store below. The API for it is
//! [`add_https_only_exception`] / [`remove_https_only_exception`]
//! (live loader + profile file, together); a settings UI that
//! calls them is deferred work, not done here.
//!
//! A corrupt or missing profile file never blocks startup: the
//! load falls back to defaults (empty HSTS store, HTTPS-only on,
//! no exceptions) and logs to stderr.

use std::path::PathBuf;

use pluto_net::{HstsStore, HttpsOnly, ResourceLoader};

/// File names inside the profile dir.
pub const HSTS_FILE: &str = "hsts.json";
pub const HTTPS_ONLY_FILE: &str = "https-only.json";

/// The profile directory: `$HOME/.config/pluto`. `None` when
/// `HOME` is unset — persistence is then skipped (in-memory
/// state still works for the session).
pub fn profile_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".config").join("pluto"))
}

fn read_file(dir: &PathBuf, name: &str) -> Option<String> {
    std::fs::read_to_string(dir.join(name)).ok()
}

/// Write a file atomically (temp-file + rename), creating the
/// profile dir if needed.
///
/// The temp name is unique per write (process id + a
/// per-process counter, created with `O_EXCL` and retried on
/// collision): two writers to the same profile dir — two Pluto
/// windows sharing `$HOME/.config/pluto`, or two test threads —
/// must never share one temp file, or their writes interleave
/// and the renamed target ends up as one JSON value plus
/// trailing bytes (Stage 10 Rec 6). The rename still publishes
/// each write whole, so a reader only ever sees a complete
/// document.
fn write_file_atomic(dir: &PathBuf, name: &str, contents: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let target = dir.join(name);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut attempt = 0;
    loop {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = dir.join(format!(".{name}.{}.{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(mut file) => {
                use std::io::Write;
                let written = file.write_all(contents.as_bytes());
                drop(file);
                let result = written.and_then(|()| std::fs::rename(&tmp, target));
                if result.is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
                return result;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // A stale temp file with this pid + counter (a pid
                // reused after a crash, or a counter wrap): take
                // the next name instead of clobbering it.
                attempt += 1;
                if attempt > 100 {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// Load the persisted security state into `loader`. Missing or
/// corrupt files fall back to defaults (logged, not fatal).
pub fn load_into(loader: &ResourceLoader) {
    let Some(dir) = profile_dir() else {
        eprintln!("[pluto] no HOME: security profile persistence disabled");
        return;
    };
    match read_file(&dir, HSTS_FILE) {
        Some(text) => loader.set_hsts_store(HstsStore::from_json(&text)),
        None => eprintln!("[pluto] no persisted HSTS store yet ({})", dir.display()),
    }
    match read_file(&dir, HTTPS_ONLY_FILE) {
        Some(text) => loader.set_https_only(HttpsOnly::from_json(&text)),
        None => eprintln!("[pluto] no persisted HTTPS-only state yet ({})", dir.display()),
    }
}

/// Save the loader's current security state to the profile.
/// Failures are logged, never fatal.
pub fn save_from(loader: &ResourceLoader) {
    let Some(dir) = profile_dir() else {
        return;
    };
    let hsts_json = loader
        .hsts_store()
        .lock()
        .unwrap()
        .to_json(std::time::SystemTime::now());
    if let Err(e) = write_file_atomic(&dir, HSTS_FILE, &hsts_json) {
        eprintln!("[pluto] could not save HSTS store: {e}");
    }
    let https_only_json = loader.https_only().lock().unwrap().to_json();
    if let Err(e) = write_file_atomic(&dir, HTTPS_ONLY_FILE, &https_only_json) {
        eprintln!("[pluto] could not save HTTPS-only state: {e}");
    }
}

/// Add a per-site HTTPS-only exception, live and persisted: the
/// host is fetched over plain HTTP from now on. Returns true when
/// the host was newly excepted.
#[allow(dead_code)] // The settings-UI seam for the escape hatch; no UI calls it yet.
pub fn add_https_only_exception(loader: &ResourceLoader, host: &str) -> bool {
    let added = loader.https_only().lock().unwrap().add_exception(host);
    if added {
        save_from(loader);
    }
    added
}

/// Remove a per-site HTTPS-only exception, live and persisted.
/// Returns true when one was held.
#[allow(dead_code)] // The settings-UI seam for the escape hatch; no UI calls it yet.
pub fn remove_https_only_exception(loader: &ResourceLoader, host: &str) -> bool {
    let removed = loader
        .https_only()
        .lock()
        .unwrap()
        .remove_exception(host);
    if removed {
        save_from(loader);
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard holding the shared HOME lock for the span between
    /// `with_home` and `restore_home` (see `TEST_HOME_LOCK`).
    struct HomeGuard {
        old: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    fn with_home(dir: &PathBuf) -> HomeGuard {
        // Serialise with every other HOME-mutating test in the
        // crate: `HOME` is process-global, and a parallel test
        // swapping it mid-span made the loader read the wrong home.
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

    fn temp_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pluto-security-test-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn profile_round_trip() {
        let home = temp_home("round-trip");
        let old = with_home(&home);
        let loader = ResourceLoader::new();
        loader
            .hsts_store()
            .lock()
            .unwrap()
            .record("example.com", true, Some("max-age=31536000; includeSubDomains"), std::time::SystemTime::now());
        loader
            .https_only()
            .lock()
            .unwrap()
            .add_exception("intranet.local");
        save_from(&loader);

        let loader2 = ResourceLoader::new();
        load_into(&loader2);
        assert!(
            loader2
                .hsts_store()
                .lock()
                .unwrap()
                .is_known_shared("sub.example.com", std::time::SystemTime::now()),
            "HSTS store survives a profile round-trip"
        );
        assert!(
            !loader2
                .https_only()
                .lock()
                .unwrap()
                .should_upgrade("intranet.local"),
            "HTTPS-only exception survives a profile round-trip"
        );
        assert!(
            loader2
                .https_only()
                .lock()
                .unwrap()
                .should_upgrade("example.com"),
            "non-excepted hosts still upgrade after a round-trip"
        );
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn missing_profile_loads_defaults() {
        let home = temp_home("missing");
        let old = with_home(&home);
        let loader = ResourceLoader::new();
        load_into(&loader); // No files exist yet: must not panic.
        assert!(
            loader
                .hsts_store()
                .lock()
                .unwrap()
                .entries
                .is_empty()
        );
        assert!(loader.https_only().lock().unwrap().enabled);
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn corrupt_profile_loads_defaults() {
        let home = temp_home("corrupt");
        let old = with_home(&home);
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(HSTS_FILE), "not json{{").unwrap();
        std::fs::write(dir.join(HTTPS_ONLY_FILE), "{{{{").unwrap();
        let loader = ResourceLoader::new();
        load_into(&loader);
        assert!(
            loader
                .hsts_store()
                .lock()
                .unwrap()
                .entries
                .is_empty()
        );
        assert!(loader.https_only().lock().unwrap().enabled);
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn exception_api_updates_loader_and_disk() {
        let home = temp_home("exception-api");
        let old = with_home(&home);
        let loader = ResourceLoader::new();
        assert!(add_https_only_exception(&loader, "intranet.local"));
        assert!(!add_https_only_exception(&loader, "intranet.local"));
        assert!(!loader.https_only().lock().unwrap().should_upgrade("intranet.local"));
        let loader2 = ResourceLoader::new();
        load_into(&loader2);
        assert!(!loader2.https_only().lock().unwrap().should_upgrade("intranet.local"));
        assert!(remove_https_only_exception(&loader, "intranet.local"));
        assert!(loader.https_only().lock().unwrap().should_upgrade("intranet.local"));
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old);
    }

    #[test]
    fn concurrent_atomic_writes_never_corrupt() {
        // Stage 10 Rec 6, same writer as `storage`'s
        // `concurrent_atomic_writes_never_corrupt` (this module
        // keeps its own copy): concurrent writers to one profile
        // dir must each land a whole document — the old fixed
        // temp name made their writes interleave (and renames
        // fail outright when another writer renamed the shared
        // temp file first).
        let dir = std::env::temp_dir().join("pluto-security-test-atomic-race");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let docs: Vec<String> = (0..4)
            .map(|w| {
                let pad = "x".repeat(512 + w * 997);
                format!("{{\"writer\":{w},\"pad\":\"{pad}\"}}")
            })
            .collect();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
        let mut handles = Vec::new();
        for w in 0..4 {
            let dir = dir.clone();
            let doc = docs[w].clone();
            let barrier = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || -> std::io::Result<()> {
                barrier.wait();
                for _ in 0..25 {
                    write_file_atomic(&dir, "race.json", &doc)?;
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().unwrap().expect("every concurrent write succeeds");
        }
        let text = std::fs::read_to_string(dir.join("race.json")).unwrap();
        assert!(
            docs.contains(&text),
            "target holds exactly one writer's document, got {} bytes",
            text.len()
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "race.json")
            .collect();
        assert!(leftovers.is_empty(), "no temp files left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
