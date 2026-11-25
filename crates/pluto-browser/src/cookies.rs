//! Cookie persistence: `cookies.json` in the profile dir (Stage 13
//! Rec 9).
//!
//! The jar lives in the shared loader (`SharedLoader::jar`); this
//! module only owns its life on disk, mirroring the
//! `security::load_into` / `save_from` pattern:
//!
//! * [`load_into`] at startup (browser start, and the broker's
//!   start): restores persistent cookies; `CookieJar::from_json`
//!   prunes expired entries and re-bases the sequence numbers.
//! * [`save_from`] after every navigation and on tab close — and
//!   the broker calls it after any fetch that took in a
//!   `Set-Cookie`, so a login survives even a crash mid-session.
//!
//! Session cookies are stripped with `clear_session_cookies`
//! **before the first save** — `to_json` serialises session
//! cookies too, so saving without stripping would make them
//! persistent (trap 2).

use pluto_net::{CookieJar, ResourceLoader};

/// File name inside the profile dir.
pub const COOKIES_FILE: &str = "cookies.json";

/// Load the persisted cookie jar into `loader`. Missing or corrupt
/// files fall back to an empty jar (logged, not fatal).
pub fn load_into(loader: &ResourceLoader) {
    let Some(dir) = crate::security::profile_dir() else {
        eprintln!("[pluto] no HOME: cookie persistence disabled");
        return;
    };
    match std::fs::read_to_string(dir.join(COOKIES_FILE)) {
        Ok(text) => match CookieJar::from_json(&text) {
            Ok(jar) => {
                let jar_arc = loader.cookie_jar();
                *jar_arc.lock().unwrap() = jar;
            }
            Err(e) => eprintln!("[pluto] corrupt {COOKIES_FILE}, starting empty: {e}"),
        },
        Err(_) => eprintln!(
            "[pluto] no persisted cookie jar yet ({})",
            dir.display()
        ),
    }
}

/// Save the loader's jar to disk. Session cookies are stripped
/// first (see the module docs) — this is the only save path, so
/// the strip-before-first-save rule holds for every caller.
pub fn save_from(loader: &ResourceLoader) {
    let Some(dir) = crate::security::profile_dir() else {
        return;
    };
    let jar_arc = loader.cookie_jar();
    let json = {
        let mut jar = jar_arc.lock().unwrap();
        jar.clear_session_cookies();
        match jar.to_json() {
            Ok(json) => json,
            Err(e) => {
                eprintln!("[pluto] could not serialise cookie jar: {e}");
                return;
            }
        }
    };
    if let Err(e) = crate::atomic::write_file_atomic(&dir, COOKIES_FILE, &json) {
        eprintln!("[pluto] could not save cookie jar: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::profile_dir;

    /// Stage 13 Rec 9: a persistent cookie survives save +
    /// reload into a fresh loader, and a session cookie is
    /// stripped before the first save (it must NOT come back).
    #[test]
    fn cookies_persist_across_restart_without_session_cookies() {
        let _lock = crate::TEST_HOME_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let home = std::env::temp_dir().join(format!(
            "pluto-cookies-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        let old = std::env::var_os("HOME");
        // SAFETY: HOME is only read by the persistence helpers,
        // which run on this thread while the lock is held.
        unsafe { std::env::set_var("HOME", &home) };
        let dir = profile_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        {
            let loader = ResourceLoader::new();
            let jar = loader.cookie_jar();
            jar.lock().unwrap().store_set_cookie(
                "https://shop.example/",
                "shop.example",
                &[
                    "perm=keep-me; Max-Age=3600; Path=/".to_string(),
                    "sess=drop-me; Path=/".to_string(),
                ],
            );
            save_from(&loader);
        }
        // The file holds the persistent cookie only.
        let text = std::fs::read_to_string(dir.join(COOKIES_FILE)).unwrap();
        assert!(text.contains("keep-me"), "persistent cookie saved");
        assert!(!text.contains("drop-me"), "session cookie stripped before save");
        {
            let loader = ResourceLoader::new();
            load_into(&loader);
            let header = loader
                .cookie_jar()
                .lock()
                .unwrap()
                .cookie_header("https://shop.example/", "shop.example", true, true, true);
            let header = header.expect("a cookie header comes back after reload");
            assert!(header.contains("perm=keep-me"), "login survives a restart: {header}");
            assert!(!header.contains("sess="), "session cookie stays dropped: {header}");
        }
        let _ = std::fs::remove_dir_all(&home);
        unsafe {
            match old {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}
