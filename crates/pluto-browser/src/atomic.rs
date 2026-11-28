//! One shared atomic file writer for every profile file (Stage 13
//! Rec 9 / trap 14).
//!
//! Three copies of `write_file_atomic` used to live in the tree
//! (`security.rs`, `storage.rs`, `blocker.rs` in `pluto-net`), each
//! with its own temp-name counter — the Stage 10 race fix had to
//! land in all three by hand. This module is the single writer for
//! the browser crate: unique pid+counter temp names with `O_EXCL`,
//! `fsync` of the file *and* the directory before the rename (so a
//! crash between write and rename cannot lose the write), and a
//! startup sweep that removes orphaned temp files (the live
//! profile's `.local-storage.json.9276.0.tmp` proved a killed write
//! leaves litter).
//!
//! `pluto-net`'s blocker copy stays: `pluto-net` cannot depend on
//! `pluto-browser`, and the blocker writer is owned by the fetch
//! layer. A fourth browser-crate writer must use this module.

use std::path::Path;

/// Write `contents` to `dir/name` atomically (temp-file + rename),
/// creating `dir` if needed. The temp name is unique per write
/// (process id + a per-process counter, created with `O_EXCL` and
/// retried on collision): two writers to the same profile dir must
/// never share one temp file. The file is `fsync`ed before the
/// rename and the directory is `fsync`ed after it, so the write is
/// durable, not merely atomic.
pub fn write_file_atomic(dir: &Path, name: &str, contents: &str) -> std::io::Result<()> {
    write_file_atomic_bytes(dir, name, contents.as_bytes())
}

/// The binary twin of [`write_file_atomic`]: the same temp-file +
/// rename + fsync dance for raw bytes (follow-up UI work,
/// 2026-11-27 — the bookmark dialog's custom favicon files are
/// image bytes, not text).
pub fn write_file_atomic_bytes(
    dir: &Path,
    name: &str,
    contents: &[u8],
) -> std::io::Result<()> {
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
                let result = file
                    .write_all(contents)
                    .and_then(|()| file.sync_all())
                    .and_then(|()| {
                        drop(file);
                        std::fs::rename(&tmp, &target)
                    })
                    .and_then(|()| sync_dir(dir));
                if result.is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
                return result;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                attempt += 1;
                if attempt > 100 {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// `fsync` a directory (best-effort on platforms without it).
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        // Opening a directory read-only and fsyncing it is the
        // portable Unix spelling (no `libc` dependency needed).
        let file = std::fs::File::open(dir)?;
        file.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// Remove orphaned atomic-write temp files (`.*.<pid>.<n>.tmp`)
/// from `dir` — leftovers of writes killed mid-flight. Returns
/// how many were removed. Never touches the live files.
pub fn sweep_stale_temps(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') && name.ends_with(".tmp") && name.matches('.').count() >= 3
            && std::fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
    }
    removed
}

/// The profile directory: `$HOME/.config/pluto`. Shared with
/// [`crate::security::profile_dir`] (kept as the canonical
/// spelling there; this is the same path, for modules that do not
/// want the security import).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_round_trips_and_sweeps_temps() {
        let dir = std::env::temp_dir().join(format!(
            "pluto-atomic-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        write_file_atomic(&dir, "a.json", "{\"x\":1}").unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("a.json")).unwrap(),
            "{\"x\":1}"
        );
        // A stale temp is swept; the live file survives.
        std::fs::write(dir.join(".a.json.99999.0.tmp"), "").unwrap();
        assert_eq!(sweep_stale_temps(&dir), 1);
        assert!(dir.join("a.json").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
