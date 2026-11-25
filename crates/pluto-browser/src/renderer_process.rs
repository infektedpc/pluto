//! The per-tab renderer process (Stage 9 Rec 9, M5 process split).
//!
//! The renderer is the same binary re-executed with `--renderer`:
//! one process per tab, speaking [`pluto_ipc`] over a Unix
//! socketpair, building display lists headlessly. The browser
//! process keeps the one wgpu device and the font system — there is
//! NO separate GPU process at M5 (ARCHITECTURE.md D4) — and the
//! renderer never opens a window, creates a wgpu device, or touches
//! the network, the disk, or the GPU directly: it asks the browser
//! for resources over IPC, and the browser validates every ask.
//!
//! ## How the socket gets into the child
//!
//! The browser creates a `UnixStream::pair()`, clears `CLOEXEC` on
//! the child end with `fcntl` (the pair is created close-on-exec,
//! so this step is what lets the fd survive `exec` — verified
//! empirically: Rust's `std::process::Command` keeps a non-CLOEXEC
//! fd open across exec), and spawns the current binary with
//! `--renderer <fd>`. The child rebuilds the stream from the raw fd
//! number ([`run_renderer_child`]) and handshakes on it. The fd
//! number itself is the only thing passed on the command line; the
//! socketpair is the only channel between the two processes.
//!
//! ## Supervision and kill semantics
//!
//! [`RendererProcess`] is the browser side of one renderer: it owns
//! the child and the socket transport, and it is the only path by
//! which the browser talks to that tab's renderer.
//!
//! * A renderer that sends a malformed IPC message is treated as
//!   COMPROMISED and killed, never recovered ([`RendererProcess::recv`]
//!   SIGKILLs on any `InvalidData` decode error — the report's
//!   broker rule). A renderer that dies on its own is observed as a
//!   clean EOF, not a compromise.
//! * A hung tab closes cleanly: [`RendererProcess::close`] asks the
//!   renderer to shut down, waits a grace period, then SIGKILLs it.
//!   `close` always reaps the child (no zombies).
//! * A failed spawn or handshake kills and reaps the child before
//!   returning the error, so no half-started renderer ever leaks.
//!
//! The iced shell does not use this yet: it still runs the
//! in-process worker (`renderer::run_worker`) until the subscription
//! is switched over to spawn one `RendererProcess` per tab and
//! bridge its events into `Message::RendererEvent`. That switch is
//! the follow-up integration; this module is the spawn + supervise
//! half of it, tested here against real `--renderer` children.
//!
//! ## Sandbox (Rec 10 — implemented)
//!
//! [`install_sandbox_hook`] installs best-effort layers in the child,
//! right after the handshake and before the renderer loop starts, in
//! order: user namespace, mount / ipc / uts namespaces (no pid
//! namespace — it breaks the renderer's first thread spawn; see
//! the layer comment), close
//! of all FDs except the IPC socketpair (and the tokio runtime's own
//! epoll/eventfd/timerfd fds), chroot to an empty tmpfs root with
//! only the font stack, DNS files, and shared-library dirs
//! bind-mounted read-only,
//! capability drop, rlimits, `PR_SET_NO_NEW_PRIVS`, then the
//! seccomp-bpf allow-list (default `KILL_PROCESS`). Every layer is
//! best-effort: on ANY failure it logs and the child continues
//! without that layer — a failed layer never fails the spawn and
//! never panics. The two layers that would break the renderer as
//! it stands today are gated, not attempted blindly: the network
//! namespace (the renderer still fetches directly through
//! `pluto-net`; a fresh empty net ns would break every real
//! navigation — opt in with `PLUTO_SANDBOX_NET_NS=1` once the
//! broker fetch lands) and the chroot (skipped if no safe empty
//! root can be built, or with `PLUTO_SANDBOX_CHROOT=0`).
//!
//! `ptrace` is never allowed in the sandboxed process, and the
//! child can never `execve` again. The GPU hole, documented: at
//! M5 the renderer never creates a wgpu device and holds no
//! `/dev/dri` fd — the browser process owns the one GPU device —
//! so device fds are closed by the FD layer. If a future spawn
//! path pre-opens a `/dev/dri` fd for in-process rasterisation,
//! that fd (and the `ioctl`/`mmap` surface that goes with it) is
//! an accepted, bounded hole until a dedicated GPU process exists
//! (post-M5); see the FD layer's comment.
//!
//! Escape hatch: `--no-sandbox` (or `PLUTO_NO_SANDBOX=1`) skips
//! the whole hook. It is UNSAFE and exists only for debugging —
//! never ship it.

use std::io;
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;

use pluto_ipc::transport::{recv_renderer_message, SocketTransport, Transport};
use pluto_ipc::{BrowserToRenderer, RendererToBrowser};

/// The child-mode flag: `pluto-browser --renderer <fd>`, where
/// `<fd>` is the number of the inherited IPC socketpair end.
pub const RENDERER_ARG: &str = "--renderer";
/// The shell-assigned tab id the `--renderer` child runs as
/// (Stage 11 Rec 4): `--tab-id <n>`, echoed by the worker in
/// every fetch request instead of the old hardcoded 0.
pub const TAB_ID_ARG: &str = "--tab-id";

/// How long the spawn handshake may take before the child is
/// treated as failed (and killed). A healthy child handshakes in
/// milliseconds; 10 s covers a cold start under load.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long [`RendererProcess::close`] waits for a clean exit after
/// `Shutdown` before SIGKILLing a hung renderer.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Test hook: when the `--renderer` child sees this env var, it
/// sends one deliberately malformed IPC frame after the handshake
/// (then runs normally) — so tests can prove the browser kills a
/// renderer that speaks garbage instead of recovering it. A real
/// renderer can never set this for itself (it cannot change its own
/// inherited environment from JS), and the only thing the hook can
/// do is get the renderer killed faster.
pub const TEST_GARBAGE_ENV: &str = "PLUTO_TEST_RENDERER_GARBAGE";

/// Test hook: when the `--renderer` child sees this env var, it
/// hangs forever after the handshake — so tests can prove a hung
/// tab still closes (SIGKILL on tab close). Same caveat as
/// [`TEST_GARBAGE_ENV`]: it can only make the renderer die.
pub const TEST_HANG_ENV: &str = "PLUTO_TEST_RENDERER_HANG";

/// Test hook: when the `--renderer` child sees this env var, the
/// seccomp allow-list drops `socket` and `connect` — so tests can
/// prove a renderer that cannot open any socket still loads a
/// real page end-to-end through the browser-side broker (Stage 13
/// Rec 11). `sendmsg`/`recvmsg` stay: the IPC transport runs over
/// a socketpair created *before* the hook, on pre-opened fds.
/// Same caveat as [`TEST_GARBAGE_ENV`]: it can only make the
/// renderer die.
pub const TEST_NO_SOCKET_ENV: &str = "PLUTO_TEST_NO_SOCKET_SYSCALLS";

/// Escape hatch (UNSAFE, debugging only): `PLUTO_NO_SANDBOX=1`, or
/// the `--no-sandbox` argument, skips the whole sandbox hook and
/// the renderer runs unconfined. Never ship it. See
/// [`sandbox_disabled`].
pub const NO_SANDBOX_ENV: &str = "PLUTO_NO_SANDBOX";
/// The CLI form of the escape hatch; see [`NO_SANDBOX_ENV`].
pub const NO_SANDBOX_ARG: &str = "--no-sandbox";

/// Dev-only diagnostic (child only, never set in production):
/// install the seccomp filter with `SECCOMP_RET_TRAP` instead of
/// `KILL_PROCESS`, plus a `SIGSYS` handler that prints the blocked
/// syscall number to stderr and exits — so a missing allow-list
/// entry shows up in the log instead of a silent dead tab.
pub const SANDBOX_TRAP_ENV: &str = "PLUTO_SANDBOX_TRAP";
/// Dev-only diagnostic (child only): install the seccomp filter
/// with `SECCOMP_RET_LOG` instead of `KILL_PROCESS`, so the
/// kernel audit log records what the filter would have killed.
pub const SANDBOX_LOG_ENV: &str = "PLUTO_SANDBOX_LOG";
/// Opt-in (child only): also unshare a fresh, empty network
/// namespace. Off by default — the renderer still fetches directly
/// through `pluto-net`, so a net ns would break real navigations
/// today; enable once the broker fetch lands.
pub const SANDBOX_NETNS_ENV: &str = "PLUTO_SANDBOX_NET_NS";
/// Set to `"0"` (child only) to skip the chroot layer.
pub const SANDBOX_CHROOT_ENV: &str = "PLUTO_SANDBOX_CHROOT";

/// Parse the `--renderer <fd> [--tab-id <n>]` arguments, if
/// present. Returns the inherited IPC socket fd and the shell-
/// assigned tab id (Stage 11 Rec 4: the id is assigned by the
/// shell and echoed by the worker — nothing hardcodes 0
/// anymore), or `None` for a normal browser launch. `--tab-id`
/// defaults to 0 when absent, for children spawned before the
/// flag existed. Any other flag (e.g. `--csd`) is ignored here —
/// the decorations parser in `main.rs` handles those.
pub fn parse_renderer_args() -> Option<(RawFd, u64)> {
    let mut args = std::env::args().skip(1);
    let mut fd: Option<RawFd> = None;
    let mut tab_id: u64 = 0;
    while let Some(arg) = args.next() {
        if arg == RENDERER_ARG {
            fd = args.next().and_then(|fd| fd.parse().ok());
        } else if arg == TAB_ID_ARG {
            tab_id = args.next().and_then(|id| id.parse().ok()).unwrap_or(0);
        }
    }
    fd.map(|fd| (fd, tab_id))
}

/// Whether the sandbox escape hatch is set: `PLUTO_NO_SANDBOX`
/// to anything but `"0"`, or a `--no-sandbox` argument. UNSAFE —
/// the renderer then runs unconfined; for debugging only.
pub fn sandbox_disabled() -> bool {
    env_flag_on(NO_SANDBOX_ENV)
        || args_disable_sandbox(std::env::args().map(|arg| arg.to_string()))
}

/// Pure half of [`sandbox_disabled`], for tests: does this argv
/// contain `--no-sandbox`?
fn args_disable_sandbox(args: impl IntoIterator<Item = String>) -> bool {
    args.into_iter().any(|arg| arg == NO_SANDBOX_ARG)
}

/// An env flag is on when it is set to anything but `"0"` / empty.
fn env_flag_on(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v != "0" && !v.is_empty())
}

/// Log a sandbox diagnostic from the renderer child. The child
/// runs before any `tracing` subscriber exists, so a bare
/// `tracing::warn!` would be dropped — `eprintln!` is the child's
/// established diagnostic channel (see [`run_renderer_child`]);
/// the `tracing` call is kept so the event also reaches a
/// subscriber if one is ever installed before the child runs.
fn sandbox_warn(message: &str) {
    tracing::warn!("[pluto-sandbox] {message}");
    eprintln!("[pluto-sandbox] {message}");
}

/// SANDBOX HOOK (Stage 9 Rec 10).
///
/// Called in the `--renderer` child right after the handshake and
/// before the renderer loop starts. Installs the layers in order —
/// user namespace, other namespaces, FD close, chroot to an empty
/// root, capability drop, rlimits, `PR_SET_NO_NEW_PRIVS`, seccomp —
/// each best-effort: on ANY failure the layer logs and the child
/// continues without it (degrade, never fail the spawn, never
/// panic). A one-line summary of which layers landed is printed
/// to stderr.
///
/// # Safety
///
/// This runs ONLY in the forked renderer child. It must never be
/// called in the browser process (or a test process): `unshare`,
/// `chroot`, and `PR_SET_SECCOMP` would confine the caller
/// itself. Tests exercise it only by spawning a real `--renderer`
/// child, or by calling the pure filter builder below.
/// Install the sandbox layers and return the per-layer results
/// (Stage 13 Rec 12a): the child reports them to the browser over
/// IPC before the first navigation, so the shell knows — instead
/// of guessing — which layers landed.
pub fn install_sandbox_hook(ipc_fd: RawFd) -> Vec<(String, bool)> {
    if sandbox_disabled() {
        sandbox_warn(
            "sandbox DISABLED via --no-sandbox / PLUTO_NO_SANDBOX=1 — UNSAFE: \
             the renderer runs unconfined. Never ship this.",
        );
        return vec![("sandbox-disabled".to_string(), false)];
    }
    let mut results: Vec<(&'static str, bool)> = Vec::new();
    attempt_layer(&mut results, "user-namespace", install_user_namespace);
    attempt_layer(&mut results, "namespaces", install_namespaces);
    attempt_layer(&mut results, "close-fds", || close_extra_fds(ipc_fd));
    if std::env::var_os(SANDBOX_CHROOT_ENV).is_some_and(|v| v == "0") {
        eprintln!("[pluto-sandbox] chroot layer skipped (PLUTO_SANDBOX_CHROOT=0)");
        results.push(("chroot", false));
    } else {
        attempt_layer(&mut results, "chroot", install_chroot);
    }
    attempt_layer(&mut results, "capabilities", drop_capabilities);
    attempt_layer(&mut results, "rlimits", install_rlimits);
    attempt_layer(&mut results, "no-new-privs", || {
        let ret = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    });
    attempt_layer(&mut results, "seccomp", install_seccomp);
    let summary: Vec<String> = results
        .iter()
        .map(|(name, ok)| format!("{name}={}", if *ok { "ok" } else { "OFF" }))
        .collect();
    eprintln!("[pluto-sandbox] layers: {}", summary.join(" "));
    results
        .into_iter()
        .map(|(name, ok)| (name.to_string(), ok))
        .collect()
}

/// Run one sandbox layer best-effort and record the outcome. A
/// failed layer is logged and the child continues without it.
fn attempt_layer(
    results: &mut Vec<(&'static str, bool)>,
    name: &'static str,
    layer: impl FnOnce() -> io::Result<()>,
) {
    match layer() {
        Ok(()) => results.push((name, true)),
        Err(e) => {
            sandbox_warn(&format!("layer failed, continuing degraded: {name}: {e}"));
            results.push((name, false));
        }
    }
}

/// Layer 1: unprivileged user namespace, then map the outer
/// uid/gid to inner uid/gid 0 — the key that unlocks the other
/// namespaces without privilege. The outer ids are read BEFORE
/// `unshare`: after it, `getuid`/`getgid` report the overflow id
/// until the map is written. A process may write its own
/// `uid_map` with a single line mapping its own euid, and its
/// `gid_map` after writing `"deny"` to `setgroups`.
fn install_user_namespace() -> io::Result<()> {
    let outer_uid = unsafe { libc::getuid() };
    let outer_gid = unsafe { libc::getgid() };
    let ret = unsafe { libc::unshare(libc::CLONE_NEWUSER) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    write_proc_file("uid_map", format!("0 {outer_uid} 1"))?;
    write_proc_file("setgroups", "deny".to_string())?;
    write_proc_file("gid_map", format!("0 {outer_gid} 1"))?;
    Ok(())
}

fn write_proc_file(name: &str, contents: String) -> io::Result<()> {
    std::fs::write(format!("/proc/self/{name}"), contents)
}

/// Layer 2: mount / ipc / uts namespaces, one at a time so
/// a single refused namespace does not take the rest down.
///
/// NO pid namespace, deliberately: `unshare(CLONE_NEWPID)` only
/// takes effect for children forked afterwards, and a process
/// with a pending PID namespace cannot create threads — its
/// first `clone3(CLONE_THREAD)` fails with `EINVAL`, because a
/// new thread would need a PID in the new namespace while the
/// thread-group leader is still in the old one. The renderer
/// spawns its layout/tab threads AFTER this hook, so a PID
/// namespace would kill the renderer at its first thread spawn
/// (verified: thread spawn fails with `EINVAL` with the flag,
/// works without it). The namespace would only have isolated
/// children the renderer never forks, so dropping it loses
/// nothing. The network namespace is likewise NOT attempted by
/// default — see [`SANDBOX_NETNS_ENV`].
fn install_namespaces() -> io::Result<()> {
    let mut any_ok = false;
    let mut first_err: Option<io::Error> = None;
    for (name, flag) in [
        ("mount", libc::CLONE_NEWNS),
        ("ipc", libc::CLONE_NEWIPC),
        ("uts", libc::CLONE_NEWUTS),
    ] {
        let ret = unsafe { libc::unshare(flag) };
        if ret == 0 {
            any_ok = true;
        } else {
            let e = io::Error::last_os_error();
            sandbox_warn(&format!("namespace unshare failed: {name}: {e}"));
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    }
    if env_flag_on(SANDBOX_NETNS_ENV) {
        let ret = unsafe { libc::unshare(libc::CLONE_NEWNET) };
        if ret == 0 {
            any_ok = true;
        } else {
            let e = io::Error::last_os_error();
            sandbox_warn(&format!("namespace unshare failed: net: {e}"));
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    } else {
        eprintln!(
            "[pluto-sandbox] net namespace skipped (broker mode fetches browser-side; \
             set PLUTO_SANDBOX_NET_NS=1 to opt in)"
        );
    }
    if any_ok {
        Ok(())
    } else {
        Err(first_err.unwrap_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "all namespace unshares failed")
        }))
    }
}

/// Layer 3: close every fd except stdin/stdout/stderr, the IPC
/// socket (`keep_fd`), and the anonymous fds the tokio runtime
/// already owns (its epoll / eventfd / timerfd / signalfd and
/// pipes, plus any other socket). This runs BEFORE the chroot:
/// it reads `/proc/self/fd`, which the chroot hides.
///
/// GPU hole, documented: at M5 the renderer never creates a wgpu
/// device and holds no `/dev/dri` fd — the browser process owns
/// the one GPU device — so device fds are closed here. If a
/// future spawn path pre-opens a `/dev/dri` fd for in-process
/// rasterisation, that fd must be added to the keep set: it is an
/// accepted, bounded hole until a dedicated GPU process exists
/// (post-M5), and it widens the seccomp surface with it
/// (`ioctl`/`mmap` on the DRM fd).
fn close_extra_fds(keep_fd: RawFd) -> io::Result<()> {
    let path = b"/proc/self/fd\0";
    let dir = unsafe { libc::opendir(path.as_ptr() as *const libc::c_char) };
    if dir.is_null() {
        return Err(io::Error::last_os_error());
    }
    let dir_fd = unsafe { libc::dirfd(dir) };
    let mut closed = 0u32;
    loop {
        let entry = unsafe { libc::readdir(dir) };
        if entry.is_null() {
            break;
        }
        let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
        let Ok(name_str) = name.to_str() else {
            continue;
        };
        let Ok(fd) = name_str.parse::<RawFd>() else {
            continue; // "." and ".."
        };
        if fd == dir_fd || fd <= 2 || fd == keep_fd {
            continue;
        }
        let mut link = format!("/proc/self/fd/{fd}").into_bytes();
        link.push(0);
        let mut buf = [0u8; 256];
        let n = unsafe {
            libc::readlink(
                link.as_ptr() as *const libc::c_char,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
            )
        };
        if n <= 0 {
            continue; // vanished mid-scan; nothing to close
        }
        let target = String::from_utf8_lossy(&buf[..n as usize]);
        let keep = target.starts_with("socket:")
            || target.starts_with("pipe:")
            || target.starts_with("anon_inode:[eventpoll]")
            || target.starts_with("anon_inode:[eventfd]")
            || target.starts_with("anon_inode:[timerfd]")
            || target.starts_with("anon_inode:[signalfd]");
        if keep {
            continue;
        }
        if unsafe { libc::close(fd) } == 0 {
            closed += 1;
        }
    }
    unsafe { libc::closedir(dir) };
    eprintln!("[pluto-sandbox] closed {closed} stray fds");
    Ok(())
}

/// Layer 4: `chroot` into an EMPTY tmpfs root with only what the
/// renderer provably needs bind-mounted read-only: the font stack
/// (`/usr/share/fonts`, `/usr/local/share/fonts`, `/etc/fonts`,
/// `/usr/share/fontconfig`, plus the user's font dirs when they
/// exist), the DNS files glibc's resolver reads
/// (`/etc/resolv.conf`, `/etc/hosts`, `/etc/nsswitch.conf`), and
/// the shared-library dirs (`/lib`, `/lib64`, `/usr/lib`) plus
/// `/etc/ld.so.cache`. The lib binds exist because glibc `dlopen`s
/// NSS modules lazily on the first DNS lookup — without them the
/// resolver fails inside the root and real-page fetches break.
/// They are world-readable system files, not user data; the root
/// still hides the home directory, the profile directory, `/proc`,
/// `/dev`, `/tmp`, `/bin`, and everything else. TLS roots are
/// compiled into `pluto-net` (webpki-roots), so no CA store bind
/// is needed. Skipped (degraded) if any step fails: the gate is
/// the mount-privatisation call, which fails with `EPERM` when
/// the mount-namespace layer did not land, so a failed mount ns
/// can never leak bind mounts onto the host.
///
/// NOTE: the empty root directory itself is left behind under
/// `/tmp` (one per renderer child) — the process holds it as its
/// root and cannot unlink it; the tmpfs mount on it dies with the
/// mount namespace. Harmless, documented.
fn install_chroot() -> io::Result<()> {
    // Privatise the mount table first (gate — see doc above).
    mount_ptr("none", "/", None, libc::MS_REC | libc::MS_PRIVATE, None)?;
    let root = make_empty_root()?;
    mount_ptr(
        "tmpfs",
        &root,
        Some("tmpfs"),
        libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
        Some("mode=0755"),
    )?;
    // Font stack (required: fontdb reads both of these).
    bind_mount_ro("/usr/share/fonts", &format!("{root}/usr/share/fonts"))?;
    bind_mount_ro("/etc/fonts", &format!("{root}/etc/fonts"))?;
    for dir in ["/usr/local/share/fonts", "/usr/share/fontconfig"] {
        if Path::new(dir).is_dir() {
            if let Err(e) = bind_mount_ro(dir, &format!("{root}{dir}")) {
                sandbox_warn(&format!("optional font bind-mount failed: {dir}: {e}"));
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        for rel in [".fonts", ".local/share/fonts", ".config/fontconfig"] {
            let src = format!("{home}/{rel}");
            if Path::new(&src).is_dir() {
                if let Err(e) = bind_mount_ro(&src, &format!("{root}{src}")) {
                    sandbox_warn(&format!("optional home font bind-mount failed: {src}: {e}"));
                }
            }
        }
    }
    // DNS files (glibc resolver; files, not dirs).
    for file in ["/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"] {
        if Path::new(file).exists() {
            if let Err(e) = bind_mount_file_ro(file, &format!("{root}{file}")) {
                sandbox_warn(&format!("optional DNS bind-mount failed: {file}: {e}"));
            }
        }
    }
    // Shared-library dirs + loader cache: glibc dlopens NSS
    // modules lazily on first DNS lookup (see the doc above).
    for dir in ["/lib", "/lib64", "/usr/lib", "/usr/lib64"] {
        if Path::new(dir).is_dir() {
            if let Err(e) = bind_mount_ro(dir, &format!("{root}{dir}")) {
                sandbox_warn(&format!("optional lib bind-mount failed: {dir}: {e}"));
            }
        }
    }
    if Path::new("/etc/ld.so.cache").exists() {
        if let Err(e) = bind_mount_file_ro("/etc/ld.so.cache", &format!("{root}/etc/ld.so.cache")) {
            sandbox_warn(&format!("optional ld.so.cache bind-mount failed: {e}"));
        }
    }
    // chroot, then move the cwd inside (a cwd outside the new root
    // would be unreachable afterwards).
    let root_c = cstr(&root)?;
    let ret = unsafe { libc::chroot(root_c.as_ptr()) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    let slash = b"/\0";
    if unsafe { libc::chdir(slash.as_ptr() as *const libc::c_char) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn cstr(s: &str) -> io::Result<std::ffi::CString> {
    std::ffi::CString::new(s)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL byte in path"))
}

fn mount_ptr(
    source: &str,
    target: &str,
    fstype: Option<&str>,
    flags: libc::c_ulong,
    data: Option<&str>,
) -> io::Result<()> {
    let src = cstr(source)?;
    let tgt = cstr(target)?;
    let fs = fstype.map(cstr).transpose()?;
    let dat = data.map(cstr).transpose()?;
    let ret = unsafe {
        libc::mount(
            src.as_ptr(),
            tgt.as_ptr(),
            fs.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            flags,
            dat.as_ref().map_or(std::ptr::null(), |d| d.as_ptr()) as *const libc::c_void,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn mkdir_p(path: &str) -> io::Result<()> {
    let mut cur = String::new();
    for part in path.split('/') {
        if part.is_empty() {
            continue;
        }
        cur.push('/');
        cur.push_str(part);
        let c = cstr(&cur)?;
        let ret = unsafe { libc::mkdir(c.as_ptr(), 0o755) };
        if ret != 0 {
            let e = io::Error::last_os_error();
            if e.kind() != io::ErrorKind::AlreadyExists {
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Bind-mount a directory read-only (a bind mount ignores
/// `MS_RDONLY` on the first call; it must be remounted).
fn bind_mount_ro(src: &str, dst: &str) -> io::Result<()> {
    mkdir_p(dst)?;
    mount_ptr(src, dst, None, libc::MS_BIND | libc::MS_REC, None)?;
    mount_ptr(
        "",
        dst,
        None,
        libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_REC,
        None,
    )
}

/// Bind-mount a single file read-only (the destination file must
/// exist first).
fn bind_mount_file_ro(src: &str, dst: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(dst).parent() {
        mkdir_p(&parent.to_string_lossy())?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(dst)?;
    mount_ptr(src, dst, None, libc::MS_BIND, None)?;
    mount_ptr(
        "",
        dst,
        None,
        libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY,
        None,
    )
}

/// `mkdtemp` an empty directory under the temp dir for the chroot
/// root. Created before the mount, owned by us, nothing in it.
fn make_empty_root() -> io::Result<String> {
    let mut template: Vec<u8> = format!(
        "{}/pluto-renderer-root-XXXXXX",
        std::env::temp_dir().display()
    )
    .into_bytes();
    template.push(0);
    let ptr = unsafe { libc::mkdtemp(template.as_mut_ptr() as *mut libc::c_char) };
    if ptr.is_null() {
        return Err(io::Error::last_os_error());
    }
    let path = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    Ok(path)
}

/// Layer 5: drop all capabilities. Order matters: the bounding
/// set is dropped FIRST (that needs `CAP_SETPCAP`, which the
/// process still holds as inner-root), then `capset` clears the
/// permitted / effective / inheritable sets (dropping one's own
/// caps needs no privilege), then the ambient set is cleared.
fn drop_capabilities() -> io::Result<()> {
    for cap in 0..=40 {
        unsafe {
            libc::prctl(
                libc::PR_CAPBSET_DROP,
                cap as libc::c_ulong,
                0,
                0,
                0,
            )
        };
    }
    #[repr(C)]
    struct CapHeader {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CapData {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    // _LINUX_CAPABILITY_VERSION_3: two CapData words, all zero.
    let header = CapHeader {
        version: 0x2008_0522,
        pid: 0,
    };
    let data = [CapData::default(); 2];
    let ret = unsafe {
        libc::syscall(
            libc::SYS_capset,
            &header as *const CapHeader,
            &data as *const CapData,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    let ret = unsafe {
        libc::prctl(
            libc::PR_CAP_AMBIENT,
            libc::PR_CAP_AMBIENT_CLEAR_ALL as libc::c_ulong,
            0,
            0,
            0,
        )
    };
    if ret != 0 {
        sandbox_warn(&format!(
            "clearing ambient capabilities failed: {}",
            io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Layer 6: generous rlimits — caps a runaway renderer cannot
/// hit in normal use. `RLIMIT_AS` 8 GiB, `RLIMIT_CPU` 600 s,
/// `RLIMIT_NOFILE` 4096, `RLIMIT_NPROC` 8192, `RLIMIT_FSIZE`
/// 64 MiB (the renderer writes no files at all). Both soft and
/// hard limits are set, so the child cannot raise them back
/// (and `setrlimit`/`prlimit64` are not in the seccomp
/// allow-list, so it cannot change them at all after the filter
/// lands). `NPROC` counts threads per real uid system-wide, so
/// it is only ever lowered toward the target, never below the
/// current soft limit — a busy desktop must not make a new
/// renderer fail to spawn threads.
const SANDBOX_RLIMIT_AS: u64 = 8 << 30;
const SANDBOX_RLIMIT_CPU_SECS: u64 = 600;
const SANDBOX_RLIMIT_NOFILE: u64 = 4096;
const SANDBOX_RLIMIT_NPROC: u64 = 8192;
const SANDBOX_RLIMIT_FSIZE: u64 = 64 << 20;

fn install_rlimits() -> io::Result<()> {
    set_rlimit_capped(libc::RLIMIT_AS, SANDBOX_RLIMIT_AS)?;
    set_rlimit_capped(libc::RLIMIT_CPU, SANDBOX_RLIMIT_CPU_SECS)?;
    set_rlimit_capped(libc::RLIMIT_NOFILE, SANDBOX_RLIMIT_NOFILE)?;
    lower_rlimit_toward(libc::RLIMIT_NPROC, SANDBOX_RLIMIT_NPROC)?;
    set_rlimit_capped(libc::RLIMIT_FSIZE, SANDBOX_RLIMIT_FSIZE)?;
    Ok(())
}

/// Set soft and hard limits to `target`, clamped to the current
/// hard limit so the call cannot fail for trying to raise it.
fn set_rlimit_capped(resource: u32, target: u64) -> io::Result<()> {
    let mut cur = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(resource, &mut cur) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let hard = cur.rlim_max as u64;
    let target = if hard == libc::RLIM_INFINITY as u64 {
        target
    } else {
        target.min(hard)
    };
    if cur.rlim_cur as u64 == target && cur.rlim_max as u64 == target {
        return Ok(());
    }
    let lim = libc::rlimit {
        rlim_cur: target as libc::rlim_t,
        rlim_max: target as libc::rlim_t,
    };
    let ret = unsafe { libc::setrlimit(resource, &lim) };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Lower `resource` toward `target`, but never below its current
/// soft limit (leave it alone if it is already at or below).
fn lower_rlimit_toward(resource: u32, target: u64) -> io::Result<()> {
    let mut cur = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(resource, &mut cur) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if (cur.rlim_cur as u64) <= target {
        return Ok(());
    }
    set_rlimit_capped(resource, target)
}

/// Layer 8: the seccomp-bpf allow-list, installed last (after
/// `PR_SET_NO_NEW_PRIVS`, which it requires). Default action is
/// `KILL_PROCESS` — the whole process dies on a disallowed
/// syscall. On architectures without an allow-list table the
/// filter is skipped (logged, degraded). The filter checks the
/// arch first (syscall numbers differ across ABIs) and kills on
/// a mismatch. See [`seccomp_allowed_syscalls`] for the list and
/// the never-allowed set.
fn install_seccomp() -> io::Result<()> {
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        eprintln!("[pluto-sandbox] seccomp skipped: no allow-list for this architecture");
        return Ok(());
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let mut default_action = libc::SECCOMP_RET_KILL_PROCESS;
        if env_flag_on(SANDBOX_TRAP_ENV) {
            install_sigsys_handler()?;
            default_action = libc::SECCOMP_RET_TRAP;
        } else if env_flag_on(SANDBOX_LOG_ENV) {
            default_action = libc::SECCOMP_RET_LOG;
        }
        let prog = build_seccomp_filter(default_action);
        let fprog = libc::sock_fprog {
            len: prog.len() as u16,
            filter: prog.as_ptr() as *mut libc::sock_filter,
        };
        let ret = unsafe {
            libc::prctl(
                libc::PR_SET_SECCOMP,
                libc::SECCOMP_MODE_FILTER,
                &fprog as *const libc::sock_fprog,
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// The seccomp allow-list: syscalls the renderer provably needs
/// after the hook — event loop and IPC (`read`/`write`,
/// `epoll`, `eventfd`, `timerfd`, `futex`), thread creation
/// (`clone`/`clone3`, `set_robust_list`, `set_tid_address`,
/// `arch_prctl`, `rseq`), memory (`mmap`/`munmap`/`mprotect`/
/// `brk`/`madvise`/`mremap`), font scanning and file reads
/// (`openat`/`openat2`, `getdents64`, `statx`/`newfstatat`,
/// `read`, `pread64`, `mmap`), time (`clock_gettime`,
/// `clock_nanosleep`, `nanosleep`), signals (`rt_sigaction`,
/// `rt_sigprocmask`, `rt_sigreturn`, `sigaltstack`), and — until
/// the broker fetch lands — direct HTTP fetches via `pluto-net`
/// (`socket`, `connect`, `sendmsg`/`recvmsg`, `setsockopt`).
/// `restart_syscall` is required: the kernel re-enters
/// interrupted syscalls through it.
///
/// Stage 13 Rec 11: the broker fetch has landed — the
/// out-of-process child fetches through the browser-side broker
/// and needs no socket at all. `socket`/`connect` stay in the
/// list only because the *in-process* worker still fetches
/// directly; [`TEST_NO_SOCKET_ENV`] drops them for broker-mode
/// tests, proving the child loads a real page with no socket.
///
/// NEVER in this list, by design: `ptrace` (a tracer can rewrite
/// syscalls and escape), `execve`/`execveat` (the renderer must
/// never run another program), `fork`/`vfork` (threads use
/// `clone`/`clone3`), `mount`/`umount2`/`chroot`/`pivot_root`/
/// `unshare`/`setns` (namespaces are fixed before the filter),
/// `bpf`/`perf_event_open`/`io_uring_setup` (kernel attack
/// surface), `keyctl`/`add_key`/`request_key`,
/// `process_vm_readv`/`process_vm_writev`,
/// `setrlimit`/`prlimit64` (rlimits are fixed before the filter),
/// `reboot`/`init_module`/`delete_module`/`swapon`, and most
/// file-creating or file-writing syscalls (`unlink`, `truncate`,
/// `open` with write is not distinguished by number — see
/// below). The exceptions are `mkdir`/`mkdirat` and
/// `rename`/`renameat`/`renameat2`, which the persistence layer
/// provably needs (verified live: `create_dir_all` and atomic
/// save during navigation) and which under the chroot can only
/// touch the empty tmpfs root.
///
/// Honest limit: seccomp sees only syscall numbers and scalar
/// args, never pathnames — `openat("/etc/passwd")` and
/// `openat("/usr/share/fonts/...")` are indistinguishable to the
/// filter. Path confinement comes from the chroot + mount
/// namespace, not from this list. The allow-list was grown
/// against the renderer's real post-hook behaviour (threads,
/// font scan, layout, JS, IPC); it is not a measured count and
/// no size is claimed for it.
///
/// With [`TEST_NO_SOCKET_ENV`] set, `socket` and `connect` are
/// dropped from the list (Stage 13 Rec 11 test hook) — the
/// renderer must then fetch through the broker, never directly.
fn seccomp_allowed_syscalls() -> Vec<libc::c_long> {
    if env_flag_on(TEST_NO_SOCKET_ENV) {
        ALLOWED_SYSCALLS
            .iter()
            .copied()
            .filter(|nr| *nr != libc::SYS_socket && *nr != libc::SYS_connect)
            .collect()
    } else {
        ALLOWED_SYSCALLS.to_vec()
    }
}

#[cfg(target_arch = "x86_64")]
const SECCOMP_ARCH: u32 = 0xC000_003E; // AUDIT_ARCH_X86_64
#[cfg(target_arch = "aarch64")]
const SECCOMP_ARCH: u32 = 0xC000_00B7; // AUDIT_ARCH_AARCH64

const ALLOWED_SYSCALLS: &[libc::c_long] = &[
    // I/O on pre-opened fds + font/file reads.
    libc::SYS_read,
    libc::SYS_write,
    libc::SYS_pread64,
    libc::SYS_preadv,
    libc::SYS_readv,
    libc::SYS_writev,
    libc::SYS_lseek,
    libc::SYS_close,
    libc::SYS_close_range,
    libc::SYS_open,
    libc::SYS_openat,
    libc::SYS_openat2,
    libc::SYS_fstat,
    libc::SYS_stat,
    libc::SYS_lstat,
    libc::SYS_newfstatat,
    libc::SYS_statx,
    libc::SYS_fstatfs,
    libc::SYS_statfs,
    libc::SYS_access,
    libc::SYS_faccessat,
    libc::SYS_faccessat2,
    libc::SYS_getdents64,
    libc::SYS_readlink,
    libc::SYS_readlinkat,
    libc::SYS_getcwd,
    // mkdir/mkdirat: the persistence layer (HSTS / localStorage /
    // permissions stores) calls create_dir_all on first use.
    // Under the chroot it can only create directories inside the
    // empty tmpfs root, which die with the process; it cannot
    // create files. (Verified: the worker calls mkdir on
    // ~/.config/pluto during navigation.)
    libc::SYS_mkdir,
    libc::SYS_mkdirat,
    // rename/renameat/renameat2: the persistence layer saves
    // atomically (write a temp file, rename over the target).
    // Under the chroot a rename can only move files inside the
    // empty tmpfs root. (Verified: the worker calls rename
    // during navigation.)
    libc::SYS_rename,
    libc::SYS_renameat,
    libc::SYS_renameat2,
    libc::SYS_fcntl,
    libc::SYS_dup,
    libc::SYS_dup2,
    libc::SYS_dup3,
    // ioctl: no device fds are held at M5, so this is near-inert
    // today; kept for the future /dev/dri hole and for terminal
    // queries on stderr.
    libc::SYS_ioctl,
    // Memory.
    libc::SYS_mmap,
    libc::SYS_munmap,
    libc::SYS_mprotect,
    libc::SYS_brk,
    libc::SYS_madvise,
    libc::SYS_mremap,
    // Threads / processes.
    libc::SYS_clone,
    libc::SYS_clone3,
    libc::SYS_futex,
    libc::SYS_set_robust_list,
    libc::SYS_get_robust_list,
    libc::SYS_set_tid_address,
    libc::SYS_sched_yield,
    libc::SYS_sched_getaffinity,
    libc::SYS_sched_setaffinity,
    libc::SYS_sched_getparam,
    libc::SYS_sched_getscheduler,
    libc::SYS_sched_get_priority_max,
    libc::SYS_sched_get_priority_min,
    libc::SYS_getpid,
    libc::SYS_gettid,
    libc::SYS_getppid,
    libc::SYS_getuid,
    libc::SYS_geteuid,
    libc::SYS_getgid,
    libc::SYS_getegid,
    libc::SYS_exit,
    libc::SYS_exit_group,
    libc::SYS_restart_syscall,
    // Signals.
    libc::SYS_rt_sigaction,
    libc::SYS_rt_sigprocmask,
    libc::SYS_rt_sigreturn,
    libc::SYS_sigaltstack,
    // Misc runtime support.
    libc::SYS_prctl,
    libc::SYS_arch_prctl,
    libc::SYS_rseq,
    libc::SYS_membarrier,
    libc::SYS_getcpu,
    libc::SYS_uname,
    libc::SYS_sysinfo,
    libc::SYS_getrandom,
    libc::SYS_getrlimit,
    libc::SYS_getrusage,
    // Time.
    libc::SYS_clock_gettime,
    libc::SYS_clock_getres,
    libc::SYS_clock_nanosleep,
    libc::SYS_nanosleep,
    libc::SYS_gettimeofday,
    libc::SYS_time,
    // Evented I/O (tokio / mio).
    libc::SYS_epoll_create,
    libc::SYS_epoll_create1,
    libc::SYS_epoll_ctl,
    libc::SYS_epoll_wait,
    libc::SYS_epoll_pwait,
    libc::SYS_eventfd,
    libc::SYS_eventfd2,
    libc::SYS_timerfd_create,
    libc::SYS_timerfd_settime,
    libc::SYS_timerfd_gettime,
    libc::SYS_signalfd,
    libc::SYS_signalfd4,
    libc::SYS_pipe,
    libc::SYS_pipe2,
    libc::SYS_poll,
    libc::SYS_ppoll,
    libc::SYS_select,
    libc::SYS_pselect6,
    // Sockets — needed only because the renderer still fetches
    // directly (pluto-net); the net namespace + broker fetch
    // remove this whole group when they land.
    libc::SYS_socket,
    libc::SYS_socketpair,
    libc::SYS_connect,
    libc::SYS_bind,
    libc::SYS_sendto,
    libc::SYS_recvfrom,
    libc::SYS_sendmsg,
    libc::SYS_recvmsg,
    // sendmmsg/recvmmsg: batched forms of send/recv, used by
    // glibc's DNS resolver for parallel queries. No new
    // capability beyond sendmsg/recvmsg on already-open sockets.
    libc::SYS_sendmmsg,
    libc::SYS_recvmmsg,
    libc::SYS_shutdown,
    libc::SYS_setsockopt,
    libc::SYS_getsockopt,
    libc::SYS_getsockname,
    libc::SYS_getpeername,
];

fn bpf_stmt(code: u16, k: u32) -> libc::sock_filter {
    libc::sock_filter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

fn bpf_jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
    libc::sock_filter { code, jt, jf, k }
}

/// Build the BPF program: arch check (`KILL_PROCESS` on
/// mismatch), then one `JEQ nr -> ALLOW` pair per allowed
/// syscall, then the default action. Pure — unit-tested without
/// ever installing a filter.
fn build_seccomp_filter(default_action: u32) -> Vec<libc::sock_filter> {
    let mut prog = Vec::with_capacity(8 + seccomp_allowed_syscalls().len() * 2);
    // struct seccomp_data: syscall nr at offset 0, arch at 4.
    // Arch check: if the arch MATCHES, skip over the KILL (jt=1);
    // if it does not, fall through into the KILL (jf=0). (Getting
    // jt/jf backwards here kills the process on its very next
    // syscall — the arch always matches on a native build.)
    prog.push(bpf_stmt(
        libc::BPF_LD as u16 | libc::BPF_W as u16 | libc::BPF_ABS as u16,
        4,
    ));
    prog.push(bpf_jump(
        libc::BPF_JMP as u16 | libc::BPF_JEQ as u16 | libc::BPF_K as u16,
        SECCOMP_ARCH,
        1,
        0,
    ));
    prog.push(bpf_stmt(
        libc::BPF_RET as u16 | libc::BPF_K as u16,
        libc::SECCOMP_RET_KILL_PROCESS,
    ));
    prog.push(bpf_stmt(
        libc::BPF_LD as u16 | libc::BPF_W as u16 | libc::BPF_ABS as u16,
        0,
    ));
    for nr in seccomp_allowed_syscalls() {
        prog.push(bpf_jump(
            libc::BPF_JMP as u16 | libc::BPF_JEQ as u16 | libc::BPF_K as u16,
            nr as u32,
            0,
            1,
        ));
        prog.push(bpf_stmt(
            libc::BPF_RET as u16 | libc::BPF_K as u16,
            libc::SECCOMP_RET_ALLOW,
        ));
    }
    prog.push(bpf_stmt(
        libc::BPF_RET as u16 | libc::BPF_K as u16,
        default_action,
    ));
    prog
}

/// `SIGSYS` handler for `PLUTO_SANDBOX_TRAP` mode: print the
/// blocked syscall number to stderr and exit. Async-signal-safe
/// (plain `write` + `_exit`, no allocation, no formatting
/// machinery). The number is read at byte offset 24 of the
/// kernel `siginfo` on 64-bit (signo/errno/code, then the
/// `_sigsys` union: `call_addr` pointer at 16, syscall at 24).
/// Diagnostic only — a wrong offset prints a wrong number.
extern "C" fn sigsys_handler(
    _sig: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    let nr = unsafe { std::ptr::read_unaligned((info as *const u8).add(24) as *const i32) };
    let mut buf = [0u8; 96];
    let mut pos = 0usize;
    let prefix = b"[pluto-sandbox] SIGSYS: seccomp blocked syscall ";
    buf[..prefix.len()].copy_from_slice(prefix);
    pos += prefix.len();
    let mut n = nr as i64;
    if n < 0 {
        buf[pos] = b'-';
        pos += 1;
        n = -n;
    }
    let mut digits = [0u8; 12];
    let mut dlen = 0usize;
    if n == 0 {
        digits[dlen] = b'0';
        dlen += 1;
    } else {
        while n > 0 {
            digits[dlen] = b'0' + (n % 10) as u8;
            dlen += 1;
            n /= 10;
        }
    }
    for i in (0..dlen).rev() {
        buf[pos] = digits[i];
        pos += 1;
    }
    buf[pos] = b'\n';
    pos += 1;
    unsafe {
        libc::write(2, buf.as_ptr() as *const libc::c_void, pos);
        libc::_exit(127);
    }
}

fn install_sigsys_handler() -> io::Result<()> {
    let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
    sa.sa_sigaction = sigsys_handler as *const () as usize;
    sa.sa_flags = libc::SA_SIGINFO;
    unsafe { libc::sigemptyset(&mut sa.sa_mask) };
    let ret = unsafe { libc::sigaction(libc::SIGSYS, &sa, std::ptr::null_mut()) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// How a tab close ended for its renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// The renderer exited on its own after `Shutdown`.
    Clean,
    /// The renderer had to be SIGKILLed (hung, or already dead in
    /// a way `Shutdown` could not reach).
    Killed,
}

/// The browser side of one per-tab renderer process: the child
/// process plus the IPC transport to it. One `RendererProcess` is
/// one tab's renderer; dropping it without `close`/`kill` leaks
/// nothing (the child is killed on drop — see `Drop` below), but
/// `close` is the clean tab-close path.
pub struct RendererProcess {
    child: Child,
    transport: SocketTransport,
}

impl RendererProcess {
    /// Spawn a renderer child from the binary at `exe` (the same
    /// binary, re-executed — see the module docs for the fd
    /// mechanism). Handshakes before returning; a child that fails
    /// to handshake is killed and reaped, and the error returned.
    /// `tab_id` is the shell-assigned tab id (Stage 11 Rec 4),
    /// passed to the child as `--tab-id <n>`.
    pub async fn spawn(exe: &Path, tab_id: u64) -> io::Result<Self> {
        Self::spawn_with_env(exe, &[], tab_id).await
    }

    /// [`RendererProcess::spawn`] with extra environment variables
    /// set on the child only. Used by tests for the
    /// [`TEST_GARBAGE_ENV`] / [`TEST_HANG_ENV`] hooks; production
    /// callers use [`RendererProcess::spawn`].
    pub async fn spawn_with_env(
        exe: &Path,
        envs: &[(&str, &str)],
        tab_id: u64,
    ) -> io::Result<Self> {
        Self::spawn_inner(exe, envs, &[], tab_id).await
    }

    /// Spawn with the `--no-sandbox` escape hatch: the child
    /// skips the whole sandbox hook and runs unconfined. UNSAFE,
    /// debugging only — see [`NO_SANDBOX_ARG`]. Never used by
    /// production callers; tests prove the hatch works.
    pub async fn spawn_unsandboxed(exe: &Path, tab_id: u64) -> io::Result<Self> {
        Self::spawn_inner(exe, &[], &[NO_SANDBOX_ARG], tab_id).await
    }

    /// The shared spawn implementation: `extra_args` are appended
    /// after `--renderer <fd> --tab-id <n>` on the child command line.
    async fn spawn_inner(
        exe: &Path,
        envs: &[(&str, &str)],
        extra_args: &[&str],
        tab_id: u64,
    ) -> io::Result<Self> {
        let (parent_end, child_end) = StdUnixStream::pair()?;
        let child_fd = child_end.as_raw_fd();
        // The pair is created CLOEXEC: clear the flag on the child
        // end so it survives exec. Verified: `std::process::Command`
        // keeps a non-CLOEXEC fd open across exec.
        clear_cloexec(child_fd)?;
        let mut command = Command::new(exe);
        command
            .arg(RENDERER_ARG)
            .arg(child_fd.to_string())
            .arg(TAB_ID_ARG)
            .arg(tab_id.to_string());
        for arg in extra_args {
            command.arg(arg);
        }
        for (key, value) in envs {
            command.env(key, value);
        }
        // The child holds its own copy of the fd after fork; the
        // parent's copy can be dropped once spawn returns.
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(e) => {
                drop(child_end);
                return Err(e);
            }
        };
        drop(child_end);
        let mut transport = match SocketTransport::from_std(parent_end) {
            Ok(transport) => transport,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        match tokio::time::timeout(HANDSHAKE_TIMEOUT, transport.handshake()).await {
            Ok(Ok(())) => Ok(Self { child, transport }),
            Ok(Err(e)) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "renderer handshake timed out — child killed",
                ))
            }
        }
    }

    /// The child's pid, for logging (and for the kill path's
    /// diagnostics).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Whether the child is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Send one command to the renderer.
    pub async fn send(&mut self, message: &BrowserToRenderer) -> io::Result<()> {
        self.transport.send_message(message).await
    }

    /// Receive the next event from the renderer, re-minting opaque
    /// origins on receipt (see `pluto_ipc::transport`).
    ///
    /// A malformed message — undecodable bytes, a stale protocol
    /// version, an over-cap frame — is treated as COMPROMISED: the
    /// child is SIGKILLed here, before the error is returned, and
    /// is never recovered. A clean EOF (`Ok(None)`) is the renderer
    /// exiting on its own, not a compromise; other I/O errors (a
    /// socket that died with the child) are returned as-is.
    pub async fn recv(&mut self) -> io::Result<Option<RendererToBrowser>> {
        match recv_renderer_message(&mut self.transport).await {
            Ok(message) => Ok(message),
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                eprintln!(
                    "[pluto] renderer pid {} sent a malformed IPC message ({e}) — treating as compromised, killing it",
                    self.pid()
                );
                let _ = self.kill();
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// SIGKILL the child and reap it. Returns the exit status.
    /// Killing an already-dead child is an error from `Child::kill`;
    /// callers that do not care should ignore it (see `close`).
    pub fn kill(&mut self) -> io::Result<std::process::ExitStatus> {
        self.child.kill()?;
        self.child.wait()
    }

    /// Close the tab: ask the renderer to shut down, wait up to
    /// [`SHUTDOWN_GRACE`] for a clean exit, then SIGKILL it if it
    /// is still alive (a hung tab closes cleanly this way). Always
    /// reaps the child. Consumes the handle — there is no renderer
    /// left to talk to afterwards.
    pub async fn close(self) -> Closed {
        self.close_with_grace(SHUTDOWN_GRACE).await
    }

    /// [`RendererProcess::close`] with an explicit grace period.
    /// Tests use a short grace so a hung-renderer close stays fast.
    pub async fn close_with_grace(mut self, grace: Duration) -> Closed {
        // Ask politely first. A hung renderer never reads the
        // socket, so the send may sit in the socket buffer — that
        // is fine, it still counts as asked.
        let _ = tokio::time::timeout(Duration::from_secs(1), self.send(&BrowserToRenderer::Shutdown)).await;
        let deadline = tokio::time::Instant::now() + grace;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return Closed::Clean,
                Ok(None) => {
                    if tokio::time::Instant::now() >= deadline {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                Err(_) => break,
            }
        }
        let _ = self.kill();
        Closed::Killed
    }
}

impl Drop for RendererProcess {
    /// Never leak a renderer: a handle dropped without `close` or
    /// `kill` still SIGKILLs the child (and reaps it). The tab-close
    /// path should still call `close` for the graceful attempt.
    fn drop(&mut self) {
        if self.is_alive() {
            let _ = self.kill();
        }
    }
}

/// Clear `CLOEXEC` on `fd` so it survives `exec` (see the module
/// docs). A no-op when the flag is already clear.
fn clear_cloexec(fd: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::FD_CLOEXEC != 0 {
        let ret = unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// The `--renderer` child entry point: run the renderer loop
/// headless over the inherited IPC socket. Returns the process exit
/// code. Called from `main()` before any GUI setup — this path
/// never creates a window, never creates a wgpu device, and never
/// touches libcosmic. It only parses, lays out, runs JS, and builds
/// display lists, exactly like the in-process worker thread, with
/// the socket transport in place of the iced command channel.
pub fn run_renderer_child(fd: RawFd, tab_id: u64) -> i32 {
    let stream = unsafe { StdUnixStream::from_raw_fd(fd) };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("[pluto-renderer] could not build a runtime: {e}");
            return 1;
        }
    };
    runtime.block_on(async move {
        let mut transport = match SocketTransport::from_std(stream) {
            Ok(transport) => transport,
            Err(e) => {
                eprintln!("[pluto-renderer] IPC socket failed: {e}");
                return 1;
            }
        };
        match tokio::time::timeout(HANDSHAKE_TIMEOUT, transport.handshake()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[pluto-renderer] handshake failed: {e}");
                return 1;
            }
            Err(_) => {
                eprintln!("[pluto-renderer] handshake timed out");
                return 1;
            }
        }
        // TEST HOOKS (see the env consts above): deliberately
        // misbehave so the browser-side kill paths are testable
        // against a real child. Neither hook can be triggered by a
        // page — only by the spawning parent's environment.
        if std::env::var_os(TEST_HANG_ENV).is_some() {
            eprintln!("[pluto-renderer] test hook: hanging forever");
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
        if std::env::var_os(TEST_GARBAGE_ENV).is_some() {
            eprintln!("[pluto-renderer] test hook: sending a malformed frame");
            let mut bad = pluto_ipc::PROTOCOL_VERSION.to_le_bytes().to_vec();
            bad.extend_from_slice(b"garbage-garbage-garbage");
            if transport.send_frame(&bad).await.is_err() {
                return 1;
            }
        }
        // SANDBOX HOOK (Stage 9 Rec 10): install the sandbox here
        // — after the handshake, before the renderer loop starts.
        // Best-effort layers (see install_sandbox_hook): a failed
        // layer degrades, it never fails the spawn. The IPC fd is
        // passed so the FD-close layer knows what to keep.
        // Stage 13 Rec 12a: report the sandbox layers to the
        // browser before the first navigation (a failed send
        // means the browser is gone — the worker loop below will
        // exit on its own).
        let layers = install_sandbox_hook(fd);
        if transport
            .send_message(&pluto_ipc::RendererToBrowser::SandboxReport { layers })
            .await
            .is_err()
        {
            return 1;
        }
        crate::renderer::run_worker_transport(transport, tab_id).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_with_missing_binary_errors() {
        assert!(
            RendererProcess::spawn(Path::new("/nonexistent/pluto-browser"), 0)
                .await
                .is_err(),
            "spawning a missing binary fails instead of hanging"
        );
    }

    // The real-child spawn tests live in
    // `tests/renderer_process_spawn.rs`: only an integration test
    // gets `CARGO_BIN_EXE_pluto-browser`, and `cargo test` builds
    // the binary for integration tests but not for unit tests, so a
    // unit test here could never find the child binary.

    // Sandbox tests: the hook itself must NEVER run in the test
    // process (unshare/chroot/seccomp would confine the test
    // runner). Only the pure parts are unit-tested here; the live
    // hook is exercised by real `--renderer` children in the
    // integration tests.

    #[test]
    fn seccomp_filter_has_arch_check_then_allow_list_then_default() {
        let prog = build_seccomp_filter(libc::SECCOMP_RET_KILL_PROCESS);
        assert!(prog.len() > 8, "filter is non-trivial");
        // First instruction loads seccomp_data.arch (offset 4).
        assert_eq!(prog[0].code, libc::BPF_LD as u16 | libc::BPF_W as u16 | libc::BPF_ABS as u16);
        assert_eq!(prog[0].k, 4);
        // Arch mismatch returns KILL_PROCESS (third instruction).
        assert_eq!(prog[2].code, libc::BPF_RET as u16 | libc::BPF_K as u16);
        assert_eq!(prog[2].k, libc::SECCOMP_RET_KILL_PROCESS);
        // The arch-check jump skips the KILL when the arch matches
        // (jt=1) and falls into it when it does not (jf=0). The
        // reverse kills the process on its first post-filter
        // syscall — the arch always matches on a native build.
        assert_eq!(prog[1].jt, 1);
        assert_eq!(prog[1].jf, 0);
        // Last instruction is the default action.
        let last = prog.last().unwrap();
        assert_eq!(last.code, libc::BPF_RET as u16 | libc::BPF_K as u16);
        assert_eq!(last.k, libc::SECCOMP_RET_KILL_PROCESS);
        // The default action is honoured when requested differently.
        let log_prog = build_seccomp_filter(libc::SECCOMP_RET_LOG);
        assert_eq!(log_prog.last().unwrap().k, libc::SECCOMP_RET_LOG);
    }

    #[test]
    fn seccomp_allow_list_covers_the_renderer_loop_and_never_ptrace_or_exec() {
        let allowed = seccomp_allowed_syscalls();
        // What the post-hook child provably needs.
        for needed in [
            libc::SYS_read,
            libc::SYS_write,
            libc::SYS_mmap,
            libc::SYS_mprotect,
            libc::SYS_brk,
            libc::SYS_clone,
            libc::SYS_clone3,
            libc::SYS_futex,
            libc::SYS_clock_gettime,
            libc::SYS_epoll_create1,
            libc::SYS_epoll_wait,
            libc::SYS_eventfd2,
            libc::SYS_openat,
            libc::SYS_getdents64,
            libc::SYS_statx,
            libc::SYS_getrandom,
            libc::SYS_exit_group,
            libc::SYS_restart_syscall,
            libc::SYS_rt_sigaction,
            libc::SYS_prctl,
            libc::SYS_arch_prctl,
            libc::SYS_socket,
            libc::SYS_connect,
            libc::SYS_mkdir,
            libc::SYS_rename,
            libc::SYS_sendmmsg,
        ] {
            assert!(
                allowed.contains(&needed),
                "allow-list is missing syscall {needed}"
            );
        }
        // Never allowed, by design. (mkdir/rename are NOT in this
        // list: the persistence layer provably needs them — see
        // the allow-list comments — and under the chroot they can
        // only touch the empty tmpfs root.)
        for forbidden in [
            libc::SYS_ptrace,
            libc::SYS_execve,
            libc::SYS_execveat,
            libc::SYS_fork,
            libc::SYS_vfork,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_chroot,
            libc::SYS_pivot_root,
            libc::SYS_unshare,
            libc::SYS_bpf,
            libc::SYS_perf_event_open,
            libc::SYS_io_uring_setup,
            libc::SYS_keyctl,
            libc::SYS_process_vm_readv,
            libc::SYS_process_vm_writev,
            libc::SYS_setrlimit,
            libc::SYS_reboot,
            libc::SYS_init_module,
            libc::SYS_delete_module,
            libc::SYS_unlink,
        ] {
            assert!(
                !allowed.contains(&forbidden),
                "allow-list must never contain syscall {forbidden}"
            );
        }
    }

    #[test]
    fn no_sandbox_arg_detection() {
        assert!(args_disable_sandbox(
            ["pluto-browser".to_string(), "--renderer".to_string(), "3".to_string(), "--no-sandbox".to_string()]
        ));
        assert!(!args_disable_sandbox(
            ["pluto-browser".to_string(), "--renderer".to_string(), "3".to_string()]
        ));
        assert!(!args_disable_sandbox(["pluto-browser".to_string()]));
    }

    #[test]
    fn sandbox_rlimits_are_generous() {
        assert!(SANDBOX_RLIMIT_AS >= 1 << 30, "address-space cap is generous");
        assert!(SANDBOX_RLIMIT_CPU_SECS >= 60, "CPU cap is generous");
        assert!(SANDBOX_RLIMIT_NOFILE >= 1024, "fd cap is generous");
        assert!(SANDBOX_RLIMIT_NPROC >= 1024, "process cap is generous");
        assert!(SANDBOX_RLIMIT_FSIZE >= 1 << 20, "file-size cap is generous");
    }
}
