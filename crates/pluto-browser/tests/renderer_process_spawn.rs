//! Real `--renderer` child tests (Stage 9 Rec 9).
//!
//! These live in an integration test — not in the unit-test module —
//! because only integration tests get `CARGO_BIN_EXE_pluto-browser`,
//! and cargo builds the binary for integration tests. A unit test
//! could never find the child binary under a plain `cargo test`.

use pluto_browser::renderer_process::{
    Closed, RendererProcess, NO_SANDBOX_ENV, SANDBOX_TRAP_ENV, TEST_GARBAGE_ENV, TEST_HANG_ENV,
};
use pluto_ipc::{BrowserToRenderer, RendererToBrowser};
use pluto_paint::DisplayList;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

/// The renderer binary under test. Cargo sets this for integration
/// tests and builds the binary first.
fn renderer_exe_for_test() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pluto-browser"))
}

#[tokio::test]
async fn spawned_renderer_child_exchanges_messages_headless() {
    let exe = renderer_exe_for_test();
    assert!(
        exe.exists(),
        "renderer binary exists at {} (cargo builds it for integration tests)",
        exe.display()
    );
    let mut renderer = RendererProcess::spawn(&exe, 7)
        .await
        .expect("renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive after handshake");
    // No window, no wgpu: the child only answers protocol. (The
    // sandbox has no display server at all — a child that tried to
    // open a window would die here instead.)
    renderer
        .send(&BrowserToRenderer::Resize {
            width: 800.0,
            height: 600.0,
        })
        .await
        .expect("resize sends");
    renderer
        .send(&BrowserToRenderer::SetVisible { visible: true })
        .await
        .expect("visibility sends");
    // `about:blank` has no fetch — pluto-net rejects non-http(s)
    // schemes before any network — so this exercises the full
    // pipeline (load → error page → display list) with zero network
    // access.
    renderer
        .send(&BrowserToRenderer::CommitNavigation {
            url: "about:blank".to_string(),
        })
        .await
        .expect("navigation sends");
    // The error page arrives as a display list first, then the
    // reason for the status line. A `TitleChanged` for the error
    // page's title may interleave between the two (the worker
    // coalesces it onto the submit) — skip it here.
    let first = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
        .await
        .expect("first reply arrives in time")
        .expect("recv succeeds")
        .expect("a message arrives");
    let bytes = match first {
        RendererToBrowser::SubmitDisplayList { bytes, .. } => bytes,
        other => panic!("expected SubmitDisplayList first, got {other:?}"),
    };
    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
    assert!(!list.is_empty(), "error page still paints");
    let mut second;
    loop {
        second = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
            .await
            .expect("second reply arrives in time")
            .expect("recv succeeds")
            .expect("a message arrives");
        if !matches!(second, RendererToBrowser::TitleChanged { .. }) {
            break;
        }
    }
    match second {
        RendererToBrowser::LoadFailed { url, reason } => {
            assert_eq!(url, "about:blank");
            assert!(!reason.is_empty());
        }
        other => panic!("expected LoadFailed, got {other:?}"),
    }
    assert!(renderer.is_alive(), "child stays alive after the exchange");
    assert_eq!(
        renderer.close().await,
        Closed::Clean,
        "a healthy renderer exits cleanly on Shutdown"
    );
}

#[tokio::test]
async fn malformed_message_from_renderer_kills_it_never_recovers() {
    let exe = renderer_exe_for_test();
    let mut renderer = RendererProcess::spawn_with_env(&exe, &[(TEST_GARBAGE_ENV, "1")], 7)
        .await
        .expect("renderer child spawns and handshakes");
    // The child sends one garbage frame after the handshake. The
    // browser must treat it as compromised and kill it — never
    // recover it, never read another message from it.
    let err = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
        .await
        .expect("recv returns in time")
        .expect_err("malformed message errors");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        !renderer.is_alive(),
        "the compromised renderer was killed, not recovered"
    );
    // And nothing further ever arrives from it.
    assert!(
        !matches!(renderer.recv().await, Ok(Some(_))),
        "no message ever follows a compromised renderer"
    );
}

#[tokio::test]
async fn hung_renderer_killed_on_tab_close() {
    let exe = renderer_exe_for_test();
    let mut renderer = RendererProcess::spawn_with_env(&exe, &[(TEST_HANG_ENV, "1")], 7)
        .await
        .expect("renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive before close");
    let started = std::time::Instant::now();
    // A short grace keeps the test fast; production `close` uses
    // SHUTDOWN_GRACE (5 s).
    let closed = renderer.close_with_grace(Duration::from_secs(1)).await;
    assert_eq!(
        closed,
        Closed::Killed,
        "a hung renderer is SIGKILLed on tab close"
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "tab close returns promptly even for a hung renderer"
    );
}

/// Stage 9 Rec 10: the sandboxed child runs the full exchange
/// with the seccomp filter in TRAP diagnostic mode
/// (`PLUTO_SANDBOX_TRAP=1`): any syscall missing from the
/// allow-list delivers SIGSYS to a handler that prints the
/// syscall number to stderr and exits, instead of a silent dead
/// tab. Passing this test proves the allow-list covers
/// everything the worker does post-hook (threads, font scan,
/// layout, IPC) — the same code path as the production
/// KILL_PROCESS filter, one return action apart.
#[tokio::test]
async fn sandboxed_renderer_exchange_under_trap_diagnostics() {
    let exe = renderer_exe_for_test();
    let mut renderer = RendererProcess::spawn_with_env(&exe, &[(SANDBOX_TRAP_ENV, "1")], 7)
        .await
        .expect("sandboxed renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive after handshake");
    renderer
        .send(&BrowserToRenderer::Resize {
            width: 800.0,
            height: 600.0,
        })
        .await
        .expect("resize sends");
    renderer
        .send(&BrowserToRenderer::CommitNavigation {
            url: "about:blank".to_string(),
        })
        .await
        .expect("navigation sends");
    let first = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
        .await
        .expect("first reply arrives in time")
        .expect("recv succeeds")
        .expect("a message arrives");
    match first {
        RendererToBrowser::SubmitDisplayList { bytes, .. } => {
            let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
            assert!(!list.is_empty(), "error page still paints under the sandbox");
        }
        other => panic!("expected SubmitDisplayList first, got {other:?}"),
    }
    assert!(renderer.is_alive(), "child stays alive after the exchange");
    assert_eq!(
        renderer.close().await,
        Closed::Clean,
        "a sandboxed renderer exits cleanly on Shutdown"
    );
}

/// Stage 9 Rec 10: the `--no-sandbox` escape hatch works — a
/// child spawned unsandboxed still runs the normal exchange.
/// (The hatch is UNSAFE and debugging-only; this test proves the
/// plumbing, not that it is safe to ship.)
#[tokio::test]
async fn no_sandbox_hatch_spawns_working_renderer() {
    let exe = renderer_exe_for_test();
    let mut renderer = RendererProcess::spawn_unsandboxed(&exe, 7)
        .await
        .expect("unsandboxed renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive after handshake");
    renderer
        .send(&BrowserToRenderer::Resize {
            width: 800.0,
            height: 600.0,
        })
        .await
        .expect("resize sends");
    assert_eq!(
        renderer.close().await,
        Closed::Clean,
        "an unsandboxed renderer exits cleanly on Shutdown"
    );
}

/// Stage 9 Rec 10: the `PLUTO_NO_SANDBOX=1` env form of the
/// escape hatch works the same way as the `--no-sandbox` flag.
#[tokio::test]
async fn no_sandbox_env_hatch_spawns_working_renderer() {
    let exe = renderer_exe_for_test();
    let mut renderer = RendererProcess::spawn_with_env(&exe, &[(NO_SANDBOX_ENV, "1")], 7)
        .await
        .expect("unsandboxed renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive after handshake");
    assert_eq!(
        renderer.close().await,
        Closed::Clean,
        "an unsandboxed renderer exits cleanly on Shutdown"
    );
}
