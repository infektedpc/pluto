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
            state_json: None,
            traversal: false,
            scroll: (0.0, 0.0),
        })
        .await
        .expect("navigation sends");
    // Stage 13 Rec 12a: the child reports its sandbox layers
    // before the first navigation — drain that report (and any
    // heartbeat) and assert it names the load-bearing layers.
    let mut saw_report = false;
    let bytes = loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
            .await
            .expect("reply arrives in time")
            .expect("recv succeeds")
            .expect("a message arrives");
        match msg {
            RendererToBrowser::SandboxReport { layers } => {
                for name in ["user-namespace", "chroot", "seccomp"] {
                    assert!(
                        layers.iter().any(|(n, _)| n == name),
                        "sandbox report names {name}: {layers:?}"
                    );
                }
                saw_report = true;
            }
            RendererToBrowser::SubmitDisplayList { bytes, .. } => break bytes,
            RendererToBrowser::Heartbeat => {}
            RendererToBrowser::RequestResource(request) => {
                // `about:blank` has no fetch: the broker answers
                // it with an Error (`invalid_url`) and the worker
                // renders the error page. The test plays the
                // broker's part for this one request — the real
                // broker path is proven by the no-socket test.
                assert_eq!(request.url, "about:blank", "only the blank page is asked");
                renderer
                    .send(&BrowserToRenderer::ResourceEvent(
                        pluto_ipc::ResourceEvent::Error {
                            request_id: request.request_id,
                            code: "invalid_url".to_string(),
                            reason: "non-http(s) scheme".to_string(),
                        },
                    ))
                    .await
                    .expect("error event sends");
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
    };
    assert!(saw_report, "sandbox report arrived before the first list");
    // The error page arrives as a display list first, then the
    // reason for the status line. A `TitleChanged` for the error
    // page's title may interleave between the two (the worker
    // coalesces it onto the submit) — skip it here.
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
            state_json: None,
            traversal: false,
            scroll: (0.0, 0.0),
        })
        .await
        .expect("navigation sends");
    // Stage 13 Rec 12a: the sandbox report precedes the list.
    let bytes = loop {
        let msg = tokio::time::timeout(Duration::from_secs(20), renderer.recv())
            .await
            .expect("reply arrives in time")
            .expect("recv succeeds")
            .expect("a message arrives");
        match msg {
            RendererToBrowser::SandboxReport { layers } => {
                assert!(
                    layers.iter().any(|(n, ok)| n == "seccomp" && *ok),
                    "seccomp landed under the trap filter: {layers:?}"
                );
            }
            RendererToBrowser::SubmitDisplayList { bytes, .. } => break bytes,
            RendererToBrowser::Heartbeat => {}
            RendererToBrowser::RequestResource(request) => {
                // `about:blank` has no fetch: the broker answers
                // it with an Error (`invalid_url`) and the worker
                // renders the error page (see the headless test).
                assert_eq!(request.url, "about:blank", "only the blank page is asked");
                renderer
                    .send(&BrowserToRenderer::ResourceEvent(
                        pluto_ipc::ResourceEvent::Error {
                            request_id: request.request_id,
                            code: "invalid_url".to_string(),
                            reason: "non-http(s) scheme".to_string(),
                        },
                    ))
                    .await
                    .expect("error event sends");
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
    };
    {
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        assert!(!list.is_empty(), "error page still paints under the sandbox");
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

/// Stage 13 Rec 11: a renderer with `socket`/`connect` removed
/// from its seccomp allow-list still loads a real HTTP page
/// end-to-end — every fetch goes through the browser-side broker
/// over IPC, so the child never opens a socket itself. If the
/// child tried a direct fetch, `socket()` would kill it
/// (KILL_PROCESS) and no `DidCommit` would ever arrive.
#[tokio::test]
async fn no_socket_renderer_loads_real_page_via_broker() {
    use pluto_browser::broker::NetworkBroker;
    use pluto_browser::renderer_process::TEST_NO_SOCKET_ENV;
    use pluto_net::ResourceLoader;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    // A tiny local HTTP server: every path serves the same page.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test server binds");
    let port = listener.local_addr().expect("listener addr").port();
    let body = "<!doctype html><title>Broker Page</title><h1>Broker Page</h1><p>served through the broker</p>";
    tokio::task::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let body = body.to_string();
            tokio::task::spawn(async move {
                // Read the request headers, then answer.
                let mut buf = [0u8; 4096];
                let mut request = Vec::new();
                loop {
                    match stream.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            request.extend_from_slice(&buf[..n]);
                            if request.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    });

    let exe = renderer_exe_for_test();
    let mut renderer =
        RendererProcess::spawn_with_env(&exe, &[(TEST_NO_SOCKET_ENV, "1")], 7)
            .await
            .expect("no-socket renderer child spawns and handshakes");
    assert!(renderer.is_alive(), "child is alive after handshake");

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
    let url = format!("http://127.0.0.1:{port}/");
    renderer
        .send(&BrowserToRenderer::CommitNavigation {
                url: url.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
        .await
        .expect("navigation sends");

    // The test process plays the shell's broker role: serve each
    // RequestResource on the real broker, forward its events to
    // the child, honour cancellations.
    let broker = Arc::new(NetworkBroker::new(Arc::new(ResourceLoader::new())));
    let (broker_tx, mut broker_rx) = tokio::sync::mpsc::channel::<pluto_ipc::ResourceEvent>(64);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut committed = false;
    let mut saw_glyphs = false;
    loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no-socket renderer did not load the page in 30 s (committed={committed}, glyphs={saw_glyphs})"
        );
        tokio::select! {
            msg = renderer.recv() => {
                let Some(msg) = msg.expect("recv succeeds") else {
                    panic!("renderer closed before the page loaded (committed={committed}, glyphs={saw_glyphs})");
                };
                match msg {
                    RendererToBrowser::RequestResource(request) => {
                        let broker = Arc::clone(&broker);
                        let tx = broker_tx.clone();
                        tokio::task::spawn(async move {
                            broker.serve(request, tx).await;
                        });
                    }
                    RendererToBrowser::CancelResource { request_id } => {
                        broker.cancel(request_id);
                    }
                    RendererToBrowser::DidCommit { url: committed_url, .. } => {
                        assert_eq!(committed_url, url, "commit is for the broker-served page");
                        committed = true;
                    }
                    RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                        if list.0.iter().any(|item| {
                            matches!(item, pluto_paint::DisplayItem::Glyphs { glyphs, .. } if !glyphs.is_empty())
                        }) {
                            saw_glyphs = true;
                        }
                    }
                    RendererToBrowser::RendererCrashed { reason } => {
                        panic!("no-socket renderer crashed: {reason}");
                    }
                    _ => {}
                }
                if committed && saw_glyphs {
                    break;
                }
            }
            event = broker_rx.recv() => {
                let Some(event) = event else { continue };
                renderer
                    .send(&BrowserToRenderer::ResourceEvent(event))
                    .await
                    .expect("ResourceEvent sends");
            }
        }
    }
    assert!(committed, "DidCommit arrived for the broker-served page");
    assert!(saw_glyphs, "a display list with glyphs arrived");
    assert!(renderer.is_alive(), "child survived the broker-mode load");
    assert_eq!(
        renderer.close().await,
        Closed::Clean,
        "a no-socket renderer exits cleanly on Shutdown"
    );
}
