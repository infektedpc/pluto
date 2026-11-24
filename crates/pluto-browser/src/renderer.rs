//! The per-tab renderer worker (Stage 4 #2): the page pipeline
//! (fetch -> parse -> layout -> display list) runs on a worker task,
//! not the UI thread.
//!
//! The worker is an iced subscription stream (the documented worker
//! pattern): it creates the command channel, hands the sender to the
//! shell as its first event, and then loops on commands.
//!
//! Stage 5 changed how the fetch runs. The old worker awaited each
//! navigation inline in the command loop, so a slow fetch blocked
//! resizes and a second navigation queued behind the first. Now the
//! loop is a `tokio::select!` over commands and navigation outcomes:
//! each navigation runs as a spawned task (fetch via the worker's
//! `ResourceLoader` — one loader per worker, one shared hyper client
//! for the browser — then parse), and its outcome comes back over a
//! channel. A new `CommitNavigation` cancels the previous
//! navigation's [`CancellationToken`] (Stage 5 #5) and bumps the
//! navigation epoch; a stale outcome — a fetch that finished its
//! parse after being cancelled — is discarded by epoch. Layout
//! happens in the loop, at the *current* viewport, so a resize
//! mid-fetch is honoured by the list that lands. The old page stays
//! visible until the new list arrives (Stage 4 #2, unchanged).
//!
//! The worker speaks the [`pluto_ipc`] protocol throughout: it
//! receives [`BrowserToRenderer`] commands and emits
//! [`RendererToBrowser`] events. The display list crosses to the
//! shell as postcard bytes plus its resource table — the same bytes
//! that cross IPC at M5, when this worker task is promoted to a
//! process. The worker never touches a socket, the disk, or the GPU
//! directly: it asks the shell for nothing and draws nothing; it
//! only builds lists.

use crate::page::{self, Document};
use cosmic::iced::futures::SinkExt;
use cosmic::iced::futures::channel::mpsc;
use pluto_ipc::{BrowserToRenderer, CommandReceiver, RendererToBrowser};
use pluto_net::ResourceLoader;
use pluto_paint::{DisplayList, ResourceTable};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// The subscription that runs the worker. The shell's `Application`
/// wires this in via `fn subscription`.
pub fn renderer_subscription() -> cosmic::iced::Subscription<crate::Message> {
    cosmic::iced::Subscription::run(worker_stream)
}

/// The worker stream: create the command channel, hand its sender
/// to the shell, then run the worker until it ends.
fn worker_stream() -> impl cosmic::iced::futures::stream::Stream<Item = crate::Message> {
    cosmic::iced::stream::channel(100, async |mut output| {
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let _ = output.send(crate::Message::RendererReady(cmd_tx)).await;
        run_worker(cmd_rx, output).await;
    })
}

/// The worker's state: the retained document, the layout viewport,
/// the frame epoch, the loader, and the in-flight navigation
/// (its epoch and its cancellation token).
struct WorkerState {
    document: Option<Document>,
    viewport: (f32, f32),
    epoch: u64,
    loader: ResourceLoader,
    nav_epoch: u64,
    nav_cancel: Option<CancellationToken>,
}

/// What a spawned navigation task reports back: the epoch it was
/// started under, and the fetch+parse outcome.
type NavOutcome = Result<(Document, String), (String, String)>;

/// Run the worker: read commands, run the pipeline, emit events.
/// Ends when the command channel closes or `Shutdown` arrives.
pub async fn run_worker(mut commands: CommandReceiver, mut events: mpsc::Sender<crate::Message>) {
    let mut state = WorkerState {
        document: None,
        viewport: page::DEFAULT_VIEWPORT,
        epoch: 0,
        loader: ResourceLoader::new(),
        nav_epoch: 0,
        nav_cancel: None,
    };
    // Navigation outcomes come back over this channel, so the loop
    // never awaits a fetch inline.
    let (nav_tx, mut nav_rx) = tokio::sync::mpsc::channel::<(u64, NavOutcome)>(4);
    loop {
        tokio::select! {
            cmd = commands.recv() => {
                match cmd {
                    None => break,
                    Some(BrowserToRenderer::Shutdown) => {
                        // Cancel any in-flight fetch before going
                        // away, so its task ends instead of
                        // finishing a load nobody will see.
                        if let Some(token) = state.nav_cancel.take() {
                            token.cancel();
                        }
                        break;
                    }
                    Some(BrowserToRenderer::CommitNavigation { url }) => {
                        start_navigation(&mut state, &nav_tx, url);
                    }
                    Some(BrowserToRenderer::Resize { width, height }) => {
                        resize(&mut state, &mut events, width, height).await;
                    }
                    Some(BrowserToRenderer::ResourceEvent(event)) => {
                        // The M5 shape: the shell streams fetch
                        // events back to the renderer. Today the
                        // worker fetches through its own loader,
                        // which consumes the events internally, so
                        // nothing arrives here — ignore it if it
                        // ever does.
                        eprintln!("[pluto] ignoring ResourceEvent from shell: {event:?}");
                    }
                    Some(BrowserToRenderer::InputEvent(event)) => {
                        // Hit-testing and focus land in M2; the worker takes
                        // no action on input yet.
                        eprintln!("[pluto] input event ignored (M2): {event:?}");
                    }
                }
            }
            Some((epoch, outcome)) = nav_rx.recv() => {
                if epoch != state.nav_epoch {
                    // Stale: superseded by a newer navigation.
                    eprintln!("[pluto] discarding stale navigation outcome (epoch {epoch})");
                    continue;
                }
                state.nav_cancel = None;
                handle_nav_outcome(&mut state, &mut events, outcome).await;
            }
        }
        // The shell is gone: there is nobody left to report to.
        if events.is_closed() {
            break;
        }
    }
}

/// Start a navigation: cancel the previous one (its token, and its
/// epoch — so a late outcome is discarded), then spawn the
/// fetch+parse task. The fetch runs on the executor; parse runs
/// inside the same task (it is fast; layout stays on the blocking
/// pool in `layout_and_submit`).
fn start_navigation(
    state: &mut WorkerState,
    nav_tx: &tokio::sync::mpsc::Sender<(u64, NavOutcome)>,
    url: String,
) {
    if let Some(token) = state.nav_cancel.take() {
        token.cancel();
    }
    state.nav_epoch += 1;
    let epoch = state.nav_epoch;
    let token = CancellationToken::new();
    state.nav_cancel = Some(token.clone());
    let loader = state.loader.clone();
    let nav_tx = nav_tx.clone();
    eprintln!("[pluto] worker fetching {url} (epoch {epoch})");
    tokio::spawn(async move {
        let outcome = match page::load_page_cancellable(&loader, url.clone(), token).await {
            Ok((document, final_url)) => Ok((document, final_url)),
            Err(reason) => Err((url, reason)),
        };
        let _ = nav_tx.send((epoch, outcome)).await;
    });
}

/// A navigation finished: commit the document and lay it out at the
/// current viewport (a resize mid-fetch is honoured here), or render
/// the error page through the same pipeline and report the reason.
async fn handle_nav_outcome(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
    outcome: NavOutcome,
) {
    match outcome {
        Ok((document, final_url)) => {
            eprintln!("[pluto] worker fetched {final_url}");
            if !emit(
                events,
                RendererToBrowser::DidCommit {
                    url: final_url.clone(),
                    status: format!("Loaded {final_url}"),
                },
            )
            .await
            {
                return;
            }
            state.document = Some(document);
            layout_and_submit(state, events).await;
        }
        Err((url, reason)) => {
            eprintln!("[pluto] worker fetch failed: {reason}");
            let html = page::error_document(&url, &reason);
            let viewport = state.viewport;
            match tokio::task::spawn_blocking(move || page::build_display_list(&html, viewport))
                .await
            {
                Ok((display, document)) => {
                    state.document = Some(document);
                    if !submit(state, events, display).await {
                        return;
                    }
                }
                Err(join_err) => {
                    let _ = emit(
                        events,
                        RendererToBrowser::RendererCrashed {
                            reason: format!("error-page layout task failed: {join_err}"),
                        },
                    )
                    .await;
                    return;
                }
            }
            let _ = emit(events, RendererToBrowser::LoadFailed { url, reason }).await;
        }
    }
}

/// Emit one renderer event to the shell. Returns false when the
/// shell is gone (the subscription's receiver dropped).
async fn emit(events: &mut mpsc::Sender<crate::Message>, event: RendererToBrowser) -> bool {
    events
        .send(crate::Message::RendererEvent(event))
        .await
        .is_ok()
}

/// Submit a display list to the shell, stamping the next epoch.
/// No images in M1, so the resource table travels empty.
async fn submit(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
    display: Arc<DisplayList>,
) -> bool {
    state.epoch += 1;
    emit(
        events,
        RendererToBrowser::SubmitDisplayList {
            epoch: state.epoch,
            bytes: display.to_bytes(),
            resources: ResourceTable::new(),
        },
    )
    .await
}

/// Lay the retained document out at the worker's viewport on the
/// blocking pool and submit the result. The document is moved into
/// the blocking task and returned with the list, so no arena copy
/// is needed. A failed layout task is reported as a crash; the
/// document is lost with it, so later resizes do nothing until the
/// next navigation.
async fn layout_and_submit(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
) -> bool {
    let Some(document) = state.document.take() else {
        return true;
    };
    let viewport = state.viewport;
    match tokio::task::spawn_blocking(move || {
        let display = page::layout_document(&document, viewport);
        (display, document)
    })
    .await
    {
        Ok((display, document)) => {
            state.document = Some(document);
            submit(state, events, display).await
        }
        Err(join_err) => {
            emit(
                events,
                RendererToBrowser::RendererCrashed {
                    reason: format!("layout task failed: {join_err}"),
                },
            )
            .await
        }
    }
}

/// A resize: record the viewport and re-lay out the retained
/// document. Degenerate sizes carry no layout; a real size arrives
/// when the window is restored.
async fn resize(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
    width: f32,
    height: f32,
) {
    if width < 1.0 || height < 1.0 {
        return;
    }
    if (width, height) == state.viewport {
        return;
    }
    state.viewport = (width, height);
    if state.document.is_some() {
        eprintln!("[pluto] worker viewport resized to {width}x{height}, re-layout");
        layout_and_submit(state, events).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic::iced::futures::stream::StreamExt;
    use pluto_ipc::BrowserToRenderer;
    use pluto_paint::DisplayItem;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
    }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents.</p>
</div>
</body>
</html>"#;

    /// Local server returning the example document, same pattern as
    /// pluto-net's and page's tests. No internet needed (hyper has
    /// no proxy support). `/slow` answers after 500 ms, so a
    /// navigation to it can be cancelled mid-flight.
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/");
                if path == "/slow" {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                let body = EXAMPLE_COM;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// Drive the worker directly (no iced): send commands in, read
    /// `RendererEvent` messages out. Returns the worker task handle
    /// and the channels, so each test drives its own worker.
    fn spawn_worker() -> (
        tokio::task::JoinHandle<()>,
        pluto_ipc::CommandSender,
        mpsc::Receiver<crate::Message>,
    ) {
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (event_tx, event_rx) = mpsc::channel(100);
        let handle = tokio::spawn(run_worker(cmd_rx, event_tx));
        (handle, cmd_tx, event_rx)
    }

    async fn next_event(rx: &mut mpsc::Receiver<crate::Message>) -> RendererToBrowser {
        match rx.next().await.expect("worker emits an event") {
            crate::Message::RendererEvent(event) => event,
            other => panic!("expected a RendererEvent, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_loads_a_page_and_submits_its_display_list() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url: url.clone() })
            .expect("worker takes the command");
        // The fetch commits first, then the display list lands.
        match next_event(&mut event_rx).await {
            RendererToBrowser::DidCommit { url: committed, .. } => {
                assert_eq!(committed, url);
            }
            other => panic!("expected DidCommit first, got {other:?}"),
        }
        let (epoch, bytes) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList {
                epoch,
                bytes,
                resources,
            } => {
                assert!(resources.is_empty(), "no images in M1");
                (epoch, bytes)
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        assert_eq!(epoch, 1, "first frame is epoch 1");
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        assert!(!list.is_empty());
        // The canvas covers the worker's default viewport.
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                height: 768.0,
                ..
            }
        ));
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    #[tokio::test]
    async fn worker_relayouts_on_resize_without_refetching() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let first_epoch = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => epoch,
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        // Resize: a new list, a newer epoch, the canvas at the new
        // size — and no second DidCommit (nothing refetched).
        cmd_tx
            .send(BrowserToRenderer::Resize {
                width: 400.0,
                height: 300.0,
            })
            .expect("worker takes Resize");
        let (epoch, bytes) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, bytes, .. } => (epoch, bytes),
            other => panic!("expected SubmitDisplayList after resize, got {other:?}"),
        };
        assert!(epoch > first_epoch, "resize advances the epoch");
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 400.0,
                height: 300.0,
                ..
            }
        ));
        // A resize to the same size is a no-op: no new list.
        cmd_tx
            .send(BrowserToRenderer::Resize {
                width: 400.0,
                height: 300.0,
            })
            .expect("worker takes Resize");
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    #[tokio::test]
    async fn worker_failed_load_renders_an_error_page() {
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: "ftp://example.com/".to_string(),
            })
            .expect("worker takes the command");
        // The error page arrives as a display list first, then the
        // reason for the status line.
        match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                assert!(!list.is_empty(), "error page still paints");
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
        match next_event(&mut event_rx).await {
            RendererToBrowser::LoadFailed { url, reason } => {
                assert_eq!(url, "ftp://example.com/");
                assert!(!reason.is_empty());
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    #[tokio::test]
    async fn worker_cancels_the_old_navigation_when_a_new_one_commits() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let slow = format!("http://127.0.0.1:{}/slow", addr.port());
        let fast = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url: slow })
            .expect("worker takes the command");
        // Before the slow fetch lands (500 ms), commit the fast one.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url: fast.clone() })
            .expect("worker takes the command");
        // The only DidCommit is the fast URL: the slow navigation
        // was cancelled by token, and any stale outcome it reports
        // is discarded by epoch — the old page never commits.
        match next_event(&mut event_rx).await {
            RendererToBrowser::DidCommit { url: committed, .. } => {
                assert_eq!(committed, fast);
            }
            other => panic!("expected DidCommit for the fast navigation, got {other:?}"),
        }
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }
}
