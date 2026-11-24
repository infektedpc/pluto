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
//! Two threads live behind the worker task, because two of its values
//! are `!Send` and the iced subscription future must stay `Send`:
//!
//! * Navigation tasks run on a dedicated OS thread with a
//!   current-thread tokio runtime. The streaming parser
//!   (`pluto_html::Parser`, html5ever `Tokenizer`/`TreeBuilder` full
//!   of `Cell`/`RefCell` and `NonAtomic` tendrils) is single-threaded
//!   by construction and cannot be held across an await in a
//!   `tokio::spawn`d future — so the whole load (fetch + streaming
//!   parse + subresource fetches) runs on that thread, and only the
//!   `Send` outcome crosses back, via `blocking_send`.
//! * Layout runs on a dedicated layout-actor thread ([`LayoutJob`]).
//!   The [`LayoutTree`](pluto_layout::LayoutTree) owns a
//!   `cosmic-text::FontSystem`, which is `!Send`, so the tree can
//!   never be held across an await in the worker task, nor moved
//!   through `spawn_blocking` (whose outputs must be `Send`). The
//!   actor owns the tree (and the document) on its thread for the
//!   tab's lifetime; the worker sends it jobs and gets `Send`
//!   results back (display lists and hit-test answers). Container scrolling is cheap there
//!   (`set_scroll_offset` + re-collect, no re-layout); a resize is a
//!   rebuild job.
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
use pluto_html::NodeId;
use pluto_ipc::{BrowserToRenderer, CommandReceiver, InputEvent, RendererToBrowser};
use pluto_layout::{ElementState, LayoutTree};
use pluto_net::ResourceLoader;
use pluto_paint::{DisplayList, ResourceTable};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::oneshot;
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

/// A job for the layout-actor thread, which owns the `!Send`
/// [`LayoutTree`] and the current [`Document`] on its thread for the
/// tab's lifetime. Every job carries a oneshot for its `Send` answer;
/// a dropped reply (actor gone) surfaces as a worker crash, never a
/// hang — the worker always awaits with the reply channel, and the
/// actor exits when the job channel closes (worker end / `Stop`).
enum LayoutJob {
    /// (Re-)lay out at `viewport` with `state`, re-applying the
    /// retained `offsets` after the rebuild. `document: Some` replaces
    /// the actor's document (a new navigation / error page);
    /// `None` re-lays out the retained one (a resize).
    Layout {
        document: Box<Option<Document>>,
        viewport: (f32, f32),
        state: ElementState,
        offsets: HashMap<NodeId, (f32, f32)>,
        reply: oneshot::Sender<Arc<DisplayList>>,
    },
    /// Scroll one container: `set_scroll_offset` + re-collect, no
    /// re-layout. `None` when the actor holds no tree.
    Scroll {
        id: NodeId,
        x: f32,
        y: f32,
        reply: oneshot::Sender<Option<Arc<DisplayList>>>,
    },
    /// Hit-test a click (page coordinates) and resolve the nearest
    /// enclosing `<a href>` against the document base URL. `None`
    /// means nothing navigable was hit.
    HitLink {
        x: f32,
        y: f32,
        reply: oneshot::Sender<Option<String>>,
    },
    /// The tree's scroll containers, excluding root/body (whose
    /// overflow propagates to the viewport instead).
    ScrollContainers {
        reply: oneshot::Sender<Vec<NodeId>>,
    },
}

/// The layout actor: owns the tree + document, answers jobs until the
/// channel closes. Runs on its own OS thread (`pluto-layout`), so the
/// `!Send` tree never crosses an await or a thread boundary.
fn layout_thread(jobs: std::sync::mpsc::Receiver<LayoutJob>) {
    let mut tree: Option<LayoutTree> = None;
    let mut document: Option<Document> = None;
    while let Ok(job) = jobs.recv() {
        match job {
            LayoutJob::Layout {
                document: new_document,
                viewport,
                state,
                offsets,
                reply,
            } => {
                if let Some(doc) = *new_document {
                    document = Some(doc);
                }
                let Some(doc) = document.as_ref() else {
                    let _ = reply.send(Arc::new(DisplayList::new()));
                    continue;
                };
                let (mut display, mut new_tree) =
                    page::layout_document_with_state(doc, viewport, &state);
                if !offsets.is_empty() {
                    for (&id, &(x, y)) in &offsets {
                        new_tree.set_scroll_offset(id, x, y);
                    }
                    // Offsets shift already-laid-out descendants
                    // (re-collect, no re-layout), so the list built
                    // before them is stale.
                    display = Arc::new(DisplayList::build(&new_tree));
                }
                tree = Some(new_tree);
                let _ = reply.send(display);
            }
            LayoutJob::Scroll { id, x, y, reply } => {
                let display = tree.as_mut().map(|tree| {
                    tree.set_scroll_offset(id, x, y);
                    Arc::new(DisplayList::build(tree))
                });
                let _ = reply.send(display);
            }
            LayoutJob::HitLink { x, y, reply } => {
                let target = tree
                    .as_ref()
                    .and_then(|tree| tree.hit_test(x, y))
                    .and_then(|id| {
                        document.as_ref().and_then(|doc| page::find_link_href(doc, id))
                    });
                let _ = reply.send(target);
            }
            LayoutJob::ScrollContainers { reply } => {
                let containers = tree
                    .as_ref()
                    .map(|tree| tree.scroll_containers())
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(id, _)| id)
                    // Root/body overflow belongs to the viewport, not
                    // to a container scroll.
                    .filter(|&id| {
                        !matches!(
                            document.as_ref().and_then(|doc| doc.dom().tag(id)),
                            Some("html" | "body")
                        )
                    })
                    .collect();
                let _ = reply.send(containers);
            }
        }
    }
}

/// The worker's state: the retained document, the layout viewport,
/// the frame epoch, the loader, and the in-flight navigation
/// (its epoch and its cancellation token).
///
/// M2: the worker keeps the current [`ElementState`] (`:hover` /
/// `:active` / `:focus`, fed by input) and the retained scroll
/// offsets (re-applied after every rebuild). The document and the
/// `!Send` layout tree live on the layout-actor thread instead (see
/// [`LayoutJob`]) — the worker holds only its job sender, plus
/// whether a page is loaded (resize gating).
struct WorkerState {
    layout_tx: std::sync::mpsc::Sender<LayoutJob>,
    has_document: bool,
    element_state: ElementState,
    scroll_offsets: HashMap<NodeId, (f32, f32)>,
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
    // The layout actor owns the !Send tree/document on its thread;
    // dropping this sender (worker end) ends the actor.
    let (layout_tx, layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
    std::thread::Builder::new()
        .name("pluto-layout".to_string())
        .spawn(move || layout_thread(layout_rx))
        .expect("layout actor thread spawns");
    let mut state = WorkerState {
        layout_tx,
        has_document: false,
        element_state: ElementState::default(),
        scroll_offsets: HashMap::new(),
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
                        // R15 (shell side): clicks hit-test the
                        // retained tree and navigate; wheel scrolls a
                        // nested scroll container when there is
                        // exactly one. Hover is NOT wired here: there
                        // is no hover variant in `InputEvent` yet, and
                        // adding one is a protocol change (v3) — see
                        // `scroll_container` / the ARCHITECTURE.md
                        // status. `:visited` stays false for the same
                        // reason the layout matcher documents: the
                        // visited-URL set has no `ElementState` field
                        // to ride in (that field lives in
                        // pluto-layout, which this workstream may not
                        // change) — follow-up, not silently faked.
                        match event {
                            InputEvent::Click { x, y } => {
                                if let Some(url) = click_target(&state, x, y).await {
                                    start_navigation(&mut state, &nav_tx, url);
                                }
                            }
                            InputEvent::WheelScrolled { dx, dy } => {
                                scroll_container(&mut state, &mut events, dx, dy).await;
                            }
                            InputEvent::KeyPressed { key } => {
                                eprintln!("[pluto] key pressed (no focus model yet): {key}");
                            }
                        }
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
/// epoch — so a late outcome is discarded), then run the fetch+parse
/// on a dedicated OS thread with a current-thread runtime. The load
/// future is `!Send` (the streaming parser holds html5ever's
/// `!Send` tokenizer/tree-builder state across awaits), so it cannot
/// be `tokio::spawn`d; the thread owns the load end to end, and only
/// the `Send` outcome crosses back, via `blocking_send`. Layout
/// stays on the layout-actor thread in `layout_and_submit`.
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
    std::thread::Builder::new()
        .name("pluto-navigation".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("navigation thread runtime builds");
            let outcome = runtime.block_on(async {
                match page::load_page_cancellable(&loader, url.clone(), token).await {
                    Ok((document, final_url)) => Ok((document, final_url)),
                    Err(reason) => Err((url, reason)),
                }
            });
            let _ = nav_tx.blocking_send((epoch, outcome));
        })
        .expect("navigation thread spawns");
}

/// A navigation finished: commit the document and lay it out at the
/// current viewport (a resize mid-fetch is honoured here), or render
/// the error page through the same pipeline and report the reason.
/// The document moves to the layout actor, which retains it (and the
/// tree) for later resizes, hit-tests, and container scrolls.
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
            layout_and_submit(state, events, Some(document)).await;
        }
        Err((url, reason)) => {
            eprintln!("[pluto] worker fetch failed: {reason}");
            let document = page::parse_document(&page::error_document(&url, &reason));
            state.scroll_offsets.clear();
            layout_and_submit(state, events, Some(document)).await;
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

/// Lay out at the worker's viewport on the layout-actor thread and
/// submit the result. `document: Some` hands the actor a new document
/// (a navigation / error page); `None` re-lays out the retained one
/// (a resize). A dead actor is reported as a crash; the old page stays
/// visible until the new list lands either way.
async fn layout_and_submit(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
    document: Option<Document>,
) -> bool {
    let (reply_tx, reply_rx) = oneshot::channel();
    let job = LayoutJob::Layout {
        document: Box::new(document),
        viewport: state.viewport,
        state: state.element_state.clone(),
        offsets: state.scroll_offsets.clone(),
        reply: reply_tx,
    };
    if state.layout_tx.send(job).is_err() {
        return emit(
            events,
            RendererToBrowser::RendererCrashed {
                reason: "layout actor is gone".to_string(),
            },
        )
        .await;
    }
    match reply_rx.await {
        Ok(display) => {
            state.has_document = true;
            submit(state, events, display).await
        }
        Err(_) => {
            emit(
                events,
                RendererToBrowser::RendererCrashed {
                    reason: "layout actor reply lost".to_string(),
                },
            )
            .await
        }
    }
}

/// Hit-test a click (page coordinates: the shell already added the
/// viewport scroll offset) on the layout actor and resolve the
/// nearest enclosing `<a href>` against the document base URL.
/// `None` means the click hit nothing navigable (or the actor is
/// gone) — it is ignored, not an error.
async fn click_target(state: &WorkerState, x: f32, y: f32) -> Option<String> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let job = LayoutJob::HitLink {
        x,
        y,
        reply: reply_tx,
    };
    if state.layout_tx.send(job).is_err() {
        return None;
    }
    reply_rx.await.ok().flatten()
}

/// Wheel-scroll a nested scroll container (R14/R15 shell side).
/// The viewport-level wheel scroll stays with the shell's Stage 4
/// shader-side scroll in the compositor; this handles the containers
/// *inside* the page: `set_scroll_offset` + re-collect + resubmit on
/// the layout actor, no re-layout.
///
/// Limits, by design: `InputEvent::WheelScrolled` carries no cursor
/// position, so with more than one scroll container there is no way
/// to know which one the wheel is over — only the exactly-one case
/// is scrolled, the rest is left to the viewport scroll (a position
/// on the event is a protocol change, follow-up). The root/body
/// (`<html>`/`<body>`) overflow propagates to the viewport instead,
/// so the actor excludes it from the container list. Offsets clamp
/// at zero on both axes, matching the shell; the bottom clamp needs
/// per-container content height, which the tree does not expose yet
/// (same documented gap as the shell scroll).
async fn scroll_container(
    state: &mut WorkerState,
    events: &mut mpsc::Sender<crate::Message>,
    dx: f32,
    dy: f32,
) {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollContainers { reply: reply_tx })
        .is_err()
    {
        return;
    }
    let Ok(containers) = reply_rx.await else {
        return;
    };
    if containers.len() != 1 {
        return;
    }
    let id = containers[0];
    let (x, y) = state.scroll_offsets.get(&id).copied().unwrap_or((0.0, 0.0));
    // NOTE: the protocol documents these deltas as lines; the shell
    // forwards pixel deltas (lines × 16), and the worker treats them
    // as page pixels — a protocol-doc fix for Stage 8, not a second
    // conversion here.
    let (next_x, next_y) = ((x + dx).max(0.0), (y + dy).max(0.0));
    if (next_x, next_y) == (x, y) {
        return;
    }
    let (scroll_tx, scroll_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::Scroll {
            id,
            x: next_x,
            y: next_y,
            reply: scroll_tx,
        })
        .is_err()
    {
        return;
    }
    if let Ok(Some(display)) = scroll_rx.await {
        state.scroll_offsets.insert(id, (next_x, next_y));
        submit(state, events, display).await;
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
    if state.has_document {
        eprintln!("[pluto] worker viewport resized to {width}x{height}, re-layout");
        layout_and_submit(state, events, None).await;
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

    /// Two-page server for the click test: `/` is a page with one
    /// big block link to `/page2`; `/page2` is a different page.
    async fn serve_two_pages(listener: TcpListener) {
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
                let body = if path == "/page2" {
                    "<!doctype html><html><head><style>body{margin:0}</style></head>\
                     <body><p>page two here</p></body></html>"
                } else {
                    "<!doctype html><html><head><style>body{margin:0}\
                     a{display:block;width:500px;height:100px}</style></head>\
                     <body><a href=\"/page2\">go to page two</a></body></html>"
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// Clicking a link (R15): the worker hit-tests the retained tree,
    /// resolves the `<a href>` against the base URL, and navigates
    /// through the normal CommitNavigation path (new epoch, old page
    /// stays visible until the new list lands).
    #[tokio::test]
    async fn worker_click_on_a_link_navigates() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_two_pages(listener));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let page1 = format!("http://127.0.0.1:{}/", addr.port());
        let page2 = format!("http://127.0.0.1:{}/page2", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url: page1 })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        // The link is a 500x100 block at (0,0); (100,50) is inside it.
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Click {
                x: 100.0,
                y: 50.0,
            }))
            .expect("worker takes the click");
        match next_event(&mut event_rx).await {
            RendererToBrowser::DidCommit { url: committed, .. } => {
                assert_eq!(committed, page2);
            }
            other => panic!("expected DidCommit for page2, got {other:?}"),
        }
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        // A click on empty page area (below the 100px link, e.g.
        // y=500 — the page is otherwise blank) navigates nowhere:
        // no event arrives within 200 ms.
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Click {
                x: 100.0,
                y: 500.0,
            }))
            .expect("worker takes the click");
        assert!(
            tokio::time::timeout(Duration::from_millis(200), next_event(&mut event_rx))
                .await
                .is_err(),
            "click on blank area must not navigate"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// One-page server with a single nested scroll container, for the
    /// wheel-scroll test: a 100px-tall `overflow:auto` div with ten
    /// lines inside it.
    async fn serve_scroll_page(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let _ = n;
                let mut body = String::from(
                    "<!doctype html><html><head><style>body{margin:0}\
                     div{height:100px;overflow:auto}</style></head><body><div>",
                );
                for i in 0..10 {
                    body.push_str(&format!("<p>line {i}</p>"));
                }
                body.push_str("</div></body></html>");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// Wheel over a page with exactly one scroll container (R14/R15):
    /// the worker scrolls that container and resubmits (new epoch,
    /// no re-layout, no refetch) instead of scrolling the viewport.
    #[tokio::test]
    async fn worker_wheel_scrolls_the_single_scroll_container() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_scroll_page(listener));
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
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::WheelScrolled {
                dx: 0.0,
                dy: 48.0,
            }))
            .expect("worker takes the wheel event");
        let (epoch, bytes) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, bytes, .. } => (epoch, bytes),
            other => panic!("expected SubmitDisplayList after wheel, got {other:?}"),
        };
        assert!(epoch > first_epoch, "wheel scroll resubmits");
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        assert!(!list.is_empty());
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }
}
