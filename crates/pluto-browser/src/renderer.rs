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
//! * Navigation runs on a dedicated per-tab OS thread with a
//!   current-thread tokio runtime inside a `LocalSet`. The streaming
//!   parser (`pluto_html::Parser`, Pluto's own tokenizer/tree-builder,
//!   full of `Cell`/`RefCell`) is single-threaded by construction
//!   and cannot be held across an await in a `tokio::spawn`d future
//!   — so each load runs as a `spawn_local` task on that thread,
//!   and only the `Send` outcome crosses back, over a channel.
//!   Script execution is delegated from the load task to the tab
//!   thread's main loop (which owns the `TabJs` — see `js_host`),
//!   so the main loop stays live while a load runs and a second
//!   navigation can cancel the first mid-flight.
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
use pluto_js::TabJs;
use pluto_layout::{ElementState, LayoutTree};
use pluto_net::ResourceLoader;
use pluto_paint::{DisplayList, ResourceTable};
use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
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
/// A job for the layout-actor thread, which owns the `!Send`
/// [`LayoutTree`](pluto_layout::LayoutTree) and the current [`Document`] on its thread for the
/// tab's lifetime. Every job carries a reply channel for its `Send`
/// answer; a dropped reply (actor gone) surfaces as a worker crash, never a
/// hang — the worker always awaits with the reply channel, and the
/// actor exits when the job channel closes (worker end / `Stop`).
///
/// Wave 2b adds the geometry jobs (`Geometry` / `FlushGeometry`):
/// the JS host's forced synchronous layout flush (the only
/// sanctioned synchronous cross-thread call in the design) and the
/// click hit-test (`HitTest`). They reply over `std` channels /
/// oneshots the tab thread blocks on, because the tab thread is a
/// plain OS thread, not an async task.
pub(crate) enum LayoutJob {
    /// (Re-)lay out at `viewport` with `state`, re-applying the
    /// retained `offsets` after the rebuild. `document: Some` replaces
    /// the actor's document (a new navigation / error page / a
    /// JS-mutated document); `None` re-lays out the retained one
    /// (a resize).
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
    /// Geometry of one node from the actor's retained tree, with NO
    /// flush (the caller already knows the layout is clean).
    /// `None` when the actor holds no tree / node.
    Geometry {
        id: NodeId,
        reply: std::sync::mpsc::Sender<Option<pluto_js::Geometry>>,
    },
    /// Forced synchronous layout flush: replace the actor's
    /// document with the live (JS-mutated) one, rebuild the tree
    /// (full re-layout in M4 — dirty-subtree layout is Stage 8),
    /// and return the node's geometry from the fresh tree. The
    /// caller does NOT clear the DOM dirty bits here — the tab
    /// thread's frame tick still re-lays out and submits a fresh
    /// display list, and clears the bits then.
    FlushGeometry {
        // Boxed: a `Document` is far larger than the other
        // variants' payloads (clippy `large_enum_variant`).
        document: Box<Document>,
        viewport: (f32, f32),
        state: ElementState,
        id: NodeId,
        reply: std::sync::mpsc::Sender<Option<pluto_js::Geometry>>,
    },
    /// Hit-test a click (page coordinates) and return the node hit,
    /// if any (the tab thread resolves links / dispatches the click
    /// event itself).
    HitTest {
        x: f32,
        y: f32,
        reply: oneshot::Sender<Option<NodeId>>,
    },
}

/// Geometry of one node from a laid-out tree, in page coordinates.
fn geometry_of(tree: &LayoutTree, id: NodeId) -> Option<pluto_js::Geometry> {
    tree.boxes_for_dom(id).first().map(|b| pluto_js::Geometry {
        x: b.x as f64,
        y: b.y as f64,
        width: b.width as f64,
        height: b.height as f64,
    })
}

/// The layout actor: owns the tree + document, answers jobs until the
/// channel closes. Runs on its own OS thread (`pluto-layout`), so the
/// `!Send` tree never crosses an await or a thread boundary.
pub(crate) fn layout_thread(jobs: std::sync::mpsc::Receiver<LayoutJob>) {
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
            LayoutJob::Geometry { id, reply } => {
                let geometry = tree.as_ref().and_then(|tree| geometry_of(tree, id));
                let _ = reply.send(geometry);
            }
            LayoutJob::FlushGeometry {
                document: new_document,
                viewport,
                state,
                id,
                reply,
            } => {
                document = Some(*new_document);
                let doc = document.as_ref().unwrap();
                let (_, new_tree) = page::layout_document_with_state(doc, viewport, &state);
                let geometry = geometry_of(&new_tree, id);
                tree = Some(new_tree);
                let _ = reply.send(geometry);
            }
            LayoutJob::HitTest { x, y, reply } => {
                let target = tree.as_ref().and_then(|tree| tree.hit_test(x, y));
                let _ = reply.send(target);
            }
        }
    }
}

// ---------------------------------------------------------------- tab thread
//
// Wave 2b (Stage 7, Rec 7–10): the per-tab worker thread. The iced
// subscription task (`run_worker`) is only a forwarder now — it
// cannot own the `TabJs` (the realm is `!Send`: `Rc` state, a
// single-threaded QuickJS runtime) or the streaming parser (also
// `!Send`), so one dedicated OS thread per tab owns them both for
// the tab's lifetime, plus the `ResourceLoader` and a current-thread
// tokio runtime for network I/O. That thread is the per-tab worker
// thread of the Stage 7 report: it owns the `AsyncRuntime` +
// `AsyncContext` (realm) + event loop + `Dom` + parser. Its main
// loop (`tab_main`) is async — loads are `spawn_local` tasks, so a
// navigation can cancel a load mid-flight, and script execution is
// delegated from the load task to the main loop over a channel
// (`js_host::ExecMsg`).
//
// Frame cadence (Rec 10): there is NO shell frame signal to drive
// rendering opportunities — the shell/compositor exposes no vsync
// tick (the viewport widget only reports bounds; the compositor
// only installs display lists). M4 therefore drives the loop on a
// ~60 Hz timer: the tab thread ticks every 16 ms, drives the
// `TabJs` turns, and re-lays out + submits only when a mutation
// marked the document dirty or an rAF callback is pending (the
// loop's own `maybe_render` already skips frames with nothing to
// do). Hidden/background tabs get the same cadence in M4 — there
// is no visibility signal to throttle on yet; throttling is a
// later-stage refinement, not faked here.

/// How often the tab thread ticks (the M4 rendering cadence:
/// ~60 Hz; see the module docs — no shell vsync exists to drive it).
const TAB_TICK: Duration = Duration::from_millis(16);

/// The outcome of one navigation's load task, sent back to the
/// tab thread's main loop. Carries the navigation epoch it belongs
/// to: a load that was cancelled by a newer navigation can still
/// have its outcome in flight (it sent it just before the abort
/// landed), and the main loop discards any outcome whose epoch is
/// not the current one — the old page never commits.
enum LoadOutcome {
    Loaded {
        epoch: u64,
        // Boxed: a `Document` is far larger than the `Failed`
        // variant's payloads (clippy `large_enum_variant`).
        document: Box<Document>,
        final_url: String,
    },
    Failed {
        epoch: u64,
        url: String,
        reason: String,
    },
}

/// The tab thread's state: everything the per-tab worker owns.
struct TabState {
    layout_tx: std::sync::mpsc::Sender<LayoutJob>,
    loader: ResourceLoader,
    /// The tokio runtime handle the `fetch()` backend spawns its
    /// network tasks on (the tab thread's current-thread runtime —
    /// see `tab_thread_main`).
    fetch_handle: tokio::runtime::Handle,
    /// The JS host for this tab (created on the first navigation,
    /// fresh realm per navigation via `TabJs::navigate`).
    tab_js: Option<TabJs>,
    /// The production `DomHost`, shared with the `TabJs` (see
    /// `js_host::SharedHost`); the tab thread updates its parser /
    /// sheet / viewport between navigations.
    host: Rc<std::sync::Mutex<crate::js_host::BrowserDomHost>>,
    /// The `fetch()` backend, shared with the `TabJs`.
    fetch_backend: Option<std::sync::Arc<crate::js_host::PlutoFetchBackend>>,
    /// The current document (re-synced from the host's live `Dom`
    /// on every re-layout, so JS mutations reach the actor).
    document: Option<Document>,
    has_document: bool,
    element_state: ElementState,
    scroll_offsets: HashMap<NodeId, (f32, f32)>,
    viewport: (f32, f32),
    /// Display-list epoch (advances per submitted list).
    epoch: u64,
    /// Navigation epoch (cancels the old navigation's fetch +
    /// scripts + timers + promises).
    nav_epoch: u64,
    nav_cancel: Option<CancellationToken>,
    /// The running load task, if any (aborted when a newer
    /// navigation commits).
    load_handle: Option<tokio::task::JoinHandle<()>>,
    /// Script-execution requests from load tasks (their sender is
    /// cloned into each `LoadHooks`).
    exec_tx: tokio::sync::mpsc::UnboundedSender<crate::js_host::ExecMsg>,
    /// Script-execution requests, received by the main loop. An
    /// `Option` so `tab_main` can take the receiver out: the
    /// `select!` arms hold the receivers locally, because the arm
    /// bodies borrow `state` mutably and `select!` builds all arm
    /// futures up front.
    exec_rx: Option<tokio::sync::mpsc::UnboundedReceiver<crate::js_host::ExecMsg>>,
    /// Load outcomes, sent back by load tasks.
    load_outcome_tx: tokio::sync::mpsc::UnboundedSender<LoadOutcome>,
    /// Load outcomes, received by the main loop (see `exec_rx`).
    load_outcome_rx: Option<tokio::sync::mpsc::UnboundedReceiver<LoadOutcome>>,
    /// Set by JS mutations (via the host); the tick reads and
    /// clears it, then asks the loop for a rendering opportunity.
    render_requested: Rc<Cell<bool>>,
}

impl TabState {
    fn new(
        layout_tx: std::sync::mpsc::Sender<LayoutJob>,
        fetch_handle: tokio::runtime::Handle,
    ) -> Self {
        let errors = Rc::new(std::sync::Mutex::new(Vec::new()));
        let render_requested = Rc::new(Cell::new(false));
        let host = Rc::new(std::sync::Mutex::new(crate::js_host::BrowserDomHost::new(
            Arc::new(std::sync::Mutex::new(pluto_html::Dom::new())),
            layout_tx.clone(),
            page::DEFAULT_VIEWPORT,
            ElementState::default(),
            "",
            Rc::clone(&errors),
            Rc::clone(&render_requested),
        )));
        let (exec_tx, exec_rx) = tokio::sync::mpsc::unbounded_channel();
        let (load_outcome_tx, load_outcome_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            layout_tx,
            loader: ResourceLoader::new(),
            fetch_handle,
            tab_js: None,
            host,
            fetch_backend: None,
            document: None,
            has_document: false,
            element_state: ElementState::default(),
            scroll_offsets: HashMap::new(),
            viewport: page::DEFAULT_VIEWPORT,
            epoch: 0,
            nav_epoch: 0,
            nav_cancel: None,
            load_handle: None,
            exec_tx,
            exec_rx: Some(exec_rx),
            load_outcome_tx,
            load_outcome_rx: Some(load_outcome_rx),
            render_requested,
        }
    }

    /// The live `TabJs`, creating it on the first navigation.
    fn tab_js(&mut self) -> &mut TabJs {
        if self.tab_js.is_none() {
            let mut js = crate::js_host::new_tab_js().expect("tab JS realm creates");
            js.set_dom_host(Box::new(crate::js_host::SharedHost(Rc::clone(&self.host))));
            let backend = std::sync::Arc::new(crate::js_host::PlutoFetchBackend::new(
                self.loader.clone(),
                self.fetch_handle.clone(),
                "",
            ));
            js.set_fetch_backend(
                std::sync::Arc::clone(&backend) as std::sync::Arc<dyn pluto_js::FetchBackend>,
            );
            self.fetch_backend = Some(backend);
            self.tab_js = Some(js);
        }
        self.tab_js.as_mut().unwrap()
    }
}

/// The tab thread: owns the `TabJs` + loader for the tab's
/// lifetime. It runs ONE `block_on` over an async main loop
/// (`tab_main`) on a current-thread runtime inside a `LocalSet`:
/// the load tasks are `spawn_local` (they hold `Rc` parser/host
/// handles, so they are `!Send` and cannot use `spawn`), and the
/// main loop stays live while a load runs — a second navigation
/// cancels the first mid-flight (token + `JoinHandle::abort`),
/// which the old inline-`block_on` load could not do (the command
/// loop was blocked inside the load until it finished).
fn tab_thread_main(
    commands: tokio::sync::mpsc::Receiver<BrowserToRenderer>,
    events: tokio::sync::mpsc::Sender<RendererToBrowser>,
    layout_tx: std::sync::mpsc::Sender<LayoutJob>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tab thread runtime builds");
    runtime.block_on(async {
        tokio::task::LocalSet::new()
            .run_until(tab_main(commands, events, layout_tx))
            .await;
    });
}

/// The tab thread's main loop: a `select!` over the frame tick
/// (~60 Hz rendering cadence — see the module docs, no shell vsync
/// exists), shell commands, script-execution requests from load
/// tasks, and load outcomes. Everything here runs on the tab
/// thread, so `TabJs` entry is always on its owning thread.
async fn tab_main(
    mut commands: tokio::sync::mpsc::Receiver<BrowserToRenderer>,
    events: tokio::sync::mpsc::Sender<RendererToBrowser>,
    layout_tx: std::sync::mpsc::Sender<LayoutJob>,
) {
    let mut state = TabState::new(layout_tx, tokio::runtime::Handle::current());
    // The receivers are taken out of the state so the `select!`
    // arms can hold them without borrowing `state` — the arm bodies
    // borrow it mutably, and `select!` builds all arm futures up
    // front.
    let mut exec_rx = state.exec_rx.take().expect("exec receiver present");
    let mut load_outcome_rx = state
        .load_outcome_rx
        .take()
        .expect("load outcome receiver present");
    let mut tick = tokio::time::interval(TAB_TICK);
    // `finish_page` blocks this loop for up to a second driving
    // the JS loop after a parse; a Burst interval would then fire
    // dozens of catch-up ticks before the next command is read.
    // Delay skips the backlog instead.
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                drive_js(&mut state);
                // Dirty-bit flow (Rec 7): a JS mutation sets the
                // host's render flag / dirty bits and returns
                // immediately (no layout); the frame tick re-lays
                // out and submits.
                if state.render_requested.get() || state.host.lock().unwrap().dom_is_dirty() {
                    state.render_requested.set(false);
                    relayout_and_submit(&mut state, &events).await;
                }
            }
            cmd = commands.recv() => {
                let Some(cmd) = cmd else { break };
                match cmd {
                    BrowserToRenderer::Shutdown => break,
                    BrowserToRenderer::CommitNavigation { url } => {
                        start_navigation(&mut state, url);
                    }
                    BrowserToRenderer::Resize { width, height } => {
                        resize(&mut state, &events, width, height).await;
                    }
                    BrowserToRenderer::ResourceEvent(event) => {
                        // The M5 shape: the shell streams fetch events back
                        // to the renderer. Today the worker fetches through
                        // its own loader, which consumes the events
                        // internally, so nothing arrives here — ignore it if
                        // it ever does.
                        eprintln!("[pluto] ignoring ResourceEvent from shell: {event:?}");
                    }
                    BrowserToRenderer::InputEvent(event) => {
                        handle_input(&mut state, &events, event).await;
                    }
                }
            }
            msg = exec_rx.recv() => {
                let Some(msg) = msg else { break };
                handle_exec(&mut state, msg);
            }
            outcome = load_outcome_rx.recv() => {
                let Some(outcome) = outcome else { break };
                handle_nav_outcome(&mut state, &events, outcome).await;
            }
        }
    }
}

/// Drive the JS loop for one tick: a few turns (one turn runs one
/// task + the microtask checkpoint + a rendering opportunity when
/// one is due). A poisoned realm is recreated fresh — a watchdog
/// kill must not wedge the tab.
fn drive_js(state: &mut TabState) {
    if state.tab_js.is_none() {
        return;
    }
    let mut poisoned = false;
    {
        let js = state.tab_js.as_mut().unwrap();
        for _ in 0..8 {
            match js.run_turn() {
                Ok(pluto_js::TurnOutcome::Idle) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[pluto] event loop error: {e}");
                    poisoned = true;
                    break;
                }
            }
            if !js.work_pending() {
                break;
            }
        }
        if js.realm_poisoned() {
            poisoned = true;
        }
    }
    if poisoned {
        recreate_realm(state);
    }
}

/// Recreate the realm after a watchdog kill (or a load whose scripts
/// killed it): fresh realm, host + fetch backend re-attached by
/// `TabJs::navigate` semantics.
fn recreate_realm(state: &mut TabState) {
    if let Some(js) = state.tab_js.as_mut()
        && js.recreate_realm().is_err()
    {
        eprintln!("[pluto] realm recreation failed; JS is dead for this tab");
    }
}

/// Run one script-execution request from a load task, on the `TabJs`
/// the main loop owns. A request from a stale navigation (its epoch
/// is not the current one — the load was cancelled after sending
/// it) is dropped, never run: an old page's script must not execute
/// in the new document's realm.
fn handle_exec(state: &mut TabState, msg: crate::js_host::ExecMsg) {
    use crate::js_host::ExecMsg;
    let epoch = match &msg {
        ExecMsg::Classic { epoch, .. }
        | ExecMsg::Queued { epoch, .. }
        | ExecMsg::ParseFinished { epoch, .. } => *epoch,
    };
    if epoch != state.nav_epoch {
        // Stale: drop without replying. The load task was aborted
        // (or is about to be); its `reply_rx.await` sees the
        // dropped sender and proceeds.
        return;
    }
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    let mut exec = crate::js_host::ScriptExecutor::new(Rc::clone(&state.host));
    match msg {
        ExecMsg::Classic {
            source,
            url,
            reply,
            ..
        } => {
            exec.exec_classic(js, &source, &url);
            let _ = reply.send(());
        }
        ExecMsg::Queued {
            node,
            src,
            bytes,
            reply,
            ..
        } => {
            exec.exec_queued(js, node, src, bytes);
            let _ = reply.send(());
        }
        ExecMsg::ParseFinished {
            document, reply, ..
        } => {
            let document = exec.finish_page(js, *document);
            let _ = reply.send(document);
        }
    }
    if state
        .tab_js
        .as_ref()
        .is_some_and(|js| js.realm_poisoned())
    {
        recreate_realm(state);
    }
}

/// Run the worker: forward shell commands to the tab thread and tab
/// events to the shell. Ends when the command channel closes, the
/// tab thread ends, or `Shutdown` arrives.
pub async fn run_worker(mut commands: CommandReceiver, mut events: mpsc::Sender<crate::Message>) {
    // The layout actor owns the !Send tree/document on its thread;
    // dropping this sender (worker end) ends the actor.
    let (layout_tx, layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
    std::thread::Builder::new()
        .name("pluto-layout".to_string())
        .spawn(move || layout_thread(layout_rx))
        .expect("layout actor thread spawns");
    let (tab_cmd_tx, tab_cmd_rx) = tokio::sync::mpsc::channel::<BrowserToRenderer>(100);
    let (tab_event_tx, mut tab_event_rx) = tokio::sync::mpsc::channel::<RendererToBrowser>(100);
    let layout_tx2 = layout_tx.clone();
    std::thread::Builder::new()
        .name("pluto-tab".to_string())
        .spawn(move || tab_thread_main(tab_cmd_rx, tab_event_tx, layout_tx2))
        .expect("tab thread spawns");
    loop {
        tokio::select! {
            cmd = commands.recv() => {
                match cmd {
                    None => {
                        let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
                        break;
                    }
                    Some(cmd) => {
                        if tab_cmd_tx.send(cmd).await.is_err() {
                            break;
                        }
                    }
                }
            }
            event = tab_event_rx.recv() => {
                match event {
                    Some(event) => {
                        if !emit(&mut events, event).await {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        // The shell is gone: there is nobody left to report to.
        if events.is_closed() {
            let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
            break;
        }
    }
}

/// Start a navigation: cancel the previous load (its token and its
/// task — a slow fetch no longer blocks the command loop), get a
/// fresh realm for the new document, then spawn the load as a
/// `spawn_local` task on the tab thread's runtime. The old page
/// stays visible until the new list lands; a stale outcome is
/// discarded by epoch in `handle_nav_outcome`.
fn start_navigation(state: &mut TabState, url: String) {
    if let Some(token) = state.nav_cancel.take() {
        token.cancel();
    }
    if let Some(handle) = state.load_handle.take() {
        handle.abort();
    }
    state.nav_epoch += 1;
    let epoch = state.nav_epoch;
    let token = CancellationToken::new();
    state.nav_cancel = Some(token.clone());
    if let Some(backend) = &state.fetch_backend {
        backend.set_epoch(epoch);
        backend.set_base_url(&url);
    }
    // Fresh realm per navigation (fresh globals, fresh wrapper
    // cache, cleared queues, bumped epoch). A failed recreation
    // keeps the old realm — the load still runs, scripts just
    // execute in the old realm; logged, not fatal.
    let js = state.tab_js();
    if js.navigate().is_err() {
        eprintln!("[pluto] realm recreation failed for navigation to {url}");
    }
    if let Some(backend) = &state.fetch_backend {
        backend.set_base_url(&url);
    }
    eprintln!("[pluto] worker fetching {url} (epoch {epoch})");
    // The host's viewport / element state for this navigation; the
    // parser itself is created by the load (it knows the final URL
    // after redirects) and the runner attaches the host to its
    // shared `Dom` via `parser_created` before the first chunk is
    // fed — the tab thread never constructs a parser of its own.
    {
        let mut host = state.host.lock().unwrap();
        host.set_viewport(state.viewport);
        host.set_element_state(state.element_state.clone());
    }
    let loader = state.loader.clone();
    let exec_tx = state.exec_tx.clone();
    let outcome_tx = state.load_outcome_tx.clone();
    let host = Rc::clone(&state.host);
    let handle = tokio::task::spawn_local(async move {
        let hooks = crate::js_host::LoadHooks::new(epoch, exec_tx, host);
        let outcome = page::load_page_cancellable_with_hooks(&loader, url.clone(), token, Some(Box::new(hooks))).await;
        let outcome = match outcome {
            Ok((document, final_url)) => LoadOutcome::Loaded {
                epoch,
                document: Box::new(document),
                final_url,
            },
            Err(reason) => LoadOutcome::Failed {
                epoch,
                url,
                reason,
            },
        };
        let _ = outcome_tx.send(outcome);
    });
    state.load_handle = Some(handle);
}

/// A navigation finished: commit the document and lay it out at the
/// current viewport (a resize mid-fetch is honoured here), or render
/// the error page through the same pipeline and report the reason.
/// The document moves to the layout actor, which retains it (and the
/// tree) for later resizes, hit-tests, and container scrolls.
/// A stale outcome (a load cancelled after it sent its result) is
/// discarded by epoch — the old page never commits.
async fn handle_nav_outcome(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    outcome: LoadOutcome,
) {
    let epoch = match &outcome {
        LoadOutcome::Loaded { epoch, .. } | LoadOutcome::Failed { epoch, .. } => *epoch,
    };
    if epoch != state.nav_epoch {
        eprintln!("[pluto] discarding stale navigation outcome (epoch {epoch})");
        return;
    }
    state.nav_cancel = None;
    state.load_handle = None;
    match outcome {
        LoadOutcome::Loaded {
            document,
            final_url,
            ..
        } => {
            eprintln!("[pluto] worker fetched {final_url}");
            state.document = Some((*document).clone());
            if !emit_tab(events, RendererToBrowser::DidCommit {
                url: final_url.clone(),
                status: format!("Loaded {final_url}"),
            })
            .await
            {
                return;
            }
            layout_and_submit(state, events, Some(*document)).await;
        }
        LoadOutcome::Failed { url, reason, .. } => {
            eprintln!("[pluto] worker fetch failed: {reason}");
            let document = page::parse_document(&page::error_document(&url, &reason));
            state.document = Some(document.clone());
            state.scroll_offsets.clear();
            layout_and_submit(state, events, Some(document)).await;
            let _ = emit_tab(
                events,
                RendererToBrowser::LoadFailed { url, reason },
            )
            .await;
        }
    }
}

/// Emit one renderer event to the async worker (which forwards it
/// to the shell). Returns false when the worker is gone.
async fn emit_tab(
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    event: RendererToBrowser,
) -> bool {
    events.send(event).await.is_ok()
}

/// Emit one renderer event to the shell (async worker side).
/// Returns false when the shell is gone.
async fn emit(events: &mut mpsc::Sender<crate::Message>, event: RendererToBrowser) -> bool {
    events
        .send(crate::Message::RendererEvent(event))
        .await
        .is_ok()
}

/// Submit a display list to the shell, stamping the next epoch.
/// No images in M1, so the resource table travels empty.
async fn submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    display: Arc<DisplayList>,
) -> bool {
    state.epoch += 1;
    emit_tab(
        events,
        RendererToBrowser::SubmitDisplayList {
            epoch: state.epoch,
            bytes: display.to_bytes(),
            resources: ResourceTable::new(),
        },
    )
    .await
}

/// Lay out at the tab's viewport on the layout-actor thread and
/// submit the result. `document: Some` hands the actor a new
/// document (a navigation / error page / a JS-mutated document);
/// `None` re-lays out the retained one (a resize). A dead actor is
/// reported as a crash; the old page stays visible until the new
/// list lands either way.
async fn layout_and_submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
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
        return emit_tab(
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
            state.host.lock().unwrap().clear_dirty();
            submit(state, events, display).await
        }
        Err(_) => emit_tab(
            events,
            RendererToBrowser::RendererCrashed {
                reason: "layout actor reply lost".to_string(),
            },
        )
        .await,
    }
}

/// Re-layout the live (possibly JS-mutated) document and submit.
/// The document is re-synced from the host's live `Dom` first, so
/// script mutations reach the actor.
async fn relayout_and_submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
) {
    if !state.has_document {
        return;
    }
    let document = state.host.lock().unwrap().current_document();
    state.document = Some(document.clone());
    layout_and_submit(state, events, Some(document)).await;
}

/// Hit-test a click (page coordinates: the shell already added the
/// viewport scroll offset) on the layout actor. `None` means nothing
/// was hit (or the actor is gone).
async fn hit_test(state: &TabState, x: f32, y: f32) -> Option<NodeId> {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::HitTest {
            x,
            y,
            reply: reply_tx,
        })
        .is_err()
    {
        return None;
    }
    reply_rx.await.ok().flatten()
}

/// Hit-test a click and resolve the nearest enclosing `<a href>`
/// against the document base URL. `None` means nothing navigable
/// was hit.
async fn click_target(state: &TabState, x: f32, y: f32) -> Option<String> {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::HitLink {
            x,
            y,
            reply: reply_tx,
        })
        .is_err()
    {
        return None;
    }
    reply_rx.await.ok().flatten()
}

async fn handle_input(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    event: InputEvent,
) {
    // R15 (shell side): clicks hit-test the retained tree and
    // navigate; wheel scrolls a nested scroll container when there
    // is exactly one. Hover is NOT wired here: there is no hover
    // variant in `InputEvent` yet, and adding one is a protocol
    // change (v3) — see `scroll_container` / the ARCHITECTURE.md
    // status. `:visited` stays false for the same reason the layout
    // matcher documents: the visited-URL set has no `ElementState`
    // field to ride in (that field lives in pluto-layout, which
    // this workstream may not change) — follow-up, not silently
    // faked.
    match event {
        InputEvent::Click { x, y } => {
            // A link navigates (new epoch, Stage 5 cancellation);
            // any other hit dispatches a bubbling click into the
            // JS event loop, so `addEventListener('click')` pages
            // (counters, accordions) respond. The tick drives the
            // dispatch and the re-layout that follows it.
            if let Some(url) = click_target(state, x, y).await {
                start_navigation(state, url);
            } else if let Some(node) = hit_test(state, x, y).await
                && let Some(js) = state.tab_js.as_mut()
            {
                js.dispatch_event(node, "click", true);
            }
        }
        InputEvent::WheelScrolled { dx, dy } => {
            scroll_container(state, events, dx, dy).await;
        }
        InputEvent::KeyPressed { key } => {
            eprintln!("[pluto] key pressed (no focus model yet): {key}");
        }
    }
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
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
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
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
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
    state.host.lock().unwrap().set_viewport((width, height));
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
    // ------------------------------------------------ M4 acceptance
    //
    // Stage 7, wave 2b: the M4 done-criteria, end to end through the
    // worker with the local test-server pattern. Text renders as
    // `Glyphs` items (one glyph per character for the spaceless
    // ASCII strings these pages use), so the tests assert on glyph
    // counts: "9" is 1 glyph, "10" is 2, "loading" is 7,
    // "hello-json" is 10. A script-driven DOM update re-lays out
    // and resubmits, so waiting for the new glyph count also proves
    // the dirty-bit flow fired.

    /// Glyphs painted by a display list.
    fn glyph_count(list: &DisplayList) -> usize {
        list.0
            .iter()
            .map(|item| match item {
                DisplayItem::Glyphs { glyphs, .. } => glyphs.len(),
                _ => 0,
            })
            .sum()
    }

    /// Wait for a `SubmitDisplayList` whose glyph count is
    /// `expected` (a script-driven DOM update re-lays out and
    /// resubmits). Other display lists are skipped; anything else
    /// is a failure. False on timeout — the update never landed.
    async fn wait_for_glyphs(
        rx: &mut mpsc::Receiver<crate::Message>,
        expected: usize,
        timeout: Duration,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return false;
            }
            let event = match tokio::time::timeout(remaining, next_event(rx)).await {
                Ok(event) => event,
                Err(_) => return false,
            };
            match event {
                RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                    if glyph_count(&list) == expected {
                        return true;
                    }
                }
                other => panic!("expected SubmitDisplayList while waiting for glyphs, got {other:?}"),
            }
        }
    }

    /// Commit a navigation and wait for its display list; returns
    /// the list's glyph count.
    async fn load_and_count(
        cmd_tx: &pluto_ipc::CommandSender,
        rx: &mut mpsc::Receiver<crate::Message>,
        url: String,
    ) -> usize {
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation { url })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        match next_event(rx).await {
            RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                glyph_count(&list)
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
    }

    /// Multi-route local server: `routes` maps a path to
    /// (content-type, body). Unknown paths 404.
    async fn serve_routes(
        listener: TcpListener,
        routes: std::sync::Arc<Vec<(String, String, String)>>,
    ) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let routes = std::sync::Arc::clone(&routes);
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
                    .unwrap_or("/")
                    .to_string();
                let (content_type, body) = routes
                    .iter()
                    .find(|(p, _, _)| *p == path)
                    .map(|(_, ct, b)| (ct.clone(), b.clone()))
                    .unwrap_or_else(|| ("text/plain".to_string(), "not found".to_string()));
                let status = if body == "not found" {
                    "404 Not Found"
                } else {
                    "200 OK"
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// M4: a counter page — click increments, text updates. The
    /// button is a 200x40 block at (0,0); the count span starts at
    /// "9" (1 glyph) and becomes "10" (2 glyphs) after the click.
    #[tokio::test]
    async fn worker_counter_click_increments_text() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block;width:200px;height:40px}</style></head><body>            <button id=\"inc\">Inc</button><span id=\"count\">9</span>            <script>let n = 9;            document.getElementById('inc').addEventListener('click', () => {            n++; document.getElementById('count').textContent = String(n); });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Inc" (3) + "9" (1) = 4 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 4);
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Click {
                x: 100.0,
                y: 20.0,
            }))
            .expect("worker takes the click");
        // "Inc" (3) + "10" (2) = 5 glyphs: the click dispatched,
        // the handler ran, the DOM mutated, and the dirty-bit flow
        // re-laid out and resubmitted.
        assert!(
            wait_for_glyphs(&mut event_rx, 5, Duration::from_secs(5)).await,
            "counter did not reach 10 after the click"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: an accordion page — click toggles the details text on
    /// and off again.
    #[tokio::test]
    async fn worker_accordion_click_toggles_text() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block;width:200px;height:40px}</style></head><body>            <button id=\"toggle\">Toggle</button><div id=\"details\"></div>            <script>let open = false;            document.getElementById('toggle').addEventListener('click', () => {            open = !open;            document.getElementById('details').textContent = open ? 'details-here-now' : ''; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Toggle" (6), details empty.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 6);
        let click = || {
            cmd_tx
                .send(BrowserToRenderer::InputEvent(InputEvent::Click {
                    x: 100.0,
                    y: 20.0,
                }))
                .expect("worker takes the click")
        };
        click();
        // "Toggle" (6) + "details-here-now" (16) = 22.
        assert!(
            wait_for_glyphs(&mut event_rx, 22, Duration::from_secs(5)).await,
            "accordion did not open after the click"
        );
        click();
        assert!(
            wait_for_glyphs(&mut event_rx, 6, Duration::from_secs(8)).await,
            "accordion did not close after the second click"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: a fetch-JSON loader page — `fetch` → `json()` → DOM
    /// update, same-origin over the local server.
    #[tokio::test]
    async fn worker_fetch_json_loader_updates_dom() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <span id=\"out\">loading</span>            <script>fetch('/data.json').then(r => r.json()).then(d => {            document.getElementById('out').textContent = d.message; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![
            ("/".to_string(), "text/html".to_string(), page.to_string()),
            (
                "/data.json".to_string(),
                "application/json".to_string(),
                "{\"message\":\"hello-json\"}".to_string(),
            ),
        ]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 7);
        // "hello-json" (10): the promise resolved as a networking
        // task, the DOM updated, and the frame tick resubmitted.
        assert!(
            wait_for_glyphs(&mut event_rx, 10, Duration::from_secs(5)).await,
            "fetch-JSON loader did not update the DOM"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: a throwing script does not kill later scripts or the
    /// loop — the script after the throw runs, and a timer set by
    /// it still fires.
    #[tokio::test]
    async fn worker_throwing_script_does_not_kill_later_scripts() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <span id=\"ok\">waiting</span>            <script>throw new Error('boom');</script>            <script>document.getElementById('ok').textContent = 'alive';            setTimeout(() => { document.getElementById('ok').textContent = 'ticked'; }, 5000);</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // The second script ran despite the throw: "alive" (5).
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 5);
        // The timer still fired: "ticked" (6) — the loop is alive.
        assert!(
            wait_for_glyphs(&mut event_rx, 6, Duration::from_secs(8)).await,
            "timer set by the later script did not fire"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: navigation cancels pending timers — a 500 ms timer set
    /// by the old page never fires after navigating away.
    #[tokio::test]
    async fn worker_navigation_cancels_pending_timer() {
        let page1 = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <span id=\"x\">page-one</span>            <script>setTimeout(() => {            document.getElementById('x').textContent = 'done'; }, 6000);</script>            </body></html>";
        let page2 = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <p>page-two-here</p></body></html>";
        let routes = std::sync::Arc::new(vec![
            ("/".to_string(), "text/html".to_string(), page1.to_string()),
            (
                "/page2".to_string(),
                "text/html".to_string(),
                page2.to_string(),
            ),
        ]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let page1_url = format!("http://127.0.0.1:{}/", addr.port());
        let page2_url = format!("http://127.0.0.1:{}/page2", addr.port());
        // "page-one" (8).
        assert_eq!(
            load_and_count(&cmd_tx, &mut event_rx, page1_url).await,
            8
        );
        // Navigate away well before the 500 ms timer fires.
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page2_url,
            })
            .expect("worker takes the command");
        match next_event(&mut event_rx).await {
            RendererToBrowser::DidCommit { .. } => {}
            other => panic!("expected DidCommit for page2, got {other:?}"),
        }
        // "page-two-here" (13).
        match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                assert_eq!(glyph_count(&list), 13);
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
        // The old timer never fires: nothing resubmits for 5 s.
        // Its 6 s deadline (measured from registration during the
        // parse, ~2.6 s before this wait starts) falls inside the
        // window, so a live timer would have fired here.
        assert!(
            tokio::time::timeout(Duration::from_millis(5000), next_event(&mut event_rx))
                .await
                .is_err(),
            "cancelled timer fired after navigation"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: a module that top-level-awaits a `fetch()` is driven by
    /// the loop across turns (never through the watchdog-poisoning
    /// `await_promise` path) and settles into the DOM.
    #[tokio::test]
    async fn worker_module_top_level_await_fetch_updates_dom() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <span id=\"m\">pending</span>            <script type=\"module\">const r = await fetch('/data.json');            const d = await r.json();            document.getElementById('m').textContent = d.message;</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![
            ("/".to_string(), "text/html".to_string(), page.to_string()),
            (
                "/data.json".to_string(),
                "application/json".to_string(),
                "{\"message\":\"hello-json\"}".to_string(),
            ),
        ]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 7);
        assert!(
            wait_for_glyphs(&mut event_rx, 10, Duration::from_secs(5)).await,
            "module top-level-awaiting fetch did not update the DOM"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: `document.write` from a parser-inserted script lands at
    /// the insertion point.
    #[tokio::test]
    async fn worker_document_write_inserts_at_parse() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <script>document.write('<p>written</p>');</script></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "written" (7) is in the first display list.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 7);
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// M4: `set_inner_html` parses with scripts inert — the
    /// injected markup renders, the injected script never runs.
    #[tokio::test]
    async fn worker_set_inner_html_keeps_scripts_inert() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <div id=\"box\"></div>            <script>document.getElementById('box').innerHTML =            '<p>safe</p><script>document.getElementById(\"box\").textContent = \"pwned\";<\\/script>';</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "safe" (4). If the injected script had run, the text
        // would be "pwned" (5) instead.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 4);
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

}
