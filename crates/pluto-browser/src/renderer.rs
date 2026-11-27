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
//! that cross IPC now that the worker runs out-of-process (Stage 9
//! Rec 9: the subscription spawns one `RendererProcess` per tab
//! and bridges it — see [`run_renderer_bridge`]). The worker never
//! touches a socket, the disk, or the GPU directly: it asks the
//! shell for nothing and draws nothing; it only builds lists.

use crate::images::{ImageOutcome, ImageRequest, ImageStore, document_title};
use crate::js_host::{MediaRequest, TrackRequest};
use crate::page::{self, Document};
use cosmic::iced::futures::SinkExt;
use cosmic::iced::futures::channel::mpsc;
use pluto_html::NodeId;
use pluto_ipc::{
    BrowserToRenderer, CommandReceiver, CredentialsMode, DeltaMode, Destination, InputEvent,
    Origin, PointerKind, Priority, RendererToBrowser, RequestMode, ResourceRequest, WheelPhase,
    origin_of,
};
use pluto_ipc::transport::{recv_browser_message, Transport};
use pluto_js::DomHost;
use pluto_js::TabJs;
use pluto_js::bindings::events::TypedEventKind;
use pluto_css::{CalcCtx, Easing, parse_value};
use pluto_js::host::WaapiOp;
use pluto_layout::animation::{
    AnimTiming, AnimatedValue, AnimationDirection, AnimationEngine, FillMode, IterationCount,
};
use pluto_layout::{
    ElementState, ImageMap, LayoutTree, OverscrollBehavior, RelayoutKind, RelayoutOutcome,
    Theme, VideoInfo,
};
use pluto_net::{ResourceLoader, SharedLoader};
use pluto_paint::{
    CompositedUpdate, DisplayList, Resource, ResourceId, ResourceTable, perf,
    svg::SvgRasterCache,
};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Rendering-step observer helpers (Stage 8 Rec 8): measuring
/// observed targets on the retained layout tree. Declared here
/// with an explicit path so `main.rs` needs no new `mod` line.
#[path = "observers.rs"]
pub(crate) mod observers;

/// The subscription that runs the worker for one tab (Stage 11
/// Rec 4): each tab gets its own subscription, keyed by the
/// shell-assigned tab id (`Subscription::run_with_id`), so the
/// shell can tell the tabs' renderer events apart. The shell's
/// `Application` wires one of these per tab in via `fn subscription`.
pub fn renderer_subscription_for_tab(
    tab_id: u64,
) -> cosmic::iced::Subscription<crate::Message> {
    // `run_with` keys the subscription by the tab id (the `D` in
    // `run_with(data, builder)` is hashed for identity), so each
    // tab's worker stream is distinct and iced starts/stops them
    // as tabs open and close.
    cosmic::iced::Subscription::run_with(tab_id, |tab_id| worker_stream(*tab_id))
}

/// The worker stream for one tab: create the command channel, hand
/// its sender to the shell, then bridge the out-of-process renderer
/// into the shell until the subscription ends (Stage 9 Rec 9).
/// The tab id rides the `RendererReady` message so the shell
/// binds the command sender to the right tab.
fn worker_stream(tab_id: u64) -> impl cosmic::iced::futures::stream::Stream<Item = crate::Message> {
    cosmic::iced::stream::channel(100, async move |mut output| {
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let _ = output
            .send(crate::Message::RendererReady(tab_id, cmd_tx))
            .await;
        match std::env::current_exe() {
            Ok(exe) => {
                run_renderer_bridge(tab_id, &exe, cmd_rx, output, inprocess_fallback_allowed())
                    .await
            }
            Err(e) => {
                // Stage 13 Rec 12d: no executable to spawn and the
                // in-process fallback is OFF by default — report
                // the crash and stay alive. The bridge is entered
                // with an unspawnable path, so it reports the
                // spawn failure as a crash and awaits a
                // navigation (a later Reload retries the spawn).
                eprintln!("[pluto] cannot find own executable ({e})");
                let allow = inprocess_fallback_allowed();
                if allow {
                    eprintln!(
                        "[pluto] PLUTO_ALLOW_INPROCESS_FALLBACK=1 — falling back to the in-process renderer"
                    );
                    run_worker(tab_id, cmd_rx, output).await;
                } else {
                    run_renderer_bridge(tab_id, std::path::Path::new(""), cmd_rx, output, allow)
                        .await;
                }
            }
        }
    })
}

/// Bridge one out-of-process renderer into the shell (Stage 9
/// Rec 9): one `RendererProcess` per tab. Shell commands flow to
/// the child over the socketpair; the child's events flow back as
/// `Message::RendererEvent(tab_id, ...)`.
///
/// If the child dies — clean EOF, a dead socket, or a malformed
/// message the transport treats as compromised — the shell gets
/// `RendererCrashed` (the stale list stays visible, per the
/// shell's existing crash handling) and the bridge stays alive:
/// the next `CommitNavigation` spawns a fresh renderer and the
/// page recovers. Commands other than navigation are dropped
/// while no renderer is alive — there is no page to act on.
///
/// If the child cannot be spawned at all, the bridge reports a
/// crash and stays alive awaiting a navigation (Stage 13 Rec 12d:
/// the in-process fallback runs only with
/// `PLUTO_ALLOW_INPROCESS_FALLBACK=1`, loudly logged, never
/// silent).
async fn run_renderer_bridge(
    tab_id: u64,
    exe: &std::path::Path,
    mut commands: CommandReceiver,
    mut output: mpsc::Sender<crate::Message>,
    allow_inprocess_fallback: bool,
) {
    // Stage 13 Rec 12b: the watchdog — the last instant any
    // event (heartbeat or otherwise) arrived from the renderer.
    let timeout = watchdog_timeout();
    let mut last_progress = tokio::time::Instant::now();
    let mut watchdog_tick = tokio::time::interval(Duration::from_secs(1));
    // Consume the immediate first tick: the first real check is
    // one second in.
    watchdog_tick.tick().await;
    let mut renderer =
        match crate::renderer_process::RendererProcess::spawn(exe, tab_id).await {
            Ok(renderer) => Some(renderer),
            Err(e) => {
                eprintln!("[pluto] renderer process spawn failed ({e})");
                if allow_inprocess_fallback {
                    eprintln!(
                        "[pluto] PLUTO_ALLOW_INPROCESS_FALLBACK=1 — falling back to the in-process renderer"
                    );
                    run_worker(tab_id, commands, output).await;
                    return;
                }
                // Fail closed (Stage 13 Rec 12d): report the crash
                // and stay alive with no renderer — the loop below
                // respawns on the next `CommitNavigation`.
                let _ = output
                    .send(crate::Message::RendererEvent(
                        tab_id,
                        RendererToBrowser::RendererCrashed {
                            reason: format!(
                                "renderer spawn failed: {e} (in-process fallback disabled)"
                            ),
                        },
                    ))
                    .await;
                None
            }
        };
    // Whether the shell has asked for shutdown: a clean EOF after
    // `Shutdown` is an orderly exit, not a crash.
    let mut shutting_down = false;
    loop {
        tokio::select! {
            cmd = commands.recv() => {
                let Some(cmd) = cmd else {
                    // The shell is gone: shut the renderer down
                    // gracefully, then end the bridge.
                    if let Some(renderer) = renderer.take() {
                        renderer.close().await;
                    }
                    return;
                };
                if matches!(cmd, BrowserToRenderer::Shutdown) {
                    shutting_down = true;
                }
                if renderer.is_some() {
                    let send_failed = renderer
                        .as_mut()
                        .unwrap()
                        .send(&cmd)
                        .await
                        .is_err();
                    if send_failed {
                        report_renderer_crash(
                            tab_id,
                            &mut output,
                            &mut renderer,
                            "renderer IPC send failed",
                            shutting_down,
                        )
                        .await;
                    }
                } else if matches!(cmd, BrowserToRenderer::CommitNavigation { .. }) {
                    // No renderer alive: a navigation revives it —
                    // a fresh child for a fresh page.
                    match crate::renderer_process::RendererProcess::spawn(exe, tab_id).await {
                        Ok(mut fresh) => {
                            if fresh.send(&cmd).await.is_err() {
                                report_renderer_crash(
                                    tab_id,
                                    &mut output,
                                    &mut Some(fresh),
                                    "respawned renderer IPC send failed",
                                    false,
                                )
                                .await;
                            } else {
                                renderer = Some(fresh);
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[pluto] renderer respawn failed ({e}) — page stays crashed"
                            );
                            let _ = output
                                .send(crate::Message::RendererEvent(
                                    tab_id,
                                    RendererToBrowser::RendererCrashed {
                                        reason: format!("renderer respawn failed: {e}"),
                                    },
                                ))
                                .await;
                        }
                    }
                }
            }
            event = async {
                match renderer.as_mut() {
                    Some(renderer) => renderer.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match event {
                    Ok(Some(event)) => {
                        // Stage 13 Rec 12b: any event — heartbeat
                        // or otherwise — proves the renderer is
                        // alive; the watchdog clock restarts here.
                        last_progress = tokio::time::Instant::now();
                        if output
                            .send(crate::Message::RendererEvent(tab_id, event))
                            .await
                            .is_err()
                        {
                            // The shell is gone.
                            if let Some(renderer) = renderer.take() {
                                renderer.close().await;
                            }
                            return;
                        }
                    }
                    Ok(None) => {
                        // Clean EOF: the renderer exited on its
                        // own. After `Shutdown` that is orderly;
                        // otherwise it is a crash.
                        if !shutting_down {
                            let _ = output
                                .send(crate::Message::RendererEvent(
                                    tab_id,
                                    RendererToBrowser::RendererCrashed {
                                        reason: "renderer exited".to_string(),
                                    },
                                ))
                                .await;
                        }
                        renderer = None;
                    }
                    Err(e) => {
                        report_renderer_crash(
                            tab_id,
                            &mut output,
                            &mut renderer,
                            &format!("renderer IPC receive failed: {e}"),
                            shutting_down,
                        )
                        .await;
                    }
                }
            }
            // Stage 13 Rec 12b: the watchdog — a renderer that
            // has sent nothing (no heartbeat, no progress) for
            // the whole timeout is hung, not slow: kill it and
            // report the crash. The bridge stays alive with no
            // renderer; the next navigation respawns. A shutdown
            // in progress never fires it (the child is exiting
            // on purpose, and its silence is expected).
            _ = watchdog_tick.tick(), if timeout.is_some() => {
                if let Some(limit) = timeout
                    && !shutting_down
                    && renderer.is_some()
                    && last_progress.elapsed() > limit
                {
                    eprintln!(
                        "[pluto] renderer watchdog: no heartbeat or progress for {}s — killing",
                        limit.as_secs()
                    );
                    report_renderer_crash(
                        tab_id,
                        &mut output,
                        &mut renderer,
                        &format!(
                            "renderer unresponsive (watchdog: no heartbeat or progress for {}s)",
                            limit.as_secs()
                        ),
                        shutting_down,
                    )
                    .await;
                }
            }
        }
    }
}

/// Report a renderer death to the shell and drop the dead handle
/// (the bridge stays alive; the next navigation respawns). A
/// shutdown-orderly death reports nothing. Dropping the handle
/// SIGKILLs + reaps a still-alive child, so no zombie leaks.
async fn report_renderer_crash(
    tab_id: u64,
    output: &mut mpsc::Sender<crate::Message>,
    renderer: &mut Option<crate::renderer_process::RendererProcess>,
    reason: &str,
    shutting_down: bool,
) {
    if !shutting_down {
        let _ = output
            .send(crate::Message::RendererEvent(
                tab_id,
                RendererToBrowser::RendererCrashed {
                    reason: reason.to_string(),
                },
            ))
            .await;
    }
    renderer.take();
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
/// A collected display list plus the resources it references.
type CollectedDisplayList = (Arc<DisplayList>, Vec<(ResourceId, Resource)>);
/// A collected display list for a viewport scroll, plus whether the
/// scroll re-collect changed anything.
type ScrolledDisplayList = (Arc<DisplayList>, bool, Vec<(ResourceId, Resource)>);

/// Wave 2b adds the geometry jobs (`Geometry` / `FlushGeometry`):
/// the JS host's forced synchronous layout flush (the only
/// sanctioned synchronous cross-thread call in the design) and the
/// click hit-test (`HitTest`). They reply over `std` channels /
/// oneshots the tab thread blocks on, because the tab thread is a
/// plain OS thread, not an async task.
/// `Mp4Video` carries the demuxer, track index, and decoder, so it is
/// inherently far larger than the other variants; boxing it would churn
/// every construction and match site for no behavior gain.
#[allow(clippy::large_enum_variant)]
pub(crate) enum LayoutJob {
    /// (Re-)lay out at `viewport` with `state`, re-applying the
    /// retained `offsets` after the rebuild — but build NO display
    /// list yet (Stage 8 Rec 8/10: the ResizeObserver and
    /// IntersectionObserver steps run after layout and before the
    /// list is built; see [`layout_and_submit`]). `document: Some` replaces the
    /// actor's document (a new navigation / error page / a
    /// JS-mutated document); `None` re-lays out the retained one
    /// (a resize).
    LayoutTree {
        document: Box<Option<Document>>,
        viewport: (f32, f32),
        state: ElementState,
        offsets: HashMap<NodeId, (f32, f32)>,
        /// Try the Stage 8 Rec 6 incremental path on the actor's
        /// retained tree (the incoming document is a snapshot of
        /// the same live `Dom` — a dirty tick or a resize — so its
        /// node ids match the retained tree). False for a fresh
        /// navigation / error page, whose ids belong to a new
        /// document: the actor must full-rebuild. A failed
        /// incremental attempt falls back to a full rebuild
        /// either way.
        ///
        /// The incremental path takes an image map only when one
        /// arrived since the last layout (Stage 12 Rec 5): the tab
        /// thread sends `images_dirty` with the fresh map, and the
        /// actor hands it to `relayout_incremental`, which treats
        /// the changed `<img>` nodes as damage. Without the flag
        /// the retained map is current. `images` is used on the
        /// full-rebuild path always.
        incremental: bool,
        /// Whether `images` is fresh (an image arrived since the
        /// last layout). The actor passes the map to the
        /// incremental path only when this is set.
        images_dirty: bool,
        /// The decoded-image map for this layout pass (Stage 11
        /// Rec 1/2): per-`<img>` intrinsic sizes + background
        /// images, built by the tab thread from its `ImageStore`.
        /// Passed to the incremental path only when `images_dirty`
        /// is set (see above); used on the full-rebuild path always.
        images: ImageMap,
        /// The shell's theme (M6 Phase 3, Rec 12): a change
        /// forces a full rebuild (restyle, never a refetch).
        theme: Theme,
        /// The viewport (page) scroll the tree mirrors (M6 Phase
        /// 3, Rec 1/3): sticky offsets are scroll-derived, so the
        /// tree re-collects when it changes.
        viewport_scroll: (f32, f32),
        /// In-flight sampled animation values (M6 Phase 3, Rec
        /// 11): the actor applies them over the resolved styles
        /// before classification, so an animated transform routes
        /// through the CompositingOnly path.
        animation_overrides: Vec<(NodeId, Vec<(String, AnimatedValue)>)>,
        /// Downloaded `@font-face` bytes waiting to be loaded
        /// (M6 Phase 3, Rec 12): the actor loads them into the
        /// retained FontSystem; a new face forces a full rebuild.
        fonts: Vec<Vec<u8>>,
        /// Which path the layout took (Stage 12 Rec 3): the actor
        /// reports the [`RelayoutOutcome`] — `ColdBuild` for a fresh
        /// tree, the incremental outcome, or `FullLayout` when the
        /// incremental attempt failed or was not attempted against
        /// a retained tree — so the tab thread can record it in the
        /// frame-time histogram and the fallback rate is measured,
        /// not guessed. For a `CompositingOnly` outcome the
        /// `composited` ids name the dirty elements, so the tab
        /// thread can ask for their new compositing state (see
        /// [`LayoutJob::CompositedState`]) instead of rebuilding
        /// the list.
        reply: oneshot::Sender<RelayoutOutcome>,
    },
    /// The compositing state (page transform + opacity) of `ids`
    /// and their descendants, from the actor's retained tree (M6
    /// Phase 2, Rec 9): the tab thread asks for it after a
    /// `CompositingOnly` layout, so it can send a
    /// `CompositingUpdate` with just the changed elements instead
    /// of rebuilding the display list. Empty when the actor has
    /// no tree.
    CompositedState {
        ids: Vec<NodeId>,
        reply: oneshot::Sender<Vec<CompositedUpdate>>,
    },
    /// Build the display list from the actor's retained tree
    /// (after the ResizeObserver step has run). The reply also
    /// carries the fresh raster Blobs (`build_with_svg_and_canvas`,
    /// Stage 15 Rec 5/8): `(ResourceId, Resource)` pairs the tab
    /// thread inserts into its image cache and announces with
    /// `ResourceReady` before submitting the list. Empty when
    /// the list has no new raster content (a scroll re-collect
    /// names the cached ids again, so nothing is returned twice).
    /// `canvas_live` names every live canvas element (its Blob
    /// item is emitted whenever it is live); `canvas_new` carries
    /// the bitmaps that changed since the last announcement;
    /// `svg_images` maps `<svg><image>` nodes to their decoded
    /// resource ids (from the last image map).
    CollectDisplayList {
        canvas_live: Vec<NodeId>,
        canvas_new: Vec<(NodeId, Resource)>,
        svg_images: HashMap<NodeId, ResourceId>,
        reply: oneshot::Sender<CollectedDisplayList>,
    },
    /// Sizes + DOM depths for observed targets, from the retained
    /// tree + document (Stage 8 Rec 8). `None` size = unrendered.
    ResizeSizes {
        ids: Vec<NodeId>,
        reply: oneshot::Sender<Vec<(NodeId, Option<pluto_js::ObservedSizes>, usize)>>,
    },
    /// Intersection geometry for observed targets, from the retained
    /// tree + document (Stage 8 Rec 10). `None` = the target (or an
    /// element root) has no box — unrendered, so no entry fires.
    /// `viewport_scroll` is the worker's mirrored viewport scroll
    /// (see `TabState::viewport_scroll`).
    IntersectionGeometry {
        queries: Vec<observers::IntersectionQuery>,
        viewport_scroll: (f32, f32),
        reply: oneshot::Sender<Vec<Option<pluto_js::ComputedIntersection>>>,
    },
    /// The innermost-first chain of scroll containers containing
    /// the point (page coordinates), each with its
    /// `overscroll-behavior` per axis (Stage 8 Rec 9). The worker
    /// applies a wheel to the chain in order, chaining the leftover
    /// delta to the next container — and to the viewport when the
    /// chain is exhausted. Root/body are excluded (their overflow
    /// propagates to the viewport).
    ScrollChain {
        x: f32,
        y: f32,
        reply: oneshot::Sender<Vec<ScrollChainEntry>>,
    },
    /// Scroll several containers at once: apply all offsets, then
    /// re-collect ONE display list (Stage 8 Rec 9 — a wheel that
    /// crosses two containers in one gesture must not submit
    /// twice). `None` when the actor holds no tree. The `Some`
    /// payload also carries the fresh raster Blobs (see
    /// [`LayoutJob::CollectDisplayList`]) — empty on a pure
    /// scroll, since sizes never change and the raster cache hits.
    /// The canvas / svg-image payload is the same as
    /// `CollectDisplayList`'s: a scroll re-collect must still
    /// name live canvases, or they would vanish on scroll.
    ScrollMany {
        scrolls: Vec<(NodeId, f32, f32)>,
        canvas_live: Vec<NodeId>,
        canvas_new: Vec<(NodeId, Resource)>,
        svg_images: HashMap<NodeId, ResourceId>,
        reply: oneshot::Sender<Option<CollectedDisplayList>>,
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
    /// Maximum scroll offset of one scroll container (Stage 10 Rec
    /// 2 — see [`LayoutTree::scrollable_size`]); `None` when the
    /// actor holds no tree / the node has no box.
    ScrollMax {
        id: NodeId,
        reply: oneshot::Sender<Option<(f32, f32)>>,
    },
    /// Maximum viewport (page) scroll offset, per axis: the tree's
    /// `content - viewport`, clamped at zero (Stage 10 Rec 2).
    /// `(0, 0)` when the actor holds no tree.
    PageMaxScroll {
        reply: oneshot::Sender<(f32, f32)>,
    },
    /// Geometry of one node from the actor's retained tree, with NO
    /// flush (the caller already knows the layout is clean).
    /// `None` when the actor holds no tree / node.
    Geometry {
        id: NodeId,
        reply: std::sync::mpsc::Sender<Option<pluto_js::Geometry>>,
    },
    /// Geometry of one node from the actor's retained tree, with NO
    /// flush — same answer as `Geometry`, but a oneshot reply so
    /// the async tab thread can await it directly (scroll-to-anchor,
    /// Stage 16 Phase 3 Rec 9). `None` when the actor holds no tree
    /// / node.
    NodeRect {
        id: NodeId,
        reply: oneshot::Sender<Option<pluto_js::Geometry>>,
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
    /// Set the viewport (page) scroll offset on the actor's
    /// retained tree (M6 Phase 3, Rec 1/3): sticky offsets are
    /// scroll-derived and fixed boxes hit-test in viewport
    /// coordinates, so the tree must know the scroll — and the
    /// re-collected display list is the reply, so the tab thread
    /// submits it in the same frame when it changed. The reply
    /// also carries whether a sticky box moved: a pure viewport
    /// scroll with no sticky content leaves the list identical,
    /// so the worker skips the submit (the shell scrolls its
    /// cached list) and the event stream stays clean — a spurious
    /// submit here used to poison the wheel/resize tests' event
    /// order. `None` when the actor holds no tree. The third
    /// payload element is the fresh raster Blobs (see
    /// [`LayoutJob::CollectDisplayList`]). The canvas / svg-image
    /// payload is the same as `CollectDisplayList`'s (a viewport
    /// scroll that moves a sticky box submits the re-collected
    /// list, which must still name live canvases).
    SetViewportScroll {
        x: f32,
        y: f32,
        canvas_live: Vec<NodeId>,
        canvas_new: Vec<(NodeId, Resource)>,
        svg_images: HashMap<NodeId, ResourceId>,
        reply: oneshot::Sender<Option<ScrolledDisplayList>>,
    },
}

/// One entry of a wheel scroll chain (Stage 8 Rec 9): the scroll
/// container and its `overscroll-behavior` per axis — whether a
/// scroll reaching this container's limit chains onward or stops
/// here — plus the container's maximum scroll offset per axis
/// (Stage 10 Rec 2), so the wheel clamps at the container's end
/// instead of scrolling past it into blank space.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScrollChainEntry {
    pub id: NodeId,
    pub overscroll_x: OverscrollBehavior,
    pub overscroll_y: OverscrollBehavior,
    pub max_scroll: (f32, f32),
}

/// The innermost-first chain of scroll containers containing the
/// point (Stage 8 Rec 9): every scroll container whose rect
/// contains the point, deepest first, ties broken by rect area
/// (the smaller, more nested container wins). Root/body are
/// excluded — their overflow propagates to the viewport, which the
/// worker scrolls with a `ScrollViewport` message instead. Each
/// entry carries the container's `overscroll-behavior` per axis,
/// read from the retained tree's computed styles.
fn scroll_chain_for(
    tree: &LayoutTree,
    doc: &Document,
    x: f32,
    y: f32,
) -> Vec<ScrollChainEntry> {
    let dom = doc.dom();
    let mut chain: Vec<(usize, f32, ScrollChainEntry)> = tree
        .scroll_containers()
        .into_iter()
        .filter(|&(id, rect)| {
            // Rect contains the point, and the container is not
            // root/body.
            x >= rect[0]
                && x < rect[0] + rect[2]
                && y >= rect[1]
                && y < rect[1] + rect[3]
                && !matches!(dom.tag(id), Some("html" | "body"))
        })
        .map(|(id, rect)| {
            // DOM depth: walk the parent chain. A detached node
            // (no depth) sorts shallowest.
            let mut depth = 0usize;
            let mut cur = dom.get(id).and_then(|n| n.parent);
            while let Some(parent) = cur {
                depth += 1;
                cur = dom.get(parent).and_then(|n| n.parent);
            }
            let (overscroll_x, overscroll_y) = tree
                .node_styles
                .get(id)
                .map(|s| {
                    (
                        s.computed.overscroll_behavior_x,
                        s.computed.overscroll_behavior_y,
                    )
                })
                .unwrap_or((OverscrollBehavior::Auto, OverscrollBehavior::Auto));
            (
                depth,
                rect[2] * rect[3],
                ScrollChainEntry {
                    id,
                    overscroll_x,
                    overscroll_y,
                    // A container the tree cannot measure (no box)
                    // scrolls nowhere: the whole delta chains on.
                    max_scroll: tree.scrollable_size(id).unwrap_or((0.0, 0.0)),
                },
            )
        })
        .collect();
    // Deepest first; a smaller rect wins a depth tie (the more
    // nested container is usually the smaller one).
    chain.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)));
    chain.into_iter().map(|(_, _, entry)| entry).collect()
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

/// Full rebuild of the layout tree for `doc`: the fallback for a
/// failed incremental attempt and the path for fresh navigations.
/// Builds the tree directly (not via `layout_document_with_state`,
/// which also builds a display list that would be thrown away here
/// — the list is built after the ResizeObserver step, by
/// `CollectDisplayList`). A layout failure renders the error page
/// through the same pipeline. `old_tree` is the retained tree
/// being replaced: its `FontSystem` is reused for the new build
/// (Stage 8 Rec 9) — a fresh `FontSystem` re-scans the system
/// fonts (~1.5 s), and a hover restyle that lands on the document
/// root would otherwise pay that per pointer event. `images` is
/// the decoded-image map for this pass (Stage 11 Rec 1/2).
fn build_full_tree(
    doc: &Document,
    viewport: (f32, f32),
    state: &ElementState,
    old_tree: Option<LayoutTree>,
    images: &ImageMap,
    theme: Theme,
) -> LayoutTree {
    let old_fonts = old_tree.and_then(|t| t.font_system);
    let result = match old_fonts {
        Some(fonts) => LayoutTree::build_with_state_reusing_font_system_and_images(
            doc.dom(),
            doc.sheet(),
            viewport.0,
            viewport.1,
            state,
            fonts,
            images,
            theme,
        ),
        None => LayoutTree::build_with_state_and_images(
            doc.dom(),
            doc.sheet(),
            viewport.0,
            viewport.1,
            state,
            images,
            theme,
        ),
    };
    match result {
        Ok(tree) => tree,
        Err(e) => {
            let err_doc = page::parse_document(&page::error_document(
                "about:blank",
                &format!("layout failed: {e}"),
            ));
            LayoutTree::build_with_state_and_images(
                err_doc.dom(),
                err_doc.sheet(),
                viewport.0,
                viewport.1,
                state,
                images,
                theme,
            )
            .unwrap_or_default()
        }
    }
}

/// The layout actor: owns the tree + document, answers jobs until the
/// channel closes. Runs on its own OS thread (`pluto-layout`), so the
/// `!Send` tree never crosses an await or a thread boundary.
pub(crate) fn layout_thread(jobs: std::sync::mpsc::Receiver<LayoutJob>) {
    let mut tree: Option<LayoutTree> = None;
    let mut document: Option<Document> = None;
    // Stage 15 Rec 5: the worker-side raster cache for inline
    // SVGs and radial CSS gradient backgrounds (`build_with_svg`).
    // Keyed by element id + bitmap size + content fingerprint —
    // a new document means new node ids, so the cache clears
    // whenever a document is installed (slot ids can be reused
    // by a fresh `Dom`, and a stale bitmap would paint the wrong
    // svg).
    let mut svg_cache = SvgRasterCache::new();
    while let Ok(job) = jobs.recv() {
        match job {
            LayoutJob::LayoutTree {
                document: new_document,
                viewport,
                state,
                offsets,
                incremental,
                images,
                images_dirty,
                theme,
                viewport_scroll,
                animation_overrides,
                fonts,
                reply,
            } => {
                if let Some(doc) = *new_document {
                    document = Some(doc);
                    svg_cache.clear();
                }
                let Some(doc) = document.as_ref() else {
                    let _ = reply.send(RelayoutOutcome {
                        kind: RelayoutKind::NothingToDo,
                        reason: "no document",
                        composited: Vec::new(),
                        pending_transitions: Vec::new(),
                        pending_animations: Vec::new(),
                        changed_styles: Vec::new(),
                    });
                    continue;
                };
                // M6 Phase 3 (Rec 3): the retained tree mirrors
                // the viewport scroll (sticky offsets are
                // scroll-derived). Applied before the incremental
                // attempt so the pass sees current offsets.
                if let Some(retained) = tree.as_mut()
                    && retained.viewport_scroll != viewport_scroll
                {
                    retained.set_viewport_scroll(viewport_scroll.0, viewport_scroll.1);
                }
                // M6 Phase 3 (Rec 12): a theme change forces a
                // full rebuild — restyle, never a refetch. The
                // retained tree records it; the fresh build below
                // (or the incremental path's `full_rebuild`) uses
                // it. `set_theme` only records — the rebuild is
                // what applies it.
                let mut theme_changed = false;
                if let Some(retained) = tree.as_mut()
                    && retained.theme != theme
                {
                    retained.set_theme(theme);
                    theme_changed = true;
                }
                // M6 Phase 3 (Rec 12): newly downloaded
                // `@font-face` bytes land in the retained
                // FontSystem; a new face forces a full rebuild
                // (like an image arrival — the fresh tree builds
                // with the new faces, falling back until they
                // arrive). On a fresh navigation there is no
                // retained tree yet — fonts arrive after the
                // commit, so the first layout never carries any.
                let mut fonts_landed = false;
                if !fonts.is_empty()
                    && let Some(retained) = tree.as_mut()
                    && retained.load_fonts(&fonts)
                {
                    fonts_landed = true;
                }
                // Stage 8 Rec 6: a dirty tick / resize hands the
                // actor a snapshot of the same live Dom the
                // retained tree was built from, so re-lay out
                // incrementally on the retained tree — dirty
                // subtree only, full rebuild only when the root
                // is unsafe (the function decides, and reports
                // which path ran). A fresh navigation's document
                // has new node ids, so it never takes this path
                // (`incremental` false). A failed incremental
                // attempt discards the retained tree and rebuilds
                // from scratch — never a stale or half-updated
                // tree.
                //
                // Stage 12 Rec 3: the kind is reported back so the
                // tab thread can record it. `ColdBuild` is the
                // default (a fresh tree with no retained tree to
                // be incremental against); a retained tree that
                // was rebuilt without attempting the incremental
                // path (resize / flush) is a `FullLayout` fallback,
                // and a failed incremental attempt is one too —
                // the fallback rate counts exactly the layouts
                // that could have been incremental but were not.
                let mut built_fresh = true;
                let mut kind = RelayoutKind::ColdBuild;
                // The incremental outcome rides back to the tab
                // thread whole (M6 Phase 2, Rec 9): a
                // `CompositingOnly` outcome names its dirty
                // elements in `composited`, so the tab thread can
                // ask for their new state instead of rebuilding
                // the list. The fallback paths have no dirty
                // elements, so they report a bare outcome.
                let mut outcome: Option<RelayoutOutcome> = None;
                // A theme change or a newly landed web font is a
                // full rebuild, not an incremental pass (the whole
                // cascade re-evaluates under the new theme / faces).
                let force_full = theme_changed || fonts_landed;
                if incremental && !force_full
                    && let Some(retained) = tree.as_mut()
                {
                    // M6 Phase 3 (Rec 11): the tab thread's
                    // in-flight sampled values ride over the
                    // resolved styles for this pass.
                    retained.animation_overrides = animation_overrides.clone();
                    match retained.relayout_incremental(
                        doc.dom(),
                        doc.sheet(),
                        viewport.0,
                        viewport.1,
                        &state,
                        // Stage 12 Rec 5: the fresh map rides the
                        // incremental path only when an image
                        // arrived; otherwise the retained map is
                        // current and `None` keeps it untouched.
                        if images_dirty { Some(&images) } else { None },
                    ) {
                        Ok(o) => {
                            eprintln!(
                                "[pluto] incremental layout: {:?} ({})",
                                o.kind, o.reason
                            );
                            kind = o.kind;
                            built_fresh = false;
                            outcome = Some(o);
                        }
                        Err(e) => {
                            eprintln!(
                                "[pluto] incremental layout failed ({e}), full rebuild"
                            );
                            kind = RelayoutKind::FullLayout;
                        }
                    }
                } else if tree.is_some() {
                    kind = RelayoutKind::FullLayout;
                }
                if built_fresh {
                    // Hand the retained tree's FontSystem to the
                    // rebuild (Stage 8 Rec 9): a fresh one would
                    // re-scan the system fonts per rebuild.
                    let old_tree = tree.take();
                    let mut new_tree =
                        build_full_tree(doc, viewport, &state, old_tree, &images, theme);
                    if !offsets.is_empty() {
                        for (&id, &(x, y)) in &offsets {
                            new_tree.set_scroll_offset(id, x, y);
                        }
                    }
                    // M6 Phase 3 (Rec 11): a fresh build resolves
                    // underlying styles only — re-apply the
                    // in-flight sampled values over its meta
                    // styles, like `full_rebuild` does.
                    new_tree.animation_overrides = animation_overrides;
                    if !new_tree.animation_overrides.is_empty() {
                        new_tree.apply_animation_overrides_to_meta();
                    }
                    // M6 Phase 3 (Rec 3): a fresh tree starts at
                    // scroll (0, 0) — mirror the job's viewport
                    // scroll so sticky is right on the first list.
                    if viewport_scroll != (0.0, 0.0) {
                        new_tree.set_viewport_scroll(viewport_scroll.0, viewport_scroll.1);
                    }
                    tree = Some(new_tree);
                } else if !offsets.is_empty() {
                    // Incremental success: the retained tree was
                    // updated in place; re-apply the scroll
                    // offsets the same way (they survive in the
                    // tree, but the job's map is authoritative).
                    if let Some(retained) = tree.as_mut() {
                        for (&id, &(x, y)) in &offsets {
                            retained.set_scroll_offset(id, x, y);
                        }
                    }
                }
                let outcome = outcome.unwrap_or(RelayoutOutcome {
                    kind,
                    reason: match kind {
                        RelayoutKind::ColdBuild => "fresh tree",
                        _ => "incremental layout failed or not attempted, full rebuild",
                    },
                    composited: Vec::new(),
                    pending_transitions: Vec::new(),
                    pending_animations: Vec::new(),
                    changed_styles: Vec::new(),
                });
                let _ = reply.send(outcome);
            }
            LayoutJob::CompositedState { ids, reply } => {
                // M6 Phase 2 (Rec 9): the compositing state of
                // the dirty elements AND their descendants — a
                // transform change moves the whole subtree, so
                // every descendant box's page transform changes
                // with it. Empty when the actor has no tree yet.
                let updates = tree
                    .as_ref()
                    .map(|tree| {
                        tree.composited_subtree_state(&ids)
                            .into_iter()
                            .map(|(id, matrix, opacity)| CompositedUpdate {
                                id,
                                matrix,
                                opacity,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let _ = reply.send(updates);
            }
            LayoutJob::CollectDisplayList {
                canvas_live,
                canvas_new,
                svg_images,
                reply,
            } => {
                let (display, blobs) = match tree.as_ref() {
                    Some(tree) => match document.as_ref() {
                        Some(doc) => {
                            let (list, blobs) =
                                DisplayList::build_with_svg_and_canvas(
                                    tree,
                                    doc.dom(),
                                    &mut svg_cache,
                                    &svg_images,
                                    &canvas_live,
                                    &canvas_new,
                                );
                            (Arc::new(list), blobs)
                        }
                        None => (Arc::new(DisplayList::build(tree)), Vec::new()),
                    },
                    None => (Arc::new(DisplayList::new()), Vec::new()),
                };
                let _ = reply.send((display, blobs));
            }
            LayoutJob::ResizeSizes { ids, reply } => {
                let rows = match (tree.as_ref(), document.as_ref()) {
                    (Some(tree), Some(doc)) => observers::resize_sizes_for(tree, doc.dom(), &ids),
                    _ => ids.iter().map(|&id| (id, None, 0)).collect(),
                };
                let _ = reply.send(rows);
            }
            LayoutJob::IntersectionGeometry {
                queries,
                viewport_scroll,
                reply,
            } => {
                let rows = match (tree.as_ref(), document.as_ref()) {
                    (Some(tree), Some(doc)) => observers::intersection_geometry_for(
                        tree,
                        doc.dom(),
                        &queries,
                        viewport_scroll,
                    ),
                    _ => queries.iter().map(|_| None).collect(),
                };
                let _ = reply.send(rows);
            }
            LayoutJob::ScrollChain { x, y, reply } => {
                let chain = match (tree.as_ref(), document.as_ref()) {
                    (Some(tree), Some(doc)) => scroll_chain_for(tree, doc, x, y),
                    _ => Vec::new(),
                };
                let _ = reply.send(chain);
            }
            LayoutJob::ScrollMany {
                scrolls,
                canvas_live,
                canvas_new,
                svg_images,
                reply,
            } => {
                let display = tree.as_mut().map(|tree| {
                    for (id, x, y) in scrolls {
                        tree.set_scroll_offset(id, x, y);
                    }
                    let (list, blobs) = match document.as_ref() {
                        Some(doc) => DisplayList::build_with_svg_and_canvas(
                            tree,
                            doc.dom(),
                            &mut svg_cache,
                            &svg_images,
                            &canvas_live,
                            &canvas_new,
                        ),
                        None => (DisplayList::build(tree), Vec::new()),
                    };
                    (Arc::new(list), blobs)
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
            LayoutJob::ScrollMax { id, reply } => {
                let max = tree.as_ref().and_then(|tree| tree.scrollable_size(id));
                let _ = reply.send(max);
            }
            LayoutJob::PageMaxScroll { reply } => {
                let max = tree
                    .as_ref()
                    .map(|tree| {
                        (
                            (tree.content_width() - tree.viewport.0).max(0.0),
                            (tree.content_height() - tree.viewport.1).max(0.0),
                        )
                    })
                    .unwrap_or((0.0, 0.0));
                let _ = reply.send(max);
            }
            LayoutJob::Geometry { id, reply } => {
                let geometry = tree.as_ref().and_then(|tree| geometry_of(tree, id));
                let _ = reply.send(geometry);
            }
            LayoutJob::NodeRect { id, reply } => {
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
                svg_cache.clear();
                let doc = document.as_ref().unwrap();
                // A forced flush rebuilds the tree: keep the
                // retained tree's images (Stage 11 Rec 1) — a
                // flush that dropped them would un-render every
                // image until the next tick's rebuild.
                let old_tree = tree.take();
                let images = old_tree
                    .as_ref()
                    .map(|t| t.images.clone())
                    .unwrap_or_default();
                // A flush keeps the retained tree's theme (a
                // theme change arrives through the layout job,
                // not through a flush).
                let theme = old_tree
                    .as_ref()
                    .map(|t| t.theme)
                    .unwrap_or(Theme::Light);
                let new_tree = build_full_tree(doc, viewport, &state, old_tree, &images, theme);
                let geometry = geometry_of(&new_tree, id);
                tree = Some(new_tree);
                let _ = reply.send(geometry);
            }
            LayoutJob::HitTest { x, y, reply } => {
                let target = tree.as_ref().and_then(|tree| tree.hit_test(x, y));
                let _ = reply.send(target);
            }
            LayoutJob::SetViewportScroll {
                x,
                y,
                canvas_live,
                canvas_new,
                svg_images,
                reply,
            } => {
                // M6 Phase 3 (Rec 1/3): the tree learns the
                // viewport scroll — sticky offsets are
                // scroll-derived, fixed boxes hit-test in viewport
                // coordinates — and the re-collected display list
                // is the reply, so the tab thread submits it in
                // the same frame. No layout runs (no observers):
                // a scroll frame is a re-collect, like
                // `ScrollMany`.
                let display = tree.as_mut().map(|tree| {
                    let sticky_moved = tree.set_viewport_scroll(x, y);
                    let (list, blobs) = match document.as_ref() {
                        Some(doc) => DisplayList::build_with_svg_and_canvas(
                            tree,
                            doc.dom(),
                            &mut svg_cache,
                            &svg_images,
                            &canvas_live,
                            &canvas_new,
                        ),
                        None => (DisplayList::build(tree), Vec::new()),
                    };
                    (Arc::new(list), sticky_moved, blobs)
                });
                let _ = reply.send(display);
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
// do). Hidden tabs stop producing frames (Stage 8 Rec 1):
// the shell forwards window focus as the visibility signal, the
// tick gate skips relayout while hidden (dirty flags stay set, so
// the first visible tick re-lays out), and JS timers keep running.

/// Frame-time histogram (Stage 10 Rec 7): the M2 done criteria
/// demand "scroll smoothly at 60+ FPS", but there is no vsync
/// signal in the shell — the tab thread ticks on a ~16 ms timer
/// and re-lays out only when dirty — and no frame-time
/// measurement existed anywhere, so the 60 FPS claim was
/// unprovable in either direction. This measures first: every
/// submitted frame records its tick-to-submit time (JS drive +
/// restyle/layout + observer steps + display-list collect +
/// submit), broken into phases, so a miss can be attributed to
/// the likely hot spots (a full restyle on every build, or a
/// display-list rebuild on every wheel tick) instead of guessed
/// at. Nothing here changes the tick cadence or any scheduling
/// behaviour — it only observes.
///
/// Overhead is deliberately negligible: one fixed-size ring of
/// pre-allocated samples (no `Vec`, no heap allocation per tick
/// — `record` is a timestamp diff plus one array write), and
/// percentiles sort a stack copy only when read, never per tick.
/// Real 60 FPS numbers can only be measured on Joel's box
/// against a long real page; the sandbox numbers prove the
/// instrumentation works, not the frame rate.
pub(crate) const FRAME_TIME_SAMPLES: usize = 240;

/// One frame's phase breakdown, in milliseconds. `layout` is the
/// layout-actor round trip (restyle + layout tree build);
/// `collect` is the ResizeObserver / IntersectionObserver steps
/// plus the display-list collect; `submit` is the postcard encode
/// and send. Wheel scrolls re-collect without re-laying out, so
/// their samples carry `layout_ms: 0.0` and
/// `layout_kind: NothingToDo` — a scroll frame's cost is its
/// collect, which is exactly the hot spot to watch. `layout_kind`
/// is the path the layout took (Stage 12 Rec 3): the histogram
/// counts kinds so the fallback rate is measured.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct FrameSample {
    pub total_ms: f32,
    pub layout_ms: f32,
    pub collect_ms: f32,
    pub submit_ms: f32,
    pub layout_kind: RelayoutKind,
}

/// Which phase of a [`FrameSample`] a percentile reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FramePhase {
    Total,
    Layout,
    Collect,
    Submit,
}

impl FramePhase {
    fn value(self, sample: &FrameSample) -> f32 {
        match self {
            FramePhase::Total => sample.total_ms,
            FramePhase::Layout => sample.layout_ms,
            FramePhase::Collect => sample.collect_ms,
            FramePhase::Submit => sample.submit_ms,
        }
    }

    fn name(self) -> &'static str {
        match self {
            FramePhase::Total => "total",
            FramePhase::Layout => "layout",
            FramePhase::Collect => "collect",
            FramePhase::Submit => "submit",
        }
    }
}

/// A fixed-size rolling histogram of frame times: the newest
/// [`FRAME_TIME_SAMPLES`] samples (four seconds at 60 Hz), kept
/// in a pre-allocated ring — `record` never allocates, and the
/// storage never grows past `capacity()`. Percentiles use the
/// nearest-rank rule (the same convention as `pluto_paint::perf`,
/// never averages), computed over a stack-sorted copy, so reading
/// them costs no heap allocation either.
#[derive(Debug, Clone)]
pub(crate) struct FrameTimeHistogram {
    samples: [FrameSample; FRAME_TIME_SAMPLES],
    next: usize,
    len: usize,
    total_recorded: u64,
    over_budget: u64,
    /// Frames recorded per [`RelayoutKind`] (indexed by
    /// `RelayoutKind::as_index`), cumulative since the tab
    /// started — the widening metric (Stage 12 Rec 3). The ring
    /// holds the last 240 samples; these counters keep the whole
    /// session, so the fallback rate does not forget the early
    /// frames that rolled off.
    layout_kinds: [u64; RelayoutKind::COUNT],
}

impl Default for FrameTimeHistogram {
    fn default() -> Self {
        Self {
            samples: [FrameSample::default(); FRAME_TIME_SAMPLES],
            next: 0,
            len: 0,
            total_recorded: 0,
            over_budget: 0,
            layout_kinds: [0; RelayoutKind::COUNT],
        }
    }
}

impl FrameTimeHistogram {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record one frame. O(1), no allocation: one ring write.
    /// Non-finite or negative totals are ignored, never counted.
    pub(crate) fn record(&mut self, sample: FrameSample) {
        if !sample.total_ms.is_finite() || sample.total_ms < 0.0 {
            return;
        }
        self.samples[self.next] = sample;
        self.next = (self.next + 1) % FRAME_TIME_SAMPLES;
        self.len = (self.len + 1).min(FRAME_TIME_SAMPLES);
        self.total_recorded += 1;
        if sample.total_ms > perf::FRAME_BUDGET_MS {
            self.over_budget += 1;
        }
        self.layout_kinds[sample.layout_kind.as_index()] += 1;
    }

    /// Frames recorded with this layout kind, since the tab
    /// started (Stage 12 Rec 3).
    pub(crate) fn layout_kind_count(&self, kind: RelayoutKind) -> u64 {
        self.layout_kinds[kind.as_index()]
    }

    /// Share of layout-taking frames that fell back to a full
    /// rebuild: `FullLayout / (SubtreeLayout + FullLayout)`.
    /// `ColdBuild` frames are excluded — a fresh tree is not a
    /// fallback — and `None` means no layout-taking frame has
    /// been recorded yet. The number the Rec 4 widening is judged
    /// by: it should fall as the safe-root predicate widens.
    pub(crate) fn fallback_rate(&self) -> Option<f32> {
        let full = self.layout_kind_count(RelayoutKind::FullLayout);
        let subtree = self.layout_kind_count(RelayoutKind::SubtreeLayout);
        if full + subtree == 0 {
            None
        } else {
            Some(full as f32 / (full + subtree) as f32)
        }
    }

    /// Samples currently held (at most `capacity()`).
    #[allow(dead_code)]
    pub(crate) fn count(&self) -> usize {
        self.len
    }

    /// The fixed ring capacity — the storage never grows past this.
    #[allow(dead_code)]
    pub(crate) fn capacity(&self) -> usize {
        FRAME_TIME_SAMPLES
    }

    /// Frames recorded since the tab started (including ones that
    /// have rolled off the ring).
    pub(crate) fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Frames in the ring that missed the 16.67 ms budget.
    #[allow(dead_code)]
    pub(crate) fn over_budget(&self) -> u64 {
        self.over_budget
    }

    /// Share of ring samples over budget, or `None` when empty.
    pub(crate) fn over_budget_ratio(&self) -> Option<f32> {
        if self.len == 0 {
            None
        } else {
            Some(self.over_budget as f32 / self.len as f32)
        }
    }

    /// Nearest-rank percentile (`p` in 0..=100) of one phase, or
    /// `None` when no samples have been recorded yet.
    pub(crate) fn percentile(&self, phase: FramePhase, p: f32) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        let mut values = [0.0f32; FRAME_TIME_SAMPLES];
        for (i, sample) in self.samples.iter().take(self.len).enumerate() {
            values[i] = phase.value(sample);
        }
        values[..self.len].sort_by(f32::total_cmp);
        let rank = (p / 100.0 * self.len as f32).ceil() as usize;
        values
            .get(rank.saturating_sub(1).min(self.len - 1))
            .copied()
    }

    pub(crate) fn p50(&self, phase: FramePhase) -> Option<f32> {
        self.percentile(phase, 50.0)
    }

    pub(crate) fn p95(&self, phase: FramePhase) -> Option<f32> {
        self.percentile(phase, 95.0)
    }

    pub(crate) fn max(&self, phase: FramePhase) -> Option<f32> {
        self.samples
            .iter()
            .take(self.len)
            .map(|s| phase.value(s))
            .fold(None, |max: Option<f32>, v| {
                Some(match max {
                    Some(m) if m >= v => m,
                    _ => v,
                })
            })
    }

    /// One stderr line with the total + per-phase percentiles and
    /// the over-budget share. `Option` reads print as `-` (no
    /// samples yet). Logged periodically by the tab tick, never
    /// per tick.
    pub(crate) fn summary(&self) -> String {
        let fmt = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
        let pct = |phase: FramePhase| {
            format!(
                "{} p50={} p95={} max={}",
                phase.name(),
                fmt(self.p50(phase)),
                fmt(self.p95(phase)),
                fmt(self.max(phase))
            )
        };
        // Stage 12 Rec 3: the layout-kind counts and the fallback
        // rate ride on the same line, so the widening verdict is
        // visible in the periodic log without a second channel.
        let fallback = self
            .fallback_rate()
            .map_or("-".to_string(), |r| format!("{:.1}%", r * 100.0));
        format!(
            "frames n={} total_recorded={} over_budget={} ({:.1}%) | {} | {} | {} | {} | layout cold={} subtree={} full={} paint={} pos={} comp={} nothing={} fallback={}",
            self.len,
            self.total_recorded,
            self.over_budget,
            self.over_budget_ratio().unwrap_or(0.0) * 100.0,
            pct(FramePhase::Total),
            pct(FramePhase::Layout),
            pct(FramePhase::Collect),
            pct(FramePhase::Submit),
            self.layout_kind_count(RelayoutKind::ColdBuild),
            self.layout_kind_count(RelayoutKind::SubtreeLayout),
            self.layout_kind_count(RelayoutKind::FullLayout),
            self.layout_kind_count(RelayoutKind::PaintOnly),
            self.layout_kind_count(RelayoutKind::PositionOnly),
            self.layout_kind_count(RelayoutKind::CompositingOnly),
            self.layout_kind_count(RelayoutKind::NothingToDo),
            fallback,
        )
    }
}

/// How often the tab thread ticks (the M4 rendering cadence:
/// ~60 Hz; see the module docs — no shell vsync exists to drive it).
const TAB_TICK: Duration = Duration::from_millis(16);

/// Heartbeat interval (Stage 13 Rec 12b): the tab thread sends
/// one `RendererToBrowser::Heartbeat` this often while alive, so
/// the bridge watchdog can tell a hung renderer from a slow one.
/// Five seconds against the 30 s watchdog timeout leaves room
/// for six missed beats before a kill.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Watchdog timeout (Stage 13 Rec 12b): silence longer than this
/// — no heartbeat, no event of any kind — means the renderer is
/// hung; the bridge kills it and reports `RendererCrashed`.
/// `PLUTO_WATCHDOG_SECS` overrides it for tests; `0` disables it.
const WATCHDOG_SECS_DEFAULT: u64 = 30;
const WATCHDOG_ENV: &str = "PLUTO_WATCHDOG_SECS";

/// Debug-only escape hatch (Stage 13 Rec 12d): the in-process
/// fallback is OFF by default — a spawn failure must surface as
/// a crashed tab, never silently run the page in-process.
const ALLOW_INPROCESS_FALLBACK_ENV: &str = "PLUTO_ALLOW_INPROCESS_FALLBACK";

fn inprocess_fallback_allowed() -> bool {
    std::env::var_os(ALLOW_INPROCESS_FALLBACK_ENV).is_some_and(|v| v == "1")
}

fn watchdog_timeout() -> Option<Duration> {
    match std::env::var(WATCHDOG_ENV) {
        Ok(value) => match value.parse::<u64>() {
            Ok(0) => None,
            Ok(secs) => Some(Duration::from_secs(secs)),
            Err(_) => Some(Duration::from_secs(WATCHDOG_SECS_DEFAULT)),
        },
        Err(_) => Some(Duration::from_secs(WATCHDOG_SECS_DEFAULT)),
    }
}

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
    /// The shell-assigned tab id (Stage 11 Rec 4): echoed in
    /// every fetch request and child-process arg, never
    /// hardcoded 0.
    tab_id: u64,
    layout_tx: std::sync::mpsc::Sender<LayoutJob>,
    loader: ResourceLoader,
    /// Where this tab's fetches go (Stage 13 Rec 11): direct
    /// loader in-process, browser-side broker out-of-process.
    fetch_source: crate::broker::FetchSource,
    /// The broker handle when `fetch_source` is `Broker`
    /// (`None` in-process) — routes inbound `ResourceEvent`s.
    broker_handle: Option<Arc<crate::broker::BrokerHandle>>,
    /// The worker's image pipeline (Stage 11 Rec 1/7/11): decoded
    /// pixels, per-URL state, animation timers, favicon
    /// bookkeeping.
    images: ImageStore,
    /// The last submitted display list, cached so an animation
    /// frame advance can re-submit it with fresh pixels and no
    /// layout at all (Stage 11 Rec 11).
    last_display: Option<Arc<DisplayList>>,
    /// The last title sent to the shell (Stage 11 Rec 8):
    /// `TitleChanged` fires at most once per submit, only when
    /// the title actually changed.
    last_title: String,
    /// The favicon URL for the current page (Rec 7), if any.
    favicon_url: Option<String>,
    /// The favicon's resource id in the worker cache, once
    /// decoded (sent to the shell via `FaviconChanged`).
    favicon_resource: Option<ResourceId>,
    /// Image requests reported by the load task (its sender is
    /// cloned into each `LoadSinks`).
    image_request_tx: tokio::sync::mpsc::UnboundedSender<ImageRequest>,
    /// Image requests, received by the main loop (see `exec_rx`).
    image_request_rx: Option<tokio::sync::mpsc::UnboundedReceiver<ImageRequest>>,
    /// Image fetch outcomes, sent back by fetch tasks.
    image_outcome_tx: tokio::sync::mpsc::UnboundedSender<ImageOutcome>,
    /// Image outcomes, received by the main loop (see `exec_rx`).
    image_outcome_rx: Option<tokio::sync::mpsc::UnboundedReceiver<ImageOutcome>>,
    /// Media fetch outcomes, sent back by fetch tasks.
    media_outcome_tx: tokio::sync::mpsc::UnboundedSender<MediaOutcome>,
    /// Media outcomes, received by the main loop (see `exec_rx`).
    media_outcome_rx: Option<tokio::sync::mpsc::UnboundedReceiver<MediaOutcome>>,
    /// True when a media intrinsic size / liveness changed since
    /// the last layout pass (Stage 16 Phase 2, Rec 5): the next
    /// tick rebuilds the image map with fresh `videos` entries.
    /// Set only on change (metadata arrival, first frame) — NOT
    /// on every presented frame, so a playing video does not
    /// re-layout at 30 fps.
    media_dirty: bool,
    /// Last announced media intrinsic sizes, for the change
    /// check above: (width, height, has_frame) per element.
    last_media_sizes: HashMap<NodeId, (u32, u32, bool)>,
    /// Document-load progress reports from the load task.
    progress_tx: tokio::sync::mpsc::UnboundedSender<(u64, Option<u64>)>,
    /// Load progress, received by the main loop (see `exec_rx`).
    progress_rx: Option<tokio::sync::mpsc::UnboundedReceiver<(u64, Option<u64>)>>,
    /// The decoded-image map of the last layout pass (Stage 11
    /// Rec 11): the animation visibility check walks it for
    /// animated nodes.
    last_image_map: ImageMap,
    /// Hash of the last announced canvas bitmap per canvas
    /// element (Stage 15, M7 Phase 3, Rec 8):
    /// `relayout_and_submit` only re-announces a canvas whose
    /// bitmap changed since its last announcement — an unchanged
    /// canvas is not re-sent at 60 fps. Cleared on navigation (a
    /// new document's keys are a fresh arena).
    last_canvas_announced: HashMap<NodeId, u64>,
    /// Animated resource ids currently on-screen (Stage 11 Rec
    /// 11): refreshed by a `Geometry` job every ~30 ticks; the
    /// advance step only runs for ids in this set.
    animation_visible: HashSet<ResourceId>,
    /// Pending animation-visibility `Geometry` replies: (cache
    /// id, receiver). Polled non-blockingly every tick — the
    /// `Geometry` job replies over a std channel, which cannot
    /// join the `select!`.
    animation_geometry_pending:
        Vec<(ResourceId, std::sync::mpsc::Receiver<Option<pluto_js::Geometry>>)>,
    /// Tick of the last animation-visibility check (every ~30
    /// ticks, ~0.5 s).
    last_animation_check_tick: u64,
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
    /// The storage + permissions backends (Stage 9 Rec 4/11),
    /// created once per tab and shared with the `TabJs`;
    /// persisted when the tab thread ends.
    tab_backends: Option<crate::js_host::TabBackends>,
    /// The current document (re-synced from the host's live `Dom`
    /// on every re-layout, so JS mutations reach the actor).
    document: Option<Document>,
    has_document: bool,
    element_state: ElementState,
    /// Committed URLs this session (Stage 10 Rec 3): every URL
    /// the worker commits — the same `final_url` it reports in
    /// `DidCommit`, which is exactly what the shell pushes into
    /// its session `history` — so this set and the shell's history
    /// hold the same URLs with no IPC change. Stage 10 Rec 8:
    /// seeded from the persisted `history.json` in `TabState::new`,
    /// so `:visited` survives a restart. Keys are
    /// fragment-stripped (`visited_key`): browsers ignore the
    /// fragment for `:visited`. `refresh_visited` projects this
    /// set onto the current document's link ids before each
    /// layout pass, into `element_state.visited`.
    visited_urls: HashSet<String>,
    scroll_offsets: HashMap<NodeId, (f32, f32)>,
    /// The currently hovered element chain — the hit element plus
    /// its ancestors (browsers hover ancestors too: hovering text
    /// inside a button highlights the button). Drives `:hover`
    /// restyle (Stage 8 Rec 9).
    hovered: Vec<NodeId>,
    /// The element under an active press (for `:active` restyle).
    active_element: Option<NodeId>,
    /// The element a pointer press started on: a left release over
    /// the same target synthesises the click (Stage 8 Rec 9).
    pointer_down_target: Option<NodeId>,
    /// The element with keyboard focus (for `:focus` restyle and
    /// Enter/Space activation).
    focused: Option<NodeId>,
    /// Whether the window is focused (Stage 8 Rec 9): the shell
    /// forwards focus alongside visibility; keyboard input is only
    /// acted on while focused.
    window_focused: bool,
    /// The scroll chain the current wheel gesture is applying to,
    /// picked on the gesture's first wheel (Stage 8 Rec 9).
    wheel_chain: Vec<ScrollChainEntry>,
    /// Ticks since the last wheel event: the gesture expires after
    /// ~200 ms (~12 ticks) with no wheel, and the next wheel
    /// re-picks its chain.
    wheel_idle_ticks: u32,
    /// Frame ticks elapsed (gesture timeout bookkeeping).
    tick_count: u64,
    /// Caret blink clock (Stage 13 Rec 4): reset on every focus
    /// change and edit, so the caret is visible right after
    /// typing; the visible half is ~530 ms.
    caret_blink_epoch: std::time::Instant,
    /// Blink phase at the last submit: the tick re-submits the
    /// cached display list (no layout) when it flips.
    caret_blink_on: bool,
    /// Last heartbeat sent to the bridge (Stage 13 Rec 12b).
    last_heartbeat: std::time::Instant,
    viewport: (f32, f32),
    /// The viewport scroll offset, mirrored from the shell (Stage 8
    /// Rec 10): every viewport scroll flows through
    /// `emit_scroll_viewport`, which updates this with the same
    /// `[0, page max]` clamp the shell's `compositor.scroll_by`
    /// applies, and `start_navigation` resets it — so the
    /// IntersectionObserver step can compute viewport-root
    /// intersections from the retained tree without a protocol
    /// change. The worker works entirely in page coordinates; the
    /// viewport root is `[viewport_scroll, viewport]` in page
    /// coordinates.
    viewport_scroll: (f32, f32),
    /// When this tab started (the `IntersectionObserver` entry
    /// `time` clock: ms since the tab started).
    tab_started: std::time::Instant,
    /// Display-list epoch (advances per submitted list).
    epoch: u64,
    /// Navigation epoch (cancels the old navigation's fetch +
    /// scripts + timers + promises).
    nav_epoch: u64,
    nav_cancel: Option<CancellationToken>,
    /// Whether the in-flight navigation replaces the current
    /// history entry (`location.replace()` — Stage 16 Phase 3,
    /// Rec 8): `DidCommit` carries it to the shell, which replaces
    /// instead of pushing. Set by `start_navigation`.
    nav_replace: bool,
    /// Pending `popstate` for a Back/Forward traversal (Stage 16
    /// Phase 3, Rec 8/9): `Some(state_json)` while a traversal
    /// navigation is in flight — the state is restored into the
    /// fresh realm at navigation start, and `popstate` fires after
    /// the commit. `None` for ordinary navigations.
    pending_popstate: Option<Option<String>>,
    /// History API requests staged by the sync host drain (Stage 16
    /// Phase 3, Rec 8): the async drain forwards them to the shell
    /// as `HistoryPush` / `HistoryReplace` / `HistoryGo`. Staged,
    /// not sent, because the sync drain has no event sender.
    staged_history_push: Vec<(String, Option<String>)>,
    staged_history_replace: Vec<(String, Option<String>)>,
    staged_history_traversal: Vec<i32>,
    /// A same-document fragment navigation staged by the sync host
    /// drain (Stage 16 Phase 3, Rec 9): `(old_url, new_url)`. The
    /// async drain updates the URL in place, scrolls to the anchor,
    /// and dispatches `hashchange` — no fetch, no new entry.
    pending_same_document: Option<(String, String)>,
    /// The tab's event sender (to the shell): the `WebSocket`
    /// backend forwards `WebSocketOpen` / `Send` / `Close` through
    /// it (Stage 16 Phase 3, Rec 11).
    events: tokio::sync::mpsc::Sender<RendererToBrowser>,
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
    /// Frame-time histogram (Stage 10 Rec 7): every submitted
    /// frame records its tick-to-submit time here (see
    /// [`FrameTimeHistogram`]). Read by the tab tick's periodic
    /// log line and by tests through [`TabState::frame_times`].
    /// The shell has no direct handle to the tab thread — and the
    /// renderer runs out-of-process — so the periodic stderr
    /// line is the shell-visible surface, not a struct field.
    frame_times: FrameTimeHistogram,
    /// `total_recorded` at the last periodic frame-time log, so
    /// the tick logs only when new frames landed (no spam on
    /// idle ticks).
    last_frame_log_total: u64,
    /// The CSS transition / keyframe / WAAPI engine (M6 Phase 3,
    /// Rec 11): one per tab, sampled from the monotonic clock on
    /// the tab tick — never by counting ticks.
    animation_engine: AnimationEngine,
    /// Currently sampled animated values per element, keyed by
    /// property: the transition `from` source, and the values the
    /// actor applies as overrides on the next layout.
    current_animated: HashMap<NodeId, HashMap<String, AnimatedValue>>,
    /// Underlying computed values merged from actor outcomes
    /// (`changed_styles`) plus in-flight samples: the
    /// `getComputedStyle` cache the host reads for JS.
    computed_style_cache: HashMap<NodeId, HashMap<String, String>>,
    /// Downloaded `@font-face` bytes waiting for the next layout
    /// (M6 Phase 3, Rec 12): the actor loads them into the
    /// retained FontSystem, then the layout rebuilds with the new
    /// faces (fallback until they arrive — `font-display: swap`
    /// behaviour falls out of the rebuild).
    pending_fonts: Vec<Vec<u8>>,
    /// `@font-face` URLs already fetched or in flight for this
    /// document (dedup — one fetch per URL per navigation).
    font_fetch_urls: HashSet<String>,
    /// Font fetch outcomes, sent back by fetch tasks.
    font_outcome_tx: tokio::sync::mpsc::UnboundedSender<FontOutcome>,
    /// Font outcomes, received by the main loop (see `exec_rx`).
    font_outcome_rx: Option<tokio::sync::mpsc::UnboundedReceiver<FontOutcome>>,
    /// The shell's theme for this tab (M6 Phase 3, Rec 12):
    /// forwarded to the actor on every layout; a change forces a
    /// full rebuild (restyle, never a refetch).
    theme: Theme,
}

/// One `@font-face` download outcome (M6 Phase 3, Rec 12): the
/// URL it was fetched for (the dedup key) plus the bytes, or the
/// failure — a failed face is never fatal, the fallback font stays.
enum FontOutcome {
    Loaded { url: String, epoch: u64, bytes: Vec<u8> },
    Failed { url: String, epoch: u64 },
}

impl TabState {
    fn new(
        layout_tx: std::sync::mpsc::Sender<LayoutJob>,
        fetch_handle: tokio::runtime::Handle,
        tab_id: u64,
        shared: &SharedLoader,
        broker: Option<Arc<crate::broker::BrokerHandle>>,
        events: tokio::sync::mpsc::Sender<RendererToBrowser>,
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
        let (image_request_tx, image_request_rx) = tokio::sync::mpsc::unbounded_channel();
        let (image_outcome_tx, image_outcome_rx) = tokio::sync::mpsc::unbounded_channel();
        let (media_outcome_tx, media_outcome_rx) = tokio::sync::mpsc::unbounded_channel();
        let (progress_tx, progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (font_outcome_tx, font_outcome_rx) = tokio::sync::mpsc::unbounded_channel();
        // Stage 9 Rec 5: the tab's loader starts with the
        // persisted HSTS store + HTTPS-only preference/exceptions
        // from the browser profile (defaults when none exists).
        // Stage 11 Rec 5: the loader is built from the shared
        // handle (HSTS store, blocker, cookie jar, HTTP cache
        // shared; CSP, referrer, HTTPS-only, preflight cache,
        // caps/dedup stay per-tab) instead of a fresh private one.
        let loader = ResourceLoader::from_shared(shared);
        // Stage 13 Rec 11: broker mode fetches through the
        // browser-side broker, so this process owns no profile
        // state — no HSTS / cookie load, no temp sweep, no
        // persistence writes (the broker owns all of it).
        let fetch_source = match &broker {
            Some(handle) => crate::broker::FetchSource::Broker(Arc::clone(handle)),
            None => crate::broker::FetchSource::Direct(Arc::new(loader.clone())),
        };
        if broker.is_none() {
            // Stage 13 Rec 10c: `document.cookie` reads/writes
            // the shared jar through the host.
            host.lock()
                .unwrap()
                .set_cookie_jar(loader.cookie_jar());
            crate::security::load_into(&loader);
            // Stage 13 Rec 9: restore the persisted cookie jar
            // into the shared loader (session cookies were
            // stripped before the last save, so nothing stale
            // comes back), and sweep stale atomic-write temp
            // files from the profile dir.
            crate::cookies::load_into(&loader);
            if let Some(dir) = crate::security::profile_dir() {
                crate::atomic::sweep_stale_temps(&dir);
            }
        }
        // Stage 10 Rec 8: seed `:visited` from the persisted
        // history, so links to pages visited in a previous
        // session already match on the first layout pass — the
        // worker's set and the shell's history hold the same
        // URLs (fragment-stripped keys, as on commit).
        let visited_urls: HashSet<String> = crate::history::load_history()
            .iter()
            .map(|url| visited_key(url).to_string())
            .collect();
        Self {
            tab_id,
            layout_tx,
            loader,
            fetch_source,
            broker_handle: broker,
            fetch_handle,
            images: ImageStore::new(),
            last_display: None,
            last_title: String::new(),
            favicon_url: None,
            favicon_resource: None,
            image_request_tx,
            image_request_rx: Some(image_request_rx),
            image_outcome_tx,
            image_outcome_rx: Some(image_outcome_rx),
            media_outcome_tx,
            media_outcome_rx: Some(media_outcome_rx),
            media_dirty: false,
            last_media_sizes: HashMap::new(),
            progress_tx,
            progress_rx: Some(progress_rx),
            last_image_map: ImageMap::new(),
            last_canvas_announced: HashMap::new(),
            animation_visible: HashSet::new(),
            animation_geometry_pending: Vec::new(),
            last_animation_check_tick: 0,
            tab_js: None,
            host,
            fetch_backend: None,
            tab_backends: None,
            document: None,
            has_document: false,
            element_state: ElementState::default(),
            visited_urls,
            scroll_offsets: HashMap::new(),
            hovered: Vec::new(),
            active_element: None,
            pointer_down_target: None,
            focused: None,
            window_focused: true,
            wheel_chain: Vec::new(),
            wheel_idle_ticks: 0,
            tick_count: 0,
            caret_blink_epoch: std::time::Instant::now(),
            caret_blink_on: true,
            last_heartbeat: std::time::Instant::now(),
            viewport: page::DEFAULT_VIEWPORT,
            viewport_scroll: (0.0, 0.0),
            tab_started: std::time::Instant::now(),
            epoch: 0,
            nav_epoch: 0,
            nav_cancel: None,
            nav_replace: false,
            pending_popstate: None,
            staged_history_push: Vec::new(),
            staged_history_replace: Vec::new(),
            staged_history_traversal: Vec::new(),
            pending_same_document: None,
            events,
            load_handle: None,
            exec_tx,
            exec_rx: Some(exec_rx),
            load_outcome_tx,
            load_outcome_rx: Some(load_outcome_rx),
            render_requested,
            frame_times: FrameTimeHistogram::new(),
            last_frame_log_total: 0,
            animation_engine: AnimationEngine::new(),
            current_animated: HashMap::new(),
            computed_style_cache: HashMap::new(),
            pending_fonts: Vec::new(),
            font_fetch_urls: HashSet::new(),
            font_outcome_tx,
            font_outcome_rx: Some(font_outcome_rx),
            theme: Theme::Light,
        }
    }

    /// The frame-time histogram for this tab (Stage 10 Rec 7):
    /// p50/p95 per phase plus the sample count, for tests and
    /// the tick's periodic log line.
    // The tick logs via `summary()` on the field directly.
    #[allow(dead_code)]
    pub(crate) fn frame_times(&self) -> &FrameTimeHistogram {
        &self.frame_times
    }

    /// The live `TabJs`, creating it on the first navigation.
    fn tab_js(&mut self) -> &mut TabJs {
        if self.tab_js.is_none() {
            let mut js = crate::js_host::new_tab_js().expect("tab JS realm creates");
            js.set_dom_host(Box::new(crate::js_host::SharedHost(Rc::clone(&self.host))));
            let backend = std::sync::Arc::new(crate::js_host::PlutoFetchBackend::new(
                self.fetch_source.clone(),
                self.fetch_handle.clone(),
                "",
                self.tab_id,
            ));
            js.set_fetch_backend(
                std::sync::Arc::clone(&backend) as std::sync::Arc<dyn pluto_js::FetchBackend>,
            );
            // Stage 16 Phase 3 (Rec 11): the `WebSocket` backend —
            // the realm's sockets open through the shell, never
            // from this process.
            js.set_websocket_backend(
                std::sync::Arc::new(crate::js_host::PlutoWebSocketBackend::new(
                    self.events.clone(),
                    self.fetch_handle.clone(),
                ))
                    as std::sync::Arc<dyn pluto_js::bindings::websocket::WebSocketBackend>,
            );
            // Stage 9 Rec 4/11: attach the profile-backed
            // storage + permissions backends (created once per
            // tab; `TabJs::navigate` re-attaches them to the
            // fresh realm on every navigation).
            let backends = crate::js_host::new_tab_backends();
            // Tag this tab's mutations for the cross-tab `storage`
            // event fan-out (Stage 13 Rec 10d).
            backends.storage.set_tab_id(self.tab_id);
            js.set_storage_backend(
                std::sync::Arc::clone(&backends.storage)
                    as std::sync::Arc<dyn pluto_js::bindings::storage::StorageBackend>,
            );
            js.set_permissions_backend(
                std::sync::Arc::clone(&backends.permissions)
                    as std::sync::Arc<dyn pluto_js::bindings::permissions::PermissionsBackend>,
            );
            self.tab_backends = Some(backends);
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
    tab_id: u64,
    shared: SharedLoader,
    broker: bool,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tab thread runtime builds");
    runtime.block_on(async {
        tokio::task::LocalSet::new()
            .run_until(tab_main(commands, events, layout_tx, tab_id, shared, broker))
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
    tab_id: u64,
    shared: SharedLoader,
    broker: bool,
) {
    // Stage 13 Rec 11: the out-of-process child fetches through
    // the browser-side broker (no sockets in this process); the
    // in-process worker keeps its direct loader.
    let broker_handle = if broker {
        Some(crate::broker::BrokerHandle::new(events.clone()))
    } else {
        None
    };
    let mut state = TabState::new(
        layout_tx,
        tokio::runtime::Handle::current(),
        tab_id,
        &shared,
        broker_handle,
        events.clone(),
    );
    // Stage 9 Rec 12: refresh the tracker lists in the
    // background (EasyList + EasyPrivacy, fetched by default).
    // The refresh works on its own copy — loaded from the
    // profile dir — and swaps it in at the end, so page loads
    // never wait on it and never see an empty blocker
    // mid-refresh; it is freshness-gated (~4 h), so only the
    // first tab in the window actually fetches. Skipped in
    // broker mode (Stage 13 Rec 11): the broker owns the
    // blocker, and this process must not write the profile.
    if state.broker_handle.is_none() {
        let blocker = state.loader.blocker();
        tokio::task::spawn(async move {
            let Some(dir) = pluto_net::blocker::profile_list_dir() else {
                return;
            };
            let mut fresh = pluto_net::blocker::Blocker::load_from_dir(&dir);
            fresh.refresh(&dir).await;
            *blocker.lock().unwrap() = fresh;
        });
    }
    // The receivers are taken out of the state so the `select!`
    // arms can hold them without borrowing `state` — the arm bodies
    // borrow it mutably, and `select!` builds all arm futures up
    // front.
    let mut exec_rx = state.exec_rx.take().expect("exec receiver present");
    let mut load_outcome_rx = state
        .load_outcome_rx
        .take()
        .expect("load outcome receiver present");
    let mut image_request_rx = state
        .image_request_rx
        .take()
        .expect("image request receiver present");
    let mut image_outcome_rx = state
        .image_outcome_rx
        .take()
        .expect("image outcome receiver present");
    let mut media_outcome_rx = state
        .media_outcome_rx
        .take()
        .expect("media outcome receiver present");
    let mut progress_rx = state
        .progress_rx
        .take()
        .expect("progress receiver present");
    let mut font_outcome_rx = state
        .font_outcome_rx
        .take()
        .expect("font outcome receiver present");
    let mut tick = tokio::time::interval(TAB_TICK);
    // `finish_page` blocks this loop for up to a second driving
    // the JS loop after a parse; a Burst interval would then fire
    // dozens of catch-up ticks before the next command is read.
    // Delay skips the backlog instead.
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                // Stage 10 Rec 7: the frame clock starts here —
                // the tick-to-submit time is tick start (before
                // the JS drive) to the submit that the dirty
                // relayout performs, so the histogram measures
                // the whole frame, not just the layout half.
                let tick_started = std::time::Instant::now();
                state.tick_count += 1;
                // Stage 13 Rec 12b: heartbeat for the bridge
                // watchdog — best-effort (a full event channel
                // means the bridge is gone; the tick continues).
                if state.last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                    state.last_heartbeat = std::time::Instant::now();
                    let _ = events.try_send(RendererToBrowser::Heartbeat);
                }
                // Wheel gesture expiry (Stage 8 Rec 9): no wheel
                // for ~200 ms (~12 ticks) ends the gesture — the
                // chain is forgotten, so the next wheel re-picks
                // its containers under the cursor.
                if !state.wheel_chain.is_empty() {
                    state.wheel_idle_ticks += 1;
                    if state.wheel_idle_ticks > 12 {
                        state.wheel_chain.clear();
                        state.wheel_idle_ticks = 0;
                    }
                }
                drive_js(&mut state);
                // Stage 12 Rec 7: a script may have asked for a
                // navigation (`location.assign` / `reload`) or an
                // image load (`Image().src` / `<img>` src) — the
                // host only recorded the request, so the tab
                // thread acts on it here, once per tick.
                drain_host_requests(&mut state);
                drain_host_requests_async(&mut state, &events).await;
                // Stage 11 Rec 11: advance due animations (the
                // visible set is refreshed by a `Geometry` job
                // every ~30 ticks — see below) and re-submit the
                // cached display list with the fresh frame pixels:
                // no layout is involved, and the rate is capped by
                // the decoded per-frame delays plus the tick.
                // Replies to pending visibility checks are polled
                // first, so a just-measured node animates (or
                // pauses) on this tick, not the next check.
                poll_animation_geometry(&mut state);
                if state.tick_count - state.last_animation_check_tick >= 30 {
                    state.last_animation_check_tick = state.tick_count;
                    check_animation_visibility(&mut state);
                }
                let now = std::time::Instant::now();
                let advanced = state
                    .images
                    .advance_animations(now, &state.animation_visible);
                if !advanced.is_empty()
                    && let Some(display) = state.last_display.clone()
                {
                    // Fresh pixels for the advanced ids travel
                    // in their own `ResourceReady` frames (same
                    // ids, new pixels — the shell overwrites its
                    // table entries), then the cached list
                    // re-submits: the ids it names are unchanged.
                    let mut sent = true;
                    for id in advanced {
                        if let Some(resource) = state.images.cache.get(id).cloned() {
                            let fits = match &resource {
                                Resource::Image { rgba, .. } => {
                                    ImageStore::resource_fits_frame(rgba.len())
                                }
                            };
                            if !fits {
                                continue;
                            }
                            sent = emit_tab(
                                &events,
                                RendererToBrowser::ResourceReady {
                                    epoch: state.nav_epoch,
                                    resource_id: id,
                                    resource,
                                },
                            )
                            .await;
                            if !sent {
                                break;
                            }
                        }
                    }
                    if sent {
                        submit(&mut state, &events, display).await;
                    }
                }
                // M6 Phase 3 (Rec 11): advance the CSS/WAAPI
                // engine from the monotonic clock (never by
                // counting ticks — 16 ms != 16.67 ms). Changed
                // values mark the frame dirty, so the layout below
                // applies them as overrides; finished events
                // dispatch into the realm.
                advance_animations(&mut state, tick_started);
                // Stage 13 Rec 4: caret blink — when the ~530 ms
                // phase flips with a focused text control, re-submit
                // the cached (caret-free) display list with fresh
                // caret items: no layout, just a new submit.
                if state.focused.is_some() {
                    let phase = caret_blink_on(&state);
                    if phase != state.caret_blink_on {
                        state.caret_blink_on = phase;
                        if let Some(display) = state.last_display.clone() {
                            let display = with_caret(&mut state, display);
                            submit(&mut state, &events, display).await;
                        }
                    }
                }
                // Stage 16 (M8 Phase 2, Rec 4/6): advance the
                // media sessions — decode ahead, present due
                // frames on the audio clock, fire drained events
                // — and send presented frames to the shell. A
                // size/liveness change marks the media map dirty
                // (layout re-runs below); a plain frame never does
                // (the shell re-uploads the texture, no layout).
                drive_media(&mut state, &events).await;
                // Dirty-bit flow (Rec 7): a JS mutation sets the
                // host's render flag / dirty bits and returns
                // immediately (no layout); the frame tick re-lays
                // out and submits — with the ResizeObserver step
                // (Rec 8) running inside `relayout_and_submit`,
                // after layout and before the display list is
                // built. Stage 8 Rec 1: a hidden tab
                // skips the relayout — the dirty flags stay set,
                // so the first visible tick re-lays out and
                // submits. A tab with no JS realm yet counts as
                // visible (the first layout must still happen).
                let visible = state
                    .tab_js
                    .as_ref()
                    .map(|js| js.is_visible())
                    .unwrap_or(true);
                let dirty = state.render_requested.get()
                    || state.host.lock().unwrap().dom_is_dirty()
                    // Stage 11 Rec 1/2: a decoded image is dirty
                    // too — the next layout pass gives it its
                    // intrinsic size (full rebuild, coalesced to
                    // one per tick by `layout_and_submit`).
                    || state.images.images_dirty
                    // Stage 16 (M8 Phase 2, Rec 5): a media
                    // size/liveness change re-lays out the same
                    // way (a playing video's frames do not set
                    // this — see `drive_media`).
                    || state.media_dirty;
                if visible && dirty {
                    state.render_requested.set(false);
                    relayout_and_submit(&mut state, &events, tick_started).await;
                } else if !visible && dirty {
                    // Perf (Stage 8 Rec 5): a hidden tab that
                    // would have re-laid out counts the skip.
                    if let Ok(mut counters) = perf::global().lock() {
                        counters.frames_hidden_skipped += 1;
                    }
                }
                // Stage 10 Rec 7: one frame-time summary line
                // every ~10 s (600 ticks at 16 ms), and only
                // when new frames landed since the last line —
                // never per tick, never on idle ticks. The
                // line is the shell-visible surface for the
                // histogram (stderr, like the perf summary).
                if state.tick_count.is_multiple_of(600)
                    && state.frame_times.total_recorded() > state.last_frame_log_total
                {
                    state.last_frame_log_total = state.frame_times.total_recorded();
                    eprintln!("[pluto frame-time] {}", state.frame_times.summary());
                }
            }
            cmd = commands.recv() => {
                let Some(cmd) = cmd else { break };
                if !handle_command(&mut state, &events, cmd, &mut commands).await {
                    break;
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
            req = image_request_rx.recv() => {
                let Some(req) = req else { break };
                handle_image_request(&mut state, req);
            }
            outcome = image_outcome_rx.recv() => {
                let Some(outcome) = outcome else { break };
                handle_image_outcome(&mut state, &events, outcome).await;
            }
            outcome = media_outcome_rx.recv() => {
                let Some(outcome) = outcome else { break };
                handle_media_outcome(&mut state, outcome);
            }
            outcome = font_outcome_rx.recv() => {
                let Some(outcome) = outcome else { break };
                handle_font_outcome(&mut state, outcome);
            }
            progress = progress_rx.recv() => {
                let Some((received, total)) = progress else { break };
                // Stage 11 Rec 9: document-load progress to the
                // shell's progress bar (indeterminate when the
                // server sent no Content-Length).
                if !emit_tab(
                    &events,
                    RendererToBrowser::LoadProgress { received, total },
                )
                .await
                {
                    break;
                }
            }
        }
    }
    // Stage 9 Rec 4/11: the tab thread is ending (tab close /
    // shutdown) — persist the tab's storage + permissions
    // backends to the profile.
    if let Some(backends) = state.tab_backends.take() {
        crate::js_host::save_tab_backends(&backends);
    }
    // Stage 13 Rec 11: in broker mode the broker owns the jar.
    if state.broker_handle.is_none() {
        crate::cookies::save_from(&state.loader);
    }
}

/// Handle one shell command. Returns false when the loop should
/// stop (`Shutdown`). Pointer moves are coalesced before handling:
/// only the latest position per frame reaches the worker (Stage 8
/// Rec 9) — a drag can queue dozens of moves between ticks, and
/// only the last one matters for hover. Non-move commands found
/// while draining are handled in order, right after.
async fn handle_command(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    cmd: BrowserToRenderer,
    commands: &mut tokio::sync::mpsc::Receiver<BrowserToRenderer>,
) -> bool {
    let cmd = match cmd {
        BrowserToRenderer::InputEvent(ev @ InputEvent::Pointer {
            kind: PointerKind::Move,
            ..
        }) => {
            let mut latest = BrowserToRenderer::InputEvent(ev);
            loop {
                match commands.try_recv() {
                    Ok(other @ BrowserToRenderer::InputEvent(InputEvent::Pointer {
                        kind: PointerKind::Move,
                        ..
                    })) => latest = other,
                    Ok(other) => {
                        if !handle_single_command(state, events, other).await {
                            return false;
                        }
                    }
                    Err(_) => break,
                }
            }
            latest
        }
        other => other,
    };
    handle_single_command(state, events, cmd).await
}

/// Handle one shell command that is not a coalescible pointer
/// move. Returns false when the loop should stop (`Shutdown`).
async fn handle_single_command(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    cmd: BrowserToRenderer,
) -> bool {
    match cmd {
        BrowserToRenderer::Shutdown => return false,
        BrowserToRenderer::CommitNavigation {
            url,
            state_json,
            traversal,
            scroll,
        } => {
            // Stage 16 Phase 3, Rec 8 (finisher): a traversal
            // restores the entry's `history.state` and scroll and
            // fires `popstate` after the commit; a plain navigation
            // carries `None` / `false` / `(0.0, 0.0)`.
            start_navigation_with_body(state, url, None, false, state_json, traversal, scroll);
        }
        BrowserToRenderer::Resize { width, height } => {
            resize(state, events, width, height).await;
        }
        BrowserToRenderer::SetVisible { visible } => {
            // Stage 8 Rec 1: the shell forwards window
            // visibility. A hidden tab keeps its JS
            // timers running (drive_js still runs each
            // tick — a timer fires while hidden) but
            // produces no frames and no display lists
            // until visible again: the tick gate below
            // skips relayout. No realm is created just to
            // record this — a tab with no JS yet is visible
            // by default.
            if let Some(js) = state.tab_js.as_mut() {
                js.set_visible(visible);
            }
        }
        BrowserToRenderer::SetFocused { focused } => {
            // Stage 8 Rec 9: the shell forwards window focus
            // alongside visibility. The shell also filters keys
            // to uncaptured ones only, so this is the
            // worker-side half: keyboard input is acted on only
            // while focused.
            state.window_focused = focused;
        }
        BrowserToRenderer::SetTheme { dark } => {
            // M6 Phase 3 (Rec 12): the OS / desktop theme
            // changed — restyle against the new theme (the actor
            // force-rebuilds on a theme change; never a refetch).
            // A tab with no document yet just records it; the
            // first layout uses it.
            state.theme = if dark { Theme::Dark } else { Theme::Light };
            if state.has_document {
                state.render_requested.set(true);
            }
        }
        BrowserToRenderer::ResourceEvent(event) => {
            // Stage 13 Rec 11: the broker streams fetch events
            // back to the renderer — deliver to the waiting
            // fetch by request id. No handle (in-process) means
            // a stray event; ignore it.
            if let Some(handle) = &state.broker_handle {
                handle.deliver(event).await;
            } else {
                eprintln!("[pluto] ignoring ResourceEvent from shell: {event:?}");
            }
        }
        BrowserToRenderer::HistorySync { length, index } => {
            // Stage 16 Phase 3, Rec 8 (finisher): the shell is
            // authoritative for session-history length / index —
            // mirror it into the realm, correcting the optimistic
            // `pushState` / `replaceState` bookkeeping.
            if let Some(js) = state.tab_js.as_mut() {
                js.set_history_length(length, index);
            }
        }
        BrowserToRenderer::WebSocketEvent { socket_id, event } => {
            // Stage 16 Phase 3, Rec 11 (finisher): a broker-pump
            // event for one socket — convert to the tab's
            // `WsEvent` (the `socket_id` rides in both, so the
            // tab routes it to the right `WebSocket` object) and
            // deliver it on the JS loop.
            use pluto_js::bindings::websocket::WsEvent;
            let ws_event = match event {
                pluto_ipc::WebSocketEvent::Opened { protocol, .. } => WsEvent::Opened {
                    id: socket_id,
                    protocol,
                },
                pluto_ipc::WebSocketEvent::Message { data, is_binary, .. } => WsEvent::Message {
                    id: socket_id,
                    data,
                    is_binary,
                },
                pluto_ipc::WebSocketEvent::Closed {
                    code,
                    reason,
                    clean,
                    ..
                } => WsEvent::Closed {
                    id: socket_id,
                    code,
                    reason,
                    clean,
                },
                pluto_ipc::WebSocketEvent::Error { message, .. } => WsEvent::Error {
                    id: socket_id,
                    message,
                },
            };
            if let Some(js) = state.tab_js.as_mut() {
                js.deliver_ws_event(ws_event);
            }
        }
        BrowserToRenderer::InputEvent(event) => {
            handle_input(state, events, event).await;
        }
    }
    true
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
    // Stage 15 (M7 Phase 1, Rec 11): the CSP gate for inline
    // scripts — the CURRENT document's policies, stashed on the
    // shared host by the load task from the document response's
    // headers, plus the document origin (the host's base URL is
    // the in-flight navigation's URL, which is the right origin
    // while the parse is still running and `state.document` is the
    // old page). The host copy — not the loader's — is the source
    // of truth here: in Broker mode the worker-side loader never
    // sees the document response.
    let csp = {
        let policies = state.host.lock().unwrap().csp_policies();
        let origin = Origin::from_url(state.host.lock().unwrap().base_url());
        Some((policies, origin))
    };
    let mut exec = crate::js_host::ScriptExecutor::new(Rc::clone(&state.host), csp);
    match msg {
        ExecMsg::Classic {
            node,
            inline,
            source,
            url,
            reply,
            ..
        } => {
            exec.exec_classic(js, node, &source, &url, inline);
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
            // Stage 16 (M8 Phase 2, Rec 4): the document is
            // parsed — register its media elements so their
            // sessions, candidates and tracks exist before any
            // script or tick touches them.
            js.register_media_elements();
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
    // Stage 12 Rec 7: a parser / defer script may have asked for
    // a navigation or an image load — act on it now, not on the
    // next tick (a `location.assign` in a parser-blocking script
    // must not wait a frame to start loading).
    drain_host_requests(state);
}

/// Drain the JS requests the host recorded (Stage 12 Rec 7
/// wiring): a navigation (`location.assign` / `replace` /
/// `reload`, `location.href = ...`) and image loads
/// (`Image().src` / an `<img>` `src` set from JS). The
/// navigation goes first: it replaces the document (fresh
/// realm), so image requests recorded in the same turn belong
/// to a page that is already gone and are dropped, not fetched
/// under the new page's epoch.
/// Drain the WAAPI queues (M6 Phase 3, Rec 11): `Element.animate`
/// starts and play-state controls, queued by JS through the host.
/// Runs on the tick, before layout: a start lands in the engine
/// with the current monotonic time, so the first tick after the
/// `animate()` call already samples it.
fn drain_animate_requests(state: &mut TabState) {
    let (starts, waapi_ops, css_ops) = {
        let mut host = state.host.lock().unwrap();
        (
            host.take_pending_animate(),
            host.take_pending_waapi_control(),
            host.take_pending_css_control(),
        )
    };
    if starts.is_empty() && waapi_ops.is_empty() && css_ops.is_empty() {
        return;
    }
    let now = std::time::Instant::now();
    let ctx = CalcCtx {
        font_size: 16.0,
        root_font_size: 16.0,
        viewport: state.viewport,
        pct_basis: None,
    };
    for (id, anim_id, spec) in starts {
        // Keyframe declaration text → sampled values (lengths
        // resolve to px here; unresolvable declarations — e.g.
        // `display` — drop out of their step).
        let mut steps: Vec<(f32, Vec<(String, AnimatedValue)>)> = Vec::new();
        for (offset, decls) in &spec.steps {
            let mut vals = Vec::new();
            for (prop, text) in decls {
                let value = parse_value(text);
                if let Some(v) = AnimatedValue::from_value(&value, &ctx) {
                    vals.push((prop.clone(), v));
                }
            }
            if !vals.is_empty() {
                steps.push((*offset, vals));
            }
        }
        if steps.is_empty() {
            // Nothing animatable: report it finished at once
            // rather than leaving a "running" object that never
            // samples.
            state
                .host
                .lock()
                .unwrap()
                .mark_waapi_finished(id, anim_id);
            continue;
        }
        let timing = AnimTiming {
            duration: (spec.duration_ms.max(0.0)) / 1000.0,
            delay: (spec.delay_ms.max(0.0)) / 1000.0,
            timing: Easing::parse(&spec.easing).unwrap_or(Easing::Linear),
            iteration_count: match spec.iterations {
                None => IterationCount::Infinite,
                Some(n) => IterationCount::Finite(n.max(0.0)),
            },
            direction: match spec.direction.as_str() {
                "reverse" => AnimationDirection::Reverse,
                "alternate" => AnimationDirection::Alternate,
                "alternate-reverse" => AnimationDirection::AlternateReverse,
                _ => AnimationDirection::Normal,
            },
            fill_mode: match spec.fill.as_str() {
                "forwards" => FillMode::Forwards,
                "backwards" => FillMode::Backwards,
                "both" => FillMode::Both,
                _ => FillMode::None,
            },
        };
        // Base values for a fill-`none` return: whatever the
        // computed cache / in-flight samples currently say.
        let mut base: HashMap<String, AnimatedValue> = HashMap::new();
        if let Some(current) = state.current_animated.get(&id) {
            for (prop, val) in current {
                base.insert(prop.clone(), val.clone());
            }
        }
        state
            .animation_engine
            .start_waapi(id, anim_id, timing, steps, base, now);
        state
            .host
            .lock()
            .unwrap()
            .note_waapi_started(id, anim_id, spec);
    }
    // Play-state controls. A cancel hands the underlying base
    // values back — restore them at once so the element snaps
    // back without waiting for a layout pass.
    let mut restored: Vec<(NodeId, Vec<(String, AnimatedValue)>)> = Vec::new();
    for (id, anim_id, op) in waapi_ops {
        match op {
            WaapiOp::Pause => {
                state.animation_engine.pause_waapi(id, anim_id, now);
            }
            WaapiOp::Play => {
                state.animation_engine.resume_waapi(id, anim_id, now);
            }
            WaapiOp::Cancel => {
                if let Some(base) = state.animation_engine.cancel_waapi(id, anim_id) {
                    restored.push((id, base.into_iter().collect()));
                    state.host.lock().unwrap().waapi_forget(id, anim_id);
                }
            }
            WaapiOp::Finish => {
                state.animation_engine.finish_waapi(id, anim_id, now);
            }
        }
    }
    for (id, name, op) in css_ops {
        match op {
            WaapiOp::Pause => {
                state.animation_engine.pause_css(id, &name, now);
            }
            WaapiOp::Play => {
                state.animation_engine.resume_css(id, &name, now);
            }
            WaapiOp::Cancel => {
                if let Some(base) = state.animation_engine.cancel_css(id, &name) {
                    restored.push((id, base.into_iter().collect()));
                    state.host.lock().unwrap().css_forget(id, &name);
                }
            }
            WaapiOp::Finish => {
                state.animation_engine.finish_css(id, &name, now);
            }
        }
    }
    if !restored.is_empty() {
        for (id, vals) in &restored {
            let entry = state.current_animated.entry(*id).or_default();
            for (prop, val) in vals {
                entry.insert(prop.clone(), val.clone());
            }
            let cache = state.computed_style_cache.entry(*id).or_default();
            for (prop, val) in vals {
                cache.insert(prop.clone(), val.to_css());
            }
        }
        {
            let mut host = state.host.lock().unwrap();
            for (id, vals) in &restored {
                let pairs: Vec<(String, String)> = vals
                    .iter()
                    .map(|(p, v)| (p.clone(), v.to_css()))
                    .collect();
                host.set_computed_styles(*id, pairs);
            }
        }
        state.render_requested.set(true);
    }
}

fn drain_host_requests(state: &mut TabState) {
    drain_animate_requests(state);
    let (navigation, images, media, tracks, history) = {
        let mut host = state.host.lock().unwrap();
        (
            host.take_pending_navigation(),
            host.take_pending_images(),
            host.take_pending_media(),
            host.take_pending_tracks(),
            host.take_pending_history(),
        )
    };
    // Stage 16 Phase 3, Rec 8 (finisher): the sync drain has no
    // event sender, so history requests stage here and the async
    // drain forwards them to the shell.
    let (pushes, replaces, traversals) = history;
    state.staged_history_push.extend(pushes);
    state.staged_history_replace.extend(replaces);
    state.staged_history_traversal.extend(traversals);
    if let Some((url, replace)) = navigation {
        // Stage 16 Phase 3, Rec 9 (finisher): a navigation whose
        // URL differs only in the fragment is same-document — no
        // fetch, no new entry. It stages here; the async drain
        // scrolls to the anchor and dispatches `hashchange`.
        if is_fragment_only_change(state, &url) {
            let old_url = state
                .host
                .lock()
                .unwrap()
                .document_url()
                .unwrap_or_default();
            state.pending_same_document = Some((old_url, url));
        } else {
            // `replace` (no new history entry) rides in
            // `nav_replace` to `DidCommit` (Stage 16 Phase 3,
            // Rec 8): a `location.replace()` commit replaces the
            // current entry in place instead of pushing.
            start_navigation_with_body(state, url, None, replace, None, false, (0.0, 0.0));
        }
    } else {
        for req in images {
            handle_js_image_request(state, req);
        }
        for req in media {
            handle_js_media_request(state, req);
        }
        for req in tracks {
            handle_js_track_request(state, req);
        }
    }
}

/// True when `url` differs from the current document's URL only
/// in the fragment (Stage 16 Phase 3, Rec 9): such a navigation
/// is same-document — no fetch, no new entry, just a scroll to
/// the anchor and a `hashchange` event. The "current URL" is the
/// host's URL (set at navigation start, moved in place by
/// `pushState` / same-document changes) — NOT `state.document`,
/// which is `None` until the first load commits and would turn a
/// hash change during the first parse into a full navigation
/// that swallows the `hashchange`.
fn is_fragment_only_change(state: &TabState, url: &str) -> bool {
    let Some(current) = state.host.lock().unwrap().document_url() else {
        return false;
    };
    same_document_url(&current, url) && current != url
}

/// True when two URLs are the same document (Stage 16 Phase 3,
/// Rec 9): equal up to the fragment — the fragment itself may
/// differ or be absent on either side.
fn same_document_url(a: &str, b: &str) -> bool {
    fn strip(u: &str) -> &str {
        u.split('#').next().unwrap_or(u)
    }
    strip(a) == strip(b)
}

/// Drain the JS requests that need an async tab thread (Stage 13
/// Rec 3/6/7): a JS-queued focus change (`el.focus()` /
/// `el.blur()`) and queued form submissions (`form.submit()` /
/// `requestSubmit()` after validation + the `submit` event).
/// Called from the tick, after the sync drain above: focus moves
/// and submissions restyle / navigate, so they need the async
/// context (and the one-tick delay is invisible).
async fn drain_host_requests_async(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
) {
    let (focus, submits) = {
        let mut host = state.host.lock().unwrap();
        (
            host.take_pending_focus(),
            host.take_pending_form_submits(),
        )
    };
    if let Some(focus) = focus {
        set_focus(state, events, focus).await;
    }
    for (form, submitter) in submits {
        submit_form(state, events, form, submitter).await;
    }
    // Stage 16 Phase 3, Rec 8 (finisher): forward the staged
    // history requests to the shell — pushes, then replaces, then
    // traversals. Each list keeps the page's order; cross-kind
    // order is not observable, since each request is a separate
    // JS call. A failed send means the tab is dead; stop.
    for (url, state_json) in std::mem::take(&mut state.staged_history_push) {
        if !emit_tab(
            events,
            RendererToBrowser::HistoryPush { url, state_json },
        )
        .await
        {
            return;
        }
    }
    for (url, state_json) in std::mem::take(&mut state.staged_history_replace) {
        if !emit_tab(
            events,
            RendererToBrowser::HistoryReplace { url, state_json },
        )
        .await
        {
            return;
        }
    }
    for delta in std::mem::take(&mut state.staged_history_traversal) {
        if !emit_tab(events, RendererToBrowser::HistoryGo { delta }).await {
            return;
        }
    }
    // Stage 16 Phase 3, Rec 9 (finisher): a staged
    // same-document navigation — update the URL in place, scroll
    // to the anchor, and dispatch `hashchange`. No fetch, no new
    // entry.
    if let Some((old_url, new_url)) = state.pending_same_document.take() {
        navigate_same_document(state, events, old_url, new_url).await;
    }
    // Storage events fan out here too (Stage 13 Rec 10d): other
    // tabs' mutations arrive as `storage` events on this tab's
    // window, never the mutating tab's.
    drain_storage_events(state);
}

/// A same-document fragment navigation (Stage 16 Phase 3, Rec 9):
/// no fetch, no new entry, no realm reset — update the URL in
/// place, scroll to the anchor (or the top when there is none),
/// and dispatch `hashchange`. The shell learns the new URL via a
/// `HistoryReplace` (carrying the current state, so the entry's
/// `history.state` survives), so the address bar and the session
/// entry agree.
///
/// No bfcache: Pluto keeps no back/forward cache — a traversal
/// re-fetches and rebuilds the document in a fresh realm (see
/// ARCHITECTURE.md, M8). A same-document navigation never leaves
/// the document, so there is nothing to cache or restore.
async fn navigate_same_document(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    old_url: String,
    new_url: String,
) {
    // The realm's `location` and the fetch base move to the new
    // URL; the document's `base_url` is left alone — a fragment
    // never changes relative-URL resolution.
    {
        let mut host = state.host.lock().unwrap();
        host.set_document_url(&new_url);
        host.set_base_url(&new_url);
    }
    if let Some(backend) = &state.fetch_backend {
        backend.set_base_url(&new_url);
    }
    // Scroll to the anchor: the element whose `id` is the
    // fragment, at the top of the viewport (clamped to the page's
    // max scroll). No anchor — or an empty fragment — scrolls to
    // the top.
    let fragment = new_url.split('#').nth(1).unwrap_or("");
    let mut target = (0.0, 0.0);
    // Resolve the anchor id first so the host lock is released
    // before the layout round trip below awaits.
    let anchor_id = if fragment.is_empty() {
        None
    } else {
        state.host.lock().unwrap().get_element_by_id(fragment)
    };
    if let Some(id) = anchor_id {
            let (reply_tx, reply_rx) = oneshot::channel();
            if state
                .layout_tx
                .send(LayoutJob::NodeRect { id, reply: reply_tx })
                .is_ok()
                && let Ok(Some(geometry)) = reply_rx.await
            {
                target = (geometry.x as f32, geometry.y as f32);
            }
        }
    let (max_x, max_y) = page_max_scroll(state).await;
    let clamped = (target.0.clamp(0.0, max_x), target.1.clamp(0.0, max_y));
    if clamped != state.viewport_scroll {
        let (dx, dy) = (
            clamped.0 - state.viewport_scroll.0,
            clamped.1 - state.viewport_scroll.1,
        );
        state.viewport_scroll = clamped;
        if emit_tab(events, RendererToBrowser::ScrollViewport { dx, dy }).await {
            // Sync the actor's tree (sticky offsets are
            // scroll-derived); the frame tick re-collects the list
            // from the synced tree.
            let _ = apply_viewport_scroll(state, events).await;
        }
    }
    // The entry's URL changed in place: tell the shell (a
    // `HistoryReplace` — no new entry), then dispatch `hashchange`
    // on the window. When a load is in flight, skip the replace:
    // the upcoming `DidCommit` carries the fragment (see
    // `handle_nav_outcome`), and a replace now would push the
    // fragment URL into the history ahead of the page being
    // loaded.
    let state_json = state
        .tab_js
        .as_ref()
        .map(|js| js.history_state())
        .unwrap_or(None);
    if state.load_handle.is_none() {
        let _ = emit_tab(
            events,
            RendererToBrowser::HistoryReplace {
                url: new_url.clone(),
                state_json,
            },
        )
        .await;
    }
    if let Some(js) = state.tab_js.as_mut() {
        js.dispatch_hashchange(old_url, new_url);
    }
}

/// Drain cross-tab `storage` events for this tab's origin and
/// dispatch them to the window (Stage 13 Rec 10d). The shared
/// store already filtered out this tab's own mutations; this
/// filters to the current document's origin (storage is
/// partitioned) and hands each event to `TabJs` (which builds a
/// plain `Event('storage')` with `key` / `oldValue` / `newValue`
/// / `url` on the window).
fn drain_storage_events(state: &mut TabState) {
    let Some(backends) = &state.tab_backends else {
        return;
    };
    use pluto_js::bindings::storage::StorageBackend;
    let events = backends.storage.drain_storage_events();
    if events.is_empty() {
        return;
    }
    let origin = state
        .document
        .as_ref()
        .map(|doc| pluto_ipc::Origin::from_url(doc.base_url()).serialize())
        .unwrap_or_default();
    let url = state
        .document
        .as_ref()
        .map(|doc| doc.base_url().to_string())
        .unwrap_or_default();
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    for event in events {
        if event.origin != origin {
            continue;
        }
        let _ = js.dispatch_storage_event(
            event.key.as_deref(),
            event.old_value.as_deref(),
            event.new_value.as_deref(),
            &url,
        );
    }
}

/// Run the worker: forward shell commands to the tab thread and tab
/// events to the shell. Ends when the command channel closes, the
/// tab thread ends, or `Shutdown` arrives.
pub async fn run_worker(
    tab_id: u64,
    mut commands: CommandReceiver,
    mut events: mpsc::Sender<crate::Message>,
) {
    let threads = spawn_tab_threads(tab_id, false);
    let tab_cmd_tx = threads.tab_cmd_tx;
    let mut tab_event_rx = threads.tab_event_rx;
    loop {
        tokio::select! {
            cmd = commands.recv() => {
                match cmd {
                    None => {
                        // The shell is gone: shut the tab thread
                        // down, then wait (bounded) for it to end
                        // before returning. The tab thread's
                        // shutdown saves (cookies, storage,
                        // permissions) must land while this
                        // worker's HOME guard is still held — an
                        // early return lets them land in the next
                        // test's home instead (Stage 13: a rare
                        // parallel-run permissions flake traced to
                        // exactly this).
                        let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
                        let _ = tokio::time::timeout(
                            Duration::from_secs(5),
                            async {
                                while tab_event_rx.recv().await.is_some() {}
                            },
                        )
                        .await;
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
                        if !emit(tab_id, &mut events, event).await {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
        // The shell is gone: there is nobody left to report to.
        // Same bounded wait as above: the tab thread's shutdown
        // saves must finish before this worker returns.
        if events.is_closed() {
            let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
            let _ = tokio::time::timeout(
                Duration::from_secs(5),
                async {
                    while tab_event_rx.recv().await.is_some() {}
                },
            )
            .await;
            break;
        }
    }
}

/// The channels to a freshly spawned tab-thread pair (layout actor
/// and tab thread), shared by the in-process worker ([`run_worker`])
/// and the cross-process renderer child ([`run_worker_transport`]).
/// The layout actor owns the `!Send` tree/document on its thread;
/// dropping the tab command sender (worker end) ends the actor.
struct TabThreads {
    tab_cmd_tx: tokio::sync::mpsc::Sender<BrowserToRenderer>,
    tab_event_rx: tokio::sync::mpsc::Receiver<RendererToBrowser>,
}

/// Spawn the layout-actor thread and the tab thread for one tab,
/// returning the channels between them and the worker bridge. Both
/// threads are per-tab and outlive the bridge: the tab thread ends
/// when its command channel closes (`Shutdown` / sender dropped),
/// and the actor ends when its job channel closes with it. The
/// shared loader (Stage 11 Rec 5) is built once here, per renderer
/// process — one tab per process today, so one handle covers the
/// tab — and handed to the tab thread, which builds its
/// `ResourceLoader` from it instead of a fresh private one.
fn spawn_tab_threads(tab_id: u64, broker: bool) -> TabThreads {
    let (layout_tx, layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
    std::thread::Builder::new()
        .name("pluto-layout".to_string())
        .spawn(move || layout_thread(layout_rx))
        .expect("layout actor thread spawns");
    let (tab_cmd_tx, tab_cmd_rx) = tokio::sync::mpsc::channel::<BrowserToRenderer>(100);
    let (tab_event_tx, tab_event_rx) = tokio::sync::mpsc::channel::<RendererToBrowser>(100);
    let layout_tx2 = layout_tx.clone();
    let shared = SharedLoader::new();
    std::thread::Builder::new()
        .name("pluto-tab".to_string())
        .spawn(move || tab_thread_main(tab_cmd_rx, tab_event_tx, layout_tx2, tab_id, shared, broker))
        .expect("tab thread spawns");
    TabThreads {
        tab_cmd_tx,
        tab_event_rx,
    }
}

/// Run the worker against a cross-process transport (Stage 9 Rec 9:
/// the `--renderer` child). The same tab thread + layout actor as
/// [`run_worker`], but commands arrive over `transport` and events
/// go back over it — the tab thread itself never knows the difference
/// (it only ever sees tokio channels).
///
/// Returns the process exit code: 0 for a clean shutdown (the
/// browser closed the socket, `Shutdown` ended the tab thread, or
/// the tab thread ended on its own), 1 when the transport failed
/// (a malformed message from the browser, or a dead tab thread — the
/// browser treats the EOF / dead child as a crash either way). A
/// malformed message from the *browser* is never recovered from
/// here either: the trusted peer is gone or corrupt, so the child
/// exits instead of guessing.
///
/// The child never creates a window or a wgpu device: the tab thread
/// only parses, lays out, runs JS, and builds display lists — the
/// same work the in-process worker does, over a different carrier.
pub(crate) async fn run_worker_transport<T: Transport>(mut transport: T, tab_id: u64) -> i32 {
    let threads = spawn_tab_threads(tab_id, true);
    let tab_cmd_tx = threads.tab_cmd_tx;
    let mut tab_event_rx = threads.tab_event_rx;
    loop {
        tokio::select! {
            cmd = recv_browser_message(&mut transport) => {
                match cmd {
                    Ok(Some(BrowserToRenderer::Shutdown)) => {
                        // Forward the shutdown, then drain the tab
                        // thread's final events (bounded: a tab
                        // thread that will not die cannot hold the
                        // process open) before exiting cleanly.
                        let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
                        let _ = tokio::time::timeout(
                            Duration::from_secs(5),
                            async {
                                while let Some(event) = tab_event_rx.recv().await {
                                    if transport.send_message(&event).await.is_err() {
                                        break;
                                    }
                                }
                            },
                        )
                        .await;
                        return 0;
                    }
                    Ok(Some(cmd)) => {
                        if tab_cmd_tx.send(cmd).await.is_err() {
                            eprintln!("[pluto-renderer] tab thread is gone");
                            return 1;
                        }
                    }
                    Ok(None) => {
                        // The browser closed the socket: shut the
                        // tab thread down and exit cleanly.
                        let _ = tab_cmd_tx.send(BrowserToRenderer::Shutdown).await;
                        return 0;
                    }
                    Err(e) => {
                        // A malformed message from the browser, or
                        // a dead socket: exit, never try to recover.
                        eprintln!("[pluto-renderer] IPC receive failed: {e}");
                        return 1;
                    }
                }
            }
            event = tab_event_rx.recv() => {
                match event {
                    Some(event) => {
                        if transport.send_message(&event).await.is_err() {
                            // The browser is gone: nothing left to
                            // report to.
                            return 0;
                        }
                    }
                    None => {
                        // The tab thread ended on its own.
                        return 0;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Stage 13 (M5 forms): focus, keys, editing, submission, caret.
// ------------------------------------------------------------------

/// Sync the form pseudo-class sets (`:valid` / `:invalid` /
/// `:required` / `:optional` / `:disabled` / `:checked` /
/// `:read-only`) from the live `Dom` into `element_state`
/// (Stage 13 Rec 8). Called at the top of every `restyle`, so a
/// value edit that flips validity re-matches selectors on the
/// next layout pass. Validity itself is computed by the same
/// `pluto_js::bindings::forms::validity_of` the JS bindings use —
/// one rule, two callers.
fn sync_form_pseudo_state(state: &mut TabState) {
    let Some(doc) = state.document.as_ref() else {
        return;
    };
    let dom = doc.dom();
    let Some(root) = dom.root() else {
        return;
    };
    let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
    let host: &dyn pluto_js::DomHost = &shared;
    let mut invalid = HashSet::new();
    let mut required = HashSet::new();
    let mut disabled = HashSet::new();
    let mut checked = HashSet::new();
    let mut read_only = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        stack.extend(dom.children(id).iter().rev());
        if !host.is_form_control(id) && dom.tag(id) != Some("button") {
            continue;
        }
        if dom.attr(id, "required").is_some() {
            required.insert(id);
        }
        if host.is_disabled(id) {
            disabled.insert(id);
        }
        if host.form_checked(id) == Some(true) {
            checked.insert(id);
        }
        if dom.attr(id, "readonly").is_some() {
            read_only.insert(id);
        }
        if pluto_js::bindings::forms::will_validate(host, id)
            && !pluto_js::bindings::forms::validity_of(host, id).valid()
        {
            invalid.insert(id);
        }
    }
    state.element_state.invalid = invalid;
    state.element_state.required = required;
    state.element_state.disabled = disabled;
    state.element_state.checked = checked;
    state.element_state.read_only = read_only;
}

/// Move keyboard focus (Stage 13 Rec 3): updates `state.focused`,
/// the `:focus` set, and the host's `activeElement` mirror, and
/// fires `blur` on the old element then `focus` on the new one
/// (both sync dispatches, so a listener sees the move at once).
/// The caret blink restarts (visible) on every focus change.
async fn set_focus(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    new: Option<NodeId>,
) {
    let old = state.focused;
    if old == new {
        return;
    }
    if let Some(old_id) = old
        && let Some(js) = state.tab_js.as_mut() {
            let _ = js.dispatch_event_sync(old_id, "blur", false);
        }
    state.focused = new;
    state.element_state.focus = new.into_iter().collect();
    state.host.lock().unwrap().set_focused(new);
    state.caret_blink_epoch = std::time::Instant::now();
    state.caret_blink_on = true;
    if let Some(new_id) = new
        && let Some(js) = state.tab_js.as_mut() {
            let _ = js.dispatch_event_sync(new_id, "focus", false);
        }
    let mut changed: Vec<NodeId> = Vec::new();
    if let Some(old_id) = old {
        changed.push(old_id);
    }
    if let Some(new_id) = new {
        changed.push(new_id);
    }
    let dirty = restyle_dirty_set(state, &changed, pseudo_reach(state, "focus"));
    restyle(state, events, &dirty).await;
}

/// The `tabindex` value of a node (`None` = no `tabindex`
/// attribute; unparseable values count as 0, per spec).
fn tabindex_of(dom: &pluto_html::Dom, id: NodeId) -> Option<i64> {
    dom.attr(id, "tabindex")
        .and_then(|v| v.trim().parse::<i64>().ok().or(Some(0)))
}

/// The Tab-order list (Stage 13 Rec 3): positive-`tabindex`
/// elements first (ascending `tabindex`, DOM order within),
/// then the rest in DOM order. `tabindex="-1"` elements are
/// focusable (click / `focus()`) but never in the Tab order,
/// and disabled controls are excluded entirely.
fn tab_order(doc: &Document) -> Vec<NodeId> {
    let dom = doc.dom();
    let all = focusable_in_order(doc);
    let mut positive: Vec<(i64, usize, NodeId)> = Vec::new();
    let mut rest: Vec<NodeId> = Vec::new();
    for (i, id) in all.into_iter().enumerate() {
        match tabindex_of(dom, id) {
            Some(t) if t > 0 => positive.push((t, i, id)),
            Some(t) if t < 0 => {}
            _ => rest.push(id),
        }
    }
    positive.sort_by_key(|(t, i, _)| (*t, *i));
    positive.into_iter().map(|(_, _, id)| id).chain(rest).collect()
}

/// The `code` for a `KeyboardEvent` from its `key` (best effort:
/// physical codes are not available from the shell yet, so the
/// code mirrors the key for named keys and `KeyX` for letters).
fn key_code(key: &str) -> String {
    match key {
        " " => "Space".to_string(),
        "Enter" | "Tab" | "Escape" | "Backspace" | "Delete" | "Home" | "End" | "PageUp"
        | "PageDown" | "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown" => key.to_string(),
        _ => {
            let mut chars = key.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphabetic() => {
                    format!("Key{}", c.to_ascii_uppercase())
                }
                _ => key.to_string(),
            }
        }
    }
}

/// Whether `key` is a text control that accepts typed input:
/// `<textarea>`, or a single-line `<input>` of a text-like type.
fn is_text_control(host: &dyn pluto_js::DomHost, key: NodeId) -> bool {
    match host.tag_name(key).as_deref() {
        Some("textarea") => true,
        Some("input") => matches!(
            host.input_type(key).as_deref(),
            Some("text" | "search" | "url" | "tel" | "email" | "password" | "number")
        ),
        _ => false,
    }
}

/// Handle one editing key on a focused text control (Stage 13
/// Rec 2/4): printable keys insert, Backspace/Delete remove,
/// arrows/Home/End move the caret, Ctrl+A selects all, Enter in a
/// `<textarea>` inserts a newline. Mutations fire a sync `input`
/// event and restyle (the value text is part of the layout, and
/// the caret restarts visible). Returns true when the key was
/// consumed (no further default action).
async fn handle_edit_key(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    focused: NodeId,
    key: &str,
    modifiers: pluto_ipc::Modifiers,
) -> bool {
    let is_textarea;
    {
        let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
        let host: &dyn pluto_js::DomHost = &shared;
        if !is_text_control(host, focused) {
            return false;
        }
        is_textarea = host.tag_name(focused).as_deref() == Some("textarea");
    }
    // Ctrl/Meta chords are not text (except Ctrl+A = select all).
    if modifiers.ctrl || modifiers.meta {
        if (key == "a" || key == "A") && !modifiers.alt {
            let len = {
                let host = state.host.lock().unwrap();
                host.form_value(focused).map(|v| v.chars().count()).unwrap_or(0)
            };
            state.host.lock().unwrap().set_form_selection(focused, 0, len);
            state.caret_blink_epoch = std::time::Instant::now();
            restyle(state, events, &[focused]).await;
        }
        return false;
    }
    let mut inserted: Option<String> = None;
    match key {
        "Backspace" => {
            state.host.lock().unwrap().delete_form_text(focused, true);
        }
        "Delete" => {
            state.host.lock().unwrap().delete_form_text(focused, false);
        }
        "ArrowLeft" => {
            let (start, end) = state
                .host
                .lock()
                .unwrap()
                .form_selection(focused)
                .unwrap_or((0, 0));
            let at = if modifiers.shift { end } else { start.min(end) };
            let at = at.saturating_sub(1);
            let (s, e) = if modifiers.shift { (start.min(at), start.max(at)) } else { (at, at) };
            state.host.lock().unwrap().set_form_selection(focused, s, e);
            state.caret_blink_epoch = std::time::Instant::now();
            restyle(state, events, &[focused]).await;
            return true;
        }
        "ArrowRight" => {
            let len = {
                let host = state.host.lock().unwrap();
                host.form_value(focused).map(|v| v.chars().count()).unwrap_or(0)
            };
            let (start, end) = state
                .host
                .lock()
                .unwrap()
                .form_selection(focused)
                .unwrap_or((0, 0));
            let at = if modifiers.shift { end } else { start.max(end) };
            let at = (at + 1).min(len);
            let (s, e) = if modifiers.shift { (start.min(at), start.max(at)) } else { (at, at) };
            state.host.lock().unwrap().set_form_selection(focused, s, e);
            state.caret_blink_epoch = std::time::Instant::now();
            restyle(state, events, &[focused]).await;
            return true;
        }
        "Home" => {
            state.host.lock().unwrap().set_form_selection(focused, 0, 0);
            state.caret_blink_epoch = std::time::Instant::now();
            restyle(state, events, &[focused]).await;
            return true;
        }
        "End" => {
            let len = {
                let host = state.host.lock().unwrap();
                host.form_value(focused).map(|v| v.chars().count()).unwrap_or(0)
            };
            state.host.lock().unwrap().set_form_selection(focused, len, len);
            state.caret_blink_epoch = std::time::Instant::now();
            restyle(state, events, &[focused]).await;
            return true;
        }
        "Enter" if is_textarea => {
            state.host.lock().unwrap().insert_form_text(focused, "\n");
            inserted = Some("\n".to_string());
        }
        _ => {
            // A single printable character inserts; anything else
            // (arrows handled above, function keys, Tab) falls
            // through to the default action below.
            if key.chars().count() == 1 && !modifiers.alt {
                state.host.lock().unwrap().insert_form_text(focused, key);
                inserted = Some(key.to_string());
            } else {
                return false;
            }
        }
    }
    // The mutation landed: fire `input` (listeners read the new
    // value off the control) and restyle so the painted text and
    // the caret move.
    if let Some(js) = state.tab_js.as_mut() {
        let _ = js.dispatch_typed_event_sync(
            focused,
            "input",
            true,
            TypedEventKind::Input { data: inserted },
        );
    }
    state.caret_blink_epoch = std::time::Instant::now();
    restyle(state, events, &[focused]).await;
    true
}

/// Percent-encode one `application/x-www-form-urlencoded` value:
/// alphanumerics + `-_.` verbatim, space as `+`, the rest `%XX`.
fn form_urlencode(value: &str) -> String {
    let mut out = String::new();
    for &b in value.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Encode an entry list as `application/x-www-form-urlencoded`.
fn encode_urlencoded(entries: &[(String, String)]) -> Vec<u8> {
    entries
        .iter()
        .map(|(n, v)| format!("{}={}", form_urlencode(n), form_urlencode(v)))
        .collect::<Vec<_>>()
        .join("&")
        .into_bytes()
}

/// Encode an entry list as `multipart/form-data` with a fresh
/// boundary (returns body + full `Content-Type` value).
fn encode_multipart(entries: &[(String, String)]) -> (Vec<u8>, String) {
    let boundary = format!(
        "----pluto-form-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        std::process::id()
    );
    let mut body = Vec::new();
    for (name, value) in entries {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (
        body,
        format!("multipart/form-data; boundary={boundary}"),
    )
}

/// Encode an entry list as `text/plain` (`name=value` lines).
fn encode_text_plain(entries: &[(String, String)]) -> Vec<u8> {
    entries
        .iter()
        .map(|(n, v)| format!("{n}={v}"))
        .collect::<Vec<_>>()
        .join("\r\n")
        .into_bytes()
}

/// First control in `form` that fails constraint validation
/// (Stage 13 Rec 8), in document order — `reportValidity`
/// focuses and reports this one, and submission stops here.
fn first_invalid_control(state: &TabState, form: NodeId) -> Option<NodeId> {
    let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
    let host: &dyn pluto_js::DomHost = &shared;
    host.form_controls(form).into_iter().find(|&control| pluto_js::bindings::forms::will_validate(host, control)
            && !pluto_js::bindings::forms::validity_of(host, control).valid())
}

/// Submit a form (Stage 13 Rec 6/7/8): validation gate first
/// (first invalid control gets an `invalid` event + focus, no
/// navigation), then the cancelable `submit` event
/// (`preventDefault` blocks navigation), then the entry list is
/// encoded and navigated THROUGH THE EXISTING LOADER (same
/// Stage 9 gates as a link click) via `start_navigation_with_body`.
async fn submit_form(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    form: NodeId,
    submitter: Option<NodeId>,
) {
    // Validation gate: report the first invalid control and stop.
    if let Some(invalid) = first_invalid_control(state, form) {
        if let Some(js) = state.tab_js.as_mut() {
            let _ = js.dispatch_event_sync(invalid, "invalid", false);
        }
        set_focus(state, events, Some(invalid)).await;
        return;
    }
    // Cancelable `submit` event (bubbles, per spec).
    let not_prevented = match state.tab_js.as_mut() {
        Some(js) => js.dispatch_event_sync(form, "submit", true).unwrap_or(true),
        None => true,
    };
    if !not_prevented {
        return;
    }
    let (entries, action, method, enctype, target) = {
        let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
        let host: &dyn pluto_js::DomHost = &shared;
        let entries = host.form_entries(form, submitter);
        let action = host.get_attribute(form, "action").unwrap_or_default();
        let method = host
            .get_attribute(form, "method")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let enctype = host
            .get_attribute(form, "enctype")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let target = host.get_attribute(form, "target").unwrap_or_default();
        (entries, action, method, enctype, target)
    };
    // Only `_self` (or no target) navigates this tab; other
    // targets (`_blank`, named frames) are not opened in M5.
    if !target.is_empty() && target != "_self" {
        eprintln!("[pluto] form target {target} not supported in M5 — submission dropped");
        return;
    }
    let base_url = state
        .document
        .as_ref()
        .map(|doc| doc.base_url().to_string())
        .unwrap_or_default();
    let url = if action.trim().is_empty() {
        base_url.clone()
    } else {
        pluto_html::resolve_url(&base_url, action.trim())
    };
    let method = if method == "post" { "post" } else { "get" };
    if method == "get" {
        // GET: entries ride the query string; the fragment (if
        // any) stays last.
        let query = String::from_utf8_lossy(&encode_urlencoded(&entries)).into_owned();
        let mut url = url;
        if !query.is_empty() {
            let (head, frag) = match url.split_once('#') {
                Some((h, f)) => (h.to_string(), Some(f.to_string())),
                None => (url.clone(), None),
            };
            url = if head.contains('?') {
                format!("{head}&{query}")
            } else {
                format!("{head}?{query}")
            };
            if let Some(f) = frag {
                url = format!("{url}#{f}");
            }
        }
        start_navigation(state, url);
    } else {
        let (body, content_type) = match enctype.as_str() {
            "multipart/form-data" => encode_multipart(&entries),
            "text/plain" => (encode_text_plain(&entries), "text/plain".to_string()),
            _ => (
                encode_urlencoded(&entries),
                "application/x-www-form-urlencoded".to_string(),
            ),
        };
        start_navigation_with_body(
            state,
            url,
            Some(page::PostBody {
                bytes: body,
                content_type,
            }),
            false,
            None,
            false,
            (0.0, 0.0),
        );
    }
}

/// Reset a form (Stage 13 Rec 6/7): cancelable `reset` event;
/// `preventDefault` keeps the current values, otherwise the
/// seeded defaults return and the form restyles.
async fn reset_form_action(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    form: NodeId,
) {
    let not_prevented = match state.tab_js.as_mut() {
        Some(js) => js.dispatch_event_sync(form, "reset", true).unwrap_or(true),
        None => true,
    };
    if !not_prevented {
        return;
    }
    state.host.lock().unwrap().reset_form(form);
    let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
    let controls = shared.form_controls(form);
    restyle(state, events, &controls).await;
}

/// Enter on a focused control (Stage 13 Rec 6): a submit button
/// submits its form, a reset button resets it, Enter in a
/// single-line text input submits its form, Enter in a
/// `<textarea>` is editing (handled by `handle_edit_key`).
/// Returns true when the key was consumed.
async fn submit_on_enter(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    focused: NodeId,
) -> bool {
    let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
    let host: &dyn pluto_js::DomHost = &shared;
    let tag = host.tag_name(focused);
    let input_type = host.input_type(focused);
    let (form, is_submit, is_reset) = match tag.as_deref() {
        Some("button") => {
            let ty = host
                .get_attribute(focused, "type")
                .unwrap_or_default()
                .to_ascii_lowercase();
            let is_submit = ty.is_empty() || ty == "submit";
            let is_reset = ty == "reset";
            (host.form_owner(focused), is_submit, is_reset)
        }
        Some("input") => match input_type.as_deref() {
            Some("submit") => (host.form_owner(focused), true, false),
            Some("reset") => (host.form_owner(focused), false, true),
            Some("button") => (None, false, false),
            _ => (host.form_owner(focused), false, false),
        },
        _ => (None, false, false),
    };
    let Some(form) = form else {
        return false;
    };
    if is_reset {
        reset_form_action(state, events, form).await;
        return true;
    }
    if is_submit {
        submit_form(state, events, form, Some(focused)).await;
        return true;
    }
    // Enter in a text-like input submits the form (the
    // single-text-input / implicit-submission rule, simplified:
    // any text input Enter submits when a form owns it).
    if tag.as_deref() == Some("input") && is_text_control(host, focused) {
        submit_form(state, events, form, None).await;
        return true;
    }
    false
}

// ------------------------------------------------------------------
// Stage 13 (M5 forms): caret painting.
// ------------------------------------------------------------------

/// Whether the caret blink is currently in its visible half
/// (~530 ms on / ~530 ms off, from the last focus change or edit).
fn caret_blink_on(state: &TabState) -> bool {
    (state.caret_blink_epoch.elapsed().as_millis() / 530).is_multiple_of(2)
}

/// Caret + selection paint items for the focused text control
/// (Stage 13 Rec 4), or `None` when no caret is shown (no focus,
/// not a text control, window unfocused, blink off). Positions
/// are estimated: ~7 px per char at the control's font size and
/// ~16 px per line — the layout actor does not expose shaped
/// text metrics to the tab thread, so an on-box check owns the
/// exact placement.
fn caret_display_items(state: &mut TabState) -> Vec<pluto_paint::DisplayItem> {
    let Some(focused) = state.focused else {
        return Vec::new();
    };
    if !state.window_focused || !caret_blink_on(state) {
        return Vec::new();
    }
    let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
    let host: &dyn pluto_js::DomHost = &shared;
    if !is_text_control(host, focused) {
        return Vec::new();
    }
    let mut host_mut = state.host.lock().unwrap();
    let Some(geometry) = host_mut.geometry(focused) else {
        return Vec::new();
    };
    let value = host_mut.form_value(focused).unwrap_or_default();
    let (sel_start, sel_end) = host_mut.form_selection(focused).unwrap_or((0, 0));
    drop(host_mut);
    const CHAR_W: f32 = 7.0;
    const LINE_H: f32 = 16.0;
    const PAD_X: f32 = 2.0;
    const PAD_Y: f32 = 2.0;
    const CARET_H: f32 = 14.0;
    let caret_chars = value.chars().count().min(sel_end);
    let x = (geometry.x as f32 + PAD_X + caret_chars as f32 * CHAR_W)
        .min(geometry.x as f32 + geometry.width as f32 - 3.0);
    // Single-line controls centre the caret vertically; a
    // textarea's caret walks lines (row = newlines before it).
    let is_textarea = {
        let shared = crate::js_host::SharedHost(Rc::clone(&state.host));
        shared.tag_name(focused).as_deref() == Some("textarea")
    };
    let y = if is_textarea {
        let row = value.chars().take(sel_end).filter(|&c| c == '\n').count();
        geometry.y as f32 + PAD_Y + row as f32 * LINE_H
    } else {
        geometry.y as f32 + ((geometry.height as f32 - CARET_H).max(0.0) / 2.0)
    };
    let mut items = Vec::new();
    // Selection first (under the caret): single-line only — a
    // multi-line selection rect is a follow-up, not a guess.
    if sel_start != sel_end && !is_textarea {
        let sx = geometry.x as f32 + PAD_X + sel_start as f32 * CHAR_W;
        let sw = (sel_end - sel_start) as f32 * CHAR_W;
        items.push(pluto_paint::DisplayItem::SelectionRect {
            x: sx,
            y,
            width: sw,
            height: CARET_H,
            color: [0.0, 0.47, 0.84, 0.4],
        });
    }
    items.push(pluto_paint::DisplayItem::Caret {
        x,
        y,
        height: CARET_H,
        color: [0.0, 0.0, 0.0, 1.0],
    });
    items
}

/// A display list with the caret items appended (Stage 13 Rec 4):
/// the cached `last_display` stays caret-free, so the blink tick
/// can re-derive it from the base each phase.
fn with_caret(state: &mut TabState, base: Arc<DisplayList>) -> Arc<DisplayList> {
    let items = caret_display_items(state);
    if items.is_empty() {
        return base;
    }
    let mut list = DisplayList(base.0.clone(), base.1);
    for item in items {
        list.push(item);
    }
    Arc::new(list)
}

/// Start a navigation: cancel the previous load (its token and its
/// task — a slow fetch no longer blocks the command loop), get a
/// fresh realm for the new document, then spawn the load as a
/// `spawn_local` task on the tab thread's runtime. The old page
/// stays visible until the new list lands; a stale outcome is
/// discarded by epoch in `handle_nav_outcome`.
fn start_navigation(state: &mut TabState, url: String) {
    start_navigation_with_body(state, url, None, false, None, false, (0.0, 0.0));
}

/// Start a navigation with the history semantics attached (Stage
/// 16 Phase 3, Rec 8): `replace` (no new history entry —
/// `location.replace()`), `state_json` (the session-history state
/// to restore into the fresh realm — `Some` for a Back/Forward
/// traversal or a reload of a stateful entry, `None` for a fresh
/// navigation, whose `history.state` is `null`), `traversal`
/// (a Back/Forward traversal fires `popstate` after the commit;
/// a reload restores the state but fires none), and `scroll` (the
/// viewport scroll to restore — the entry's saved scroll for a
/// traversal / reload, `(0.0, 0.0)` for a fresh navigation; the
/// worker's layout mirror must match the shell's compositor, or
/// fixed / sticky elements lay out at the wrong offset).
fn start_navigation_with_body(
    state: &mut TabState,
    url: String,
    post_body: Option<page::PostBody>,
    replace: bool,
    state_json: Option<String>,
    traversal: bool,
    scroll: (f32, f32),
) {
    // Stage 9 Rec 11: visiting an origin extends its
    // permission grants (they expire after ~90 days of
    // non-visit). Opaque origins are skipped by `note_visit`.
    if let Some(backends) = &state.tab_backends {
        use pluto_js::bindings::permissions::PermissionsBackend;
        backends
            .permissions
            .note_visit(&pluto_ipc::Origin::from_url(&url));
    }
    if let Some(token) = state.nav_cancel.take() {
        token.cancel();
    }
    if let Some(handle) = state.load_handle.take() {
        handle.abort();
    }
    state.nav_epoch += 1;
    let epoch = state.nav_epoch;
    state.nav_replace = replace;
    state.pending_popstate = if traversal {
        Some(state_json.clone())
    } else {
        None
    };
    let token = CancellationToken::new();
    state.nav_cancel = Some(token.clone());
    // Stage 9 Rec 15: the navigation's referrer is the current
    // document (link-click behaviour). The loader keeps it as the
    // referrer context until the new document's response
    // re-points it; with no document yet (first load) no Referer
    // is sent. (Typed-URL navigations cannot be told apart from
    // link navigations here — see `service::referer_for_request`.)
    if let Some(doc) = &state.document {
        state.fetch_source.set_referrer_document(doc.base_url());
    }
    // Perf (Stage 8 Rec 5): a navigation — address bar, reload,
    // or link click — starts the first-paint clock here, not just
    // in the shell's `commit` (link clicks bypass `commit`).
    if let Ok(mut counters) = perf::global().lock() {
        counters.note_navigation_start();
    }
    // A new page drops the old page's input state: hover/active/
    // focus chains and any in-flight wheel gesture belong to node
    // ids that no longer exist.
    state.hovered.clear();
    state.active_element = None;
    state.pointer_down_target = None;
    state.focused = None;
    state.wheel_chain.clear();
    state.wheel_idle_ticks = 0;
    // Stage 11 Rec 1/7/8: a new page drops the old page's image
    // state (in-flight fetches are stale — their outcomes carry
    // the old epoch), its favicon, and its title (re-emitted when
    // the new page's title is read after the first submit).
    state.images.note_navigation();
    state.favicon_url = None;
    state.favicon_resource = None;
    state.last_title = String::new();
    state.last_display = None;
    state.last_image_map = ImageMap::new();
    state.last_canvas_announced.clear();
    state.animation_visible.clear();
    state.animation_geometry_pending.clear();
    // The shell resets its scroll on commit; mirror it here (the
    // viewport scroll is mirrored, not messaged — see
    // `viewport_scroll`). A traversal / reload restores the
    // entry's saved scroll instead (Stage 16 Phase 3, Rec 8 —
    // finisher); the shell restores the same scroll into its
    // compositor when the new list lands, so the two agree.
    state.viewport_scroll = scroll;
    // M6 Phase 3 (Rec 11/12): navigation clears running
    // animations/transitions, sampled values, the computed-style
    // cache, and pending fonts — the new document starts clean
    // and fetches its own faces.
    state.animation_engine.clear();
    state.current_animated.clear();
    state.computed_style_cache.clear();
    state.pending_fonts.clear();
    state.font_fetch_urls.clear();
    state.host.lock().unwrap().clear_animations();
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
    // A traversal / stateful reload restores the entry's state
    // into the fresh realm before any script runs, so the new
    // document's scripts read the right `history.state`.
    if let Some(state_json) = state_json {
        state.tab_js().set_history_state(Some(state_json));
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
        host.set_base_url(&url);
        host.set_element_state(state.element_state.clone());
    }
    let fetch_source = state.fetch_source.clone();
    let exec_tx = state.exec_tx.clone();
    let outcome_tx = state.load_outcome_tx.clone();
    let host = Rc::clone(&state.host);
    // Stage 11 Rec 1/9: the load reports image discoveries and
    // document-fetch progress to the tab thread's sinks; the tab
    // thread owns the image pipeline, the shell owns the bar.
    // Stage 12 Rec 1: the load also pre-fetches module import
    // graphs into the fresh realm's store (`navigate()` above
    // replaced it — grab it after, not before).
    let module_sources = state.tab_js().module_sources();
    let sinks = page::LoadSinks {
        images: Some(state.image_request_tx.clone()),
        progress: Some(state.progress_tx.clone()),
        module_sources: Some(module_sources),
    };
    let tab_id = state.tab_id;
    let viewport = state.viewport;
    // Stage 13 Rec 11: the previous document's URL is the
    // referrer for the mixed-download check (the broker owns the
    // loader-side referrer context in broker mode).
    let referrer = state
        .document
        .as_ref()
        .map(|doc| doc.base_url().to_string())
        .unwrap_or_default();
    let handle = tokio::task::spawn_local(async move {
        let hooks = crate::js_host::LoadHooks::new(epoch, exec_tx, host);
        let outcome = page::load_page_cancellable_with_hooks(
            &fetch_source,
            url.clone(),
            token,
            Some(Box::new(hooks)),
            sinks,
            tab_id,
            epoch,
            viewport,
            post_body,
            &referrer,
        )
        .await;
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
    // Stage 9 Rec 5: persist the HSTS store (entries may have
    // been recorded during this navigation's fetches) and the
    // HTTPS-only state after every completed navigation.
    // Stage 13 Rec 9: same for the cookie jar — a login taken in
    // during this navigation survives a restart (session cookies
    // are stripped before the write). Skipped in broker mode
    // (Stage 13 Rec 11): the broker owns the jar + HSTS store.
    if state.broker_handle.is_none() {
        crate::security::save_from(&state.loader);
        crate::cookies::save_from(&state.loader);
    }
    match outcome {
        LoadOutcome::Loaded {
            document,
            final_url,
            ..
        } => {
            eprintln!("[pluto] worker fetched {final_url}");
            // Stage 10 Rec 3: a committed URL is visited from now
            // on — record it before laying out, so the new page's
            // own links to it already match `:visited`.
            state
                .visited_urls
                .insert(visited_key(&final_url).to_string());
            state.document = Some((*document).clone());
            // Stage 11 Rec 7: determine the favicon now that the
            // document is committed — link[rel=icon], else the
            // origin's /favicon.ico, once per origin per session.
            // A missing icon is remembered, never re-requested.
            maybe_request_favicon(state, &final_url);
            // M6 Phase 3 (Rec 12): the committed document's
            // `@font-face` sources fetch through the loader now.
            maybe_request_font_faces(state);
            if !emit_tab(events, RendererToBrowser::DidCommit {
                // Stage 16 Phase 3, Rec 9 (finisher): the
                // committed URL is the realm's effective URL — a
                // same-document fragment change during the load
                // (a script setting `location.hash` during the
                // first parse) moves the realm's URL before the
                // commit lands, and the entry must carry the
                // fragment; the fetch-side URL has none.
                url: {
                    let host_url = state.host.lock().unwrap().document_url();
                    match host_url {
                        Some(u) if same_document_url(&u, &final_url) => u,
                        _ => final_url.clone(),
                    }
                },
                status: format!("Loaded {final_url}"),
                // Stage 16 Phase 3, Rec 8 (finisher): the shell
                // replaces the current entry in place on a
                // `location.replace()` commit instead of pushing.
                replace: state.nav_replace,
            })
            .await
            {
                return;
            }
            layout_and_submit(state, events, Some(*document), false, std::time::Instant::now()).await;
            // Stage 16 Phase 3, Rec 8 (finisher): a traversal's
            // `popstate` fires after the commit — the state was
            // restored into the fresh realm at navigation start.
            // A failed traversal load fires none (the error page
            // has no realm state to report); a stale outcome never
            // reaches here (the epoch guard above returns first).
            if let Some(state_json) = state.pending_popstate.take()
                && let Some(js) = state.tab_js.as_mut() {
                    js.dispatch_popstate(state_json);
                }
        }
        LoadOutcome::Failed { url, reason, .. } => {
            eprintln!("[pluto] worker fetch failed: {reason}");
            let document = page::parse_document(&page::error_document(&url, &reason));
            state.document = Some(document.clone());
            state.scroll_offsets.clear();
            layout_and_submit(state, events, Some(document), false, std::time::Instant::now()).await;
            let _ = emit_tab(
                events,
                RendererToBrowser::LoadFailed { url, reason },
            )
            .await;
        }
    }
}

/// Stage 11 Rec 7: determine the favicon URL for the committed
/// document and fetch it through the image pipeline — once per
/// origin per session. A missing icon is remembered (in
/// `favicon_missing`) and never re-requested; an attempt whose
/// outcome went stale (a newer navigation superseded it) is
/// un-marked in the outcome handler so a later visit retries.
fn maybe_request_favicon(state: &mut TabState, final_url: &str) {
    let Some(doc) = state.document.as_ref() else {
        return;
    };
    let Some(favicon_url) = ImageStore::favicon_url_for(doc.dom(), final_url) else {
        return;
    };
    let origin = origin_of(final_url);
    if state.images.favicon_missing.contains(&origin) {
        return;
    }
    if !state.images.favicon_attempted.insert(origin) {
        return;
    }
    state.favicon_url = Some(favicon_url.clone());
    let req = ImageRequest {
        url: favicon_url,
        base_url: final_url.to_string(),
        is_imageset: false,
        integrity: None,
    };
    if state.images.request(&req) {
        ImageStore::spawn_fetch(
            state.fetch_source.clone(),
            req,
            state.tab_id,
            state.nav_epoch,
            CancellationToken::new(),
            state.image_outcome_tx.clone(),
        );
    }
}

/// M6 Phase 3 (Rec 12): fetch the committed document's
/// `@font-face` sources through the existing loader — the same
/// Stage 9 gates as any other subresource (CSP / referrer /
/// HTTPS-only / caps), never a second fetch path. At most 16 faces
/// per document, ~10 MiB per face; dedup by resolved URL. Outcomes
/// arrive on `font_outcome_rx`; the bytes land in `pending_fonts`
/// for the next layout (fallback until they arrive).
fn maybe_request_font_faces(state: &mut TabState) {
    let Some(doc) = state.document.as_ref() else {
        return;
    };
    let base_url = doc.base_url().to_string();
    let mut count = 0;
    for face in doc.sheet().font_faces.iter() {
        if count >= 16 {
            break;
        }
        let Some(src) = face.src.as_ref() else {
            continue;
        };
        let Some(raw) = parse_font_src_url(src) else {
            continue;
        };
        let Ok(url) = resolve_font_url(&base_url, &raw) else {
            continue;
        };
        if !state.font_fetch_urls.insert(url.clone()) {
            continue;
        }
        count += 1;
        let request =
            page::font_resource_request(&url, &base_url, state.tab_id, state.nav_epoch);
        let fetch = state.fetch_source.clone();
        let cancel = state
            .nav_cancel
            .clone()
            .unwrap_or_default();
        let epoch = state.nav_epoch;
        let tx = state.font_outcome_tx.clone();
        tokio::task::spawn(async move {
            let outcome = match fetch.fetch_with(request, cancel).await {
                Ok(response) => {
                    if response.body.len() > 10 * 1024 * 1024 {
                        FontOutcome::Failed { url, epoch }
                    } else {
                        FontOutcome::Loaded {
                            url,
                            epoch,
                            bytes: response.body,
                        }
                    }
                }
                Err(_) => FontOutcome::Failed { url, epoch },
            };
            let _ = tx.send(outcome);
        });
    }
}

/// First `url(...)` in a `@font-face` `src` list. `local(...)`
/// sources are never fetched — the loop looks for `url(` only —
/// and an empty `url()` is skipped.
fn parse_font_src_url(src: &str) -> Option<String> {
    let mut rest = src;
    loop {
        let lower = rest.to_ascii_lowercase();
        let pos = lower.find("url(")?;
        let after = &rest[pos + 4..];
        let end = after.find(')')?;
        let inner = after[..end]
            .trim()
            .trim_matches(|c| c == '"' || c == '\'');
        if !inner.is_empty() {
            return Some(inner.to_string());
        }
        rest = &after[end + 1..];
    }
}

/// Resolve a font URL against the document base. Absolute URLs
/// (including `data:`) pass through; relative ones join the base.
fn resolve_font_url(base_url: &str, raw: &str) -> Result<String, url::ParseError> {
    if raw.contains("://") || raw.starts_with("data:") {
        return Ok(raw.to_string());
    }
    let base = url::Url::parse(base_url)?;
    Ok(base.join(raw)?.to_string())
}

/// One `@font-face` outcome: a loaded face waits in
/// `pending_fonts` for the next layout (which rebuilds with it);
/// a failed face leaves the fallback in place, never fatal.
fn handle_font_outcome(state: &mut TabState, outcome: FontOutcome) {
    match outcome {
        FontOutcome::Loaded { url, epoch, bytes } => {
            if epoch != state.nav_epoch {
                return;
            }
            eprintln!("[pluto] font loaded: {url}");
            state.pending_fonts.push(bytes);
            state.render_requested.set(true);
        }
        FontOutcome::Failed { url, epoch } => {
            if epoch != state.nav_epoch {
                return;
            }
            eprintln!("[pluto] font failed: {url}");
        }
    }
}

/// One image URL reported by the load (Stage 11 Rec 1): dedup
/// by URL, then fetch + decode on detached tasks (the outcome
/// arrives on `image_outcome_rx`). The fetch takes the current
/// navigation's cancellation token, so a newer navigation aborts
/// in-flight image fetches along with the document load; the
/// epoch check in the outcome handler is the backstop.
fn handle_image_request(state: &mut TabState, req: ImageRequest) {
    if state.images.request(&req) {
        spawn_image_fetch(state, req);
    }
}

/// Fetch + decode one image on detached tasks, for a request the
/// store accepted (see `handle_image_request`).
fn spawn_image_fetch(state: &mut TabState, req: ImageRequest) {
    let cancel = state
        .nav_cancel
        .clone()
        .unwrap_or_default();
    ImageStore::spawn_fetch(
        state.fetch_source.clone(),
        req,
        state.tab_id,
        state.nav_epoch,
        cancel,
        state.image_outcome_tx.clone(),
    );
}

/// One image load requested by JS (`Image().src` / an `<img>`
/// `src` set from JS — Stage 12 Rec 7 wiring, drained from the
/// host). A fresh URL fetches like any scanner-reported image;
/// an already-decoded URL reports `load` straight away with the
/// cached size; an already-broken URL reports `error`; a URL
/// still in flight reports nothing now — the outcome handler
/// notifies when it lands.
fn handle_js_image_request(state: &mut TabState, req: ImageRequest) {
    if state.images.request(&req) {
        spawn_image_fetch(state, req);
        return;
    }
    if let Some((width, height)) = state.images.ready_size(&req.url) {
        notify_image_targets(state, &req.url, Some((width, height)));
    } else if state.images.is_broken(&req.url) {
        notify_image_targets(state, &req.url, None);
    }
}

// --- Stage 16 (M8 Phase 2, Rec 4/5): media fetches ---

/// Request-id counter for media fetches (separate from the
/// image counter — the broker keys responses by id, and the
/// two pipelines must never collide).
static NEXT_MEDIA_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// One media fetch outcome, from a spawned task to the tab
/// thread: the whole fetched file (buffered-whole media until
/// Rec 12 streaming), or the failure reason.
pub enum MediaOutcome {
    /// A media candidate's bytes (or failure) for `key`.
    Media {
        key: NodeId,
        epoch: u64,
        result: Result<MediaDelivery, String>,
    },
    /// A `<track>` file's bytes (or failure) for `key`.
    Track {
        key: NodeId,
        track_idx: usize,
        epoch: u64,
        result: Result<Vec<u8>, String>,
    },
}

/// A delivered media candidate: the bytes, the response
/// Content-Type, and the tainting decision for this fetch.
pub struct MediaDelivery {
    pub bytes: Vec<u8>,
    pub mime: String,
    /// True when the origin is clean for this element:
    /// same-origin, `data:`, or a CORS-approved cross-origin
    /// fetch. The renderer computes it (see
    /// `media_origin_clean`); `pluto-net` enforces CORS, the
    /// renderer records the outcome.
    pub origin_clean: bool,
    /// True when the fetch was CORS-approved (mode `Cors` +
    /// success). M8 always fetches media `NoCors` (no
    /// `crossorigin` attribute support yet), so this is false
    /// today — the field exists so the record is explicit and
    /// the `crossorigin` work can flip it without changing the
    /// delivery shape.
    pub cors_approved: bool,
}

/// Whether a media URL is origin-clean (Stage 16 Phase 2,
/// Rec 4): `data:` URLs never taint; a URL is clean when its
/// origin equals the document base URL's origin. Cross-origin
/// media taints unless the fetch was CORS-approved — the
/// caller passes that in separately (see `MediaDelivery`).
fn media_origin_clean(base_url: &str, url: &str, cors_approved: bool) -> bool {
    if cors_approved {
        return true;
    }
    if url.starts_with("data:") {
        return true;
    }
    Origin::from_url(url) == Origin::from_url(base_url)
}

/// Build a media subresource request (mirrors
/// `image_resource_request` / `subresource_request`): the
/// destination is `Video` / `Audio` by element, mode is
/// `NoCors` (the element has no `crossorigin` attribute in
/// M8), priority Medium.
fn media_resource_request(
    url: &str,
    destination: Destination,
    base_url: &str,
    tab_id: u64,
    epoch: u64,
) -> ResourceRequest {
    let origin = origin_of(base_url);
    ResourceRequest {
        request_id: NEXT_MEDIA_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
        tab_id,
        epoch,
        url: url.to_string(),
        method: "GET".to_string(),
        headers: Vec::new(),
        is_imageset: false,
        destination,
        priority: Priority::Medium,
        mode: RequestMode::NoCors,
        credentials: CredentialsMode::SameOrigin,
        origin: origin.clone(),
        top_level_site: origin,
        upgrade_insecure_requests: false,
        integrity: None,
        body: None,
    }
}

/// Handle one JS media-candidate request: a `data:` URL is
/// decoded inline (the broker never sees it — same belt as
/// the preload scanner's `data:` skip); anything else spawns
/// a broker fetch whose outcome lands on `media_outcome_rx`.
fn handle_js_media_request(state: &mut TabState, req: MediaRequest) {
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    if let Some(bytes) = decode_data_url(&req.url) {
        js.deliver_media(req.key, bytes, req.mime_hint, true, true);
        return;
    }
    let is_video = state
        .host
        .lock()
        .unwrap()
        .tag_name(req.key)
        .as_deref()
        == Some("video");
    let destination = if is_video {
        Destination::Video
    } else {
        Destination::Audio
    };
    let fetch = state.fetch_source.clone();
    let request = media_resource_request(
        &req.url,
        destination,
        &req.base_url,
        state.tab_id,
        state.nav_epoch,
    );
    let cancel = state
        .nav_cancel
        .clone()
        .unwrap_or_default();
    let outcome_tx = state.media_outcome_tx.clone();
    let key = req.key;
    let epoch = state.nav_epoch;
    let base_url = req.base_url.clone();
    tokio::task::spawn(async move {
        let result = match fetch.fetch_with(request, cancel).await {
            Ok(response) => {
                let mime = response
                    .content_type()
                    .unwrap_or("")
                    .to_string();
                // Mode is always NoCors in M8 (see
                // `media_resource_request`), so a cross-origin
                // success is NOT CORS-approved — the clean bit
                // comes from same-origin / data: only.
                let cors_approved = false;
                let origin_clean = media_origin_clean(&base_url, &response.final_url.to_string(), cors_approved);
                Ok(MediaDelivery {
                    bytes: response.body,
                    mime,
                    origin_clean,
                    cors_approved,
                })
            }
            Err(e) => Err(format!("{e}")),
        };
        let _ = outcome_tx.send(MediaOutcome::Media {
            key,
            epoch,
            result,
        });
    });
}

/// Handle one JS `<track>` request: same shape as media
/// (inline `data:` decode, else a broker fetch with
/// `Destination::Video` — tracks ride the media pipeline;
/// the destination only gates CSP media-src, which is the
/// right gate for a track file).
fn handle_js_track_request(state: &mut TabState, req: TrackRequest) {
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    if let Some(bytes) = decode_data_url(&req.url) {
        js.deliver_track(req.key, req.track_idx, bytes);
        return;
    }
    let fetch = state.fetch_source.clone();
    let request = media_resource_request(
        &req.url,
        Destination::Video,
        &req.base_url,
        state.tab_id,
        state.nav_epoch,
    );
    let cancel = state
        .nav_cancel
        .clone()
        .unwrap_or_default();
    let outcome_tx = state.media_outcome_tx.clone();
    let key = req.key;
    let track_idx = req.track_idx;
    let epoch = state.nav_epoch;
    tokio::task::spawn(async move {
        let result = match fetch.fetch_with(request, cancel).await {
            Ok(response) => Ok(response.body),
            Err(e) => Err(format!("{e}")),
        };
        let _ = outcome_tx.send(MediaOutcome::Track {
            key,
            track_idx,
            epoch,
            result,
        });
    });
}

/// Apply one media fetch outcome: stale epochs are dropped
/// (a navigation since the fetch started), the rest go to
/// the element's session via `TabJs`.
fn handle_media_outcome(state: &mut TabState, outcome: MediaOutcome) {
    let epoch = match &outcome {
        MediaOutcome::Media { epoch, .. } => *epoch,
        MediaOutcome::Track { epoch, .. } => *epoch,
    };
    if epoch != state.nav_epoch {
        return;
    }
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    match outcome {
        MediaOutcome::Media { key, result, .. } => match result {
            Ok(delivery) => js.deliver_media(
                key,
                delivery.bytes,
                delivery.mime,
                delivery.origin_clean,
                delivery.cors_approved,
            ),
            Err(reason) => js.notify_media_failed(key, reason),
        },
        MediaOutcome::Track {
            key,
            track_idx,
            result,
            ..
        } => match result {
            Ok(bytes) => js.deliver_track(key, track_idx, bytes),
            Err(_) => js.notify_track_failed(key, track_idx),
        },
    }
}

/// Advance every media session one tick and send presented
/// frames to the shell (Stage 16, M8 Phase 2, Rec 4/6). The
/// session's audio clock decides what presents; this only
/// moves the frames. A size/liveness change marks the media
/// map dirty (layout re-runs on the next dirty check); a
/// plain frame never does.
async fn drive_media(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
) {
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    let presented = js.media_tick();
    let sizes = js.media_intrinsic_sizes();
    if sizes != state.last_media_sizes {
        state.last_media_sizes = sizes;
        state.media_dirty = true;
    }
    for (_key, frame) in presented {
        if !emit_tab(
            events,
            RendererToBrowser::VideoFrame {
                epoch: state.nav_epoch,
                frame,
            },
        )
        .await
        {
            break;
        }
    }
}

/// Decode a `data:` URL to bytes (`data:[mime][;base64],...`),
/// or `None` when the URL is not a `data:` URL or is
/// malformed. Media and track `data:` sources decode inline
/// here — the broker never sees them.
fn decode_data_url(url: &str) -> Option<Vec<u8>> {
    let rest = url.trim_start().strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if meta.contains(";base64") {
        base64_decode(data)
    } else {
        percent_decode(data)
    }
}

/// Minimal base64 decode (no padding-sensitive general
/// decoder; invalid input → None).
fn base64_decode(data: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut acc: u32 = 0;
    let mut nbits = 0;
    for c in data.chars() {
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            '=' => break,
            _ => return None,
        };
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Some(out)
}

/// Percent-decode a data-URL body (`%XX` → byte).
fn percent_decode(data: &str) -> Option<Vec<u8>> {
    let bytes = data.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return None;
            }
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
            let v = u8::from_str_radix(hex, 16).ok()?;
            out.push(v);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Some(out)
}

/// Whether an image URL is origin-clean for canvas tainting
/// (Stage 15, M7 Phase 3, Rec 9): `data:` URLs never taint,
/// and a URL is clean when its origin equals the document base
/// URL's origin. Cross-origin images taint — the pipeline
/// records no CORS approval for `<img>`, so there is no clean
/// cross-origin case.
fn image_origin_clean(base_url: &str, url: &str) -> bool {
    if url.starts_with("data:") {
        return true;
    }
    Origin::from_url(url) == Origin::from_url(base_url)
}

/// Report one image outcome to JS (Stage 12 Rec 7): every `<img>`
/// whose current `src` resolves to `url` gets a `load` (with the
/// intrinsic size) or `error` event, via `TabJs`. A favicon never
/// reaches here (it has no element). Elements whose `src` moved
/// on since the fetch started do not match, so they hear nothing
/// about the stale load.
///
/// A decoded load also feeds the pixels to the JS host (Stage 15,
/// M7 Phase 3, Rec 8): a later `drawImage(img)` then has the
/// bitmap synchronously, with the origin-clean flag Rec 9
/// requires. The host keeps its own copy of the bytes (see
/// `ImageStore::ready_pixels` for the bound).
fn notify_image_targets(state: &mut TabState, url: &str, outcome: Option<(u32, u32)>) {
    let targets = state.host.lock().unwrap().image_targets(url);
    if targets.is_empty() {
        return;
    }
    if outcome.is_some() {
        let base_url = state.host.lock().unwrap().base_url().to_string();
        let clean = image_origin_clean(&base_url, url);
        if let Some(pixels) = state.images.ready_pixels(url) {
            let mut host = state.host.lock().unwrap();
            for target in &targets {
                host.set_image_pixels(
                    *target,
                    pixels.width,
                    pixels.height,
                    pixels.rgba.clone(),
                    clean,
                );
            }
        }
    }
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    for target in targets {
        match outcome {
            Some((width, height)) => js.notify_image_loaded(target, width, height),
            None => js.notify_image_failed(target),
        }
    }
}

/// One image fetch + decode outcome landed on the tab thread
/// (Stage 11 Rec 1/7/11). Stale epochs are dropped before
/// applying — a newer navigation superseded the page that wanted
/// the image — and a stale favicon attempt is un-marked so a
/// later visit retries it. A decoded image is inserted into the
/// cache and sent to the shell out-of-band in its own
/// `ResourceReady` frame, but only when the encoded frame fits
/// the IPC cap (oversize → broken placeholder; the layout rebuild
/// still happens, so the alt text shows). A favicon outcome also
/// emits `FaviconChanged`, after its `ResourceReady` (per the
/// IPC docs). The `images_dirty` flag the apply sets drives the
/// next tick's full rebuild — this handler never lays out.
async fn handle_image_outcome(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    outcome: ImageOutcome,
) {
    let url = match &outcome {
        ImageOutcome::Decoded { url, .. } | ImageOutcome::Failed { url, .. } => url.clone(),
    };
    let epoch = match &outcome {
        ImageOutcome::Decoded { epoch, .. } | ImageOutcome::Failed { epoch, .. } => *epoch,
    };
    if epoch != state.nav_epoch {
        state.images.favicon_attempted.remove(&origin_of(&url));
        return;
    }
    let is_favicon = state.favicon_url.as_deref() == Some(url.as_str());
    let applied = state.images.apply_outcome(outcome, is_favicon);
    // Stage 12 Rec 7: report the outcome to JS — a decoded image
    // fires `load` on every `<img>` whose current `src` is this
    // URL, a failure fires `error`. Favicons have no element, so
    // they never notify (`FaviconChanged` below is their report).
    // The listeners run on the next tick's `drive_js`, not here.
    if !is_favicon {
        match &applied {
            Some(a) => notify_image_targets(state, &url, Some((a.width, a.height))),
            None => notify_image_targets(state, &url, None),
        }
    }
    match applied {
        Some(applied) => {
            if let Some(resource) = state.images.cache.get(applied.id).cloned() {
                let fits = match &resource {
                    Resource::Image { rgba, .. } => {
                        ImageStore::resource_fits_frame(rgba.len())
                    }
                };
                if fits {
                    if !emit_tab(
                        events,
                        RendererToBrowser::ResourceReady {
                            epoch,
                            resource_id: applied.id,
                            resource,
                        },
                    )
                    .await
                    {
                        return;
                    }
                } else {
                    eprintln!(
                        "[pluto] image {url} exceeds the IPC frame cap — broken placeholder"
                    );
                }
            }
            if is_favicon {
                state.favicon_resource = Some(applied.id);
                let _ = emit_tab(
                    events,
                    RendererToBrowser::FaviconChanged {
                        resource_id: Some(applied.id),
                    },
                )
                .await;
            }
        }
        None => {
            if is_favicon {
                state.images.favicon_missing.insert(origin_of(&url));
                let _ = emit_tab(
                    events,
                    RendererToBrowser::FaviconChanged { resource_id: None },
                )
                .await;
            }
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
async fn emit(
    tab_id: u64,
    events: &mut mpsc::Sender<crate::Message>,
    event: RendererToBrowser,
) -> bool {
    events
        .send(crate::Message::RendererEvent(tab_id, event))
        .await
        .is_ok()
}

/// Insert fresh raster Blobs (Stage 15, M7 Phase 2, Rec 5)
/// into the tab's image cache and announce each with
/// `ResourceReady` BEFORE the display list that names them
/// submits — the shell validates the list against its per-tab
/// table, so a Blob id without pixels would fail validation and
/// drop the frame. Entries the cache already holds (same id,
/// same pixels — the deterministic ids make a scroll re-collect
/// name the same id) are skipped: the shell already has them.
/// An entry under the same id with different pixels (a
/// re-raster after a content change or resize) replaces the
/// cached pixels and re-announces, like the animated-frame
/// precedent above. Oversize-for-frame resources are skipped —
/// they cannot cross IPC (the raster cap in `pluto_paint::svg`
/// is set so this never happens in practice; the check is the
/// belt to that suspender). Returns false when the shell is gone.
/// Drain the host's canvas state for one collect (Stage 15, M7
/// Phase 3, Rec 8): every live canvas element, the bitmaps that
/// changed since the last announcement (as `Resource::Image`s),
/// and the svg-image resource map from the last image map. A
/// bitmap whose hash matches the last announced one is not
/// re-sent — the shell already has those pixels, so an unchanged
/// canvas costs nothing per frame. The drain clears the hosts'
/// dirty flags, so a second call in the same tick finds nothing
/// new.
/// What `collect_canvas_state` returns: live canvas keys, new/changed
/// canvas bitmaps, and the svg-image resource map.
type CanvasState = (
    Vec<NodeId>,
    Vec<(NodeId, Resource)>,
    HashMap<NodeId, ResourceId>,
);

fn collect_canvas_state(
    state: &mut TabState,
) -> CanvasState {
    let (live, dirty) = {
        let mut host = state.host.lock().unwrap();
        (host.canvas_live_keys(), host.canvas_take_dirty())
    };
    let mut fresh = Vec::new();
    for (key, width, height, rgba) in dirty {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        rgba.hash(&mut hasher);
        let hash = hasher.finish();
        if state.last_canvas_announced.get(&key).copied() != Some(hash) {
            state.last_canvas_announced.insert(key, hash);
            fresh.push((
                key,
                Resource::Image {
                    width,
                    height,
                    rgba,
                },
            ));
        }
    }
    let svg_images: HashMap<NodeId, ResourceId> = state
        .last_image_map
        .by_node
        .iter()
        .map(|(id, info)| (*id, ResourceId(info.resource)))
        .collect();
    (live, fresh, svg_images)
}

async fn announce_blobs(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    blobs: Vec<(ResourceId, Resource)>,
) -> bool {
    for (id, resource) in blobs {
        if state.images.cache.get(id) == Some(&resource) {
            continue;
        }
        let fits = match &resource {
            Resource::Image { rgba, .. } => ImageStore::resource_fits_frame(rgba.len()),
        };
        if !fits {
            continue;
        }
        if !state.images.cache.insert_with_id(id, resource.clone()) {
            continue;
        }
        if !emit_tab(
            events,
            RendererToBrowser::ResourceReady {
                epoch: state.nav_epoch,
                resource_id: id,
                resource,
            },
        )
        .await
        {
            return false;
        }
    }
    true
}

/// Submit a display list to the shell, stamping the next epoch.
/// Stage 11 Rec 1: the resource table travels EMPTY — decoded
/// resources crossed IPC out-of-band in `ResourceReady` frames,
/// and the shell validates the list against the per-tab table it
/// accumulated from them.
async fn submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    display: Arc<DisplayList>,
) -> bool {
    state.epoch += 1;
    // Perf (Stage 8 Rec 5): the postcard encode is the IPC cost.
    let started = std::time::Instant::now();
    let bytes = display.to_bytes();
    perf::record_span(
        perf::SpanKind::Ipc,
        started.elapsed().as_secs_f32() * 1000.0,
    );
    emit_tab(
        events,
        RendererToBrowser::SubmitDisplayList {
            epoch: state.epoch,
            bytes,
            resources: ResourceTable::new(),
        },
    )
    .await
}

/// Lay out at the tab's viewport on the layout-actor thread and
/// submit the result. `document: Some` hands the actor a new
/// document (a navigation / error page / a JS-mutated document);
/// `None` re-lays out the retained one (a resize). `incremental`
/// tells the actor the document is a snapshot of the same live
/// `Dom` the retained tree was built from (a dirty tick or a
/// resize), so it may re-lay out incrementally (Stage 8 Rec 6);
/// false for a fresh navigation, whose node ids are new. A dead
/// actor is reported as a crash; the old page stays visible until
/// the new list lands either way.
///
/// Stage 8 Rec 8 ordering: (1) build the layout tree, (2) run
/// the ResizeObserver gather/broadcast step against it, (3) build
/// the display list from the tree, (4) submit. A ResizeObserver
/// callback that mutates the DOM marks the document dirty again,
/// so the *next* tick re-lays out and submits a fresh list — the
/// callback's write lands one tick later, not in the list being
/// built here (the actor's tree is already built by then; see
/// `run_resize_observer_step`).
/// M6 Phase 3 (Rec 11): fold one layout outcome into the tab's
/// animation state — merge `changed_styles` into the computed-style
/// cache (and the host, for `getComputedStyle`), start the pending
/// transitions (from the currently sampled value when one is already
/// running on that property) and keyframe animations. Runs for every
/// outcome kind, including `CompositingOnly`: an animated transform
/// change reports its pending transitions on that outcome too.
fn process_animation_outcome(
    state: &mut TabState,
    outcome: &RelayoutOutcome,
    now: std::time::Instant,
) {
    for (id, pairs) in &outcome.changed_styles {
        let entry = state.computed_style_cache.entry(*id).or_default();
        for (prop, css) in pairs {
            entry.insert(prop.clone(), css.clone());
        }
    }
    {
        let mut host = state.host.lock().unwrap();
        for (id, pairs) in &outcome.changed_styles {
            host.set_computed_styles(*id, pairs.clone());
        }
    }
    for pt in &outcome.pending_transitions {
        let from = state
            .current_animated
            .get(&pt.id)
            .and_then(|m| m.get(&pt.spec.property))
            .cloned()
            .unwrap_or_else(|| pt.from.clone());
        state
            .animation_engine
            .start_transition(pt.id, pt.spec.clone(), from, pt.to.clone(), now);
    }
    let ctx = CalcCtx {
        font_size: 16.0,
        root_font_size: 16.0,
        viewport: state.viewport,
        pct_basis: None,
    };
    for pa in &outcome.pending_animations {
        let base: HashMap<String, AnimatedValue> = pa.base.iter().cloned().collect();
        state.animation_engine.start_animation(
            pa.id,
            pa.spec.clone(),
            &pa.keyframes,
            base,
            &ctx,
            now,
        );
    }
    {
        let mut host = state.host.lock().unwrap();
        for pa in &outcome.pending_animations {
            host.note_css_animation_started(pa.id, pa.spec.name.clone());
        }
    }
}

/// Advance the animation engine one tick (M6 Phase 3, Rec 11).
/// Sampled values update `current_animated` + the computed-style
/// cache and mark the frame dirty (the next layout applies them as
/// overrides); finished `transitionend` / `animationend` events
/// dispatch into the realm (a dispatch to a detached target is
/// dropped, never fatal). While hidden the engine clock freezes —
/// the tick still runs, but no animation jumps on return.
fn advance_animations(state: &mut TabState, now: std::time::Instant) {
    if state.animation_engine.is_empty() {
        return;
    }
    let visible = state
        .tab_js
        .as_ref()
        .map(|js| js.is_visible())
        .unwrap_or(true);
    if !visible {
        state.animation_engine.pause(now);
        return;
    }
    state.animation_engine.resume(now);
    let outcome = state.animation_engine.advance(now);
    if outcome.changed.is_empty()
        && outcome.finished_events.is_empty()
        && outcome.waapi_finished.is_empty()
    {
        return;
    }
    for (id, vals) in &outcome.changed {
        let entry = state.current_animated.entry(*id).or_default();
        for (prop, val) in vals {
            entry.insert(prop.clone(), val.clone());
        }
        let cache = state.computed_style_cache.entry(*id).or_default();
        for (prop, val) in vals {
            cache.insert(prop.clone(), val.to_css());
        }
    }
    {
        let mut host = state.host.lock().unwrap();
        for (id, vals) in &outcome.changed {
            let pairs: Vec<(String, String)> =
                vals.iter().map(|(p, v)| (p.clone(), v.to_css())).collect();
            host.set_computed_styles(*id, pairs);
        }
        for (id, wid) in &outcome.waapi_finished {
            host.mark_waapi_finished(*id, *wid);
        }
    }
    if let Some(js) = state.tab_js.as_mut() {
        for (id, ev) in &outcome.finished_events {
            js.dispatch_event(*id, ev, false);
        }
    }
    // An `animationend` retires one CSS name: refresh the host's
    // list from the engine (the finished name drops out, so
    // `getAnimations` stops reporting it).
    {
        let mut host = state.host.lock().unwrap();
        for (id, ev) in &outcome.finished_events {
            if *ev == "animationend" {
                let names = state.animation_engine.running_css_animations(*id);
                host.set_css_animations(*id, names);
            }
        }
    }
    if !outcome.changed.is_empty() {
        state.render_requested.set(true);
    }
}

async fn layout_and_submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    document: Option<Document>,
    incremental: bool,
    started: std::time::Instant,
) -> bool {
    // Stage 10 Rec 3: project the committed-URL set onto the
    // document's links before the state is cloned into the
    // layout job, so `:visited` is fresh for every pass.
    refresh_visited(state);
    // Stage 11 Rec 1/2, wired for the incremental path by Stage 12
    // Rec 5: an image arrival no longer forces a FULL rebuild with
    // a fresh image map — the fresh map rides the layout job with
    // its dirty flag, and the actor hands it to the incremental
    // path, which treats the changed `<img>` nodes as damage. The
    // map is built from the document this pass lays out (or the
    // retained one on a resize, where `document` is `None`) whenever
    // the actor will need it — an image arrival, or any
    // non-incremental pass (a fresh tree builds with the map, never
    // empty). The dirty flag is consumed here: an image that lands
    // mid-layout sets it again, and the next tick rebuilds.
    let images_dirty = state.images.images_dirty;
    let media_dirty = state.media_dirty;
    let images = if images_dirty || media_dirty || !incremental {
        state.images.images_dirty = false;
        state.media_dirty = false;
        let doc = document.as_ref().or(state.document.as_ref());
        let mut images = match doc {
            Some(doc) => state.images.build_image_map(
                doc.dom(),
                doc.sheet(),
                doc.base_url(),
                state.viewport.0,
                1.0,
            ),
            None => ImageMap::new(),
        };
        // Stage 16 (M8 Phase 2, Rec 5): the decoded media sizes
        // ride the same map layout reads — the intrinsic size
        // once HAVE_METADATA, and whether a frame is presentable.
        if let Some(js) = state.tab_js.as_ref() {
            for (key, (w, h, live)) in js.media_intrinsic_sizes() {
                images.videos.insert(
                    key,
                    VideoInfo {
                        width: w,
                        height: h,
                        has_frame: live,
                    },
                );
            }
        }
        images
    } else {
        ImageMap::new()
    };
    if images_dirty || media_dirty || !incremental {
        state.last_image_map = images.clone();
    }
    // Stage 11 Rec 8: read the title BEFORE the document moves
    // into the layout job — it is compared after the submit, and
    // `TitleChanged` fires at most once per submit, only when
    // the title actually changed (a title-setting loop cannot
    // spam IPC: one coalesced event per frame at most).
    let title = document
        .as_ref()
        .or(state.document.as_ref())
        .map(|doc| document_title(doc.dom()))
        .unwrap_or_default();
    // Perf (Stage 8 Rec 5): the whole build — layout tree,
    // ResizeObserver step, IntersectionObserver step, display-list
    // collect — is the worker build span. The submit's postcard
    // encode is measured separately, in `submit`. `started` is
    // the caller's frame clock (the tick start on the tick path),
    // so the recorded total is tick-to-submit, not just the build.
    // (1) Layout tree on the actor (no display list yet).
    let layout_started = std::time::Instant::now();
    let (tree_tx, tree_rx) = oneshot::channel();
    let job = LayoutJob::LayoutTree {
        document: Box::new(document),
        viewport: state.viewport,
        state: state.element_state.clone(),
        offsets: state.scroll_offsets.clone(),
        incremental,
        images,
        images_dirty,
        theme: state.theme,
        // M6 Phase 3 (Rec 3): the tree mirrors the viewport
        // scroll on every layout, so sticky offsets stay current
        // even when the scroll changed without a scroll frame
        // (resize / clamp).
        viewport_scroll: state.viewport_scroll,
        animation_overrides: state
            .current_animated
            .iter()
            .map(|(id, vals)| {
                (
                    *id,
                    vals.iter()
                        .map(|(p, v)| (p.clone(), v.clone()))
                        .collect(),
                )
            })
            .collect(),
        fonts: std::mem::take(&mut state.pending_fonts),
        reply: tree_tx,
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
    // Stage 12 Rec 3: the actor reports which path the layout
    // took; the frame sample carries it into the histogram.
    let outcome = match tree_rx.await {
        Ok(outcome) => outcome,
        Err(_) => {
            return emit_tab(
                events,
                RendererToBrowser::RendererCrashed {
                    reason: "layout actor reply lost".to_string(),
                },
            )
            .await;
        }
    };
    let layout_kind = outcome.kind;
    state.has_document = true;
    // M6 Phase 3 (Rec 11): fold the outcome's animation work
    // into the engine + computed-style cache before any branch —
    // a `CompositingOnly` outcome carries pending transitions too
    // (an animated transform change reports them there).
    process_animation_outcome(state, &outcome, started);
    // M6 Phase 2 (Rec 9): a transform/opacity-only change takes
    // the cheap path — no observers, no list rebuild, no submit.
    // Ask the actor for the changed elements' new compositing
    // state (the dirty elements plus their descendants, whose
    // page transforms move with them), send it as a
    // `CompositingUpdate` against the active list's epoch, patch
    // the worker's cached list the same way so a later
    // caret-blink / GIF re-submit does not clobber the update,
    // and record the frame. The observer steps are skipped: a
    // transform does not change sizes, and intersection geometry
    // is re-measured on the next full frame.
    if layout_kind == RelayoutKind::CompositingOnly {
        let (state_tx, state_rx) = oneshot::channel();
        if state
            .layout_tx
            .send(LayoutJob::CompositedState {
                ids: outcome.composited,
                reply: state_tx,
            })
            .is_err()
        {
            return emit_tab(
                events,
                RendererToBrowser::RendererCrashed {
                    reason: "layout actor is gone".to_string(),
                },
            )
            .await;
        }
        let updates = match state_rx.await {
            Ok(updates) => updates,
            Err(_) => {
                return emit_tab(
                    events,
                    RendererToBrowser::RendererCrashed {
                        reason: "layout actor reply lost".to_string(),
                    },
                )
                .await;
            }
        };
        state.host.lock().unwrap().clear_dirty();
        if let Some(display) = state.last_display.as_mut() {
            Arc::make_mut(display).apply_compositing_update(&updates);
        }
        let sent = emit_tab(
            events,
            RendererToBrowser::CompositingUpdate {
                epoch: state.epoch,
                updates,
            },
        )
        .await;
        state.frame_times.record(FrameSample {
            total_ms: started.elapsed().as_secs_f32() * 1000.0,
            layout_ms: layout_started.elapsed().as_secs_f32() * 1000.0,
            collect_ms: 0.0,
            submit_ms: 0.0,
            layout_kind,
        });
        return sent;
    }
    // Follow-up 2 (Stage 10): the viewport scroll mirror goes
    // stale when a resize shrinks the page — re-clamp it to the
    // new page maximum now that the tree has landed, and send the
    // correction to the shell so the two stay in sync (the shell's
    // compositor re-clamps on submit too; this keeps the mirror
    // exact for the IntersectionObserver step below).
    let (max_x, max_y) = page_max_scroll(state).await;
    let clamped = (
        state.viewport_scroll.0.clamp(0.0, max_x),
        state.viewport_scroll.1.clamp(0.0, max_y),
    );
    if clamped != state.viewport_scroll {
        let (dx, dy) = (
            clamped.0 - state.viewport_scroll.0,
            clamped.1 - state.viewport_scroll.1,
        );
        state.viewport_scroll = clamped;
        if !emit_tab(
            events,
            RendererToBrowser::ScrollViewport { dx, dy },
        )
        .await
        {
            return false;
        }
        // M6 Phase 3 (Rec 3): the clamp moved the viewport
        // scroll without a scroll frame — sync the actor's tree
        // now (the list is discarded; the collect step below
        // rebuilds it from the synced tree).
        let (reply_tx, reply_rx) = oneshot::channel();
        if state
            .layout_tx
            .send(LayoutJob::SetViewportScroll {
                x: clamped.0,
                y: clamped.1,
                // The list is discarded (the collect step below
                // rebuilds it), so the canvas drain stays with
                // the collect — passing live keys here would only
                // name Blobs in a list nobody submits.
                canvas_live: Vec::new(),
                canvas_new: Vec::new(),
                svg_images: HashMap::new(),
                reply: reply_tx,
            })
            .is_ok()
        {
            // The list is discarded (the collect step below
            // rebuilds it from the synced tree) — but a fresh
            // Blob in the reply must still be announced, or the
            // re-collect names an id the shell never received.
            if let Ok(Some((_, _, blobs))) = reply_rx.await
                && !announce_blobs(state, events, blobs).await
            {
                return false;
            }
        }
    }
    // Stage 10 Rec 7: the layout phase ends when the tree reply
    // lands; the collect phase runs from here to the list.
    let layout_ms = layout_started.elapsed().as_secs_f32() * 1000.0;
    let collect_started = std::time::Instant::now();
    // (2) ResizeObserver step, after layout, before the list.
    run_resize_observer_step(state).await;
    // (2b) IntersectionObserver step, after layout, before the
    // list: the geometry is computed here, but the callbacks are a
    // queued task — they run after the submit, on a later turn.
    run_intersection_observer_step(state).await;
    // (3) Display list from the retained tree. The canvas drain
    // runs first: the list names canvas Blobs, so fresh bitmaps
    // must be announced (below, via `announce_blobs`) before the
    // list that names them is submitted.
    let (canvas_live, canvas_new, svg_images) = collect_canvas_state(state);
    let (list_tx, list_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::CollectDisplayList {
            canvas_live,
            canvas_new,
            svg_images,
            reply: list_tx,
        })
        .is_err()
    {
        return emit_tab(
            events,
            RendererToBrowser::RendererCrashed {
                reason: "layout actor is gone".to_string(),
            },
        )
        .await;
    }
    match list_rx.await {
        Ok((display, blobs)) => {
            // Stage 10 Rec 7: the collect phase ends when the
            // list lands (it covers the observer steps above
            // plus the collect round trip).
            let collect_ms = collect_started.elapsed().as_secs_f32() * 1000.0;
            perf::record_span(
                perf::SpanKind::WorkerBuild,
                started.elapsed().as_secs_f32() * 1000.0,
            );
            state.host.lock().unwrap().clear_dirty();
            // Stage 11 Rec 11: cache the submitted list so an
            // animation frame advance can re-submit it with
            // fresh pixels and no layout at all.
            state.last_display = Some(Arc::clone(&display));
            // Stage 13 Rec 4: the caret + selection paint rides
            // on the submitted list, not the cached one — the
            // blink tick re-derives it from the caret-free base.
            let display = with_caret(state, display);
            state.caret_blink_on = caret_blink_on(state);
            // Stage 15 Rec 5: the fresh raster Blobs travel in
            // their own `ResourceReady` frames before the list
            // that names them — otherwise the shell's validation
            // would reject the submit for a missing resource.
            if !announce_blobs(state, events, blobs).await {
                return false;
            }
            let submit_started = std::time::Instant::now();
            let submitted = submit(state, events, display).await;
            // Stage 11 Rec 8: the title is emitted here, after
            // the submit, coalesced to one event per frame at
            // most — only when it changed since the last submit.
            if submitted && title != state.last_title {
                state.last_title = title.clone();
                if !emit_tab(events, RendererToBrowser::TitleChanged { title }).await {
                    return false;
                }
            }
            // Stage 10 Rec 7: one ring write per submitted
            // frame — total is tick-to-submit (the caller's
            // clock), phases are layout / collect / submit.
            // Stage 12 Rec 3: the kind rides along, so the
            // histogram counts layout paths per frame.
            state.frame_times.record(FrameSample {
                total_ms: started.elapsed().as_secs_f32() * 1000.0,
                layout_ms,
                collect_ms,
                submit_ms: submit_started.elapsed().as_secs_f32() * 1000.0,
                layout_kind,
            });
            submitted
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

/// Stage 8 Rec 8: the ResizeObserver gather/broadcast step, run
/// after layout and before the display list is built. The `TabJs`
/// registry owns the observations; this step only supplies the
/// sizes the layout actor just measured (border + content boxes)
/// and the DOM depths, then lets the loop deliver the callbacks
/// through the JS event loop. No observations → no actor round
/// trip at all. A resize-loop error is logged (the loop already
/// recorded it in the tab's script errors).
async fn run_resize_observer_step(state: &mut TabState) {
    let targets = match state.tab_js.as_mut() {
        Some(js) => js.resize_observation_targets(),
        None => return,
    };
    if targets.is_empty() {
        return;
    }
    let ids: Vec<NodeId> = targets.iter().map(|(_, target, _)| *target).collect();
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ResizeSizes {
            ids,
            reply: reply_tx,
        })
        .is_err()
    {
        return;
    }
    let Ok(rows) = reply_rx.await else {
        return;
    };
    let mut sizes = HashMap::new();
    let mut depths = HashMap::new();
    for (id, size, depth) in rows {
        if let Some(size) = size {
            sizes.insert(id, size);
        }
        depths.insert(id, depth);
    }
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    let outcome = js.run_resize_observers_with_sizes(&sizes, &depths);
    if outcome.loop_error {
        eprintln!("[pluto] ResizeObserver loop completed with undelivered notifications.");
    }
}

/// Stage 8 Rec 10: the IntersectionObserver gather step, run after
/// layout (and after the ResizeObserver step) and before the display
/// list is built. The `TabJs` registry owns the observations; this
/// step only supplies the intersections the layout actor just
/// measured from the retained tree, then lets the registry queue
/// the delivery task — the callbacks run after the submit, on a
/// later turn, never inline here. No observations → no actor round
/// trip at all. Runs on the retained tree, so scrolls can re-run it
/// without a re-layout (see `handle_wheel`).
async fn run_intersection_observer_step(state: &mut TabState) {
    let targets = match state.tab_js.as_mut() {
        Some(js) => js.intersection_observation_targets(),
        None => return,
    };
    if targets.is_empty() {
        return;
    }
    let queries: Vec<observers::IntersectionQuery> = targets
        .iter()
        .map(|(_, target, root, root_margin)| observers::IntersectionQuery {
            target: *target,
            root: *root,
            root_margin: *root_margin,
        })
        .collect();
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::IntersectionGeometry {
            queries,
            viewport_scroll: state.viewport_scroll,
            reply: reply_tx,
        })
        .is_err()
    {
        return;
    }
    let Ok(rows) = reply_rx.await else {
        return;
    };
    let mut geometry = HashMap::new();
    for ((observer_id, target, _, _), row) in targets.iter().zip(rows) {
        if let Some(geo) = row {
            geometry.insert((*observer_id, *target), geo);
        }
    }
    let time_ms = state.tab_started.elapsed().as_secs_f64() * 1000.0;
    let Some(js) = state.tab_js.as_mut() else {
        return;
    };
    js.run_intersection_observers_with_geometry(&geometry, time_ms, state.viewport_scroll);
}

/// Re-layout the live (possibly JS-mutated) document and submit.
/// The document is re-synced from the host's live `Dom` first, so
/// script mutations reach the actor.
/// Stage 11 Rec 11: poll the pending animation-visibility
/// `Geometry` replies (non-blocking — the job replies over a std
/// channel). A reply updates the animated id's visibility: the
/// node's border box, in page coordinates, against the viewport
/// rect `[viewport_scroll, viewport_scroll + viewport]`. A node
/// with no box (unrendered) is not visible. Replies are consumed
/// in place; a still-pending job stays queued for the next tick.
fn poll_animation_geometry(state: &mut TabState) {
    let (vx, vy) = state.viewport_scroll;
    let (vw, vh) = state.viewport;
    let mut i = 0;
    while i < state.animation_geometry_pending.len() {
        let (id, rx) = &state.animation_geometry_pending[i];
        match rx.try_recv() {
            Ok(geometry) => {
                let visible = match geometry {
                    Some(g) => {
                        let x0 = g.x as f32;
                        let y0 = g.y as f32;
                        let x1 = x0 + g.width as f32;
                        let y1 = y0 + g.height as f32;
                        x1 > vx && x0 < vx + vw && y1 > vy && y0 < vy + vh
                    }
                    None => false,
                };
                if visible {
                    state.animation_visible.insert(*id);
                } else {
                    state.animation_visible.remove(id);
                }
                state.animation_geometry_pending.remove(i);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                i += 1;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // The actor is gone (or never answered): treat
                // the animation as visible rather than freezing
                // it forever on a dead reply.
                state.animation_visible.insert(*id);
                state.animation_geometry_pending.remove(i);
            }
        }
    }
}

/// Stage 11 Rec 11: ask the layout actor for the geometry of
/// every animated node in the last layout pass, so the advance
/// step can pause off-screen animations. One `Geometry` job per
/// node, every ~30 ticks (~0.5 s) — replies land on
/// `animation_geometry_pending` and are polled by the tick.
/// Nodes whose check is already pending are not re-asked until
/// their reply lands.
fn check_animation_visibility(state: &mut TabState) {
    if !state.has_document {
        return;
    }
    let pending: HashSet<ResourceId> = state
        .animation_geometry_pending
        .iter()
        .map(|(id, _)| *id)
        .collect();
    let mut animated_nodes: Vec<(NodeId, ResourceId)> = Vec::new();
    for (&node, info) in &state.last_image_map.by_node {
        let id = ResourceId(info.resource);
        if state.images.cache.is_animated(id) && !pending.contains(&id) {
            animated_nodes.push((node, id));
        }
    }
    for (node, id) in animated_nodes {
        let (tx, rx) = std::sync::mpsc::channel();
        if state
            .layout_tx
            .send(LayoutJob::Geometry { id: node, reply: tx })
            .is_err()
        {
            return;
        }
        state.animation_geometry_pending.push((id, rx));
    }
}

async fn relayout_and_submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    started: std::time::Instant,
) {
    if !state.has_document {
        return;
    }
    let document = state.host.lock().unwrap().current_document();
    state.document = Some(document.clone());
    // Same-lineage snapshot of the live Dom: the actor may lay
    // out incrementally on its retained tree (Stage 8 Rec 6).
    layout_and_submit(state, events, Some(document), true, started).await;
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

/// Input handling (Stage 8 Rec 9, InputEvent v3): pointer
/// move/down/up (hover + active restyle, click synthesis on a
/// left release over its press target), wheel (nested containers
/// first, viewport remainder last), and keys (Tab focus movement,
/// Enter/Space activation, scroll keys).
async fn handle_input(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    event: InputEvent,
) {
    match event {
        InputEvent::Pointer {
            kind,
            x,
            y,
            button,
            ..
        } => match kind {
            PointerKind::Move => handle_pointer_move(state, events, x, y).await,
            PointerKind::Down => handle_pointer_down(state, events, x, y, button).await,
            PointerKind::Up => handle_pointer_up(state, events, x, y, button).await,
        },
        InputEvent::Wheel {
            x,
            y,
            dx,
            dy,
            delta_mode,
            phase,
            ..
        } => handle_wheel(state, events, x, y, dx, dy, delta_mode, phase).await,
        InputEvent::Key {
            key,
            pressed,
            modifiers,
            ..
        } => handle_key(state, events, &key, pressed, modifiers).await,
    }
}

/// The `:visited` key for a URL (Stage 10 Rec 3): the URL with
/// its `#fragment` stripped. Browsers ignore the fragment for
/// `:visited` — visiting `page#top` marks `page#bottom` visited
/// too — so both the recorded commits and the link lookups go
/// through this key.
fn visited_key(url: &str) -> &str {
    url.split('#').next().unwrap_or(url)
}

/// The link ids in `doc` whose resolved `href` is in
/// `visited_urls` (Stage 10 Rec 3): the worker-side projection
/// of the committed-URL set onto the document, resolved against
/// the document base URL with the same `find_link_href` the
/// click path uses. This is the `:visited` half of
/// `ElementState` — the matcher only ever sees ids, never URLs.
fn visited_link_ids(doc: &Document, visited_urls: &HashSet<String>) -> HashSet<pluto_html::NodeId> {
    let mut out = HashSet::new();
    if visited_urls.is_empty() {
        return out;
    }
    let dom = doc.dom();
    let mut stack = dom.root().map(|r| vec![r]).unwrap_or_default();
    while let Some(id) = stack.pop() {
        if dom.tag(id) == Some("a")
            && let Some(href) = page::find_link_href(doc, id)
            && visited_urls.contains(visited_key(&href))
        {
            out.insert(id);
        }
        stack.extend(dom.children(id).iter().rev());
    }
    out
}

/// Recompute `state.element_state.visited` from the current
/// document and the session's committed-URL set (Stage 10 Rec 3).
/// Called at the top of `layout_and_submit` — the choke point
/// for every layout pass (navigation, resize, dirty tick,
/// relayout) — so a Back/Forward revisit, which re-lays out the
/// old page after later URLs were committed, picks up the new
/// visited links too.
fn refresh_visited(state: &mut TabState) {
    if let Some(doc) = state.document.as_ref() {
        state.element_state.visited = visited_link_ids(doc, &state.visited_urls);
    }
}

/// The hover chain for a hit target (Stage 8 Rec 9): the element
/// plus its ancestors, from the tab's synced document. Browsers
/// hover ancestors too — hovering text inside a button highlights
/// the button — so the chain, not just the hit element, is what
/// `:hover` restyles. Falls back to the target alone when there
/// is no document yet.
fn hover_chain(state: &TabState, target: NodeId) -> Vec<NodeId> {
    let mut chain = vec![target];
    if let Some(doc) = state.document.as_ref() {
        let dom = doc.dom();
        let mut cur = dom.get(target).and_then(|n| n.parent);
        while let Some(parent) = cur {
            chain.push(parent);
            cur = dom.get(parent).and_then(|n| n.parent);
        }
    }
    chain
}

/// Expand a restyle set by the sheet's pseudo-class reach (Stage 8
/// Rec 9): a `:hover`/`:active`/`:focus` change may restyle
/// descendants and siblings of the changed elements (per the
/// rules' combinators) and ancestors (`:focus-within`). The
/// changed set itself is always included.
fn restyle_dirty_set(
    state: &TabState,
    changed: &[NodeId],
    reach: pluto_css::PseudoReach,
) -> Vec<NodeId> {
    let mut dirty: HashSet<NodeId> = changed.iter().copied().collect();
    if (reach.descendants || reach.siblings || reach.ancestors)
        && let Some(doc) = state.document.as_ref() {
            let dom = doc.dom();
            for &id in changed {
                if reach.descendants {
                    collect_descendants(dom, id, &mut dirty);
                }
                if reach.siblings
                    && let Some(parent) = dom.get(id).and_then(|n| n.parent) {
                        for &sib in dom.children(parent) {
                            dirty.insert(sib);
                        }
                    }
                if reach.ancestors {
                    let mut cur = dom.get(id).and_then(|n| n.parent);
                    while let Some(parent) = cur {
                        dirty.insert(parent);
                        cur = dom.get(parent).and_then(|n| n.parent);
                    }
                }
            }
        }
    // Stage 10 Rec 4: `:has()` anchors are ancestors of the changed
    // element, so they belong in this same expansion. The re-check
    // is still stubbed (returns no nodes — `:has()` matching is not
    // implemented), so this changes nothing today; the gate and the
    // TODO live in `has_anchor_recheck` below.
    for id in has_anchor_recheck(state, changed) {
        dirty.insert(id);
    }
    dirty.into_iter().collect()
}

/// The `:has()` anchor re-check for a restyle (Stage 10 Rec 4,
/// bookkeeping only). Gate first, walk second: the sheet's
/// `:has()` inventory is asked whether any feature of the changed
/// nodes appears in a `:has()` argument, and only then would the
/// ancestors be walked re-checking anchors.
///
/// TODO(Stage 10 Rec 4): the walk itself is NOT IMPLEMENTED.
/// `HasInvalidationIndex::recheck_anchors` (in `pluto-css`) is the
/// stub that will return the candidate rules' anchors to mark
/// dirty; until `:has()` matching lands it returns empty, so this
/// function returns no nodes and `:has()` still never matches —
/// behaviour identical to before this bookkeeping existed. Do not
/// "just wire up matching" here: a naive re-run of
/// `selector_matches` on every ancestor after every change is the
/// quadratic trap the Rec 4 design names.
fn has_anchor_recheck(state: &TabState, changed: &[NodeId]) -> Vec<NodeId> {
    let Some(doc) = state.document.as_ref() else {
        return Vec::new();
    };
    let index = &doc.sheet().pseudo_class_index.has;
    if index.is_empty() {
        return Vec::new();
    }
    let features: Vec<pluto_css::HasArgFeature> = changed
        .iter()
        .flat_map(|&id| has_features_for(state, id))
        .collect();
    if !index.could_affect_any(&features) {
        return Vec::new();
    }
    // Gate passed — a `:has()` argument mentions a changed feature.
    // The ancestor walk goes here once matching exists; today the
    // stub returns no candidate rules, so no anchor is marked.
    let _ = index.recheck_anchors(&features);
    Vec::new()
}

/// The `:has()`-relevant features of one element (Stage 10 Rec 4):
/// its tag, classes, id, and attribute names from the DOM, plus
/// the dynamic pseudo-class state the shell tracks for it
/// (`:hover` / `:active` / `:focus` / `:visited`). The later
/// `:has()` build reads these through
/// `HasInvalidationIndex::could_affect`; structural features
/// (`:first-child`, `:empty`, ...) are not named here — a
/// sibling/child-structure change alters them without any feature
/// on the changed element itself, so the later build must treat
/// structure changes as affecting any argument that mentions one
/// (the inventory records those names; see `pluto-css`).
fn has_features_for(state: &TabState, id: NodeId) -> Vec<pluto_css::HasArgFeature> {
    let mut out = Vec::new();
    if let Some(doc) = state.document.as_ref() {
        let dom = doc.dom();
        if let Some(node) = dom.get(id) {
            if let Some(tag) = node.tag() {
                out.push(pluto_css::HasArgFeature::type_selector(tag));
            }
            if let Some(class) = node.attr("class") {
                for c in class.split_whitespace() {
                    out.push(pluto_css::HasArgFeature::class(c));
                }
            }
            if let Some(i) = node.attr("id") {
                out.push(pluto_css::HasArgFeature::id(i));
            }
            for (name, _) in &node.attrs {
                out.push(pluto_css::HasArgFeature::attribute(name));
            }
        }
    }
    if state.element_state.hover.contains(&id) {
        out.push(pluto_css::HasArgFeature::pseudo_class("hover"));
    }
    if state.element_state.active.contains(&id) {
        out.push(pluto_css::HasArgFeature::pseudo_class("active"));
    }
    if state.element_state.focus.contains(&id) {
        out.push(pluto_css::HasArgFeature::pseudo_class("focus"));
    }
    if state.element_state.visited.contains(&id) {
        out.push(pluto_css::HasArgFeature::pseudo_class("visited"));
    }
    out
}

/// All descendants of `id` (exclusive), into `out`.
fn collect_descendants(
    dom: &pluto_html::Dom,
    id: NodeId,
    out: &mut HashSet<NodeId>,
) {
    for &child in dom.children(id) {
        out.insert(child);
        collect_descendants(dom, child, out);
    }
}

/// Apply a restyle: sync the host's element-state copy (the
/// forced-flush path reads it), mark the dirty nodes on the live
/// Dom, and re-layout + submit. The incremental path re-resolves
/// styles for the dirty nodes; a failed attempt falls back to a
/// full rebuild. No-op when there is no document yet.
async fn restyle(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    dirty: &[NodeId],
) {
    // Stage 13 Rec 8: form pseudo-classes (`:valid` / `:invalid` /
    // `:required` / `:disabled` / `:checked` / `:read-only`) are
    // derived state — re-derive them from the live `Dom` before
    // every restyle, so a value edit flips them on the next pass.
    sync_form_pseudo_state(state);
    {
        let mut host = state.host.lock().unwrap();
        host.set_element_state(state.element_state.clone());
        host.mark_dirty(dirty);
    }
    relayout_and_submit(state, events, std::time::Instant::now()).await;
}

/// The sheet's pseudo-class reach for `which` (`hover`, `active`,
/// or `focus`): which elements beyond the changed one a restyle
/// may touch. Empty when there is no document yet.
fn pseudo_reach(state: &TabState, which: &str) -> pluto_css::PseudoReach {
    state
        .document
        .as_ref()
        .map(|doc| {
            let index = &doc.sheet().pseudo_class_index;
            match which {
                "active" => index.active_reach,
                "focus" => index.focus_reach,
                _ => index.hover_reach,
            }
        })
        .unwrap_or_default()
}

/// A pointer move: hover restyle. The chain changes only when the
/// hit element (or its ancestors) change — moving within the same
/// element is a no-op. Moving off every element clears the chain.
async fn handle_pointer_move(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    x: f32,
    y: f32,
) {
    let Some(target) = hit_test(state, x, y).await else {
        if !state.hovered.is_empty() {
            let old = std::mem::take(&mut state.hovered);
            state.element_state.hover.clear();
            let dirty = restyle_dirty_set(state, &old, pseudo_reach(state, "hover"));
            restyle(state, events, &dirty).await;
        }
        return;
    };
    let chain = hover_chain(state, target);
    if chain == state.hovered {
        return;
    }
    let old = std::mem::replace(&mut state.hovered, chain.clone());
    state.element_state.hover = chain.iter().copied().collect();
    let changed: Vec<NodeId> = old.iter().chain(&chain).copied().collect();
    let dirty = restyle_dirty_set(state, &changed, pseudo_reach(state, "hover"));
    restyle(state, events, &dirty).await;
}

/// A pointer press: `:active` restyle on the hit chain, and the
/// press target is recorded for click synthesis. The press also
/// hovers the chain — a press without a preceding move still
/// highlights. Only a left press records a click target.
async fn handle_pointer_down(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    x: f32,
    y: f32,
    button: Option<u8>,
) {
    // Stage 16 Phase 1, Rec 7: a trusted pointer press is a user
    // gesture — record it BEFORE hit-testing, so a press on empty
    // page still counts (the shell never forwards chrome clicks
    // here). `pointerdown` itself is not dispatched to JS yet;
    // activation is recorded from the `InputEvent` regardless.
    if let Some(js) = state.tab_js.as_mut() {
        js.note_user_activation();
    }
    let Some(target) = hit_test(state, x, y).await else {
        return;
    };
    // Stage 13 Rec 3: a pointer press moves focus to the hit
    // element BEFORE the click dispatch (the click itself lands
    // on pointer-up) — a click on an input focuses it first.
    if state.focused != Some(target)
        && let Some(doc) = state.document.as_ref()
            && is_focusable(doc.dom(), target) {
                set_focus(state, events, Some(target)).await;
            }
    if button == Some(0) {
        state.pointer_down_target = Some(target);
    }
    let chain = hover_chain(state, target);
    let old_hover = std::mem::replace(&mut state.hovered, chain.clone());
    state.element_state.hover = chain.iter().copied().collect();
    state.active_element = Some(target);
    state.element_state.active = chain.iter().copied().collect();
    let changed: Vec<NodeId> = old_hover.iter().chain(&chain).copied().collect();
    let mut reach = pseudo_reach(state, "hover");
    reach |= pseudo_reach(state, "active");
    let dirty = restyle_dirty_set(state, &changed, reach);
    restyle(state, events, &dirty).await;
}

/// A pointer release: clear `:active`, then synthesise the click
/// — a left release over its press target navigates a link, or
/// dispatches a bubbling click into the JS event loop (counters,
/// accordions). A release over a different element is no click.
async fn handle_pointer_up(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    x: f32,
    y: f32,
    button: Option<u8>,
) {
    if let Some(active) = state.active_element.take() {
        let chain = hover_chain(state, active);
        state.element_state.active.clear();
        let dirty = restyle_dirty_set(state, &chain, pseudo_reach(state, "active"));
        restyle(state, events, &dirty).await;
    }
    if button != Some(0) {
        return;
    }
    let Some(down) = state.pointer_down_target.take() else {
        return;
    };
    let Some(target) = hit_test(state, x, y).await else {
        return;
    };
    if target != down {
        return;
    }
    // A link navigates (new epoch, Stage 5 cancellation); any
    // other hit dispatches a bubbling click into the JS event
    // loop. The tick drives the dispatch and the re-layout that
    // follows it.
    if let Some(url) = click_target(state, x, y).await {
        start_navigation(state, url);
    } else if let Some(js) = state.tab_js.as_mut() {
        // Stage 12 Rec 7: a typed mouse event, so the click
        // listener reads `clientX` / `clientY` / `button`
        // instead of a bare `click`.
        js.dispatch_typed_event(
            target,
            "click",
            true,
            TypedEventKind::Mouse {
                client_x: x as f64,
                client_y: y as f64,
                button: 0,
            },
        );
    }
}

/// The innermost-first scroll chain under the cursor (page
/// coordinates), from the layout actor. Empty when the actor
/// holds no tree.
async fn scroll_chain(state: &TabState, x: f32, y: f32) -> Vec<ScrollChainEntry> {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollChain {
            x,
            y,
            reply: reply_tx,
        })
        .is_err()
    {
        return Vec::new();
    }
    reply_rx.await.unwrap_or_default()
}

/// Maximum viewport (page) scroll offset from the layout actor
/// (Stage 10 Rec 2): `content - viewport` per axis, clamped at
/// zero. `(0, 0)` when the actor is gone or holds no tree — the
/// page scrolls nowhere rather than past its end.
async fn page_max_scroll(state: &TabState) -> (f32, f32) {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::PageMaxScroll { reply: reply_tx })
        .is_err()
    {
        return (0.0, 0.0);
    }
    reply_rx.await.unwrap_or((0.0, 0.0))
}

/// Maximum scroll offset of one scroll container from the layout
/// actor (Stage 10 Rec 2). `(0, 0)` when the actor is gone or the
/// node has no box — the container scrolls nowhere, and the whole
/// delta chains on.
async fn container_max_scroll(state: &TabState, id: NodeId) -> (f32, f32) {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollMax {
            id,
            reply: reply_tx,
        })
        .is_err()
    {
        return (0.0, 0.0);
    }
    reply_rx.await.ok().flatten().unwrap_or((0.0, 0.0))
}

/// Apply a viewport scroll and submit the re-collected display
/// list (M6 Phase 3, Rec 1/3): the actor's tree learns the new
/// scroll (sticky offsets are scroll-derived), and the list
/// submits in the same frame — no layout, no observers, like a
/// container scroll frame (`apply_scrolls`).
async fn apply_viewport_scroll(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
) -> bool {
    let started = std::time::Instant::now();
    let (canvas_live, canvas_new, svg_images) = collect_canvas_state(state);
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::SetViewportScroll {
            x: state.viewport_scroll.0,
            y: state.viewport_scroll.1,
            canvas_live,
            canvas_new,
            svg_images,
            reply: reply_tx,
        })
        .is_err()
    {
        return false;
    }
    match reply_rx.await {
        Ok(Some((display, sticky_moved, blobs))) => {
            // The tree is synced either way — but the list only
            // submits when a sticky box actually moved. Otherwise
            // the list is byte-identical to the cached one and a
            // submit would be a spurious event in the stream (the
            // shell scrolls its cached list on `ScrollViewport`).
            // The fresh Blobs are still announced: the drain
            // already consumed them, and the list names the
            // canvas Blob ids — skipping the announcement would
            // leave the shell without pixels the next list names.
            // (A canvas pixel change does not alter the list
            // bytes — only the resource — so skipping the submit
            // stays correct.)
            if !sticky_moved {
                return announce_blobs(state, events, blobs).await;
            }
            let collect_ms = started.elapsed().as_secs_f32() * 1000.0;
            state.last_display = Some(Arc::clone(&display));
            if !announce_blobs(state, events, blobs).await {
                return false;
            }
            let submit_started = std::time::Instant::now();
            let submitted = submit(state, events, display).await;
            state.frame_times.record(FrameSample {
                total_ms: started.elapsed().as_secs_f32() * 1000.0,
                layout_ms: 0.0,
                collect_ms,
                submit_ms: submit_started.elapsed().as_secs_f32() * 1000.0,
                layout_kind: RelayoutKind::NothingToDo,
            });
            submitted
        }
        _ => false,
    }
}

/// Apply several container scrolls at once and submit the single
/// re-collected display list (Stage 8 Rec 9). False when the actor
/// is gone or holds no tree.
async fn apply_scrolls(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    scrolls: Vec<(NodeId, f32, f32)>,
) -> bool {
    // Stage 10 Rec 7: a scroll frame re-collects without
    // re-laying out, so its cost is the ScrollMany round trip
    // (collect) plus the submit — no layout phase.
    let started = std::time::Instant::now();
    let (canvas_live, canvas_new, svg_images) = collect_canvas_state(state);
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollMany {
            scrolls: scrolls.clone(),
            canvas_live,
            canvas_new,
            svg_images,
            reply: reply_tx,
        })
        .is_err()
    {
        return false;
    }
    match reply_rx.await {
        Ok(Some((display, blobs))) => {
            let collect_ms = started.elapsed().as_secs_f32() * 1000.0;
            for (id, x, y) in scrolls {
                state.scroll_offsets.insert(id, (x, y));
            }
            if !announce_blobs(state, events, blobs).await {
                return false;
            }
            let submit_started = std::time::Instant::now();
            let submitted = submit(state, events, display).await;
            state.frame_times.record(FrameSample {
                total_ms: started.elapsed().as_secs_f32() * 1000.0,
                layout_ms: 0.0,
                collect_ms,
                submit_ms: submit_started.elapsed().as_secs_f32() * 1000.0,
                // No layout ran on a scroll frame (Stage 12 Rec 3).
                layout_kind: RelayoutKind::NothingToDo,
            });
            submitted
        }
        _ => false,
    }
}

/// Apply one axis of a wheel delta to a container offset (Stage 8
/// Rec 9, clamped Stage 10 Rec 2). Returns (new offset, remaining
/// delta). The offset clamps to `[0, max]`: a delta that would
/// carry it past either end is only partly consumed, and the
/// remainder chains onward — unless the container's
/// `overscroll-behavior` stops the chain, in which case the
/// remainder is dropped. A container already at its limit consumes
/// nothing, so the whole delta chains on.
fn consume_axis(offset: f32, delta: f32, max: f32, stops: bool) -> (f32, f32) {
    let next = (offset + delta).clamp(0.0, max);
    let consumed = next - offset;
    let mut remaining = delta - consumed;
    if stops {
        remaining = 0.0;
    }
    (next, remaining)
}

/// Emit a `ScrollViewport` delta and mirror it in
/// `state.viewport_scroll` (Stage 8 Rec 10, clamped Stage 10 Rec
/// 2): every viewport scroll flows through here, clamped to `[0,
/// page max]` per axis — the same `[0, max]` clamp the shell's
/// `compositor.scroll_by` applies — so the worker's mirror of the
/// shell scroll stays exact, and the IntersectionObserver step can
/// use it without a protocol change. Only the consumed part of the
/// delta is emitted: the shell clamps too, but sending the
/// unclamped remainder would desync the mirror from the shell.
/// Callers re-run the IntersectionObserver step after a scroll
/// (the geometry changed even though nothing re-laid out).
async fn emit_scroll_viewport(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    dx: f32,
    dy: f32,
) {
    let (max_x, max_y) = page_max_scroll(state).await;
    let next = (
        (state.viewport_scroll.0 + dx).clamp(0.0, max_x),
        (state.viewport_scroll.1 + dy).clamp(0.0, max_y),
    );
    let consumed = (
        next.0 - state.viewport_scroll.0,
        next.1 - state.viewport_scroll.1,
    );
    state.viewport_scroll = next;
    if consumed.0 != 0.0 || consumed.1 != 0.0 {
        emit_tab(
            events,
            RendererToBrowser::ScrollViewport {
                dx: consumed.0,
                dy: consumed.1,
            },
        )
        .await;
        // M6 Phase 3 (Rec 1/3): the actor's tree learns the new
        // scroll and the re-collected list submits — sticky
        // re-positions in the same frame as the scroll.
        apply_viewport_scroll(state, events).await;
    }
}

/// A wheel event (Stage 8 Rec 9): the worker owns the whole wheel.
/// The delta is scaled by mode (lines × 16 px, pages × the
/// viewport size, pixels as-is); a new gesture picks the
/// innermost-first container chain under the cursor, and the delta
/// walks it — each container consumes what it can up to its
/// maximum, the leftover chains onward, and the viewport takes the
/// rest (via `ScrollViewport`, which the shell clamps to `[0, page
/// max]` — the same clamp the worker's mirror applies).
#[allow(clippy::too_many_arguments)]
async fn handle_wheel(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
    delta_mode: DeltaMode,
    phase: WheelPhase,
) {
    let (mut rx, mut ry) = match delta_mode {
        DeltaMode::Pixel => (dx, dy),
        DeltaMode::Line => (dx * 16.0, dy * 16.0),
        DeltaMode::Page => (dx * state.viewport.0, dy * state.viewport.1),
    };
    if phase == WheelPhase::Began || state.wheel_chain.is_empty() {
        state.wheel_chain = scroll_chain(state, x, y).await;
    }
    state.wheel_idle_ticks = 0;
    let mut scrolls: Vec<(NodeId, f32, f32)> = Vec::new();
    for entry in state.wheel_chain.clone() {
        if rx == 0.0 && ry == 0.0 {
            break;
        }
        let (ox, oy) = state
            .scroll_offsets
            .get(&entry.id)
            .copied()
            .unwrap_or((0.0, 0.0));
        let (nx, rem_x) = consume_axis(
            ox,
            rx,
            entry.max_scroll.0,
            entry.overscroll_x.stops_chaining(),
        );
        let (ny, rem_y) = consume_axis(
            oy,
            ry,
            entry.max_scroll.1,
            entry.overscroll_y.stops_chaining(),
        );
        if (nx, ny) != (ox, oy) {
            scrolls.push((entry.id, nx, ny));
        }
        rx = rem_x;
        ry = rem_y;
    }
    let scrolled = !scrolls.is_empty();
    if scrolled {
        apply_scrolls(state, events, scrolls).await;
    }
    if rx != 0.0 || ry != 0.0 {
        emit_scroll_viewport(state, events, rx, ry).await;
    }
    // A scroll moved the page under the observed targets: re-run
    // the IntersectionObserver step on the retained tree (no
    // re-layout — the step reads the tree as it stands).
    if scrolled || rx != 0.0 || ry != 0.0 {
        run_intersection_observer_step(state).await;
    }
}

/// The tab's scroll containers, excluding root/body (whose
/// overflow propagates to the viewport). Empty when the actor
/// holds no tree.
async fn scroll_containers(state: &TabState) -> Vec<NodeId> {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollContainers { reply: reply_tx })
        .is_err()
    {
        return Vec::new();
    }
    reply_rx.await.unwrap_or_default()
}

/// A key press / release (Stage 8 Rec 9, Stage 13 Rec 2/4).
/// Releases dispatch a real `keyup` on the focused element.
/// Presses dispatch a cancelable `keydown` first (sync, so
/// `preventDefault()` suppresses the following input and the
/// default action); then editing keys mutate the focused text
/// control (`input` event + restyle), Enter submits, and the
/// remaining keys keep their M2 default actions. Repeats re-fire
/// the action (holding an arrow scrolls repeatedly). Keys are
/// ignored while the window is unfocused — the shell filters to
/// uncaptured keys only, so this is the worker-side half.
async fn handle_key(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    key: &str,
    pressed: bool,
    modifiers: pluto_ipc::Modifiers,
) {
    if !state.window_focused {
        return;
    }
    if pressed {
        // Stage 16 Phase 1, Rec 7: a trusted key press is a user
        // gesture. Recorded before dispatch: a `preventDefault()`d
        // keydown still counts (the gesture happened), and a
        // keyup never counts.
        if let Some(js) = state.tab_js.as_mut() {
            js.note_user_activation();
        }
    }
    if !pressed {
        // A real `keyup` on the focused element (Stage 13 Rec 2).
        if let Some(focused) = state.focused
            && let Some(js) = state.tab_js.as_mut() {
                let _ = js.dispatch_typed_event_sync(
                    focused,
                    "keyup",
                    true,
                    TypedEventKind::Keyboard {
                        key: key.to_string(),
                        code: key_code(key),
                    },
                );
            }
        return;
    }
    // Cancelable `keydown` on the focused element: a prevented
    // keydown suppresses the input AND the default action.
    if let Some(focused) = state.focused {
        let not_prevented = match state.tab_js.as_mut() {
            Some(js) => js
                .dispatch_typed_event_sync(
                    focused,
                    "keydown",
                    true,
                    TypedEventKind::Keyboard {
                        key: key.to_string(),
                        code: key_code(key),
                    },
                )
                .unwrap_or(true),
            None => true,
        };
        if !not_prevented {
            return;
        }
    }
    // Editing keys consume the key on a focused text control.
    if let Some(focused) = state.focused
        && handle_edit_key(state, events, focused, key, modifiers).await
    {
        return;
    }
    // Enter on a focused control submits / resets its form.
    if key == "Enter"
        && let Some(focused) = state.focused
            && submit_on_enter(state, events, focused).await
        {
            return;
        }
    match key {
        "Tab" => move_focus(state, events, !modifiers.shift).await,
        "Enter" | " " => activate_focused(state, events, key).await,
        "ArrowDown" => scroll_focused_or_viewport(state, events, 0.0, 40.0).await,
        "ArrowUp" => scroll_focused_or_viewport(state, events, 0.0, -40.0).await,
        "ArrowRight" => scroll_focused_or_viewport(state, events, 40.0, 0.0).await,
        "ArrowLeft" => scroll_focused_or_viewport(state, events, -40.0, 0.0).await,
        "PageDown" => {
            scroll_focused_or_viewport(state, events, 0.0, state.viewport.1).await
        }
        "PageUp" => {
            scroll_focused_or_viewport(state, events, 0.0, -state.viewport.1).await
        }
        // `Home` and `End` jump to the ends of the page: a huge
        // delta is "scroll to the top / bottom" — `emit_scroll_viewport`
        // clamps it at the page maximum (zero at the top), so the
        // emitted delta is exactly the remaining distance.
        "Home" => {
            emit_scroll_viewport(state, events, 0.0, -1e9).await;
            run_intersection_observer_step(state).await;
        }
        "End" => {
            emit_scroll_viewport(state, events, 0.0, 1e9).await;
            run_intersection_observer_step(state).await;
        }
        _ => {}
    }
}

/// The focusable elements in DOM order: links with `href`,
/// buttons, inputs, selects, textareas, and anything with a
/// `tabindex`. Pre-order walk from the document root.
fn focusable_in_order(doc: &Document) -> Vec<NodeId> {
    let mut out = Vec::new();
    let dom = doc.dom();
    if let Some(root) = dom.root() {
        walk_focusable(dom, root, &mut out);
    }
    out
}

fn walk_focusable(dom: &pluto_html::Dom, id: NodeId, out: &mut Vec<NodeId>) {
    if is_focusable(dom, id) {
        out.push(id);
    }
    for &child in dom.children(id) {
        walk_focusable(dom, child, out);
    }
}

fn is_focusable(dom: &pluto_html::Dom, id: NodeId) -> bool {
    // Stage 13 Rec 3: disabled controls are never focusable —
    // not by click, not by Tab, not by `focus()`.
    if dom.is_control_disabled(id) {
        return false;
    }
    if dom.attr(id, "tabindex").is_some() {
        return true;
    }
    match dom.tag(id) {
        Some("button" | "input" | "select" | "textarea") => true,
        Some("a") => dom
            .attr(id, "href")
            .map(|href| !href.trim().is_empty())
            .unwrap_or(false),
        _ => false,
    }
}

/// Move keyboard focus (Stage 8 Rec 9, Stage 13 Rec 3): Tab
/// forward, Shift+Tab backward, wrapping at the ends, in Tab
/// order (positive `tabindex` first, `tabindex="-1"` and disabled
/// controls excluded). The move goes through `set_focus`, so
/// `blur` / `focus` events fire and `activeElement` stays true.
async fn move_focus(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    forward: bool,
) {
    let Some(doc) = state.document.as_ref() else {
        return;
    };
    let focusable = tab_order(doc);
    if focusable.is_empty() {
        return;
    }
    let next = match state
        .focused
        .and_then(|f| focusable.iter().position(|&id| id == f))
    {
        Some(i) if forward => focusable[(i + 1) % focusable.len()],
        Some(i) => focusable[(i + focusable.len() - 1) % focusable.len()],
        None if forward => focusable[0],
        None => *focusable.last().unwrap(),
    };
    set_focus(state, events, Some(next)).await;
}

/// Activate the focused element (Stage 8 Rec 9): a link navigates,
/// a button/input/select/textarea dispatches a click into the JS
/// event loop. Space on a non-activatable element scrolls a page
/// down; Enter on one does nothing.
async fn activate_focused(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    key: &str,
) {
    let Some(focused) = state.focused else {
        if key == " " {
            scroll_focused_or_viewport(state, events, 0.0, state.viewport.1).await;
        }
        return;
    };
    let (tag, href) = {
        let Some(doc) = state.document.as_ref() else {
            return;
        };
        (
            doc.dom().tag(focused).map(|t| t.to_string()),
            page::find_link_href(doc, focused),
        )
    };
    match tag.as_deref() {
        Some("a") => {
            if let Some(url) = href {
                start_navigation(state, url);
            }
        }
        Some("button" | "input" | "select" | "textarea") => {
            if let Some(js) = state.tab_js.as_mut() {
                // Stage 12 Rec 7: a typed keyboard event, so
                // the click listener reads which key activated
                // the element (`Enter` / ` `) instead of a bare
                // `click`. The event type stays `click` — this
                // is a click synthesised from the keyboard.
                let code = match key {
                    "Enter" => "Enter",
                    " " => "Space",
                    other => other,
                };
                js.dispatch_typed_event(
                    focused,
                    "click",
                    true,
                    TypedEventKind::Keyboard {
                        key: key.to_string(),
                        code: code.to_string(),
                    },
                );
            }
        }
        _ => {
            if key == " " {
                scroll_focused_or_viewport(state, events, 0.0, state.viewport.1).await;
            }
        }
    }
}

/// Scroll the focused element when it is a scroll container,
/// otherwise the viewport (Stage 8 Rec 9). A container at its
/// limit chains to the viewport — keyboard scroll has no
/// `overscroll-behavior` stop (unlike the wheel), so a key always
/// moves something when anything can move.
async fn scroll_focused_or_viewport(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    dx: f32,
    dy: f32,
) {
    if let Some(focused) = state.focused {
        let containers = scroll_containers(state).await;
        if containers.contains(&focused) {
            let (ox, oy) = state
                .scroll_offsets
                .get(&focused)
                .copied()
                .unwrap_or((0.0, 0.0));
            // Clamped to the container's maximum (Stage 10 Rec 2):
            // a key that would carry it past the end consumes
            // nothing, so it falls through to the viewport below —
            // keyboard scroll has no `overscroll-behavior` stop.
            let (max_x, max_y) = container_max_scroll(state, focused).await;
            let (nx, ny) = (
                (ox + dx).clamp(0.0, max_x),
                (oy + dy).clamp(0.0, max_y),
            );
            if (nx, ny) != (ox, oy) {
                apply_scrolls(state, events, vec![(focused, nx, ny)]).await;
                run_intersection_observer_step(state).await;
                return;
            }
        }
    }
    emit_scroll_viewport(state, events, dx, dy).await;
    run_intersection_observer_step(state).await;
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
    // M6 Phase 3 (Rec 11): the engine resolves viewport units
    // against the current viewport.
    state.animation_engine.set_ctx(CalcCtx {
        font_size: 16.0,
        root_font_size: 16.0,
        viewport: (width, height),
        pct_basis: None,
    });
    if state.has_document {
        eprintln!("[pluto] worker viewport resized to {width}x{height}, re-layout");
        layout_and_submit(state, events, None, true, std::time::Instant::now()).await;
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

    /// A page with one radial-gradient inline SVG (Stage 15,
    /// M7 Phase 2, Rec 5): the worker must rasterise it and
    /// announce the Blob's pixels before the submit that names
    /// them. Served at `/svg` by the test server below.
    const SVG_PAGE: &str = r#"<!doctype html>
<html>
<head><title>SVG</title></head>
<body>
<svg width="100" height="100" viewBox="0 0 100 100"><defs><radialGradient id="rg"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="white"/></radialGradient></defs><rect x="0" y="0" width="100" height="100" fill="url(#rg)"/></svg>
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
                let body = if path == "/svg" { SVG_PAGE } else { EXAMPLE_COM };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// Guard holding `TEST_HOME_LOCK` with `HOME` pointed at a
    /// fresh temp dir for the whole span a worker tab is alive
    /// (see `TEST_HOME_LOCK` in `lib.rs`). A worker tab reads the
    /// profile at `TabState::new` and writes it back on shutdown,
    /// so it is a HOME-mutating test even though it never touches
    /// `HOME` itself — without this, a parallel
    /// permissions/storage/bookmarks/security round-trip test could
    /// swap `HOME` under it and corrupt the profile (Stage 10
    /// Rec 6). Restores `HOME` on drop; the guard must be held
    /// until the worker task has been awaited.
    struct WorkerHomeGuard {
        old: Option<std::ffi::OsString>,
        dir: std::path::PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for WorkerHomeGuard {
        fn drop(&mut self) {
            // Restore HOME first; the lock field drops after this,
            // so no other HOME-mutating test can interleave.
            unsafe {
                match self.old.take() {
                    Some(v) => std::env::set_var("HOME", v),
                    None => std::env::remove_var("HOME"),
                }
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn worker_home() -> WorkerHomeGuard {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let lock = crate::TEST_HOME_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "pluto-renderer-test-home-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let old = std::env::var_os("HOME");
        // SAFETY: HOME is only read by the persistence helpers,
        // which run on the tab thread while the lock is held.
        unsafe {
            std::env::set_var("HOME", &dir);
        }
        WorkerHomeGuard { old, dir, _lock: lock }
    }

    /// Drive the worker directly (no iced): send commands in, read
    /// `RendererEvent` messages out. Returns the worker task handle
    /// and the channels, so each test drives its own worker — plus
    /// the `WorkerHomeGuard`, which the caller must hold until the
    /// worker task has ended (the tab reads the profile at startup
    /// and writes it back on shutdown).
    fn spawn_worker() -> (
        WorkerHomeGuard,
        tokio::task::JoinHandle<()>,
        pluto_ipc::CommandSender,
        mpsc::Receiver<crate::Message>,
    ) {
        let home = worker_home();
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (event_tx, event_rx) = mpsc::channel(100);
        let handle = tokio::spawn(run_worker(0, cmd_rx, event_tx));
        (home, handle, cmd_tx, event_rx)
    }

    /// The cross-process worker bridge over an in-process transport
    /// (Stage 9 Rec 9): the same tab thread + layout actor as the
    /// in-process worker, driven through `run_worker_transport` with
    /// a channel transport standing in for the socketpair. A
    /// navigation to `about:blank` exercises the full pipeline with
    /// zero network (pluto-net rejects non-http(s) before any
    /// socket), and `Shutdown` ends the bridge with exit code 0.
    #[tokio::test]
    async fn worker_transport_bridge_exchanges_messages() {
        use pluto_ipc::transport::{channel_transport_pair, recv_renderer_message};
        let _home = worker_home();
        let (mut browser, renderer) = channel_transport_pair();
        let handle = tokio::spawn(run_worker_transport(renderer, 0));
        browser
            .send_message(&BrowserToRenderer::CommitNavigation {
                url: "about:blank".to_string(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .await
            .expect("navigation sends");
        // Stage 13 Rec 11: the broker-mode child asks the browser
        // for the document — even `about:blank`, which has no fetch.
        // Answer that ask the way the broker would (Error
        // `invalid_url`) and the worker renders the error page.
        // The error page arrives as a display list first, then the
        // reason — the same order as the in-process worker.
        let answer_blank_ask = async |browser: &mut pluto_ipc::transport::ChannelTransport,
                                      request: pluto_ipc::ResourceRequest| {
            assert_eq!(request.url, "about:blank", "only the blank page is asked");
            browser
                .send_message(&BrowserToRenderer::ResourceEvent(
                    pluto_ipc::ResourceEvent::Error {
                        request_id: request.request_id,
                        code: "invalid_url".to_string(),
                        reason: "non-http(s) scheme".to_string(),
                    },
                ))
                .await
                .expect("error event sends");
        };
        let bytes = loop {
            let msg = tokio::time::timeout(
                Duration::from_secs(20),
                recv_renderer_message(&mut browser),
            )
            .await
            .expect("first reply arrives in time")
            .expect("recv succeeds")
            .expect("a message arrives");
            match msg {
                RendererToBrowser::SubmitDisplayList { bytes, .. } => break bytes,
                RendererToBrowser::RequestResource(request) => {
                    answer_blank_ask(&mut browser, request).await;
                }
                RendererToBrowser::Heartbeat | RendererToBrowser::SandboxReport { .. } => {}
                other => panic!("expected SubmitDisplayList first, got {other:?}"),
            }
        };
        {
            let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
            assert!(!list.is_empty(), "error page still paints");
        }
        // The error page's title may interleave as a
        // `TitleChanged` between the display list and the
        // failure (the worker coalesces it onto the submit).
        let mut second;
        loop {
            second = tokio::time::timeout(
                Duration::from_secs(20),
                recv_renderer_message(&mut browser),
            )
            .await
            .expect("second reply arrives in time")
            .expect("recv succeeds")
            .expect("a message arrives");
            match second {
                RendererToBrowser::RequestResource(request) => {
                    answer_blank_ask(&mut browser, request).await;
                }
                RendererToBrowser::Heartbeat | RendererToBrowser::SandboxReport { .. } => {}
                RendererToBrowser::TitleChanged { .. } => continue,
                _ => break,
            }
        }
        match second {
            RendererToBrowser::LoadFailed { url, reason } => {
                assert_eq!(url, "about:blank");
                assert!(!reason.is_empty());
            }
            other => panic!("expected LoadFailed, got {other:?}"),
        }
        browser
            .send_message(&BrowserToRenderer::Shutdown)
            .await
            .expect("shutdown sends");
        assert_eq!(
            handle.await.expect("bridge task ends"),
            0,
            "a clean Shutdown exits 0"
        );
    }

    /// Stage 13 Rec 12d: when the renderer process cannot be
    /// spawned at all, the bridge FAILS CLOSED by default — it
    /// reports a crash and stays alive awaiting a navigation,
    /// never silently running the page in-process.
    #[tokio::test]
    async fn renderer_bridge_fails_closed_when_spawn_fails() {
        let _home = worker_home();
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (output_tx, mut output_rx) = mpsc::channel::<crate::Message>(100);
        let handle = tokio::spawn(run_renderer_bridge(
            0,
            std::path::Path::new("/nonexistent/pluto-browser"),
            cmd_rx,
            output_tx,
            false,
        ));
        let event = tokio::time::timeout(Duration::from_secs(20), next_event(&mut output_rx))
            .await
            .expect("a renderer event arrives in time");
        assert!(
            matches!(
                event,
                RendererToBrowser::RendererCrashed { ref reason }
                    if reason.contains("in-process fallback disabled")
            ),
            "spawn failure reports a fail-closed crash, got {event:?}"
        );
        // The bridge stays alive: a later navigation retries the
        // spawn and reports the failure again instead of dying.
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: "about:blank".to_string(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("bridge takes the navigation");
        let event = tokio::time::timeout(Duration::from_secs(20), next_event(&mut output_rx))
            .await
            .expect("a renderer event arrives in time");
        assert!(
            matches!(
                event,
                RendererToBrowser::RendererCrashed { ref reason }
                    if reason.contains("respawn failed")
            ),
            "a later navigation retries the spawn, got {event:?}"
        );
        drop(cmd_tx);
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("bridge ends after the shell goes away")
            .expect("bridge task joins");
    }

    /// Stage 9 Rec 9 (gated by Stage 13 Rec 12d): with the
    /// fallback allowed — and only then — the bridge falls back
    /// to the in-process worker when the renderer process cannot
    /// be spawned at all. The flag is passed explicitly (never
    /// via process env — parallel tests would race on it).
    #[tokio::test]
    async fn renderer_bridge_falls_back_to_in_process_worker_when_spawn_fails() {
        let _home = worker_home();
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (output_tx, mut output_rx) = mpsc::channel::<crate::Message>(100);
        let handle = tokio::spawn(run_renderer_bridge(
            0,
            std::path::Path::new("/nonexistent/pluto-browser"),
            cmd_rx,
            output_tx,
            true,
        ));
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: "about:blank".to_string(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("bridge takes the navigation");
        let event = tokio::time::timeout(Duration::from_secs(20), next_event(&mut output_rx))
            .await
            .expect("a renderer event arrives in time");
        // about:blank loads: DidCommit or the first display list —
        // either proves the fallback worker served the navigation.
        assert!(
            matches!(
                event,
                RendererToBrowser::DidCommit { .. }
                    | RendererToBrowser::SubmitDisplayList { .. }
            ),
            "fallback worker emits a navigation event, got {event:?}"
        );
        drop(cmd_tx);
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("bridge ends after the shell goes away")
            .expect("bridge task joins");
    }

    /// Read the next *structural* worker event, skipping the
    /// informational events that interleave freely with the
    /// navigation sequence: `LoadProgress` (reported during the
    /// document fetch, before `DidCommit`), `TitleChanged`
    /// (coalesced onto the display-list submit, after it),
    /// `FaviconChanged` (an async fetch that lands whenever it
    /// lands), `ResourceReady` (one per decoded image, in
    /// arrival order), `Heartbeat` (Stage 13 Rec 12b: every 5 s —
    /// silence-window tests would see one), and `SandboxReport`
    /// (Stage 13 Rec 12a: once, before the first navigation).
    /// Tests that assert on progress, titles, favicons,
    /// resources, heartbeats, or sandbox reports read the raw
    /// channel instead of this helper.
    async fn next_event(rx: &mut mpsc::Receiver<crate::Message>) -> RendererToBrowser {
        loop {
            match rx.next().await.expect("worker emits an event") {
                crate::Message::RendererEvent(_, event) => match event {
                    RendererToBrowser::LoadProgress { .. }
                    | RendererToBrowser::TitleChanged { .. }
                    | RendererToBrowser::FaviconChanged { .. }
                    | RendererToBrowser::ResourceReady { .. }
                    | RendererToBrowser::Heartbeat
                    | RendererToBrowser::SandboxReport { .. } => continue,
                    event => return event,
                },
                other => panic!("expected a RendererEvent, got {other:?}"),
            }
        }
    }

    /// Send a left click (press + release) at page coordinates
    /// (Stage 8 Rec 9: a click is a Down + Up pair on the same
    /// target, not one `Click` event).
    fn send_click(cmd_tx: &pluto_ipc::CommandSender, x: f32, y: f32) {
        let pointer = |kind: PointerKind, buttons: u16| {
            BrowserToRenderer::InputEvent(InputEvent::Pointer {
                kind,
                x,
                y,
                button: Some(0),
                buttons,
                modifiers: pluto_ipc::Modifiers::default(),
                timestamp_ms: 0,
                pointer_id: 0,
            })
        };
        cmd_tx
            .send(pointer(PointerKind::Down, 1))
            .expect("worker takes the pointer down");
        cmd_tx
            .send(pointer(PointerKind::Up, 0))
            .expect("worker takes the pointer up");
    }

    /// Send one wheel event at page coordinates (Stage 8 Rec 9):
    /// a `Began` pixel-mode wheel, as the shell synthesises for a
    /// first wheel after idle.
    fn send_wheel(cmd_tx: &pluto_ipc::CommandSender, x: f32, y: f32, dx: f32, dy: f32) {
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Wheel {
                x,
                y,
                dx,
                dy,
                delta_mode: DeltaMode::Pixel,
                phase: WheelPhase::Began,
                modifiers: pluto_ipc::Modifiers::default(),
                timestamp_ms: 0,
            }))
            .expect("worker takes the wheel event");
    }

    /// Send one pointer move at page coordinates (Stage 8 Rec 9).
    fn send_move(cmd_tx: &pluto_ipc::CommandSender, x: f32, y: f32) {
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Pointer {
                kind: PointerKind::Move,
                x,
                y,
                button: None,
                buttons: 0,
                modifiers: pluto_ipc::Modifiers::default(),
                timestamp_ms: 0,
                pointer_id: 0,
            }))
            .expect("worker takes the pointer move");
    }

    /// Send one key press (Stage 8 Rec 9). `modifiers` defaults to
    /// none; pass a `Modifiers` with `shift: true` for Shift+Tab.
    fn send_key(
        cmd_tx: &pluto_ipc::CommandSender,
        key: &str,
        code: &str,
        modifiers: pluto_ipc::Modifiers,
    ) {
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Key {
                key: key.to_string(),
                code: code.to_string(),
                pressed: true,
                repeat: false,
                modifiers,
                timestamp_ms: 0,
            }))
            .expect("worker takes the key event");
    }

    /// Send a key release (Stage 13 Rec 2: a real `keyup`).
    fn send_key_release(
        cmd_tx: &pluto_ipc::CommandSender,
        key: &str,
        code: &str,
        modifiers: pluto_ipc::Modifiers,
    ) {
        cmd_tx
            .send(BrowserToRenderer::InputEvent(InputEvent::Key {
                key: key.to_string(),
                code: code.to_string(),
                pressed: false,
                repeat: false,
                modifiers,
                timestamp_ms: 0,
            }))
            .expect("worker takes the key event");
    }

    /// Wait for a `SubmitDisplayList` satisfying `want` (other
    /// lists are skipped; anything else is a failure).
    async fn wait_for_list(
        rx: &mut mpsc::Receiver<crate::Message>,
        timeout: Duration,
        want: impl Fn(&DisplayList) -> bool,
    ) -> Option<DisplayList> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let event = match tokio::time::timeout(remaining, next_event(rx)).await {
                Ok(event) => event,
                Err(_) => return None,
            };
            match event {
                RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                    if want(&list) {
                        return Some(list);
                    }
                }
                other => panic!("expected SubmitDisplayList while waiting, got {other:?}"),
            }
        }
    }

    /// True when no `DidCommit` arrives within `timeout`
    /// (other events — caret-blink resubmits included — are
    /// consumed and ignored).
    async fn no_commit_within(
        rx: &mut mpsc::Receiver<crate::Message>,
        timeout: Duration,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return true;
            }
            let event = match tokio::time::timeout(remaining, next_event(rx)).await {
                Ok(event) => event,
                Err(_) => return true,
            };
            if matches!(event, RendererToBrowser::DidCommit { .. }) {
                return false;
            }
        }
    }

    /// Every glyph in a display list as (colour, x, y).
    fn glyph_colors(list: &DisplayList) -> Vec<([f32; 4], f32, f32)> {
        let mut out = Vec::new();
        for item in list.0.iter() {
            if let DisplayItem::Glyphs { color, glyphs } = item {
                for g in glyphs {
                    out.push((*color, g.x, g.y));
                }
            }
        }
        out
    }

    /// Wait for a `SubmitDisplayList` containing a glyph of
    /// `color` (a restyle resubmits), and return that list. Other
    /// display lists are skipped; anything else panics.
    async fn wait_for_glyph_color(
        rx: &mut mpsc::Receiver<crate::Message>,
        color: [f32; 4],
        timeout: Duration,
    ) -> Option<DisplayList> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let event = match tokio::time::timeout(remaining, next_event(rx)).await {
                Ok(event) => event,
                Err(_) => return None,
            };
            match event {
                RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                    if glyph_colors(&list).iter().any(|(c, _, _)| *c == color) {
                        return Some(list);
                    }
                }
                other => panic!("expected SubmitDisplayList while waiting for a colour, got {other:?}"),
            }
        }
    }

    /// True when no event arrives within `timeout`.
    async fn no_event_within(rx: &mut mpsc::Receiver<crate::Message>, timeout: Duration) -> bool {
        tokio::time::timeout(timeout, next_event(rx)).await.is_err()
    }

    #[tokio::test]
    async fn worker_loads_a_page_and_submits_its_display_list() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: url.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
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

    /// Stage 15 (M7 Phase 2, Rec 5): a page with a
    /// radial-gradient inline SVG produces a `ResourceReady`
    /// BEFORE the `SubmitDisplayList` that names the Blob — the
    /// shell validates the list against its per-tab table, so a
    /// Blob without pixels would fail validation and drop the
    /// frame. Mirrors
    /// `worker_loads_a_page_and_submits_its_display_list`, with
    /// the `/svg` page. (`next_event` skips `ResourceReady`, so
    /// this test reads the channel raw until the submit lands.)
    #[tokio::test]
    async fn worker_announces_svg_blob_before_submit() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/svg", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: url.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let mut ready: Vec<(ResourceId, Resource)> = Vec::new();
        let (epoch, bytes) = loop {
            let msg = tokio::time::timeout(Duration::from_secs(20), event_rx.recv())
                .await
                .expect("an event arrives in time")
                .expect("worker emits an event");
            let event = match msg {
                crate::Message::RendererEvent(_, event) => event,
                other => panic!("expected a RendererEvent, got {other:?}"),
            };
            match event {
                RendererToBrowser::ResourceReady {
                    resource_id,
                    resource,
                    ..
                } => {
                    ready.push((resource_id, resource));
                }
                RendererToBrowser::SubmitDisplayList {
                    epoch,
                    bytes,
                    resources,
                } => {
                    assert!(resources.is_empty(), "resources travel out-of-band");
                    break (epoch, bytes);
                }
                // LoadProgress / TitleChanged / FaviconChanged /
                // Heartbeat / SandboxReport interleave freely.
                _ => {}
            }
        };
        assert_eq!(epoch, 1, "first frame is epoch 1");
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        let blob_ids: Vec<ResourceId> = list
            .0
            .iter()
            .filter_map(|i| match i {
                DisplayItem::Blob { resource, .. } => Some(*resource),
                _ => None,
            })
            .collect();
        assert_eq!(blob_ids.len(), 1, "one radial svg = one Blob");
        let (_, ready_resource) = ready
            .iter()
            .find(|(id, _)| blob_ids.contains(id))
            .expect("the Blob's pixels arrived in a ResourceReady before the submit");
        assert!(
            matches!(
                ready_resource,
                Resource::Image {
                    width: 100,
                    height: 100,
                    rgba,
                } if !rgba.is_empty()
            ),
            "the announced resource is the 100x100 raster"
        );
        let mut table = ResourceTable::new();
        for (id, resource) in &ready {
            table.insert(*id, resource.clone());
        }
        list.validate(&table)
            .expect("the submitted list validates against the announced resources");
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
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

    /// Follow-up 2 (Stage 10): shrinking the viewport re-clamps
    /// the worker's viewport scroll mirror — a `ScrollViewport`
    /// correction (negative, to the new maximum) arrives before
    /// the resized display list.
    #[tokio::test]
    async fn worker_resize_reclamps_the_viewport_scroll() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_overscroll_page(listener, "auto"));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let first_epoch = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => epoch,
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        // The page is 1100px tall in a 768px viewport (max 332):
        // wheel down 200px in the tall spacer, where no container
        // sits, so the whole delta scrolls the viewport.
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, 200.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!((dy - 200.0).abs() < 0.01, "viewport scrolled 200, got {dy}");
            }
            other => panic!("expected ScrollViewport after wheel down, got {other:?}"),
        }
        // Shrink the viewport to 1000px tall: the new maximum is
        // 100, below the current 200 — the resize must emit the
        // -100 correction before the new list lands.
        cmd_tx
            .send(BrowserToRenderer::Resize {
                width: 1024.0,
                height: 1000.0,
            })
            .expect("worker takes Resize");
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - (-100.0)).abs() < 0.01,
                    "correction is -100, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport correction after resize, got {other:?}"),
        }
        let (epoch, bytes) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, bytes, .. } => (epoch, bytes),
            other => panic!("expected SubmitDisplayList after resize, got {other:?}"),
        };
        assert!(epoch > first_epoch, "resize advances the epoch");
        let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                height: 1000.0,
                ..
            }
        ));
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    #[tokio::test]
    async fn worker_failed_load_renders_an_error_page() {
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: "ftp://example.com/".to_string(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let slow = format!("http://127.0.0.1:{}/slow", addr.port());
        let fast = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: slow,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        // Before the slow fetch lands (500 ms), commit the fast one.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: fast.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page1 = format!("http://127.0.0.1:{}/", addr.port());
        let page2 = format!("http://127.0.0.1:{}/page2", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page1,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
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
        // The press/release restyle the link (:hover/:active) and
        // resubmit first; the navigation's commit follows.
        send_click(&cmd_tx, 100.0, 50.0);
        let committed = loop {
            match next_event(&mut event_rx).await {
                RendererToBrowser::DidCommit { url, .. } => break url,
                RendererToBrowser::SubmitDisplayList { .. } => continue,
                other => panic!("expected DidCommit for page2, got {other:?}"),
            }
        };
        assert_eq!(committed, page2);
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        // A click on empty page area (below the 100px link, e.g.
        // y=500 — the page is otherwise blank) navigates nowhere:
        // no event arrives within 200 ms.
        send_click(&cmd_tx, 100.0, 500.0);
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

    /// Wait for a `TitleChanged` carrying `want` (Stage 16 Phase
    /// 3): pages under test report JS-observed state by setting
    /// `document.title`, since the worker tests have no JS-eval
    /// channel. Other events are skipped; a timeout fails the test.
    async fn wait_for_title(rx: &mut mpsc::Receiver<crate::Message>, want: &str) {
        let found = tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                match rx.next().await.expect("worker emits an event") {
                    crate::Message::RendererEvent(
                        _,
                        RendererToBrowser::TitleChanged { title },
                    ) if title == want => return,
                    crate::Message::RendererEvent(_, _) => continue,
                    other => panic!("expected a RendererEvent, got {other:?}"),
                }
            }
        })
        .await;
        assert!(found.is_ok(), "timed out waiting for title {want:?}");
    }

    /// Serve one fixed page body for the history tests below.
    async fn serve_fixed_page(listener: TcpListener, body: &'static str) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(_) = stream.read(&mut buf).await else {
                    return;
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

    /// Stage 16 Phase 3, Rec 8 (finisher): a page's `pushState`
    /// reaches the shell as a `HistoryPush` — the drain forwards
    /// the staged request instead of dropping it.
    #[tokio::test]
    async fn worker_history_push_state_reaches_shell() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_fixed_page(
            listener,
            "<!doctype html><html><head><style>body{margin:0}</style></head>\
             <body><script>history.pushState({view:\"home\"}, \"\", \"app#view\");</script>\
             <p>push page</p></body></html>",
        ));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        // The push runs during the parse, so its `HistoryPush`
        // may arrive before or after the commit — collect both,
        // in either order.
        let mut committed = false;
        let mut pushed: Option<(String, Option<String>)> = None;
        while !committed || pushed.is_none() {
            match next_event(&mut event_rx).await {
                RendererToBrowser::DidCommit { .. } => committed = true,
                RendererToBrowser::HistoryPush { url, state_json } => {
                    pushed = Some((url, state_json));
                }
                RendererToBrowser::SubmitDisplayList { .. } => continue,
                other => panic!("expected DidCommit/HistoryPush, got {other:?}"),
            }
        }
        let (url, state_json) = pushed.expect("HistoryPush arrived");
        assert_eq!(url, format!("{page}app#view"));
        assert_eq!(state_json.as_deref(), Some(r#"{"view":"home"}"#));
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 16 Phase 3, Rec 8 (finisher): a traversal commit
    /// restores the entry's state into the fresh realm and fires
    /// `popstate` after the commit — the page's listener reads
    /// the restored `history.state`.
    #[tokio::test]
    async fn worker_traversal_restores_state_and_fires_popstate() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_fixed_page(
            listener,
            "<!doctype html><html><head><style>body{margin:0}</style></head>\
             <body><script>window.addEventListener('popstate', (e) => { \
             document.title = 'popped:' + JSON.stringify(e.state); });</script>\
             <p>popstate page</p></body></html>",
        ));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        // Drain the first page's submit before the traversal:
        // it is already queued ahead of the traversal's commit.
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        // A Back/Forward commit to the same page, carrying the
        // entry's state: the worker restores it and fires
        // `popstate` after the commit.
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: Some(r#"{"view":"home"}"#.to_string()),
                traversal: true,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        wait_for_title(&mut event_rx, r#"popped:{"view":"home"}"#).await;
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 16 Phase 3, Rec 9 (finisher): setting
    /// `location.hash` is a same-document navigation — no fetch,
    /// no new entry: `hashchange` fires and no second `DidCommit`
    /// arrives.
    #[tokio::test]
    async fn worker_hash_change_is_same_document() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_fixed_page(
            listener,
            "<!doctype html><html><head><style>body{margin:0}</style></head>\
             <body><div id=\"sec2\" style=\"height:2000px\">section two</div>\
             <script>window.addEventListener('hashchange', () => { \
             document.title = 'hashed:' + location.hash; }); \
             location.hash = 'sec2';</script></body></html>",
        ));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        // The `hashchange` (and its title) may arrive before or
        // after the commit — collect both, in either order.
        let mut committed = false;
        let mut titled = false;
        let found = tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                match event_rx.next().await.expect("worker emits an event") {
                    crate::Message::RendererEvent(_, RendererToBrowser::DidCommit { .. }) => {
                        committed = true;
                    }
                    crate::Message::RendererEvent(
                        _,
                        RendererToBrowser::TitleChanged { title },
                    ) if title == "hashed:#sec2" => titled = true,
                    crate::Message::RendererEvent(_, _) => continue,
                    other => panic!("expected a RendererEvent, got {other:?}"),
                }
                if committed && titled {
                    return;
                }
            }
        })
        .await;
        assert!(
            found.is_ok(),
            "timed out waiting for DidCommit + hashed title"
        );
        // A full reload would commit again — nothing arrives
        // within 200 ms (the established quiet-window idiom).
        assert!(
            tokio::time::timeout(Duration::from_millis(200), next_event(&mut event_rx))
                .await
                .is_err(),
            "a hash change must not commit a second load"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 16 Phase 3, Rec 8 (finisher): the shell's
    /// `HistorySync` mirrors the session-history length / index
    /// into the realm — `history.length` reads the shell's value,
    /// not the realm's optimistic bookkeeping.
    #[tokio::test]
    async fn worker_history_sync_mirrors_shell_length() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_fixed_page(
            listener,
            "<!doctype html><html><head><style>body{margin:0}</style></head>\
             <body><script>setInterval(() => { \
             if (history.length !== 1) document.title = 'len:' + history.length; }, 100);</script>\
             <p>sync page</p></body></html>",
        ));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        cmd_tx
            .send(BrowserToRenderer::HistorySync {
                length: 7,
                index: 3,
            })
            .expect("worker takes HistorySync");
        wait_for_title(&mut event_rx, "len:7").await;
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 16 Phase 3, Rec 11 (finisher): `new WebSocket`
    /// asks the shell to open the socket (a `WebSocketOpen`
    /// leaves the worker), and a `WebSocketEvent` from the shell
    /// reaches the socket's `onopen` handler.
    #[tokio::test]
    async fn worker_websocket_open_round_trips() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let ws_url = format!("ws://127.0.0.1:{}/chat", addr.port());
        let body = format!(
            "<!doctype html><html><head><style>body{{margin:0}}</style></head>\
             <body><script>const ws = new WebSocket('{ws_url}'); \
             ws.onopen = () => {{ document.title = 'ws-open'; }};</script>\
             <p>socket page</p></body></html>"
        );
        let body: &'static str = Box::leak(body.into_boxed_str());
        tokio::spawn(serve_fixed_page(listener, body));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let page = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: page.clone(),
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        // `new WebSocket` runs during the parse, so its
        // `WebSocketOpen` may arrive before or after the commit —
        // collect both, in either order.
        let mut socket_id = None;
        let mut committed = false;
        while socket_id.is_none() || !committed {
            match next_event(&mut event_rx).await {
                RendererToBrowser::WebSocketOpen {
                    socket_id: id, url, ..
                } => {
                    assert_eq!(url, ws_url);
                    socket_id = Some(id);
                }
                RendererToBrowser::DidCommit { .. } => committed = true,
                RendererToBrowser::SubmitDisplayList { .. } => continue,
                other => panic!("expected WebSocketOpen/DidCommit, got {other:?}"),
            }
        }
        let socket_id = socket_id.expect("WebSocketOpen arrived");
        // The shell's answer (the broker's handshake, simulated
        // here): the socket opens, and the page's handler runs.
        cmd_tx
            .send(BrowserToRenderer::WebSocketEvent {
                socket_id,
                event: pluto_ipc::WebSocketEvent::Opened {
                    socket_id,
                    protocol: String::new(),
                },
            })
            .expect("worker takes WebSocketEvent");
        wait_for_title(&mut event_rx, "ws-open").await;
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let first_epoch = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => epoch,
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        // The container is a 100px-tall div at (0,0); (50,50) is
        // inside it. The wheel scrolls the container (new epoch)
        // instead of the viewport.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 48.0);
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

    /// One-page server with a single nested scroll container whose
    /// `overscroll-behavior` is `behavior` (Stage 8 Rec 9): a
    /// 100px-tall `overflow:auto` div with ten lines inside it,
    /// over a 1000px spacer — the page is 1100px tall in a 768px
    /// viewport, so the viewport itself can scroll 332px.
    async fn serve_overscroll_page(listener: TcpListener, behavior: &'static str) {
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
                    "<!doctype html><html><head><style>body{margin:0}\n                     div.c{height:100px;overflow:auto;overscroll-behavior:",
                );
                body.push_str(behavior);
                body.push_str("}</style></head><body><div class=\"c\">");
                for i in 0..10 {
                    body.push_str(&format!("<p>line {i}</p>"));
                }
                body.push_str("</div><div style=\"height:1000px\"></div></body></html>");
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    /// Stage 8 Rec 9: hovering a link restyles it (`:hover`), the
    /// restyle resubmits with the new colour; moving within the
    /// same link resubmits nothing; moving off restores the colour.
    #[tokio::test]
    async fn worker_hover_restyle_recolors_the_link() {
        let page = "<!doctype html><html><head><style>body{margin:0}            a{display:block;width:200px;height:40px} a:hover{color:red}</style></head><body>            <a href=\"/page2\">Link</a></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 4);
        // Hover the link: the glyphs turn red.
        send_move(&cmd_tx, 100.0, 20.0);
        assert!(
            wait_for_glyph_color(&mut event_rx, [1.0, 0.0, 0.0, 1.0], Duration::from_secs(5))
                .await
                .is_some(),
            "hover did not recolor the link"
        );
        // Moving within the same link: the chain is unchanged, so
        // no resubmit arrives within 200 ms.
        send_move(&cmd_tx, 100.0, 25.0);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "move within the hovered link must not resubmit"
        );
        // Moving off the link: the colour leaves red again.
        send_move(&cmd_tx, 400.0, 400.0);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut restored = false;
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(
                deadline.saturating_duration_since(tokio::time::Instant::now()),
                next_event(&mut event_rx),
            )
            .await
            {
                Ok(RendererToBrowser::SubmitDisplayList { bytes, .. }) => {
                    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                    if !glyph_colors(&list)
                        .iter()
                        .any(|(c, _, _)| *c == [1.0, 0.0, 0.0, 1.0])
                    {
                        restored = true;
                        break;
                    }
                }
                Ok(other) => panic!("expected SubmitDisplayList, got {other:?}"),
                Err(_) => break,
            }
        }
        assert!(restored, "moving off the link did not restore its colour");
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 8 Rec 9: a wheel scrolled up past a container's top
    /// limit chains the leftover to the viewport — unless the
    /// container's `overscroll-behavior` stops the chain, in which
    /// case the leftover is dropped and no `ScrollViewport` arrives.
    async fn wheel_overscroll_case(behavior: &'static str, expect_viewport: bool) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_overscroll_page(listener, behavior));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let first_epoch = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => epoch,
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        // Wheel down 48px inside the container: the container
        // scrolls (new epoch), nothing chains to the viewport.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 48.0);
        let (epoch, _) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, bytes, .. } => (epoch, bytes),
            other => panic!("expected SubmitDisplayList after wheel down, got {other:?}"),
        };
        assert!(epoch > first_epoch, "wheel down resubmits");
        // Wheel down 200px below the container (y=500, in the tall
        // spacer): no container under the cursor, so the whole
        // delta scrolls the viewport — the page is 1100px tall in
        // a 768px viewport, so 200px is inside its maximum. The
        // viewport must be scrolled down first: a leftover that
        // chains to a viewport already at its limit (or at zero,
        // for a negative delta) has nowhere to go.
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, 200.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!((dy - 200.0).abs() < 0.01, "viewport scrolled 200, got {dy}");
            }
            other => panic!("expected ScrollViewport below the container, got {other:?}"),
        }
        // Wheel up 100px: the container consumes 48 (back to 0)
        // and 52px are left over.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, -100.0);
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::SubmitDisplayList { .. }
        ));
        if expect_viewport {
            match next_event(&mut event_rx).await {
                RendererToBrowser::ScrollViewport { dx, dy } => {
                    assert_eq!(dx, 0.0);
                    assert!((dy - (-52.0)).abs() < 0.01, "leftover delta is -52, got {dy}");
                }
                other => panic!("expected ScrollViewport with the leftover, got {other:?}"),
            }
        } else {
            assert!(
                no_event_within(&mut event_rx, Duration::from_millis(200)).await,
                "overscroll-behavior:{behavior} must drop the leftover, not chain it"
            );
        }
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    #[tokio::test]
    async fn worker_wheel_overscroll_none_drops_the_remainder() {
        wheel_overscroll_case("none", false).await;
    }

    #[tokio::test]
    async fn worker_wheel_overscroll_auto_chains_to_the_viewport() {
        wheel_overscroll_case("auto", true).await;
    }

    /// `consume_axis` (Stage 10 Rec 2): the offset clamps to
    /// `[0, max]` and only the unconsumed part of the delta chains
    /// on — unless `overscroll-behavior` stops the chain.
    #[test]
    fn consume_axis_clamps_to_both_limits() {
        // At the top, a negative delta is unconsumed (chains on).
        assert_eq!(consume_axis(0.0, -50.0, 100.0, false), (0.0, -50.0));
        // A delta inside the range is fully consumed.
        assert_eq!(consume_axis(0.0, 50.0, 100.0, false), (50.0, 0.0));
        // Past the bottom, only the room up to the maximum is
        // consumed; the rest chains on.
        assert_eq!(consume_axis(80.0, 50.0, 100.0, false), (100.0, 30.0));
        // Already at the maximum, nothing is consumed.
        assert_eq!(consume_axis(100.0, 50.0, 100.0, false), (100.0, 50.0));
        // A negative delta from the maximum is fully consumed.
        assert_eq!(consume_axis(100.0, -30.0, 100.0, false), (70.0, 0.0));
        // `overscroll-behavior` drops the remainder either way.
        assert_eq!(consume_axis(100.0, 50.0, 100.0, true), (100.0, 0.0));
        assert_eq!(consume_axis(0.0, -50.0, 100.0, true), (0.0, 0.0));
    }

    /// A 100px-tall `overflow:auto` div with ten lines, over a
    /// 1000px spacer: page content 1100px, viewport maximum 332px.
    fn tall_container_page() -> String {
        let mut page = String::from(
            "<!doctype html><html><head><style>body{margin:0}div.c{height:100px;overflow:auto}</style></head><body><div class=\"c\">",
        );
        for i in 0..10 {
            page.push_str(&format!("<p>line {i}</p>"));
        }
        page.push_str("</div><div style=\"height:1000px\"></div></body></html>");
        page
    }

    /// Stage 10 Rec 2: a wheel past the end of a short page stops
    /// at the page maximum — the viewport never scrolls into blank
    /// space — and a further wheel in the same direction emits
    /// nothing; the zero clamp still holds on the way back up.
    #[tokio::test]
    async fn worker_wheel_stops_at_the_bottom_of_a_short_page() {
        let page = "<!doctype html><html><head><style>body{margin:0}#tall{height:1000px}</style></head><body><div id=\"tall\"></div></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 0);
        // The page is 1000px tall in a 768px viewport: maximum 232.
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, 100000.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - 232.0).abs() < 0.01,
                    "viewport stopped at its 232 maximum, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport after the wheel, got {other:?}"),
        }
        // Already at the maximum: another wheel down consumes
        // nothing and emits nothing.
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, 100000.0);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "wheel past the page maximum must emit nothing"
        );
        // Back up: the whole maximum, then nothing past zero.
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, -100000.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - (-232.0)).abs() < 0.01,
                    "viewport returned to zero, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport after the wheel up, got {other:?}"),
        }
        send_wheel(&cmd_tx, 50.0, 500.0, 0.0, -100000.0);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "wheel past the top must emit nothing"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 10 Rec 2: a wheel past the end of a short nested
    /// container stops at the container's maximum and chains the
    /// leftover to the viewport — which stops at its own maximum;
    /// a further wheel in the same direction emits nothing.
    #[tokio::test]
    async fn worker_wheel_stops_at_the_bottom_of_a_nested_container() {
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            tall_container_page(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        let mut last_epoch = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => epoch,
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        };
        // A wheel down of 100000px inside the container: the
        // container consumes up to its own maximum, the leftover
        // chains to the viewport, which stops at its 332 maximum.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 100000.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => {
                assert!(epoch > last_epoch, "wheel down resubmits");
                last_epoch = epoch;
            }
            other => panic!("expected SubmitDisplayList after wheel down, got {other:?}"),
        }
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - 332.0).abs() < 0.01,
                    "leftover chained to the viewport maximum, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport with the chained leftover, got {other:?}"),
        }
        // Both scrollers are at their maximum: another wheel down
        // consumes nothing anywhere and emits nothing.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 100000.0);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "wheel past both maxima must emit nothing"
        );
        // Wheel up: the container gives back its whole maximum
        // (resubmit) and the viewport its 332.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, -100000.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, .. } => {
                assert!(epoch > last_epoch, "wheel up resubmits");
            }
            other => panic!("expected SubmitDisplayList after wheel up, got {other:?}"),
        }
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - (-332.0)).abs() < 0.01,
                    "viewport returned to zero, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport after the wheel up, got {other:?}"),
        }
        // Back at both minima, another wheel up emits nothing.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, -100000.0);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "wheel past both minima must emit nothing"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 10 Rec 2: `End` jumps to the bottom of the page (the
    /// viewport maximum) and `Home` back to the top; pressing
    /// either again at its end emits nothing.
    #[tokio::test]
    async fn worker_end_and_home_jump_to_the_bottom_and_top() {
        let page = "<!doctype html><html><head><style>body{margin:0}#tall{height:1000px}</style></head><body><div id=\"tall\"></div></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 0);
        let no_modifiers = pluto_ipc::Modifiers::default();
        // The page is 1000px tall in a 768px viewport: maximum 232.
        send_key(&cmd_tx, "End", "End", no_modifiers);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - 232.0).abs() < 0.01,
                    "End jumped to the 232 maximum, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport after End, got {other:?}"),
        }
        // Already at the bottom: End again emits nothing.
        send_key(&cmd_tx, "End", "End", no_modifiers);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "End at the bottom must emit nothing"
        );
        send_key(&cmd_tx, "Home", "Home", no_modifiers);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!(
                    (dy - (-232.0)).abs() < 0.01,
                    "Home returned to zero, got {dy}"
                );
            }
            other => panic!("expected ScrollViewport after Home, got {other:?}"),
        }
        // Already at the top: Home again emits nothing.
        send_key(&cmd_tx, "Home", "Home", no_modifiers);
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(200)).await,
            "Home at the top must emit nothing"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 8 Rec 9: Tab moves keyboard focus between buttons
    /// (`:focus` recolors the focused one, lower on the page each
    /// Tab), and Enter activates the focused button (its click
    /// handler runs).
    #[tokio::test]
    async fn worker_tab_moves_focus_and_enter_activates() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block} button:focus{color:red}</style></head><body>            <button id=\"b1\">One</button><button id=\"b2\">Two</button><span id=\"c\">0</span>            <script>document.getElementById('b1').addEventListener('click', () => {            document.getElementById('c').textContent = 'clicked'; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "One" (3) + "Two" (3) + "0" (1) = 7 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 7);
        let no_modifiers = pluto_ipc::Modifiers::default();
        // Tab: focus lands on the first button — its glyphs turn
        // red. Record their height: the second button sits lower.
        send_key(&cmd_tx, "Tab", "Tab", no_modifiers);
        let first_list = wait_for_glyph_color(
            &mut event_rx,
            [1.0, 0.0, 0.0, 1.0],
            Duration::from_secs(5),
        )
        .await
        .expect("Tab did not focus the first button");
        let first_red_y = glyph_colors(&first_list)
            .iter()
            .filter(|(c, _, _)| *c == [1.0, 0.0, 0.0, 1.0])
            .map(|(_, _, y)| *y)
            .fold(f32::INFINITY, f32::min);
        // Enter: the focused button's click handler runs — the span
        // becomes "clicked" (7 glyphs): 3 + 3 + 7 = 13.
        send_key(&cmd_tx, "Enter", "Enter", no_modifiers);
        assert!(
            wait_for_glyphs(&mut event_rx, 13, Duration::from_secs(5)).await,
            "Enter did not activate the focused button"
        );
        // Tab again: focus moves to the second button — the red
        // glyphs sit lower on the page than the first button's.
        send_key(&cmd_tx, "Tab", "Tab", no_modifiers);
        let second_list = wait_for_glyph_color(
            &mut event_rx,
            [1.0, 0.0, 0.0, 1.0],
            Duration::from_secs(5),
        )
        .await
        .expect("second Tab did not focus the second button");
        let second_red_y = glyph_colors(&second_list)
            .iter()
            .filter(|(c, _, _)| *c == [1.0, 0.0, 0.0, 1.0])
            .map(|(_, _, y)| *y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            second_red_y > first_red_y,
            "focus did not move down: first red y={first_red_y}, second red y={second_red_y}"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 8 Rec 10: an `IntersectionObserver` fires its initial
    /// entry for a target in view — the callback runs as a queued
    /// task after paint (so the text it writes lands on a later
    /// submit), and no second entry arrives while the geometry is
    /// unchanged.
    #[tokio::test]
    async fn worker_intersection_observer_fires_initial_entry() {
        let page = "<!doctype html><html><head><style>body{margin:0}#t{width:100px;height:100px}</style></head><body><div id=\"t\">Target</div><div id=\"out\"></div><script>new IntersectionObserver((es) => { document.getElementById('out').textContent = es[0].isIntersecting ? 'IN' : 'OUT'; }).observe(document.getElementById('t'));</script></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Target" is 6 glyphs; the callback writes "IN" (2 more).
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 6);
        assert!(
            wait_for_glyphs(&mut event_rx, 8, Duration::from_secs(5)).await,
            "IntersectionObserver initial entry did not write IN"
        );
        // Unchanged geometry: no further submit within 300 ms.
        assert!(
            no_event_within(&mut event_rx, Duration::from_millis(300)).await,
            "identical geometry must not deliver a second entry"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 8 Rec 10: scrolling the viewport re-runs the
    /// IntersectionObserver step on the retained tree — a target
    /// that scrolls into view flips `isIntersecting` and the
    /// callback writes the new state.
    #[tokio::test]
    async fn worker_intersection_observer_scroll_changes_the_entry() {
        let page = "<!doctype html><html><head><style>body{margin:0}#sp{height:1000px}#t{width:100px;height:100px}</style></head><body><div id=\"sp\"></div><div id=\"t\">Target</div><div id=\"out\"></div><script>new IntersectionObserver((es) => { document.getElementById('out').textContent = es[0].isIntersecting ? 'IN' : 'OUT'; }).observe(document.getElementById('t'));</script></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Target" is 6 glyphs; out of view the callback writes
        // "OUT" (3 more).
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 6);
        assert!(
            wait_for_glyphs(&mut event_rx, 9, Duration::from_secs(5)).await,
            "IntersectionObserver initial entry did not write OUT"
        );
        // Wheel down 332px over the spacer: no container, so the
        // whole delta chains to the viewport (target 1000..1100
        // enters the 332..1100 visible region). 332, not 500: the
        // page is 1100px tall in a 768px viewport, so the viewport
        // maximum is 332 (Stage 10 Rec 2 clamps anything larger).
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 332.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!((dy - 332.0).abs() < 0.01, "viewport scrolled 332, got {dy}");
            }
            other => panic!("expected ScrollViewport after the wheel, got {other:?}"),
        }
        // The scroll re-ran the step; the callback flips OUT -> IN
        // (2 glyphs instead of 3).
        assert!(
            wait_for_glyphs(&mut event_rx, 8, Duration::from_secs(5)).await,
            "scrolling the target into view did not write IN"
        );
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
            .send(BrowserToRenderer::CommitNavigation {
                url,
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Inc" (3) + "9" (1) = 4 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 4);
        send_click(&cmd_tx, 100.0, 20.0);
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

    /// Stage 12 Rec 7: a pointer click dispatches a *typed*
    /// mouse event — the listener reads `clientX` / `clientY` /
    /// `button` off it, not a bare `click`. The button is a
    /// 200x40 block at (0,0); (100,20) is inside it.
    #[tokio::test]
    async fn worker_typed_mouse_click_carries_client_coordinates() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block;width:200px;height:40px}</style></head><body>            <button id=\"go\">Go</button><span id=\"out\">?</span>            <script>document.getElementById('go').addEventListener('click', (e) => {            document.getElementById('out').textContent = e.clientX + ',' + e.clientY + ',' + e.button; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Go" (2) + "?" (1) = 3 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 3);
        send_click(&cmd_tx, 100.0, 20.0);
        // "Go" (2) + "100,20,0" (8) = 10 glyphs: the click
        // listener saw the mouse coordinates and button.
        assert!(
            wait_for_glyphs(&mut event_rx, 10, Duration::from_secs(5)).await,
            "mouse click did not carry clientX/clientY/button to JS"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 16 Phase 1, Rec 7: a pointer press records user
    /// activation — the click listener (which runs on pointer-up,
    /// after the pointer-down recorded the gesture) reads
    /// `navigator.userActivation.isActive` as true. A key press
    /// does the same: a `keydown` handler reads it live.
    #[tokio::test]
    async fn worker_pointer_and_key_record_user_activation() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block;width:200px;height:40px}</style></head><body>            <button id=\"go\">Go</button><span id=\"out\">?</span>            <script>document.getElementById('go').addEventListener('click', () => {            document.getElementById('out').textContent = navigator.userActivation.isActive ? 'ACTIVE' : 'IDLE'; });            document.addEventListener('keydown', () => {            document.getElementById('out').textContent = navigator.userActivation.isActive ? 'KEY-ACTIVE' : 'KEY-IDLE'; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Go" (2) + "?" (1) = 3 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 3);
        send_click(&cmd_tx, 100.0, 20.0);
        // "Go" (2) + "ACTIVE" (6) = 8 glyphs: the pointer press
        // recorded activation before the click dispatched.
        assert!(
            wait_for_glyphs(&mut event_rx, 8, Duration::from_secs(5)).await,
            "pointer press did not record user activation"
        );
        // The click focused the button, so the keydown bubbles
        // from it to the document listener.
        send_key(&cmd_tx, "a", "KeyA", pluto_ipc::Modifiers::default());
        // "Go" (2) + "KEY-ACTIVE" (10) = 12 glyphs.
        assert!(
            wait_for_glyphs(&mut event_rx, 12, Duration::from_secs(5)).await,
            "key press did not record user activation"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 12 Rec 7: keyboard activation (Enter on a focused
    /// button) dispatches a *typed* keyboard event — the click
    /// listener reads which key activated the element.
    #[tokio::test]
    async fn worker_keyboard_activation_click_carries_key() {
        let page = "<!doctype html><html><head><style>body{margin:0}            button{display:block;width:200px;height:40px}</style></head><body>            <button id=\"go\">Go</button><span id=\"out\">?</span>            <script>document.getElementById('go').addEventListener('click', (e) => {            document.getElementById('out').textContent = e.key || 'nokey'; });</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Go" (2) + "?" (1) = 3 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 3);
        // Tab focuses the button; Enter activates it.
        send_key(&cmd_tx, "Tab", "Tab", pluto_ipc::Modifiers::default());
        send_key(&cmd_tx, "Enter", "Enter", pluto_ipc::Modifiers::default());
        // "Go" (2) + "Enter" (5) = 7 glyphs: the click listener
        // saw the activating key.
        assert!(
            wait_for_glyphs(&mut event_rx, 7, Duration::from_secs(5)).await,
            "keyboard activation did not carry the key to JS"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Multi-route local server with binary bodies (like
    /// `serve_routes`, but the body is bytes — a PNG cannot
    /// round-trip through a `String`).
    async fn serve_bytes_routes(
        listener: TcpListener,
        routes: std::sync::Arc<Vec<(String, String, Vec<u8>)>>,
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
                let (content_type, body): (String, Vec<u8>) = routes
                    .iter()
                    .find(|(p, _, _)| *p == path)
                    .map(|(_, ct, b)| (ct.clone(), b.clone()))
                    .unwrap_or_else(|| {
                        ("text/plain".to_string(), b"not found".to_vec())
                    });
                let status = if body == b"not found" {
                    "404 Not Found"
                } else {
                    "200 OK"
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes()).await;
                let _ = stream.write_all(&body).await;
            });
        }
    }

    /// Encode a small RGBA image to real PNG bytes (same helper
    /// shape as the image store's decode tests — no checked-in
    /// fixture, no hand-built CRCs).
    fn png_2x2_red_bytes() -> Vec<u8> {
        use image::codecs::png::PngEncoder;
        use image::{ExtendedColorType, ImageEncoder};
        let mut out = Vec::new();
        let rgba: Vec<u8> = [255u8, 0, 0, 255].repeat(4);
        PngEncoder::new(&mut out)
            .write_image(&rgba, 2, 2, ExtendedColorType::Rgba8)
            .expect("test PNG encodes");
        out
    }

    /// Stage 12 Rec 7, end to end: a script creates `new
    /// Image()`, attaches `onload` *before* setting `src`, and
    /// the shell-reported load fires the handler. The listener
    /// is attached before the load starts, so there is no race
    /// with the fetch — the `src` set is what starts it.
    #[tokio::test]
    async fn worker_js_image_load_event_fires() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body>            <span id=\"out\">?</span>            <script>const img = new Image();            img.onload = () => { document.getElementById('out').textContent = 'loaded'; };            img.onerror = () => { document.getElementById('out').textContent = 'failed'; };            img.src = '/pic.png';</script>            </body></html>";
        let routes = std::sync::Arc::new(vec![
            ("/".to_string(), "text/html".to_string(), page.as_bytes().to_vec()),
            (
                "/pic.png".to_string(),
                "image/png".to_string(),
                png_2x2_red_bytes(),
            ),
        ]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_bytes_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "?" (1) before the image loads.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 1);
        // "loaded" (6): the `load` event fired on the JS-created
        // image and the handler updated the DOM.
        assert!(
            wait_for_glyphs(&mut event_rx, 6, Duration::from_secs(5)).await,
            "Image() load event did not fire after the shell-reported load"
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Toggle" (6), details empty.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 6);
        let click = || {
            send_click(&cmd_tx, 100.0, 20.0);
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
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
                state_json: None,
                traversal: false,
                scroll: (0.0, 0.0),
            })
            .expect("worker takes the command");
        // DidCommit and the first SubmitDisplayList for page2 may
        // arrive in either order — the commit path and the layout
        // path emit on different tasks, so their order on the
        // channel is not a contract. Accept both orders (and skip
        // any stale page1 resubmit on the way); the point of this
        // test is the timer, not the event order.
        let mut saw_commit = false;
        let mut saw_list = false;
        for _ in 0..4 {
            if saw_commit && saw_list {
                break;
            }
            match next_event(&mut event_rx).await {
                RendererToBrowser::DidCommit { .. } => saw_commit = true,
                RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                    let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                    if glyph_count(&list) == 13 {
                        saw_list = true;
                    }
                    // Any other list is a stale page1 resubmit.
                }
                other => panic!("unexpected event after navigating to page2: {other:?}"),
            }
        }
        assert!(
            saw_commit && saw_list,
            "page2 commit + display list both arrive"
        );
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "safe" (4). If the injected script had run, the text
        // would be "pwned" (5) instead.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 4);
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 10 Rec 6: the permissions round-trip under a
    /// concurrently-built tab. The old fixed temp name let a
    /// tab's shutdown save interleave with a permission-grant
    /// save into one corrupt `permissions.json` (the observed
    /// flake: "trailing characters after JSON value", a grant
    /// read back as Prompt). With a unique temp name per write,
    /// every published profile file parses — even when the tab
    /// and the grant loop save at the same time. The final
    /// content is last-writer-wins, so only parse-validity (never
    /// corruption) is asserted, over several rounds.
    #[tokio::test]
    async fn permissions_round_trip_survives_concurrent_tab_save() {
        let _home = worker_home();
        // A real tab: `TabState::new` reads the security profile,
        // `tab_js()` builds the tab backends, loading permissions.
        let (layout_tx, _layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
        let mut state = TabState::new(layout_tx, tokio::runtime::Handle::current(), 0, &SharedLoader::new(), None, tokio::sync::mpsc::channel(100).0);
        state.tab_js();
        let backends = std::sync::Arc::new(
            state.tab_backends.take().expect("tab backends built"),
        );
        let dir = crate::security::profile_dir().expect("profile dir resolves");
        for round in 0..10 {
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
            let mut handles = Vec::new();
            for w in 0..4 {
                let barrier = std::sync::Arc::clone(&barrier);
                let dir = dir.clone();
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..10 {
                        let origin = pluto_ipc::Origin::from_url(&format!(
                            "https://race{round}-{w}-{i}.example/"
                        ));
                        let mut store = crate::js_host::permissions::load_permissions();
                        store.grant_persistent_storage_at(
                            &origin,
                            std::time::SystemTime::now(),
                        );
                        crate::js_host::permissions::save_permissions(&store);
                        // Every published state must parse: a
                        // corrupt publication is the race.
                        let text = std::fs::read_to_string(
                            dir.join(crate::js_host::permissions::PERMISSIONS_FILE),
                        )
                        .expect("permissions file exists");
                        crate::js_host::storage::parse_json(&text)
                            .expect("published permissions file parses");
                    }
                }));
            }
            // The tab's shutdown save, racing the grants.
            let tab_barrier = std::sync::Arc::clone(&barrier);
            let tab_backends = std::sync::Arc::clone(&backends);
            handles.push(std::thread::spawn(move || {
                tab_barrier.wait();
                for _ in 0..10 {
                    crate::js_host::save_tab_backends(&tab_backends);
                }
            }));
            for h in handles {
                h.join().unwrap();
            }
            let text = std::fs::read_to_string(
                dir.join(crate::js_host::permissions::PERMISSIONS_FILE),
            )
            .expect("permissions file exists");
            crate::js_host::storage::parse_json(&text)
                .expect("permissions file parses after the round");
        }
    }

    #[test]
    fn visited_key_strips_fragment() {
        assert_eq!(visited_key("https://example.com/a#b"), "https://example.com/a");
        assert_eq!(visited_key("https://example.com/a"), "https://example.com/a");
        assert_eq!(visited_key("https://example.com/a#"), "https://example.com/a");
        assert_eq!(visited_key("about:blank"), "about:blank");
    }

    /// Stage 10 Rec 8: a fresh tab seeds its `:visited` set from
    /// the persisted history — a restart (a new `TabState`) sees
    /// the previous session's URLs, fragment-stripped, with no
    /// navigation in between. The `WorkerHomeGuard` holds
    /// `TEST_HOME_LOCK` while the tab reads the profile.
    #[tokio::test]
    async fn visited_urls_seeded_from_persisted_history() {
        let _home = worker_home();
        crate::history::save_history(&[
            "https://example.com/a".to_string(),
            "https://example.com/b#frag".to_string(),
        ]);
        let (layout_tx, _layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
        let state = TabState::new(layout_tx, tokio::runtime::Handle::current(), 0, &SharedLoader::new(), None, tokio::sync::mpsc::channel(100).0);
        assert!(state.visited_urls.contains("https://example.com/a"));
        // The fragment is stripped on seed, as on commit.
        assert!(state.visited_urls.contains("https://example.com/b"));
        assert!(!state.visited_urls.contains("https://example.com/b#frag"));
        assert_eq!(state.visited_urls.len(), 2);
        // And a tab with no persisted history starts unvisited.
        let _ = std::fs::remove_file(
            crate::security::profile_dir()
                .unwrap()
                .join(crate::history::HISTORY_FILE),
        );
        let (layout_tx, _layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
        let fresh = TabState::new(layout_tx, tokio::runtime::Handle::current(), 0, &SharedLoader::new(), None, tokio::sync::mpsc::channel(100).0);
        assert!(fresh.visited_urls.is_empty());
    }

    #[test]
    fn visited_link_ids_match_resolved_hrefs_only() {
        // Stage 10 Rec 3: the worker-side projection of the
        // committed-URL set onto a document's links — absolute
        // hrefs, root-relative hrefs resolved against the base
        // URL, fragment-stripped keys, and href-less anchors
        // excluded.
        let doc = page::parse_document_with_base(
            "<a href=\"https://example.com/a\">A</a><a href=\"/b\">B</a>\
             <a href=\"https://example.com/c#frag\">C</a><a href=\"https://other.example/d\">D</a>\
             <a>No href</a>",
            "https://example.com/index",
        );
        let mut visited = HashSet::new();
        visited.insert("https://example.com/a".to_string());
        visited.insert("https://example.com/b".to_string());
        // The `#frag` link resolves with its fragment; the key
        // strips it, so it counts as visited.
        visited.insert("https://example.com/c".to_string());
        let ids = visited_link_ids(&doc, &visited);
        assert_eq!(ids.len(), 3, "exactly the three visited links match");
        // Every `<a>` in the document is matched iff its resolved
        // href (fragment-stripped) is in the visited set.
        let dom = doc.dom();
        let mut stack = dom.root().map(|r| vec![r]).unwrap_or_default();
        let mut links = 0;
        while let Some(id) = stack.pop() {
            if dom.tag(id) == Some("a") {
                links += 1;
                let href = page::find_link_href(&doc, id);
                let expect = href
                    .as_ref()
                    .is_some_and(|h| visited.contains(visited_key(h)));
                assert_eq!(
                    ids.contains(&id),
                    expect,
                    "link {href:?} visited-ness must follow the set"
                );
            }
            stack.extend(dom.children(id).iter().rev());
        }
        assert_eq!(links, 5, "all five anchors walked");
        // Empty set: nothing matches.
        assert!(visited_link_ids(&doc, &HashSet::new()).is_empty());
    }

    /// Stage 10 Rec 7: the histogram reports correct nearest-rank
    /// p50/p95 on synthetic durations, per phase, plus the
    /// over-budget count — the numbers the 60 FPS verdict reads.
    #[test]
    fn frame_time_histogram_percentiles_on_synthetic_durations() {
        let mut hist = FrameTimeHistogram::new();
        // Empty: every read is None, never a zero masquerading
        // as a measurement.
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.p50(FramePhase::Total), None);
        assert_eq!(hist.p95(FramePhase::Total), None);
        assert_eq!(hist.max(FramePhase::Total), None);
        assert_eq!(hist.over_budget_ratio(), None);
        // Totals 1..=10 ms; layout is half the total, collect a
        // quarter, submit a flat 0.1 ms — distinct per phase, so
        // a crossed wire between phases fails the test.
        for v in 1..=10 {
            hist.record(FrameSample {
                total_ms: v as f32,
                layout_ms: v as f32 * 0.5,
                collect_ms: v as f32 * 0.25,
                submit_ms: 0.1,
                layout_kind: RelayoutKind::NothingToDo,
            });
        }
        assert_eq!(hist.count(), 10);
        assert_eq!(hist.total_recorded(), 10);
        // Nearest-rank, never averages: p50 of 1..=10 is 5.0
        // (an average would say 5.5).
        assert_eq!(hist.p50(FramePhase::Total), Some(5.0));
        assert_eq!(hist.p95(FramePhase::Total), Some(10.0));
        assert_eq!(hist.max(FramePhase::Total), Some(10.0));
        assert_eq!(hist.p50(FramePhase::Layout), Some(2.5));
        assert_eq!(hist.p95(FramePhase::Layout), Some(5.0));
        assert_eq!(hist.p50(FramePhase::Collect), Some(1.25));
        assert_eq!(hist.p95(FramePhase::Collect), Some(2.5));
        assert_eq!(hist.p50(FramePhase::Submit), Some(0.1));
        assert_eq!(hist.p95(FramePhase::Submit), Some(0.1));
        assert_eq!(hist.over_budget(), 0);
        // One 20 ms frame misses the 16.67 ms budget; the rest do
        // not. Non-finite / negative samples are ignored, not
        // counted.
        hist.record(FrameSample {
            total_ms: 20.0,
            layout_ms: 20.0,
            collect_ms: 0.0,
            submit_ms: 0.0,
            layout_kind: RelayoutKind::NothingToDo,
        });
        hist.record(FrameSample {
            total_ms: f32::NAN,
            layout_ms: 0.0,
            collect_ms: 0.0,
            submit_ms: 0.0,
            layout_kind: RelayoutKind::NothingToDo,
        });
        hist.record(FrameSample {
            total_ms: -1.0,
            layout_ms: 0.0,
            collect_ms: 0.0,
            submit_ms: 0.0,
            layout_kind: RelayoutKind::NothingToDo,
        });
        assert_eq!(hist.count(), 11);
        assert_eq!(hist.total_recorded(), 11);
        assert_eq!(hist.over_budget(), 1);
        assert!((hist.over_budget_ratio().unwrap() - 1.0 / 11.0).abs() < 0.001);
        assert_eq!(hist.max(FramePhase::Total), Some(20.0));
    }

    /// Stage 10 Rec 7: the measurement must not itself become
    /// the bottleneck — the storage is one fixed-size,
    /// pre-allocated ring (no `Vec`, no heap field), `record` is
    /// a single array write, and recording far past capacity
    /// rolls the oldest samples off instead of growing.
    #[test]
    fn frame_time_histogram_is_fixed_size_and_never_grows() {
        assert_eq!(FRAME_TIME_SAMPLES, 240, "four seconds at 60 Hz");
        let mut hist = FrameTimeHistogram::new();
        assert_eq!(hist.capacity(), 240);
        // The whole histogram is the ring plus four counters and
        // seven per-kind counters (Stage 14 Rec 9 adds the
        // `CompositingOnly` kind to Stage 12 Rec 3's six): 240
        // samples × 20 bytes, no heap allocation to grow.
        assert_eq!(
            std::mem::size_of::<FrameTimeHistogram>(),
            FRAME_TIME_SAMPLES * std::mem::size_of::<FrameSample>() + 88
        );
        for v in 0..1000 {
            hist.record(FrameSample {
                total_ms: v as f32,
                layout_ms: 0.0,
                collect_ms: 0.0,
                submit_ms: 0.0,
                layout_kind: RelayoutKind::NothingToDo,
            });
        }
        assert_eq!(hist.count(), 240, "the ring caps at capacity");
        assert_eq!(hist.capacity(), 240, "recording never grows it");
        assert_eq!(hist.total_recorded(), 1000, "the count keeps the truth");
        // Samples 760..=999 remain; nearest-rank p50 is index 119.
        assert_eq!(hist.p50(FramePhase::Total), Some(879.0));
        assert_eq!(hist.max(FramePhase::Total), Some(999.0));
    }

    /// Stage 10 Rec 7: the periodic summary line carries the
    /// verdict numbers (total + per-phase percentiles, the
    /// over-budget share) in one line, and renders with no
    /// samples instead of panicking.
    #[test]
    fn frame_time_histogram_summary_names_phases_and_budget() {
        let empty = FrameTimeHistogram::new();
        let line = empty.summary();
        assert!(line.contains("frames n=0"), "empty summary: {line}");
        assert!(line.contains("total p50=-"), "empty summary: {line}");
        let mut hist = FrameTimeHistogram::new();
        hist.record(FrameSample {
            total_ms: 20.0,
            layout_ms: 12.0,
            collect_ms: 7.0,
            submit_ms: 1.0,
            layout_kind: RelayoutKind::ColdBuild,
        });
        let line = hist.summary();
        assert!(line.contains("frames n=1"), "summary: {line}");
        assert!(line.contains("over_budget=1"), "summary: {line}");
        assert!(line.contains("total p50=20.00"), "summary: {line}");
        assert!(line.contains("layout p50=12.00"), "summary: {line}");
        assert!(line.contains("collect p50=7.00"), "summary: {line}");
        assert!(line.contains("submit p50=1.00"), "summary: {line}");
    }

    /// Stage 12 Rec 3: the histogram counts layout kinds per
    /// frame, and the fallback rate is FullLayout /
    /// (SubtreeLayout + FullLayout) — cold builds excluded, so a
    /// fresh navigation does not read as a fallback. The summary
    /// line carries the counts and the rate, and renders `-` for
    /// the rate when no layout-taking frame exists yet.
    #[test]
    fn frame_time_histogram_layout_kinds_and_fallback_rate() {
        let empty = FrameTimeHistogram::new();
        assert_eq!(empty.fallback_rate(), None);
        assert_eq!(empty.layout_kind_count(RelayoutKind::FullLayout), 0);
        assert!(
            empty.summary().contains("fallback=-"),
            "empty summary: {}",
            empty.summary()
        );
        let mut hist = FrameTimeHistogram::new();
        let sample = |kind: RelayoutKind| FrameSample {
            total_ms: 10.0,
            layout_ms: 5.0,
            collect_ms: 3.0,
            submit_ms: 1.0,
            layout_kind: kind,
        };
        // Two cold builds (first layouts), three subtree layouts,
        // one full fallback, one paint-only, one position-only,
        // one scroll frame (nothing-to-do).
        for _ in 0..2 {
            hist.record(sample(RelayoutKind::ColdBuild));
        }
        for _ in 0..3 {
            hist.record(sample(RelayoutKind::SubtreeLayout));
        }
        hist.record(sample(RelayoutKind::FullLayout));
        hist.record(sample(RelayoutKind::PaintOnly));
        hist.record(sample(RelayoutKind::PositionOnly));
        hist.record(sample(RelayoutKind::NothingToDo));
        assert_eq!(hist.count(), 9);
        assert_eq!(hist.layout_kind_count(RelayoutKind::ColdBuild), 2);
        assert_eq!(hist.layout_kind_count(RelayoutKind::SubtreeLayout), 3);
        assert_eq!(hist.layout_kind_count(RelayoutKind::FullLayout), 1);
        assert_eq!(hist.layout_kind_count(RelayoutKind::PaintOnly), 1);
        assert_eq!(hist.layout_kind_count(RelayoutKind::PositionOnly), 1);
        assert_eq!(hist.layout_kind_count(RelayoutKind::NothingToDo), 1);
        // 1 full out of 1 + 3 layout-taking frames — the cold
        // builds do not count on either side.
        assert!((hist.fallback_rate().unwrap() - 0.25).abs() < 0.001);
        let line = hist.summary();
        assert!(line.contains("cold=2"), "summary: {line}");
        assert!(line.contains("subtree=3"), "summary: {line}");
        assert!(line.contains("full=1"), "summary: {line}");
        assert!(line.contains("fallback=25.0%"), "summary: {line}");
        // Ignored samples (non-finite total) never reach the kind
        // counters either.
        hist.record(FrameSample {
            total_ms: f32::NAN,
            layout_ms: 0.0,
            collect_ms: 0.0,
            submit_ms: 0.0,
            layout_kind: RelayoutKind::FullLayout,
        });
        assert_eq!(hist.layout_kind_count(RelayoutKind::FullLayout), 1);
        assert_eq!(hist.total_recorded(), 9);
    }

    /// Stage 10 Rec 7: the instrumented paths still function —
    /// a real `layout_and_submit` through a live layout actor
    /// submits its display list AND records exactly one frame,
    /// and a scroll re-collect (`apply_scrolls`, the wheel path)
    /// records a second frame with a zero layout phase (scrolls
    /// re-collect without re-laying out). This is the tick path
    /// with measurement enabled, minus the timer itself.
    #[tokio::test]
    async fn frame_time_histogram_records_a_real_submitted_frame() {
        let _home = worker_home();
        let (layout_tx, layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
        std::thread::Builder::new()
            .name("pluto-layout".to_string())
            .spawn(move || layout_thread(layout_rx))
            .expect("layout actor thread spawns");
        let mut state = TabState::new(layout_tx, tokio::runtime::Handle::current(), 0, &SharedLoader::new(), None, tokio::sync::mpsc::channel(100).0);
        let (events_tx, mut events_rx) =
            tokio::sync::mpsc::channel::<RendererToBrowser>(100);
        let doc = page::parse_document(EXAMPLE_COM);
        let submitted = layout_and_submit(
            &mut state,
            &events_tx,
            Some(doc),
            false,
            std::time::Instant::now(),
        )
        .await;
        assert!(submitted, "the instrumented build still submits");
        let event = tokio::time::timeout(Duration::from_secs(20), events_rx.recv())
            .await
            .expect("a display list arrives in time")
            .expect("events channel stays open");
        assert!(
            matches!(event, RendererToBrowser::SubmitDisplayList { .. }),
            "the frame reaches the shell, got {event:?}"
        );
        assert_eq!(state.frame_times().count(), 1);
        assert_eq!(state.frame_times().total_recorded(), 1);
        // Stage 12 Rec 3: the first layout is a cold build (no
        // retained tree), and the recorded kind says so — not a
        // fallback, so the fallback rate stays undefined.
        assert_eq!(
            state.frame_times().layout_kind_count(RelayoutKind::ColdBuild),
            1
        );
        assert_eq!(state.frame_times().fallback_rate(), None);
        let p50 = state
            .frame_times()
            .p50(FramePhase::Total)
            .expect("one recorded frame has a p50");
        assert!(p50.is_finite() && p50 > 0.0, "p50 total is a real time: {p50}");
        assert!(
            state.frame_times().p50(FramePhase::Layout).is_some(),
            "the layout phase was attributed"
        );
        assert!(
            state.frame_times().summary().contains("frames n=1"),
            "the summary reads the recorded frame"
        );
        // The wheel path: re-collect with no layout phase.
        let scrolled = apply_scrolls(&mut state, &events_tx, vec![]).await;
        assert!(scrolled, "the instrumented scroll still submits");
        assert_eq!(state.frame_times().count(), 2);
        // Stage 12 Rec 3: the scroll frame records NothingToDo
        // (no layout ran), and the summary carries the kinds.
        assert_eq!(
            state.frame_times().layout_kind_count(RelayoutKind::NothingToDo),
            1
        );
        assert!(
            state.frame_times().summary().contains("cold=1"),
            "summary: {}",
            state.frame_times().summary()
        );
        assert_eq!(
            state.frame_times().p50(FramePhase::Layout),
            Some(0.0),
            "half the samples are scrolls with zero layout time"
        );
        assert!(
            state.frame_times().p50(FramePhase::Collect).is_some(),
            "the scroll's collect phase was attributed"
        );
    }

    /// Stage 13 Rec 2/4: typing into a focused `<input>` inserts
    /// the text (painted value glyphs), fires `input` events the
    /// page counts, fires real `keyup` events on release, and
    /// paints a caret bar while the blink is on.
    #[tokio::test]
    async fn worker_typing_inserts_text_and_fires_input_and_keyup() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body><input id=\"q\"><span id=\"c\">0</span><span id=\"k\">0</span><script>document.getElementById('q').addEventListener('input', () => { const c = document.getElementById('c'); c.textContent = String(Number(c.textContent) + 1); }); document.getElementById('q').addEventListener('keyup', () => { const k = document.getElementById('k'); k.textContent = String(Number(k.textContent) + 1); });</script></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // Two "0" spans = 2 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 2);
        let no_modifiers = pluto_ipc::Modifiers::default();
        // Tab focuses the input; type "hi" (press + release each).
        send_key(&cmd_tx, "Tab", "Tab", no_modifiers);
        send_key(&cmd_tx, "h", "KeyH", no_modifiers);
        send_key_release(&cmd_tx, "h", "KeyH", no_modifiers);
        send_key(&cmd_tx, "i", "KeyI", no_modifiers);
        send_key_release(&cmd_tx, "i", "KeyI", no_modifiers);
        // "hi" (2) + input count "2" (1) + keyup count "2" (1) = 4
        // glyphs, with a caret bar painted (blink just restarted,
        // so it is on).
        let list = wait_for_list(&mut event_rx, Duration::from_secs(5), |list| {
            glyph_count(list) == 4
                && list
                    .0
                    .iter()
                    .any(|item| matches!(item, DisplayItem::Caret { .. }))
        })
        .await
        .expect("typing did not paint 'hi' + counts + caret");
        assert!(
            list.0.iter().any(|item| matches!(
                item,
                DisplayItem::Caret { .. }
            )),
            "caret bar is painted while focused"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 13 Rec 6: Enter in a text input submits the form as
    /// a GET navigation through the loader — the query string
    /// carries the entry list and the results page loads.
    #[tokio::test]
    async fn worker_form_submit_get_navigates_with_query() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body><form action=\"/search\" method=\"get\"><input id=\"q\" name=\"q\"><button>Go</button></form></body></html>";
        let routes = std::sync::Arc::new(vec![
            ("/".to_string(), "text/html".to_string(), page.to_string()),
            (
                "/search?q=pluto".to_string(),
                "text/html".to_string(),
                "<!doctype html><html><head><style>body{margin:0}</style></head><body>FOUND</body></html>".to_string(),
            ),
        ]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Go" = 2 glyphs.
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 2);
        let no_modifiers = pluto_ipc::Modifiers::default();
        send_key(&cmd_tx, "Tab", "Tab", no_modifiers);
        for (key, code) in [
            ("p", "KeyP"),
            ("l", "KeyL"),
            ("u", "KeyU"),
            ("t", "KeyT"),
            ("o", "KeyO"),
        ] {
            send_key(&cmd_tx, key, code, no_modifiers);
        }
        // "Go" (2) + "pluto" (5) = 7 glyphs before submit.
        assert!(
            wait_for_glyphs(&mut event_rx, 7, Duration::from_secs(5)).await,
            "typed value did not paint"
        );
        send_key(&cmd_tx, "Enter", "Enter", no_modifiers);
        // The submission navigates: DidCommit, then the results
        // page ("FOUND" = 5 glyphs).
        assert!(matches!(
            next_event(&mut event_rx).await,
            RendererToBrowser::DidCommit { .. }
        ));
        match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                assert_eq!(glyph_count(&list), 5, "results page paints FOUND");
            }
            other => panic!("expected SubmitDisplayList after submit, got {other:?}"),
        }
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }

    /// Stage 13 Rec 8: Enter on an empty `required` input does
    /// NOT navigate — validation blocks submission before any
    /// navigation, and the form stays put.
    #[tokio::test]
    async fn worker_form_validation_blocks_empty_required_submit() {
        let page = "<!doctype html><html><head><style>body{margin:0}</style></head><body><form action=\"/search\" method=\"get\"><input id=\"q\" name=\"q\" required><button>Go</button></form></body></html>";
        let routes = std::sync::Arc::new(vec![(
            "/".to_string(),
            "text/html".to_string(),
            page.to_string(),
        )]);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_routes(listener, routes));
        let (_home, handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 2);
        let no_modifiers = pluto_ipc::Modifiers::default();
        send_key(&cmd_tx, "Tab", "Tab", no_modifiers);
        send_key(&cmd_tx, "Enter", "Enter", no_modifiers);
        assert!(
            no_commit_within(&mut event_rx, Duration::from_millis(500)).await,
            "empty required input must not navigate"
        );
        cmd_tx
            .send(BrowserToRenderer::Shutdown)
            .expect("worker takes Shutdown");
        handle.await.expect("worker task ends cleanly");
    }


    /// Stage 13 Rec 12b: the bridge watchdog kills a hung renderer.
    /// The child hangs after its handshake (`PLUTO_TEST_RENDERER_HANG`),
    /// so it never sends a heartbeat or any event; with
    /// `PLUTO_WATCHDOG_SECS=2` the bridge must report a watchdog
    /// crash within a few seconds instead of waiting forever.
    /// The env vars are set on this test process (the child inherits
    /// them) and removed afterwards; no other test reads them.
    #[tokio::test]
    async fn bridge_watchdog_kills_hung_renderer() {
        // The child binary: the test executable lives in
        // target/debug/deps, the pluto-browser binary in
        // target/debug. Skip (do not fail) when it was not built
        // — `cargo test --lib` does not build bins.
        let exe = match std::env::current_exe() {
            Ok(path) => path
                .parent()
                .and_then(|deps| deps.parent())
                .map(|debug| debug.join("pluto-browser"))
                .filter(|p| p.exists()),
            Err(_) => None,
        };
        let Some(exe) = exe else {
            eprintln!("[pluto] skipping watchdog test: pluto-browser binary not built");
            return;
        };
        // SAFETY: single-threaded test setup — no other thread
        // reads these vars while this test runs.
        unsafe {
            std::env::set_var(crate::renderer_process::TEST_HANG_ENV, "1");
            std::env::set_var(WATCHDOG_ENV, "2");
        }
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (out_tx, mut out_rx) = mpsc::channel::<crate::Message>(100);
        let bridge = tokio::spawn(async move {
            run_renderer_bridge(77, &exe, cmd_rx, out_tx, false).await
        });
        let crashed = tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                match out_rx.recv().await {
                    Ok(crate::Message::RendererEvent(
                        _,
                        RendererToBrowser::RendererCrashed { reason },
                    )) => break reason,
                    Ok(_) => continue,
                    Err(_) => panic!("bridge output closed without a crash report"),
                }
            }
        })
        .await
        .expect("watchdog reports within 20 s");
        assert!(
            crashed.contains("watchdog"),
            "crash reason names the watchdog: {crashed}"
        );
        // Closing the command channel ends the bridge: the hung
        // child is already dead (killed by the watchdog), so this
        // must return promptly, not hang.
        drop(cmd_tx);
        tokio::time::timeout(Duration::from_secs(10), bridge)
            .await
            .expect("bridge exits after the command channel closes")
            .expect("bridge task joins");
        unsafe {
            std::env::remove_var(crate::renderer_process::TEST_HANG_ENV);
            std::env::remove_var(WATCHDOG_ENV);
        }
    }

    // M6 Phase 3 (Rec 12): `@font-face` source parsing — the first
    // `url(...)` wins, `local(...)` is never fetched, empty urls
    // are skipped.
    #[test]
    fn font_src_url_parsing() {
        assert_eq!(
            parse_font_src_url("url(fonts/body.woff2) format('woff2')"),
            Some("fonts/body.woff2".to_string())
        );
        assert_eq!(
            parse_font_src_url(r#"url("fonts/body.woff2")"#),
            Some("fonts/body.woff2".to_string())
        );
        assert_eq!(
            parse_font_src_url("local('My Font'), url(fallback.woff2)"),
            Some("fallback.woff2".to_string())
        );
        assert_eq!(parse_font_src_url("local('My Font')"), None);
        assert_eq!(parse_font_src_url("url()"), None);
        assert_eq!(parse_font_src_url("url(  ) format('woff2')"), None);
        // An empty first url is skipped, not returned.
        assert_eq!(
            parse_font_src_url("url(), url(second.woff2)"),
            Some("second.woff2".to_string())
        );
    }

    #[test]
    fn font_url_resolution() {
        assert_eq!(
            resolve_font_url("https://example.com/css/main.css", "fonts/a.woff2").unwrap(),
            "https://example.com/css/fonts/a.woff2"
        );
        assert_eq!(
            resolve_font_url("https://example.com/css/main.css", "/fonts/a.woff2").unwrap(),
            "https://example.com/fonts/a.woff2"
        );
        assert_eq!(
            resolve_font_url("https://example.com/css/main.css", "https://cdn.example.com/a.woff2")
                .unwrap(),
            "https://cdn.example.com/a.woff2"
        );
        assert_eq!(
            resolve_font_url("https://example.com/", "data:font/woff2;base64,AAAA").unwrap(),
            "data:font/woff2;base64,AAAA"
        );
        assert!(resolve_font_url("not a url", "fonts/a.woff2").is_err());
    }
}
