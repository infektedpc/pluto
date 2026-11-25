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

use crate::page::{self, Document};
use cosmic::iced::futures::SinkExt;
use cosmic::iced::futures::channel::mpsc;
use pluto_html::NodeId;
use pluto_ipc::{
    BrowserToRenderer, CommandReceiver, DeltaMode, InputEvent, PointerKind, RendererToBrowser,
    WheelPhase,
};
use pluto_ipc::transport::{recv_browser_message, Transport};
use pluto_js::TabJs;
use pluto_layout::{ElementState, LayoutTree, OverscrollBehavior};
use pluto_net::ResourceLoader;
use pluto_paint::{DisplayList, ResourceTable, perf};
use std::cell::Cell;
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

/// The subscription that runs the worker. The shell's `Application`
/// wires this in via `fn subscription`.
pub fn renderer_subscription() -> cosmic::iced::Subscription<crate::Message> {
    cosmic::iced::Subscription::run(worker_stream)
}

/// The worker stream: create the command channel, hand its sender
/// to the shell, then bridge the out-of-process renderer into the
/// shell until the subscription ends (Stage 9 Rec 9).
fn worker_stream() -> impl cosmic::iced::futures::stream::Stream<Item = crate::Message> {
    cosmic::iced::stream::channel(100, async |mut output| {
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let _ = output.send(crate::Message::RendererReady(cmd_tx)).await;
        match std::env::current_exe() {
            Ok(exe) => run_renderer_bridge(&exe, cmd_rx, output).await,
            Err(e) => {
                eprintln!(
                    "[pluto] cannot find own executable ({e}) — falling back to the in-process renderer"
                );
                run_worker(cmd_rx, output).await;
            }
        }
    })
}

/// Bridge one out-of-process renderer into the shell (Stage 9
/// Rec 9): one `RendererProcess` per tab (one tab today; M3 grows
/// this to a pool). Shell commands flow to the child over the
/// socketpair; the child's events flow back as
/// `Message::RendererEvent`.
///
/// If the child dies — clean EOF, a dead socket, or a malformed
/// message the transport treats as compromised — the shell gets
/// `RendererCrashed` (the stale list stays visible, per the
/// shell's existing crash handling) and the bridge stays alive:
/// the next `CommitNavigation` spawns a fresh renderer and the
/// page recovers. Commands other than navigation are dropped
/// while no renderer is alive — there is no page to act on.
///
/// If the child cannot be spawned at all, the bridge falls back
/// to the in-process worker (`run_worker`) so the browser still
/// runs — loudly logged, never silent.
async fn run_renderer_bridge(
    exe: &std::path::Path,
    mut commands: CommandReceiver,
    mut output: mpsc::Sender<crate::Message>,
) {
    let mut renderer = match crate::renderer_process::RendererProcess::spawn(exe).await {
        Ok(renderer) => Some(renderer),
        Err(e) => {
            eprintln!(
                "[pluto] renderer process spawn failed ({e}) — falling back to the in-process renderer"
            );
            run_worker(commands, output).await;
            return;
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
                    match crate::renderer_process::RendererProcess::spawn(exe).await {
                        Ok(mut fresh) => {
                            if fresh.send(&cmd).await.is_err() {
                                report_renderer_crash(
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
                        if output
                            .send(crate::Message::RendererEvent(event))
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
                            &mut output,
                            &mut renderer,
                            &format!("renderer IPC receive failed: {e}"),
                            shutting_down,
                        )
                        .await;
                    }
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
    output: &mut mpsc::Sender<crate::Message>,
    renderer: &mut Option<crate::renderer_process::RendererProcess>,
    reason: &str,
    shutting_down: bool,
) {
    if !shutting_down {
        let _ = output
            .send(crate::Message::RendererEvent(
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
/// Wave 2b adds the geometry jobs (`Geometry` / `FlushGeometry`):
/// the JS host's forced synchronous layout flush (the only
/// sanctioned synchronous cross-thread call in the design) and the
/// click hit-test (`HitTest`). They reply over `std` channels /
/// oneshots the tab thread blocks on, because the tab thread is a
/// plain OS thread, not an async task.
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
        incremental: bool,
        reply: oneshot::Sender<()>,
    },
    /// Build the display list from the actor's retained tree
    /// (after the ResizeObserver step has run).
    CollectDisplayList {
        reply: oneshot::Sender<Arc<DisplayList>>,
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
    /// twice). `None` when the actor holds no tree.
    ScrollMany {
        scrolls: Vec<(NodeId, f32, f32)>,
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

/// One entry of a wheel scroll chain (Stage 8 Rec 9): the scroll
/// container and its `overscroll-behavior` per axis — whether a
/// scroll reaching this container's limit chains onward or stops
/// here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScrollChainEntry {
    pub id: NodeId,
    pub overscroll_x: OverscrollBehavior,
    pub overscroll_y: OverscrollBehavior,
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
/// root would otherwise pay that per pointer event.
fn build_full_tree(
    doc: &Document,
    viewport: (f32, f32),
    state: &ElementState,
    old_tree: Option<LayoutTree>,
) -> LayoutTree {
    let old_fonts = old_tree.and_then(|t| t.font_system);
    let result = match old_fonts {
        Some(fonts) => LayoutTree::build_with_state_reusing_font_system(
            doc.dom(),
            doc.sheet(),
            viewport.0,
            viewport.1,
            state,
            fonts,
        ),
        None => LayoutTree::build_with_state(doc.dom(), doc.sheet(), viewport.0, viewport.1, state),
    };
    match result {
        Ok(tree) => tree,
        Err(e) => {
            let err_doc = page::parse_document(&page::error_document(
                "about:blank",
                &format!("layout failed: {e}"),
            ));
            LayoutTree::build_with_state(
                err_doc.dom(),
                err_doc.sheet(),
                viewport.0,
                viewport.1,
                state,
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
    while let Ok(job) = jobs.recv() {
        match job {
            LayoutJob::LayoutTree {
                document: new_document,
                viewport,
                state,
                offsets,
                incremental,
                reply,
            } => {
                if let Some(doc) = *new_document {
                    document = Some(doc);
                }
                let Some(doc) = document.as_ref() else {
                    let _ = reply.send(());
                    continue;
                };
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
                let mut built_fresh = true;
                if incremental
                    && let Some(retained) = tree.as_mut()
                {
                    match retained.relayout_incremental(
                        doc.dom(),
                        doc.sheet(),
                        viewport.0,
                        viewport.1,
                        &state,
                    ) {
                        Ok(outcome) => {
                            eprintln!(
                                "[pluto] incremental layout: {:?} ({})",
                                outcome.kind, outcome.reason
                            );
                            built_fresh = false;
                        }
                        Err(e) => {
                            eprintln!(
                                "[pluto] incremental layout failed ({e}), full rebuild"
                            );
                        }
                    }
                }
                if built_fresh {
                    // Hand the retained tree's FontSystem to the
                    // rebuild (Stage 8 Rec 9): a fresh one would
                    // re-scan the system fonts per rebuild.
                    let old_tree = tree.take();
                    let mut new_tree = build_full_tree(doc, viewport, &state, old_tree);
                    if !offsets.is_empty() {
                        for (&id, &(x, y)) in &offsets {
                            new_tree.set_scroll_offset(id, x, y);
                        }
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
                let _ = reply.send(());
            }
            LayoutJob::CollectDisplayList { reply } => {
                let display = tree
                    .as_ref()
                    .map(|tree| Arc::new(DisplayList::build(tree)))
                    .unwrap_or_else(|| Arc::new(DisplayList::new()));
                let _ = reply.send(display);
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
            LayoutJob::ScrollMany { scrolls, reply } => {
                let display = tree.as_mut().map(|tree| {
                    for (id, x, y) in scrolls {
                        tree.set_scroll_offset(id, x, y);
                    }
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
// do). Hidden tabs stop producing frames (Stage 8 Rec 1):
// the shell forwards window focus as the visibility signal, the
// tick gate skips relayout while hidden (dirty flags stay set, so
// the first visible tick re-lays out), and JS timers keep running.

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
    /// The storage + permissions backends (Stage 9 Rec 4/11),
    /// created once per tab and shared with the `TabJs`;
    /// persisted when the tab thread ends.
    tab_backends: Option<crate::js_host::TabBackends>,
    /// The current document (re-synced from the host's live `Dom`
    /// on every re-layout, so JS mutations reach the actor).
    document: Option<Document>,
    has_document: bool,
    element_state: ElementState,
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
    viewport: (f32, f32),
    /// The viewport scroll offset, mirrored from the shell (Stage 8
    /// Rec 10): every viewport scroll flows through
    /// `emit_scroll_viewport`, which updates this with the same
    /// ≥0 clamp the shell's `compositor.scroll_by` applies, and
    /// `start_navigation` resets it — so the IntersectionObserver
    /// step can compute viewport-root intersections from the
    /// retained tree without a protocol change. The worker works
    /// entirely in page coordinates; the viewport root is
    /// `[viewport_scroll, viewport]` in page coordinates.
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
        // Stage 9 Rec 5: the tab's loader starts with the
        // persisted HSTS store + HTTPS-only preference/exceptions
        // from the browser profile (defaults when none exists).
        let loader = ResourceLoader::new();
        crate::security::load_into(&loader);
        Self {
            layout_tx,
            loader,
            fetch_handle,
            tab_js: None,
            host,
            fetch_backend: None,
            tab_backends: None,
            document: None,
            has_document: false,
            element_state: ElementState::default(),
            scroll_offsets: HashMap::new(),
            hovered: Vec::new(),
            active_element: None,
            pointer_down_target: None,
            focused: None,
            window_focused: true,
            wheel_chain: Vec::new(),
            wheel_idle_ticks: 0,
            tick_count: 0,
            viewport: page::DEFAULT_VIEWPORT,
            viewport_scroll: (0.0, 0.0),
            tab_started: std::time::Instant::now(),
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
            // Stage 9 Rec 4/11: attach the profile-backed
            // storage + permissions backends (created once per
            // tab; `TabJs::navigate` re-attaches them to the
            // fresh realm on every navigation).
            let backends = crate::js_host::new_tab_backends();
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
    // Stage 9 Rec 12: refresh the tracker lists in the
    // background (EasyList + EasyPrivacy, fetched by default).
    // The refresh works on its own copy — loaded from the
    // profile dir — and swaps it in at the end, so page loads
    // never wait on it and never see an empty blocker
    // mid-refresh; it is freshness-gated (~4 h), so only the
    // first tab in the window actually fetches.
    {
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
    let mut tick = tokio::time::interval(TAB_TICK);
    // `finish_page` blocks this loop for up to a second driving
    // the JS loop after a parse; a Burst interval would then fire
    // dozens of catch-up ticks before the next command is read.
    // Delay skips the backlog instead.
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                state.tick_count += 1;
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
                    || state.host.lock().unwrap().dom_is_dirty();
                if visible && dirty {
                    state.render_requested.set(false);
                    relayout_and_submit(&mut state, &events).await;
                } else if !visible && dirty {
                    // Perf (Stage 8 Rec 5): a hidden tab that
                    // would have re-laid out counts the skip.
                    if let Ok(mut counters) = perf::global().lock() {
                        counters.frames_hidden_skipped += 1;
                    }
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
        }
    }
    // Stage 9 Rec 4/11: the tab thread is ending (tab close /
    // shutdown) — persist the tab's storage + permissions
    // backends to the profile.
    if let Some(backends) = state.tab_backends.take() {
        crate::js_host::save_tab_backends(&backends);
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
        BrowserToRenderer::CommitNavigation { url } => {
            start_navigation(state, url);
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
        BrowserToRenderer::ResourceEvent(event) => {
            // The M5 shape: the shell streams fetch events back
            // to the renderer. Today the worker fetches through
            // its own loader, which consumes the events
            // internally, so nothing arrives here — ignore it if
            // it ever does.
            eprintln!("[pluto] ignoring ResourceEvent from shell: {event:?}");
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
    let threads = spawn_tab_threads();
    let tab_cmd_tx = threads.tab_cmd_tx;
    let mut tab_event_rx = threads.tab_event_rx;
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

/// The channels to a freshly spawned tab-thread pair (layout actor
/// + tab thread), shared by the in-process worker ([`run_worker`])
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
/// and the actor ends when its job channel closes with it.
fn spawn_tab_threads() -> TabThreads {
    let (layout_tx, layout_rx) = std::sync::mpsc::channel::<LayoutJob>();
    std::thread::Builder::new()
        .name("pluto-layout".to_string())
        .spawn(move || layout_thread(layout_rx))
        .expect("layout actor thread spawns");
    let (tab_cmd_tx, tab_cmd_rx) = tokio::sync::mpsc::channel::<BrowserToRenderer>(100);
    let (tab_event_tx, tab_event_rx) = tokio::sync::mpsc::channel::<RendererToBrowser>(100);
    let layout_tx2 = layout_tx.clone();
    std::thread::Builder::new()
        .name("pluto-tab".to_string())
        .spawn(move || tab_thread_main(tab_cmd_rx, tab_event_tx, layout_tx2))
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
pub(crate) async fn run_worker_transport<T: Transport>(mut transport: T) -> i32 {
    let threads = spawn_tab_threads();
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

/// Start a navigation: cancel the previous load (its token and its
/// task — a slow fetch no longer blocks the command loop), get a
/// fresh realm for the new document, then spawn the load as a
/// `spawn_local` task on the tab thread's runtime. The old page
/// stays visible until the new list lands; a stale outcome is
/// discarded by epoch in `handle_nav_outcome`.
fn start_navigation(state: &mut TabState, url: String) {
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
    let token = CancellationToken::new();
    state.nav_cancel = Some(token.clone());
    // Stage 9 Rec 15: the navigation's referrer is the current
    // document (link-click behaviour). The loader keeps it as the
    // referrer context until the new document's response
    // re-points it; with no document yet (first load) no Referer
    // is sent. (Typed-URL navigations cannot be told apart from
    // link navigations here — see `service::referer_for_request`.)
    if let Some(doc) = &state.document {
        state.loader.set_referrer_document(doc.base_url());
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
    // The shell resets its scroll on commit; mirror it here (the
    // viewport scroll is mirrored, not messaged — see
    // `viewport_scroll`).
    state.viewport_scroll = (0.0, 0.0);
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
    // Stage 9 Rec 5: persist the HSTS store (entries may have
    // been recorded during this navigation's fetches) and the
    // HTTPS-only state after every completed navigation.
    crate::security::save_from(&state.loader);
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
            layout_and_submit(state, events, Some(*document), false).await;
        }
        LoadOutcome::Failed { url, reason, .. } => {
            eprintln!("[pluto] worker fetch failed: {reason}");
            let document = page::parse_document(&page::error_document(&url, &reason));
            state.document = Some(document.clone());
            state.scroll_offsets.clear();
            layout_and_submit(state, events, Some(document), false).await;
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
async fn layout_and_submit(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    document: Option<Document>,
    incremental: bool,
) -> bool {
    // Perf (Stage 8 Rec 5): the whole build — layout tree,
    // ResizeObserver step, IntersectionObserver step, display-list
    // collect — is the worker build span. The submit's postcard
    // encode is measured separately, in `submit`.
    let build_started = std::time::Instant::now();
    // (1) Layout tree on the actor (no display list yet).
    let (tree_tx, tree_rx) = oneshot::channel();
    let job = LayoutJob::LayoutTree {
        document: Box::new(document),
        viewport: state.viewport,
        state: state.element_state.clone(),
        offsets: state.scroll_offsets.clone(),
        incremental,
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
    if tree_rx.await.is_err() {
        return emit_tab(
            events,
            RendererToBrowser::RendererCrashed {
                reason: "layout actor reply lost".to_string(),
            },
        )
        .await;
    }
    state.has_document = true;
    // (2) ResizeObserver step, after layout, before the list.
    run_resize_observer_step(state).await;
    // (2b) IntersectionObserver step, after layout, before the
    // list: the geometry is computed here, but the callbacks are a
    // queued task — they run after the submit, on a later turn.
    run_intersection_observer_step(state).await;
    // (3) Display list from the retained tree.
    let (list_tx, list_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::CollectDisplayList { reply: list_tx })
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
        Ok(display) => {
            perf::record_span(
                perf::SpanKind::WorkerBuild,
                build_started.elapsed().as_secs_f32() * 1000.0,
            );
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
    js.run_intersection_observers_with_geometry(&geometry, time_ms);
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
    // Same-lineage snapshot of the live Dom: the actor may lay
    // out incrementally on its retained tree (Stage 8 Rec 6).
    layout_and_submit(state, events, Some(document), true).await;
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
    if reach.descendants || reach.siblings || reach.ancestors {
        if let Some(doc) = state.document.as_ref() {
            let dom = doc.dom();
            for &id in changed {
                if reach.descendants {
                    collect_descendants(dom, id, &mut dirty);
                }
                if reach.siblings {
                    if let Some(parent) = dom.get(id).and_then(|n| n.parent) {
                        for &sib in dom.children(parent) {
                            dirty.insert(sib);
                        }
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
    }
    dirty.into_iter().collect()
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
    {
        let mut host = state.host.lock().unwrap();
        host.set_element_state(state.element_state.clone());
        host.mark_dirty(dirty);
    }
    relayout_and_submit(state, events).await;
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
    let Some(target) = hit_test(state, x, y).await else {
        return;
    };
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
        js.dispatch_event(target, "click", true);
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

/// Apply several container scrolls at once and submit the single
/// re-collected display list (Stage 8 Rec 9). False when the actor
/// is gone or holds no tree.
async fn apply_scrolls(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    scrolls: Vec<(NodeId, f32, f32)>,
) -> bool {
    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .layout_tx
        .send(LayoutJob::ScrollMany {
            scrolls: scrolls.clone(),
            reply: reply_tx,
        })
        .is_err()
    {
        return false;
    }
    match reply_rx.await {
        Ok(Some(display)) => {
            for (id, x, y) in scrolls {
                state.scroll_offsets.insert(id, (x, y));
            }
            submit(state, events, display).await
        }
        _ => false,
    }
}

/// Apply one axis of a wheel delta to a container offset (Stage 8
/// Rec 9). Returns (new offset, remaining delta). A negative delta
/// consumes the offset down to zero; the remainder chains onward —
/// unless the container's `overscroll-behavior` stops the chain,
/// in which case the remainder is dropped. A positive delta is
/// always fully consumed: the tree exposes no per-container
/// content height yet, so there is no bottom limit to hit (the
/// same documented gap as the shell scroll).
fn consume_axis(offset: f32, delta: f32, stops: bool) -> (f32, f32) {
    let next = (offset + delta).max(0.0);
    let consumed = next - offset;
    let mut remaining = delta - consumed;
    if stops {
        remaining = 0.0;
    }
    (next, remaining)
}

/// Emit a `ScrollViewport` delta and mirror it in
/// `state.viewport_scroll` (Stage 8 Rec 10): every viewport scroll
/// flows through here, with the same ≥0 clamp the shell's
/// `compositor.scroll_by` applies — so the worker's mirror of the
/// shell scroll stays exact, and the IntersectionObserver step can
/// use it without a protocol change. Callers re-run the
/// IntersectionObserver step after a scroll (the geometry changed
/// even though nothing re-laid out).
async fn emit_scroll_viewport(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    dx: f32,
    dy: f32,
) {
    state.viewport_scroll = (
        (state.viewport_scroll.0 + dx).max(0.0),
        (state.viewport_scroll.1 + dy).max(0.0),
    );
    emit_tab(
        events,
        RendererToBrowser::ScrollViewport { dx, dy },
    )
    .await;
}

/// A wheel event (Stage 8 Rec 9): the worker owns the whole wheel.
/// The delta is scaled by mode (lines × 16 px, pages × the
/// viewport size, pixels as-is); a new gesture picks the
/// innermost-first container chain under the cursor, and the delta
/// walks it — each container consumes what it can, the leftover
/// chains onward, and the viewport takes the rest (via
/// `ScrollViewport`, which the shell clamps at zero).
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
        let (nx, rem_x) = consume_axis(ox, rx, entry.overscroll_x.stops_chaining());
        let (ny, rem_y) = consume_axis(oy, ry, entry.overscroll_y.stops_chaining());
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

/// A key press (Stage 8 Rec 9). Releases do nothing in M2 (no
/// `keyup` listeners yet), and keys are ignored while the window
/// is unfocused — the shell filters to uncaptured keys only, so
/// this is the worker-side half. Repeats re-fire the action
/// (holding an arrow scrolls repeatedly).
async fn handle_key(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    key: &str,
    pressed: bool,
    modifiers: pluto_ipc::Modifiers,
) {
    if !pressed || !state.window_focused {
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
        // The shell clamps the viewport scroll at zero, so a huge
        // negative delta is "scroll to the top". `End` needs the
        // page's content height, which the tree does not expose
        // yet — skipped (the same documented gap as the bottom
        // clamp).
        "Home" => {
            emit_scroll_viewport(state, events, 0.0, -1e9).await;
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

/// Move keyboard focus (Stage 8 Rec 9): Tab forward, Shift+Tab
/// backward, wrapping at the ends. The old and new elements (plus
/// the new element's ancestors, for `:focus-within`) are marked
/// dirty and restyled.
async fn move_focus(
    state: &mut TabState,
    events: &tokio::sync::mpsc::Sender<RendererToBrowser>,
    forward: bool,
) {
    let Some(doc) = state.document.as_ref() else {
        return;
    };
    let focusable = focusable_in_order(doc);
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
    let old = state.focused.replace(next);
    state.element_state.focus = [next].into_iter().collect();
    let mut changed: Vec<NodeId> = vec![next];
    if let Some(old) = old {
        changed.push(old);
    }
    let dirty = restyle_dirty_set(state, &changed, pseudo_reach(state, "focus"));
    restyle(state, events, &dirty).await;
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
                js.dispatch_event(focused, "click", true);
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
            let (nx, ny) = ((ox + dx).max(0.0), (oy + dy).max(0.0));
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
    if state.has_document {
        eprintln!("[pluto] worker viewport resized to {width}x{height}, re-layout");
        layout_and_submit(state, events, None, true).await;
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
        let (mut browser, renderer) = channel_transport_pair();
        let handle = tokio::spawn(run_worker_transport(renderer));
        browser
            .send_message(&BrowserToRenderer::CommitNavigation {
                url: "about:blank".to_string(),
            })
            .await
            .expect("navigation sends");
        // The error page arrives as a display list first, then the
        // reason — the same order as the in-process worker.
        let first = tokio::time::timeout(
            Duration::from_secs(20),
            recv_renderer_message(&mut browser),
        )
        .await
        .expect("first reply arrives in time")
        .expect("recv succeeds")
        .expect("a message arrives");
        match first {
            RendererToBrowser::SubmitDisplayList { bytes, .. } => {
                let list = DisplayList::from_bytes(&bytes).expect("list bytes decode");
                assert!(!list.is_empty(), "error page still paints");
            }
            other => panic!("expected SubmitDisplayList first, got {other:?}"),
        }
        let second = tokio::time::timeout(
            Duration::from_secs(20),
            recv_renderer_message(&mut browser),
        )
        .await
        .expect("second reply arrives in time")
        .expect("recv succeeds")
        .expect("a message arrives");
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

    /// Stage 9 Rec 9: when the renderer process cannot be spawned
    /// at all, the bridge falls back to the in-process worker —
    /// the browser still runs (loudly logged), and the navigation
    /// still produces renderer events.
    #[tokio::test]
    async fn renderer_bridge_falls_back_to_in_process_worker_when_spawn_fails() {
        let (cmd_tx, cmd_rx) = pluto_ipc::command_channel();
        let (output_tx, mut output_rx) = mpsc::channel::<crate::Message>(100);
        let handle = tokio::spawn(run_renderer_bridge(
            std::path::Path::new("/nonexistent/pluto-browser"),
            cmd_rx,
            output_tx,
        ));
        cmd_tx
            .send(BrowserToRenderer::CommitNavigation {
                url: "about:blank".to_string(),
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

    async fn next_event(rx: &mut mpsc::Receiver<crate::Message>) -> RendererToBrowser {
        match rx.next().await.expect("worker emits an event") {
            crate::Message::RendererEvent(event) => event,
            other => panic!("expected a RendererEvent, got {other:?}"),
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
    /// 100px-tall `overflow:auto` div with ten lines inside it.
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
                    "<!doctype html><html><head><style>body{margin:0}\n                     div{height:100px;overflow:auto;overscroll-behavior:",
                );
                body.push_str(behavior);
                body.push_str("}</style></head><body><div>");
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
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
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
        // Wheel down 48px inside the container: the container
        // scrolls (new epoch), nothing chains to the viewport.
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 48.0);
        let (epoch, _) = match next_event(&mut event_rx).await {
            RendererToBrowser::SubmitDisplayList { epoch, bytes, .. } => (epoch, bytes),
            other => panic!("expected SubmitDisplayList after wheel down, got {other:?}"),
        };
        assert!(epoch > first_epoch, "wheel down resubmits");
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
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
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
        let (handle, cmd_tx, mut event_rx) = spawn_worker();
        let url = format!("http://127.0.0.1:{}/", addr.port());
        // "Target" is 6 glyphs; out of view the callback writes
        // "OUT" (3 more).
        assert_eq!(load_and_count(&cmd_tx, &mut event_rx, url).await, 6);
        assert!(
            wait_for_glyphs(&mut event_rx, 9, Duration::from_secs(5)).await,
            "IntersectionObserver initial entry did not write OUT"
        );
        // Wheel down 500px over the spacer: no container, so the
        // whole delta chains to the viewport (target 1000..1100
        // enters the 500..1268 visible region).
        send_wheel(&cmd_tx, 50.0, 50.0, 0.0, 500.0);
        match next_event(&mut event_rx).await {
            RendererToBrowser::ScrollViewport { dx, dy } => {
                assert_eq!(dx, 0.0);
                assert!((dy - 500.0).abs() < 0.01, "viewport scrolled 500, got {dy}");
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
