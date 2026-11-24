//! Browser <-> renderer message protocol (Stage 4 #1).
//!
//! The protocol is the seam that makes "threads now, processes
//! later" real: the same message enums travel over channels
//! in-process today and over `ipc-channel`/Unix sockets at M5, when
//! the per-tab worker thread is promoted to a process. Nothing in
//! the protocol knows which transport carries it.
//!
//! Messages are versioned serde enums. [`encode_message`] prefixes
//! the payload with [`PROTOCOL_VERSION`] (a little-endian `u32`);
//! [`decode_message`] rejects bytes from another version instead of
//! misreading them. Bump `PROTOCOL_VERSION` when a message changes
//! shape, and keep the old shape decodable only if a mixed-version
//! fleet ever needs it (Pluto ships one binary, so a bump is a
//! clean break — the check exists to fail loudly, not to migrate).
//!
//! The display list itself travels inside
//! [`RendererToBrowser::SubmitDisplayList`] as postcard bytes plus
//! the [`ResourceTable`] its image/blob ids resolve against — the
//! same bytes the shell validates ([`DisplayList::validate`]) before
//! installing. The list is *not* re-serialised per hop: the worker
//! serialises once, the shell deserialises once.

use pluto_paint::ResourceTable;
use serde::{Deserialize, Serialize};

/// The protocol version every encoded message carries. Bump on any
/// breaking change to the enums below.
///
/// v2 (Stage 5) froze `RequestResource` v1: destination, priority,
/// mode, credentials, origin and top-level site, plus the streaming
/// `ResourceEvent` reply and `CancelResource`. v1 bytes are rejected
/// by [`decode_message`] — Pluto ships one binary, so a bump is a
/// clean break, not a migration.
pub const PROTOCOL_VERSION: u32 = 2;

/// What went wrong decoding a versioned message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    #[error("message is empty")]
    Empty,
    #[error("unsupported protocol version {found} (expected {PROTOCOL_VERSION})")]
    UnsupportedVersion { found: u32 },
    #[error("message bytes do not decode: {0}")]
    Decode(String),
}

/// Encode a message as `[version:u32 LE][postcard bytes]`.
pub fn encode_message<T: Serialize>(message: &T) -> Vec<u8> {
    let mut out = PROTOCOL_VERSION.to_le_bytes().to_vec();
    out.extend_from_slice(&postcard::to_allocvec(message).expect("message serialises"));
    out
}

/// Decode bytes produced by [`encode_message`], rejecting another
/// version before touching the payload.
pub fn decode_message<T>(bytes: &[u8]) -> Result<T, ProtocolError>
where
    T: for<'de> Deserialize<'de>,
{
    if bytes.len() < 4 {
        return Err(ProtocolError::Empty);
    }
    let version = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"));
    if version != PROTOCOL_VERSION {
        return Err(ProtocolError::UnsupportedVersion { found: version });
    }
    postcard::from_bytes(&bytes[4..]).map_err(|e| ProtocolError::Decode(e.to_string()))
}

/// Messages from the browser (shell) to the renderer worker.
/// The browser owns navigation, input, and the viewport; the
/// renderer owns the document and never touches a socket, the disk,
/// or the GPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BrowserToRenderer {
    /// Start loading this URL: fetch, parse, lay out at the current
    /// viewport, and report back with `DidCommit` + `SubmitDisplayList`.
    CommitNavigation { url: String },
    /// The layout viewport changed (window resize, first frame).
    /// Re-lay out the retained document; no refetch.
    Resize { width: f32, height: f32 },
    /// An input event for the page. Routing (hit-testing, focus)
    /// lands in M2; the vocabulary exists now so the protocol does
    /// not need a breaking change then.
    InputEvent(InputEvent),
    /// A streamed resource reply for a `RequestResource` ask.
    /// `pluto-net` sends `ResponseHeaders` once, then `BodyChunk`s
    /// as bytes arrive, then exactly one terminal event (`Done`,
    /// `Error` or `Cancelled`). The renderer never sees a socket.
    ResourceEvent(ResourceEvent),
    /// Shut the worker down. The shell sends this when the tab closes.
    Shutdown,
}

/// Messages from the renderer worker to the browser (shell).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RendererToBrowser {
    /// A new display list is ready: postcard bytes of the
    /// [`DisplayList`](pluto_paint::DisplayList) plus the resource
    /// table its image/blob ids resolve against. `epoch` is the
    /// worker's monotonically increasing frame id; the shell only
    /// installs a list newer than the one it holds, so a slow worker
    /// can never resurrect a stale frame.
    SubmitDisplayList {
        epoch: u64,
        bytes: Vec<u8>,
        resources: ResourceTable,
    },
    /// Navigation committed: the URL that actually loaded (after
    /// redirects) and the status line text. Sent when the fetch
    /// lands, before the display list.
    DidCommit { url: String, status: String },
    /// A load failed; the worker already rendered the error page and
    /// sent it as a `SubmitDisplayList` — this carries the reason
    /// for the status line.
    LoadFailed { url: String, reason: String },
    /// The page `<title>` changed.
    TitleChanged { title: String },
    /// The worker died or gave up; the shell shows an error page.
    RendererCrashed { reason: String },
    /// The renderer needs a resource it may not fetch itself
    /// (external stylesheet, image, script — M2). The browser
    /// fetches and answers with `BrowserToRenderer::ResourceEvent`
    /// messages; `request_id` matches the answer to the ask.
    /// Frozen as v1 in Stage 5: every field of [`ResourceRequest`]
    /// is needed by a later layer (cache, cookies, CORS, CSP,
    /// preload scanner), so the shape is fixed now, before Stage 6's
    /// parser depends on it.
    RequestResource(ResourceRequest),
    /// Cancel an in-flight `RequestResource`. The browser answers
    /// with a terminal `ResourceEvent::Cancelled` for that id.
    CancelResource { request_id: u64 },
}

/// A resource ask, frozen as v1 in Stage 5. This is the payload of
/// [`RendererToBrowser::RequestResource`] and the parameter of
/// `pluto-net`'s fetch API — one shape, so the wire format and the
/// service API can never drift apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub request_id: u64,
    /// Which tab asked (0 while Pluto is single-tab).
    pub tab_id: u64,
    /// The navigation epoch the ask belongs to. A reply whose epoch
    /// is no longer current is stale and must be dropped.
    pub epoch: u64,
    pub url: String,
    pub destination: Destination,
    pub priority: Priority,
    pub mode: RequestMode,
    pub credentials: CredentialsMode,
    /// The document origin making the ask, e.g.
    /// `https://example.com` — drives the mixed-content gate and,
    /// at M5, CORS.
    pub origin: String,
    /// The top-level site the ask belongs to (the document origin
    /// for now; proper eTLD+1 partitioning is an M5 refinement).
    /// Cache keys and cookie jars are partitioned on this from day
    /// one.
    pub top_level_site: String,
    /// Honour `upgrade-insecure-requests`: when true, http
    /// subresources on an https page are upgraded to https instead
    /// of going through the mixed-content gate. Set by the
    /// document's CSP at M10; false until then.
    pub upgrade_insecure_requests: bool,
}

impl ResourceRequest {
    /// A top-level navigation ask for `url`: highest priority, the
    /// document destination, navigation mode. `origin` and
    /// `top_level_site` are the URL's own origin.
    pub fn navigate(request_id: u64, tab_id: u64, epoch: u64, url: &str) -> Self {
        let origin = origin_of(url);
        Self {
            request_id,
            tab_id,
            epoch,
            url: url.to_string(),
            destination: Destination::Document,
            priority: Priority::VeryHigh,
            mode: RequestMode::Navigate,
            credentials: CredentialsMode::SameOrigin,
            origin: origin.clone(),
            top_level_site: origin,
            upgrade_insecure_requests: false,
        }
    }
}

/// The origin (`scheme://authority`) of a URL string, or the string
/// itself when it does not parse (the fetch then fails as an invalid
/// URL, as before).
pub fn origin_of(url: &str) -> String {
    let Ok(uri) = url.parse::<http::Uri>() else {
        return url.to_string();
    };
    match (uri.scheme_str(), uri.authority()) {
        (Some(scheme), Some(authority)) => format!("{scheme}://{authority}"),
        _ => url.to_string(),
    }
}

/// What kind of resource a [`RendererToBrowser::RequestResource`]
/// is asking for. The destination drives priority, byte caps, the
/// mixed-content gate, and (at M5) the CORS mode — it is not a
/// label, it is load-bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Destination {
    /// A top-level navigation. Only the browser decides what a
    /// status code means for a navigation (commit, error page,
    /// download); the renderer only asks for the bytes.
    Document,
    Stylesheet,
    Script,
    Image,
    Font,
    /// A `fetch()` call (Stage 7). CORS-gated at M5.
    Fetch,
    /// Anything else (e.g. a future destination the parser does
    /// not know yet). Gets the most conservative caps.
    Other,
}

/// Fetch priority, mirroring Chromium's five bands
/// (VeryHigh/High/Medium/Low/VeryLow). The scheduler in `pluto-net`
/// orders by (priority, discovery order); `fetchpriority` maps onto
/// this directly in Stage 6. A hint, not a guarantee — bandwidth is
/// still subject to connection limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
}

impl Priority {
    /// Whether this priority is delayable: held behind
    /// non-delayable work while the scheduler is saturated.
    /// Prefetch and below-fold images are delayable; the document
    /// and parser-blocking CSS/JS are not.
    pub fn is_delayable(self) -> bool {
        matches!(self, Priority::Low | Priority::VeryLow)
    }
}

/// The fetch mode of a request: whether it may cross origins, and
/// whether that crossing needs a CORS check. Only `Navigate` and
/// `NoCors` are used before `fetch()` arrives (Stage 7); the rest
/// exist now so the wire shape does not change then.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestMode {
    /// A top-level navigation. Same-origin policy does not apply.
    Navigate,
    /// A classic subresource load (stylesheet, script, image).
    /// Cross-origin is allowed; the response is opaque to script.
    NoCors,
    /// Same-origin only; cross-origin fails before the network.
    SameOrigin,
    /// A CORS fetch: cross-origin allowed only if the response
    /// passes the CORS check (M5).
    Cors,
}

/// Whether credentials (cookies, and later `Authorization`) travel
/// with the request. The cookie jar itself is M5; until then only
/// `SameOrigin` and `Omit` have any effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialsMode {
    Omit,
    SameOrigin,
    Include,
}

/// The streaming reply to a `RequestResource`, sent as
/// [`BrowserToRenderer::ResourceEvent`]. Exactly one terminal event
/// (`Done`, `Error` or `Cancelled`) closes each `request_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceEvent {
    /// Response headers landed. Sent once, before any chunk.
    /// `headers` are filtered — `Set-Cookie` is stripped before the
    /// renderer ever sees it (the jar, M5, is the only place that
    /// reads it). `status` is the raw HTTP status as a number.
    /// `url` is the URL the bytes actually came from, after
    /// redirects — the caller reports it as the final URL.
    ResponseHeaders {
        request_id: u64,
        status: u16,
        url: String,
        headers: Vec<(String, String)>,
        content_length: Option<u64>,
    },
    /// The next bytes of the (decoded) body.
    BodyChunk { request_id: u64, bytes: Vec<u8> },
    /// The body is complete.
    Done { request_id: u64 },
    /// The fetch failed. `code` is machine-readable
    /// (`invalid_url`, `too_many_redirects`, `insecure_redirect`,
    /// `too_large`, `timeout`, `cancelled`, `request`, `body`);
    /// `reason` is human-readable. The renderer renders an error
    /// page or a broken-resource placeholder.
    Error {
        request_id: u64,
        code: String,
        reason: String,
    },
    /// The fetch was cancelled (renderer `CancelResource`,
    /// navigation epoch bump, tab close, or timeout).
    Cancelled { request_id: u64 },
}

/// An input event for the page, in page coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputEvent {
    /// Mouse click at page coordinates.
    Click { x: f32, y: f32 },
    /// Mouse wheel scroll, in lines (dx, dy).
    WheelScrolled { dx: f32, dy: f32 },
    /// A key was pressed (key name as the shell reports it).
    KeyPressed { key: String },
}

/// The in-process transport for the app -> worker direction: an
/// unbounded tokio mpsc channel carrying [`BrowserToRenderer`]
/// messages. Unbounded so a send from the shell's `update` never
/// blocks the UI thread; the worker drains it in order. The worker
/// -> app direction is the iced subscription the worker runs as —
/// the subscription *is* that channel in iced, so there is no
/// second queue to keep in sync.
pub type CommandSender = tokio::sync::mpsc::UnboundedSender<BrowserToRenderer>;
pub type CommandReceiver = tokio::sync::mpsc::UnboundedReceiver<BrowserToRenderer>;

/// Create the app -> worker command channel.
pub fn command_channel() -> (CommandSender, CommandReceiver) {
    tokio::sync::mpsc::unbounded_channel()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_css::Stylesheet;
    use pluto_html::parse as parse_html;
    use pluto_layout::LayoutTree;
    use pluto_paint::{DisplayItem, DisplayList, Resource, ResourceId};

    fn example_list() -> DisplayList {
        let html = "<html><head><style>body { background-color: #f0f0f2; }</style></head><body><p>Hello</p></body></html>";
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("body { background-color: #f0f0f2; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        DisplayList::build(&tree)
    }

    #[test]
    fn protocol_version_is_two() {
        assert_eq!(PROTOCOL_VERSION, 2);
    }

    #[test]
    fn browser_messages_round_trip_through_versioned_bytes() {
        let messages = [
            BrowserToRenderer::CommitNavigation {
                url: "https://example.com".to_string(),
            },
            BrowserToRenderer::Resize {
                width: 800.0,
                height: 600.0,
            },
            BrowserToRenderer::InputEvent(InputEvent::Click { x: 10.0, y: 20.0 }),
            BrowserToRenderer::InputEvent(InputEvent::WheelScrolled { dx: 0.0, dy: 3.0 }),
            BrowserToRenderer::InputEvent(InputEvent::KeyPressed {
                key: "Enter".to_string(),
            }),
            BrowserToRenderer::Shutdown,
        ];
        for message in messages {
            let bytes = encode_message(&message);
            assert_eq!(
                u32::from_le_bytes(bytes[..4].try_into().unwrap()),
                PROTOCOL_VERSION
            );
            let back: BrowserToRenderer = decode_message(&bytes).expect("browser message decodes");
            assert_eq!(
                postcard::to_allocvec(&back).unwrap(),
                postcard::to_allocvec(&message).unwrap()
            );
        }
    }

    #[test]
    fn renderer_messages_round_trip_through_versioned_bytes() {
        let list = example_list();
        let mut resources = ResourceTable::new();
        let id = ResourceId(1);
        resources.insert(
            id,
            Resource::Image {
                width: 1,
                height: 1,
                rgba: vec![255, 0, 0, 255],
            },
        );
        let mut list_with_image = list.clone();
        list_with_image.push(DisplayItem::Image {
            resource: id,
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 16.0,
        });
        let submit = RendererToBrowser::SubmitDisplayList {
            epoch: 42,
            bytes: list_with_image.to_bytes(),
            resources,
        };
        let bytes = encode_message(&submit);
        let back: RendererToBrowser = decode_message(&bytes).expect("renderer message decodes");
        match back {
            RendererToBrowser::SubmitDisplayList {
                epoch,
                bytes,
                resources,
            } => {
                assert_eq!(epoch, 42);
                assert_eq!(DisplayList::from_bytes(&bytes).unwrap(), list_with_image);
                assert_eq!(resources.len(), 1);
                assert!(resources.get(id).is_some());
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
        // And the other renderer messages round-trip too.
        for message in [
            RendererToBrowser::DidCommit {
                url: "https://example.com/".to_string(),
                status: "Loaded https://example.com/".to_string(),
            },
            RendererToBrowser::LoadFailed {
                url: "https://example.com".to_string(),
                reason: "connection refused".to_string(),
            },
            RendererToBrowser::TitleChanged {
                title: "Example Domain".to_string(),
            },
            RendererToBrowser::RendererCrashed {
                reason: "worker task ended".to_string(),
            },
            RendererToBrowser::RequestResource(ResourceRequest {
                request_id: 7,
                tab_id: 0,
                epoch: 3,
                url: "https://example.com/style.css".to_string(),
                destination: Destination::Stylesheet,
                priority: Priority::VeryHigh,
                mode: RequestMode::NoCors,
                credentials: CredentialsMode::SameOrigin,
                origin: "https://example.com".to_string(),
                top_level_site: "https://example.com".to_string(),
                upgrade_insecure_requests: false,
            }),
            RendererToBrowser::CancelResource { request_id: 7 },
        ] {
            let bytes = encode_message(&message);
            let _: RendererToBrowser = decode_message(&bytes).expect("renderer message decodes");
        }
    }

    #[test]
    fn resource_events_round_trip_through_versioned_bytes() {
        for message in [
            BrowserToRenderer::ResourceEvent(ResourceEvent::ResponseHeaders {
                request_id: 7,
                status: 200,
                url: "https://example.com/style.css".to_string(),
                headers: vec![("content-type".to_string(), "text/css".to_string())],
                content_length: Some(42),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::BodyChunk {
                request_id: 7,
                bytes: b"body { color: red; }".to_vec(),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Done { request_id: 7 }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Error {
                request_id: 7,
                code: "request".to_string(),
                reason: "connection refused".to_string(),
            }),
            BrowserToRenderer::ResourceEvent(ResourceEvent::Cancelled { request_id: 7 }),
        ] {
            let bytes = encode_message(&message);
            let back: BrowserToRenderer = decode_message(&bytes).expect("resource event decodes");
            assert_eq!(
                postcard::to_allocvec(&back).unwrap(),
                postcard::to_allocvec(&message).unwrap()
            );
        }
    }

    #[test]
    fn decode_rejects_version_one_bytes() {
        // A v1 message (version prefix 1) must fail loudly, not be
        // misread as v2: the shapes changed in the Stage 5 freeze.
        let mut v1 = encode_message(&BrowserToRenderer::Shutdown);
        v1[0] = 1;
        assert_eq!(
            decode_message::<BrowserToRenderer>(&v1),
            Err(ProtocolError::UnsupportedVersion { found: 1 })
        );
    }

    #[test]
    fn decode_rejects_wrong_version_and_garbage() {
        let good = encode_message(&BrowserToRenderer::Shutdown);
        let mut wrong_version = good.clone();
        wrong_version[0] = PROTOCOL_VERSION as u8 + 1;
        assert_eq!(
            decode_message::<BrowserToRenderer>(&wrong_version),
            Err(ProtocolError::UnsupportedVersion {
                found: PROTOCOL_VERSION + 1
            })
        );
        assert_eq!(
            decode_message::<BrowserToRenderer>(&[]),
            Err(ProtocolError::Empty)
        );
        assert_eq!(
            decode_message::<BrowserToRenderer>(&[PROTOCOL_VERSION as u8, 0]),
            Err(ProtocolError::Empty)
        );
        assert!(matches!(
            decode_message::<BrowserToRenderer>(&good[..good.len() - 1]),
            Err(ProtocolError::Decode(_))
        ));
        let mut garbage = vec![PROTOCOL_VERSION as u8, 0, 0, 0];
        garbage.extend_from_slice(b"garbage");
        assert!(matches!(
            decode_message::<BrowserToRenderer>(&garbage),
            Err(ProtocolError::Decode(_))
        ));
    }

    #[tokio::test]
    async fn command_channel_carries_browser_messages() {
        let (tx, mut rx) = command_channel();
        tx.send(BrowserToRenderer::CommitNavigation {
            url: "https://example.com".to_string(),
        })
        .expect("send succeeds while the receiver lives");
        tx.send(BrowserToRenderer::Resize {
            width: 400.0,
            height: 300.0,
        })
        .expect("send succeeds");
        match rx.recv().await.expect("first message arrives") {
            BrowserToRenderer::CommitNavigation { url } => {
                assert_eq!(url, "https://example.com");
            }
            other => panic!("expected CommitNavigation, got {other:?}"),
        }
        match rx.recv().await.expect("second message arrives") {
            BrowserToRenderer::Resize { width, height } => {
                assert_eq!((width, height), (400.0, 300.0));
            }
            other => panic!("expected Resize, got {other:?}"),
        }
        // Dropping the receiver makes further sends fail cleanly.
        drop(rx);
        assert!(tx.send(BrowserToRenderer::Shutdown).is_err());
    }
}
