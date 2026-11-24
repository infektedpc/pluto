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
pub const PROTOCOL_VERSION: u32 = 1;

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
    /// The renderer needs a subresource it may not fetch itself
    /// (external stylesheet, image, script — M2). The browser
    /// fetches and answers with a `ResourceLoaded` message (added
    /// with M2's external-resource fetching); `request_id` matches
    /// the answer to the ask.
    RequestResource {
        request_id: u64,
        url: String,
        kind: ResourceKind,
    },
}

/// What kind of subresource a [`RendererToBrowser::RequestResource`]
/// is asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceKind {
    Stylesheet,
    Image,
    Script,
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
    fn protocol_version_is_one() {
        assert_eq!(PROTOCOL_VERSION, 1);
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
            RendererToBrowser::RequestResource {
                request_id: 7,
                url: "https://example.com/style.css".to_string(),
                kind: ResourceKind::Stylesheet,
            },
        ] {
            let bytes = encode_message(&message);
            let _: RendererToBrowser = decode_message(&bytes).expect("renderer message decodes");
        }
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
            decode_message::<BrowserToRenderer>(&[1, 0]),
            Err(ProtocolError::Empty)
        );
        assert!(matches!(
            decode_message::<BrowserToRenderer>(&good[..good.len() - 1]),
            Err(ProtocolError::Decode(_))
        ));
        assert!(matches!(
            decode_message::<BrowserToRenderer>(b"\x01\x00\x00\x00garbage"),
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
