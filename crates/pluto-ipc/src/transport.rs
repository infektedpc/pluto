//! The transport layer for the browser <-> renderer protocol
//! (Stage 9 Rec 9, M5 process split).
//!
//! The protocol messages ([`BrowserToRenderer`] /
//! [`RendererToBrowser`]) never know which transport carries them:
//! the same typed messages travel over an in-process channel today
//! and over a Unix socketpair once the renderer is a process. A
//! transport moves *frames*; the typed layer ([`Transport::send_message`]
//! / [`Transport::recv_message`]) wraps each frame in the versioned
//! encoding from the crate root ([`encode_message`] /
//! [`decode_message`]) — so a version mismatch is rejected at
//! decode time on either transport, never misread.
//!
//! ## Framing
//!
//! Every frame on the wire is `[length: u32 LE][payload]`, where
//! `payload` is exactly the bytes [`encode_message`] produces
//! (`[version: u32 LE][postcard]`). The length prefix is capped at
//! [`MAX_FRAME_BYTES`]: a peer that announces a larger frame is
//! treated as malformed, so a compromised renderer cannot make the
//! browser allocate unboundedly by naming a huge length.
//!
//! ## Handshake
//!
//! Before any frame, both ends exchange an 8-byte handshake — the
//! magic [`HANDSHAKE_MAGIC`] plus the protocol version as a
//! little-endian `u32` ([`SocketTransport::handshake`]). A peer
//! speaking another version (a stale renderer binary, the wrong
//! peer entirely) is rejected before the first message, never
//! silently accepted. Each side writes first, then reads: the
//! handshake is 8 bytes, far under the socket buffer, so neither
//! side can deadlock waiting for the other to write.
//!
//! ## Opaque origins across the boundary
//!
//! [`Origin::Opaque`] ids are process-local counters (see the type's
//! docs): an id minted in the renderer means nothing in the browser,
//! and trusting a deserialized id would let two different opaque
//! origins alias — or one origin be forged — across the boundary.
//! The wire never carries the id at all (an opaque origin
//! serializes as the string `"null"`), and the receiving side
//! re-parses every serialized origin on receipt
//! ([`remint_opaque_origins`], applied by
//! [`recv_renderer_message`]): a `"null"` origin is re-parsed into
//! a FRESH process-local opaque identity minted by the receiver,
//! never into the sender's identity, and tuple origins are
//! re-normalized (default ports dropped) at the same time. The
//! renderer can therefore never smuggle an opaque identity across,
//! and the browser can never confuse two renderers' opaque origins
//! for the same one.
//!
//! ## Transports
//!
//! * [`ChannelTransport`] — the in-process transport: two tokio
//!   mpsc channels carrying frames. Kept for unit tests (and for
//!   the in-process worker, which still uses the typed
//!   [`command_channel`] directly).
//! * [`SocketTransport`] — the cross-process transport: a Unix
//!   socketpair, one end in the browser process, one inherited by
//!   the `--renderer` child (see `pluto-browser`'s
//!   `renderer_process` module). Both sides use this same code.

use std::io;
use std::os::unix::net::UnixStream as StdUnixStream;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use crate::{
    decode_message, encode_message, BrowserToRenderer, Origin, RendererToBrowser,
    PROTOCOL_VERSION,
};

/// The handshake magic: the first four bytes each side sends.
/// A peer that does not speak Pluto IPC fails here, before any
/// frame is read.
pub const HANDSHAKE_MAGIC: [u8; 4] = *b"PLTO";

/// The largest single frame either side accepts, in bytes.
/// Display lists are the biggest frames (a decoded image in the
/// resource table can be tens of MiB); 64 MiB leaves ample headroom
/// while bounding what a malformed length prefix can make the
/// receiver allocate.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// A transport moves length-prefixed frames between the browser
/// and a renderer. The typed helpers encode/decode protocol
/// messages on top of frames; implementors only move bytes.
///
/// `async fn` in this trait is deliberate (edition 2024 RPITIT):
/// it keeps the trait dependency-free instead of pulling in
/// `async-trait`, and no caller needs auto trait bounds on it.
#[allow(async_fn_in_trait)]
pub trait Transport: Send {
    /// Send one frame (the caller supplies the already-encoded
    /// bytes — for protocol messages, use [`Transport::send_message`]).
    async fn send_frame(&mut self, frame: &[u8]) -> io::Result<()>;

    /// Receive one frame. `Ok(None)` is a clean EOF at a frame
    /// boundary (the peer closed the socket / dropped the channel);
    /// a partial frame at EOF is an error, never a silent `None`.
    async fn recv_frame(&mut self) -> io::Result<Option<Vec<u8>>>;

    /// Send one protocol message: version-prefixed postcard bytes
    /// ([`encode_message`]) as a single frame.
    async fn send_message<T: Serialize>(&mut self, message: &T) -> io::Result<()> {
        self.send_frame(&encode_message(message)).await
    }

    /// Receive one protocol message, rejecting another version or
    /// undecodable bytes as [`io::ErrorKind::InvalidData`] — never
    /// misreading them. `Ok(None)` is a clean EOF (see
    /// [`Transport::recv_frame`]).
    async fn recv_message<T>(&mut self) -> io::Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let Some(frame) = self.recv_frame().await? else {
            return Ok(None);
        };
        decode_message(&frame)
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// The in-process transport: two unbounded tokio mpsc channels
/// carrying frames. The same messages, the same framing, the same
/// version check as the socket transport — only the carrier differs,
/// so unit tests exercise the real protocol without a process.
pub struct ChannelTransport {
    tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    rx: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>,
}

/// Create a connected pair of in-process transports: what one end
/// sends, the other receives.
pub fn channel_transport_pair() -> (ChannelTransport, ChannelTransport) {
    let (a_tx, a_rx) = tokio::sync::mpsc::unbounded_channel();
    let (b_tx, b_rx) = tokio::sync::mpsc::unbounded_channel();
    (
        ChannelTransport { tx: a_tx, rx: b_rx },
        ChannelTransport { tx: b_tx, rx: a_rx },
    )
}

impl Transport for ChannelTransport {
    async fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        self.tx.send(frame.to_vec()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "channel transport: peer dropped",
            )
        })
    }

    async fn recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        Ok(self.rx.recv().await)
    }
}

/// The cross-process transport: a Unix stream socket, one end in
/// the browser process and one inherited by the `--renderer` child.
/// Frames are `[length: u32 LE][payload]`; see the module docs.
pub struct SocketTransport {
    stream: UnixStream,
}

impl SocketTransport {
    /// Wrap an already-connected std Unix stream (the browser side
    /// of a `socketpair`, or the fd the `--renderer` child
    /// inherited). The stream is put in non-blocking mode for
    /// tokio.
    pub fn from_std(stream: StdUnixStream) -> io::Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream: UnixStream::from_std(stream)?,
        })
    }

    /// Create a connected pair of socket transports (used by tests;
    /// the browser spawn path builds its pair from
    /// `StdUnixStream::pair` directly so it can clear `CLOEXEC` on
    /// the child end before exec).
    pub fn from_pair() -> io::Result<(Self, Self)> {
        let (a, b) = StdUnixStream::pair()?;
        Ok((Self::from_std(a)?, Self::from_std(b)?))
    }

    /// The version handshake: write our magic + protocol version,
    /// then read the peer's and verify both. A stale renderer (or
    /// the wrong peer) is rejected here, before the first frame —
    /// never silently accepted. Both sides write first, then read.
    pub async fn handshake(&mut self) -> io::Result<()> {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&HANDSHAKE_MAGIC);
        out[4..].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        self.stream.write_all(&out).await?;
        let mut peer = [0u8; 8];
        self.stream.read_exact(&mut peer).await?;
        if peer[..4] != HANDSHAKE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IPC handshake: peer did not send the Pluto magic — not a Pluto renderer",
            ));
        }
        let version = u32::from_le_bytes(peer[4..].try_into().expect("4 bytes"));
        if version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IPC handshake: peer speaks protocol version {version}, expected {PROTOCOL_VERSION} (stale renderer or wrong peer)"
                ),
            ));
        }
        Ok(())
    }
}

impl Transport for SocketTransport {
    async fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        if frame.len() > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IPC frame of {} bytes exceeds the {MAX_FRAME_BYTES}-byte cap",
                    frame.len()
                ),
            ));
        }
        let len = (frame.len() as u32).to_le_bytes();
        self.stream.write_all(&len).await?;
        self.stream.write_all(frame).await?;
        Ok(())
    }

    async fn recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        // Read the first length byte with a plain `read`: 0 bytes
        // is a clean EOF at a frame boundary; any bytes at all mean
        // a frame is starting, and a short read after that is a
        // truncation, not an EOF (`read_exact` below reports it).
        let mut first = [0u8; 1];
        if self.stream.read(&mut first).await? == 0 {
            return Ok(None);
        }
        let mut rest = [0u8; 3];
        self.stream.read_exact(&mut rest).await?;
        let mut len_buf = [0u8; 4];
        len_buf[0] = first[0];
        len_buf[1..].copy_from_slice(&rest);
        let len = u32::from_le_bytes(len_buf) as usize;
        if len > MAX_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IPC frame length {len} exceeds the {MAX_FRAME_BYTES}-byte cap — malformed peer"
                ),
            ));
        }
        let mut payload = vec![0u8; len];
        self.stream.read_exact(&mut payload).await?;
        Ok(Some(payload))
    }
}

/// Receive the next browser -> renderer message over `transport`.
/// Browser messages carry no origins, so no re-minting applies.
pub async fn recv_browser_message(
    transport: &mut impl Transport,
) -> io::Result<Option<BrowserToRenderer>> {
    transport.recv_message().await
}

/// Receive the next renderer -> browser message over `transport`,
/// re-minting opaque origins on receipt (see
/// [`remint_opaque_origins`]).
pub async fn recv_renderer_message(
    transport: &mut impl Transport,
) -> io::Result<Option<RendererToBrowser>> {
    let Some(mut message) = transport.recv_message().await? else {
        return Ok(None);
    };
    remint_opaque_origins(&mut message);
    Ok(Some(message))
}

/// Re-mint opaque origins in a message received from a renderer
/// process. The wire carries origins only as serialized strings,
/// and an opaque origin serializes as `"null"` — it carries no
/// identity — so re-parsing each string here mints a FRESH
/// process-local opaque identity on the receiving side
/// ([`Origin::from_url`]) instead of trusting any id the renderer
/// might have held. Tuple origins are re-normalized at the same
/// time (an explicit default port is dropped), so a renderer cannot
/// smuggle a non-canonical origin past the browser's comparisons
/// either.
///
/// Only `RequestResource` carries origins today; other variants
/// pass through untouched. A future message that carries an
/// [`Origin`] value directly must be added here — never trust a
/// deserialized `Origin::Opaque` id across the boundary.
pub fn remint_opaque_origins(message: &mut RendererToBrowser) {
    if let RendererToBrowser::RequestResource(request) = message {
        request.origin = Origin::from_url(&request.origin).serialize();
        request.top_level_site = Origin::from_url(&request.top_level_site).serialize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CredentialsMode, Destination, Priority, RequestMode, ResourceRequest,
    };
    use pluto_css::Stylesheet;
    use pluto_html::parse as parse_html;
    use pluto_layout::LayoutTree;
    use pluto_paint::{DisplayItem, DisplayList, Resource, ResourceId, ResourceTable};
    use tokio::io::AsyncWriteExt;

    fn example_list() -> DisplayList {
        let html = "<html><head><style>body { background-color: #f0f0f2; }</style></head><body><p>Hello</p></body></html>";
        let dom = parse_html(html);
        let sheet = Stylesheet::parse("body { background-color: #f0f0f2; }");
        let tree = LayoutTree::build(&dom, &sheet, 800.0, 600.0).expect("layout builds");
        DisplayList::build(&tree)
    }

    fn example_request() -> ResourceRequest {
        ResourceRequest {
            request_id: 7,
            tab_id: 0,
            epoch: 3,
            url: "https://example.com:443/style.css".to_string(),
                method: "GET".to_string(),
                headers: Vec::new(),
                is_imageset: false,
            destination: Destination::Stylesheet,
            priority: Priority::VeryHigh,
            mode: RequestMode::NoCors,
            credentials: CredentialsMode::SameOrigin,
            // Deliberately non-canonical: an explicit default port
            // the receiver must normalize away.
            origin: "https://example.com:443".to_string(),
            top_level_site: "https://example.com".to_string(),
            upgrade_insecure_requests: false,
            integrity: None,
        }
    }

    /// Round-trip real protocol messages in both directions over a
    /// pair of transports: browser -> renderer and renderer ->
    /// browser, including a real display list and a resource ask.
    async fn round_trip_both_directions(browser: &mut impl Transport, renderer: &mut impl Transport) {
        // Browser -> renderer: navigation, resize, shutdown.
        for message in [
            BrowserToRenderer::CommitNavigation {
                url: "https://example.com".to_string(),
            },
            BrowserToRenderer::Resize {
                width: 800.0,
                height: 600.0,
            },
            BrowserToRenderer::Shutdown,
        ] {
            browser.send_message(&message).await.expect("send succeeds");
            let back: BrowserToRenderer = renderer
                .recv_message()
                .await
                .expect("recv succeeds")
                .expect("a message arrives");
            assert_eq!(
                postcard::to_allocvec(&back).unwrap(),
                postcard::to_allocvec(&message).unwrap()
            );
        }
        // Renderer -> browser: a display list with a real resource,
        // then a resource ask. The ask goes through
        // `recv_renderer_message`, so its origins are re-minted on
        // receipt: the explicit :443 is normalized away.
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
        renderer.send_message(&submit).await.expect("send succeeds");
        match recv_renderer_message(browser)
            .await
            .expect("recv succeeds")
            .expect("a message arrives")
        {
            RendererToBrowser::SubmitDisplayList {
                epoch,
                bytes,
                resources,
            } => {
                assert_eq!(epoch, 42);
                assert_eq!(
                    DisplayList::from_bytes(&bytes).unwrap(),
                    list_with_image
                );
                assert_eq!(resources.len(), 1);
                assert!(resources.get(id).is_some());
            }
            other => panic!("expected SubmitDisplayList, got {other:?}"),
        }
        let request = example_request();
        renderer
            .send_message(&RendererToBrowser::RequestResource(request))
            .await
            .expect("send succeeds");
        match recv_renderer_message(browser)
            .await
            .expect("recv succeeds")
            .expect("a message arrives")
        {
            RendererToBrowser::RequestResource(back) => {
                // The URL itself is untouched; the origin is
                // re-normalized on receipt (:443 dropped).
                assert_eq!(back.url, "https://example.com:443/style.css");
                assert_eq!(back.origin, "https://example.com");
                assert_eq!(back.top_level_site, "https://example.com");
            }
            other => panic!("expected RequestResource, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn channel_transport_round_trips_real_messages() {
        let (mut browser, mut renderer) = channel_transport_pair();
        round_trip_both_directions(&mut browser, &mut renderer).await;
        // Dropping one end is a clean EOF for the other, not an error.
        drop(renderer);
        assert_eq!(browser.recv_frame().await.expect("recv ok"), None);
    }

    #[tokio::test]
    async fn socket_transport_round_trips_real_messages() {
        let (mut browser, mut renderer) = SocketTransport::from_pair().expect("socketpair");
        round_trip_both_directions(&mut browser, &mut renderer).await;
        // Closing one end is a clean EOF for the other.
        drop(renderer);
        assert_eq!(browser.recv_frame().await.expect("recv ok"), None);
    }

    #[tokio::test]
    async fn handshake_accepts_matching_versions() {
        let (mut a, mut b) = SocketTransport::from_pair().expect("socketpair");
        let (a_result, b_result) = tokio::join!(a.handshake(), b.handshake());
        a_result.expect("side A handshakes");
        b_result.expect("side B handshakes");
    }

    #[tokio::test]
    async fn handshake_rejects_wrong_version_and_bad_magic() {
        // A peer speaking the previous version (v3): rejected.
        let (mut transport, mut fake_peer) = {
            let (a, b) = StdUnixStream::pair().expect("socketpair");
            b.set_nonblocking(true).expect("peer non-blocking");
            (
                SocketTransport::from_std(a).expect("transport"),
                UnixStream::from_std(b).expect("peer"),
            )
        };
        let mut wrong_version = [0u8; 8];
        wrong_version[..4].copy_from_slice(&HANDSHAKE_MAGIC);
        wrong_version[4..].copy_from_slice(&(PROTOCOL_VERSION - 1).to_le_bytes());
        fake_peer.write_all(&wrong_version).await.expect("peer writes");
        let err = transport.handshake().await.expect_err("stale peer rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("protocol version"), "{err}");

        // A peer with the wrong magic: rejected even before the
        // version is considered.
        let (mut transport, mut fake_peer) = {
            let (a, b) = StdUnixStream::pair().expect("socketpair");
            b.set_nonblocking(true).expect("peer non-blocking");
            (
                SocketTransport::from_std(a).expect("transport"),
                UnixStream::from_std(b).expect("peer"),
            )
        };
        let mut bad_magic = [0u8; 8];
        bad_magic[..4].copy_from_slice(b"NOPE");
        bad_magic[4..].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        fake_peer.write_all(&bad_magic).await.expect("peer writes");
        let err = transport.handshake().await.expect_err("bad magic rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[tokio::test]
    async fn transport_rejects_malformed_payload() {
        let (mut browser, mut renderer) = SocketTransport::from_pair().expect("socketpair");
        // A well-formed frame whose payload is not a v4 message:
        // correct version prefix, garbage postcard.
        let mut bad = PROTOCOL_VERSION.to_le_bytes().to_vec();
        bad.extend_from_slice(b"garbage-garbage-garbage");
        renderer.send_frame(&bad).await.expect("frame sends");
        let err = browser
            .recv_message::<BrowserToRenderer>()
            .await
            .expect_err("garbage payload rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        // And a frame carrying the previous protocol version:
        // rejected by the version prefix, never misread.
        let mut stale = (PROTOCOL_VERSION - 1).to_le_bytes().to_vec();
        stale.extend_from_slice(&postcard::to_allocvec(&BrowserToRenderer::Shutdown).unwrap());
        renderer.send_frame(&stale).await.expect("frame sends");
        let err = browser
            .recv_message::<BrowserToRenderer>()
            .await
            .expect_err("stale version rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("version"), "{err}");
    }

    #[tokio::test]
    async fn transport_rejects_oversize_frame_length() {
        // Announce a frame over the cap: the receiver must refuse
        // before allocating for it. The length prefix is written by
        // hand on the raw peer stream, going around `send_frame`,
        // which caps sends too.
        let (a, b) = StdUnixStream::pair().expect("socketpair");
        let mut browser = SocketTransport::from_std(a).expect("transport");
        b.set_nonblocking(true).expect("peer non-blocking");
        let mut peer = UnixStream::from_std(b).expect("peer");
        let huge = (MAX_FRAME_BYTES + 1) as u32;
        peer.write_all(&huge.to_le_bytes())
            .await
            .expect("length sends");
        let err = browser.recv_frame().await.expect_err("oversize refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[tokio::test]
    async fn remint_mints_fresh_opaque_origins_on_receipt() {
        // An opaque origin crosses only as the string "null": the
        // receiver must mint a FRESH local identity for it, never
        // reuse (or trust) the sender's. Two receipts of the same
        // "null" string therefore yield two different opaque
        // identities on the receiving side.
        let mut request = example_request();
        request.origin = "null".to_string();
        request.top_level_site = "null".to_string();
        let mut message = RendererToBrowser::RequestResource(request);
        remint_opaque_origins(&mut message);
        let RendererToBrowser::RequestResource(first) = &message else {
            panic!("expected RequestResource");
        };
        assert_eq!(first.origin, "null");
        // Parsing the received string mints a fresh opaque origin
        // each time — the two receipts are never the same identity.
        let a = Origin::from_url(&first.origin);
        let b = Origin::from_url(&first.origin);
        assert!(a.is_opaque());
        assert_ne!(a, b, "each receipt mints a fresh opaque origin");
        assert!(!a.is_same_origin(&b));

        // And a tuple origin survives the re-mint exactly (already
        // normalized by the transport test above).
        let mut tuple = RendererToBrowser::RequestResource(example_request());
        remint_opaque_origins(&mut tuple);
        let RendererToBrowser::RequestResource(back) = tuple else {
            panic!("expected RequestResource");
        };
        assert_eq!(back.origin, "https://example.com");
    }
}
