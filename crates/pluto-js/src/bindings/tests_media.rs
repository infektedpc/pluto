//! Media binding tests (Stage 16, M8 Phase 2, Rec 4): the
//! `HTMLMediaElement` surface exercised through `TabJs` + the
//! [`TestHost`], with the REAL pluto-media fixtures — the Big
//! Buck Bunny MP4 decodes through openh264, the MP3 through
//! symphonia, the WebM/VP9 file is honestly refused. A passing
//! test means the whole crossing works (JS → session → decode
//! → events → frames), not just the Rust side.
//!
//! The `TestHost` never fetches (`request_media` is the trait
//! default no-op), so the tests deliver bytes directly via
//! `TabJs::deliver_media` — the same entry point the renderer's
//! broker fetch uses. The audio clock is the `Instant`
//! fallback (no audio device in the sandbox; `CpalSink`
//! degrades to `NullSink`).

use crate::bindings::test_host::TestHost;
use crate::bindings::tests::{DOC, read_bool, read_num, read_str, run};
use crate::host::DomHost;
use slotmap::Key as _;
use crate::realm::RealmConfig;
use crate::tab::TabJs;
use std::sync::LazyLock;

/// Fixtures are stored base64-encoded (`.b64`) because the
/// snapshot push route is text-only; the decoded bytes are
/// byte-identical to the original media files. Minimal
/// standard-base64 decoder (test-only; avoids a new dependency
/// for one helper). Whitespace is skipped, `=` pads.
fn decode_base64(input: &str) -> Vec<u8> {
    fn val(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0, // `=` pad contributes zero bits
        }
    }
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.as_chunks::<4>().0 {
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        let n = ((val(chunk[0]) as u32) << 18)
            | ((val(chunk[1]) as u32) << 12)
            | ((val(chunk[2]) as u32) << 6)
            | (val(chunk[3]) as u32);
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    out
}

static BBB_MP4: LazyLock<Vec<u8>> = LazyLock::new(|| {
    decode_base64(include_str!(
        "../../../pluto-media/tests/fixtures/bbb-360-10s.mp4.b64"
    ))
});
static BBB_WEBM: LazyLock<Vec<u8>> = LazyLock::new(|| {
    decode_base64(include_str!(
        "../../../pluto-media/tests/fixtures/bbb-360-10s.webm.b64"
    ))
});
static SAMPLE_MP3: LazyLock<Vec<u8>> = LazyLock::new(|| {
    decode_base64(include_str!(
        "../../../pluto-media/tests/fixtures/sample-3s.mp3.b64"
    ))
});

/// A tab whose document holds one `<video id="vid">` with the
/// given attributes, plus any `<source>` / `<track>` children
/// described by `children` (tag, attrs). Returns the tab and
/// the video's `DomKey`.
fn video_tab(
    video_attrs: &[(&str, &str)],
    children: &[(&str, &[(&str, &str)])],
) -> (TabJs, crate::host::DomKey) {
    let mut host = TestHost::from_html(DOC);
    let video = host.dom.create_element("video");
    host.dom.set_attribute(video, "id", "vid");
    for (name, value) in video_attrs {
        host.dom.set_attribute(video, name, value);
    }
    for (tag, attrs) in children {
        let child = host.dom.create_element(tag);
        for (name, value) in *attrs {
            host.dom.set_attribute(child, name, value);
        }
        host.dom.append_child(video, child);
    }
    let body = host.body().unwrap();
    host.dom.append_child(body, video);
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    tab.register_media_elements();
    tab.run_until_idle().unwrap();
    (tab, video)
}

#[test]
fn video_loads_real_mp4_and_reports_metadata() {
    let (mut tab, video) = video_tab(
        &[("src", "https://example.com/bbb-360-10s.mp4")],
        &[],
    );
    run(
        &mut tab,
        "const v = document.getElementById('vid'); \
         const seen = []; \
         for (const e of ['loadedmetadata','loadeddata','canplay','canplaythrough']) \
           v.addEventListener(e, () => seen.push(e)); \
         globalThis.__seen = seen; globalThis.__v = v;",
    );
    tab.deliver_media(
        video,
        BBB_MP4.to_vec(),
        "video/mp4".to_string(),
        true,
        true,
    );
    tab.run_until_idle().unwrap();
    // HAVE_ENOUGH_DATA (4) / NETWORK_IDLE (1): the load primed
    // decode-ahead, so metadata + data + canplay all fired.
    assert_eq!(read_num(&mut tab, "__v.readyState"), 4.0);
    assert_eq!(read_num(&mut tab, "__v.networkState"), 1.0);
    assert!((read_num(&mut tab, "__v.duration") - 10.0).abs() < 0.1);
    assert_eq!(read_num(&mut tab, "__v.videoWidth"), 640.0);
    assert_eq!(read_num(&mut tab, "__v.videoHeight"), 360.0);
    assert_eq!(
        read_str(&mut tab, "__v.currentSrc"),
        "https://example.com/bbb-360-10s.mp4"
    );
    assert_eq!(
        read_str(&mut tab, "__seen.join(',')"),
        "loadedmetadata,loadeddata,canplay,canplaythrough"
    );
    assert!(read_bool(&mut tab, "__v.paused"));
    // Whole-file buffered range, interim shape.
    assert_eq!(read_num(&mut tab, "__v.buffered.length"), 1.0);
    assert_eq!(read_num(&mut tab, "__v.buffered.end(0)"), 10.0);
}

#[test]
fn video_play_presents_real_frames_on_the_audio_clock() {
    let (mut tab, video) = video_tab(
        &[("src", "https://example.com/bbb-360-10s.mp4")],
        &[],
    );
    tab.deliver_media(
        video,
        BBB_MP4.to_vec(),
        "video/mp4".to_string(),
        true,
        true,
    );
    tab.run_until_idle().unwrap();
    run(
        &mut tab,
        "const v = document.getElementById('vid'); \
         globalThis.__v = v; v.play();",
    );
    // No user activation: the policy starts muted instead of
    // rejecting (MutedOnly) — sound needs activation, motion
    // does not.
    assert!(read_bool(&mut tab, "__v.muted"));
    assert!(!read_bool(&mut tab, "__v.paused"));
    // Let the audio clock (Instant fallback — no device here)
    // run ~300 ms, then tick: at most one frame per tick call,
    // so tick until one presents.
    let mut presented = None;
    for _ in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let frames = tab.media_tick();
        if let Some((_, frame)) = frames.into_iter().next() {
            presented = Some(frame);
            break;
        }
    }
    let frame = presented.expect("a frame presented within ~1 s of playing");
    assert_eq!((frame.width, frame.height), (640, 360));
    assert!(
        frame.pts_ms >= 0.0 && frame.pts_ms < 1500.0,
        "pts_ms={} (audio-clock timestamp, not a tick count)",
        frame.pts_ms
    );
    assert_eq!(frame.element_id, video.data().as_ffi());
    // The media clock moved: currentTime is past zero.
    assert!(read_num(&mut tab, "__v.currentTime") > 0.0);
    // Pause stops the clock: a tick now presents nothing new.
    run(&mut tab, "__v.pause();");
    assert!(read_bool(&mut tab, "__v.paused"));
    let after_pause = tab.media_tick();
    assert!(
        after_pause.is_empty(),
        "no frames present while paused: {after_pause:?}"
    );
}

#[test]
fn video_unsupported_source_falls_through_to_next_source() {
    let (mut tab, video) = video_tab(
        &[],
        &[
            ("source", &[("src", "https://example.com/bad.webm"), ("type", "video/mp4")]),
            ("source", &[("src", "https://example.com/good.mp4")]),
        ],
    );
    // Selection picked the first candidate (its `type` claims
    // MP4, so it is not skipped before the fetch).
    assert_eq!(
        read_str(&mut tab, "document.getElementById('vid').currentSrc"),
        "https://example.com/bad.webm"
    );
    // The bytes are actually WebM/VP9: unsupported → the next
    // candidate is tried (currentSrc moves, no error yet).
    run(
        &mut tab,
        "const v = document.getElementById('vid'); \
         let errors = 0; v.addEventListener('error', () => { errors += 1; }); \
         globalThis.__errors = () => errors; globalThis.__v = v;",
    );
    tab.deliver_media(video, BBB_WEBM.to_vec(), "video/webm".to_string(), true, true);
    tab.run_until_idle().unwrap();
    assert_eq!(
        read_str(&mut tab, "__v.currentSrc"),
        "https://example.com/good.mp4"
    );
    assert_eq!(read_num(&mut tab, "__errors()"), 0.0);
    // The good candidate loads: metadata arrives, no error.
    tab.deliver_media(
        video,
        BBB_MP4.to_vec(),
        "video/mp4".to_string(),
        true,
        true,
    );
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__v.readyState"), 4.0);
    assert_eq!(read_num(&mut tab, "__v.videoWidth"), 640.0);
    assert_eq!(read_num(&mut tab, "__errors()"), 0.0);
}

#[test]
fn video_no_supported_source_fires_error_with_no_source() {
    let (mut tab, video) = video_tab(
        &[],
        &[("source", &[("src", "https://example.com/bad.webm"), ("type", "video/mp4")])],
    );
    run(
        &mut tab,
        "const v = document.getElementById('vid'); \
         let errors = 0; v.addEventListener('error', () => { errors += 1; }); \
         globalThis.__errors = () => errors; globalThis.__v = v;",
    );
    tab.deliver_media(video, BBB_WEBM.to_vec(), "video/webm".to_string(), true, true);
    tab.run_until_idle().unwrap();
    assert_eq!(read_num(&mut tab, "__errors()"), 1.0);
    // NETWORK_NO_SOURCE (3) / HAVE_NOTHING (0), error code 4
    // (MEDIA_ERR_SRC_NOT_SUPPORTED).
    assert_eq!(read_num(&mut tab, "__v.networkState"), 3.0);
    assert_eq!(read_num(&mut tab, "__v.readyState"), 0.0);
    assert_eq!(read_num(&mut tab, "__v.error.code"), 4.0);
}

#[test]
fn audio_mp3_loads_and_canplaytype_is_honest() {
    let mut host = TestHost::from_html(DOC);
    let audio = host.dom.create_element("audio");
    host.dom.set_attribute(audio, "id", "aud");
    host.dom
        .set_attribute(audio, "src", "https://example.com/sample-3s.mp3");
    let body = host.body().unwrap();
    host.dom.append_child(body, audio);
    let mut tab = TabJs::new(&RealmConfig::default()).unwrap();
    tab.set_dom_host(Box::new(host));
    tab.register_media_elements();
    tab.run_until_idle().unwrap();
    tab.deliver_media(
        audio,
        SAMPLE_MP3.to_vec(),
        "audio/mpeg".to_string(),
        true,
        true,
    );
    tab.run_until_idle().unwrap();
    run(
        &mut tab,
        "const a = document.getElementById('aud'); globalThis.__a = a; \
         const v = document.createElement('video'); globalThis.__v = v;",
    );
    assert_eq!(read_num(&mut tab, "__a.readyState"), 4.0);
    let duration = read_num(&mut tab, "__a.duration");
    assert!((2.5..3.5).contains(&duration), "duration={duration}");
    // Proven on fixtures → "probably"; no decoder → "".
    assert_eq!(
        read_str(&mut tab, "__v.canPlayType('video/mp4; codecs=\"avc1.42E01E\"')"),
        "probably"
    );
    assert_eq!(
        read_str(&mut tab, "__v.canPlayType('audio/mpeg')"),
        "probably"
    );
    assert_eq!(
        read_str(&mut tab, "__v.canPlayType('video/webm; codecs=\"vp9\"')"),
        ""
    );
    assert_eq!(
        read_str(&mut tab, "__v.canPlayType('audio/ogg; codecs=\"opus\"')"),
        ""
    );
    assert_eq!(
        read_str(&mut tab, "__v.canPlayType('video/mp4; codecs=\"av01.0.05M.08\"')"),
        ""
    );
}

#[test]
fn track_data_url_parses_cues_into_text_tracks() {
    let vtt = "WEBVTT%0A%0A00%3A00.000%20--%3E%2000%3A01.000%0AHello%20track";
    let src = format!("data:text/vtt,{vtt}");
    let (mut tab, _video) = video_tab(
        &[("src", "https://example.com/bbb-360-10s.mp4")],
        &[("track", &[
            ("kind", "subtitles"),
            ("label", "English"),
            ("srclang", "en"),
            ("src", src.as_str()),
        ])],
    );
    run(
        &mut tab,
        "const v = document.getElementById('vid'); globalThis.__v = v;",
    );
    assert_eq!(read_num(&mut tab, "__v.textTracks.length"), 1.0);
    assert_eq!(read_str(&mut tab, "__v.textTracks[0].kind"), "subtitles");
    assert_eq!(read_str(&mut tab, "__v.textTracks[0].label"), "English");
    assert_eq!(read_str(&mut tab, "__v.textTracks[0].language"), "en");
    assert_eq!(read_num(&mut tab, "__v.textTracks[0].cues.length"), 1.0);
    assert_eq!(
        read_str(&mut tab, "__v.textTracks[0].cues[0].text"),
        "Hello track"
    );
    assert_eq!(read_num(&mut tab, "__v.textTracks[0].cues[0].startTime"), 0.0);
    assert_eq!(read_num(&mut tab, "__v.textTracks[0].cues[0].endTime"), 1.0);
}
