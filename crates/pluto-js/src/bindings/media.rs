//! `HTMLMediaElement` / `HTMLVideoElement` / `HTMLAudioElement`
//! bindings (Stage 16, M8 Phase 2, Rec 4).
//!
//! One [`MediaState`] per `<video>` / `<audio>` element, keyed by
//! `DomKey` in [`BindingState::media_states`] — the same pattern
//! as `image_states`: the JS object holds only the key, the
//! session + resource-selection state lives here in Rust.
//!
//! Resource selection (spec §4.8, simplified to what the shell
//! can fetch): the element's own `src` attribute first, then its
//! `<source>` children in document order. A candidate whose `type`
//! attribute names an unsupported codec/container is skipped
//! before any fetch ([`pluto_media::can_play_type`]); the first
//! remaining candidate is fetched by the renderer (via the
//! `DomHost::request_media` hook — the binding never fetches
//! itself, so mixed-content / CSP / blocker gates apply exactly
//! as for preloads). On a load failure the next candidate is
//! tried; when none remain the session goes `NETWORK_NO_SOURCE`
//! and fires `error` — never silence.
//!
//! The media clock is the session's audio clock
//! ([`pluto_media::MediaClock`]); `TabJs::media_tick` advances
//! every session once per tab tick, fires the drained events, and
//! returns the frames presented this tick for the renderer to
//! send over IPC. Video is never scheduled by counting ticks.
//!
//! `<track>` discovery rides here too: track elements are found
//! at registration, their files are fetched by the renderer
//! (`request_track`), and the bytes are parsed as WebVTT into
//! the session's `TextTrack`s. Cue *display* is deferred —
//! `textTracks` exposes the parsed cues, nothing renders them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pluto_ipc::VideoFrame;
use pluto_media::{
    CanPlay, MediaEvent, MediaSession, TrackKind, can_play_type,
};
use rquickjs::function::This;
use rquickjs::{Array, Ctx, FromJs, Function, Object, Value};
use slotmap::Key as _;

use crate::activation::{AutoplayDecision, AutoplayPolicy};
use crate::bindings::{
    BindingState, dom_key_of_this, dom_key_of_this_opt, js, js_null, throw_type_error,
    to_dom_string,
};
use crate::bindings::dom::{def_accessor, def_getter, def_method, host_of, resolve_url};
use crate::bindings::events::dispatch_to_target;
use crate::host::DomKey;

/// One resource-selection candidate: a resolved URL plus the
/// `type` hint from the `<source>` (empty for the element's own
/// `src`, which carries no hint).
#[derive(Debug, Clone)]
pub struct MediaCandidate {
    pub url: String,
    pub mime: String,
}

/// Per-element media state (see the module docs).
pub struct MediaState {
    /// The pluto-media state machine: ready/network state,
    /// clock, sink, decode-ahead, events.
    pub session: MediaSession,
    /// Candidates in selection order (own `src`, then
    /// `<source>` children).
    pub candidates: Vec<MediaCandidate>,
    /// Index of the candidate currently loading / loaded.
    pub candidate_idx: Option<usize>,
    /// Resolved URL of the current candidate ("" when none).
    pub current_src: String,
    /// A fetch for the current candidate is outstanding.
    pub loading: bool,
    /// The `preload` attribute, mirrored here so `load()` and
    /// registration can read it without a host round-trip.
    pub preload: String,
    /// The `autoplay` attribute.
    pub autoplay: bool,
    /// The `controls` attribute (no controls UI in M8 — the
    /// attribute is reflected, nothing is drawn).
    pub controls: bool,
    /// The `defaultMuted` attribute.
    pub default_muted: bool,
    /// The last fetch's tainting decision: true when the
    /// origin is clean for this element (same-origin, `data:`,
    /// or a CORS-approved cross-origin fetch). Recorded here
    /// for the readback / canvas-tainting rules; the renderer
    /// computes it, the session stores it.
    pub cors_approved: bool,
    /// `<track>` children discovered at registration:
    /// (track element key, kind, label, language). The file
    /// bytes land in the session's `TextTrack`s via
    /// [`deliver_track_bytes`].
    pub track_elements: Vec<(DomKey, String, String, String)>,
}

impl MediaState {
    pub fn new() -> Self {
        Self {
            session: MediaSession::new(),
            candidates: Vec::new(),
            candidate_idx: None,
            current_src: String::new(),
            loading: false,
            preload: "auto".to_string(),
            autoplay: false,
            controls: false,
            default_muted: false,
            cors_approved: false,
            track_elements: Vec::new(),
        }
    }
}

impl Default for MediaState {
    fn default() -> Self {
        Self::new()
    }
}

/// The JS-visible `MediaError` code for a failed source
/// selection (kept in sync with `pluto_media::MEDIA_ERR_*`).
fn error_code(state: &MediaState) -> Option<(u32, String)> {
    state.session.error()
}

/// Build the plain-object form of `el.error` (`{code, message}`)
/// or `null` when there is no error.
fn error_object<'js>(
    ctx: &Ctx<'js>,
    state: &MediaState,
) -> rquickjs::Result<Value<'js>> {
    let Some((code, message)) = error_code(state) else {
        return Ok(js_null(ctx));
    };
    let obj = Object::new(ctx.clone())?;
    obj.set("code", code as f64)?;
    obj.set("message", message)?;
    Ok(Value::from(obj))
}

/// Build a `TimeRanges`-shaped plain object for `el.buffered`:
/// `{length, start(i), end(i)}` over the session's buffered
/// ranges (one whole-file range once loaded, interim).
fn buffered_object<'js>(
    ctx: &Ctx<'js>,
    state: &MediaState,
) -> rquickjs::Result<Value<'js>> {
    let ranges = Rc::new(state.session.buffered());
    let obj = Object::new(ctx.clone())?;
    obj.set("length", ranges.len() as f64)?;
    let r = Rc::clone(&ranges);
    let start = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, idx: f64| -> rquickjs::Result<f64> {
            let i = idx as usize;
            Ok(r.get(i).map(|(s, _)| *s).unwrap_or(0.0))
        },
    )?;
    obj.set("start", start)?;
    let r = Rc::clone(&ranges);
    let end = Function::new(
        ctx.clone(),
        move |_ctx: Ctx<'js>, idx: f64| -> rquickjs::Result<f64> {
            let i = idx as usize;
            Ok(r.get(i).map(|(_, e)| *e).unwrap_or(0.0))
        },
    )?;
    obj.set("end", end)?;
    Ok(Value::from(obj))
}

/// Build the `textTracks` list: one plain object per session
/// track — `{kind, label, language, mode, cues: [{startTime,
/// endTime, text}]}`. Fresh objects each access (the session
/// owns the tracks; JS never mutates them through this list).
fn text_tracks_array<'js>(
    ctx: &Ctx<'js>,
    state: &MediaState,
) -> rquickjs::Result<Value<'js>> {
    let arr = Array::new(ctx.clone())?;
    for (i, track) in state.session.tracks().iter().enumerate() {
        let obj = Object::new(ctx.clone())?;
        obj.set("kind", track_kind_name(track.kind))?;
        obj.set("label", track.label.clone())?;
        obj.set("language", track.language.clone())?;
        obj.set("mode", track.mode.clone())?;
        let cues = Array::new(ctx.clone())?;
        for (j, cue) in track.cues.iter().enumerate() {
            let c = Object::new(ctx.clone())?;
            c.set("startTime", cue.start)?;
            c.set("endTime", cue.end)?;
            c.set("text", cue.text.clone())?;
            if let Some(id) = &cue.id {
                c.set("id", id.clone())?;
            }
            cues.set(j, Value::from(c))?;
        }
        obj.set("cues", Value::from(cues))?;
        arr.set(i, Value::from(obj))?;
    }
    Ok(Value::from(arr))
}

fn track_kind_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Subtitles => "subtitles",
        TrackKind::Captions => "captions",
        TrackKind::Descriptions => "descriptions",
        TrackKind::Chapters => "chapters",
        TrackKind::Metadata => "metadata",
    }
}

/// A `DomKey` as the `u64` the IPC `VideoFrame` carries
/// (`element_id`): the slotmap key's foreign-function id, the
/// same value paint's `canvas_blob_id` / `video_blob_id` fold
/// into resource ids.
fn dom_key_u64(key: DomKey) -> u64 {
    key.data().as_ffi()
}

fn track_kind_from_str(s: &str) -> TrackKind {
    match s {
        "captions" => TrackKind::Captions,
        "descriptions" => TrackKind::Descriptions,
        "chapters" => TrackKind::Chapters,
        "metadata" => TrackKind::Metadata,
        _ => TrackKind::Subtitles,
    }
}

/// Map a [`MediaEvent`] to its DOM event name. `Error` carries
/// the message; the event itself is a plain `error`.
fn media_event_name(event: &MediaEvent) -> &'static str {
    match event {
        MediaEvent::LoadedMetadata => "loadedmetadata",
        MediaEvent::LoadedData => "loadeddata",
        MediaEvent::CanPlay => "canplay",
        MediaEvent::CanPlayThrough => "canplaythrough",
        MediaEvent::Play => "play",
        MediaEvent::Playing => "playing",
        MediaEvent::Pause => "pause",
        MediaEvent::TimeUpdate => "timeupdate",
        MediaEvent::Seeking => "seeking",
        MediaEvent::Seeked => "seeked",
        MediaEvent::Ended => "ended",
        MediaEvent::Error(_) => "error",
        MediaEvent::Progress => "progress",
    }
}

/// Fire one media event on an element: dispatch to listeners,
/// then call the `on<event>` property handler if one is set.
/// Mirrors `fire_image_event` (dom.rs) — a never-wrapped element
/// has no listeners, so nothing happens.
pub fn fire_media_event<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    event: &MediaEvent,
) -> rquickjs::Result<()> {
    let kind = media_event_name(event);
    let Some(id) = state.borrow().wrapper_ids.get(&key).copied() else {
        return Ok(()); // Never wrapped in JS: nothing can listen.
    };
    let wrappers = crate::bindings::registry_map(ctx, "wrappers")?;
    let wrapper = crate::bindings::map_get(ctx, &wrappers, id)?;
    let Some(obj) = wrapper.as_object() else {
        return Ok(());
    };
    let ev: Object = ctx.eval(format!(
        "new Event({kind:?}, {{ bubbles: false, cancelable: false }})"
    ))?;
    dispatch_to_target(ctx, state, id, &ev, kind)?;
    let prop = format!("on{kind}");
    if let Ok(handler) = obj.get::<_, Value>(prop.as_str())
        && handler.is_function()
    {
        let func = Function::from_js(ctx, handler)?;
        let _: Value = func.call((This(Value::from(obj.clone())), Value::from(ev)))?;
    }
    Ok(())
}

/// Drain a session's queued events and fire each on the element.
fn drain_and_fire<'js>(
    ctx: &Ctx<'js>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
) -> rquickjs::Result<()> {
    let events = {
        let mut s = state.borrow_mut();
        let Some(ms) = s.media_states.get_mut(&key) else {
            return Ok(());
        };
        ms.session.drain_events()
    };
    for event in events {
        fire_media_event(ctx, state, key, &event)?;
    }
    Ok(())
}

/// The autoplay decision for `play()` / autoplay: the tab's
/// policy is the default (no `block_all` setter exists anywhere
/// — if one ever lands, this must read the shared policy from
/// `TabJs` instead of constructing the default here).
fn autoplay_decision(muted: bool, state: &Rc<RefCell<BindingState>>) -> AutoplayDecision {
    let activation = state.borrow().user_activation.clone();
    AutoplayPolicy::default().decide(muted, &activation.borrow())
}

/// Install the shared `HTMLMediaElement` methods on `media`
/// (proto: `Element.prototype`). `HTMLVideoElement` /
/// `HTMLAudioElement` chain off this prototype (see
/// `install_prototypes` in dom.rs).
pub fn install_media_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    media: &Object<'js>,
) -> Result<(), String> {
    // src (getter: the resolved URL of the current candidate;
    // setter: set the attribute and restart resource selection).
    let b = Rc::clone(bindings);
    let get_src = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        if let Some(ms) = b.borrow().media_states.get(&key)
            && !ms.current_src.is_empty()
        {
            return Ok(js(&ctx, ms.current_src.clone()));
        }
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().get_attribute(key, "src") {
            Some(v) => js(&ctx, resolve_url(&b, &v)),
            None => js(&ctx, ""),
        })
    };
    let b = Rc::clone(bindings);
    let set_src = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let src = to_dom_string(&ctx, &value)?;
        set_media_src(&ctx, &b, key, &src)
    };
    def_accessor(ctx, media, "src", get_src, set_src)?;

    // currentSrc ("" when no candidate has been selected).
    let b = Rc::clone(bindings);
    def_getter(ctx, media, "currentSrc", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        Ok(js(
            &ctx,
            b.borrow()
                .media_states
                .get(&key)
                .map(|ms| ms.current_src.clone())
                .unwrap_or_default(),
        ))
    })?;

    // readyState / networkState (numeric, per spec).
    for (prop, is_ready) in [("readyState", true), ("networkState", false)] {
        let b = Rc::clone(bindings);
        def_getter(
            ctx,
            media,
            prop,
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                    return Ok(0.0);
                };
                Ok(b.borrow()
                    .media_states
                    .get(&key)
                    .map(|ms| {
                        if is_ready {
                            ms.session.ready_state() as u8 as f64
                        } else {
                            ms.session.network_state() as u8 as f64
                        }
                    })
                    .unwrap_or(0.0))
            },
        )?;
    }

    // duration (NaN until metadata arrives), currentTime
    // (getter + setter → seek), paused / ended / seeking.
    let b = Rc::clone(bindings);
    def_getter(ctx, media, "duration", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(f64::NAN);
        };
        Ok(b.borrow()
            .media_states
            .get(&key)
            .and_then(|ms| ms.session.duration())
            .unwrap_or(f64::NAN))
    })?;
    let b = Rc::clone(bindings);
    let get_time = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(0.0);
        };
        Ok(b.borrow()
            .media_states
            .get(&key)
            .map(|ms| ms.session.current_time())
            .unwrap_or(0.0))
    };
    let b = Rc::clone(bindings);
    let set_time = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let t = value.as_number().unwrap_or(0.0);
        let mut s = b.borrow_mut();
        if let Some(ms) = s.media_states.get_mut(&key) {
            ms.session.seek(t);
        }
        drop(s);
        // The seek fires `seeking` / `seeked` from the session;
        // drain them now so JS sees the events in order.
        drain_and_fire(&ctx, &b, key)
    };
    def_accessor(ctx, media, "currentTime", get_time, set_time)?;
    for (prop, which) in [("paused", 0u8), ("ended", 1u8), ("seeking", 2u8)] {
        let b = Rc::clone(bindings);
        def_getter(
            ctx,
            media,
            prop,
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
                let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                    return Ok(prop == "paused");
                };
                Ok(b.borrow()
                    .media_states
                    .get(&key)
                    .map(|ms| match which {
                        0 => ms.session.paused(),
                        1 => ms.session.ended(),
                        _ => ms.session.seeking(),
                    })
                    .unwrap_or(prop == "paused"))
            },
        )?;
    }

    // volume (0..1, clamped by the session) / muted (sink-level
    // mute — the M12 auto-mute hook).
    let b = Rc::clone(bindings);
    let get_volume = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(1.0);
        };
        Ok(b.borrow()
            .media_states
            .get(&key)
            .map(|ms| ms.session.volume() as f64)
            .unwrap_or(1.0))
    };
    let b = Rc::clone(bindings);
    let set_volume = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let v = value.as_number().unwrap_or(1.0) as f32;
        if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
            ms.session.set_volume(v);
        }
        Ok(())
    };
    def_accessor(ctx, media, "volume", get_volume, set_volume)?;
    let b = Rc::clone(bindings);
    let get_muted = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(false);
        };
        Ok(b.borrow()
            .media_states
            .get(&key)
            .map(|ms| ms.session.muted())
            .unwrap_or(false))
    };
    let b = Rc::clone(bindings);
    let set_muted = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let m = value.as_bool().unwrap_or(false);
        if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
            ms.session.set_muted(m);
        }
        Ok(())
    };
    def_accessor(ctx, media, "muted", get_muted, set_muted)?;

    // loop / autoplay / controls / defaultMuted / preload:
    // content attributes, mirrored in the state so `load()` and
    // registration read them without a host round-trip.
    for prop in ["loop", "autoplay", "controls", "defaultMuted"] {
        let b = Rc::clone(bindings);
        let get_bool = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
            let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                return Ok(false);
            };
            Ok(b.borrow().media_states.get(&key).map(|ms| match prop {
                "loop" => ms.session.loop_(),
                "autoplay" => ms.autoplay,
                "controls" => ms.controls,
                _ => ms.default_muted,
            }).unwrap_or(false))
        };
        let b = Rc::clone(bindings);
        let set_bool = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let v = value.as_bool().unwrap_or(false);
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let attr = if prop == "defaultMuted" { "muted" } else { prop };
            if v {
                host.borrow_mut().set_attribute(key, attr, "");
            } else {
                host.borrow_mut().remove_attribute(key, attr);
            }
            if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
                match prop {
                    "loop" => ms.session.set_loop(v),
                    "autoplay" => ms.autoplay = v,
                    "controls" => ms.controls = v,
                    _ => {
                        ms.default_muted = v;
                        if v {
                            ms.session.set_muted(true);
                        }
                    }
                }
            }
            Ok(())
        };
        def_accessor(ctx, media, prop, get_bool, set_bool)?;
    }
    let b = Rc::clone(bindings);
    let get_preload = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, "auto"));
        };
        Ok(js(
            &ctx,
            b.borrow()
                .media_states
                .get(&key)
                .map(|ms| ms.preload.clone())
                .unwrap_or_else(|| "auto".to_string()),
        ))
    };
    let b = Rc::clone(bindings);
    let set_preload = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let v = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        host.borrow_mut().set_attribute(key, "preload", &v);
        if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
            ms.preload = v;
        }
        Ok(())
    };
    def_accessor(ctx, media, "preload", get_preload, set_preload)?;

    // error / buffered / textTracks (getter-only, built fresh).
    let b = Rc::clone(bindings);
    def_getter(ctx, media, "error", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let s = b.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(js_null(&ctx));
        };
        error_object(&ctx, ms)
    })?;
    let b = Rc::clone(bindings);
    def_getter(ctx, media, "buffered", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let s = b.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(js_null(&ctx));
        };
        buffered_object(&ctx, ms)
    })?;
    let b = Rc::clone(bindings);
    def_getter(ctx, media, "textTracks", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        let s = b.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        text_tracks_array(&ctx, ms)
    })?;

    // canPlayType(type) → "" / "maybe" / "probably".
    let b = Rc::clone(bindings);
    let can_play = move |ctx: Ctx<'js>, This(_this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<Value<'js>> {
        let _ = &b;
        let mime = to_dom_string(&ctx, &value)?;
        let out = match can_play_type(&mime) {
            CanPlay::Empty => "",
            CanPlay::Maybe => "maybe",
            CanPlay::Probably => "probably",
        };
        Ok(js(&ctx, out))
    };
    def_method(
        ctx,
        media,
        "canPlayType",
        Function::new(ctx.clone(), can_play)
            .map_err(|e| format!("install canPlayType: {e}"))?,
    )?;

    // play() → Promise: rejects NotAllowedError when autoplay
    // blocks, NotSupportedError when there is nothing to play.
    let b = Rc::clone(bindings);
    let play = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let (promise, resolve, reject) = ctx.promise()?;
        let reject_with = |ctx: &Ctx<'js>, name: &str, msg: &str| -> rquickjs::Result<()> {
            let value: Value = ctx.eval(format!("new {name}({msg:?})"))?;
            let _: Value = reject.call((value,))?;
            Ok(())
        };
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            reject_with(&ctx, "NotSupportedError", "the element is no longer in the document")?;
            return Ok(promise.into_value());
        };
        let muted = b
            .borrow()
            .media_states
            .get(&key)
            .map(|ms| ms.session.muted())
            .unwrap_or(false);
        match autoplay_decision(muted, &b) {
            AutoplayDecision::Block => {
                reject_with(&ctx, "NotAllowedError", "play() was blocked by the autoplay policy")?;
                return Ok(promise.into_value());
            }
            AutoplayDecision::MutedOnly if !muted => {
                // Sound would play without activation: start
                // muted instead of rejecting (spec: the play
                // request proceeds, muted).
                if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
                    ms.session.set_muted(true);
                }
            }
            _ => {}
        }
        let started = {
            let mut s = b.borrow_mut();
            match s.media_states.get_mut(&key) {
                Some(ms) => ms.session.play(),
                None => false,
            }
        };
        if !started {
            reject_with(&ctx, "NotSupportedError", "no media source is loaded")?;
            return Ok(promise.into_value());
        }
        let _: Value = resolve.call((js_null(&ctx),))?;
        drain_and_fire(&ctx, &b, key)?;
        Ok(promise.into_value())
    };
    def_method(
        ctx,
        media,
        "play",
        Function::new(ctx.clone(), play).map_err(|e| format!("install play: {e}"))?,
    )?;

    // pause() — fires `pause` from the session on the next drain.
    let b = Rc::clone(bindings);
    let pause = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        if let Some(ms) = b.borrow_mut().media_states.get_mut(&key) {
            ms.session.pause();
        }
        drain_and_fire(&ctx, &b, key)
    };
    def_method(
        ctx,
        media,
        "pause",
        Function::new(ctx.clone(), pause).map_err(|e| format!("install pause: {e}"))?,
    )?;

    // load() — restart resource selection from the current
    // attributes (spec: the `load()` method).
    let b = Rc::clone(bindings);
    let load = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        start_media_load(&ctx, &b, key)?;
        drain_and_fire(&ctx, &b, key)
    };
    def_method(
        ctx,
        media,
        "load",
        Function::new(ctx.clone(), load).map_err(|e| format!("install load: {e}"))?,
    )?;
    Ok(())
}

/// Install `HTMLVideoElement` extras on `video` (proto: the
/// media prototype): `videoWidth` / `videoHeight` (0 until
/// metadata) and `poster` (content attribute).
pub fn install_video_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    video: &Object<'js>,
) -> Result<(), String> {
    for (prop, is_width) in [("videoWidth", true), ("videoHeight", false)] {
        let b = Rc::clone(bindings);
        def_getter(
            ctx,
            video,
            prop,
            move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<f64> {
                let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                    return Ok(0.0);
                };
                Ok(b.borrow()
                    .media_states
                    .get(&key)
                    .and_then(|ms| ms.session.video_size())
                    .map(|(w, h)| if is_width { w } else { h } as f64)
                    .unwrap_or(0.0))
            },
        )?;
    }
    let b = Rc::clone(bindings);
    let get_poster = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(match host.borrow().get_attribute(key, "poster") {
            Some(v) => js(&ctx, resolve_url(&b, &v)),
            None => js(&ctx, ""),
        })
    };
    let b = Rc::clone(bindings);
    let set_poster = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let poster = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        host.borrow_mut().set_attribute(key, "poster", &poster);
        // The poster is painted from the image store; a new
        // poster needs a re-layout to attach it.
        host.borrow_mut().request_restlye_and_layout(key);
        Ok(())
    };
    def_accessor(ctx, video, "poster", get_poster, set_poster)?;
    Ok(())
}

/// Install `HTMLAudioElement` extras: none beyond the shared
/// media prototype in M8 (kept as a hook for M12's audio-only
/// controls surface).
pub fn install_audio_methods<'js>(
    _ctx: &Ctx<'js>,
    _bindings: &Rc<RefCell<BindingState>>,
    _audio: &Object<'js>,
) -> Result<(), String> {
    Ok(())
}

/// Set the element's `src` attribute from JS and restart
/// resource selection (the `src` setter's work).
fn set_media_src(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    src: &str,
) -> rquickjs::Result<()> {
    let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
    if !host.borrow_mut().set_attribute(key, "src", src) {
        return Err(throw_type_error(ctx, "the element is no longer in the document"));
    }
    start_media_load(ctx, state, key)
}

/// Collect the resource-selection candidates for `key` from the
/// live DOM: the element's own `src`, then its `<source>`
/// children in document order (each with its `type` hint).
/// Candidates whose `type` names an unsupported codec/container
/// are skipped here, before any fetch.
fn collect_candidates(
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
) -> Vec<MediaCandidate> {
    let host = match host_of(state) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };
    let host = host.borrow();
    let mut out: Vec<MediaCandidate> = Vec::new();
    if let Some(src) = host.get_attribute(key, "src")
        && !src.is_empty()
    {
        out.push(MediaCandidate {
            url: resolve_url(state, &src),
            mime: String::new(),
        });
    }
    for child in host.children(key) {
        if host.tag_name(child).as_deref() != Some("source") {
            continue;
        }
        let Some(src) = host.get_attribute(child, "src") else {
            continue;
        };
        if src.is_empty() {
            continue;
        }
        let mime = host
            .get_attribute(child, "type")
            .unwrap_or_default();
        if !mime.is_empty() && can_play_type(&mime) == CanPlay::Empty {
            continue; // Known-unsupported: skip before fetching.
        }
        out.push(MediaCandidate {
            url: resolve_url(state, &src),
            mime,
        });
    }
    out
}

/// (Re)start resource selection for `key`: rebuild the
/// candidate list from the DOM, pick the first playable one,
/// and ask the host to fetch it. With no candidates the session
/// goes `NETWORK_NO_SOURCE` and fires `error` (the caller drains
/// the event).
pub fn start_media_load(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
) -> rquickjs::Result<()> {
    let candidates = collect_candidates(state, key);
    let (preload, autoplay) = {
        let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
        let host = host.borrow();
        (
            host.get_attribute(key, "preload")
                .unwrap_or_else(|| "auto".to_string()),
            host.get_attribute(key, "autoplay").is_some(),
        )
    };
    {
        let mut s = state.borrow_mut();
        let ms = s.media_states.entry(key).or_default();
        ms.candidates = candidates;
        ms.candidate_idx = None;
        ms.current_src = String::new();
        ms.loading = false;
        ms.preload = preload;
        ms.autoplay = autoplay;
    }
    try_next_candidate(ctx, state, key)
}

/// Try the next candidate after the current one (or the first
/// one, when selection just started). A `preload="none"`
/// element fetches nothing until `load()` / `play()` asks for
/// it — but selection still runs, so `currentSrc` and the
/// network state are honest.
fn try_next_candidate(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
) -> rquickjs::Result<()> {
    let (next_idx, preload) = {
        let s = state.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(());
        };
        let next = match ms.candidate_idx {
            None => 0,
            Some(i) => i + 1,
        };
        (next, ms.preload.clone())
    };
    let (url, mime) = {
        let s = state.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(());
        };
        match ms.candidates.get(next_idx) {
            Some(c) => (c.url.clone(), c.mime.clone()),
            None => {
                // Selection exhausted: no usable source.
                drop(s);
                let mut s = state.borrow_mut();
                if let Some(ms) = s.media_states.get_mut(&key) {
                    ms.session.no_source();
                    ms.candidate_idx = None;
                    ms.current_src = String::new();
                    ms.loading = false;
                }
                return Ok(());
            }
        }
    };
    {
        let mut s = state.borrow_mut();
        if let Some(ms) = s.media_states.get_mut(&key) {
            ms.candidate_idx = Some(next_idx);
            ms.current_src = url.clone();
            ms.loading = preload != "none";
        }
    }
    if preload == "none" {
        return Ok(());
    }
    let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
    let base = host.borrow().document_url().unwrap_or_default();
    host.borrow_mut()
        .request_media(key, &url, &base, &mime);
    Ok(())
}

/// Deliver one candidate's fetched bytes to the session
/// (called by `TabJs::deliver_media` on the loop). On a
/// successful load the session primes decode-ahead and fires
/// `loadedmetadata` / `loadeddata` / `canplay` /
/// `canplaythrough`; on an unsupported source the next
/// candidate is tried; a network failure tries the next
/// candidate too (a later candidate may be a different host).
/// `origin_clean` is the renderer's tainting decision for this
/// fetch (same-origin / `data:` / CORS-approved).
pub fn deliver_media_bytes(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    bytes: Vec<u8>,
    mime: String,
    origin_clean: bool,
    cors_approved: bool,
) -> rquickjs::Result<()> {
    let unsupported = {
        let mut s = state.borrow_mut();
        let Some(ms) = s.media_states.get_mut(&key) else {
            return Ok(()); // Element gone: drop the delivery.
        };
        ms.loading = false;
        ms.cors_approved = cors_approved;
        ms.session.load(bytes, &mime, origin_clean);
        ms.session
            .error()
            .map(|(code, _)| code == pluto_media::MEDIA_ERR_SRC_NOT_SUPPORTED)
            .unwrap_or(false)
    };
    if unsupported {
        // Clear this candidate's failure first (its queued
        // `error` event included): when a next candidate
        // exists no `error` fires on the element at all, and
        // when selection is exhausted `no_source()` pushes
        // exactly one fresh `error` — never two.
        {
            let mut s = state.borrow_mut();
            if let Some(ms) = s.media_states.get_mut(&key) {
                ms.session.clear_failure();
            }
        }
        try_next_candidate(ctx, state, key)?;
    } else {
        // Autoplay: a loaded element with the attribute starts
        // when the policy allows (muted-only starts muted).
        let (autoplay, muted) = {
            let s = state.borrow();
            match s.media_states.get(&key) {
                Some(ms) => (ms.autoplay, ms.session.muted()),
                None => (false, false),
            }
        };
        if autoplay {
            match autoplay_decision(muted, state) {
                AutoplayDecision::Block => {}
                AutoplayDecision::MutedOnly if !muted => {
                    let mut s = state.borrow_mut();
                    if let Some(ms) = s.media_states.get_mut(&key) {
                        ms.session.set_muted(true);
                        ms.session.play();
                    }
                }
                _ => {
                    let mut s = state.borrow_mut();
                    if let Some(ms) = s.media_states.get_mut(&key) {
                        ms.session.play();
                    }
                }
            }
        }
    }
    drain_and_fire(ctx, state, key)
}

/// Report a candidate fetch failure (called by
/// `TabJs::notify_media_failed`): try the next candidate; when
/// none remain the session goes `NETWORK_NO_SOURCE`.
pub fn notify_media_failed(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    _reason: &str,
) -> rquickjs::Result<()> {
    {
        let mut s = state.borrow_mut();
        if let Some(ms) = s.media_states.get_mut(&key) {
            ms.loading = false;
        }
    }
    try_next_candidate(ctx, state, key)?;
    drain_and_fire(ctx, state, key)
}

/// Deliver a `<track>` file's bytes: parse the WebVTT into the
/// session's `TextTrack`s (cue display is deferred — parsing is
/// the deliverable here). A bad file is recorded as a failed
/// track, never a media error.
pub fn deliver_track_bytes(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
    key: DomKey,
    track_idx: usize,
    bytes: Vec<u8>,
) -> rquickjs::Result<()> {
    let (kind, label, language) = {
        let s = state.borrow();
        let Some(ms) = s.media_states.get(&key) else {
            return Ok(());
        };
        match ms.track_elements.get(track_idx) {
            Some((_, k, l, lang)) => (
                track_kind_from_str(k),
                l.clone(),
                lang.clone(),
            ),
            None => return Ok(()),
        }
    };
    let vtt = String::from_utf8_lossy(&bytes);
    let mut s = state.borrow_mut();
    if let Some(ms) = s.media_states.get_mut(&key) {
        // A parse failure leaves the track absent (its cues
        // stay empty) — the media itself is unaffected.
        let _ = ms.session.add_track_vtt(kind, label, language, &vtt);
    }
    drop(s);
    let _ = ctx;
    Ok(())
}

/// A `<track>` fetch failed: nothing to do — the track stays
/// cue-less and the media plays on.
pub fn notify_track_failed(
    _ctx: &Ctx<'_>,
    _state: &Rc<RefCell<BindingState>>,
    _key: DomKey,
    _track_idx: usize,
) -> rquickjs::Result<()> {
    Ok(())
}

/// Advance every media session one tab tick: decode ahead,
/// present the due frame, fire drained events. Returns the
/// frames presented this tick, keyed by element, for the
/// renderer to send over IPC. The clock — not the tick count
/// — decides what presents.
pub fn media_tick(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
) -> rquickjs::Result<Vec<(DomKey, VideoFrame)>> {
    let keys: Vec<DomKey> = state.borrow().media_states.keys().copied().collect();
    let mut presented = Vec::new();
    for key in keys {
        let frame = {
            let mut s = state.borrow_mut();
            let Some(ms) = s.media_states.get_mut(&key) else {
                continue;
            };
            ms.session.advance().presented_frame
        };
        if let Some(mut frame) = frame {
            // The session stamps its own id at decode, but the
            // tick's key is authoritative (it covers frames
            // decoded before registration stamped them 0).
            frame.element_id = dom_key_u64(key);
            presented.push((key, frame));
        }
        drain_and_fire(ctx, state, key)?;
    }
    Ok(presented)
}

/// Intrinsic sizes for layout: (width, height, has_frame) per
/// video element with metadata, read from the sessions.
pub fn media_intrinsic_sizes(
    state: &Rc<RefCell<BindingState>>,
) -> HashMap<DomKey, (u32, u32, bool)> {
    let s = state.borrow();
    let mut out = HashMap::new();
    for (key, ms) in s.media_states.iter() {
        if let Some((w, h)) = ms.session.video_size() {
            out.insert(*key, (w, h, ms.session.has_frame()));
        }
    }
    out
}

/// Register every `<video>` / `<audio>` in the document:
/// create the state, collect candidates + track elements, and
/// start the first candidate's fetch (unless `preload="none"`).
/// Called by the renderer after parse, and again after a
/// `src` / `<source>` mutation re-runs it for one element
/// (via `load()`). `data:` track sources are parsed inline —
/// no fetch needed; remote tracks go through the host.
pub fn register_media_elements(
    ctx: &Ctx<'_>,
    state: &Rc<RefCell<BindingState>>,
) -> rquickjs::Result<()> {
    let elements = {
        let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
        let host = host.borrow();
        let Some(root) = host.document_node() else {
            return Ok(());
        };
        let mut found = Vec::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(tag) = host.tag_name(id)
                && (tag == "video" || tag == "audio") {
                    found.push(id);
                }
            stack.extend(host.children(id));
        }
        found
    };
    for key in elements {
        // Mirror the content attributes into the state first.
        let (preload, autoplay, controls, default_muted, looped, muted) = {
            let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
            let host = host.borrow();
            (
                host.get_attribute(key, "preload")
                    .unwrap_or_else(|| "auto".to_string()),
                host.get_attribute(key, "autoplay").is_some(),
                host.get_attribute(key, "controls").is_some(),
                host.get_attribute(key, "muted").is_some(),
                host.get_attribute(key, "loop").is_some(),
                host.get_attribute(key, "muted").is_some(),
            )
        };
        {
            let mut s = state.borrow_mut();
            let ms = s.media_states.entry(key).or_default();
            ms.session.set_element_id(dom_key_u64(key));
            ms.preload = preload;
            ms.autoplay = autoplay;
            ms.controls = controls;
            ms.default_muted = default_muted;
            ms.session.set_loop(looped);
            if muted {
                ms.session.set_muted(true);
            }
        }
        // `<track>` children: record + fetch (or parse inline
        // for `data:`).
        let tracks = {
            let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
            let host = host.borrow();
            let mut out = Vec::new();
            for child in host.children(key) {
                if host.tag_name(child).as_deref() != Some("track") {
                    continue;
                }
                let kind = host
                    .get_attribute(child, "kind")
                    .unwrap_or_else(|| "subtitles".to_string());
                let label = host.get_attribute(child, "label").unwrap_or_default();
                let language = host
                    .get_attribute(child, "srclang")
                    .unwrap_or_default();
                let src = host.get_attribute(child, "src").unwrap_or_default();
                out.push((child, kind, label, language, src));
            }
            out
        };
        for (track_key, kind, label, language, src) in tracks {
            let idx = {
                let mut s = state.borrow_mut();
                let ms = s.media_states.entry(key).or_default();
                ms.track_elements
                    .push((track_key, kind.clone(), label.clone(), language.clone()));
                ms.track_elements.len() - 1
            };
            if src.starts_with("data:") {
                // Inline: decode the data URL here, no fetch.
                if let Some(comma) = src.find(',') {
                    let (meta, data) = (&src[5..comma], &src[comma + 1..]);
                    let bytes = if meta.contains(";base64") {
                        base64_decode(data)
                    } else {
                        percent_decode(data)
                    };
                    if let Some(bytes) = bytes {
                        deliver_track_bytes(ctx, state, key, idx, bytes)?;
                    }
                }
            } else if !src.is_empty() {
                let host = host_of(state).map_err(|e| throw_type_error(ctx, &e))?;
                let base = host.borrow().document_url().unwrap_or_default();
                let url = resolve_url(state, &src);
                host.borrow_mut()
                    .request_track(key, idx, &url, &base);
            }
        }
        start_media_load(ctx, state, key)?;
        drain_and_fire(ctx, state, key)?;
    }
    Ok(())
}

/// Minimal base64 decode (data-URL track sources only — no
/// padding-sensitive general decoder; invalid input → None).
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

/// Percent-decode a data-URL body (`%XX` → byte, `+` stays
/// literal — data URLs do not use form encoding).
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
