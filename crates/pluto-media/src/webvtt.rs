//! WebVTT cue parsing + the `TextTrack` model (Stage 16
//! Phase 2, Rec 4).
//!
//! Parsing only: cues are parsed, validated, and exposed to
//! JS as `TextTrack` / `TextTrackCue` objects. Cue *display*
//! (rendering cues over the video, styling, regions) is
//! deferred — there is no cue renderer in this phase, and none
//! is faked. A malformed file is an error, never a silent
//! empty track: the session surfaces it as a track `error`
//! event on the media element.
//!
//! Grammar covered (WebVTT spec, the common subset): the
//! `WEBVTT` header (with optional header metadata), cue ids,
//! `HH:MM:SS.mmm` / `MM:SS.mmm` timings, cue settings after
//! the timings, multi-line payload text, `NOTE` comments.
//! Not covered: regions (`Region:` settings), styles
//! (`STYLE` blocks — parsed and skipped, not applied), and
//! cue payload markup (kept verbatim in `text`).

/// One parsed cue.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    /// Cue id, when the block carries one.
    pub id: Option<String>,
    /// Start time, seconds.
    pub start: f64,
    /// End time, seconds.
    pub end: f64,
    /// Settings after the timings (`line:90% align:start` …),
    /// verbatim, space-joined.
    pub settings: String,
    /// Payload text, lines joined with `\n`, markup verbatim.
    pub text: String,
}

/// The kind of a text track (`subtitles`, `captions`,
/// `descriptions`, `chapters`, `metadata`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Subtitles,
    Captions,
    Descriptions,
    Chapters,
    Metadata,
}

impl TrackKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TrackKind::Subtitles => "subtitles",
            TrackKind::Captions => "captions",
            TrackKind::Descriptions => "descriptions",
            TrackKind::Chapters => "chapters",
            TrackKind::Metadata => "metadata",
        }
    }

    pub fn from_attr(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "captions" => TrackKind::Captions,
            "descriptions" => TrackKind::Descriptions,
            "chapters" => TrackKind::Chapters,
            "metadata" => TrackKind::Metadata,
            _ => TrackKind::Subtitles,
        }
    }
}

/// A text track: its `<track>` attributes plus parsed cues.
#[derive(Debug, Clone)]
pub struct TextTrack {
    pub kind: TrackKind,
    pub label: String,
    pub language: String,
    /// `mode`: `"disabled"` (default), `"hidden"`, `"showing"`.
    /// Parsing fills `cues` regardless of mode; display is
    /// deferred, so mode is stored, not acted on.
    pub mode: String,
    pub cues: Vec<Cue>,
}

impl TextTrack {
    pub fn new(kind: TrackKind, label: String, language: String) -> Self {
        Self {
            kind,
            label,
            language,
            mode: "disabled".to_string(),
            cues: Vec::new(),
        }
    }

    /// Cues active at `time_secs` (start ≤ t < end), in order.
    pub fn active_cues(&self, time_secs: f64) -> Vec<&Cue> {
        self.cues
            .iter()
            .filter(|c| c.start <= time_secs && time_secs < c.end)
            .collect()
    }
}

/// Parse a WebVTT document. Returns the cues in file order.
/// Errors name the offending line; a file without the `WEBVTT`
/// header is an error, not a guess.
pub fn parse_webvtt(input: &str) -> Result<Vec<Cue>, String> {
    let text = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = text.lines().peekable();
    let first = lines.next().unwrap_or("").trim();
    if !first.starts_with("WEBVTT") {
        return Err("WebVTT file does not start with a WEBVTT header".to_string());
    }
    // Header metadata lines run until the first blank line.
    for line in lines.by_ref() {
        if line.trim().is_empty() {
            break;
        }
    }
    let mut cues = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    // Blocks are separated by blank lines; flush at the end too.
    for line in lines.chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !block.is_empty() {
                if let Some(cue) = parse_block(&block)? {
                    cues.push(cue);
                }
                block.clear();
            }
        } else {
            block.push(line);
        }
    }
    Ok(cues)
}

/// Parse one blank-line-delimited block. `None` for NOTE
/// comments, STYLE/REGION blocks (skipped, not errors).
fn parse_block(block: &[&str]) -> Result<Option<Cue>, String> {
    let first = block[0];
    if first.starts_with("NOTE") {
        return Ok(None);
    }
    if first == "STYLE" || first == "REGION" {
        return Ok(None);
    }
    // [id] / timings / payload… — the timings line is the first
    // line containing "-->".
    let timing_idx = block
        .iter()
        .position(|l| l.contains("-->"))
        .ok_or_else(|| format!("WebVTT block has no timing line: {first}"))?;
    let id = if timing_idx == 0 {
        None
    } else {
        Some(block[..timing_idx].join("\n"))
    };
    let timing = block[timing_idx];
    let (start, end, settings) = parse_timings(timing)?;
    let text = block[timing_idx + 1..].join("\n");
    Ok(Some(Cue {
        id,
        start,
        end,
        settings,
        text,
    }))
}

/// Parse `start --> end [settings]`. Times are `MM:SS.mmm` or
/// `HH:MM:SS.mmm`.
fn parse_timings(line: &str) -> Result<(f64, f64, String), String> {
    let (left, right) = line
        .split_once("-->")
        .ok_or_else(|| format!("bad WebVTT timing line: {line}"))?;
    let start = parse_timestamp(left.trim())
        .ok_or_else(|| format!("bad WebVTT start time: {left}"))?;
    let right = right.trim();
    let mut parts = right.splitn(2, |c: char| c.is_whitespace());
    let end_s = parts.next().unwrap_or("");
    let end = parse_timestamp(end_s)
        .ok_or_else(|| format!("bad WebVTT end time: {end_s}"))?;
    let settings = parts.next().unwrap_or("").trim().to_string();
    if end < start {
        return Err(format!("WebVTT cue ends before it starts: {line}"));
    }
    Ok((start, end, settings))
}

/// Parse one WebVTT timestamp to seconds.
fn parse_timestamp(s: &str) -> Option<f64> {
    let (main, frac) = s.split_once('.')?;
    if frac.len() != 3 || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let ms: f64 = frac.parse().ok()?;
    let fields: Vec<&str> = main.split(':').collect();
    let (h, m, sec) = match fields.len() {
        2 => (0.0, fields[0].parse::<f64>().ok()?, fields[1].parse::<f64>().ok()?),
        3 => (
            fields[0].parse::<f64>().ok()?,
            fields[1].parse::<f64>().ok()?,
            fields[2].parse::<f64>().ok()?,
        ),
        _ => return None,
    };
    if !(0.0..60.0).contains(&m) || !(0.0..60.0).contains(&sec) {
        return None;
    }
    Some(h * 3600.0 + m * 60.0 + sec + ms / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "WEBVTT\n\n1\n00:00.500 --> 00:02.000 line:90%\nHello, world.\nSecond line.\n\n00:03.000 --> 00:05.500 align:start\nNo id on this one.\n\nNOTE a comment\n\n01:00:00.000 --> 01:00:04.000\nAn hour in.\n";

    #[test]
    fn parses_ids_timings_settings_and_text() {
        let cues = parse_webvtt(SAMPLE).expect("parses");
        assert_eq!(cues.len(), 3);
        assert_eq!(cues[0].id.as_deref(), Some("1"));
        assert!((cues[0].start - 0.5).abs() < 1e-9);
        assert!((cues[0].end - 2.0).abs() < 1e-9);
        assert_eq!(cues[0].settings, "line:90%");
        assert_eq!(cues[0].text, "Hello, world.\nSecond line.");
        assert_eq!(cues[1].id, None);
        assert!((cues[1].start - 3.0).abs() < 1e-9);
        assert_eq!(cues[2].start, 3600.0);
        assert!((cues[2].end - 3604.0).abs() < 1e-9);
    }

    #[test]
    fn active_cues_follow_the_clock() {
        let cues = parse_webvtt(SAMPLE).expect("parses");
        let track = TextTrack {
            kind: TrackKind::Subtitles,
            label: "English".to_string(),
            language: "en".to_string(),
            mode: "showing".to_string(),
            cues,
        };
        assert_eq!(track.active_cues(1.0).len(), 1);
        assert_eq!(track.active_cues(2.0).len(), 0, "end is exclusive");
        assert_eq!(track.active_cues(4.0).len(), 1);
        assert_eq!(track.active_cues(10.0).len(), 0);
    }

    #[test]
    fn missing_header_and_bad_timings_are_errors() {
        assert!(parse_webvtt("00:00.000 --> 00:01.000\nHi\n").is_err());
        assert!(parse_webvtt("WEBVTT\n\n00:05.000 --> 00:01.000\nBackwards.\n").is_err());
        assert!(parse_webvtt("WEBVTT\n\nno timings here\n").is_err());
    }

    #[test]
    fn note_and_style_blocks_are_skipped() {
        let vtt = "WEBVTT\n\nSTYLE\n::cue { color: red; }\n\n00:00.000 --> 00:01.000\nVisible.\n";
        let cues = parse_webvtt(vtt).expect("parses");
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Visible.");
    }
}
