//! Downloads (Stage 9 Rec 13, report §h "Downloads safety"):
//!
//! The download decision + save pipeline for the shell. No UI lives
//! here: [`decide`] answers "render inline or download?" and, for a
//! download, builds the [`DownloadPrompt`] the shell renders (prompt
//! with filename + origin + size before saving; "Save" / "Save As" /
//! "Cancel"). The finisher wires this into navigation (`page.rs`);
//! this module only decides, sanitises, sniffs, hashes, and saves.
//!
//! Decision order (report §h; a design synthesis, not a quoted spec
//! algorithm):
//!
//! 1. `Content-Disposition: attachment` (optionally `filename="…"` or
//!    RFC 5987 `filename*=UTF-8''…`) → always a download, regardless
//!    of type. `Content-Disposition: inline` → render if the type is
//!    renderable.
//! 2. `Content-Type: application/octet-stream` (or a missing / unknown
//!    type) → download; never render.
//! 3. Type Pluto cannot render inline → download.
//! 4. Otherwise render inline.
//! 5. A `download` attribute on `<a>` forces a download with the
//!    suggested filename (here: the disposition filename, else the
//!    URL basename — the anchor's own suggested name is threaded in
//!    by the finisher if it differs).
//!
//! MIME sniffing ([`sniff_mime`]) runs ONLY when the supplied type is
//! absent, is `application/octet-stream`, or is unknown — never when
//! `X-Content-Type-Options: nosniff` is present — and it may only
//! NARROW toward download, never promote `text/plain` to `text/html`
//! or to script. Concretely: sniffing never flips a
//! [`DownloadDecision::RenderInline`] to a download for a declared
//! renderable type (sniffing is not consulted there at all), and it
//! never flips a download to inline (an `application/octet-stream`
//! whose bytes are a PNG stays a download). What sniffing *does* do
//! is tell the prompt whether the payload is an ELF / script /
//! archive, so the shell can demand the extra "Keep" confirmation
//! even when the server declared no type at all.
//!
//! Safety rules (Pluto policy choices, not standards — say so):
//!
//! * Save with mode `0644`, never `+x`, regardless of what the
//!   server suggests. The `+x` bit is only ever set by the user
//!   outside the browser.
//! * ELF / script / archive payloads need an explicit "Keep" for
//!   the download to complete ([`DownloadPrompt::needs_keep_confirmation`]).
//! * Never auto-open or auto-execute. There is no auto-open path in
//!   this module at all.
//! * Mixed downloads from a secure context are BLOCKED, not
//!   auto-upgraded (Mixed Content L2 §5.2) — see
//!   [`is_mixed_download_blocked`], which reuses
//!   [`pluto_net::mixed::is_mixed_download`] rather than
//!   re-implementing it.
//! * Remote reputation (Safe Browsing-style) is deferred: no API key,
//!   no list-update mechanism. File hashes are never sent to any
//!   third party.
//! * Partial downloads keep a `.part` file and resume with `Range`
//!   if the server supports it; on completion the size is verified
//!   against `Content-Length` when present.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// The longest filename Pluto will save (bytes-as-chars cap; the
/// report says "cap length" without a number — 255 is the usual
/// per-component filesystem limit, so a capped name always fits).
pub const MAX_FILENAME_CHARS: usize = 255;

/// What the shell should do with a navigation's response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadDecision {
    /// Render the response inline in the page.
    RenderInline,
    /// Save the response to disk; the shell prompts first with the
    /// [`DownloadPrompt`] (filename + origin + size), and honours
    /// `needs_keep_confirmation` before completing the save.
    Download(DownloadPrompt),
    /// A mixed download from a secure context: blocked, not
    /// auto-upgraded (Mixed Content L2 §5.2). No bytes are saved.
    BlockMixed,
}

/// What the shell shows in the save prompt. No UI is rendered here;
/// the shell renders this later ("Save" / "Save As" / "Cancel").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadPrompt {
    /// Sanitised filename (see [`sanitize_filename`]): never an
    /// absolute path, never containing separators, `..`, leading
    /// `.`, or control characters.
    pub filename: String,
    /// Serialized origin of the download URL
    /// (`scheme://host`, `:port` when non-default), via
    /// `pluto_ipc::origin_of`.
    pub origin: String,
    /// Expected size in bytes when the response carried a
    /// `Content-Length`; `None` when unknown at decision time.
    /// The shell fills this in from the response headers — the
    /// decision inputs carry no length.
    pub size: Option<u64>,
    /// Extra "Keep" confirmation marker: true for ELF / script /
    /// archive payloads. The download must not complete until the
    /// user explicitly keeps it.
    pub needs_keep_confirmation: bool,
}

/// The decision inputs, in order — see the module docs.
pub fn decide(
    content_disposition: Option<&str>,
    content_type: Option<&str>,
    url: &str,
    has_download_attr: bool,
) -> DownloadDecision {
    decide_with_sniff(content_disposition, content_type, url, has_download_attr, &[], false)
}

/// The full decision, with MIME sniffing. `body_prefix` is a prefix
/// of the response body (up to 1445 bytes per the MIME Sniffing
/// Standard's summary — the exact figure is unverified, so callers
/// should pass whatever prefix they have; the magic table only
/// needs the first few bytes). `nosniff` is true when the response
/// carried `X-Content-Type-Options: nosniff`.
///
/// Sniffing is consulted ONLY when the declared type is absent,
/// `application/octet-stream`, or unknown, and NEVER when `nosniff`
/// is set. It can only mark the prompt as needing "Keep"
/// confirmation (sniffed ELF / script / archive); it never flips
/// the inline/download outcome — see the module docs.
pub fn decide_with_sniff(
    content_disposition: Option<&str>,
    content_type: Option<&str>,
    url: &str,
    has_download_attr: bool,
    body_prefix: &[u8],
    nosniff: bool,
) -> DownloadDecision {
    // (1) `Content-Disposition: attachment` → always a download,
    // regardless of type.
    let disposition = content_disposition.and_then(parse_content_disposition);
    if disposition.as_ref().is_some_and(|d| d.is_attachment()) {
        return DownloadDecision::Download(make_prompt(disposition.as_ref(), content_type, url, false));
    }
    // (5) `<a download>` forces a download.
    if has_download_attr {
        return DownloadDecision::Download(make_prompt(disposition.as_ref(), content_type, url, false));
    }
    let essence = mime_essence(content_type);
    // (2) Missing type, `application/octet-stream`, or an unknown
    // type → download; never render.
    match essence.as_deref() {
        None => {
            return DownloadDecision::Download(make_prompt(
                disposition.as_ref(),
                content_type,
                url,
                sniffed_keep(body_prefix, nosniff),
            ));
        }
        Some("application/octet-stream") => {
            return DownloadDecision::Download(make_prompt(
                disposition.as_ref(),
                content_type,
                url,
                sniffed_keep(body_prefix, nosniff),
            ));
        }
        Some(mime) if !is_renderable_type(mime) => {
            return DownloadDecision::Download(make_prompt(
                disposition.as_ref(),
                content_type,
                url,
                sniffed_keep(body_prefix, nosniff),
            ));
        }
        // (3)/(4) A renderable type renders inline — unless the
        // type is one Pluto does not know at all, which the
        // `!is_renderable_type` arm above already caught.
        Some(_) => {}
    }
    DownloadDecision::RenderInline
}

/// The navigation entry point for the finisher: the mixed-download
/// block first (a secure context downloading over http is blocked,
/// not upgraded), then the plain [`decide`].
///
/// `page_origin` is the document origin (`https://…` = a secure
/// context); `url` is the URL the response actually came from
/// (after redirects).
pub fn decide_for_navigation(
    page_origin: &str,
    content_disposition: Option<&str>,
    content_type: Option<&str>,
    url: &str,
    has_download_attr: bool,
) -> DownloadDecision {
    if is_mixed_download_blocked(page_origin, url, content_disposition) {
        return DownloadDecision::BlockMixed;
    }
    decide(content_disposition, content_type, url, has_download_attr)
}

/// Is this response a mixed download (Mixed Content L2 §5.2)?
/// Thin wrapper over [`pluto_net::mixed::is_mixed_download`] —
/// the rule itself lives there and is NOT re-implemented here.
/// A secure context (`https://` page origin) downloading an
/// `attachment` over plain `http` is blocked, not auto-upgraded.
pub fn is_mixed_download_blocked(
    page_origin: &str,
    url: &str,
    content_disposition: Option<&str>,
) -> bool {
    let Ok(uri) = url.parse::<http::Uri>() else {
        return false;
    };
    let mut headers = http::header::HeaderMap::new();
    if let Some(value) = content_disposition
        && let Ok(parsed) = value.parse::<http::header::HeaderValue>()
    {
        headers.insert(http::header::CONTENT_DISPOSITION, parsed);
    }
    pluto_net::mixed::is_mixed_download(page_origin, &uri, &headers)
}

/// Build the prompt for a download decision. The filename comes
/// from `Content-Disposition` first, else the URL basename, else
/// `"download"`; it is always sanitised. `sniffed_keep` is the
/// extra "Keep" signal from MIME sniffing (already gated on the
/// sniffing rules by the caller).
fn make_prompt(
    disposition: Option<&ParsedDisposition>,
    content_type: Option<&str>,
    url: &str,
    sniffed_keep: bool,
) -> DownloadPrompt {
    let filename = disposition
        .and_then(|d| d.filename.clone())
        .map(|f| sanitize_filename(&f))
        .filter(|f| !f.is_empty())
        .unwrap_or_else(|| filename_from_url(url));
    let needs_keep_confirmation = sniffed_keep
        || payload_needs_keep(content_type, &filename, None);
    DownloadPrompt {
        filename,
        origin: pluto_ipc::origin_of(url),
        // No Content-Length reaches the decision inputs; the shell
        // fills this in from the response headers.
        size: None,
        needs_keep_confirmation,
    }
}

/// "Keep" confirmation for a payload, from the declared type, the
/// (sanitised) filename extension, and/or the sniffed kind: ELF /
/// script / archive payloads. Anything else — including a PDF, an
/// image, or plain text — needs no extra confirmation.
pub fn payload_needs_keep(
    content_type: Option<&str>,
    filename: &str,
    sniffed: Option<SniffedKind>,
) -> bool {
    if matches!(
        sniffed,
        Some(SniffedKind::Elf) | Some(SniffedKind::Script) | Some(SniffedKind::Zip)
    ) {
        return true;
    }
    if let Some(mime) = mime_essence(content_type) {
        if matches!(
            mime.as_str(),
            "application/x-executable"
                | "application/x-elf"
                | "application/x-msdownload"
                | "application/x-sh"
                | "application/x-shellscript"
                | "text/x-shellscript"
                | "application/zip"
                | "application/x-zip-compressed"
                | "application/x-tar"
                | "application/gzip"
                | "application/x-gzip"
                | "application/x-7z-compressed"
                | "application/x-rar-compressed"
                | "application/x-compressed"
        ) {
            return true;
        }
    }
    let lower = filename.to_ascii_lowercase();
    let ext = lower.rsplit('.').next().unwrap_or("");
    // Only an extension after a real dot counts: a bare name with
    // no dot has `rsplit` return the whole name, which must not
    // match (e.g. a file literally named "zip" is not an archive).
    if lower.contains('.') {
        matches!(
            ext,
            "exe" | "elf" | "com" | "scr" | "msi" | "sh" | "bash" | "zsh" | "py" | "pl" | "rb"
                | "ps1" | "bat" | "cmd" | "dmg" | "pkg" | "deb" | "rpm" | "appimage"
                | "zip" | "tar" | "gz" | "tgz" | "bz2" | "xz" | "7z" | "rar"
        )
    } else {
        false
    }
}

/// The "Keep" signal from sniffing alone: sniffed ELF / script /
/// archive bytes, and only when sniffing is allowed at all
/// (`nosniff` false). The caller (see [`decide_with_sniff`]) only
/// passes a prefix when the declared type is absent /
/// `application/octet-stream` / unknown, so this never fires for a
/// declared renderable type.
fn sniffed_keep(body_prefix: &[u8], nosniff: bool) -> bool {
    if nosniff {
        return false;
    }
    matches!(
        sniff_mime(body_prefix),
        Some(SniffedKind::Elf) | Some(SniffedKind::Script) | Some(SniffedKind::Zip)
    )
}

/// Is this MIME essence one Pluto can render inline? The set is
/// deliberately small: HTML documents, plain text, stylesheets,
/// and still images. Everything else — PDF, JSON/XML, audio/video
/// (M8, not built yet), fonts, scripts, archives, executables —
/// Pluto cannot render inline, so it downloads.
pub fn is_renderable_type(mime: &str) -> bool {
    matches!(
        mime,
        "text/html" | "text/plain" | "text/css" | "image/png" | "image/jpeg" | "image/gif"
            | "image/webp" | "image/svg+xml"
    )
}

/// The MIME essence of a `Content-Type` header value: lowercased,
/// parameters (`; charset=…`) stripped, surrounding whitespace
/// trimmed. `None` for a missing/empty value.
pub fn mime_essence(content_type: Option<&str>) -> Option<String> {
    let value = content_type?.split(';').next()?.trim().to_ascii_lowercase();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

// --- Content-Disposition parsing ---

/// A parsed `Content-Disposition` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDisposition {
    /// First token, lowercased: `attachment` or `inline`.
    pub disposition: String,
    /// Suggested filename, if any — `filename*` (RFC 5987) wins
    /// over `filename`.
    pub filename: Option<String>,
}

impl ParsedDisposition {
    /// `true` when the disposition type is `attachment`.
    pub fn is_attachment(&self) -> bool {
        self.disposition == "attachment"
    }
}

/// Parse a `Content-Disposition` header value: disposition type
/// plus `filename` / RFC 5987 `filename*` parameters. Parameter
/// names are case-insensitive; quoted values honour backslash
/// escapes. Returns `None` for an empty value.
pub fn parse_content_disposition(value: &str) -> Option<ParsedDisposition> {
    let mut parts = split_disposition_params(value);
    let disposition = parts.next()?.trim().to_ascii_lowercase();
    if disposition.is_empty() {
        return None;
    }
    let mut filename: Option<String> = None;
    let mut filename_star: Option<String> = None;
    for part in parts {
        let Some((name, raw)) = part.split_once('=') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let raw = raw.trim();
        match name.as_str() {
            "filename" => filename = Some(unquote(raw)),
            "filename*" => filename_star = parse_rfc5987(raw),
            _ => {}
        }
    }
    // RFC 6266: `filename*` takes precedence over `filename`.
    Some(ParsedDisposition {
        disposition,
        filename: filename_star.or(filename),
    })
}

/// Split a header value on `;`, ignoring `;` inside quoted strings.
fn split_disposition_params(value: &str) -> impl Iterator<Item = &str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, c) in value.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_quotes => escaped = true,
            '"' => in_quotes = !in_quotes,
            ';' if !in_quotes => {
                out.push(&value[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&value[start..]);
    out.into_iter()
}

/// Remove surrounding quotes from a parameter value, honouring
/// backslash escapes inside the quotes. Unquoted values pass
/// through unchanged.
fn unquote(raw: &str) -> String {
    let raw = raw.trim();
    if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
        let inner = &raw[1..raw.len() - 1];
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        raw.to_string()
    }
}

/// Parse an RFC 5987 extended parameter value:
/// `charset'language'percent-encoded-value`. Only UTF-8 is
/// accepted (any other charset → `None`, falling back to plain
/// `filename`). Percent-decoding is lossy UTF-8.
fn parse_rfc5987(raw: &str) -> Option<String> {
    let raw = unquote(raw);
    let mut it = raw.splitn(3, '\'');
    let charset = it.next()?;
    let _language = it.next()?;
    let encoded = it.next()?;
    if !charset.eq_ignore_ascii_case("utf-8") {
        return None;
    }
    Some(percent_decode(encoded))
}

/// Percent-decode `%XX` sequences; a malformed `%` passes through
/// literally. Decodes to bytes, then lossy UTF-8.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            )
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Suggested filename from a `Content-Disposition` header value
/// (`filename` or RFC 5987 `filename*`), unsanitised — pass the
/// result through [`sanitize_filename`] before saving.
pub fn filename_from_disposition(disposition: &str) -> Option<String> {
    parse_content_disposition(disposition).and_then(|d| d.filename)
}

// --- Filename sanitisation ---

/// Sanitise a server-suggested filename before it ever touches the
/// filesystem:
///
/// * path separators (`/` and `\`) are stripped — only the last
///   path segment survives, so an absolute path (`/etc/passwd`) or
///   a traversal (`../../secret.txt`) can never escape the download
///   folder;
/// * `.` and `..` segments are dropped;
/// * leading `.`s are stripped (no hidden files);
/// * control characters are removed;
/// * trailing dots/spaces are stripped;
/// * the name is capped at [`MAX_FILENAME_CHARS`] characters;
/// * an empty result becomes `"download"`.
///
/// The caller still joins the result onto the download directory —
/// the sanitised name itself carries no directory component.
pub fn sanitize_filename(raw: &str) -> String {
    let mut name = raw.replace('\\', "/");
    // Keep only the last non-empty, non-dot segment.
    let mut last = String::new();
    for segment in name.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            continue;
        }
        last = segment.to_string();
    }
    name = last;
    // Strip control characters.
    name.retain(|c| !c.is_control());
    // Strip leading dots (no hidden files) and trim whitespace.
    let trimmed = name.trim().trim_start_matches('.');
    // Strip trailing dots/spaces (unsafe on some filesystems).
    let trimmed = trimmed.trim_end_matches(['.', ' ']);
    let mut out: String = trimmed.chars().take(MAX_FILENAME_CHARS).collect();
    if out.is_empty() {
        out.push_str("download");
    }
    out
}

/// Filename from a URL: the last path segment, percent-decoded and
/// sanitised; `"download"` when the URL has no usable segment
/// (e.g. `https://example.com/`).
pub fn filename_from_url(url: &str) -> String {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    // Only a real path segment counts: a bare authority
    // (`https://example.com`, no `/`) has no filename.
    let Some(path) = after_scheme.split_once('/').map(|(_, p)| p) else {
        return "download".to_string();
    };
    let path = path.split(['?', '#']).next().unwrap_or("");
    let segment = path.split('/').next_back().unwrap_or("");
    let decoded = percent_decode(segment);
    sanitize_filename(&decoded)
}

// --- MIME sniffing ---

/// What the magic-byte table recognised in a body prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SniffedKind {
    /// PNG image (`89 50 4E 47 0D 0A 1A 0A`).
    Png,
    /// JPEG image (`FF D8 FF`).
    Jpeg,
    /// GIF image (`GIF8`).
    Gif,
    /// PDF document (`%PDF`).
    Pdf,
    /// ELF executable (`\x7fELF`).
    Elf,
    /// `#!` script.
    Script,
    /// ZIP archive (`PK\x03\x04`).
    Zip,
}

impl SniffedKind {
    /// Is this a kind Pluto can render inline? Images yes; PDF,
    /// ELF, scripts, and archives no — they download.
    pub fn is_renderable(self) -> bool {
        matches!(self, SniffedKind::Png | SniffedKind::Jpeg | SniffedKind::Gif)
    }
}

/// Match a body prefix against the magic table (report §h).
/// Returns `None` when the prefix matches nothing (too short or
/// an unrecognised signature) — "unknown" is not a sniffed type,
/// and the caller must not treat it as one.
pub fn sniff_mime(prefix: &[u8]) -> Option<SniffedKind> {
    if prefix.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(SniffedKind::Png)
    } else if prefix.starts_with(b"\xff\xd8\xff") {
        Some(SniffedKind::Jpeg)
    } else if prefix.starts_with(b"GIF8") {
        Some(SniffedKind::Gif)
    } else if prefix.starts_with(b"%PDF") {
        Some(SniffedKind::Pdf)
    } else if prefix.starts_with(b"\x7fELF") {
        Some(SniffedKind::Elf)
    } else if prefix.starts_with(b"#!") {
        Some(SniffedKind::Script)
    } else if prefix.starts_with(b"PK\x03\x04") {
        Some(SniffedKind::Zip)
    } else {
        None
    }
}

// --- Save pipeline: temp file + SHA-256 during write + mode 0644 ---

/// SHA-256 of a byte slice, lowercase hex. Used by the sink and
/// exposed for tests / callers that already hold the bytes.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, bytes);
    to_hex(digest.as_ref())
}

/// Lowercase hex encoding of raw bytes.
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Force mode `0644` on a path (never `+x`, regardless of what the
/// server suggested). The `+x` bit is only ever set by the user
/// outside the browser.
#[cfg(unix)]
pub fn ensure_mode_0644(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644))
}

/// Non-Unix fallback: permissions are platform-default; there is
/// no `+x` bit to clear.
#[cfg(not(unix))]
pub fn ensure_mode_0644(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// A download being written to a temp file, with its SHA-256
/// computed as the bytes arrive (report §h: "SHA-256 of the file
/// contents, computed as the temp file is written"). Nothing is
/// auto-opened or auto-executed; the finished file keeps mode
/// `0644` and waits for the shell — and, for ELF / script /
/// archive payloads, for the user's explicit "Keep".
pub struct DownloadSink {
    file: File,
    hasher: ring::digest::Context,
    bytes_written: u64,
    path: PathBuf,
}

/// A completed temp-file download: path, SHA-256 (hex) of the
/// exact bytes written, and byte count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedDownload {
    /// Temp-file path. Pass to [`finalize_download`] to move it
    /// into the download folder under its sanitised name.
    pub path: PathBuf,
    /// SHA-256 (lowercase hex) of the bytes written.
    pub sha256_hex: String,
    /// Bytes written.
    pub size: u64,
}

impl DownloadSink {
    /// Create a fresh temp file in `dir` (created if missing).
    /// The name is Pluto-internal (`.pluto-download-<pid>-<nanos>`)
    /// — never the server-suggested name, so a hostile name cannot
    /// influence the temp path. Mode is set to `0644` at creation.
    pub fn create_in(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!(".pluto-download-{}-{nanos}", std::process::id());
        let path = dir.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        ensure_mode_0644(&path)?;
        Ok(DownloadSink {
            file,
            hasher: ring::digest::Context::new(&ring::digest::SHA256),
            bytes_written: 0,
            path,
        })
    }

    /// Temp-file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Write one chunk: appends to the temp file AND feeds the
    /// running SHA-256. The hash therefore covers exactly the
    /// bytes on disk.
    pub fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes_written += bytes.len() as u64;
        Ok(())
    }

    /// Flush, re-assert mode `0644`, and finish the hash.
    pub fn finish(self) -> io::Result<FinishedDownload> {
        self.file.sync_all()?;
        drop(self.file);
        ensure_mode_0644(&self.path)?;
        let digest = self.hasher.finish();
        Ok(FinishedDownload {
            path: self.path,
            sha256_hex: to_hex(digest.as_ref()),
            size: self.bytes_written,
        })
    }
}

/// Move a finished temp download into `dest_dir` under its
/// sanitised `filename`, keeping mode `0644`. If a file with that
/// name already exists, `name (1).ext`, `name (2).ext`, … is used
/// instead — an existing file is never overwritten. The temp file
/// is consumed (renamed, or copied + removed if the rename
/// crosses filesystems). Returns the final path.
pub fn finalize_download(
    finished: &FinishedDownload,
    dest_dir: &Path,
    filename: &str,
) -> io::Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let clean = sanitize_filename(filename);
    let dest = unique_dest(dest_dir, &clean);
    match std::fs::rename(&finished.path, &dest) {
        Ok(()) => {}
        Err(_) => {
            std::fs::copy(&finished.path, &dest)?;
            std::fs::remove_file(&finished.path)?;
        }
    }
    ensure_mode_0644(&dest)?;
    Ok(dest)
}

/// First unused `dest_dir/name`, appending ` (n)` before the
/// extension when the plain name is taken.
fn unique_dest(dest_dir: &Path, filename: &str) -> PathBuf {
    let candidate = dest_dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let (stem, ext) = match filename.rfind('.') {
        Some(i) if i > 0 => (&filename[..i], Some(&filename[i..])),
        _ => (filename, None),
    };
    let mut n = 1u32;
    loop {
        let numbered = match ext {
            Some(ext) => format!("{stem} ({n}){ext}"),
            None => format!("{stem} ({n})"),
        };
        let candidate = dest_dir.join(numbered);
        if !candidate.exists() {
            return candidate;
        }
        n += 1;
    }
}

// --- `.part` resume helper ---

/// The `.part` path for a download whose final path is `dest`:
/// `report.pdf` → `report.pdf.part`. Interrupted downloads keep
/// this file; a fresh download starts from it.
pub fn part_path_for(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// The `Range` request header value to resume a download that
/// already has `existing_bytes` on disk (`bytes=N-`), or `None`
/// when there is nothing to resume from (a fresh download sends
/// no `Range` header).
pub fn resume_range_header(existing_bytes: u64) -> Option<String> {
    if existing_bytes == 0 {
        None
    } else {
        Some(format!("bytes={existing_bytes}-"))
    }
}

/// Is a partial download already complete? True when the server
/// gave a `Content-Length` and the bytes on disk meet or exceed
/// it. With no `Content-Length` there is nothing to compare
/// against, so this is false — the caller must verify some other
/// way (or re-fetch).
pub fn resume_is_complete(existing_bytes: u64, content_length: Option<u64>) -> bool {
    content_length.is_some_and(|len| existing_bytes >= len)
}

/// Does `size` match the `Content-Length` the server advertised?
/// `None` (no length advertised) always passes — there is nothing
/// to check against. Call after a download completes.
pub fn size_matches_content_length(size: u64, content_length: Option<u64>) -> bool {
    content_length.is_none_or(|len| size == len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Decision order ---

    fn download_prompt(d: DownloadDecision) -> DownloadPrompt {
        match d {
            DownloadDecision::Download(p) => p,
            other => panic!("expected Download, got {other:?}"),
        }
    }

    #[test]
    fn attachment_always_downloads_regardless_of_type() {
        // Even a renderable type downloads when the disposition
        // says attachment.
        let d = decide(
            Some("attachment; filename=\"page.html\""),
            Some("text/html"),
            "https://example.com/page",
            false,
        );
        assert_eq!(download_prompt(d).filename, "page.html");
        // Bare `attachment`, no filename: URL basename.
        let d = decide(
            Some("attachment"),
            Some("text/html; charset=utf-8"),
            "https://example.com/report.pdf",
            false,
        );
        assert_eq!(download_prompt(d).filename, "report.pdf");
        // Case-insensitive disposition type.
        assert!(matches!(
            decide(
                Some("Attachment; filename=x.bin"),
                Some("text/html"),
                "https://example.com/x",
                false
            ),
            DownloadDecision::Download(_)
        ));
    }

    #[test]
    fn octet_stream_and_missing_and_unknown_types_download() {
        for content_type in [
            Some("application/octet-stream"),
            Some("application/octet-stream; charset=binary"),
            None,
            Some(""),
            Some("application/x-foo"),
            Some("application/pdf"),
            Some("application/json"),
            Some("video/mp4"),
            Some("audio/mpeg"),
            Some("application/javascript"),
            Some("text/javascript"),
        ] {
            let d = decide(content_type, content_type, "https://example.com/f", false);
            assert!(
                matches!(d, DownloadDecision::Download(_)),
                "type {content_type:?} should download, got {d:?}"
            );
        }
    }

    #[test]
    fn renderable_types_render_inline() {
        for mime in [
            "text/html",
            "text/html; charset=utf-8",
            "TEXT/PLAIN",
            "text/css",
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "image/svg+xml",
        ] {
            let d = decide(None, Some(mime), "https://example.com/f", false);
            assert_eq!(d, DownloadDecision::RenderInline, "type {mime} should render inline");
        }
        // `inline` disposition + renderable type renders inline.
        assert_eq!(
            decide(Some("inline"), Some("text/html"), "https://example.com/f", false),
            DownloadDecision::RenderInline
        );
        // But `inline` + octet-stream still downloads.
        assert!(matches!(
            decide(Some("inline"), Some("application/octet-stream"), "https://example.com/f", false),
            DownloadDecision::Download(_)
        ));
    }

    #[test]
    fn download_attribute_forces_download() {
        // Even a plain HTML page downloads when `<a download>` was used.
        let d = decide(None, Some("text/html"), "https://example.com/page", true);
        assert!(matches!(d, DownloadDecision::Download(_)));
        // Without the attribute the same response renders inline.
        assert_eq!(
            decide(None, Some("text/html"), "https://example.com/page", false),
            DownloadDecision::RenderInline
        );
    }

    #[test]
    fn prompt_carries_filename_origin_and_no_size_yet() {
        let p = download_prompt(decide(
            Some("attachment; filename=\"setup.exe\""),
            Some("application/octet-stream"),
            "https://cdn.example.com:8443/setup.exe",
            false,
        ));
        assert_eq!(p.filename, "setup.exe");
        assert_eq!(p.origin, "https://cdn.example.com:8443");
        // The decision inputs carry no Content-Length.
        assert_eq!(p.size, None);
        // An executable payload needs the extra "Keep".
        assert!(p.needs_keep_confirmation);
    }

    // --- Filename sanitisation ---

    #[test]
    fn sanitize_strips_paths_traversals_and_leading_dots() {
        assert_eq!(sanitize_filename("/etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("C:\\Users\\x\\evil.exe"), "evil.exe");
        assert_eq!(sanitize_filename("../../secret.txt"), "secret.txt");
        assert_eq!(sanitize_filename("..\\..\\secret.txt"), "secret.txt");
        assert_eq!(sanitize_filename(".hidden"), "hidden");
        assert_eq!(sanitize_filename("...dotfile"), "dotfile");
        assert_eq!(sanitize_filename("a/b/c.txt"), "c.txt");
        assert_eq!(sanitize_filename("reports/annual.pdf"), "annual.pdf");
    }

    #[test]
    fn sanitize_strips_control_chars_and_caps_length() {
        assert_eq!(sanitize_filename("a\x00b\x1fc.txt"), "abc.txt");
        assert_eq!(sanitize_filename("a\tb.txt"), "ab.txt");
        let long = "x".repeat(300);
        let capped = sanitize_filename(&format!("{long}.txt"));
        assert_eq!(capped.chars().count(), MAX_FILENAME_CHARS);
        // A normal name passes through untouched.
        assert_eq!(sanitize_filename("report-2026.pdf"), "report-2026.pdf");
    }

    #[test]
    fn sanitize_empty_or_dot_only_becomes_download() {
        assert_eq!(sanitize_filename(""), "download");
        assert_eq!(sanitize_filename(".."), "download");
        assert_eq!(sanitize_filename("/"), "download");
        assert_eq!(sanitize_filename("..."), "download");
    }

    #[test]
    fn filename_from_url_uses_last_segment() {
        assert_eq!(filename_from_url("https://example.com/files/report.pdf"), "report.pdf");
        assert_eq!(
            filename_from_url("https://example.com/files/my%20file.zip?x=1#frag"),
            "my file.zip"
        );
        assert_eq!(filename_from_url("https://example.com/"), "download");
        assert_eq!(filename_from_url("https://example.com"), "download");
    }

    #[test]
    fn rfc5987_filename_star_wins_and_percent_decodes() {
        // `filename*` (UTF-8, percent-encoded) beats plain `filename`.
        let name = filename_from_disposition(
            "attachment; filename=\"fallback.txt\"; filename*=UTF-8''%E2%82%AC%20rates.pdf",
        );
        assert_eq!(name.as_deref(), Some("€ rates.pdf"));
        // A non-UTF-8 charset falls back to plain `filename`.
        let name = filename_from_disposition(
            "attachment; filename=\"plain.txt\"; filename*=ISO-8859-1''caf%E9.txt",
        );
        assert_eq!(name.as_deref(), Some("plain.txt"));
        // Quoted filename with an escaped quote.
        let name = filename_from_disposition("attachment; filename=\"a\\\"b.txt\"");
        assert_eq!(name.as_deref(), Some("a\"b.txt"));
        // No filename at all.
        assert_eq!(filename_from_disposition("attachment"), None);
        assert_eq!(filename_from_disposition("inline"), None);
    }

    // --- MIME sniffing ---

    #[test]
    fn sniff_magic_table() {
        assert_eq!(
            sniff_mime(b"\x89PNG\r\n\x1a\nrest"),
            Some(SniffedKind::Png)
        );
        assert_eq!(sniff_mime(b"\xff\xd8\xff\xe0"), Some(SniffedKind::Jpeg));
        assert_eq!(sniff_mime(b"GIF89a"), Some(SniffedKind::Gif));
        assert_eq!(sniff_mime(b"%PDF-1.7"), Some(SniffedKind::Pdf));
        assert_eq!(sniff_mime(b"\x7fELF\x02\x01"), Some(SniffedKind::Elf));
        assert_eq!(sniff_mime(b"#!/bin/sh\n"), Some(SniffedKind::Script));
        assert_eq!(sniff_mime(b"PK\x03\x04\x14\x00"), Some(SniffedKind::Zip));
        // Too short / unrecognised → no sniffed type.
        assert_eq!(sniff_mime(b""), None);
        assert_eq!(sniff_mime(b"\x89PN"), None);
        assert_eq!(sniff_mime(b"<html>"), None);
        assert_eq!(sniff_mime(b"hello world"), None);
        // Sniffed images are renderable kinds; PDF/ELF/script/ZIP are not.
        assert!(SniffedKind::Png.is_renderable());
        assert!(!SniffedKind::Pdf.is_renderable());
        assert!(!SniffedKind::Elf.is_renderable());
    }

    #[test]
    fn sniffing_never_promotes_to_inline() {
        // `application/octet-stream` whose bytes are a PNG stays a
        // download — sniffing may only narrow toward download.
        let d = decide_with_sniff(
            None,
            Some("application/octet-stream"),
            "https://example.com/f",
            false,
            b"\x89PNG\r\n\x1a\n",
            false,
        );
        assert!(matches!(d, DownloadDecision::Download(_)));
        // A declared renderable type is never re-sniffed: `text/plain`
        // whose body starts with `%PDF` still renders inline.
        let d = decide_with_sniff(
            None,
            Some("text/plain"),
            "https://example.com/f",
            false,
            b"%PDF-1.7",
            false,
        );
        assert_eq!(d, DownloadDecision::RenderInline);
    }

    #[test]
    fn sniffing_marks_keep_for_untyped_dangerous_payloads() {
        // Unknown type + ELF bytes → download needing "Keep".
        let p = download_prompt(decide_with_sniff(
            None,
            None,
            "https://example.com/f",
            false,
            b"\x7fELF\x02\x01",
            false,
        ));
        assert!(p.needs_keep_confirmation);
        // Same bytes, `nosniff` set → sniffing is skipped entirely,
        // so no Keep marker from the bytes (neutral name + no type).
        let p = download_prompt(decide_with_sniff(
            None,
            None,
            "https://example.com/f",
            false,
            b"\x7fELF\x02\x01",
            true,
        ));
        assert!(!p.needs_keep_confirmation);
        // ZIP bytes likewise.
        let p = download_prompt(decide_with_sniff(
            None,
            Some("application/octet-stream"),
            "https://example.com/f",
            false,
            b"PK\x03\x04",
            false,
        ));
        assert!(p.needs_keep_confirmation);
        // But sniffed PNG bytes do not demand "Keep".
        let p = download_prompt(decide_with_sniff(
            None,
            Some("application/octet-stream"),
            "https://example.com/f",
            false,
            b"\x89PNG\r\n\x1a\n",
            false,
        ));
        assert!(!p.needs_keep_confirmation);
    }

    // --- Keep-confirmation detection ---

    #[test]
    fn keep_confirmation_detection() {
        // Executable / script / archive types and extensions.
        assert!(payload_needs_keep(Some("application/x-executable"), "tool", None));
        assert!(payload_needs_keep(Some("application/x-msdownload"), "tool", None));
        assert!(payload_needs_keep(Some("application/zip"), "a.bin", None));
        assert!(payload_needs_keep(None, "setup.exe", None));
        assert!(payload_needs_keep(None, "INSTALL.SH", None));
        assert!(payload_needs_keep(None, "archive.tar.gz", None));
        assert!(payload_needs_keep(None, "script.py", None));
        assert!(payload_needs_keep(None, "x", Some(SniffedKind::Elf)));
        assert!(payload_needs_keep(None, "x", Some(SniffedKind::Script)));
        assert!(payload_needs_keep(None, "x", Some(SniffedKind::Zip)));
        // Ordinary payloads do not.
        assert!(!payload_needs_keep(Some("text/plain"), "notes.txt", None));
        assert!(!payload_needs_keep(Some("image/png"), "photo.png", None));
        assert!(!payload_needs_keep(Some("application/pdf"), "doc.pdf", None));
        assert!(!payload_needs_keep(None, "photo.png", Some(SniffedKind::Png)));
        assert!(!payload_needs_keep(None, "doc", Some(SniffedKind::Pdf)));
        // A bare name that merely equals an extension word is not one.
        assert!(!payload_needs_keep(None, "zip", None));
    }

    // --- Save pipeline: temp file + SHA-256 + mode 0644 ---

    #[test]
    fn sha256_known_vector() {
        // SHA-256("abc") — the FIPS 180-4 example.
        assert_eq!(
            hash_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hash_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sink_hashes_during_write_and_saves_0644_no_exec() {
        let dir = std::env::temp_dir().join(format!(
            "pluto-download-test-{}-sink",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut sink = DownloadSink::create_in(&dir).expect("create sink");
        sink.write_chunk(b"abc").unwrap();
        sink.write_chunk(b"def").unwrap();
        assert_eq!(sink.bytes_written(), 6);
        // Mode is 0644 (no exec bits) from creation, before finish.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(sink.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o644, "temp file must be 0644, got {mode:o}");
            assert_eq!(mode & 0o111, 0, "temp file must never be +x");
        }
        let finished = sink.finish().expect("finish sink");
        assert_eq!(finished.size, 6);
        assert_eq!(finished.sha256_hex, hash_bytes(b"abcdef"));
        // The hash covers exactly the bytes on disk.
        assert_eq!(std::fs::read(&finished.path).unwrap(), b"abcdef");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&finished.path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o644);
            assert_eq!(mode & 0o111, 0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn finalize_moves_to_sanitised_name_0644_and_never_overwrites() {
        let dir = std::env::temp_dir().join(format!(
            "pluto-download-test-{}-finalize",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // A hostile server-suggested name is sanitised at finalize.
        let mut sink = DownloadSink::create_in(&dir).unwrap();
        sink.write_chunk(b"data").unwrap();
        let finished = sink.finish().unwrap();
        let dest = finalize_download(&finished, &dir, "../../etc/passwd").unwrap();
        assert_eq!(dest.file_name().unwrap(), "passwd");
        assert_eq!(std::fs::read(&dest).unwrap(), b"data");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dest).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o644);
            assert_eq!(mode & 0o111, 0);
        }
        // A second file with the same name gets a numbered name,
        // never overwriting the first.
        let mut sink = DownloadSink::create_in(&dir).unwrap();
        sink.write_chunk(b"data2").unwrap();
        let finished = sink.finish().unwrap();
        let dest2 = finalize_download(&finished, &dir, "passwd").unwrap();
        assert_eq!(dest2.file_name().unwrap(), "passwd (1)");
        assert_eq!(std::fs::read(&dest).unwrap(), b"data");
        assert_eq!(std::fs::read(&dest2).unwrap(), b"data2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- `.part` resume helper ---

    #[test]
    fn resume_helpers() {
        assert_eq!(resume_range_header(0), None);
        assert_eq!(resume_range_header(12345).as_deref(), Some("bytes=12345-"));
        assert_eq!(
            part_path_for(Path::new("/tmp/dl/report.pdf")),
            PathBuf::from("/tmp/dl/report.pdf.part")
        );
        // Complete when the bytes on disk meet the advertised length.
        assert!(resume_is_complete(100, Some(100)));
        assert!(resume_is_complete(150, Some(100)));
        assert!(!resume_is_complete(99, Some(100)));
        assert!(!resume_is_complete(100, None));
        // Size check against Content-Length.
        assert!(size_matches_content_length(100, Some(100)));
        assert!(!size_matches_content_length(99, Some(100)));
        assert!(size_matches_content_length(99, None));
    }

    // --- Mixed downloads (reuses pluto_net::mixed::is_mixed_download) ---

    #[test]
    fn mixed_download_blocked_not_upgraded() {
        // Secure context + http URL + attachment → blocked.
        assert!(is_mixed_download_blocked(
            "https://example.com",
            "http://cdn.example.com/f.zip",
            Some("attachment; filename=\"f.zip\"")
        ));
        assert_eq!(
            decide_for_navigation(
                "https://example.com",
                Some("attachment"),
                Some("application/zip"),
                "http://cdn.example.com/f.zip",
                false
            ),
            DownloadDecision::BlockMixed
        );
        // Same response over https is a normal download.
        assert!(matches!(
            decide_for_navigation(
                "https://example.com",
                Some("attachment"),
                Some("application/zip"),
                "https://cdn.example.com/f.zip",
                false
            ),
            DownloadDecision::Download(_)
        ));
        // An http page has no secure context to protect.
        assert!(!is_mixed_download_blocked(
            "http://example.com",
            "http://cdn.example.com/f.zip",
            Some("attachment")
        ));
        // No disposition → not a download → not blocked here.
        assert!(!is_mixed_download_blocked(
            "https://example.com",
            "http://cdn.example.com/f.zip",
            None
        ));
        // `inline` is not a download.
        assert!(!is_mixed_download_blocked(
            "https://example.com",
            "http://cdn.example.com/f.zip",
            Some("inline")
        ));
    }
}
