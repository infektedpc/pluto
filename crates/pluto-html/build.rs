//! Build-time generation of the named character-reference table
//! from the vendored WHATWG `entities.json` (source of truth — no
//! hand-typed entity names). Emits `entities_table.rs` into OUT_DIR
//! as a sorted `&[(&str, &str)]` (name without the leading `&`,
//! with or without the trailing `;`, as it appears in the file).

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = Path::new(&manifest).join("entities.json");
    let text = fs::read_to_string(&src).expect("read entities.json");
    println!("cargo:rerun-if-changed=entities.json");

    // Minimal JSON scan: entries look like
    //   "&AElig;": { "codepoints": [198], "characters": "\u00C6" },
    // We only need the key and the "characters" string value.
    let mut entries: Vec<(String, String)> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        // Parse a JSON string starting at i.
        let (s, next) = match parse_json_string(&text, i) {
            Some(v) => v,
            None => break,
        };
        i = next;
        if !s.starts_with('&') {
            continue;
        }
        // Expect `:` then `{` ... `"characters": "..."`.
        let rest = &text[i..];
        let Some(colon) = rest.find(':') else { break };
        let after = &rest[colon + 1..];
        let Some(cpos) = after.find("\"characters\"") else {
            continue;
        };
        let after_chars = &after[cpos + "\"characters\"".len()..];
        let Some(c2) = after_chars.find(':') else { continue };
        let val_start = after_chars[c2 + 1..].trim_start();
        let Some(q) = val_start.find('"') else { continue };
        let abs = text.len() - val_start.len() + q;
        let (value, _) = match parse_json_string(&text, abs) {
            Some(v) => v,
            None => continue,
        };
        entries.push((s.trim_start_matches('&').to_string(), value));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.dedup_by(|a, b| a.0 == b.0);

    let out_dir = env::var("OUT_DIR").unwrap();
    let mut out = String::new();
    out.push_str("// Generated from entities.json by build.rs — do not edit.\n");
    out.push_str("pub static ENTITIES: &[(&str, &str)] = &[\n");
    for (name, value) in &entries {
        out.push_str(&format!("    ({:?}, {:?}),\n", name, value));
    }
    out.push_str("];\n");
    fs::write(Path::new(&out_dir).join("entities_table.rs"), out).unwrap();
}

/// Parse a JSON string literal at byte offset `start` (which must
/// point at the opening `"`). Returns (decoded string, offset just
/// past the closing `"`).
fn parse_json_string(text: &str, start: usize) -> Option<(String, usize)> {
    let b = text.as_bytes();
    if b.get(start) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'"' => return Some((out, i + 1)),
            b'\\' => {
                let e = *b.get(i + 1)?;
                match e {
                    b'"' => { out.push('"'); i += 2; }
                    b'\\' => { out.push('\\'); i += 2; }
                    b'/' => { out.push('/'); i += 2; }
                    b'n' => { out.push('\n'); i += 2; }
                    b't' => { out.push('\t'); i += 2; }
                    b'r' => { out.push('\r'); i += 2; }
                    b'b' => { out.push('\u{8}'); i += 2; }
                    b'f' => { out.push('\u{c}'); i += 2; }
                    b'u' => {
                        let hex = text.get(i + 2..i + 6)?;
                        let cp = u32::from_str_radix(hex, 16).ok()?;
                        // Surrogate pairs in entities.json are two
                        // \uXXXX escapes; decode each half and combine
                        // below via push of the pair when needed.
                        if (0xD800..0xDC00).contains(&cp) {
                            let lo_hex = text.get(i + 8..i + 12)?;
                            if text.get(i + 6..i + 8) == Some("\\u") {
                                let lo = u32::from_str_radix(lo_hex, 16).ok()?;
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                i += 12;
                            } else {
                                out.push('\u{FFFD}');
                                i += 6;
                            }
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            i += 6;
                        }
                    }
                    _ => return None,
                }
            }
            _ => {
                // Copy one UTF-8 char.
                let ch_len = utf8_len(b[i]);
                let s = text.get(i..i + ch_len)?;
                out.push_str(s);
                i += ch_len;
            }
        }
    }
    None
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}
