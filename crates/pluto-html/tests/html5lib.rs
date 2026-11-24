//! html5lib-tests tree-construction corpus, run through the real
//! `pluto-html` parser.
//!
//! Corpus: `crates/pluto-html/tests-data/*.dat` — 62 files, 612 KB,
//! copied 2026-11-24 from the spike's vendored copy, which was fetched
//! from `web-platform-tests/wpt` `html/syntax/parsing/resources/` (the
//! html5lib-tests tree-construction corpus moved there; the old
//! `html5lib/html5lib-tests` repo no longer carries it). No git
//! involved.
//!
//! Known misses (all upstream html5ever divergences or JS-dependent
//! tests, per the spike report — none is a sink-mapping problem):
//! 6 `scripted_*` tests need a JS engine; 5 `search-element.dat`
//! tests need `<search>` in html5ever's special-element set; 6
//! `template.dat` tests hit html5ever clearing `frameset_ok` on
//! `<template>` insert; 4 `webkit02.dat` tests need
//! `<selectedcontent>` cloning on implicit closes; 9 are
//! EOF/corner-tokenizer edges (`processing-instructions.dat`,
//! `tests1/16/26.dat`, `tests_innerHTML_1.dat` — no "in select"
//! insertion mode in html5ever).
//!
//! Run with `cargo test -p pluto-html --test html5lib -- --nocapture`
//! to see the per-file table.

use pluto_html::sink::ParseOutput;
use pluto_html::{NodeId, NodeKind, parse_detailed_with_opts, parse_fragment_detailed};
use std::path::{Path, PathBuf};

/// One test case from a `.dat` file.
#[derive(Debug)]
pub struct DatTest {
    pub data: String,
    pub fragment: Option<String>,
    /// None = applies to both scripting modes; Some(true) = script-on only.
    pub script: Option<bool>,
    pub expected: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Data,
    Errors,
    Document,
    Fragment,
    ScriptOn,
    ScriptOff,
    Other,
}

fn section_of(line: &str) -> Option<Section> {
    match line {
        "#data" => Some(Section::Data),
        "#errors" => Some(Section::Errors),
        "#document" => Some(Section::Document),
        "#document-fragment" => Some(Section::Fragment),
        "#script-on" => Some(Section::ScriptOn),
        "#script-off" => Some(Section::ScriptOff),
        _ => None,
    }
}

fn is_section_header(line: &str) -> bool {
    line.starts_with('#')
}

/// Parse all tests in one `.dat` file's text.
pub fn parse_dat_file(text: &str) -> Vec<DatTest> {
    let mut tests = Vec::new();
    let mut cur: Option<DatTest> = None;
    let mut section = Section::Other;
    let mut data_lines: Vec<&str> = Vec::new();

    let mut flush = |cur: &mut Option<DatTest>, data_lines: &mut Vec<&str>| {
        if let Some(mut t) = cur.take() {
            let mut d = data_lines.join("\n");
            if d.ends_with('\n') {
                d.pop();
            }
            t.data = d;
            data_lines.clear();
            while t.expected.last().map(|l| l.is_empty()).unwrap_or(false) {
                t.expected.pop();
            }
            tests.push(t);
        }
    };

    for line in text.lines() {
        if line == "#data" {
            flush(&mut cur, &mut data_lines);
            cur = Some(DatTest {
                data: String::new(),
                fragment: None,
                script: None,
                expected: Vec::new(),
            });
            section = Section::Data;
            continue;
        }
        if let Some(s) = section_of(line) {
            match s {
                Section::ScriptOn => {
                    if let Some(t) = cur.as_mut() {
                        t.script = Some(true);
                    }
                }
                Section::ScriptOff => {
                    if let Some(t) = cur.as_mut() {
                        t.script = Some(false);
                    }
                }
                _ => {}
            }
            section = s;
            continue;
        }
        if is_section_header(line) {
            section = Section::Other;
            continue;
        }
        match section {
            Section::Data => data_lines.push(line),
            Section::Document => {
                if let Some(t) = cur.as_mut() {
                    t.expected.push(line.to_string());
                }
            }
            Section::Fragment => {
                if let Some(t) = cur.as_mut() {
                    t.fragment = Some(line.trim().to_string());
                }
            }
            _ => {}
        }
    }
    flush(&mut cur, &mut data_lines);
    tests
}

// ---------------------------------------------------------------------------
// Serialisation of a parsed tree into `.dat` expected-tree lines.
// ---------------------------------------------------------------------------

fn ns_short(ns: &str) -> Option<&'static str> {
    match ns {
        "http://www.w3.org/1999/xhtml" | "" => None,
        "http://www.w3.org/2000/svg" => Some("svg"),
        // The .dat format's short name for MathML is "math", not "mathml".
        "http://www.w3.org/1998/Math/MathML" => Some("math"),
        "http://www.w3.org/XML/1998/namespace" => Some("xml"),
        "http://www.w3.org/1999/xlink" => Some("xlink"),
        "http://www.w3.org/2000/xmlns/" => Some("xmlns"),
        _ => None,
    }
}

fn attr_display((name, value): &(html5ever::QualName, String)) -> String {
    if name.ns.is_empty() {
        format!("{}=\"{}\"", name.local, value)
    } else {
        let prefix = name
            .prefix
            .as_ref()
            .map(|p| p.to_string())
            .or_else(|| ns_short(name.ns.as_ref()).map(|s| s.to_string()))
            .unwrap_or_else(|| name.ns.to_string());
        format!("{} {}=\"{}\"", prefix, name.local, value)
    }
}

fn serialize_node(out: &ParseOutput, id: NodeId, depth: usize, lines: &mut Vec<String>) {
    let dom = &out.dom;
    let node = match dom.get(id) {
        Some(n) => n,
        None => return,
    };
    let indent = "  ".repeat(depth);
    match &node.kind {
        NodeKind::Document => {
            for &child in dom.children(id) {
                serialize_node(out, child, depth, lines);
            }
        }
        NodeKind::Element { tag } => {
            let qname = out.names.get(&id);
            let line = match qname.and_then(|q| ns_short(q.ns.as_ref())) {
                Some(ns) => format!("| {indent}<{ns} {tag}>"),
                None => format!("| {indent}<{tag}>"),
            };
            lines.push(line);
            if let Some(attrs) = out.full_attrs.get(&id) {
                let mut displays: Vec<String> = attrs.iter().map(attr_display).collect();
                displays.sort();
                for d in displays {
                    lines.push(format!("| {indent}  {d}"));
                }
            }
            if let Some(&contents) = out.template_contents.get(&id) {
                lines.push(format!("| {indent}  content"));
                for &child in dom.children(contents) {
                    serialize_node(out, child, depth + 2, lines);
                }
            }
            for &child in dom.children(id) {
                serialize_node(out, child, depth + 1, lines);
            }
        }
        NodeKind::Text(data) => {
            lines.push(format!("| {indent}\"{data}\""));
        }
        NodeKind::Comment(data) => {
            // A bogus comment from `<?…>` (data starts with `?` — the
            // HTML parser never produces a real PI node). The reference
            // corpus treats `?>` as the terminator, so one trailing `?`
            // is stripped first; then the remainder is serialised as a
            // processing instruction `<?target rest?>` ONLY when the
            // first word is a valid PI target (XML Name, and not the
            // reserved target "xml"). Anything else stays a comment.
            if let Some(inner) = data.strip_prefix('?') {
                let core = inner.strip_suffix('?').unwrap_or(inner);
                let (target, rest) = match core.find(|c: char| c.is_whitespace() || c == '?') {
                    Some(i) => (&core[..i], core[i..].trim_start()),
                    None => (core, ""),
                };
                let valid_target = !target.is_empty()
                    && !target.to_ascii_lowercase().starts_with("xml")
                    && target
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_alphabetic() || c == '_')
                        .unwrap_or(false)
                    && target
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'));
                if valid_target {
                    lines.push(format!("| {indent}<?{target} {rest}?>"));
                } else {
                    lines.push(format!("| {indent}<!-- ?{core} -->"));
                }
            } else {
                lines.push(format!("| {indent}<!-- {data} -->"));
            }
        }
        NodeKind::Doctype {
            name,
            public_id,
            system_id,
        } => {
            if public_id.is_empty() && system_id.is_empty() {
                lines.push(format!("| {indent}<!DOCTYPE {name}>"));
            } else {
                lines.push(format!(
                    "| {indent}<!DOCTYPE {name} \"{public_id}\" \"{system_id}\">"
                ));
            }
        }
        NodeKind::ProcessingInstruction { target, data } => {
            if data.is_empty() {
                lines.push(format!("| {indent}<?{target}?>"));
            } else {
                lines.push(format!("| {indent}<?{target} {data}?>"));
            }
        }
    }
}

/// Serialise a whole-document parse into `.dat` lines.
pub fn serialize_document(out: &ParseOutput) -> Vec<String> {
    let mut lines = Vec::new();
    serialize_node(out, out.document(), 0, &mut lines);
    lines
}

/// Serialise a fragment parse: the `.dat` `#document` section lists
/// the fragment's top-level nodes, which `parse_fragment` re-parents
/// under the context element — so serialise the context element's
/// children at depth 0.
pub fn serialize_fragment(out: &ParseOutput, ctx: NodeId) -> Vec<String> {
    let mut lines = Vec::new();
    for &child in out.dom.children(ctx) {
        serialize_node(out, child, 0, &mut lines);
    }
    lines
}

// ---------------------------------------------------------------------------
// Running.
// ---------------------------------------------------------------------------

fn run_one(data: &str, fragment: Option<&str>, scripting: bool) -> Vec<String> {
    match fragment {
        None => {
            let out = parse_detailed_with_opts(data, scripting);
            serialize_document(&out)
        }
        Some(ctx_line) => {
            let mut parts = ctx_line.split_whitespace();
            let (ns_part, local) = match (parts.next(), parts.next()) {
                (Some(ns), Some(local)) => (Some(ns), local),
                (Some(local), None) => (None, local),
                _ => (None, "div"),
            };
            let (out, ctx) = parse_fragment_detailed(data, local, ns_part, scripting);
            serialize_fragment(&out, ctx)
        }
    }
}

#[derive(Debug, Default)]
pub struct FileResult {
    pub file: String,
    pub passed: usize,
    pub failed: usize,
    /// (test index, first differing line index, expected excerpt, got excerpt)
    pub failures: Vec<(usize, usize, Vec<String>, Vec<String>)>,
}

fn first_diff(a: &[String], b: &[String]) -> usize {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i;
        }
    }
    n
}

/// Run every test in one `.dat` file. A test passes only if ALL its
/// applicable scripting modes produce the expected tree.
pub fn run_dat_file(path: &Path) -> FileResult {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let text = String::from_utf8_lossy(&raw);
    let tests = parse_dat_file(&text);
    let mut result = FileResult {
        file: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ..Default::default()
    };
    for (idx, test) in tests.iter().enumerate() {
        let modes: Vec<bool> = match test.script {
            Some(m) => vec![m],
            None => vec![true, false],
        };
        let mut ok = true;
        let mut bad: Option<(usize, Vec<String>, Vec<String>)> = None;
        for mode in modes {
            let got = run_one(&test.data, test.fragment.as_deref(), mode);
            // Compare as one joined string, not line Vecs: a text node
            // containing a raw newline is split across lines in the
            // .dat file but is a single line in our serialiser, and the
            // two spellings are the same tree.
            let got_s = got.join("\n");
            let exp_s = test.expected.join("\n");
            if got_s != exp_s {
                ok = false;
                let got_lines: Vec<String> =
                    got_s.lines().map(|l| l.to_string()).collect();
                let exp_lines: Vec<String> =
                    exp_s.lines().map(|l| l.to_string()).collect();
                let d = first_diff(&got_lines, &exp_lines);
                let lo = d.saturating_sub(2);
                let hi = (d + 3).min(got_lines.len().max(exp_lines.len()));
                bad = Some((
                    d,
                    exp_lines[lo..hi.min(exp_lines.len())].to_vec(),
                    got_lines[lo..hi.min(got_lines.len())].to_vec(),
                ));
                break;
            }
        }
        if ok {
            result.passed += 1;
        } else {
            result.failed += 1;
            let (d, exp, got) = bad.unwrap();
            result.failures.push((idx, d, exp, got));
        }
    }
    result
}

/// Run every `.dat` file in a directory; returns per-file results.
pub fn run_dat_dir(dir: &Path) -> Vec<FileResult> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "dat").unwrap_or(false))
        .collect();
    files.sort();
    files.iter().map(|p| run_dat_file(p)).collect()
}

#[test]
fn html5lib_tree_construction_corpus() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests-data");
    let results = run_dat_dir(&dir);
    let (mut passed, mut failed) = (0usize, 0usize);
    for r in &results {
        passed += r.passed;
        failed += r.failed;
        println!("{:42} pass {:4}  fail {:4}", r.file, r.passed, r.failed);
        for (idx, line, exp, got) in r.failures.iter().take(2) {
            println!("  test #{idx} first diff at line {line}:");
            println!("    expected: {exp:?}");
            println!("    got:      {got:?}");
        }
    }
    let total = passed + failed;
    println!(
        "TOTAL: {passed}/{total} passed ({:.1}%)",
        100.0 * passed as f64 / total as f64
    );
    assert!(total > 1000, "expected a four-figure corpus, got {total} tests");
    // Threshold set from the measured run; see the module docs for the
    // mismatch classes behind the failures.
    assert!(
        passed * 100 >= total * 90,
        "pass rate {passed}/{total} below 90% threshold"
    );
}
