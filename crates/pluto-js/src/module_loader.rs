//! ES module loader (Stage 12 Rec 1).
//!
//! The rquickjs loader API is **synchronous**: `Loader::load` runs
//! inside `Module::evaluate`, during linking — it cannot fetch over
//! the network. So the loader never fetches. The browser's load task
//! pre-fetches the whole import graph **before** the module executes
//! (see `fetch_script` in `pluto-browser::page`, which scans each
//! fetched module with [`scan_module_imports`] and fetches the
//! imports through the normal loader, with the Stage 9 gates —
//! there is no second fetch path), and the sources land in the
//! shared [`ModuleSources`] map. The realm's loader serves from
//! that map and returns a loading error on a cache miss, so a
//! module whose import was never fetched rejects cleanly instead
//! of hanging or poisoning the realm.
//!
//! Resolution is URL-only: relative (`./`, `../`) and absolute
//! (`/`, `https://…`) specifiers resolve against the importing
//! module's URL. **Bare specifiers are rejected** — there is no
//! import map in M4 (TODO M5); a bare `import "lodash"` rejects
//! with a message saying so. `data:` specifiers are rejected the
//! same way (M4 does not decode inline modules).
//!
//! Dynamic `import()` goes through the same loader, so a
//! string-literal dynamic import that the pre-fetch scan saw works;
//! a runtime-computed specifier was never pre-fetched, so the
//! loader misses and the dynamic `import()` promise rejects with
//! the loading error. That is the documented M4 limit, not a bug.
//!
//! No bytecode cache lives in the loader, deliberately: QuickJS
//! dedups loaded modules by resolved name per runtime (verified
//! by `probe_engine_dedups_loaded_modules_by_name` — a second
//! entry importing the same URL does not re-run the module body
//! and never calls the loader again), so an import is compiled at
//! most once per realm and a loader-side cache could never get a
//! second hit. The entry-module cache (`TabJs::bytecode`) stays —
//! it serves repeated *explicit* loads of the same entry, which
//! the engine does not dedup for the host.
//!
//! Classic scripts never go through this loader or the bytecode
//! cache: the cache API is module-shaped, and compiling a classic
//! script as a module would change its semantics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rquickjs::{
    Ctx, Error, Module,
    loader::{ImportAttributes, Loader, Resolver},
};

/// Pre-fetched module sources, keyed by absolute URL (no fragment).
/// Shared between the browser's load task (writer) and the realm's
/// loader (reader). One instance per navigation: `TabJs::navigate`
/// replaces it, so a dying load task's late inserts land in the old
/// map, never in the new document's.
#[derive(Debug, Clone, Default)]
pub struct ModuleSources {
    inner: Arc<Mutex<HashMap<String, String>>>,
}

impl ModuleSources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, url: &str, source: String) {
        self.inner.lock().unwrap().insert(url.to_string(), source);
    }

    pub fn get(&self, url: &str) -> Option<String> {
        self.inner.lock().unwrap().get(url).cloned()
    }

    pub fn contains(&self, url: &str) -> bool {
        self.inner.lock().unwrap().contains_key(url)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// True when `spec` is a bare specifier: no scheme, not relative
/// (`./`, `../`), not root-absolute (`/`), not scheme-relative
/// (`//`). Bare specifiers resolve only through an import map,
/// which M4 does not have.
pub fn is_bare_specifier(spec: &str) -> bool {
    let spec = spec.trim();
    if spec.is_empty() {
        return true;
    }
    if spec.starts_with("./")
        || spec.starts_with("../")
        || spec.starts_with('/')
    {
        return false;
    }
    if spec.contains("://") || spec.starts_with("//") {
        return false;
    }
    // A scheme (`https:foo`, `blob:…`) without `//` is absolute.
    if let Some(colon) = spec.find(':') {
        let scheme = &spec[..colon];
        if !scheme.is_empty()
            && scheme
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        {
            return false;
        }
    }
    true
}

/// Resolve one import specifier against the importing module's URL
/// (the same resolution the pre-fetch and the loader's resolver
/// both use, so they can never disagree). Returns the absolute URL
/// with the fragment stripped (two spellings that differ only by
/// fragment are the same module).
pub fn resolve_module_specifier(base: &str, spec: &str) -> Result<String, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("empty module specifier".to_string());
    }
    if is_bare_specifier(spec) {
        return Err(format!(
            "bare module specifier '{spec}' requires an import map (M5)"
        ));
    }
    if spec.starts_with("data:") {
        return Err("data: module specifiers are not supported in M4".to_string());
    }
    if let Ok(base_url) = url::Url::parse(base) {
        return base_url.join(spec).map(|mut u| {
            u.set_fragment(None);
            u.to_string()
        }).map_err(|e| {
            format!("cannot resolve module specifier '{spec}' against '{base}': {e}")
        });
    }
    // Unparseable base (e.g. `about:blank`): the same naive join
    // the preload scanner uses, then strip the fragment.
    let mut joined = pluto_html::resolve_url(base, spec);
    if let Some(i) = joined.find('#') {
        joined.truncate(i);
    }
    Ok(joined)
}

/// Scan JS source for module specifiers, in order, deduped:
/// static `import … from '…'`, side-effect `import '…'`,
/// `export … from '…'`, and string-literal dynamic `import('…')`.
/// A comment / string / template-literal aware walk — an `import`
/// inside a string, a comment, or a template literal (outside
/// `${…}`) is not a specifier. A dynamic `import()` with a
/// non-literal argument is ignored (it cannot be pre-fetched).
/// Bare specifiers are returned as written; the caller decides
/// what to do with them (the pre-fetch skips them, the resolver
/// rejects them).
pub fn scan_module_imports(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;

    // Frames: code (with its own brace depth, so a `}` at depth 0
    // closes the enclosing template interpolation) and template
    // bodies (only backtick / `${` / escapes are significant).
    enum Frame {
        Code { brace_depth: usize },
        Template,
    }
    let mut frames: Vec<Frame> = vec![Frame::Code { brace_depth: 0 }];

    // Read one string literal starting at `i` (quote already
    // checked); returns its contents and the index just past the
    // closing quote. `None` when unterminated. Slices the original
    // source, so non-ASCII stays intact.
    fn scan_string(source: &str, mut i: usize, quote: u8) -> Option<(String, usize)> {
        let bytes = source.as_bytes();
        let start = i + 1;
        i += 1; // opening quote
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2, // escaped char never closes the string
                b if b == quote => return Some((source[start..i].to_string(), i + 1)),
                _ => i += 1,
            }
        }
        None
    }

    // At `i`, read an identifier; returns (ident, next_i).
    fn read_ident(bytes: &[u8], mut i: usize) -> (String, usize) {
        let mut s = String::new();
        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'$' {
                s.push(b as char);
                i += 1;
            } else {
                break;
            }
        }
        (s, i)
    }

    // Skip whitespace and comments from `i`; returns the next
    // significant index.
    fn skip_ws_comments(bytes: &[u8], mut i: usize) -> usize {
        loop {
            while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                i += 1;
            }
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            } else {
                break;
            }
        }
        i
    }

    // After `import` / `export`, scan forward for `from 'spec'`
    // (or a bare string literal right after `import`); returns the
    // specifier when found. `import.meta` and `export default …`
    // yield nothing.
    fn scan_from_spec(
        source: &str,
        mut i: usize,
        after_import: bool,
    ) -> Option<(String, usize)> {
        let bytes = source.as_bytes();
        i = skip_ws_comments(bytes, i);
        if after_import && i < bytes.len() && (bytes[i] == b'\'' || bytes[i] == b'"') {
            // Side-effect import: `import 'spec'`.
            return scan_string(source, i, bytes[i]);
        }
        loop {
            i = skip_ws_comments(bytes, i);
            if i >= bytes.len() || bytes[i] == b';' {
                return None;
            }
            if bytes[i] == b'\'' || bytes[i] == b'"' {
                return None;
            }
            if bytes[i] == b'{' {
                // Binding list: skip to the matching `}`.
                let mut depth = 0usize;
                while i < bytes.len() {
                    match bytes[i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        b'\'' | b'"' => {
                            let q = bytes[i];
                            match scan_string(source, i, q) {
                                Some((_, next)) => i = next,
                                None => return None,
                            }
                            continue;
                        }
                        _ => {}
                    }
                    i += 1;
                }
                continue;
            }
            let (ident, next) = read_ident(bytes, i);
            if ident == "from" {
                let j = skip_ws_comments(bytes, next);
                if j < bytes.len() && (bytes[j] == b'\'' || bytes[j] == b'"') {
                    return scan_string(source, j, bytes[j]);
                }
                return None;
            }
            if ident.is_empty() {
                i += 1;
            } else {
                i = next;
            }
        }
    }

    while i < bytes.len() {
        match frames.last() {
            Some(Frame::Template) => {
                match bytes[i] {
                    b'\\' => i += 2,
                    b'`' => {
                        frames.pop();
                        i += 1;
                    }
                    b'$' if i + 1 < bytes.len() && bytes[i + 1] == b'{' => {
                        frames.push(Frame::Code { brace_depth: 0 });
                        i += 2;
                    }
                    _ => i += 1,
                }
                continue;
            }
            _ => {}
        }
        // Top frame is Code.
        let b = bytes[i];
        match b {
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            b'\'' | b'"' => match scan_string(source, i, b) {
                Some((_, next)) => i = next,
                None => break,
            },
            b'`' => {
                frames.push(Frame::Template);
                i += 1;
            }
            b'{' => {
                if let Some(Frame::Code { brace_depth }) = frames.last_mut() {
                    *brace_depth += 1;
                }
                i += 1;
            }
            b'}' => {
                let mut close_interpolation = false;
                if let Some(Frame::Code { brace_depth }) = frames.last_mut() {
                    if *brace_depth > 0 {
                        *brace_depth -= 1;
                    } else {
                        close_interpolation = true;
                    }
                }
                if close_interpolation && frames.len() > 1 {
                    // Closes the `${…}` of the template below. A
                    // stray `}` at the top level (invalid code) is
                    // ignored — the scanner never panics on it.
                    frames.pop();
                }
                i += 1;
            }
            _ if b.is_ascii_alphabetic() || b == b'_' || b == b'$' => {
                let (ident, next) = read_ident(bytes, i);
                if ident == "import" || ident == "export" {
                    let is_import = ident == "import";
                    let mut j = skip_ws_comments(bytes, next);
                    if is_import && j < bytes.len() && bytes[j] == b'(' {
                        // Dynamic import(): only a string literal
                        // can be pre-fetched.
                        j = skip_ws_comments(bytes, j + 1);
                        if j < bytes.len() && (bytes[j] == b'\'' || bytes[j] == b'"') {
                            if let Some((spec, after)) = scan_string(source, j, bytes[j]) {
                                push_spec(&mut out, spec);
                                i = after;
                                continue;
                            }
                        }
                        i = next;
                        continue;
                    }
                    if let Some((spec, after)) = scan_from_spec(source, next, is_import) {
                        push_spec(&mut out, spec);
                        i = after;
                        continue;
                    }
                    i = next;
                } else {
                    i = next;
                }
            }
            _ => i += 1,
        }
    }
    out
}

fn push_spec(out: &mut Vec<String>, spec: String) {
    if !spec.trim().is_empty() && !out.contains(&spec) {
        out.push(spec);
    }
}

/// The module resolver: URL-only, no import maps (M5).
pub struct PlutoResolver;

impl Resolver for PlutoResolver {
    fn resolve<'js>(
        &mut self,
        _ctx: &Ctx<'js>,
        base: &str,
        name: &str,
        _attributes: Option<ImportAttributes<'js>>,
    ) -> rquickjs::Result<String> {
        resolve_module_specifier(base, name)
            .map_err(|msg| Error::new_resolving_message(base, name, msg))
    }
}

/// The module loader: serves pre-fetched sources from
/// [`ModuleSources`], compiling from source with
/// `Module::declare`. A miss is a loading error — the importing
/// module rejects, the realm lives. No bytecode cache (see the
/// module docs: the engine dedups loaded modules per runtime, so
/// a loader-side cache could never hit twice).
pub struct PlutoLoader {
    sources: ModuleSources,
}

impl PlutoLoader {
    pub fn new(sources: ModuleSources) -> Self {
        Self { sources }
    }
}

impl Loader for PlutoLoader {
    fn load<'js>(
        &mut self,
        ctx: &Ctx<'js>,
        name: &str,
        attributes: Option<ImportAttributes<'js>>,
    ) -> rquickjs::Result<Module<'js, rquickjs::module::Declared>> {
        if let Some(attrs) = &attributes {
            if let Ok(Some(t)) = attrs.get_type() {
                if t != "javascript" {
                    return Err(Error::new_loading_message(
                        name,
                        format!(
                            "import attribute type '{t}' is not supported in M4 (only JavaScript modules)"
                        ),
                    ));
                }
            }
        }
        let Some(source) = self.sources.get(name) else {
            return Err(Error::new_loading_message(
                name,
                "module was not pre-fetched (its import was never seen by the loader, or its fetch failed)",
            ));
        };
        Module::declare(ctx.clone(), name, source.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_static_imports() {
        let src = r#"
            import def from './def.mjs';
            import { a, b as c } from "../lib/util.js";
            import * as ns from '/abs.mjs';
            import './side.mjs';
            export { x } from './x.mjs';
            export * from "https://example.com/y.mjs";
        "#;
        assert_eq!(
            scan_module_imports(src),
            vec![
                "./def.mjs",
                "../lib/util.js",
                "/abs.mjs",
                "./side.mjs",
                "./x.mjs",
                "https://example.com/y.mjs",
            ]
        );
    }

    #[test]
    fn scan_ignores_imports_in_strings_and_comments() {
        let src = r#"
            // import './not-real.mjs';
            /* import { q } from './also-not.mjs'; */
            const s = "import './string.mjs'";
            const t = 'export { z } from "./string2.mjs"';
            const u = `import './template.mjs'`;
            import './real.mjs';
        "#;
        assert_eq!(scan_module_imports(src), vec!["./real.mjs"]);
    }

    #[test]
    fn scan_dynamic_import_literal_only() {
        let src = r#"
            const a = await import('./chunk.mjs');
            const name = './computed.mjs';
            const b = await import(name);
            const c = await import(`./tpl-${name}.mjs`);
        "#;
        assert_eq!(scan_module_imports(src), vec!["./chunk.mjs"]);
    }

    #[test]
    fn scan_template_interpolation_import_is_found() {
        // An import inside `${…}` is live code, not template text.
        let src = "const s = `prefix ${import('./live.mjs')} suffix`;";
        assert_eq!(scan_module_imports(src), vec!["./live.mjs"]);
    }

    #[test]
    fn scan_dedups_and_keeps_order() {
        let src = "import './a.mjs'; import './b.mjs'; import './a.mjs';";
        assert_eq!(
            scan_module_imports(src),
            vec!["./a.mjs", "./b.mjs"]
        );
    }

    #[test]
    fn resolve_relative_and_absolute() {
        assert_eq!(
            resolve_module_specifier("https://example.com/app/main.mjs", "./util.mjs"),
            Ok("https://example.com/app/util.mjs".to_string())
        );
        assert_eq!(
            resolve_module_specifier("https://example.com/app/main.mjs", "../shared.mjs"),
            Ok("https://example.com/shared.mjs".to_string())
        );
        assert_eq!(
            resolve_module_specifier("https://example.com/app/main.mjs", "/root.mjs"),
            Ok("https://example.com/root.mjs".to_string())
        );
        assert_eq!(
            resolve_module_specifier(
                "https://example.com/app/main.mjs",
                "https://cdn.example.com/lib.mjs"
            ),
            Ok("https://cdn.example.com/lib.mjs".to_string())
        );
        assert_eq!(
            resolve_module_specifier(
                "https://example.com/app/main.mjs",
                "./frag.mjs#section"
            ),
            Ok("https://example.com/app/frag.mjs".to_string())
        );
    }

    #[test]
    fn resolve_rejects_bare_and_data() {
        assert!(
            resolve_module_specifier("https://example.com/main.mjs", "lodash")
                .unwrap_err()
                .contains("import map")
        );
        assert!(
            resolve_module_specifier("https://example.com/main.mjs", "data:text/javascript,export default 1")
                .unwrap_err()
                .contains("data:")
        );
        assert!(is_bare_specifier("react"));
        assert!(is_bare_specifier("@scope/pkg"));
        assert!(!is_bare_specifier("./rel.mjs"));
        assert!(!is_bare_specifier("https://example.com/a.mjs"));
    }

    /// QuickJS dedups loaded modules by resolved name per runtime:
    /// a second, differently-named entry importing the same URL
    /// does NOT re-run the module body and never calls the loader
    /// again. This is why the loader has no bytecode cache — an
    /// import is compiled at most once per realm, so a loader-side
    /// cache could never get a second hit.
    #[test]
    fn engine_dedups_loaded_modules_by_name() {
        use crate::engine::{EvalKind, JsEngine, JsValue};
        use crate::realm::{RealmConfig, TabRealm};
        let mut realm = TabRealm::new(&RealmConfig::default()).unwrap();
        let sources = ModuleSources::new();
        sources.insert(
            "https://example.com/util.mjs",
            "globalThis.__util_runs = (globalThis.__util_runs || 0) + 1; export const x = 3;"
                .to_string(),
        );
        realm.install_module_loader(sources).unwrap();
        realm
            .eval(
                EvalKind::Module,
                "import { x } from './util.mjs'; globalThis.__v = x;",
                "https://example.com/main1.mjs",
            )
            .unwrap();
        assert_eq!(
            realm
                .eval(EvalKind::Classic, "globalThis.__util_runs", "read.js")
                .unwrap(),
            JsValue::Number(1.0)
        );
        realm
            .eval(
                EvalKind::Module,
                "import { x } from './util.mjs'; globalThis.__w = x;",
                "https://example.com/main2.mjs",
            )
            .unwrap();
        assert_eq!(
            realm
                .eval(EvalKind::Classic, "globalThis.__util_runs", "read.js")
                .unwrap(),
            JsValue::Number(1.0),
            "import body must run exactly once per realm"
        );
    }
}
