//! Page pipeline (M1.4): fetch -> parse -> layout -> paint.
//!
//! Fetch and parse happen once per navigation; layout is cheap and
//! repeatable, so the parsed [`Document`] is retained and re-laid-out
//! whenever the viewport changes — a resize never refetches.
//!
//! The layout viewport is a single stored value owned by the shell:
//! the page area of the window (window minus chrome), reported by the
//! viewport widget. It starts at [`DEFAULT_VIEWPORT`] (the default
//! window size in this libcosmic pin) and is corrected by the first
//! bounds report. There is no fixed layout size anymore — the old
//! 1024x768 constant was the mobile 980-1024px fallback applied to a
//! desktop window, and it left the canvas stopping at 1024px on a
//! maximised window.
//!
//! Errors never crash the app: a failed fetch or layout renders as a
//! real page (an error document run through the same pipeline), and the
//! status line says what happened.

use pluto_css::Stylesheet;
use pluto_html::{Dom, NodeId};
use pluto_ipc::ResourceRequest;
use pluto_layout::LayoutTree;
use pluto_net::ResourceLoader;
use pluto_paint::DisplayList;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio_util::sync::CancellationToken;

/// Startup guess for the layout viewport: the default window size (no
/// `Settings::size()` in this libcosmic pin, so the window opens
/// 1024x768). The viewport widget reports its real bounds on the first
/// frame and the shell corrects this; it is never used as a layout
/// size after that.
pub const DEFAULT_VIEWPORT: (f32, f32) = (1024.0, 768.0);

/// A parsed page: the DOM plus its collected `<style>` rules.
/// Retained across resizes so layout can re-run without refetching.
#[derive(Debug, Clone)]
pub struct Document {
    dom: Dom,
    sheet: Stylesheet,
}

/// The page loaded at startup and when the address bar is empty.
pub const DEFAULT_URL: &str = "https://example.com";

/// Turn address-bar input into a fetchable URL. A bare host gets
/// `https://`; anything with a scheme is left alone.
pub fn normalize_url(input: &str) -> String {
    let input = input.trim();
    if input.is_empty() {
        return DEFAULT_URL.to_string();
    }
    if input.contains("://") {
        return input.to_string();
    }
    format!("https://{input}")
}

/// Fetch `url` (owned, so the future is `'static` for `Task::perform`)
/// and parse the HTML. Layout happens separately in
/// [`layout_document`], at the shell's current viewport, so a resize
/// never refetches. Returns the parsed document and the final URL
/// after redirects. `loader` is the worker's loader (one per worker,
/// shared client + scheduler — Stage 5): the old fresh-loader-per-
/// navigation defeated pooling and resumption, so it is gone.
/// Cancelling `cancel` aborts the fetch (Stage 5 #5 — the worker
/// cancels the old navigation's token when a new one commits). The
/// request is a top-level navigation (VeryHigh priority, Document
/// destination) with the page's own URL as its origin, so the
/// mixed-content gate never blocks it.
///
/// Logs each stage to stderr: the terminal is the only window into a
/// load that stalls (no logging framework is wired up yet).
pub async fn load_page_cancellable(
    loader: &ResourceLoader,
    url: String,
    cancel: CancellationToken,
) -> Result<(Document, String), String> {
    static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
    let request =
        ResourceRequest::navigate(NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed), 0, 0, &url);
    eprintln!("[pluto] fetching {url}");
    let response = match loader.fetch_with(request, cancel).await {
        Ok(response) => response,
        Err(e) => {
            eprintln!("[pluto] fetch failed: {e}");
            return Err(e.to_string());
        }
    };
    eprintln!(
        "[pluto] fetched {} ({} bytes), parsing",
        response.final_url,
        response.body.len()
    );
    Ok((
        parse_document(&response.text()),
        response.final_url.to_string(),
    ))
}

/// Parse HTML and collect its `<style>` rules into a [`Document`].
/// External `<link rel="stylesheet">` is M2.
pub fn parse_document(html: &str) -> Document {
    let dom = pluto_html::parse(html);
    let sheet = extract_stylesheet(&dom);
    Document { dom, sheet }
}

/// Lay `doc` out at `viewport` (CSS px, logical — the page area of
/// the window) and build the display list. The canvas background is
/// painted over the whole viewport by the display list, so a short
/// page on a wide window is covered to the edge. A layout failure
/// renders as an error page at the same viewport instead of an empty
/// viewport.
pub fn layout_document(doc: &Document, viewport: (f32, f32)) -> Arc<DisplayList> {
    match LayoutTree::build(&doc.dom, &doc.sheet, viewport.0, viewport.1) {
        Ok(tree) => Arc::new(DisplayList::build(&tree)),
        Err(e) => {
            let err_doc = parse_document(&error_document(
                "about:blank",
                &format!("layout failed: {e}"),
            ));
            let tree = LayoutTree::build(&err_doc.dom, &err_doc.sheet, viewport.0, viewport.1)
                .unwrap_or_default();
            Arc::new(DisplayList::build(&tree))
        }
    }
}

/// Parse HTML and lay it out at `viewport`, returning the display
/// list and the parsed document (retained for later resizes).
pub fn build_display_list(html: &str, viewport: (f32, f32)) -> (Arc<DisplayList>, Document) {
    let doc = parse_document(html);
    let display = layout_document(&doc, viewport);
    (display, doc)
}

/// Collect every `<style>` element's text in the document and parse it
/// as one stylesheet. External `<link rel="stylesheet">` is M2.
pub fn extract_stylesheet(dom: &Dom) -> Stylesheet {
    let mut css = String::new();
    if let Some(root) = dom.root() {
        collect_style_text(dom, root, &mut css);
    }
    Stylesheet::parse(&css)
}

fn collect_style_text(dom: &Dom, id: NodeId, out: &mut String) {
    if dom.tag(id) == Some("style") {
        for &child in dom.children(id) {
            if let Some(text) = dom.text(child) {
                out.push_str(text);
                out.push('\n');
            }
        }
        return;
    }
    for &child in dom.children(id) {
        collect_style_text(dom, child, out);
    }
}

/// A small error page, rendered through the same pipeline as a real one.
pub fn error_document(url: &str, err: &str) -> String {
    format!(
        r#"<!doctype html>
<html>
<head>
    <title>Pluto: could not load page</title>
    <style type="text/css">
    body {{
        background-color: #ffffff;
        margin: 0;
        padding: 2em;
    }}
    h1 {{
        font-size: 24px;
    }}
    p {{
        font-size: 16px;
    }}
    </style>
</head>
<body>
    <h1>Could not load {url}</h1>
    <p>{err}</p>
    <p><a href="https://example.com">Go to example.com</a></p>
</body>
</html>"#,
        url = escape_html(url),
        err = escape_html(err),
    )
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pluto_paint::DisplayItem;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const EXAMPLE_COM: &str = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
    }
    a:link, a:visited {
        color: #38488f;
    }
    </style>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>"#;

    #[test]
    fn normalize_url_cases() {
        assert_eq!(normalize_url("example.com"), "https://example.com");
        assert_eq!(
            normalize_url("https://example.com"),
            "https://example.com"
        );
        assert_eq!(normalize_url("http://localhost:8080/"), "http://localhost:8080/");
        assert_eq!(normalize_url("  example.com  "), "https://example.com");
        assert_eq!(normalize_url(""), DEFAULT_URL);
    }

    #[test]
    fn build_display_list_example_com() {
        let (list, _doc) = build_display_list(EXAMPLE_COM, (1024.0, 768.0));
        assert!(!list.is_empty());
        // Canvas background first, then the div fill, then glyph runs.
        assert!(matches!(
            list.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                height: 768.0,
                ..
            }
        ));
        assert!(list.0.iter().any(|i| matches!(
            i,
            DisplayItem::Glyphs { glyphs, .. } if !glyphs.is_empty()
        )));
    }

    /// The live example.com page as served 2026-11-23 (577 bytes):
    /// the paragraph wraps at the viewport width (its `max-width`
    /// is beyond M1, so the body is full-width). Regression for the
    /// resize fix: the canvas must cover the viewport at any size,
    /// and the text must rewrap when the viewport narrows.
    const NEW_EXAMPLE_COM: &str = "<!doctype html><html lang=en><head><meta charset=utf-8><link rel=icon href=data:,><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>Example Domain</title><style>html{color-scheme:light dark;background:light-dark(#eee,#222)}body{font:16px/1.6 system-ui,sans-serif;max-width:26em;margin:auto;padding:25vh 2em 2em;text-align:center}</style></head><body><p>This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.</p><script src=/s.js></script></body></html>";

    fn glyph_max_y(list: &DisplayList) -> f32 {
        list.0.iter().fold(0.0f32, |acc, item| match item {
            DisplayItem::Glyphs { glyphs, .. } => glyphs.iter().fold(acc, |a, g| a.max(g.y)),
            _ => acc,
        })
    }

    #[test]
    fn display_list_reflows_with_viewport_width() {
        let (wide, _doc) = build_display_list(NEW_EXAMPLE_COM, (1024.0, 768.0));
        let (narrow, _doc) = build_display_list(NEW_EXAMPLE_COM, (400.0, 768.0));
        // The canvas covers each viewport exactly — no 1024px seam
        // on a wide window, no shortfall on a narrow one.
        for (list, w) in [(&wide, 1024.0), (&narrow, 400.0)] {
            assert!(matches!(
                list.0[0],
                DisplayItem::FillRect {
                    width,
                    height: 768.0,
                    ..
                } if width == w
            ));
        }
        // Same text in both, but the narrow viewport wraps it into
        // more lines, so the lowest glyph sits lower — and each wrap
        // consumes the space at the break, so the narrow layout has
        // (slightly) fewer glyph entries, not the same count.
        let glyph_total = |list: &DisplayList| {
            list.0.iter().fold(0, |acc, item| match item {
                DisplayItem::Glyphs { glyphs, .. } => acc + glyphs.len(),
                _ => acc,
            })
        };
        assert_eq!(glyph_total(&wide), 155);
        assert!(glyph_total(&narrow) < glyph_total(&wide));
        assert!(
            glyph_max_y(&narrow) > glyph_max_y(&wide),
            "narrow layout should wrap into more lines"
        );
    }

    #[test]
    fn layout_at_same_viewport_is_deterministic() {
        // Re-laying out at the same size must produce the identical
        // display list — so the shell's skip-if-unchanged is safe,
        // and a redundant re-layout would be invisible anyway.
        let doc = parse_document(NEW_EXAMPLE_COM);
        let first = layout_document(&doc, (1024.0, 768.0));
        let second = layout_document(&doc, (1024.0, 768.0));
        assert_eq!(first.0.len(), second.0.len());
        for (a, b) in first.0.iter().zip(second.0.iter()) {
            match (a, b) {
                (
                    DisplayItem::FillRect {
                        x: ax,
                        y: ay,
                        width: aw,
                        height: ah,
                        color: ac,
                    },
                    DisplayItem::FillRect {
                        x: bx,
                        y: by,
                        width: bw,
                        height: bh,
                        color: bc,
                    },
                ) => {
                    assert_eq!((*ax, *ay, *aw, *ah, *ac), (*bx, *by, *bw, *bh, *bc));
                }
                (
                    DisplayItem::Glyphs {
                        color: ac,
                        glyphs: ag,
                    },
                    DisplayItem::Glyphs {
                        color: bc,
                        glyphs: bg,
                    },
                ) => {
                    assert_eq!(ac, bc);
                    assert_eq!(ag.len(), bg.len());
                    for (g, h) in ag.iter().zip(bg.iter()) {
                        assert_eq!((g.x, g.y), (h.x, h.y));
                        assert_eq!(g.key, h.key);
                    }
                }
                _ => panic!("display list items differ between identical layouts"),
            }
        }
    }

    #[test]
    fn error_document_escapes() {
        let doc = error_document("https://x/?a=1&b=2", "oops <b>bold</b>");
        assert!(doc.contains("https://x/?a=1&amp;b=2"));
        assert!(doc.contains("oops &lt;b&gt;bold&lt;/b&gt;"));
        assert!(!doc.contains("<b>bold</b>"));
        // And the escaped doc still builds a display list.
        assert!(!build_display_list(&doc, (1024.0, 768.0)).0.is_empty());
    }

    /// Local server returning the example.com document, same pattern as
    /// pluto-net's tests. Proves fetch -> parse -> layout -> paint
    /// end to end without the internet (hyper has no proxy support).
    async fn serve(listener: TcpListener) {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let _ = n;
                let body = EXAMPLE_COM;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            });
        }
    }

    #[tokio::test]
    async fn load_page_from_local_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve(listener));
        let url = format!("http://127.0.0.1:{}/", addr.port());
        let loader = ResourceLoader::new();
        let (doc, final_url) =
            load_page_cancellable(&loader, url.clone(), CancellationToken::new())
                .await
                .expect("page loads");
        assert_eq!(final_url, url);
        // Layout at a non-default viewport: the viewport flows
        // through the fetch path, and the canvas covers it exactly.
        let display = layout_document(&doc, (800.0, 600.0));
        assert!(!display.is_empty());
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect {
                width: 800.0,
                height: 600.0,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn load_page_invalid_url_errors() {
        let loader = ResourceLoader::new();
        assert!(
            load_page_cancellable(
                &loader,
                "ftp://example.com/".to_string(),
                CancellationToken::new()
            )
            .await
            .is_err()
        );
    }
}
