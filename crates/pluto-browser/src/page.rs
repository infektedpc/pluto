//! Page pipeline (M1.4): fetch -> parse -> layout -> paint.
//!
//! One function per stage, wired by [`load_page`], which is what the
//! browser's `Navigate` task runs. The layout viewport is the default
//! window size (1024x768); resizing does not re-layout in M1 — the
//! viewport reports its bounds to the renderer, but the display list
//! itself is built once per navigation.
//!
//! Errors never crash the app: a failed fetch or layout renders as a
//! real page (an error document run through the same pipeline), and the
//! status line says what happened.

use pluto_css::Stylesheet;
use pluto_html::{Dom, NodeId};
use pluto_layout::LayoutTree;
use pluto_net::ResourceLoader;
use pluto_paint::DisplayList;
use std::sync::Arc;

/// Layout viewport: the default window size (no `Settings::size()` in
/// this libcosmic pin, so the window opens 1024x768).
pub const VIEWPORT: (f32, f32) = (1024.0, 768.0);

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

/// Fetch `url` (owned, so the future is `'static` for `Task::perform`),
/// parse the HTML, lay it out, and paint it. Returns the display list
/// and the final URL after redirects. The loader is fresh per navigation
/// in M1 (no shared connection pool yet).
///
/// Logs each stage to stderr: the terminal is the only window into a
/// load that stalls (no logging framework is wired up yet).
pub async fn load_page(url: String) -> Result<(Arc<DisplayList>, String), String> {
    eprintln!("[pluto] fetching {url}");
    let loader = ResourceLoader::new();
    let response = match loader.fetch(&url).await {
        Ok(response) => response,
        Err(e) => {
            eprintln!("[pluto] fetch failed: {e}");
            return Err(e.to_string());
        }
    };
    eprintln!(
        "[pluto] fetched {} ({} bytes), building display list",
        response.final_url,
        response.body.len()
    );
    let display = build_display_list(&response.text());
    eprintln!("[pluto] display list ready: {} items", display.0.len());
    Ok((display, response.final_url.to_string()))
}

/// Parse HTML, collect its `<style>` rules, lay it out, and build the
/// display list. A layout failure renders as an error page instead of
/// an empty viewport.
pub fn build_display_list(html: &str) -> Arc<DisplayList> {
    let dom = pluto_html::parse(html);
    let sheet = extract_stylesheet(&dom);
    match LayoutTree::build(&dom, &sheet, VIEWPORT.0, VIEWPORT.1) {
        Ok(tree) => Arc::new(DisplayList::build(&tree)),
        Err(e) => {
            let doc = error_document("about:blank", &format!("layout failed: {e}"));
            let dom = pluto_html::parse(&doc);
            let sheet = extract_stylesheet(&dom);
            let tree = LayoutTree::build(&dom, &sheet, VIEWPORT.0, VIEWPORT.1)
                .unwrap_or_default();
            Arc::new(DisplayList::build(&tree))
        }
    }
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
        let list = build_display_list(EXAMPLE_COM);
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

    #[test]
    fn error_document_escapes() {
        let doc = error_document("https://x/?a=1&b=2", "oops <b>bold</b>");
        assert!(doc.contains("https://x/?a=1&amp;b=2"));
        assert!(doc.contains("oops &lt;b&gt;bold&lt;/b&gt;"));
        assert!(!doc.contains("<b>bold</b>"));
        // And the escaped doc still builds a display list.
        assert!(!build_display_list(&doc).is_empty());
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
        let (display, final_url) = load_page(url.clone()).await.expect("page loads");
        assert_eq!(final_url, url);
        assert!(!display.is_empty());
        assert!(matches!(
            display.0[0],
            DisplayItem::FillRect {
                width: 1024.0,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn load_page_invalid_url_errors() {
        assert!(load_page("ftp://example.com/".to_string())
            .await
            .is_err());
    }
}
