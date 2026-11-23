//! Parsing & cascade (layer 3): winnow CSS parser.
//!
//! M1 scope: simple compound selectors (`div`, `.note`, `#main`, `*`,
//! `div.note`), and the properties `display width height margin padding
//! background-color color font-size`. Values are typed where M1 models
//! them (`px` lengths, percentages, `#hex`/`rgb()`/named colors); anything
//! else is kept verbatim as [`Value::Raw`] so a later milestone can use it.
//!
//! The parser is fault-tolerant like a browser: a malformed rule or
//! declaration is skipped and recorded in [`Stylesheet::errors`], the rest
//! still parses. Unsupported in M1 (recorded, skipped): combinators and
//! pseudo-classes in selectors, `@media` and other at-rules, `@font-face`.
//! Those are M2 (selectors) and M6 (@media, @font-face).

use std::fmt;

/// A parsed CSS value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A bare keyword: `block`, `none`, `auto`, `flex`, ...
    Keyword(String),
    /// A `px` length (bare `0` counts as `Length(0.0)`).
    Length(f32),
    /// A `%` value.
    Percentage(f32),
    /// An RGB color, alpha always opaque in M1.
    Color(u8, u8, u8),
    /// Anything M1 does not model yet (`2em`, `calc(...)`, `url(...)`,
    /// `rgba(...)`, font stacks, ...), kept verbatim.
    Raw(String),
}

/// One simple selector: part of a compound selector, all parts of which
/// must match the same element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimpleSelector {
    Universal,
    Type(String),
    Class(String),
    Id(String),
}

/// A compound selector: `div`, `.note`, `#main`, `*`, `div.note`.
/// Combinators (`div p`, `div > p`) are M2 and rejected by the parser.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Selector {
    pub parts: Vec<SimpleSelector>,
}

impl Selector {
    /// True when every part matches this element (`tag` lowercased,
    /// `id` and `classes` as given).
    pub fn matches(&self, tag: &str, id: Option<&str>, classes: &[&str]) -> bool {
        self.parts.iter().all(|part| match part {
            SimpleSelector::Universal => true,
            SimpleSelector::Type(t) => t == tag,
            SimpleSelector::Class(c) => classes.contains(&c.as_str()),
            SimpleSelector::Id(want) => id == Some(want.as_str()),
        })
    }
}

/// One `property: value;` pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub property: String,
    pub value: Value,
}

/// One `selector, selector { declarations }` rule.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: Vec<Declaration>,
}

/// A parsed stylesheet: rules in source order (cascade order is applied
/// by the styler, not stored here), plus what was skipped and why.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    /// Human-readable notes for every skipped rule, declaration, or
    /// at-rule. Empty means the whole sheet parsed cleanly.
    pub errors: Vec<String>,
}

impl Stylesheet {
    /// Parse CSS text. Never fails outright: malformed rules are skipped
    /// and recorded in [`Stylesheet::errors`].
    pub fn parse(css: &str) -> Self {
        let mut sheet = Stylesheet::default();
        let mut input = css;
        loop {
            skip_ws_comments(&mut input);
            if input.is_empty() {
                break;
            }
            if input.starts_with('@') {
                skip_at_rule(&mut input, &mut sheet.errors);
                continue;
            }
            let Some(brace) = input.find('{') else {
                if !input.trim().is_empty() {
                    sheet
                        .errors
                        .push(format!("rule without '{{' skipped: {:?}", input.trim()));
                }
                break;
            };
            let (sel_text, rest) = input.split_at(brace);
            let selectors = parse_selector_list(sel_text, &mut sheet.errors);
            let rest = &rest[1..];
            let Some(close) = rest.find('}') else {
                sheet.errors.push("unclosed rule block".to_string());
                break;
            };
            let (block, rest) = rest.split_at(close);
            input = &rest[1..];
            if selectors.is_empty() {
                continue; // all selectors rejected; error already recorded
            }
            let mut rule = Rule {
                selectors,
                declarations: Vec::new(),
            };
            parse_declarations(block, &mut rule, &mut sheet.errors);
            sheet.rules.push(rule);
        }
        sheet
    }
}

fn skip_ws_comments(input: &mut &str) {
    loop {
        *input = input.trim_start();
        if input.starts_with("/*") {
            match input.find("*/") {
                Some(p) => *input = &input[p + 2..],
                None => {
                    *input = "";
                    return;
                }
            }
        } else {
            return;
        }
    }
}

/// Skip an at-rule (`@media ... { ... }`, `@import ...;`). M1 does not
/// model any of them; the skip is recorded, not silent.
fn skip_at_rule(input: &mut &str, errors: &mut Vec<String>) {
    errors.push("at-rule skipped (unsupported in M1)".to_string());
    match input.find('{') {
        Some(brace) => {
            // A ';' before the '{' means a statement at-rule: skip to it.
            if let Some(semi) = input.find(';') {
                if semi < brace {
                    *input = &input[semi + 1..];
                    return;
                }
            }
            // Block at-rule: skip balanced braces.
            let mut depth = 0;
            for (i, c) in input.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            *input = &input[i + 1..];
                            return;
                        }
                    }
                    _ => {}
                }
            }
            *input = "";
        }
        None => *input = "",
    }
}

fn parse_selector_list(text: &str, errors: &mut Vec<String>) -> Vec<Selector> {
    let mut out = Vec::new();
    for item in text.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        match parse_selector(item) {
            Some(sel) => out.push(sel),
            None => errors.push(format!(
                "selector skipped (unsupported in M1): {item:?}"
            )),
        }
    }
    out
}

/// Parse one compound selector. Returns `None` for anything with a
/// combinator, pseudo-class, attribute selector, or other M2 syntax.
fn parse_selector(item: &str) -> Option<Selector> {
    let bytes = item.as_bytes();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'*' => {
                parts.push(SimpleSelector::Universal);
                i += 1;
            }
            b'#' | b'.' => {
                let kind = bytes[i];
                i += 1;
                let start = i;
                while i < bytes.len() && is_name_char(bytes[i]) {
                    i += 1;
                }
                if start == i {
                    return None;
                }
                let name = &item[start..i];
                parts.push(if kind == b'#' {
                    SimpleSelector::Id(name.to_string())
                } else {
                    SimpleSelector::Class(name.to_string())
                });
            }
            b if b.is_ascii_alphabetic() => {
                let start = i;
                while i < bytes.len() && is_name_char(bytes[i]) {
                    i += 1;
                }
                parts.push(SimpleSelector::Type(
                    item[start..i].to_ascii_lowercase(),
                ));
            }
            _ => return None,
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(Selector { parts })
    }
}

fn is_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

fn parse_declarations(block: &str, rule: &mut Rule, errors: &mut Vec<String>) {
    for chunk in block.split(';') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let Some((prop, raw)) = chunk.split_once(':') else {
            errors.push(format!("declaration without ':' skipped: {chunk:?}"));
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let raw = raw.trim();
        if prop.is_empty() || raw.is_empty() {
            errors.push(format!("empty declaration skipped: {chunk:?}"));
            continue;
        }
        if prop == "margin" || prop == "padding" {
            match expand_shorthand(&prop, raw) {
                Some(decls) => rule.declarations.extend(decls),
                None => rule.declarations.push(Declaration {
                    property: prop,
                    value: Value::Raw(raw.to_string()),
                }),
            }
        } else {
            rule.declarations.push(Declaration {
                property: prop,
                value: parse_value(raw),
            });
        }
    }
}

/// Expand `margin`/`padding: 1-4 values` into the four longhands.
/// Returns `None` when a token is not a length, percentage, or `auto`.
fn expand_shorthand(prop: &str, raw: &str) -> Option<Vec<Declaration>> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        match parse_value(t) {
            Value::Length(v) => vals.push(Value::Length(v)),
            Value::Percentage(v) => vals.push(Value::Percentage(v)),
            Value::Keyword(k) if k == "auto" => vals.push(Value::Keyword(k)),
            _ => return None,
        }
    }
    let (top, right, bottom, left) = match vals.len() {
        1 => (
            vals[0].clone(),
            vals[0].clone(),
            vals[0].clone(),
            vals[0].clone(),
        ),
        2 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[0].clone(),
            vals[1].clone(),
        ),
        3 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[2].clone(),
            vals[1].clone(),
        ),
        4 => (
            vals[0].clone(),
            vals[1].clone(),
            vals[2].clone(),
            vals[3].clone(),
        ),
        _ => unreachable!(),
    };
    Some(vec![
        Declaration {
            property: format!("{prop}-top"),
            value: top,
        },
        Declaration {
            property: format!("{prop}-right"),
            value: right,
        },
        Declaration {
            property: format!("{prop}-bottom"),
            value: bottom,
        },
        Declaration {
            property: format!("{prop}-left"),
            value: left,
        },
    ])
}

/// Interpret one raw value string as a typed [`Value`].
pub fn parse_value(raw: &str) -> Value {
    let s = raw.trim();
    let lower = s.to_ascii_lowercase();
    if let Some(num) = lower.strip_suffix("px") {
        if let Ok(v) = num.trim().parse::<f32>() {
            return Value::Length(v);
        }
    }
    if let Some(num) = lower.strip_suffix('%') {
        if let Ok(v) = num.trim().parse::<f32>() {
            return Value::Percentage(v);
        }
    }
    if let Ok(v) = s.parse::<f32>() {
        return Value::Length(v); // bare number: unitless zero, e.g. `margin: 0`
    }
    if let Some(hex) = lower.strip_prefix('#') {
        if let Some(c) = parse_hex_color(hex) {
            return Value::Color(c.0, c.1, c.2);
        }
    }
    if lower.starts_with("rgb(") && lower.ends_with(')') {
        if let Some(c) = parse_rgb(&lower[4..lower.len() - 1]) {
            return Value::Color(c.0, c.1, c.2);
        }
    }
    if let Some(c) = named_color(&lower) {
        return Value::Color(c.0, c.1, c.2);
    }
    if s.split_whitespace().count() == 1
        && s.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
    {
        return Value::Keyword(lower);
    }
    Value::Raw(s.to_string())
}

fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let nibble = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let b = hex.as_bytes();
    match b.len() {
        3 => Some((
            nibble(b[0])? * 17,
            nibble(b[1])? * 17,
            nibble(b[2])? * 17,
        )),
        6 => Some((
            nibble(b[0])? * 16 + nibble(b[1])?,
            nibble(b[2])? * 16 + nibble(b[3])?,
            nibble(b[4])? * 16 + nibble(b[5])?,
        )),
        _ => None,
    }
}

fn parse_rgb(inner: &str) -> Option<(u8, u8, u8)> {
    let mut parts = inner.split(',');
    let r: u8 = parts.next()?.trim().parse().ok()?;
    let g: u8 = parts.next()?.trim().parse().ok()?;
    let b: u8 = parts.next()?.trim().parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b))
}

fn named_color(name: &str) -> Option<(u8, u8, u8)> {
    Some(match name {
        "black" => (0, 0, 0),
        "white" => (255, 255, 255),
        "red" => (255, 0, 0),
        "green" => (0, 128, 0),
        "lime" => (0, 255, 0),
        "blue" => (0, 0, 255),
        "yellow" => (255, 255, 0),
        "cyan" | "aqua" => (0, 255, 255),
        "magenta" | "fuchsia" => (255, 0, 255),
        "gray" | "grey" => (128, 128, 128),
        "silver" => (192, 192, 192),
        "maroon" => (128, 0, 0),
        "navy" => (0, 0, 128),
        "olive" => (128, 128, 0),
        "teal" => (0, 128, 128),
        "purple" => (128, 0, 128),
        "orange" => (255, 165, 0),
        "pink" => (255, 192, 203),
        "brown" => (165, 42, 42),
        "gold" => (255, 215, 0),
        "beige" => (245, 245, 220),
        _ => return None,
    })
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Keyword(k) => write!(f, "{k}"),
            Value::Length(v) => write!(f, "{v}px"),
            Value::Percentage(v) => write!(f, "{v}%"),
            Value::Color(r, g, b) => write!(f, "rgb({r}, {g}, {b})"),
            Value::Raw(r) => write!(f, "{r}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real example.com `<style>` block (see pluto-html tests).
    const EXAMPLE_CSS: &str = r#"
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
        border-radius: 0.5em;
        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);
    }
    a:link, a:visited {
        color: #38488f;
        text-decoration: none;
    }
    @media (max-width: 700px) {
        div {
            margin: 0 auto;
            width: auto;
        }
    }
    "#;

    fn decl<'a>(rule: &'a Rule, prop: &str) -> Option<&'a Value> {
        rule.declarations
            .iter()
            .find(|d| d.property == prop)
            .map(|d| &d.value)
    }

    #[test]
    fn parses_example_com_stylesheet() {
        let sheet = Stylesheet::parse(EXAMPLE_CSS);
        // body + div parse; a:link/a:visited (two bad selectors) and @media
        // are skipped, recorded.
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.errors.len(), 3);
        assert!(sheet.errors.iter().any(|e| e.contains("a:link")));
        assert!(sheet.errors.iter().any(|e| e.contains("a:visited")));
        assert!(sheet.errors.iter().any(|e| e.contains("at-rule")));

        let body = &sheet.rules[0];
        assert_eq!(body.selectors.len(), 1);
        assert!(body.selectors[0].matches("body", None, &[]));
        assert_eq!(
            decl(body, "background-color"),
            Some(&Value::Color(240, 240, 242))
        );
        assert_eq!(decl(body, "margin-top"), Some(&Value::Length(0.0)));
        assert_eq!(decl(body, "margin-left"), Some(&Value::Length(0.0)));
        // font-family is a multi-word stack: kept verbatim.
        assert!(matches!(decl(body, "font-family"), Some(Value::Raw(_))));

        let div = &sheet.rules[1];
        assert_eq!(decl(div, "width"), Some(&Value::Length(600.0)));
        assert_eq!(
            decl(div, "background-color"),
            Some(&Value::Color(253, 253, 255))
        );
        // `margin: 5em auto` cannot expand (em is M6): kept as Raw.
        assert!(matches!(decl(div, "margin"), Some(Value::Raw(_))));
        assert!(matches!(decl(div, "padding"), Some(Value::Raw(_))));
        assert!(matches!(decl(div, "border-radius"), Some(Value::Raw(_))));
    }

    #[test]
    fn shorthand_expansion() {
        let sheet = Stylesheet::parse("div { margin: 1px 2px 3px 4px; padding: 10px 20px; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "margin-top"), Some(&Value::Length(1.0)));
        assert_eq!(decl(rule, "margin-right"), Some(&Value::Length(2.0)));
        assert_eq!(decl(rule, "margin-bottom"), Some(&Value::Length(3.0)));
        assert_eq!(decl(rule, "margin-left"), Some(&Value::Length(4.0)));
        assert_eq!(decl(rule, "padding-top"), Some(&Value::Length(10.0)));
        assert_eq!(decl(rule, "padding-bottom"), Some(&Value::Length(10.0)));
        assert_eq!(decl(rule, "padding-right"), Some(&Value::Length(20.0)));
        assert_eq!(decl(rule, "padding-left"), Some(&Value::Length(20.0)));
        assert!(decl(rule, "margin").is_none());
    }

    #[test]
    fn selectors_and_matching() {
        let sheet = Stylesheet::parse("div, .note, #main, *, div.note { display: block; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(rule.selectors.len(), 5);
        assert!(rule.selectors[0].matches("div", None, &[]));
        assert!(!rule.selectors[0].matches("p", None, &[]));
        assert!(rule.selectors[1].matches("p", None, &["note"]));
        assert!(!rule.selectors[1].matches("p", None, &["other"]));
        assert!(rule.selectors[2].matches("div", Some("main"), &[]));
        assert!(!rule.selectors[2].matches("div", Some("other"), &[]));
        assert!(rule.selectors[3].matches("anything", None, &[]));
        assert!(rule.selectors[4].matches("div", None, &["note"]));
        assert!(!rule.selectors[4].matches("p", None, &["note"]));
        assert_eq!(decl(rule, "display"), Some(&Value::Keyword("block".into())));
    }

    #[test]
    fn comments_and_fault_tolerance() {
        let sheet = Stylesheet::parse(
            "/* header */ div { color: red; bogus } p { width: 10px; }",
        );
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.errors.len(), 1);
        assert_eq!(decl(&sheet.rules[0], "color"), Some(&Value::Color(255, 0, 0)));
        assert_eq!(decl(&sheet.rules[1], "width"), Some(&Value::Length(10.0)));
    }

    #[test]
    fn colors() {
        assert_eq!(parse_value("#fff"), Value::Color(255, 255, 255));
        assert_eq!(parse_value("#f0f0f2"), Value::Color(240, 240, 242));
        assert_eq!(parse_value("rgb(1, 2, 3)"), Value::Color(1, 2, 3));
        assert_eq!(parse_value("red"), Value::Color(255, 0, 0));
        assert_eq!(parse_value("50%"), Value::Percentage(50.0));
        assert_eq!(parse_value("12px"), Value::Length(12.0));
        assert_eq!(parse_value("auto"), Value::Keyword("auto".into()));
        assert_eq!(parse_value("2em"), Value::Raw("2em".into()));
    }
}
