//! Parsing & cascade (layer 3): CSS parser.
//!
//! M2 scope: complex selectors (combinators, attribute selectors,
//! structural/link/dynamic/logical pseudo-classes, pseudo-elements),
//! specificity, cascade data (origin / `!important` / layers / source
//! order), conditional group rules (`@media` / `@supports`), `@import`,
//! `@font-face`, `@layer`, `@keyframes` (stored), and a real UA
//! stylesheet ([`UA_CSS`]) parsed by this same parser.
//!
//! The parser is fault-tolerant per CSS Syntax L3: a bad declaration is
//! dropped to the next `;`, a bad selector drops the whole rule (one
//! invalid complex selector invalidates the whole selector list), an
//! unknown at-rule is skipped to `;` or its balanced `}`, and an
//! unclosed block is closed at EOF. Everything skipped is recorded in
//! [`Stylesheet::errors`] with a byte offset where cheap.
//!
//! Matching note: [`Selector::matches`] only answers for compound-only
//! selectors against a bare `(tag, id, classes)` triple. Anything
//! needing tree context (combinators, structural/dynamic pseudo-classes,
//! attribute selectors, `:has()`) returns `false` here; the real
//! right-to-left matcher lives in `pluto-layout` (wave 2) and consumes
//! [`Selector::compounds`] / [`Selector::combinators`] directly.

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
    /// An `em` length, resolved against the element's own computed
    /// font size at style time.
    Em(f32),
    /// A `rem` length, resolved against the root element's font size.
    Rem(f32),
    /// A `vh` length, resolved against the viewport height at layout
    /// time (a *used* value, not a computed one).
    Vh(f32),
    /// A `vw` length, resolved against the viewport width at layout time.
    Vw(f32),
    /// An RGB color, alpha always opaque.
    Color(u8, u8, u8),
    /// An RGBA color (`rgba()` / `hsla()` / 4- and 8-digit hex). The
    /// alpha channel is preserved here; consumers that cannot paint
    /// alpha yet should use the RGB channels and note the drop.
    ColorA(u8, u8, u8, f32),
    /// `border-radius: 1-4 lengths` (elliptical `/` radii are dropped).
    BorderRadius(BorderRadius),
    /// `box-shadow: <shadow>, ...` (`none` is the empty list).
    BoxShadow(Vec<BoxShadow>),
    /// Anything not modelled yet (`calc(...)`, `url(...)`, font stacks,
    /// ...), kept verbatim.
    Raw(String),
}

/// One length that can appear inside a structured value
/// ([`BorderRadius`], [`BoxShadow`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CssLength {
    Px(f32),
    Percentage(f32),
    Em(f32),
    Rem(f32),
    Vh(f32),
    Vw(f32),
}

impl CssLength {
    /// Convert a typed [`Value`] to a [`CssLength`], or `None` when the
    /// value is not a length in any unit this type models.
    pub fn from_value(value: &Value) -> Option<Self> {
        Some(match value {
            Value::Length(v) => CssLength::Px(*v),
            Value::Percentage(v) => CssLength::Percentage(*v),
            Value::Em(v) => CssLength::Em(*v),
            Value::Rem(v) => CssLength::Rem(*v),
            Value::Vh(v) => CssLength::Vh(*v),
            Value::Vw(v) => CssLength::Vw(*v),
            _ => return None,
        })
    }
}

/// `border-radius` corners, clockwise from the top-left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderRadius {
    pub top_left: CssLength,
    pub top_right: CssLength,
    pub bottom_right: CssLength,
    pub bottom_left: CssLength,
}

/// One `box-shadow` layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxShadow {
    pub inset: bool,
    pub offset_x: CssLength,
    pub offset_y: CssLength,
    pub blur: CssLength,
    pub spread: CssLength,
    /// `(r, g, b, alpha)`; `None` means "current color".
    pub color: Option<(u8, u8, u8, f32)>,
}

/// An `an+b` argument to `:nth-child()` and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnPlusB {
    pub a: i32,
    pub b: i32,
}

/// An attribute-selector operator: `[a]`, `[a=v]`, `[a~=v]`, `[a|=v]`,
/// `[a^=v]`, `[a$=v]`, `[a*=v]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrOp {
    Exists,
    Equals,
    Includes,
    Dash,
    Prefix,
    Suffix,
    Substring,
}

/// A pseudo-class. Structural and link variants are static (checkable
/// from the DOM); dynamic variants need event/hit-test state; logical
/// variants take forgiving selector lists; `:has()` is parsed and
/// stored but matching is not implemented (see
/// [`Selector::has_unimplemented`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PseudoClass {
    Root,
    FirstChild,
    LastChild,
    NthChild(AnPlusB),
    NthLastChild(AnPlusB),
    OnlyChild,
    Empty,
    FirstOfType,
    LastOfType,
    NthOfType(AnPlusB),
    NthLastOfType(AnPlusB),
    OnlyOfType,
    Link,
    Visited,
    AnyLink,
    Hover,
    Active,
    Focus,
    FocusVisible,
    FocusWithin,
    Not(Vec<Selector>),
    Is(Vec<Selector>),
    Where(Vec<Selector>),
    Has(Vec<Selector>),
}

impl PseudoClass {
    /// Specificity contribution: `:where()` is zero, `:is()` / `:not()` /
    /// `:has()` take the max of their arguments, everything else is one
    /// pseudo-class (0, 1, 0).
    pub fn specificity(&self) -> (u32, u32, u32) {
        match self {
            PseudoClass::Where(_) => (0, 0, 0),
            PseudoClass::Is(args) | PseudoClass::Not(args) | PseudoClass::Has(args) => args
                .iter()
                .map(|s| s.specificity())
                .max()
                .unwrap_or((0, 0, 0)),
            _ => (0, 1, 0),
        }
    }
}

/// A pseudo-element (`::before`, ...). Parsed and stored on the
/// selector; generated boxes are a layout concern, not a CSS one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PseudoElement {
    Before,
    After,
    FirstLine,
    FirstLetter,
    Marker,
    Selection,
}

/// One simple selector: part of a compound selector, all parts of which
/// must match the same element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimpleSelector {
    Universal,
    Type(String),
    Class(String),
    Id(String),
    Attribute {
        name: String,
        op: AttrOp,
        value: Option<String>,
        /// `Some(false)` for the `i` flag, `Some(true)` for `s`,
        /// `None` for the default (case-sensitive, except quirks mode).
        case_sensitive: Option<bool>,
    },
    PseudoClass(PseudoClass),
    /// Kept in the compound only transiently while parsing; after a
    /// successful parse it is moved to [`Selector::pseudo_element`].
    /// It still appears here in a partially built selector.
    PseudoElement(PseudoElement),
}

impl SimpleSelector {
    /// Specificity contribution of one simple selector.
    pub fn specificity(&self) -> (u32, u32, u32) {
        match self {
            SimpleSelector::Universal => (0, 0, 0),
            SimpleSelector::Type(_) => (0, 0, 1),
            SimpleSelector::Class(_) => (0, 1, 0),
            SimpleSelector::Id(_) => (1, 0, 0),
            SimpleSelector::Attribute { .. } => (0, 1, 0),
            SimpleSelector::PseudoClass(pc) => pc.specificity(),
            SimpleSelector::PseudoElement(_) => (0, 0, 1),
        }
    }

    /// Indexing rank for [`Selector::key_selector`]: id beats
    /// class/attribute/pseudo-class beats type beats universal.
    fn key_rank(&self) -> u8 {
        match self {
            SimpleSelector::Id(_) => 4,
            SimpleSelector::Class(_)
            | SimpleSelector::Attribute { .. }
            | SimpleSelector::PseudoClass(_) => 3,
            SimpleSelector::Type(_) => 2,
            SimpleSelector::Universal => 1,
            SimpleSelector::PseudoElement(_) => 0,
        }
    }
}

/// A compound selector: `div`, `.note`, `#main`, `*`, `div.note`,
/// `a[href^="https"]:hover` — simple selectors with no whitespace
/// between them, all matching the same element.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompoundSelector {
    pub parts: Vec<SimpleSelector>,
}

/// A combinator joining two compounds in a complex selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    /// `A B` — B is a descendant of A.
    Descendant,
    /// `A > B` — B is a child of A.
    Child,
    /// `A + B` — B immediately follows A.
    NextSibling,
    /// `A ~ B` — B follows A among siblings.
    SubsequentSibling,
}

/// A complex selector: compounds joined by combinators, e.g.
/// `div.note > p:first-child`, plus an optional trailing
/// pseudo-element (`p::before`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Selector {
    /// Compounds left to right; `combinators[i]` joins
    /// `compounds[i]` and `compounds[i + 1]`.
    pub compounds: Vec<CompoundSelector>,
    pub combinators: Vec<Combinator>,
    pub pseudo_element: Option<PseudoElement>,
    /// True when the selector contains `:has(...)`. `:has()` is parsed
    /// and stored, but matching is NOT implemented: a selector with
    /// this flag must never be treated as matching.
    pub has_unimplemented: bool,
}

impl Selector {
    /// Specificity as a lexicographic triple `(ids, classes, types)`:
    /// IDs; classes + attributes + pseudo-classes; types +
    /// pseudo-elements. Compared lexicographically, never as a single
    /// number — 256 classes do not beat one ID. `:is()` / `:not()` /
    /// `:has()` contribute the max of their arguments; `:where()`
    /// contributes zero.
    pub fn specificity(&self) -> (u32, u32, u32) {
        let mut spec = (0u32, 0u32, 0u32);
        for compound in &self.compounds {
            for part in &compound.parts {
                let (a, b, c) = part.specificity();
                spec.0 += a;
                spec.1 += b;
                spec.2 += c;
            }
        }
        if self.pseudo_element.is_some() {
            spec.2 += 1;
        }
        spec
    }

    /// The key selector for rule indexing (Stylo-style): the most
    /// selective simple selector of the rightmost compound —
    /// id > class/attribute/pseudo-class > type > universal. Layout
    /// buckets rules by this and tests only the matching buckets.
    pub fn key_selector(&self) -> Option<&SimpleSelector> {
        self.compounds
            .last()?
            .parts
            .iter()
            .max_by_key(|part| part.key_rank())
    }

    /// True when every part matches this element (`tag` lowercased,
    /// `id` and `classes` as given). Compound-only selectors only:
    /// any combinator, pseudo-element, `:has()`, attribute selector,
    /// or pseudo-class needing DOM/event context returns `false` —
    /// the real right-to-left matcher (wave 2, in `pluto-layout`)
    /// handles those against the DOM.
    pub fn matches(&self, tag: &str, id: Option<&str>, classes: &[&str]) -> bool {
        if self.compounds.len() != 1 || !self.combinators.is_empty() {
            return false;
        }
        if self.pseudo_element.is_some() || self.has_unimplemented {
            return false;
        }
        self.compounds[0].parts.iter().all(|part| match part {
            SimpleSelector::Universal => true,
            SimpleSelector::Type(t) => t == tag,
            SimpleSelector::Class(c) => classes.contains(&c.as_str()),
            SimpleSelector::Id(want) => id == Some(want.as_str()),
            // Attribute selectors and pseudo-classes need DOM or
            // event state this triple does not carry.
            _ => false,
        })
    }
}

/// Cascade origin (Cascade 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Origin {
    /// The browser's own defaults ([`UA_CSS`]).
    Ua,
    /// The user's own stylesheet (not wired up yet).
    User,
    /// The document's own stylesheets (what the parser produces).
    #[default]
    Author,
}

impl Origin {
    /// Origin + importance rank, lowest first (Cascade 5 §6.1):
    /// UA normal < user normal < author normal < author `!important`
    /// < user `!important` < UA `!important`. Note the reversal:
    /// important user beats important author.
    pub fn rank(self, important: bool) -> u8 {
        match (self, important) {
            (Origin::Ua, false) => 0,
            (Origin::User, false) => 1,
            (Origin::Author, false) => 2,
            (Origin::Author, true) => 3,
            (Origin::User, true) => 4,
            (Origin::Ua, true) => 5,
        }
    }
}

/// One `property: value;` pair, plus whether it was `!important`.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub property: String,
    pub value: Value,
    pub important: bool,
}

/// A `@media` condition: `and` / `,` lists plus `max-width` /
/// `min-width` (and height) features with `px`/`em` values. Anything
/// else is kept as [`MediaCondition::Unknown`] and evaluates to
/// `None` (unknown), never silently to true or false.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaCondition {
    /// A bare media type: `screen`, `print`, `all`, ...
    MediaType(String),
    /// `(max-width: 700px)` — `value` is `None` for boolean features
    /// such as `(color)`.
    Feature { name: String, value: Option<Value> },
    And(Vec<MediaCondition>),
    /// A comma list.
    Or(Vec<MediaCondition>),
    Not(Box<MediaCondition>),
    Unknown(String),
}

impl MediaCondition {
    /// Parse a media query list. Never fails: unparseable input becomes
    /// [`MediaCondition::Unknown`].
    pub fn parse(raw: &str) -> Self {
        let text = raw.trim();
        if text.is_empty() {
            return MediaCondition::Unknown(String::new());
        }
        let comma_parts = split_top_level(text, ',');
        if comma_parts.len() > 1 {
            return MediaCondition::Or(
                comma_parts
                    .iter()
                    .map(|p| MediaCondition::parse(p))
                    .collect(),
            );
        }
        let and_parts = split_top_level_keyword(text, "and");
        if and_parts.len() > 1 {
            return MediaCondition::And(
                and_parts
                    .iter()
                    .map(|p| MediaCondition::parse(p))
                    .collect(),
            );
        }
        let lower = text.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("not ") {
            let rest_len = rest.len();
            let _ = rest_len;
            let inner = &text[4..];
            return MediaCondition::Not(Box::new(MediaCondition::parse(inner)));
        }
        if let Some(rest) = lower.strip_prefix("only ") {
            let _ = rest;
            return MediaCondition::parse(&text[5..]);
        }
        if text.starts_with('(') && text.ends_with(')') {
            let inner = text[1..text.len() - 1].trim();
            if let Some((name, value)) = inner.split_once(':') {
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim();
                return MediaCondition::Feature {
                    name,
                    value: if value.is_empty() {
                        None
                    } else {
                        Some(parse_value(value))
                    },
                };
            }
            return MediaCondition::Feature {
                name: inner.to_ascii_lowercase(),
                value: None,
            };
        }
        if text
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '-')
        {
            return MediaCondition::MediaType(lower);
        }
        MediaCondition::Unknown(text.to_string())
    }

    /// Evaluate against a viewport. `None` means "cannot tell"
    /// (unknown feature or type); `em` in a feature resolves against
    /// the initial 16px font size.
    pub fn eval(&self, viewport_width: f32, viewport_height: f32) -> Option<bool> {
        match self {
            MediaCondition::MediaType(t) => match t.as_str() {
                "screen" | "all" => Some(true),
                "print" => Some(false),
                _ => None,
            },
            MediaCondition::Feature { name, value } => {
                let px = |v: &Value| -> Option<f32> {
                    match v {
                        Value::Length(x) => Some(*x),
                        Value::Em(x) => Some(x * 16.0),
                        Value::Percentage(p) => Some(p / 100.0 * viewport_width),
                        _ => None,
                    }
                };
                let v = value.as_ref()?;
                let limit = px(v)?;
                match name.as_str() {
                    "max-width" => Some(viewport_width <= limit),
                    "min-width" => Some(viewport_width >= limit),
                    "max-height" => Some(viewport_height <= limit),
                    "min-height" => Some(viewport_height >= limit),
                    "width" => Some((viewport_width - limit).abs() < 0.5),
                    "height" => Some((viewport_height - limit).abs() < 0.5),
                    _ => None,
                }
            }
            MediaCondition::And(parts) => {
                let mut unknown = false;
                for p in parts {
                    match p.eval(viewport_width, viewport_height) {
                        Some(false) => return Some(false),
                        None => unknown = true,
                        Some(true) => {}
                    }
                }
                if unknown {
                    None
                } else {
                    Some(true)
                }
            }
            MediaCondition::Or(parts) => {
                let mut unknown = false;
                for p in parts {
                    match p.eval(viewport_width, viewport_height) {
                        Some(true) => return Some(true),
                        None => unknown = true,
                        Some(false) => {}
                    }
                }
                if unknown {
                    None
                } else {
                    Some(false)
                }
            }
            MediaCondition::Not(inner) => {
                inner.eval(viewport_width, viewport_height).map(|b| !b)
            }
            MediaCondition::Unknown(_) => None,
        }
    }
}

/// A raw `@supports` condition plus a best-effort evaluation for
/// `(property: value)` pairs the parser understands. Anything else
/// (`selector(...)`, unknown functions, unparseable syntax) evaluates
/// to `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportsCondition {
    pub raw: String,
}

impl SupportsCondition {
    pub fn parse(raw: &str) -> Self {
        SupportsCondition {
            raw: raw.trim().to_string(),
        }
    }

    /// Best-effort eval. A `(property: value)` pair is `Some(true)`
    /// when the property is known and the value parses to a typed
    /// (non-`Raw`) value, `Some(false)` when the property is unknown,
    /// and `None` when the value cannot be judged (e.g. a font stack
    /// kept as `Raw`). `and` / `or` / `not` combine per spec, with
    /// `None` propagating unless another branch decides the result.
    pub fn eval(&self) -> Option<bool> {
        eval_supports(&self.raw)
    }
}

/// Properties the parser recognises for `@supports` evaluation and
/// for the `font` shorthand's keyword checks. This is a judgement
/// list, not an exhaustive CSS property table.
pub const KNOWN_PROPERTIES: &[&str] = &[
    "display",
    "position",
    "top",
    "right",
    "bottom",
    "left",
    "width",
    "height",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "border",
    "border-radius",
    "border-collapse",
    "border-spacing",
    "box-shadow",
    "box-sizing",
    "color",
    "background",
    "background-color",
    "font",
    "font-family",
    "font-size",
    "font-style",
    "font-variant",
    "font-weight",
    "line-height",
    "text-align",
    "text-decoration",
    "white-space",
    "vertical-align",
    "overflow",
    "overflow-x",
    "overflow-y",
    "flex",
    "flex-direction",
    "flex-wrap",
    "flex-grow",
    "flex-shrink",
    "flex-basis",
    "justify-content",
    "align-items",
    "align-content",
    "grid",
    "grid-template-columns",
    "grid-template-rows",
    "grid-column",
    "grid-row",
    "grid-area",
    "gap",
    "grid-gap",
    "row-gap",
    "column-gap",
    "list-style",
    "list-style-type",
    "float",
    "clear",
    "visibility",
    "opacity",
    "z-index",
    "cursor",
    "transform",
    "transition",
    "content",
];

fn eval_supports(text: &str) -> Option<bool> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let lower = text.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("not ") {
        let _ = rest;
        return eval_supports(&text[4..]).map(|b| !b);
    }
    let and_parts = split_top_level_keyword(text, "and");
    if and_parts.len() > 1 {
        let mut unknown = false;
        for p in &and_parts {
            match eval_supports(p) {
                Some(false) => return Some(false),
                None => unknown = true,
                Some(true) => {}
            }
        }
        return if unknown { None } else { Some(true) };
    }
    let or_parts = split_top_level_keyword(text, "or");
    if or_parts.len() > 1 {
        let mut unknown = false;
        for p in &or_parts {
            match eval_supports(p) {
                Some(true) => return Some(true),
                None => unknown = true,
                Some(false) => {}
            }
        }
        return if unknown { None } else { Some(false) };
    }
    if text.starts_with('(') && text.ends_with(')') && balanced_parens(text) {
        let inner = text[1..text.len() - 1].trim();
        let inner_lower = inner.to_ascii_lowercase();
        if inner_lower.starts_with("selector(") || inner_lower.starts_with("font-tech(") {
            return None;
        }
        if let Some((prop, value)) = split_top_level_once(inner, ':') {
            let prop = prop.trim().to_ascii_lowercase();
            let value = value.trim();
            if value.is_empty() {
                return Some(false);
            }
            if !KNOWN_PROPERTIES.contains(&prop.as_str()) {
                return Some(false);
            }
            return match parse_value(value) {
                Value::Raw(_) => None,
                _ => Some(true),
            };
        }
        return None;
    }
    None
}

fn balanced_parens(text: &str) -> bool {
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

/// One `@import` rule. Fetching is the integration's job, not the
/// parser's: this only records what was imported, from where, and
/// under what conditions. `layer` is `Some("")` for the anonymous
/// `layer` keyword, `Some(name)` for `layer(name)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Import {
    pub url: String,
    pub layer: Option<String>,
    pub supports: Option<String>,
    pub media: Option<String>,
}

/// One `@font-face` descriptor block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FontFace {
    pub family: Option<String>,
    pub src: Option<String>,
    pub unicode_range: Option<String>,
    pub display: Option<String>,
    pub weight: Option<String>,
    pub style: Option<String>,
}

/// One `@keyframes` rule: name plus the raw block text (steps are
/// parsed when animations land, not here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keyframes {
    pub name: String,
    pub raw: String,
}

/// One `selector, selector { declarations }` rule, with its cascade
/// data attached. `media` / `supports` are the conditions of the
/// enclosing `@media` / `@supports` group rules, if any — rules are
/// NOT flattened at parse time, because viewport/feature state can
/// change after parsing.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub selectors: Vec<Selector>,
    pub declarations: Vec<Declaration>,
    pub origin: Origin,
    /// Cascade layer name, or `None` for unlayered rules. Unlayered
    /// rules beat layered rules for normal declarations.
    pub layer: Option<String>,
    /// Global sequence across the whole stylesheet, in document order
    /// (UA sheet: 0..). Last of equal key wins.
    pub source_order: u32,
    pub media: Option<MediaCondition>,
    pub supports: Option<String>,
}

impl Rule {
    /// The cascade sort key for one declaration of this rule, as a
    /// [`CascadeKey`]. `layer_order` is resolved by the caller from
    /// [`Stylesheet::layer_order`]: higher wins for normal
    /// declarations, unlayered is `i32::MAX`. (For `!important`
    /// declarations the layer order reverses — the caller must invert
    /// it; see Cascade 5.) Specificity here is the max over the
    /// rule's selectors; the per-matching-selector key is the
    /// layout pass's job.
    pub fn sort_key(&self, decl: &Declaration, layer_order: i32, inline: bool) -> CascadeKey {
        CascadeKey::new(self, decl, layer_order, inline)
    }
}

/// The cascade sort key for one declaration: `(origin rank,
/// important, layer order, inline, specificity, source order)`,
/// compared lexicographically — only consult the next criterion on a
/// tie. Higher wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CascadeKey {
    pub origin_rank: u8,
    pub important: bool,
    pub layer_order: i32,
    pub inline: bool,
    pub specificity: (u32, u32, u32),
    pub source_order: u32,
}

impl CascadeKey {
    pub fn new(rule: &Rule, decl: &Declaration, layer_order: i32, inline: bool) -> Self {
        let (origin_rank, important, lo, inl, specificity, source_order) =
            sort_key(rule, decl, layer_order, inline);
        CascadeKey {
            origin_rank,
            important,
            layer_order: lo,
            inline: inl,
            specificity,
            source_order,
        }
    }
}

/// The cascade sort key for one declaration of `rule`, as a plain
/// tuple: `(origin rank, important, layer order, inline,
/// specificity, source order)`. See [`CascadeKey`] for the layer
/// order convention and the max-specificity approximation.
pub fn sort_key(
    rule: &Rule,
    decl: &Declaration,
    layer_order: i32,
    inline: bool,
) -> (u8, bool, i32, bool, (u32, u32, u32), u32) {
    let specificity = rule
        .selectors
        .iter()
        .map(|s| s.specificity())
        .max()
        .unwrap_or((0, 0, 0));
    (
        rule.origin.rank(decl.important),
        decl.important,
        layer_order,
        inline,
        specificity,
        rule.source_order,
    )
}

/// A parsed stylesheet: rules in source order (cascade order is
/// applied by the styler, not stored here), plus what was skipped and
/// why, plus the at-rule data the styler and loader need.
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
    /// Human-readable notes for every skipped rule, declaration, or
    /// at-rule. Empty means the whole sheet parsed cleanly.
    pub errors: Vec<String>,
    /// Cascade layer names in declaration order (first declaration
    /// fixes the order). Block-form `@layer` also declares here.
    pub layers: Vec<String>,
    /// `@import` records, in document order. Fetching is not done here.
    pub imports: Vec<Import>,
    /// `@font-face` descriptor blocks, in document order.
    pub font_faces: Vec<FontFace>,
    /// `@keyframes` rules, in document order (name + raw block text).
    pub keyframes: Vec<Keyframes>,
}

impl Stylesheet {
    /// Parse CSS text. Never fails outright: malformed rules are
    /// skipped and recorded in [`Stylesheet::errors`].
    pub fn parse(css: &str) -> Self {
        Self::parse_with_origin(css, Origin::Author)
    }

    /// Parse the built-in UA stylesheet ([`UA_CSS`]) with origin
    /// [`Origin::Ua`] and `source_order` starting at 0. This replaces
    /// the hardcoded defaults previously kept in `pluto-layout`.
    pub fn ua() -> Stylesheet {
        Self::parse_with_origin(UA_CSS, Origin::Ua)
    }

    fn parse_with_origin(css: &str, origin: Origin) -> Self {
        let mut sheet = Stylesheet::default();
        let mut order = 0u32;
        let mut seen_rule = false;
        let ctx = RuleCtx::default();
        parse_rules(css, &mut sheet, &mut order, &ctx, origin, &mut seen_rule, true);
        sheet
    }

    /// Resolve a layer name to its sort position: the index in
    /// [`Stylesheet::layers`] (later-declared wins for normal
    /// declarations), `i32::MAX` for unlayered rules, and
    /// `layers.len()` for a name that was never declared (defensive;
    /// block-form `@layer` always declares first).
    pub fn layer_order(&self, layer: &Option<String>) -> i32 {
        match layer {
            None => i32::MAX,
            Some(name) => self
                .layers
                .iter()
                .position(|l| l == name)
                .map(|i| i as i32)
                .unwrap_or(self.layers.len() as i32),
        }
    }
}

/// Rule context threaded through nested group rules: the enclosing
/// layer and conditions. Carried, not flattened.
#[derive(Debug, Clone, Default)]
struct RuleCtx {
    layer: Option<String>,
    media: Option<MediaCondition>,
    supports: Option<String>,
}

/// The browser's built-in stylesheet, as real CSS text parsed by the
/// same parser as author sheets (origin UA, source order 0..). Values
/// follow the usual UA defaults: body margin 8px, `h1` 2em bold with
/// 0.67em margins, `p`/`ul` 1em vertical margins, `ul` disc with
/// 40px left padding, links `#0000EE` underlined, `head`/`title`/
/// `style`/`script`/`meta`/`link`/`base` hidden, table/form defaults.
pub const UA_CSS: &str = r#"
html { display: block; }
head, title, style, script, meta, link, base { display: none; }
body { display: block; margin: 8px; }
div, section, article, header, footer, nav, main, aside, figure, figcaption, blockquote, pre, hr, form, fieldset, address, dl, dt, dd, details, summary { display: block; }
span, a, b, i, em, strong, small, code, sub, sup, u, s, mark, abbr, cite, q, time, kbd, samp, var, bdi, bdo, data, wbr, label, output { display: inline; }
h1 { display: block; font-size: 2em; font-weight: bold; margin: 0.67em 0; }
h2 { display: block; font-size: 1.5em; font-weight: bold; margin: 0.83em 0; }
h3 { display: block; font-size: 1.17em; font-weight: bold; margin: 1em 0; }
h4 { display: block; font-size: 1em; font-weight: bold; margin: 1.33em 0; }
h5 { display: block; font-size: 0.83em; font-weight: bold; margin: 1.67em 0; }
h6 { display: block; font-size: 0.67em; font-weight: bold; margin: 2.33em 0; }
p { display: block; margin: 1em 0; }
ul { display: block; list-style-type: disc; margin: 1em 0; padding-left: 40px; }
ol { display: block; list-style-type: decimal; margin: 1em 0; padding-left: 40px; }
li { display: list-item; }
a { color: #0000EE; text-decoration: underline; }
a:visited { color: #551A8B; }
b, strong { font-weight: bold; }
i, em { font-style: italic; }
small { font-size: 0.83em; }
blockquote { display: block; margin: 1em 40px; }
figure { display: block; margin: 1em 40px; }
pre { display: block; margin: 1em 0; white-space: pre; }
hr { display: block; margin: 0.5em auto; border: 1px inset; }
fieldset { display: block; margin: 0 2px; padding: 0.35em 0.75em 0.625em; border: 1px solid; }
table { display: table; border-collapse: separate; border-spacing: 2px; }
thead { display: table-header-group; }
tbody { display: table-row-group; }
tfoot { display: table-footer-group; }
tr { display: table-row; }
td, th { display: table-cell; padding: 1px; }
th { font-weight: bold; text-align: center; }
caption { display: table-caption; text-align: center; }
col { display: table-column; }
colgroup { display: table-column-group; }
button, input, select, textarea { display: inline-block; }
button { padding: 1px 6px; }
input { padding: 1px; }
img { display: inline; }
"#;

// ---- top-level parsing ----

/// Parse rules from `text` into `sheet`, threading the cascade
/// context (`ctx`), the global source-order counter, and the
/// "have we seen a real rule yet" flag used by `@import`.
/// `top_level` is false inside group rules (`@import` is only valid
/// at the top level).
fn parse_rules(
    text: &str,
    sheet: &mut Stylesheet,
    order: &mut u32,
    ctx: &RuleCtx,
    origin: Origin,
    seen_rule: &mut bool,
    top_level: bool,
) {
    let mut cur = Cursor::new(text);
    loop {
        cur.skip_ws_comments();
        if cur.eof() {
            break;
        }
        let start_pos = cur.pos;
        if cur.rest().starts_with('@') {
            handle_at_rule(
                &mut cur, sheet, order, ctx, origin, seen_rule, top_level, start_pos,
            );
            continue;
        }
        let Some(prelude) = cur.read_until('{') else {
            if !cur.rest().trim().is_empty() {
                sheet.errors.push(format!(
                    "rule without '{{' skipped at byte {start_pos}: {:?}",
                    cur.rest().trim()
                ));
            }
            break;
        };
        let block_start = cur.pos;
        let (block, closed) = match cur.read_balanced_block() {
            Some(b) => (b, true),
            None => {
                sheet.errors.push(format!(
                    "unclosed rule block at byte {start_pos} (closed at EOF)"
                ));
                (&text[block_start..], false)
            }
        };
        let selectors = match parse_selector_list(prelude, &mut sheet.errors) {
            Some(sels) => sels,
            None => {
                if !closed {
                    break;
                }
                continue;
            }
        };
        let mut rule = Rule {
            selectors,
            declarations: Vec::new(),
            origin,
            layer: ctx.layer.clone(),
            source_order: *order,
            media: ctx.media.clone(),
            supports: ctx.supports.clone(),
        };
        *order += 1;
        *seen_rule = true;
        parse_declarations(block, &mut rule, &mut sheet.errors);
        sheet.rules.push(rule);
        if !closed {
            break;
        }
    }
}

/// Handle one at-rule; `cur` is positioned at the `@`.
#[allow(clippy::too_many_arguments)]
fn handle_at_rule(
    cur: &mut Cursor<'_>,
    sheet: &mut Stylesheet,
    order: &mut u32,
    ctx: &RuleCtx,
    origin: Origin,
    seen_rule: &mut bool,
    top_level: bool,
    start_pos: usize,
) {
    // Read the at-rule name: `@media`, `@-webkit-keyframes`, ...
    let rest = cur.rest();
    let mut name_end = 1;
    for (i, c) in rest.char_indices().skip(1) {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            name_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    let name = rest[1..name_end].to_ascii_lowercase();
    cur.pos += name_end;
    let Some((off, term)) = cur.find_top_level(&[';', '{']) else {
        sheet
            .errors
            .push(format!("unterminated at-rule at byte {start_pos}: @{name}"));
        cur.pos = cur.s.len();
        return;
    };
    let prelude = cur.rest()[..off].trim().to_string();
    match term {
        ';' => {
            cur.pos += off + 1;
            match name.as_str() {
                "charset" => {
                    // Ignored: UTF-8 is assumed. Never an error.
                }
                "import" => {
                    if !top_level || *seen_rule {
                        sheet.errors.push(format!(
                            "@import ignored at byte {start_pos}: must precede all rules"
                        ));
                    } else {
                        match parse_import(&prelude) {
                            Some(import) => sheet.imports.push(import),
                            None => sheet.errors.push(format!(
                                "unparseable @import skipped at byte {start_pos}: {prelude:?}"
                            )),
                        }
                    }
                }
                "namespace" => {
                    sheet.errors.push(format!(
                        "@namespace skipped at byte {start_pos} (XML only): {prelude:?}"
                    ));
                    *seen_rule = true;
                }
                "layer" => {
                    // Statement form: `@layer a, b;` declares order.
                    // First declaration fixes it.
                    for layer in prelude.split(',') {
                        let layer = layer.trim();
                        if !layer.is_empty() && !sheet.layers.contains(&layer.to_string()) {
                            sheet.layers.push(layer.to_string());
                        }
                    }
                }
                _ => {
                    sheet.errors.push(format!(
                        "unknown at-rule skipped at byte {start_pos}: @{name}"
                    ));
                    *seen_rule = true;
                }
            }
        }
        '{' => {
            cur.pos += off + 1; // step past the '{'
            let block_start = cur.pos;
            let (block, closed) = match cur.read_balanced_block() {
                Some(b) => (b, true),
                None => {
                    sheet.errors.push(format!(
                        "unclosed at-rule block at byte {start_pos} (closed at EOF): @{name}"
                    ));
                    (&cur.s[block_start..], false)
                }
            };
            match name.as_str() {
                "media" => {
                    *seen_rule = true;
                    let cond = MediaCondition::parse(&prelude);
                    let mut inner = ctx.clone();
                    inner.media = Some(match ctx.media.clone() {
                        Some(outer) => MediaCondition::And(vec![outer, cond]),
                        None => cond,
                    });
                    parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                }
                "supports" => {
                    *seen_rule = true;
                    let mut inner = ctx.clone();
                    inner.supports = Some(match ctx.supports.clone() {
                        Some(outer) => format!("({outer}) and ({prelude})"),
                        None => prelude.clone(),
                    });
                    parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                }
                "layer" => {
                    *seen_rule = true;
                    // Block form: `@layer name { ... }` (one name;
                    // anonymous `@layer { ... }` gets "").
                    let layer_name = prelude.trim().to_string();
                    if layer_name.contains(',') {
                        sheet.errors.push(format!(
                            "@layer block with multiple names skipped at byte {start_pos}: {prelude:?}"
                        ));
                    } else {
                        if !sheet.layers.contains(&layer_name) {
                            sheet.layers.push(layer_name.clone());
                        }
                        let mut inner = ctx.clone();
                        inner.layer = Some(layer_name);
                        parse_rules(block, sheet, order, &inner, origin, seen_rule, false);
                    }
                }
                "font-face" => {
                    *seen_rule = true;
                    sheet.font_faces.push(parse_font_face(block));
                }
                "container" | "scope" | "property" => {
                    *seen_rule = true;
                    sheet.errors.push(format!(
                        "@{name} skipped at byte {start_pos} (not implemented yet)"
                    ));
                }
                _ if name == "keyframes" || name.ends_with("-keyframes") => {
                    *seen_rule = true;
                    let kf_name = prelude
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                    sheet.keyframes.push(Keyframes {
                        name: kf_name,
                        raw: block.to_string(),
                    });
                }
                _ => {
                    *seen_rule = true;
                    sheet.errors.push(format!(
                        "unknown at-rule skipped at byte {start_pos}: @{name}"
                    ));
                }
            }
            if !closed {
                // EOF was hit inside the block; nothing follows.
                cur.pos = cur.s.len();
            }
        }
        _ => unreachable!(),
    }
}

/// Parse an `@import` prelude (text after `@import`, before `;`):
/// `url("x.css")` or `"x.css"`, then optional `layer` /
/// `layer(name)`, `supports(...)`, and a media query list.
fn parse_import(prelude: &str) -> Option<Import> {
    let mut rest = prelude.trim();
    let url;
    if rest.to_ascii_lowercase().starts_with("url(") {
        let inner_start = 4;
        let mut depth = 1i32; // already inside url('s paren
        let mut end = None;
        for (i, c) in rest.char_indices().skip(inner_start) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end?;
        let inner = rest[inner_start..end].trim();
        url = unquote(inner).unwrap_or_else(|| inner.to_string());
        rest = rest[end + 1..].trim_start();
    } else if rest.starts_with('"') || rest.starts_with('\'') {
        let (u, after) = parse_quoted(rest)?;
        url = u;
        rest = after.trim_start();
    } else {
        return None;
    }
    if url.is_empty() {
        return None;
    }
    let mut import = Import {
        url,
        ..Import::default()
    };
    // Optional `layer` / `layer(name)`, then `supports(...)`, then media.
    loop {
        let lower = rest.to_ascii_lowercase();
        if lower.starts_with("layer(") {
            let end = rest.find(')')?;
            import.layer = Some(rest[6..end].trim().to_string());
            rest = rest[end + 1..].trim_start();
        } else if lower.starts_with("layer")
            && rest[5..]
                .chars()
                .next()
                .map(|c| c.is_whitespace())
                .unwrap_or(true)
        {
            import.layer = Some(String::new());
            rest = rest[5..].trim_start();
        } else if lower.starts_with("supports(") {
            let inner_start = 9;
            let mut depth = 1i32; // already inside supports('s paren
            let mut end = None;
            for (i, c) in rest.char_indices().skip(inner_start) {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let end = end?;
            import.supports = Some(rest[inner_start..end].trim().to_string());
            rest = rest[end + 1..].trim_start();
        } else {
            break;
        }
    }
    if !rest.is_empty() {
        import.media = Some(rest.to_string());
    }
    Some(import)
}

/// Parse a `@font-face` descriptor block into a [`FontFace`].
/// Descriptors are `name: value;` pairs; unknown ones are ignored.
fn parse_font_face(block: &str) -> FontFace {
    let mut face = FontFace::default();
    for chunk in split_top_level(block, ';') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let Some((prop, raw)) = split_top_level_once(chunk, ':') else {
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let value = unquote(raw.trim()).unwrap_or_else(|| raw.trim().to_string());
        match prop.as_str() {
            "font-family" => face.family = Some(value),
            "src" => face.src = Some(value),
            "unicode-range" => face.unicode_range = Some(value),
            "font-display" => face.display = Some(value),
            "font-weight" => face.weight = Some(value),
            "font-style" => face.style = Some(value),
            _ => {}
        }
    }
    face
}

/// Parse a quoted string at the start of `s`: returns the unquoted
/// value and the remainder after the closing quote.
fn parse_quoted(s: &str) -> Option<(String, &str)> {
    let quote = s.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    let mut chars = s[1..].char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                if let Some((_, esc)) = chars.next() {
                    out.push(esc);
                }
            }
            c if c == quote => {
                return Some((out, &s[1 + i + c.len_utf8()..]));
            }
            c => out.push(c),
        }
    }
    None
}

/// Strip matching quotes from a string, if it is quoted.
fn unquote(s: &str) -> Option<String> {
    parse_quoted(s).map(|(v, _)| v)
}

// ---- cursor over the stylesheet text ----

/// Byte cursor over a stylesheet (or a block of one), aware of
/// comments, strings, and nesting depth.
struct Cursor<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Cursor { s, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn skip_ws_comments(&mut self) {
        loop {
            let trimmed = self.rest().trim_start();
            self.pos = self.s.len() - trimmed.len();
            if self.rest().starts_with("/*") {
                match self.rest().find("*/") {
                    Some(p) => self.pos += p + 2,
                    None => {
                        self.pos = self.s.len();
                        return;
                    }
                }
            } else {
                return;
            }
        }
    }

    /// Read up to `target` (outside strings); returns the slice
    /// before it and leaves the cursor just past `target`. `None`
    /// when never found. Note: only strings are skipped — brackets
    /// and parens do NOT hide the target, because a `{` in a rule
    /// prelude always opens the block even inside an (unclosed)
    /// attribute selector or functional pseudo-class.
    fn read_until(&mut self, target: char) -> Option<&'a str> {
        let start = self.pos;
        let mut iter = self.s[self.pos..].char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                _ if c == target => {
                    let end = self.pos + off;
                    let out = &self.s[start..end];
                    self.pos = end + c.len_utf8();
                    return Some(out);
                }
                _ => {}
            }
        }
        None
    }

    /// Read a balanced `{ ... }` block; the cursor must be just past
    /// the opening `{`. Returns the inner text and leaves the cursor
    /// just past the matching `}`. `None` when the block never closes
    /// (cursor ends at EOF).
    fn read_balanced_block(&mut self) -> Option<&'a str> {
        let start = self.pos;
        let mut depth = 0i32;
        let mut iter = self.s[self.pos..].char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                '/' if iter.peek().map(|(_, p)| *p) == Some('*') => {
                    iter.next();
                    let mut prev = ' ';
                    for (_, cc) in iter.by_ref() {
                        if prev == '*' && cc == '/' {
                            break;
                        }
                        prev = cc;
                    }
                }
                '{' => depth += 1,
                '}' => {
                    if depth == 0 {
                        let end = self.pos + off;
                        let inner = &self.s[start..end];
                        self.pos = end + 1;
                        return Some(inner);
                    }
                    depth -= 1;
                }
                _ => {}
            }
        }
        self.pos = self.s.len();
        None
    }

    /// Offset (relative to the cursor) and character of the first
    /// top-level `;` or `{` (outside strings/parens/brackets).
    /// Used to tell statement at-rules from block at-rules.
    fn find_top_level(&self, targets: &[char]) -> Option<(usize, char)> {
        let mut depth = 0i32;
        let mut iter = self.rest().char_indices().peekable();
        while let Some((off, c)) = iter.next() {
            match c {
                '"' | '\'' => {
                    skip_string(&mut iter, c);
                }
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ if depth == 0 && targets.contains(&c) => {
                    return Some((off, c));
                }
                _ => {}
            }
        }
        None
    }
}

/// Skip a quoted string in a `char_indices` iterator (the opening
/// quote was already consumed).
fn skip_string(
    iter: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    quote: char,
) {
    while let Some((_, c)) = iter.next() {
        if c == '\\' {
            iter.next();
            continue;
        }
        if c == quote {
            break;
        }
    }
}

/// Split `s` on top-level occurrences of `sep` (outside strings,
/// parens, and brackets). The separator itself is dropped.
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                skip_string(&mut iter, c);
            }
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(&s[start..off]);
                start = off + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Split on the first top-level occurrence of `sep`.
fn split_top_level_once(s: &str, sep: char) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                skip_string(&mut iter, c);
            }
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ if c == sep && depth == 0 => {
                return Some((&s[..off], &s[off + c.len_utf8()..]));
            }
            _ => {}
        }
    }
    None
}

/// Split on top-level whitespace (outside strings and parens).
fn split_top_level_ws(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    let mut iter = s.char_indices().peekable();
    while let Some((off, c)) = iter.next() {
        match c {
            '"' | '\'' => {
                if start.is_none() {
                    start = Some(off);
                }
                skip_string(&mut iter, c);
            }
            '(' | '[' => {
                if start.is_none() {
                    start = Some(off);
                }
                depth += 1;
            }
            ')' | ']' => depth -= 1,
            _ if c.is_whitespace() && depth == 0 => {
                if let Some(st) = start.take() {
                    out.push(&s[st..off]);
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(off);
                }
            }
        }
    }
    if let Some(st) = start {
        out.push(&s[st..]);
    }
    out
}

/// Split on a top-level keyword (`and` / `or`) matched as a whole
/// word, case-insensitively, outside strings and parens.
fn split_top_level_keyword<'a>(s: &'a str, keyword: &str) -> Vec<&'a str> {
    let lower = s.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_string: Option<char> = None;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if let Some(q) = in_string {
            if c == '\\' {
                i += 2;
                continue;
            }
            if c as u8 == q as u8 {
                in_string = None;
            }
            i += 1;
            continue;
        }
        match c {
            '"' | '\'' => {
                in_string = Some(c);
                i += 1;
            }
            '(' | '[' => {
                depth += 1;
                i += 1;
            }
            ')' | ']' => {
                depth -= 1;
                i += 1;
            }
            _ if depth == 0 && lower[i..].starts_with(keyword) => {
                let before_ok = i == 0
                    || !(bytes[i - 1] as char).is_ascii_alphanumeric()
                        && bytes[i - 1] != b'-'
                        && bytes[i - 1] != b'_';
                let after = i + keyword.len();
                let after_ok = after >= bytes.len()
                    || !(bytes[after] as char).is_ascii_alphanumeric()
                        && bytes[after] != b'-'
                        && bytes[after] != b'_';
                if before_ok && after_ok {
                    out.push(&s[start..i]);
                    start = after;
                    i = after;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    out.push(&s[start..]);
    out
}

// ---- selector parsing ----

/// Parse a selector list. ONE invalid complex selector invalidates
/// the WHOLE list (CSS Syntax L3): returns `None` and records one
/// error, and the caller drops the entire rule.
fn parse_selector_list(text: &str, errors: &mut Vec<String>) -> Option<Vec<Selector>> {
    let mut out = Vec::new();
    for item in split_top_level(text, ',') {
        let item = item.trim();
        if item.is_empty() {
            errors.push(format!(
                "invalid selector list (empty item), whole rule dropped: {text:?}"
            ));
            return None;
        }
        match parse_complex_selector(item) {
            Some(sel) => out.push(sel),
            None => {
                errors.push(format!(
                    "invalid selector {item:?} in list, whole rule dropped: {text:?}"
                ));
                return None;
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse a forgiving selector list (inside `:is()` / `:not()` /
/// `:where()` / `:has()`): invalid entries are dropped, the valid
/// ones are kept.
fn parse_forgiving_selector_list(text: &str) -> Vec<Selector> {
    let mut out = Vec::new();
    for item in split_top_level(text, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        // `:has()` takes relative selectors (`> p`); a leading
        // combinator is stored against an implied universal subject.
        let item = if item.starts_with('>') || item.starts_with('+') || item.starts_with('~') {
            format!("* {item}")
        } else {
            item.to_string()
        };
        if let Some(sel) = parse_complex_selector(&item) {
            out.push(sel);
        }
    }
    out
}

/// Parse one complex selector: compounds joined by combinators.
fn parse_complex_selector(text: &str) -> Option<Selector> {
    let chars: Vec<char> = text.chars().collect();
    let mut p = SelectorParser {
        chars: &chars,
        i: 0,
    };
    let mut sel = Selector::default();
    p.skip_ws();
    if p.eof() {
        return None;
    }
    if matches!(p.peek(), Some('>') | Some('+') | Some('~')) {
        return None; // leading combinator
    }
    sel.compounds.push(p.parse_compound()?);
    loop {
        let had_ws = p.skip_ws();
        if p.eof() {
            break;
        }
        let comb = match p.peek() {
            Some('>') => {
                p.i += 1;
                Combinator::Child
            }
            Some('+') => {
                p.i += 1;
                Combinator::NextSibling
            }
            Some('~') => {
                p.i += 1;
                Combinator::SubsequentSibling
            }
            Some(_) if had_ws => Combinator::Descendant,
            _ => return None,
        };
        p.skip_ws();
        if p.eof() {
            return None; // trailing combinator
        }
        sel.compounds.push(p.parse_compound()?);
        sel.combinators.push(comb);
    }
    // A pseudo-element may only appear on the last compound, as its
    // final part; move it onto the selector.
    for compound in sel.compounds.iter().take(sel.compounds.len().saturating_sub(1)) {
        if compound
            .parts
            .iter()
            .any(|part| matches!(part, SimpleSelector::PseudoElement(_)))
        {
            return None;
        }
    }
    let pseudo = sel
        .compounds
        .last()
        .and_then(|last| last.parts.last())
        .and_then(|part| match part {
            SimpleSelector::PseudoElement(pe) => Some(*pe),
            _ => None,
        });
    if let Some(pe) = pseudo {
        let last = sel.compounds.last_mut().unwrap();
        last.parts.pop();
        sel.pseudo_element = Some(pe);
        if last.parts.is_empty() {
            // `::before` alone is not a selector; keep the
            // compound only if something else matched. A bare
            // pseudo-element has no originating element here.
            return None;
        }
    }
    sel.has_unimplemented = sel.compounds.iter().any(|c| {
        c.parts.iter().any(|part| {
            matches!(part, SimpleSelector::PseudoClass(PseudoClass::Has(_)))
        })
    });
    Some(sel)
}

struct SelectorParser<'a> {
    chars: &'a [char],
    i: usize,
}

impl<'a> SelectorParser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn eof(&self) -> bool {
        self.i >= self.chars.len()
    }

    /// Skip whitespace; true when any was skipped.
    fn skip_ws(&mut self) -> bool {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.i += 1;
        }
        self.i > start
    }

    fn parse_compound(&mut self) -> Option<CompoundSelector> {
        let mut parts = Vec::new();
        let mut saw_pseudo_element = false;
        loop {
            match self.peek() {
                None => break,
                Some(c) if c.is_whitespace() || c == '>' || c == '+' || c == '~' || c == ',' => {
                    break
                }
                Some('*') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    self.i += 1;
                    parts.push(SimpleSelector::Universal);
                }
                Some('#') | Some('.') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    let kind = self.peek().unwrap();
                    self.i += 1;
                    let name = self.parse_name()?;
                    parts.push(if kind == '#' {
                        SimpleSelector::Id(name)
                    } else {
                        SimpleSelector::Class(name)
                    });
                }
                Some('[') => {
                    if saw_pseudo_element {
                        return None;
                    }
                    parts.push(self.parse_attribute()?);
                }
                Some(':') => match self.parse_pseudo()? {
                    PseudoOut::Element(pe) => {
                        if saw_pseudo_element {
                            return None;
                        }
                        saw_pseudo_element = true;
                        parts.push(SimpleSelector::PseudoElement(pe));
                    }
                    PseudoOut::Class(pc) => {
                        if saw_pseudo_element {
                            return None;
                        }
                        parts.push(SimpleSelector::PseudoClass(pc));
                    }
                },
                Some(c) if is_ident_start(c) => {
                    if saw_pseudo_element {
                        return None;
                    }
                    let name = self.parse_name()?;
                    parts.push(SimpleSelector::Type(name.to_ascii_lowercase()));
                }
                _ => return None,
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(CompoundSelector { parts })
        }
    }

    /// An identifier: `[A-Za-z0-9_-]` plus non-ASCII, with `\x`
    /// escapes. `None` when empty.
    fn parse_name(&mut self) -> Option<String> {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                self.i += 1;
                let esc = self.peek()?;
                out.push(esc);
                self.i += 1;
            } else if c.is_ascii_alphanumeric() || c == '-' || c == '_' || !c.is_ascii() {
                out.push(c);
                self.i += 1;
            } else {
                break;
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    fn parse_attribute(&mut self) -> Option<SimpleSelector> {
        debug_assert_eq!(self.peek(), Some('['));
        self.i += 1;
        self.skip_ws();
        let name = self.parse_name()?;
        self.skip_ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Some(SimpleSelector::Attribute {
                name,
                op: AttrOp::Exists,
                value: None,
                case_sensitive: None,
            });
        }
        let op = match (self.peek(), self.chars.get(self.i + 1).copied()) {
            (Some('='), _) => {
                self.i += 1;
                AttrOp::Equals
            }
            (Some('~'), Some('=')) => {
                self.i += 2;
                AttrOp::Includes
            }
            (Some('|'), Some('=')) => {
                self.i += 2;
                AttrOp::Dash
            }
            (Some('^'), Some('=')) => {
                self.i += 2;
                AttrOp::Prefix
            }
            (Some('$'), Some('=')) => {
                self.i += 2;
                AttrOp::Suffix
            }
            (Some('*'), Some('=')) => {
                self.i += 2;
                AttrOp::Substring
            }
            _ => return None,
        };
        self.skip_ws();
        let value = match self.peek() {
            Some('"') | Some('\'') => {
                let quote = self.peek().unwrap();
                self.i += 1;
                let mut out = String::new();
                loop {
                    match self.peek() {
                        None => return None,
                        Some(c) if c == quote => {
                            self.i += 1;
                            break;
                        }
                        Some('\\') => {
                            self.i += 1;
                            let esc = self.peek()?;
                            out.push(esc);
                            self.i += 1;
                        }
                        Some(c) => {
                            out.push(c);
                            self.i += 1;
                        }
                    }
                }
                out
            }
            Some(_) => self.parse_name()?,
            None => return None,
        };
        self.skip_ws();
        // An `i`/`s` case flag only counts when followed by `]` (or
        // whitespace then `]`).
        let mut case_sensitive = None;
        if let Some(flag) = self.peek()
            && matches!(flag, 'i' | 'I' | 's' | 'S')
            && self
                .chars
                .get(self.i + 1)
                .copied()
                .is_some_and(|c| c == ']' || c.is_whitespace())
        {
            case_sensitive = Some(matches!(flag, 's' | 'S'));
            self.i += 1;
            self.skip_ws();
        }
        if self.peek() != Some(']') {
            return None;
        }
        self.i += 1;
        Some(SimpleSelector::Attribute {
            name,
            op,
            value: Some(value),
            case_sensitive,
        })
    }

    /// Read the balanced `( ... )` after a functional pseudo-class.
    fn read_paren_arg(&mut self) -> Option<String> {
        if self.peek() != Some('(') {
            return None;
        }
        self.i += 1;
        let start = self.i;
        let mut depth = 1i32;
        while let Some(c) = self.peek() {
            match c {
                '"' | '\'' => {
                    let quote = c;
                    self.i += 1;
                    loop {
                        match self.peek() {
                            None => return None,
                            Some('\\') => self.i += 2,
                            Some(c2) if c2 == quote => {
                                self.i += 1;
                                break;
                            }
                            Some(_) => self.i += 1,
                        }
                    }
                }
                '(' => {
                    depth += 1;
                    self.i += 1;
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        let arg: String = self.chars[start..self.i].iter().collect();
                        self.i += 1;
                        return Some(arg);
                    }
                    self.i += 1;
                }
                _ => self.i += 1,
            }
        }
        None
    }

    fn parse_pseudo(&mut self) -> Option<PseudoOut> {
        debug_assert_eq!(self.peek(), Some(':'));
        self.i += 1;
        if self.peek() == Some(':') {
            self.i += 1;
            let name = self.parse_name()?.to_ascii_lowercase();
            return match name.as_str() {
                "before" => Some(PseudoOut::Element(PseudoElement::Before)),
                "after" => Some(PseudoOut::Element(PseudoElement::After)),
                "first-line" => Some(PseudoOut::Element(PseudoElement::FirstLine)),
                "first-letter" => Some(PseudoOut::Element(PseudoElement::FirstLetter)),
                "marker" => Some(PseudoOut::Element(PseudoElement::Marker)),
                "selection" => Some(PseudoOut::Element(PseudoElement::Selection)),
                _ => None,
            };
        }
        let name = self.parse_name()?.to_ascii_lowercase();
        // Legacy single-colon pseudo-elements.
        match name.as_str() {
            "before" => return Some(PseudoOut::Element(PseudoElement::Before)),
            "after" => return Some(PseudoOut::Element(PseudoElement::After)),
            "first-line" => return Some(PseudoOut::Element(PseudoElement::FirstLine)),
            "first-letter" => return Some(PseudoOut::Element(PseudoElement::FirstLetter)),
            _ => {}
        }
        let pc = match name.as_str() {
            "root" => PseudoClass::Root,
            "first-child" => PseudoClass::FirstChild,
            "last-child" => PseudoClass::LastChild,
            "only-child" => PseudoClass::OnlyChild,
            "empty" => PseudoClass::Empty,
            "first-of-type" => PseudoClass::FirstOfType,
            "last-of-type" => PseudoClass::LastOfType,
            "only-of-type" => PseudoClass::OnlyOfType,
            "link" => PseudoClass::Link,
            "visited" => PseudoClass::Visited,
            "any-link" => PseudoClass::AnyLink,
            "hover" => PseudoClass::Hover,
            "active" => PseudoClass::Active,
            "focus" => PseudoClass::Focus,
            "focus-visible" => PseudoClass::FocusVisible,
            "focus-within" => PseudoClass::FocusWithin,
            "nth-child" | "nth-last-child" | "nth-of-type" | "nth-last-of-type" => {
                let arg = self.read_paren_arg()?;
                let ab = parse_an_plus_b(&arg)?;
                match name.as_str() {
                    "nth-child" => PseudoClass::NthChild(ab),
                    "nth-last-child" => PseudoClass::NthLastChild(ab),
                    "nth-of-type" => PseudoClass::NthOfType(ab),
                    _ => PseudoClass::NthLastOfType(ab),
                }
            }
            "not" | "is" | "where" | "has" => {
                let arg = self.read_paren_arg()?;
                let list = parse_forgiving_selector_list(&arg);
                if list.is_empty() {
                    return None;
                }
                match name.as_str() {
                    "not" => PseudoClass::Not(list),
                    "is" => PseudoClass::Is(list),
                    "where" => PseudoClass::Where(list),
                    _ => PseudoClass::Has(list),
                }
            }
            _ => return None, // unknown pseudo-class invalidates the selector
        };
        Some(PseudoOut::Class(pc))
    }
}

enum PseudoOut {
    Element(PseudoElement),
    Class(PseudoClass),
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '-' || !c.is_ascii()
}

/// Parse an `an+b` argument: `odd`, `even`, `3`, `n`, `-n+3`, `2n+1`.
fn parse_an_plus_b(s: &str) -> Option<AnPlusB> {
    let t = s.trim().to_ascii_lowercase();
    if t == "odd" {
        return Some(AnPlusB { a: 2, b: 1 });
    }
    if t == "even" {
        return Some(AnPlusB { a: 2, b: 0 });
    }
    if let Ok(n) = t.parse::<i32>() {
        return Some(AnPlusB { a: 0, b: n });
    }
    let t: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    let npos = t.find('n')?;
    let (a_part, rest) = t.split_at(npos);
    let a = match a_part {
        "" | "+" => 1,
        "-" => -1,
        _ => a_part.parse::<i32>().ok()?,
    };
    let rest = &rest[1..];
    let b = if rest.is_empty() {
        0
    } else {
        rest.parse::<i32>().ok()?
    };
    Some(AnPlusB { a, b })
}

// ---- declarations & shorthands ----

fn decl(property: String, value: Value, important: bool) -> Declaration {
    Declaration {
        property,
        value,
        important,
    }
}

fn parse_declarations(block: &str, rule: &mut Rule, errors: &mut Vec<String>) {
    for chunk in split_top_level(block, ';') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let Some((prop, raw)) = split_top_level_once(chunk, ':') else {
            errors.push(format!("declaration without ':' skipped: {chunk:?}"));
            continue;
        };
        let prop = prop.trim().to_ascii_lowercase();
        let mut raw = raw.trim();
        if prop.is_empty() || raw.is_empty() {
            errors.push(format!("empty declaration skipped: {chunk:?}"));
            continue;
        }
        // A second top-level ':' in the value means a missing `;`
        // merged two declarations into one invalid one: both are lost.
        if split_top_level_once(raw, ':').is_some() {
            errors.push(format!(
                "declaration merged by missing ';' dropped: {chunk:?}"
            ));
            continue;
        }
        let mut important = false;
        if let Some(stripped) = strip_important(raw) {
            if stripped.is_empty() {
                errors.push(format!("empty declaration skipped: {chunk:?}"));
                continue;
            }
            raw = stripped;
            important = true;
        }
        match prop.as_str() {
            "margin" | "padding" => match expand_shorthand(&prop, raw, important) {
                Some(decls) => rule.declarations.extend(decls),
                None => rule
                    .declarations
                    .push(decl(prop, parse_value(raw), important)),
            },
            "font" => match expand_font_shorthand(raw, important) {
                Some(decls) => rule.declarations.extend(decls),
                None => errors.push(format!("invalid font shorthand dropped: {chunk:?}")),
            },
            "border-radius" => match parse_border_radius(raw) {
                Some(v) => rule.declarations.push(decl(prop, v, important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            "box-shadow" => match parse_box_shadow(raw) {
                Some(v) => rule.declarations.push(decl(prop, v, important)),
                None => rule
                    .declarations
                    .push(decl(prop, Value::Raw(raw.to_string()), important)),
            },
            _ => rule
                .declarations
                .push(decl(prop, parse_value(raw), important)),
        }
    }
}

/// Strip a trailing `!important` (case-insensitive, whitespace
/// allowed around the `!`). Returns the value without it.
fn strip_important(raw: &str) -> Option<&str> {
    let t = raw.trim_end();
    let lower = t.to_ascii_lowercase();
    let stripped = lower.strip_suffix("important")?;
    let stripped = stripped.trim_end();
    let stripped = stripped.strip_suffix('!')?;
    Some(t[..stripped.len()].trim_end())
}

/// Expand `margin`/`padding: 1-4 values` into the four longhands.
/// Returns `None` when a token is not a length, percentage, or `auto`
/// (the declaration is then kept as a single `Raw` value).
fn expand_shorthand(prop: &str, raw: &str, important: bool) -> Option<Vec<Declaration>> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        match parse_value(t) {
            Value::Length(v) => vals.push(Value::Length(v)),
            Value::Percentage(v) => vals.push(Value::Percentage(v)),
            Value::Em(v) => vals.push(Value::Em(v)),
            Value::Rem(v) => vals.push(Value::Rem(v)),
            Value::Vh(v) => vals.push(Value::Vh(v)),
            Value::Vw(v) => vals.push(Value::Vw(v)),
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
        decl(format!("{prop}-top"), top, important),
        decl(format!("{prop}-right"), right, important),
        decl(format!("{prop}-bottom"), bottom, important),
        decl(format!("{prop}-left"), left, important),
    ])
}

const FONT_SIZE_KEYWORDS: &[&str] = &[
    "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "xxx-large",
    "larger", "smaller",
];

/// Expand `font: [style] [variant] [weight] size[/line-height] family`
/// into its longhands. Size + family are required; anything else
/// invalid returns `None` (the declaration is dropped).
fn expand_font_shorthand(raw: &str, important: bool) -> Option<Vec<Declaration>> {
    let tokens = split_top_level_ws(raw);
    if tokens.is_empty() {
        return None;
    }
    let mut i = 0;
    let mut style: Option<String> = None;
    let mut variant: Option<String> = None;
    let mut weight: Option<Value> = None;
    // Style / variant / weight keywords before the size.
    while i < tokens.len() {
        let tok = tokens[i];
        let lower = tok.to_ascii_lowercase();
        match lower.as_str() {
            "italic" | "oblique" => {
                if style.is_some() {
                    return None;
                }
                style = Some(lower);
                i += 1;
            }
            "small-caps" => {
                if variant.is_some() {
                    return None;
                }
                variant = Some(lower);
                i += 1;
            }
            "bold" | "bolder" | "lighter" => {
                if weight.is_some() {
                    return None;
                }
                weight = Some(Value::Keyword(lower));
                i += 1;
            }
            "normal" => {
                i += 1; // `normal` for style/variant/weight: no decl needed
            }
            _ => {
                if let Ok(n) = tok.parse::<f32>() {
                    if (1.0..=1000.0).contains(&n) && weight.is_none() {
                        weight = Some(Value::Raw(tok.to_string()));
                        i += 1;
                    } else {
                        return None;
                    }
                } else {
                    break; // must be the size token
                }
            }
        }
    }
    if i >= tokens.len() {
        return None; // no size, no family
    }
    // Size token, optionally with `/line-height` (which may also be
    // split across tokens: `12px / 1.5`).
    let mut size_tok = tokens[i].to_string();
    i += 1;
    let mut line_height: Option<Value> = None;
    if size_tok.ends_with('/') {
        size_tok.pop();
        if i < tokens.len() {
            line_height = Some(parse_value(tokens[i]));
            i += 1;
        } else {
            return None;
        }
    } else if let Some((size_part, lh_part)) = size_tok.split_once('/') {
        let lh_part = lh_part.to_string();
        size_tok = size_part.to_string();
        if lh_part.is_empty() {
            if i < tokens.len() {
                line_height = Some(parse_value(tokens[i]));
                i += 1;
            } else {
                return None;
            }
        } else {
            line_height = Some(parse_value(&lh_part));
        }
    }
    let size = match parse_value(&size_tok) {
        v @ (Value::Length(_)
        | Value::Percentage(_)
        | Value::Em(_)
        | Value::Rem(_)
        | Value::Vh(_)
        | Value::Vw(_)) => v,
        Value::Keyword(k) if FONT_SIZE_KEYWORDS.contains(&k.as_str()) => Value::Keyword(k),
        _ => return None,
    };
    if matches!(line_height.as_ref(), Some(Value::Raw(_))) {
        return None;
    }
    if i >= tokens.len() {
        return None; // no family
    }
    let family = tokens[i..].join(" ");
    if family.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    if let Some(s) = style {
        out.push(decl(
            "font-style".to_string(),
            Value::Keyword(s),
            important,
        ));
    }
    if let Some(v) = variant {
        out.push(decl(
            "font-variant".to_string(),
            Value::Keyword(v),
            important,
        ));
    }
    if let Some(w) = weight {
        out.push(decl("font-weight".to_string(), w, important));
    }
    out.push(decl("font-size".to_string(), size, important));
    if let Some(lh) = line_height {
        out.push(decl("line-height".to_string(), lh, important));
    }
    out.push(decl(
        "font-family".to_string(),
        Value::Raw(family),
        important,
    ));
    Some(out)
}

/// Parse `border-radius: 1-4 lengths`. The elliptical part after `/`
/// is dropped (not modelled yet). `None` keeps the `Raw` fallback.
fn parse_border_radius(raw: &str) -> Option<Value> {
    let before_slash = raw.split('/').next()?;
    let tokens: Vec<&str> = before_slash.split_whitespace().collect();
    if tokens.is_empty() || tokens.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(tokens.len());
    for t in &tokens {
        vals.push(CssLength::from_value(&parse_value(t))?);
    }
    let (tl, tr, br, bl) = match vals.len() {
        1 => (vals[0], vals[0], vals[0], vals[0]),
        2 => (vals[0], vals[1], vals[0], vals[1]),
        3 => (vals[0], vals[1], vals[2], vals[1]),
        4 => (vals[0], vals[1], vals[2], vals[3]),
        _ => unreachable!(),
    };
    Some(Value::BorderRadius(BorderRadius {
        top_left: tl,
        top_right: tr,
        bottom_right: br,
        bottom_left: bl,
    }))
}

/// Parse `box-shadow: <shadow>, ...`. `None` keeps the `Raw` fallback.
fn parse_box_shadow(raw: &str) -> Option<Value> {
    if raw.trim().eq_ignore_ascii_case("none") {
        return Some(Value::BoxShadow(Vec::new()));
    }
    let mut shadows = Vec::new();
    for part in split_top_level(raw, ',') {
        shadows.push(parse_one_box_shadow(part.trim())?);
    }
    if shadows.is_empty() {
        return None;
    }
    Some(Value::BoxShadow(shadows))
}

fn parse_one_box_shadow(s: &str) -> Option<BoxShadow> {
    let mut inset = false;
    let mut lengths = Vec::new();
    let mut color = None;
    for tok in split_top_level_ws(s) {
        if tok.eq_ignore_ascii_case("inset") {
            inset = true;
            continue;
        }
        if let Some(l) = CssLength::from_value(&parse_value(tok)) {
            lengths.push(l);
            continue;
        }
        if color.is_none() && let Some(c) = parse_color_token(tok) {
            color = Some(c);
            continue;
        }
        return None;
    }
    if lengths.len() < 2 || lengths.len() > 4 {
        return None;
    }
    Some(BoxShadow {
        inset,
        offset_x: lengths[0],
        offset_y: lengths[1],
        blur: lengths.get(2).copied().unwrap_or(CssLength::Px(0.0)),
        spread: lengths.get(3).copied().unwrap_or(CssLength::Px(0.0)),
        color,
    })
}

fn parse_color_token(tok: &str) -> Option<(u8, u8, u8, f32)> {
    match parse_value(tok) {
        Value::Color(r, g, b) => Some((r, g, b, 1.0)),
        Value::ColorA(r, g, b, a) => Some((r, g, b, a)),
        _ => None,
    }
}

// ---- values ----

/// Interpret one raw value string as a typed [`Value`].
pub fn parse_value(raw: &str) -> Value {
    let s = raw.trim();
    let lower = s.to_ascii_lowercase();
    // NOTE: `rem` must be checked before `em` ("2rem" ends in "em").
    if let Some(num) = lower.strip_suffix("rem")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Rem(v);
    }
    if let Some(num) = lower.strip_suffix("em")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Em(v);
    }
    if let Some(num) = lower.strip_suffix("vh")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Vh(v);
    }
    if let Some(num) = lower.strip_suffix("vw")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Vw(v);
    }
    if let Some(num) = lower.strip_suffix("px")
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Length(v);
    }
    if let Some(num) = lower.strip_suffix('%')
        && let Ok(v) = num.trim().parse::<f32>()
    {
        return Value::Percentage(v);
    }
    if let Ok(v) = s.parse::<f32>() {
        return Value::Length(v); // bare number: unitless zero, e.g. `margin: 0`
    }
    if let Some(hex) = lower.strip_prefix('#')
        && let Some((r, g, b, a)) = parse_hex_color(hex)
    {
        if a >= 1.0 {
            return Value::Color(r, g, b);
        }
        return Value::ColorA(r, g, b, a);
    }
    if lower.starts_with("rgba(")
        && lower.ends_with(')')
        && let Some(c) = parse_rgba(&lower[5..lower.len() - 1])
    {
        return Value::ColorA(c.0, c.1, c.2, c.3);
    }
    if lower.starts_with("rgb(")
        && lower.ends_with(')')
        && let Some(c) = parse_rgb(&lower[4..lower.len() - 1])
    {
        return Value::Color(c.0, c.1, c.2);
    }
    if lower.starts_with("hsla(")
        && lower.ends_with(')')
        && let Some(c) = parse_hsl(&lower[5..lower.len() - 1], true)
    {
        return Value::ColorA(c.0, c.1, c.2, c.3);
    }
    if lower.starts_with("hsl(")
        && lower.ends_with(')')
        && let Some(c) = parse_hsl(&lower[4..lower.len() - 1], false)
    {
        return Value::Color(c.0, c.1, c.2);
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

/// `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`. Alpha is 1.0 unless the
/// 4/8-digit form says otherwise.
fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8, f32)> {
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
            1.0,
        )),
        4 => Some((
            nibble(b[0])? * 17,
            nibble(b[1])? * 17,
            nibble(b[2])? * 17,
            nibble(b[3])? as f32 * 17.0 / 255.0,
        )),
        6 => Some((
            nibble(b[0])? * 16 + nibble(b[1])?,
            nibble(b[2])? * 16 + nibble(b[3])?,
            nibble(b[4])? * 16 + nibble(b[5])?,
            1.0,
        )),
        8 => Some((
            nibble(b[0])? * 16 + nibble(b[1])?,
            nibble(b[2])? * 16 + nibble(b[3])?,
            nibble(b[4])? * 16 + nibble(b[5])?,
            (nibble(b[6])? * 16 + nibble(b[7])?) as f32 / 255.0,
        )),
        _ => None,
    }
}

fn parse_channel(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct.trim().parse().ok()?;
        return Some((v / 100.0 * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    s.parse::<u8>().ok()
}

fn parse_alpha(s: &str) -> Option<f32> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let v: f32 = pct.trim().parse().ok()?;
        return Some((v / 100.0).clamp(0.0, 1.0));
    }
    s.parse::<f32>().ok().map(|v| v.clamp(0.0, 1.0))
}

fn parse_rgb(inner: &str) -> Option<(u8, u8, u8)> {
    let mut parts = inner.split(',');
    let r = parse_channel(parts.next()?)?;
    let g = parse_channel(parts.next()?)?;
    let b = parse_channel(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b))
}

fn parse_rgba(inner: &str) -> Option<(u8, u8, u8, f32)> {
    let mut parts = inner.split(',');
    let r = parse_channel(parts.next()?)?;
    let g = parse_channel(parts.next()?)?;
    let b = parse_channel(parts.next()?)?;
    let a = parse_alpha(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((r, g, b, a))
}

/// `hsl(h, s%, l%)` / `hsla(h, s%, l%, a)`. Hue may carry a `deg`
/// suffix; `s`/`l` are percentages.
fn parse_hsl(inner: &str, with_alpha: bool) -> Option<(u8, u8, u8, f32)> {
    let mut parts = inner.split(',');
    let h_raw = parts.next()?.trim();
    let h_raw = h_raw.strip_suffix("deg").unwrap_or(h_raw);
    let h: f32 = h_raw.parse().ok()?;
    let s_raw = parts.next()?.trim();
    let s_pct: f32 = s_raw.strip_suffix('%')?.trim().parse().ok()?;
    let l_raw = parts.next()?.trim();
    let l_pct: f32 = l_raw.strip_suffix('%')?.trim().parse().ok()?;
    let a = if with_alpha {
        parse_alpha(parts.next()?)?
    } else {
        1.0
    };
    if parts.next().is_some() {
        return None;
    }
    let (r, g, b) = hsl_to_rgb(h, s_pct / 100.0, l_pct / 100.0);
    Some((r, g, b, a))
}

fn hsl_to_rgb(h_deg: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = (((h_deg % 360.0) + 360.0) % 360.0) / 360.0;
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let conv = |v: f32| ((v + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    (conv(r), conv(g), conv(b))
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

impl fmt::Display for CssLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CssLength::Px(v) => write!(f, "{v}px"),
            CssLength::Percentage(v) => write!(f, "{v}%"),
            CssLength::Em(v) => write!(f, "{v}em"),
            CssLength::Rem(v) => write!(f, "{v}rem"),
            CssLength::Vh(v) => write!(f, "{v}vh"),
            CssLength::Vw(v) => write!(f, "{v}vw"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Keyword(k) => write!(f, "{k}"),
            Value::Length(v) => write!(f, "{v}px"),
            Value::Percentage(v) => write!(f, "{v}%"),
            Value::Em(v) => write!(f, "{v}em"),
            Value::Rem(v) => write!(f, "{v}rem"),
            Value::Vh(v) => write!(f, "{v}vh"),
            Value::Vw(v) => write!(f, "{v}vw"),
            Value::Color(r, g, b) => write!(f, "rgb({r}, {g}, {b})"),
            Value::ColorA(r, g, b, a) => write!(f, "rgba({r}, {g}, {b}, {a})"),
            Value::BorderRadius(br) => {
                let corners = [br.top_left, br.top_right, br.bottom_right, br.bottom_left];
                if corners.iter().all(|c| *c == corners[0]) {
                    write!(f, "{}", corners[0])
                } else if corners[0] == corners[2] && corners[1] == corners[3] {
                    write!(f, "{} {}", corners[0], corners[1])
                } else if corners[1] == corners[3] {
                    write!(f, "{} {} {}", corners[0], corners[1], corners[2])
                } else {
                    write!(
                        f,
                        "{} {} {} {}",
                        corners[0], corners[1], corners[2], corners[3]
                    )
                }
            }
            Value::BoxShadow(shadows) => {
                if shadows.is_empty() {
                    return write!(f, "none");
                }
                let mut first = true;
                for sh in shadows {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    if sh.inset {
                        write!(f, "inset ")?;
                    }
                    write!(f, "{} {}", sh.offset_x, sh.offset_y)?;
                    if sh.blur != CssLength::Px(0.0) || sh.spread != CssLength::Px(0.0) {
                        write!(f, " {}", sh.blur)?;
                    }
                    if sh.spread != CssLength::Px(0.0) {
                        write!(f, " {}", sh.spread)?;
                    }
                    if let Some((r, g, b, a)) = sh.color {
                        if a >= 1.0 {
                            write!(f, " rgb({r}, {g}, {b})")?;
                        } else {
                            write!(f, " rgba({r}, {g}, {b}, {a})")?;
                        }
                    }
                }
                Ok(())
            }
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

    fn decl_entry<'a>(rule: &'a Rule, prop: &str) -> Option<&'a Declaration> {
        rule.declarations.iter().find(|d| d.property == prop)
    }

    #[test]
    fn parses_example_com_stylesheet() {
        let sheet = Stylesheet::parse(EXAMPLE_CSS);
        // M2 truth: everything parses now — body, div, a:link/a:visited,
        // and the @media div rule. Zero errors.
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 4);

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
        // `margin: 5em auto` now expands (em is typed in M2).
        assert_eq!(decl(div, "margin-top"), Some(&Value::Em(5.0)));
        assert_eq!(
            decl(div, "margin-right"),
            Some(&Value::Keyword("auto".into()))
        );
        assert_eq!(decl(div, "margin-bottom"), Some(&Value::Em(5.0)));
        assert_eq!(decl(div, "padding-top"), Some(&Value::Em(2.0)));
        assert_eq!(
            decl(div, "border-radius"),
            Some(&Value::BorderRadius(BorderRadius {
                top_left: CssLength::Em(0.5),
                top_right: CssLength::Em(0.5),
                bottom_right: CssLength::Em(0.5),
                bottom_left: CssLength::Em(0.5),
            }))
        );
        assert_eq!(
            decl(div, "box-shadow"),
            Some(&Value::BoxShadow(vec![BoxShadow {
                inset: false,
                offset_x: CssLength::Px(2.0),
                offset_y: CssLength::Px(3.0),
                blur: CssLength::Px(7.0),
                spread: CssLength::Px(2.0),
                color: Some((0, 0, 0, 0.02)),
            }]))
        );

        // a:link, a:visited now parse (link pseudo-classes stored).
        let link = &sheet.rules[2];
        assert_eq!(link.selectors.len(), 2);
        assert_eq!(
            decl(link, "color"),
            Some(&Value::Color(0x38, 0x48, 0x8f))
        );

        // The @media rule is kept with its condition attached.
        let media_div = &sheet.rules[3];
        assert_eq!(
            media_div.media,
            Some(MediaCondition::Feature {
                name: "max-width".to_string(),
                value: Some(Value::Length(700.0)),
            })
        );
        assert_eq!(
            media_div.media.as_ref().unwrap().eval(600.0, 800.0),
            Some(true)
        );
        assert_eq!(
            media_div.media.as_ref().unwrap().eval(800.0, 800.0),
            Some(false)
        );
        assert_eq!(decl(media_div, "width"), Some(&Value::Keyword("auto".into())));
        assert_eq!(decl(media_div, "margin-top"), Some(&Value::Length(0.0)));
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
        // M2: `2em` is typed now (was Raw in M1).
        assert_eq!(parse_value("2em"), Value::Em(2.0));
        assert_eq!(parse_value("1.5rem"), Value::Rem(1.5));
        assert_eq!(parse_value("25vh"), Value::Vh(25.0));
        assert_eq!(parse_value("100vw"), Value::Vw(100.0));
        assert_eq!(
            parse_value("rgba(10, 20, 30, 0.5)"),
            Value::ColorA(10, 20, 30, 0.5)
        );
        // hsla(0, 100%, 50%, 0.25) is red at quarter alpha.
        assert_eq!(
            parse_value("hsla(0, 100%, 50%, 0.25)"),
            Value::ColorA(255, 0, 0, 0.25)
        );
        assert_eq!(parse_value("hsl(120, 100%, 50%)"), Value::Color(0, 255, 0));
        assert_eq!(
            parse_value("#ff000080"),
            Value::ColorA(255, 0, 0, 128.0 / 255.0)
        );
    }

    // ---- R5: complex selectors ----

    #[test]
    fn combinators_parse() {
        let sheet = Stylesheet::parse(
            "div p { color: red; } div > p { color: green; } \
             h1 + p { color: blue; } h1 ~ p { color: yellow; } \
             div.note > p:first-child { color: pink; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 5);
        let combos = [
            Combinator::Descendant,
            Combinator::Child,
            Combinator::NextSibling,
            Combinator::SubsequentSibling,
            Combinator::Child,
        ];
        for (i, want) in combos.iter().enumerate() {
            let sel = &sheet.rules[i].selectors[0];
            assert_eq!(sel.compounds.len(), 2, "rule {i}");
            assert_eq!(sel.combinators, vec![*want], "rule {i}");
        }
        // Compound-only matches() still works; combinator selectors
        // need DOM context and return false here.
        assert!(!sheet.rules[0].selectors[0].matches("p", None, &[]));
        assert!(!sheet.rules[4].selectors[0].matches("p", None, &[]));
        // The :first-child pseudo-class is stored on the last compound.
        let last = &sheet.rules[4].selectors[0];
        assert!(matches!(
            last.compounds[1].parts.last(),
            Some(SimpleSelector::PseudoClass(PseudoClass::FirstChild))
        ));
    }

    #[test]
    fn specificity_cases() {
        let spec = |css: &str| {
            let sheet = Stylesheet::parse(&format!("{css} {{ color: red; }}"));
            assert!(sheet.errors.is_empty(), "{css}: {:?}", sheet.errors);
            sheet.rules[0].selectors[0].specificity()
        };
        assert_eq!(spec("#a"), (1, 0, 0));
        assert_eq!(spec(".a.b"), (0, 2, 0));
        assert_eq!(spec("div"), (0, 0, 1));
        assert_eq!(spec("div.note#main"), (1, 1, 1));
        assert_eq!(spec("*"), (0, 0, 0));
        assert_eq!(spec("ul li a"), (0, 0, 3));
        assert_eq!(spec("a:hover"), (0, 1, 1));
        assert_eq!(spec("[href]"), (0, 1, 0));
        assert_eq!(spec("p::before"), (0, 0, 2));
        // :where() contributes zero; :is()/:not() take the max arg.
        assert_eq!(spec(":where(.a)"), (0, 0, 0));
        assert_eq!(spec(":where(#a)"), (0, 0, 0));
        assert_eq!(spec(":is(.a, #b)"), (1, 0, 0));
        assert_eq!(spec(":is(.a, .b)"), (0, 1, 0));
        assert_eq!(spec(":not(.a)"), (0, 1, 0));
        assert_eq!(spec("div:not(#a)"), (1, 0, 1));
        assert_eq!(spec(":nth-child(2n+1)"), (0, 1, 0));
        // Lexicographic: 256 classes do NOT beat one id.
        let many_classes = ".c".repeat(256);
        assert_eq!(spec(&many_classes), (0, 256, 0));
        assert!(spec(&many_classes) < spec("#a"));
    }

    #[test]
    fn attribute_selectors_parse() {
        let sheet = Stylesheet::parse(
            r#"[href] { color: red; } a[href^="https"] { color: green; }
               [data-x~="y" i] { color: blue; } [lang|="en"] { color: pink; }
               a[href$=".pdf" s] { color: yellow; } [title*="hi"] { color: purple; }"#,
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 6);
        let attr = |rule: usize| match &sheet.rules[rule].selectors[0].compounds[0].parts[..] {
            [SimpleSelector::Attribute { op, .. }] => *op,
            [_, SimpleSelector::Attribute { op, .. }] => *op,
            _ => panic!("rule {rule} has no attribute selector"),
        };
        assert_eq!(attr(0), AttrOp::Exists);
        assert_eq!(attr(1), AttrOp::Prefix);
        assert_eq!(attr(2), AttrOp::Includes);
        assert_eq!(attr(3), AttrOp::Dash);
        assert_eq!(attr(4), AttrOp::Suffix);
        assert_eq!(attr(5), AttrOp::Substring);
        // Case flags: `i` is insensitive, `s` is sensitive.
        let flag = |rule: usize| {
            let parts = &sheet.rules[rule].selectors[0].compounds[0].parts;
            match &parts[..] {
                [SimpleSelector::Attribute {
                    case_sensitive, ..
                }] => *case_sensitive,
                [_, SimpleSelector::Attribute {
                    case_sensitive, ..
                }] => *case_sensitive,
                _ => panic!("rule {rule}"),
            }
        };
        assert_eq!(flag(2), Some(false));
        assert_eq!(flag(4), Some(true));
    }

    #[test]
    fn pseudo_classes_and_elements_parse() {
        let sheet = Stylesheet::parse(
            "p::before { content: \"x\"; } a:hover { color: red; } \
             li:nth-child(2n+1) { color: green; } :root { color: blue; } \
             p:empty { color: pink; } div:has(> p) { color: yellow; } \
             a:any-link { color: purple; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 7);
        assert_eq!(
            sheet.rules[0].selectors[0].pseudo_element,
            Some(PseudoElement::Before)
        );
        assert!(matches!(
            sheet.rules[1].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::Hover)
        ));
        assert!(matches!(
            sheet.rules[2].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::NthChild(AnPlusB { a: 2, b: 1 }))
        ));
        // :has() is parsed, stored, and flagged as unimplemented.
        let has = &sheet.rules[5].selectors[0];
        assert!(has.has_unimplemented);
        assert!(!has.matches("div", None, &[]));
        assert!(matches!(
            sheet.rules[6].selectors[0].compounds[0].parts[1],
            SimpleSelector::PseudoClass(PseudoClass::AnyLink)
        ));
    }

    #[test]
    fn key_selector_picks_most_selective() {
        let sheet = Stylesheet::parse("div#main .note { color: red; } div p { color: blue; }");
        assert!(sheet.errors.is_empty());
        assert!(matches!(
            sheet.rules[0].selectors[0].key_selector(),
            Some(SimpleSelector::Class(c)) if c == "note"
        ));
        assert!(matches!(
            sheet.rules[1].selectors[0].key_selector(),
            Some(SimpleSelector::Type(t)) if t == "p"
        ));
    }

    #[test]
    fn one_bad_selector_kills_the_whole_list() {
        // CHANGED BEHAVIOUR (M2, CSS Syntax L3): previously the good
        // selectors were kept; now one invalid complex invalidates
        // the whole list and the rule is dropped.
        let sheet = Stylesheet::parse(
            "div, .bad[foo, p { color: red; } span { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);
        assert!(sheet.errors[0].contains("whole rule dropped"));
        assert!(sheet.rules[0].selectors[0].matches("span", None, &[]));

        let sheet = Stylesheet::parse("div, span:hover-ish { color: red; }");
        assert_eq!(sheet.rules.len(), 0);
        assert_eq!(sheet.errors.len(), 1);
    }

    // ---- R6: cascade data ----

    #[test]
    fn important_flag_parsed() {
        let sheet =
            Stylesheet::parse("p { color: red !important; margin: 0 !IMPORTANT; width: 1px; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert!(decl_entry(rule, "color").unwrap().important);
        assert!(decl_entry(rule, "margin-top").unwrap().important);
        assert!(!decl_entry(rule, "width").unwrap().important);
    }

    #[test]
    fn cascade_key_ordering() {
        let sheet = Stylesheet::parse(
            "p { color: red; } p { color: blue !important; }",
        );
        let (normal, important) = (&sheet.rules[0], &sheet.rules[1]);
        let normal_decl = &normal.declarations[0];
        let important_decl = &important.declarations[0];
        let kn = normal.sort_key(normal_decl, i32::MAX, false);
        let ki = important.sort_key(important_decl, i32::MAX, false);
        assert!(ki > kn);
        assert_eq!(kn.origin_rank, 2); // author normal
        assert_eq!(ki.origin_rank, 3); // author !important
        // Source order breaks ties between equal keys.
        let later = Rule {
            selectors: normal.selectors.clone(),
            declarations: vec![],
            origin: Origin::Author,
            layer: None,
            source_order: 99,
            media: None,
            supports: None,
        };
        assert_eq!(later.source_order, 99);
    }

    #[test]
    fn layer_statement_and_block_order() {
        let sheet = Stylesheet::parse(
            "@layer b, a; @layer a { p { color: red; } } @layer b { p { color: blue; } } \
             p { color: green; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        // First declaration fixes the order.
        assert_eq!(sheet.layers, vec!["b".to_string(), "a".to_string()]);
        assert_eq!(sheet.rules.len(), 3);
        assert_eq!(sheet.rules[0].layer, Some("a".to_string()));
        assert_eq!(sheet.rules[1].layer, Some("b".to_string()));
        assert_eq!(sheet.rules[2].layer, None);
        assert_eq!(sheet.layer_order(&Some("b".to_string())), 0);
        assert_eq!(sheet.layer_order(&Some("a".to_string())), 1);
        assert_eq!(sheet.layer_order(&None), i32::MAX);
    }

    #[test]
    fn import_recorded_and_late_import_ignored() {
        let sheet = Stylesheet::parse(
            r#"@import url("base.css") layer(foo) supports(display: grid) screen and (max-width: 600px);
               @import "theme.css";
               p { color: red; }"#,
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.imports.len(), 2);
        assert_eq!(sheet.imports[0].url, "base.css");
        assert_eq!(sheet.imports[0].layer, Some("foo".to_string()));
        assert_eq!(
            sheet.imports[0].supports,
            Some("display: grid".to_string())
        );
        assert_eq!(
            sheet.imports[0].media,
            Some("screen and (max-width: 600px)".to_string())
        );
        assert_eq!(sheet.imports[1].url, "theme.css");
        assert_eq!(sheet.imports[1].layer, None);

        let late = Stylesheet::parse("p { color: red; } @import url(\"x.css\");");
        assert!(late.imports.is_empty());
        assert_eq!(late.errors.len(), 1);
        assert!(late.errors[0].contains("@import ignored"));
    }

    #[test]
    fn charset_and_namespace_do_not_choke() {
        let sheet = Stylesheet::parse(
            "@charset \"UTF-8\"; @namespace url(http://www.w3.org/1999/xhtml); p { color: red; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        // @charset is silently ignored; @namespace is recorded, skipped.
        assert_eq!(sheet.errors.len(), 1);
        assert!(sheet.errors[0].contains("@namespace"));
    }

    // ---- R6/R7: @media / @supports / @font-face / @keyframes ----

    #[test]
    fn media_conditions_parse_and_eval() {
        let sheet = Stylesheet::parse(
            "@media screen and (min-width: 100px) { p { color: red; } } \
             @media (max-width: 700px), (min-width: 1200px) { p { color: blue; } }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.rules.len(), 2);
        let and = sheet.rules[0].media.as_ref().unwrap();
        assert!(matches!(and, MediaCondition::And(_)));
        assert_eq!(and.eval(50.0, 800.0), Some(false));
        assert_eq!(and.eval(200.0, 800.0), Some(true));
        let or = sheet.rules[1].media.as_ref().unwrap();
        assert!(matches!(or, MediaCondition::Or(_)));
        assert_eq!(or.eval(600.0, 800.0), Some(true));
        assert_eq!(or.eval(800.0, 800.0), Some(false));
        assert_eq!(or.eval(1300.0, 800.0), Some(true));
        // `em` in a media feature resolves against 16px.
        let em = MediaCondition::parse("(max-width: 40em)");
        assert_eq!(em.eval(640.0, 800.0), Some(true));
        assert_eq!(em.eval(641.0, 800.0), Some(false));
        // Unknown features evaluate to None, never true/false.
        assert_eq!(
            MediaCondition::parse("(pointer: fine)").eval(800.0, 600.0),
            None
        );
        assert_eq!(
            MediaCondition::parse("print").eval(800.0, 600.0),
            Some(false)
        );
    }

    #[test]
    fn supports_condition_best_effort_eval() {
        assert_eq!(
            SupportsCondition::parse("(display: grid)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("(display: grid) and (color: red)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("(foo: bar)").eval(),
            Some(false)
        );
        assert_eq!(
            SupportsCondition::parse("(display: grid) or (foo: bar)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("not (foo: bar)").eval(),
            Some(true)
        );
        assert_eq!(
            SupportsCondition::parse("selector(.foo)").eval(),
            None
        );
        // Rules inside @supports keep the raw condition attached.
        let sheet = Stylesheet::parse("@supports (display: grid) { p { color: red; } }");
        assert!(sheet.errors.is_empty());
        assert_eq!(
            sheet.rules[0].supports,
            Some("(display: grid)".to_string())
        );
    }

    #[test]
    fn font_face_and_keyframes_stored() {
        let sheet = Stylesheet::parse(
            "@font-face { font-family: Foo; src: url(foo.woff2); font-display: swap; \
             font-weight: 400; } \
             @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } } \
             p { color: red; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        assert_eq!(sheet.font_faces.len(), 1);
        assert_eq!(sheet.font_faces[0].family, Some("Foo".to_string()));
        assert_eq!(sheet.font_faces[0].src, Some("url(foo.woff2)".to_string()));
        assert_eq!(sheet.font_faces[0].display, Some("swap".to_string()));
        assert_eq!(sheet.font_faces[0].weight, Some("400".to_string()));
        assert_eq!(sheet.keyframes.len(), 1);
        assert_eq!(sheet.keyframes[0].name, "spin");
        assert!(sheet.keyframes[0].raw.contains("from"));
        assert_eq!(sheet.rules.len(), 1);
    }

    // ---- R7: UA stylesheet ----

    #[test]
    fn ua_stylesheet_parses_cleanly() {
        let ua = Stylesheet::ua();
        assert!(ua.errors.is_empty(), "errors: {:?}", ua.errors);
        assert!(!ua.rules.is_empty());
        // Every UA rule is origin Ua with source_order 0.. in order.
        for (i, rule) in ua.rules.iter().enumerate() {
            assert_eq!(rule.origin, Origin::Ua);
            assert_eq!(rule.source_order, i as u32);
        }
        let has = |tag: &str| {
            ua.rules.iter().any(|r| {
                r.selectors
                    .iter()
                    .any(|s| s.compounds.len() == 1 && s.compounds[0].parts.iter().any(|p| matches!(p, SimpleSelector::Type(t) if t == tag)))
            })
        };
        assert!(has("body"));
        assert!(has("h1"));
        assert!(has("a"));
        assert!(has("ul"));
        assert!(has("table"));
        let body = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.iter().any(|s| {
                    s.compounds.len() == 1
                        && s.compounds[0].parts == vec![SimpleSelector::Type("body".into())]
                })
            })
            .unwrap();
        assert_eq!(decl(body, "margin-top"), Some(&Value::Length(8.0)));
        let h1 = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.iter().any(|s| {
                    s.compounds.len() == 1
                        && s.compounds[0].parts == vec![SimpleSelector::Type("h1".into())]
                })
            })
            .unwrap();
        assert_eq!(decl(h1, "font-size"), Some(&Value::Em(2.0)));
        assert_eq!(decl(h1, "margin-top"), Some(&Value::Em(0.67)));
        let a = ua
            .rules
            .iter()
            .find(|r| {
                r.selectors.len() == 1
                    && r.selectors[0].compounds.len() == 1
                    && r.selectors[0].compounds[0].parts
                        == vec![SimpleSelector::Type("a".into())]
            })
            .unwrap();
        assert_eq!(decl(a, "color"), Some(&Value::Color(0, 0, 238)));
    }

    // ---- R13/R16: values & shorthands ----

    #[test]
    fn font_shorthand_expands() {
        let sheet = Stylesheet::parse("p { font: italic bold 12px/1.5 sans-serif; }");
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let rule = &sheet.rules[0];
        assert_eq!(
            decl(rule, "font-style"),
            Some(&Value::Keyword("italic".into()))
        );
        assert_eq!(
            decl(rule, "font-weight"),
            Some(&Value::Keyword("bold".into()))
        );
        assert_eq!(decl(rule, "font-size"), Some(&Value::Length(12.0)));
        assert_eq!(decl(rule, "line-height"), Some(&Value::Length(1.5)));
        assert_eq!(decl(rule, "font-family"), Some(&Value::Raw("sans-serif".into())));
        assert!(decl(rule, "font").is_none());

        let minimal = Stylesheet::parse("p { font: 16px serif; }");
        assert!(minimal.errors.is_empty());
        assert_eq!(
            decl(&minimal.rules[0], "font-size"),
            Some(&Value::Length(16.0))
        );
        assert_eq!(
            decl(&minimal.rules[0], "font-family"),
            Some(&Value::Raw("serif".into()))
        );
        assert!(decl(&minimal.rules[0], "line-height").is_none());

        // Missing size/family: invalid, dropped with an error.
        let bad = Stylesheet::parse("p { font: bold; color: red; }");
        assert_eq!(bad.rules[0].declarations.len(), 1);
        assert_eq!(bad.errors.len(), 1);
        assert_eq!(
            decl(&bad.rules[0], "color"),
            Some(&Value::Color(255, 0, 0))
        );
    }

    #[test]
    fn shorthand_with_em_expands() {
        let sheet = Stylesheet::parse("div { margin: 5em auto; padding: 2em; }");
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "margin-top"), Some(&Value::Em(5.0)));
        assert_eq!(
            decl(rule, "margin-right"),
            Some(&Value::Keyword("auto".into()))
        );
        assert_eq!(decl(rule, "padding-top"), Some(&Value::Em(2.0)));
        assert_eq!(decl(rule, "padding-left"), Some(&Value::Em(2.0)));
    }

    #[test]
    fn border_radius_and_box_shadow_parse() {
        let sheet = Stylesheet::parse(
            "div { border-radius: 4px 8px; box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02); }",
        );
        assert!(sheet.errors.is_empty());
        let rule = &sheet.rules[0];
        assert_eq!(
            decl(rule, "border-radius"),
            Some(&Value::BorderRadius(BorderRadius {
                top_left: CssLength::Px(4.0),
                top_right: CssLength::Px(8.0),
                bottom_right: CssLength::Px(4.0),
                bottom_left: CssLength::Px(8.0),
            }))
        );
        assert_eq!(
            decl(rule, "box-shadow"),
            Some(&Value::BoxShadow(vec![BoxShadow {
                inset: false,
                offset_x: CssLength::Px(2.0),
                offset_y: CssLength::Px(3.0),
                blur: CssLength::Px(7.0),
                spread: CssLength::Px(2.0),
                color: Some((0, 0, 0, 0.02)),
            }]))
        );
        let inset = Stylesheet::parse("div { box-shadow: inset 0 1px 2px #000; }");
        assert!(inset.errors.is_empty());
        match decl(&inset.rules[0], "box-shadow") {
            Some(Value::BoxShadow(shadows)) => {
                assert_eq!(shadows.len(), 1);
                assert!(shadows[0].inset);
                assert_eq!(shadows[0].color, Some((0, 0, 0, 1.0)));
            }
            other => panic!("unexpected box-shadow: {other:?}"),
        }
        // Unparseable shapes fall back to Raw, never an error.
        let bad = Stylesheet::parse("div { border-radius: bogus; box-shadow: bogus; }");
        assert!(bad.errors.is_empty());
        assert!(matches!(decl(&bad.rules[0], "border-radius"), Some(Value::Raw(_))));
        assert!(matches!(decl(&bad.rules[0], "box-shadow"), Some(Value::Raw(_))));
    }

    #[test]
    fn grid_and_flex_properties_accepted() {
        let sheet = Stylesheet::parse(
            "div { display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: auto; \
             gap: 10px; grid-column: 1 / 3; flex-direction: column; flex-wrap: wrap; }",
        );
        assert!(sheet.errors.is_empty(), "errors: {:?}", sheet.errors);
        let rule = &sheet.rules[0];
        assert_eq!(decl(rule, "display"), Some(&Value::Keyword("grid".into())));
        assert!(decl(rule, "grid-template-columns").is_some());
        assert!(decl(rule, "grid-template-rows").is_some());
        assert_eq!(decl(rule, "gap"), Some(&Value::Length(10.0)));
        assert!(decl(rule, "grid-column").is_some());
        assert_eq!(
            decl(rule, "flex-direction"),
            Some(&Value::Keyword("column".into()))
        );
        assert_eq!(
            decl(rule, "flex-wrap"),
            Some(&Value::Keyword("wrap".into()))
        );
    }

    // ---- error-recovery granularity ----

    #[test]
    fn error_recovery_granularity() {
        // Bad declaration: only that declaration is dropped.
        let sheet = Stylesheet::parse("p { color: red; bogus:; width: 10px; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.rules[0].declarations.len(), 2);
        assert_eq!(sheet.errors.len(), 1);

        // Missing semicolon merges two declarations: both are lost,
        // the rest of the rule survives.
        let sheet = Stylesheet::parse("p { color: red width: 10px; height: 5px; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            None,
            "merged declaration must be dropped"
        );
        assert_eq!(
            decl(&sheet.rules[0], "height"),
            Some(&Value::Length(5.0))
        );
        assert_eq!(sheet.errors.len(), 1);

        // Bad selector: the whole rule is dropped, other rules survive.
        let sheet = Stylesheet::parse("div > > p { color: red; } p { color: blue; }");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);

        // Unknown at-rule (statement and block): skipped, rest survives.
        let sheet = Stylesheet::parse(
            "@foo bar; p { color: red; } @bar { div { width: 1px; } } p { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 2);
        assert_eq!(sheet.errors.len(), 2);

        // Unclosed block: closed at EOF, the rule is still kept.
        let sheet = Stylesheet::parse("p { color: red; ");
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            Some(&Value::Color(255, 0, 0))
        );
        assert_eq!(sheet.errors.len(), 1);

        // @container is parsed-and-skipped with correct brace matching.
        let sheet = Stylesheet::parse(
            "@container (min-width: 400px) { p { color: red; } } p { color: blue; }",
        );
        assert_eq!(sheet.rules.len(), 1);
        assert_eq!(sheet.errors.len(), 1);
        assert_eq!(
            decl(&sheet.rules[0], "color"),
            Some(&Value::Color(0, 0, 255))
        );
    }
}
