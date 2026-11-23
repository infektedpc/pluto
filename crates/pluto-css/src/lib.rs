//! Parsing & cascade (layer 3): winnow CSS parser.
//!
//! Zero-copy CSS tokenization with winnow, then a cascade that turns
//! matched rules into computed styles for the layout tree. The tokenizer
//! and cascade land in the css round; this file defines the types they
//! produce.

/// One `property: value;` pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub property: String,
    pub value: String,
}

/// One `selector { declarations }` rule.
#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub selectors: Vec<String>,
    pub declarations: Vec<Declaration>,
}

/// A parsed stylesheet: rules in source order (cascade order is applied
/// by the styler, not stored here).
#[derive(Debug, Clone, Default)]
pub struct Stylesheet {
    pub rules: Vec<Rule>,
}

impl Stylesheet {
    /// Parse CSS text. Implemented in the css round (winnow tokenizer).
    pub fn parse(_css: &str) -> Self {
        Self::default()
    }
}
