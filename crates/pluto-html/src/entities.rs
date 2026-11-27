//! Named character references, generated at build time from the
//! vendored WHATWG `entities.json` (see `build.rs`). The table is
//! the source of truth; no count is asserted anywhere (published
//! sources disagree on the total).

include!(concat!(env!("OUT_DIR"), "/entities_table.rs"));

/// Look up a named reference by its name *without* the leading `&`
/// (e.g. `"amp;"` or the legacy `"amp"`). Returns the replacement
/// characters.
pub fn lookup(name: &str) -> Option<&'static str> {
    ENTITIES
        .binary_search_by(|(n, _)| n.cmp(&name))
        .ok()
        .map(|i| ENTITIES[i].1)
}

/// Windows-1252 replacements for numeric references 0x80..=0x9F
/// (WHATWG numeric character reference table). Entries that are
/// `None` keep their own code point.
const C1_REPLACEMENTS: [Option<char>; 32] = [
    Some('\u{20AC}'), // 0x80 EURO SIGN
    None,             // 0x81
    Some('\u{201A}'), // 0x82
    Some('\u{0192}'), // 0x83
    Some('\u{201E}'), // 0x84
    Some('\u{2026}'), // 0x85
    Some('\u{2020}'), // 0x86
    Some('\u{2021}'), // 0x87
    Some('\u{02C6}'), // 0x88
    Some('\u{2030}'), // 0x89
    Some('\u{0160}'), // 0x8A
    Some('\u{2039}'), // 0x8B
    Some('\u{0152}'), // 0x8C
    None,             // 0x8D
    Some('\u{017D}'), // 0x8E
    None,             // 0x8F
    None,             // 0x90
    Some('\u{2018}'), // 0x91
    Some('\u{2019}'), // 0x92
    Some('\u{201C}'), // 0x93
    Some('\u{201D}'), // 0x94
    Some('\u{2022}'), // 0x95
    Some('\u{2013}'), // 0x96
    Some('\u{2014}'), // 0x97
    Some('\u{02DC}'), // 0x98
    Some('\u{2122}'), // 0x99
    Some('\u{0161}'), // 0x9A
    Some('\u{203A}'), // 0x9B
    Some('\u{0153}'), // 0x9C
    None,             // 0x9D
    Some('\u{017E}'), // 0x9E
    Some('\u{0178}'), // 0x9F
];

/// Map a numeric character reference value per the WHATWG table:
/// NUL, surrogates and values > U+10FFFF become U+FFFD; C1 controls
/// get their Windows-1252 substitution; anything else (including
/// noncharacters) is returned as-is.
pub fn numeric_char(value: u32) -> char {
    if value == 0 || value > 0x10FFFF || (0xD800..0xE000).contains(&value) {
        return '\u{FFFD}';
    }
    if (0x80..=0x9F).contains(&value)
        && let Some(c) = C1_REPLACEMENTS[(value - 0x80) as usize] {
            return c;
        }
    char::from_u32(value).unwrap_or('\u{FFFD}')
}
