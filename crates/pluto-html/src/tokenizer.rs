//! Pluto's own WHATWG HTML tokenizer: a character-driven state
//! machine emitting `Doctype / StartTag / EndTag / Comment /
//! Character / EOF` tokens.
//!
//! Incremental by design: the tokenizer holds a character buffer
//! and a position; [`Tokenizer::feed_str`] appends input and
//! [`Tokenizer::next_token`] returns `None` when the buffer ends
//! mid-token and more input may still arrive (call
//! [`Tokenizer::set_eof`] at end of input, after which a truncated
//! token is finished per spec instead of waiting). Parse errors are
//! counted, never fatal.
//!
//! Attribute rules follow the spec exactly: unquoted values may
//! contain `/` (it is only the self-closing marker *between*
//! attributes, never inside a value), and may not contain
//! `" ' = < > \`` (those are parse errors, the character is still
//! consumed into the value); `&` starts a character reference in
//! values and in text. A self-closing `/>` on a non-void HTML
//! element is a parse error and is ignored by the tree builder (the
//! flag is carried on the token).

use std::collections::VecDeque;

use crate::entities;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagToken {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub self_closing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Doctype {
        name: Option<String>,
        public_id: Option<String>,
        system_id: Option<String>,
        force_quirks: bool,
    },
    StartTag(TagToken),
    EndTag(TagToken),
    Comment(String),
    Character(String),
    Eof,
}

/// Raw-text / RCDATA content model the tokenizer switches to after
/// a start tag. Set by the tree builder (via the tokenizer's
/// `pending_content` after a start tag is emitted); `noscript`
/// honours [`Tokenizer::scripting_enabled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentModel {
    Rcdata,
    RawText,
    ScriptData,
    PlainText,
}

enum EndTagCheck {
    /// Not an appropriate end tag.
    No,
    /// The tag name matches so far but the delimiter character has
    /// not arrived yet (more input may come).
    NeedMore,
    /// Appropriate end tag (tag name).
    Yes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Data,
    Rcdata,
    RawText,
    ScriptData,
    ScriptDataEscaped,
    ScriptDataDoubleEscaped,
    PlainText,
}

pub struct Tokenizer {
    input: Vec<char>,
    pos: usize,
    eof: bool,
    state: State,
    /// Tag whose end tag terminates the current raw/RCDATA/script
    /// content (lowercase), if any.
    raw_tag: Option<String>,
    /// Content model requested by the most recently emitted start
    /// tag; applied on the next `next_token` call.
    pending_content: Option<ContentModel>,
    /// A CR held back from the last `feed_str` (may be half of a
    /// CRLF split across chunks; resolved on the next feed / EOF).
    pending_cr: bool,
    /// Whether the document is parsed with scripting enabled
    /// (controls `<noscript>` raw-text handling).
    pub scripting_enabled: bool,
    /// Whether `<![CDATA[` sections are character data (true only
    /// in foreign content, set by the tree builder before each pull).
    pub cdata_allowed: bool,
    queued: VecDeque<Token>,
    eof_emitted: bool,
    pub parse_errors: usize,
    pub first_errors: Vec<String>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            pos: 0,
            eof: false,
            state: State::Data,
            raw_tag: None,
            pending_content: None,
            pending_cr: false,
            scripting_enabled: true,
            cdata_allowed: false,
            queued: VecDeque::new(),
            eof_emitted: false,
            parse_errors: 0,
            first_errors: Vec::new(),
        }
    }

    pub fn with_scripting(scripting_enabled: bool) -> Self {
        let mut t = Self::new();
        t.scripting_enabled = scripting_enabled;
        t
    }

    pub fn set_scripting(&mut self, scripting_enabled: bool) {
        self.scripting_enabled = scripting_enabled;
    }

    /// Append more input characters.
    pub fn feed_str(&mut self, text: &str) {
        // Spec 13.2.3 (input stream): CRLF and lone CR normalise to
        // LF. A CR at the end of a chunk is held back: the next
        // chunk may start with LF (one newline, not two).
        let mut chars = text.chars().peekable();
        if self.pending_cr {
            self.pending_cr = false;
            match chars.peek() {
                Some('\n') => {
                    chars.next(); // CRLF split across chunks
                    self.input.push('\n');
                }
                _ => self.input.push('\n'),
            }
        }
        let mut prev_cr = false;
        for c in chars {
            if c == '\r' {
                if prev_cr {
                    self.input.push('\n');
                }
                prev_cr = true;
            } else if c == '\n' {
                // Covers CRLF (prev_cr consumed) and lone LF.
                self.input.push('\n');
                prev_cr = false;
            } else {
                if prev_cr {
                    self.input.push('\n');
                    prev_cr = false;
                }
                self.input.push(c);
            }
        }
        if prev_cr {
            self.pending_cr = true;
        }
    }

    /// No more input will arrive; truncated tokens finish per spec.
    pub fn set_eof(&mut self) {
        if self.pending_cr {
            self.pending_cr = false;
            self.input.push('\n');
        }
        self.eof = true;
    }

    pub fn is_eof(&self) -> bool {
        self.eof && self.pos >= self.input.len() && self.queued.is_empty()
    }

    /// Current read position in the input buffer (index of the next
    /// unconsumed character). The parser's insertion point at a
    /// script pause: everything from here on is still network input
    /// that has not been tokenised yet.
    pub fn input_pos(&self) -> usize {
        self.pos
    }

    /// Insert `text` into the input buffer at char index `at`
    /// (clamped to the buffer end), ahead of any buffered input
    /// not yet consumed. This is the `document.write` insertion
    /// point: written markup is tokenised before the network bytes
    /// that follow it. Returns the number of characters inserted
    /// after input-stream normalisation (CRLF / CR → LF, as in
    /// [`Tokenizer::feed_str`]).
    ///
    /// Caveat: a trailing CR in `text` normalises to LF at once,
    /// so a CRLF split with its LF already sitting in the network
    /// buffer would yield two newlines instead of one — a
    /// pathological `document.write` ending in a bare CR.
    pub fn insert_chars_at(&mut self, at: usize, text: &str) -> usize {
        let mut chars = text.chars().peekable();
        let mut norm = String::new();
        while let Some(c) = chars.next() {
            if c == '\r' {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                norm.push('\n');
            } else {
                norm.push(c);
            }
        }
        let at = at.min(self.input.len());
        let inserted: Vec<char> = norm.chars().collect();
        let n = inserted.len();
        self.input.splice(at..at, inserted);
        n
    }

    /// Seed the content model for a fragment parse from the
    /// context element's tag (spec: the tokenizer starts in the
    /// context element's content model, e.g. RCDATA for <title>).
    pub fn set_fragment_content_model(&mut self, tag_name: &str) {
        // The fragment algorithm seeds the tokenizer's content model
        // from the context element's tag name — but it does NOT set
        // the "last start tag": in a fragment the tokenizer has no
        // last start tag, so `</script>` (etc.) inside the fragment
        // content is text, never an appropriate end tag (tests4 #8:
        // a script-context fragment keeps `</script>` as text). The
        // shared set_pending_content also records the raw tag name,
        // which would make the tokenizer close the content early.
        self.pending_content = match tag_name {
            "title" | "textarea" => Some(ContentModel::Rcdata),
            "style" | "xmp" | "iframe" | "noembed" | "noframes" => {
                Some(ContentModel::RawText)
            }
            "script" => Some(ContentModel::ScriptData),
            "plaintext" => Some(ContentModel::PlainText),
            "noscript" if self.scripting_enabled => Some(ContentModel::RawText),
            _ => None,
        };
    }

    /// The tree builder calls this after processing a start tag when
    /// the element must NOT switch the tokenizer to raw/RCDATA
    /// (e.g. a foreign-content element whose name collides with a
    /// raw-text element).
    pub fn cancel_pending_content(&mut self) {
        self.pending_content = None;
    }

    /// Emit the single EOF token (exactly once).
    fn eof_token(&mut self) -> Option<Token> {
        if self.eof_emitted {
            None
        } else {
            self.eof_emitted = true;
            Some(Token::Eof)
        }
    }

    fn error(&mut self, msg: &str) {
        self.parse_errors += 1;
        if self.first_errors.len() < 10 {
            self.first_errors.push(msg.to_string());
        }
    }

    // -- input primitives -------------------------------------------------

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<char> {
        self.input.get(self.pos + off).copied()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    /// True when the buffer is exhausted and more input may arrive.
    fn need_more(&self) -> bool {
        self.pos >= self.input.len() && !self.eof
    }

    fn starts_with_ignore_case(&self, s: &str) -> bool {
        s.chars().enumerate().all(|(i, c)| {
            self.peek_at(i).map(|p| p.eq_ignore_ascii_case(&c)).unwrap_or(false)
        })
    }

    fn consume_str_ignore_case(&mut self, s: &str) {
        for _ in s.chars() {
            self.pos += 1;
        }
    }

    // -- character references ---------------------------------------------

    /// Consume a character reference at the current position (which
    /// must be `&`). Returns the decoded string, or `None` if this
    /// is not a reference (caller emits a literal `&`). On
    /// incomplete input at end-of-buffer without EOF, returns
    /// `Some("")` with the position left at `&` — signalled by the
    /// caller via [`Tokenizer::need_more`] after this returns empty
    /// and the position did not advance... (simpler: this function
    /// never consumes on failure; `None` means "literal &".)
    ///
    /// `in_attr`: the attribute-value legacy rule — a semicolon-less
    /// named reference followed by `=` or an ASCII alphanumeric is
    /// NOT a reference in attributes.
    fn consume_char_ref(&mut self, in_attr: bool) -> Option<String> {
        // self.input[self.pos] == '&'
        let save = self.pos;
        // Need at least one more char, or EOF.
        if self.peek_at(1).is_none() {
            if self.need_more() {
                // Incomplete: do not consume; caller will retry after
                // more input arrives. Signal via a sentinel: leave
                // pos at save and return None, but the caller checks
                // need_more() before treating it as literal.
                return None;
            }
            return None;
        }
        let c1 = self.peek_at(1).unwrap();
        if c1 == '#' {
            return self.consume_numeric_ref(save);
        }
        if !c1.is_ascii_alphanumeric() {
            return None;
        }
        // Longest-match over the entity table: collect the maximal
        // run of ASCII alphanumerics, then try with/without ';'.
        let mut end = self.pos + 1;
        while end < self.input.len() && self.input[end].is_ascii_alphanumeric() {
            end += 1;
        }
        let has_more_input = end >= self.input.len() && !self.eof;
        let name: String = self.input[self.pos + 1..end].iter().collect();
        // Try the longest name with ';' first.
        let mut best: Option<(String, bool)> = None; // (value, had_semicolon)
        // Candidate names, longest first: full name + ';' (if the
        // next char is ';'), then progressively shorter prefixes.
        let mut candidates: Vec<String> = Vec::new();
        if self.input.get(end) == Some(&';') {
            candidates.push(format!("{name};"));
        }
        for len in (1..=name.len()).rev() {
            candidates.push(name[..len].to_string());
        }
        // Also try the full name + ';' even when ';' is not present
        // yet but input may be incomplete — handled by has_more_input.
        if has_more_input && !candidates.iter().any(|c| c.ends_with(';')) {
            // The ';' might still arrive; wait.
            self.pos = save;
            return Some(String::new());
        }
        for cand in &candidates {
            if let Some(v) = entities::lookup(cand) {
                let had_semi = cand.ends_with(';');
                best = Some((v.to_string(), had_semi));
                // Longest-first ordering already gives the longest
                // match; but a ';' form beats a shorter no-';' form
                // only if it is at least as long — candidates are
                // ordered full+';' first, then by decreasing length,
                // which is the spec's longest-match order.
                break;
            }
        }
        match best {
            Some((value, had_semi)) => {
                if !had_semi && in_attr {
                    // The character after the matched name decides.
                    let matched_len = if value.is_empty() { 0 } else { 0 };
                    let _ = matched_len;
                    // Find how many name chars were consumed: the
                    // candidate that matched.
                    let consumed = candidates
                        .iter()
                        .find(|c| entities::lookup(c).is_some())
                        .map(|c| c.len())
                        .unwrap_or(0);
                    let after = self.input.get(self.pos + 1 + consumed).copied();
                    if let Some(a) = after {
                        if a == '=' || a.is_ascii_alphanumeric() {
                            return None;
                        }
                    }
                }
                let consumed = candidates
                    .iter()
                    .find(|c| entities::lookup(c).is_some())
                    .map(|c| c.len())
                    .unwrap_or(0);
                self.pos += 1 + consumed;
                Some(value)
            }
            None => {
                self.pos = save;
                None
            }
        }
    }

    fn consume_numeric_ref(&mut self, save: usize) -> Option<String> {
        // At '#'; next char decides decimal vs hex.
        let c2 = self.peek_at(2);
        let (hex, digits_start) = match c2 {
            Some(c) if c == 'x' || c == 'X' => (true, self.pos + 3),
            Some(c) if c.is_ascii_digit() => (false, self.pos + 2),
            _ => return None,
        };
        let mut end = digits_start;
        while end < self.input.len() {
            let c = self.input[end];
            if (hex && c.is_ascii_hexdigit()) || (!hex && c.is_ascii_digit()) {
                end += 1;
            } else {
                break;
            }
        }
        if end == digits_start {
            self.pos = save;
            return None;
        }
        if end >= self.input.len() && !self.eof {
            // Digits may continue; wait for more input.
            self.pos = save;
            return Some(String::new());
        }
        let digits: String = self.input[digits_start..end].iter().collect();
        let has_semi = self.input.get(end) == Some(&';');
        let value = if hex {
            u32::from_str_radix(&digits, 16).unwrap_or(0x110000)
        } else {
            u32::from_str_radix(&digits, 10).unwrap_or(0x110000)
        };
        if !has_semi {
            self.error("numeric character reference without semicolon");
        }
        self.pos = end + if has_semi { 1 } else { 0 };
        Some(entities::numeric_char(value).to_string())
    }

    /// Decode character references inside a raw string slice (used
    /// for attribute values already collected — not used on the hot
    /// path; the tokenizer decodes inline instead).
    #[allow(dead_code)]
    fn decode_refs_in(&mut self, s: &str, in_attr: bool) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '&' {
                out.push(c);
                continue;
            }
            // Re-run the reference logic on the remaining string.
            let rest: String = chars.clone().collect();
            let mut sub = Tokenizer::new();
            sub.input = format!("&{rest}").chars().collect();
            sub.eof = true;
            match sub.consume_char_ref(in_attr) {
                Some(v) => {
                    out.push_str(&v);
                    let consumed = sub.pos;
                    for _ in 0..consumed.saturating_sub(1) {
                        chars.next();
                    }
                }
                None => out.push('&'),
            }
        }
        out
    }

    // -- public pull API ---------------------------------------------------

    /// Next token, or `None` when the buffer ends mid-token and more
    /// input may still arrive. After [`Tokenizer::set_eof`], this
    /// returns `Some(Token::Eof)` at end of input and never `None`
    /// for lack of input.
    pub fn next_token(&mut self) -> Option<Token> {
        if let Some(t) = self.queued.pop_front() {
            return Some(t);
        }
        // Transactional pull: if the tokeniser runs out of input
        // mid-token, roll the whole attempt back (position, state,
        // pending content model, error count) so the next feed
        // re-parses the token from its start. Without this, a token
        // split across chunks would resume in the wrong state.
        let save_pos = self.pos;
        let save_state = self.state;
        let save_raw = self.raw_tag.clone();
        let save_pending = self.pending_content;
        let save_errors = self.parse_errors;
        let save_first_len = self.first_errors.len();
        if let Some(cm) = self.pending_content.take() {
            match cm {
                ContentModel::Rcdata => self.state = State::Rcdata,
                ContentModel::RawText => self.state = State::RawText,
                ContentModel::ScriptData => self.state = State::ScriptData,
                ContentModel::PlainText => self.state = State::PlainText,
            }
        }
        if self.pos >= self.input.len() {
            if self.eof {
                self.state = State::Data;
                return self.eof_token();
            }
            self.pending_content = save_pending;
            self.state = save_state;
            return None;
        }
        let result = match self.state {
            State::Data => self.data_token(),
            State::Rcdata => self.rcdata_token(),
            State::RawText => self.raw_text_token(),
            State::ScriptData => self.script_data_token(false),
            State::ScriptDataEscaped => self.script_data_token(true),
            State::ScriptDataDoubleEscaped => self.script_double_escaped_token(),
            State::PlainText => self.plain_text_token(),
        };
        if result.is_none() {
            self.pos = save_pos;
            self.state = save_state;
            self.raw_tag = save_raw;
            self.pending_content = save_pending;
            self.parse_errors = save_errors;
            self.first_errors.truncate(save_first_len);
        }
        result
    }

    // -- data state ----------------------------------------------------------

    fn data_token(&mut self) -> Option<Token> {
        let mut text = String::new();
        loop {
            let Some(c) = self.peek() else {
                if self.eof {
                    break;
                }
                if text.is_empty() {
                    return None;
                }
                break;
            };
            match c {
                '&' => {
                    let save = self.pos;
                    match self.consume_char_ref(false) {
                        Some(v) => {
                            if v.is_empty() {
                                // Incomplete reference at buffer end;
                                // wait for more input.
                                if text.is_empty() {
                                    return None;
                                }
                                break;
                            }
                            text.push_str(&v);
                        }
                        None => {
                            if self.pos == save && self.need_more() && self.peek_at(1).is_none() {
                                if text.is_empty() {
                                    return None;
                                }
                                break;
                            }
                            text.push('&');
                            self.pos += 1;
                        }
                    }
                }
                '<' => {
                    if text.is_empty() {
                        return self.tag_open_token();
                    }
                    break;
                }
                '\0' => {
                    // Data state: parse error, emit U+0000 as-is;
                    // the tree builder drops NULs (in-body rule).
                    self.error("null character in data");
                    text.push('\0');
                    self.pos += 1;
                }
                _ => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        if text.is_empty() {
            if self.eof {
                Some(Token::Eof)
            } else {
                None
            }
        } else {
            Some(Token::Character(text))
        }
    }

    /// At '<' in data state.
    fn tag_open_token(&mut self) -> Option<Token> {
        // self.input[self.pos] == '<'
        match self.peek_at(1) {
            None => {
                if self.eof {
                    self.pos += 1;
                    Some(Token::Character("<".to_string()))
                } else {
                    None
                }
            }
            Some(c) if c.is_ascii_alphabetic() => {
                self.pos += 1; // consume '<'
                self.parse_tag(false)
            }
            Some('/') => {
                self.pos += 1; // consume '<'
                self.end_tag_open()
            }
            Some('!') => {
                self.pos += 2; // consume '<!'
                self.markup_declaration_open()
            }
            Some('?') => {
                // Spec 13.2.5.72 (processing instruction open): a
                // letter or '_' after '?' starts a PI target, and
                // an unterminated PI is dropped at EOF; a bare `<?`
                // at EOF is dropped the same way. Any other
                // character falls back to a bogus comment, which
                // still emits at EOF (e.g. `<?#` -> `<!-- ?# -->`).
                // A target starting with "xml" (any case) is
                // reserved, so `<?xml version` is a bogus comment,
                // not a PI, and is kept (comments01 #13); a valid
                // target such as `<?start` is dropped at EOF.
                match self.peek_at(2) {
                    None => {
                        if self.eof {
                            return self.eof_token();
                        }
                        None
                    }
                    Some(c) => {
                        let mut pi = c.is_ascii_alphabetic() || c == '_';
                        if pi {
                            // Scan the target; "xml" (any case) at
                            // its start makes it a bogus comment.
                            let mut target = String::new();
                            let mut k = 2;
                            while let Some(ch) = self.peek_at(k) {
                                if ch.is_ascii_alphanumeric()
                                    || matches!(ch, '_' | '-' | '.' | ':')
                                {
                                    target.push(ch);
                                    k += 1;
                                } else {
                                    break;
                                }
                            }
                            if target.to_ascii_lowercase().starts_with("xml") {
                                pi = false;
                            }
                        }
                        self.pos += 1; // consume '<', '?' stays in comment data
                        self.bogus_comment(true, "", pi)
                    }
                }
            }
            Some(_) => {
                self.pos += 1;
                Some(Token::Character("<".to_string()))
            }
        }
    }

    fn end_tag_open(&mut self) -> Option<Token> {
        // At '/' (after '<' consumed).
        match self.peek_at(1) {
            None => {
                if self.eof {
                    // "</" at EOF: literal text "</".
                    self.pos += 1;
                    Some(Token::Character("</".to_string()))
                } else {
                    None
                }
            }
            Some(c) if c.is_ascii_alphabetic() => {
                self.pos += 1; // consume '/'
                self.parse_tag(true)
            }
            Some('>') => {
                self.error("empty end tag");
                self.pos += 2; // consume '/>'
                self.data_token()
            }
            Some(_) => {
                self.error("bogus end tag treated as comment");
                // Bogus comment: the '/' is consumed by the
                // end-tag-open state, so the data starts with the
                // character after it (spec 13.2.5.7: reconsume in
                // the bogus comment state).
                self.pos += 1; // consume '/'
                self.bogus_comment(false, "", false)
            }
        }
    }

    // -- tag parsing ---------------------------------------------------------

    fn parse_tag(&mut self, is_end: bool) -> Option<Token> {
        // Tag name: consume ASCII alphanumerics (lowercased).
        let mut name = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        // EOF in tag name state: eof-in-tag — drop
                        // the tag and emit EOF.
                        return self.eof_token();
                    }
                    return None;
                }
                Some('\0') => {
                    // Spec 13.2.5.8: NUL in a tag name is a parse
                    // error, replaced by U+FFFD.
                    self.error("null in tag name");
                    name.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c)
                    if !c.is_ascii_whitespace() && c != '/' && c != '>' =>
                {
                    // Tag names accept anything else (letters,
                    // digits, '-', ':', '.', etc.); ASCII uppercase
                    // is lowercased.
                    name.push(c.to_ascii_lowercase());
                    self.pos += 1;
                }
                _ => break,
            }
        }
        if name.is_empty() {
            // Should not happen (caller checked first char).
            return self.data_token();
        }
        let mut tag = TagToken {
            name: name.clone(),
            attrs: Vec::new(),
            self_closing: false,
        };
        // Attribute loop.
        loop {
            // Skip whitespace.
            loop {
                match self.peek() {
                    Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                    None => {
                        if self.eof {
                            // EOF in before-attribute-name state:
                            // eof-in-tag — drop the tag.
                            return self.eof_token();
                        }
                        return None;
                    }
                    _ => break,
                }
            }
            match self.peek() {
                None => {
                    if self.eof {
                        return self.eof_token();
                    }
                    return None;
                }
                Some('>') => {
                    self.pos += 1;
                    break;
                }
                Some('/') => {
                    if self.peek_at(1) == Some('>') {
                        self.pos += 2;
                        if is_end {
                            self.error("self-closing flag on end tag");
                        } else {
                            tag.self_closing = true;
                        }
                        break;
                    } else if self.peek_at(1).is_none() && !self.eof {
                        return None;
                    } else {
                        self.error("unexpected solidus in tag");
                        self.pos += 1;
                        // Reconsume in before-attribute-name state.
                        continue;
                    }
                }
                Some('=') => {
                    // Attribute name starting with '=': parse error,
                    // name includes '='.
                    self.error("attribute name starts with =");
                    match self.parse_attribute(&mut tag, true) {
                        None => return None,
                        Some(()) => continue,
                    }
                }
                Some(_) => match self.parse_attribute(&mut tag, false) {
                    None => return None,
                    Some(()) => continue,
                },
            }
        }
        if is_end {
            if tag.self_closing {
                // Already reported.
            }
            Some(Token::EndTag(tag))
        } else {
            self.set_pending_content(&tag);
            Some(Token::StartTag(tag))
        }
    }

    /// Parse one attribute (name + optional value) into `tag`.
    /// `leading_eq`: the name starts with an '=' already seen.
    fn parse_attribute(&mut self, tag: &mut TagToken, leading_eq: bool) -> Option<()> {
        let mut name = String::new();
        if leading_eq {
            name.push('=');
            self.pos += 1;
        }
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    return None;
                }
                Some(c) if c.is_ascii_whitespace() || c == '/' || c == '>' => break,
                Some('=') => break,
                Some(c) if c == '"' || c == '\'' || c == '<' => {
                    self.error("unexpected character in attribute name");
                    name.push(c.to_ascii_lowercase());
                    self.pos += 1;
                }
                Some(c) => {
                    name.push(c.to_ascii_lowercase());
                    self.pos += 1;
                }
            }
        }
        // Skip whitespace before '='.
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                None => {
                    if self.eof {
                        break;
                    }
                    return None;
                }
                _ => break,
            }
        }
        let mut value = String::new();
        if self.peek() == Some('=') {
            self.pos += 1;
            // Skip whitespace after '='.
            loop {
                match self.peek() {
                    Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                    None => {
                        if self.eof {
                            break;
                        }
                        return None;
                    }
                    _ => break,
                }
            }
            match self.peek() {
                Some('"') => {
                    self.pos += 1;
                    loop {
                        match self.peek() {
                            None => {
                                if self.eof {
                                    break;
                                }
                                return None;
                            }
                            Some('"') => {
                                self.pos += 1;
                                break;
                            }
                            Some('&') => match self.consume_char_ref(true) {
                                Some(v) => {
                                    if v.is_empty() {
                                        return None;
                                    }
                                    value.push_str(&v);
                                }
                                None => {
                                    value.push('&');
                                    self.pos += 1;
                                }
                            },
                            Some('\0') => {
                                self.error("null in attribute value");
                                value.push('\u{FFFD}');
                                self.pos += 1;
                            }
                            Some(c) => {
                                value.push(c);
                                self.pos += 1;
                            }
                        }
                    }
                }
                Some('\'') => {
                    self.pos += 1;
                    loop {
                        match self.peek() {
                            None => {
                                if self.eof {
                                    break;
                                }
                                return None;
                            }
                            Some('\'') => {
                                self.pos += 1;
                                break;
                            }
                            Some('&') => match self.consume_char_ref(true) {
                                Some(v) => {
                                    if v.is_empty() {
                                        return None;
                                    }
                                    value.push_str(&v);
                                }
                                None => {
                                    value.push('&');
                                    self.pos += 1;
                                }
                            },
                            Some('\0') => {
                                self.error("null in attribute value");
                                value.push('\u{FFFD}');
                                self.pos += 1;
                            }
                            Some(c) => {
                                value.push(c);
                                self.pos += 1;
                            }
                        }
                    }
                }
                Some('>') => {
                    self.error("missing attribute value");
                    // Value stays empty; '>' handled by tag loop.
                }
                _ => {
                    // Unquoted value: '/' is a value character; the
                    // self-closing marker is only recognised between
                    // attributes, never here.
                    loop {
                        match self.peek() {
                            None => {
                                if self.eof {
                                    break;
                                }
                                return None;
                            }
                            Some(c) if c.is_ascii_whitespace() || c == '>' => break,
                            Some('&') => match self.consume_char_ref(true) {
                                Some(v) => {
                                    if v.is_empty() {
                                        return None;
                                    }
                                    value.push_str(&v);
                                }
                                None => {
                                    value.push('&');
                                    self.pos += 1;
                                }
                            },
                            Some('\0') => {
                                self.error("null in attribute value");
                                value.push('\u{FFFD}');
                                self.pos += 1;
                            }
                            Some(c) if c == '"' || c == '\'' || c == '=' || c == '<' || c == '`' => {
                                self.error("unexpected character in unquoted attribute value");
                                value.push(c);
                                self.pos += 1;
                            }
                            Some(c) => {
                                value.push(c);
                                self.pos += 1;
                            }
                        }
                    }
                }
            }
        }
        // First duplicate wins (spec).
        if !tag.attrs.iter().any(|(n, _)| *n == name) {
            tag.attrs.push((name, value));
        }
        Some(())
    }

    /// Decide the content model after a start tag.
    fn set_pending_content(&mut self, tag: &TagToken) {
        self.pending_content = match tag.name.as_str() {
            "title" | "textarea" => Some(ContentModel::Rcdata),
            "style" | "xmp" | "iframe" | "noembed" | "noframes" => {
                Some(ContentModel::RawText)
            }
            "script" => Some(ContentModel::ScriptData),
            "plaintext" => Some(ContentModel::PlainText),
            "noscript" if self.scripting_enabled => Some(ContentModel::RawText),
            _ => None,
        };
        if self.pending_content.is_some() {
            self.raw_tag = Some(tag.name.clone());
        }
    }

    // -- markup declaration / comments / doctype ------------------------------

    fn markup_declaration_open(&mut self) -> Option<Token> {
        // At the char after '<!'. If the remaining buffer is a strict
        // prefix of a known opener and more input may arrive, wait.
        if !self.eof {
            let rest: String = self.input[self.pos..].iter().collect();
            for opener in ["--", "doctype", "DOCTYPE", "[CDATA["] {
                if opener.starts_with(&rest) && rest.len() < opener.len() {
                    return None;
                }
            }
        }
        if self.starts_with_ignore_case("--") {
            // Need 2 chars; if buffer ends, wait.
            if self.peek_at(1).is_none() && !self.eof {
                return None;
            }
            self.pos += 2;
            return self.comment_token();
        }
        if self.starts_with_ignore_case("doctype") {
            if self.peek_at(6).is_none() && !self.eof {
                // "doctype" may be split across chunks; wait unless
                // the available prefix already mismatches (handled
                // by starts_with returning false on missing chars).
                return None;
            }
            self.consume_str_ignore_case("doctype");
            return self.doctype_token();
        }
        if self.starts_with_ignore_case("[cdata[") {
            if self.peek_at(6).is_none() && !self.eof {
                return None;
            }
            self.consume_str_ignore_case("[CDATA[");
            if self.cdata_allowed {
                return self.cdata_section();
            }
            self.error("CDATA section in HTML content treated as bogus comment");
            // Spec 13.2.5.42: the comment token's data starts with
            // "[CDATA["; the rest follows in the bogus comment state.
            return self.bogus_comment(false, "[CDATA[", false);
        }
        // Anything else: bogus comment.
        self.error("bogus markup declaration treated as comment");
        self.bogus_comment(false, "", false)
    }

    fn cdata_section(&mut self) -> Option<Token> {
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    if text.is_empty() {
                        return None;
                    }
                    break;
                }
                Some(']') if self.peek_at(1) == Some(']') && self.peek_at(2) == Some('>') => {
                    self.pos += 3;
                    break;
                }
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        Some(Token::Character(text))
    }

    /// Bogus comment: consume until '>' (or EOF). If `from_question`,
    /// the data starts with '?' (the `<?` that opened it). `initial`
    /// is data already consumed before this state (e.g. "[CDATA[").
    fn bogus_comment(
        &mut self,
        from_question: bool,
        initial: &str,
        drop_at_eof: bool,
    ) -> Option<Token> {
        let mut data = initial.to_string();
        if from_question {
            data.push('?');
            self.pos += 1; // consume '?'
        }
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        // An unterminated processing instruction
                        // (a `<?` followed by a valid PI target
                        // start) is dropped entirely at EOF (spec
                        // 13.2.5.72-76: EOF in a PI state emits
                        // nothing). Any other bogus comment still
                        // emits at EOF.
                        if drop_at_eof {
                            return self.eof_token();
                        }
                        break;
                    }
                    if data.is_empty() && !from_question {
                        return None;
                    }
                    break;
                }
                Some('>') => {
                    self.pos += 1;
                    break;
                }
                Some('\0') => {
                    data.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c) => {
                    data.push(c);
                    self.pos += 1;
                }
            }
        }
        Some(Token::Comment(data))
    }

    fn comment_token(&mut self) -> Option<Token> {
        // After '<!--'. Implements the comment state machine in a
        // compact scan: look for '-->' / '--!>' / abrupt '>'.
        let mut data = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        self.error("eof in comment");
                        break;
                    }
                    if data.is_empty() {
                        return None;
                    }
                    break;
                }
                Some('>') => {
                    // Abrupt closing of an empty comment "<!-->".
                    if data.is_empty() {
                        self.pos += 1;
                        break;
                    }
                    data.push('>');
                    self.pos += 1;
                }
                Some('-') => {
                    // Possible end: '-->' or '--!>', or the dashes
                    // at EOF (they are the end marker, not data:
                    // `<!--x--` at EOF is a comment with data "x").
                    // A '-' followed by '>' right after '<!--' is
                    // the abrupt closing of an empty comment
                    // (`<!--->` -> empty comment, comments01 #9).
                    match self.peek_at(1) {
                        Some('-') => match self.peek_at(2) {
                            Some('>') => {
                                self.pos += 3;
                                return Some(Token::Comment(data));
                            }
                            Some('!') => match self.peek_at(3) {
                                Some('>') => {
                                    self.pos += 4;
                                    return Some(Token::Comment(data));
                                }
                                None if self.eof => {
                                    return Some(Token::Comment(data));
                                }
                                None => {
                                    if data.is_empty() {
                                        return None;
                                    }
                                    break;
                                }
                                _ => {
                                    data.push('-');
                                    self.pos += 1;
                                }
                            },
                            None if self.eof => {
                                // The '--' at EOF is the end marker:
                                // consume it, the comment ends here.
                                self.pos += 2;
                                return Some(Token::Comment(data));
                            }
                            None => {
                                if data.is_empty() {
                                    return None;
                                }
                                break;
                            }
                            _ => {
                                data.push('-');
                                self.pos += 1;
                            }
                        },
                        Some('>') if data.is_empty() => {
                            self.pos += 2;
                            return Some(Token::Comment(data));
                        }
                        None if self.eof => {
                            // A lone trailing '-' at EOF is the
                            // (incomplete) end marker, not data.
                            self.pos += 1;
                            return Some(Token::Comment(data));
                        }
                        None => {
                            if data.is_empty() {
                                return None;
                            }
                            break;
                        }
                        _ => {
                            data.push('-');
                            self.pos += 1;
                        }
                    }
                }
                Some('\0') => {
                    data.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some('<') => {
                    // '<' inside a comment is literal (nested-comment
                    // parse errors aside).
                    data.push('<');
                    self.pos += 1;
                }
                Some(c) => {
                    data.push(c);
                    self.pos += 1;
                }
            }
        }
        Some(Token::Comment(data))
    }

    fn doctype_token(&mut self) -> Option<Token> {
        // After 'doctype'. Parse name / public / system ids.
        let mut name: Option<String> = None;
        let mut public_id: Option<String> = None;
        let mut system_id: Option<String> = None;
        let mut force_quirks = false;

        // Skip whitespace.
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                None => {
                    if self.eof {
                        break;
                    }
                    return None;
                }
                _ => break,
            }
        }
        match self.peek() {
            Some('>') => {
                self.pos += 1;
                force_quirks = true;
                return Some(Token::Doctype {
                    name,
                    public_id,
                    system_id,
                    force_quirks,
                });
            }
            None => {
                if self.eof {
                    force_quirks = true;
                    return Some(Token::Doctype {
                        name,
                        public_id,
                        system_id,
                        force_quirks,
                    });
                }
                return None;
            }
            _ => {}
        }
        // Name.
        let mut n = String::new();
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() || c == '>' => break,
                None => {
                    if self.eof {
                        break;
                    }
                    return None;
                }
                Some(c) => {
                    n.push(c.to_ascii_lowercase());
                    self.pos += 1;
                }
            }
        }
        name = Some(n);
        // After name: optional PUBLIC / SYSTEM.
        loop {
            // Skip whitespace.
            loop {
                match self.peek() {
                    Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                    None => {
                        if self.eof {
                            break;
                        }
                        return None;
                    }
                    _ => break,
                }
            }
            match self.peek() {
                Some('>') => {
                    self.pos += 1;
                    return Some(Token::Doctype {
                        name,
                        public_id,
                        system_id,
                        force_quirks,
                    });
                }
                None => {
                    if self.eof {
                        force_quirks = true;
                        return Some(Token::Doctype {
                            name,
                            public_id,
                            system_id,
                            force_quirks,
                        });
                    }
                    return None;
                }
                Some(_) => {
                    if self.starts_with_ignore_case("public") {
                        if self.peek_at(5).is_none() && !self.eof {
                            return None;
                        }
                        self.consume_str_ignore_case("public");
                        match self.read_doctype_id() {
                            None => return None,
                            Some((id, q)) => {
                                public_id = Some(id);
                                force_quirks = force_quirks || q;
                                // Optional SYSTEM after PUBLIC.
                                match self.maybe_system_id() {
                                    None => return None,
                                    Some((sid, q2)) => {
                                        system_id = sid;
                                        force_quirks = force_quirks || q2;
                                    }
                                }
                            }
                        }
                    } else if self.starts_with_ignore_case("system") {
                        if self.peek_at(5).is_none() && !self.eof {
                            return None;
                        }
                        self.consume_str_ignore_case("system");
                        match self.read_doctype_id() {
                            None => return None,
                            Some((id, q)) => {
                                system_id = Some(id);
                                force_quirks = force_quirks || q;
                            }
                        }
                    } else {
                        self.error("bogus doctype");
                        force_quirks = true;
                        // Skip to '>'.
                        loop {
                            match self.peek() {
                                Some('>') => {
                                    self.pos += 1;
                                    return Some(Token::Doctype {
                                        name,
                                        public_id,
                                        system_id,
                                        force_quirks,
                                    });
                                }
                                None => {
                                    if self.eof {
                                        return Some(Token::Doctype {
                                            name,
                                            public_id,
                                            system_id,
                                            force_quirks,
                                        });
                                    }
                                    return None;
                                }
                                _ => self.pos += 1,
                            }
                        }
                    }
                }
            }
        }
    }

    /// After PUBLIC/SYSTEM keyword: skip ws, read a quoted (or
    /// missing-quote) identifier. Returns (id, force_quirks).
    fn read_doctype_id(&mut self) -> Option<(String, bool)> {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                None => {
                    if self.eof {
                        return Some((String::new(), true));
                    }
                    return None;
                }
                _ => break,
            }
        }
        match self.peek() {
            Some('"') | Some('\'') => {
                let q = self.next_char().unwrap();
                let mut id = String::new();
                loop {
                    match self.peek() {
                        None => {
                            if self.eof {
                                return Some((id, true));
                            }
                            return None;
                        }
                        Some(c) if c == q => {
                            self.pos += 1;
                            return Some((id, false));
                        }
                        Some('>') => {
                            self.error("unexpected > in doctype id");
                            return Some((id, true));
                        }
                        Some(c) => {
                            id.push(c);
                            self.pos += 1;
                        }
                    }
                }
            }
            Some('>') => {
                self.error("missing doctype id");
                Some((String::new(), true))
            }
            _ => {
                self.error("missing quotes in doctype id");
                Some((String::new(), true))
            }
        }
    }

    /// After a PUBLIC id, optionally read a SYSTEM id.
    fn maybe_system_id(&mut self) -> Option<(Option<String>, bool)> {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                None => {
                    if self.eof {
                        return Some((None, false));
                    }
                    return None;
                }
                _ => break,
            }
        }
        if self.starts_with_ignore_case("system") {
            if self.peek_at(5).is_none() && !self.eof {
                return None;
            }
            self.consume_str_ignore_case("system");
            match self.read_doctype_id() {
                None => None,
                Some((id, q)) => Some((Some(id), q)),
            }
        } else if matches!(self.peek(), Some('"') | Some('\'')) {
            // The system identifier may follow the public identifier
            // without a SYSTEM keyword.
            match self.read_doctype_id() {
                None => None,
                Some((id, q)) => Some((Some(id), q)),
            }
        } else {
            Some((None, false))
        }
    }

    // -- RCDATA / raw text / script data -------------------------------------

    /// Check whether the input at `pos` is an appropriate end tag
    /// for `raw_tag` (case-insensitive name followed by whitespace,
    /// '/', '>' or EOF).
    fn appropriate_end_tag_at(&self) -> EndTagCheck {
        let Some(tag) = self.raw_tag.as_ref() else {
            return EndTagCheck::No;
        };
        if self.peek() != Some('<') {
            return EndTagCheck::No;
        }
        match self.peek_at(1) {
            None if !self.eof => return EndTagCheck::NeedMore,
            Some('/') => {}
            _ => return EndTagCheck::No,
        }
        let mut i = 2usize;
        for c in tag.chars() {
            match self.peek_at(i) {
                Some(p) if p.eq_ignore_ascii_case(&c) => i += 1,
                None if !self.eof => return EndTagCheck::NeedMore,
                _ => return EndTagCheck::No,
            }
        }
        // After the name a terminator is required: '>' closes the
        // tag; whitespace or '/' opens the attribute section (which
        // may run to EOF — the tag is then dropped by eof-in-tag);
        // any other character (a letter, as in `</scripts>`) means
        // this is not an end tag at all. EOF immediately after the
        // bare name is literal text (e.g. `</SCRIPT` at EOF).
        match self.peek_at(i) {
            Some('>') => return EndTagCheck::Yes,
            Some(c) if c.is_ascii_whitespace() || c == '/' => {
                i += 1;
            }
            None if self.eof => return EndTagCheck::No,
            None => return EndTagCheck::NeedMore,
            Some(_) => return EndTagCheck::No,
        }
        // Scan the attribute section (quote-aware) for '>' or EOF.
        let mut quote: Option<char> = None;
        loop {
            match self.peek_at(i) {
                None if self.eof => return EndTagCheck::Yes,
                None => return EndTagCheck::NeedMore,
                Some(q @ ('"' | '\'')) => {
                    if quote == Some(q) {
                        quote = None;
                    } else if quote.is_none() {
                        quote = Some(q);
                    }
                    i += 1;
                }
                Some('>') if quote.is_none() => return EndTagCheck::Yes,
                Some(_) => i += 1,
            }
        }
    }

    /// Consume characters up to (but not including) an appropriate
    /// end tag, or EOF. Emits the characters as one token and queues
    /// the end tag. `decode_refs`: RCDATA decodes character
    /// references; raw text does not.
    fn raw_content_token(&mut self, decode_refs: bool) -> Option<Token> {
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    if text.is_empty() {
                        return None;
                    }
                    break;
                }
                Some('<') => {
                    match self.appropriate_end_tag_at() {
                        EndTagCheck::Yes => break,
                        EndTagCheck::NeedMore => {
                            if text.is_empty() {
                                return None;
                            }
                            break;
                        }
                        EndTagCheck::No => {
                            text.push('<');
                            self.pos += 1;
                        }
                    }
                }
                Some('&') if decode_refs => match self.consume_char_ref(false) {
                    Some(v) => {
                        if v.is_empty() {
                            if text.is_empty() {
                                return None;
                            }
                            break;
                        }
                        text.push_str(&v);
                    }
                    None => {
                        text.push('&');
                        self.pos += 1;
                    }
                },
                Some('\0') => {
                    self.error("null in raw text");
                    text.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        if text.is_empty() {
            // At an appropriate end tag (or EOF with nothing).
            if self.peek() == Some('<')
                && matches!(self.appropriate_end_tag_at(), EndTagCheck::Yes)
            {
                self.state = State::Data;
                self.raw_tag = None;
                // Parse the end tag now (it is complete or EOF).
                self.pos += 1; // '<'
                return self.end_tag_open();
            }
            if self.eof {
                self.state = State::Data;
                return self.eof_token();
            }
            return None;
        }
        Some(Token::Character(text))
    }

    fn rcdata_token(&mut self) -> Option<Token> {
        self.raw_content_token(true)
    }

    fn raw_text_token(&mut self) -> Option<Token> {
        self.raw_content_token(false)
    }

    fn plain_text_token(&mut self) -> Option<Token> {
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    if text.is_empty() {
                        return None;
                    }
                    break;
                }
                Some('\0') => {
                    self.error("null in plaintext");
                    text.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        if text.is_empty() {
            self.eof_token()
        } else {
            Some(Token::Character(text))
        }
    }

    /// Script data, with the `<!--` escaped states. `escaped`:
    /// currently inside a `<!-- ... -->` escape.
    fn script_data_token(&mut self, escaped: bool) -> Option<Token> {
        // `escaped` follows the state: the `<!--` transition below
        // enters the escaped state mid-token, so a later `<script`
        // in the same call must still enter double-escaped state.
        let mut escaped = escaped;
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    if text.is_empty() {
                        return None;
                    }
                    break;
                }
                Some('<') => {
                    if self.peek_at(1).is_none() && !self.eof {
                        if text.is_empty() {
                            return None;
                        }
                        break;
                    }
                    match self.appropriate_end_tag_at() {
                        EndTagCheck::Yes => break,
                        EndTagCheck::NeedMore => {
                            if text.is_empty() {
                                return None;
                            }
                            break;
                        }
                        EndTagCheck::No => {}
                    }
                    // '<!--' enters escaped state; '<script' in
                    // escaped state enters double-escaped state.
                    if self.peek_at(1) == Some('!')
                        && self.peek_at(2) == Some('-')
                        && self.peek_at(3) == Some('-')
                    {
                        if self.peek_at(3).is_none() && !self.eof {
                            if text.is_empty() {
                                return None;
                            }
                            break;
                        }
                        text.push_str("<!--");
                        self.pos += 4;
                        self.state = State::ScriptDataEscaped;
                        escaped = true;
                        continue;
                    }
                    if escaped
                        && self.peek_at(1).is_some()
                        && self.starts_with_at(1, "script")
                        && self.script_start_delim_at(7)
                    {
                        // Preserve the input's case (`<sCrIpt>`).
                        for k in 0..7 {
                            if let Some(c) = self.peek_at(k) {
                                text.push(c);
                            }
                        }
                        self.pos += 7;
                        self.state = State::ScriptDataDoubleEscaped;
                        // Hand off to the double-escaped state on
                        // the next call (`-->` must not exit there).
                        break;
                    }
                    text.push('<');
                    self.pos += 1;
                }
                Some('-') if escaped => {
                    if self.peek_at(1) == Some('-')
                        && self.peek_at(2) == Some('>')
                    {
                        text.push_str("-->");
                        self.pos += 3;
                        self.state = State::ScriptData;
                        escaped = false;
                        continue;
                    }
                    text.push('-');
                    self.pos += 1;
                }
                Some('\0') => {
                    self.error("null in script data");
                    text.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        if text.is_empty() {
            if self.peek() == Some('<')
                && matches!(self.appropriate_end_tag_at(), EndTagCheck::Yes)
            {
                self.state = State::Data;
                self.raw_tag = None;
                self.pos += 1; // '<'
                return self.end_tag_open();
            }
            if self.eof {
                self.state = State::Data;
                return self.eof_token();
            }
            return None;
        }
        Some(Token::Character(text))
    }

    fn script_double_escaped_token(&mut self) -> Option<Token> {
        // Like script data, but '</script' exits to escaped state
        // and '-->' does not exit.
        let mut text = String::new();
        loop {
            match self.peek() {
                None => {
                    if self.eof {
                        break;
                    }
                    if text.is_empty() {
                        return None;
                    }
                    break;
                }
                Some('<') => {
                    match self.appropriate_end_tag_at() {
                        EndTagCheck::Yes => {
                            // Spec 13.2.5.31 (script data double
                            // escape end state): `</script` followed
                            // by whitespace, '/' or '>' is TEXT, not
                            // an end tag; the tokenizer returns to
                            // the escaped state.
                            let name_len =
                                self.raw_tag.as_ref().map(|t| t.len()).unwrap_or(0);
                            let raw_name: String = (0..name_len)
                                .filter_map(|k| self.peek_at(2 + k))
                                .collect();
                            text.push('<');
                            text.push('/');
                            text.push_str(&raw_name);
                            self.pos += 2 + name_len;
                            // Consume the terminator as text too.
                            if let Some(c) = self.peek() {
                                if c.is_ascii_whitespace() || c == '/' || c == '>' {
                                    text.push(c);
                                    self.pos += 1;
                                }
                            }
                            self.state = State::ScriptDataEscaped;
                            return Some(Token::Character(text));
                        }
                        EndTagCheck::NeedMore => {
                            if text.is_empty() {
                                return None;
                            }
                            break;
                        }
                        EndTagCheck::No => {
                            text.push('<');
                            self.pos += 1;
                        }
                    }
                }
                Some('-') => {
                    // Spec 13.2.5.30: `-->` in the double-escaped
                    // state exits to the (single) escaped state, so
                    // a following `</script>` is a real end tag.
                    if self.peek_at(1) == Some('-') && self.peek_at(2) == Some('>') {
                        text.push_str("-->");
                        self.pos += 3;
                        self.state = State::ScriptDataEscaped;
                        return Some(Token::Character(text));
                    }
                    text.push('-');
                    self.pos += 1;
                }
                Some('\0') => {
                    self.error("null in script data");
                    text.push('\u{FFFD}');
                    self.pos += 1;
                }
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
        if text.is_empty() {
            // An appropriate end tag here is handled in the loop
            // above (emitted as text); reaching this point means
            // the buffer ended mid-`</script` or at EOF.
            if self.eof {
                self.state = State::Data;
                return self.eof_token();
            }
            return None;
        }
        Some(Token::Character(text))
    }

    fn starts_with_at(&self, off: usize, s: &str) -> bool {
        s.chars().enumerate().all(|(i, c)| {
            self.peek_at(off + i)
                .map(|p| p.eq_ignore_ascii_case(&c))
                .unwrap_or(false)
        })
    }

    fn script_start_delim_at(&self, off: usize) -> bool {
        match self.peek_at(off) {
            Some(c) => c.is_ascii_whitespace() || c == '/' || c == '>',
            None => self.eof,
        }
    }
}
