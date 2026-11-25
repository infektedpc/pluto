//! Subresource Integrity (Stage 9 Rec 14; W3C SRI).
//!
//! Parse an element's `integrity` attribute into its hash entries
//! and verify streamed bytes against them — incrementally, chunk by
//! chunk, as the body streams in. The body is never buffered just
//! to hash it (that is the whole point: the check rides the
//! streaming path, it does not sit on top of a collected body).
//!
//! The check itself happens at fetch time: the streaming path in
//! `service.rs` feeds each `BodyChunk` to [`SriVerifier::update`]
//! and fails the fetch when [`SriVerifier::verify`] is false at the
//! end of the body. This module only parses metadata and hashes /
//! compares — it never fetches, never decides what a failure means
//! for the page, and never touches the DOM.
//!
//! Rules (W3C SRI, "parse metadata" + "does the response match"):
//! * The attribute is a space-separated list of
//!   `<algorithm>-<base64 digest>` entries. Whitespace-tolerant:
//!   entries are split on any run of whitespace.
//! * Only `sha256` / `sha384` / `sha512` are supported. Unknown
//!   algorithms (`md5`, `sha1`, ...) are ignored, not errors.
//! * A malformed entry is ignored, not an error: no `-`, an empty
//!   digest, undecodable base64, or a decoded digest whose length
//!   does not match the algorithm's output length.
//! * A `?` in the digest starts per-entry options, which are not
//!   part of the digest and are ignored.
//! * Verification passes when ANY listed entry matches the bytes.
//! * No integrity metadata at all means no check —
//!   [`SriVerifier::verify`] returns `true`, not a failure.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use ring::digest::{self, Context};

/// A hash algorithm named in an `integrity` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Algorithm {
    Sha256,
    Sha384,
    Sha512,
}

impl Algorithm {
    /// The `ring` digest algorithm this maps to.
    pub fn ring_algorithm(self) -> &'static digest::Algorithm {
        match self {
            Algorithm::Sha256 => &digest::SHA256,
            Algorithm::Sha384 => &digest::SHA384,
            Algorithm::Sha512 => &digest::SHA512,
        }
    }

    /// Expected digest length in bytes for this algorithm.
    pub fn digest_len(self) -> usize {
        match self {
            Algorithm::Sha256 => 32,
            Algorithm::Sha384 => 48,
            Algorithm::Sha512 => 64,
        }
    }

    /// Parse an algorithm name (ASCII case-insensitive). `None`
    /// for anything unsupported — `md5`, `sha1`, a typo.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sha256" => Some(Algorithm::Sha256),
            "sha384" => Some(Algorithm::Sha384),
            "sha512" => Some(Algorithm::Sha512),
            _ => None,
        }
    }
}

/// One parsed `integrity` entry: `<algorithm>-<base64 digest>`,
/// with the digest already decoded to raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityEntry {
    pub algorithm: Algorithm,
    /// Raw digest bytes (base64-decoded).
    pub digest: Vec<u8>,
}

/// Parse an `integrity` attribute value into its entries.
///
/// Whitespace-tolerant (split on any whitespace run); malformed
/// entries and unknown algorithms are silently skipped, so a value
/// with no usable entries parses to an empty list — never an error.
pub fn parse_integrity(value: &str) -> Vec<IntegrityEntry> {
    value
        .split_whitespace()
        .filter_map(|token| {
            let (algo, encoded) = token.split_once('-')?;
            let algorithm = Algorithm::parse(algo)?;
            // Options after `?` are not part of the digest.
            let encoded = encoded.split('?').next().unwrap_or("");
            if encoded.is_empty() {
                return None;
            }
            // Strict base64: correct padding, standard alphabet.
            let digest = STANDARD.decode(encoded).ok()?;
            if digest.len() != algorithm.digest_len() {
                return None;
            }
            Some(IntegrityEntry { algorithm, digest })
        })
        .collect()
}

/// Incrementally verifies streamed bytes against parsed
/// `integrity` metadata (Stage 9 Rec 14).
///
/// One live hash context per distinct algorithm in the entries —
/// entries sharing an algorithm share one context, so a body is
/// hashed once per algorithm no matter how many entries name it.
/// Feed each body chunk to [`SriVerifier::update`] as it arrives
/// and call [`SriVerifier::verify`] when the body is complete.
#[derive(Clone)]
pub struct SriVerifier {
    entries: Vec<IntegrityEntry>,
    /// One live context per distinct algorithm in `entries`.
    contexts: Vec<(Algorithm, Context)>,
}

// Manual `Debug`: `ring::digest::Context` is not `Debug`.
impl std::fmt::Debug for SriVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SriVerifier")
            .field("entries", &self.entries)
            .field(
                "algorithms",
                &self.contexts.iter().map(|(a, _)| a).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SriVerifier {
    /// Build a verifier from already-parsed entries.
    pub fn new(entries: Vec<IntegrityEntry>) -> Self {
        let mut algorithms: Vec<Algorithm> =
            entries.iter().map(|entry| entry.algorithm).collect();
        algorithms.sort();
        algorithms.dedup();
        let contexts = algorithms
            .into_iter()
            .map(|algorithm| (algorithm, Context::new(algorithm.ring_algorithm())))
            .collect();
        Self { entries, contexts }
    }

    /// Parse an `integrity` attribute value and build a verifier
    /// from it.
    pub fn from_attribute(value: &str) -> Self {
        Self::new(parse_integrity(value))
    }

    /// Whether any integrity metadata was supplied at all. With
    /// none there is nothing to check — [`SriVerifier::verify`]
    /// passes.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Feed the next chunk of body bytes into every hash context.
    /// Call once per streamed chunk, in body order.
    pub fn update(&mut self, bytes: &[u8]) {
        for (_, context) in self.contexts.iter_mut() {
            context.update(bytes);
        }
    }

    /// Whether the streamed bytes match ANY listed entry (W3C SRI:
    /// a single match passes). With no entries at all this returns
    /// `true`: no integrity metadata means no check, not a failure.
    pub fn verify(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }
        // `Context::finish` consumes; clone the small hash state so
        // `verify` can stay `&self` (and be called more than once).
        let finished: Vec<(Algorithm, digest::Digest)> = self
            .contexts
            .iter()
            .map(|(algorithm, context)| (*algorithm, context.clone().finish()))
            .collect();
        self.entries.iter().any(|entry| {
            finished
                .iter()
                .find(|(algorithm, _)| *algorithm == entry.algorithm)
                .is_some_and(|(_, digest)| digest.as_ref() == entry.digest.as_slice())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ring` one-shot digest of `bytes`, for building expected
    /// `integrity` strings in tests.
    fn one_shot(algorithm: Algorithm, bytes: &[u8]) -> Vec<u8> {
        digest::digest(algorithm.ring_algorithm(), bytes)
            .as_ref()
            .to_vec()
    }

    /// Render one entry the way an author would write it in HTML.
    fn entry_text(algorithm: Algorithm, digest: &[u8]) -> String {
        let name = match algorithm {
            Algorithm::Sha256 => "sha256",
            Algorithm::Sha384 => "sha384",
            Algorithm::Sha512 => "sha512",
        };
        format!("{name}-{}", STANDARD.encode(digest))
    }

    #[test]
    fn parses_multiple_entries() {
        let body = b"hello world";
        let value = format!(
            "{} {} {}",
            entry_text(Algorithm::Sha256, &one_shot(Algorithm::Sha256, body)),
            entry_text(Algorithm::Sha384, &one_shot(Algorithm::Sha384, body)),
            entry_text(Algorithm::Sha512, &one_shot(Algorithm::Sha512, body)),
        );
        let entries = parse_integrity(&value);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].algorithm, Algorithm::Sha256);
        assert_eq!(entries[1].algorithm, Algorithm::Sha384);
        assert_eq!(entries[2].algorithm, Algorithm::Sha512);
        assert_eq!(entries[0].digest, one_shot(Algorithm::Sha256, body));
    }

    #[test]
    fn parsing_is_whitespace_tolerant() {
        let digest = one_shot(Algorithm::Sha256, b"x");
        let value = format!(
            "  {} \t\n {}  ",
            entry_text(Algorithm::Sha256, &digest),
            entry_text(Algorithm::Sha384, &one_shot(Algorithm::Sha384, b"x")),
        );
        assert_eq!(parse_integrity(&value).len(), 2);
    }

    #[test]
    fn malformed_entries_are_ignored() {
        // No `-`; empty digest; undecodable base64; a digest of the
        // wrong length for its algorithm; a `?`-options entry whose
        // digest part is still valid (options are ignored, so it
        // parses). One good entry survives among the junk.
        let good = one_shot(Algorithm::Sha256, b"ok");
        let short = STANDARD.encode([0u8; 5]); // valid base64, wrong length
        let value = format!(
            "sha256 {} sha256- sha256-!!! sha256-{short} sha256-{}?foo=bar {}",
            STANDARD.encode([0u8; 32]), // right length, wrong bytes — still parses
            STANDARD.encode(&good),
            entry_text(Algorithm::Sha256, &good),
        );
        let entries = parse_integrity(&value);
        // The `?`-options entry and the good entry parse; the
        // other five tokens are dropped.
        assert_eq!(entries.len(), 2, "{entries:?}");
        assert!(entries.iter().all(|e| e.digest == good));
    }

    #[test]
    fn unknown_algorithms_are_ignored() {
        let value = format!(
            "md5-{} sha1-{} sha999-{} {}",
            STANDARD.encode([0u8; 16]),
            STANDARD.encode([0u8; 20]),
            STANDARD.encode([0u8; 32]),
            entry_text(Algorithm::Sha384, &one_shot(Algorithm::Sha384, b"y")),
        );
        let entries = parse_integrity(&value);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].algorithm, Algorithm::Sha384);
    }

    #[test]
    fn empty_and_garbage_values_parse_to_nothing() {
        assert!(parse_integrity("").is_empty());
        assert!(parse_integrity("   ").is_empty());
        assert!(parse_integrity("not-an-entry").is_empty());
    }

    #[test]
    fn known_vector_sha256_of_hello() {
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let entries = parse_integrity(
            "sha256-LPJNul+wow4m6DsqxbninhsWHlwfp0JecwQzYpOLmCQ=",
        );
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].algorithm, Algorithm::Sha256);
        let mut verifier = SriVerifier::new(entries);
        verifier.update(b"hello");
        assert!(verifier.verify());
    }

    #[test]
    fn incremental_hashing_matches_one_shot() {
        let body = b"The quick brown fox jumps over the lazy dog";
        for algorithm in [Algorithm::Sha256, Algorithm::Sha384, Algorithm::Sha512] {
            let expected = one_shot(algorithm, body);
            let mut verifier = SriVerifier::new(vec![IntegrityEntry {
                algorithm,
                digest: expected,
            }]);
            // Feed in uneven chunks, including an empty one.
            verifier.update(&body[..1]);
            verifier.update(&[]);
            verifier.update(&body[1..20]);
            verifier.update(&body[20..]);
            assert!(verifier.verify(), "{algorithm:?} chunked");
            // And the same bytes in one shot agree.
            let mut whole = SriVerifier::new(vec![IntegrityEntry {
                algorithm,
                digest: one_shot(algorithm, body),
            }]);
            whole.update(body);
            assert!(whole.verify(), "{algorithm:?} whole");
        }
    }

    #[test]
    fn verify_passes_when_any_entry_matches() {
        let body = b"payload";
        let wrong = vec![0u8; 32];
        let right = one_shot(Algorithm::Sha256, body);
        let value = format!(
            "{} {}",
            entry_text(Algorithm::Sha256, &wrong),
            entry_text(Algorithm::Sha384, &one_shot(Algorithm::Sha384, body)),
        );
        // The sha256 entry is wrong but the sha384 entry matches.
        let mut verifier = SriVerifier::from_attribute(&value);
        verifier.update(body);
        assert!(verifier.verify());
        // Only the wrong entry: fails.
        let mut verifier = SriVerifier::from_attribute(&entry_text(Algorithm::Sha256, &wrong));
        verifier.update(body);
        assert!(!verifier.verify());
        // The right digest alone: passes. (`right` is used above
        // only through `one_shot`; keep the direct check too.)
        let mut verifier = SriVerifier::new(vec![IntegrityEntry {
            algorithm: Algorithm::Sha256,
            digest: right,
        }]);
        verifier.update(body);
        assert!(verifier.verify());
    }

    #[test]
    fn verify_fails_on_tampered_bytes() {
        let body = b"original bytes";
        let mut verifier = SriVerifier::from_attribute(&entry_text(
            Algorithm::Sha512,
            &one_shot(Algorithm::Sha512, body),
        ));
        verifier.update(b"original byteS"); // one bit flipped
        assert!(!verifier.verify());
    }

    #[test]
    fn empty_integrity_is_no_check_not_a_failure() {
        // No metadata at all: verify passes, before and after bytes.
        let verifier = SriVerifier::from_attribute("");
        assert!(verifier.is_empty());
        assert!(verifier.verify());
        let mut verifier = SriVerifier::from_attribute("   ");
        verifier.update(b"anything at all");
        assert!(verifier.verify());
        // Same for a verifier built from no entries, and for a
        // value whose entries were all malformed/unknown.
        let mut verifier = SriVerifier::new(vec![]);
        verifier.update(b"anything at all");
        assert!(verifier.verify());
        let verifier = SriVerifier::from_attribute("md5-AAAAAAAAAAAAAAAAAAAAAA==");
        assert!(verifier.is_empty());
        assert!(verifier.verify());
    }
}
