//! A serialisable glyph cache key (Stage 4 #4).
//!
//! `cosmic_text::CacheKey` does not implement serde, and one of its
//! fields (`font_id: fontdb::ID`) is an opaque slotmap key with no
//! public constructor from a number — so the display list carries this
//! plain-data mirror instead, and the renderer converts back at draw
//! time. The conversion is lossless in-process: every field of the
//! `CacheKey` is represented, and `to_cache_key` resolves the font id
//! through the renderer's font database (both sides load the same
//! system fonts in the same order, so the ids agree — the same
//! assumption the atlas already relies on).
//!
//! The font id is stored as the `u64` from `fontdb::ID`'s `Display`
//! impl, which writes the slotmap key's ffi value. That format is not
//! promised by fontdb's docs, but it is the implementation in the
//! pinned fontdb 0.23, and a key that fails to parse degrades to
//! `u64::MAX` (which simply never resolves, so the glyph is skipped
//! rather than mis-drawn).

use cosmic_text::{CacheKey, CacheKeyFlags, FontSystem, SubpixelBin};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The serialisable form of a [`CacheKey`]: plain numbers only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GlyphKey {
    /// `fontdb::ID` as a `u64` (see module docs).
    pub font_id: u64,
    pub glyph_id: u16,
    /// `f32` bits of the font size.
    pub font_size_bits: u32,
    /// [`SubpixelBin`] as 0..=3.
    pub x_bin: u8,
    /// [`SubpixelBin`] as 0..=3.
    pub y_bin: u8,
    /// `fontdb::Weight.0`.
    pub font_weight: u16,
    /// [`CacheKeyFlags`] bits.
    pub flags: u32,
}

fn bin_to_u8(bin: SubpixelBin) -> u8 {
    match bin {
        SubpixelBin::Zero => 0,
        SubpixelBin::One => 1,
        SubpixelBin::Two => 2,
        SubpixelBin::Three => 3,
    }
}

fn u8_to_bin(v: u8) -> Option<SubpixelBin> {
    match v {
        0 => Some(SubpixelBin::Zero),
        1 => Some(SubpixelBin::One),
        2 => Some(SubpixelBin::Two),
        3 => Some(SubpixelBin::Three),
        _ => None,
    }
}

impl From<CacheKey> for GlyphKey {
    fn from(key: CacheKey) -> Self {
        Self {
            font_id: key.font_id.to_string().parse().unwrap_or(u64::MAX),
            glyph_id: key.glyph_id,
            font_size_bits: key.font_size_bits,
            x_bin: bin_to_u8(key.x_bin),
            y_bin: bin_to_u8(key.y_bin),
            font_weight: key.font_weight.0,
            flags: key.flags.bits(),
        }
    }
}

impl GlyphKey {
    /// Rebuild the [`CacheKey`], resolving the font id through a map
    /// built by [`font_id_map`]. Returns `None` when the font is not
    /// in the map or a bin/flag value is out of range — the caller
    /// skips that glyph, the same as a cache miss.
    pub fn to_cache_key(&self, font_ids: &HashMap<u64, fontdb::ID>) -> Option<CacheKey> {
        Some(CacheKey {
            font_id: *font_ids.get(&self.font_id)?,
            glyph_id: self.glyph_id,
            font_size_bits: self.font_size_bits,
            x_bin: u8_to_bin(self.x_bin)?,
            y_bin: u8_to_bin(self.y_bin)?,
            font_weight: fontdb::Weight(self.font_weight),
            flags: CacheKeyFlags::from_bits_truncate(self.flags),
        })
    }
}

/// Map every loaded font's ffi id to its [`fontdb::ID`], for
/// [`GlyphKey::to_cache_key`]. Built once per `FontSystem` — scanning
/// the face list per glyph would be pure waste.
pub fn font_id_map(font_system: &FontSystem) -> HashMap<u64, fontdb::ID> {
    font_system
        .db()
        .faces()
        .map(|face| (face.id.to_string().parse().unwrap_or(u64::MAX), face.id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic_text::{Attrs, Buffer, Metrics, Shaping};

    fn shaped_keys() -> Vec<CacheKey> {
        let mut font_system = FontSystem::new();
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(16.0, 19.2));
        buffer.set_text(
            "HOSCAxofjgyp0123456789",
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer.set_size(None, None);
        buffer.shape_until_scroll(&mut font_system, false);
        let mut keys = Vec::new();
        for run in buffer.layout_runs() {
            for g in run.glyphs {
                keys.push(g.physical((0.0, 0.0), 1.0).cache_key);
            }
        }
        keys
    }

    #[test]
    fn glyph_key_round_trips_through_cache_key() {
        let keys = shaped_keys();
        assert!(!keys.is_empty(), "shaping produced no glyphs");
        let font_system = FontSystem::new();
        let map = font_id_map(&font_system);
        assert!(!map.is_empty(), "font database has no faces");
        for key in keys {
            let wire = GlyphKey::from(key);
            let back = wire
                .to_cache_key(&map)
                .expect("wire key resolves against the same font system");
            assert_eq!(back, key, "glyph key round-trip must be lossless");
            // And the wire form itself survives postcard bytes.
            let bytes = postcard::to_allocvec(&wire).expect("GlyphKey serialises");
            let decoded: GlyphKey = postcard::from_bytes(&bytes).expect("GlyphKey deserialises");
            assert_eq!(decoded, wire);
        }
    }

    #[test]
    fn glyph_key_rejects_unknown_font_and_bad_bins() {
        let keys = shaped_keys();
        let key = GlyphKey::from(keys[0]);
        // Unknown font id: no map entry, no cache key.
        assert!(key.to_cache_key(&HashMap::new()).is_none());
        // Out-of-range bins: not a SubpixelBin, no cache key.
        let font_system = FontSystem::new();
        let map = font_id_map(&font_system);
        assert!(GlyphKey { x_bin: 9, ..key }.to_cache_key(&map).is_none());
        assert!(GlyphKey { y_bin: 255, ..key }.to_cache_key(&map).is_none());
        // Flag bits outside the known set are truncated, not rejected
        // (bitflags semantics) — the key still resolves.
        assert!(
            GlyphKey {
                flags: u32::MAX,
                ..key
            }
            .to_cache_key(&map)
            .is_some()
        );
    }
}
