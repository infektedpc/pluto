//! Image decoding and the browser-owned image cache (Stage 4 #6).
//!
//! Decoding is the most crash-prone CPU work after JS, so it runs on
//! the shared rayon pool (or wherever the caller runs it off the UI
//! thread) and lands in the [`ImageCache`], which owns the
//! [`ResourceTable`] the display list references by id. A decoded
//! image is stored once as RGBA bytes; the renderer uploads it to a
//! wgpu texture from the table, never from the wire.
//!
//! Nothing in M1 produces an image yet (`<img>` and
//! `background-image` are M3) — this is the decode path and cache M3
//! plugs into, proven here by tests that round-trip real PNG bytes.

use crate::{Resource, ResourceId, ResourceTable};
use rayon::prelude::*;

/// One decoded image: RGBA8 bytes, row-major, `width * height * 4`
/// bytes. The layout is the `Resource::Image` layout, so a decoded
/// image drops straight into the resource table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("image decode failed: {0}")]
    Decode(#[from] image::ImageError),
    #[error("decoded image has zero width or height")]
    Empty,
    #[error("decoded image byte length {0} does not match {1}x{2} RGBA")]
    LengthMismatch(usize, u32, u32),
}

/// Decode one image's bytes (PNG, JPEG, GIF, WebP — the formats the
/// workspace `image` dependency enables) to RGBA8. Pure and
/// thread-safe: call it on the rayon pool, never on the UI thread.
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, ImageError> {
    let dynamic = image::load_from_memory(bytes)?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Err(ImageError::Empty);
    }
    let raw = rgba.into_raw();
    if raw.len() != (width as usize) * (height as usize) * 4 {
        return Err(ImageError::LengthMismatch(raw.len(), width, height));
    }
    Ok(DecodedImage {
        width,
        height,
        rgba: raw,
    })
}

/// Decode several images in parallel on the shared rayon pool.
/// Order is preserved: result `i` is the decode of input `i`.
pub fn decode_images_parallel(inputs: &[&[u8]]) -> Vec<Result<DecodedImage, ImageError>> {
    inputs.par_iter().map(|bytes| decode_image(bytes)).collect()
}

/// The browser-owned image cache: decoded images keyed by
/// [`ResourceId`], backed by the [`ResourceTable`] display items
/// reference. Ids are assigned in insertion order, starting at 1
/// (0 is reserved as "no resource").
#[derive(Debug, Default)]
pub struct ImageCache {
    table: ResourceTable,
    next_id: u32,
}

impl ImageCache {
    pub fn new() -> Self {
        Self {
            table: ResourceTable::default(),
            next_id: 1,
        }
    }

    /// Decode `bytes` and store the result, returning its id.
    pub fn insert_bytes(&mut self, bytes: &[u8]) -> Result<ResourceId, ImageError> {
        let decoded = decode_image(bytes)?;
        Ok(self.insert_decoded(decoded))
    }

    /// Store an already-decoded image, returning its id.
    pub fn insert_decoded(&mut self, image: DecodedImage) -> ResourceId {
        let id = ResourceId(self.next_id);
        // Ids are u32 and insertion is one-per-call; wrapping is not a
        // practical concern (4 billion images), but never hand out 0.
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.table.insert(
            id,
            Resource::Image {
                width: image.width,
                height: image.height,
                rgba: image.rgba,
            },
        );
        id
    }

    /// The table the display list references (serialisable with it).
    pub fn table(&self) -> &ResourceTable {
        &self.table
    }

    pub fn get(&self, id: ResourceId) -> Option<&Resource> {
        self.table.get(id)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::png::PngEncoder;
    use image::{ExtendedColorType, ImageEncoder};

    /// Encode a small RGBA image to PNG bytes, so the decode tests
    /// exercise real format bytes without a checked-in fixture.
    fn png_bytes(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(rgba, width, height, ExtendedColorType::Rgba8)
            .expect("test PNG encodes");
        out
    }

    #[test]
    fn decode_png_round_trips_exact_pixels() {
        // 2x2: red, green, blue, transparent white.
        let rgba: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 0,
        ];
        let bytes = png_bytes(2, 2, &rgba);
        let decoded = decode_image(&bytes).expect("PNG decodes");
        assert_eq!((decoded.width, decoded.height), (2, 2));
        assert_eq!(decoded.rgba, rgba);
    }

    #[test]
    fn decode_rejects_garbage_and_truncated_bytes() {
        assert!(decode_image(b"not an image at all").is_err());
        let bytes = png_bytes(2, 2, &[255u8; 16]);
        assert!(decode_image(&bytes[..bytes.len() / 2]).is_err());
        assert!(decode_image(&[]).is_err());
    }

    #[test]
    fn parallel_decode_matches_serial_decode() {
        let inputs: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                let px = [i * 30, 100, 200, 255];
                let rgba = [px, px, px, px].concat();
                png_bytes(2, 2, &rgba)
            })
            .collect();
        let refs: Vec<&[u8]> = inputs.iter().map(|b| b.as_slice()).collect();
        let parallel = decode_images_parallel(&refs);
        let serial: Vec<_> = refs.iter().map(|b| decode_image(b)).collect();
        assert_eq!(parallel.len(), serial.len());
        for (p, s) in parallel.iter().zip(serial.iter()) {
            assert_eq!(p.as_ref().ok(), s.as_ref().ok());
        }
        // And every parallel result is the right image.
        for (i, result) in parallel.iter().enumerate() {
            let img = result.as_ref().expect("parallel decode succeeds");
            assert_eq!(img.rgba[0], (i as u8) * 30);
        }
    }

    #[test]
    fn image_cache_assigns_ids_and_serves_the_table() {
        let mut cache = ImageCache::new();
        assert!(cache.is_empty());
        let red = png_bytes(1, 1, &[255, 0, 0, 255]);
        let green = png_bytes(1, 1, &[0, 255, 0, 255]);
        let id_red = cache.insert_bytes(&red).expect("red decodes");
        let id_green = cache.insert_bytes(&green).expect("green decodes");
        assert_ne!(id_red, id_green);
        assert_ne!(id_red, ResourceId(0), "0 is never handed out");
        assert_eq!(cache.len(), 2);
        match cache.get(id_red).expect("red is cached") {
            Resource::Image {
                width,
                height,
                rgba,
            } => {
                assert_eq!((*width, *height), (1, 1));
                assert_eq!(rgba, &[255, 0, 0, 255]);
            }
        }
        assert!(cache.get(ResourceId(999)).is_none());
        // The table round-trips through postcard with its images.
        let bytes = postcard::to_allocvec(cache.table()).expect("table serialises");
        let back: ResourceTable = postcard::from_bytes(&bytes).expect("table deserialises");
        assert_eq!(back.len(), 2);
        assert_eq!(back.get(id_green), cache.get(id_green));
    }
}
