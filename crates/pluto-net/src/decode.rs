//! Incremental gzip/deflate decoding (Stage 5 #9).
//!
//! hyper does not decompress response bodies. Pluto advertises
//! `Accept-Encoding: gzip, deflate` and decodes in `pluto-net`
//! before handing bytes to the parser — otherwise every CSS/JS
//! payload costs 3–5× bandwidth.
//!
//! Decoding is incremental: flate2's *write* decoders take
//! compressed chunks and hand back decoded bytes, so the streaming
//! path never buffers a whole body. `deflate` here means the
//! zlib-wrapped form (RFC 1950), which is what HTTP's
//! `Content-Encoding: deflate` specifies — not raw deflate.
//! An unknown encoding is an error, not a passthrough: a body the
//! parser cannot read must fail loudly instead of arriving
//! compressed.

use std::io::Write;

use crate::NetError;

/// A streaming decompressor for one response body.
pub enum Decoder {
    Identity,
    Gzip(flate2::write::GzDecoder<Vec<u8>>),
    Deflate(flate2::write::ZlibDecoder<Vec<u8>>),
}

impl Decoder {
    /// Pick a decoder from a `Content-Encoding` header value.
    /// Absent, empty or `identity` means no decoding.
    pub fn from_encoding(encoding: Option<&str>) -> Result<Self, NetError> {
        match encoding.map(str::trim) {
            None | Some("") | Some("identity") => Ok(Decoder::Identity),
            Some("gzip") => Ok(Decoder::Gzip(flate2::write::GzDecoder::new(Vec::new()))),
            Some("deflate") => Ok(Decoder::Deflate(
                flate2::write::ZlibDecoder::new(Vec::new()),
            )),
            Some(other) => Err(NetError::Body(format!(
                "unsupported content-encoding: {other}"
            ))),
        }
    }

    /// Feed one compressed chunk; returns the bytes it decoded
    /// (possibly empty — headers and block boundaries decode to
    /// nothing until enough input arrives).
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<u8>, NetError> {
        match self {
            Decoder::Identity => Ok(chunk.to_vec()),
            Decoder::Gzip(d) => {
                d.write_all(chunk)
                    .map_err(|e| NetError::Body(e.to_string()))?;
                Ok(std::mem::take(d.get_mut()))
            }
            Decoder::Deflate(d) => {
                d.write_all(chunk)
                    .map_err(|e| NetError::Body(e.to_string()))?;
                Ok(std::mem::take(d.get_mut()))
            }
        }
    }

    /// End of stream: flush and validate the trailer (gzip CRC).
    /// Returns any last decoded bytes.
    pub fn finish(self) -> Result<Vec<u8>, NetError> {
        match self {
            Decoder::Identity => Ok(Vec::new()),
            Decoder::Gzip(mut d) => {
                d.try_finish().map_err(|e| NetError::Body(e.to_string()))?;
                Ok(std::mem::take(d.get_mut()))
            }
            Decoder::Deflate(mut d) => {
                d.try_finish().map_err(|e| NetError::Body(e.to_string()))?;
                Ok(std::mem::take(d.get_mut()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    }

    fn zlib(bytes: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn gzip_decodes_incrementally() {
        let plain = b"hello world, hello world, hello world";
        let compressed = gzip(plain);
        let mut d = Decoder::from_encoding(Some("gzip")).unwrap();
        // Feed one byte at a time: output arrives in drips, but the
        // concatenation is exact.
        let mut out = Vec::new();
        for byte in &compressed {
            out.extend_from_slice(&d.feed(std::slice::from_ref(byte)).unwrap());
        }
        out.extend_from_slice(&d.finish().unwrap());
        assert_eq!(out, plain);
    }

    #[test]
    fn deflate_decodes() {
        let plain = b"body { color: red; }";
        let compressed = zlib(plain);
        let mut d = Decoder::from_encoding(Some("deflate")).unwrap();
        let mut out = d.feed(&compressed).unwrap();
        out.extend_from_slice(&d.finish().unwrap());
        assert_eq!(out, plain);
    }

    #[test]
    fn identity_passes_through_and_unknown_errors() {
        let mut d = Decoder::from_encoding(None).unwrap();
        assert_eq!(d.feed(b"abc").unwrap(), b"abc");
        assert!(Decoder::from_encoding(Some("br")).is_err());
        assert!(Decoder::from_encoding(Some("gzip, deflate")).is_err());
    }

    #[test]
    fn corrupt_gzip_errors_on_finish() {
        let mut compressed = gzip(b"hello");
        compressed.truncate(compressed.len() - 4); // cut the CRC
        let mut d = Decoder::from_encoding(Some("gzip")).unwrap();
        let _ = d.feed(&compressed);
        assert!(d.finish().is_err());
    }
}
