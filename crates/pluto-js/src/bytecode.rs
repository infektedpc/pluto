//! In-memory bytecode cache for module scripts (Stage 7 cheap win (b)).
//!
//! Compile once per script URL, reuse on repeat loads. The integration
//! wave wires this to fetched scripts; this module is the cache plus the
//! proof that a second load of the same script skips recompilation.
//!
//! # Keying
//!
//! Key = (script URL, SHA-256 of the source). The hash matters: a URL
//! can be re-fetched with different bytes (a redeploy, a `?v=2` that
//! the loader did not normalise), and the cache must never serve stale
//! bytecode for new source. SHA-256 (not a fast non-cryptographic hash)
//! because scripts come from the network and a cache-poisoning
//! collision must not be constructible by a page.
//!
//! # Bounds and eviction
//!
//! Bounded by total bytecode bytes (`capacity_bytes`). Eviction is LRU
//! (least-recently-used entry first). An entry larger than the whole
//! capacity is never stored. The cache is in-memory only and per-realm:
//! bytecode is tied to the engine version that compiled it, and a fresh
//! realm (navigation) starts with an empty cache — bytecode never
//! crosses a navigation, let alone a process boundary or disk.
//!
//! # Scope
//!
//! Modules only. Classic scripts are not cached in this wave: the
//! rquickjs bytecode API exposed here (`Module::write` / `Module::load`)
//! is module-shaped, and compiling a classic script as a module would
//! change its semantics (strict mode, deferred evaluation). If a later
//! wave needs classic-script caching it must compile the classic script
//! through the engine's script-bytecode path, not by reusing this one.

use std::collections::{HashMap, VecDeque};

use sha2::{Digest, Sha256};

use crate::engine::{EvalKind, JsEngine};
use crate::error::JsError;

/// SHA-256 of a script's source bytes.
pub type ContentHash = [u8; 32];

/// One cache entry's key: which script, and which exact bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// The script's URL, as the loader resolved it (absolute, no
    /// fragment). Two spellings of the same URL are two keys — the
    /// loader normalises before it calls us.
    pub url: String,
    /// SHA-256 of the source the bytecode was compiled from.
    pub hash: ContentHash,
}

/// SHA-256 of `source`.
pub fn hash_source(source: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.finalize().into()
}

/// Cache statistics, for tests and for the host to log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Cache hits (bytecode reused, no compile).
    pub hits: u64,
    /// Cache misses (compiled).
    pub misses: u64,
    /// Entries evicted to make room.
    pub evictions: u64,
    /// Entries currently stored.
    pub entries: usize,
    /// Bytecode bytes currently stored.
    pub used_bytes: usize,
}

/// Whether a [`BytecodeCache::load_or_compile`] call compiled or reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHit {
    /// Bytecode came from the cache; the engine never saw the source.
    Hit,
    /// Bytecode was compiled fresh (and stored, unless oversized).
    Miss,
}

/// Bounded LRU cache of compiled module bytecode, keyed by
/// (URL, SHA-256 of source).
#[derive(Debug, Default)]
pub struct BytecodeCache {
    capacity_bytes: usize,
    used_bytes: usize,
    entries: HashMap<CacheKey, Vec<u8>>,
    /// LRU order: front is least-recently-used.
    order: VecDeque<CacheKey>,
    stats: CacheStats,
}

impl BytecodeCache {
    /// A cache holding at most `capacity_bytes` of bytecode in total.
    /// `capacity_bytes == 0` disables storage (every load compiles).
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    /// Current statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            used_bytes: self.used_bytes,
            ..self.stats
        }
    }

    /// Look up compiled bytecode for (`url`, `source`) without compiling.
    /// A hit also refreshes the entry's LRU position.
    pub fn get(&mut self, url: &str, source: &str) -> Option<&[u8]> {
        let key = CacheKey {
            url: url.to_string(),
            hash: hash_source(source),
        };
        if self.entries.contains_key(&key) {
            self.touch(&key);
            self.stats.hits += 1;
            self.entries.get(&key).map(Vec::as_slice)
        } else {
            None
        }
    }

    /// Store compiled `bytecode` for (`url`, `source`), evicting LRU
    /// entries as needed. Oversized entries (larger than the whole
    /// capacity) are dropped, not stored. Re-storing an existing key
    /// replaces it and refreshes its LRU position.
    pub fn put(&mut self, url: &str, source: &str, bytecode: Vec<u8>) {
        if bytecode.len() > self.capacity_bytes {
            return;
        }
        let key = CacheKey {
            url: url.to_string(),
            hash: hash_source(source),
        };
        if let Some(old) = self.entries.insert(key.clone(), bytecode) {
            self.used_bytes -= old.len();
            self.order.retain(|k| k != &key);
        }
        let len = self.entries[&key].len();
        self.used_bytes += len;
        self.order.push_back(key);
        while self.used_bytes > self.capacity_bytes {
            let Some(lru) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&lru) {
                self.used_bytes -= evicted.len();
                self.stats.evictions += 1;
            }
        }
    }

    /// Load a module, compiling on a miss: on a hit, evaluate the cached
    /// bytecode; on a miss, compile the source, store the bytecode, and
    /// evaluate the source. Returns the completion value and whether the
    /// bytecode was reused. This is the call the integration wave makes
    /// per fetched module script.
    pub fn load_or_compile<E: JsEngine>(
        &mut self,
        engine: &mut E,
        url: &str,
        source: &str,
    ) -> Result<(crate::engine::JsValue, CacheHit), JsError> {
        if let Some(bytecode) = self.get(url, source).map(<[u8]>::to_vec) {
            let value = engine.eval_module_bytecode(url, &bytecode)?;
            return Ok((value, CacheHit::Hit));
        }
        self.stats.misses += 1;
        let bytecode = engine.compile_module(url, source)?;
        let value = engine.eval_module(url, source)?;
        self.put(url, source, bytecode);
        Ok((value, CacheHit::Miss))
    }

    /// Evaluate a classic script through the cache's keying only for
    /// bookkeeping: classic scripts are NOT cached (see module docs), so
    /// this always evaluates the source. It exists so callers do not
    /// have to special-case the kind; the miss is counted so the stats
    /// stay honest about what was compiled.
    pub fn load_classic<E: JsEngine>(
        &mut self,
        engine: &mut E,
        source: &str,
        filename: &str,
    ) -> Result<crate::engine::JsValue, JsError> {
        self.stats.misses += 1;
        engine.eval(EvalKind::Classic, source, filename)
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used_bytes = 0;
    }

    fn touch(&mut self, key: &CacheKey) {
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::JsValue;

    /// A fake engine that counts compiles and serves fixed values, so the
    /// cache logic is tested without a real VM.
    struct CountingEngine {
        compiles: usize,
        bytecode_evals: usize,
    }

    impl JsEngine for CountingEngine {
        fn eval(&mut self, _kind: EvalKind, _source: &str, _filename: &str) -> Result<JsValue, JsError> {
            Ok(JsValue::Number(1.0))
        }

        fn eval_module(&mut self, _url: &str, _source: &str) -> Result<JsValue, JsError> {
            Ok(JsValue::Undefined)
        }

        fn compile_module(&mut self, _url: &str, source: &str) -> Result<Vec<u8>, JsError> {
            self.compiles += 1;
            Ok(format!("bytecode:{source}").into_bytes())
        }

        fn eval_module_bytecode(&mut self, _url: &str, _bytecode: &[u8]) -> Result<JsValue, JsError> {
            self.bytecode_evals += 1;
            Ok(JsValue::Undefined)
        }

        fn drain_jobs(&mut self) -> Result<usize, JsError> {
            Ok(0)
        }

        fn memory_usage_bytes(&self) -> u64 {
            0
        }
    }

    #[test]
    fn second_load_of_same_script_reuses_bytecode() {
        let mut cache = BytecodeCache::new(1024 * 1024);
        let mut engine = CountingEngine {
            compiles: 0,
            bytecode_evals: 0,
        };
        let source = "export const x = 1;";
        let (_, hit1) = cache
            .load_or_compile(&mut engine, "https://example.com/a.mjs", source)
            .unwrap();
        assert_eq!(hit1, CacheHit::Miss);
        assert_eq!(engine.compiles, 1);
        let (_, hit2) = cache
            .load_or_compile(&mut engine, "https://example.com/a.mjs", source)
            .unwrap();
        assert_eq!(hit2, CacheHit::Hit);
        assert_eq!(engine.compiles, 1, "second load must not recompile");
        assert_eq!(engine.bytecode_evals, 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn changed_source_under_same_url_recompiles() {
        let mut cache = BytecodeCache::new(1024 * 1024);
        let mut engine = CountingEngine {
            compiles: 0,
            bytecode_evals: 0,
        };
        cache
            .load_or_compile(&mut engine, "https://example.com/a.mjs", "export const x = 1;")
            .unwrap();
        let (_, hit) = cache
            .load_or_compile(&mut engine, "https://example.com/a.mjs", "export const x = 2;")
            .unwrap();
        assert_eq!(hit, CacheHit::Miss);
        assert_eq!(engine.compiles, 2);
        assert_eq!(cache.stats().entries, 2);
    }

    #[test]
    fn eviction_is_lru_and_bounded() {
        // Capacity fits two small entries (28 bytes each) but not three.
        let mut cache = BytecodeCache::new(60);
        let mut engine = CountingEngine {
            compiles: 0,
            bytecode_evals: 0,
        };
        for (url, src) in [
            ("https://example.com/a.mjs", "export const a = 1;"),
            ("https://example.com/b.mjs", "export const b = 2;"),
            ("https://example.com/c.mjs", "export const c = 3;"),
        ] {
            cache.load_or_compile(&mut engine, url, src).unwrap();
        }
        assert!(cache.stats().used_bytes <= 60);
        assert!(cache.stats().evictions >= 1);
        // The first entry was least-recently-used: loading it again misses.
        let (_, hit) = cache
            .load_or_compile(&mut engine, "https://example.com/a.mjs", "export const a = 1;")
            .unwrap();
        assert_eq!(hit, CacheHit::Miss);
        // The most recent entry is still cached.
        let (_, hit) = cache
            .load_or_compile(&mut engine, "https://example.com/c.mjs", "export const c = 3;")
            .unwrap();
        assert_eq!(hit, CacheHit::Hit);
    }

    #[test]
    fn oversized_entry_is_not_stored() {
        let mut cache = BytecodeCache::new(4);
        cache.put("https://example.com/big.mjs", "export const x = 1;", vec![0u8; 100]);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().used_bytes, 0);
    }

    #[test]
    fn hash_is_stable_and_content_sensitive() {
        assert_eq!(hash_source("abc"), hash_source("abc"));
        assert_ne!(hash_source("abc"), hash_source("abd"));
        assert_eq!(hash_source("abc").len(), 32);
    }
}
