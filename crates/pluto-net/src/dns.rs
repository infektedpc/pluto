//! Pluto-owned DNS TTL cache (Stage 11 Rec 6).
//!
//! The shared client's connector used to resolve through the OS
//! `GaiResolver` on every connect — hyper-util's
//! `connect/dns.rs` calls `getaddrinfo` per connection, and the
//! `client.rs:116` comment claiming a DNS cache lived behind the
//! shared client was wrong. This module puts a fixed-TTL cache in
//! front of `GaiResolver`: `CachedResolver` implements
//! `tower::Service<Name>` (hyper-util's sealed `Resolve` is
//! implemented for any such service), and `HttpConnector` is built
//! with it in [`crate::client::build_connector`].
//!
//! TTLs are fixed policy, not the record's TTL: `getaddrinfo`
//! returns no TTL, so there is nothing to honour. Positive
//! answers live 60 s, failures 5 s. A negative entry only
//! suppresses re-resolution for 5 s — a transient `EAI_AGAIN` is
//! retried quickly, and a genuinely missing name is not
//! re-queried on every subresource of the same page.

use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use hyper_util::client::legacy::connect::dns::{GaiResolver, Name};
use tower_service::Service;

/// Positive DNS answers live this long in the cache.
pub const DNS_POSITIVE_TTL: Duration = Duration::from_secs(60);
/// Failed resolutions live this long (negative caching).
pub const DNS_NEGATIVE_TTL: Duration = Duration::from_secs(5);

#[derive(Debug)]
enum DnsEntry {
    Positive {
        addrs: Vec<SocketAddr>,
        expires: Instant,
    },
    Negative {
        expires: Instant,
    },
}

/// A fixed-TTL DNS cache: host → addresses or a cached failure.
/// Clone shares the table. `get` / `insert_*` are the direct
/// API (used by `dns_prefetch` and the tests); [`CachedResolver`]
/// is the `Service<Name>` front end the connector uses.
#[derive(Clone, Debug, Default)]
pub struct DnsCache {
    inner: Arc<Mutex<HashMap<String, DnsEntry>>>,
}

impl DnsCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look a host up without resolving: `Some(Ok(addrs))` for a
    /// fresh positive entry, `Some(Err(()))` for a fresh negative
    /// one, `None` on a miss or an expired entry (expired entries
    /// are evicted on sight).
    pub fn get(&self, host: &str) -> Option<Result<Vec<SocketAddr>, ()>> {
        let mut map = self.inner.lock().unwrap();
        match map.get(host) {
            Some(DnsEntry::Positive { addrs, expires }) if *expires > Instant::now() => {
                Some(Ok(addrs.clone()))
            }
            Some(DnsEntry::Negative { expires }) if *expires > Instant::now() => Some(Err(())),
            Some(_) => {
                map.remove(host);
                None
            }
            None => None,
        }
    }

    /// Record a successful resolution.
    pub fn insert_positive(&self, host: &str, addrs: Vec<SocketAddr>) {
        self.inner.lock().unwrap().insert(
            host.to_string(),
            DnsEntry::Positive {
                addrs,
                expires: Instant::now() + DNS_POSITIVE_TTL,
            },
        );
    }

    /// Record a failed resolution (negative caching).
    pub fn insert_negative(&self, host: &str) {
        self.inner.lock().unwrap().insert(
            host.to_string(),
            DnsEntry::Negative {
                expires: Instant::now() + DNS_NEGATIVE_TTL,
            },
        );
    }

    /// Number of unexpired entries (expired ones are evicted).
    pub fn len(&self) -> usize {
        let mut map = self.inner.lock().unwrap();
        map.retain(|_, e| match e {
            DnsEntry::Positive { expires, .. } | DnsEntry::Negative { expires } => {
                *expires > Instant::now()
            }
        });
        map.len()
    }

    /// Whether the cache holds no unexpired entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A `tower::Service<Name>` resolver that answers from a
/// [`DnsCache`] and falls back to the OS `GaiResolver` (Stage 11
/// Rec 6). Clone shares the cache — the shared client and the
/// preconnect path resolve through the same table.
#[derive(Clone)]
pub struct CachedResolver {
    cache: DnsCache,
    inner: GaiResolver,
}

impl CachedResolver {
    pub fn new(cache: DnsCache) -> Self {
        Self {
            cache,
            inner: GaiResolver::new(),
        }
    }

    /// The cache this resolver reads and writes.
    pub fn cache(&self) -> &DnsCache {
        &self.cache
    }
}

impl Service<Name> for CachedResolver {
    type Response = std::vec::IntoIter<SocketAddr>;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, name: Name) -> Self::Future {
        let cache = self.cache.clone();
        let mut inner = self.inner.clone();
        let host = name.as_str().to_string();
        Box::pin(async move {
            match cache.get(&host) {
                Some(Ok(addrs)) => {
                    eprintln!("[pluto-broker] dns cache hit host={} addrs={}", host, addrs.len());
                    Ok(addrs.into_iter())
                }
                Some(Err(())) => {
                    eprintln!("[pluto-broker] dns cache negative host={}", host);
                    Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("cached DNS failure for {host}"),
                    ))
                }
                None => {
                    eprintln!("[pluto-broker] dns resolve start host={}", host);
                    match inner.call(name).await {
                        Ok(addrs) => {
                            let addrs: Vec<SocketAddr> = addrs.collect();
                            cache.insert_positive(&host, addrs.clone());
                            eprintln!("[pluto-broker] dns resolve ok host={} addrs={}", host, addrs.len());
                            Ok(addrs.into_iter())
                        }
                        Err(e) => {
                            cache.insert_negative(&host);
                            eprintln!("[pluto-broker] dns resolve err host={} err={}", host, e);
                            Err(e)
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn addr(ip: [u8; 4], port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::from(ip)), port)
    }

    #[test]
    fn positive_entries_hit_until_they_expire() {
        let cache = DnsCache::new();
        assert!(cache.get("example.com").is_none());
        cache.insert_positive("example.com", vec![addr([93, 184, 216, 34], 0)]);
        assert_eq!(
            cache.get("example.com").unwrap().unwrap(),
            vec![addr([93, 184, 216, 34], 0)]
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn negative_entries_suppress_re_resolution() {
        let cache = DnsCache::new();
        cache.insert_negative("missing.example");
        assert!(cache.get("missing.example").unwrap().is_err());
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn cached_resolver_serves_from_cache_without_touching_the_os() {
        let cache = DnsCache::new();
        cache.insert_positive("cached.example", vec![addr([127, 0, 0, 1], 0)]);
        let mut resolver = CachedResolver::new(cache.clone());
        let name: Name = "cached.example".parse().unwrap();
        let addrs: Vec<SocketAddr> = resolver.call(name).await.unwrap().collect();
        assert_eq!(addrs, vec![addr([127, 0, 0, 1], 0)]);
        // A cached failure is reported without a lookup either.
        cache.insert_negative("gone.example");
        let name: Name = "gone.example".parse().unwrap();
        assert!(resolver.call(name).await.is_err());
    }

    #[tokio::test]
    async fn cached_resolver_falls_back_to_the_os_for_localhost() {
        // `localhost` resolves via the OS (hosts file / NSS) and
        // the answer lands in the cache for the next call.
        let cache = DnsCache::new();
        let mut resolver = CachedResolver::new(cache.clone());
        let name: Name = "localhost".parse().unwrap();
        let addrs: Vec<SocketAddr> = resolver.call(name).await.unwrap().collect();
        assert!(!addrs.is_empty(), "localhost must resolve");
        assert!(addrs.iter().any(|a| a.ip().is_loopback()));
        assert_eq!(cache.len(), 1);
        let name: Name = "localhost".parse().unwrap();
        let again: Vec<SocketAddr> = resolver.call(name).await.unwrap().collect();
        assert_eq!(again, addrs, "second call is a cache hit");
    }
}
