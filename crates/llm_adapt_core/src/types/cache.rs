//! Prompt-cache markers.
//!
//! Most vendors expose explicit caching via per-block markers. We model only
//! the TTLs that are exposed in pricing today (5 minutes — the default — and
//! 1 hour). Handlers that don't understand caching ignore the marker.

use serde::{Deserialize, Serialize};

/// How long the cache entry should live on the provider side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    /// Default ephemeral cache (~5 minutes on Anthropic).
    Ephemeral5m,
    /// Long-lived ephemeral cache (~1 hour on Anthropic). Premium pricing.
    Ephemeral1h,
}

/// A block-level opt-in to caching the prefix up to and including this block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMarker {
    pub ttl: CacheTtl,
}

impl CacheMarker {
    pub fn ephemeral_5m() -> Self {
        Self { ttl: CacheTtl::Ephemeral5m }
    }

    pub fn ephemeral_1h() -> Self {
        Self { ttl: CacheTtl::Ephemeral1h }
    }
}
