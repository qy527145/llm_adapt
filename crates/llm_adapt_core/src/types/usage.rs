//! Token usage and billing breakdown.
//!
//! Designed to mirror the granularity Agent operators care about: separate
//! cache read vs cache write, and split cache writes by TTL tier.

use serde::{Deserialize, Serialize};

/// Per-request token accounting.
///
/// `input_tokens` counts only tokens NOT served from cache, mirroring how the
/// major providers bill base input.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Reasoning / thinking tokens (Anthropic extended thinking, o-series).
    #[serde(default)]
    pub thinking_tokens: u32,
    #[serde(default)]
    pub cache: CacheUsage,
}

impl Usage {
    /// All tokens billed against the prompt side
    /// (base + cache reads + cache writes). Useful for back-of-envelope cost
    /// calculations.
    pub fn total_input_billed(&self) -> u32 {
        self.input_tokens + self.cache.read_tokens + self.cache.write.total
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheUsage {
    /// Tokens served from the provider's prompt cache (cheap reads).
    pub read_tokens: u32,
    pub write: CacheWriteUsage,
}

/// Cache-write accounting, split by TTL tier where the vendor reports it.
///
/// For vendors that do not split by TTL (OpenAI), `total` carries the only
/// reliable number and the two ephemeral counters stay at zero.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheWriteUsage {
    /// Total cache-write tokens across all TTL tiers. Always equals
    /// `ephemeral_5m + ephemeral_1h` when both are reported.
    pub total: u32,
    /// Tokens written into the short-lived (default) cache tier.
    pub ephemeral_5m: u32,
    /// Tokens written into the long-lived premium cache tier.
    pub ephemeral_1h: u32,
}

/// Rate-limit hints surfaced from response headers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub remaining_requests: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub reset_seconds: Option<u64>,
}
