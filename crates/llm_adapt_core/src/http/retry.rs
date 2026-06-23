//! Exponential-backoff retry policy used by [`crate::http::HttpExecutor`].

use std::time::Duration;

use crate::error::LLMError;

/// Retry policy. The executor decides whether to retry based on
/// [`should_retry`](Self::should_retry); the actual delay between attempts is
/// computed by [`delay`](Self::delay).
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retries (the initial attempt is *not* counted).
    pub max_retries: u32,
    /// Initial backoff after the first failure.
    pub initial_backoff: Duration,
    /// Cap on the per-attempt backoff.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(8),
        }
    }
}

impl RetryPolicy {
    /// No retries — useful in tests where we want deterministic single-attempt behaviour.
    pub fn none() -> Self {
        Self { max_retries: 0, ..Self::default() }
    }

    pub fn should_retry(&self, attempt: u32, err: &LLMError) -> bool {
        attempt < self.max_retries && err.is_retryable()
    }

    /// Exponential backoff with a hard cap, no jitter (callers can layer jitter
    /// on top if they need it).
    pub fn delay(&self, attempt: u32) -> Duration {
        let exp = attempt.saturating_sub(1);
        let mult = 1u64 << exp.min(20);
        let ms = self.initial_backoff.as_millis() as u64 * mult;
        let capped = ms.min(self.max_backoff.as_millis() as u64);
        Duration::from_millis(capped)
    }
}
