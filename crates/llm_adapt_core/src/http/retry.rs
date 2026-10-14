//! [`crate::http::HttpExecutor`] 使用的指数退避重试策略。

use std::time::Duration;

use crate::error::LLMError;

/// 重试策略。执行器依据 [`should_retry`](Self::should_retry) 决定是否重试；
/// 每次尝试之间的实际延迟由 [`delay`](Self::delay) 计算。
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// 最大重试次数（首次尝试*不计*在内）。
    pub max_retries: u32,
    /// 首次失败后的初始退避时长。
    pub initial_backoff: Duration,
    /// 单次退避的上限。
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
    /// 不重试 —— 测试里想要确定的「只尝试一次」行为时有用。
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    pub fn should_retry(&self, attempt: u32, err: &LLMError) -> bool {
        attempt < self.max_retries && err.is_retryable()
    }

    /// 带硬上限的指数退避，不加抖动（调用方需要的话可以自己叠一层抖动）。
    pub fn delay(&self, attempt: u32) -> Duration {
        let exp = attempt.saturating_sub(1);
        let mult = 1u64 << exp.min(20);
        let ms = self.initial_backoff.as_millis() as u64 * mult;
        let capped = ms.min(self.max_backoff.as_millis() as u64);
        Duration::from_millis(capped)
    }
}
