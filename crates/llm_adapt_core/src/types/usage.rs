//! 用量与限流。
//!
//! [`Usage`] 一族是 wire 概念（各家协议怎么报 token），定义在 IR 层，这里再导出。
//! [`RateLimitInfo`] 从响应头读出、属于应用侧关心的东西，留在这里。

pub use crate::protocol::ir::{CacheUsage, CacheWriteUsage, Usage, UsageSource};

use serde::{Deserialize, Serialize};

/// 从响应头里读出的限流提示。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub remaining_requests: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub reset_seconds: Option<u64>,
}
