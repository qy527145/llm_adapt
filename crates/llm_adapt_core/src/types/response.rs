//! 统一的非流式响应。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::conversation::AssistantMessage;
use super::usage::{RateLimitInfo, Usage};

// FinishReason 定义在 IR 层 —— 各家协议 stop_reason 字符串的归一表也在那边。
// 注意它现在带 `Other { value }`，不再有损。
pub use crate::protocol::ir::FinishReason;

/// 统一的非流式响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// 助手的完整回复，表示为角色感知的消息。
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub request_id: Option<String>,
    /// 提供商解析出的模型名（使用别名时可能与请求里的不同）。
    pub actual_model: Option<String>,
    pub latency_ms: u128,
    #[serde(default)]
    pub rate_limit_info: RateLimitInfo,
    /// 提供商的自定义元数据（如 `system_fingerprint`）。
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChatResponse {
    /// 便捷方法：助手消息的文本内容拼接结果。
    pub fn text(&self) -> String {
        self.message.text_content()
    }
}
