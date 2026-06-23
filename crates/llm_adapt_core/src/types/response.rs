//! Unified non-streaming response.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::conversation::AssistantMessage;
use super::usage::{RateLimitInfo, Usage};

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    Safety,
    StopSequence,
    Other,
}

/// Unified non-streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant's full reply, as a role-aware message.
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub request_id: Option<String>,
    /// Provider-resolved model name (may differ from the request when aliases are used).
    pub actual_model: Option<String>,
    pub latency_ms: u128,
    #[serde(default)]
    pub rate_limit_info: RateLimitInfo,
    /// Free-form provider metadata (e.g. `system_fingerprint`).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChatResponse {
    /// Convenience: concatenated text content of the assistant message.
    pub fn text(&self) -> String {
        self.message.text_content()
    }
}
