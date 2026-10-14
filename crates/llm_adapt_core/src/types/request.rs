//! 统一的请求表示。

use serde::{Deserialize, Serialize};

use super::conversation::Conversation;
use super::tool::ToolDefinition;

// 这两个是 wire 概念（决定请求体长什么样），定义在 IR 层，这里再导出。
pub use crate::protocol::ir::{ResponseFormat, ToolChoice};

/// 所有提供商共用的顶层请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// 提供商侧的模型标识。
    pub model: String,
    /// 用于路由的键，对应 [`crate::handler::HandlerRegistry`]。
    pub api_format: String,
    /// 角色感知的对话历史。
    pub conversation: Conversation,

    // ---- 生成参数 ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    // ---- 能力开关 ----
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub enable_thinking: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,

    // ---- 工具 ----
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: ToolChoice,

    // ---- 响应形状 ----
    #[serde(default)]
    pub response_format: ResponseFormat,

    // ---- 追踪 ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl ChatRequest {
    /// 常见场景构造：给定对话，构造 OpenAI 兼容请求。
    pub fn openai(model: impl Into<String>, conversation: Conversation) -> Self {
        Self::new(model, "openai_compat", conversation)
    }

    /// 常见场景构造：Anthropic Messages 请求。`max_tokens` 默认 1024，
    /// 因为 Anthropic 要求必须带它。
    pub fn anthropic(model: impl Into<String>, conversation: Conversation) -> Self {
        let mut req = Self::new(model, "anthropic_v2", conversation);
        req.max_tokens = Some(1024);
        req
    }

    pub fn new(
        model: impl Into<String>,
        api_format: impl Into<String>,
        conversation: Conversation,
    ) -> Self {
        Self {
            model: model.into(),
            api_format: api_format.into(),
            conversation,
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            max_tokens: None,
            seed: None,
            stream: false,
            enable_thinking: false,
            thinking_budget: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
            response_format: ResponseFormat::default(),
            user_id: None,
        }
    }
}
