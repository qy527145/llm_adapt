//! Unified request representation.

use serde::{Deserialize, Serialize};

use super::conversation::Conversation;
use super::tool::{ToolChoice, ToolDefinition};

/// Top-level request shared by every provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Provider-side model identifier.
    pub model: String,
    /// Routing key into [`crate::handler::HandlerRegistry`].
    pub api_format: String,
    /// Role-aware conversation history.
    pub conversation: Conversation,

    // ---- generation controls ----
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

    // ---- capability toggles ----
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub enable_thinking: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,

    // ---- tooling ----
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: ToolChoice,

    // ---- response shape ----
    #[serde(default)]
    pub response_format: ResponseFormat,

    // ---- tracing ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl ChatRequest {
    /// Common-case constructor: OpenAI-compatible request with the given
    /// conversation.
    pub fn openai(model: impl Into<String>, conversation: Conversation) -> Self {
        Self::new(model, "openai_compat", conversation)
    }

    /// Common-case constructor: Anthropic Messages request. `max_tokens`
    /// defaults to 1024 because Anthropic requires it.
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

/// Desired shape of the assistant's response.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    Json,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: bool,
    },
}
