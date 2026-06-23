//! Tool execution result content.
//!
//! Used by [`crate::types::UserBlock::ToolResult`]. We model three shapes so
//! callers don't have to lossily stringify multi-modal or structured results
//! when the underlying vendor accepts them natively.

use serde::{Deserialize, Serialize};

/// Body of a tool execution result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultContent {
    /// Plain text — accepted by every vendor.
    Text(String),
    /// Structured JSON. Sent as a serialised string to vendors that don't have
    /// a structured tool-result type.
    Json(serde_json::Value),
    /// Multi-modal blocks. Anthropic accepts this natively; OpenAI degrades to
    /// concatenated text (images logged + dropped).
    Blocks(Vec<ToolResultBlock>),
}

impl ToolResultContent {
    pub fn text(s: impl Into<String>) -> Self {
        ToolResultContent::Text(s.into())
    }

    pub fn json(value: serde_json::Value) -> Self {
        ToolResultContent::Json(value)
    }
}

/// One block inside a multi-modal tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        text: String,
    },
    Image {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}
