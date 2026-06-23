//! Tool definitions sent to the model.
//!
//! Tool *calls* and *results* live elsewhere — calls on
//! [`crate::types::AssistantBlock::ToolCall`], results on
//! [`crate::types::UserBlock::ToolResult`].

use serde::{Deserialize, Serialize};

use super::cache::CacheMarker;

/// A tool the model may invoke. `parameters` is a JSON Schema object describing
/// the tool's arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Optional cache hint. Vendors that support tool-definition caching
    /// (Anthropic) apply it to the tool list prefix up to and including this
    /// definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMarker>,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            cache: None,
        }
    }

    pub fn with_cache(mut self, marker: CacheMarker) -> Self {
        self.cache = Some(marker);
        self
    }
}

/// How aggressively the model should pick tools.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides freely (default).
    #[default]
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call some tool (any).
    Required,
    /// Model must call this specific tool.
    Specific { name: String },
}
