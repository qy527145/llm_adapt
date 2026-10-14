//! 发给模型的工具定义。
//!
//! 工具的*调用*与*结果*在别处 —— 调用见
//! [`crate::types::AssistantBlock::ToolCall`]，结果见
//! [`crate::types::UserBlock::ToolResult`]。

use serde::{Deserialize, Serialize};

use super::cache::CacheMarker;

/// 模型可以调用的一个工具。`parameters` 是描述该工具参数的 JSON Schema 对象。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// 可选的缓存提示。支持工具定义缓存的厂商（Anthropic）会把它应用到
    /// 工具列表前缀上，直到并包含这个定义。
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

/// 模型挑选工具的积极程度。
///
/// 定义在 IR 层，这里再导出。
pub use crate::protocol::ir::ToolChoice;
