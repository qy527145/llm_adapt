//! Model capability metadata.
//!
//! Provides a shared, mutable table of `(api_format, model) -> ModelCapabilities`
//! that upper layers (Agents, CLI, debug UIs) consult to decide whether a request
//! is admissible (e.g. fits the context window) or which features to expose.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::protocol::ir::ProtocolId;

/// Capability and quota metadata for a single model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCapabilities {
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    pub supports_vision: bool,
    pub supports_prompt_cache: bool,
    pub max_parallel_tool_calls: u32,
}

impl ModelCapabilities {
    /// Fallback used when no entry is registered for a model — conservative
    /// values that won't lie about capability presence.
    pub fn unknown() -> Self {
        Self {
            context_window: 8_192,
            max_output_tokens: 4_096,
            supports_tools: false,
            supports_thinking: false,
            supports_vision: false,
            supports_prompt_cache: false,
            max_parallel_tool_calls: 0,
        }
    }
}

/// Thread-safe capability table. Cloning is cheap; mutation goes through `RwLock`.
#[derive(Default, Clone)]
pub struct ModelCapabilityTable {
    inner: Arc<RwLock<HashMap<(String, String), ModelCapabilities>>>,
}

impl ModelCapabilityTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-populated table with capabilities for the major models we test against.
    pub fn with_defaults() -> Self {
        let table = Self::new();
        for (api, model, caps) in defaults() {
            table.insert(api, model, caps);
        }
        table
    }

    pub fn insert(
        &self,
        protocol: impl AsRef<str>,
        model: impl Into<String>,
        caps: ModelCapabilities,
    ) {
        let key = (ProtocolId::new(protocol).to_string(), model.into());
        let mut guard = self.inner.write().expect("ModelCapabilityTable poisoned");
        guard.insert(key, caps);
    }

    /// Lookup. Falls back to [`ModelCapabilities::unknown`] when the key isn't present.
    ///
    /// 协议名先做别名归一，所以老配置里的 `anthropic_v2` / `openai_compat`
    /// 与新名 `anthropic` / `openai_chat` 都能命中同一行。
    pub fn get(&self, protocol: &str, model: &str) -> ModelCapabilities {
        let key = (ProtocolId::new(protocol).to_string(), model.to_string());
        let guard = self.inner.read().expect("ModelCapabilityTable poisoned");
        guard
            .get(&key)
            .cloned()
            .unwrap_or_else(ModelCapabilities::unknown)
    }

    pub fn list(&self) -> Vec<((String, String), ModelCapabilities)> {
        let guard = self.inner.read().expect("ModelCapabilityTable poisoned");
        let mut entries: Vec<_> = guard
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }
}

fn defaults() -> Vec<(&'static str, &'static str, ModelCapabilities)> {    let openai_4o = ModelCapabilities {
        context_window: 128_000,
        max_output_tokens: 16_384,
        supports_tools: true,
        supports_thinking: false,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    let openai_4o_mini = ModelCapabilities {
        context_window: 128_000,
        max_output_tokens: 16_384,
        supports_tools: true,
        supports_thinking: false,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    let openai_o1 = ModelCapabilities {
        context_window: 128_000,
        max_output_tokens: 65_536,
        supports_tools: true,
        supports_thinking: true,
        supports_vision: false,
        supports_prompt_cache: false,
        max_parallel_tool_calls: 1,
    };
    let sonnet = ModelCapabilities {
        context_window: 200_000,
        max_output_tokens: 8_192,
        supports_tools: true,
        supports_thinking: true,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    let haiku = ModelCapabilities {
        context_window: 200_000,
        max_output_tokens: 8_192,
        supports_tools: true,
        supports_thinking: false,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    let opus = ModelCapabilities {
        context_window: 200_000,
        max_output_tokens: 4_096,
        supports_tools: true,
        supports_thinking: false,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    // Responses 侧的旗舰模型。OpenAI 的提示词缓存是自动的，没有显式断点，
    // 这里仍标 true —— 它表示「这个模型会命中缓存」，不是「能打断点」。
    let gpt5 = ModelCapabilities {
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_tools: true,
        supports_thinking: true,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };
    let gpt5_mini = ModelCapabilities {
        context_window: 400_000,
        max_output_tokens: 128_000,
        supports_tools: true,
        supports_thinking: true,
        supports_vision: true,
        supports_prompt_cache: true,
        max_parallel_tool_calls: 16,
    };

    vec![
        (ProtocolId::OPENAI_CHAT, "gpt-4o", openai_4o.clone()),
        (ProtocolId::OPENAI_CHAT, "gpt-4o-2024-08-06", openai_4o),
        (ProtocolId::OPENAI_CHAT, "gpt-4o-mini", openai_4o_mini),
        (ProtocolId::OPENAI_CHAT, "o1-mini", openai_o1),
        (ProtocolId::ANTHROPIC, "claude-3-5-sonnet-20241022", sonnet),
        (ProtocolId::ANTHROPIC, "claude-3-5-haiku-20241022", haiku),
        (ProtocolId::ANTHROPIC, "claude-3-opus-20240229", opus),
        (ProtocolId::OPENAI_RESPONSES, "gpt-5", gpt5),
        (ProtocolId::OPENAI_RESPONSES, "gpt-5-mini", gpt5_mini),
    ]
}
