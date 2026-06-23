//! Model capability metadata.
//!
//! Provides a shared, mutable table of `(api_format, model) -> ModelCapabilities`
//! that upper layers (Agents, CLI, debug UIs) consult to decide whether a request
//! is admissible (e.g. fits the context window) or which features to expose.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

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
        api_format: impl Into<String>,
        model: impl Into<String>,
        caps: ModelCapabilities,
    ) {
        let key = (api_format.into(), model.into());
        let mut guard = self.inner.write().expect("ModelCapabilityTable poisoned");
        guard.insert(key, caps);
    }

    /// Lookup. Falls back to [`ModelCapabilities::unknown`] when the key isn't present.
    pub fn get(&self, api_format: &str, model: &str) -> ModelCapabilities {
        let guard = self.inner.read().expect("ModelCapabilityTable poisoned");
        guard
            .get(&(api_format.to_string(), model.to_string()))
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

fn defaults() -> Vec<(&'static str, &'static str, ModelCapabilities)> {
    let openai_4o = ModelCapabilities {
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

    vec![
        ("openai_compat", "gpt-4o", openai_4o.clone()),
        ("openai_compat", "gpt-4o-2024-08-06", openai_4o),
        ("openai_compat", "gpt-4o-mini", openai_4o_mini),
        ("openai_compat", "o1-mini", openai_o1),
        ("anthropic_v2", "claude-3-5-sonnet-20241022", sonnet),
        ("anthropic_v2", "claude-3-5-haiku-20241022", haiku),
        ("anthropic_v2", "claude-3-opus-20240229", opus),
    ]
}
