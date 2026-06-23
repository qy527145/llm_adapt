//! Streaming event representation.

use serde::{Deserialize, Serialize};

use super::response::FinishReason;
use super::usage::Usage;
use crate::error::LLMError;

/// One unit of streamed information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    /// Incremental text delta.
    TextDelta {
        text: String,
    },
    /// Incremental reasoning/thinking delta. `signature_delta` is set when the
    /// vendor signs thinking content (Anthropic) — concatenating all
    /// `signature_delta` fragments yields the full signature.
    ThinkingDelta {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature_delta: Option<String>,
    },
    /// Incremental tool-call delta. Vendors typically send the `id` and
    /// `name_delta` once at the start of a tool call, then stream
    /// `arguments_delta` fragments which concatenate into a JSON object.
    ToolCallDelta {
        id: String,
        name_delta: Option<String>,
        arguments_delta: Option<String>,
    },
    /// Usage report. Most providers emit this once near the end.
    Usage {
        usage: Usage,
    },
    /// Stream ended. Mirrors the non-streaming `finish_reason`.
    Finish {
        reason: FinishReason,
    },
    /// Provider sent an explicit error frame inside the stream.
    Error {
        error: String,
    },
}

impl StreamChunk {
    pub fn text(text: impl Into<String>) -> Self {
        Self::TextDelta { text: text.into() }
    }
}

/// Convert an in-stream `Error` frame into a fatal [`LLMError`] when the caller
/// wants to bail out of the stream.
pub fn chunk_error_to_llm_error(msg: &str) -> LLMError {
    LLMError::provider(msg)
}
