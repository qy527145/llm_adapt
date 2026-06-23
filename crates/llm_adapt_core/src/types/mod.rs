//! Unified type system shared by every handler.

pub mod cache;
pub mod conversation;
pub mod request;
pub mod response;
pub mod stream;
pub mod tool;
pub mod tool_result;
pub mod usage;

pub use cache::{CacheMarker, CacheTtl};
pub use conversation::{
    AssistantBlock, AssistantMessage, Conversation, SystemBlock, SystemPrompt, Turn, UserBlock,
    UserMessage,
};
pub use request::{ChatRequest, ResponseFormat};
pub use response::{ChatResponse, FinishReason};
pub use stream::StreamChunk;
pub use tool::{ToolChoice, ToolDefinition};
pub use tool_result::{ToolResultBlock, ToolResultContent};
pub use usage::{CacheUsage, CacheWriteUsage, RateLimitInfo, Usage};
