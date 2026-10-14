//! 所有 handler 共用的统一类型系统。

pub mod cache;
pub mod conversation;
pub mod convert;
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
pub use convert::{lift_request, lower_response};
pub use request::{ChatRequest, ResponseFormat};
pub use response::{ChatResponse, FinishReason};
pub use stream::{BlockAssembler, StreamChunk};
pub use tool::{ToolChoice, ToolDefinition};
pub use tool_result::{ToolResultBlock, ToolResultContent};
pub use usage::{CacheUsage, CacheWriteUsage, RateLimitInfo, Usage, UsageSource};
