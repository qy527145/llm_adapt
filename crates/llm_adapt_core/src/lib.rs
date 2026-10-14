//! # llm_adapt_core
//!
//! Production-grade Rust abstraction over LLM provider APIs. Application code
//! talks to one unified, role-aware type system; per-protocol [`protocol::Codec`]
//! implementations translate to and from each vendor wire format.
//!
//! ## Quick start
//!
//! ```no_run
//! use llm_adapt_core::{ChatRequest, ClientConfig, Conversation, LLMClient};
//!
//! # async fn run() -> Result<(), llm_adapt_core::LLMError> {
//! let client = LLMClient::new(ClientConfig::new("https://api.openai.com", "sk-..."))?;
//! let request = ChatRequest::openai("gpt-4o-mini", Conversation::single_user("Hello!"));
//! let resp = client.chat(&request).await?;
//! println!("{}", resp.text());
//! # Ok(()) }
//! ```
//!
//! ## Architecture
//!
//! 分层严格单向：`http`（怎么发出去）→ `protocol`（发什么、回的什么）→
//! `types`（给人用的角色感知 API）。`client` 把三层串起来。
//!
//! * [`protocol`] —— 协议无关的 IR（[`protocol::ir`]）与 [`Codec`] 六方法、
//!   [`ProtocolRegistry`]；内置 anthropic / openai_chat 两个协议
//! * [`http`] —— 与协议无关的 HTTP 执行器（reqwest、可重试、可挂钩子）
//! * [`sse`] —— 统一的 SSE 切帧与驱动
//! * [`types`] —— 角色感知的请求/响应/流模型，一等公民的缓存标记与按 TTL 分档的用量
//! * [`capability`] —— 每个模型的上下文窗口与能力元数据
//! * [`error`] —— 所有可失败操作统一返回 [`LLMError`]
//!
//! ## 两种保真度
//!
//! [`LLMClient`] 提供两组入口：ergonomic（`chat` / `chat_stream`）省事，
//! IR（`chat_ir` / `chat_stream_ir`）保真。IR 路径不会丢掉
//! [`protocol::ir::Block::Unmodeled`] 这类角色感知层表达不了的内容块。

pub mod capability;
pub mod client;
pub mod error;
pub mod http;
pub mod protocol;
pub mod sse;
pub mod types;

pub use capability::{ModelCapabilities, ModelCapabilityTable};
pub use client::{LLMClient, LLMClientBuilder};
pub use error::{ErrorContext, LLMError};
pub use http::{
    ByteStream, ClientConfig, HttpExecutor, HttpMethod, HttpRequest, ProxyConfig, RawHttpResponse,
    RequestHook, RetryPolicy,
};
pub use protocol::ir::{
    Block, CacheMarker, CacheTtl, CacheUsage, CacheWriteUsage, FinishReason, ImageSource,
    ResponseFormat, Role, ThinkingConfig, ToolChoice, ToolKind, ToolResultBlock, ToolResultContent,
    Usage, UsageSource, WireMessage, WireRequest, WireResponse,
};
pub use protocol::{
    decode_stream, Codec, Protocol, ProtocolId, ProtocolRegistry, StreamEncoder,
};
pub use sse::{ChunkStream, SseChunkStream, SseFrame, SseMapper, SseOutcome};
pub use types::{
    AssistantBlock, AssistantMessage, BlockAssembler, ChatRequest, ChatResponse, Conversation,
    RateLimitInfo, StreamChunk, SystemBlock, SystemPrompt, ToolDefinition, Turn, UserBlock,
    UserMessage,
};
