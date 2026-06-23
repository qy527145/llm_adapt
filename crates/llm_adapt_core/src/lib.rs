//! # llm_adapt_core
//!
//! Production-grade Rust abstraction over LLM provider APIs. Application code
//! talks to one unified, role-aware type system; per-provider
//! [`Handler`](handler) implementations translate to and from vendor wire
//! formats.
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
//! * [`types`] — unified, role-aware request/response/stream model with
//!   first-class cache markers and TTL-split usage
//! * [`error`] — single [`LLMError`] enum surfaced by all fallible operations
//! * [`handler`] — three handler traits (request / non-stream / stream) and a
//!   thread-safe [`HandlerRegistry`]
//! * [`http`] — provider-agnostic HTTP executor (reqwest-based, retry-aware,
//!   hookable)
//! * [`handlers`] — built-in OpenAI- and Anthropic-compatible handlers
//! * [`capability`] — per-model capability metadata
//! * [`client`] — high-level facade [`LLMClient`] tying everything together

pub mod capability;
pub mod client;
pub mod error;
pub mod handler;
pub mod handlers;
pub mod http;
pub mod types;

pub use capability::{ModelCapabilities, ModelCapabilityTable};
pub use client::{LLMClient, LLMClientBuilder};
pub use error::{ErrorContext, LLMError};
pub use handler::{
    ByteStream, ChunkStream, HandlerEntry, HandlerRegistry, HttpMethod, HttpRequest,
    NonStreamResponseHandler, RawHttpResponse, RequestHandler, StreamResponseHandler,
};
pub use http::{ClientConfig, HttpExecutor, ProxyConfig, RequestHook, RetryPolicy};
pub use types::{
    AssistantBlock, AssistantMessage, CacheMarker, CacheTtl, CacheUsage, CacheWriteUsage,
    ChatRequest, ChatResponse, Conversation, FinishReason, RateLimitInfo, ResponseFormat,
    StreamChunk, SystemBlock, SystemPrompt, ToolChoice, ToolDefinition, ToolResultBlock,
    ToolResultContent, Turn, Usage, UserBlock, UserMessage,
};
