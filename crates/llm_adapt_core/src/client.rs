//! 高层门面：把协议注册表、HTTP 执行器与模型能力表串起来。
//!
//! 两组入口，按需要的保真度选：
//!
//! * **ergonomic**（`chat` / `chat_stream` / `preview`）—— 吃 `ChatRequest`，吐
//!   `ChatResponse`。省事，但角色感知层表达不了的内容块会在 `lower` 时被丢掉。
//! * **IR**（`chat_ir` / `chat_stream_ir` / `preview_ir`）—— 直接吃吐 IR。
//!   `Unmodeled` / `RedactedThinking` / 服务端工具的活动块全程保留。
//!
//! 需要保真的调用方（比如 agent 客户端）应该用 IR 那组。

use std::sync::Arc;
use std::time::Instant;

use crate::capability::{ModelCapabilities, ModelCapabilityTable};
use crate::error::LLMError;
use crate::http::{ClientConfig, HttpExecutor, HttpRequest, RawHttpResponse, RetryPolicy};
use crate::protocol::ir::{WireRequest, WireResponse};
use crate::protocol::{decode_stream, Codec, ProtocolId, ProtocolRegistry};
use crate::sse::ChunkStream;
use crate::types::{lift_request, lower_response, ChatRequest, ChatResponse};

/// Application-facing client. Inexpensive to clone (Arcs internally).
#[derive(Clone)]
pub struct LLMClient {
    registry: ProtocolRegistry,
    executor: Arc<HttpExecutor>,
    capabilities: ModelCapabilityTable,
}

/// Builder for [`LLMClient`].
pub struct LLMClientBuilder {
    config: ClientConfig,
    registry: ProtocolRegistry,
    capabilities: ModelCapabilityTable,
    retry: RetryPolicy,
    install_defaults: bool,
}

impl LLMClient {
    pub fn builder(config: ClientConfig) -> LLMClientBuilder {
        LLMClientBuilder {
            config,
            registry: ProtocolRegistry::new(),
            capabilities: ModelCapabilityTable::with_defaults(),
            retry: RetryPolicy::default(),
            install_defaults: true,
        }
    }

    /// Quick start: build a default-configured client.
    pub fn new(config: ClientConfig) -> Result<Self, LLMError> {
        Self::builder(config).build()
    }

    pub fn registry(&self) -> &ProtocolRegistry {
        &self.registry
    }

    pub fn executor(&self) -> &HttpExecutor {
        &self.executor
    }

    pub fn capabilities(&self) -> &ModelCapabilityTable {
        &self.capabilities
    }

    /// 取某个协议的 codec。未注册时报 [`LLMError::HandlerNotFound`]。
    pub fn codec(&self, protocol: &ProtocolId) -> Result<Arc<dyn Codec>, LLMError> {
        self.registry.get(protocol)
    }

    pub fn lookup_capabilities(&self, request: &ChatRequest) -> ModelCapabilities {
        self.capabilities
            .get(&request.api_format, &request.model)
    }

    // ------------------------------------------------------------------
    // IR 入口（保真）
    // ------------------------------------------------------------------

    /// 把一个 IR 请求编码成具体 HTTP 请求，但不发送。`preview` 的基础。
    pub fn preview_ir(
        &self,
        protocol: &ProtocolId,
        request: &WireRequest,
    ) -> Result<HttpRequest, LLMError> {
        let codec = self.codec(protocol)?;
        codec.encode_request(request, self.executor.config())
    }

    /// 非流式调用，走 IR。返回的 [`WireResponse`] 不丢任何块。
    pub async fn chat_ir(
        &self,
        protocol: &ProtocolId,
        request: &WireRequest,
    ) -> Result<WireResponse, LLMError> {
        let codec = self.codec(protocol)?;
        let mut effective = request.clone();
        effective.stream = false;
        let http_req = codec.encode_request(&effective, self.executor.config())?;
        let raw = self.executor.execute(&http_req).await?;
        // 先把兜底 id 取出来，body 随后会被移动进解码器。
        let fallback_id = request_id_from_headers(&raw);
        let mut resp = codec.decode_response(raw.body)?;
        if resp.id.is_none() {
            resp.id = fallback_id;
        }
        Ok(resp)
    }

    /// 流式调用，走 IR。`stream` 会被强制置为 true。
    pub async fn chat_stream_ir(
        &self,
        protocol: &ProtocolId,
        request: &WireRequest,
    ) -> Result<ChunkStream, LLMError> {
        let codec = self.codec(protocol)?;
        let mut effective = request.clone();
        effective.stream = true;
        let http_req = codec.encode_request(&effective, self.executor.config())?;
        let (_meta, bytes) = self.executor.execute_stream(&http_req).await?;
        Ok(decode_stream(&*codec, bytes))
    }

    // ------------------------------------------------------------------
    // ergonomic 入口（省事，但 lower 时可能丢块）
    // ------------------------------------------------------------------

    /// Render the unified request into a concrete HTTP request without sending
    /// it. Used by `llm-adapt preview` and the debug UIs.
    pub fn preview(&self, request: &ChatRequest) -> Result<HttpRequest, LLMError> {
        let protocol = ProtocolId::new(&request.api_format);
        self.preview_ir(&protocol, &lift_request(request))
    }

    /// Send a non-streaming chat request and parse the response.
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LLMError> {
        if request.stream {
            return Err(LLMError::validation(
                "chat() called with stream=true; use chat_stream() instead",
            ));
        }
        let protocol = ProtocolId::new(&request.api_format);
        let start = Instant::now();
        let ir = self.chat_ir(&protocol, &lift_request(request)).await?;
        let mut resp = lower_response(ir);
        resp.latency_ms = start.elapsed().as_millis();
        Ok(resp)
    }

    /// Send a streaming chat request and return a stream of normalised events.
    pub async fn chat_stream(&self, request: &ChatRequest) -> Result<ChunkStream, LLMError> {
        let protocol = ProtocolId::new(&request.api_format);
        self.chat_stream_ir(&protocol, &lift_request(request)).await
    }
}

/// 从响应头里找上游的请求 id。各家头名不同，逐个认。
fn request_id_from_headers(raw: &RawHttpResponse) -> Option<String> {
    for (k, v) in &raw.headers {
        if k.eq_ignore_ascii_case("x-request-id")
            || k.eq_ignore_ascii_case("openai-request-id")
            || k.eq_ignore_ascii_case("anthropic-request-id")
            || k.eq_ignore_ascii_case("request-id")
        {
            return Some(v.clone());
        }
    }
    None
}

impl LLMClientBuilder {
    /// Disable automatic registration of feature-gated handlers. Useful when
    /// the caller wants full control over the registry.
    pub fn without_default_handlers(mut self) -> Self {
        self.install_defaults = false;
        self
    }

    pub fn registry(mut self, registry: ProtocolRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn capabilities(mut self, table: ModelCapabilityTable) -> Self {
        self.capabilities = table;
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    pub fn config(mut self, config: ClientConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<LLMClient, LLMError> {
        if self.install_defaults {
            crate::protocol::register_defaults(&self.registry);
        }
        let executor = HttpExecutor::with_retry(self.config, self.retry)?;
        Ok(LLMClient {
            registry: self.registry,
            executor: Arc::new(executor),
            capabilities: self.capabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Conversation;

    #[test]
    fn preview_renders_without_sending() {
        let client = LLMClient::new(ClientConfig::new("https://api.openai.com", "sk-x")).unwrap();
        let req = ChatRequest::openai("gpt-4o", Conversation::single_user("hi"));
        let http = client.preview(&req).unwrap();
        assert_eq!(http.url, "https://api.openai.com/v1/chat/completions");
        assert!(http.body_as_str().unwrap().contains("\"hi\""));
    }

    /// 老配置里的 `anthropic_v2` / `openai_compat` 必须仍然找得到 codec。
    #[test]
    fn legacy_api_format_aliases_still_resolve() {
        let client = LLMClient::new(ClientConfig::new("https://x", "k")).unwrap();
        for alias in ["anthropic_v2", "anthropic", "openai_compat", "openai_chat"] {
            let protocol = ProtocolId::new(alias);
            assert!(client.codec(&protocol).is_ok(), "别名 {alias} 应能解析");
        }
    }

    #[test]
    fn unknown_protocol_reports_handler_not_found() {
        let client = LLMClient::new(ClientConfig::new("https://x", "k")).unwrap();
        let err = match client.codec(&ProtocolId::new("nope")) {
            Ok(_) => panic!("未注册的协议不该解析成功"),
            Err(e) => e,
        };
        assert!(matches!(err, LLMError::HandlerNotFound { .. }));
    }

    #[test]
    fn capabilities_lookup_tolerates_aliases() {
        let client = LLMClient::new(ClientConfig::new("https://x", "k")).unwrap();
        let req = ChatRequest::anthropic(
            "claude-3-5-sonnet-20241022",
            Conversation::single_user("hi"),
        );
        let caps = client.lookup_capabilities(&req);
        assert!(caps.supports_tools, "应命中 anthropic 的能力表");
        assert_eq!(caps.context_window, 200_000);
    }
}
