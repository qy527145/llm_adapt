//! High-level facade combining [`HandlerRegistry`], [`HttpExecutor`], and the
//! model capability table.

use std::sync::Arc;
use std::time::Instant;

use crate::capability::{ModelCapabilities, ModelCapabilityTable};
use crate::error::LLMError;
use crate::handler::{ChunkStream, HandlerRegistry, HttpRequest};
use crate::http::{ClientConfig, HttpExecutor, RetryPolicy};
use crate::types::{ChatRequest, ChatResponse};

/// Application-facing client. Inexpensive to clone (Arcs internally).
#[derive(Clone)]
pub struct LLMClient {
    registry: HandlerRegistry,
    executor: Arc<HttpExecutor>,
    capabilities: ModelCapabilityTable,
}

/// Builder for [`LLMClient`].
pub struct LLMClientBuilder {
    config: ClientConfig,
    registry: HandlerRegistry,
    capabilities: ModelCapabilityTable,
    retry: RetryPolicy,
    install_defaults: bool,
}

impl LLMClient {
    pub fn builder(config: ClientConfig) -> LLMClientBuilder {
        LLMClientBuilder {
            config,
            registry: HandlerRegistry::new(),
            capabilities: ModelCapabilityTable::with_defaults(),
            retry: RetryPolicy::default(),
            install_defaults: true,
        }
    }

    /// Quick start: build a default-configured client.
    pub fn new(config: ClientConfig) -> Result<Self, LLMError> {
        Self::builder(config).build()
    }

    pub fn registry(&self) -> &HandlerRegistry {
        &self.registry
    }

    pub fn executor(&self) -> &HttpExecutor {
        &self.executor
    }

    pub fn capabilities(&self) -> &ModelCapabilityTable {
        &self.capabilities
    }

    pub fn lookup_capabilities(&self, request: &ChatRequest) -> ModelCapabilities {
        self.capabilities.get(&request.api_format, &request.model)
    }

    /// Render the unified request into a concrete HTTP request without sending
    /// it. Used by `llm-adapt preview` and the debug UIs.
    pub fn preview(&self, request: &ChatRequest) -> Result<HttpRequest, LLMError> {
        let entry = self.registry.get(&request.api_format)?;
        entry.request.build_request(request, self.executor.config())
    }

    /// Send a non-streaming chat request and parse the response.
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LLMError> {
        if request.stream {
            return Err(LLMError::validation(
                "chat() called with stream=true; use chat_stream() instead",
            ));
        }
        let entry = self.registry.get(&request.api_format)?;
        let http_req = entry.request.build_request(request, self.executor.config())?;
        let start = Instant::now();
        let raw = self.executor.execute(&http_req).await?;
        let mut resp = entry.non_stream.parse_response(raw.body)?;
        resp.latency_ms = start.elapsed().as_millis();
        // Surface the network-level request id if the handler didn't set one.
        if resp.request_id.is_none() {
            for (k, v) in &raw.headers {
                if k.eq_ignore_ascii_case("x-request-id")
                    || k.eq_ignore_ascii_case("openai-request-id")
                    || k.eq_ignore_ascii_case("anthropic-request-id")
                {
                    resp.request_id = Some(v.clone());
                    break;
                }
            }
        }
        Ok(resp)
    }

    /// Send a streaming chat request and return a stream of normalised events.
    pub async fn chat_stream(&self, request: &ChatRequest) -> Result<ChunkStream, LLMError> {
        let entry = self.registry.get(&request.api_format)?;
        let mut effective = request.clone();
        effective.stream = true;
        let http_req = entry.request.build_request(&effective, self.executor.config())?;
        let (_meta, bytes) = self.executor.execute_stream(&http_req).await?;
        Ok(entry.stream.parse_stream(bytes))
    }
}

impl LLMClientBuilder {
    /// Disable automatic registration of feature-gated handlers. Useful when
    /// the caller wants full control over the registry.
    pub fn without_default_handlers(mut self) -> Self {
        self.install_defaults = false;
        self
    }

    pub fn registry(mut self, registry: HandlerRegistry) -> Self {
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
            crate::handlers::register_defaults(&self.registry);
        }
        let executor = HttpExecutor::with_retry(self.config, self.retry)?;
        Ok(LLMClient {
            registry: self.registry,
            executor: Arc::new(executor),
            capabilities: self.capabilities,
        })
    }
}
