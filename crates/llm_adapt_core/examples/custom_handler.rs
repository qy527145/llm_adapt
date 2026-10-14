//! `cargo run -p llm_adapt_core --example custom_handler`
//!
//! 演示如何注册一个自定义协议：包一层内置 codec，在每个请求上注入 `X-Trace-Id` 头。
//!
//! 这是「自定义协议」最常见的形态 —— 大多数兼容厂商不过是某个内置协议
//! 加几个头、几个字段偏移。真要从头写一个协议，实现同一个 [`Codec`] 即可；
//! 入站方向的三个方法不实现就用默认值（返回 `Unsupported`）。

use std::sync::Arc;

use bytes::Bytes;
use llm_adapt_core::{
    ChatRequest, ClientConfig, Codec, Conversation, HttpRequest, LLMClient, LLMError, ProtocolId,
    ProtocolRegistry, SseMapper, WireRequest, WireResponse,
};

/// 包一层 codec，只改请求头。
struct TracedCodec {
    inner: Arc<dyn Codec>,
    /// 注册表里的键 —— 与 `inner.protocol()` **不同**，否则会覆盖掉内置协议。
    id: ProtocolId,
}

impl Codec for TracedCodec {
    fn protocol(&self) -> ProtocolId {
        self.id.clone()
    }

    fn encode_request(
        &self,
        request: &WireRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        let mut http = self.inner.encode_request(request, config)?;
        http.headers
            .push(("x-trace-id".into(), format!("trace-{}", request.model)));
        Ok(http)
    }

    fn decode_response(&self, body: Bytes) -> Result<WireResponse, LLMError> {
        self.inner.decode_response(body)
    }

    fn new_stream_decoder(&self) -> Box<dyn SseMapper> {
        self.inner.new_stream_decoder()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 先拿到内置的 openai_chat codec 作为内层实现。
    let builtins = ProtocolRegistry::new();
    llm_adapt_core::protocol::register_defaults(&builtins);
    let inner = builtins.get(&ProtocolId::new("openai_chat"))?;

    let registry = ProtocolRegistry::new();
    registry.register(Arc::new(TracedCodec {
        inner,
        id: ProtocolId::new("openai_traced"),
    }));

    let client = LLMClient::builder(ClientConfig::new("https://api.openai.com", "sk-demo"))
        .without_default_handlers()
        .registry(registry)
        .build()?;

    let mut request = ChatRequest::openai("gpt-4o-mini", Conversation::single_user("Hi"));
    request.api_format = "openai_traced".into();

    // preview 只渲染、不发送 —— 用来核对自定义 codec 产出的报文。
    let http = client.preview(&request)?;
    println!("URL: {}", http.url);
    for (k, v) in &http.headers {
        println!("{k}: {v}");
    }
    Ok(())
}
