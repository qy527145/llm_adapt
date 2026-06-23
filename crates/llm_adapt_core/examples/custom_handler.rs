//! `cargo run -p llm_adapt_core --example custom_handler`
//!
//! Shows how to register a custom protocol handler. Here we wrap the built-in
//! OpenAI handlers and inject an extra `X-Trace-Id` header on every request.

use bytes::Bytes;
use llm_adapt_core::{
    ByteStream, ChatRequest, ChatResponse, ChunkStream, ClientConfig, Conversation,
    HandlerRegistry, HttpRequest, LLMClient, LLMError, NonStreamResponseHandler, RequestHandler,
    StreamResponseHandler,
};

struct TracingRequest<R>(R);

impl<R: RequestHandler> RequestHandler for TracingRequest<R> {
    fn build_request(
        &self,
        request: &ChatRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        let mut http = self.0.build_request(request, config)?;
        http.headers
            .push(("x-trace-id".into(), format!("trace-{}", request.model)));
        Ok(http)
    }
}

struct WrappedNonStream<N>(N);
impl<N: NonStreamResponseHandler> NonStreamResponseHandler for WrappedNonStream<N> {
    fn parse_response(&self, body: Bytes) -> Result<ChatResponse, LLMError> {
        self.0.parse_response(body)
    }
}

struct WrappedStream<S>(S);
impl<S: StreamResponseHandler> StreamResponseHandler for WrappedStream<S> {
    fn parse_stream(&self, stream: ByteStream) -> ChunkStream {
        self.0.parse_stream(stream)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use llm_adapt_core::handlers::openai;

    let registry = HandlerRegistry::new();
    registry.register_protocol(
        "openai_traced",
        TracingRequest(openai::OpenAIRequestHandler),
        WrappedNonStream(openai::OpenAINonStreamHandler),
        WrappedStream(openai::OpenAIStreamHandler),
    );

    let client = LLMClient::builder(ClientConfig::new("https://api.openai.com", "sk-demo"))
        .without_default_handlers()
        .registry(registry)
        .build()?;

    let mut request = ChatRequest::openai("gpt-4o-mini", Conversation::single_user("Hi"));
    request.api_format = "openai_traced".into();

    let http = client.preview(&request)?;
    println!("URL: {}", http.url);
    for (k, v) in &http.headers {
        println!("{k}: {v}");
    }
    Ok(())
}
