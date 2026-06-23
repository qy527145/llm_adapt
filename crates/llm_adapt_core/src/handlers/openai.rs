//! OpenAI-compatible protocol handlers (`api_format = "openai_compat"`).
//!
//! Tested against OpenAI's `/v1/chat/completions` endpoint and works with any
//! provider that mirrors that schema (Together, Groq, DeepSeek, OpenRouter, ...).
//!
//! OpenAI does not surface explicit `cache_control` markers (its caching is
//! automatic on structurally identical prefixes), nor does it expose signed
//! thinking content. The handler silently ignores cache markers and drops
//! `Thinking` blocks with a `tracing::warn!`.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{ErrorContext, LLMError};
use crate::handler::{
    ByteStream, ChunkStream, HttpMethod, HttpRequest, NonStreamResponseHandler, RequestHandler,
    StreamResponseHandler,
};
use crate::http::ClientConfig;
use crate::types::{
    AssistantBlock, AssistantMessage, CacheUsage, CacheWriteUsage, ChatRequest, ChatResponse,
    Conversation, FinishReason, ResponseFormat, StreamChunk, SystemBlock, ToolChoice,
    ToolResultBlock, ToolResultContent, Turn, Usage, UserBlock,
};

/// Registry key for this protocol.
pub const API_FORMAT: &str = "openai_compat";

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct OpenAIRequestHandler;

impl RequestHandler for OpenAIRequestHandler {
    fn build_request(
        &self,
        request: &ChatRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        if config.base_url.is_empty() {
            return Err(LLMError::config("OpenAI: base_url is empty"));
        }
        let url = format!(
            "{}/v1/chat/completions",
            config.base_url.trim_end_matches('/')
        );

        let body = build_openai_body(request)?;
        let body_bytes = serde_json::to_vec(&body)?;

        let mut headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
            (
                "authorization".to_string(),
                format!("Bearer {}", config.api_key),
            ),
        ];
        if request.stream {
            headers.push(("accept".to_string(), "text/event-stream".to_string()));
        }

        Ok(HttpRequest {
            method: HttpMethod::Post,
            url,
            headers,
            body: body_bytes,
        })
    }
}

fn build_openai_body(req: &ChatRequest) -> Result<Value, LLMError> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(req.model));
    body.insert("messages".into(), json!(conversation_to_openai(&req.conversation)));
    body.insert("stream".into(), json!(req.stream));
    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if let Some(m) = req.max_tokens {
        body.insert("max_tokens".into(), json!(m));
    }
    if let Some(s) = req.seed {
        body.insert("seed".into(), json!(s));
    }
    if !req.stop_sequences.is_empty() {
        body.insert("stop".into(), json!(req.stop_sequences));
    }
    if let Some(user) = &req.user_id {
        body.insert("user".into(), json!(user));
    }
    if req.stream {
        body.insert(
            "stream_options".into(),
            json!({ "include_usage": true }),
        );
    }

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body.insert("tools".into(), json!(tools));

        match &req.tool_choice {
            ToolChoice::Auto => body.insert("tool_choice".into(), json!("auto")),
            ToolChoice::None => body.insert("tool_choice".into(), json!("none")),
            ToolChoice::Required => body.insert("tool_choice".into(), json!("required")),
            ToolChoice::Specific { name } => body.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": name}}),
            ),
        };
    }

    match &req.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::Json => {
            body.insert(
                "response_format".into(),
                json!({"type": "json_object"}),
            );
        }
        ResponseFormat::JsonSchema { name, schema, strict } => {
            body.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": *strict,
                    }
                }),
            );
        }
    }

    Ok(Value::Object(body))
}

fn conversation_to_openai(conv: &Conversation) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(system) = &conv.system {
        let mut text = String::new();
        for block in &system.blocks {
            match block {
                SystemBlock::Text { text: t, cache } => {
                    if cache.is_some() {
                        tracing::debug!(target: "llm_adapt::openai",
                            "ignoring CacheMarker on system prompt (auto-cached by OpenAI)");
                    }
                    text.push_str(t);
                }
            }
        }
        if !text.is_empty() {
            out.push(json!({ "role": "system", "content": text }));
        }
    }

    for turn in &conv.turns {
        match turn {
            Turn::User(msg) => append_user_turn(&mut out, msg),
            Turn::Assistant(msg) => append_assistant_turn(&mut out, msg),
        }
    }
    out
}

fn append_user_turn(out: &mut Vec<Value>, msg: &crate::types::UserMessage) {
    // OpenAI splits a single "user turn" into:
    //  - 0..n {"role":"user", content: [...]}   (text + images)
    //  - 0..n {"role":"tool", tool_call_id: id, content: "..."}  (one per result)
    let mut user_blocks: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();

    for block in &msg.blocks {
        match block {
            UserBlock::Text { text, cache } => {
                if cache.is_some() {
                    tracing::debug!(target: "llm_adapt::openai",
                        "ignoring CacheMarker on user text (auto-caching is automatic)");
                }
                user_blocks.push(json!({"type": "text", "text": text}));
            }
            UserBlock::Image { url, detail, cache } => {
                if cache.is_some() {
                    tracing::debug!(target: "llm_adapt::openai", "ignoring CacheMarker on user image");
                }
                let mut img = json!({"url": url});
                if let Some(d) = detail {
                    img["detail"] = json!(d);
                }
                user_blocks.push(json!({"type": "image_url", "image_url": img}));
            }
            UserBlock::ToolResult { call_id, content, is_error, cache: _ } => {
                let serialised = tool_result_to_openai_string(content);
                let mut payload = json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": serialised,
                });
                if *is_error {
                    // OpenAI has no native `is_error` flag; prefix the content
                    // so the model still sees the signal.
                    payload["content"] = json!(format!("[error] {}", serialised));
                }
                tool_results.push(payload);
            }
        }
    }

    if !user_blocks.is_empty() {
        // Collapse single text-only to plain string for maximum compatibility.
        let content = if user_blocks.len() == 1 && user_blocks[0]["type"] == json!("text") {
            user_blocks[0]["text"].clone()
        } else {
            json!(user_blocks)
        };
        out.push(json!({"role": "user", "content": content}));
    }
    out.extend(tool_results);
}

fn append_assistant_turn(out: &mut Vec<Value>, msg: &AssistantMessage) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut dropped_thinking = false;
    for block in &msg.blocks {
        match block {
            AssistantBlock::Text { text, .. } => text_parts.push(text),
            AssistantBlock::Thinking { .. } => dropped_thinking = true,
            AssistantBlock::ToolCall { id, name, arguments, .. } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }));
            }
        }
    }
    if dropped_thinking {
        tracing::warn!(target: "llm_adapt::openai",
            "dropping AssistantBlock::Thinking (not representable in chat/completions)");
    }
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), json!("assistant"));
    if !text_parts.is_empty() {
        obj.insert("content".into(), json!(text_parts.concat()));
    } else if !tool_calls.is_empty() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert("content".into(), json!(""));
    }
    if !tool_calls.is_empty() {
        obj.insert("tool_calls".into(), json!(tool_calls));
    }
    out.push(Value::Object(obj));
}

fn tool_result_to_openai_string(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(s) => s.clone(),
        ToolResultContent::Json(v) => v.to_string(),
        ToolResultContent::Blocks(blocks) => {
            let mut text = String::new();
            let mut dropped_images = 0;
            for b in blocks {
                match b {
                    ToolResultBlock::Text { text: t } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                    ToolResultBlock::Image { .. } => dropped_images += 1,
                }
            }
            if dropped_images > 0 {
                tracing::warn!(target: "llm_adapt::openai",
                    "dropped {} image block(s) from tool_result (OpenAI tool messages are text-only)",
                    dropped_images);
            }
            text
        }
    }
}

// ---------------------------------------------------------------------------
// Non-stream response handler
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct OpenAINonStreamHandler;

#[derive(Debug, Deserialize)]
struct OpenAIResponseDto {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIChoiceDto>,
    #[serde(default)]
    usage: Option<OpenAIUsageDto>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoiceDto {
    #[serde(default)]
    message: OpenAIMessageDto,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIMessageDto {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCallDto>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIToolCallDto {
    id: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    function: OpenAIFunctionDto,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIFunctionDto {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIUsageDto {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAIPromptDetailsDto>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIPromptDetailsDto {
    #[serde(default)]
    cached_tokens: u32,
}

impl NonStreamResponseHandler for OpenAINonStreamHandler {
    fn parse_response(&self, body: Bytes) -> Result<ChatResponse, LLMError> {
        let dto: OpenAIResponseDto = serde_json::from_slice(&body)
            .map_err(|e| LLMError::Parse {
                message: format!("openai: invalid JSON: {e}"),
                ctx: ErrorContext::default(),
            })?;

        let choice = dto.choices.into_iter().next().ok_or_else(|| LLMError::Parse {
            message: "openai: response had no choices".into(),
            ctx: ErrorContext::default(),
        })?;

        let mut blocks = Vec::new();
        if let Some(text) = choice.message.content.filter(|s| !s.is_empty()) {
            blocks.push(AssistantBlock::Text { text, cache: None });
        }
        for tc in choice.message.tool_calls {
            blocks.push(AssistantBlock::ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
                cache: None,
            });
        }

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolCall,
            Some("content_filter") => FinishReason::Safety,
            Some("stop_sequence") => FinishReason::StopSequence,
            _ => FinishReason::Other,
        };

        let usage = dto
            .usage
            .map(|u| {
                let cached = u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens).unwrap_or(0);
                let input_uncached = u.prompt_tokens.saturating_sub(cached);
                Usage {
                    input_tokens: input_uncached,
                    output_tokens: u.completion_tokens,
                    thinking_tokens: 0,
                    cache: CacheUsage {
                        read_tokens: cached,
                        write: CacheWriteUsage::default(),
                    },
                }
            })
            .unwrap_or_default();

        let mut metadata = std::collections::HashMap::new();
        if let Some(fp) = dto.system_fingerprint {
            metadata.insert("system_fingerprint".into(), serde_json::json!(fp));
        }

        Ok(ChatResponse {
            message: AssistantMessage { blocks },
            finish_reason,
            usage,
            request_id: dto.id,
            actual_model: dto.model,
            latency_ms: 0,
            rate_limit_info: Default::default(),
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// Stream handler (SSE)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct OpenAIStreamHandler;

impl StreamResponseHandler for OpenAIStreamHandler {
    fn parse_stream(&self, stream: ByteStream) -> ChunkStream {
        Box::pin(SseChunkStream {
            inner: stream,
            buffer: bytes::BytesMut::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
        })
    }
}

struct SseChunkStream {
    inner: ByteStream,
    buffer: bytes::BytesMut,
    pending: std::collections::VecDeque<Result<StreamChunk, LLMError>>,
    done: bool,
}

impl SseChunkStream {
    fn process_buffer(&mut self) {
        while let Some(pos) = find_event_boundary(&self.buffer) {
            let raw = self.buffer.split_to(pos);
            let sep_len = if self.buffer.starts_with(b"\r\n\r\n") { 4 } else { 2 };
            let _ = self.buffer.split_to(sep_len);

            let event = std::str::from_utf8(&raw).unwrap_or("");
            let mut data_parts: Vec<&str> = Vec::new();
            for line in event.split('\n') {
                let line = line.trim_end_matches('\r');
                if let Some(rest) = line.strip_prefix("data:") {
                    data_parts.push(rest.trim_start());
                }
            }
            if data_parts.is_empty() {
                continue;
            }
            let data = data_parts.join("\n");
            if data == "[DONE]" {
                self.done = true;
                self.pending.push_back(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                }));
                continue;
            }
            match parse_openai_stream_event(&data) {
                Ok(chunks) => {
                    for c in chunks {
                        self.pending.push_back(Ok(c));
                    }
                }
                Err(e) => self.pending.push_back(Err(e)),
            }
        }
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    let win_lf = buf.windows(2).position(|w| w == b"\n\n");
    let win_crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    match (win_lf, win_crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDto {
    #[serde(default)]
    choices: Vec<OpenAIStreamChoiceDto>,
    #[serde(default)]
    usage: Option<OpenAIUsageDto>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoiceDto {
    #[serde(default)]
    delta: OpenAIDeltaDto,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIDeltaDto {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIStreamToolCallDto>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCallDto {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIStreamFunctionDto>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIStreamFunctionDto {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn parse_openai_stream_event(data: &str) -> Result<Vec<StreamChunk>, LLMError> {
    let dto: OpenAIStreamDto = serde_json::from_str(data).map_err(|e| LLMError::Parse {
        message: format!("openai stream: invalid JSON: {e}"),
        ctx: ErrorContext::default(),
    })?;
    let mut out = Vec::new();
    for choice in dto.choices {
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                out.push(StreamChunk::TextDelta { text });
            }
        }
        for tc in choice.delta.tool_calls {
            let id = tc.id.unwrap_or_else(|| format!("tc_idx_{}", tc.index.unwrap_or(0)));
            let (name_delta, arguments_delta) = match tc.function {
                Some(f) => (f.name, f.arguments),
                None => (None, None),
            };
            out.push(StreamChunk::ToolCallDelta {
                id,
                name_delta,
                arguments_delta,
            });
        }
        if let Some(reason) = choice.finish_reason {
            let r = match reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCall,
                "content_filter" => FinishReason::Safety,
                _ => FinishReason::Other,
            };
            out.push(StreamChunk::Finish { reason: r });
        }
    }
    if let Some(u) = dto.usage {
        let cached = u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens).unwrap_or(0);
        let input_uncached = u.prompt_tokens.saturating_sub(cached);
        out.push(StreamChunk::Usage {
            usage: Usage {
                input_tokens: input_uncached,
                output_tokens: u.completion_tokens,
                thinking_tokens: 0,
                cache: CacheUsage {
                    read_tokens: cached,
                    write: CacheWriteUsage::default(),
                },
            },
        });
    }
    Ok(out)
}

impl Stream for SseChunkStream {
    type Item = Result<StreamChunk, LLMError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(item) = self.pending.pop_front() {
                return Poll::Ready(Some(item));
            }
            if self.done {
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    self.buffer.extend_from_slice(&bytes);
                    self.process_buffer();
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    if !self.buffer.is_empty() {
                        let trailing = self.buffer.split().freeze();
                        let text = std::str::from_utf8(&trailing).unwrap_or("");
                        for line in text.split('\n') {
                            if let Some(rest) = line.trim_end_matches('\r').strip_prefix("data:") {
                                let data = rest.trim_start();
                                if data == "[DONE]" || data.is_empty() {
                                    continue;
                                }
                                if let Ok(chunks) = parse_openai_stream_event(data) {
                                    for c in chunks {
                                        self.pending.push_back(Ok(c));
                                    }
                                }
                            }
                        }
                        if !self.pending.is_empty() {
                            continue;
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Register OpenAI-compatible handlers under `"openai_compat"`.
pub fn register(registry: &crate::handler::HandlerRegistry) {
    registry.register_protocol(
        API_FORMAT,
        OpenAIRequestHandler,
        OpenAINonStreamHandler,
        OpenAIStreamHandler,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessage, Conversation, ToolResultBlock, ToolResultContent, Turn, UserBlock,
        UserMessage,
    };

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.openai.com", "sk-test")
    }

    #[test]
    fn builds_simple_request() {
        let mut req = ChatRequest::openai("gpt-4o", Conversation::single_user("hi"));
        req.temperature = Some(0.5);
        let http = OpenAIRequestHandler.build_request(&req, &cfg()).unwrap();
        assert_eq!(http.method, HttpMethod::Post);
        assert_eq!(http.url, "https://api.openai.com/v1/chat/completions");
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn tool_result_blocks_degrade_to_string() {
        let mut conv = Conversation::with_system("you are terse", "what's the temp?");
        let mut user = UserMessage::default();
        user.push(UserBlock::ToolResult {
            call_id: "t1".into(),
            content: ToolResultContent::Blocks(vec![
                ToolResultBlock::Text { text: "Tokyo: 22C".into() },
                ToolResultBlock::Image { url: "https://example.com/x.png".into(), detail: None },
            ]),
            is_error: false,
            cache: None,
        });
        conv.push(Turn::User(user));

        let req = ChatRequest::openai("gpt-4o", conv);
        let http = OpenAIRequestHandler.build_request(&req, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        // System + user (text) + tool (degraded)
        let messages = body["messages"].as_array().unwrap();
        let tool_msg = messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["tool_call_id"], "t1");
        assert_eq!(tool_msg["content"], "Tokyo: 22C"); // image silently dropped
    }

    #[test]
    fn assistant_thinking_dropped_silently() {
        let mut conv = Conversation::single_user("continue");
        let mut assistant = AssistantMessage::default();
        assistant.push(AssistantBlock::Thinking { text: "secret".into(), signature: None });
        assistant.push(AssistantBlock::Text { text: "ok".into(), cache: None });
        conv.push(Turn::Assistant(assistant));
        conv.push(Turn::User(UserMessage::text("go on")));

        let req = ChatRequest::openai("gpt-4o", conv);
        let http = OpenAIRequestHandler.build_request(&req, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        // Assistant message should contain "ok" but not "secret" anywhere.
        let dump = body.to_string();
        assert!(dump.contains("\"ok\""));
        assert!(!dump.contains("secret"));
    }

    #[test]
    fn parses_simple_response() {
        let body = br#"{
            "id": "chatcmpl-1",
            "model": "gpt-4o-2024",
            "choices": [{
                "message": {"role": "assistant", "content": "hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                       "prompt_tokens_details": {"cached_tokens": 1}}
        }"#;
        let resp = OpenAINonStreamHandler
            .parse_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.text(), "hello!");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        // 3 prompt - 1 cached read = 2 uncached input
        assert_eq!(resp.usage.input_tokens, 2);
        assert_eq!(resp.usage.cache.read_tokens, 1);
        assert_eq!(resp.usage.output_tokens, 2);
    }

    #[tokio::test]
    async fn parses_sse_stream() {
        use futures_util::StreamExt;
        let lines = b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_vec();
        let s: ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(lines))]));
        let mut out = OpenAIStreamHandler.parse_stream(s);
        let mut collected = String::new();
        while let Some(c) = out.next().await {
            match c.unwrap() {
                StreamChunk::TextDelta { text } => collected.push_str(&text),
                StreamChunk::Finish { .. } => break,
                _ => {}
            }
        }
        assert_eq!(collected, "hello");
    }
}
