//! Anthropic Messages API handlers (`api_format = "anthropic_v2"`).
//!
//! Implements the v2 messages format with native support for:
//! * extended thinking (signed content blocks),
//! * tool use,
//! * prompt caching with TTL (5m / 1h) on per-block `cache_control` markers,
//! * usage reporting that splits cache writes by TTL tier.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{ErrorContext, LLMError};
use crate::handler::{
    ByteStream, ChunkStream, HttpMethod, HttpRequest, NonStreamResponseHandler, RequestHandler,
    StreamResponseHandler,
};
use crate::http::ClientConfig;
use crate::types::{
    AssistantBlock, AssistantMessage, CacheMarker, CacheTtl, CacheUsage, CacheWriteUsage,
    ChatRequest, ChatResponse, Conversation, FinishReason, StreamChunk, SystemBlock,
    SystemPrompt, ToolChoice, ToolResultBlock, ToolResultContent, Turn, Usage, UserBlock,
    UserMessage,
};

pub const API_FORMAT: &str = "anthropic_v2";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Request handler
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct AnthropicRequestHandler;

impl RequestHandler for AnthropicRequestHandler {
    fn build_request(
        &self,
        request: &ChatRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        if config.base_url.is_empty() {
            return Err(LLMError::config("Anthropic: base_url is empty"));
        }
        let max_tokens = request.max_tokens.ok_or_else(|| {
            LLMError::validation("Anthropic requires max_tokens to be set on the request")
        })?;

        let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));

        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(request.model));
        body.insert("max_tokens".into(), json!(max_tokens));
        body.insert("stream".into(), json!(request.stream));

        if let Some(system) = &request.conversation.system {
            if let Some(system_value) = system_prompt_to_anthropic(system) {
                body.insert("system".into(), system_value);
            }
        }

        body.insert(
            "messages".into(),
            json!(turns_to_anthropic(&request.conversation)),
        );

        if let Some(t) = request.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(p) = request.top_p {
            body.insert("top_p".into(), json!(p));
        }
        if !request.stop_sequences.is_empty() {
            body.insert("stop_sequences".into(), json!(request.stop_sequences));
        }
        if request.enable_thinking {
            let budget = request.thinking_budget.unwrap_or(1024);
            body.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
        if let Some(user) = &request.user_id {
            body.insert("metadata".into(), json!({"user_id": user}));
        }

        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    let mut v = json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    });
                    if let Some(marker) = &t.cache {
                        v["cache_control"] = cache_control_value(marker);
                    }
                    v
                })
                .collect();
            body.insert("tools".into(), json!(tools));
            match &request.tool_choice {
                ToolChoice::Auto => {
                    body.insert("tool_choice".into(), json!({"type": "auto"}));
                }
                ToolChoice::None => { /* omit tools to disable */ }
                ToolChoice::Required => {
                    body.insert("tool_choice".into(), json!({"type": "any"}));
                }
                ToolChoice::Specific { name } => {
                    body.insert("tool_choice".into(), json!({"type": "tool", "name": name}));
                }
            }
        }

        let mut headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("x-api-key".to_string(), config.api_key.clone()),
            ("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
        ];
        if request.stream {
            headers.push(("accept".to_string(), "text/event-stream".to_string()));
        }

        Ok(HttpRequest {
            method: HttpMethod::Post,
            url,
            headers,
            body: serde_json::to_vec(&Value::Object(body))?,
        })
    }
}

fn cache_control_value(marker: &CacheMarker) -> Value {
    // Default TTL on Anthropic is 5m, so we omit `ttl` for that case.
    match marker.ttl {
        CacheTtl::Ephemeral5m => json!({"type": "ephemeral"}),
        CacheTtl::Ephemeral1h => json!({"type": "ephemeral", "ttl": "1h"}),
    }
}

fn system_prompt_to_anthropic(system: &SystemPrompt) -> Option<Value> {
    if system.blocks.is_empty() {
        return None;
    }
    let has_cache = system.blocks.iter().any(|b| match b {
        SystemBlock::Text { cache, .. } => cache.is_some(),
    });
    if !has_cache {
        // Simple form: plain string.
        let text: String = system
            .blocks
            .iter()
            .map(|b| match b {
                SystemBlock::Text { text, .. } => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("");
        if text.is_empty() {
            return None;
        }
        return Some(json!(text));
    }
    // Array form to carry cache markers.
    let arr: Vec<Value> = system
        .blocks
        .iter()
        .map(|b| match b {
            SystemBlock::Text { text, cache } => {
                let mut v = json!({"type": "text", "text": text});
                if let Some(marker) = cache {
                    v["cache_control"] = cache_control_value(marker);
                }
                v
            }
        })
        .collect();
    Some(Value::Array(arr))
}

fn turns_to_anthropic(conv: &Conversation) -> Vec<Value> {
    conv.turns
        .iter()
        .map(|t| match t {
            Turn::User(msg) => user_to_anthropic(msg),
            Turn::Assistant(msg) => assistant_to_anthropic(msg),
        })
        .collect()
}

fn user_to_anthropic(msg: &UserMessage) -> Value {
    let blocks: Vec<Value> = msg
        .blocks
        .iter()
        .map(|b| match b {
            UserBlock::Text { text, cache } => attach_cache(
                json!({"type": "text", "text": text}),
                cache.as_ref(),
            ),
            UserBlock::Image { url, detail: _, cache } => {
                let source = if let Some(rest) = url.strip_prefix("data:") {
                    if let Some((media, data)) = rest.split_once(";base64,") {
                        json!({"type": "base64", "media_type": media, "data": data})
                    } else {
                        json!({"type": "url", "url": url})
                    }
                } else {
                    json!({"type": "url", "url": url})
                };
                attach_cache(json!({"type": "image", "source": source}), cache.as_ref())
            }
            UserBlock::ToolResult { call_id, content, is_error, cache } => {
                let mut v = json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": tool_result_content_to_anthropic(content),
                });
                if *is_error {
                    v["is_error"] = json!(true);
                }
                attach_cache(v, cache.as_ref())
            }
        })
        .collect();
    json!({"role": "user", "content": blocks})
}

fn tool_result_content_to_anthropic(content: &ToolResultContent) -> Value {
    match content {
        ToolResultContent::Text(s) => json!(s),
        ToolResultContent::Json(v) => {
            // Anthropic accepts a text block whose text is the serialised JSON.
            json!([{"type": "text", "text": v.to_string()}])
        }
        ToolResultContent::Blocks(blocks) => Value::Array(
            blocks
                .iter()
                .map(|b| match b {
                    ToolResultBlock::Text { text } => json!({"type": "text", "text": text}),
                    ToolResultBlock::Image { url, .. } => {
                        let source = if let Some(rest) = url.strip_prefix("data:") {
                            if let Some((media, data)) = rest.split_once(";base64,") {
                                json!({"type": "base64", "media_type": media, "data": data})
                            } else {
                                json!({"type": "url", "url": url})
                            }
                        } else {
                            json!({"type": "url", "url": url})
                        };
                        json!({"type": "image", "source": source})
                    }
                })
                .collect(),
        ),
    }
}

fn assistant_to_anthropic(msg: &AssistantMessage) -> Value {
    let blocks: Vec<Value> = msg
        .blocks
        .iter()
        .map(|b| match b {
            AssistantBlock::Text { text, cache } => attach_cache(
                json!({"type": "text", "text": text}),
                cache.as_ref(),
            ),
            AssistantBlock::Thinking { text, signature } => {
                let mut v = json!({"type": "thinking", "thinking": text});
                if let Some(sig) = signature {
                    v["signature"] = json!(sig);
                }
                v
            }
            AssistantBlock::ToolCall { id, name, arguments, cache } => {
                let input: Value =
                    serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()));
                attach_cache(
                    json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                    cache.as_ref(),
                )
            }
        })
        .collect();
    json!({"role": "assistant", "content": blocks})
}

fn attach_cache(mut v: Value, marker: Option<&CacheMarker>) -> Value {
    if let Some(m) = marker {
        v["cache_control"] = cache_control_value(m);
    }
    v
}

// ---------------------------------------------------------------------------
// Non-stream response handler
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct AnthropicNonStreamHandler;

#[derive(Debug, Deserialize)]
struct AnthropicResponseDto {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    content: Vec<AnthropicBlockDto>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsageDto>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlockDto {
    Text { text: String },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
struct AnthropicUsageDto {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_creation: Option<AnthropicCacheCreationDto>,
}

#[derive(Debug, Default, Deserialize)]
struct AnthropicCacheCreationDto {
    #[serde(default)]
    ephemeral_5m_input_tokens: u32,
    #[serde(default)]
    ephemeral_1h_input_tokens: u32,
}

fn build_usage(u: &AnthropicUsageDto) -> Usage {
    let (e5, e1h) = match &u.cache_creation {
        Some(c) => (c.ephemeral_5m_input_tokens, c.ephemeral_1h_input_tokens),
        None => (0, 0),
    };
    let write_total = if u.cache_creation_input_tokens > 0 {
        u.cache_creation_input_tokens
    } else {
        e5.saturating_add(e1h)
    };
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        thinking_tokens: 0,
        cache: CacheUsage {
            read_tokens: u.cache_read_input_tokens,
            write: CacheWriteUsage {
                total: write_total,
                ephemeral_5m: e5,
                ephemeral_1h: e1h,
            },
        },
    }
}

impl NonStreamResponseHandler for AnthropicNonStreamHandler {
    fn parse_response(&self, body: Bytes) -> Result<ChatResponse, LLMError> {
        let dto: AnthropicResponseDto = serde_json::from_slice(&body).map_err(|e| LLMError::Parse {
            message: format!("anthropic: invalid JSON: {e}"),
            ctx: ErrorContext::default(),
        })?;

        let mut blocks = Vec::new();
        for b in dto.content {
            match b {
                AnthropicBlockDto::Text { text } => {
                    blocks.push(AssistantBlock::Text { text, cache: None });
                }
                AnthropicBlockDto::Thinking { thinking, signature } => {
                    blocks.push(AssistantBlock::Thinking {
                        text: thinking,
                        signature,
                    });
                }
                AnthropicBlockDto::ToolUse { id, name, input } => {
                    blocks.push(AssistantBlock::ToolCall {
                        id,
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                        cache: None,
                    });
                }
                AnthropicBlockDto::Unknown => {}
            }
        }

        let finish_reason = match dto.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("tool_use") => FinishReason::ToolCall,
            Some("stop_sequence") => FinishReason::StopSequence,
            _ => FinishReason::Other,
        };

        let usage = dto.usage.as_ref().map(build_usage).unwrap_or_default();

        Ok(ChatResponse {
            message: AssistantMessage { blocks },
            finish_reason,
            usage,
            request_id: dto.id,
            actual_model: dto.model,
            latency_ms: 0,
            rate_limit_info: Default::default(),
            metadata: Default::default(),
        })
    }
}

// ---------------------------------------------------------------------------
// Stream handler
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct AnthropicStreamHandler;

impl StreamResponseHandler for AnthropicStreamHandler {
    fn parse_stream(&self, stream: ByteStream) -> ChunkStream {
        Box::pin(AnthropicSseStream {
            inner: stream,
            buffer: bytes::BytesMut::new(),
            pending: std::collections::VecDeque::new(),
            tool_calls: std::collections::HashMap::new(),
            done: false,
        })
    }
}

struct AnthropicSseStream {
    inner: ByteStream,
    buffer: bytes::BytesMut,
    pending: std::collections::VecDeque<Result<StreamChunk, LLMError>>,
    /// Maps content-block index → tool_use id.
    tool_calls: std::collections::HashMap<u32, String>,
    done: bool,
}

impl AnthropicSseStream {
    fn process_buffer(&mut self) {
        while let Some(pos) = find_event_boundary(&self.buffer) {
            let raw = self.buffer.split_to(pos);
            let sep_len = if self.buffer.starts_with(b"\r\n\r\n") { 4 } else { 2 };
            let _ = self.buffer.split_to(sep_len);

            let text = std::str::from_utf8(&raw).unwrap_or("");
            let mut event_name: Option<&str> = None;
            let mut data_parts: Vec<&str> = Vec::new();
            for line in text.split('\n') {
                let line = line.trim_end_matches('\r');
                if let Some(rest) = line.strip_prefix("event:") {
                    event_name = Some(rest.trim());
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data_parts.push(rest.trim_start());
                }
            }
            if data_parts.is_empty() {
                continue;
            }
            let data = data_parts.join("\n");
            match parse_anthropic_event(event_name, &data, &mut self.tool_calls) {
                Ok(chunks) => {
                    for c in chunks {
                        if matches!(c, StreamChunk::Finish { .. }) {
                            self.done = true;
                        }
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
struct AnthropicEventDto {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<Value>,
    #[serde(default)]
    content_block: Option<Value>,
    #[serde(default)]
    usage: Option<AnthropicUsageDto>,
    #[serde(default)]
    error: Option<Value>,
}

fn parse_anthropic_event(
    event_name: Option<&str>,
    data: &str,
    tool_calls: &mut std::collections::HashMap<u32, String>,
) -> Result<Vec<StreamChunk>, LLMError> {
    let dto: AnthropicEventDto = serde_json::from_str(data).map_err(|e| LLMError::Parse {
        message: format!("anthropic stream: invalid JSON: {e}"),
        ctx: ErrorContext::default(),
    })?;
    let kind = dto.kind.as_deref().or(event_name).unwrap_or("");
    let mut out = Vec::new();
    match kind {
        "content_block_start" => {
            if let (Some(idx), Some(block)) = (dto.index, dto.content_block) {
                let btype = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if btype == "tool_use" {
                    if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                        tool_calls.insert(idx, id.to_string());
                        let name = block.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
                        out.push(StreamChunk::ToolCallDelta {
                            id: id.to_string(),
                            name_delta: name,
                            arguments_delta: None,
                        });
                    }
                } else if btype == "thinking" {
                    // Initial signature may arrive on content_block_start.
                    let sig = block
                        .get("signature")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    if sig.is_some() {
                        out.push(StreamChunk::ThinkingDelta {
                            text: String::new(),
                            signature_delta: sig,
                        });
                    }
                }
            }
        }
        "content_block_delta" => {
            if let Some(delta) = dto.delta {
                let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::TextDelta { text: text.to_string() });
                        }
                    }
                    "thinking_delta" => {
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::ThinkingDelta {
                                text: text.to_string(),
                                signature_delta: None,
                            });
                        }
                    }
                    "signature_delta" => {
                        if let Some(sig) = delta.get("signature").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::ThinkingDelta {
                                text: String::new(),
                                signature_delta: Some(sig.to_string()),
                            });
                        }
                    }
                    "input_json_delta" => {
                        if let (Some(idx), Some(args)) =
                            (dto.index, delta.get("partial_json").and_then(|v| v.as_str()))
                        {
                            if let Some(id) = tool_calls.get(&idx) {
                                out.push(StreamChunk::ToolCallDelta {
                                    id: id.clone(),
                                    name_delta: None,
                                    arguments_delta: Some(args.to_string()),
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "message_delta" => {
            if let Some(usage) = dto.usage.as_ref() {
                out.push(StreamChunk::Usage { usage: build_usage(usage) });
            }
            if let Some(delta) = dto.delta {
                if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                    let r = match reason {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "tool_use" => FinishReason::ToolCall,
                        "stop_sequence" => FinishReason::StopSequence,
                        _ => FinishReason::Other,
                    };
                    out.push(StreamChunk::Finish { reason: r });
                }
            }
        }
        "message_stop" => {
            if !out.iter().any(|c| matches!(c, StreamChunk::Finish { .. })) {
                out.push(StreamChunk::Finish { reason: FinishReason::Stop });
            }
        }
        "error" => {
            let msg = dto
                .error
                .as_ref()
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("anthropic stream error")
                .to_string();
            out.push(StreamChunk::Error { error: msg });
        }
        _ => {}
    }
    Ok(out)
}

impl Stream for AnthropicSseStream {
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
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Register Anthropic handlers under `"anthropic_v2"`.
pub fn register(registry: &crate::handler::HandlerRegistry) {
    registry.register_protocol(
        API_FORMAT,
        AnthropicRequestHandler,
        AnthropicNonStreamHandler,
        AnthropicStreamHandler,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Conversation, SystemPrompt};

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.anthropic.com", "sk-ant-test")
    }

    #[test]
    fn builds_messages_request() {
        let req = ChatRequest::anthropic(
            "claude-3-5-sonnet-20241022",
            Conversation::single_user("hi"),
        );
        let http = AnthropicRequestHandler.build_request(&req, &cfg()).unwrap();
        assert_eq!(http.url, "https://api.anthropic.com/v1/messages");
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        let hdrs: std::collections::HashMap<_, _> =
            http.headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(hdrs.get("x-api-key").copied(), Some("sk-ant-test"));
        assert_eq!(hdrs.get("anthropic-version").copied(), Some(ANTHROPIC_VERSION));
    }

    #[test]
    fn cache_marker_5m_emits_default_ephemeral() {
        let conv = Conversation {
            system: Some(SystemPrompt::text("S").with_cache(CacheMarker::ephemeral_5m())),
            turns: vec![Turn::User(UserMessage::text("hi"))],
        };
        let req = ChatRequest::anthropic("claude", conv);
        let http = AnthropicRequestHandler.build_request(&req, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        let sys = &body["system"];
        assert!(sys.is_array(), "system should be array form when cache marker present");
        let cc = &sys[0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
        assert!(cc.get("ttl").is_none(), "5m is default; ttl field should be omitted");
    }

    #[test]
    fn cache_marker_1h_emits_ttl() {
        let conv = Conversation {
            system: Some(SystemPrompt::text("S").with_cache(CacheMarker::ephemeral_1h())),
            turns: vec![Turn::User(UserMessage::text("hi"))],
        };
        let req = ChatRequest::anthropic("claude", conv);
        let http = AnthropicRequestHandler.build_request(&req, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        let cc = &body["system"][0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
        assert_eq!(cc["ttl"], "1h");
    }

    #[test]
    fn anthropic_requires_max_tokens() {
        let mut req = ChatRequest::openai("c", Conversation::single_user("hi"));
        req.api_format = API_FORMAT.into();
        let err = AnthropicRequestHandler.build_request(&req, &cfg()).unwrap_err();
        assert!(matches!(err, LLMError::Validation { .. }));
    }

    #[test]
    fn parses_response_with_thinking_signature_and_cache_usage() {
        let body = br#"{
            "id":"msg_1",
            "model":"claude-3-5-sonnet-20241022",
            "content":[
                {"type":"thinking","thinking":"hmm","signature":"sig_abc"},
                {"type":"text","text":"sure"},
                {"type":"tool_use","id":"tu_1","name":"add","input":{"a":1,"b":2}}
            ],
            "stop_reason":"tool_use",
            "usage":{
                "input_tokens":10,
                "output_tokens":5,
                "cache_read_input_tokens":3,
                "cache_creation_input_tokens":50,
                "cache_creation":{
                    "ephemeral_5m_input_tokens":30,
                    "ephemeral_1h_input_tokens":20
                }
            }
        }"#;
        let resp = AnthropicNonStreamHandler
            .parse_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::ToolCall);
        // Cache usage split by TTL
        assert_eq!(resp.usage.cache.read_tokens, 3);
        assert_eq!(resp.usage.cache.write.total, 50);
        assert_eq!(resp.usage.cache.write.ephemeral_5m, 30);
        assert_eq!(resp.usage.cache.write.ephemeral_1h, 20);
        // Thinking with signature preserved
        let thinking_sig = resp.message.blocks.iter().find_map(|b| match b {
            AssistantBlock::Thinking { signature, .. } => signature.clone(),
            _ => None,
        });
        assert_eq!(thinking_sig.as_deref(), Some("sig_abc"));
        // Tool call args
        let tool_args = resp.message.blocks.iter().find_map(|b| match b {
            AssistantBlock::ToolCall { arguments, .. } => Some(arguments.clone()),
            _ => None,
        });
        assert!(tool_args.unwrap().contains("\"a\":1"));
    }

    #[test]
    fn tool_result_blocks_serialize_natively() {
        let mut conv = Conversation::single_user("ok");
        let mut user = UserMessage::default();
        user.push(UserBlock::ToolResult {
            call_id: "t1".into(),
            content: ToolResultContent::Blocks(vec![
                ToolResultBlock::Text { text: "Tokyo".into() },
                ToolResultBlock::Image { url: "https://x/y.png".into(), detail: None },
            ]),
            is_error: false,
            cache: None,
        });
        conv.push(Turn::User(user));
        let req = ChatRequest::anthropic("c", conv);
        let http = AnthropicRequestHandler.build_request(&req, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        let tool_block = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|m| m["content"].as_array().unwrap())
            .find(|b| b["type"] == "tool_result")
            .unwrap();
        assert!(tool_block["content"].is_array());
        assert_eq!(tool_block["content"][0]["type"], "text");
        assert_eq!(tool_block["content"][1]["type"], "image");
    }

    #[tokio::test]
    async fn parses_sse_stream() {
        use futures_util::StreamExt;
        let bytes = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"he\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"y\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_vec();
        let s: ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(bytes))]));
        let mut out = AnthropicStreamHandler.parse_stream(s);
        let mut acc = String::new();
        let mut saw_finish = false;
        while let Some(c) = out.next().await {
            match c.unwrap() {
                StreamChunk::TextDelta { text } => acc.push_str(&text),
                StreamChunk::Finish { .. } => saw_finish = true,
                _ => {}
            }
        }
        assert_eq!(acc, "hey");
        assert!(saw_finish);
    }

    #[tokio::test]
    async fn stream_thinking_with_signature() {
        use futures_util::StreamExt;
        // Initial content_block_start with thinking type + signature, then thinking_delta, then signature_delta.
        let bytes = b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"signature\":\"sig_init\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"step1\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"_more\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n".to_vec();
        let s: ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(bytes))]));
        let mut out = AnthropicStreamHandler.parse_stream(s);
        let mut full_sig = String::new();
        let mut full_thinking = String::new();
        while let Some(c) = out.next().await {
            match c.unwrap() {
                StreamChunk::ThinkingDelta { text, signature_delta } => {
                    full_thinking.push_str(&text);
                    if let Some(s) = signature_delta {
                        full_sig.push_str(&s);
                    }
                }
                StreamChunk::Finish { .. } => break,
                _ => {}
            }
        }
        assert_eq!(full_thinking, "step1");
        assert_eq!(full_sig, "sig_init_more");
    }
}
