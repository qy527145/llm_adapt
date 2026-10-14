//! Anthropic Messages API 的编解码（`protocol = "anthropic"`）。
//!
//! 原生支持：扩展思考（带签名的思考块）、工具调用、按块 `cache_control` 的
//! 提示词缓存（5m / 1h 两档）、按 TTL 分档的缓存写入用量。
//!
//! 入站（网关）方向的 `decode_request` / `encode_response` / `new_stream_encoder`
//! 在 [`inbound`] 里实现：别的客户端把 `/v1/messages` 报文发给我们时，解成 IR
//! 再按目标协议编回去。

use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{ErrorContext, LLMError};
use crate::http::{ClientConfig, HttpMethod, HttpRequest};
use crate::protocol::codec::{Codec, StreamEncoder};
use crate::protocol::ir::{
    Block, CacheMarker, CacheTtl, CacheUsage, CacheWriteUsage, FinishReason, ImageSource, Protocol,
    ProtocolId, ToolChoice, ToolResultBlock, ToolResultContent, Usage, WireRequest, WireResponse,
};
use crate::sse::{SseFrame, SseMapper, SseOutcome};
use crate::types::StreamChunk;

mod inbound;

pub use inbound::AnthropicStreamEncoder;


pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// 扩展思考未给预算时的默认值。
pub const DEFAULT_THINKING_BUDGET: u32 = 1024;

#[derive(Debug, Default, Clone)]
pub struct AnthropicCodec;

impl Codec for AnthropicCodec {
    fn protocol(&self) -> ProtocolId {
        Protocol::Anthropic.into()
    }

    fn encode_request(
        &self,
        request: &WireRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        let body = build_body(request, config)?;
        let mut headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
            ("x-api-key".to_string(), config.api_key.clone()),
            (
                "anthropic-version".to_string(),
                ANTHROPIC_VERSION.to_string(),
            ),
        ];
        if request.stream {
            headers.push(("accept".to_string(), "text/event-stream".to_string()));
        }
        Ok(HttpRequest {
            method: HttpMethod::Post,
            url: crate::protocol::codec::endpoint_url(config, Protocol::Anthropic.default_path()),
            headers,
            body: serde_json::to_vec(&body)?,
            correlation: None,
        })
    }

    fn decode_response(&self, body: Bytes) -> Result<WireResponse, LLMError> {
        decode_response(&body)
    }

    fn new_stream_decoder(&self) -> Box<dyn SseMapper> {
        Box::new(AnthropicSseMapper::default())
    }

    fn decode_request(&self, body: &[u8]) -> Result<WireRequest, LLMError> {
        inbound::decode_request(body)
    }

    fn encode_response(&self, response: &WireResponse) -> Result<Vec<u8>, LLMError> {
        inbound::encode_response(response)
    }

    fn new_stream_encoder(&self) -> Option<Box<dyn StreamEncoder>> {
        Some(Box::new(AnthropicStreamEncoder::new()))
    }
}

pub fn build_body(request: &WireRequest, config: &ClientConfig) -> Result<Value, LLMError> {
    if config.base_url.is_empty() {
        return Err(LLMError::config("Anthropic: base_url is empty"));
    }
    let max_tokens = request.max_tokens.ok_or_else(|| {
        LLMError::validation("Anthropic requires max_tokens to be set on the request")
    })?;

    // 未建模的顶层字段先铺进去，随后被建模字段覆盖 —— 让「模型认识的那个」永远赢，
    // 免得一个笔误的 extra 键静默改掉请求的实际语义。
    let mut body = serde_json::Map::new();
    for (k, v) in &request.extra {
        body.insert(k.clone(), v.clone());
    }

    body.insert("model".into(), json!(request.model));
    body.insert("max_tokens".into(), json!(max_tokens));
    body.insert("stream".into(), json!(request.stream));

    if let Some(system) = system_to_wire(&request.system) {
        body.insert("system".into(), system);
    }

    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role.as_str(),
                "content": m.content.iter().map(block_to_wire).collect::<Vec<_>>(),
            })
        })
        .collect();
    body.insert("messages".into(), json!(messages));

    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if !request.stop.is_empty() {
        body.insert("stop_sequences".into(), json!(request.stop));
    }
    if let Some(thinking) = &request.thinking {
        let budget = thinking.budget_tokens.unwrap_or(DEFAULT_THINKING_BUDGET);
        body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
    }
    if let Some(user) = &request.user_id {
        body.insert("metadata".into(), json!({"user_id": user}));
    }

    if !request.tools.is_empty() {
        let tools: Vec<Value> = request.tools.iter().map(tool_to_wire).collect();
        body.insert("tools".into(), json!(tools));
        match &request.tool_choice {
            ToolChoice::Auto => {
                body.insert("tool_choice".into(), json!({"type": "auto"}));
            }
            // 注释与旧实现保持一致：None 时不发 tool_choice。
            ToolChoice::None => {}
            ToolChoice::Required => {
                body.insert("tool_choice".into(), json!({"type": "any"}));
            }
            ToolChoice::Specific { name } => {
                body.insert("tool_choice".into(), json!({"type": "tool", "name": name}));
            }
        }
    }

    Ok(Value::Object(body))
}

fn cache_control_value(marker: &CacheMarker) -> Value {
    // Anthropic 的默认 TTL 就是 5m，所以那一档省略 `ttl` 字段。
    match marker.ttl {
        CacheTtl::Ephemeral5m => json!({"type": "ephemeral"}),
        CacheTtl::Ephemeral1h => json!({"type": "ephemeral", "ttl": "1h"}),
    }
}

fn attach_cache(mut v: Value, marker: Option<&CacheMarker>) -> Value {
    if let Some(m) = marker {
        v["cache_control"] = cache_control_value(m);
    }
    v
}

/// system 分段 → wire。
///
/// 没有任何缓存标记时用最简的裸字符串形态；有标记才升格成块数组 —— 与旧实现一致，
/// 也让绝大多数请求的报文保持干净。
fn system_to_wire(blocks: &[Block]) -> Option<Value> {
    if blocks.is_empty() {
        return None;
    }
    let texts: Vec<(&str, Option<&CacheMarker>)> = blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text, cache } => Some((text.as_str(), cache.as_ref())),
            other => {
                tracing::warn!(
                    target: "llm_adapt::anthropic",
                    "system 只接受 text 块，已忽略 {:?}",
                    other
                );
                None
            }
        })
        .collect();

    if !texts.iter().any(|(_, c)| c.is_some()) {
        // 换行拼接：见 `WireRequest::system_text` 的说明，三协议必须一致。
        let joined = texts.iter().map(|(t, _)| *t).collect::<Vec<_>>().join("\n");
        if joined.is_empty() {
            return None;
        }
        return Some(json!(joined));
    }

    Some(Value::Array(
        texts
            .into_iter()
            .map(|(text, cache)| attach_cache(json!({"type": "text", "text": text}), cache))
            .collect(),
    ))
}

fn tool_to_wire(t: &crate::protocol::ir::ToolSpec) -> Value {
    // 服务端工具（web_search 等）用厂商原样的声明对象，不套 function 外壳。
    if let crate::protocol::ir::ToolKind::ServerTool { spec } = &t.kind {
        return spec.clone();
    }
    let mut v = json!({
        "name": t.name,
        "description": t.description.clone().unwrap_or_default(),
        "input_schema": t.input_schema,
    });
    if let Some(marker) = &t.cache {
        v["cache_control"] = cache_control_value(marker);
    }
    v
}

fn image_source_to_wire(source: &ImageSource) -> Value {
    match source {
        ImageSource::Url { url } => json!({"type": "url", "url": url}),
        ImageSource::Base64 { media_type, data } => {
            json!({"type": "base64", "media_type": media_type, "data": data})
        }
    }
}

fn tool_result_content_to_wire(content: &ToolResultContent) -> Value {
    match content {
        ToolResultContent::Text { text: s } => json!(s),
        ToolResultContent::Json { value: v } => {
            // Anthropic 接受一个 text 块，正文是序列化后的 JSON。
            json!([{"type": "text", "text": v.to_string()}])
        }
        ToolResultContent::Blocks { blocks } => Value::Array(
            blocks
                .iter()
                .map(|b| match b {
                    ToolResultBlock::Text { text } => json!({"type": "text", "text": text}),
                    ToolResultBlock::Image { url, .. } => json!({
                        "type": "image",
                        "source": image_source_to_wire(&ImageSource::from_url(url)),
                    }),
                })
                .collect(),
        ),
    }
}

fn block_to_wire(b: &Block) -> Value {
    match b {
        Block::Text { text, cache } => {
            attach_cache(json!({"type": "text", "text": text}), cache.as_ref())
        }
        Block::Image {
            source,
            detail: _,
            cache,
        } => attach_cache(
            json!({"type": "image", "source": image_source_to_wire(source)}),
            cache.as_ref(),
        ),
        Block::ToolUse {
            id,
            name,
            input,
            cache,
        } => attach_cache(
            json!({"type": "tool_use", "id": id, "name": name, "input": input}),
            cache.as_ref(),
        ),
        Block::ToolResult {
            tool_use_id,
            content,
            is_error,
            cache,
        } => {
            let mut v = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result_content_to_wire(content),
            });
            if *is_error {
                v["is_error"] = json!(true);
            }
            attach_cache(v, cache.as_ref())
        }
        Block::Thinking { text, signature } => {
            let mut v = json!({"type": "thinking", "thinking": text});
            if let Some(sig) = signature {
                v["signature"] = json!(sig);
            }
            v
        }
        Block::RedactedThinking { data } => json!({"type": "redacted_thinking", "data": data}),
        // 未建模的块原样写回 —— 同协议直通时一个字节都不改。
        Block::Unmodeled { raw } => raw.clone(),
    }
}

// ---------------------------------------------------------------------------
// 响应解码
// ---------------------------------------------------------------------------

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
        ..Default::default()
    }
}

/// 单个内容块 → IR。认不出的类型整体进 [`Block::Unmodeled`]，不丢。
pub fn block_from_wire(v: Value) -> Block {
    let kind = v.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "text" => Block::Text {
            text: v
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            cache: None,
        },
        "thinking" => Block::Thinking {
            text: v
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            signature: v
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_string),
        },
        "redacted_thinking" => Block::RedactedThinking {
            data: v
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        "tool_use" => Block::ToolUse {
            id: v
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: v
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            input: v.get("input").cloned().unwrap_or_else(|| json!({})),
            cache: None,
        },
        _ => Block::Unmodeled { raw: v },
    }
}

fn decode_response(body: &[u8]) -> Result<WireResponse, LLMError> {
    #[derive(Deserialize)]
    struct Dto {
        id: Option<String>,
        model: Option<String>,
        #[serde(default)]
        content: Vec<Value>,
        #[serde(default)]
        stop_reason: Option<String>,
        #[serde(default)]
        usage: Option<AnthropicUsageDto>,
        #[serde(default, flatten)]
        extra: indexmap::IndexMap<String, Value>,
    }

    let dto: Dto = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("anthropic: invalid JSON: {e}"),
        ctx: ErrorContext::default(),
    })?;

    // `stop_reason` / `usage` 是已建模字段，不该落进 extra。
    let mut extra = dto.extra;
    extra.shift_remove("content");
    extra.shift_remove("stop_reason");
    extra.shift_remove("usage");
    extra.shift_remove("uuid");

    Ok(WireResponse {
        id: dto.id,
        model: dto.model,
        content: dto.content.into_iter().map(block_from_wire).collect(),
        finish_reason: FinishReason::parse_opt(dto.stop_reason.as_deref()),
        usage: dto.usage.as_ref().map(build_usage).unwrap_or_default(),
        extra,
    })
}

// ---------------------------------------------------------------------------
// 流式解码
// ---------------------------------------------------------------------------

/// Anthropic 的 SSE 映射。
///
/// 需要记住 content-block index → tool_use id 的对应：`input_json_delta` 只带
/// index、不带 id，而 `StreamChunk::ToolCallDelta` 必须带 id。
#[derive(Debug, Default, Clone)]
pub struct AnthropicSseMapper {
    tool_calls: std::collections::HashMap<u32, String>,
}

impl SseMapper for AnthropicSseMapper {
    fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError> {
        parse_event(frame.event.as_deref(), &frame.data, &mut self.tool_calls)
    }
}

#[derive(Debug, Deserialize)]
struct EventDto {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    delta: Option<Value>,
    #[serde(default)]
    content_block: Option<Value>,
    #[serde(default)]
    message: Option<Value>,
    #[serde(default)]
    usage: Option<AnthropicUsageDto>,
    #[serde(default)]
    error: Option<Value>,
}

fn parse_event(
    event_name: Option<&str>,
    data: &str,
    tool_calls: &mut std::collections::HashMap<u32, String>,
) -> Result<SseOutcome, LLMError> {
    let dto: EventDto = serde_json::from_str(data).map_err(|e| LLMError::Parse {
        message: format!("anthropic stream: invalid JSON: {e}"),
        ctx: ErrorContext::default(),
    })?;
    let kind = dto.kind.as_deref().or(event_name).unwrap_or("");
    let mut out = Vec::new();
    match kind {
        "message_start" => {
            if let Some(msg) = dto.message {
                out.push(StreamChunk::MessageStart {
                    id: msg
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    model: msg
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }
        "content_block_start" => {
            if let (Some(idx), Some(block)) = (dto.index, dto.content_block) {
                // 认不出的块（服务端工具活动等）由 `block_from_wire` 整体兜成
                // `Unmodeled`，UI 因此能把它渲染出来而不是当成不存在。
                let block = block_from_wire(block);
                if let Block::ToolUse { id, .. } = &block {
                    // `input_json_delta` 只带 index、不带 id，这里先把对应关系记下。
                    tool_calls.insert(idx, id.clone());
                }
                out.push(StreamChunk::BlockStart { index: idx, block });
            }
        }
        "content_block_delta" => {
            let idx = dto.index.unwrap_or(0);
            if let Some(delta) = dto.delta {
                let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::TextDelta {
                                index: idx,
                                text: text.to_string(),
                            });
                        }
                    }
                    "thinking_delta" => {
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::ThinkingDelta {
                                index: idx,
                                text: text.to_string(),
                                signature_delta: None,
                            });
                        }
                    }
                    "signature_delta" => {
                        if let Some(sig) = delta.get("signature").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::ThinkingDelta {
                                index: idx,
                                text: String::new(),
                                signature_delta: Some(sig.to_string()),
                            });
                        }
                    }
                    "input_json_delta" => {
                        if let Some(args) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            out.push(StreamChunk::ToolCallDelta {
                                index: idx,
                                id: tool_calls.get(&idx).cloned().unwrap_or_default(),
                                name_delta: None,
                                arguments_delta: Some(args.to_string()),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            if let Some(idx) = dto.index {
                out.push(StreamChunk::BlockStop { index: idx });
            }
        }
        "message_delta" => {
            if let Some(usage) = dto.usage.as_ref() {
                out.push(StreamChunk::Usage {
                    usage: build_usage(usage),
                });
            }
            if let Some(delta) = dto.delta {
                if let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                    out.push(StreamChunk::Finish {
                        reason: FinishReason::parse(reason),
                    });
                }
            }
        }
        "message_stop" => {
            if !out.iter().any(|c| matches!(c, StreamChunk::Finish { .. })) {
                out.push(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                });
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
    Ok(SseOutcome {
        chunks: out,
        // `message_stop` 是 Anthropic 的终止帧。不用「见到 Finish 就停」：
        // `message_delta` 也会带 stop_reason 并产出 Finish，但后面还可能跟事件。
        terminal: kind == "message_stop",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::decode_stream;
    use crate::protocol::ir::ToolSpec;
    use crate::types::{CacheMarker, ChatRequest, Conversation, SystemPrompt, Turn, UserMessage};

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.anthropic.com", "sk-ant-test")
    }

    /// 走一遍 ergonomic → IR → wire，测试里少写样板。
    fn encode(req: &ChatRequest) -> Value {
        let ir = crate::types::lift_request(req);
        let http = AnthropicCodec
            .encode_request(&ir, &cfg())
            .expect("encode_request");
        serde_json::from_slice(&http.body).unwrap()
    }

    fn build(req: &ChatRequest) -> HttpRequest {
        let ir = crate::types::lift_request(req);
        AnthropicCodec.encode_request(&ir, &cfg()).unwrap()
    }

    #[test]
    fn builds_messages_request() {
        let req = ChatRequest::anthropic(
            "claude-3-5-sonnet-20241022",
            Conversation::single_user("hi"),
        );
        let http = build(&req);
        assert_eq!(http.url, "https://api.anthropic.com/v1/messages");
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        let hdrs: std::collections::HashMap<_, _> = http
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(hdrs.get("x-api-key").copied(), Some("sk-ant-test"));
        assert_eq!(
            hdrs.get("anthropic-version").copied(),
            Some(ANTHROPIC_VERSION)
        );
    }

    #[test]
    fn cache_marker_5m_emits_default_ephemeral() {
        let conv = Conversation {
            system: Some(SystemPrompt::text("S").with_cache(CacheMarker::ephemeral_5m())),
            turns: vec![Turn::User(UserMessage::text("hi"))],
        };
        let body = encode(&ChatRequest::anthropic("claude", conv));
        let sys = &body["system"];
        assert!(
            sys.is_array(),
            "有缓存标记时 system 应为数组形态，实际 {sys}"
        );
        let cc = &sys[0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
        assert!(cc.get("ttl").is_none(), "5m 是默认档，不应出现 ttl 字段");
    }

    #[test]
    fn cache_marker_1h_emits_ttl() {
        let conv = Conversation {
            system: Some(SystemPrompt::text("S").with_cache(CacheMarker::ephemeral_1h())),
            turns: vec![Turn::User(UserMessage::text("hi"))],
        };
        let body = encode(&ChatRequest::anthropic("claude", conv));
        let cc = &body["system"][0]["cache_control"];
        assert_eq!(cc["type"], "ephemeral");
        assert_eq!(cc["ttl"], "1h");
    }

    #[test]
    fn anthropic_requires_max_tokens() {
        let mut req = ChatRequest::openai("c", Conversation::single_user("hi"));
        req.max_tokens = None;
        let ir = crate::types::lift_request(&req);
        let err = AnthropicCodec.encode_request(&ir, &cfg()).unwrap_err();
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
        let resp = AnthropicCodec
            .decode_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::ToolCall);
        // 缓存用量按 TTL 分档
        assert_eq!(resp.usage.cache.read_tokens, 3);
        assert_eq!(resp.usage.cache.write.total, 50);
        assert_eq!(resp.usage.cache.write.ephemeral_5m, 30);
        assert_eq!(resp.usage.cache.write.ephemeral_1h, 20);
        // 思考签名保留
        let thinking_sig = resp.content.iter().find_map(|b| match b {
            Block::Thinking { signature, .. } => signature.clone(),
            _ => None,
        });
        assert_eq!(thinking_sig.as_deref(), Some("sig_abc"));
        // 工具入参已结构化
        let input = resp
            .content
            .iter()
            .find_map(|b| match b {
                Block::ToolUse { input, .. } => Some(input.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(input["a"], 1);
    }

    /// 认不出的内容块必须整体存活，而不是被丢弃。
    #[test]
    fn unknown_response_blocks_survive_as_unmodeled() {
        let body = br#"{
            "id":"msg_2",
            "content":[
                {"type":"text","text":"let me search"},
                {"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"rust"}},
                {"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"text","text":"..."}]}
            ],
            "stop_reason":"end_turn"
        }"#;
        let resp = AnthropicCodec
            .decode_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.content.len(), 3);
        let unmodeled = resp
            .content
            .iter()
            .filter(|b| matches!(b, Block::Unmodeled { .. }))
            .count();
        assert_eq!(unmodeled, 2, "服务端工具的两个块都该以 Unmodeled 存活");
    }

    /// 未建模的块写回 wire 时必须逐字节不变。
    #[test]
    fn unmodeled_blocks_round_trip_unchanged() {
        let raw = json!({
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [{"type": "text", "text": "x"}],
            "cache_control": null
        });
        let block = Block::Unmodeled { raw: raw.clone() };
        let wire = block_to_wire(&block);
        assert_eq!(wire, raw);
    }

    #[test]
    fn tool_result_blocks_serialize_natively() {
        let mut conv = Conversation::single_user("ok");
        let mut user = UserMessage::default();
        user.push(crate::types::UserBlock::ToolResult {
            call_id: "t1".into(),
            content: ToolResultContent::Blocks {
                blocks: vec![
                    ToolResultBlock::Text {
                        text: "Tokyo".into(),
                    },
                    ToolResultBlock::Image {
                        url: "https://x/y.png".into(),
                        detail: None,
                    },
                ],
            },
            is_error: false,
            cache: None,
        });
        conv.push(Turn::User(user));
        let body = encode(&ChatRequest::anthropic("c", conv));
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

    /// 顶层未建模参数必须透传进请求体。
    #[test]
    fn extra_top_level_fields_pass_through() {
        let mut ir = WireRequest::new("claude");
        ir.max_tokens = Some(16);
        ir.extra.insert("top_k".into(), json!(42));
        ir.extra
            .insert("service_tier".into(), json!("standard_only"));
        // extra 不能盖掉建模字段
        ir.extra.insert("model".into(), json!("hijacked"));
        let http = AnthropicCodec.encode_request(&ir, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["top_k"], 42);
        assert_eq!(body["service_tier"], "standard_only");
        assert_eq!(body["model"], "claude", "建模字段必须赢过 extra");
    }

    #[test]
    fn server_tools_are_declared_verbatim() {
        let mut ir = WireRequest::new("claude");
        ir.max_tokens = Some(16);
        let spec = json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 3});
        ir.tools.push(ToolSpec {
            name: "web_search".into(),
            description: None,
            input_schema: json!({}),
            cache: None,
            kind: crate::protocol::ir::ToolKind::ServerTool { spec: spec.clone() },
        });
        let http = AnthropicCodec.encode_request(&ir, &cfg()).unwrap();
        let body: Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["tools"][0], spec, "服务端工具声明应原样下发");
    }

    #[tokio::test]
    async fn parses_sse_stream() {
        use futures_util::StreamExt;
        let bytes = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"he\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"y\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(bytes))]));
        let mut out = decode_stream(&AnthropicCodec, s);
        let mut acc = String::new();
        let mut saw_finish = false;
        while let Some(c) = out.next().await {
            match c.unwrap() {
                StreamChunk::TextDelta { text, .. } => acc.push_str(&text),
                StreamChunk::Finish { .. } => saw_finish = true,
                _ => {}
            }
        }
        assert_eq!(acc, "hey");
        assert!(saw_finish);
    }

    /// 初始签名（在 `content_block_start` 上）与后续 `signature_delta` 片段
    /// 必须归并进**同一个**思考块 —— 签名不完整，多轮带工具的对话会被上游拒掉。
    #[tokio::test]
    async fn stream_thinking_and_signature_merge_into_one_block() {
        use futures_util::StreamExt;
        // 初始 content_block_start 带 thinking 类型 + signature，随后 thinking_delta，
        // 再随后 signature_delta。
        let bytes = b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"signature\":\"sig_init\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"step1\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"_more\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(bytes))]));
        let mut out = decode_stream(&AnthropicCodec, s);
        let mut asm = crate::types::BlockAssembler::new();
        while let Some(c) = out.next().await {
            let c = c.unwrap();
            let done = matches!(c, StreamChunk::Finish { .. });
            asm.push(&c);
            if done {
                break;
            }
        }
        let blocks = asm.blocks();
        assert_eq!(blocks.len(), 1, "思考内容应归并成一个块：{blocks:?}");
        match &blocks[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "step1");
                assert_eq!(signature.as_deref(), Some("sig_init_more"));
            }
            other => panic!("期望 thinking，得到 {other:?}"),
        }
    }

    /// 认不出的流式内容块必须整体存活（`Unmodeled`），而不是被丢掉。
    #[tokio::test]
    async fn unknown_stream_blocks_survive_as_unmodeled() {
        use futures_util::StreamExt;
        let bytes = b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srv_1\",\"name\":\"web_search\",\"input\":{}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srv_1\",\"content\":[]}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(bytes))]));
        let mut out = decode_stream(&AnthropicCodec, s);
        let mut asm = crate::types::BlockAssembler::new();
        while let Some(c) = out.next().await {
            asm.push(&c.unwrap());
        }
        let blocks = asm.blocks();
        assert_eq!(blocks.len(), 2);
        assert!(blocks.iter().all(|b| matches!(b, Block::Unmodeled { .. })));
    }
}
