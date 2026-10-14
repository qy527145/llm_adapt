//! Anthropic 入站（网关）方向：`/v1/messages` 报文 → IR → `/v1/messages` 报文。
//!
//! 与出站方向共用同一套 wire 知识：内容块的编码直接复用 `super::block_to_wire`，
//! 因此同协议往返（解出来再编回去）逐字节一致 —— 未建模的块原样写回。
//!
//! 三条产品约束（原样搬自 apilot，都是踩过坑的）：
//! * 认不出的**顶层**字段进 [`WireRequest::extra`]，认不出的**内容块**整块进
//!   [`Block::Unmodeled`]；解不出就报错会把用户挡在门外。
//! * 流式 `message_start` 必须最先，`content_block_start` 必须早于同 index 的任何
//!   delta，`message_delta` + `message_stop` 收尾。

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::protocol::ir::{
    Block, CacheMarker, FinishReason, ImageSource, Protocol, ThinkingConfig, ToolChoice,
    ToolKind, ToolResultBlock, ToolResultContent, ToolSpec, Usage, WireMessage, WireRequest,
    WireResponse,
};
use crate::protocol::shared::{synth_id, sse_named};
use crate::protocol::codec::StreamEncoder;
use crate::types::StreamChunk;

/// 已建模的顶层字段；不在这张表里的一律进 `extra` 原样透传。
const KNOWN_TOP_LEVEL: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
    "thinking",
];

// ---------------------------------------------------------------------------
// 请求解码（wire → IR）
// ---------------------------------------------------------------------------

pub(crate) fn decode_request(body: &[u8]) -> Result<WireRequest, LLMError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("anthropic: invalid JSON: {e}"),
        ctx: Default::default(),
    })?;
    let obj = v
        .as_object()
        .ok_or_else(|| LLMError::validation("anthropic: 请求体不是 JSON 对象"))?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| LLMError::validation("anthropic: 缺少 model 字段"))?
        .to_string();

    let mut req = WireRequest::new(model);
    req.stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    req.max_tokens = obj
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|m| m as u32);
    req.temperature = obj.get("temperature").and_then(Value::as_f64).map(|f| f as f32);
    req.top_p = obj.get("top_p").and_then(Value::as_f64).map(|f| f as f32);
    req.stop = obj
        .get("stop_sequences")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if let Some(sys) = obj.get("system") {
        req.system = decode_content(sys);
    }

    req.messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| LLMError::validation("anthropic: 缺少 messages 字段"))?
        .iter()
        .map(decode_message)
        .collect::<Result<Vec<_>, _>>()?;

    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        req.tools = tools.iter().filter_map(decode_tool).collect();
    }

    if let Some(tc) = obj.get("tool_choice") {
        req.tool_choice = decode_tool_choice(tc);
    }

    if let Some(th) = obj.get("thinking") {
        if th.get("type").and_then(Value::as_str) == Some("enabled") {
            req.thinking = Some(ThinkingConfig {
                budget_tokens: th
                    .get("budget_tokens")
                    .and_then(Value::as_u64)
                    .map(|b| b as u32),
                effort: None,
            });
        }
    }

    // `metadata.user_id` 是 Anthropic 唯一建模的用户标识；同名的其它键留在 extra 里。
    if let Some(uid) = obj
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str)
    {
        req.user_id = Some(uid.to_string());
    }

    for (k, val) in obj {
        if !KNOWN_TOP_LEVEL.contains(&k.as_str()) {
            req.extra.insert(k.clone(), val.clone());
        }
    }

    Ok(req)
}

/// Anthropic 的 content 既可以是字符串，也可以是块数组，还可以缺省。
fn decode_content(v: &Value) -> Vec<Block> {
    match v {
        Value::Null => Vec::new(),
        Value::String(s) => vec![Block::text(s.clone())],
        Value::Array(items) => items.iter().map(decode_block).collect(),
        // 非法的 content 形态不该让整条请求失败：整块兜成 Unmodeled，同协议直通时
        // 原样写回，跨协议时自然降级。
        other => vec![Block::Unmodeled { raw: other.clone() }],
    }
}

/// 从块上的 `cache_control` 还原缓存断点。
fn cache_marker(v: &Value) -> Option<CacheMarker> {
    let cc = v.get("cache_control")?;
    if cc.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return None;
    }
    // 只有 1h 档显式写 ttl；其余（含缺省）都是默认的 5m 档。
    Some(match cc.get("ttl").and_then(Value::as_str) {
        Some("1h") => CacheMarker::ephemeral_1h(),
        _ => CacheMarker::ephemeral_5m(),
    })
}

fn decode_block(v: &Value) -> Block {
    let cache = cache_marker(v);
    let kind = v.get("type").and_then(Value::as_str).unwrap_or("text");
    match kind {
        "text" => Block::Text {
            text: v
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            cache,
        },
        "image" => Block::Image {
            source: decode_image_source(v.get("source")),
            detail: None,
            cache,
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
            cache,
        },
        "tool_result" => Block::ToolResult {
            tool_use_id: v
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content: decode_tool_result_content(v.get("content").unwrap_or(&Value::Null)),
            is_error: v
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            cache,
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
        // 解不出不等于这次请求该失败 —— Claude Code 带附件时发的 `document` 就走这里。
        _ => Block::Unmodeled { raw: v.clone() },
    }
}

fn decode_image_source(source: Option<&Value>) -> ImageSource {
    let Some(src) = source else {
        return ImageSource::Url {
            url: String::new(),
        };
    };
    match src.get("type").and_then(Value::as_str) {
        Some("url") => ImageSource::from_url(
            src.get("url").and_then(Value::as_str).unwrap_or_default(),
        ),
        _ => ImageSource::Base64 {
            media_type: src
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png")
                .to_string(),
            data: src
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
    }
}

fn decode_tool_result_content(v: &Value) -> ToolResultContent {
    match v {
        Value::String(s) => ToolResultContent::Text { text: s.clone() },
        Value::Array(items) => ToolResultContent::Blocks {
            blocks: items
                .iter()
                .filter_map(|b| match b.get("type").and_then(Value::as_str) {
                    Some("text") => Some(ToolResultBlock::Text {
                        text: b
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    Some("image") => Some(ToolResultBlock::Image {
                        url: decode_image_source(b.get("source")).to_url(),
                        detail: None,
                    }),
                    // 结果里出现别的块（嵌套工具调用等）丢掉：它不该出现在这里，
                    // 硬塞进 [Text|Image] 反而会造出非法 wire。
                    _ => None,
                })
                .collect(),
        },
        _ => ToolResultContent::Text {
            text: String::new(),
        },
    }
}

fn decode_message(v: &Value) -> Result<WireMessage, LLMError> {
    let role = match v.get("role").and_then(Value::as_str) {
        Some("assistant") => crate::protocol::ir::Role::Assistant,
        // Anthropic 只有 user / assistant；tool_result 也放在 user 消息里。
        _ => crate::protocol::ir::Role::User,
    };
    Ok(WireMessage::new(
        role,
        decode_content(v.get("content").unwrap_or(&Value::Null)),
    ))
}

fn decode_tool(v: &Value) -> Option<ToolSpec> {
    let name = v.get("name")?.as_str()?.to_string();
    // 服务端工具（web_search 等）没有 `input_schema`，只有厂商的 `type` 声明；
    // 本地函数工具反之。用这一点区分，声明对象原样收进 `ToolKind::ServerTool::spec`。
    if v.get("input_schema").is_none() && v.get("type").is_some() {
        return Some(ToolSpec {
            name,
            description: v
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            input_schema: json!({}),
            cache: cache_marker(v),
            kind: ToolKind::ServerTool { spec: v.clone() },
        });
    }
    Some(ToolSpec {
        name,
        description: v
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string),
        input_schema: v
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        cache: cache_marker(v),
        kind: ToolKind::Function,
    })
}

fn decode_tool_choice(v: &Value) -> ToolChoice {
    match v.get("type").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        // Anthropic 用 "any" 表达「必须调用某个工具」。
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => v
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Specific { name: n.to_string() })
            .unwrap_or_default(),
        _ => ToolChoice::Auto,
    }
}

// ---------------------------------------------------------------------------
// 响应编码（IR → wire）
// ---------------------------------------------------------------------------

pub(crate) fn encode_response(resp: &WireResponse) -> Result<Vec<u8>, LLMError> {
    let mut obj = Map::new();
    obj.insert("id".into(), json!(resp.id.clone().unwrap_or_default()));
    obj.insert("type".into(), json!("message"));
    obj.insert("role".into(), json!("assistant"));
    obj.insert("model".into(), json!(resp.model.clone().unwrap_or_default()));
    obj.insert(
        "content".into(),
        Value::Array(resp.content.iter().map(super::block_to_wire).collect()),
    );
    // 终止原因归一后写回 Anthropic 的取值（Stop → end_turn 等）。
    obj.insert(
        "stop_reason".into(),
        json!(resp.finish_reason.to_wire(Protocol::Anthropic)),
    );
    obj.insert("stop_sequence".into(), Value::Null);
    obj.insert("usage".into(), encode_usage(&resp.usage));
    Ok(serde_json::to_vec(&Value::Object(obj))?)
}

/// IR 用量 → Anthropic 口径。
///
/// 两边口径一致（`input_tokens` 都不含缓存），直接对应；缓存字段只在非零时输出，
/// 免得给不认识的客户端塞一堆 0。
fn encode_usage(u: &Usage) -> Value {
    let mut m = Map::new();
    m.insert("input_tokens".into(), json!(u.input_tokens));
    m.insert("output_tokens".into(), json!(u.output_tokens));
    if u.cache.read_tokens > 0 {
        m.insert(
            "cache_read_input_tokens".into(),
            json!(u.cache.read_tokens),
        );
    }
    if u.cache.write.total > 0 {
        m.insert(
            "cache_creation_input_tokens".into(),
            json!(u.cache.write.total),
        );
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// 流式编码（IR delta → SSE 帧）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
    Other,
}

/// 把 [`StreamChunk`] 编成客户端可读的 Anthropic SSE 序列。
///
/// 关键顺序约束由本编码器统一补齐，而不是指望上游遵守：
/// `message_start` 最先、`content_block_start` 早于同 index 的任何 delta、
/// `content_block_stop` 后不再发该 index 的 delta、`message_stop` 收尾。
pub struct AnthropicStreamEncoder {
    started: bool,
    id: String,
    model: String,
    /// 当前处于开启状态的块 index。
    open_block: Option<u32>,
    finished: bool,
    usage: Usage,
    finish_reason: Option<FinishReason>,
    /// 各 index 的块类型（BlockStart 时记下），用于在缺 BlockStart 时补对形状。
    kinds: HashMap<u32, BlockKind>,
    /// index → (id, name)，补 tool_use 开头要用。
    tool_meta: HashMap<u32, (String, String)>,
}

impl Default for AnthropicStreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicStreamEncoder {
    pub fn new() -> Self {
        Self {
            started: false,
            id: String::new(),
            model: String::new(),
            open_block: None,
            finished: false,
            usage: Usage::default(),
            finish_reason: None,
            kinds: HashMap::new(),
            tool_meta: HashMap::new(),
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        if self.id.is_empty() {
            self.id = synth_id("msg");
        }
        let payload = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            }
        });
        out.push(sse_named("message_start", &payload.to_string()));
    }

    fn close_open_block(&mut self, out: &mut Vec<String>) {
        if let Some(idx) = self.open_block.take() {
            let payload = json!({ "type": "content_block_stop", "index": idx });
            out.push(sse_named("content_block_stop", &payload.to_string()));
        }
    }

    /// 确保 index 对应的块已开启；未开启则按记录的类型补一个 `content_block_start`。
    fn ensure_open(&mut self, index: u32, out: &mut Vec<String>) {
        if self.open_block == Some(index) {
            return;
        }
        self.close_open_block(out);
        let content_block = match self.kinds.get(&index).copied() {
            Some(BlockKind::ToolUse) => {
                let (id, name) = self
                    .tool_meta
                    .get(&index)
                    .cloned()
                    .unwrap_or_default();
                json!({ "type": "tool_use", "id": id, "name": name, "input": {} })
            }
            Some(BlockKind::Thinking) => json!({ "type": "thinking", "thinking": "" }),
            _ => json!({ "type": "text", "text": "" }),
        };
        let payload = json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        });
        out.push(sse_named("content_block_start", &payload.to_string()));
        self.open_block = Some(index);
    }
}

impl StreamEncoder for AnthropicStreamEncoder {
    fn encode(&mut self, chunk: &StreamChunk) -> Result<Vec<String>, LLMError> {
        let mut out = Vec::new();
        match chunk {
            StreamChunk::MessageStart { id, model } => {
                self.id = id.clone();
                self.model = model.clone();
                self.ensure_started(&mut out);
            }

            StreamChunk::BlockStart { index, block } => {
                self.ensure_started(&mut out);
                self.close_open_block(&mut out);
                let content_block = match block {
                    Block::Text { .. } => {
                        self.kinds.insert(*index, BlockKind::Text);
                        json!({ "type": "text", "text": "" })
                    }
                    Block::Thinking { signature, .. } => {
                        self.kinds.insert(*index, BlockKind::Thinking);
                        let mut v = json!({ "type": "thinking", "thinking": "" });
                        // 初始签名（Responses 的 encrypted_content 可能落在这里）。
                        if let Some(sig) = signature {
                            v["signature"] = json!(sig);
                        }
                        v
                    }
                    Block::ToolUse { id, name, .. } => {
                        self.kinds.insert(*index, BlockKind::ToolUse);
                        self.tool_meta.insert(*index, (id.clone(), name.clone()));
                        json!({ "type": "tool_use", "id": id, "name": name, "input": {} })
                    }
                    // 未建模的块原样重发：它本来可能就是 Anthropic 的形状，
                    // 「我们没建模」不是改它的理由。
                    Block::Unmodeled { raw } => {
                        self.kinds.insert(*index, BlockKind::Other);
                        raw.clone()
                    }
                    // 其余（图片 / 工具结果 / 加密思考）没有空的占位形态，退化成文本块。
                    _ => {
                        self.kinds.insert(*index, BlockKind::Other);
                        json!({ "type": "text", "text": "" })
                    }
                };
                let payload = json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": content_block,
                });
                out.push(sse_named("content_block_start", &payload.to_string()));
                self.open_block = Some(*index);
            }

            StreamChunk::TextDelta { index, text } => {
                self.ensure_started(&mut out);
                self.ensure_open(*index, &mut out);
                let payload = json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": text },
                });
                out.push(sse_named("content_block_delta", &payload.to_string()));
            }

            StreamChunk::ThinkingDelta {
                index,
                text,
                signature_delta,
            } => {
                self.ensure_started(&mut out);
                self.ensure_open(*index, &mut out);
                if !text.is_empty() {
                    let payload = json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "thinking_delta", "thinking": text },
                    });
                    out.push(sse_named("content_block_delta", &payload.to_string()));
                }
                if let Some(sig) = signature_delta {
                    let payload = json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": { "type": "signature_delta", "signature": sig },
                    });
                    out.push(sse_named("content_block_delta", &payload.to_string()));
                }
            }

            StreamChunk::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                self.ensure_started(&mut out);
                // 先把 id / name 记下，补开头时才有东西可用。
                if !id.is_empty() || name_delta.is_some() {
                    let meta = self
                        .tool_meta
                        .entry(*index)
                        .or_insert_with(|| (id.clone(), String::new()));
                    if meta.0.is_empty() {
                        meta.0 = id.clone();
                    }
                    if let Some(n) = name_delta {
                        meta.1.push_str(n);
                    }
                    self.kinds.entry(*index).or_insert(BlockKind::ToolUse);
                }
                self.ensure_open(*index, &mut out);
                if let Some(args) = arguments_delta {
                    if !args.is_empty() {
                        let payload = json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": { "type": "input_json_delta", "partial_json": args },
                        });
                        out.push(sse_named("content_block_delta", &payload.to_string()));
                    }
                }
            }

            StreamChunk::BlockStop { index } => {
                self.ensure_started(&mut out);
                if self.open_block == Some(*index) {
                    self.close_open_block(&mut out);
                }
            }

            StreamChunk::Usage { usage } => self.usage.merge_non_zero(usage),

            StreamChunk::Finish { reason } => self.finish_reason = Some(reason.clone()),

            StreamChunk::Error { error } => {
                self.ensure_started(&mut out);
                let payload = json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": error },
                });
                out.push(sse_named("error", &payload.to_string()));
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<String>, LLMError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        let mut out = Vec::new();
        self.ensure_started(&mut out);
        self.close_open_block(&mut out);

        let reason = self
            .finish_reason
            .clone()
            .unwrap_or(FinishReason::Stop)
            .to_wire(Protocol::Anthropic);
        let mut usage = json!({ "output_tokens": self.usage.output_tokens });
        // input 侧在 message_start 已给过，这里只按 Anthropic 惯例补 output 侧。
        if self.usage.cache.read_tokens > 0 {
            usage["cache_read_input_tokens"] = json!(self.usage.cache.read_tokens);
        }
        let payload = json!({
            "type": "message_delta",
            "delta": { "stop_reason": reason, "stop_sequence": Value::Null },
            "usage": usage,
        });
        out.push(sse_named("message_delta", &payload.to_string()));

        out.push(sse_named(
            "message_stop",
            &json!({ "type": "message_stop" }).to_string(),
        ));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::http::ClientConfig;
    use crate::protocol::anthropic::AnthropicCodec;
    use crate::protocol::codec::Codec;

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.anthropic.com", "sk-ant")
    }

    /// 解出来再用出站编码器编回去，语义字段应当一致。
    fn roundtrip(body: &str) -> (WireRequest, Value) {
        let ir = decode_request(body.as_bytes()).expect("入站解码");
        let http = AnthropicCodec
            .encode_request(&ir, &cfg())
            .expect("出站编码");
        let out: Value = serde_json::from_slice(&http.body).unwrap();
        (ir, out)
    }

    #[test]
    fn decodes_and_reencodes_core_fields() {
        let (_ir, out) = roundtrip(
            r#"{"model":"claude-sonnet-5","max_tokens":100,"temperature":0.5,
                "system":"be brief",
                "messages":[{"role":"user","content":"hi"},
                            {"role":"assistant","content":[{"type":"text","text":"ok"}]}]}"#,
        );
        assert_eq!(out["model"], "claude-sonnet-5");
        assert_eq!(out["max_tokens"], 100);
        assert_eq!(out["system"], "be brief");
        assert_eq!(out["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(out["messages"][1]["role"], "assistant");
    }

    #[test]
    fn base64_and_url_images_survive() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}},
            {"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}"#;
        let (ir, _) = roundtrip(body);
        let imgs: Vec<&Block> = ir.messages[0]
            .content
            .iter()
            .filter(|b| matches!(b, Block::Image { .. }))
            .collect();
        assert_eq!(imgs.len(), 2);
        match imgs[0] {
            Block::Image { source, .. } => assert_eq!(
                *source,
                ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "AAA".into()
                }
            ),
            _ => unreachable!(),
        }
    }

    #[test]
    fn cache_control_is_modeled_on_decode() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}"#;
        let (ir, _) = roundtrip(body);
        assert_eq!(
            ir.messages[0].content[0],
            Block::text_cached("x", CacheMarker::ephemeral_1h())
        );
    }

    #[test]
    fn unknown_top_level_fields_land_in_extra() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","max_tokens":1,"messages":[],"metadata":{"user_id":"u1"},
                "service_tier":"standard_only"}"#,
        );
        assert_eq!(ir.user_id.as_deref(), Some("u1"));
        assert_eq!(ir.extra.get("service_tier"), Some(&json!("standard_only")));
    }

    /// 解不出的内容块整块原样留着，同协议编回去一个字节都不改。
    #[test]
    fn unknown_content_block_survives_verbatim() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AAA"},"title":"a.pdf"},
            {"type":"text","text":"look"}]}]}"#;
        let (ir, out) = roundtrip(body);
        assert!(
            matches!(ir.messages[0].content[0], Block::Unmodeled { .. }),
            "未知块应整块留下"
        );
        assert_eq!(out["messages"][0]["content"][0]["type"], "document");
        assert_eq!(out["messages"][0]["content"][0]["title"], "a.pdf");
        assert_eq!(out["messages"][0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn thinking_budget_and_tool_use_decode() {
        let body = r#"{"model":"m","max_tokens":1,"thinking":{"type":"enabled","budget_tokens":2048},
            "messages":[{"role":"assistant","content":[
                {"type":"thinking","thinking":"hmm","signature":"sig"},
                {"type":"tool_use","id":"tu_1","name":"search","input":{"q":"rust"}}]}],
            "tools":[{"name":"search","description":"d","input_schema":{"type":"object"}}]}"#;
        let (ir, out) = roundtrip(body);
        assert_eq!(ir.thinking.as_ref().unwrap().budget_tokens, Some(2048));
        assert_eq!(ir.tools.len(), 1);
        assert!(ir.tools[0].is_function());
        match &ir.messages[0].content[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "hmm");
                assert_eq!(signature.as_deref(), Some("sig"));
            }
            other => panic!("期望 thinking，得到 {other:?}"),
        }
        assert_eq!(out["thinking"]["budget_tokens"], 2048);
        assert_eq!(out["messages"][0]["content"][1]["name"], "search");
    }

    #[test]
    fn server_tool_declaration_is_preserved() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[],
            "tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}"#;
        let ir = decode_request(body.as_bytes()).unwrap();
        assert!(ir.tools[0].is_server_tool());
    }

    #[test]
    fn encode_response_round_trips() {
        let raw = br#"{"id":"msg_9","model":"claude-opus-5","type":"message",
            "content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}],
            "stop_reason":"tool_use",
            "usage":{"input_tokens":7,"output_tokens":3,"cache_read_input_tokens":30}}"#;
        let resp = AnthropicCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        let out: Value = serde_json::from_slice(&encode_response(&resp).unwrap()).unwrap();
        assert_eq!(out["id"], "msg_9");
        assert_eq!(out["type"], "message");
        assert_eq!(out["role"], "assistant");
        assert_eq!(out["content"][0]["text"], "ok");
        assert_eq!(out["content"][1]["type"], "tool_use");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["usage"]["input_tokens"], 7);
        assert_eq!(out["usage"]["cache_read_input_tokens"], 30);
        assert!(out["usage"].get("cache_creation_input_tokens").is_none());
    }

    // --- 流式编码 ---

    fn events(frames: &[String]) -> Vec<String> {
        frames
            .iter()
            .filter_map(|f| {
                f.lines()
                    .find_map(|l| l.strip_prefix("event: ").map(str::to_string))
            })
            .collect()
    }

    #[test]
    fn stream_encoder_emits_well_formed_sequence() {
        let mut enc = AnthropicStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(
            enc.encode(&StreamChunk::MessageStart {
                id: "msg_x".into(),
                model: "claude".into(),
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::BlockStart {
                index: 0,
                block: Block::text(""),
            })
            .unwrap(),
        );
        out.extend(enc.encode(&StreamChunk::text("Hi")).unwrap());
        out.extend(enc.encode(&StreamChunk::BlockStop { index: 0 }).unwrap());
        out.extend(
            enc.encode(&StreamChunk::Usage {
                usage: Usage {
                    output_tokens: 2,
                    ..Default::default()
                },
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::Finish {
                reason: FinishReason::Stop,
            })
            .unwrap(),
        );
        out.extend(enc.finish().unwrap());

        let n = events(&out);
        assert_eq!(
            n,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // message_start 必须最先，message_stop 必须最后。
        assert!(out[0].starts_with("event: message_start"));
        assert!(out.last().unwrap().starts_with("event: message_stop"));
        // message_delta 带正确的 stop_reason 与 output_tokens。
        let md = out.iter().find(|f| f.contains("message_delta")).unwrap();
        assert!(md.contains("\"end_turn\""));
        assert!(md.contains("\"output_tokens\":2"));
        // 每帧以空行结尾（可直接写 socket）。
        assert!(out.iter().all(|f| f.ends_with("\n\n")));
    }

    #[test]
    fn stream_encoder_tool_call_produces_input_json_delta() {
        let mut enc = AnthropicStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(
            enc.encode(&StreamChunk::BlockStart {
                index: 0,
                block: Block::ToolUse {
                    id: "tu_1".into(),
                    name: "search".into(),
                    input: json!({}),
                    cache: None,
                },
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::ToolCallDelta {
                index: 0,
                id: "tu_1".into(),
                name_delta: None,
                arguments_delta: Some(r#"{"q":"x"}"#.into()),
            })
            .unwrap(),
        );
        let joined = out.join("");
        assert!(joined.contains("\"type\":\"tool_use\""));
        assert!(joined.contains("\"id\":\"tu_1\""));
        assert!(joined.contains("input_json_delta"));
    }

    #[test]
    fn stream_encoder_finish_is_idempotent() {
        let mut enc = AnthropicStreamEncoder::new();
        enc.encode(&StreamChunk::text("x")).unwrap();
        assert!(!enc.finish().unwrap().is_empty());
        assert!(enc.finish().unwrap().is_empty());
    }
}
