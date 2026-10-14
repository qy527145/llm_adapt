//! OpenAI Chat Completions 入站（网关）方向。
//!
//! 与 Anthropic 侧同一个思路：认不出的顶层字段进 `extra`，认不出的内容块进
//! [`Block::Unmodeled`]。Chat 特有的一点是**消息形态的折返**：
//!
//! * `role: "system"` / `"developer"` 提升到顶层 `system`（IR 与 Anthropic 一致）；
//! * `role: "tool"` 消息折回 user 消息里的 `tool_result` 块；
//! * assistant 的 `reasoning_content` 折回 `Thinking`，`tool_calls` 折回 `ToolUse`。
//!
//! `content` 上的 `cache_control` 在 Chat 里没有对应物（OpenAI 自动缓存），
//! 解码时被忽略 —— 这是 [`crate::protocol::convert`] 损耗清单里的一条。

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::protocol::codec::StreamEncoder;
use crate::protocol::ir::{
    Block, FinishReason, ImageSource, Protocol, ThinkingConfig, ToolChoice, ToolKind, ToolResultContent,
    ToolSpec, Usage, WireMessage, WireRequest, WireResponse,
};
use crate::protocol::shared::{now_secs, synth_id, sse_frame};
use crate::types::StreamChunk;

/// 已建模的顶层字段；其余进 `extra` 透传。
///
/// `stream_options` 刻意**不在**表里：编码时要保留用户设置的其他键，只在 extra
/// 的基础上强制打开 usage。
const KNOWN_TOP_LEVEL: &[&str] = &[
    "model",
    "messages",
    "tools",
    "tool_choice",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
    "reasoning_effort",
];

// ---------------------------------------------------------------------------
// 请求解码（wire → IR）
// ---------------------------------------------------------------------------

pub(crate) fn decode_request(body: &[u8]) -> Result<WireRequest, LLMError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("openai_chat: invalid JSON: {e}"),
        ctx: Default::default(),
    })?;
    let obj = v
        .as_object()
        .ok_or_else(|| LLMError::validation("openai_chat: 请求体不是 JSON 对象"))?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| LLMError::validation("openai_chat: 缺少 model 字段"))?
        .to_string();

    let mut req = WireRequest::new(model);
    req.stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // 旧字段 max_tokens 与新字段 max_completion_tokens 都认。
    req.max_tokens = obj
        .get("max_tokens")
        .or_else(|| obj.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map(|m| m as u32);
    req.temperature = obj.get("temperature").and_then(Value::as_f64).map(|f| f as f32);
    req.top_p = obj.get("top_p").and_then(Value::as_f64).map(|f| f as f32);

    req.stop = match obj.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    };

    if let Some(effort) = obj.get("reasoning_effort").and_then(Value::as_str) {
        req.thinking = Some(ThinkingConfig {
            budget_tokens: None,
            effort: Some(effort.to_string()),
        });
    }

    let messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| LLMError::validation("openai_chat: 缺少 messages 字段"))?;
    for m in messages {
        decode_message(m, &mut req);
    }

    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        req.tools = tools.iter().filter_map(decode_tool).collect();
    }
    if let Some(tc) = obj.get("tool_choice") {
        req.tool_choice = decode_tool_choice(tc);
    }

    for (k, val) in obj {
        if !KNOWN_TOP_LEVEL.contains(&k.as_str()) {
            req.extra.insert(k.clone(), val.clone());
        }
    }

    Ok(req)
}

fn decode_message(v: &Value, req: &mut WireRequest) {
    let role = v.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = v.get("content").unwrap_or(&Value::Null);

    match role {
        "system" | "developer" => {
            // developer 是 OpenAI 给推理模型准备的 system 替身，语义等价。
            req.system.extend(decode_content(content));
        }
        "tool" => {
            // 独立的 tool 消息 → user 消息里的 tool_result 块。
            req.messages.push(WireMessage::new(
                crate::protocol::ir::Role::User,
                vec![Block::ToolResult {
                    tool_use_id: v
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    content: tool_result_text(content),
                    is_error: false,
                    cache: None,
                }],
            ));
        }
        "assistant" => {
            let mut blocks = Vec::new();
            // 推理内容先于正文，保持与实际生成顺序一致。
            if let Some(rc) = v.get("reasoning_content").and_then(Value::as_str) {
                if !rc.is_empty() {
                    blocks.push(Block::Thinking {
                        text: rc.to_string(),
                        signature: None,
                    });
                }
            }
            blocks.extend(decode_content(content));
            if let Some(calls) = v.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let func = call.get("function").unwrap_or(&Value::Null);
                    blocks.push(Block::ToolUse {
                        id: call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: func
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        input: parse_arguments(
                            func.get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}"),
                        ),
                        cache: None,
                    });
                }
            }
            req.messages
                .push(WireMessage::new(crate::protocol::ir::Role::Assistant, blocks));
        }
        _ => {
            req.messages.push(WireMessage::new(
                crate::protocol::ir::Role::User,
                decode_content(content),
            ));
        }
    }
}

fn decode_content(v: &Value) -> Vec<Block> {
    match v {
        Value::Null => Vec::new(),
        Value::String(s) => vec![Block::text(s.clone())],
        Value::Array(items) => items.iter().map(decode_part).collect(),
        other => vec![Block::Unmodeled { raw: other.clone() }],
    }
}

fn decode_part(v: &Value) -> Block {
    match v.get("type").and_then(Value::as_str).unwrap_or("text") {
        "text" | "input_text" => Block::text(
            v.get("text").and_then(Value::as_str).unwrap_or_default(),
        ),
        "image_url" => {
            let url = v
                .get("image_url")
                .and_then(|i| i.get("url"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Block::Image {
                // `data:` URL 拆成结构化的 Base64 来源；远程 URL 原样。
                source: ImageSource::from_url(url),
                detail: v
                    .get("image_url")
                    .and_then(|i| i.get("detail"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                cache: None,
            }
        }
        // refusal 是一条拒绝说明，当普通文本带走。
        "refusal" => Block::text(
            v.get("refusal").and_then(Value::as_str).unwrap_or_default(),
        ),
        // 认不出的 part 整块留着（与 Anthropic 侧同一个处置）。
        _ => Block::Unmodeled { raw: v.clone() },
    }
}

fn tool_result_text(v: &Value) -> ToolResultContent {
    match v {
        Value::String(s) => ToolResultContent::Text { text: s.clone() },
        Value::Array(items) => {
            let joined: String = items
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .or_else(|| p.get("refusal"))
                        .and_then(Value::as_str)
                })
                .collect();
            ToolResultContent::Text { text: joined }
        }
        _ => ToolResultContent::Text {
            text: String::new(),
        },
    }
}

fn decode_tool(v: &Value) -> Option<ToolSpec> {
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function");
    let func = v.get("function").unwrap_or(v);
    let name = func.get("name").and_then(Value::as_str)?.to_string();
    let kind = if ty == "function" {
        ToolKind::Function
    } else {
        // 非函数工具（web_search 之类的厂商扩展）：原样保留声明，别当本地函数。
        ToolKind::ServerTool { spec: v.clone() }
    };
    Some(ToolSpec {
        name,
        description: func
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_string),
        input_schema: func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"})),
        cache: None,
        kind,
    })
}

fn decode_tool_choice(v: &Value) -> ToolChoice {
    match v {
        Value::String(s) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Value::Object(_) => v
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| v.get("name"))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Specific { name: n.to_string() })
            .unwrap_or_default(),
        _ => ToolChoice::Auto,
    }
}

/// 解析工具入参的 JSON 字符串。截断时上游会给不合法的 JSON，退回空对象并告警 ——
/// 为一个工具入参让整次对话崩掉，代价远大于收益。
fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return json!({});
    }
    serde_json::from_str(raw).unwrap_or_else(|_| json!({}))
}

// ---------------------------------------------------------------------------
// 响应编码（IR → wire）
// ---------------------------------------------------------------------------

pub(crate) fn encode_response(resp: &WireResponse) -> Result<Vec<u8>, LLMError> {
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));

    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning: Option<String> = None;

    for b in &resp.content {
        match b {
            Block::Text { text: t, .. } => text.push_str(t),
            Block::Thinking { text: t, .. } => {
                let acc = reasoning.get_or_insert_with(String::new);
                acc.push_str(t);
            }
            Block::ToolUse {
                id, name, input, ..
            } => tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    // OpenAI 的 arguments 是 JSON **字符串**，不是对象。
                    "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                },
            })),
            // 加密思考与未建模块在 Chat 里没有对应表示；响应侧丢弃，不静默改形状。
            _ => {}
        }
    }

    message.insert(
        "content".into(),
        if text.is_empty() {
            Value::Null
        } else {
            json!(text)
        },
    );
    if let Some(r) = reasoning {
        message.insert("reasoning_content".into(), json!(r));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }

    let body = json!({
        "id": resp.id.clone().unwrap_or_else(|| synth_id("chatcmpl")),
        "object": "chat.completion",
        "created": now_secs(),
        "model": resp.model.clone().unwrap_or_default(),
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": resp.finish_reason.to_wire(Protocol::OpenAiChat),
        }],
        "usage": encode_usage(&resp.usage),
    });
    Ok(serde_json::to_vec(&body)?)
}

/// IR 用量 → OpenAI 口径。
///
/// 反向折算：`prompt_tokens` 必须**含**缓存，客户端拿它做分母算命中率。
fn encode_usage(u: &Usage) -> Value {
    let prompt_tokens = u.total_input_billed();
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": prompt_tokens.saturating_add(u.output_tokens),
    });
    if u.cache.read_tokens > 0 {
        usage["prompt_tokens_details"] = json!({ "cached_tokens": u.cache.read_tokens });
    }
    if u.thinking_tokens > 0 {
        usage["completion_tokens_details"] = json!({ "reasoning_tokens": u.thinking_tokens });
    }
    usage
}

// ---------------------------------------------------------------------------
// 流式编码（IR delta → SSE 帧）
// ---------------------------------------------------------------------------

/// 把 [`StreamChunk`] 编成客户端可读的 Chat Completions SSE 序列。
///
/// Chat 的流式协议没有块边界，但它有 `tool_calls[].index` 这个位置概念：
/// IR 里不同的块 index 映射到从 0 连续递增的 tool_call index。收尾固定为
/// `finish_reason` 帧 + usage 帧 + `data: [DONE]`。
#[derive(Default)]
pub struct ChatStreamEncoder {
    id: String,
    model: String,
    started: bool,
    finished: bool,
    /// IR 的 block index → OpenAI 的 tool_call index。
    tool_index: HashMap<u32, u32>,
    next_tool_index: u32,
    usage: Usage,
    finish_reason: Option<FinishReason>,
}

impl ChatStreamEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn chunk(&self, delta: Value, finish_reason: Value) -> String {
        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        });
        // Chat 不用 `event:` 行，只有 data。
        sse_frame(None, &payload.to_string())
    }

    fn ensure_started(&mut self, out: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        if self.id.is_empty() {
            self.id = synth_id("chatcmpl");
        }
        // 首个 chunk 只带 role，符合 OpenAI 的习惯。
        out.push(self.chunk(json!({ "role": "assistant", "content": "" }), Value::Null));
    }

    fn tool_idx(&mut self, block_index: u32) -> u32 {
        if let Some(i) = self.tool_index.get(&block_index) {
            return *i;
        }
        let i = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_index.insert(block_index, i);
        i
    }
}

impl StreamEncoder for ChatStreamEncoder {
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
                if let Block::ToolUse { id, name, .. } = block {
                    let ti = self.tool_idx(*index);
                    out.push(self.chunk(
                        json!({ "tool_calls": [{
                            "index": ti,
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": "" },
                        }]}),
                        Value::Null,
                    ));
                }
            }

            StreamChunk::TextDelta { text, .. } => {
                self.ensure_started(&mut out);
                if !text.is_empty() {
                    out.push(self.chunk(json!({ "content": text }), Value::Null));
                }
            }

            StreamChunk::ThinkingDelta { text, .. } => {
                self.ensure_started(&mut out);
                if !text.is_empty() {
                    out.push(self.chunk(json!({ "reasoning_content": text }), Value::Null));
                }
            }

            StreamChunk::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                self.ensure_started(&mut out);
                // 上游可能没发过 BlockStart：这里补一个带 id / name 的开头，
                // 否则下游拿不到 tool_call 的 id，也无法把结果关联回调用。
                let ti = match self.tool_index.get(index) {
                    Some(i) => *i,
                    None => {
                        let ti = self.tool_idx(*index);
                        let call_id = if id.is_empty() {
                            format!("call_{ti}")
                        } else {
                            id.clone()
                        };
                        out.push(self.chunk(
                            json!({ "tool_calls": [{
                                "index": ti,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name_delta.clone().unwrap_or_default(),
                                    "arguments": "",
                                },
                            }]}),
                            Value::Null,
                        ));
                        ti
                    }
                };
                if let Some(args) = arguments_delta {
                    if !args.is_empty() {
                        out.push(self.chunk(
                            json!({ "tool_calls": [{
                                "index": ti,
                                "function": { "arguments": args },
                            }]}),
                            Value::Null,
                        ));
                    }
                }
            }

            StreamChunk::BlockStop { .. } => {}

            StreamChunk::Usage { usage } => self.usage.merge_non_zero(usage),

            StreamChunk::Finish { reason } => self.finish_reason = Some(reason.clone()),

            StreamChunk::Error { error } => {
                self.ensure_started(&mut out);
                out.push(sse_frame(
                    None,
                    &json!({ "error": { "type": "api_error", "message": error } }).to_string(),
                ));
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

        let reason = self
            .finish_reason
            .clone()
            .unwrap_or(FinishReason::Stop)
            .to_wire(Protocol::OpenAiChat);
        // 带 finish_reason 的空 delta chunk。
        out.push(self.chunk(json!({}), json!(reason)));

        // 独立的 usage chunk：choices 为空数组，这是 OpenAI 的约定。
        let prompt_tokens = self.usage.total_input_billed();
        out.push(sse_frame(
            None,
            &json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": now_secs(),
                "model": self.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": self.usage.output_tokens,
                    "total_tokens": prompt_tokens.saturating_add(self.usage.output_tokens),
                },
            })
            .to_string(),
        ));

        out.push(sse_frame(None, "[DONE]"));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::http::ClientConfig;
    use crate::protocol::codec::Codec;
    use crate::protocol::oai_chat::OpenAiChatCodec;

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.openai.com", "sk")
    }

    fn roundtrip(body: &str) -> (WireRequest, Value) {
        let ir = decode_request(body.as_bytes()).expect("入站解码");
        let http = OpenAiChatCodec
            .encode_request(&ir, &cfg())
            .expect("出站编码");
        (ir, serde_json::from_slice(&http.body).unwrap())
    }

    #[test]
    fn system_is_lifted_and_user_content_decoded() {
        let (ir, out) = roundtrip(
            r#"{"model":"gpt-4o","messages":[
                {"role":"system","content":"be brief"},
                {"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(ir.system.len(), 1);
        assert_eq!(ir.system[0].as_text(), Some("be brief"));
        assert_eq!(ir.messages.len(), 1, "system 不该留在 messages 里");
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["content"], "hi");
    }

    #[test]
    fn tool_role_message_becomes_tool_result_block() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","messages":[
                {"role":"assistant","tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"f","arguments":"{\"a\":1}"}}]},
                {"role":"tool","tool_call_id":"c1","content":"sunny"}]}"#,
        );
        assert!(matches!(
            &ir.messages[0].content[0],
            Block::ToolUse { id, name, .. } if id == "c1" && name == "f"
        ));
        match &ir.messages[1].content[0] {
            Block::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                assert_eq!(tool_use_id, "c1");
                assert_eq!(content.as_text(), "sunny");
            }
            other => panic!("期望 tool_result，得到 {other:?}"),
        }
    }

    #[test]
    fn data_url_image_splits_into_base64_source() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AAAA"}}]}]}"#,
        );
        match &ir.messages[0].content[0] {
            Block::Image { source, .. } => assert_eq!(
                *source,
                ImageSource::Base64 {
                    media_type: "image/jpeg".into(),
                    data: "AAAA".into()
                }
            ),
            other => panic!("期望 image，得到 {other:?}"),
        }
    }

    #[test]
    fn reasoning_content_becomes_thinking_first() {
        let (ir, _) = roundtrip(
            r#"{"model":"deepseek-reasoner","messages":[{"role":"assistant",
                "reasoning_content":"let me think","content":"answer"}]}"#,
        );
        assert!(matches!(ir.messages[0].content[0], Block::Thinking { .. }));
        assert_eq!(ir.messages[0].content[1].as_text(), Some("answer"));
    }

    /// `stream_options` 落进 `extra`；出站编码强制打开 `include_usage`。
    ///
    /// 注意：出站 chat codec 是**整体覆写** `stream_options`（只留 `include_usage`），
    /// 不是像 apilot 那样在用户设置上合并 —— 这是既有出站行为，本任务不改。
    #[test]
    fn stream_options_land_in_extra_and_include_usage_is_forced() {
        let (ir, out) = roundtrip(
            r#"{"model":"m","messages":[],"stream":true,"stream_options":{"custom":1}}"#,
        );
        assert!(ir.extra.contains_key("stream_options"));
        assert_eq!(out["stream_options"]["include_usage"], true);
    }

    #[test]
    fn encode_response_reports_inclusive_prompt_tokens() {
        let raw = br#"{"id":"c1","model":"gpt-4o","choices":[
            {"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1000,"completion_tokens":5,
                     "prompt_tokens_details":{"cached_tokens":800}}}"#;
        let resp = OpenAiChatCodec.decode_response(Bytes::from_static(raw)).unwrap();
        let out: Value = serde_json::from_slice(&encode_response(&resp).unwrap()).unwrap();
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["choices"][0]["message"]["content"], "hi");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        // prompt_tokens 必须含缓存（OpenAI 口径）。
        assert_eq!(out["usage"]["prompt_tokens"], 1000);
        assert_eq!(out["usage"]["prompt_tokens_details"]["cached_tokens"], 800);
    }

    #[test]
    fn encode_response_with_only_tools_has_null_content() {
        let resp = WireResponse {
            id: Some("c".into()),
            model: Some("m".into()),
            content: vec![Block::ToolUse {
                id: "t1".into(),
                name: "f".into(),
                input: json!({"a": 1}),
                cache: None,
            }],
            finish_reason: FinishReason::ToolCall,
            usage: Usage::default(),
            extra: Default::default(),
        };
        let out: Value = serde_json::from_slice(&encode_response(&resp).unwrap()).unwrap();
        assert_eq!(out["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            r#"{"a":1}"#
        );
    }

    // --- 流式编码 ---

    #[test]
    fn stream_encoder_terminates_with_done() {
        let mut enc = ChatStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(enc.encode(&StreamChunk::text("hi")).unwrap());
        out.extend(
            enc.encode(&StreamChunk::Finish {
                reason: FinishReason::Stop,
            })
            .unwrap(),
        );
        out.extend(enc.finish().unwrap());

        assert!(out[0].contains("\"role\":\"assistant\""));
        assert!(out.iter().any(|f| f.contains("\"content\":\"hi\"")));
        assert!(out.last().unwrap().contains("[DONE]"));
        assert!(out.iter().all(|f| f.starts_with("data: ")));
    }

    #[test]
    fn stream_encoder_numbers_tool_calls_from_zero() {
        let mut enc = ChatStreamEncoder::new();
        let mut out = Vec::new();
        for (i, name) in [(5u32, "f1"), (9u32, "f2")] {
            out.extend(
                enc.encode(&StreamChunk::BlockStart {
                    index: i,
                    block: Block::ToolUse {
                        id: format!("call_{name}"),
                        name: name.into(),
                        input: json!({}),
                        cache: None,
                    },
                })
                .unwrap(),
            );
        }
        let indices: Vec<u64> = out
            .iter()
            .filter_map(|f| serde_json::from_str::<Value>(f.trim_start_matches("data: ").trim_end()).ok())
            .filter_map(|v| {
                v["choices"][0]["delta"]["tool_calls"][0]["index"].as_u64()
            })
            .collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn stream_encoder_synthesizes_tool_header_for_orphan_arguments() {
        let mut enc = ChatStreamEncoder::new();
        let out = enc
            .encode(&StreamChunk::ToolCallDelta {
                index: 0,
                id: String::new(),
                name_delta: None,
                arguments_delta: Some("{}".into()),
            })
            .unwrap();
        let joined = out.join("");
        assert!(joined.contains("\"id\":\"call_0\""), "应合成 tool_call id");
    }
}
