//! OpenAI Responses API 的编解码（`protocol = "openai_responses"`，`POST /v1/responses`）。
//!
//! 与 Chat Completions 的主要差别：
//!
//! * system 叫 `instructions`；
//! * 输入是**扁平的 item 数组**：工具调用与工具结果都是顶层 item，不是消息里的块；
//! * `max_tokens` 叫 `max_output_tokens`；
//! * 工具声明的 `name` / `parameters` 在对象顶层，不套 `function` 外壳；
//! * 流式没有 `[DONE]`，结束靠 `response.completed`。
//!
//! ## 思考内容的往返
//!
//! Responses 的 reasoning item 带一个不透明的 `encrypted_content`（服务端加密后的
//! 思考内容），语义与 Anthropic 的 thinking `signature` 完全一致：都必须在多轮里
//! **原样回传**，否则模型拿不回自己的推理链。因此 IR 用同一个字段承载它
//! （[`Block::Thinking::signature`]）。
//!
//! **已知缺口**：解码时若上游没给 `encrypted_content`，重新编码时会跳过该 reasoning
//! item（只有一个摘要、没有加密内容的 reasoning item 上游不接受）。要让思考内容
//! 完整往返，调用方需要在请求里带上 `include: ["reasoning.encrypted_content"]`
//! —— 通过 [`WireRequest::extra`] 传即可，见本文件末尾的测试。
//!
//! 出站方向的 `decode_request` / `encode_response` / `new_stream_encoder` 尚未实现
//! （网关方向），由 trait 默认实现返回 `Unsupported`。

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::error::LLMError;
use crate::http::{ClientConfig, HttpMethod, HttpRequest};
use crate::protocol::codec::Codec;
use crate::protocol::ir::{
    Block, CacheUsage, FinishReason, ImageSource, Protocol, ProtocolId, ResponseFormat, ToolChoice,
    ToolKind, ToolSpec, Usage, WireMessage, WireRequest, WireResponse,
};
use crate::sse::SseMapper;

mod stream;

pub use stream::ResponsesSseMapper;

#[derive(Debug, Default, Clone)]
pub struct OpenAiResponsesCodec;

impl Codec for OpenAiResponsesCodec {
    fn protocol(&self) -> ProtocolId {
        Protocol::OpenAiResponses.into()
    }

    fn encode_request(
        &self,
        request: &WireRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        let body = build_body(request, config)?;
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
            url: crate::protocol::codec::endpoint_url(
                config,
                Protocol::OpenAiResponses.default_path(),
            ),
            headers,
            body: serde_json::to_vec(&body)?,
        })
    }

    fn decode_response(&self, body: Bytes) -> Result<WireResponse, LLMError> {
        decode_response(&body)
    }

    fn new_stream_decoder(&self) -> Box<dyn SseMapper> {
        Box::new(ResponsesSseMapper::default())
    }
}

pub fn build_body(request: &WireRequest, config: &ClientConfig) -> Result<Value, LLMError> {
    if config.base_url.is_empty() {
        return Err(LLMError::config("OpenAI Responses: base_url is empty"));
    }
    reject_unsupported_response_format(&request.response_format)?;

    // 未建模字段先铺，随后被建模字段覆盖。
    let mut body = Map::new();
    for (k, v) in &request.extra {
        body.insert(k.clone(), v.clone());
    }

    body.insert("model".into(), json!(request.model));

    if !request.system.is_empty() {
        // `instructions` 是一个字符串，多个 system 分段用换行拼。
        let text = request.system_text();
        if !text.is_empty() {
            body.insert("instructions".into(), json!(text));
        }
    }

    body.insert(
        "input".into(),
        Value::Array(messages_to_items(&request.messages)),
    );

    if !request.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(tool_to_wire).collect()),
        );
    }
    if !request.tools.is_empty() {
        body.insert(
            "tool_choice".into(),
            match &request.tool_choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Specific { name } => json!({"type": "function", "name": name}),
            },
        );
    }

    if let Some(m) = request.max_tokens {
        body.insert("max_output_tokens".into(), json!(m));
    }
    if let Some(t) = request.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = request.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if request.stream {
        body.insert("stream".into(), json!(true));
    }
    if let Some(thinking) = &request.thinking {
        // Responses 只用 effort；预算由模型自己定。
        let effort = thinking.effort.clone().unwrap_or_else(|| "medium".into());
        body.insert("reasoning".into(), json!({"effort": effort}));
        // **让思考内容能往返。** reasoning item 的 `encrypted_content` 只有在
        // `include` 里点名时才随响应下发；拿不到它，下一轮就回传不了推理链，
        // 模型等于每轮丢掉自己的思考 —— 直接体现为多轮质量下降与缓存失效。
        //
        // 用 `entry().or_insert`：调用方自己传了 `include`（经 `extra` 透传）就以它为准。
        body.entry("include".to_string())
            .or_insert_with(|| json!(["reasoning.encrypted_content"]));
    }

    Ok(Value::Object(body))
}

/// Responses 用 `text.format` 而不是 `response_format`。两种结构化输出都还没建模，
/// 与其静默丢掉让调用方以为生效了，不如明确报错。
fn reject_unsupported_response_format(fmt: &ResponseFormat) -> Result<(), LLMError> {
    match fmt {
        ResponseFormat::Text => Ok(()),
        other => Err(LLMError::unsupported(format!(
            "openai_responses 渠道尚未支持结构化输出（{other:?}）"
        ))),
    }
}

fn tool_to_wire(t: &ToolSpec) -> Value {
    match &t.kind {
        // 服务端工具（web_search / code_interpreter / ...）用厂商原样的声明对象。
        ToolKind::ServerTool { spec } => spec.clone(),
        // Responses 的 function 工具字段在顶层，不套 `function`。
        // namespace 的组名不参与出站编码：函数名只允许 `[a-zA-Z0-9_-]`。
        ToolKind::Function | ToolKind::Namespace { .. } => {
            let mut v = json!({
                "type": "function",
                "name": t.name,
                "parameters": t.input_schema,
            });
            if let Some(d) = &t.description {
                v["description"] = json!(d);
            }
            v
        }
        // 自由格式工具（如 apply_patch）：Responses 的 `custom` 类型。
        ToolKind::Freeform => {
            let mut v = json!({"type": "custom", "name": t.name});
            if let Some(d) = &t.description {
                v["description"] = json!(d);
            }
            v
        }
    }
}

/// IR 的交替消息 → Responses 的扁平 item 列表。
///
/// **块顺序必须保留**：同一轮里模型可能先思考、再说话、再调工具，
/// 这里按原顺序 flush 出对应的 item，而不是把所有 function_call 提到前面。
fn messages_to_items(messages: &[WireMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            crate::protocol::ir::Role::Assistant => {
                let mut text_run: Vec<Value> = Vec::new();
                for b in &msg.content {
                    match b {
                        Block::Text { text, .. } => {
                            text_run.push(json!({"type": "output_text", "text": text}));
                        }
                        Block::ToolUse {
                            id, name, input, ..
                        } => {
                            flush_text_run(&mut text_run, &mut out);
                            out.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                            }));
                        }
                        Block::Thinking { text, signature } => {
                            flush_text_run(&mut text_run, &mut out);
                            match signature {
                                Some(encrypted) => out.push(json!({
                                    "type": "reasoning",
                                    "summary": [{"type": "summary_text", "text": text}],
                                    "encrypted_content": encrypted,
                                })),
                                None => tracing::debug!(
                                    target: "llm_adapt::oai_responses",
                                    "跳过没有 encrypted_content 的 reasoning item —— \
                                     上游不接受只有摘要的 reasoning item。\
                                     要让思考内容往返，请在请求里带 \
                                     include: [\"reasoning.encrypted_content\"]"
                                ),
                            }
                        }
                        // 未建模块（服务端工具活动、Codex 的 additional_tools 等）
                        // 原样写回。同协议直通时它们本来就无需改动。
                        Block::Unmodeled { raw } => {
                            flush_text_run(&mut text_run, &mut out);
                            out.push(raw.clone());
                        }
                        _ => {}
                    }
                }
                flush_text_run(&mut text_run, &mut out);
            }
            _ => {
                let role_str = if msg.role == crate::protocol::ir::Role::System {
                    "system"
                } else {
                    "user"
                };
                let mut parts: Vec<Value> = Vec::new();
                for b in &msg.content {
                    match b {
                        Block::Text { text, .. } => {
                            parts.push(json!({"type": "input_text", "text": text}));
                        }
                        Block::Image { source, .. } => {
                            // Responses 的 input_image 只要 URL；内联的渲染成 data: URL。
                            parts.push(json!({
                                "type": "input_image",
                                "image_url": image_url(source),
                            }));
                        }
                        Block::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            flush_parts(&mut parts, role_str, &mut out);
                            out.push(json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": content.as_text(),
                            }));
                        }
                        other => {
                            tracing::debug!(
                                target: "llm_adapt::oai_responses",
                                "Responses 的 input 表达不了该块，已跳过：{:?}",
                                other
                            );
                        }
                    }
                }
                flush_parts(&mut parts, role_str, &mut out);
            }
        }
    }
    out
}

fn image_url(source: &ImageSource) -> String {
    source.to_url()
}

fn flush_text_run(run: &mut Vec<Value>, out: &mut Vec<Value>) {
    if run.is_empty() {
        return;
    }
    let content = std::mem::take(run);
    out.push(json!({"type": "message", "role": "assistant", "content": content}));
}

fn flush_parts(parts: &mut Vec<Value>, role: &str, out: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    let content = std::mem::take(parts);
    out.push(json!({"type": "message", "role": role, "content": content}));
}

// ---------------------------------------------------------------------------
// 响应解码
// ---------------------------------------------------------------------------

/// 请求里已被显式建模、编码时会重新写出的顶层字段。
///
/// 不在这里的一律由 [`WireRequest::extra`] 原样透传 —— 尤其有状态语义的
/// `store` / `previous_response_id` / `include`（后者是让思考内容能往返的关键），
/// 刻意不做建模，让它们无损穿过本层。

/// Responses 的 usage → IR 口径。
///
/// 与 Chat 同款陷阱：`input_tokens` **含** `input_tokens_details.cached_tokens`，
/// 必须做减法才是 IR 约定的「未命中缓存的输入」。
pub fn decode_usage(v: Option<&Value>) -> Usage {
    let Some(u) = v else {
        return Usage::default();
    };
    let input_tokens = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
    let output_tokens = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
    let cached = u
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = u
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    Usage {
        input_tokens: input_tokens.saturating_sub(cached) as u32,
        output_tokens: output_tokens as u32,
        thinking_tokens: reasoning as u32,
        cache: CacheUsage {
            read_tokens: cached as u32,
            write: Default::default(),
        },
        ..Default::default()
    }
}

fn finish_from_status(status: Option<&str>, saw_tool_call: bool) -> FinishReason {
    match status {
        Some("incomplete") => FinishReason::Length,
        Some("failed") => FinishReason::Other {
            value: "failed".into(),
        },
        _ if saw_tool_call => FinishReason::ToolCall,
        _ => FinishReason::Stop,
    }
}

/// 一个 `output[]` item → 0..1 个 IR 块。
///
/// 认不出的 item 类型（`web_search_call`、`local_shell_call`、`item_reference`……）
/// 整体进 [`Block::Unmodeled`]。上游每加一种新 item 类型就让整次对话失败，
/// 代价远大于「我们暂时表达不了它」。
fn item_to_block(item: &Value) -> Option<Block> {
    match item.get("type").and_then(Value::as_str).unwrap_or("") {
        "message" => {
            let parts = item.get("content").and_then(Value::as_array)?;
            let text: String = parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .or_else(|| p.get("output_text"))
                        .and_then(Value::as_str)
                })
                .collect();
            if text.is_empty() {
                None
            } else {
                Some(Block::Text { text, cache: None })
            }
        }
        "reasoning" => {
            let summary = item
                .get("summary")
                .and_then(Value::as_array)
                .map(|s| {
                    s.iter()
                        .filter_map(|x| x.get("text").and_then(Value::as_str))
                        .collect::<String>()
                })
                .unwrap_or_default();
            let encrypted = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .map(str::to_string);
            if summary.is_empty() && encrypted.is_none() {
                return None;
            }
            // `encrypted_content` 与 Anthropic 的 thinking signature 是同一个东西：
            // 不透明、必须原样回传。共用 `signature` 字段承载。
            Some(Block::Thinking {
                text: summary,
                signature: encrypted,
            })
        }
        "function_call" => Some(Block::ToolUse {
            id: call_id_of(item),
            name: item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            input: parse_arguments(
                item.get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}"),
            ),
            cache: None,
        }),
        _ => Some(Block::Unmodeled { raw: item.clone() }),
    }
}

/// Responses 用 `call_id` 关联调用与结果，`id` 只是 item 自身的标识。
fn call_id_of(item: &Value) -> String {
    item.get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "llm_adapt::oai_responses",
                "工具入参不是合法 JSON（{e}），已退回空对象"
            );
            json!({})
        }
    }
}

/// 响应里已被显式建模、不该落进 `extra` 的顶层字段。
///
/// 不在这张表里的一律进 `extra` 原样透传 —— 把字段列进来等于承诺「我们已经读懂它」。
const KNOWN_RESPONSE_FIELDS: &[&str] = &["id", "model", "output", "status", "usage", "object"];

fn decode_response(body: &[u8]) -> Result<WireResponse, LLMError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("openai_responses: invalid JSON: {e}"),
        ctx: Default::default(),
    })?;

    let mut content = Vec::new();
    let mut saw_tool_call = false;
    if let Some(output) = v.get("output").and_then(Value::as_array) {
        for item in output {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                saw_tool_call = true;
            }
            if let Some(block) = item_to_block(item) {
                content.push(block);
            }
        }
    }

    let status = v.get("status").and_then(Value::as_str);
    let mut extra = indexmap::IndexMap::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if !KNOWN_RESPONSE_FIELDS.contains(&k.as_str()) {
                extra.insert(k.clone(), val.clone());
            }
        }
    }
    // 让排障时能看到是哪种 status 导致的结束。
    if let Some(s) = status {
        extra.insert("status".into(), json!(s));
    }

    Ok(WireResponse {
        id: v.get("id").and_then(Value::as_str).map(str::to_string),
        model: v.get("model").and_then(Value::as_str).map(str::to_string),
        content,
        finish_reason: finish_from_status(status, saw_tool_call),
        usage: decode_usage(v.get("usage")),
        extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::decode_stream;
    use crate::protocol::ir::{Role, ThinkingConfig};
    use crate::types::{BlockAssembler, ChatRequest, Conversation};

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.openai.com", "sk-test")
    }

    fn encode(req: &ChatRequest) -> Value {
        let ir = crate::types::lift_request(req);
        let http = OpenAiResponsesCodec
            .encode_request(&ir, &cfg())
            .expect("encode_request");
        serde_json::from_slice(&http.body).unwrap()
    }

    fn body(ir: &WireRequest) -> Value {
        let http = OpenAiResponsesCodec.encode_request(ir, &cfg()).unwrap();
        serde_json::from_slice(&http.body).unwrap()
    }

    #[test]
    fn request_hits_the_responses_endpoint() {
        let http = OpenAiResponsesCodec
            .encode_request(
                &crate::types::lift_request(&ChatRequest::new(
                    "gpt-5",
                    "openai_responses",
                    Conversation::single_user("hi"),
                )),
                &cfg(),
            )
            .unwrap();
        assert_eq!(http.url, "https://api.openai.com/v1/responses");
    }

    #[test]
    fn simple_user_message_becomes_a_message_item() {
        let b = encode(&ChatRequest::new(
            "gpt-5",
            "openai_responses",
            Conversation::single_user("hi"),
        ));
        assert_eq!(b["model"], "gpt-5");
        assert_eq!(b["input"][0]["type"], "message");
        assert_eq!(b["input"][0]["role"], "user");
        assert_eq!(b["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(b["input"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn system_becomes_instructions() {
        let conv = Conversation::with_system("be terse", "hi");
        let b = encode(&ChatRequest::new("gpt-5", "openai_responses", conv));
        assert_eq!(b["instructions"], "be terse");
        // 不该同时出现 role:system 的 message
        assert!(!b["input"].to_string().contains("system"));
    }

    #[test]
    fn tools_use_flat_schema_not_nested_function() {
        let mut ir = WireRequest::new("gpt-5");
        ir.tools
            .push(ToolSpec::function("f", "d", json!({"type": "object"})));
        let b = body(&ir);
        assert_eq!(b["tools"][0]["type"], "function");
        assert_eq!(b["tools"][0]["name"], "f");
        assert!(
            b["tools"][0].get("function").is_none(),
            "Responses 不套 function 外壳"
        );
    }

    /// 需求 2：原生服务端工具按厂商原样下发。
    #[test]
    fn server_tools_are_declared_verbatim() {
        let mut ir = WireRequest::new("gpt-5");
        let spec = json!({"type": "web_search"});
        ir.tools.push(ToolSpec {
            name: "web_search".into(),
            description: None,
            input_schema: json!({}),
            cache: None,
            kind: ToolKind::ServerTool { spec: spec.clone() },
        });
        let b = body(&ir);
        assert_eq!(b["tools"][0], spec);
    }

    /// 助手轮的块顺序必须保住：思考 → 文本 → 工具调用。
    #[test]
    fn assistant_block_order_is_preserved() {
        let mut ir = WireRequest::new("gpt-5");
        ir.messages.push(WireMessage::new(
            Role::Assistant,
            vec![
                Block::Thinking {
                    text: "thought".into(),
                    signature: Some("enc_abc".into()),
                },
                Block::text("let me search"),
                Block::ToolUse {
                    id: "call_1".into(),
                    name: "web_search".into(),
                    input: json!({"q": "rust"}),
                    cache: None,
                },
            ],
        ));
        let b = body(&ir);
        let items = b["input"].as_array().unwrap();
        let types: Vec<&str> = items
            .iter()
            .map(|i| i["type"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(types, vec!["reasoning", "message", "function_call"]);
        assert_eq!(items[0]["encrypted_content"], "enc_abc");
        assert_eq!(items[2]["call_id"], "call_1");
        assert_eq!(items[2]["name"], "web_search");
    }

    /// 没有 `encrypted_content` 的 reasoning item 上游不接受，必须跳过而不是发出。
    #[test]
    fn reasoning_without_encrypted_content_is_skipped() {
        let mut ir = WireRequest::new("gpt-5");
        ir.messages.push(WireMessage::new(
            Role::Assistant,
            vec![Block::Thinking {
                text: "only summary".into(),
                signature: None,
            }],
        ));
        let b = body(&ir);
        assert_eq!(b["input"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn tool_result_becomes_function_call_output_item() {
        let mut ir = WireRequest::new("gpt-5");
        ir.messages.push(WireMessage::new(
            Role::User,
            vec![Block::ToolResult {
                tool_use_id: "call_1".into(),
                content: crate::protocol::ir::ToolResultContent::Text {
                    text: "result".into(),
                },
                is_error: false,
                cache: None,
            }],
        ));
        let b = body(&ir);
        let item = &b["input"][0];
        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["output"], "result");
    }

    #[test]
    fn tool_choice_maps_to_responses_shape() {
        let mut ir = WireRequest::new("gpt-5");
        ir.tools
            .push(ToolSpec::function("f", "d", json!({"type": "object"})));
        ir.tool_choice = ToolChoice::Specific { name: "f".into() };
        let b = body(&ir);
        assert_eq!(b["tool_choice"]["type"], "function");
        assert_eq!(b["tool_choice"]["name"], "f");
    }

    /// 未建模的顶层字段（`store` / `previous_response_id` / `include`）必须透传 ——
    /// 尤其 `include`，它是让思考内容能往返的关键。
    #[test]
    fn unmodeled_top_level_fields_pass_through() {
        let mut ir = WireRequest::new("gpt-5");
        ir.extra.insert("store".into(), json!(false));
        ir.extra
            .insert("previous_response_id".into(), json!("resp_1"));
        ir.extra
            .insert("include".into(), json!(["reasoning.encrypted_content"]));
        // extra 不能盖掉建模字段
        ir.extra.insert("model".into(), json!("hijacked"));
        let b = body(&ir);
        assert_eq!(b["store"], false);
        assert_eq!(b["previous_response_id"], "resp_1");
        assert_eq!(b["include"][0], "reasoning.encrypted_content");
        assert_eq!(b["model"], "gpt-5", "建模字段必须赢过 extra");
    }

    /// 开思考就必须自动带上 include，否则拿不到 encrypted_content，
    /// 下一轮回传不了推理链 —— 直接表现为多轮质量下降与缓存失效。
    #[test]
    fn enabling_thinking_requests_encrypted_reasoning() {
        let mut ir = WireRequest::new("gpt-5");
        ir.thinking = Some(ThinkingConfig {
            budget_tokens: None,
            effort: Some("high".into()),
        });
        let b = body(&ir);
        assert_eq!(b["reasoning"]["effort"], "high");
        assert_eq!(b["include"][0], "reasoning.encrypted_content");
    }

    /// 默认 effort 也该带上 include。
    #[test]
    fn default_effort_still_requests_encrypted_reasoning() {
        let mut ir = WireRequest::new("gpt-5");
        ir.thinking = Some(ThinkingConfig::default());
        let b = body(&ir);
        assert_eq!(b["reasoning"]["effort"], "medium");
        assert_eq!(b["include"][0], "reasoning.encrypted_content");
    }

    /// 调用方自己指定 include 时以它为准，不被默认值顶掉。
    #[test]
    fn explicit_include_wins_over_the_default() {
        let mut ir = WireRequest::new("gpt-5");
        ir.thinking = Some(ThinkingConfig::default());
        ir.extra
            .insert("include".into(), json!(["web_search_call.action.sources"]));
        let b = body(&ir);
        assert_eq!(b["include"][0], "web_search_call.action.sources");
    }

    /// 没开思考就不该塞这个字段。
    #[test]
    fn include_is_absent_without_thinking() {
        let ir = WireRequest::new("gpt-5");
        assert!(body(&ir).get("include").is_none());
    }

    #[test]
    fn max_tokens_maps_to_max_output_tokens() {
        let mut ir = WireRequest::new("gpt-5");
        ir.max_tokens = Some(256);
        assert_eq!(body(&ir)["max_output_tokens"], 256);
    }

    #[test]
    fn decodes_message_output() {
        let raw = br#"{"id":"resp_1","model":"gpt-5","status":"completed","output":[
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi there"}]}
        ]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        assert_eq!(resp.concat_text(), "hi there");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn decodes_function_call_output_as_tool_use() {
        let raw = br#"{"id":"r","model":"gpt-5","status":"completed","output":[
            {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1}"}
        ]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::ToolCall);
        match &resp.content[0] {
            Block::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "f");
                assert_eq!(input["a"], 1);
            }
            other => panic!("期望 tool_use，得到 {other:?}"),
        }
    }

    /// reasoning 的 `encrypted_content` 必须映射到 `signature`，才能原样回传。
    #[test]
    fn reasoning_encrypted_content_maps_to_signature() {
        let raw = br#"{"id":"r","model":"gpt-5","status":"completed","output":[
            {"type":"reasoning","summary":[{"type":"summary_text","text":"hmm"}],
             "encrypted_content":"enc_xyz"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"ans"}]}
        ]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        match &resp.content[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "hmm");
                assert_eq!(signature.as_deref(), Some("enc_xyz"));
            }
            other => panic!("期望 thinking，得到 {other:?}"),
        }
    }

    /// 未知 item 类型（服务端工具活动）必须整体存活。
    #[test]
    fn unknown_output_items_survive_as_unmodeled() {
        let raw = br#"{"id":"r","model":"gpt-5","status":"completed","output":[
            {"type":"web_search_call","id":"ws_1","status":"completed"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}
        ]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        assert_eq!(resp.content.len(), 2);
        assert!(matches!(resp.content[0], Block::Unmodeled { .. }));
    }

    #[test]
    fn incomplete_status_maps_to_length() {
        let raw = br#"{"id":"r","model":"gpt-5","status":"incomplete","output":[]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    /// 与 Chat 相同的陷阱：input_tokens 含缓存，必须折算。
    #[test]
    fn cached_tokens_are_subtracted_from_input_tokens() {
        let raw = br#"{"id":"r","model":"gpt-5","status":"completed","output":[],
            "usage":{"input_tokens":1000,"output_tokens":50,"total_tokens":1050,
                     "input_tokens_details":{"cached_tokens":750},
                     "output_tokens_details":{"reasoning_tokens":20}}}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        assert_eq!(resp.usage.input_tokens, 250);
        assert_eq!(resp.usage.cache.read_tokens, 750);
        assert_eq!(resp.usage.thinking_tokens, 20);
        assert_eq!(resp.usage.output_tokens, 50);
    }

    /// 结构化输出尚未建模：明确报错，不静默丢弃。
    #[test]
    fn structured_output_is_rejected_not_silently_dropped() {
        let mut ir = WireRequest::new("gpt-5");
        ir.response_format = ResponseFormat::Json;
        let err = OpenAiResponsesCodec
            .encode_request(&ir, &cfg())
            .unwrap_err();
        assert!(matches!(err, LLMError::Unsupported { .. }), "实际 {err:?}");
    }

    // ------------------------------------------------------------------
    // 流式
    // ------------------------------------------------------------------

    async fn collect(sse: &'static str) -> (Vec<Block>, Option<FinishReason>, Usage) {
        use futures_util::StreamExt;
        let s: crate::http::ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(
            Bytes::from_static(sse.as_bytes()),
        )]));
        let mut out = decode_stream(&OpenAiResponsesCodec, s);
        let mut asm = BlockAssembler::new();
        while let Some(c) = out.next().await {
            asm.push(&c.expect("mapper 不应报错"));
        }
        let blocks = asm.blocks();
        let finish = asm.finish_reason().cloned();
        let usage = asm.usage().clone();
        (blocks, finish, usage)
    }

    #[tokio::test]
    async fn stream_text_then_completion() {
        let sse = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n\n\
                   event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_0\"}}\n\n\
                   event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"he\"}\n\n\
                   event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"llo\"}\n\n\
                   event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_0\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}}\n\n\
                   event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}}\n\n";
        let (blocks, finish, usage) = collect(sse).await;
        assert_eq!(blocks, vec![Block::text("hello")]);
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(usage.input_tokens, 10);
    }

    /// 工具调用：`output_item.added` 给 id/name，arguments 靠分片拼，
    /// `output_item.done` 用完整 item 校准一次。
    #[tokio::test]
    async fn stream_tool_call_assembles_arguments() {
        let sse = "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_0\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"\"}}\n\n\
                   event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\"}\n\n\
                   event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"\\\"rust\\\"}\"}\n\n\
                   event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_0\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"rust\\\"}\"}}\n\n\
                   event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n";
        let (blocks, finish, _) = collect(sse).await;
        match &blocks[0] {
            Block::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "search");
                assert_eq!(input["q"], "rust");
            }
            other => panic!("期望 tool_use，得到 {other:?}"),
        }
        assert_eq!(finish, Some(FinishReason::ToolCall));
    }

    #[tokio::test]
    async fn stream_reasoning_summary_becomes_thinking() {
        let sse = "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_0\",\"summary\":[],\"encrypted_content\":\"enc_1\"}}\n\n\
                   event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"think\"}\n\n\
                   event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n";
        let (blocks, _, _) = collect(sse).await;
        match &blocks[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "think");
                assert_eq!(
                    signature.as_deref(),
                    Some("enc_1"),
                    "encrypted_content 必须保留，否则思考内容无法回传"
                );
            }
            other => panic!("期望 thinking，得到 {other:?}"),
        }
    }

    /// 服务端工具的活动（`web_search_call` item）必须可见，不能被丢掉。
    #[tokio::test]
    async fn stream_unknown_items_survive() {
        let sse = "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"in_progress\"}}\n\n\
                   event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n";
        let (blocks, _, _) = collect(sse).await;
        assert!(
            matches!(blocks[0], Block::Unmodeled { .. }),
            "服务端工具活动应整体存活：{blocks:?}"
        );
    }

    #[tokio::test]
    async fn stream_failed_reports_error() {
        let sse = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"boom\"}}}\n\n";
        let (_, _, _) = collect(sse).await;
        // 走一遍不应 panic；错误通过 BlockAssembler::error() 暴露
        use futures_util::StreamExt;
        let s: crate::http::ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(
            Bytes::from_static(sse.as_bytes()),
        )]));
        let mut out = decode_stream(&OpenAiResponsesCodec, s);
        let mut asm = BlockAssembler::new();
        while let Some(c) = out.next().await {
            asm.push(&c.unwrap());
        }
        assert!(asm.error().is_some());
    }
}
