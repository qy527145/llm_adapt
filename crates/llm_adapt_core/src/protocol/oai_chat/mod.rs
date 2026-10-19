//! OpenAI Chat Completions 协议的编解码（`protocol = "openai_chat"`）。
//!
//! 对 `/v1/chat/completions` 实测，并且适用于任何镜像该 schema 的厂商
//! （Together、Groq、DeepSeek、OpenRouter、Ollama……）。
//!
//! OpenAI 不暴露显式 `cache_control` 标记（它的缓存对结构相同的前缀自动生效），
//! 也不返回带签名的思考内容。因此本 codec 会忽略缓存标记，并在丢弃 `Thinking`
//! 块时告警——不静默。
//!
//! 入站（网关）方向的 `decode_request` / `encode_response` / `new_stream_encoder`
//! 在 [`inbound`] 里实现（`role: "tool"` 消息折回 user 消息里的 `tool_result` 块等）。

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{ErrorContext, LLMError};
use crate::http::{ClientConfig, HttpMethod, HttpRequest};
use crate::protocol::codec::{Codec, StreamEncoder};
use crate::protocol::ir::{
    Block, CacheUsage, CacheWriteUsage, FinishReason, Protocol, ProtocolId, ResponseFormat,
    ToolChoice, ToolResultContent, Usage, WireRequest, WireResponse,
};
use crate::sse::{SseFrame, SseMapper, SseOutcome};
use crate::types::StreamChunk;

mod inbound;

pub use inbound::ChatStreamEncoder;


#[derive(Debug, Default, Clone)]
pub struct OpenAiChatCodec;

impl Codec for OpenAiChatCodec {
    fn protocol(&self) -> ProtocolId {
        Protocol::OpenAiChat.into()
    }

    fn encode_request(
        &self,
        request: &WireRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError> {
        if config.base_url.is_empty() {
            return Err(LLMError::config("OpenAI: base_url is empty"));
        }
        let body = build_body(request)?;
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
            url: crate::protocol::codec::endpoint_url(config, Protocol::OpenAiChat.default_path()),
            headers,
            body: serde_json::to_vec(&body)?,
            correlation: None,
        })
    }

    fn decode_response(&self, body: Bytes) -> Result<WireResponse, LLMError> {
        decode_response(&body)
    }

    fn new_stream_decoder(&self) -> Box<dyn SseMapper> {
        Box::new(OpenAiSseMapper::default())
    }

    fn decode_request(&self, body: &[u8]) -> Result<WireRequest, LLMError> {
        inbound::decode_request(body)
    }

    fn encode_response(&self, response: &WireResponse) -> Result<Vec<u8>, LLMError> {
        inbound::encode_response(response)
    }

    fn new_stream_encoder(&self) -> Option<Box<dyn StreamEncoder>> {
        Some(Box::new(ChatStreamEncoder::new()))
    }
}

pub fn build_body(req: &WireRequest) -> Result<Value, LLMError> {
    reject_server_tools(&req.tools)?;

    let mut body = serde_json::Map::new();
    // 未建模字段先铺，随后被建模字段覆盖。
    for (k, v) in &req.extra {
        body.insert(k.clone(), v.clone());
    }

    body.insert("model".into(), json!(req.model));
    body.insert("messages".into(), json!(messages_to_wire(req)));
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
    if !req.stop.is_empty() {
        body.insert("stop".into(), json!(req.stop));
    }
    if let Some(user) = &req.user_id {
        body.insert("user".into(), json!(user));
    }
    if req.stream {
        // 不带 `include_usage` 的话，流式响应最后一帧就没有 usage。
        //
        // **合并而不是覆盖**：调用方可能通过 `extra` 传了自己的 `stream_options`
        // （比如 `chunk_include_obfuscation`），直接盖掉会静默丢掉那个键。
        let mut so = match body.get("stream_options") {
            Some(Value::Object(o)) => o.clone(),
            _ => serde_json::Map::new(),
        };
        so.insert("include_usage".into(), json!(true));
        body.insert("stream_options".into(), Value::Object(so));
    }

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req.tools.iter().map(tool_to_wire).collect();
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
            body.insert("response_format".into(), json!({"type": "json_object"}));
        }
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
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

/// Chat Completions 里没有服务端工具的对应物。**明确报错**，而不是把它们丢掉
/// 让模型静默失去能力，也不是伪装成一个本地函数让它看起来能用。
///
/// 调用方据此可以给出明确提示，或把该会话切到 anthropic / openai_responses 渠道。
/// 参考：`docs/PROTOCOL_MATRIX.md` 的损耗清单。
fn reject_server_tools(tools: &[crate::protocol::ir::ToolSpec]) -> Result<(), LLMError> {
    let unsupported: Vec<&str> = tools
        .iter()
        .filter(|t| t.is_server_tool())
        .map(|t| t.name.as_str())
        .collect();
    if unsupported.is_empty() {
        return Ok(());
    }
    Err(LLMError::unsupported(format!(
        "openai_chat 渠道不支持服务端工具（{}）；请改用 anthropic 或 openai_responses 渠道",
        unsupported.join(", ")
    )))
}

/// 本地函数工具 → wire。
fn tool_to_wire(t: &crate::protocol::ir::ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description.clone().unwrap_or_default(),
            "parameters": t.input_schema,
        }
    })
}

fn messages_to_wire(req: &WireRequest) -> Vec<Value> {
    let mut out = Vec::new();

    // system 在 Chat 里是 messages 的第一条。多段用换行拼（见 `WireRequest::system_text`）。
    for b in &req.system {
        if let Block::Text { cache, .. } = b {
            if cache.is_some() {
                tracing::debug!(
                    target: "llm_adapt::openai_chat",
                    "忽略 system 上的 CacheMarker（OpenAI 自动缓存）"
                );
            }
        }
    }
    let system_text = req.system_text();
    if !system_text.is_empty() {
        out.push(json!({ "role": "system", "content": system_text }));
    }

    for msg in &req.messages {
        match msg.role {
            crate::protocol::ir::Role::System => {
                // 顶层 system 才是规范位置；出现在 messages 里的按普通 system 发出。
                let text = msg.concat_text();
                if !text.is_empty() {
                    out.push(json!({ "role": "system", "content": text }));
                }
            }
            crate::protocol::ir::Role::User => append_user_turn(&mut out, &msg.content),
            crate::protocol::ir::Role::Assistant => append_assistant_turn(&mut out, &msg.content),
        }
    }
    out
}

/// 一条 IR user 消息在 Chat 里会裂成：
/// * 0..1 条 `{"role":"user", content: [...]}`（文本 + 图片）
/// * 0..n 条 `{"role":"tool", tool_call_id, content}`（每个工具结果一条）
fn append_user_turn(out: &mut Vec<Value>, content: &[Block]) {
    let mut user_blocks: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();

    for block in content {
        match block {
            Block::Text { text, cache } => {
                if cache.is_some() {
                    tracing::debug!(
                        target: "llm_adapt::openai_chat",
                        "忽略 user 文本上的 CacheMarker（自动缓存）"
                    );
                }
                user_blocks.push(json!({"type": "text", "text": text}));
            }
            Block::Image {
                source,
                detail,
                cache,
            } => {
                if cache.is_some() {
                    tracing::debug!(
                        target: "llm_adapt::openai_chat",
                        "忽略 user 图片上的 CacheMarker"
                    );
                }
                let mut img = json!({"url": source.to_url()});
                if let Some(d) = detail {
                    img["detail"] = json!(d);
                }
                user_blocks.push(json!({"type": "image_url", "image_url": img}));
            }
            Block::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                let serialised = tool_result_to_openai_string(content);
                let mut payload = json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": serialised,
                });
                if *is_error {
                    // OpenAI 没有原生的 is_error 标记，用前缀把信号带给模型。
                    payload["content"] = json!(format!("[error] {}", serialised));
                }
                tool_results.push(payload);
            }
            other => {
                // user 消息里出现工具调用/思考块是不合法的，但宁可告警也不静默丢。
                tracing::warn!(
                    target: "llm_adapt::openai_chat",
                    "user 消息里出现无法表达的块，已丢弃：{:?}",
                    other
                );
            }
        }
    }

    if !user_blocks.is_empty() {
        // 只有单段文本时压成裸字符串，兼容性最好。
        let content = if user_blocks.len() == 1 && user_blocks[0]["type"] == json!("text") {
            user_blocks[0]["text"].clone()
        } else {
            json!(user_blocks)
        };
        out.push(json!({"role": "user", "content": content}));
    }
    out.extend(tool_results);
}

fn append_assistant_turn(out: &mut Vec<Value>, content: &[Block]) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut dropped_thinking = 0usize;

    for block in content {
        match block {
            Block::Text { text, .. } => text_parts.push(text),
            Block::Thinking { .. } | Block::RedactedThinking { .. } => dropped_thinking += 1,
            Block::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": input.to_string() },
                }));
            }
            other => {
                tracing::warn!(
                    target: "llm_adapt::openai_chat",
                    "assistant 消息里出现无法表达的块，已丢弃：{:?}",
                    other
                );
            }
        }
    }
    if dropped_thinking > 0 {
        tracing::warn!(
            target: "llm_adapt::openai_chat",
            "丢弃 {dropped_thinking} 个思考块（chat/completions 无法表达）"
        );
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
    let dropped = content.dropped_image_count();
    if dropped > 0 {
        tracing::warn!(
            target: "llm_adapt::openai_chat",
            "从 tool_result 丢弃 {dropped} 个图片块（OpenAI 的 tool 消息只能放文本）"
        );
    }
    content.as_text()
}

// ---------------------------------------------------------------------------
// 响应解码
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct UsageDto {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetailsDto>,
    /// 输出侧的明细。推理模型（DeepSeek、o 系列）在这里报推理 token 数。
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetailsDto>,
}

#[derive(Debug, Default, Deserialize)]
struct PromptDetailsDto {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Debug, Default, Deserialize)]
struct CompletionDetailsDto {
    #[serde(default)]
    reasoning_tokens: u32,
}

fn build_usage(u: &UsageDto) -> Usage {
    // OpenAI 的 `prompt_tokens` **含**缓存命中部分，IR 的口径是不含 —— 这里折算。
    // 忘了减会把缓存部分重复计费。
    let cached = u
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    // 推理 token **含在** `completion_tokens` 里，这里只是把它单独标出来
    // （与 Responses 侧的 `output_tokens_details.reasoning_tokens` 同一口径）。
    let reasoning = u
        .completion_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .unwrap_or(0);
    Usage {
        input_tokens: u.prompt_tokens.saturating_sub(cached),
        output_tokens: u.completion_tokens,
        thinking_tokens: reasoning,
        cache: CacheUsage {
            read_tokens: cached,
            write: CacheWriteUsage::default(),
        },
        ..Default::default()
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ToolCallDto {
    id: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    function: FunctionDto,
}

#[derive(Debug, Deserialize, Serialize)]
struct FunctionDto {
    name: String,
    #[serde(default)]
    arguments: String,
}

fn decode_response(body: &[u8]) -> Result<WireResponse, LLMError> {
    #[derive(Deserialize)]
    struct Dto {
        id: Option<String>,
        model: Option<String>,
        #[serde(default)]
        choices: Vec<Choice>,
        #[serde(default)]
        usage: Option<UsageDto>,
        #[serde(default, flatten)]
        extra: indexmap::IndexMap<String, Value>,
    }
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        message: Message,
        finish_reason: Option<String>,
    }
    #[derive(Default, Deserialize)]
    struct Message {
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ToolCallDto>,
    }

    let dto: Dto = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("openai: invalid JSON: {e}"),
        ctx: ErrorContext::default(),
    })?;

    let choice = dto
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LLMError::Parse {
            message: "openai: response had no choices".into(),
            ctx: ErrorContext::default(),
        })?;

    let mut content = Vec::new();
    if let Some(text) = choice.message.content.filter(|s| !s.is_empty()) {
        content.push(Block::Text { text, cache: None });
    }
    for tc in choice.message.tool_calls {
        content.push(Block::ToolUse {
            id: tc.id,
            name: tc.function.name,
            input: parse_arguments(&tc.function.arguments),
            cache: None,
        });
    }

    let mut extra = dto.extra;
    extra.shift_remove("choices");
    extra.shift_remove("usage");
    extra.shift_remove("object");
    extra.shift_remove("created");

    Ok(WireResponse {
        id: dto.id,
        model: dto.model,
        content,
        finish_reason: FinishReason::parse_opt(choice.finish_reason.as_deref()),
        usage: dto.usage.as_ref().map(build_usage).unwrap_or_default(),
        extra,
    })
}

/// 上游偶尔会在截断时给出不合法的 JSON 入参。解析失败就退回空对象并告警 ——
/// 让整次对话因为一个工具入参崩掉，代价远大于收益。
fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "llm_adapt::openai_chat",
                "工具入参不是合法 JSON（{e}），已退回空对象。原始内容：{}",
                truncate(raw, 200)
            );
            json!({})
        }
    }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ---------------------------------------------------------------------------
// 流式解码
// ---------------------------------------------------------------------------

/// OpenAI 的 SSE 映射。
///
/// Chat Completions **不标注内容块边界**：文本、思考与工具调用混在同一个扁平 delta 里。
/// 这里按「首次出现」给它们分配块 index，并在 `finish_reason` 到达时补发 `BlockStop`，
/// 让下游能用统一的 [`crate::types::BlockAssembler`] 组装，而不必为这个协议写特例。
///
/// ## `BlockStart` 里的名字与首个 `name_delta`
///
/// 工具名在第一片 `tool_calls` 里给全，所以它既进了 `BlockStart` 的快照，又在同一个
/// 分片里作为 `name_delta` 出现。**这两个只能发一个** —— 下游的累加器（
/// [`crate::types::BlockAssembler`]，以及网关方向的 anthropic 编码器）都是
/// 「`BlockStart` 建槽、`name_delta` 往后追加」，同一份字节发两遍就拼成 `bashbash`。
/// 这里的选择是：快照发（`name_delta: None`），因为只有快照能覆盖「厂商把名字拆成
/// 多片」的情形 —— 那种情况下后续分片照常追加，拼出来仍然是对的。
#[derive(Debug, Default)]
pub struct OpenAiSseMapper {
    started: bool,
    /// 文本块占的 index（首次出现文本时分配，全流最多一个）。
    text_slot: Option<u32>,
    /// 思考块占的 index（与文本块同理，全流最多一个）。
    ///
    /// 多数厂商把思考放在正文之前一次给完，但**协议并不保证顺序**：DeepSeek 的
    /// 逐字推理流里 `reasoning_content` 与 `content` 可能交替出现。与文本块一样，
    /// 全流只占一个 index，交错的部分由 `BlockAssembler` 按 index 归并回同一个块。
    thinking_slot: Option<u32>,
    /// OpenAI 的 `tool_call.index` → 我们分配的块 index。
    tool_slots: std::collections::HashMap<u32, u32>,
    /// 收尾原因是不是已经报过了。
    ///
    /// `[DONE]` 表示**传输**结束，不是「模型为什么停下」—— 它只能合成一个 `stop`，
    /// 而那个合成值会盖掉刚收到没多久的真实原因（`tool_calls`、`length` 全都变成
    /// `stop`）。所以报过就不再报。
    finish_reported: bool,
    next: u32,
}

impl OpenAiSseMapper {
    fn alloc(&mut self) -> u32 {
        let i = self.next;
        self.next += 1;
        i
    }

    fn text_block(&mut self, out: &mut Vec<StreamChunk>) -> u32 {
        if let Some(i) = self.text_slot {
            return i;
        }
        let i = self.alloc();
        self.text_slot = Some(i);
        out.push(StreamChunk::BlockStart {
            index: i,
            block: Block::text(""),
        });
        i
    }

    fn thinking_block(&mut self, out: &mut Vec<StreamChunk>) -> u32 {
        if let Some(i) = self.thinking_slot {
            return i;
        }
        let i = self.alloc();
        self.thinking_slot = Some(i);
        out.push(StreamChunk::BlockStart {
            index: i,
            block: Block::Thinking {
                text: String::new(),
                // Chat Completions 没有思考签名的位置，这里恒定为空。
                signature: None,
            },
        });
        i
    }

    /// 建工具块（已建过就返回原 index）。返回值第二项 = **这一次是不是它建的**，
    /// 调用方据此决定首个 `name_delta` 要不要发（见类型文档）。
    fn tool_block(
        &mut self,
        tc_index: u32,
        id: &str,
        name: Option<&str>,
        out: &mut Vec<StreamChunk>,
    ) -> (u32, bool) {
        if let Some(i) = self.tool_slots.get(&tc_index) {
            return (*i, false);
        }
        let i = self.alloc();
        self.tool_slots.insert(tc_index, i);
        out.push(StreamChunk::BlockStart {
            index: i,
            block: Block::ToolUse {
                id: id.to_string(),
                name: name.unwrap_or_default().to_string(),
                input: json!({}),
                cache: None,
            },
        });
        (i, true)
    }

    /// 给所有还开着的块补 `BlockStop`。重复调用是安全的。
    fn close_all(&mut self, out: &mut Vec<StreamChunk>) {
        if let Some(i) = self.thinking_slot.take() {
            out.push(StreamChunk::BlockStop { index: i });
        }
        if let Some(i) = self.text_slot.take() {
            out.push(StreamChunk::BlockStop { index: i });
        }
        let mut slots: Vec<u32> = self.tool_slots.drain().map(|(_, v)| v).collect();
        slots.sort_unstable();
        for i in slots {
            out.push(StreamChunk::BlockStop { index: i });
        }
    }

    fn map_data(&mut self, data: &str) -> Result<Vec<StreamChunk>, LLMError> {
        let dto: StreamDto = serde_json::from_str(data).map_err(|e| LLMError::Parse {
            message: format!("openai stream: invalid JSON: {e}"),
            ctx: ErrorContext::default(),
        })?;
        let mut out = Vec::new();
        if !self.started {
            if let Some(id) = &dto.id {
                self.started = true;
                out.push(StreamChunk::MessageStart {
                    id: id.clone(),
                    model: dto.model.clone().unwrap_or_default(),
                });
            }
        }
        for choice in dto.choices {
            // 思考先于正文：厂商逐字吐推理内容时两者交替出现，但**先看思考**更接近
            // 「模型先想再答」的实况。
            if let Some(text) = choice.delta.reasoning_content.filter(|s| !s.is_empty()) {
                let index = self.thinking_block(&mut out);
                out.push(StreamChunk::ThinkingDelta {
                    index,
                    text,
                    signature_delta: None,
                });
            }
            if let Some(text) = choice.delta.content.filter(|s| !s.is_empty()) {
                let index = self.text_block(&mut out);
                out.push(StreamChunk::TextDelta { index, text });
            }
            for tc in choice.delta.tool_calls {
                let tc_index = tc.index.unwrap_or(0);
                let (name_delta, arguments_delta) = match tc.function {
                    Some(f) => (f.name, f.arguments),
                    None => (None, None),
                };
                // 首片带真实 id，后续分片只有 index —— 用稳定的合成 id 补上，
                // 让下游能把同一工具调用的分片归并起来。
                let id = tc.id.unwrap_or_else(|| format!("tc_idx_{tc_index}"));
                let (index, opened) =
                    self.tool_block(tc_index, &id, name_delta.as_deref(), &mut out);
                out.push(StreamChunk::ToolCallDelta {
                    index,
                    id,
                    // 首片的名字已经随 `BlockStart` 发过了，见类型文档。
                    name_delta: if opened { None } else { name_delta },
                    arguments_delta,
                });
            }
            if let Some(reason) = choice.finish_reason {
                self.close_all(&mut out);
                self.finish_reported = true;
                out.push(StreamChunk::Finish {
                    reason: FinishReason::parse(&reason),
                });
            }
        }
        if let Some(u) = dto.usage {
            out.push(StreamChunk::Usage {
                usage: build_usage(&u),
            });
        }
        Ok(out)
    }
}

impl SseMapper for OpenAiSseMapper {
    fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError> {
        if frame.data.trim() == "[DONE]" {
            let mut out = Vec::new();
            self.close_all(&mut out);
            // 已经报过真实原因就不要用合成的 `stop` 盖掉它，见字段上的说明。
            if !self.finish_reported {
                self.finish_reported = true;
                out.push(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                });
            }
            return Ok(SseOutcome {
                chunks: out,
                terminal: true,
            });
        }
        Ok(SseOutcome::chunks(self.map_data(&frame.data)?))
    }
}

#[derive(Debug, Deserialize)]
struct StreamDto {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<UsageDto>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: DeltaDto,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DeltaDto {
    #[serde(default)]
    content: Option<String>,
    /// 思考增量。
    ///
    /// OpenAI 官方端点没有这个字段，但**镜像 chat completions schema 的厂商普遍用它**
    /// 承载推理模型的思考：DeepSeek 与 Kimi 叫 `reasoning_content`，GLM 也叫这个。
    /// 不认它就等于把这些模型的思考整段丢掉（而且在 JSON 层就丢了，下游无从补救）。
    ///
    /// 注：OpenRouter 另有一个 `reasoning` 字段，语义相同但拼法不同；等真接上那个
    /// 渠道时再加别名 —— 现在不猜。
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCallDto>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCallDto {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionDto>,
}

#[derive(Debug, Default, Deserialize)]
struct StreamFunctionDto {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::decode_stream;
    use crate::protocol::ir::{Role, ToolKind, ToolSpec, WireMessage};
    use crate::types::{
        BlockAssembler, ChatRequest, Conversation, ToolResultBlock, ToolResultContent, Turn,
        UserBlock, UserMessage,
    };

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.openai.com", "sk-test")
    }

    fn encode(req: &ChatRequest) -> Value {
        let ir = crate::types::lift_request(req);
        let http = OpenAiChatCodec
            .encode_request(&ir, &cfg())
            .expect("encode_request");
        serde_json::from_slice(&http.body).unwrap()
    }

    fn build(req: &ChatRequest) -> HttpRequest {
        let ir = crate::types::lift_request(req);
        OpenAiChatCodec.encode_request(&ir, &cfg()).unwrap()
    }

    /// 调用方自己传的 `stream_options` 键不能被我们盖掉。
    #[test]
    fn stream_options_are_merged_not_overwritten() {
        let mut ir = WireRequest::new("gpt-4o");
        ir.stream = true;
        ir.extra.insert(
            "stream_options".into(),
            json!({"chunk_include_obfuscation": true}),
        );
        let b = build_body(&ir).unwrap();
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert_eq!(
            b["stream_options"]["chunk_include_obfuscation"], true,
            "调用方设的键必须保住"
        );
    }

    #[test]
    fn builds_simple_request() {
        let mut req = ChatRequest::openai("gpt-4o", Conversation::single_user("hi"));
        req.temperature = Some(0.5);
        let http = build(&req);
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
            content: ToolResultContent::Blocks {
                blocks: vec![
                    ToolResultBlock::Text {
                        text: "Tokyo: 22C".into(),
                    },
                    ToolResultBlock::Image {
                        url: "https://example.com/x.png".into(),
                        detail: None,
                    },
                ],
            },
            is_error: false,
            cache: None,
        });
        conv.push(Turn::User(user));

        let body = encode(&ChatRequest::openai("gpt-4o", conv));
        let messages = body["messages"].as_array().unwrap();
        let tool_msg = messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["tool_call_id"], "t1");
        assert_eq!(tool_msg["content"], "Tokyo: 22C"); // 图片被丢弃
    }

    #[test]
    fn assistant_thinking_dropped_silently() {
        let mut conv = Conversation::single_user("continue");
        let mut assistant = crate::types::AssistantMessage::default();
        assistant.push(crate::types::AssistantBlock::Thinking {
            text: "secret".into(),
            signature: None,
        });
        assistant.push(crate::types::AssistantBlock::Text {
            text: "ok".into(),
            cache: None,
        });
        conv.push(Turn::Assistant(assistant));
        conv.push(Turn::User(UserMessage::text("go on")));

        let body = encode(&ChatRequest::openai("gpt-4o", conv));
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
        let resp = OpenAiChatCodec
            .decode_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.concat_text(), "hello!");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        // 3 prompt - 1 cached = 2 未缓存输入
        assert_eq!(resp.usage.input_tokens, 2);
        assert_eq!(resp.usage.cache.read_tokens, 1);
        assert_eq!(resp.usage.output_tokens, 2);
    }

    #[test]
    fn system_fingerprint_lands_in_extra() {
        let body = br#"{"id":"c1","choices":[{"message":{"role":"assistant","content":"x"}}],
                       "system_fingerprint":"fp_abc"}"#;
        let resp = OpenAiChatCodec
            .decode_response(Bytes::from_static(body))
            .unwrap();
        assert_eq!(resp.extra.get("system_fingerprint"), Some(&json!("fp_abc")));
    }

    /// 服务端工具在 chat/completions 里没有对应物：必须明确报错。
    #[test]
    fn server_tools_are_rejected_not_silently_dropped() {
        let mut ir = WireRequest::new("gpt-4o");
        ir.tools.push(ToolSpec {
            name: "web_search".into(),
            description: None,
            input_schema: json!({}),
            cache: None,
            kind: ToolKind::ServerTool {
                spec: json!({"type": "web_search"}),
            },
        });
        let err = OpenAiChatCodec.encode_request(&ir, &cfg()).unwrap_err();
        assert!(matches!(err, LLMError::Unsupported { .. }), "实际 {err:?}");
        assert!(err.to_string().contains("web_search"));
    }

    #[test]
    fn assistant_tool_use_becomes_tool_calls() {
        let mut req = WireRequest::new("gpt-4o");
        req.messages.push(WireMessage::new(
            Role::Assistant,
            vec![Block::ToolUse {
                id: "t1".into(),
                name: "f".into(),
                input: json!({"a":1}),
                cache: None,
            }],
        ));
        let body = build_body(&req).unwrap();
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], Value::Null);
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(msg["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
    }

    #[tokio::test]
    async fn parses_sse_stream() {
        use futures_util::StreamExt;
        let lines = b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(lines))]));
        let mut out = decode_stream(&OpenAiChatCodec, s);
        let mut collected = String::new();
        while let Some(c) = out.next().await {
            match c.unwrap() {
                StreamChunk::TextDelta { text, .. } => collected.push_str(&text),
                StreamChunk::Finish { .. } => break,
                _ => {}
            }
        }
        assert_eq!(collected, "hello");
    }

    /// usage 帧在 finish_reason 之后到达，不能被提前截断。
    #[tokio::test]
    async fn usage_after_finish_is_not_lost() {
        use futures_util::StreamExt;
        let lines = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(lines))]));
        let mut out = decode_stream(&OpenAiChatCodec, s);
        let mut usage = None;
        while let Some(c) = out.next().await {
            if let StreamChunk::Usage { usage: u } = c.unwrap() {
                usage = Some(u);
            }
        }
        let usage = usage.expect("usage 帧必须送达");
        assert_eq!(usage.input_tokens, 9);
        assert_eq!(usage.output_tokens, 2);
    }

    /// 把一段 SSE 原文喂进 mapper，再用 `BlockAssembler` 装回内容块。
    /// 这是**真下游**的装配方式（carter 用的就是它），所以回归测试必须走这条线，
    /// 只断言 mapper 吐了哪些 chunk 不够 —— 名字拼两遍的 bug 正是死在装配那一步。
    fn assemble(sse: &[u8]) -> Vec<Block> {
        assemble_with_finish(sse).0
    }

    /// 同 [`assemble`]，另外把装配器记着的收尾原因带出来。
    fn assemble_with_finish(sse: &[u8]) -> (Vec<Block>, Option<FinishReason>) {
        let codec = OpenAiChatCodec;
        let mut mapper = codec.new_stream_decoder();
        let mut asm = BlockAssembler::new();
        for frame in split_frames(sse) {
            let outcome = mapper.map(frame).expect("这一帧应当能解析");
            for c in &outcome.chunks {
                asm.push(c);
            }
            if outcome.terminal {
                break;
            }
        }
        let finish = asm.finish_reason().cloned();
        (asm.into_blocks(), finish)
    }

    /// 按空行切 SSE 帧（与 `sse::find_boundary` 同一判据，够测试用）。
    fn split_frames(sse: &[u8]) -> Vec<SseFrame> {
        String::from_utf8_lossy(sse)
            .split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .map(|block| {
                let mut event = None;
                let mut data: Vec<&str> = Vec::new();
                for line in block.split('\n') {
                    if let Some(rest) = line.strip_prefix("event:") {
                        event = Some(rest.trim().to_string());
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        data.push(rest.trim_start());
                    }
                }
                SseFrame {
                    event,
                    data: data.join("\n"),
                }
            })
            .collect()
    }

    /// 工具名只能出现一次。
    ///
    /// 上游把名字同时放进首个 `tool_calls` 分片与块首快照里，两个都往下发的话，
    /// 「快照建槽 + 分片追加」的装配会把名字拼成 `bashbash` —— 于是模型收到
    /// 「unknown tool: bashbash」，反复重试同一个调用（DeepSeek 上实测如此）。
    #[test]
    fn tool_name_is_not_doubled_by_the_assembler() {
        let sse = b"data: {\"id\":\"c1\",\"model\":\"deepseek-flash\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
        let blocks = assemble(sse);
        match &blocks[0] {
            Block::ToolUse { name, input, id, .. } => {
                assert_eq!(name, "bash", "工具名不能被拼两遍");
                assert_eq!(input["command"], "ls", "入参分片照常拼接");
                assert_eq!(id, "call_1");
            }
            other => panic!("期望 ToolUse，得到 {other:?}"),
        }
    }

    /// 推理 token 要单独标出来（含在 completion_tokens 里，不重复计）。
    #[tokio::test]
    async fn reasoning_tokens_are_reported_separately() {
        use futures_util::StreamExt;
        let lines = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":4253,\"completion_tokens\":183,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":119}}}\n\ndata: [DONE]\n\n".to_vec();
        let s: crate::http::ByteStream =
            Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(lines))]));
        let mut out = decode_stream(&OpenAiChatCodec, s);
        let mut usage = None;
        while let Some(c) = out.next().await {
            if let StreamChunk::Usage { usage: u } = c.unwrap() {
                usage = Some(u);
            }
        }
        let usage = usage.expect("usage 帧必须送达");
        assert_eq!(usage.input_tokens, 4253, "没命中缓存时输入不折算");
        assert_eq!(usage.output_tokens, 183, "推理 token 含在输出里，不另加");
        assert_eq!(usage.thinking_tokens, 119);
    }

    /// 收尾原因不能被 `[DONE]` 合成的那一个盖掉。
    ///
    /// `[DONE]` 说的是「传输结束」，不是「模型为什么停下」。它合成的 `stop` 会盖住
    /// 刚收到的 `tool_calls`，于是「模型要调工具」这件事在收尾信息里消失 ——
    /// 界面把「要调用工具」显示成「正常结束」，而模型其实还没答完。
    #[test]
    fn done_does_not_clobber_the_real_finish_reason() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
        let (_, finish) = assemble_with_finish(sse);
        assert_eq!(finish, Some(FinishReason::ToolCall));
    }

    /// 上游压根没给收尾原因时，`[DONE]` 合成的 `stop` 仍要报出来。
    #[test]
    fn done_still_reports_a_synthetic_finish_when_upstream_gave_none() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        let (_, finish) = assemble_with_finish(sse);
        assert_eq!(finish, Some(FinishReason::Stop));
    }

    /// 厂商把名字拆成多片时，后续分片照常追加（快照不吞掉它们）。
    #[test]
    fn tool_name_split_across_chunks_is_joined() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"get_\",\"arguments\":\"\"}}]}}]}\n\ndata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"weather\",\"arguments\":\"{}\"}}]}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
        let blocks = assemble(sse);
        match &blocks[0] {
            Block::ToolUse { name, .. } => assert_eq!(name, "get_weather"),
            other => panic!("期望 ToolUse，得到 {other:?}"),
        }
    }

    /// `reasoning_content` 是思考，不是正文。
    ///
    /// OpenAI 官方没有这个字段，但镜像该 schema 的厂商（DeepSeek / Kimi / GLM）用它
    /// 承载推理模型的思考。认不出来就等于整段丢掉，界面上的「思考」永远不会出现。
    #[test]
    fn reasoning_content_becomes_thinking() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"Let me \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":null,\"reasoning_content\":\"check the dirs\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let blocks = assemble(sse);
        // 思考与正文各占一块，且思考排在前面（模型先想再答）。
        match &blocks[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "Let me check the dirs");
                assert!(signature.is_none(), "chat completions 没有思考签名的位置");
            }
            other => panic!("第一块应当是思考，得到 {other:?}"),
        }
        match &blocks[1] {
            Block::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("第二块应当是正文，得到 {other:?}"),
        }
    }

    /// 空的 `reasoning_content`（首片常常是空串）不该凭空开一个思考块。
    #[test]
    fn empty_reasoning_content_does_not_open_a_block() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        let blocks = assemble(sse);
        assert_eq!(blocks.len(), 1, "只有正文一块：{blocks:?}");
        assert!(matches!(&blocks[0], Block::Text { .. }));
    }
}
