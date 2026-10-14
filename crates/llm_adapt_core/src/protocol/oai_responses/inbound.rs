//! OpenAI Responses 入站（网关）方向。
//!
//! Responses 的请求是**扁平的 item 数组**：消息、工具调用、工具结果、思考都是顶层
//! item，不像 Anthropic / Chat 那样装在消息里。解码时把它们折回 IR 的
//! `system` / `messages`，并把连续同角色的 item 合并成一条消息（Anthropic 要求
//! assistant 的思考块排在消息最前，合并正好满足）。
//!
//! Codex 的 Responses Lite 会把工具藏在 input 里的 `additional_tools` 条目（甚至
//! 里层再套 `namespace`）。不把它们捞出来并进 `tools`，转成 Chat 后上游就是一个工具
//! 都没有，Codex 直接瘫掉 —— 这是踩过的坑。

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::error::LLMError;
use crate::protocol::codec::StreamEncoder;
use crate::protocol::ir::{
    Block, FinishReason, ImageSource, Role, ToolChoice, ToolKind, ToolResultContent,
    ToolSpec, Usage, WireMessage, WireRequest, WireResponse,
};
use crate::protocol::shared::{now_secs, synth_id, sse_named};
use crate::types::StreamChunk;

/// 已建模的顶层字段；其余（`store` / `previous_response_id` / `include` …）进 `extra`
/// 原样透传。有状态语义的字段刻意不建模，让它们无损穿过本层。
const KNOWN_TOP_LEVEL: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "max_output_tokens",
    "temperature",
    "top_p",
    "stream",
    "reasoning",
];

// ---------------------------------------------------------------------------
// 请求解码（wire → IR）
// ---------------------------------------------------------------------------

pub(crate) fn decode_request(body: &[u8]) -> Result<WireRequest, LLMError> {
    let v: Value = serde_json::from_slice(body).map_err(|e| LLMError::Parse {
        message: format!("openai_responses: invalid JSON: {e}"),
        ctx: Default::default(),
    })?;
    let obj = v
        .as_object()
        .ok_or_else(|| LLMError::validation("openai_responses: 请求体不是 JSON 对象"))?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| LLMError::validation("openai_responses: 缺少 model 字段"))?
        .to_string();

    let mut req = WireRequest::new(model);
    req.stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    req.max_tokens = obj
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .map(|m| m as u32);
    req.temperature = obj.get("temperature").and_then(Value::as_f64).map(|f| f as f32);
    req.top_p = obj.get("top_p").and_then(Value::as_f64).map(|f| f as f32);

    if let Some(instr) = obj.get("instructions").and_then(Value::as_str) {
        if !instr.is_empty() {
            req.system = vec![Block::text(instr)];
        }
    }

    if let Some(r) = obj.get("reasoning") {
        if let Some(e) = r.get("effort").and_then(Value::as_str) {
            req.thinking = Some(crate::protocol::ir::ThinkingConfig {
                budget_tokens: None,
                effort: Some(e.to_string()),
            });
        }
    }

    // 顶层 tools 先解，`additional_tools` 条目再往里追加。
    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        for t in tools {
            decode_tools_into(t, None, &mut req.tools);
        }
    }
    if let Some(tc) = obj.get("tool_choice") {
        req.tool_choice = decode_tool_choice(tc);
    }

    match obj.get("input") {
        Some(Value::String(s)) => req.messages.push(WireMessage::user_text(s.clone())),
        Some(Value::Array(items)) => {
            for item in items {
                decode_item(item, &mut req);
            }
        }
        _ => {}
    }

    for (k, val) in obj {
        if !KNOWN_TOP_LEVEL.contains(&k.as_str()) {
            req.extra.insert(k.clone(), val.clone());
        }
    }

    Ok(req)
}

fn decode_item(item: &Value, req: &mut WireRequest) {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => push_merged(
            req,
            Role::Assistant,
            vec![Block::ToolUse {
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
            }],
        ),
        Some("function_call_output") => {
            let text = match item.get("output").cloned().unwrap_or(Value::Null) {
                Value::String(s) => s,
                Value::Null => String::new(),
                other => other.to_string(),
            };
            push_merged(
                req,
                Role::User,
                vec![Block::ToolResult {
                    tool_use_id: item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    content: ToolResultContent::text(text),
                    is_error: false,
                    cache: None,
                }],
            );
        }
        // reasoning 的 `encrypted_content` 与 Anthropic 的 thinking signature 同义，
        // 折回 `Thinking::signature` 就能在跨协议时原样带走。
        Some("reasoning") => {
            let text = summary_text(item);
            let encrypted = item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .map(str::to_string);
            if text.is_empty() && encrypted.is_none() {
                return;
            }
            push_merged(
                req,
                Role::Assistant,
                vec![Block::Thinking {
                    text,
                    signature: encrypted,
                }],
            );
        }
        // Codex 的 Responses Lite：工具 schema 塞在 input 里当成一条会话条目发，
        // 同时把顶层 tools 置空。必须把这些工具捞出来。
        Some("additional_tools") => {
            if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                let mut defs = Vec::new();
                for t in tools {
                    decode_tools_into(t, None, &mut defs);
                }
                for def in defs {
                    // 与顶层重复的工具保留先到的，避免同名 schema 打架。
                    if !req.tools.iter().any(|e| e.name == def.name) {
                        req.tools.push(def);
                    }
                }
            }
        }
        Some("message") | None => {
            let content = decode_message_content(item.get("content").unwrap_or(&Value::Null));
            match item.get("role").and_then(Value::as_str) {
                Some("assistant") => push_merged(req, Role::Assistant, content),
                Some("system") | Some("developer") => req.system.extend(content),
                _ => push_merged(req, Role::User, content),
            }
        }
        // 未建模的条目类型只跳过、不报错：上游每加一种新条目（web_search_call、
        // local_shell_call、item_reference……）就报错会让客户端升级后全线崩掉。
        Some(_) => {}
    }
}

/// 把一段块追加到末尾同角色的消息上，避免产出连续的同角色消息。
fn push_merged(req: &mut WireRequest, role: Role, blocks: Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = req.messages.last_mut() {
        if last.role == role {
            last.content.extend(blocks);
            return;
        }
    }
    req.messages.push(WireMessage::new(role, blocks));
}

fn summary_text(item: &Value) -> String {
    item.get("summary")
        .and_then(Value::as_array)
        .map(|s| {
            s.iter()
                .filter_map(|x| x.get("text").and_then(Value::as_str))
                .collect::<String>()
        })
        .unwrap_or_default()
}

fn decode_message_content(v: &Value) -> Vec<Block> {
    match v {
        Value::Null => Vec::new(),
        Value::String(s) => vec![Block::text(s.clone())],
        Value::Array(parts) => parts
            .iter()
            .map(|p| {
                match p
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("input_text")
                {
                    "input_text" | "output_text" | "text" => {
                        Block::text(p.get("text").and_then(Value::as_str).unwrap_or_default())
                    }
                    "input_image" => Block::Image {
                        source: ImageSource::from_url(
                            p.get("image_url").and_then(Value::as_str).unwrap_or_default(),
                        ),
                        detail: None,
                        cache: None,
                    },
                    // 不认识的 content 整块留着。
                    _ => Block::Unmodeled { raw: p.clone() },
                }
            })
            .collect(),
        other => vec![Block::Unmodeled { raw: other.clone() }],
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
    serde_json::from_str(raw).unwrap_or_else(|_| json!({}))
}

/// 把一个 Responses 工具条目展开成 0..n 个 [`ToolSpec`]。
///
/// `type: "namespace"` 是 1→N 的条目：Codex 把成组工具打成
/// `{"type":"namespace","name":"functions","tools":[…]}`，真正的工具在里层。当成普通
/// 工具解的话，模型手里只剩一个叫 `functions`、schema 是空对象的壳子。这里展开成
/// 成员工具，组名挂在 `ToolKind::Namespace` 上（出站编码不用它，只供监控展示）。
fn decode_tools_into(v: &Value, ns: Option<&str>, out: &mut Vec<ToolSpec>) {
    if v.get("type").and_then(Value::as_str) == Some("namespace") {
        let ns = v.get("name").and_then(Value::as_str).or(ns);
        if let Some(nested) = v.get("tools").and_then(Value::as_array) {
            for t in nested {
                decode_tools_into(t, ns, out);
            }
        }
        return;
    }
    if let Some(def) = decode_tool(v, ns) {
        out.push(def);
    }
}

fn decode_tool(v: &Value, ns: Option<&str>) -> Option<ToolSpec> {
    // Responses 的 function 工具字段在顶层；部分中转会套一层 "function"，两者都认。
    let src = v.get("function").unwrap_or(v);
    let name = src.get("name").and_then(Value::as_str)?.to_string();
    let description = src
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_string);
    let input_schema = src
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));

    let kind = if let Some(group) = ns {
        // 成员工具：组名带上，出站时只用裸 name。
        ToolKind::Namespace {
            name: group.to_string(),
        }
    } else {
        match v.get("type").and_then(Value::as_str) {
            // 自由格式工具（apply_patch / code mode 的 exec）：回的不是 JSON 而是原文。
            Some("custom") => ToolKind::Freeform,
            Some("function") | None => ToolKind::Function,
            // 其余（web_search / code_interpreter 之类的服务端工具）：声明原样保留。
            Some(_) => ToolKind::ServerTool { spec: v.clone() },
        }
    };

    Some(ToolSpec {
        name,
        description,
        input_schema,
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
    let mut output: Vec<Value> = Vec::new();
    let mut text_run: Vec<Value> = Vec::new();

    for b in &resp.content {
        match b {
            Block::Text { text, .. } => {
                text_run.push(json!({ "type": "output_text", "text": text, "annotations": [] }));
            }
            Block::Thinking { text, signature } => {
                flush_text_run(&mut text_run, &mut output);
                // `encrypted_content` 必须存在（null 也算）：Codex 的 ResponseItem
                // 上这个字段没有 serde default，缺了整个 item 反序列化失败。
                output.push(json!({
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": text }],
                    "encrypted_content": signature,
                }));
            }
            Block::ToolUse {
                id, name, input, ..
            } => {
                flush_text_run(&mut text_run, &mut output);
                output.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    // Responses 的 arguments 是字符串，不是对象。
                    "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                }));
            }
            // 未建模的块原样写回。
            Block::Unmodeled { raw } => {
                flush_text_run(&mut text_run, &mut output);
                output.push(raw.clone());
            }
            _ => {}
        }
    }
    flush_text_run(&mut text_run, &mut output);

    let status = match resp.finish_reason {
        FinishReason::Length => "incomplete",
        _ => "completed",
    };

    let prompt_total = resp.usage.total_input_billed();
    let body = json!({
        "id": resp.id.clone().unwrap_or_else(|| synth_id("resp")),
        "object": "response",
        "created_at": now_secs(),
        "model": resp.model.clone().unwrap_or_default(),
        "status": status,
        "output": output,
        "parallel_tool_calls": true,
        "usage": {
            "input_tokens": prompt_total,
            "output_tokens": resp.usage.output_tokens,
            "total_tokens": prompt_total.saturating_add(resp.usage.output_tokens),
            "input_tokens_details": { "cached_tokens": resp.usage.cache.read_tokens },
            "output_tokens_details": { "reasoning_tokens": resp.usage.thinking_tokens },
        },
    });
    Ok(serde_json::to_vec(&body)?)
}

fn flush_text_run(run: &mut Vec<Value>, out: &mut Vec<Value>) {
    if run.is_empty() {
        return;
    }
    let content = std::mem::take(run);
    out.push(json!({
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": content,
    }));
}

// ---------------------------------------------------------------------------
// 流式编码（IR delta → SSE 帧）
// ---------------------------------------------------------------------------

/// 一个 output item 的种类与已攒内容。
///
/// **必须攒着**：Codex 只从 `response.output_item.done` 的完整 item 里取工具调用，
/// delta 事件纯粹用于界面回显（见 PROTOCOL_MATRIX 的硬契约一节）。
#[derive(Clone)]
enum ItemKind {
    Message { text: String },
    Reasoning { text: String, encrypted: Option<String> },
    /// `(call_id, name, arguments)`
    Tool(String, String, String),
}

/// 把 [`StreamChunk`] 编成客户端可读的 Responses SSE 序列。
///
/// 三条硬契约（发错客户端静默失效）：
/// 1. `output_item.done` 必须带**完整** item，工具调用的 `arguments` 是拼全的字符串；
/// 2. reasoning 的 `output_item.added` / `done` 必须带 `encrypted_content`（null 也算），
///    且 `reasoning_summary_text.delta` 必带 `summary_index`；
/// 3. 收尾在 `finish()` 里按 output_index 补齐没发过 done 的 item —— Chat 那边根本不产
///    `BlockStop`，跨协议进来时只能靠这里兜底，否则 Codex 收不到工具调用。
#[derive(Default)]
pub struct ResponsesStreamEncoder {
    id: String,
    model: String,
    started: bool,
    finished: bool,
    /// IR block index → Responses output_index。
    output_index: HashMap<u32, u64>,
    next_output_index: u64,
    /// output_index → 已发送 `output_item.added` 且攒着内容。
    items: HashMap<u64, ItemKind>,
    /// 已发过 done 的 output_index，避免收尾时重复发。
    item_done: HashMap<u64, bool>,
    /// IR block index → (call_id, name)，缺 BlockStart 时兜底要用。
    tool_meta: HashMap<u32, (String, String)>,
    usage: Usage,
    finish_reason: Option<FinishReason>,
}

impl ResponsesStreamEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn ev(&self, name: &str, mut payload: Value) -> String {
        payload["type"] = json!(name);
        sse_named(name, &payload.to_string())
    }

    fn ensure_started(&mut self, out: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        if self.id.is_empty() {
            self.id = synth_id("resp");
        }
        out.push(self.ev(
            "response.created",
            json!({
                "response": {
                    "id": self.id,
                    "object": "response",
                    "model": self.model,
                    "status": "in_progress",
                    "output": [],
                }
            }),
        ));
    }

    fn oi(&mut self, block_index: u32) -> u64 {
        if let Some(i) = self.output_index.get(&block_index) {
            return *i;
        }
        let i = self.next_output_index;
        self.next_output_index += 1;
        self.output_index.insert(block_index, i);
        i
    }

    /// 把一个 item 还原成 Responses 的 item 对象。字段形状对齐 Codex 的 `ResponseItem`。
    fn item_json(oi: u64, kind: &ItemKind) -> Value {
        match kind {
            ItemKind::Message { text } => json!({
                "type": "message",
                "id": format!("msg_{oi}"),
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": text }],
            }),
            ItemKind::Reasoning { text, encrypted } => json!({
                "type": "reasoning",
                "id": format!("rs_{oi}"),
                "summary": [{ "type": "summary_text", "text": text }],
                // 缺了这个字段 Codex 整个 item 解不出来 —— 没有加密内容时给 null。
                "encrypted_content": encrypted,
            }),
            ItemKind::Tool(call_id, name, args) => json!({
                "type": "function_call",
                "id": format!("fc_{oi}"),
                "call_id": call_id,
                "name": name,
                // 空参数给 "{}"：Codex 会把 arguments 当 JSON 解析，空串直接报错。
                "arguments": if args.is_empty() { "{}" } else { args.as_str() },
                "status": "completed",
            }),
        }
    }

    fn emit_done(&mut self, oi: u64, out: &mut Vec<String>) {
        if self.item_done.contains_key(&oi) {
            return;
        }
        let Some(kind) = self.items.get(&oi).cloned() else {
            return;
        };
        self.item_done.insert(oi, true);
        let item = Self::item_json(oi, &kind);
        out.push(self.ev(
            "response.output_item.done",
            json!({ "output_index": oi, "item": item }),
        ));
    }

    fn ensure_message_item(&mut self, oi: u64, out: &mut Vec<String>) {
        if self.items.contains_key(&oi) {
            return;
        }
        self.items.insert(oi, ItemKind::Message { text: String::new() });
        out.push(self.ev(
            "response.output_item.added",
            json!({
                "output_index": oi,
                "item": {
                    "type": "message",
                    "id": format!("msg_{oi}"),
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            }),
        ));
    }

    fn ensure_reasoning_item(&mut self, oi: u64, encrypted: Option<String>, out: &mut Vec<String>) {
        if self.items.contains_key(&oi) {
            return;
        }
        self.items.insert(
            oi,
            ItemKind::Reasoning {
                text: String::new(),
                encrypted: encrypted.clone(),
            },
        );
        out.push(self.ev(
            "response.output_item.added",
            json!({
                "output_index": oi,
                "item": {
                    "type": "reasoning",
                    "id": format!("rs_{oi}"),
                    "summary": [],
                    "encrypted_content": encrypted,
                }
            }),
        ));
    }

    fn ensure_tool_item(&mut self, block_index: u32, oi: u64, out: &mut Vec<String>) {
        if self.items.contains_key(&oi) {
            return;
        }
        let (id, name) = self
            .tool_meta
            .get(&block_index)
            .cloned()
            .unwrap_or_else(|| (format!("call_{oi}"), String::new()));
        self.items
            .insert(oi, ItemKind::Tool(id.clone(), name.clone(), String::new()));
        out.push(self.ev(
            "response.output_item.added",
            json!({
                "output_index": oi,
                "item": {
                    "type": "function_call",
                    "id": format!("fc_{oi}"),
                    "call_id": id,
                    "name": name,
                    "arguments": "",
                }
            }),
        ));
    }
}

impl StreamEncoder for ResponsesStreamEncoder {
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
                match block {
                    Block::ToolUse { id, name, .. } => {
                        let oi = self.oi(*index);
                        self.tool_meta.insert(*index, (id.clone(), name.clone()));
                        self.ensure_tool_item(*index, oi, &mut out);
                    }
                    // 思考块若在 BlockStart 就带了加密内容（Responses 原生形状），先建 item
                    // 把它存下，否则收尾的 done 里 encrypted_content 会是 null。
                    Block::Thinking { signature, .. } => {
                        let oi = self.oi(*index);
                        self.ensure_reasoning_item(oi, signature.clone(), &mut out);
                    }
                    _ => {}
                }
            }

            StreamChunk::TextDelta { index, text } => {
                self.ensure_started(&mut out);
                let oi = self.oi(*index);
                self.ensure_message_item(oi, &mut out);
                if let Some(ItemKind::Message { text: acc }) = self.items.get_mut(&oi) {
                    acc.push_str(text);
                }
                if !text.is_empty() {
                    out.push(self.ev(
                        "response.output_text.delta",
                        json!({ "output_index": oi, "delta": text }),
                    ));
                }
            }

            StreamChunk::ThinkingDelta { index, text, .. } => {
                self.ensure_started(&mut out);
                let oi = self.oi(*index);
                self.ensure_reasoning_item(oi, None, &mut out);
                if let Some(ItemKind::Reasoning { text: acc, .. }) = self.items.get_mut(&oi) {
                    acc.push_str(text);
                }
                if !text.is_empty() {
                    out.push(self.ev(
                        "response.reasoning_summary_text.delta",
                        // summary_index 是必填：缺了 Codex 直接忽略这个事件。
                        json!({ "output_index": oi, "summary_index": 0, "delta": text }),
                    ));
                }
            }

            StreamChunk::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                self.ensure_started(&mut out);
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
                }
                let oi = self.oi(*index);
                self.ensure_tool_item(*index, oi, &mut out);
                if let Some(ItemKind::Tool(call_id, name, args)) = self.items.get_mut(&oi) {
                    // 补开头时可能用了占位 call_id / 空 name，这里以真实值校准。
                    if let Some((mid, mname)) = self.tool_meta.get(index) {
                        if !mid.is_empty() {
                            *call_id = mid.clone();
                        }
                        if !mname.is_empty() {
                            *name = mname.clone();
                        }
                    }
                    if let Some(a) = arguments_delta {
                        args.push_str(a);
                    }
                }
                if let Some(a) = arguments_delta {
                    if !a.is_empty() {
                        out.push(self.ev(
                            "response.function_call_arguments.delta",
                            json!({
                                "output_index": oi,
                                "item_id": format!("fc_{oi}"),
                                "delta": a,
                            }),
                        ));
                    }
                }
            }

            StreamChunk::BlockStop { index } => {
                if let Some(oi) = self.output_index.get(index).copied() {
                    self.emit_done(oi, &mut out);
                }
            }

            StreamChunk::Usage { usage } => self.usage.merge_non_zero(usage),

            StreamChunk::Finish { reason } => self.finish_reason = Some(reason.clone()),

            StreamChunk::Error { error } => {
                self.ensure_started(&mut out);
                out.push(self.ev(
                    "error",
                    json!({ "type": "api_error", "message": error }),
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

        // 收尾补 done：Chat 上游的解码器不产 BlockStop，不在这里兜一把 Codex 永远
        // 收不到完整 item（表现是「文字出来了、工具不执行」）。按 output_index 升序。
        let mut pending: Vec<u64> = self
            .items
            .keys()
            .copied()
            .filter(|oi| !self.item_done.contains_key(oi))
            .collect();
        pending.sort_unstable();
        for oi in pending {
            self.emit_done(oi, &mut out);
        }

        let status = match self.finish_reason {
            Some(FinishReason::Length) => "incomplete",
            _ => "completed",
        };
        let prompt_total = self.usage.total_input_billed();
        out.push(self.ev(
            "response.completed",
            json!({
                "response": {
                    "id": self.id,
                    "object": "response",
                    "model": self.model,
                    "status": status,
                    "usage": {
                        "input_tokens": prompt_total,
                        "output_tokens": self.usage.output_tokens,
                        "total_tokens": prompt_total.saturating_add(self.usage.output_tokens),
                        "input_tokens_details": { "cached_tokens": self.usage.cache.read_tokens },
                        "output_tokens_details": { "reasoning_tokens": self.usage.thinking_tokens },
                    },
                }
            }),
        ));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use crate::http::ClientConfig;
    use crate::protocol::codec::Codec;
    use crate::protocol::oai_responses::OpenAiResponsesCodec;

    fn cfg() -> ClientConfig {
        ClientConfig::new("https://api.openai.com", "sk")
    }

    fn roundtrip(body: &str) -> (WireRequest, Value) {
        let ir = decode_request(body.as_bytes()).expect("入站解码");
        let http = OpenAiResponsesCodec
            .encode_request(&ir, &cfg())
            .expect("出站编码");
        (ir, serde_json::from_slice(&http.body).unwrap())
    }

    #[test]
    fn string_input_becomes_user_message() {
        let (ir, out) = roundtrip(r#"{"model":"gpt-5","input":"hi"}"#);
        assert_eq!(ir.messages[0].concat_text(), "hi");
        assert_eq!(out["input"][0]["type"], "message");
        assert_eq!(out["input"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn instructions_become_system() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","instructions":"be brief","input":"hi"}"#,
        );
        assert_eq!(ir.system[0].as_text(), Some("be brief"));
    }

    /// Codex 的 Responses Lite：工具藏在 input 的 additional_tools 条目里。
    #[test]
    fn additional_tools_are_lifted() {
        let (ir, _) = roundtrip(
            r#"{"model":"gpt-5.6-sol","tools":null,"input":[
                {"type":"additional_tools","role":"developer","tools":[
                    {"type":"function","name":"shell","parameters":{"type":"object"}},
                    {"type":"custom","name":"apply_patch"}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"go"}]}]}"#,
        );
        let names: Vec<&str> = ir.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["shell", "apply_patch"]);
        assert!(ir.tools[0].is_function());
        assert!(matches!(ir.tools[1].kind, ToolKind::Freeform));
        // 载体条目本身不该变成一条消息。
        assert_eq!(ir.messages.len(), 1);
    }

    #[test]
    fn namespace_tools_expand_into_members() {
        let (ir, _) = roundtrip(
            r#"{"model":"gpt-6","input":[
                {"type":"additional_tools","tools":[
                    {"type":"namespace","name":"functions","tools":[
                        {"type":"function","name":"exec","parameters":{"type":"object"}}]}]}]}"#,
        );
        assert_eq!(ir.tools.len(), 1);
        assert_eq!(ir.tools[0].name, "exec");
        assert!(matches!(
            &ir.tools[0].kind,
            ToolKind::Namespace { name } if name == "functions"
        ));
    }

    #[test]
    fn function_call_and_output_fold_into_messages() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","input":[
                {"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"q\":\"x\"}"},
                {"type":"function_call_output","call_id":"call_1","output":"result"}]}"#,
        );
        match &ir.messages[0].content[0] {
            Block::ToolUse { id, name, input, .. } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "search");
                assert_eq!(input["q"], "x");
            }
            other => panic!("期望 tool_use，得到 {other:?}"),
        }
        match &ir.messages[1].content[0] {
            Block::ToolResult { tool_use_id, .. } => assert_eq!(tool_use_id, "call_1"),
            other => panic!("期望 tool_result，得到 {other:?}"),
        }
    }

    /// reasoning 的 encrypted_content 折回 signature，才能跨协议原样带走。
    #[test]
    fn reasoning_item_maps_to_thinking_with_signature() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","input":[
                {"type":"reasoning","summary":[{"type":"summary_text","text":"thought"}],
                 "encrypted_content":"enc_1"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"ans"}]}]}"#,
        );
        // reasoning 与 message 都是 assistant，应合并成一条消息，思考在最前。
        assert_eq!(ir.messages.len(), 1);
        assert_eq!(ir.messages[0].role, Role::Assistant);
        match &ir.messages[0].content[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "thought");
                assert_eq!(signature.as_deref(), Some("enc_1"));
            }
            other => panic!("期望 thinking，得到 {other:?}"),
        }
    }

    #[test]
    fn unmodeled_input_items_are_skipped() {
        let (ir, _) = roundtrip(
            r#"{"model":"m","input":[
                {"type":"web_search_call","id":"ws_1","status":"completed"},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#,
        );
        assert_eq!(ir.messages.len(), 1);
        assert_eq!(ir.messages[0].concat_text(), "hi");
    }

    #[test]
    fn unknown_top_level_fields_pass_through() {
        let (ir, out) = roundtrip(
            r#"{"model":"m","input":"x","store":false,"previous_response_id":"r1"}"#,
        );
        assert_eq!(ir.extra.get("store"), Some(&json!(false)));
        assert_eq!(out["store"], false);
        assert_eq!(out["previous_response_id"], "r1");
    }

    #[test]
    fn encode_response_orders_output_and_carries_encrypted_content() {
        let raw = br#"{"id":"r","model":"m","status":"completed","output":[
            {"type":"reasoning","summary":[{"type":"summary_text","text":"hmm"}],"encrypted_content":"enc"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"ans"}]},
            {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1}"}]}"#;
        let resp = OpenAiResponsesCodec
            .decode_response(Bytes::from_static(raw))
            .unwrap();
        let out: Value = serde_json::from_slice(&encode_response(&resp).unwrap()).unwrap();
        let types: Vec<&str> = out["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect();
        assert_eq!(types, vec!["reasoning", "message", "function_call"]);
        // encrypted_content 字段必须存在（Codex 的 ResponseItem 缺它就解不出来）。
        assert_eq!(out["output"][0]["encrypted_content"], "enc");
        assert_eq!(out["output"][2]["call_id"], "c1");
    }

    #[test]
    fn encode_response_always_emits_encrypted_content_key() {
        let resp = WireResponse {
            id: Some("r".into()),
            model: Some("m".into()),
            content: vec![Block::Thinking {
                text: "only summary".into(),
                signature: None,
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            extra: Default::default(),
        };
        let out: Value = serde_json::from_slice(&encode_response(&resp).unwrap()).unwrap();
        assert!(
            out["output"][0].get("encrypted_content").is_some(),
            "没有加密内容时也要给 null，否则客户端整个 item 解不出来"
        );
        assert!(out["output"][0]["encrypted_content"].is_null());
    }

    // --- 流式编码 ---

    fn parsed(frames: &[String]) -> Vec<(String, Value)> {
        frames
            .iter()
            .map(|f| {
                let mut ev = String::new();
                let mut data = String::new();
                for line in f.lines() {
                    if let Some(r) = line.strip_prefix("event: ") {
                        ev = r.to_string();
                    } else if let Some(r) = line.strip_prefix("data: ") {
                        data = r.to_string();
                    }
                }
                (ev, serde_json::from_str(&data).unwrap())
            })
            .collect()
    }

    #[test]
    fn stream_encoder_created_first_completed_last() {
        let mut enc = ResponsesStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(enc.encode(&StreamChunk::text("hi")).unwrap());
        out.extend(enc.finish().unwrap());
        let evs = parsed(&out);
        assert_eq!(evs[0].0, "response.created");
        assert_eq!(evs.last().unwrap().0, "response.completed");
        assert_eq!(out.iter().filter(|f| f.contains("response.created")).count(), 1);
    }

    /// 工具调用：done 必须带拼全的 item（Codex 只从 done 的 item 里取）。
    #[test]
    fn stream_encoder_tool_done_carries_complete_item() {
        let mut enc = ResponsesStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(
            enc.encode(&StreamChunk::BlockStart {
                index: 0,
                block: Block::ToolUse {
                    id: "call_1".into(),
                    name: "exec_command".into(),
                    input: json!({}),
                    cache: None,
                },
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::ToolCallDelta {
                index: 0,
                id: "call_1".into(),
                name_delta: None,
                arguments_delta: Some(r#"{"cmd":"#.into()),
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::ToolCallDelta {
                index: 0,
                id: "call_1".into(),
                name_delta: None,
                arguments_delta: Some(r#""ls"}"#.into()),
            })
            .unwrap(),
        );
        out.extend(enc.encode(&StreamChunk::BlockStop { index: 0 }).unwrap());
        out.extend(enc.finish().unwrap());

        let evs = parsed(&out);
        let done = evs
            .iter()
            .find(|(e, _)| e == "response.output_item.done")
            .expect("必须发 output_item.done");
        assert_eq!(done.1["item"]["call_id"], "call_1");
        assert_eq!(done.1["item"]["name"], "exec_command");
        assert_eq!(done.1["item"]["arguments"], r#"{"cmd":"ls"}"#);
        // done 必须早于 response.completed。
        let di = evs.iter().position(|(e, _)| e == "response.output_item.done").unwrap();
        let ci = evs.iter().position(|(e, _)| e == "response.completed").unwrap();
        assert!(di < ci);
    }

    #[test]
    fn stream_encoder_reasoning_has_summary_index_and_added_first() {
        let mut enc = ResponsesStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(
            enc.encode(&StreamChunk::ThinkingDelta {
                index: 0,
                text: "想一下".into(),
                signature_delta: None,
            })
            .unwrap(),
        );
        out.extend(enc.encode(&StreamChunk::BlockStop { index: 0 }).unwrap());
        out.extend(enc.finish().unwrap());
        let evs = parsed(&out);

        let ai = evs.iter().position(|(e, _)| e == "response.output_item.added").unwrap();
        let di = evs
            .iter()
            .position(|(e, _)| e == "response.reasoning_summary_text.delta")
            .unwrap();
        assert!(ai < di, "added 必须早于 summary delta");
        assert_eq!(evs[di].1["summary_index"], 0);

        let done = evs
            .iter()
            .find(|(e, _)| e == "response.output_item.done")
            .unwrap();
        assert_eq!(done.1["item"]["summary"][0]["text"], "想一下");
        assert!(done.1["item"].get("encrypted_content").is_some());
    }

    /// Chat 上游的解码器不产 BlockStop；收尾必须兜底补 done，否则工具不执行。
    #[test]
    fn finish_flushes_items_without_block_stop() {
        let mut enc = ResponsesStreamEncoder::new();
        let mut out = Vec::new();
        out.extend(
            enc.encode(&StreamChunk::BlockStart {
                index: 2,
                block: Block::ToolUse {
                    id: "call_9".into(),
                    name: "shell".into(),
                    input: json!({}),
                    cache: None,
                },
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::ToolCallDelta {
                index: 2,
                id: "call_9".into(),
                name_delta: None,
                arguments_delta: Some(r#"{"a":1}"#.into()),
            })
            .unwrap(),
        );
        out.extend(
            enc.encode(&StreamChunk::Finish {
                reason: FinishReason::ToolCall,
            })
            .unwrap(),
        );
        out.extend(enc.finish().unwrap());

        let evs = parsed(&out);
        let done: Vec<&(String, Value)> = evs
            .iter()
            .filter(|(e, _)| e == "response.output_item.done")
            .collect();
        assert_eq!(done.len(), 1, "收尾必须补且只补一次 done");
        assert_eq!(done[0].1["item"]["call_id"], "call_9");
        assert_eq!(done[0].1["item"]["arguments"], r#"{"a":1}"#);
    }
}
