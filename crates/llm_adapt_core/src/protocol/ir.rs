//! 协议无关的中间表示（IR）。
//!
//! 所有协议转换都经过这里：ergonomic 层（[`crate::types`]）→ IR → 各协议的 wire 格式。
//! N 个协议只需 2N 个编解码方向，而不是 N² 个两两转换器。
//!
//! **设计原则：宁可少建模，也不要在转换里悄悄丢东西。** 认不出的内容块进
//! [`Block::Unmodeled`]，认不出的顶层参数进 [`WireRequest::extra`]，两者都原样透传。
//! 同协议直通时它们一个字节都不用改。
//!
//! ## 为什么工具结果不单独占一个 Role
//!
//! 本 IR 只有 `System/User/Assistant` 三个角色，工具结果作为
//! [`Block::ToolResult`] 留在 user 消息里（Anthropic 的规范形态）。OpenAI Chat 需要
//! 独立的 `role: "tool"` 消息，由该协议的 codec 在编码时拆出去。反过来（网关方向）
//! 解码时再折回来。这样 IR 是「更可表达」的那一侧，不会被较弱的协议拉平。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 协议标识
// ---------------------------------------------------------------------------

/// 内置协议。JSON 名一律显式写死，不用 `rename_all` 推导。
///
/// 类型名里的 `OpenAi` 在 `snake_case` 规则下会变成 `open_ai_chat`，而
/// [`Protocol::as_str`]（DB 列、配置、IPC、前端联合类型）里一律是 `openai_chat`。
/// 两套名字并存会让反序列化静默失败 —— apilot 在那边已经踩过一次。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    /// Anthropic Messages API（`POST /v1/messages`）
    #[serde(rename = "anthropic")]
    Anthropic,
    /// OpenAI Chat Completions（`POST /v1/chat/completions`）
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    /// OpenAI Responses API（`POST /v1/responses`）
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

impl Protocol {
    pub const ALL: [Protocol; 3] = [
        Protocol::Anthropic,
        Protocol::OpenAiChat,
        Protocol::OpenAiResponses,
    ];

    /// 稳定短名：配置、日志、DB、IPC 都用它。
    pub fn as_str(&self) -> &'static str {
        match self {
            Protocol::Anthropic => "anthropic",
            Protocol::OpenAiChat => "openai_chat",
            Protocol::OpenAiResponses => "openai_responses",
        }
    }

    /// [`Protocol::as_str`] 的逆运算。两者必须成对维护。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(Protocol::Anthropic),
            "openai_chat" => Some(Protocol::OpenAiChat),
            "openai_responses" => Some(Protocol::OpenAiResponses),
            _ => None,
        }
    }

    /// 该协议的标准请求路径。
    pub fn default_path(&self) -> &'static str {
        match self {
            Protocol::Anthropic => "/v1/messages",
            Protocol::OpenAiChat => "/v1/chat/completions",
            Protocol::OpenAiResponses => "/v1/responses",
        }
    }

    /// 流式响应结束的标记（用于日志归因与测试断言）。
    pub fn stream_terminator(&self) -> &'static str {
        match self {
            Protocol::Anthropic => "message_stop",
            Protocol::OpenAiChat => "[DONE]",
            Protocol::OpenAiResponses => "response.completed",
        }
    }

    /// 依据请求路径推断协议。识别不出返回 `None`，由调用方决定回落策略。
    pub fn from_path(path: &str) -> Option<Self> {
        let p = path.trim_end_matches('/');
        match p {
            "/v1/messages" | "/messages" => Some(Protocol::Anthropic),
            "/v1/chat/completions" | "/chat/completions" => Some(Protocol::OpenAiChat),
            "/v1/responses" | "/responses" => Some(Protocol::OpenAiResponses),
            _ => None,
        }
    }
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 注册表键：内置协议用 [`Protocol`]，也可以是任意自定义协议的字符串。
///
/// [`ProtocolId::new`] 会把常见别名归一成规范名，便于配置文件与命令行平滑迁移。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolId(String);

impl ProtocolId {
    pub const ANTHROPIC: &'static str = "anthropic";
    pub const OPENAI_CHAT: &'static str = "openai_chat";
    pub const OPENAI_RESPONSES: &'static str = "openai_responses";

    /// 构造一个 id，并把已知别名归一成规范名。
    ///
    /// 归一表：
    /// * `anthropic` / `anthropic_v2` / `anthropic_messages` / `messages` → `anthropic`
    /// * `openai_chat` / `openai_compat` / `openai` / `custom` / `chat_completions` → `openai_chat`
    /// * `openai_responses` / `responses` → `openai_responses`
    ///
    /// 不在表里的一律按下转移小写后的原样保留（自定义协议）。
    pub fn new(raw: impl AsRef<str>) -> Self {
        Self(normalize(raw.as_ref()))
    }

    /// 不归一，原样保留。自定义协议想避开别名表时用这个。
    pub fn from_raw(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// 若是内置协议则返回对应枚举。
    pub fn protocol(&self) -> Option<Protocol> {
        Protocol::parse(&self.0)
    }

    pub fn is_builtin(&self) -> bool {
        self.protocol().is_some()
    }
}

impl From<Protocol> for ProtocolId {
    fn from(p: Protocol) -> Self {
        ProtocolId(p.as_str().to_string())
    }
}

impl std::fmt::Display for ProtocolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn normalize(raw: &str) -> String {
    let s = raw.trim().to_ascii_lowercase();
    match s.as_str() {
        "anthropic" | "anthropic_v2" | "anthropic_messages" | "messages" => {
            ProtocolId::ANTHROPIC.to_string()
        }
        "openai_chat" | "openai_compat" | "openai" | "custom" | "chat_completions" => {
            ProtocolId::OPENAI_CHAT.to_string()
        }
        "openai_responses" | "responses" | "openai_resp" => {
            ProtocolId::OPENAI_RESPONSES.to_string()
        }
        _ => s,
    }
}

// ---------------------------------------------------------------------------
// 内容模型
// ---------------------------------------------------------------------------

/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// 提示词缓存的存活时长。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    /// 默认档（Anthropic 约 5 分钟）。
    Ephemeral5m,
    /// 长存活档（Anthropic 约 1 小时）。单价更高。
    Ephemeral1h,
}

/// 块级缓存断点：标记「缓存到本块为止的前缀」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheMarker {
    pub ttl: CacheTtl,
}

impl CacheMarker {
    pub fn ephemeral_5m() -> Self {
        Self {
            ttl: CacheTtl::Ephemeral5m,
        }
    }

    pub fn ephemeral_1h() -> Self {
        Self {
            ttl: CacheTtl::Ephemeral1h,
        }
    }
}

/// 图片内容的来源。
///
/// `data:` URL 在进入 IR 时就拆成 [`ImageSource::Base64`]，不再以「带前缀的字符串」
/// 这种约定在内部流转 —— 那是个容易漏判的隐式协议。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// 远程 URL。
    Url { url: String },
    /// 内联 base64。
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    /// 渲染回一个可直接放进 wire 的 URL 字符串（内联的变成 `data:` URL）。
    pub fn to_url(&self) -> String {
        match self {
            ImageSource::Url { url } => url.clone(),
            ImageSource::Base64 { media_type, data } => {
                format!("data:{media_type};base64,{data}")
            }
        }
    }

    /// 从可能是 `data:` URL 的字符串还原。
    pub fn from_url(url: &str) -> Self {
        if let Some(rest) = url.strip_prefix("data:") {
            if let Some((media_type, data)) = rest.split_once(";base64,") {
                return ImageSource::Base64 {
                    media_type: media_type.to_string(),
                    data: data.to_string(),
                };
            }
        }
        ImageSource::Url {
            url: url.to_string(),
        }
    }

    pub fn media_type(&self) -> Option<&str> {
        match self {
            ImageSource::Base64 { media_type, .. } => Some(media_type),
            ImageSource::Url { .. } => None,
        }
    }
}

/// 回传的工具结果正文。
///
/// 三种形态对应 wire 上的三种写法，**不做归一**：Anthropic 的 `tool_result.content`
/// 既接受裸字符串也接受块数组，把它们压成同一种会在 wire 上产生无谓差异。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultContent {
    /// 纯文本——所有厂商都接受。
    ///
    /// **必须是结构体变体而不是 `Text(String)`。** serde 的内部标签（`tag = "kind"`）
    /// 不支持「标签 + newtype 变体」的组合：序列化时会在运行时直接报
    /// `cannot serialize tagged newtype variant ... containing a string`，
    /// 也就是任何带工具结果的请求都存不下、传不出去。这类错误编译期看不出来，
    /// 只有真去序列化才会炸 —— 所以下面配了一个整请求往返的测试守它。
    Text { text: String },
    /// 结构化 JSON。不认结构化的厂商会拿到序列化后的字符串。
    Json { value: serde_json::Value },
    /// 多模态块。Anthropic 原生接受；OpenAI 降级成拼接文本（图片丢弃并告警）。
    Blocks { blocks: Vec<ToolResultBlock> },
}

impl ToolResultContent {
    pub fn text(s: impl Into<String>) -> Self {
        ToolResultContent::Text { text: s.into() }
    }

    pub fn json(value: serde_json::Value) -> Self {
        ToolResultContent::Json { value }
    }

    /// 只看文本部分。用于降级到纯文本的协议。
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContent::Text { text } => text.clone(),
            ToolResultContent::Json { value } => value.to_string(),
            ToolResultContent::Blocks { blocks } => blocks
                .iter()
                .filter_map(|b| match b {
                    ToolResultBlock::Text { text } => Some(text.as_str()),
                    ToolResultBlock::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// 多模态块里被丢弃的图片数量（降级时用来告警）。
    pub fn dropped_image_count(&self) -> usize {
        match self {
            ToolResultContent::Blocks { blocks } => blocks
                .iter()
                .filter(|b| matches!(b, ToolResultBlock::Image { .. }))
                .count(),
            _ => 0,
        }
    }
}

/// 多模态工具结果里的一个块。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        text: String,
    },
    Image {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

/// 消息内容块。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    Image {
        source: ImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    /// 模型发起的工具调用。
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    /// 回传给模型的工具执行结果。
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    /// 扩展思考。`signature` 是 Anthropic 的签名，跨协议无法保真。
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// 被上游加密的思考内容，只能原样回传。
    RedactedThinking { data: String },
    /// 未建模的内容块，整块原样留着。
    ///
    /// 客户端与上游总在先用上新类型：Anthropic 的 `server_tool_use` /
    /// `web_search_tool_result`、OpenAI Responses 的 `web_search_call`、
    /// Claude Code 带附件时发的 `document`…… IR 不认识它们 **不等于** 这次请求该失败。
    /// 同协议直通时这些块本来就一个字节都不用改；解不出来就报错，等于把用户挡在门外。
    Unmodeled { raw: serde_json::Value },
}

impl Block {
    pub fn text(s: impl Into<String>) -> Self {
        Block::Text {
            text: s.into(),
            cache: None,
        }
    }

    pub fn text_cached(s: impl Into<String>, cache: CacheMarker) -> Self {
        Block::Text {
            text: s.into(),
            cache: Some(cache),
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Block::Text { text, .. } => Some(text),
            _ => None,
        }
    }

    pub fn is_text(&self) -> bool {
        matches!(self, Block::Text { .. })
    }

    pub fn as_tool_use(&self) -> Option<(&str, &str, &serde_json::Value)> {
        match self {
            Block::ToolUse {
                id, name, input, ..
            } => Some((id, name, input)),
            _ => None,
        }
    }
}

/// 一条消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: Role,
    /// 始终显式序列化（空数组也要出现）。
    pub content: Vec<Block>,
}

impl WireMessage {
    pub fn new(role: Role, content: Vec<Block>) -> Self {
        Self { role, content }
    }

    pub fn user_text(s: impl Into<String>) -> Self {
        Self::new(Role::User, vec![Block::text(s)])
    }

    /// 把该消息里所有文本块拼起来。用于估算 token 与生成缓存键。
    pub fn concat_text(&self) -> String {
        self.content
            .iter()
            .filter_map(Block::as_text)
            .collect::<Vec<_>>()
            .join("")
    }
}

// ---------------------------------------------------------------------------
// 工具声明
// ---------------------------------------------------------------------------

/// 工具的种类。
///
/// `Function` 是本地函数工具（我们要自己执行）；其余是**服务端工具**——由模型厂商
/// 自己执行，我们只负责声明和理解它回传的活动块。形状参考 codex 的 `ToolSpec`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolKind {
    /// 本地函数工具。
    Function,
    /// 厂商托管的服务端工具（web_search / web_fetch / code_execution / ...）。
    /// `spec` 是原样的厂商声明对象，由各 codec 负责填装与降级。
    ServerTool { spec: serde_json::Value },
    /// 自由格式工具（无 JSON Schema），如 codex 的 `apply_patch`。
    Freeform,
    /// 一组打包工具（codex Responses Lite 的 `type: "namespace"`）。
    /// 出站编码一律只用裸 `name`：函数名只允许 `[a-zA-Z0-9_-]`，带点的会被上游拒掉。
    Namespace { name: String },
}

impl Default for ToolKind {
    fn default() -> Self {
        ToolKind::Function
    }
}

/// 下发给模型的工具声明。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 统一成 JSON Schema。各协议表述差异在 codec 里抹平。
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMarker>,
    #[serde(default)]
    pub kind: ToolKind,
}

impl ToolSpec {
    /// 一个本地函数工具。
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
            cache: None,
            kind: ToolKind::Function,
        }
    }

    pub fn is_function(&self) -> bool {
        matches!(self.kind, ToolKind::Function)
    }

    pub fn is_server_tool(&self) -> bool {
        matches!(self.kind, ToolKind::ServerTool { .. })
    }
}

/// 模型选工具的激进程度。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolChoice {
    /// 模型自由决定（默认）。
    #[default]
    Auto,
    /// 禁止调用任何工具。
    None,
    /// 必须调用某个工具。
    Required,
    /// 必须调用指定的这个工具。
    Specific { name: String },
}

// ---------------------------------------------------------------------------
// 请求
// ---------------------------------------------------------------------------

/// 期望的响应形状。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    Json,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: bool,
    },
}

/// 扩展思考开关。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThinkingConfig {
    /// Anthropic 的 `thinking.budget_tokens`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// OpenAI 系的 `reasoning.effort`（low / medium / high）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

/// 协议无关的一次推理请求。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireRequest {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    /// Anthropic 把 system 放在顶层、OpenAI 放在 messages 里；这里统一为顶层。
    /// 空数组也要显式序列化。
    pub system: Vec<Block>,
    pub messages: Vec<WireMessage>,
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    #[serde(default)]
    pub response_format: ResponseFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// 协议原生但未建模的顶层字段，原样透传。
    #[serde(default, flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

impl WireRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            stream: false,
            system: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            temperature: None,
            top_p: None,
            max_tokens: None,
            seed: None,
            stop: Vec::new(),
            response_format: ResponseFormat::Text,
            thinking: None,
            user_id: None,
            extra: IndexMap::new(),
        }
    }

    /// 把 system 分段里的文本块拼成一个字符串。
    ///
    /// **用换行拼接，不用空串。** system 是分段来的（人设 / 技能 / 记忆 / 运行环境），
    /// 空串会把它们粘成一坨，模型读到的是连成一片的字。三个协议必须一致，
    /// 否则同一段对话换渠道后提示词就变了。
    pub fn system_text(&self) -> String {
        self.system
            .iter()
            .filter_map(Block::as_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 请求里出现的全部文本，用于粗略估算输入 token。
    pub fn all_text(&self) -> String {
        let mut out = String::new();
        for b in &self.system {
            if let Some(t) = b.as_text() {
                out.push_str(t);
            }
        }
        for m in &self.messages {
            out.push_str(&m.concat_text());
            // 工具调用的入参也要计入，否则工具密集的会话会被严重低估。
            for b in &m.content {
                if let Block::ToolUse { input, .. } = b {
                    out.push_str(&input.to_string());
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 响应
// ---------------------------------------------------------------------------

/// 模型停止生成的原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    Safety,
    StopSequence,
    /// 各家协议的其它取值，原样保留以免丢信息。
    Other {
        value: String,
    },
}

impl FinishReason {
    /// 从各家协议的字符串归一。
    pub fn parse(s: &str) -> Self {
        match s {
            "stop" | "end_turn" | "stop_sequence" => FinishReason::Stop,
            "length" | "max_tokens" | "max_output_tokens" => FinishReason::Length,
            "tool_calls" | "tool_use" | "function_call" => FinishReason::ToolCall,
            "content_filter" | "refusal" | "safety" => FinishReason::Safety,
            other => FinishReason::Other {
                value: other.to_string(),
            },
        }
    }

    /// 从可选的原生字符串归一。上游没给时返回 `Other { value: "" }`。
    pub fn parse_opt(s: Option<&str>) -> Self {
        match s {
            Some(s) => Self::parse(s),
            None => FinishReason::Other {
                value: String::new(),
            },
        }
    }

    /// 归一后写回指定协议的字符串。
    pub fn to_wire(&self, proto: Protocol) -> String {
        match self {
            FinishReason::Stop => match proto {
                Protocol::Anthropic => "end_turn".into(),
                _ => "stop".into(),
            },
            FinishReason::Length => match proto {
                Protocol::Anthropic => "max_tokens".into(),
                Protocol::OpenAiChat => "length".into(),
                Protocol::OpenAiResponses => "max_output_tokens".into(),
            },
            FinishReason::ToolCall => match proto {
                Protocol::Anthropic => "tool_use".into(),
                _ => "tool_calls".into(),
            },
            FinishReason::Safety => match proto {
                Protocol::Anthropic => "refusal".into(),
                _ => "content_filter".into(),
            },
            FinishReason::StopSequence => match proto {
                Protocol::Anthropic => "stop_sequence".into(),
                _ => "stop".into(),
            },
            FinishReason::Other { value } => value.clone(),
        }
    }
}

/// 一次推理的完整响应。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub content: Vec<Block>,
    pub finish_reason: FinishReason,
    #[serde(default)]
    pub usage: Usage,
    /// 协议原生但未建模的顶层字段，原样透传（如 OpenAI 的 `system_fingerprint`）。
    #[serde(default, flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

impl WireResponse {
    /// 把所有文本块拼成最终答案。
    pub fn concat_text(&self) -> String {
        self.content
            .iter()
            .filter_map(Block::as_text)
            .collect::<Vec<_>>()
            .join("")
    }

    /// 按出现顺序取出全部工具调用。
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(Block::as_tool_use).collect()
    }
}

// ---------------------------------------------------------------------------
// 用量
// ---------------------------------------------------------------------------

/// 用量数字的来源，决定计费可信度。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageSource {
    /// 上游如实返回。
    #[default]
    Upstream,
    /// 上游没给，本地估算。
    LocalEstimate,
}

/// 缓存写入用量，按 TTL 档位拆分。
///
/// 不按 TTL 拆分的厂商（OpenAI）只填 `total`，两个档位留 0。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheWriteUsage {
    /// 全部档位合计。两个档位都有值时它等于二者之和。
    pub total: u32,
    pub ephemeral_5m: u32,
    pub ephemeral_1h: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheUsage {
    /// 命中缓存、按低价计费的输入 token。
    pub read_tokens: u32,
    pub write: CacheWriteUsage,
}

/// 单次请求的 token 用量。
///
/// **口径约定：`input_tokens` 只算未命中缓存的输入**（Anthropic 语义）。
/// OpenAI 系的 `prompt_tokens` 是**含**缓存的，由各自 codec 负责减去
/// `prompt_tokens_details.cached_tokens`。这条约定被破坏会导致缓存部分被重复计费。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub thinking_tokens: u32,
    #[serde(default)]
    pub cache: CacheUsage,
    #[serde(default)]
    pub source: UsageSource,
}

impl Usage {
    /// 计入 prompt 侧的全部 token（基础输入 + 缓存读 + 缓存写）。
    pub fn total_input_billed(&self) -> u32 {
        self.input_tokens
            .saturating_add(self.cache.read_tokens)
            .saturating_add(self.cache.write.total)
    }

    pub fn total(&self) -> u32 {
        self.total_input_billed().saturating_add(self.output_tokens)
    }

    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// 非零字段覆盖式合并。
    ///
    /// 流式响应的 usage 常被拆成多个事件下发（Claude 的 `message_start` 给 input、
    /// `message_delta` 给 output），**零值不得覆盖已有正数**。
    pub fn merge_non_zero(&mut self, other: &Usage) {
        if other.input_tokens > 0 {
            self.input_tokens = other.input_tokens;
        }
        if other.output_tokens > 0 {
            self.output_tokens = other.output_tokens;
        }
        if other.thinking_tokens > 0 {
            self.thinking_tokens = other.thinking_tokens;
        }
        if other.cache.read_tokens > 0 {
            self.cache.read_tokens = other.cache.read_tokens;
        }
        if other.cache.write.total > 0 {
            self.cache.write = other.cache.write.clone();
        }
        if other.source == UsageSource::LocalEstimate {
            self.source = UsageSource::LocalEstimate;
        }
    }

    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(other.thinking_tokens);
        self.cache.read_tokens = self
            .cache
            .read_tokens
            .saturating_add(other.cache.read_tokens);
        self.cache.write.total = self
            .cache
            .write
            .total
            .saturating_add(other.cache.write.total);
        self.cache.write.ephemeral_5m = self
            .cache
            .write
            .ephemeral_5m
            .saturating_add(other.cache.write.ephemeral_5m);
        self.cache.write.ephemeral_1h = self
            .cache
            .write
            .ephemeral_1h
            .saturating_add(other.cache.write.ephemeral_1h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JSON 名必须与 `as_str()` 完全一致，否则配置/DB/IPC 三处会静默对不上。
    #[test]
    fn protocol_json_name_matches_as_str() {
        for p in Protocol::ALL {
            assert_eq!(
                serde_json::to_string(&p).unwrap(),
                format!("\"{}\"", p.as_str())
            );
            assert_eq!(
                serde_json::from_str::<Protocol>(&format!("\"{}\"", p.as_str())).unwrap(),
                p
            );
        }
    }

    #[test]
    fn protocol_parse_is_inverse_of_as_str() {
        for p in Protocol::ALL {
            assert_eq!(Protocol::parse(p.as_str()), Some(p));
        }
        assert_eq!(Protocol::parse("gemini"), None);
    }

    /// 别名归一：老配置里的 `anthropic_v2` / `openai_compat` 必须仍然指向同一个协议。
    #[test]
    fn protocol_id_normalizes_legacy_aliases() {
        for (input, expected) in [
            ("anthropic", "anthropic"),
            ("ANTHROPIC_V2", "anthropic"),
            ("anthropic_messages", "anthropic"),
            ("openai_compat", "openai_chat"),
            ("openai", "openai_chat"),
            ("custom", "openai_chat"),
            ("openai_responses", "openai_responses"),
            ("responses", "openai_responses"),
        ] {
            assert_eq!(ProtocolId::new(input).as_str(), expected, "别名 {input}");
        }
        // 表外的自定义协议原样保留（仅小写化）
        assert_eq!(ProtocolId::new("My_Vendor_V1").as_str(), "my_vendor_v1");
        assert_eq!(
            ProtocolId::from_raw("My_Vendor_V1").as_str(),
            "My_Vendor_V1"
        );
    }

    #[test]
    fn protocol_id_round_trips_through_protocol() {
        for p in Protocol::ALL {
            let id = ProtocolId::from(p);
            assert_eq!(id.protocol(), Some(p));
            assert!(id.is_builtin());
        }
        assert!(!ProtocolId::new("my_vendor").is_builtin());
    }

    /// 空集合必须显式序列化成 `[]`，不能整个消失 ——
    /// 跨语言消费方把它们当必有数组去 `.map()` 时会直接崩。
    #[test]
    fn empty_collections_are_serialized_explicitly() {
        let req = WireRequest::new("m");
        let v = serde_json::to_value(&req).unwrap();
        for k in ["system", "messages", "tools", "stop"] {
            assert_eq!(v[k], serde_json::json!([]), "{k} 必须是显式空数组");
        }
        let msg = WireMessage::new(Role::User, vec![]);
        assert_eq!(
            serde_json::to_value(&msg).unwrap()["content"],
            serde_json::json!([])
        );
    }

    #[test]
    fn unmodeled_block_survives_round_trip() {
        let block = Block::Unmodeled {
            raw: serde_json::json!({"type": "web_search_tool_result", "content": [1, 2]}),
        };
        let s = serde_json::to_string(&block).unwrap();
        assert_eq!(serde_json::from_str::<Block>(&s).unwrap(), block);
    }

    #[test]
    fn extra_fields_pass_through_unmodeled() {
        let raw = r#"{"model":"m","stream":false,"system":[],"messages":[],"tools":[],
                      "stop":[],"top_k":42,"service_tier":"flex"}"#;
        let req: WireRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.extra.get("top_k"), Some(&serde_json::json!(42)));
        assert_eq!(
            req.extra.get("service_tier"),
            Some(&serde_json::json!("flex"))
        );
        // 重新编码后它们仍在
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["top_k"], 42);
        assert_eq!(v["service_tier"], "flex");
    }

    /// 三个协议必须用同一种方式拼 system，否则同一段对话换渠道提示词就变了。
    #[test]
    fn system_segments_join_with_newlines() {
        let mut req = WireRequest::new("m");
        req.system = vec![
            Block::text("人设"),
            Block::text("技能"),
            Block::text("记忆"),
        ];
        assert_eq!(
            req.system_text(),
            "人设
技能
记忆"
        );
        assert_eq!(WireRequest::new("m").system_text(), "");
    }

    #[test]
    fn data_url_becomes_base64_source() {
        let src = ImageSource::from_url("data:image/png;base64,AAAB");
        assert_eq!(
            src,
            ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "AAAB".into()
            }
        );
        assert_eq!(src.to_url(), "data:image/png;base64,AAAB");
        let url = ImageSource::from_url("https://x/y.png");
        assert_eq!(
            url,
            ImageSource::Url {
                url: "https://x/y.png".into()
            }
        );
        assert_eq!(url.to_url(), "https://x/y.png");
    }

    /// **整请求 / 整响应往返。** 这条测试的意义不在于「值相等」，而在于
    /// 序列化路径真的能跑通 —— IR 里任何一个块只要 serde 表示不合法，
    /// 就会在这里炸出来（例如内部标签 + newtype 变体那种组合，编译期完全看不出来）。
    #[test]
    fn full_request_and_response_round_trip() {
        let mut req = WireRequest::new("m");
        req.system = vec![Block::text("sys")];
        req.messages = vec![
            WireMessage::user_text("hi"),
            WireMessage::new(
                Role::Assistant,
                vec![
                    Block::Thinking {
                        text: "hmm".into(),
                        signature: Some("sig".into()),
                    },
                    Block::text("let me look"),
                    Block::ToolUse {
                        id: "t1".into(),
                        name: "read".into(),
                        input: serde_json::json!({"path": "a.rs"}),
                        cache: None,
                    },
                ],
            ),
            WireMessage::new(
                Role::User,
                vec![
                    Block::ToolResult {
                        tool_use_id: "t1".into(),
                        content: ToolResultContent::text("contents"),
                        is_error: false,
                        cache: None,
                    },
                    Block::ToolResult {
                        tool_use_id: "t2".into(),
                        content: ToolResultContent::json(serde_json::json!({"ok": true})),
                        is_error: true,
                        cache: Some(CacheMarker::ephemeral_5m()),
                    },
                    Block::ToolResult {
                        tool_use_id: "t3".into(),
                        content: ToolResultContent::Blocks {
                            blocks: vec![
                                ToolResultBlock::Text { text: "x".into() },
                                ToolResultBlock::Image {
                                    url: "https://x/y.png".into(),
                                    detail: None,
                                },
                            ],
                        },
                        is_error: false,
                        cache: None,
                    },
                ],
            ),
            WireMessage::new(
                Role::User,
                vec![
                    Block::Image {
                        source: ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "AA".into(),
                        },
                        detail: Some("high".into()),
                        cache: None,
                    },
                    Block::RedactedThinking { data: "enc".into() },
                    Block::Unmodeled {
                        raw: serde_json::json!({"type": "web_search_call"}),
                    },
                ],
            ),
        ];
        req.tools
            .push(ToolSpec::function("f", "d", serde_json::json!({})));
        req.thinking = Some(ThinkingConfig {
            budget_tokens: Some(1024),
            effort: None,
        });
        req.response_format = ResponseFormat::Json;
        req.tool_choice = ToolChoice::Specific { name: "f".into() };
        req.extra.insert("store".into(), serde_json::json!(false));

        let json = serde_json::to_string(&req).expect("请求必须能序列化");
        let back: WireRequest = serde_json::from_str(&json).expect("请求必须能反序列化");
        assert_eq!(back, req);

        let mut resp = WireResponse {
            id: Some("r".into()),
            model: Some("m".into()),
            content: vec![
                Block::text("done"),
                Block::Thinking {
                    text: "t".into(),
                    signature: None,
                },
                Block::ToolUse {
                    id: "t9".into(),
                    name: "f".into(),
                    input: serde_json::json!({}),
                    cache: None,
                },
                Block::Unmodeled {
                    raw: serde_json::json!({"type": "message", "x": 1}),
                },
            ],
            finish_reason: FinishReason::Other {
                value: "pause_turn".into(),
            },
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
                thinking_tokens: 3,
                cache: CacheUsage {
                    read_tokens: 4,
                    write: CacheWriteUsage {
                        total: 5,
                        ephemeral_5m: 5,
                        ephemeral_1h: 0,
                    },
                },
                ..Default::default()
            },
            extra: IndexMap::new(),
        };
        resp.extra
            .insert("status".into(), serde_json::json!("completed"));
        let json = serde_json::to_string(&resp).expect("响应必须能序列化");
        let back: WireResponse = serde_json::from_str(&json).expect("响应必须能反序列化");
        assert_eq!(back, resp);
    }

    #[test]
    fn usage_merge_never_overwrites_with_zero() {
        let mut acc = Usage {
            input_tokens: 10,
            ..Default::default()
        };
        acc.merge_non_zero(&Usage {
            output_tokens: 5,
            ..Default::default()
        });
        assert_eq!(acc.input_tokens, 10, "零值不得覆盖已有的正数");
        assert_eq!(acc.output_tokens, 5);
    }

    #[test]
    fn finish_reason_normalizes_and_keeps_unknown() {
        assert_eq!(FinishReason::parse("end_turn"), FinishReason::Stop);
        assert_eq!(FinishReason::parse("tool_use"), FinishReason::ToolCall);
        assert_eq!(
            FinishReason::parse("pause_turn"),
            FinishReason::Other {
                value: "pause_turn".into()
            }
        );
        assert_eq!(FinishReason::Stop.to_wire(Protocol::Anthropic), "end_turn");
        assert_eq!(FinishReason::Stop.to_wire(Protocol::OpenAiChat), "stop");
        assert_eq!(
            FinishReason::Other {
                value: "pause_turn".into()
            }
            .to_wire(Protocol::Anthropic),
            "pause_turn"
        );
    }
}
