//! 厂商托管的服务端工具（web_search 等）。
//!
//! 与本地函数工具的区别：**它由厂商自己执行**。我们只负责在请求里声明它，
//! 并在响应里把它回传的活动块（`server_tool_use` / `web_search_tool_result` /
//! `web_search_call`）认出来呈现给用户 —— 本地没有任何东西可执行。
//!
//! ## 不支持时要出声
//!
//! 三家协议对服务端工具的支持不同（Chat Completions 根本没有对应物）。
//! [`resolve_specs`] 因此返回**两份**结果：能声明的、以及被丢弃并需要告知用户的。
//! 静默丢弃是最坏的选项 —— 用户以为开了联网搜索，模型其实一个结果都拿不到，
//! 而模型自己也不知道，只会一本正经地编。

use serde_json::{json, Value};

use crate::protocol::ir::{Protocol, ToolKind, ToolSpec};

/// 内置认识的服务端工具。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerTool {
    /// 联网搜索。
    WebSearch,
    /// 抓取指定 URL。
    WebFetch,
    /// 服务端代码执行。
    CodeExecution,
    /// 文件检索（Responses 的 file_search，需要先建 vector store）。
    FileSearch,
}

impl ServerTool {
    /// 从配置里的名字解析。接受常见的几种写法。
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "web_search" | "web-search" | "search" => Some(ServerTool::WebSearch),
            "web_fetch" | "fetch" | "url_fetch" => Some(ServerTool::WebFetch),
            "code_execution" | "code_interpreter" | "python" => Some(ServerTool::CodeExecution),
            "file_search" => Some(ServerTool::FileSearch),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ServerTool::WebSearch => "web_search",
            ServerTool::WebFetch => "web_fetch",
            ServerTool::CodeExecution => "code_execution",
            ServerTool::FileSearch => "file_search",
        }
    }

    /// 该协议下这个工具的声明对象。`None` = 这个协议没有对应物。
    ///
    /// **类型字符串写死在此处而不是拼出来。** 厂商给这些工具带版本号
    /// （`web_search_20250305`），拼字符串必然拼错；要换新版本就在配置里
    /// 用 `server_tool_specs` 原样给一个对象。
    pub fn spec(&self, protocol: Protocol, max_uses: Option<u32>) -> Option<Value> {
        match (self, protocol) {
            // Anthropic：带版本号的类型名 + 可选的调用次数上限。
            (ServerTool::WebSearch, Protocol::Anthropic) => {
                let mut v = json!({"type": "web_search_20250305", "name": "web_search"});
                if let Some(n) = max_uses {
                    v["max_uses"] = json!(n);
                }
                Some(v)
            }
            (ServerTool::WebFetch, Protocol::Anthropic) => {
                let mut v = json!({"type": "web_fetch_20250910", "name": "web_fetch"});
                if let Some(n) = max_uses {
                    v["max_uses"] = json!(n);
                }
                Some(v)
            }
            (ServerTool::CodeExecution, Protocol::Anthropic) => {
                Some(json!({"type": "code_execution_20250522", "name": "code_execution"}))
            }

            // Responses：类型名就是裸的 `web_search`，没有版本号。
            (ServerTool::WebSearch, Protocol::OpenAiResponses) => {
                Some(json!({"type": "web_search"}))
            }
            (ServerTool::CodeExecution, Protocol::OpenAiResponses) => {
                Some(json!({"type": "code_interpreter", "container": {"type": "auto"}}))
            }
            (ServerTool::FileSearch, Protocol::OpenAiResponses) => {
                Some(json!({"type": "file_search"}))
            }

            // Chat Completions 与其余组合：没有对应物。
            _ => None,
        }
    }
}

/// 把配置里的名字解析成可以下发的 [`ToolSpec`]。
///
/// 返回 `(可下发的, 被丢弃的名字)`。调用方**必须**把第二项上报给用户：
/// 静默丢弃等于让用户以为开了联网搜索而实际没有。
pub fn resolve_specs(
    protocol: Protocol,
    names: &[String],
    max_uses: Option<u32>,
) -> (Vec<ToolSpec>, Vec<String>) {
    let mut ok = Vec::new();
    let mut dropped = Vec::new();
    for raw in names {
        let Some(tool) = ServerTool::parse(raw) else {
            dropped.push(raw.clone());
            continue;
        };
        match tool.spec(protocol, max_uses) {
            Some(spec) => ok.push(ToolSpec {
                name: tool.as_str().to_string(),
                description: None,
                input_schema: json!({}),
                cache: None,
                kind: ToolKind::ServerTool { spec },
            }),
            None => dropped.push(raw.clone()),
        }
    }
    (ok, dropped)
}

/// 把配置里原样给出的 spec 对象包成 [`ToolSpec`]。
///
/// 逃生舱：厂商上新版本、或要用的参数超出内置表时，用它原样下发。
pub fn spec_from_raw(raw: &Value) -> Option<ToolSpec> {
    let ty = raw.get("type").and_then(Value::as_str)?;
    // Responses 的服务端工具没有 `name` 字段，用 type 当名字。
    let name = raw
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(ty)
        .to_string();
    Some(ToolSpec {
        name,
        description: None,
        input_schema: json!({}),
        cache: None,
        kind: ToolKind::ServerTool { spec: raw.clone() },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_web_search_carries_version_and_max_uses() {
        let spec = ServerTool::WebSearch.spec(Protocol::Anthropic, Some(3)).unwrap();
        assert_eq!(spec["type"], "web_search_20250305");
        assert_eq!(spec["name"], "web_search");
        assert_eq!(spec["max_uses"], 3);
    }

    #[test]
    fn responses_web_search_is_the_bare_type() {
        let spec = ServerTool::WebSearch.spec(Protocol::OpenAiResponses, Some(9)).unwrap();
        assert_eq!(spec, json!({"type": "web_search"}), "Responses 不带版本号也不认 max_uses");
    }

    /// Chat Completions 没有服务端工具的对应物 —— 必须报出来，不能静默丢。
    #[test]
    fn chat_reports_server_tools_as_unsupported() {
        let (ok, dropped) =
            resolve_specs(Protocol::OpenAiChat, &["web_search".to_string()], None);
        assert!(ok.is_empty());
        assert_eq!(dropped, vec!["web_search".to_string()]);
    }

    #[test]
    fn unknown_names_are_reported_not_guessed() {
        let names = vec!["web_search".to_string(), "telepathy".to_string()];
        let (ok, dropped) = resolve_specs(Protocol::Anthropic, &names, None);
        assert_eq!(ok.len(), 1);
        assert_eq!(dropped, vec!["telepathy".to_string()]);
    }

    #[test]
    fn raw_spec_escape_hatch_uses_type_as_name_when_unnamed() {
        let raw = json!({"type": "web_search_20260101", "max_uses": 7});
        let spec = spec_from_raw(&raw).unwrap();
        assert_eq!(spec.name, "web_search_20260101");
        assert!(spec.is_server_tool());
        assert_eq!(spec.kind, ToolKind::ServerTool { spec: raw });
    }
}
