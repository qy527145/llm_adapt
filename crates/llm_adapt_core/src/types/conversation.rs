//! 角色感知的对话模型。
//!
//! `Conversation` 是交给模型的标准历史表示：可选的 system prompt，后面跟着
//! 交替出现的 user 与 assistant 轮次。块级类型在类型系统层面就杜绝了非法组合
//! （如 `Thinking` 不能出现在 user 消息里，`ToolResult` 不能出现在 assistant
//! 消息里）。

use serde::{Deserialize, Serialize};

use super::cache::CacheMarker;
use super::tool_result::ToolResultContent;

// ---------------------------------------------------------------------------
// 顶层对话
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub turns: Vec<Turn>,
}

impl Conversation {
    /// 一次性对话：只有一条 user 文本轮次，没有 system prompt。
    pub fn single_user(text: impl Into<String>) -> Self {
        Self {
            system: None,
            turns: vec![Turn::User(UserMessage::text(text))],
        }
    }

    /// 一次性对话：带 system prompt，外加一条 user 文本轮次。
    pub fn with_system(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            system: Some(SystemPrompt::text(system)),
            turns: vec![Turn::User(UserMessage::text(user))],
        }
    }

    /// 追加一轮。返回 `&mut self` 以便链式调用。
    pub fn push(&mut self, turn: Turn) -> &mut Self {
        self.turns.push(turn);
        self
    }
}

/// 对话中的一轮。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Turn {
    User(UserMessage),
    Assistant(AssistantMessage),
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemPrompt {
    pub blocks: Vec<SystemBlock>,
}

impl SystemPrompt {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![SystemBlock::Text { text: text.into(), cache: None }],
        }
    }

    /// 返回所有 `Text` 块拼接后的纯文本内容。
    pub fn text_content(&self) -> String {
        self.blocks
            .iter()
            .map(|b| match b {
                SystemBlock::Text { text, .. } => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn with_cache(mut self, marker: CacheMarker) -> Self {
        if let Some(last) = self.blocks.last_mut() {
            match last {
                SystemBlock::Text { cache, .. } => *cache = Some(marker),
            }
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
}

// ---------------------------------------------------------------------------
// 用户消息
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserMessage {
    pub blocks: Vec<UserBlock>,
}

impl UserMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![UserBlock::Text { text: text.into(), cache: None }],
        }
    }

    /// 添加一个工具结果并返回 self 以便链式调用。
    pub fn with_tool_result(
        mut self,
        call_id: impl Into<String>,
        content: ToolResultContent,
    ) -> Self {
        self.blocks.push(UserBlock::ToolResult {
            call_id: call_id.into(),
            content,
            is_error: false,
            cache: None,
        });
        self
    }

    pub fn push(&mut self, block: UserBlock) -> &mut Self {
        self.blocks.push(block);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    Image {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    ToolResult {
        call_id: String,
        content: ToolResultContent,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
}

// ---------------------------------------------------------------------------
// 助手消息
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub blocks: Vec<AssistantBlock>,
}

impl AssistantMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![AssistantBlock::Text { text: text.into(), cache: None }],
        }
    }

    /// 拼接所有 `Text` 块；非文本块会被跳过。
    pub fn text_content(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// 本条消息里的所有工具调用，按顺序。
    pub fn tool_calls(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.blocks.iter().filter_map(|b| match b {
            AssistantBlock::ToolCall { id, name, arguments, .. } => {
                Some((id.as_str(), name.as_str(), arguments.as_str()))
            }
            _ => None,
        })
    }

    /// `Thinking` 内容拼接起来的结果（如果有）。
    pub fn thinking_content(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Thinking { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn push(&mut self, block: AssistantBlock) -> &mut Self {
        self.blocks.push(block);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
    /// 推理内容。多轮带工具的对话里回灌消息时，Anthropic 要求带上 `signature`。
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// 模型请求的一次工具调用。
    ///
    /// `arguments` 是厂商送出的**原始** JSON 字符串。流式增量以分片后缀的形式
    /// 到来、拼起来才是这个最终形态；存成解析后的 `Value` 会丢掉这一性质。
    /// 需要 `serde_json::Value` 时用 [`AssistantBlock::parsed_arguments`]。
    ToolCall {
        id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
}

impl AssistantBlock {
    /// 工具调用且 JSON 合法时返回 `Some(Ok(v))`，参数字符串 JSON 非法时返回
    /// `Some(Err(_))`，非工具块返回 `None`。
    pub fn parsed_arguments(&self) -> Option<Result<serde_json::Value, serde_json::Error>> {
        match self {
            AssistantBlock::ToolCall { arguments, .. } => Some(serde_json::from_str(arguments)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsed_arguments_only_on_tool_call() {
        let text = AssistantBlock::Text { text: "hi".into(), cache: None };
        assert!(text.parsed_arguments().is_none());

        let call = AssistantBlock::ToolCall {
            id: "t1".into(),
            name: "add".into(),
            arguments: r#"{"a":1,"b":2}"#.into(),
            cache: None,
        };
        let parsed = call.parsed_arguments().unwrap().unwrap();
        assert_eq!(parsed["a"], 1);
        assert_eq!(parsed["b"], 2);
    }

    #[test]
    fn assistant_message_helpers() {
        let mut m = AssistantMessage::text("hello");
        m.push(AssistantBlock::Thinking { text: "hm".into(), signature: None });
        m.push(AssistantBlock::ToolCall {
            id: "t1".into(),
            name: "f".into(),
            arguments: "{}".into(),
            cache: None,
        });
        assert_eq!(m.text_content(), "hello");
        assert_eq!(m.thinking_content(), "hm");
        let calls: Vec<_> = m.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "f");
    }

    #[test]
    fn system_prompt_with_cache_marker() {
        let s = SystemPrompt::text("you are concise").with_cache(CacheMarker::ephemeral_1h());
        match &s.blocks[0] {
            SystemBlock::Text { cache, .. } => {
                assert_eq!(cache.unwrap().ttl, super::super::cache::CacheTtl::Ephemeral1h);
            }
        }
    }
}
