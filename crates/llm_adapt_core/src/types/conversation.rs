//! Role-aware conversation model.
//!
//! `Conversation` is the canonical history representation passed to the model:
//! an optional system prompt followed by an alternating sequence of user and
//! assistant turns. The block-level types prevent illegal compositions at the
//! type system level (e.g. `Thinking` cannot appear in a user message,
//! `ToolResult` cannot appear in an assistant message).

use serde::{Deserialize, Serialize};

use super::cache::CacheMarker;
use super::tool_result::ToolResultContent;

// ---------------------------------------------------------------------------
// Top-level conversation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub turns: Vec<Turn>,
}

impl Conversation {
    /// One-shot conversation with a single user text turn and no system prompt.
    pub fn single_user(text: impl Into<String>) -> Self {
        Self {
            system: None,
            turns: vec![Turn::User(UserMessage::text(text))],
        }
    }

    /// One-shot conversation with a system prompt and a single user text turn.
    pub fn with_system(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self {
            system: Some(SystemPrompt::text(system)),
            turns: vec![Turn::User(UserMessage::text(user))],
        }
    }

    /// Append a turn. Returns `&mut self` for chaining.
    pub fn push(&mut self, turn: Turn) -> &mut Self {
        self.turns.push(turn);
        self
    }
}

/// One turn in the conversation.
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

    /// Returns the concatenated plain-text content of all `Text` blocks.
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
// User message
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

    /// Add a tool result and return self for chaining.
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
// Assistant message
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

    /// Concatenate all `Text` blocks; non-text blocks are skipped.
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

    /// All tool calls in this message, in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.blocks.iter().filter_map(|b| match b {
            AssistantBlock::ToolCall { id, name, arguments, .. } => {
                Some((id.as_str(), name.as_str(), arguments.as_str()))
            }
            _ => None,
        })
    }

    /// Concatenated `Thinking` content, if any.
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
    /// Reasoning content. `signature` is required by Anthropic when echoing
    /// the message back in a multi-turn conversation that involves tools.
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// A tool invocation requested by the model.
    ///
    /// `arguments` is the **raw** JSON-encoded string emitted by the vendor.
    /// Streaming deltas arrive as suffixes that concatenate to this final form;
    /// storing the parsed `Value` would lose that property. Use
    /// [`AssistantBlock::parsed_arguments`] to materialise a `serde_json::Value`.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache: Option<CacheMarker>,
    },
}

impl AssistantBlock {
    /// `Some(Ok(v))` for a tool call with valid JSON arguments,
    /// `Some(Err(_))` if the arguments string is malformed JSON,
    /// `None` for non-tool blocks.
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
