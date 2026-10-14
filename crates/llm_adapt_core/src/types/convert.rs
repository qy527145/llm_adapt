//! 角色感知层（[`crate::types`]）与 IR（[`crate::protocol::ir`]）的互转。
//!
//! * [`lift_request`] —— ergonomic → IR。**恒无损。**
//! * [`lower_response`] —— IR → ergonomic。**有损，且是设计如此**：角色感知层
//!   表达不了 [`Block::Unmodeled`] / [`Block::RedactedThinking`]，也表达不了服务端
//!   工具的活动块。被丢掉的块会记一条 `debug` 日志，不会静默消失。
//!
//! 需要无损的调用方直接用 IR（`LLMClient` 的 `*_ir` 系列），或者自己调 codec。

use std::collections::HashMap;

use crate::protocol::ir::{
    Block, ImageSource, Role, ThinkingConfig, ToolKind, ToolSpec, WireMessage, WireRequest,
    WireResponse,
};

use super::conversation::{AssistantBlock, AssistantMessage, SystemBlock, SystemPrompt, Turn, UserBlock};
use super::request::ChatRequest;
use super::response::ChatResponse;
use super::tool::ToolDefinition;

// ---------------------------------------------------------------------------
// lift：ergonomic → IR
// ---------------------------------------------------------------------------

/// 把一个角色感知的请求摊平成 IR。恒无损。
pub fn lift_request(req: &ChatRequest) -> WireRequest {
    let mut out = WireRequest::new(req.model.clone());
    out.stream = req.stream;
    out.system = req
        .conversation
        .system
        .as_ref()
        .map(lift_system)
        .unwrap_or_default();
    out.messages = req.conversation.turns.iter().map(lift_turn).collect();
    out.tools = req.tools.iter().map(lift_tool).collect();
    out.tool_choice = req.tool_choice.clone();
    out.temperature = req.temperature;
    out.top_p = req.top_p;
    out.max_tokens = req.max_tokens;
    out.seed = req.seed;
    out.stop = req.stop_sequences.clone();
    out.response_format = req.response_format.clone();
    // 布尔开关 + 可选预算 → 可选配置。关掉时必须是 None，否则会在 wire 上多出
    // 一个「thinking: 关」的字段，某些上游会因此报错。
    out.thinking = req.enable_thinking.then(|| ThinkingConfig {
        budget_tokens: req.thinking_budget,
        effort: None,
    });
    out.user_id = req.user_id.clone();
    out
}

fn lift_system(system: &SystemPrompt) -> Vec<Block> {
    system
        .blocks
        .iter()
        .map(|b| match b {
            SystemBlock::Text { text, cache } => Block::Text {
                text: text.clone(),
                cache: *cache,
            },
        })
        .collect()
}

fn lift_turn(turn: &Turn) -> WireMessage {
    match turn {
        Turn::User(msg) => WireMessage::new(
            Role::User,
            msg.blocks.iter().map(lift_user_block).collect(),
        ),
        Turn::Assistant(msg) => WireMessage::new(
            Role::Assistant,
            msg.blocks.iter().map(lift_assistant_block).collect(),
        ),
    }
}

fn lift_user_block(b: &UserBlock) -> Block {
    match b {
        UserBlock::Text { text, cache } => Block::Text {
            text: text.clone(),
            cache: *cache,
        },
        UserBlock::Image { url, detail, cache } => Block::Image {
            // `data:` URL 在这里就被拆出来，之后内部只流转结构化的来源。
            source: ImageSource::from_url(url),
            detail: detail.clone(),
            cache: *cache,
        },
        UserBlock::ToolResult {
            call_id,
            content,
            is_error,
            cache,
        } => Block::ToolResult {
            tool_use_id: call_id.clone(),
            content: content.clone(),
            is_error: *is_error,
            cache: *cache,
        },
    }
}

fn lift_assistant_block(b: &AssistantBlock) -> Block {
    match b {
        AssistantBlock::Text { text, cache } => Block::Text {
            text: text.clone(),
            cache: *cache,
        },
        AssistantBlock::Thinking { text, signature } => Block::Thinking {
            text: text.clone(),
            signature: signature.clone(),
        },
        AssistantBlock::ToolCall {
            id,
            name,
            arguments,
            cache,
        } => Block::ToolUse {
            id: id.clone(),
            name: name.clone(),
            // `arguments` 是上游流式拼出来的原始 JSON 字符串；这里解析成结构，
            // 解析失败退化成空对象 —— 与各 codec 原有的容错行为一致。
            input: serde_json::from_str(arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
            cache: *cache,
        },
    }
}

fn lift_tool(t: &ToolDefinition) -> ToolSpec {
    ToolSpec {
        name: t.name.clone(),
        description: Some(t.description.clone()),
        input_schema: t.parameters.clone(),
        cache: t.cache,
        kind: ToolKind::Function,
    }
}

// ---------------------------------------------------------------------------
// lower：IR → ergonomic
// ---------------------------------------------------------------------------

/// 把 IR 响应压成角色感知的响应。表达不了的块会被丢弃（记 debug 日志）。
pub fn lower_response(resp: WireResponse) -> ChatResponse {
    let mut blocks = Vec::new();
    for b in resp.content {
        if let Some(ab) = lower_block(b) {
            blocks.push(ab);
        }
    }
    // 未建模的顶层字段（如 `system_fingerprint`）落到 metadata，不丢。
    let metadata: HashMap<String, serde_json::Value> = resp.extra.into_iter().collect();
    ChatResponse {
        message: AssistantMessage { blocks },
        finish_reason: resp.finish_reason,
        usage: resp.usage,
        request_id: resp.id,
        actual_model: resp.model,
        latency_ms: 0,
        rate_limit_info: Default::default(),
        metadata,
    }
}

fn lower_block(b: Block) -> Option<AssistantBlock> {
    match b {
        Block::Text { text, .. } => Some(AssistantBlock::Text { text, cache: None }),
        Block::Thinking { text, signature } => {
            Some(AssistantBlock::Thinking { text, signature })
        }
        Block::ToolUse {
            id, name, input, ..
        } => Some(AssistantBlock::ToolCall {
            id,
            name,
            arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
            cache: None,
        }),
        other => {
            tracing::debug!(
                target: "llm_adapt::convert",
                "角色感知层表达不了 {} 块，lower 时丢弃（IR 路径仍保有它）",
                block_kind(&other)
            );
            None
        }
    }
}

fn block_kind(b: &Block) -> &'static str {
    match b {
        Block::Text { .. } => "text",
        Block::Image { .. } => "image",
        Block::ToolUse { .. } => "tool_use",
        Block::ToolResult { .. } => "tool_result",
        Block::Thinking { .. } => "thinking",
        Block::RedactedThinking { .. } => "redacted_thinking",
        Block::Unmodeled { .. } => "unmodeled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ir::{CacheMarker, FinishReason, ToolResultBlock, ToolResultContent};
    use crate::types::{Conversation, SystemPrompt, UserMessage};

    #[test]
    fn lift_round_trips_system_cache_and_tools() {
        let mut conv = Conversation::single_user("hi");
        conv.system = Some(SystemPrompt::text("sys").with_cache(CacheMarker::ephemeral_1h()));
        let mut req = ChatRequest::anthropic("claude", conv);
        req.max_tokens = Some(64);
        req.seed = Some(7);
        req.stop_sequences = vec!["END".into()];
        req.enable_thinking = true;
        req.thinking_budget = Some(2048);
        req.tools = vec![ToolDefinition::new("f", "d", serde_json::json!({"type": "object"}))];

        let ir = lift_request(&req);
        assert_eq!(ir.model, "claude");
        assert_eq!(ir.max_tokens, Some(64));
        assert_eq!(ir.seed, Some(7));
        assert_eq!(ir.stop, vec!["END".to_string()]);
        assert_eq!(ir.system.len(), 1);
        assert_eq!(ir.system[0], Block::text_cached("sys", CacheMarker::ephemeral_1h()));
        assert_eq!(ir.tools.len(), 1);
        assert!(ir.tools[0].is_function());
        assert_eq!(
            ir.thinking,
            Some(ThinkingConfig { budget_tokens: Some(2048), effort: None })
        );
    }

    #[test]
    fn thinking_disabled_lifts_to_none() {
        let req = ChatRequest::anthropic("claude", Conversation::single_user("hi"));
        assert_eq!(lift_request(&req).thinking, None);
    }

    #[test]
    fn lift_splits_data_url_into_structured_source() {
        let mut conv = Conversation::single_user("look");
        let mut user = UserMessage::default();
        user.push(UserBlock::Image {
            url: "data:image/png;base64,AAAB".into(),
            detail: Some("high".into()),
            cache: None,
        });
        conv.push(Turn::User(user));
        let req = ChatRequest::openai("gpt", conv);
        let ir = lift_request(&req);
        let img = ir.messages[1]
            .content
            .iter()
            .find(|b| matches!(b, Block::Image { .. }))
            .unwrap();
        match img {
            Block::Image { source, detail, .. } => {
                assert_eq!(
                    *source,
                    ImageSource::Base64 { media_type: "image/png".into(), data: "AAAB".into() }
                );
                assert_eq!(detail.as_deref(), Some("high"));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn lift_keeps_tool_result_shape_unormalized() {
        let mut conv = Conversation::single_user("ok");
        let mut user = UserMessage::default();
        // 裸字符串形态
        user.push(UserBlock::ToolResult {
            call_id: "t1".into(),
            content: ToolResultContent::Text("plain".into()),
            is_error: true,
            cache: None,
        });
        // 块数组形态
        user.push(UserBlock::ToolResult {
            call_id: "t2".into(),
            content: ToolResultContent::Blocks(vec![ToolResultBlock::Text { text: "x".into() }]),
            is_error: false,
            cache: None,
        });
        conv.push(Turn::User(user));
        let ir = lift_request(&ChatRequest::anthropic("c", conv));

        let shapes: Vec<&ToolResultContent> = ir.messages[1]
            .content
            .iter()
            .filter_map(|b| match b {
                Block::ToolResult { content, .. } => Some(content),
                _ => None,
            })
            .collect();
        assert!(matches!(shapes[0], ToolResultContent::Text(_)), "裸字符串形态必须保留");
        assert!(matches!(shapes[1], ToolResultContent::Blocks(_)), "块数组形态必须保留");
    }

    #[test]
    fn lower_maps_blocks_and_keeps_unknown_fields_as_metadata() {
        let mut extra = indexmap::IndexMap::new();
        extra.insert("system_fingerprint".to_string(), serde_json::json!("fp_1"));
        let ir = WireResponse {
            id: Some("r1".into()),
            model: Some("gpt".into()),
            content: vec![
                Block::Thinking { text: "hmm".into(), signature: Some("sig".into()) },
                Block::text("hello"),
                Block::ToolUse {
                    id: "t1".into(),
                    name: "f".into(),
                    input: serde_json::json!({"a": 1}),
                    cache: None,
                },
                Block::Unmodeled { raw: serde_json::json!({"type": "web_search_call"}) },
            ],
            finish_reason: FinishReason::ToolCall,
            usage: Default::default(),
            extra,
        };
        let resp = lower_response(ir);
        assert_eq!(resp.text(), "hello");
        assert_eq!(resp.finish_reason, FinishReason::ToolCall);
        assert_eq!(resp.request_id.as_deref(), Some("r1"));
        assert_eq!(resp.actual_model.as_deref(), Some("gpt"));
        assert_eq!(
            resp.metadata.get("system_fingerprint"),
            Some(&serde_json::json!("fp_1"))
        );
        // Unmodeled 被丢弃，其它三个保留
        assert_eq!(resp.message.blocks.len(), 3);
        let calls: Vec<_> = resp.message.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, "{\"a\":1}");
    }
}
