//! 流式事件表示。
//!
//! 这是**唯一的**流式词汇 —— 客户端与网关共用它，不另设「IR 流」再投影一层。
//! 多一层投影的代价是：下层的 `lower()` 必须逐条复刻上层实现的怪癖才能保持行为
//! 一致，脆且难维护。宁可让这一个枚举自己带上块边界与序号。
//!
//! ## 块边界为什么必须有
//!
//! 流式增量是碎片。只按「文本增量 / 工具增量」拼接会丢掉**块与块的边界**与**顺序**：
//! 同一轮里模型可能先思考、再说话、再调工具，回灌历史时这个顺序是硬要求
//! （Anthropic 要求 thinking 块排在 assistant 消息最前，且签名原样回传）。
//! 因此每个增量都带 `index`，块的开始/结束由 [`StreamChunk::BlockStart`] /
//! [`StreamChunk::BlockStop`] 显式标注；不原生标注块边界的协议（如 Chat Completions）
//! 由各自的 mapper 合成。
//!
//! 把增量组装回有序内容块用 [`BlockAssembler`]。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::protocol::ir::{Block, FinishReason, Usage};

/// 一个流式事件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    /// 流开始，携带上游给的 message id 与 model（有的协议不给，就不发这一条）。
    MessageStart { id: String, model: String },
    /// 一个新内容块开始。`block` 带类型与初始内容（文本块此时通常是空的）。
    BlockStart { index: u32, block: Block },
    /// 文本增量。
    TextDelta { index: u32, text: String },
    /// 思考增量。
    ///
    /// `signature_delta` 在厂商对思考内容签名时出现（Anthropic）：把全部
    /// `signature_delta` 片段拼起来就是完整签名。签名必须原样回灌，否则多轮带工具的
    /// 对话会被上游拒掉。
    ThinkingDelta {
        index: u32,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature_delta: Option<String>,
    },
    /// 工具调用增量。
    ///
    /// 厂商通常在第一片给 `id` 与 `name_delta`，之后只给 `arguments_delta` 分片；
    /// 分片拼接起来才是完整 JSON。`id` 在后续分片上可能由 mapper 合成（见各协议的注解）。
    ToolCallDelta {
        index: u32,
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name_delta: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        arguments_delta: Option<String>,
    },
    /// 内容块结束。
    BlockStop { index: u32 },
    /// 用量上报。多数厂商在临近结束时给一次。
    Usage { usage: Usage },
    /// 流结束。与非流式响应的 `finish_reason` 对应。
    Finish { reason: FinishReason },
    /// 厂商在流内发来的错误帧。
    Error { error: String },
}

impl StreamChunk {
    /// 该增量属于哪个块（块边界事件返回 `None`）。
    pub fn index(&self) -> Option<u32> {
        match self {
            StreamChunk::TextDelta { index, .. }
            | StreamChunk::ThinkingDelta { index, .. }
            | StreamChunk::ToolCallDelta { index, .. } => Some(*index),
            _ => None,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        StreamChunk::TextDelta {
            index: 0,
            text: text.into(),
        }
    }
}

/// 把一个块在流式过程中攒下来的内容。
#[derive(Debug, Clone)]
enum Slot {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    Tool {
        id: String,
        name: String,
        arguments: String,
    },
    /// 由 `BlockStart` 直接给定的整块（未建模块、服务端工具活动等）。
    Opaque(Block),
}

/// 把 [`StreamChunk`] 流装配成有序内容块。
///
/// 按 `index` 归并，输出顺序 = 各 index **首次出现**的顺序，因此块间交错也能保真。
/// 未带 `BlockStart` 就出现的增量会按增量类型自动建块 —— mapper 漏发 BlockStart
/// 不该让整条流报废。
#[derive(Debug, Default)]
pub struct BlockAssembler {
    order: Vec<u32>,
    slots: HashMap<u32, Slot>,
    id: Option<String>,
    model: Option<String>,
    finish_reason: Option<FinishReason>,
    usage: Usage,
    error: Option<String>,
}

impl BlockAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// 吃下一个增量。块边界与用量/结束事件都接受，多余的事件安全忽略。
    pub fn push(&mut self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::MessageStart { id, model } => {
                self.id = Some(id.clone());
                self.model = Some(model.clone());
            }
            StreamChunk::BlockStart { index, block } => {
                // 只在槽位缺席时建块：先到的增量已经建好了带内容的槽位，
                // 用一个空壳 BlockStart 把它覆盖掉就丢内容了。
                //
                // 建的是**按块类型展开的**槽位而不是不透明块 —— 否则后续增量会另起
                // 一个槽位，把 `BlockStart` 带下来的 id / name 丢掉。
                if !self.slots.contains_key(index) {
                    self.order.push(*index);
                    self.slots.insert(*index, slot_from_block(block));
                }
            }
            StreamChunk::TextDelta { index, text } => {
                let slot = self.ensure(*index, || Slot::Text(String::new()));
                if let Slot::Text(acc) = slot {
                    acc.push_str(text);
                } else {
                    self.slots.insert(*index, Slot::Text(text.clone()));
                }
            }
            StreamChunk::ThinkingDelta {
                index,
                text,
                signature_delta,
            } => {
                let slot = self.ensure(*index, || Slot::Thinking {
                    text: String::new(),
                    signature: String::new(),
                });
                match slot {
                    Slot::Thinking {
                        text: acc,
                        signature,
                    } => {
                        acc.push_str(text);
                        if let Some(s) = signature_delta {
                            signature.push_str(s);
                        }
                    }
                    _ => {
                        self.slots.insert(
                            *index,
                            Slot::Thinking {
                                text: text.clone(),
                                signature: signature_delta.clone().unwrap_or_default(),
                            },
                        );
                    }
                }
            }
            StreamChunk::ToolCallDelta {
                index,
                id,
                name_delta,
                arguments_delta,
            } => {
                let slot = self.ensure(*index, || Slot::Tool {
                    id: id.clone(),
                    name: String::new(),
                    arguments: String::new(),
                });
                match slot {
                    Slot::Tool {
                        id: slot_id,
                        name,
                        arguments,
                    } => {
                        // 首片才有真实 id；后续分片 mapper 会合成一个稳定的占位 id，
                        // 不能让它覆盖掉真实 id。
                        if slot_id.is_empty() {
                            *slot_id = id.clone();
                        }
                        if let Some(n) = name_delta {
                            name.push_str(n);
                        }
                        if let Some(a) = arguments_delta {
                            arguments.push_str(a);
                        }
                    }
                    _ => {
                        self.slots.insert(
                            *index,
                            Slot::Tool {
                                id: id.clone(),
                                name: name_delta.clone().unwrap_or_default(),
                                arguments: arguments_delta.clone().unwrap_or_default(),
                            },
                        );
                    }
                }
            }
            StreamChunk::BlockStop { .. } => {}
            StreamChunk::Usage { usage } => self.usage.merge_non_zero(usage),
            StreamChunk::Finish { reason } => self.finish_reason = Some(reason.clone()),
            StreamChunk::Error { error } => self.error = Some(error.clone()),
        }
    }

    pub fn extend<'a>(&mut self, chunks: impl IntoIterator<Item = &'a StreamChunk>) {
        for c in chunks {
            self.push(c);
        }
    }

    fn ensure(&mut self, index: u32, make: impl FnOnce() -> Slot) -> &mut Slot {
        if !self.slots.contains_key(&index) {
            self.order.push(index);
            self.slots.insert(index, make());
        }
        self.slots.get_mut(&index).expect("刚插入的槽位")
    }

    /// 当前已装配出的有序内容块。
    pub fn blocks(&self) -> Vec<Block> {
        self.order
            .iter()
            .filter_map(|i| self.slots.get(i))
            .filter_map(slot_to_block)
            .collect()
    }

    /// 收尾并取出内容块。
    pub fn into_blocks(self) -> Vec<Block> {
        self.blocks()
    }

    pub fn message_id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

fn slot_from_block(block: &Block) -> Slot {
    match block {
        Block::Text { text, .. } => Slot::Text(text.clone()),
        Block::Thinking { text, signature } => Slot::Thinking {
            text: text.clone(),
            signature: signature.clone().unwrap_or_default(),
        },
        Block::ToolUse {
            id, name, input, ..
        } => Slot::Tool {
            id: id.clone(),
            name: name.clone(),
            // `BlockStart` 里的 input 通常是空对象（入参靠后续分片拼），
            // 空对象还原成空串，避免拼出 `{}{"a":1}` 这种坏 JSON。
            arguments: if input.is_null() || input == &serde_json::json!({}) {
                String::new()
            } else {
                input.to_string()
            },
        },
        // 未建模块 / 服务端工具活动：整块原样保留，不参与累加。
        other => Slot::Opaque(other.clone()),
    }
}

fn slot_to_block(slot: &Slot) -> Option<Block> {
    match slot {
        Slot::Text(text) => {
            if text.is_empty() {
                None
            } else {
                Some(Block::Text {
                    text: text.clone(),
                    cache: None,
                })
            }
        }
        Slot::Thinking { text, signature } => {
            if text.is_empty() && signature.is_empty() {
                return None;
            }
            Some(Block::Thinking {
                text: text.clone(),
                signature: if signature.is_empty() {
                    None
                } else {
                    Some(signature.clone())
                },
            })
        }
        Slot::Tool {
            id,
            name,
            arguments,
        } => Some(Block::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: parse_arguments(arguments),
            cache: None,
        }),
        Slot::Opaque(block) => Some(block.clone()),
    }
}

/// 分片拼起来的入参可能不合法（流被截断）。解析失败退化成空对象并告警，
/// 不因为一个工具入参让整轮对话崩掉。
fn parse_arguments(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::json!({});
    }
    match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(target: "llm_adapt::stream", "工具入参不是合法 JSON（{e}），已退回空对象");
            serde_json::json!({})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_at(index: u32, text: &str) -> StreamChunk {
        StreamChunk::TextDelta {
            index,
            text: text.to_string(),
        }
    }

    #[test]
    fn assembles_interleaved_blocks_in_first_appearance_order() {
        let mut a = BlockAssembler::new();
        a.push(&StreamChunk::MessageStart {
            id: "m1".into(),
            model: "gpt".into(),
        });
        // 思考 → 文本 → 工具，各自多个分片交错到达
        a.push(&StreamChunk::BlockStart {
            index: 0,
            block: Block::Thinking {
                text: String::new(),
                signature: None,
            },
        });
        a.push(&StreamChunk::BlockStart {
            index: 1,
            block: Block::text(""),
        });
        a.push(&StreamChunk::ThinkingDelta {
            index: 0,
            text: "hmm".into(),
            signature_delta: Some("sig".into()),
        });
        a.push(&text_at(1, "he"));
        a.push(&StreamChunk::ToolCallDelta {
            index: 2,
            id: "t1".into(),
            name_delta: Some("f".into()),
            arguments_delta: Some("{\"a\":".into()),
        });
        a.push(&text_at(1, "llo"));
        a.push(&StreamChunk::ToolCallDelta {
            index: 2,
            id: "t1".into(),
            name_delta: None,
            arguments_delta: Some("1}".into()),
        });
        a.push(&StreamChunk::Finish {
            reason: FinishReason::ToolCall,
        });

        let blocks = a.blocks();
        assert_eq!(blocks.len(), 3);
        match &blocks[0] {
            Block::Thinking { text, signature } => {
                assert_eq!(text, "hmm");
                assert_eq!(signature.as_deref(), Some("sig"));
            }
            other => panic!("0 应为 thinking，得到 {other:?}"),
        }
        assert_eq!(blocks[1], Block::text("hello"), "分片应按 index 归并");
        match &blocks[2] {
            Block::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "t1");
                assert_eq!(name, "f");
                assert_eq!(input["a"], 1);
            }
            other => panic!("2 应为 tool_use，得到 {other:?}"),
        }
        assert_eq!(a.message_id(), Some("m1"));
        assert_eq!(a.finish_reason(), Some(&FinishReason::ToolCall));
    }

    /// 后续分片带的合成占位 id 不能覆盖首片的真实 id。
    #[test]
    fn synthetic_continuation_ids_do_not_clobber_the_real_one() {
        let mut a = BlockAssembler::new();
        a.push(&StreamChunk::ToolCallDelta {
            index: 0,
            id: "call_real".into(),
            name_delta: Some("f".into()),
            arguments_delta: Some("{".into()),
        });
        a.push(&StreamChunk::ToolCallDelta {
            index: 0,
            id: "tc_idx_0".into(),
            name_delta: None,
            arguments_delta: Some("}".into()),
        });
        match &a.blocks()[0] {
            Block::ToolUse { id, .. } => assert_eq!(id, "call_real"),
            other => panic!("期望 tool_use，得到 {other:?}"),
        }
    }

    /// mapper 漏发 BlockStart 不该让整条流报废。
    #[test]
    fn deltas_without_block_start_still_assemble() {
        let mut a = BlockAssembler::new();
        a.push(&text_at(0, "hi"));
        assert_eq!(a.blocks(), vec![Block::text("hi")]);
    }

    /// 流式增量也必须能过 serde —— 它要跨 IPC 送到 UI、也要进流式时间轴。
    /// 和请求/响应那边同一个道理：serde 表示不合法的块编译期看不出来，
    /// 只有真去序列化才炸。
    #[test]
    fn every_chunk_variant_round_trips() {
        let chunks = vec![
            StreamChunk::MessageStart {
                id: "m".into(),
                model: "gpt".into(),
            },
            StreamChunk::BlockStart {
                index: 0,
                block: Block::text(""),
            },
            StreamChunk::BlockStart {
                index: 1,
                block: Block::Thinking {
                    text: String::new(),
                    signature: Some("s".into()),
                },
            },
            StreamChunk::BlockStart {
                index: 2,
                block: Block::ToolUse {
                    id: "t".into(),
                    name: "f".into(),
                    input: serde_json::json!({}),
                    cache: None,
                },
            },
            StreamChunk::BlockStart {
                index: 3,
                block: Block::Unmodeled {
                    raw: serde_json::json!({"type": "web_search_call"}),
                },
            },
            StreamChunk::TextDelta {
                index: 0,
                text: "hi".into(),
            },
            StreamChunk::ThinkingDelta {
                index: 1,
                text: "hmm".into(),
                signature_delta: Some("sig".into()),
            },
            StreamChunk::ToolCallDelta {
                index: 2,
                id: "t".into(),
                name_delta: Some("f".into()),
                arguments_delta: Some("{}".into()),
            },
            StreamChunk::BlockStop { index: 2 },
            StreamChunk::Usage {
                usage: Usage::default(),
            },
            StreamChunk::Finish {
                reason: FinishReason::Stop,
            },
            StreamChunk::Error {
                error: "boom".into(),
            },
        ];
        for c in chunks {
            let json = serde_json::to_string(&c).expect("每个增量变体都必须能序列化");
            let back: StreamChunk = serde_json::from_str(&json).expect("必须能反序列化");
            assert_eq!(back, c, "往返后应一模一样：{json}");
        }
    }

    #[test]
    fn usage_merges_without_zero_overwrite() {
        let mut a = BlockAssembler::new();
        a.push(&StreamChunk::Usage {
            usage: Usage {
                input_tokens: 10,
                ..Default::default()
            },
        });
        a.push(&StreamChunk::Usage {
            usage: Usage {
                output_tokens: 3,
                ..Default::default()
            },
        });
        assert_eq!(a.usage().input_tokens, 10);
        assert_eq!(a.usage().output_tokens, 3);
    }

    #[test]
    fn empty_text_blocks_are_dropped() {
        let mut a = BlockAssembler::new();
        a.push(&StreamChunk::BlockStart {
            index: 0,
            block: Block::text(""),
        });
        assert!(a.blocks().is_empty(), "空文本块不该产出内容");
    }
}
