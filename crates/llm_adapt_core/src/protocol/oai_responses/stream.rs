//! OpenAI Responses 的 SSE 映射。
//!
//! Responses 的每个事件都带 `event:` 行，data 里也有 `type`，两者取其一即可。
//! 它没有 `[DONE]`，终止靠 `response.completed`（或 `response.failed` / `error`）。
//!
//! 与 Chat 不同，Responses **原生标注块边界**：每个 item 有 `output_index`，
//! `output_item.added` / `output_item.done` 天然对应 `BlockStart` / `BlockStop`。

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

use crate::error::LLMError;
use crate::protocol::ir::{Block, FinishReason};
use crate::sse::{SseFrame, SseMapper, SseOutcome};
use crate::types::StreamChunk;

use super::decode_usage;

#[derive(Debug, Default)]
pub struct ResponsesSseMapper {
    started: bool,
    /// `output_index` → 已发过 `BlockStart`。
    opened: HashSet<u64>,
    /// `output_index` → `call_id`（arguments 分片只带 index，得自己补 id）。
    tool_ids: HashMap<u64, String>,
    /// `item_id` → `output_index`（有些事件只给 item_id）。
    id_to_index: HashMap<String, u64>,
    /// `output_index` → 已从分片拼出的 arguments。
    tool_args: HashMap<u64, String>,
    saw_tool_call: bool,
    finish_reason: Option<FinishReason>,
    finished: bool,
}

impl ResponsesSseMapper {
    fn ensure_started(&mut self, id: &str, model: &str, out: &mut Vec<StreamChunk>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(StreamChunk::MessageStart {
            id: id.to_string(),
            model: model.to_string(),
        });
    }

    /// 把一个 item 开成内容块。认不出的 item 类型整体进 `Unmodeled`——
    /// 服务端工具的活动（`web_search_call` 等）就靠这条路径露出来。
    fn open_item(&mut self, output_index: u64, item: &Value, out: &mut Vec<StreamChunk>) {
        if !self.opened.insert(output_index) {
            return;
        }
        let idx = output_index as u32;
        if let Some(item_id) = item.get("id").and_then(Value::as_str) {
            self.id_to_index.insert(item_id.to_string(), output_index);
        }
        let block = match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "message" => Block::text(""),
            "reasoning" => Block::Thinking {
                text: String::new(),
                // encrypted_content 在 added 事件里通常就带了；不带也没关系，
                // `output_item.done` 或非流式路径会补上。
                signature: item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            },
            "function_call" => {
                self.saw_tool_call = true;
                let call_id = super::call_id_of(item);
                self.tool_ids.insert(output_index, call_id.clone());
                self.tool_args.entry(output_index).or_default();
                Block::ToolUse {
                    id: call_id,
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input: json!({}),
                    cache: None,
                }
            }
            _ => Block::Unmodeled { raw: item.clone() },
        };
        out.push(StreamChunk::BlockStart { index: idx, block });
    }

    /// `output_index`：优先用事件自带的，退化到 `item_id` 反查。
    fn resolve_index(&self, v: &Value) -> u64 {
        v.get("output_index")
            .and_then(Value::as_u64)
            .or_else(|| {
                v.get("item_id")
                    .and_then(Value::as_str)
                    .and_then(|id| self.id_to_index.get(id).copied())
            })
            .unwrap_or(0)
    }

    fn on_event(&mut self, v: &Value, out: &mut Vec<StreamChunk>) {
        let name = v.get("type").and_then(Value::as_str).unwrap_or("");
        match name {
            "response.created" | "response.in_progress" => {
                if let Some(r) = v.get("response") {
                    let id = r.get("id").and_then(Value::as_str).unwrap_or_default();
                    let model = r.get("model").and_then(Value::as_str).unwrap_or_default();
                    self.ensure_started(id, model, out);
                }
            }

            "response.output_item.added" => {
                let oi = v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                self.open_item(oi, v.get("item").unwrap_or(&Value::Null), out);
            }

            "response.output_text.delta" => {
                let oi = self.resolve_index(v);
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or_default();
                if !delta.is_empty() {
                    out.push(StreamChunk::TextDelta {
                        index: oi as u32,
                        text: delta.to_string(),
                    });
                }
            }

            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let oi = self.resolve_index(v);
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or_default();
                if !delta.is_empty() {
                    out.push(StreamChunk::ThinkingDelta {
                        index: oi as u32,
                        text: delta.to_string(),
                        signature_delta: None,
                    });
                }
            }

            "response.function_call_arguments.delta" => {
                let oi = self.resolve_index(v);
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or_default();
                if !delta.is_empty() {
                    self.tool_args.entry(oi).or_default().push_str(delta);
                    out.push(StreamChunk::ToolCallDelta {
                        index: oi as u32,
                        id: self.tool_ids.get(&oi).cloned().unwrap_or_default(),
                        name_delta: None,
                        arguments_delta: Some(delta.to_string()),
                    });
                }
            }

            "response.output_item.done" => {
                let oi = v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                let item = v.get("item").unwrap_or(&Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    self.saw_tool_call = true;
                    let full = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    // 上游理论上一定会先发 arguments 分片；万一漏了，这里用完整
                    // item 补一次，否则工具会拿到空参数。
                    let accumulated = self.tool_args.entry(oi).or_default();
                    if accumulated.is_empty() && !full.is_empty() {
                        accumulated.push_str(full);
                        out.push(StreamChunk::ToolCallDelta {
                            index: oi as u32,
                            id: super::call_id_of(item),
                            name_delta: None,
                            arguments_delta: Some(full.to_string()),
                        });
                    }
                }
                if self.opened.contains(&oi) {
                    out.push(StreamChunk::BlockStop { index: oi as u32 });
                }
            }

            "response.completed" | "response.incomplete" => {
                if let Some(r) = v.get("response") {
                    self.ensure_started(
                        r.get("id").and_then(Value::as_str).unwrap_or_default(),
                        r.get("model").and_then(Value::as_str).unwrap_or_default(),
                        out,
                    );
                    let usage = decode_usage(r.get("usage"));
                    if !usage.is_empty() {
                        out.push(StreamChunk::Usage { usage });
                    }
                    let reason = match r.get("status").and_then(Value::as_str) {
                        Some("incomplete") => FinishReason::Length,
                        _ if self.saw_tool_call => FinishReason::ToolCall,
                        _ => FinishReason::Stop,
                    };
                    self.finish_reason = Some(reason.clone());
                    out.push(StreamChunk::Finish { reason });
                }
                self.finished = true;
            }

            "response.failed" => {
                self.finished = true;
                let msg = v
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Responses 请求失败");
                out.push(StreamChunk::Error {
                    error: msg.to_string(),
                });
            }

            "error" => {
                self.finished = true;
                out.push(StreamChunk::Error {
                    error: v
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("上游返回未知错误")
                        .to_string(),
                });
            }

            _ => {}
        }
    }
}

impl SseMapper for ResponsesSseMapper {
    fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError> {
        let v: Value = serde_json::from_str(&frame.data).map_err(|e| LLMError::Parse {
            message: format!("openai_responses stream: invalid JSON: {e}"),
            ctx: Default::default(),
        })?;
        let mut out = Vec::new();
        self.on_event(&v, &mut out);
        let terminal = self.finished;
        Ok(SseOutcome {
            chunks: out,
            terminal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(mapper: &mut ResponsesSseMapper, data: &str) -> Vec<StreamChunk> {
        let frame = SseFrame {
            event: None,
            data: data.to_string(),
        };
        mapper.map(frame).expect("mapper 不应报错").chunks
    }

    #[test]
    fn completed_is_terminal() {
        let mut m = ResponsesSseMapper::default();
        let chunks = feed(
            &mut m,
            r#"{"type":"response.completed","response":{"id":"r","model":"gpt-5","status":"completed"}}"#,
        );
        assert!(matches!(chunks.last(), Some(StreamChunk::Finish { .. })));
        let out = m
            .map(SseFrame {
                event: None,
                data: r#"{"type":"response.completed","response":{}}"#.into(),
            })
            .unwrap();
        assert!(out.terminal);
    }

    #[test]
    fn function_call_arguments_fall_back_to_the_done_item() {
        // 上游只发了 added + done，没有 arguments 分片 —— 必须用完整 item 补一次，
        // 否则工具拿到空参数。
        let mut m = ResponsesSseMapper::default();
        feed(
            &mut m,
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_0","call_id":"call_1","name":"f","arguments":""}}"#,
        );
        let chunks = feed(
            &mut m,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_0","call_id":"call_1","name":"f","arguments":"{\"a\":1}"}}"#,
        );
        let deltas: Vec<&StreamChunk> = chunks
            .iter()
            .filter(|c| matches!(c, StreamChunk::ToolCallDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 1, "应补发一次完整参数");
        match deltas[0] {
            StreamChunk::ToolCallDelta {
                arguments_delta, ..
            } => assert_eq!(arguments_delta.as_deref(), Some(r#"{"a":1}"#)),
            _ => unreachable!(),
        }
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::BlockStop { .. })));
    }

    #[test]
    fn item_id_resolves_the_output_index() {
        let mut m = ResponsesSseMapper::default();
        feed(
            &mut m,
            r#"{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"fc_3","call_id":"c3","name":"f","arguments":""}}"#,
        );
        let chunks = feed(
            &mut m,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_3","delta":"{}"}"#,
        );
        match &chunks[0] {
            StreamChunk::ToolCallDelta { index, id, .. } => {
                assert_eq!(*index, 3, "只给 item_id 时应反查到 output_index");
                assert_eq!(id, "c3");
            }
            other => panic!("期望 ToolCallDelta，得到 {other:?}"),
        }
    }

    /// 空 usage 不该塞一个全零的 Usage 事件出去 ——
    /// `merge_non_zero` 会把它合并成噪音。
    #[test]
    fn empty_usage_is_not_emitted() {
        let mut m = ResponsesSseMapper::default();
        let chunks = feed(
            &mut m,
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        );
        assert!(!chunks.iter().any(|c| matches!(c, StreamChunk::Usage { .. })));
    }
}
