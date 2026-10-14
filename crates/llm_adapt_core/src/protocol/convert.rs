//! 跨协议转换器（网关方向）。
//!
//! 任意两种协议之间的转换都是 `decode(A) → IR → encode(B)`，不需要两两配对实现。
//! 九种组合（含三种同协议）全部由这里调度。
//!
//! ## 同协议短路：字节不变，而不是「解出来再编回去」
//!
//! 同协议时请求/响应转换直接返回入参字节，**不**走 IR。理由很实际：经 IR 一来一回会
//! 丢掉 IR 没建模的字段（顶层字段、块内字段如 `cache_control`）。直通时客户端拿到的
//! 是自己那份报文，一个字节都不该变。apilot 是先按 IR 往返、后来才补上这条短路的，
//! 代价是一次真金白银的字段丢失。
//!
//! ## 思考块的剥离
//!
//! Anthropic 的 `Thinking.signature` 由 Anthropic 侧签名，跨协议无法保真；把没有签名的
//! 思考块回传给 Anthropic 又会被拒。因此**只在跨协议**时调 [`strip_unportable_thinking`]：
//! 同协议直通保留签名（这正是 Claude Code 走 Anthropic 渠道必须直通的原因之一）。

use bytes::Bytes;

use crate::error::LLMError;
use crate::http::ClientConfig;
use crate::protocol::codec::{ProtocolRegistry, StreamEncoder};
use crate::protocol::ir::{Block, Protocol, ProtocolId, WireRequest};
use crate::sse::{SseFrame, SseMapper, find_boundary, parse_frame};

/// 跨协议进到 Anthropic 请求时，若原请求没带 `max_tokens` 就补这个值。
///
/// Anthropic 的 `max_tokens` 是必填项，缺了直接 400。Chat / Responses 的请求本来可以
/// 不带（模型自己定上限），转到 Anthropic 就必须给一个 —— 给一个足够大的默认值比让
/// 整次请求失败合理（apilot 同样处理）。
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 8192;

/// 只用于取 body 的占位配置：转换只关心请求体，URL / 密钥由调用方在真正出站时决定。
///
/// `base_url` 必须非空 —— 各 codec 的 `encode_request` 会据此校验（空 base_url 是配置
/// 错误），但那条校验在网关转换语境下没有意义，给一个占位地址即可。
fn gateway_config() -> ClientConfig {
    ClientConfig::new("http://gateway.invalid", "")
}

/// 跨协议转换的总入口：持有协议注册表，按 `from` / `to` 两个 [`ProtocolId`] 调度。
#[derive(Clone)]
pub struct ProtocolConverter {
    registry: ProtocolRegistry,
}

impl ProtocolConverter {
    pub fn new(registry: ProtocolRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &ProtocolRegistry {
        &self.registry
    }

    /// 请求方向：`decode(from) → IR → encode(to)`。同协议短路，**字节不变**。
    pub fn convert_request(
        &self,
        from: &ProtocolId,
        to: &ProtocolId,
        body: &[u8],
    ) -> Result<Vec<u8>, LLMError> {
        if from == to {
            return Ok(body.to_vec());
        }
        let decode_codec = self.registry.get(from)?;
        let mut req = decode_codec.decode_request(body)?;
        // 只在跨协议时剥离不可移植的思考块。同协议已在上面短路返回。
        if let Some(target) = to.protocol() {
            strip_unportable_thinking(&mut req, target);
        }
        self.encode_request_body(to, &req)
    }

    /// 响应方向：`decode(from) → IR → encode(to)`。同协议短路，**字节不变**。
    pub fn convert_response(
        &self,
        from: &ProtocolId,
        to: &ProtocolId,
        body: &[u8],
    ) -> Result<Vec<u8>, LLMError> {
        if from == to {
            return Ok(body.to_vec());
        }
        let resp = self
            .registry
            .get(from)?
            .decode_response(Bytes::copy_from_slice(body))?;
        self.registry.get(to)?.encode_response(&resp)
    }

    /// 流式方向：把源协议的 SSE 逐帧转成目标协议的 SSE 帧，返回**已渲染好的帧文本**。
    ///
    /// 同协议短路：整段入参作为**单个**帧原样返回（字节不变）—— 重新切帧再拼回去会
    /// 丢掉分帧边界，某些客户端对此敏感。
    pub fn convert_stream(
        &self,
        from: &ProtocolId,
        to: &ProtocolId,
        body: &[u8],
    ) -> Result<Vec<String>, LLMError> {
        if from == to {
            return Ok(vec![String::from_utf8_lossy(body).into_owned()]);
        }
        let mut conv = self.stream_converter(from, to)?;
        let mut frames = conv.push(body)?;
        frames.extend(conv.finish()?);
        Ok(frames)
    }

    /// 造一个可增量喂字节的流式转换器。源协议要有解码器，目标协议要有编码器。
    pub fn stream_converter(
        &self,
        from: &ProtocolId,
        to: &ProtocolId,
    ) -> Result<StreamConverter, LLMError> {
        let source = self.registry.get(from)?;
        let target = self.registry.get(to)?;
        let encoder = target
            .new_stream_encoder()
            .ok_or_else(|| LLMError::unsupported(format!("{to}: 流式响应编码尚未实现")))?;
        Ok(StreamConverter::new(source.new_stream_decoder(), encoder))
    }

    /// 把 IR 请求编成目标协议的 body 字节。
    ///
    /// 用 `encode_request` 取 body 而不是各协议私有的 body builder：这样自定义协议
    /// （只要实现了 `Codec`）也能参与转换，不必认识每一个内置协议的内部函数。
    fn encode_request_body(
        &self,
        to: &ProtocolId,
        req: &WireRequest,
    ) -> Result<Vec<u8>, LLMError> {
        let codec = self.registry.get(to)?;
        let mut req = req.clone();
        if to.protocol() == Some(Protocol::Anthropic) && req.max_tokens.is_none() {
            req.max_tokens = Some(DEFAULT_ANTHROPIC_MAX_TOKENS);
        }
        Ok(codec.encode_request(&req, &gateway_config())?.body)
    }
}

/// 剥离无法跨协议保真的思考内容。
///
/// * `Thinking` / `RedactedThinking` 一律从消息里剥掉：签名由 Anthropic 侧签发，
///   别的协议装不下它，而把**没有签名**的思考块回传给 Anthropic 会被拒。
/// * 目标不是 Anthropic 时再清掉 `thinking.budget_tokens`：那是 Anthropic 特有的
///   预算参数；反过来（→ Anthropic）不能清 —— 它恰恰是那边唯一能用的思考参数。
///
/// 调用点只在**跨协议**转换上；同协议直通必须原样保留，否则多轮带工具的对话会崩。
pub fn strip_unportable_thinking(req: &mut WireRequest, target: Protocol) {
    for msg in &mut req.messages {
        msg.content.retain(|b| {
            !matches!(
                b,
                Block::Thinking { .. } | Block::RedactedThinking { .. }
            )
        });
    }
    if target != Protocol::Anthropic {
        if let Some(t) = &mut req.thinking {
            t.budget_tokens = None;
        }
    }
}

/// 增量式的流式协议转换器：源协议 SSE 字节进，目标协议 SSE 帧出。
///
/// 内部状态有两处：源解码器的状态（工具 id 关联、已开块等）与目标编码器的状态
/// （已发的 `message_start`、块边界等）。所以它**有状态**，一个流实例只能从一个
/// 方向用一次；不要跨请求复用。
pub struct StreamConverter {
    decoder: Box<dyn SseMapper>,
    encoder: Box<dyn StreamEncoder>,
    /// 尚未凑成完整帧的残留字节（TCP 分片、缺尾分隔符都可能）。
    buffer: Vec<u8>,
    /// 源协议已给出终止帧。
    finished: bool,
}

impl StreamConverter {
    pub fn new(decoder: Box<dyn SseMapper>, encoder: Box<dyn StreamEncoder>) -> Self {
        Self {
            decoder,
            encoder,
            buffer: Vec::new(),
            finished: false,
        }
    }

    /// 喂一段任意切分的源协议字节，返回本次能产出的目标协议帧。
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, LLMError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.buffer.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some((pos, sep)) = find_boundary(&self.buffer) {
            let raw: Vec<u8> = self.buffer.drain(..pos).collect();
            self.buffer.drain(..sep);
            let frame = parse_frame(&raw);
            if frame.data.is_empty() {
                continue;
            }
            self.dispatch(frame, &mut out)?;
            if self.finished {
                break;
            }
        }
        Ok(out)
    }

    /// 源流结束：把残留的半帧也交出去，再让目标编码器收尾（`message_stop` / `[DONE]`）。
    pub fn finish(&mut self) -> Result<Vec<String>, LLMError> {
        let mut out = Vec::new();
        if !self.buffer.is_empty() {
            let raw = std::mem::take(&mut self.buffer);
            let frame = parse_frame(&raw);
            if !frame.data.is_empty() {
                self.dispatch(frame, &mut out)?;
            }
        }
        self.finished = true;
        out.extend(self.encoder.finish()?);
        Ok(out)
    }

    fn dispatch(&mut self, frame: SseFrame, out: &mut Vec<String>) -> Result<(), LLMError> {
        let outcome = self.decoder.map(frame)?;
        for chunk in &outcome.chunks {
            out.extend(self.encoder.encode(chunk)?);
        }
        if outcome.terminal {
            self.finished = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ByteStream;
    use crate::protocol::codec::decode_stream;
    use crate::protocol::codec::Codec;
    use crate::protocol::oai_chat::OpenAiChatCodec;
    use crate::protocol::oai_responses::OpenAiResponsesCodec;
    use crate::protocol::register_defaults;
    use crate::protocol::ir::{Role, ToolKind};
    use crate::types::BlockAssembler;
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::Value;

    fn converter() -> ProtocolConverter {
        let r = ProtocolRegistry::new();
        register_defaults(&r);
        ProtocolConverter::new(r)
    }

    fn pid(s: &str) -> ProtocolId {
        ProtocolId::new(s)
    }

    /// 三种协议的 id，供九方向组合遍历。
    const IDS: [&str; 3] = ["anthropic", "openai_chat", "openai_responses"];

    // ------------------------------------------------------------------
    // 真实报文（含工具调用 / 图片 / 思考 / 未建模块）
    // ------------------------------------------------------------------

    const ANTHROPIC_REQ: &str = r#"{
        "model":"claude-sonnet-5","max_tokens":1024,"stream":false,
        "system":[{"type":"text","text":"be brief"}],
        "messages":[
            {"role":"user","content":[
                {"type":"text","text":"hi"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"hmm","signature":"sig"},
                {"type":"text","text":"let me search"},
                {"type":"tool_use","id":"tu_1","name":"search","input":{"q":"x"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"sunny"}]}
        ],
        "tools":[{"name":"search","description":"d","input_schema":{"type":"object"}}],
        "thinking":{"type":"enabled","budget_tokens":1024},
        "service_tier":"standard_only"
    }"#;

    const CHAT_REQ: &str = r#"{
        "model":"gpt-4o",
        "messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":[
                {"type":"text","text":"hi"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"sunny"}
        ],
        "tools":[{"type":"function","function":{"name":"search","description":"d","parameters":{"type":"object"}}}]
    }"#;

    const RESPONSES_REQ: &str = r#"{
        "model":"gpt-5",
        "input":[
            {"type":"message","role":"system","content":[{"type":"input_text","text":"be brief"}]},
            {"type":"message","role":"user","content":[
                {"type":"input_text","text":"hi"},
                {"type":"input_image","image_url":"data:image/png;base64,AA"}]},
            {"type":"reasoning","summary":[{"type":"summary_text","text":"hmm"}],"encrypted_content":"enc"},
            {"type":"function_call","call_id":"c1","name":"search","arguments":"{\"q\":\"x\"}"},
            {"type":"function_call_output","call_id":"c1","output":"sunny"}
        ],
        "tools":[{"type":"function","name":"search","parameters":{"type":"object"}}]
    }"#;

    const ANTHROPIC_RESP: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude",
        "content":[{"type":"text","text":"done"},
                   {"type":"tool_use","id":"tu_1","name":"f","input":{"a":1}}],
        "stop_reason":"tool_use","stop_sequence":null,
        "usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":3}}"#;

    const CHAT_RESP: &str = r#"{"id":"c1","object":"chat.completion","model":"gpt-4o",
        "choices":[{"index":0,"message":{"role":"assistant","content":"done",
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},
            "finish_reason":"tool_calls"}],
        "usage":{"prompt_tokens":1000,"completion_tokens":5,
                 "prompt_tokens_details":{"cached_tokens":800}}}"#;

    const RESPONSES_RESP: &str = r#"{"id":"r1","object":"response","model":"gpt-5","status":"completed",
        "output":[
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]},
            {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1}"}],
        "usage":{"input_tokens":10,"output_tokens":5}}"#;

    fn body_for(id: &str) -> &'static str {
        match id {
            "anthropic" => ANTHROPIC_REQ,
            "openai_chat" => CHAT_REQ,
            _ => RESPONSES_REQ,
        }
    }

    fn resp_for(id: &str) -> &'static str {
        match id {
            "anthropic" => ANTHROPIC_RESP,
            "openai_chat" => CHAT_RESP,
            _ => RESPONSES_RESP,
        }
    }

    // ------------------------------------------------------------------
    // 3. 同协议转换必须字节一致（不能解出来再编回去）
    // ------------------------------------------------------------------

    #[test]
    fn same_protocol_request_conversion_is_byte_identical() {
        let c = converter();
        for id in IDS {
            let raw = body_for(id);
            let out = c.convert_request(&pid(id), &pid(id), raw.as_bytes()).unwrap();
            assert_eq!(out, raw.as_bytes(), "{id} 同协议请求必须字节不变");
        }
    }

    #[test]
    fn same_protocol_response_conversion_is_byte_identical() {
        let c = converter();
        for id in IDS {
            let raw = resp_for(id);
            let out = c.convert_response(&pid(id), &pid(id), raw.as_bytes()).unwrap();
            assert_eq!(out, raw.as_bytes(), "{id} 同协议响应必须字节不变");
        }
    }

    #[test]
    fn same_protocol_stream_conversion_is_byte_identical() {
        let c = converter();
        let sse = "event: message_start\ndata: {\"type\":\"message_stop\"}\n\ncompletely-unparsed";
        let out = c
            .convert_stream(&pid("anthropic"), &pid("anthropic"), sse.as_bytes())
            .unwrap();
        assert_eq!(out, vec![sse.to_string()]);
    }

    // ------------------------------------------------------------------
    // 2. 九方向都不报错
    // ------------------------------------------------------------------

    #[test]
    fn all_nine_request_directions_convert() {
        let c = converter();
        for from in IDS {
            for to in IDS {
                let out = c
                    .convert_request(&pid(from), &pid(to), body_for(from).as_bytes())
                    .unwrap_or_else(|e| panic!("{from} -> {to} 请求转换失败: {e}"));
                let v: Value = serde_json::from_slice(&out)
                    .unwrap_or_else(|e| panic!("{from} -> {to} 产出不是 JSON: {e}"));
                assert!(v.get("model").is_some(), "{from} -> {to} 丢了 model");
            }
        }
    }

    #[test]
    fn all_nine_response_directions_convert() {
        let c = converter();
        for from in IDS {
            for to in IDS {
                let out = c
                    .convert_response(&pid(from), &pid(to), resp_for(from).as_bytes())
                    .unwrap_or_else(|e| panic!("{from} -> {to} 响应转换失败: {e}"));
                let v: Value = serde_json::from_slice(&out)
                    .unwrap_or_else(|e| panic!("{from} -> {to} 产出不是 JSON: {e}"));
                assert!(v.is_object(), "{from} -> {to} 响应应为 JSON 对象");
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. 损耗清单逐条变成测试
    // ------------------------------------------------------------------

    /// 损耗 #1：Anthropic 的思考签名跨协议无法保真，跨协议时整块剥离。
    #[test]
    fn loss_thinking_is_stripped_when_leaving_anthropic() {
        let c = converter();
        let out = c
            .convert_request(&pid("anthropic"), &pid("openai_chat"), ANTHROPIC_REQ.as_bytes())
            .unwrap();
        let dump = String::from_utf8(out).unwrap();
        assert!(!dump.contains("\"hmm\""), "思考正文不该跨协议带走");
        assert!(!dump.contains("\"signature\""), "思考签名不该跨协议带走");
        // 同协议直通时签名必须原样保留。
        let kept = c
            .convert_request(&pid("anthropic"), &pid("anthropic"), ANTHROPIC_REQ.as_bytes())
            .unwrap();
        assert!(String::from_utf8(kept).unwrap().contains("\"signature\":\"sig\""));
    }

    /// 损耗 #1 后半：目标不是 Anthropic 时清掉 budget_tokens。
    #[test]
    fn loss_budget_tokens_cleared_when_target_is_not_anthropic() {
        let mut req = WireRequest::new("m");
        req.thinking = Some(crate::protocol::ir::ThinkingConfig {
            budget_tokens: Some(2048),
            effort: None,
        });
        let mut to_responses = req.clone();
        strip_unportable_thinking(&mut to_responses, Protocol::OpenAiResponses);
        assert!(to_responses.thinking.unwrap().budget_tokens.is_none());

        // 反过来（→ Anthropic）必须保留：那边唯一的思考参数就是它。
        let mut to_anthropic = req.clone();
        strip_unportable_thinking(&mut to_anthropic, Protocol::Anthropic);
        assert_eq!(
            to_anthropic.thinking.unwrap().budget_tokens,
            Some(2048)
        );
    }

    /// 损耗 #2：`cache_control` 只在直通时保留。
    #[test]
    fn loss_cache_control_only_survives_passthrough() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}"#;
        let c = converter();
        let kept = c.convert_request(&pid("anthropic"), &pid("anthropic"), body.as_bytes()).unwrap();
        assert!(String::from_utf8(kept).unwrap().contains("cache_control"));
        let crossed = c
            .convert_request(&pid("anthropic"), &pid("openai_chat"), body.as_bytes())
            .unwrap();
        assert!(!String::from_utf8(crossed).unwrap().contains("cache_control"));
    }

    /// 损耗 #3：Responses 的 `additional_tools` 折叠进普通 tools（否则 Codex 无工具可用）。
    #[test]
    fn loss_additional_tools_are_folded_into_tools() {
        let body = r#"{"model":"gpt-5.6-sol","tools":null,"input":[
            {"type":"additional_tools","role":"developer","tools":[
                {"type":"function","name":"shell","parameters":{"type":"object"}}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"list"}]}]}"#;
        let c = converter();
        let out = c
            .convert_request(&pid("openai_responses"), &pid("openai_chat"), body.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        let tools = v["tools"].as_array().expect("转换后必须带上 tools");
        assert_eq!(tools[0]["function"]["name"], "shell");
    }

    /// 损耗 #4：finish_reason 归一后细节变粗，但枚举外的值原样带回。
    #[test]
    fn loss_finish_reason_normalized_but_unknown_preserved() {
        let c = converter();
        let out = c
            .convert_response(&pid("anthropic"), &pid("openai_chat"), ANTHROPIC_RESP.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "tool_use 应归一成 tool_calls");

        // 枚举外的取值（pause_turn）不该被吞。
        let raw = br#"{"id":"m","model":"m","content":[],"stop_reason":"pause_turn"}"#;
        let out = c
            .convert_response(&pid("anthropic"), &pid("openai_chat"), raw)
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "pause_turn");
    }

    /// 损耗 #5：未建模的顶层字段靠 `extra` 带过去。
    #[test]
    fn loss_unmodeled_top_level_fields_cross_via_extra() {
        let c = converter();
        // ANTHROPIC_REQ 带了 service_tier，Anthropic 没建模它 → 应出现在 Chat 请求体里。
        let out = c
            .convert_request(&pid("anthropic"), &pid("openai_chat"), ANTHROPIC_REQ.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["service_tier"], "standard_only");
    }

    /// 损耗 #6：用量统一成「不含缓存的 fresh 输入」，跨回 Anthropic 口径一致。
    #[test]
    fn loss_usage_normalized_to_fresh_input() {
        let c = converter();
        let out = c
            .convert_response(&pid("openai_chat"), &pid("anthropic"), CHAT_RESP.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        // OpenAI prompt_tokens 1000 含缓存 800 → fresh 200，缓存单列。
        assert_eq!(v["usage"]["input_tokens"], 200);
        assert_eq!(v["usage"]["cache_read_input_tokens"], 800);
    }

    /// 损耗 #8：自由格式工具转到 Chat 退化成空 schema 的 function。
    #[test]
    fn loss_freeform_tool_degrades_to_a_function() {
        let body = r#"{"model":"m","input":"x","tools":[
            {"type":"custom","name":"apply_patch","description":"patch"}]}"#;
        // IR 层认得出它是 Freeform。
        let ir = OpenAiResponsesCodec.decode_request(body.as_bytes()).unwrap();
        assert!(matches!(ir.tools[0].kind, ToolKind::Freeform));

        let c = converter();
        let out = c
            .convert_request(&pid("openai_responses"), &pid("openai_chat"), body.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "apply_patch");
        // 退化成 JSON schema 的 function —— 文档里写明的损耗。
        assert!(v["tools"][0]["function"]["parameters"].is_object());
    }

    /// 损耗 #9：未建模块同协议逐字保留，跨协议才丢。
    #[test]
    fn loss_unmodeled_block_survives_only_same_protocol() {
        let body = r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AAA"},"title":"a.pdf"},
            {"type":"text","text":"look"}]}]}"#;
        let c = converter();
        let kept = c.convert_request(&pid("anthropic"), &pid("anthropic"), body.as_bytes()).unwrap();
        assert!(String::from_utf8(kept).unwrap().contains("document"));
        let crossed = c
            .convert_request(&pid("anthropic"), &pid("openai_chat"), body.as_bytes())
            .unwrap();
        assert!(!String::from_utf8(crossed).unwrap().contains("document"));
    }

    /// 工具关联 id 必须原样跨协议（否则结果会错配到别的调用上）。
    #[test]
    fn tool_call_ids_survive_cross_protocol() {
        let c = converter();
        let out = c
            .convert_request(&pid("anthropic"), &pid("openai_responses"), ANTHROPIC_REQ.as_bytes())
            .unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        let items = v["input"].as_array().unwrap();
        assert!(
            items.iter().any(|i| i["type"] == "function_call" && i["call_id"] == "tu_1"),
            "工具调用 id 必须原样保留：{v}"
        );
        assert!(
            items
                .iter()
                .any(|i| i["type"] == "function_call_output" && i["call_id"] == "tu_1"),
            "工具结果 id 必须与调用对上：{v}"
        );
    }

    /// 手搓一条 IR 请求，验证 `strip_unportable_thinking` 直接从消息里剥块。
    #[test]
    fn strip_removes_thinking_blocks_and_redacted() {
        let mut req = WireRequest::new("m");
        req.messages.push(crate::protocol::ir::WireMessage::new(
            Role::Assistant,
            vec![
                Block::Thinking {
                    text: "t".into(),
                    signature: Some("s".into()),
                },
                Block::RedactedThinking { data: "d".into() },
                Block::text("answer"),
            ],
        ));
        strip_unportable_thinking(&mut req, Protocol::OpenAiChat);
        assert_eq!(req.messages[0].content.len(), 1);
        assert_eq!(req.messages[0].content[0].as_text(), Some("answer"));
    }

    // ------------------------------------------------------------------
    // 6. 流式往返：A 的帧 → B 的编码器 → B 的解码器，块应一致
    // ------------------------------------------------------------------

    async fn assemble(codec: &dyn Codec, sse: &str) -> Vec<Block> {
        let bytes: ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(
            sse.to_string(),
        ))]));
        let mut stream = decode_stream(codec, bytes);
        let mut asm = BlockAssembler::new();
        while let Some(c) = stream.next().await {
            asm.push(&c.expect("解码不应失败"));
        }
        asm.blocks()
    }

    const ANTHROPIC_SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude\",\"usage\":{\"input_tokens\":3}}}\n\n\
        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
        event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n\
        event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
        event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n\
        event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    #[tokio::test]
    async fn stream_anthropic_to_responses_round_trips_text() {
        let c = converter();
        let frames = c
            .convert_stream(&pid("anthropic"), &pid("openai_responses"), ANTHROPIC_SSE.as_bytes())
            .unwrap();
        let out = frames.join("");
        assert!(out.contains("response.created"), "responses 流必须先发 created");
        assert!(out.contains("response.completed"), "responses 流必须收尾 completed");
        let blocks = assemble(&OpenAiResponsesCodec, &out).await;
        assert_eq!(blocks, vec![Block::text("hello")]);
    }

    #[tokio::test]
    async fn stream_anthropic_to_chat_round_trips_text() {
        let c = converter();
        let frames = c
            .convert_stream(&pid("anthropic"), &pid("openai_chat"), ANTHROPIC_SSE.as_bytes())
            .unwrap();
        let out = frames.join("");
        assert!(out.contains("[DONE]"), "chat 流必须以 [DONE] 收尾");
        let blocks = assemble(&OpenAiChatCodec, &out).await;
        assert_eq!(blocks, vec![Block::text("hello")]);
    }

    /// 工具调用跨流式协议：Anthropic 的 input_json_delta → Responses 的完整 done item，
    /// 再由 Responses 解码器还原成同样的工具调用。
    #[tokio::test]
    async fn stream_tool_call_survives_anthropic_to_responses() {
        let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\"}}\n\n\
            event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"search\",\"input\":{}}}\n\n\
            event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n\
            event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"x\\\"}\"}}\n\n\
            event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
            event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n\
            event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let c = converter();
        let frames = c
            .convert_stream(&pid("anthropic"), &pid("openai_responses"), sse.as_bytes())
            .unwrap();
        let out = frames.join("");
        assert!(out.contains("response.output_item.done"));
        let blocks = assemble(&OpenAiResponsesCodec, &out).await;
        match &blocks[0] {
            Block::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "search");
                assert_eq!(input["q"], "x");
            }
            other => panic!("期望 tool_use，得到 {other:?}"),
        }
    }

    /// 同协议流式也不该被重新切帧 —— 原样一段返回。
    #[tokio::test]
    async fn stream_same_protocol_is_passthrough() {
        let c = converter();
        let frames = c
            .convert_stream(&pid("openai_chat"), &pid("openai_chat"), ANTHROPIC_SSE.as_bytes())
            .unwrap();
        assert_eq!(frames.len(), 1);
    }

    // ------------------------------------------------------------------
    // StreamConverter 增量喂字节（TCP 分片）也不该丢帧
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn stream_converter_handles_split_chunks() {
        let c = converter();
        let mut conv = c
            .stream_converter(&pid("anthropic"), &pid("openai_chat"))
            .unwrap();
        let bytes = ANTHROPIC_SSE.as_bytes();
        let mid = bytes.len() / 2;
        let mut frames = conv.push(&bytes[..mid]).unwrap();
        frames.extend(conv.push(&bytes[mid..]).unwrap());
        frames.extend(conv.finish().unwrap());
        let out = frames.join("");
        assert!(out.contains("[DONE]"));
        let blocks = assemble(&OpenAiChatCodec, &out).await;
        assert_eq!(blocks, vec![Block::text("hello")]);
    }
}
