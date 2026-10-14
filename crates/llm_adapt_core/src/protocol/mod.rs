//! 协议层：协议无关的 IR，以及各协议的编解码实现。
//!
//! 分层位置：`http`（怎么发出去）→ `protocol`（发什么、回的什么）→ `types`
//! （给人用的角色感知 API）。方向严格单向，`protocol` 不认识 `types`。
//!
//! * [`ir`] —— 协议无关的中间表示。所有转换都经过它。
//! * [`codec`] —— [`Codec`] trait 与 [`ProtocolRegistry`]。

pub mod anthropic;
pub mod codec;
pub mod ir;
pub mod oai_chat;
pub mod oai_responses;

pub use codec::{decode_stream, Codec, ProtocolRegistry, StreamEncoder};
pub use ir::*;

use codec::ProtocolRegistry as Registry;

/// 把所有内置协议注册进注册表。
pub fn register_defaults(registry: &Registry) {
    registry.register(std::sync::Arc::new(anthropic::AnthropicCodec));
    registry.register(std::sync::Arc::new(oai_chat::OpenAiChatCodec));
    registry.register(std::sync::Arc::new(oai_responses::OpenAiResponsesCodec));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::ClientConfig;
    use crate::protocol::ir::{Block, WireRequest};

    fn registry() -> ProtocolRegistry {
        let r = ProtocolRegistry::new();
        register_defaults(&r);
        r
    }

    /// 自定义路径要能覆盖协议默认路径，且 base_url 的尾斜杠不出乱子。
    #[test]
    fn custom_path_overrides_the_protocol_default() {
        let mut cfg = ClientConfig::new("https://gw.example.com/anthropic/", "k");
        let mut ir = WireRequest::new("m");
        ir.max_tokens = Some(16);

        // 默认：base_url 去尾斜杠 + 协议默认路径
        let codec = registry().get(&ProtocolId::new("anthropic")).unwrap();
        let http = codec.encode_request(&ir, &cfg).unwrap();
        assert_eq!(http.url, "https://gw.example.com/anthropic/v1/messages");

        // 覆盖：用指定路径（相对写法要自动补前导斜杠）
        cfg.path = Some("v1/beta/messages".into());
        let http = codec.encode_request(&ir, &cfg).unwrap();
        assert_eq!(http.url, "https://gw.example.com/anthropic/v1/beta/messages");
    }

    #[test]
    fn all_builtin_protocols_are_registered() {
        assert_eq!(
            registry().list().iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            vec!["anthropic", "openai_chat", "openai_responses"]
        );
    }

    /// **跨协议不变量**：三个协议必须用同一种方式拼 system。
    /// 否则同一段对话换个渠道，提示词就变了 —— 而调用方完全看不出来。
    #[test]
    fn all_protocols_join_system_segments_identically() {
        let cfg = ClientConfig::new("https://example.invalid", "k");
        let mut ir = WireRequest::new("m");
        ir.max_tokens = Some(16);
        ir.system = vec![Block::text("人设"), Block::text("技能"), Block::text("记忆")];

        for id in registry().list() {
            let codec = registry().get(&id).expect("已注册");
            let http = codec.encode_request(&ir, &cfg).expect("编码成功");
            let dump = String::from_utf8(http.body).expect("UTF-8");
            assert!(
                dump.contains("人设\\n技能\\n记忆"),
                "{id} 的 system 拼接方式与其它协议不一致：{dump}"
            );
        }
    }
}
