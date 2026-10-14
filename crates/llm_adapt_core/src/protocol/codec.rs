//! 协议编解码接口。
//!
//! [`Codec`] 从第一天就定义**六个方法**：出站三个（客户端方向）与入站三个
//! （网关方向）。入站的三个先给默认实现返回「未实现」，由各协议在后续补齐。
//! 这样做的理由很实际：如果先只按客户端形状设计，等要接网关时 trait 会被迫改形，
//! 所有实现和调用点都要返工。先定形、后补实现，返工成本是零。

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bytes::Bytes;

use crate::error::LLMError;
use crate::http::{ByteStream, ClientConfig, HttpRequest};
use crate::sse::{ChunkStream, SseChunkStream, SseMapper};
use crate::types::StreamChunk;

use super::ir::{ProtocolId, WireRequest, WireResponse};

/// 用某个 codec 的解码器把网络字节流驱动成 chunk 流。
///
/// 是自由函数而不是 `Codec` 的方法，是因为 `SseChunkStream` 需要按值持有 mapper，
/// 而 trait 对象得先 `Box` 出来。
pub fn decode_stream(codec: &dyn Codec, bytes: ByteStream) -> ChunkStream {
    Box::pin(SseChunkStream::new(bytes, codec.new_stream_decoder()))
}

/// 拼出请求 URL：`base_url` + 协议默认路径，或 [`ClientConfig::path`] 指定的路径。
///
/// 独立成函数是因为三个 codec 都要用，而「base_url 带不带尾斜杠」「路径从哪来」
/// 这类细节一旦各写各的就会分叉。
pub fn endpoint_url(config: &ClientConfig, default_path: &str) -> String {
    let base = config.base_url.trim_end_matches('/');
    match &config.path {
        Some(p) if !p.is_empty() => format!("{base}{}", ensure_leading_slash(p)),
        _ => format!("{base}{default_path}"),
    }
}

fn ensure_leading_slash(p: &str) -> String {
    if p.starts_with('/') {
        p.to_string()
    } else {
        format!("/{p}")
    }
}

/// 一个协议的全部编解码能力。
///
/// 对象安全：注册表里存的是 `Arc<dyn Codec>`，因此各协议实现不能带泛型方法。
pub trait Codec: Send + Sync {
    /// 本 codec 负责的协议 id。
    fn protocol(&self) -> ProtocolId;

    // ------------------------------------------------------------------
    // 出站方向（客户端）：我们要发请求给上游，并读懂它回什么
    // ------------------------------------------------------------------

    /// IR → 具体的 HTTP 请求（含最终字节形态的 body）。
    fn encode_request(
        &self,
        request: &WireRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError>;

    /// 原始响应体 → IR。
    fn decode_response(&self, body: Bytes) -> Result<WireResponse, LLMError>;

    /// 流式响应解码器：SSE 帧 → [`StreamChunk`]。
    fn new_stream_decoder(&self) -> Box<dyn SseMapper>;

    // ------------------------------------------------------------------
    // 入站方向（网关）：别的客户端把某个协议的报文发给我们
    // ------------------------------------------------------------------

    /// 原始请求体 → IR。
    fn decode_request(&self, _body: &[u8]) -> Result<WireRequest, LLMError> {
        Err(LLMError::unsupported(format!(
            "{}: 入站请求解码尚未实现",
            self.protocol()
        )))
    }

    /// IR → 原始响应体。
    fn encode_response(&self, _response: &WireResponse) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::unsupported(format!(
            "{}: 入站响应编码尚未实现",
            self.protocol()
        )))
    }

    /// 流式响应编码器：IR 增量 → SSE 帧。`None` 表示本协议尚不支持。
    fn new_stream_encoder(&self) -> Option<Box<dyn StreamEncoder>> {
        None
    }
}

/// IR 增量 → SSE 帧。输出的是**已渲染好的完整帧文本**（含 `event:` / `data:` 行，
/// 以一个空行结尾），调用方直接写进 socket 即可。
pub trait StreamEncoder: Send {
    fn encode(&mut self, chunk: &StreamChunk) -> Result<Vec<String>, LLMError>;

    /// 流结束时的收尾帧（如 Anthropic 的 `message_stop`、OpenAI 的 `[DONE]`）。
    fn finish(&mut self) -> Result<Vec<String>, LLMError>;
}

/// 协议注册表：`ProtocolId` → `Codec`。
///
/// 克隆廉价（内部只有 `Arc`）。注册时用 `Arc<dyn Codec>`，因此可以运行时替换实现
/// （测试里塞假 codec、网关里按渠道切实现）。
#[derive(Default, Clone)]
pub struct ProtocolRegistry {
    inner: Arc<RwLock<HashMap<ProtocolId, Arc<dyn Codec>>>>,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册（或覆盖）一个协议的 codec。
    pub fn register(&self, codec: Arc<dyn Codec>) {
        let id = codec.protocol();
        let mut guard = self.inner.write().expect("ProtocolRegistry 锁中毒");
        guard.insert(id, codec);
    }

    /// 取一个 codec。未注册时报 [`LLMError::HandlerNotFound`]。
    pub fn get(&self, id: &ProtocolId) -> Result<Arc<dyn Codec>, LLMError> {
        let guard = self.inner.read().expect("ProtocolRegistry 锁中毒");
        guard.get(id).cloned().ok_or_else(|| LLMError::HandlerNotFound {
            api_format: id.to_string(),
        })
    }

    /// 是否已注册。
    pub fn contains(&self, id: &ProtocolId) -> bool {
        self.inner
            .read()
            .expect("ProtocolRegistry 锁中毒")
            .contains_key(id)
    }

    /// 全部已注册的协议 id，已排序。
    pub fn list(&self) -> Vec<ProtocolId> {
        let guard = self.inner.read().expect("ProtocolRegistry 锁中毒");
        let mut keys: Vec<ProtocolId> = guard.keys().cloned().collect();
        keys.sort();
        keys
    }

    pub fn len(&self) -> usize {
        self.inner.read().expect("ProtocolRegistry 锁中毒").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for ProtocolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProtocolRegistry")
            .field("protocols", &self.list())
            .finish()
    }
}
