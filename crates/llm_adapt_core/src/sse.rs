//! 统一的 SSE 帧解析与驱动。
//!
//! 此前 `handlers/openai.rs` 与 `handlers/anthropic.rs` 各自复制了一份
//! `find_event_boundary` + 手写 `Stream` 实现；再加第三个协议就会变成三份。
//! 这里把「切帧」与「驱动字节流」收拢成一份，协议差异通过 [`SseMapper`] 注入。
//!
//! 「流结束」由 mapper 显式声明，而不是靠「见到 `Finish` 就停」。这不是洁癖：
//! OpenAI 在 `finish_reason` 之后还会发一帧带 usage 的数据
//! （`stream_options.include_usage`），若见 `Finish` 即停，那一帧会随机丢在半路。

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::BytesMut;
use futures_util::Stream;

use crate::error::LLMError;
use crate::http::ByteStream;
use crate::types::StreamChunk;

/// 解析后的 chunk 流：应用层消费的流式形态。
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>;

/// 一个完整的 SSE 帧。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseFrame {
    /// `event:` 字段。OpenAI 不用它，Anthropic 用它区分事件类型。
    pub event: Option<String>,
    /// `data:` 字段。多行 `data:` 按 SSE 规范以 `\n` 连接。
    pub data: String,
}

/// mapper 对一帧的产出。
#[derive(Debug, Default)]
pub struct SseOutcome {
    pub chunks: Vec<StreamChunk>,
    /// 该帧表示协议层流结束（OpenAI 的 `[DONE]`、Anthropic 的 `message_stop`）。
    /// 置位后驱动层不再处理后续字节。
    pub terminal: bool,
}

impl SseOutcome {
    /// 非终止帧的产出。
    pub fn chunks(chunks: Vec<StreamChunk>) -> Self {
        Self {
            chunks,
            terminal: false,
        }
    }
}

/// 协议侧的 SSE 映射器。每个协议实现一个。
///
/// 要求 `Unpin`，因为 [`SseChunkStream`] 需要在 `poll_next` 里拿 `&mut` 访问。
/// 各协议的 mapper 都是普通结构体，天然满足。
pub trait SseMapper: Send {
    /// 把一帧映射成 0..n 个 chunk。返回 `Err` 表示这一帧无法解析。
    fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError>;
}

/// 让 `Box<dyn SseMapper>` 也能当 mapper 用 ——
/// [`Codec::new_stream_decoder`](crate::protocol::Codec::new_stream_decoder)
/// 返回的正是它，而 trait 对象不会自动实现自己的 trait。
impl SseMapper for Box<dyn SseMapper> {
    fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError> {
        (**self).map(frame)
    }
}

/// 找出缓冲里下一个帧分隔符，返回 `(起始位置, 分隔符长度)`。
///
/// `\n\n` 与 `\r\n\r\n` 在同一位置互斥（`\r\n\r\n` 里没有连续两个 `\n`），
/// 因此按起始位置取靠前的那个即可。
///
/// `pub(crate)`：网关方向的流式协议转换器（[`crate::protocol::convert`]）也要按同样
/// 的规则切帧，切帧逻辑不能各写一份 —— 少一个尾分隔符的容错差异会让两边对不上。
pub(crate) fn find_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
    let crlf = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| (p, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// 解析一个帧的原始字节。`data` 为空表示这是注释行 / 心跳，驱动层会跳过。
pub(crate) fn parse_frame(raw: &[u8]) -> SseFrame {
    let text = std::str::from_utf8(raw).unwrap_or("");
    let mut event = None;
    let mut data_parts: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_parts.push(rest.trim_start());
        }
    }
    SseFrame {
        event,
        data: data_parts.join("\n"),
    }
}

/// 把 [`ByteStream`] 驱动成 [`ChunkStream`]。
pub struct SseChunkStream<M: SseMapper> {
    inner: ByteStream,
    buffer: BytesMut,
    pending: VecDeque<Result<StreamChunk, LLMError>>,
    mapper: M,
    /// 协议层终止帧已见（OpenAI 的 `[DONE]` / Anthropic 的 `message_stop`）。
    terminal_seen: bool,
    /// 不再处理任何字节：pending 排空后返回 `None`。
    finished: bool,
}

impl<M: SseMapper> SseChunkStream<M> {
    pub fn new(inner: ByteStream, mapper: M) -> Self {
        Self {
            inner,
            buffer: BytesMut::new(),
            pending: VecDeque::new(),
            mapper,
            terminal_seen: false,
            finished: false,
        }
    }

    /// 是否见到了协议层终止帧。
    pub fn is_finished(&self) -> bool {
        self.terminal_seen
    }

    /// 把已到达的字节切成完整帧并交给 mapper。
    fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        while let Some((pos, sep)) = find_boundary(&self.buffer) {
            let raw = self.buffer.split_to(pos);
            let _ = self.buffer.split_to(sep);
            let frame = parse_frame(&raw);
            if frame.data.is_empty() {
                continue;
            }
            self.dispatch(frame, false);
            if self.terminal_seen {
                return;
            }
        }
    }

    fn dispatch(&mut self, frame: SseFrame, partial: bool) {
        match self.mapper.map(frame) {
            Ok(outcome) => {
                for c in outcome.chunks {
                    self.pending.push_back(Ok(c));
                }
                if outcome.terminal {
                    self.terminal_seen = true;
                    self.finished = true;
                }
            }
            // 残缺的尾帧（EOF 时缺分隔符）解析失败就静默丢弃：它不是协议错误，
            // 只是服务端没把最后一个 `\n\n` 发出来。
            Err(e) => {
                if !partial {
                    self.pending.push_back(Err(e));
                }
            }
        }
    }

    /// EOF：把缺尾分隔符的残留帧也交出去。
    fn flush_trailing(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let raw = self.buffer.split().freeze();
        let frame = parse_frame(&raw);
        if !frame.data.is_empty() {
            self.dispatch(frame, true);
        }
    }
}

impl<M: SseMapper + Unpin> Stream for SseChunkStream<M> {
    type Item = Result<StreamChunk, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(item) = this.pending.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => this.feed(&bytes),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    this.flush_trailing();
                    this.finished = true;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures_util::StreamExt;

    use crate::types::FinishReason;

    /// 每个非终止帧产出一个 `TextDelta`（内容是帧的 data），便于断言顺序与截断。
    #[derive(Default)]
    struct Echo {
        seen: Vec<String>,
    }

    impl SseMapper for Echo {
        fn map(&mut self, frame: SseFrame) -> Result<SseOutcome, LLMError> {
            if frame.data.trim() == "[DONE]" {
                return Ok(SseOutcome {
                    chunks: vec![StreamChunk::Finish {
                        reason: FinishReason::Stop,
                    }],
                    terminal: true,
                });
            }
            self.seen.push(frame.data.clone());
            Ok(SseOutcome::chunks(vec![StreamChunk::TextDelta {
                index: 0,
                text: frame.data,
            }]))
        }
    }

    fn drain_frames(buf: &mut BytesMut) -> Vec<SseFrame> {
        let mut out = Vec::new();
        while let Some((pos, sep)) = find_boundary(buf) {
            let raw = buf.split_to(pos);
            let _ = buf.split_to(sep);
            let f = parse_frame(&raw);
            if !f.data.is_empty() {
                out.push(f);
            }
        }
        out
    }

    #[test]
    fn splits_on_lf_and_crlf() {
        let mut buf = BytesMut::from(&b"event: a\ndata: 1\n\ndata: 2\r\n\r\n"[..]);
        let frames = drain_frames(&mut buf);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].event.as_deref(), Some("a"));
        assert_eq!(frames[0].data, "1");
        assert_eq!(frames[1].event, None);
        assert_eq!(frames[1].data, "2");
    }

    #[test]
    fn joins_multiline_data_and_skips_heartbeats() {
        let mut buf = BytesMut::from(&b"data: {\"a\":\ndata: 1}\n\n: ping\n\ndata: x\n\n"[..]);
        let frames = drain_frames(&mut buf);
        assert_eq!(frames.len(), 2, "注释/心跳帧应被跳过");
        assert_eq!(frames[0].data, "{\"a\":\n1}");
        assert_eq!(frames[1].data, "x");
    }

    #[test]
    fn incomplete_tail_stays_in_buffer() {
        let mut buf = BytesMut::from(&b"data: done\n\ndata: partial"[..]);
        let frames = drain_frames(&mut buf);
        assert_eq!(frames.len(), 1);
        assert_eq!(buf.as_ref(), b"data: partial");
    }

    async fn collect(frames: &'static [u8]) -> (Vec<StreamChunk>, bool) {
        let s: ByteStream = Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from_static(
            frames,
        ))]));
        let mut stream = SseChunkStream::new(s, Echo::default());
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.expect("Echo mapper never errors"));
        }
        (out, stream.is_finished())
    }

    #[tokio::test]
    async fn flushes_trailing_frame_at_eof() {
        // 服务端没发最后一个 `\n\n` —— 尾帧仍应被解析出来。
        let (chunks, finished) = collect(b"data: first\n\ndata: tail").await;
        let texts: Vec<String> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::TextDelta { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "tail"]);
        assert!(!finished, "没有终止帧，只是 EOF");
    }

    #[tokio::test]
    async fn terminal_frame_stops_processing() {
        let (chunks, finished) = collect(b"data: a\n\ndata: [DONE]\n\ndata: after\n\n").await;
        assert!(finished);
        let texts: Vec<String> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::TextDelta { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["a"], "终止帧之后的字节不应再产出");
        assert!(matches!(chunks.last(), Some(StreamChunk::Finish { .. })));
    }

    #[tokio::test]
    async fn frames_split_across_tcp_chunks_are_reassembled() {
        let s: ByteStream = Box::pin(futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"data: he")),
            Ok(Bytes::from_static(b"l")),
            Ok(Bytes::from_static(b"lo\n\n")),
        ]));
        let mut stream = SseChunkStream::new(s, Echo::default());
        let mut texts = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamChunk::TextDelta { text, .. } = item.unwrap() {
                texts.push(text);
            }
        }
        assert_eq!(texts, vec!["hello"]);
    }
}
