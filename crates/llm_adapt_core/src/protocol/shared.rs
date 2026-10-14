//! 入站（网关）方向的跨协议公共小工具。
//!
//! 出站方向各协议各写各的没有分叉风险（编码的是自己那套 wire 格式），入站方向则
//! 共享三类东西：SSE 帧的**渲染**、无中生有的 id、以及当前时间戳。它们放这里，
//! 免得三个 `inbound` 各写一份 —— 各写一份的代价是同一段帧渲染出现三种细微不同的
//! 写法（尾随空行少一个、`event:` 后多一个空格），而客户端对 SSE 的容错并不一致。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 渲染一个**完整**的 SSE 帧。
///
/// 返回的字符串含 `event:`（可选）/`data:` 行，并以空行结尾 —— 调用方直接写进
/// socket 即可，不需要再拼分隔符。多个 `data:` 行的形态这里用不到：三个协议下发
/// 的每一帧都只有一个 `data:` 行。
///
/// `event: None` 时不输出 event 行。Chat Completions 就见不得 event 行，
/// 给它加上某些严格客户端反而会拒收。
pub(crate) fn sse_frame(event: Option<&str>, data: &str) -> String {
    match event {
        Some(name) => format!("event: {name}\ndata: {data}\n\n"),
        None => format!("data: {data}\n\n"),
    }
}

/// 带 `event:` 行的帧。Anthropic 与 Responses 都用它。
pub(crate) fn sse_named(event: &str, data: &str) -> String {
    sse_frame(Some(event), data)
}

/// 当前 Unix 秒。wire 上的 `created` / `created_at` 只要一个单调递增的数字即可，
/// 不需要毫秒精度。
pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 进程内的单调计数，喂给 [`synth_id`]。
static ID_SEQ: AtomicU64 = AtomicU64::new(0);

/// 合成一个客户端可见、进程内唯一、跨重启也几乎不撞的 id。
///
/// 入站的流式编码器在 IR 没给 id 时（上游或上一个协议的 mapper 没提供）必须自己造
/// 一个 —— `message_start` / `response.created` / `chatcmpl-*` 都不能是空串，
/// 否则客户端拿它做去重、做关联时会出问题。这里不引入 uuid 依赖：秒级时间戳 +
/// 进程内自增序号，同一进程内绝不重复，重启后时间戳又不同。
pub(crate) fn synth_id(prefix: &str) -> String {
    let n = ID_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{:x}{:x}", now_secs(), n)
}
