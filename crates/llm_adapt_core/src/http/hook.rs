//! 每次 HTTP 调用前后触发的扩展钩子。

use super::{HttpRequest, RawHttpResponse};

/// [`crate::http::HttpExecutor`] 在执行请求前后触发的回调。
///
/// 钩子刻意做成同步的：它们的定位是廉价观察（打日志、记指标、为调试 UI
/// 抓原始帧）。重活应该丢到后台任务里去。
pub trait RequestHook: Send + Sync {
    fn before_request(&self, _request: &HttpRequest) {}

    /// 收到响应头时。**带上产生它的那次请求**。
    ///
    /// 响应本身不带请求 id，而钩子可能同时服务多个并发请求（并行子 agent）。
    /// 只给响应的话，hook 只能靠「上一条请求」这种脆弱假设来配对 —— 并发一上来就错。
    fn after_raw_response(&self, _request: &HttpRequest, _response: &RawHttpResponse) {}

    /// 流式响应每收到一块原始字节就调一次（同样带上请求）。
    ///
    /// `after_raw_response` 只覆盖非流式 —— 流式的响应体是逐块吐出来的，
    /// 等拼完再回调就等于把整段响应缓冲在内存里，而那可能很长。
    /// 注意这是**网络块**不是 SSE 帧：帧边界不保证与块边界对齐。
    fn on_stream_chunk(&self, _request: &HttpRequest, _chunk: &bytes::Bytes) {}

    /// 流结束时回调一次 —— 正常收完、出错、或被提前 drop 都算。
    ///
    /// 没有它，抓包方就不知道一条流式交换在哪里收尾，只能靠超时猜。
    fn on_stream_end(&self, _request: &HttpRequest) {}
}
