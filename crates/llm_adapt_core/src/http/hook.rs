//! 每次 HTTP 调用前后触发的扩展钩子。

use super::{HttpRequest, RawHttpResponse};

/// [`crate::http::HttpExecutor`] 在执行请求前后触发的回调。
///
/// 钩子刻意做成同步的：它们的定位是廉价观察（打日志、记指标、为调试 UI
/// 抓原始帧）。重活应该丢到后台任务里去。
pub trait RequestHook: Send + Sync {
    fn before_request(&self, _request: &HttpRequest) {}
    fn after_raw_response(&self, _response: &RawHttpResponse) {}

    /// 流式响应每收到一块原始字节就调一次。
    ///
    /// `after_raw_response` 只覆盖非流式 —— 流式的响应体是逐块吐出来的，
    /// 等拼完再回调就等于把整段响应缓冲在内存里，而那可能很长。
    /// 要注意这是**网络块**不是 SSE 帧：帧的边界不保证与块的边界对齐。
    fn on_stream_chunk(&self, _chunk: &bytes::Bytes) {}
}
