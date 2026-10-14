//! 每次 HTTP 调用前后触发的扩展钩子。

use super::{HttpRequest, RawHttpResponse};

/// [`crate::http::HttpExecutor`] 在执行请求前后触发的回调。
///
/// 钩子刻意做成同步的：它们的定位是廉价观察（打日志、记指标、为调试 UI
/// 抓原始帧）。重活应该丢到后台任务里去。
pub trait RequestHook: Send + Sync {
    fn before_request(&self, _request: &HttpRequest) {}
    fn after_raw_response(&self, _response: &RawHttpResponse) {}
}
