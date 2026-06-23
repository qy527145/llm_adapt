//! Extension hooks invoked around every HTTP call.

use crate::handler::{HttpRequest, RawHttpResponse};

/// Callbacks fired by [`crate::http::HttpExecutor`] around request execution.
///
/// Hooks are synchronous on purpose: they're meant for cheap inspection
/// (logging, metrics, raw-frame capture for the debug UIs). Heavy work should
/// be offloaded to a background task.
pub trait RequestHook: Send + Sync {
    fn before_request(&self, _request: &HttpRequest) {}
    fn after_raw_response(&self, _response: &RawHttpResponse) {}
}
