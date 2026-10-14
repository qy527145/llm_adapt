//! HTTP layer.
//!
//! The HTTP layer is independent of any provider: handlers produce
//! [`crate::handler::HttpRequest`] descriptions, and [`HttpExecutor`] turns them
//! into actual network calls via `reqwest`.

pub mod config;
pub mod hook;
pub mod retry;

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{ErrorContext, LLMError};

pub use config::{ClientConfig, ProxyConfig};
pub use hook::RequestHook;
pub use retry::RetryPolicy;

// ---------------------------------------------------------------------------
// 传输层的公共类型
//
// 这几个类型属于 HTTP 层（协议层要用它们，但它们是「怎么发出去」，不是
// 「发什么」），因此定义在这里。`handler` 会重新导出它们以保持既有路径可用。
// ---------------------------------------------------------------------------

/// HTTP 方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

/// 供应商中立的 HTTP 请求描述。
///
/// 协议层产出它，[`HttpExecutor`] 把它变成真正的网络调用。它的 `body` 是**最终的
/// 字节**——「原始报文检视」功能拿到的就是它，不存在二次序列化。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// 把请求体当 UTF-8 渲染，用于调试 / `preview`。
    /// 空 body 或非 UTF-8 时返回 `None`。
    pub fn body_as_str(&self) -> Option<&str> {
        if self.body.is_empty() {
            return None;
        }
        std::str::from_utf8(&self.body).ok()
    }
}

/// 未经解码的原始响应。hook 拿到它做日志与调试。
#[derive(Debug, Clone)]
pub struct RawHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

/// 网络层吐出的原始响应字节流。
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, LLMError>> + Send>>;


/// Stateful HTTP executor. Holds a `reqwest::Client`, configuration, retry
/// policy, and a list of hooks. Cloning is cheap — internals are `Arc`-shared.
#[derive(Clone)]
pub struct HttpExecutor {
    client: Client,
    config: ClientConfig,
    retry: RetryPolicy,
    hooks: Vec<Arc<dyn RequestHook>>,
}

impl HttpExecutor {
    /// Build an executor from the given configuration. Returns
    /// [`LLMError::Config`] if the underlying `reqwest::Client` cannot be built
    /// (typically a bad proxy URL).
    pub fn new(config: ClientConfig) -> Result<Self, LLMError> {
        Self::with_retry(config, RetryPolicy::default())
    }

    pub fn with_retry(config: ClientConfig, retry: RetryPolicy) -> Result<Self, LLMError> {
        let client = build_reqwest_client(&config)?;
        Ok(Self {
            client,
            config,
            retry,
            hooks: Vec::new(),
        })
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    pub fn retry_policy(&self) -> &RetryPolicy {
        &self.retry
    }

    /// Register a hook. Hooks fire in registration order.
    pub fn add_hook(&mut self, hook: Arc<dyn RequestHook>) {
        self.hooks.push(hook);
    }

    /// Execute a non-streaming request. On a successful HTTP response (2xx) the
    /// raw body is returned; non-2xx responses are converted into the matching
    /// [`LLMError`] variant.
    pub async fn execute(&self, req: &HttpRequest) -> Result<RawHttpResponse, LLMError> {
        let mut attempt = 0u32;
        loop {
            for hook in &self.hooks {
                hook.before_request(req);
            }
            let start = Instant::now();
            let result = self.send_once(req).await;
            match result {
                Ok(raw) => {
                    for hook in &self.hooks {
                        hook.after_raw_response(&raw);
                    }
                    if let Some(err) = error_from_status(&raw) {
                        if self.retry.should_retry(attempt, &err) {
                            attempt += 1;
                            tokio::time::sleep(self.retry.delay(attempt)).await;
                            continue;
                        }
                        return Err(err);
                    }
                    return Ok(raw);
                }
                Err(mut err) => {
                    if let LLMError::Timeout { ref mut elapsed_ms, .. } = err {
                        *elapsed_ms = start.elapsed().as_millis();
                    }
                    if self.retry.should_retry(attempt, &err) {
                        attempt += 1;
                        tokio::time::sleep(self.retry.delay(attempt)).await;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    /// Execute a streaming request. Returns a [`ByteStream`] of body chunks plus
    /// a [`StreamMeta`] describing the response head (status, headers). Stream
    /// requests are not retried — callers are expected to drive them to completion.
    pub async fn execute_stream(
        &self,
        req: &HttpRequest,
    ) -> Result<(StreamMeta, ByteStream), LLMError> {
        for hook in &self.hooks {
            hook.before_request(req);
        }
        let reqwest_req = self.to_reqwest(req)?;
        let resp = self.client.execute(reqwest_req).await.map_err(LLMError::from)?;
        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string()))
            .collect();
        if !resp.status().is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            let raw = RawHttpResponse { status, headers, body };
            for hook in &self.hooks {
                hook.after_raw_response(&raw);
            }
            return Err(error_from_status(&raw).unwrap_or_else(|| LLMError::Provider {
                message: format!("unexpected status {status}"),
                ctx: ErrorContext::default().with_status(status),
            }));
        }
        let meta = StreamMeta { status, headers };
        let stream = resp
            .bytes_stream()
            .map_err(LLMError::from)
            .boxed();
        Ok((meta, stream))
    }

    async fn send_once(&self, req: &HttpRequest) -> Result<RawHttpResponse, LLMError> {
        let reqwest_req = self.to_reqwest(req)?;
        let resp = self.client.execute(reqwest_req).await.map_err(LLMError::from)?;
        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string()))
            .collect();
        let body = resp.bytes().await.map_err(LLMError::from)?;
        Ok(RawHttpResponse { status, headers, body })
    }

    fn to_reqwest(&self, req: &HttpRequest) -> Result<reqwest::Request, LLMError> {
        let method = match req.method {
            HttpMethod::Get => reqwest::Method::GET,
            HttpMethod::Post => reqwest::Method::POST,
            HttpMethod::Put => reqwest::Method::PUT,
            HttpMethod::Delete => reqwest::Method::DELETE,
            HttpMethod::Patch => reqwest::Method::PATCH,
        };
        let mut builder = self.client.request(method, &req.url);
        for (k, v) in &self.config.default_headers {
            builder = builder.header(k, v);
        }
        for (k, v) in &req.headers {
            builder = builder.header(k, v);
        }
        if !req.body.is_empty() {
            builder = builder.body(req.body.clone());
        }
        builder
            .build()
            .map_err(|e| LLMError::Config { message: e.to_string() })
    }
}

/// Metadata about a streaming response head.
#[derive(Debug, Clone)]
pub struct StreamMeta {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

fn build_reqwest_client(config: &ClientConfig) -> Result<Client, LLMError> {
    let mut builder = Client::builder()
        .connect_timeout(config.connect_timeout)
        .timeout(config.request_timeout)
        .danger_accept_invalid_certs(!config.verify_ssl);
    if let Some(proxy) = &config.proxy {
        let scheme = proxy.url.clone();
        let proxy = reqwest::Proxy::all(&scheme)
            .map_err(|e| LLMError::Config { message: format!("invalid proxy '{scheme}': {e}") })?;
        builder = builder.proxy(proxy);
    } else {
        // Explicitly disable picking up system proxy unless requested.
        builder = builder.no_proxy();
    }
    builder
        .build()
        .map_err(|e| LLMError::Config { message: format!("failed to build HTTP client: {e}") })
}

fn error_from_status(raw: &RawHttpResponse) -> Option<LLMError> {
    if (200..300).contains(&raw.status) {
        return None;
    }
    let body_preview = std::str::from_utf8(&raw.body)
        .unwrap_or("<non-utf8 body>")
        .chars()
        .take(512)
        .collect::<String>();
    let request_id = raw
        .headers
        .iter()
        .find(|(k, _)| {
            k.eq_ignore_ascii_case("x-request-id")
                || k.eq_ignore_ascii_case("anthropic-request-id")
                || k.eq_ignore_ascii_case("openai-request-id")
        })
        .map(|(_, v)| v.clone());
    let retry_after_ms = raw
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, v)| v.parse::<u64>().ok())
        .map(|s| s * 1000);
    let mut ctx = ErrorContext::default().with_status(raw.status);
    if let Some(id) = request_id {
        ctx = ctx.with_request_id(id);
    }
    Some(match raw.status {
        401 | 403 => LLMError::Auth { message: body_preview, ctx },
        429 => LLMError::RateLimit {
            message: body_preview,
            retry_after_ms,
            ctx: ErrorContext { retryable: true, ..ctx },
        },
        408 => LLMError::Timeout {
            elapsed_ms: 0,
            ctx: ErrorContext { retryable: true, ..ctx },
        },
        413 => LLMError::ContextOverflow { message: body_preview, ctx },
        500..=599 => LLMError::Provider {
            message: body_preview,
            ctx: ErrorContext { retryable: true, ..ctx },
        },
        _ => LLMError::Provider { message: body_preview, ctx },
    })
}

/// Convenience for buffering a streaming body — used by tests and the CLI
/// `--show-raw` flag.
pub async fn collect_byte_stream(mut stream: ByteStream) -> Result<Bytes, LLMError> {
    let mut buf = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}
