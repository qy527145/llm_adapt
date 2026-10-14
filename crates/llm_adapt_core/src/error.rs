//! 所有 `llm_adapt` 操作统一的错误类型。
//!
//! 核心 crate 里每个可能失败的 API 都返回 [`LLMError`]。错误按稳定的变体分类，
//! 便于调用方按类别 match，同时携带可选元数据（请求 id、HTTP 状态、可重试提示）
//! 以便给出更丰富的诊断。

use std::fmt;

use thiserror::Error;

/// 附加在多数错误变体上的额外上下文。所有字段都是可选的，
/// 这样无论掌握多少信息，任意一层都能构造错误。
#[derive(Debug, Default, Clone)]
pub struct ErrorContext {
    /// 上游提供的请求 id（如果响应里带了的话）。
    pub request_id: Option<String>,
    /// 触发该错误的 HTTP 状态码（如果适用）。
    pub http_status: Option<u16>,
    /// 用同样的 payload 重试是否安全。
    pub retryable: bool,
    /// 厂商专属错误码（用字符串形式以保持格式无关）。
    pub vendor_code: Option<String>,
}

impl ErrorContext {
    pub fn retryable() -> Self {
        Self {
            retryable: true,
            ..Self::default()
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn with_vendor_code(mut self, code: impl Into<String>) -> Self {
        self.vendor_code = Some(code.into());
        self
    }
}

/// `llm_adapt_core` 抛出的所有错误。
#[derive(Debug, Error)]
pub enum LLMError {
    /// 在构造任何请求之前发现的配置错误（缺 key、URL 非法等）。
    #[error("configuration error: {message}")]
    Config { message: String },

    /// 传入的 [`crate::types::ChatRequest`] 不满足校验规则。
    #[error("validation error: {message}")]
    Validation { message: String },

    /// 网络/传输层失败（DNS、TLS、broken pipe……）。
    #[error("network error: {message}")]
    Network { message: String, ctx: ErrorContext },

    /// 请求超过了配置的超时时间。
    #[error("request timed out after {elapsed_ms} ms")]
    Timeout { elapsed_ms: u128, ctx: ErrorContext },

    /// 认证/授权失败（HTTP 401/403）。
    #[error("authentication error: {message}")]
    Auth { message: String, ctx: ErrorContext },

    /// 触发限流（HTTP 429）。上游发了 `Retry-After` 头时，
    /// `retry_after_ms` 与之对应。
    #[error("rate limited: {message}")]
    RateLimit {
        message: String,
        retry_after_ms: Option<u64>,
        ctx: ErrorContext,
    },

    /// 上游以安全为由拒绝了请求。
    #[error("safety violation: {message}")]
    Safety { message: String, ctx: ErrorContext },

    /// 超出了上下文窗口。
    #[error("context overflow: {message}")]
    ContextOverflow { message: String, ctx: ErrorContext },

    /// 响应体无法解析成稳定的结构。
    #[error("parse error: {message}")]
    Parse { message: String, ctx: ErrorContext },

    /// 上游返回了非成功响应，且不归入其它任何类别。
    #[error("provider error: {message}")]
    Provider { message: String, ctx: ErrorContext },

    /// 请求的 `api_format` 没有注册对应的 handler。
    #[error("no handler registered for api_format = {api_format}")]
    HandlerNotFound { api_format: String },

    /// 该实现尚未支持这项能力（例如入站方向的编解码还没写）。
    /// 与 `Validation` 区分开：这不是调用方用错了，是本库还没做。
    #[error("unsupported: {message}")]
    Unsupported { message: String },
}

impl LLMError {
    /// 返回内嵌的 [`ErrorContext`]（如果有）。
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            LLMError::Network { ctx, .. }
            | LLMError::Timeout { ctx, .. }
            | LLMError::Auth { ctx, .. }
            | LLMError::RateLimit { ctx, .. }
            | LLMError::Safety { ctx, .. }
            | LLMError::ContextOverflow { ctx, .. }
            | LLMError::Parse { ctx, .. }
            | LLMError::Provider { ctx, .. } => Some(ctx),
            _ => None,
        }
    }

    /// 便捷方法：这个错误能否原样重试。
    pub fn is_retryable(&self) -> bool {
        self.context().map(|c| c.retryable).unwrap_or(false)
    }

    pub fn config(msg: impl fmt::Display) -> Self {
        LLMError::Config {
            message: msg.to_string(),
        }
    }

    pub fn validation(msg: impl fmt::Display) -> Self {
        LLMError::Validation {
            message: msg.to_string(),
        }
    }

    pub fn parse(msg: impl fmt::Display) -> Self {
        LLMError::Parse {
            message: msg.to_string(),
            ctx: ErrorContext::default(),
        }
    }

    pub fn provider(msg: impl fmt::Display) -> Self {
        LLMError::Provider {
            message: msg.to_string(),
            ctx: ErrorContext::default(),
        }
    }

    pub fn unsupported(msg: impl fmt::Display) -> Self {
        LLMError::Unsupported {
            message: msg.to_string(),
        }
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> Self {
        LLMError::Parse {
            message: format!("json: {err}"),
            ctx: ErrorContext::default(),
        }
    }
}

impl From<reqwest::Error> for LLMError {
    fn from(err: reqwest::Error) -> Self {
        let status = err.status().map(|s| s.as_u16());
        let ctx = ErrorContext {
            http_status: status,
            retryable: err.is_timeout() || err.is_connect(),
            ..ErrorContext::default()
        };
        if err.is_timeout() {
            LLMError::Timeout { elapsed_ms: 0, ctx }
        } else {
            LLMError::Network {
                message: err.to_string(),
                ctx,
            }
        }
    }
}
