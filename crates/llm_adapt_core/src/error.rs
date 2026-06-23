//! Unified error type for all `llm_adapt` operations.
//!
//! Every fallible API in the core crate returns [`LLMError`]. Errors are classified
//! into stable variants so callers can match on category, while carrying optional
//! metadata (request id, HTTP status, retryability hint) for richer diagnostics.

use std::fmt;

use thiserror::Error;

/// Extra context attached to most error variants. All fields are optional so that
/// errors can be constructed at any layer regardless of how much information is
/// available.
#[derive(Debug, Default, Clone)]
pub struct ErrorContext {
    /// Provider-supplied request id, if the response carried one.
    pub request_id: Option<String>,
    /// HTTP status code that triggered the error, if applicable.
    pub http_status: Option<u16>,
    /// Whether the operation is safe to retry with the same payload.
    pub retryable: bool,
    /// Vendor-specific error code (string form to keep it format-agnostic).
    pub vendor_code: Option<String>,
}

impl ErrorContext {
    pub fn retryable() -> Self {
        Self { retryable: true, ..Self::default() }
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

/// All errors surfaced by `llm_adapt_core`.
#[derive(Debug, Error)]
pub enum LLMError {
    /// Misconfiguration detected before any request is built (missing key, bad URL, etc.).
    #[error("configuration error: {message}")]
    Config { message: String },

    /// The supplied [`crate::types::ChatRequest`] did not satisfy validation rules.
    #[error("validation error: {message}")]
    Validation { message: String },

    /// Network/transport level failure (DNS, TLS, broken pipe, ...).
    #[error("network error: {message}")]
    Network { message: String, ctx: ErrorContext },

    /// Request exceeded the configured timeout.
    #[error("request timed out after {elapsed_ms} ms")]
    Timeout { elapsed_ms: u128, ctx: ErrorContext },

    /// Authentication/authorisation failure (HTTP 401/403).
    #[error("authentication error: {message}")]
    Auth { message: String, ctx: ErrorContext },

    /// Rate limit hit (HTTP 429). `retry_after_ms` mirrors the `Retry-After` header
    /// when the provider sent one.
    #[error("rate limited: {message}")]
    RateLimit {
        message: String,
        retry_after_ms: Option<u64>,
        ctx: ErrorContext,
    },

    /// Provider rejected the request on safety grounds.
    #[error("safety violation: {message}")]
    Safety { message: String, ctx: ErrorContext },

    /// Context window was exceeded.
    #[error("context overflow: {message}")]
    ContextOverflow { message: String, ctx: ErrorContext },

    /// A response body could not be parsed into a stable structure.
    #[error("parse error: {message}")]
    Parse { message: String, ctx: ErrorContext },

    /// Provider returned a non-success response that does not fit any other category.
    #[error("provider error: {message}")]
    Provider { message: String, ctx: ErrorContext },

    /// No handler was registered for the requested `api_format`.
    #[error("no handler registered for api_format = {api_format}")]
    HandlerNotFound { api_format: String },
}

impl LLMError {
    /// Returns the embedded [`ErrorContext`], if any.
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

    /// Convenience: whether this error is safe to retry as-is.
    pub fn is_retryable(&self) -> bool {
        self.context().map(|c| c.retryable).unwrap_or(false)
    }

    pub fn config(msg: impl fmt::Display) -> Self {
        LLMError::Config { message: msg.to_string() }
    }

    pub fn validation(msg: impl fmt::Display) -> Self {
        LLMError::Validation { message: msg.to_string() }
    }

    pub fn parse(msg: impl fmt::Display) -> Self {
        LLMError::Parse { message: msg.to_string(), ctx: ErrorContext::default() }
    }

    pub fn provider(msg: impl fmt::Display) -> Self {
        LLMError::Provider { message: msg.to_string(), ctx: ErrorContext::default() }
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
            LLMError::Network { message: err.to_string(), ctx }
        }
    }
}
