//! Handler trait system.
//!
//! Three independent, object-safe traits split the work of talking to a provider:
//!
//! * [`RequestHandler`] — translates a unified [`ChatRequest`] into a vendor-shaped
//!   HTTP request description ([`HttpRequest`]).
//! * [`NonStreamResponseHandler`] — turns a raw response body into [`ChatResponse`].
//! * [`StreamResponseHandler`] — turns a byte stream into a stream of [`StreamChunk`]s.
//!
//! Combined with [`registry::HandlerRegistry`], this lets the runtime dispatch on
//! `(api_format, stream)` without compile-time knowledge of every provider.

use std::pin::Pin;

use bytes::Bytes;
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use crate::error::LLMError;
use crate::http::ClientConfig;
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

pub mod registry;
pub use registry::{HandlerEntry, HandlerRegistry};

/// HTTP method used by the request description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

/// Provider-neutral description of an HTTP request. Handlers produce these;
/// the HTTP executor turns them into real network calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Render the request body as UTF-8 text for debug/preview purposes.
    /// Returns `None` if the body is empty or not valid UTF-8.
    pub fn body_as_str(&self) -> Option<&str> {
        if self.body.is_empty() {
            return None;
        }
        std::str::from_utf8(&self.body).ok()
    }
}

/// Raw, undecoded response from the HTTP layer. Hooks receive this for logging
/// and debugging.
#[derive(Debug, Clone)]
pub struct RawHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

/// Stream of raw response bytes coming from the network.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, LLMError>> + Send>>;
/// Stream of parsed [`StreamChunk`]s ready for application code.
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>;

/// Builds an [`HttpRequest`] from a unified [`ChatRequest`].
///
/// Implementations are pure functions of the request + client configuration and
/// must not perform IO. Object safety is required: handlers are stored in the
/// registry as `Arc<dyn RequestHandler>`.
pub trait RequestHandler: Send + Sync {
    fn build_request(
        &self,
        request: &ChatRequest,
        config: &ClientConfig,
    ) -> Result<HttpRequest, LLMError>;
}

/// Parses a non-streaming response body into [`ChatResponse`].
pub trait NonStreamResponseHandler: Send + Sync {
    fn parse_response(&self, body: Bytes) -> Result<ChatResponse, LLMError>;
}

/// Wraps a [`ByteStream`] into a [`ChunkStream`] of stable [`StreamChunk`] events.
pub trait StreamResponseHandler: Send + Sync {
    fn parse_stream(&self, stream: ByteStream) -> ChunkStream;
}
