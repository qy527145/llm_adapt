//! Handler registry: maps `api_format` strings to the three handler kinds.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::{NonStreamResponseHandler, RequestHandler, StreamResponseHandler};
use crate::error::LLMError;

/// One registry slot: the three handlers that cooperate for a given protocol.
#[derive(Clone)]
pub struct HandlerEntry {
    pub request: Arc<dyn RequestHandler>,
    pub non_stream: Arc<dyn NonStreamResponseHandler>,
    pub stream: Arc<dyn StreamResponseHandler>,
}

/// Thread-safe registry of handlers keyed by `api_format`.
///
/// Cloning a registry is cheap (it holds an `Arc`). All mutation goes through
/// the inner `RwLock`.
#[derive(Default, Clone)]
pub struct HandlerRegistry {
    inner: Arc<RwLock<HashMap<String, HandlerEntry>>>,
}

impl HandlerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register / overwrite the entry for `api_format`.
    pub fn register(&self, api_format: impl Into<String>, entry: HandlerEntry) {
        let key = api_format.into();
        let mut guard = self.inner.write().expect("HandlerRegistry lock poisoned");
        guard.insert(key, entry);
    }

    /// Convenience: register a complete protocol from three concrete handlers.
    pub fn register_protocol<R, N, S>(
        &self,
        api_format: impl Into<String>,
        request: R,
        non_stream: N,
        stream: S,
    ) where
        R: RequestHandler + 'static,
        N: NonStreamResponseHandler + 'static,
        S: StreamResponseHandler + 'static,
    {
        self.register(
            api_format,
            HandlerEntry {
                request: Arc::new(request),
                non_stream: Arc::new(non_stream),
                stream: Arc::new(stream),
            },
        );
    }

    /// Get a clone of the entry (cheap — only Arcs). Errors with
    /// [`LLMError::HandlerNotFound`] if the key is unknown.
    pub fn get(&self, api_format: &str) -> Result<HandlerEntry, LLMError> {
        let guard = self.inner.read().expect("HandlerRegistry lock poisoned");
        guard
            .get(api_format)
            .cloned()
            .ok_or_else(|| LLMError::HandlerNotFound {
                api_format: api_format.to_string(),
            })
    }

    /// List all currently registered `api_format` keys.
    pub fn list(&self) -> Vec<String> {
        let guard = self.inner.read().expect("HandlerRegistry lock poisoned");
        let mut keys: Vec<String> = guard.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Number of registered protocols.
    pub fn len(&self) -> usize {
        self.inner.read().expect("HandlerRegistry lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
