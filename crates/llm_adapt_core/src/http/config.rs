//! HTTP-layer configuration shared across all handlers.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Proxy configuration. Both HTTP(S) and SOCKS5 proxies are accepted; the
/// scheme in `url` (e.g. `socks5://...`) determines which is used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub url: String,
}

/// Top-level HTTP/client configuration. Handlers receive a borrowed
/// `&ClientConfig` when building requests so they can pick up `base_url`,
/// `api_key`, and global headers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Base URL for the provider (e.g. `https://api.openai.com`). Handlers
    /// append their own path (e.g. `/v1/chat/completions`).
    pub base_url: String,
    /// API key / bearer token. Some handlers may also pull this from
    /// `default_headers` if the auth scheme is non-standard.
    pub api_key: String,
    /// Optional proxy.
    #[serde(default)]
    pub proxy: Option<ProxyConfig>,
    /// Connect-phase timeout.
    #[serde(default = "default_connect_timeout", with = "humantime_serde_compat")]
    pub connect_timeout: Duration,
    /// Total request timeout (non-streaming).
    #[serde(default = "default_request_timeout", with = "humantime_serde_compat")]
    pub request_timeout: Duration,
    /// Inactivity timeout while a stream is open.
    #[serde(default = "default_stream_timeout", with = "humantime_serde_compat")]
    pub stream_timeout: Duration,
    /// Whether to verify TLS server certificates. Default true.
    #[serde(default = "default_true")]
    pub verify_ssl: bool,
    /// Headers attached to every request before handler-specific headers.
    #[serde(default)]
    pub default_headers: Vec<(String, String)>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            proxy: None,
            connect_timeout: default_connect_timeout(),
            request_timeout: default_request_timeout(),
            stream_timeout: default_stream_timeout(),
            verify_ssl: true,
            default_headers: Vec::new(),
        }
    }
}

impl ClientConfig {
    /// Convenience builder for tests and quick examples.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            ..Self::default()
        }
    }
}

fn default_connect_timeout() -> Duration { Duration::from_secs(10) }
fn default_request_timeout() -> Duration { Duration::from_secs(60) }
fn default_stream_timeout() -> Duration { Duration::from_secs(60) }
fn default_true() -> bool { true }

/// Minimal `Duration` ⇄ seconds (u64) serde helper.
///
/// We don't pull in `humantime_serde` to keep dependencies lean; the config
/// stores plain integer seconds.
mod humantime_serde_compat {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
