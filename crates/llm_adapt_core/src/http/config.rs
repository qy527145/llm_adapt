//! 各 handler 共用的 HTTP 层配置。

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// 代理配置。HTTP(S) 与 SOCKS5 代理都支持；
/// 用哪个由 `url` 里的 scheme（如 `socks5://...`）决定。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub url: String,
}

/// 顶层 HTTP/客户端配置。handler 在构造请求时拿到借用的 `&ClientConfig`，
/// 从中取 `base_url`、`api_key` 与全局 headers。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// 提供商的基础 URL（如 `https://api.openai.com`）。handler 自行拼接
    /// 各自的路径（如 `/v1/chat/completions`）。
    pub base_url: String,
    /// API key / bearer token。若认证方式非标准，某些 handler 也可能
    /// 从 `default_headers` 里取。
    pub api_key: String,
    /// 可选代理。
    #[serde(default)]
    pub proxy: Option<ProxyConfig>,
    /// 建立连接阶段的超时。
    #[serde(default = "default_connect_timeout", with = "humantime_serde_compat")]
    pub connect_timeout: Duration,
    /// 请求总超时（非流式）。
    #[serde(default = "default_request_timeout", with = "humantime_serde_compat")]
    pub request_timeout: Duration,
    /// 流打开期间的空闲超时。
    #[serde(default = "default_stream_timeout", with = "humantime_serde_compat")]
    pub stream_timeout: Duration,
    /// 是否校验 TLS 服务端证书。默认 true。
    #[serde(default = "default_true")]
    pub verify_ssl: bool,
    /// 附加到每个请求上的 headers，排在 handler 专属 headers 之前。
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
    /// 给测试与快速示例用的便捷构造。
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

/// 极简的 `Duration` ⇄ 秒（u64）serde 辅助。
///
/// 不引入 `humantime_serde` 以保持依赖精简；配置里存的就是纯整数秒。
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
