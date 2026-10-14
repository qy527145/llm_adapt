//! 各 CLI / TUI / Web 命令共用的配置模型。
//!
//! 配置落盘于 `~/.llm-adapt/config.toml`（可用 `LLM_ADAPT_HOME` 覆盖）。
//! API key 可用 `env:VAR_NAME` 语法引用环境变量，由 [`Profile::resolved_api_key`]
//! 惰性解析。

pub mod store;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// 顶层落盘配置。
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// 未提供 `--profile` 时命令使用的 profile 名。
    #[serde(default)]
    pub active_profile: Option<String>,
    /// 全部已知 profile，按名称索引。`BTreeMap` 保证落盘顺序稳定。
    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
    /// `llm-adapt web`（下一阶段的 Web 面板）的默认端口。
    #[serde(default = "default_web_port")]
    pub web_port: u16,
}

fn default_web_port() -> u16 { 8787 }

impl Settings {
    /// 返回 active profile（若只有一个，就返回那一个）。
    pub fn active(&self) -> Option<(&str, &Profile)> {
        let name = self.active_profile.as_deref().or_else(|| {
            if self.profiles.len() == 1 {
                self.profiles.keys().next().map(|s| s.as_str())
            } else {
                None
            }
        })?;
        self.profiles.get(name).map(|p| (name, p))
    }
}

/// 一个具名环境（如 `dev`、`prod`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub base_url: String,
    /// API key。用 `env:VAR_NAME` 从环境变量读取。
    pub api_key: String,
    /// 省略 `--api-format` 时使用的默认 `api_format`（handler key）。
    #[serde(default = "default_api_format")]
    pub api_format: String,
    /// 默认模型名。
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub proxy: Option<String>,
    #[serde(default = "default_connect_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_request_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_stream_secs")]
    pub stream_timeout_secs: u64,
    #[serde(default = "default_true")]
    pub verify_ssl: bool,
    #[serde(default)]
    pub default_headers: BTreeMap<String, String>,
}

fn default_api_format() -> String { "openai_compat".into() }
fn default_connect_secs() -> u64 { 10 }
fn default_request_secs() -> u64 { 60 }
fn default_stream_secs() -> u64 { 60 }
fn default_true() -> bool { true }

impl Default for Profile {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com".into(),
            api_key: String::new(),
            api_format: default_api_format(),
            default_model: None,
            proxy: None,
            connect_timeout_secs: default_connect_secs(),
            request_timeout_secs: default_request_secs(),
            stream_timeout_secs: default_stream_secs(),
            verify_ssl: true,
            default_headers: BTreeMap::new(),
        }
    }
}

impl Profile {
    /// 把 `env:VAR_NAME` 占位符解析成实际值。
    pub fn resolved_api_key(&self) -> String {
        if let Some(var) = self.api_key.strip_prefix("env:") {
            std::env::var(var).unwrap_or_default()
        } else {
            self.api_key.clone()
        }
    }

    /// 用于展示的掩码渲染。
    pub fn masked_api_key(&self) -> String {
        let raw = if self.api_key.starts_with("env:") {
            self.api_key.clone()
        } else {
            self.resolved_api_key()
        };
        mask_secret(&raw)
    }

    /// 由该 profile 构造运行时的 [`llm_adapt_core::ClientConfig`]。
    pub fn to_client_config(&self) -> llm_adapt_core::ClientConfig {
        use std::time::Duration;

        let proxy = self
            .proxy
            .as_ref()
            .filter(|p| !p.is_empty())
            .map(|p| llm_adapt_core::ProxyConfig { url: p.clone() });

        llm_adapt_core::ClientConfig {
            base_url: self.base_url.clone(),
            path: None,
            api_key: self.resolved_api_key(),
            proxy,
            connect_timeout: Duration::from_secs(self.connect_timeout_secs),
            request_timeout: Duration::from_secs(self.request_timeout_secs),
            stream_timeout: Duration::from_secs(self.stream_timeout_secs),
            verify_ssl: self.verify_ssl,
            default_headers: self
                .default_headers
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

/// 只保留 secret 末 4 个字符，其余打码，同时保留大致长度信息。
pub fn mask_secret(s: &str) -> String {
    if s.is_empty() {
        return "<unset>".into();
    }
    if s.len() <= 8 {
        return "*".repeat(s.len());
    }
    let tail: String = s.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{}…{}", &s[..2], tail)
}
