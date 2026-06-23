//! Configuration model shared between every CLI / TUI / Web command.
//!
//! On disk, configuration lives at `~/.llm-adapt/config.toml` (override via
//! `LLM_ADAPT_HOME`). API keys may reference environment variables using the
//! `env:VAR_NAME` syntax, which gets resolved lazily by [`Profile::resolved_api_key`].

pub mod store;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level on-disk config.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Name of the profile that commands use when `--profile` is not supplied.
    #[serde(default)]
    pub active_profile: Option<String>,
    /// All known profiles, keyed by name. `BTreeMap` keeps the on-disk order stable.
    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
    /// Default port for `llm-adapt web` (the next-phase Web panel).
    #[serde(default = "default_web_port")]
    pub web_port: u16,
}

fn default_web_port() -> u16 { 8787 }

impl Settings {
    /// Return the active profile (or the only one, when there's only one).
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

/// One named environment (e.g. `dev`, `prod`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub base_url: String,
    /// API key. Use `env:VAR_NAME` to read from an environment variable.
    pub api_key: String,
    /// Default `api_format` (handler key) used when `--api-format` is omitted.
    #[serde(default = "default_api_format")]
    pub api_format: String,
    /// Default model name.
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
    /// Resolve `env:VAR_NAME` placeholders to the actual value.
    pub fn resolved_api_key(&self) -> String {
        if let Some(var) = self.api_key.strip_prefix("env:") {
            std::env::var(var).unwrap_or_default()
        } else {
            self.api_key.clone()
        }
    }

    /// Masked rendering for display purposes.
    pub fn masked_api_key(&self) -> String {
        let raw = if self.api_key.starts_with("env:") {
            self.api_key.clone()
        } else {
            self.resolved_api_key()
        };
        mask_secret(&raw)
    }

    /// Build a runtime [`llm_adapt_core::ClientConfig`] from this profile.
    pub fn to_client_config(&self) -> llm_adapt_core::ClientConfig {
        use std::time::Duration;

        let proxy = self
            .proxy
            .as_ref()
            .filter(|p| !p.is_empty())
            .map(|p| llm_adapt_core::ProxyConfig { url: p.clone() });

        llm_adapt_core::ClientConfig {
            base_url: self.base_url.clone(),
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

/// Mask all but the last 4 chars of a secret, preserving rough length info.
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
