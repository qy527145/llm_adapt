//! `llm-adapt preview` —— 把统一请求渲染成各家 provider 专属的
//! HTTP 形态，但不发送。

use anyhow::{Context, Result};
use clap::Args;
use llm_adapt_core::{
    CacheMarker, ChatRequest, Conversation, HttpMethod, HttpRequest, LLMClient, Protocol,
    ProtocolId, SystemPrompt, Turn, UserMessage,
};
use serde::Serialize;

use crate::commands::config_cmd::pick_profile;
use crate::config::store;
use crate::output::{OutputFormat, emit_json};

#[derive(Debug, Args)]
pub struct PreviewArgs {
    /// 使用的 profile；默认用 active 的那个。
    #[arg(long)]
    pub profile: Option<String>,
    /// 覆盖模型名（否则用 profile 的默认值）。
    #[arg(long, short)]
    pub model: Option<String>,
    /// 覆盖 api_format（handler key）。
    #[arg(long)]
    pub api_format: Option<String>,
    /// 用户 prompt 内容。
    #[arg(long, short)]
    pub prompt: String,
    /// 可选的 system 消息。
    #[arg(long)]
    pub system: Option<String>,
    /// 给 system prompt 挂一个 CacheMarker（`5m` 或 `1h`）。
    /// 仅对支持显式缓存的 handler（Anthropic）生效。
    #[arg(long, value_parser = ["5m", "1h"])]
    pub cache_system: Option<String>,
    /// 温度。
    #[arg(long)]
    pub temperature: Option<f32>,
    /// 最大输出 token 数（Anthropic 必填）。
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// 假装这是一个流式请求。
    #[arg(long, default_value_t = false)]
    pub stream: bool,
    /// 输出格式：`human`（默认）、`curl` 或 `json`。
    #[arg(long, default_value = "human")]
    pub format: String,
}

#[derive(Debug, Serialize)]
struct PreviewView {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

pub fn run(args: PreviewArgs, fmt: OutputFormat) -> Result<()> {
    let settings = store::load()?;
    let (profile_name, profile) = pick_profile(&settings, args.profile.as_deref())?;
    let api_format = args
        .api_format
        .clone()
        .unwrap_or_else(|| profile.api_format.clone());
    let model = args
        .model
        .clone()
        .or_else(|| profile.default_model.clone())
        .context("no model — pass --model or set default_model on the profile")?;

    let client = LLMClient::new(profile.to_client_config())?;
    let mut request = build_request(&api_format, &model, &args)?;
    request.stream = args.stream;

    let http = client.preview(&request)?;
    output_preview(&args.format, fmt, profile_name, &http)?;
    Ok(())
}

fn build_request(api_format: &str, model: &str, args: &PreviewArgs) -> Result<ChatRequest> {
    let mut conv = Conversation {
        system: args.system.as_ref().map(|s| {
            let mut sp = SystemPrompt::text(s);
            if let Some(ttl) = &args.cache_system {
                let marker = match ttl.as_str() {
                    "1h" => CacheMarker::ephemeral_1h(),
                    _ => CacheMarker::ephemeral_5m(),
                };
                sp = sp.with_cache(marker);
            }
            sp
        }),
        turns: vec![],
    };
    conv.push(Turn::User(UserMessage::text(&args.prompt)));

    let mut req = ChatRequest::new(model, api_format, conv);
    if let Some(t) = args.temperature {
        req.temperature = Some(t);
    }
    if let Some(m) = args.max_tokens {
        req.max_tokens = Some(m);
    } else if is_anthropic(api_format) {
        req.max_tokens = Some(1024);
    }
    Ok(req)
}

/// 协议名先归一，`anthropic` / `anthropic_v2` 一视同仁。
fn is_anthropic(api_format: &str) -> bool {
    ProtocolId::new(api_format).protocol() == Some(Protocol::Anthropic)
}

fn output_preview(
    format: &str,
    fmt: OutputFormat,
    profile: &str,
    http: &HttpRequest,
) -> Result<()> {
    let body_json: serde_json::Value =
        serde_json::from_slice(&http.body).unwrap_or(serde_json::Value::Null);
    if matches!(fmt, OutputFormat::Json) {
        let view = PreviewView {
            method: format!("{:?}", http.method).to_uppercase(),
            url: http.url.clone(),
            headers: http.headers.clone(),
            body: body_json,
        };
        emit_json(&view)?;
        return Ok(());
    }
    match format {
        "curl" => println!("{}", to_curl(http)),
        "json" => {
            let view = PreviewView {
                method: format!("{:?}", http.method).to_uppercase(),
                url: http.url.clone(),
                headers: http.headers.clone(),
                body: body_json,
            };
            emit_json(&view)?;
        }
        _ => {
            println!("# preview using profile [{profile}]");
            println!("{} {}", method_str(http.method), http.url);
            for (k, v) in &http.headers {
                let v_shown = if k.eq_ignore_ascii_case("authorization")
                    || k.eq_ignore_ascii_case("x-api-key")
                {
                    crate::config::mask_secret(v)
                } else {
                    v.clone()
                };
                println!("{k}: {v_shown}");
            }
            println!();
            println!("{}", serde_json::to_string_pretty(&body_json)?);
        }
    }
    Ok(())
}

fn method_str(m: HttpMethod) -> &'static str {
    match m {
        HttpMethod::Get => "GET",
        HttpMethod::Post => "POST",
        HttpMethod::Put => "PUT",
        HttpMethod::Delete => "DELETE",
        HttpMethod::Patch => "PATCH",
    }
}

fn to_curl(http: &HttpRequest) -> String {
    let mut s = format!(
        "curl -X {} {}",
        method_str(http.method),
        shell_quote(&http.url)
    );
    for (k, v) in &http.headers {
        s.push_str(&format!(" \\\n  -H {}", shell_quote(&format!("{k}: {v}"))));
    }
    if !http.body.is_empty() {
        let body = std::str::from_utf8(&http.body).unwrap_or("");
        s.push_str(&format!(" \\\n  --data-raw {}", shell_quote(body)));
    }
    s
}

fn shell_quote(s: &str) -> String {
    // 单引号对一切都安全，唯独字面单引号本身例外。
    if s.contains('\'') {
        let escaped = s.replace('\'', "'\\''");
        format!("'{escaped}'")
    } else {
        format!("'{s}'")
    }
}
