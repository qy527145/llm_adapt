//! `llm-adapt preview` — render a unified request into its provider-specific
//! HTTP shape without sending it.

use anyhow::{Context, Result};
use clap::Args;
use llm_adapt_core::{
    CacheMarker, ChatRequest, Conversation, HttpMethod, HttpRequest, LLMClient, SystemPrompt,
    Turn, UserMessage,
};
use serde::Serialize;

use crate::commands::config_cmd::pick_profile;
use crate::config::store;
use crate::output::{emit_json, OutputFormat};

#[derive(Debug, Args)]
pub struct PreviewArgs {
    /// Profile to use; defaults to the active one.
    #[arg(long)]
    pub profile: Option<String>,
    /// Override the model name (otherwise uses profile default).
    #[arg(long, short)]
    pub model: Option<String>,
    /// Override the api_format (handler key).
    #[arg(long)]
    pub api_format: Option<String>,
    /// User prompt content.
    #[arg(long, short)]
    pub prompt: String,
    /// Optional system message.
    #[arg(long)]
    pub system: Option<String>,
    /// Attach a CacheMarker to the system prompt (`5m` or `1h`).
    /// Only honoured by handlers that support explicit caching (Anthropic).
    #[arg(long, value_parser = ["5m", "1h"])]
    pub cache_system: Option<String>,
    /// Temperature.
    #[arg(long)]
    pub temperature: Option<f32>,
    /// Max output tokens (required for Anthropic).
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Pretend the request is a streaming one.
    #[arg(long, default_value_t = false)]
    pub stream: bool,
    /// Output format: `human` (default), `curl`, or `json`.
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
    } else if api_format == "anthropic_v2" {
        req.max_tokens = Some(1024);
    }
    Ok(req)
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
                let v_shown = if k.eq_ignore_ascii_case("authorization") || k.eq_ignore_ascii_case("x-api-key") {
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
    let mut s = format!("curl -X {} {}", method_str(http.method), shell_quote(&http.url));
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
    // Single quotes are safe for everything except a literal single quote.
    if s.contains('\'') {
        let escaped = s.replace('\'', "'\\''");
        format!("'{escaped}'")
    } else {
        format!("'{s}'")
    }
}
