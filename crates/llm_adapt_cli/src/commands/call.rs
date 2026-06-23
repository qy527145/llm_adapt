//! `llm-adapt call` — actually send a request and print the response.

use anyhow::{Context, Result};
use clap::Args;
use futures_util::StreamExt;
use llm_adapt_core::{
    CacheMarker, ChatRequest, Conversation, LLMClient, StreamChunk, SystemPrompt, Turn,
    UserMessage,
};
use serde::Serialize;

use crate::commands::config_cmd::pick_profile;
use crate::config::store;
use crate::output::{emit_json, OutputFormat};

#[derive(Debug, Args)]
pub struct CallArgs {
    #[arg(long)]
    pub profile: Option<String>,
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long)]
    pub api_format: Option<String>,
    #[arg(long, short)]
    pub prompt: String,
    #[arg(long)]
    pub system: Option<String>,
    /// Attach a CacheMarker to the system prompt (`5m` or `1h`).
    /// Only honoured by handlers that support explicit caching (Anthropic).
    #[arg(long, value_parser = ["5m", "1h"])]
    pub cache_system: Option<String>,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Use the streaming endpoint.
    #[arg(long, default_value_t = false)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
struct StreamSummary {
    text: String,
    thinking: String,
    thinking_signature: String,
    finish: Option<String>,
}

pub fn run(args: CallArgs, fmt: OutputFormat) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move { run_async(args, fmt).await })
}

async fn run_async(args: CallArgs, fmt: OutputFormat) -> Result<()> {
    let settings = store::load()?;
    let (_profile_name, profile) = pick_profile(&settings, args.profile.as_deref())?;
    let api_format = args.api_format.clone().unwrap_or_else(|| profile.api_format.clone());
    let model = args
        .model
        .clone()
        .or_else(|| profile.default_model.clone())
        .context("no model — pass --model or set default_model on the profile")?;

    let client = LLMClient::new(profile.to_client_config())?;

    let conversation = build_conversation(&args);
    let mut req = ChatRequest::new(&model, &api_format, conversation);
    if let Some(t) = args.temperature { req.temperature = Some(t); }
    if let Some(m) = args.max_tokens { req.max_tokens = Some(m); }
    if req.max_tokens.is_none() && api_format == "anthropic_v2" {
        req.max_tokens = Some(1024);
    }

    if args.stream {
        req.stream = true;
        let mut stream = client.chat_stream(&req).await?;
        let mut text = String::new();
        let mut thinking = String::new();
        let mut thinking_sig = String::new();
        let mut finish: Option<String> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            match &chunk {
                StreamChunk::TextDelta { text: t } => {
                    if matches!(fmt, OutputFormat::Human) {
                        print!("{t}");
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    }
                    text.push_str(t);
                }
                StreamChunk::ThinkingDelta { text: t, signature_delta } => {
                    thinking.push_str(t);
                    if let Some(s) = signature_delta {
                        thinking_sig.push_str(s);
                    }
                }
                StreamChunk::Finish { reason } => finish = Some(format!("{reason:?}")),
                _ => {}
            }
        }
        match fmt {
            OutputFormat::Human => {
                println!();
                if !thinking.is_empty() {
                    eprintln!("\n[thinking] {thinking}");
                }
                if let Some(r) = finish {
                    eprintln!("[finish] {r}");
                }
            }
            OutputFormat::Json => emit_json(&StreamSummary {
                text,
                thinking,
                thinking_signature: thinking_sig,
                finish,
            })?,
        }
    } else {
        let resp = client.chat(&req).await?;
        match fmt {
            OutputFormat::Human => {
                println!("{}", resp.text());
                let thinking = resp.message.thinking_content();
                if !thinking.is_empty() {
                    eprintln!("\n[thinking]\n{thinking}");
                }
                eprintln!(
                    "\n[finish={reason:?}] in={inp} out={out} \
                     cache_r={cr} cache_w_5m={cw5} cache_w_1h={cw1} latency={lat}ms",
                    reason = resp.finish_reason,
                    inp = resp.usage.input_tokens,
                    out = resp.usage.output_tokens,
                    cr = resp.usage.cache.read_tokens,
                    cw5 = resp.usage.cache.write.ephemeral_5m,
                    cw1 = resp.usage.cache.write.ephemeral_1h,
                    lat = resp.latency_ms,
                );
            }
            OutputFormat::Json => emit_json(&resp)?,
        }
    }
    Ok(())
}

fn build_conversation(args: &CallArgs) -> Conversation {
    let system = args.system.as_ref().map(|s| {
        let mut sp = SystemPrompt::text(s);
        if let Some(ttl) = &args.cache_system {
            let marker = match ttl.as_str() {
                "1h" => CacheMarker::ephemeral_1h(),
                _ => CacheMarker::ephemeral_5m(),
            };
            sp = sp.with_cache(marker);
        }
        sp
    });
    Conversation {
        system,
        turns: vec![Turn::User(UserMessage::text(&args.prompt))],
    }
}
