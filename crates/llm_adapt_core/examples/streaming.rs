//! `cargo run -p llm_adapt_core --example streaming`
//!
//! Demonstrates the unified streaming API against either OpenAI or Anthropic
//! depending on which env var is set.

use std::io::Write;

use futures_util::StreamExt;
use llm_adapt_core::{ChatRequest, ClientConfig, Conversation, LLMClient, StreamChunk};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conv = Conversation::single_user("Count from one to five with a space between numbers.");

    let (base_url, key_var, mut request) = match std::env::var("OPENAI_API_KEY") {
        Ok(_) => (
            "https://api.openai.com",
            "OPENAI_API_KEY",
            ChatRequest::openai("gpt-4o-mini", conv),
        ),
        Err(_) => (
            "https://api.anthropic.com",
            "ANTHROPIC_API_KEY",
            ChatRequest::anthropic("claude-3-5-haiku-20241022", conv),
        ),
    };
    let api_key = std::env::var(key_var)
        .unwrap_or_else(|_| panic!("set {key_var} or OPENAI_API_KEY"));
    request.stream = true;

    let client = LLMClient::new(ClientConfig::new(base_url, api_key))?;
    let mut stream = client.chat_stream(&request).await?;

    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk::TextDelta { text } => {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            StreamChunk::Finish { reason } => {
                println!("\n[finish: {reason:?}]");
                break;
            }
            StreamChunk::Usage { usage } => {
                eprintln!(
                    "[usage] in={} out={} cache_r={} cache_w={}",
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache.read_tokens,
                    usage.cache.write.total,
                );
            }
            _ => {}
        }
    }
    Ok(())
}
