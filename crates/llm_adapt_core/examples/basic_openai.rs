//! `cargo run -p llm_adapt_core --example basic_openai`
//!
//! Requires `OPENAI_API_KEY` in the environment.

use llm_adapt_core::{ChatRequest, ClientConfig, Conversation, LLMClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let client = LLMClient::new(ClientConfig::new("https://api.openai.com", api_key))?;

    let request = ChatRequest::openai(
        "gpt-4o-mini",
        Conversation::single_user("Give me a one-line Rust haiku."),
    );
    let response = client.chat(&request).await?;

    println!("{}", response.text());
    eprintln!(
        "[usage] in={} out={} cache_r={} latency={}ms",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.cache.read_tokens,
        response.latency_ms,
    );
    Ok(())
}
