//! `cargo run -p llm_adapt_core --example basic_anthropic`
//!
//! Requires `ANTHROPIC_API_KEY` in the environment.

use llm_adapt_core::{
    CacheMarker, ChatRequest, ClientConfig, Conversation, LLMClient, SystemPrompt, Turn,
    UserMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
    let client = LLMClient::new(ClientConfig::new("https://api.anthropic.com", api_key))?;

    // Mark the system prompt for caching to demonstrate the per-block marker.
    let conv = Conversation {
        system: Some(SystemPrompt::text("You are concise.").with_cache(CacheMarker::ephemeral_5m())),
        turns: vec![Turn::User(UserMessage::text(
            "Summarise the difference between Rust's `&str` and `String` in one sentence.",
        ))],
    };
    let mut request = ChatRequest::anthropic("claude-3-5-haiku-20241022", conv);
    request.max_tokens = Some(256);

    let response = client.chat(&request).await?;
    println!("{}", response.text());
    eprintln!(
        "[usage] in={} out={} cache_r={} cache_w_5m={} cache_w_1h={}",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.cache.read_tokens,
        response.usage.cache.write.ephemeral_5m,
        response.usage.cache.write.ephemeral_1h,
    );
    Ok(())
}
