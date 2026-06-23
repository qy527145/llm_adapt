//! `cargo run -p llm_adapt_core --example tool_calling`
//!
//! Walks one full tool-use round-trip: ask → tool call → tool result → final answer.

use llm_adapt_core::{
    AssistantBlock, ChatRequest, ClientConfig, Conversation, FinishReason, LLMClient,
    ToolDefinition, ToolResultContent, Turn, UserBlock, UserMessage,
};
use serde_json::json;

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("set OPENAI_API_KEY");
    let client = LLMClient::new(ClientConfig::new("https://api.openai.com", api_key))?;

    let mut request = ChatRequest::openai(
        "gpt-4o-mini",
        Conversation::single_user("What is the weather like in Tokyo?"),
    );
    request.tools = vec![weather_tool()];

    // Round 1 — expect a tool call.
    let resp = client.chat(&request).await?;
    assert_eq!(resp.finish_reason, FinishReason::ToolCall, "expected a tool call");

    let (tool_id, tool_name, args_str) = resp
        .message
        .blocks
        .iter()
        .find_map(|b| match b {
            AssistantBlock::ToolCall { id, name, arguments, .. } => {
                Some((id.clone(), name.clone(), arguments.clone()))
            }
            _ => None,
        })
        .expect("no tool call in response");
    println!("model wants to call {tool_name}({args_str}) with id={tool_id}");

    // Round 2 — echo the assistant turn back, then a structured tool result.
    request
        .conversation
        .push(Turn::Assistant(resp.message.clone()));

    let mut tool_turn = UserMessage::default();
    tool_turn.push(UserBlock::ToolResult {
        call_id: tool_id,
        content: ToolResultContent::Json(json!({"temperature_c": 22, "condition": "Sunny"})),
        is_error: false,
        cache: None,
    });
    request.conversation.push(Turn::User(tool_turn));

    let final_resp = client.chat(&request).await?;
    println!("\nfinal answer:\n{}", final_resp.text());
    Ok(())
}
