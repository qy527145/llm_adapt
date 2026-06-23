//! `llm-adapt handlers list`

use anyhow::Result;
use clap::Subcommand;
use llm_adapt_core::{HandlerRegistry, ModelCapabilityTable};
use serde::Serialize;

use crate::output::{emit_json, OutputFormat};

#[derive(Debug, Subcommand)]
pub enum HandlersCmd {
    /// List all registered protocol handlers.
    List,
    /// Show capability metadata for each (api_format, model) pair we know about.
    Capabilities,
}

#[derive(Debug, Serialize)]
struct HandlerView {
    api_format: String,
}

pub fn run(cmd: HandlersCmd, fmt: OutputFormat) -> Result<()> {
    match cmd {
        HandlersCmd::List => list(fmt),
        HandlersCmd::Capabilities => capabilities(fmt),
    }
}

fn list(fmt: OutputFormat) -> Result<()> {
    let registry = HandlerRegistry::new();
    llm_adapt_core::handlers::register_defaults(&registry);
    let views: Vec<HandlerView> = registry
        .list()
        .into_iter()
        .map(|k| HandlerView { api_format: k })
        .collect();
    match fmt {
        OutputFormat::Json => emit_json(&serde_json::json!({"handlers": views}))?,
        OutputFormat::Human => {
            println!("# registered handlers ({total})", total = views.len());
            for v in &views {
                println!("  • {}", v.api_format);
            }
        }
    }
    Ok(())
}

fn capabilities(fmt: OutputFormat) -> Result<()> {
    let table = ModelCapabilityTable::with_defaults();
    let rows = table.list();
    match fmt {
        OutputFormat::Json => emit_json(&serde_json::json!({
            "capabilities": rows.iter().map(|((api, m), c)| serde_json::json!({
                "api_format": api,
                "model": m,
                "capabilities": c,
            })).collect::<Vec<_>>(),
        }))?,
        OutputFormat::Human => {
            for ((api, model), caps) in &rows {
                println!("[{api}] {model}");
                println!(
                    "  ctx={ctx}  out={out}  tools={tools}  thinking={thinking}  vision={vision}  cache={cache}",
                    ctx = caps.context_window,
                    out = caps.max_output_tokens,
                    tools = caps.supports_tools,
                    thinking = caps.supports_thinking,
                    vision = caps.supports_vision,
                    cache = caps.supports_prompt_cache,
                );
            }
        }
    }
    Ok(())
}
