//! `llm-adapt` — unified CLI / TUI / Web entry point for `llm_adapt`.
//!
//! Phase 1 ships CLI subcommands (`config`, `preview`, `call`, `handlers`);
//! `tui` and `web` are stubs whose implementations land in phase 2.

mod commands;
mod config;
mod output;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::commands::{call, config_cmd, handlers, preview, tui, web};
use crate::output::OutputFormat;

#[derive(Debug, Parser)]
#[command(
    name = "llm-adapt",
    about = "Unified entry point for the llm_adapt LLM API abstraction layer",
    version,
)]
struct Cli {
    /// Emit machine-readable JSON on stdout instead of human prose.
    #[arg(long, global = true, default_value_t = false)]
    json: bool,

    /// Increase log verbosity (-v, -vv).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Manage profiles (~/.llm-adapt/config.toml).
    Config {
        #[command(subcommand)]
        cmd: config_cmd::ConfigCmd,
    },
    /// Render a request without sending it. Useful for vetting the wire format.
    Preview(preview::PreviewArgs),
    /// Send a request and print the response (streaming or one-shot).
    Call(call::CallArgs),
    /// Inspect registered protocol handlers and model capabilities.
    Handlers {
        #[command(subcommand)]
        cmd: handlers::HandlersCmd,
    },
    /// (next phase) Launch the interactive TUI.
    Tui,
    /// (next phase) Launch the Web management panel.
    Web(web::WebArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let fmt = OutputFormat::from_flag(cli.json);

    match cli.command {
        Command::Config { cmd } => config_cmd::run(cmd, fmt),
        Command::Preview(args) => preview::run(args, fmt),
        Command::Call(args) => call::run(args, fmt),
        Command::Handlers { cmd } => handlers::run(cmd, fmt),
        Command::Tui => tui::run(fmt),
        Command::Web(args) => web::run(args, fmt),
    }
}

fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        _ => tracing::Level::DEBUG,
    };
    let _ = tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
