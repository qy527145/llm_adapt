//! `llm-adapt` —— `llm_adapt` 的统一 CLI / TUI / Web 入口。
//!
//! 第一阶段先落地 CLI 子命令（`config`、`preview`、`call`、`handlers`）；
//! `tui` 与 `web` 是占位，实现留到第二阶段。

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
    version
)]
struct Cli {
    /// 在 stdout 上输出机器可读的 JSON，而非人类可读的散文。
    #[arg(long, global = true, default_value_t = false)]
    json: bool,

    /// 提高日志详细度（-v、-vv）。
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// 管理 profile（~/.llm-adapt/config.toml）。
    Config {
        #[command(subcommand)]
        cmd: config_cmd::ConfigCmd,
    },
    /// 只渲染请求而不发送，便于核对 wire 格式。
    Preview(preview::PreviewArgs),
    /// 发送请求并打印响应（流式或一次性）。
    Call(call::CallArgs),
    /// 查看已注册的协议 handler 与模型能力。
    Handlers {
        #[command(subcommand)]
        cmd: handlers::HandlersCmd,
    },
    /// （下一阶段）启动交互式 TUI。
    Tui,
    /// （下一阶段）启动 Web 管理面板。
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
