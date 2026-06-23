//! Placeholder for the next-phase Web management panel.

use anyhow::Result;
use clap::Args;

use crate::output::OutputFormat;

#[derive(Debug, Args)]
pub struct WebArgs {
    /// Port to bind the Web panel to. Defaults to the value in config.toml.
    #[arg(long)]
    pub port: Option<u16>,
    /// Host/interface to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Open the browser automatically after the server starts.
    #[arg(long, default_value_t = false)]
    pub open: bool,
}

pub fn run(args: WebArgs, _fmt: OutputFormat) -> Result<()> {
    let port = args.port.unwrap_or(8787);
    eprintln!(
        "web: the visual debug panel ({host}:{port}, open={open}) is planned for the next phase.\n\
         The HTTP backend and embedded SPA will ship in a follow-up release.",
        host = args.host,
        port = port,
        open = args.open,
    );
    Ok(())
}
