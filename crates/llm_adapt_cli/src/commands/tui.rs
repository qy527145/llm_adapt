//! Placeholder for the next-phase TUI implementation.

use anyhow::Result;

use crate::output::OutputFormat;

pub fn run(_fmt: OutputFormat) -> Result<()> {
    eprintln!(
        "tui: interactive terminal mode is planned for the next phase.\n\
         The core library is fully functional today — use `llm-adapt call/preview/handlers` in the meantime."
    );
    Ok(())
}
