//! 下一阶段 TUI 实现的占位。

use anyhow::Result;

use crate::output::OutputFormat;

pub fn run(_fmt: OutputFormat) -> Result<()> {
    eprintln!(
        "tui: interactive terminal mode is planned for the next phase.\n\
         The core library is fully functional today — use `llm-adapt call/preview/handlers` in the meantime."
    );
    Ok(())
}
