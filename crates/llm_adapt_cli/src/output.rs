//! Output helpers shared by every CLI command.
//!
//! All commands accept a `--json` flag; when set, callers go through
//! [`emit_json`]. The non-JSON path stays plain-text and friendly to humans.

use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Human,
    Json,
}

impl OutputFormat {
    pub fn from_flag(json: bool) -> Self {
        if json { Self::Json } else { Self::Human }
    }
}

pub fn emit_json<T: Serialize>(value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    println!("{text}");
    Ok(())
}

/// Render a slice of `(label, value)` pairs as aligned `key: value` lines.
pub fn print_kv(rows: &[(&str, String)]) {
    let width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (k, v) in rows {
        println!("{k:<width$}  {v}", width = width);
    }
}
