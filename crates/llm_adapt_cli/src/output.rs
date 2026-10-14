//! 各 CLI 命令共用的输出辅助。
//!
//! 所有命令都接受 `--json` 旗标；置位时调用方走 [`emit_json`]。
//! 非 JSON 路径保持纯文本、对人类友好。

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

/// 把一组 `(label, value)` 渲染成对齐的 `key: value` 行。
pub fn print_kv(rows: &[(&str, String)]) {
    let width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (k, v) in rows {
        println!("{k:<width$}  {v}", width = width);
    }
}
