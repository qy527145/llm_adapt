//! CLI 配置文件的持久化层。

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::Settings;

/// 解析 `~/.llm-adapt/`（或 `LLM_ADAPT_HOME` 指向的覆盖路径）。
pub fn config_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("LLM_ADAPT_HOME") {
        return Ok(PathBuf::from(custom));
    }
    let home = dirs::home_dir().context("could not locate home directory")?;
    Ok(home.join(".llm-adapt"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

/// 加载配置；文件不存在时返回 [`Settings::default`]。
pub fn load() -> Result<Settings> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(Settings::default());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let parsed: Settings =
        toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(parsed)
}

/// 持久化设置，必要时创建父目录。
pub fn save(settings: &Settings) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let text = toml::to_string_pretty(settings).context("serialize settings")?;
    fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// 从 TOML 文件导入设置。导入的文件会完全替换当前配置。
pub fn import_from(file: &Path) -> Result<Settings> {
    let text = fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
    let parsed: Settings =
        toml::from_str(&text).with_context(|| format!("parse {}", file.display()))?;
    save(&parsed)?;
    Ok(parsed)
}

/// 把当前设置导出到指定文件。
pub fn export_to(file: &Path, settings: &Settings) -> Result<()> {
    let text = toml::to_string_pretty(settings).context("serialize settings")?;
    if let Some(parent) = file.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).ok();
        }
    }
    fs::write(file, text).with_context(|| format!("write {}", file.display()))?;
    Ok(())
}
