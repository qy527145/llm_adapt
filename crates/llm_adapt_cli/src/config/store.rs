//! Persistence layer for the CLI config file.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::Settings;

/// Resolve `~/.llm-adapt/` (or the override pointed at by `LLM_ADAPT_HOME`).
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

/// Load the config, returning [`Settings::default`] if the file does not exist.
pub fn load() -> Result<Settings> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(Settings::default());
    }
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let parsed: Settings = toml::from_str(&text)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(parsed)
}

/// Persist the settings, creating the parent directory if necessary.
pub fn save(settings: &Settings) -> Result<PathBuf> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let text = toml::to_string_pretty(settings).context("serialize settings")?;
    fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Import settings from a TOML file. The imported file completely replaces
/// the current config.
pub fn import_from(file: &Path) -> Result<Settings> {
    let text = fs::read_to_string(file)
        .with_context(|| format!("read {}", file.display()))?;
    let parsed: Settings = toml::from_str(&text)
        .with_context(|| format!("parse {}", file.display()))?;
    save(&parsed)?;
    Ok(parsed)
}

/// Export the current settings into the given file.
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
