//! `llm-adapt config ...`

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Subcommand;
use serde::Serialize;

use crate::config::{store, Profile, Settings};
use crate::output::{emit_json, print_kv, OutputFormat};

#[derive(Debug, Subcommand)]
pub enum ConfigCmd {
    /// List all profiles. API keys are masked.
    List,
    /// Show a single profile (or the active one if `--name` is omitted).
    Show {
        #[arg(long)]
        name: Option<String>,
    },
    /// Set the active profile.
    Use { name: String },
    /// Create or update a single field. The field must be one of:
    /// `base_url`, `api_key`, `api_format`, `default_model`, `proxy`, `verify_ssl`.
    Set {
        profile: String,
        key: String,
        value: String,
    },
    /// Remove a profile.
    Remove { name: String },
    /// Replace the whole config file with the contents of the given TOML file.
    Import { file: PathBuf },
    /// Dump the whole config to a file (or stdout when omitted).
    Export {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Print the path where the config file lives.
    Path,
}

#[derive(Debug, Serialize)]
struct ProfileView<'a> {
    name: &'a str,
    base_url: &'a str,
    api_format: &'a str,
    default_model: Option<&'a str>,
    api_key: String,
    proxy: Option<&'a str>,
    verify_ssl: bool,
    is_active: bool,
}

pub fn run(cmd: ConfigCmd, fmt: OutputFormat) -> Result<()> {
    match cmd {
        ConfigCmd::List => list(fmt),
        ConfigCmd::Show { name } => show(name, fmt),
        ConfigCmd::Use { name } => use_profile(name, fmt),
        ConfigCmd::Set { profile, key, value } => set_field(profile, key, value, fmt),
        ConfigCmd::Remove { name } => remove(name, fmt),
        ConfigCmd::Import { file } => import(file, fmt),
        ConfigCmd::Export { out } => export(out, fmt),
        ConfigCmd::Path => {
            let path = store::config_path()?;
            match fmt {
                OutputFormat::Human => println!("{}", path.display()),
                OutputFormat::Json => emit_json(&serde_json::json!({"path": path.display().to_string()}))?,
            }
            Ok(())
        }
    }
}

fn list(fmt: OutputFormat) -> Result<()> {
    let settings = store::load()?;
    if settings.profiles.is_empty() {
        match fmt {
            OutputFormat::Human => {
                println!("no profiles yet — create one with:");
                println!("  llm-adapt config set <profile> api_key sk-...");
            }
            OutputFormat::Json => emit_json(&serde_json::json!({ "profiles": [] }))?,
        }
        return Ok(());
    }
    let active = settings.active_profile.as_deref();
    let views: Vec<ProfileView<'_>> = settings
        .profiles
        .iter()
        .map(|(name, p)| ProfileView {
            name,
            base_url: &p.base_url,
            api_format: &p.api_format,
            default_model: p.default_model.as_deref(),
            api_key: p.masked_api_key(),
            proxy: p.proxy.as_deref(),
            verify_ssl: p.verify_ssl,
            is_active: active == Some(name.as_str()),
        })
        .collect();
    match fmt {
        OutputFormat::Json => emit_json(&serde_json::json!({"profiles": views}))?,
        OutputFormat::Human => {
            for v in &views {
                let marker = if v.is_active { " (active)" } else { "" };
                println!("[{name}]{marker}", name = v.name);
                print_kv(&[
                    ("  base_url", v.base_url.into()),
                    ("  api_format", v.api_format.into()),
                    ("  api_key", v.api_key.clone()),
                    ("  default_model", v.default_model.unwrap_or("-").into()),
                    ("  proxy", v.proxy.unwrap_or("-").into()),
                    ("  verify_ssl", v.verify_ssl.to_string()),
                ]);
                println!();
            }
        }
    }
    Ok(())
}

fn show(name: Option<String>, fmt: OutputFormat) -> Result<()> {
    let settings = store::load()?;
    let (name, profile) = pick_profile(&settings, name.as_deref())?;
    let view = ProfileView {
        name,
        base_url: &profile.base_url,
        api_format: &profile.api_format,
        default_model: profile.default_model.as_deref(),
        api_key: profile.masked_api_key(),
        proxy: profile.proxy.as_deref(),
        verify_ssl: profile.verify_ssl,
        is_active: settings.active_profile.as_deref() == Some(name),
    };
    match fmt {
        OutputFormat::Json => emit_json(&view)?,
        OutputFormat::Human => {
            print_kv(&[
                ("name", view.name.into()),
                ("base_url", view.base_url.into()),
                ("api_format", view.api_format.into()),
                ("api_key", view.api_key.clone()),
                ("default_model", view.default_model.unwrap_or("-").into()),
                ("proxy", view.proxy.unwrap_or("-").into()),
                ("verify_ssl", view.verify_ssl.to_string()),
            ]);
        }
    }
    Ok(())
}

fn use_profile(name: String, fmt: OutputFormat) -> Result<()> {
    let mut settings = store::load()?;
    if !settings.profiles.contains_key(&name) {
        anyhow::bail!("no profile named '{name}'");
    }
    settings.active_profile = Some(name.clone());
    let path = store::save(&settings)?;
    match fmt {
        OutputFormat::Human => println!("active profile = {name} (saved to {})", path.display()),
        OutputFormat::Json => emit_json(&serde_json::json!({"active": name}))?,
    }
    Ok(())
}

fn set_field(profile: String, key: String, value: String, fmt: OutputFormat) -> Result<()> {
    let mut settings = store::load()?;
    let entry = settings.profiles.entry(profile.clone()).or_insert_with(Profile::default);
    match key.as_str() {
        "base_url"        => entry.base_url = value.clone(),
        "api_key"         => entry.api_key = value.clone(),
        "api_format"      => entry.api_format = value.clone(),
        "default_model"   => entry.default_model = Some(value.clone()),
        "proxy"           => entry.proxy = Some(value.clone()).filter(|s| !s.is_empty()),
        "verify_ssl"      => {
            entry.verify_ssl = value
                .parse::<bool>()
                .with_context(|| format!("'{value}' is not a bool"))?;
        }
        other => anyhow::bail!("unknown config key '{other}'"),
    }
    if settings.active_profile.is_none() {
        settings.active_profile = Some(profile.clone());
    }
    let path = store::save(&settings)?;
    match fmt {
        OutputFormat::Human => {
            println!("updated profile '{profile}': {key} ← {}",
                if key == "api_key" { crate::config::mask_secret(&value) } else { value.clone() });
            println!("({})", path.display());
        }
        OutputFormat::Json => emit_json(&serde_json::json!({
            "profile": profile, "key": key, "value": value
        }))?,
    }
    Ok(())
}

fn remove(name: String, fmt: OutputFormat) -> Result<()> {
    let mut settings = store::load()?;
    if settings.profiles.remove(&name).is_none() {
        anyhow::bail!("no profile named '{name}'");
    }
    if settings.active_profile.as_deref() == Some(&name) {
        settings.active_profile = settings.profiles.keys().next().cloned();
    }
    store::save(&settings)?;
    match fmt {
        OutputFormat::Human => println!("removed profile '{name}'"),
        OutputFormat::Json => emit_json(&serde_json::json!({"removed": name}))?,
    }
    Ok(())
}

fn import(file: PathBuf, fmt: OutputFormat) -> Result<()> {
    let settings = store::import_from(&file)?;
    match fmt {
        OutputFormat::Human => println!(
            "imported {} profile(s) from {}",
            settings.profiles.len(),
            file.display()
        ),
        OutputFormat::Json => emit_json(&serde_json::json!({
            "imported_profiles": settings.profiles.len(),
            "from": file.display().to_string(),
        }))?,
    }
    Ok(())
}

fn export(out: Option<PathBuf>, _fmt: OutputFormat) -> Result<()> {
    let settings = store::load()?;
    let text = toml::to_string_pretty(&settings)?;
    match out {
        Some(path) => {
            store::export_to(&path, &settings)?;
            println!("exported to {}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// Find a profile by explicit name, otherwise the active one.
pub fn pick_profile<'a>(
    settings: &'a Settings,
    name: Option<&'a str>,
) -> Result<(&'a str, &'a Profile)> {
    if let Some(n) = name {
        let p = settings
            .profiles
            .get(n)
            .with_context(|| format!("no profile named '{n}'"))?;
        return Ok((n, p));
    }
    settings
        .active()
        .context("no active profile — run `llm-adapt config use <name>` or pass --profile")
}
