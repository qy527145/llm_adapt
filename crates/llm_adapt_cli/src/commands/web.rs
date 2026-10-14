//! 下一阶段 Web 管理面板的占位。

use anyhow::Result;
use clap::Args;

use crate::output::OutputFormat;

#[derive(Debug, Args)]
pub struct WebArgs {
    /// Web 面板绑定的端口。默认取 config.toml 中的值。
    #[arg(long)]
    pub port: Option<u16>,
    /// 绑定的 host / 网卡接口。
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// 服务启动后自动打开浏览器。
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
