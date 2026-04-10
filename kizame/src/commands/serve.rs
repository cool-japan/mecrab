//! Serve command - HTTP API server
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use clap::Args;
use mecrab::MeCrab;
use std::path::PathBuf;

#[derive(Args)]
pub struct ServeArgs {
    /// Server bind address
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    pub host: String,

    /// Server port
    #[arg(short = 'p', long, default_value = "3000")]
    pub port: u16,

    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    pub dicdir: Option<PathBuf>,
}

pub fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mecrab = MeCrab::builder().dicdir(args.dicdir).build()?;

    let addr_str = format!("{}:{}", args.host, args.port);
    let addr = addr_str
        .parse()
        .map_err(|e| format!("Invalid address '{}': {}", addr_str, e))?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(crate::server::run_server(mecrab, addr))
}
