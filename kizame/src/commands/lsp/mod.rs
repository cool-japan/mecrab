//! LSP command - Language Server Protocol server for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Provides IDE integration via the Language Server Protocol:
//! - Hover: show morpheme analysis for word under cursor
//! - Completion: complete Japanese words
//! - Diagnostics: highlight unknown words

mod handlers;
mod server;

pub use server::MeCrabLanguageServer;

use clap::Args;
use std::path::PathBuf;
use tower_lsp::{LspService, Server};

/// Arguments for the LSP server subcommand
#[derive(Args)]
pub struct LspArgs {
    /// Path to MeCab dictionary directory (e.g. /usr/local/lib/mecab/dic/ipadic-utf8)
    #[arg(long, help = "Path to MeCab dictionary directory")]
    pub dict: Option<PathBuf>,
}

/// Entry point: start the MeCrab LSP server over stdin/stdout.
///
/// # Errors
///
/// Returns an error if the Tokio runtime cannot be created.
pub fn run_lsp(args: LspArgs) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let (service, socket) =
            LspService::new(|client| MeCrabLanguageServer::new(client, args.dict));
        Server::new(stdin, stdout, socket).serve(service).await;
        Ok(())
    })
}
