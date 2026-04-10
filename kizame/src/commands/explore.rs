//! Explore command - interactive lattice debugger TUI
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct ExploreArgs {
    /// Text to analyze and explore
    #[arg(required = true)]
    pub text: String,

    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    pub dicdir: Option<PathBuf>,

    /// Path to semantic pool file (semantic.bin)
    #[arg(short = 's', long)]
    pub semantic_pool: Option<PathBuf>,

    /// Compute and display marginal probabilities via the forward-backward
    /// algorithm.  Each node in the cost panel will show "Prob: X.XX" when
    /// this flag is set.
    #[arg(long, default_value_t = false)]
    pub show_probs: bool,
}

pub fn run_explore(args: ExploreArgs) -> Result<(), Box<dyn std::error::Error>> {
    crate::tui::run_explore(&args.text, args.dicdir, args.semantic_pool, args.show_probs)
}
