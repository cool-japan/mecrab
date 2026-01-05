//! MeCrab Builder - Semantic Dictionary Pipeline
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This crate provides the data pipeline for building semantic-enriched
//! dictionaries from Wikidata/Wikipedia dumps. It implements ED-005.
//!
//! ## Architecture
//!
//! ```text
//! Wikidata JSON Dump ──┐
//!                      ├──▶ Surface → URI Index ──▶ Merge with Dictionary CSV
//! Wikipedia Abstracts ─┘                                      │
//!                                                             ▼
//!                                                    Extended Dictionary
//!                                                    (with semantic URIs)
//! ```

mod wikidata;

pub use wikidata::{
    BuildConfig, BuildProgress, BuildResult, WikidataEntry, WikidataIndex, WikidataProcessor,
};

use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during dictionary building
#[derive(Error, Debug)]
pub enum BuildError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV parsing error
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// HTTP request error
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Processing error
    #[error("Processing error: {0}")]
    Processing(String),
}

/// Result type for builder operations
pub type Result<T> = std::result::Result<T, BuildError>;

/// Create a spinner progress bar with the given message
pub fn create_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Create a progress bar for counting items
pub fn create_progress_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Build a semantic dictionary from sources
///
/// This is the main entry point for the build pipeline.
///
/// # Arguments
///
/// * `config` - Build configuration
///
/// # Returns
///
/// Build result with statistics
pub async fn build_dictionary(config: BuildConfig) -> Result<BuildResult> {
    let processor = WikidataProcessor::new(config)?;
    processor.run().await
}

/// Synchronous wrapper for build_dictionary
///
/// This creates a tokio runtime and runs the async build.
pub fn build_dictionary_sync(config: BuildConfig) -> Result<BuildResult> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(build_dictionary(config))
}

/// Check if a path looks like a Wikidata dump
pub fn is_wikidata_dump(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.contains("wikidata") && (name.ends_with(".json.gz") || name.ends_with(".json"))
}

/// Check if a path looks like a Wikipedia dump
pub fn is_wikipedia_dump(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.contains("wikipedia") || name.contains("abstract")
}
