//! Extended API modules for MeCrab.
//!
//! This module houses the higher-level API surface that wraps the core
//! [`crate::MeCrab`] analyser:
//!
//! - [`batch`]  – parallel / sequential batch processing of multiple texts.
//! - [`iter`]   – lazy iterator adapters for streaming large text sets.
//! - [`format`] – `AnalysisResult` output formatting (JSON, JSON-LD, RDF).

pub mod batch;
pub mod format;
pub mod iter;
