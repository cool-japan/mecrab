//! Wikidata processing for semantic dictionary building
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module handles:
//! - Streaming Wikidata JSON dumps
//! - Building surface → URI index with max-confidence dedup
//! - Entity type filtering via P31 claims
//! - Confidence calibration with ambiguity penalty
//! - Parallel chunk processing via rayon
//! - Merging with dictionary CSV (POS-based URI filtering)
//! - Outputting extended dictionary format

pub mod index;
pub mod parser;
pub mod processor;

pub use index::WikidataIndex;
#[allow(unused_imports)]
pub use parser::{
    Claim, ClaimValue, Claims, LabelValue, Mainsnak, SitelinkValue, WikidataEntry,
    parse_wikidata_dump, parse_wikidata_dump_streaming, pos_to_allowed_entity_types,
};
pub use processor::{BuildConfig, BuildProgress, BuildResult, WikidataProcessor};
