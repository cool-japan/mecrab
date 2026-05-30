//! JSON streaming parser for Wikidata dumps
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Provides:
//! - Wikidata JSON schema structs (`WikidataEntry`, `Claims`, `ClaimValue`, …)
//! - `WikidataEntry` surface extraction and type-filtering helpers
//! - Streaming + chunked parallel parse helpers used by `WikidataProcessor`

use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::index::WikidataIndex;
use crate::{BuildError, Result, create_spinner};

// ─────────────────────────────────────────────────────────────
// Wikidata JSON schema structs
// ─────────────────────────────────────────────────────────────

/// A claim datavalue (object values carry Q-IDs in .value.id)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimValue {
    #[serde(rename = "type")]
    pub value_type: Option<String>,
    pub value: Option<serde_json::Value>,
}

/// Main snak in a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mainsnak {
    pub datavalue: Option<ClaimValue>,
}

/// A single Wikidata claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub mainsnak: Mainsnak,
}

/// Claims data – only P31 ("instance of") is extracted for entity type filtering
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Claims {
    #[serde(rename = "P31", default)]
    pub instance_of: Vec<Claim>,
}

/// A Wikidata entity entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikidataEntry {
    /// Wikidata ID (e.g., "Q1490")
    pub id: String,
    /// Labels in different languages
    #[serde(default)]
    pub labels: HashMap<String, LabelValue>,
    /// Aliases in different languages
    #[serde(default)]
    pub aliases: HashMap<String, Vec<LabelValue>>,
    /// Sitelinks (Wikipedia pages)
    #[serde(default)]
    pub sitelinks: HashMap<String, SitelinkValue>,
    /// Claims (P31 "instance of" used for entity-type filtering)
    #[serde(default)]
    pub claims: Claims,
}

/// A label/alias value in a given language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelValue {
    pub language: String,
    pub value: String,
}

/// A sitelink value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitelinkValue {
    pub site: String,
    pub title: String,
}

// ─────────────────────────────────────────────────────────────
// WikidataEntry impl
// ─────────────────────────────────────────────────────────────

impl WikidataEntry {
    /// Get all Japanese surface forms for this entity
    pub fn japanese_surfaces(&self) -> Vec<String> {
        let mut surfaces = Vec::new();

        if let Some(label) = self.labels.get("ja") {
            surfaces.push(label.value.clone());
        }

        if let Some(aliases) = self.aliases.get("ja") {
            for alias in aliases {
                if !surfaces.contains(&alias.value) {
                    surfaces.push(alias.value.clone());
                }
            }
        }

        surfaces
    }

    /// Popularity score based on sitelink count (log-scaled, [0.1, 1.0])
    pub fn popularity(&self) -> f32 {
        let count = self.sitelinks.len();
        // 0 sitelinks → 0.1, 100+ sitelinks → 1.0
        (0.1 + (count as f32 / 100.0).min(0.9)).min(1.0)
    }

    /// Returns Wikidata Q-IDs of entity types (from P31 "instance of" claims)
    pub fn entity_type_ids(&self) -> Vec<String> {
        self.claims
            .instance_of
            .iter()
            .filter_map(|claim| {
                let dv = claim.mainsnak.datavalue.as_ref()?;
                let val = dv.value.as_ref()?;
                // Value object: {"entity-type":"item","numeric-id":...,"id":"Q5"}
                val.get("id")?.as_str().map(|s| s.to_string())
            })
            .collect()
    }

    /// Check if this entity passes the given type filter.
    /// An empty filter means "allow all".
    pub fn matches_type_filter(&self, filter: &[String]) -> bool {
        if filter.is_empty() {
            return true;
        }
        let types = self.entity_type_ids();
        types.iter().any(|t| filter.contains(t))
    }

    /// Calibrated confidence for a specific surface form.
    ///
    /// Factors considered:
    /// - Entity popularity (sitelinks, log-scaled via `popularity()`)
    /// - Primary-label boost (+20%) when `surface` is the Japanese label
    ///
    /// An additional ambiguity penalty is applied post-hoc by
    /// [`WikidataIndex::recalibrate()`].
    pub fn calibrated_confidence(&self, surface: &str) -> f32 {
        let base = self.popularity();

        let is_primary = self
            .labels
            .get("ja")
            .map(|l| l.value == surface)
            .unwrap_or(false);
        let label_boost = if is_primary { 1.2_f32 } else { 1.0_f32 };

        (base * label_boost).clamp(0.05, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────
// POS-based entity-type filtering helper
// ─────────────────────────────────────────────────────────────

/// Map an IPADIC POS feature string to allowed Wikidata entity-type Q-IDs.
///
/// Returns:
/// - `None`         → no restriction (allow any URI)
/// - `Some(vec![])` → no entity URIs allowed (common noun etc.)
/// - `Some(ids)`    → only URIs whose types intersect with `ids`
pub fn pos_to_allowed_entity_types(pos_feature: &str) -> Option<Vec<&'static str>> {
    if pos_feature.contains("人名") {
        // persons and musical artists
        Some(vec!["Q5", "Q215380"])
    } else if pos_feature.contains("地名") {
        // city, country, region, state
        Some(vec!["Q515", "Q6256", "Q82794", "Q35657"])
    } else if pos_feature.contains("組織") {
        // business, organisation, political party
        Some(vec!["Q4830453", "Q43229", "Q7278"])
    } else if pos_feature.contains("固有名詞") {
        // general proper noun: allow everything
        None
    } else {
        // common noun etc.: no entity URIs
        Some(vec![])
    }
}

// ─────────────────────────────────────────────────────────────
// Chunked parallel parse (used by WikidataProcessor)
// ─────────────────────────────────────────────────────────────

/// Parse a Wikidata JSON dump (plain or gzip), merging results into `index`.
///
/// Lines are read sequentially (I/O bound), parsed in parallel chunks via
/// rayon, then merged sequentially into the index.
pub fn parse_wikidata_dump(
    path: &Path,
    entity_type_filter: &[String],
    calibration_enabled: bool,
    index: &mut WikidataIndex,
) -> Result<()> {
    let pb = create_spinner("Loading Wikidata dump...");

    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let chunk_size = 10_000usize;
    let mut chunk: Vec<String> = Vec::with_capacity(chunk_size);
    let mut count = 0u64;

    let mut lines_iter = reader.lines();

    loop {
        chunk.clear();

        // Collect up to chunk_size usable lines
        for line_result in lines_iter.by_ref().take(chunk_size) {
            let line = line_result?;
            let trimmed = line.trim().to_string();
            if trimmed != "[" && trimmed != "]" && !trimmed.is_empty() {
                chunk.push(trimmed);
            }
        }

        if chunk.is_empty() {
            break;
        }

        // Parallel parse + surface extraction
        // Each element: (surface, uri, confidence, entity_type_ids)
        // entity_type_ids contains the P31 Q-IDs for the URI (may be empty).
        let parsed: Vec<(String, String, f32, Vec<String>)> = chunk
            .par_iter()
            .flat_map(|line| {
                let json_str = line.trim_end_matches(',');
                match serde_json::from_str::<WikidataEntry>(json_str) {
                    Ok(entry) if entry.matches_type_filter(entity_type_filter) => {
                        let uri = format!("http://www.wikidata.org/entity/{}", entry.id);
                        let type_ids = entry.entity_type_ids();
                        entry
                            .japanese_surfaces()
                            .into_iter()
                            .map(|surface| {
                                let conf = entry.calibrated_confidence(&surface);
                                (surface, uri.clone(), conf, type_ids.clone())
                            })
                            .collect::<Vec<_>>()
                    }
                    _ => vec![],
                }
            })
            .collect();

        // Sequential merge into index (index is not Sync)
        for (surface, uri, confidence, type_ids) in parsed {
            index.add(&surface, &uri, confidence);
            index.add_entity_types(&uri, type_ids);
        }

        count += chunk.len() as u64;
        if count % 100_000 < chunk_size as u64 {
            pb.set_message(format!("Processed {} lines...", count));
        }
    }

    // Post-processing: ambiguity-penalty recalibration
    if calibration_enabled {
        index.recalibrate();
    }

    pb.finish_with_message(format!(
        "Indexed {} lines, {} surface forms, {} mappings",
        count,
        index.len(),
        index.total_mappings()
    ));

    Ok(())
}

/// Memory-efficient streaming version of [`parse_wikidata_dump`].
///
/// Uses a bounded sync channel to limit in-flight memory:
/// - A dedicated reader thread fills the channel (back-pressure when full).
/// - The calling thread consumes batches and processes them with rayon.
///
/// Use this variant when processing dumps > 10 GB to avoid OOM.
///
/// # Arguments
/// * `path`             – path to the Wikidata JSON dump (plain or .gz)
/// * `channel_capacity` – max line-level buffer size (clamped to ≥ 1 000)
pub fn parse_wikidata_dump_streaming(
    path: &Path,
    entity_type_filter: &[String],
    calibration_enabled: bool,
    verbose: bool,
    channel_capacity: usize,
    index: &mut WikidataIndex,
) -> std::result::Result<(), BuildError> {
    use std::sync::mpsc::sync_channel;

    let channel_capacity = channel_capacity.max(1_000);
    // One slot per 1 000-line batch → bounded to channel_capacity / 1 000 batches
    let slot_count = (channel_capacity / 1_000).max(1);
    let (sender, receiver) = sync_channel::<Vec<String>>(slot_count);

    let path_clone = path.to_path_buf();

    let pb = if verbose {
        Some(crate::create_spinner("Streaming Wikidata dump..."))
    } else {
        None
    };

    // ── Reader thread ────────────────────────────────────────
    let reader_handle = std::thread::spawn(move || -> std::io::Result<u64> {
        let file = std::fs::File::open(&path_clone)?;
        let gz_ext = path_clone
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gz"));

        let reader: Box<dyn std::io::BufRead + Send> = if gz_ext {
            Box::new(std::io::BufReader::with_capacity(
                1 << 20,
                flate2::read::GzDecoder::new(file),
            ))
        } else {
            Box::new(std::io::BufReader::with_capacity(1 << 20, file))
        };

        let mut batch: Vec<String> = Vec::with_capacity(1_000);
        let mut count = 0u64;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() && trimmed != "[" && trimmed != "]" {
                batch.push(trimmed);
                count += 1;

                if batch.len() >= 1_000 {
                    let send_batch = std::mem::replace(&mut batch, Vec::with_capacity(1_000));
                    // Block here when channel is full (back-pressure)
                    if sender.send(send_batch).is_err() {
                        // Receiver dropped – consumer exited early
                        break;
                    }
                }
            }
        }

        if !batch.is_empty() {
            sender.send(batch).ok();
        }

        Ok(count)
    });

    // ── Consumer loop ────────────────────────────────────────
    let mut batches_consumed = 0u64;

    for batch in &receiver {
        // Each element: (surface, uri, confidence, entity_type_ids)
        let parsed: Vec<(String, String, f32, Vec<String>)> = batch
            .par_iter()
            .flat_map(|line| {
                let json_str = line.trim_end_matches(',');
                match serde_json::from_str::<WikidataEntry>(json_str) {
                    Ok(entry) if entry.matches_type_filter(entity_type_filter) => {
                        let uri = format!("http://www.wikidata.org/entity/{}", entry.id);
                        let type_ids = entry.entity_type_ids();
                        entry
                            .japanese_surfaces()
                            .into_iter()
                            .map(|surface| {
                                let conf = entry.calibrated_confidence(&surface);
                                (surface, uri.clone(), conf, type_ids.clone())
                            })
                            .collect::<Vec<_>>()
                    }
                    _ => vec![],
                }
            })
            .collect();

        for (surface, uri, confidence, type_ids) in parsed {
            index.add(&surface, &uri, confidence);
            index.add_entity_types(&uri, type_ids);
        }

        batches_consumed += 1;
        if let Some(ref pb) = pb {
            if batches_consumed % 100 == 0 {
                pb.set_message(format!(
                    "~{} entries processed...",
                    batches_consumed * 1_000
                ));
            }
        }
    }

    // ── Wait for reader ──────────────────────────────────────
    match reader_handle.join() {
        Ok(Ok(_lines)) => {}
        Ok(Err(e)) => return Err(BuildError::Io(e)),
        Err(_) => return Err(BuildError::Processing("Reader thread panicked".to_string())),
    }

    if calibration_enabled {
        index.recalibrate();
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!(
            "Streaming complete: {} surface forms, {} mappings",
            index.len(),
            index.total_mappings()
        ));
    }

    Ok(())
}
